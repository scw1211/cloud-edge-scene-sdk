"""Fail-closed guided deployment controller for the product demo.

The browser never submits shell text, credentials, ports, or model paths.  A
real deployment is possible only through a server-side profile that was
prepared by an operator.  The controller exposes a small state machine and a
fixed set of read-only/deployment commands for already provisioned nodes.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SCHEMA_VERSION = "guided-deployment-controller/v1"
PROFILE_SCHEMA_VERSION = "guided-deployment-profiles/v1"
PROFILE_SCHEMA_VERSION_V2 = "guided-deployment-profiles/v2"
SESSION_SCHEMA_VERSION = "guided-deployment-session/v1"
STAGES = (
    ("select_scenario", "选择使用场景"),
    ("connect_nodes", "连接云端与边缘设备"),
    ("preflight", "自动检查运行环境"),
    ("generate_plan", "生成推荐部署方案"),
    ("deploy", "安装并启动系统"),
    ("verify", "运行样例并验证结果"),
)
TEMPLATES = {
    "recommended": "推荐部署",
    "low_memory": "低内存部署",
    "high_performance": "高性能部署",
    "offline": "离线部署",
}
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_SAFE_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9.:-]{0,252}")
_SAFE_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,31}")


class DeploymentError(RuntimeError):
    """A user-facing, fail-closed deployment error."""

    def __init__(self, code: str, message: str, suggestion: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggestion = suggestion

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "suggestion": self.suggestion,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    with temporary.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validate_remote_path(value: Any, field: str) -> str:
    text = str(value or "")
    if not text.startswith("/") or "\x00" in text or "\n" in text or "\r" in text:
        raise ValueError(f"{field} must be an absolute path")
    return text


def _validated_node(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    host = str(value.get("host", ""))
    user = str(value.get("user", ""))
    if not _SAFE_HOST.fullmatch(host) or not _SAFE_USER.fullmatch(user):
        raise ValueError(f"{field} contains an invalid host or user")
    port = int(value.get("ssh_port", 22))
    if not 1 <= port <= 65535:
        raise ValueError(f"{field}.ssh_port is invalid")
    node = {
        "host": host,
        "user": user,
        "ssh_port": port,
        "identity_file": _validate_remote_path(value.get("identity_file"), f"{field}.identity_file"),
        "sdk_root": _validate_remote_path(value.get("sdk_root"), f"{field}.sdk_root"),
        "label": str(value.get("label") or field),
    }
    if field == "edge":
        node["llama_binary"] = _validate_remote_path(
            value.get("llama_binary"), "edge.llama_binary"
        )
    return node


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _profile_sha256(profile: dict[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in profile.items() if key != "profile_sha256"}
    )


def _profile_endpoint(value: Any, field: str) -> str:
    """Use the public endpoint validator while loading a server-side profile."""

    try:
        return _validated_endpoint(value, field)
    except DeploymentError as exc:
        raise ValueError(f"{field} is invalid: {exc.message}") from exc


def _validated_profile_node(value: Any, field: str) -> dict[str, Any]:
    """Validate one v2 inventory node and retain secrets only server-side."""

    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    node_id = str(value.get("id", ""))
    role = str(value.get("role", ""))
    if not _SAFE_ID.fullmatch(node_id):
        raise ValueError(f"{field}.id is invalid")
    if role not in {"cloud", "edge"}:
        raise ValueError(f"{field}.role must be cloud or edge")
    ssh = value.get("ssh", {})
    if ssh is None:
        ssh = {}
    if not isinstance(ssh, dict):
        raise ValueError(f"{field}.ssh must be an object")
    # Both the flat v1-style spelling and a nested ssh object are accepted.
    # The normalized result is never returned by catalog().
    remote = {
        "host": ssh.get("host", value.get("host")),
        "user": ssh.get("user", value.get("user")),
        "ssh_port": ssh.get("ssh_port", value.get("ssh_port", 22)),
        "identity_file": ssh.get("identity_file", value.get("identity_file")),
        "sdk_root": value.get("sdk_root"),
        "label": value.get("label") or node_id,
    }
    if role == "edge":
        remote["llama_binary"] = value.get("llama_binary")
    node = _validated_node(remote, role)
    node.update(
        {
            "id": node_id,
            "role": role,
            "service_url": _profile_endpoint(
                value.get("service_url"), f"{field}.service_url"
            ),
        }
    )
    return node


def _normalize_topology(
    profile: dict[str, Any], value: Any, *, public_error: bool
) -> dict[str, Any]:
    """Return a stable ID-only topology after fail-closed authorization checks."""

    def fail(message: str) -> None:
        if public_error:
            raise DeploymentError(
                "TOPOLOGY_INVALID",
                "部署拓扑无效",
                message,
            )
        raise ValueError(message)

    if not isinstance(value, dict):
        fail("topology must be an object")
    allowed_keys = {"primary_cloud_id", "cloud_node_ids", "edge_bindings"}
    if set(value) - allowed_keys:
        fail("topology contains unsupported fields")
    primary = str(value.get("primary_cloud_id", ""))
    raw_cloud_ids = value.get("cloud_node_ids")
    raw_bindings = value.get("edge_bindings")
    if raw_bindings is None:
        raw_bindings = []
    if not isinstance(raw_bindings, list):
        fail("edge_bindings must be a list")
    bindings: list[dict[str, str]] = []
    for index, raw in enumerate(raw_bindings):
        if not isinstance(raw, dict) or set(raw) - {"edge_id", "cloud_id"}:
            fail(f"edge_bindings[{index}] is invalid")
        edge_id = str(raw.get("edge_id", ""))
        cloud_id = str(raw.get("cloud_id", ""))
        if not _SAFE_ID.fullmatch(edge_id) or not _SAFE_ID.fullmatch(cloud_id):
            fail(f"edge_bindings[{index}] contains an invalid node id")
        bindings.append({"edge_id": edge_id, "cloud_id": cloud_id})
    if raw_cloud_ids is None:
        # v2 defaults only require a primary and bindings. Include all declared
        # cloud nodes so standby nodes can be part of the default deployment.
        raw_cloud_ids = [
            node_id
            for node_id, node in profile["node_by_id"].items()
            if node["role"] == "cloud"
        ]
    if not isinstance(raw_cloud_ids, list):
        fail("cloud_node_ids must be a list")
    cloud_ids = [str(item) for item in raw_cloud_ids]
    if not cloud_ids or len(set(cloud_ids)) != len(cloud_ids):
        fail("cloud_node_ids must contain at least one unique cloud id")
    if not bindings:
        fail("edge_bindings must contain at least one edge")
    edge_ids = [item["edge_id"] for item in bindings]
    if len(set(edge_ids)) != len(edge_ids):
        fail("each edge may be bound only once")
    if primary not in cloud_ids:
        fail("primary_cloud_id must identify one selected cloud")
    node_by_id = profile["node_by_id"]
    for cloud_id in cloud_ids:
        node = node_by_id.get(cloud_id)
        if node is None or node["role"] != "cloud":
            fail(f"unauthorized or non-cloud node: {cloud_id}")
    for binding in bindings:
        edge = node_by_id.get(binding["edge_id"])
        if edge is None or edge["role"] != "edge":
            fail(f"unauthorized or non-edge node: {binding['edge_id']}")
        if binding["cloud_id"] not in cloud_ids:
            fail(f"edge {binding['edge_id']} is bound to an unselected cloud")
    return {
        "primary_cloud_id": primary,
        "cloud_node_ids": sorted(cloud_ids),
        "edge_bindings": sorted(bindings, key=lambda item: item["edge_id"]),
    }


def _validate_profile_v1(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("profile must be an object")
    profile_id = str(value.get("id", ""))
    if not _SAFE_ID.fullmatch(profile_id):
        raise ValueError("profile id is invalid")
    cloud_url = _profile_endpoint(value.get("cloud_url"), "profile.cloud_url")
    edge_url = _profile_endpoint(value.get("edge_url"), "profile.edge_url")
    cloud = _validated_node(value.get("cloud"), "cloud")
    cloud.update({"id": "cloud", "role": "cloud", "service_url": cloud_url})
    edge = _validated_node(value.get("edge"), "edge")
    edge.update({"id": "edge", "role": "edge", "service_url": edge_url})
    profile = {
        "id": profile_id,
        "label": str(value.get("label") or profile_id),
        "enabled": value.get("enabled", True) is True,
        "allow_deploy": value.get("allow_deploy", False) is True,
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "legacy_v1": True,
        "node_by_id": {"cloud": cloud, "edge": edge},
    }
    profile["default_topology"] = _normalize_topology(
        profile,
        {
            "primary_cloud_id": "cloud",
            "cloud_node_ids": ["cloud"],
            "edge_bindings": [{"edge_id": "edge", "cloud_id": "cloud"}],
        },
        public_error=False,
    )
    profile["profile_sha256"] = _profile_sha256(profile)
    return profile


def _validate_profile_v2(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("profile must be an object")
    profile_id = str(value.get("id", ""))
    if not _SAFE_ID.fullmatch(profile_id):
        raise ValueError("profile id is invalid")
    records = value.get("nodes")
    if not isinstance(records, list) or not records:
        raise ValueError("v2 profile nodes must be a non-empty list")
    node_by_id: dict[str, dict[str, Any]] = {}
    endpoints: set[str] = set()
    host_roles: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        node = _validated_profile_node(record, f"profile.nodes[{index}]")
        if node["id"] in node_by_id:
            raise ValueError("duplicate node id in profile")
        if node["service_url"] in endpoints:
            raise ValueError("duplicate node service_url in profile")
        host_role = (str(node["host"]).lower(), node["role"])
        if host_role in host_roles:
            raise ValueError("duplicate host-role in profile")
        node_by_id[node["id"]] = node
        endpoints.add(node["service_url"])
        host_roles.add(host_role)
    profile = {
        "id": profile_id,
        "label": str(value.get("label") or profile_id),
        "enabled": value.get("enabled", True) is True,
        "allow_deploy": value.get("allow_deploy", False) is True,
        "profile_schema_version": PROFILE_SCHEMA_VERSION_V2,
        "legacy_v1": False,
        "node_by_id": node_by_id,
    }
    profile["default_topology"] = _normalize_topology(
        profile, value.get("default_topology"), public_error=False
    )
    profile["profile_sha256"] = _profile_sha256(profile)
    return profile


def _validated_endpoint(value: Any, field: str) -> str:
    """Validate a health-check base URL without accepting credentials or paths."""

    text = str(value or "").strip().rstrip("/")
    parsed = urlparse(text)
    if parsed.scheme != "http" or not parsed.hostname:
        raise DeploymentError(
            "INVALID_ENDPOINT",
            f"{field}地址无效",
            "请输入形如 http://192.168.1.10:18100 的服务地址。",
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise DeploymentError(
            "INVALID_ENDPOINT",
            f"{field}地址不能包含账号、查询参数或片段",
            "这里只填写服务根地址，认证信息不能写在URL中。",
        )
    if parsed.path not in {"", "/"}:
        raise DeploymentError(
            "INVALID_ENDPOINT",
            f"{field}地址不能包含接口路径",
            "请删除 /health 等路径，系统会自动执行只读健康检查。",
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise DeploymentError(
            "INVALID_ENDPOINT", f"{field}端口无效", "端口必须在1到65535之间。"
        ) from exc
    if port is not None and not 1 <= port <= 65535:
        raise DeploymentError(
            "INVALID_ENDPOINT", f"{field}端口无效", "端口必须在1到65535之间。"
        )
    return text


class GuidedDeploymentController:
    """Persistent six-stage controller with an injectable command runner."""

    def __init__(
        self,
        *,
        profiles_path: Path | None,
        state_root: Path,
        connection_defaults: dict[str, str] | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.state_root = state_root.resolve()
        self.runner = runner
        self.profiles_path = profiles_path.resolve() if profiles_path else None
        self.profiles: dict[str, dict[str, Any]] = {}
        self.profile_identity: dict[str, Any] | None = None
        self.connection_defaults = (
            {
                "edge_url": _validated_endpoint(
                    (connection_defaults or {}).get("edge_url", ""), "边缘服务"
                ),
                "cloud_url": _validated_endpoint(
                    (connection_defaults or {}).get("cloud_url", ""), "云端服务"
                ),
                "cloud_advertised_url": _validated_endpoint(
                    (connection_defaults or {}).get(
                        "cloud_advertised_url",
                        (connection_defaults or {}).get("cloud_url", ""),
                    ),
                    "云端对外服务",
                ),
            }
            if connection_defaults
            else None
        )
        if self.profiles_path is not None:
            payload = _read_json(self.profiles_path)
            profile_schema_version = payload.get("schema_version")
            if profile_schema_version not in {
                PROFILE_SCHEMA_VERSION,
                PROFILE_SCHEMA_VERSION_V2,
            }:
                raise ValueError("deployment profile schema_version mismatch")
            records = payload.get("profiles")
            if not isinstance(records, list):
                raise ValueError("deployment profiles must be a list")
            for record in records:
                profile = (
                    _validate_profile_v1(record)
                    if profile_schema_version == PROFILE_SCHEMA_VERSION
                    else _validate_profile_v2(record)
                )
                if profile["id"] in self.profiles:
                    raise ValueError("duplicate deployment profile id")
                self.profiles[profile["id"]] = profile
            raw = self.profiles_path.read_bytes()
            self.profile_identity = {
                "path": str(self.profiles_path),
                "bytes": len(raw),
                "sha256": _sha256_bytes(raw),
            }

    def catalog(self) -> dict[str, Any]:
        profiles = [
            {
                "id": item["id"],
                "label": item["label"],
                "enabled": item["enabled"],
                "allow_deploy": item["allow_deploy"],
                "profile_schema_version": item["profile_schema_version"],
                "nodes": [
                    {
                        "id": node["id"],
                        "role": node["role"],
                        "label": node["label"],
                        "service_url": node["service_url"],
                        "credential_ready": bool(node.get("identity_file")),
                    }
                    for node in sorted(
                        item["node_by_id"].values(), key=lambda node: node["id"]
                    )
                ],
                "default_topology": item["default_topology"],
            }
            for item in self.profiles.values()
            if item["enabled"]
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "live_deployment_available": bool(profiles),
            "profile_source": (
                {
                    "configured": True,
                    "bytes": self.profile_identity["bytes"],
                    "sha256": self.profile_identity["sha256"],
                }
                if self.profile_identity
                else None
            ),
            "profiles": profiles,
            "scenarios": [
                {"id": "traffic", "label": "交通监控", "enabled": True},
                {"id": "industrial", "label": "工业异常检测", "enabled": False},
                {"id": "both", "label": "交通 + 工业", "enabled": False},
            ],
            "templates": [
                {"id": key, "label": label, "enabled": key == "recommended"}
                for key, label in TEMPLATES.items()
            ],
            "stages": [{"id": key, "label": label} for key, label in STAGES],
            "claim_boundary": (
                "仅支持服务端预先配置、已具备SSH密钥和运行资产的设备；"
                "浏览器只选择已授权节点ID，不接收SSH秘密；"
                "未配置档案时只能查看流程，不会模拟部署成功。"
            ),
            "runtime_connection_defaults": self.connection_defaults,
        }

    @staticmethod
    def _health_summary(health: dict[str, Any]) -> dict[str, Any]:
        runtime = health.get("runtime") if isinstance(health.get("runtime"), dict) else {}
        scenes = runtime.get("scenes") if isinstance(runtime.get("scenes"), list) else []
        plugins = runtime.get("plugins") if isinstance(runtime.get("plugins"), list) else []
        model: dict[str, Any] | None = None
        product_count: int | None = None
        for plugin in plugins:
            if not isinstance(plugin, dict):
                continue
            plugin_health = plugin.get("health")
            if not isinstance(plugin_health, dict):
                continue
            if plugin.get("scene") == "industrial_anomaly" and isinstance(
                plugin_health.get("product_count"), int
            ):
                product_count = int(plugin_health["product_count"])
            edge_llm = plugin_health.get("edge_llm")
            active = edge_llm.get("active") if isinstance(edge_llm, dict) else None
            if isinstance(active, dict):
                deployment = active.get("deployment")
                deployment = deployment if isinstance(deployment, dict) else {}
                metrics = active.get("metrics")
                metrics = metrics if isinstance(metrics, dict) else {}
                model = {
                    "release_id": active.get("release_id"),
                    "quantization": deployment.get("quantization"),
                    "artifact_bytes": deployment.get("artifact_bytes"),
                    "input_tokens": deployment.get("input_tokens"),
                    "output_tokens": deployment.get("output_tokens"),
                    "thinking": deployment.get("thinking"),
                    "traffic_accuracy": metrics.get("nano_traffic_accuracy"),
                    "industrial_accuracy": metrics.get("nano_industrial_accuracy"),
                    "mean_request_ms": metrics.get("nano_overall_mean_latency_ms"),
                    "strict_peak_bytes": metrics.get("nano_strict_peak_bytes"),
                }
                break
        return {
            "status": health.get("status"),
            "ready": health.get("ready") is True,
            "role": health.get("role"),
            "framework_version": health.get("framework_version"),
            "generation": runtime.get("generation"),
            "scenes": scenes,
            "model": model,
            "industrial_product_count": product_count,
        }

    def _probe_node(self, role: str, endpoint: str, timeout: float) -> dict[str, Any]:
        started = time.perf_counter_ns()
        try:
            health = self._request_json(endpoint + "/health", timeout=timeout)
        except Exception as exc:
            return {
                "online": False,
                "role": role,
                "endpoint": endpoint,
                "latency_ms": round((time.perf_counter_ns() - started) / 1_000_000, 3),
                "error_code": type(exc).__name__,
                "message": "连接超时" if isinstance(exc, TimeoutError) else "健康检查失败",
            }
        return {
            "online": True,
            "role": role,
            "endpoint": endpoint,
            "latency_ms": round((time.perf_counter_ns() - started) / 1_000_000, 3),
            "health": self._health_summary(health),
        }

    def probe_endpoints(
        self, *, edge_url: Any, cloud_url: Any, timeout_seconds: float = 3.0
    ) -> dict[str, Any]:
        """Probe user-selected service roots; never deploy or mutate either node."""

        endpoints = {
            "edge": _validated_endpoint(edge_url, "边缘服务"),
            "cloud": _validated_endpoint(cloud_url, "云端服务"),
        }
        timeout = max(0.5, min(float(timeout_seconds), 5.0))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                role: pool.submit(self._probe_node, role, endpoint, timeout)
                for role, endpoint in endpoints.items()
            }
            nodes = {role: future.result() for role, future in futures.items()}
        all_online = all(node["online"] for node in nodes.values())
        return {
            "schema_version": "runtime-connection-check/v1",
            "configured": True,
            "online": all_online,
            "partial_online": any(node["online"] for node in nodes.values())
            and not all_online,
            "observed_at": _utc_now(),
            "read_only": True,
            "nodes": nodes,
        }

    def probe_topology(
        self, *, nodes: Any, timeout_seconds: float = 3.0
    ) -> dict[str, Any]:
        """Probe up to 32 caller-described HTTP endpoints without mutation.

        This endpoint is deliberately limited to the localhost-bound demo
        controller. It applies the same HTTP-root validation as the original
        two-node connection check and must not be exposed as a network scanner.
        """

        if not isinstance(nodes, list) or not 1 <= len(nodes) <= 32:
            raise DeploymentError(
                "INVALID_TOPOLOGY_PROBE",
                "节点列表无效",
                "只读连接检查每次需要1到32个节点。",
            )
        normalized: list[dict[str, str]] = []
        node_ids: set[str] = set()
        endpoints: set[str] = set()
        for index, raw in enumerate(nodes):
            if not isinstance(raw, dict) or set(raw) - {
                "id",
                "role",
                "label",
                "endpoint",
            }:
                raise DeploymentError(
                    "INVALID_TOPOLOGY_PROBE",
                    "节点描述无效",
                    f"第{index + 1}个节点只能包含id、role、label和endpoint。",
                )
            node_id = str(raw.get("id", ""))
            role = str(raw.get("role", ""))
            if not _SAFE_ID.fullmatch(node_id) or role not in {"cloud", "edge"}:
                raise DeploymentError(
                    "INVALID_TOPOLOGY_PROBE",
                    "节点身份无效",
                    f"第{index + 1}个节点的id或role无效。",
                )
            endpoint = _validated_endpoint(
                raw.get("endpoint"), f"节点{node_id}服务"
            )
            if node_id in node_ids or endpoint in endpoints:
                raise DeploymentError(
                    "INVALID_TOPOLOGY_PROBE",
                    "节点列表存在重复",
                    "节点ID和服务地址必须唯一。",
                )
            label = str(raw.get("label") or node_id).strip()
            if not label or len(label) > 128:
                raise DeploymentError(
                    "INVALID_TOPOLOGY_PROBE",
                    "节点标签无效",
                    "节点标签长度必须为1到128个字符。",
                )
            node_ids.add(node_id)
            endpoints.add(endpoint)
            normalized.append(
                {"id": node_id, "role": role, "label": label, "endpoint": endpoint}
            )
        timeout = max(0.5, min(float(timeout_seconds), 5.0))
        with ThreadPoolExecutor(max_workers=min(8, len(normalized))) as pool:
            futures = {
                item["id"]: pool.submit(
                    self._probe_node, item["role"], item["endpoint"], timeout
                )
                for item in normalized
            }
            observed = {node_id: future.result() for node_id, future in futures.items()}
        public_nodes = {}
        labels = {item["id"]: item["label"] for item in normalized}
        for node_id, result in observed.items():
            public_nodes[node_id] = {"id": node_id, "label": labels[node_id], **result}
        online_count = sum(1 for item in public_nodes.values() if item["online"])
        return {
            "schema_version": "runtime-topology-connection-check/v2",
            "configured": True,
            "online": online_count == len(public_nodes),
            "partial_online": 0 < online_count < len(public_nodes),
            "node_count": len(public_nodes),
            "online_count": online_count,
            "observed_at": _utc_now(),
            "read_only": True,
            "probe_boundary": (
                "localhost demo only; HTTP service-root validation; no deployment mutation"
            ),
            "nodes": public_nodes,
        }

    def runtime_snapshot(
        self, *, edge_url: Any, cloud_url: Any, timeout_seconds: float = 4.0
    ) -> dict[str, Any]:
        """Return a bounded, read-only summary of the running cloud-edge system."""

        edge = _validated_endpoint(edge_url, "边缘服务")
        cloud = _validated_endpoint(cloud_url, "云端服务")
        timeout = max(0.5, min(float(timeout_seconds), 6.0))
        requests = {
            "edge_metrics": (edge, "/api/v1/framework/metrics"),
            "edge_outbox": (edge, "/api/v1/framework/outbox"),
            "edge_reviews": (edge, "/api/v1/collaboration/reviews"),
            "edge_release": (edge, "/api/v1/framework/edge-llm/release"),
            "cloud_metrics": (cloud, "/api/v1/framework/metrics"),
            "cloud_aggregations": (
                cloud,
                "/api/v1/collaboration/aggregations",
            ),
        }

        def fetch(base: str, path: str) -> dict[str, Any]:
            try:
                return {"ok": True, "value": self._request_json(base + path, timeout)}
            except Exception as exc:
                return {
                    "ok": False,
                    "error_code": type(exc).__name__,
                    "message": "接口暂时不可用",
                }

        with ThreadPoolExecutor(max_workers=len(requests)) as pool:
            futures = {
                key: pool.submit(fetch, base, path)
                for key, (base, path) in requests.items()
            }
            observed = {key: future.result() for key, future in futures.items()}

        def value(key: str) -> dict[str, Any]:
            item = observed[key]
            raw = item.get("value") if item.get("ok") is True else None
            return raw if isinstance(raw, dict) else {}

        edge_metrics = value("edge_metrics")
        edge_counters = edge_metrics.get("counters")
        edge_counters = edge_counters if isinstance(edge_counters, dict) else {}
        outbox = value("edge_outbox")
        reviews = value("edge_reviews")
        review_summary = reviews.get("summary")
        review_summary = review_summary if isinstance(review_summary, dict) else {}
        recent_raw = reviews.get("recent")
        recent_raw = recent_raw if isinstance(recent_raw, list) else []
        release = value("edge_release")
        cloud_metrics = value("cloud_metrics")
        cloud_counters = cloud_metrics.get("counters")
        cloud_counters = cloud_counters if isinstance(cloud_counters, dict) else {}
        cloud_distributions = cloud_metrics.get("distributions")
        cloud_distributions = (
            cloud_distributions if isinstance(cloud_distributions, dict) else {}
        )
        cloud_runtime = cloud_distributions.get("cloud_service_runtime_ms")
        cloud_runtime = cloud_runtime if isinstance(cloud_runtime, dict) else {}
        aggregations = value("cloud_aggregations")

        recent = []
        # Keep the monitoring snapshot bounded so the UI and tests share one
        # stable, read-only contract for "recent requests".
        for item in recent_raw[:8]:
            if not isinstance(item, dict):
                continue
            recent.append(
                {
                    "event_id": item.get("event_id"),
                    "scene": item.get("scene"),
                    "requested_route": item.get("requested_route"),
                    "state": item.get("state"),
                    "preliminary_latency_ms": item.get("preliminary_latency_ms"),
                    "planned_request_bytes": item.get("planned_request_bytes"),
                    "decision_changed": item.get("decision_changed"),
                    "completion_stage": item.get("completion_stage"),
                    "attempts": item.get("attempts"),
                    "last_error": str(item.get("last_error") or "")[:180] or None,
                }
            )

        outbox_states = outbox.get("states")
        outbox_states = outbox_states if isinstance(outbox_states, dict) else {}
        review_states = review_summary.get("states")
        review_states = review_states if isinstance(review_states, dict) else {}
        aggregation_states = aggregations.get("states")
        aggregation_states = (
            aggregation_states if isinstance(aggregation_states, dict) else {}
        )
        return {
            "schema_version": "runtime-monitor-snapshot/v1",
            "observed_at": _utc_now(),
            "read_only": True,
            "source_status": {
                key: {
                    "ok": item.get("ok") is True,
                    "error_code": item.get("error_code"),
                }
                for key, item in observed.items()
            },
            "edge": {
                "requests_total": edge_counters.get("edge_requests_total"),
                "local_autonomy_total": edge_counters.get("local_autonomy_total"),
                "cloud_delivery_successes_total": edge_counters.get(
                    "async_cloud_delivery_successes_total"
                ),
                "cloud_delivery_failures_total": edge_counters.get(
                    "async_cloud_delivery_failures_total"
                ),
                "outbox": {
                    "active": outbox.get("active"),
                    "states": outbox_states,
                    "oldest_active_age_ms": outbox.get("oldest_active_age_ms"),
                },
                "reviews": {
                    "total": review_summary.get("total"),
                    "states": review_states,
                    "final_business_completed": review_summary.get(
                        "final_business_completed",
                        review_summary.get("completed"),
                    ),
                },
                "release": {
                    "active_release_id": release.get("active_release_id"),
                    "applied_revision": release.get("applied_revision"),
                    "running": release.get("running"),
                    "last_error": release.get("last_error"),
                },
            },
            "cloud": {
                "requests_total": cloud_counters.get("cloud_requests_total"),
                "coordination_events_total": cloud_counters.get(
                    "coordination_events_total"
                ),
                "conflicts_initial_total": cloud_counters.get(
                    "coordination_conflicts_initial_total"
                ),
                "conflicts_residual_total": cloud_counters.get(
                    "coordination_conflicts_residual_total"
                ),
                "resolution_successes_total": cloud_counters.get(
                    "coordination_conflict_resolution_successes_total"
                ),
                "runtime_ms": {
                    "p50": cloud_runtime.get("p50"),
                    "p95": cloud_runtime.get("p95"),
                },
                "aggregations": {
                    "states": aggregation_states,
                    "event_count": aggregations.get("event_count"),
                    "retry_waiting": aggregations.get("retry_waiting"),
                },
            },
            "recent_events": recent,
        }

    def runtime_status(self, profile_id: str | None = None) -> dict[str, Any]:
        """Read current health without turning configured profiles into "online"."""

        if not self.profiles:
            if self.connection_defaults:
                result = self.probe_endpoints(
                    edge_url=self.connection_defaults["edge_url"],
                    cloud_url=self.connection_defaults["cloud_url"],
                )
                result["cloud_advertised_url"] = self.connection_defaults[
                    "cloud_advertised_url"
                ]
                return result
            return {
                "schema_version": "guided-runtime-status/v1",
                "configured": False,
                "online": False,
                "reason": "no_deployment_profile",
                "nodes": {},
            }
        if profile_id:
            profile = self.profiles.get(profile_id)
        else:
            profile = next((item for item in self.profiles.values() if item["enabled"]), None)
        if profile is None or profile["enabled"] is not True:
            raise DeploymentError(
                "PROFILE_NOT_FOUND", "找不到可用设备档案", "请刷新页面或联系管理员。"
            )
        selected_nodes = self._nodes_for_topology(profile, profile["default_topology"])

        def inspect(node: dict[str, Any]) -> dict[str, Any]:
            node_id = node["id"]
            try:
                health = self._request_json(node["service_url"] + "/health", timeout=2.0)
                return {
                    "online": True,
                    "id": node_id,
                    "role": node["role"],
                    "label": node["label"],
                    "health": health,
                }
            except Exception as exc:
                return {
                    "online": False,
                    "id": node_id,
                    "role": node["role"],
                    "label": node["label"],
                    "error_type": type(exc).__name__,
                }
        with ThreadPoolExecutor(max_workers=min(8, len(selected_nodes))) as pool:
            futures = {
                node["id"]: pool.submit(inspect, node) for node in selected_nodes
            }
            nodes = {node_id: future.result() for node_id, future in futures.items()}
        return {
            "schema_version": "guided-runtime-status/v1",
            "configured": True,
            "online": all(item["online"] for item in nodes.values()),
            "profile_id": profile["id"],
            "profile_label": profile["label"],
            "observed_at": _utc_now(),
            "nodes": nodes,
        }

    def create_session(self, request: dict[str, Any]) -> dict[str, Any]:
        scenario = str(request.get("scenario", "traffic"))
        template = str(request.get("template", "recommended"))
        profile_id = str(request.get("profile_id", ""))
        if scenario != "traffic":
            raise DeploymentError(
                "SCENARIO_NOT_READY", "当前只开放交通场景部署", "请先使用交通监控场景。"
            )
        if template != "recommended":
            raise DeploymentError(
                "TEMPLATE_NOT_READY", "当前只冻结了推荐部署模板", "请选择推荐部署。"
            )
        profile = self.profiles.get(profile_id)
        if profile is None or not profile["enabled"]:
            raise DeploymentError(
                "PROFILE_REQUIRED",
                "尚未配置可用的真实设备档案",
                "请让管理员在服务端配置SSH密钥与设备档案，然后刷新页面。",
            )
        requested_topology = request.get("topology")
        topology = _normalize_topology(
            profile,
            profile["default_topology"]
            if requested_topology is None
            else requested_topology,
            public_error=True,
        )
        session_id = secrets.token_hex(12)
        now = _utc_now()
        stages = [
            {
                "id": key,
                "label": label,
                "status": "completed" if key in {"select_scenario", "connect_nodes"} else "pending",
                "started_at": now if key in {"select_scenario", "connect_nodes"} else None,
                "ended_at": now if key in {"select_scenario", "connect_nodes"} else None,
                "result": (
                    {"scenario": scenario, "template": template}
                    if key == "select_scenario"
                    else {"profile_id": profile_id, "topology": topology}
                    if key == "connect_nodes"
                    else None
                ),
            }
            for key, label in STAGES
        ]
        session = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "status": "in_progress",
            "scenario": scenario,
            "template": template,
            "profile_id": profile_id,
            "profile_identity": (
                {
                    "bytes": self.profile_identity["bytes"],
                    "sha256": self.profile_identity["sha256"],
                }
                if self.profile_identity
                else None
            ),
            "profile_sha256": profile["profile_sha256"],
            "topology": topology,
            "topology_sha256": _canonical_sha256(topology),
            "stages": stages,
            "artifacts": {},
            "deployment": {
                "install_order": [],
                "installed_node_ids": [],
                "active_node_ids": [],
            },
        }
        self._save(session)
        return session

    def get_session(self, session_id: str) -> dict[str, Any]:
        if not _SAFE_ID.fullmatch(session_id):
            raise DeploymentError("INVALID_SESSION", "部署记录编号无效", "请重新进入部署向导。")
        path = self.state_root / session_id / "session.json"
        if not path.is_file():
            raise DeploymentError("SESSION_NOT_FOUND", "找不到部署记录", "请重新创建部署任务。")
        session = _read_json(path)
        if session.get("session_id") != session_id:
            raise DeploymentError("SESSION_MISMATCH", "部署记录身份不一致", "请停止操作并联系管理员。")
        return session

    def run_action(
        self, session_id: str, action: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        if action not in {"preflight", "generate_plan", "deploy", "verify", "rollback"}:
            raise DeploymentError("ACTION_NOT_ALLOWED", "不支持该部署操作", "请使用页面提供的按钮。")
        if action == "rollback":
            return self._rollback(session, request)
        expected = self._first_incomplete_stage(session)
        if expected != action:
            raise DeploymentError(
                "STAGE_ORDER",
                f"当前应先完成“{dict(STAGES).get(expected, expected)}”",
                "请按页面顺序继续。",
            )
        stage = self._stage(session, action)
        stage["status"] = "running"
        stage["started_at"] = _utc_now()
        self._save(session)
        try:
            if action == "preflight":
                result = self._preflight(session)
            elif action == "generate_plan":
                result = self._generate_plan(session)
            elif action == "deploy":
                result = self._deploy(session, request)
            else:
                result = self._verify(session)
        except DeploymentError as exc:
            if action == "verify":
                active_ids = list(
                    session.get("deployment", {}).get("active_node_ids", [])
                )
                if active_ids:
                    try:
                        profile = self._profile(session)
                    except DeploymentError:
                        profile = None
                    if profile is not None:
                        self._compensate_started_nodes(
                            session, profile, active_ids
                        )
            stage["status"] = "failed"
            stage["ended_at"] = _utc_now()
            stage["error"] = exc.as_dict()
            session["status"] = "failed"
            self._save(session)
            raise
        stage["status"] = "completed"
        stage["ended_at"] = _utc_now()
        stage["result"] = result
        stage.pop("error", None)
        session["status"] = "completed" if action == "verify" else "in_progress"
        self._save(session)
        return session

    def _profile(self, session: dict[str, Any]) -> dict[str, Any]:
        profile = self.profiles.get(str(session["profile_id"]))
        if (
            profile is None
            or _profile_sha256(profile) != profile.get("profile_sha256")
            or profile.get("profile_sha256") != session.get("profile_sha256")
        ):
            raise DeploymentError(
                "PROFILE_DRIFT", "设备档案已变化", "请停止部署并重新创建部署任务。"
            )
        topology = session.get("topology")
        if not isinstance(topology, dict) or _canonical_sha256(topology) != session.get(
            "topology_sha256"
        ):
            raise DeploymentError(
                "TOPOLOGY_DRIFT", "部署拓扑记录不一致", "请停止部署并重新创建部署任务。"
            )
        return profile

    @staticmethod
    def _cloud_order(topology: dict[str, Any]) -> list[str]:
        primary = topology["primary_cloud_id"]
        return [primary] + sorted(
            node_id
            for node_id in topology["cloud_node_ids"]
            if node_id != primary
        )

    @staticmethod
    def _nodes_for_topology(
        profile: dict[str, Any], topology: dict[str, Any]
    ) -> list[dict[str, Any]]:
        node_by_id = profile["node_by_id"]
        cloud_nodes = [
            node_by_id[node_id]
            for node_id in GuidedDeploymentController._cloud_order(topology)
        ]
        edge_nodes = [
            node_by_id[binding["edge_id"]] for binding in topology["edge_bindings"]
        ]
        return cloud_nodes + edge_nodes

    @staticmethod
    def _binding_by_edge(topology: dict[str, Any]) -> dict[str, str]:
        return {
            binding["edge_id"]: binding["cloud_id"]
            for binding in topology["edge_bindings"]
        }

    @staticmethod
    def _stage(session: dict[str, Any], stage_id: str) -> dict[str, Any]:
        return next(item for item in session["stages"] if item["id"] == stage_id)

    @staticmethod
    def _first_incomplete_stage(session: dict[str, Any]) -> str | None:
        for stage in session["stages"]:
            if stage["status"] != "completed":
                return str(stage["id"])
        return None

    def _save(self, session: dict[str, Any]) -> None:
        session["updated_at"] = _utc_now()
        _write_json(self.state_root / str(session["session_id"]) / "session.json", session)

    @staticmethod
    def _ssh_prefix(node: dict[str, Any]) -> list[str]:
        return [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ControlMaster=no",
            "-o",
            "ControlPersist=no",
            "-o",
            "ControlPath=none",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=8",
            "-i",
            node["identity_file"],
            "-p",
            str(node["ssh_port"]),
            f"{node['user']}@{node['host']}",
        ]

    def _remote_python(
        self, node: dict[str, Any], source: str, *, timeout: float = 30.0
    ) -> dict[str, Any]:
        command = self._ssh_prefix(node) + ["python3", "-"]
        try:
            completed = self.runner(
                command,
                input=source,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DeploymentError(
                "SSH_COMMAND_FAILED",
                f"无法连接设备“{node['label']}”",
                "请确认设备在线、SSH密钥有效且远端命令能在限定时间内完成。",
            ) from exc
        if completed.returncode != 0:
            raise DeploymentError(
                "SSH_COMMAND_FAILED",
                f"无法检查设备“{node['label']}”",
                "请确认设备在线、SSH密钥有效且Python 3可用。技术详情中保留了原始错误。",
            )
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise DeploymentError(
                "REMOTE_OUTPUT_INVALID", "设备返回了无法识别的检查结果", "请核对远端Python环境。"
            ) from exc
        if not isinstance(value, dict):
            raise DeploymentError("REMOTE_OUTPUT_INVALID", "设备检查结果格式错误", "请核对远端环境。")
        value["technical_stderr"] = completed.stderr[-4000:]
        return value

    def _preflight(self, session: dict[str, Any]) -> dict[str, Any]:
        profile = self._profile(session)
        topology = session["topology"]
        results: dict[str, Any] = {}
        for node in self._nodes_for_topology(profile, topology):
            role = node["role"]
            paths = [node["sdk_root"]]
            if role == "edge":
                paths.append(node["llama_binary"])
            source = (
                "import json,os,platform,shutil,socket\n"
                f"paths={json.dumps(paths)}\n"
                "disk=shutil.disk_usage(paths[0]) if os.path.exists(paths[0]) else None\n"
                "print(json.dumps({'ok':all(os.path.exists(p) for p in paths),"
                "'system':platform.system(),'architecture':platform.machine(),"
                "'python':platform.python_version(),'paths':{p:os.path.exists(p) for p in paths},"
                "'memory_bytes':int(os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')),"
                "'disk_free_bytes':disk.free if disk else 0,"
                "'cuda_visible':os.path.exists('/usr/local/cuda') or os.path.exists('/dev/nvhost-gpu'),"
                "'hostname':socket.gethostname()},sort_keys=True))\n"
            )
            result = self._remote_python(node, source)
            if result.get("ok") is not True:
                raise DeploymentError(
                    "ASSET_MISSING", f"{node['label']}缺少运行目录或程序", "请由管理员补齐预装资产后重试。"
                )
            results[node["id"]] = {"role": role, **result}
        return {
            "status": "passed",
            "node_count": len(results),
            "nodes": results,
        }

    def _generate_plan(self, session: dict[str, Any]) -> dict[str, Any]:
        profile = self._profile(session)
        topology = session["topology"]
        node_by_id = profile["node_by_id"]
        nodes = self._nodes_for_topology(profile, topology)
        plan = {
            "schema_version": "guided-deployment-plan/v2",
            "scene": "交通监控",
            "deployment_template": TEMPLATES[session["template"]],
            "cloud_model": "从云端当前服务配置读取",
            "edge_model": "从边缘当前发布清单读取",
            "traffic_capability": "启用",
            "deployment_mode": "在线协同",
            "profile_id": profile["id"],
            "profile_sha256": session["profile_sha256"],
            "topology_sha256": session["topology_sha256"],
            "primary_cloud_id": topology["primary_cloud_id"],
            "nodes": [
                {
                    "id": node["id"],
                    "role": node["role"],
                    "label": node["label"],
                    "service_url": node["service_url"],
                    "service_unit": "cloud-edge-traffic-{}".format(node["role"]),
                }
                for node in nodes
            ],
            "edge_bindings": [
                {
                    **binding,
                    "cloud_service_url": node_by_id[binding["cloud_id"]][
                        "service_url"
                    ],
                }
                for binding in topology["edge_bindings"]
            ],
            "dag": [
                {
                    "layer": 0,
                    "role": "cloud",
                    "node_ids": self._cloud_order(topology),
                },
                {
                    "layer": 1,
                    "role": "edge",
                    "depends_on_layer": 0,
                    "node_ids": [
                        binding["edge_id"] for binding in topology["edge_bindings"]
                    ],
                },
            ],
            "apply_boundary": (
                "server-authorized pre-provisioned nodes only; clouds before edges"
            ),
        }
        session_dir = self.state_root / str(session["session_id"])
        plan_path = session_dir / "deployment_plan.json"
        _write_json(plan_path, plan)
        session["artifacts"]["deployment_plan"] = str(plan_path)
        return plan

    def _remote_install(
        self, node: dict[str, Any], *, cloud_url: str | None
    ) -> dict[str, Any]:
        role = node["role"]
        environment = {}
        if role == "edge":
            environment = {
                "CLOUD_URL": str(cloud_url or ""),
                "LLAMA_SERVER_PATH": node["llama_binary"],
            }
        source = (
            "import json,os,subprocess\n"
            f"root={json.dumps(node['sdk_root'])}\n"
            f"role={json.dumps(role)}\n"
            f"extra={json.dumps(environment)}\n"
            "env=dict(os.environ);env.update(extra)\n"
            "cmd=['bash',os.path.join(root,'scripts','install_traffic_systemd.sh'),role]\n"
            "p=subprocess.run(cmd,cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)\n"
            "print(json.dumps({'returncode':p.returncode,'stdout':p.stdout[-8000:],'stderr':p.stderr[-8000:]},sort_keys=True))\n"
        )
        result = self._remote_python(node, source, timeout=360.0)
        if int(result.get("returncode", -1)) != 0:
            raise DeploymentError(
                "INSTALL_FAILED", f"{node['label']}安装未完成", "请展开技术详情检查权限或服务配置后重试。"
            )
        return result

    def _remote_stop(self, node: dict[str, Any]) -> dict[str, Any]:
        source = (
            "import json,subprocess\n"
            f"unit={json.dumps('cloud-edge-traffic-' + node['role'])}\n"
            "p=subprocess.run(['sudo','systemctl','disable','--now',unit],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)\n"
            "print(json.dumps({'returncode':p.returncode,'stdout':p.stdout[-4000:],'stderr':p.stderr[-4000:]},sort_keys=True))\n"
        )
        result = self._remote_python(node, source, timeout=60.0)
        if int(result.get("returncode", -1)) != 0:
            raise DeploymentError(
                "ROLLBACK_FAILED",
                f"{node['label']}回滚失败",
                "请查看技术详情并人工处理。",
            )
        return result

    def _compensate_started_nodes(
        self,
        session: dict[str, Any],
        profile: dict[str, Any],
        started_node_ids: list[str],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        failures: dict[str, Any] = {}
        active = list(session["deployment"].get("active_node_ids", []))
        for node_id in reversed(started_node_ids):
            node = profile["node_by_id"][node_id]
            try:
                results[node_id] = self._remote_stop(node)
                if node_id in active:
                    active.remove(node_id)
            except Exception as exc:
                failures[node_id] = (
                    exc.as_dict()
                    if isinstance(exc, DeploymentError)
                    else {
                        "code": type(exc).__name__,
                        "message": "补偿操作未完成",
                        "suggestion": "请查看技术日志并人工核对该节点。",
                    }
                )
        session["deployment"]["active_node_ids"] = active
        compensation = {
            "attempted_node_ids": list(reversed(started_node_ids)),
            "completed_node_ids": list(results),
            "failures": failures,
            "completed_at": _utc_now(),
        }
        session["deployment"]["compensation"] = compensation
        self._save(session)
        return compensation

    def _deploy(self, session: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        profile = self._profile(session)
        if profile["allow_deploy"] is not True:
            raise DeploymentError(
                "DEPLOY_NOT_AUTHORIZED", "该设备档案只允许检查", "请由管理员显式开放部署权限。"
            )
        if request.get("confirmation") != "START_TRAFFIC_DEPLOYMENT":
            raise DeploymentError(
                "CONFIRMATION_REQUIRED", "尚未确认开始真实部署", "请勾选确认后重新点击开始部署。"
            )
        topology = session["topology"]
        node_by_id = profile["node_by_id"]
        binding_by_edge = self._binding_by_edge(topology)
        order = self._cloud_order(topology) + [
            binding["edge_id"] for binding in topology["edge_bindings"]
        ]
        completed: dict[str, Any] = {}
        started: list[str] = []
        session["deployment"]["apply_started_at"] = _utc_now()
        session["deployment"]["planned_node_ids"] = order
        self._save(session)
        try:
            for node_id in order:
                node = node_by_id[node_id]
                cloud_url = None
                if node["role"] == "edge":
                    cloud_url = node_by_id[binding_by_edge[node_id]]["service_url"]
                completed[node_id] = {
                    "role": node["role"],
                    "cloud_id": binding_by_edge.get(node_id),
                    "result": self._remote_install(node, cloud_url=cloud_url),
                }
                started.append(node_id)
                session["deployment"]["install_order"] = list(started)
                session["deployment"]["installed_node_ids"] = list(started)
                session["deployment"]["active_node_ids"] = list(started)
                self._save(session)
        except Exception as exc:
            self._compensate_started_nodes(session, profile, started)
            if isinstance(exc, DeploymentError):
                raise
            raise DeploymentError(
                "INSTALL_FAILED",
                "节点安装过程中出现未预期错误",
                "已对本次任务成功启动的节点执行逆序补偿，请检查技术日志。",
            ) from exc
        session["deployment"]["apply_completed_at"] = _utc_now()
        self._save(session)
        return {
            "status": "services_started",
            "primary_cloud_id": topology["primary_cloud_id"],
            "install_order": order,
            "nodes": completed,
        }

    @staticmethod
    def _request_json(url: str, timeout: float = 8.0) -> dict[str, Any]:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            body = response.read(1024 * 1024)
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("health response must be an object")
        return value

    def _verify(self, session: dict[str, Any]) -> dict[str, Any]:
        profile = self._profile(session)
        topology = session["topology"]
        selected_nodes = self._nodes_for_topology(profile, topology)
        selected_ids = [node["id"] for node in selected_nodes]
        active_ids = list(session.get("deployment", {}).get("active_node_ids", []))
        if set(active_ids) != set(selected_ids):
            raise DeploymentError(
                "DEPLOYMENT_INCOMPLETE",
                "部署节点尚未全部启动",
                "请先完成部署，或检查失败后的自动补偿记录。",
            )
        health_by_node: dict[str, Any] = {}
        for node in selected_nodes:
            try:
                health_by_node[node["id"]] = self._request_json(
                    node["service_url"] + "/ready"
                )
            except Exception as exc:
                raise DeploymentError(
                    "HEALTH_CHECK_FAILED",
                    f"{node['label']}已启动，但健康检查未通过",
                    "请检查服务日志和云边网络后重试。",
                ) from exc

        smoke_by_edge: dict[str, Any] = {}
        for binding in topology["edge_bindings"]:
            node = profile["node_by_id"][binding["edge_id"]]
            output = "/tmp/cloud-edge-guided-{}-{}.json".format(
                session["session_id"], node["id"]
            )
            source = (
                "import json,os,subprocess\n"
                f"root={json.dumps(node['sdk_root'])}\n"
                f"edge_url={json.dumps(node['service_url'])}\n"
                f"output={json.dumps(output)}\n"
                "cmd=['python3',os.path.join(root,'scenes/freeway_traffic/run_partitioned_current_state_edges.py'),"
                "'--project-root',root,'--edge-url',edge_url,'--split','test','--sample-start','100',"
                "'--sample-stop','101','--top-k','10','--output',output]\n"
                "p=subprocess.run(cmd,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)\n"
                "value=json.load(open(output,encoding='utf-8')) if p.returncode==0 and os.path.isfile(output) else None\n"
                "print(json.dumps({'returncode':p.returncode,'sample':value,'stdout':p.stdout[-4000:],'stderr':p.stderr[-4000:]},sort_keys=True))\n"
            )
            smoke = self._remote_python(node, source, timeout=210.0)
            if int(smoke.get("returncode", -1)) != 0 or not isinstance(
                smoke.get("sample"), dict
            ):
                raise DeploymentError(
                    "SAMPLE_VERIFICATION_FAILED",
                    f"{node['label']}健康检查通过，但真实交通样例未完成",
                    "部署不能判定成功；请检查场景数据、模型和云端聚合日志。",
                )
            smoke_by_edge[node["id"]] = smoke["sample"]
        result = {
            "status": "passed",
            "primary_cloud_id": topology["primary_cloud_id"],
            "node_health": health_by_node,
            "traffic_samples": smoke_by_edge,
            "verified_at": _utc_now(),
        }
        # Keep the original single-cloud/single-edge result aliases so v1
        # consumers do not need a flag day migration.
        if len(topology["cloud_node_ids"]) == 1 and len(smoke_by_edge) == 1:
            cloud_id = topology["cloud_node_ids"][0]
            edge_id = topology["edge_bindings"][0]["edge_id"]
            result.update(
                {
                    "cloud_health": health_by_node[cloud_id],
                    "edge_health": health_by_node[edge_id],
                    "traffic_sample": smoke_by_edge[edge_id],
                }
            )
        self._write_delivery_artifacts(session, profile, result)
        return result

    def _write_delivery_artifacts(
        self, session: dict[str, Any], profile: dict[str, Any], verification: dict[str, Any]
    ) -> None:
        root = self.state_root / str(session["session_id"]) / "deliverables"
        selected_nodes = self._nodes_for_topology(profile, session["topology"])
        topology = {
            **session["topology"],
            "nodes": [
                {
                    "id": node["id"],
                    "role": node["role"],
                    "label": node["label"],
                    "endpoint": node["service_url"],
                }
                for node in selected_nodes
            ],
            "scene": "traffic",
        }
        manifest = {
            "session_id": session["session_id"],
            "scenario": session["scenario"],
            "template": session["template"],
            "profile_identity": session["profile_identity"],
            "profile_sha256": session["profile_sha256"],
            "topology_sha256": session["topology_sha256"],
            "verified_at": verification["verified_at"],
        }
        active_model = {
            "source": "current service health responses",
            "nodes": verification["node_health"],
        }
        files = {
            "deployment_manifest.json": manifest,
            "active_model_manifest.json": active_model,
            "health_check_report.json": verification,
            "system_topology.json": topology,
        }
        for name, value in files.items():
            path = root / name
            _write_json(path, value)
            session["artifacts"][name] = str(path)
        report = (
            "<!doctype html><meta charset='utf-8'><title>部署报告</title>"
            "<style>body{font:16px system-ui;max-width:820px;margin:60px auto;color:#1f2328}"
            "h1{font-size:32px}pre{background:#f6f8fa;padding:18px;border-radius:8px;white-space:pre-wrap}</style>"
            f"<h1>云边协同部署报告</h1><p>部署记录：{html.escape(str(session['session_id']))}</p>"
            "<p>交通真实样例验证通过，系统方可判定为部署完成。</p>"
            f"<pre>{html.escape(json.dumps(verification, ensure_ascii=False, indent=2))}</pre>"
        )
        report_path = root / "deployment_report.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        session["artifacts"]["deployment_report.html"] = str(report_path)
        rollback_path = root / "rollback.sh"
        rollback_lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
        for node_id in reversed(session["deployment"]["active_node_ids"]):
            node = profile["node_by_id"][node_id]
            remote = "sudo systemctl disable --now cloud-edge-traffic-{}".format(
                node["role"]
            )
            rollback_lines.append(
                " ".join(shlex.quote(item) for item in self._ssh_prefix(node) + [remote])
            )
        rollback_path.write_text("\n".join(rollback_lines) + "\n", encoding="utf-8")
        rollback_path.chmod(0o700)
        session["artifacts"]["rollback.sh"] = str(rollback_path)

    def _rollback(self, session: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        if request.get("confirmation") != "ROLLBACK_TRAFFIC_DEPLOYMENT":
            raise DeploymentError("CONFIRMATION_REQUIRED", "尚未确认回滚", "请确认后再执行回滚。")
        profile = self._profile(session)
        deployment = session.get("deployment")
        deployment = deployment if isinstance(deployment, dict) else {}
        active_ids = list(deployment.get("active_node_ids", []))
        installed_ids = set(deployment.get("installed_node_ids", []))
        if not active_ids:
            raise DeploymentError(
                "NOTHING_TO_ROLLBACK",
                "本次任务没有可回滚的活动节点",
                "系统不会停止未由本次任务成功安装的服务。",
            )
        if any(node_id not in installed_ids for node_id in active_ids):
            raise DeploymentError(
                "ROLLBACK_SCOPE_INVALID",
                "回滚范围校验失败",
                "系统拒绝停止未由本次任务成功安装的节点。",
            )
        results: dict[str, Any] = {}
        rolled_back: list[str] = []
        for node_id in reversed(active_ids):
            node = profile["node_by_id"].get(node_id)
            if node is None:
                raise DeploymentError(
                    "ROLLBACK_SCOPE_INVALID",
                    "回滚节点已不在授权档案中",
                    "请停止操作并由管理员核对部署记录。",
                )
            results[node_id] = {"role": node["role"], "result": self._remote_stop(node)}
            rolled_back.append(node_id)
            deployment["active_node_ids"] = [
                item for item in active_ids if item not in rolled_back
            ]
            self._save(session)
        session["status"] = "rolled_back"
        session["rollback"] = {
            "completed_at": _utc_now(),
            "rollback_order": rolled_back,
            "nodes": results,
        }
        self._save(session)
        return session


__all__ = [
    "DeploymentError",
    "GuidedDeploymentController",
    "PROFILE_SCHEMA_VERSION",
    "PROFILE_SCHEMA_VERSION_V2",
    "SESSION_SCHEMA_VERSION",
]
