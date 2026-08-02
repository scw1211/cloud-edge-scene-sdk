"""用途：一键启动真实交通云边栈，运行常驻 ASTGCN 样本并产出闭环验收证据。"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid

import numpy as np

from traffic_system.benchmark_utils import (
    SystemMemorySampler,
    read_pss_kb,
    read_rss_kb,
    summarize,
)
from traffic_system.generate_joint_edge_events import parse_sample_spec
from traffic_system.scene_event import traffic_event_from_output
from traffic_system.traffic_perception_runtime import JointTrafficPerceptionRuntime


DECIDE_PATH = "/api/v1/collaboration/decide"
METRICS_PATH = "/api/v1/framework/metrics"
NON_AUTHORITATIVE_REVIEW_STAGES = {"partial_final", "local_only_timeout"}


def _json_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(dict(value), file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    os.replace(str(temporary), str(path))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_port(value: int) -> int:
    if value < 0 or value > 65535:
        raise ValueError("port must be zero or within [1, 65535]")
    return _free_port() if value == 0 else value


def _find_llama_server_pid(endpoint: str) -> int:
    parsed = urlsplit(endpoint)
    port = parsed.port
    if port is None:
        return 0
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            command = (process_dir / "cmdline").read_text(
                encoding="utf-8", errors="ignore"
            ).split("\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        parts = [value for value in command if value]
        if not parts or Path(parts[0]).name != "llama-server":
            continue
        for index, value in enumerate(parts):
            if value == "--port" and index + 1 < len(parts) and parts[index + 1] == str(port):
                return int(process_dir.name)
            if value == "--port={}".format(port):
                return int(process_dir.name)
    return 0


def _request_json(
    base_url: str,
    path: str,
    payload: Optional[Mapping[str, Any]] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout_seconds: float = 2.0,
) -> Dict[str, Any]:
    body = None
    method = "GET"
    request_headers = {"Accept": "application/json", **dict(headers or {})}
    if payload is not None:
        body = json.dumps(
            dict(payload), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
        method = "POST"
    request = Request(
        base_url.rstrip("/") + path,
        data=body,
        headers=request_headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP {} {}: {}".format(exc.code, path, detail)) from exc
    if not isinstance(value, dict):
        raise ValueError("{} returned a non-object JSON value".format(path))
    return value


def _wait_for(
    description: str,
    operation,
    predicate,
    timeout_seconds: float,
) -> Tuple[Dict[str, Any], float]:
    started = time.perf_counter()
    deadline = time.monotonic() + timeout_seconds
    last_value: Optional[Dict[str, Any]] = None
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            last_value = operation()
            if predicate(last_value):
                return last_value, round((time.perf_counter() - started) * 1000.0, 6)
        except (OSError, ValueError, RuntimeError, URLError) as exc:
            last_error = exc
        time.sleep(0.05)
    raise TimeoutError(
        "timed out waiting for {}: value={!r}, error={!r}".format(
            description, last_value, last_error
        )
    )


def _descendant_pids(root_pid: int) -> List[int]:
    pending = [int(root_pid)]
    discovered: List[int] = []
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if not (Path("/proc") / str(pid)).exists():
            continue
        discovered.append(pid)
        children_path = Path("/proc") / str(pid) / "task" / str(pid) / "children"
        try:
            children = children_path.read_text(
                encoding="utf-8", errors="ignore"
            ).strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            children = ""
        if children:
            pending.extend(int(value) for value in children.split())
    return discovered


class StackMemorySampler:
    """Measure disjoint perception, Edge-Qwen, edge service and cloud processes."""

    def __init__(
        self,
        perception_pid: int,
        edge_qwen_pid: int,
        edge_service_pid: int,
        cloud_service_pid: int,
        interval_seconds: float = 0.02,
    ) -> None:
        self.roots = {
            "perception": int(perception_pid),
            "edge_qwen": int(edge_qwen_pid),
            "edge_service": int(edge_service_pid),
            "cloud_service": int(cloud_service_pid),
        }
        self.interval_seconds = max(0.01, float(interval_seconds))
        self.rss_samples: Dict[str, List[float]] = {
            name: [] for name in self.roots
        }
        self.pss_samples: Dict[str, List[float]] = {
            name: [] for name in self.roots
        }
        self.edge_stack_rss: List[float] = []
        self.edge_stack_pss: List[float] = []
        self.full_stack_rss: List[float] = []
        self.full_stack_pss: List[float] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _component_memory(self, name: str, proportional: bool) -> float:
        root = self.roots[name]
        pids = [root] if name == "perception" else _descendant_pids(root)
        reader = read_pss_kb if proportional else read_rss_kb
        return sum(reader(pid) for pid in pids) / 1024.0

    def _sample(self, include_pss: bool) -> None:
        rss = {
            name: self._component_memory(name, False) for name in self.roots
        }
        for name, value in rss.items():
            self.rss_samples[name].append(value)
        edge_rss = rss["perception"] + rss["edge_qwen"] + rss["edge_service"]
        self.edge_stack_rss.append(edge_rss)
        self.full_stack_rss.append(edge_rss + rss["cloud_service"])
        if not include_pss:
            return
        pss = {
            name: self._component_memory(name, True) for name in self.roots
        }
        for name, value in pss.items():
            self.pss_samples[name].append(value)
        edge_pss = pss["perception"] + pss["edge_qwen"] + pss["edge_service"]
        self.edge_stack_pss.append(edge_pss)
        self.full_stack_pss.append(edge_pss + pss["cloud_service"])

    def _run(self) -> None:
        next_pss = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            include_pss = now >= next_pss
            self._sample(include_pss)
            if include_pss:
                next_pss = now + 1.0
            self._stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        self._sample(True)
        self._thread.start()

    def stop(self) -> None:
        self._sample(True)
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def result(self) -> Dict[str, Any]:
        def peak(values: Iterable[float]) -> float:
            data = list(values)
            return round(max(data), 4) if data else 0.0

        return {
            "component_peak_rss_mb": {
                name: peak(values) for name, values in self.rss_samples.items()
            },
            "component_peak_pss_mb": {
                name: peak(values) for name, values in self.pss_samples.items()
            },
            "edge_stack_peak_rss_mb": peak(self.edge_stack_rss),
            "edge_stack_peak_pss_mb": peak(self.edge_stack_pss),
            "full_edge_cloud_stack_peak_rss_mb": peak(self.full_stack_rss),
            "full_edge_cloud_stack_peak_pss_mb": peak(self.full_stack_pss),
            "scope": (
                "edge stack = resident ASTGCN process + Edge-Qwen supervisor/llama-server "
                "+ edge HTTP service; full stack additionally includes the cloud HTTP service"
            ),
            "pss_note": "PSS apportions shared memory and is the preferred multi-process RAM estimate.",
        }


class ManagedTrafficStack:
    def __init__(self, project_root: Path, run_dir: Path, args: argparse.Namespace) -> None:
        self.project_root = project_root
        self.run_dir = run_dir
        self.args = args
        self.manage_cloud = not bool(str(args.external_cloud_url).strip())
        self.manage_edge_qwen = not bool(str(args.external_edge_llm_url).strip())
        self.external_edge_qwen_pid = 0
        self.cloud_port = _resolve_port(args.cloud_port) if self.manage_cloud else 0
        self.edge_port = _resolve_port(args.edge_port)
        self.llama_port = _resolve_port(args.llama_port) if self.manage_edge_qwen else 0
        ports = {self.edge_port}
        if self.manage_edge_qwen:
            ports.add(self.llama_port)
        if self.manage_cloud:
            ports.add(self.cloud_port)
        expected_port_count = 1 + int(self.manage_edge_qwen) + int(self.manage_cloud)
        if len(ports) != expected_port_count:
            raise ValueError("cloud, edge and llama ports must differ")
        if self.manage_cloud:
            self.cloud_url = "http://127.0.0.1:{}".format(self.cloud_port)
        else:
            self.cloud_url = str(args.external_cloud_url).rstrip("/")
            if not self.cloud_url.startswith(("http://", "https://")):
                raise ValueError("external-cloud-url must use http or https")
        self.edge_url = "http://127.0.0.1:{}".format(self.edge_port)
        if self.manage_edge_qwen:
            self.llama_url = "http://127.0.0.1:{}".format(self.llama_port)
        else:
            self.llama_url = str(args.external_edge_llm_url).rstrip("/")
            if not self.llama_url.startswith(("http://", "https://")):
                raise ValueError("external-edge-llm-url must use http or https")
            self.external_edge_qwen_pid = _find_llama_server_pid(self.llama_url)
            if self.external_edge_qwen_pid <= 0:
                raise RuntimeError("cannot identify external llama-server process")
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_handles: Dict[str, Any] = {}
        self.startup_ms: Dict[str, float] = {}
        self.config_paths = self._write_configs()

    def _write_configs(self) -> Dict[str, Path]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        release_registry = Path(self.args.release_registry).resolve()
        runtime_source = Path(self.args.edge_llm_runtime_config).resolve()
        edge_plugin_source = Path(self.args.edge_plugin_config).resolve()
        cloud_plugin_source = Path(self.args.cloud_plugin_config).resolve()
        for path in (release_registry, runtime_source, edge_plugin_source):
            if not path.is_file():
                raise FileNotFoundError(path)
        if self.manage_cloud and not cloud_plugin_source.is_file():
            raise FileNotFoundError(cloud_plugin_source)

        runtime_config = _json_object(runtime_source)
        runtime_config["endpoint"] = self.llama_url
        runtime_path = self.run_dir / "edge_llm_runtime.json"
        _write_json(runtime_path, runtime_config)

        edge_plugins = _json_object(edge_plugin_source)
        traffic_plugins = [
            plugin
            for plugin in edge_plugins.get("plugins", [])
            if isinstance(plugin, dict)
            and str(plugin.get("spec", "")).endswith(":TrafficPlugin")
        ]
        if len(traffic_plugins) != 1:
            raise ValueError("edge plugin config must enable exactly one TrafficPlugin")
        options = traffic_plugins[0].setdefault("options", {})
        options["edge_llm_release_registry_path"] = str(release_registry)
        options["edge_llm_runtime_config_path"] = str(runtime_path)
        if self.args.edge_llm_mode:
            options["edge_llm_mode"] = self.args.edge_llm_mode
        edge_plugins_path = self.run_dir / "scene_plugins_edge.json"
        _write_json(edge_plugins_path, edge_plugins)

        cloud_config = {
            "schema_version": 1,
            "role": "cloud",
            "plugin_config": str(cloud_plugin_source),
            "listen": {
                "host": "127.0.0.1",
                "port": self.cloud_port,
                "max_body_bytes": 8 * 1024 * 1024,
                "access_log": False,
            },
            "storage": {
                "feedback": str(self.run_dir / "cloud_feedback.jsonl"),
                "idempotency": str(self.run_dir / "cloud_idempotency.sqlite3"),
                "artifacts": str(self.run_dir / "cloud_artifacts"),
                "aggregations": str(
                    self.run_dir / "cloud_aggregations.sqlite3"
                ),
            },
            "idempotency": {"ttl_seconds": 3600, "max_entries": 100000},
        }
        if str(self.args.cloud_llm_runtime_config).strip():
            cloud_config["cloud_llm"] = {
                "enabled": True,
                "runtime_config": str(
                    Path(self.args.cloud_llm_runtime_config).resolve()
                ),
                "min_risk_level": str(self.args.cloud_llm_min_risk_level),
            }
        cloud_config_path = self.run_dir / "cloud_service.json"
        _write_json(cloud_config_path, cloud_config)

        edge_config = {
            "schema_version": 1,
            "role": "edge",
            "plugin_config": str(edge_plugins_path),
            "listen": {
                "host": "127.0.0.1",
                "port": self.edge_port,
                "max_body_bytes": 8 * 1024 * 1024,
                "access_log": False,
            },
            "storage": {
                "outbox": str(self.run_dir / "edge_outbox.sqlite3"),
                "performance_profiles": str(
                    self.run_dir / "edge_performance_profiles.json"
                ),
                "feedback": str(self.run_dir / "edge_feedback.jsonl"),
                "idempotency": str(self.run_dir / "edge_idempotency.sqlite3"),
                "reviews": str(self.run_dir / "edge_reviews.sqlite3"),
                "monitoring": str(self.run_dir / "edge_monitoring.sqlite3"),
            },
            "scheduler": {"confidence_threshold": 0.75, "jitter_guard": 1.645},
            "cloud": {
                "base_url": self.cloud_url,
                "timeout_seconds": self.args.cloud_timeout_seconds,
                "max_attempts": 2,
                "retry_backoff_seconds": 0.025,
            },
            "release_watch": {
                "enabled": True,
                "registry": str(release_registry),
                "interval_seconds": 2.0,
            },
            "network_probe": {
                "interval_seconds": 0.1,
                "window_size": 10,
                "failure_threshold": 2,
                "uplink_mbps": 100.0,
                "downlink_mbps": 100.0,
                "expected_response_bytes": 2048,
                "cloud_queue_ms": 1.0,
                "cloud_compute_ms": 12.0,
            },
            "replay": {
                "interval_seconds": 0.2,
                "batch_size": 64,
                "lease_seconds": 30.0,
                "max_backoff_seconds": 60.0,
            },
            "idempotency": {"ttl_seconds": 3600, "max_entries": 100000},
        }
        edge_config_path = self.run_dir / "edge_service.json"
        _write_json(edge_config_path, edge_config)
        return {
            "runtime": runtime_path,
            "edge_plugins": edge_plugins_path,
            "cloud": cloud_config_path,
            "edge": edge_config_path,
        }

    def _start_process(self, name: str, command: Sequence[str]) -> subprocess.Popen:
        log_path = self.run_dir / "{}.log".format(name)
        handle = log_path.open("w", encoding="utf-8")
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        source_roots = [
            str(self.project_root),
            str(self.project_root / "scenes" / "freeway_traffic"),
        ]
        inherited_pythonpath = environment.get("PYTHONPATH", "").strip()
        if inherited_pythonpath:
            source_roots.append(inherited_pythonpath)
        environment["PYTHONPATH"] = os.pathsep.join(source_roots)
        process = subprocess.Popen(
            list(command),
            cwd=str(self.project_root),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        self.processes[name] = process
        self.log_handles[name] = handle
        return process

    def start(self) -> Dict[str, Any]:
        supervisor = (
            self._start_process(
            "edge_qwen",
            [
                sys.executable,
                "-m",
                "edge_llm_factory",
                "serve-release",
                "--registry",
                str(Path(self.args.release_registry).resolve()),
                "--runtime-config",
                str(self.config_paths["runtime"]),
                "--binary",
                str(Path(self.args.llama_binary).resolve()),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.llama_port),
                "--context-tokens",
                str(self.args.llama_context_tokens),
                "--threads",
                str(self.args.llama_threads),
                "--gpu-layers",
                str(self.args.llama_gpu_layers),
                "--startup-timeout-seconds",
                str(self.args.startup_timeout_seconds),
            ],
            )
            if self.manage_edge_qwen
            else None
        )
        _, self.startup_ms["edge_qwen"] = _wait_for(
            "Edge-Qwen",
            lambda: _request_json(self.llama_url, "/health", timeout_seconds=1.0),
            lambda value: value.get("status") == "ok",
            self.args.startup_timeout_seconds,
        )
        if supervisor is not None and supervisor.poll() is not None:
            raise RuntimeError("Edge-Qwen supervisor exited after startup")

        if self.manage_cloud:
            cloud = self._start_process(
                "cloud_service",
                [
                    sys.executable,
                    "-m",
                    "cloud_edge_framework.cloud_service",
                    "--config",
                    str(self.config_paths["cloud"]),
                    "--project_root",
                    str(self.project_root),
                ],
            )
            _, self.startup_ms["cloud_service"] = _wait_for(
                "cloud service",
                lambda: _request_json(self.cloud_url, "/ready", timeout_seconds=1.0),
                lambda value: value.get("ready") is True,
                self.args.startup_timeout_seconds,
            )
            if cloud.poll() is not None:
                raise RuntimeError("cloud service exited after startup")
        else:
            _, self.startup_ms["external_cloud_probe"] = _wait_for(
                "external cloud service",
                lambda: _request_json(self.cloud_url, "/ready", timeout_seconds=1.0),
                lambda value: value.get("ready") is True,
                self.args.startup_timeout_seconds,
            )

        edge = self._start_process(
            "edge_service",
            [
                sys.executable,
                "-m",
                "cloud_edge_framework.edge_service",
                "--config",
                str(self.config_paths["edge"]),
                "--project_root",
                str(self.project_root),
            ],
        )
        edge_health, self.startup_ms["edge_service"] = _wait_for(
            "edge service and cloud probe",
            lambda: _request_json(self.edge_url, "/health", timeout_seconds=1.0),
            lambda value: value.get("ready") is True
            and value.get("cloud_available") is True,
            self.args.startup_timeout_seconds,
        )
        if edge.poll() is not None:
            raise RuntimeError("edge service exited after startup")
        return edge_health

    def stop(self) -> None:
        for name in ("edge_service", "cloud_service", "edge_qwen"):
            process = self.processes.get(name)
            if process is None or process.poll() is not None:
                continue
            try:
                os.kill(process.pid, signal.SIGINT)
                process.wait(timeout=8.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=3.0)
        for handle in self.log_handles.values():
            handle.close()

    def pids(self) -> Dict[str, int]:
        result = {name: process.pid for name, process in self.processes.items()}
        if not self.manage_edge_qwen:
            result["edge_qwen"] = self.external_edge_qwen_pid
        return result


def _select_samples(spec: str, sample_count: int, automatic_count: int) -> List[int]:
    if spec.strip().lower() != "auto":
        values = parse_sample_spec(spec)
    else:
        if automatic_count <= 0 or automatic_count > sample_count:
            raise ValueError("sample_count must be within the selected split size")
        values = np.linspace(
            0, sample_count - 1, num=automatic_count, dtype=np.int64
        ).tolist()
    normalized = [int(value) for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError("sample selection produced duplicate ids")
    return normalized


def _post_traffic_event(
    edge_url: str,
    native_event: Mapping[str, Any],
    request_id: str,
    timeout_seconds: float,
) -> Tuple[Dict[str, Any], float, int, int]:
    envelope = traffic_event_from_output(dict(native_event))
    payload = {"event": envelope}
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    request = Request(
        edge_url.rstrip("/") + DECIDE_PATH,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Idempotency-Key": request_id,
            "X-Trace-Id": "trace_{}".format(request_id),
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "edge request failed with HTTP {}: {}".format(exc.code, detail)
        ) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result = json.loads(response_body.decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("edge decision response must be an object")
    return result, elapsed_ms, len(body), len(response_body)


def _counter(values: Iterable[Any]) -> Dict[str, int]:
    counts = Counter(str(value) for value in values)
    return dict(sorted(counts.items()))


def _mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    return round(statistics.fmean(data), 6) if data else 0.0


def _edge_llm_release_evidence(edge_health: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = edge_health.get("runtime", {})
    plugins = runtime.get("plugins", []) if isinstance(runtime, dict) else []
    for plugin in plugins:
        if not isinstance(plugin, dict) or plugin.get("scene") != "traffic":
            continue
        health = plugin.get("health", {})
        edge_llm = health.get("edge_llm", {}) if isinstance(health, dict) else {}
        active = edge_llm.get("active", {}) if isinstance(edge_llm, dict) else {}
        metrics = active.get("metrics", {}) if isinstance(active, dict) else {}
        if not isinstance(metrics, dict) or "system_ram_footprint_mb" not in metrics:
            continue
        footprint_mb = float(metrics["system_ram_footprint_mb"])
        deployment = active.get("deployment", {})
        return {
            "available": True,
            "release_id": active.get("release_id"),
            "system_ram_footprint_mb": footprint_mb,
            "below_1_5_gb": footprint_mb <= 1536.0,
            "average_ttft_ms": float(metrics.get("average_ttft_ms", 0.0)),
            "input_tokens": int(deployment.get("max_input_tokens", 0)),
            "output_tokens": int(deployment.get("max_output_tokens", 0)),
            "scope": (
                "standalone Jetson system-RAM delta from before model startup through "
                "one-token inference; includes the resident model and unified GPU memory"
            ),
        }
    return {"available": False}


def _record_from_response(
    sample_id: int,
    native_event: Mapping[str, Any],
    perception_forward_ms: float,
    perception_ms: float,
    response: Mapping[str, Any],
    client_wall_ms: float,
    ingress_request_bytes: int,
    ingress_response_bytes: int,
) -> Dict[str, Any]:
    local = dict(response["local_decision"])
    final = dict(response["final_decision"])
    local_metadata = dict(local.get("metadata", {}))
    final_metadata = dict(final.get("metadata", {}))
    transport = final_metadata.get("transport", {})
    if not isinstance(transport, dict):
        transport = {}
    accounting = dict(response["closed_loop_accounting"])
    pipeline_stages = dict(accounting.get("pipeline_stage_ms", {}))
    data_plane = dict(response["data_plane"])
    evidence_plan = dict(response.get("evidence_plan", {}))
    action_authorization = local_metadata.get("action_authorization", {})
    if not isinstance(action_authorization, dict):
        action_authorization = {}
    deferred_action_types = action_authorization.get("deferred_action_types", [])
    if not isinstance(deferred_action_types, list):
        deferred_action_types = []
    scheduler_selected_wait = bool(
        data_plane.get(
            "scheduler_selected_wait",
            response.get("schedule", {}).get("waits_for_cloud", False),
        )
    )
    # The delivery path is deliberately asynchronous for multi-edge summaries.
    # Business completion follows action authorization.  A scheduler may make a
    # conservative wait recommendation, but a reversible/locally-authorized
    # action is already usable when provisional is returned.  Report the more
    # conservative scheduler interpretation separately instead of conflating it
    # with the action contract.
    requires_authoritative_final = bool(deferred_action_types)
    scheduler_conservative_requires_authoritative_final = bool(
        scheduler_selected_wait or deferred_action_types
    )
    observed_full_loop_ms = perception_ms + client_wall_ms
    return {
        "sample_id": sample_id,
        "partition_id": int(native_event["partition_id"]),
        "event_id": str(native_event["event_id"]),
        "region_risk_level": str(
            native_event["region_summary"]["region_risk_level"]
        ),
        "max_node_risk_level": str(
            native_event["region_summary"]["max_node_risk_level"]
        ),
        "upload_required": bool(native_event["upload_required"]),
        "model_forward_ms": round(float(perception_forward_ms), 6),
        "perception_ms": round(float(perception_ms), 6),
        "client_edge_http_wall_ms": round(float(client_wall_ms), 6),
        "observed_external_full_loop_ms": round(observed_full_loop_ms, 6),
        "framework_accounted_closed_loop_ms": float(
            accounting["accounted_closed_loop_ms"]
        ),
        "edge_service_wall_ms": float(response["edge_service_wall_ms"]),
        "framework_runtime_ms": float(response["framework_runtime_ms"]),
        "pipeline_stage_ms": {
            name: float(pipeline_stages.get(name, 0.0))
            for name in (
                "normalization",
                "edge_decision",
                "data_plane_preparation",
                "scheduling",
                "route_execution",
            )
        },
        "schedule_route": str(response["schedule"]["route"]),
        "scheduler_selected_route": str(
            data_plane.get("scheduler_selected_route", response["schedule"]["route"])
        ),
        "scheduler_selected_wait": scheduler_selected_wait,
        "requires_authoritative_final": requires_authoritative_final,
        "scheduler_conservative_requires_authoritative_final": (
            scheduler_conservative_requires_authoritative_final
        ),
        "deferred_action_types": [str(value) for value in deferred_action_types],
        "summary_delivery_required": bool(
            data_plane.get("summary_delivery_required", False)
        ),
        "evidence_required_level": str(
            evidence_plan.get("required_level", "unknown")
        ),
        "large_evidence_requested": str(
            evidence_plan.get("required_level", "summary")
        )
        in {"feature", "raw"},
        "executed_route": str(final["route"]),
        "local_decision": str(local["decision"]),
        "final_decision": str(final["decision"]),
        "local_source": str(local_metadata.get("source", "unknown")),
        "edge_decision_path": str(
            local_metadata.get("edge_decision_path", "unknown")
        ),
        "edge_llm_selected": bool(local_metadata.get("edge_llm_selected", False)),
        "edge_llm_selection_reason": str(
            local_metadata.get("edge_llm_selection_reason", "unknown")
        ),
        "edge_llm_accepted": local_metadata.get("source")
        == "edge_qwen_single_token",
        "edge_llm_latency_ms": float(
            local_metadata.get("edge_llm_latency_ms", 0.0) or 0.0
        ),
        "edge_llm_prompt_tokens": int(
            local_metadata.get("edge_llm_prompt_tokens", 0) or 0
        ),
        "edge_llm_output_tokens": int(
            local_metadata.get("edge_llm_output_tokens", 0) or 0
        ),
        "edge_llm_runtime_error": local_metadata.get("edge_llm_runtime_error"),
        "cloud_error": final_metadata.get("cloud_error"),
        "feedback_sync_error": final_metadata.get("feedback_sync_error"),
        "cloud_http_round_trip_ms": float(
            transport.get("http_round_trip_ms", 0.0) or 0.0
        ),
        "cloud_request_bytes": int(transport.get("request_bytes", 0) or 0),
        "cloud_response_bytes": int(transport.get("response_bytes", 0) or 0),
        "ingress_request_bytes": int(ingress_request_bytes),
        "ingress_response_bytes": int(ingress_response_bytes),
        "legacy_full_request_bytes": int(data_plane["legacy_full_request_bytes"]),
        "selected_request_bytes": int(data_plane["selected_request_bytes"]),
        "request_reduction_ratio": float(data_plane["request_reduction_ratio"]),
        "deadline_ms": float(response["schedule"]["deadline_ms"]),
        "deadline_met": observed_full_loop_ms
        <= float(response["schedule"]["deadline_ms"]),
        "feedback_count": int(response.get("feedback_count", 0)),
        "pending_review_count": int(response.get("pending_review_count", 0)),
    }


def _build_report(result: Mapping[str, Any]) -> str:
    latency = result["latency"]
    quality = result["execution"]
    memory = result["memory"]
    lines = [
        "# 交通云边框架真实闭环验收",
        "",
        "- 运行标识：`{}`".format(result["run_id"]),
        "- 平台：`{}`，ASTGCN 设备：`{}`。".format(
            result["platform"]["hostname"], result["perception"]["device"]
        ),
        "- 使用 {} 个不重复测试时刻、{} 个 METIS 区域，共发起 {} 次正式边缘请求。".format(
            result["sample_count"],
            result["partition_count"],
            quality["expected_request_count"],
        ),
        "- 同时报告三种互不混用的口径：输入到本地 provisional、按业务策略完成、输入到全局 final。",
        "",
        "## 结果",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        "| 请求成功率 | {:.2%} |".format(quality["success_rate"]),
        "| 云端完整汇聚率 | {:.2%} |".format(
            quality["complete_aggregation_rate"]
        ),
        "| provisional→final 完成率 | {:.2%} |".format(
            quality["final_completion_rate"]
        ),
        "| 存在初始冲突的样本组比例 | {:.2%} |".format(
            quality["conflict_group_rate"]
        ),
        "| 初始冲突 → 协调后残余冲突 | {} → {} |".format(
            quality["initial_conflict_count"],
            quality["residual_conflict_count"],
        ),
        "| 冲突解决成功率 | {:.2%} |".format(
            quality["conflict_resolution_success_rate"]
        ),
        "| 输入到本地 provisional 的 0.2 s 达标率 | {:.2%} |".format(
            quality["deadline_success_rate_all_requests"]
        ),
        "| 输入到本地 provisional 平均时延 | {:.3f} ms |".format(
            latency["observed_external_full_loop"]["average_ms"]
        ),
        "| 输入到本地 provisional P95 | {:.3f} ms |".format(
            latency["observed_external_full_loop"]["p95_ms"]
        ),
        "| 按业务策略完成平均时延 | {:.3f} ms |".format(
            latency["observed_business_completion"]["average_ms"]
        ),
        "| 按业务策略完成 P95 | {:.3f} ms |".format(
            latency["observed_business_completion"]["p95_ms"]
        ),
        "| 调度器保守等待口径平均时延 | {:.3f} ms |".format(
            latency["observed_scheduler_conservative_completion"]["average_ms"]
        ),
        "| 按业务策略完成的 0.2 s 达标率 | {:.2%} |".format(
            quality["business_deadline_success_rate_all_requests"]
        ),
        "| 样本级按业务策略完成平均时延（四边取最大） | {:.3f} ms |".format(
            latency["sample_business_completion"]["average_ms"]
        ),
        "| 样本级按业务策略完成 P95（四边取最大） | {:.3f} ms |".format(
            latency["sample_business_completion"]["p95_ms"]
        ),
        "| 样本级按业务策略完成的 0.2 s 达标率 | {:.2%} |".format(
            quality["sample_business_deadline_success_rate"]
        ),
        "| 输入到全局 final 平均时延 | {:.3f} ms |".format(
            latency["observed_input_to_global_final"]["average_ms"]
        ),
        "| 输入到全局 final P95 | {:.3f} ms |".format(
            latency["observed_input_to_global_final"]["p95_ms"]
        ),
        "| 输入到全局 final 的 0.2 s 达标率 | {:.2%} |".format(
            quality["global_final_deadline_success_rate_all_requests"]
        ),
        "| 样本级全局 final 平均时延（四边取最大） | {:.3f} ms |".format(
            latency["sample_global_final"]["average_ms"]
        ),
        "| 不等待云端即可完成的业务比例 | {:.2%} |".format(
            quality["local_business_completion_rate"]
        ),
        "| 等待权威 final 的业务比例 | {:.2%} |".format(
            quality["authoritative_final_required_rate"]
        ),
        "| 调度器建议下不等待 final 的比例 | {:.2%} |".format(
            quality["scheduler_conservative_local_completion_rate"]
        ),
        "| 轻量摘要上云比例 | {:.2%} |".format(
            quality["summary_delivery_required_rate"]
        ),
        "| 大证据请求比例 | {:.2%} |".format(
            quality["large_evidence_requested_rate"]
        ),
        "| ASTGCN 感知平均时延 | {:.3f} ms |".format(
            latency["perception"]["average_ms"]
        ),
        "| Edge-Qwen 触发率 | {:.2%} |".format(quality["edge_llm_selection_rate"]),
        "| Edge-Qwen 接受率（触发后） | {:.2%} |".format(
            quality["edge_llm_acceptance_rate_when_selected"]
        ),
        "| 云端同步路由率 | {:.2%} |".format(quality["cloud_sync_rate"]),
        "| 云端 9B 结构化复核次数 | {} |".format(
            quality["cloud_llm_review_count"]
        ),
        "| 云请求平均压缩率 | {:.2%} |".format(
            result["communication"]["mean_request_reduction_ratio"]
        ),
        "| 边侧进程组峰值 PSS | {:.1f} MB |".format(
            memory["processes"]["edge_stack_peak_pss_mb"]
        ),
        "| Edge-Qwen 峰值 PSS | {:.1f} MB |".format(
            memory["processes"]["component_peak_pss_mb"]["edge_qwen"]
        ),
        "| Edge-Qwen 单次推理系统内存足迹 | {:.1f} MB |".format(
            memory["edge_llm_single_inference"]["system_ram_footprint_mb"]
        ),
        "| Edge-Qwen 单次推理内存不超过 1.5 GB | {} |".format(
            "是" if memory["edge_llm_single_inference"]["below_1_5_gb"] else "否"
        ),
        "",
        "## 分布",
        "",
        "- 调度路由：`{}`".format(
            json.dumps(quality["schedule_routes"], ensure_ascii=False)
        ),
        "- 原始调度意图：`{}`".format(
            json.dumps(quality["scheduler_selected_routes"], ensure_ascii=False)
        ),
        "- 边缘决策路径：`{}`".format(
            json.dumps(quality["edge_decision_paths"], ensure_ascii=False)
        ),
        "- 区域风险：`{}`".format(
            json.dumps(quality["regional_risk_levels"], ensure_ascii=False)
        ),
        "- 节点最高风险：`{}`".format(
            json.dumps(quality["max_node_risk_levels"], ensure_ascii=False)
        ),
        "",
        "## 口径说明",
        "",
        "- `observed_external_full_loop_ms` 是输入到本地 provisional；它包含感知和边缘 HTTP，不包含异步云端 final。",
        "- `observed_business_completion_ms` 按动作授权计时：本地已授权事件止于 provisional，动作明确带 `requires_cloud_confirmation` 时止于 authoritative final。",
        "- `observed_scheduler_conservative_completion_ms` 额外把调度器的等待建议算成阻塞；它是更保守的对照口径，不改变动作授权语义。",
        "- `observed_input_to_global_final_ms` 强制所有事件都等 final，仅用于说明全局一致性闭环，不等同于普通监测业务响应时间。",
        "- 样本级指标把同一 sample_id 的四个边缘完成时间取最大值，再跨样本统计；这是多边缘任务完成的保守主口径。",
        "- 在线多边缘样本的轻量摘要仍异步上云；不等待云端不等于不上传摘要。动态调度控制的是是否阻塞、是否请求大证据和是否调用大模型。",
        "- 初始冲突组比例是协调前诊断量；题目要求的最终多节点决策冲突应同时查看协调后残余冲突，不能把两者互换。",
        "- 每个测试时刻只执行一次完整 ASTGCN，并发提交四个区域事件；本机四个逻辑边仍共享一个 `parallel=1` 的 Edge-Qwen 服务，因此 `timestamp_group_wall_ms` 是单机竞争压力，不代表两台 Jetson 的分布式时延。",
        "- `provisional_to_final` 是异步云端最终回填时延；普通监测业务可先使用 provisional，高风险不可逆动作必须等待 final。",
        "- PSS 用于按比例分摊多进程共享页；边侧进程组包含 ASTGCN、Edge-Qwen 和边缘 HTTP 服务，不含云服务。",
        "- 1.5 GB 指标采用 active release 的 Jetson 独立测试系统内存足迹；共驻运行中的进程 PSS 不包含 CUDA 统一内存，二者不可混用。",
        "- 当前云端在线决策器由部署配置决定；是否为 9B Teacher 见 `cloud_backend` 字段，不能把专用协调器写成大模型推理。",
        "- 该验收证明软件闭环和运行性能，不等同于 SUMO 或真实道路上的控制收益。",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run a real resident-ASTGCN traffic framework acceptance benchmark."
    )
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--config", default="configurations/PEMS08_astgcn.conf")
    parser.add_argument(
        "--data-npz",
        default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz",
    )
    parser.add_argument(
        "--checkpoint",
        default="experiments/PEMS08/joint_risk_astgcn_metis4_flowprio2_frozen/best.pt",
    )
    parser.add_argument(
        "--risk-calibrator", default="models/region_risk_conformal.json"
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--samples",
        default="auto",
        help="auto for split-wide evenly spaced samples, or start:end:step/list",
    )
    parser.add_argument("--sample-count", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--request-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--cloud-timeout-seconds", type=float, default=0.5)
    parser.add_argument("--startup-timeout-seconds", type=float, default=90.0)
    parser.add_argument(
        "--aggregation-timeout-ms",
        type=int,
        default=1000,
        help=(
            "同一样本多边缘摘要的汇聚窗口；单机验收共享一个串行 "
            "Edge-Qwen，默认放宽到 1000 ms，真实多机实验应单独报告。"
        ),
    )
    parser.add_argument(
        "--final-wait-seconds",
        type=float,
        default=20.0,
        help="等待所有 provisional 决策收到云端 final 回填的最长时间。",
    )
    parser.add_argument("--cloud-port", type=int, default=0)
    parser.add_argument("--external-cloud-url", default="")
    parser.add_argument("--edge-port", type=int, default=0)
    parser.add_argument("--llama-port", type=int, default=0)
    parser.add_argument(
        "--external-edge-llm-url",
        default="",
        help="Reuse an already running llama-server instead of loading another model.",
    )
    parser.add_argument(
        "--edge-plugin-config",
        default=str(root / "deployment/framework/scene_plugins_edge.json"),
    )
    parser.add_argument(
        "--cloud-plugin-config",
        default=str(root / "deployment/framework/scene_plugins.json"),
    )
    parser.add_argument(
        "--cloud-llm-runtime-config",
        default="",
        help="可选：启用真实云端 Qwen 结构化复核。",
    )
    parser.add_argument(
        "--cloud-llm-min-risk-level",
        choices=("low", "medium", "high", "severe"),
        default="severe",
    )
    parser.add_argument(
        "--release-registry",
        default=str(root / "runtime/edge_llm_release_store.json"),
    )
    parser.add_argument(
        "--edge-llm-runtime-config",
        default=str(
            root / "deployment/edge_llm/runtime/freeway_traffic_llama_cpp.json"
        ),
    )
    parser.add_argument(
        "--llama-binary",
        default="llama-server",
        help="llama-server 可执行文件路径；独立 SDK 启动器要求显式填写。",
    )
    parser.add_argument("--llama-context-tokens", type=int, default=128)
    parser.add_argument("--llama-threads", type=int, default=4)
    parser.add_argument("--llama-gpu-layers", type=int, default=99)
    parser.add_argument(
        "--edge-llm-mode",
        choices=("disabled", "shadow", "selective", "primary"),
        default="",
        help="Optionally override the deployed plugin mode for controlled profiling.",
    )
    parser.add_argument("--require-edge-llm", action="store_true")
    parser.add_argument(
        "--output",
        default="results/framework/traffic_framework_acceptance_latest.json",
    )
    parser.add_argument(
        "--report",
        default="results/framework/traffic_framework_acceptance_latest.md",
    )
    return parser.parse_args()


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    if args.aggregation_timeout_ms <= 0:
        raise ValueError("aggregation-timeout-ms must be positive")
    if min(
        args.request_timeout_seconds,
        args.cloud_timeout_seconds,
        args.startup_timeout_seconds,
        args.final_wait_seconds,
    ) <= 0:
        raise ValueError("timeouts must be positive")
    project_root = Path(args.project_root).resolve()
    output_path = _project_path(project_root, args.output)
    report_path = _project_path(project_root, args.report)
    run_id = "traffic-{}-{}".format(
        time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8]
    )
    run_dir = output_path.parent / "traffic_acceptance_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    for name in (
        "edge_plugin_config",
        "cloud_plugin_config",
        "release_registry",
        "edge_llm_runtime_config",
        "llama_binary",
    ):
        value = getattr(args, name)
        setattr(args, name, str(_project_path(project_root, value)))
    if str(args.cloud_llm_runtime_config).strip():
        args.cloud_llm_runtime_config = str(
            _project_path(project_root, args.cloud_llm_runtime_config)
        )

    import torch

    torch.set_num_threads(args.torch_threads)
    system_memory = SystemMemorySampler(0.02)
    system_memory.start()
    stack = ManagedTrafficStack(project_root, run_dir, args)
    memory_sampler: Optional[StackMemorySampler] = None
    startup_health: Dict[str, Any] = {}
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    timestamp_records: List[Dict[str, Any]] = []
    edge_health: Dict[str, Any] = {}
    edge_metrics: Dict[str, Any] = {}
    cloud_metrics: Dict[str, Any] = {}
    cloud_health: Dict[str, Any] = {}
    review_records: Dict[str, Dict[str, Any]] = {}
    aggregation_records: Dict[str, Dict[str, Any]] = {}
    final_wait_ms = 0.0
    perception: Optional[JointTrafficPerceptionRuntime] = None
    warmup: Dict[str, Any] = {}
    measured_started = 0.0
    try:
        startup_health = stack.start()
        perception = JointTrafficPerceptionRuntime(
            config_path=_project_path(project_root, args.config),
            data_path=_project_path(project_root, args.data_npz),
            checkpoint_path=_project_path(project_root, args.checkpoint),
            risk_calibrator_path=_project_path(project_root, args.risk_calibrator),
            split=args.split,
            device_name=args.device,
            top_k=args.top_k,
        )
        sample_ids = perception.validate_sample_ids(
            _select_samples(args.samples, perception.sample_count, args.sample_count)
        )
        warmup_forward_ms = perception.warmup(sample_ids[0])
        warmup_result = perception.infer_sample(sample_ids[0])
        warmup_event = copy.deepcopy(warmup_result.events[0])
        warmup_event["event_id"] = warmup_event["event_id"] + "_warmup_" + run_id
        # Warm-up must not occupy the real sample's durable aggregation member.
        # Keep the original perception sample but isolate its aggregation key.
        warmup_event["sample_split"] = "warmup_" + run_id
        warmup_event["num_partitions"] = 1
        warmup_request_id = "{}-warmup".format(run_id)
        warmup_response, warmup_http_ms, _, _ = _post_traffic_event(
            stack.edge_url,
            warmup_event,
            warmup_request_id,
            args.request_timeout_seconds,
        )
        warmup = {
            "astgcn_forward_ms": warmup_forward_ms,
            "resident_perception_ms": warmup_result.perception_ms,
            "edge_http_wall_ms": round(warmup_http_ms, 6),
            "edge_decision_path": warmup_response["local_decision"]
            .get("metadata", {})
            .get("edge_decision_path", "unknown"),
        }

        pids = stack.pids()
        memory_sampler = StackMemorySampler(
            perception_pid=os.getpid(),
            edge_qwen_pid=pids["edge_qwen"],
            edge_service_pid=pids["edge_service"],
            cloud_service_pid=pids.get("cloud_service", 0),
        )
        memory_sampler.start()
        measured_started = time.perf_counter()
        for sample_index, sample_id in enumerate(sample_ids, start=1):
            timestamp_started = time.perf_counter()
            perception_result = perception.infer_sample(sample_id)
            timestamp_success = 0
            measured_events = []
            for native_event in perception_result.events:
                measured_event = copy.deepcopy(native_event)
                measured_event["event_id"] = "{}_{}".format(
                    native_event["event_id"], run_id
                )
                measured_event["aggregation_timeout_ms"] = int(
                    args.aggregation_timeout_ms
                )
                measured_events.append(measured_event)

            def submit_event(measured_event):
                request_id = "{}-s{:04d}-p{}".format(
                    run_id, sample_id, measured_event["partition_id"]
                )
                try:
                    response, client_wall_ms, request_bytes, response_bytes = (
                        _post_traffic_event(
                            stack.edge_url,
                            measured_event,
                            request_id,
                            args.request_timeout_seconds,
                        )
                    )
                    return (
                        "record",
                        _record_from_response(
                            sample_id,
                            measured_event,
                            perception_result.model_forward_ms,
                            perception_result.perception_ms,
                            response,
                            client_wall_ms,
                            request_bytes,
                            response_bytes,
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    return (
                        "failure",
                        {
                            "sample_id": sample_id,
                            "partition_id": int(measured_event["partition_id"]),
                            "event_id": str(measured_event["event_id"]),
                            "error": "{}: {}".format(type(exc).__name__, exc),
                            "timeout": isinstance(
                                exc, (TimeoutError, socket.timeout, URLError)
                            ),
                        },
                    )

            # Four logical edge partitions perceive the same traffic timestamp.
            # Concurrent submission matches the distributed deployment and avoids
            # turning Edge-Qwen latency into artificial aggregation skew.
            with ThreadPoolExecutor(
                max_workers=perception.partition_count
            ) as executor:
                futures = [
                    executor.submit(submit_event, measured_event)
                    for measured_event in measured_events
                ]
                for future in as_completed(futures):
                    result_type, result_value = future.result()
                    if result_type == "record":
                        records.append(result_value)
                        timestamp_success += 1
                    else:
                        failures.append(result_value)
            timestamp_wall_ms = (time.perf_counter() - timestamp_started) * 1000.0
            timestamp_records.append(
                {
                    "sample_id": sample_id,
                    "success_count": timestamp_success,
                    "expected_count": perception.partition_count,
                    "model_forward_ms": perception_result.model_forward_ms,
                    "perception_ms": perception_result.perception_ms,
                    "timestamp_group_wall_ms": round(timestamp_wall_ms, 6),
                }
            )
            print(
                "[{}/{}] sample={} perception={:.3f}ms requests={}/{} group={:.3f}ms".format(
                    sample_index,
                    len(sample_ids),
                    sample_id,
                    perception_result.perception_ms,
                    timestamp_success,
                    perception.partition_count,
                    timestamp_wall_ms,
                ),
                flush=True,
            )

        records.sort(key=lambda row: (row["sample_id"], row["partition_id"]))

        def read_reviews():
            return {
                record["event_id"]: _request_json(
                    stack.edge_url,
                    "/api/v1/collaboration/reviews/{}".format(
                        record["event_id"]
                    ),
                    timeout_seconds=args.request_timeout_seconds,
                )
                for record in records
            }

        review_records, final_wait_ms = _wait_for(
            "all edge provisional decisions to receive authoritative cloud final",
            read_reviews,
            lambda value: len(value) == len(records)
            and all(
                item.get("state") == "completed"
                and str(item.get("completion_stage", ""))
                not in NON_AUTHORITATIVE_REVIEW_STAGES
                for item in value.values()
            ),
            args.final_wait_seconds,
        )
        group_ids = set()
        for record in records:
            review = review_records[record["event_id"]]
            final_decision = review.get("final_decision") or {}
            metadata = final_decision.get("metadata", {})
            aggregation = (
                metadata.get("aggregation", {})
                if isinstance(metadata, dict)
                else {}
            )
            cloud_llm_review = (
                metadata.get("cloud_llm_review")
                if isinstance(metadata, dict)
                else None
            )
            group_id = aggregation.get("group_id")
            if group_id:
                group_ids.add(str(group_id))
            record.update(
                {
                    "review_state": str(review.get("state", "")),
                    "review_completion_mode": str(
                        review.get("completion_mode", "")
                    ),
                    "review_completion_stage": str(
                        review.get("completion_stage", "")
                    ),
                    "eventual_completion_ms": float(
                        review.get("eventual_completion_ms", 0.0) or 0.0
                    ),
                    "decision_changed": bool(
                        review.get("decision_changed", False)
                    ),
                    "authoritative_final_decision": str(
                        final_decision.get("decision", "")
                    ),
                    "authoritative_final_route": str(
                        final_decision.get("route", "")
                    ),
                    "aggregation_group_id": str(group_id or ""),
                    "aggregation_state": str(
                        aggregation.get("state", "")
                    ),
                    "aggregation_completion_reason": str(
                        aggregation.get("completion_reason", "")
                    ),
                    "cloud_llm_review": (
                        dict(cloud_llm_review)
                        if isinstance(cloud_llm_review, dict)
                        else None
                    ),
                }
            )
            record["observed_input_to_global_final_ms"] = round(
                float(record["perception_ms"])
                + float(record["eventual_completion_ms"]),
                6,
            )
            record["observed_business_completion_ms"] = round(
                (
                    float(record["observed_input_to_global_final_ms"])
                    if record["requires_authoritative_final"]
                    else float(record["observed_external_full_loop_ms"])
                ),
                6,
            )
            record["observed_scheduler_conservative_completion_ms"] = round(
                (
                    float(record["observed_input_to_global_final_ms"])
                    if record[
                        "scheduler_conservative_requires_authoritative_final"
                    ]
                    else float(record["observed_external_full_loop_ms"])
                ),
                6,
            )
        for group_id in sorted(group_ids):
            aggregation_records[group_id] = _request_json(
                stack.cloud_url,
                "/api/v1/collaboration/aggregations/{}".format(group_id),
                timeout_seconds=args.request_timeout_seconds,
            )
        incomplete_groups = [
            value
            for value in aggregation_records.values()
            if value.get("state") != "completed"
            or set(value.get("received_members", []))
            != set(value.get("expected_members", []))
        ]
        if len(aggregation_records) != len(sample_ids) or incomplete_groups:
            raise RuntimeError(
                "full aggregation failed: groups={}, samples={}, incomplete={}".format(
                    len(aggregation_records),
                    len(sample_ids),
                    [item.get("group_id") for item in incomplete_groups],
                )
            )

        measured_seconds = time.perf_counter() - measured_started
        edge_health = _request_json(
            stack.edge_url, "/health", timeout_seconds=args.request_timeout_seconds
        )
        edge_metrics = _request_json(
            stack.edge_url, METRICS_PATH, timeout_seconds=args.request_timeout_seconds
        )
        cloud_metrics = _request_json(
            stack.cloud_url, METRICS_PATH, timeout_seconds=args.request_timeout_seconds
        )
        cloud_health = _request_json(
            stack.cloud_url, "/health", timeout_seconds=args.request_timeout_seconds
        )
        memory_sampler.stop()
        memory_result = memory_sampler.result()
        memory_sampler = None
    finally:
        if memory_sampler is not None:
            memory_sampler.stop()
            memory_result = memory_sampler.result()
        stack.stop()
        system_memory.stop()

    if perception is None:
        raise RuntimeError("traffic perception runtime did not initialize")
    expected_request_count = len(sample_ids) * perception.partition_count
    success_count = len(records)
    selected_count = sum(record["edge_llm_selected"] for record in records)
    accepted_count = sum(record["edge_llm_accepted"] for record in records)
    deadline_met_count = sum(record["deadline_met"] for record in records)
    cloud_sync_count = sum(
        record["executed_route"] == "cloud_sync" for record in records
    )
    local_business_completion_count = sum(
        not record["requires_authoritative_final"] for record in records
    )
    scheduler_conservative_local_completion_count = sum(
        not record["scheduler_conservative_requires_authoritative_final"]
        for record in records
    )
    authoritative_final_count = sum(
        record.get("review_state") == "completed"
        and record.get("review_completion_stage")
        not in NON_AUTHORITATIVE_REVIEW_STAGES
        for record in records
    )
    business_deadline_met_count = sum(
        record.get("observed_business_completion_ms", float("inf")) <= 200.0
        for record in records
    )
    global_final_deadline_met_count = sum(
        record.get("observed_input_to_global_final_ms", float("inf")) <= 200.0
        for record in records
    )
    sample_business_completion_ms = []
    sample_scheduler_conservative_completion_ms = []
    sample_global_final_ms = []
    for sample_id in sample_ids:
        sample_records = [
            record for record in records if record["sample_id"] == sample_id
        ]
        if len(sample_records) != perception.partition_count:
            continue
        sample_business_completion_ms.append(
            max(record["observed_business_completion_ms"] for record in sample_records)
        )
        sample_scheduler_conservative_completion_ms.append(
            max(
                record["observed_scheduler_conservative_completion_ms"]
                for record in sample_records
            )
        )
        sample_global_final_ms.append(
            max(
                record["observed_input_to_global_final_ms"]
                for record in sample_records
            )
        )
    sample_business_deadline_met_count = sum(
        value <= 200.0 for value in sample_business_completion_ms
    )
    sample_global_final_deadline_met_count = sum(
        value <= 200.0 for value in sample_global_final_ms
    )
    coordination_results = [
        value.get("result", {})
        for value in aggregation_records.values()
        if isinstance(value.get("result"), dict)
    ]
    initial_conflict_count = sum(
        int(value.get("initial_conflict_count", 0))
        for value in coordination_results
    )
    residual_conflict_count = sum(
        int(value.get("residual_conflict_count", 0))
        for value in coordination_results
    )
    conflict_group_count = sum(
        int(value.get("initial_conflict_count", 0)) > 0
        for value in coordination_results
    )
    conflict_resolution_success_rate = (
        (initial_conflict_count - residual_conflict_count)
        / initial_conflict_count
        if initial_conflict_count
        else 1.0
    )
    if args.require_edge_llm and selected_count == 0:
        raise RuntimeError("no measured event selected Edge-Qwen")
    edge_llm_runtime_error_count = sum(
        bool(record["edge_llm_runtime_error"]) for record in records
    )
    if args.require_edge_llm and edge_llm_runtime_error_count:
        raise RuntimeError(
            "Edge-Qwen had {} runtime errors".format(edge_llm_runtime_error_count)
        )

    result: Dict[str, Any] = {
        "schema_version": 2,
        "task": "traffic_framework_real_closed_loop_acceptance",
        "run_id": run_id,
        "created_at_epoch_ms": int(time.time() * 1000),
        "platform": {
            "hostname": socket.gethostname(),
            "python": sys.version.split()[0],
            "pid": os.getpid(),
        },
        "artifacts_dir": str(run_dir),
        "sample_selection": {
            "split": args.split,
            "spec": args.samples,
            "sample_ids": sample_ids,
            "unique": len(sample_ids) == len(set(sample_ids)),
        },
        "sample_count": len(sample_ids),
        "partition_count": perception.partition_count,
        "perception": {
            "device": str(perception.device),
            "torch_threads": torch.get_num_threads(),
            "load_latency_ms": perception.load_latency_ms,
            "config": str(perception.config_path),
            "data": str(perception.data_path),
            "checkpoint": str(perception.checkpoint_path),
            "risk_calibrator": str(perception.risk_calibrator_path),
        },
        "services": {
            "edge_url": stack.edge_url,
            "cloud_url": stack.cloud_url,
            "edge_qwen_url": stack.llama_url,
            "cloud_managed_by_benchmark_process": stack.manage_cloud,
            "edge_qwen_managed_by_benchmark_process": stack.manage_edge_qwen,
            "startup_ms": stack.startup_ms,
            "startup_health": startup_health,
            "final_edge_health": edge_health,
            "final_cloud_health": cloud_health,
        },
        "cloud_backend": {
            "configured_plugin": str(Path(args.cloud_plugin_config).resolve()),
            "online_decision_path": "traffic topology-fused ExtraTrees coordinator",
            "uses_qwen9b_teacher_online": bool(
                str(args.cloud_llm_runtime_config).strip()
            ),
            "note": (
                "Qwen 9B structured review was enabled for this run."
                if str(args.cloud_llm_runtime_config).strip()
                else "Qwen 9B review is disabled in this low-latency run."
            ),
        },
        "warmup": warmup,
        "execution": {
            "expected_request_count": expected_request_count,
            "success_count": success_count,
            "failure_count": len(failures),
            "timeout_count": sum(bool(row.get("timeout")) for row in failures),
            "success_rate": round(success_count / expected_request_count, 6),
            "final_completion_count": authoritative_final_count,
            "final_completion_rate": round(
                authoritative_final_count / max(1, expected_request_count),
                6,
            ),
            "cloud_correction_count": sum(
                bool(record.get("decision_changed")) for record in records
            ),
            "cloud_correction_rate": round(
                sum(
                    bool(record.get("decision_changed"))
                    for record in records
                )
                / max(1, success_count),
                6,
            ),
            "complete_aggregation_count": sum(
                value.get("state") == "completed"
                and set(value.get("received_members", []))
                == set(value.get("expected_members", []))
                for value in aggregation_records.values()
            ),
            "complete_aggregation_rate": round(
                sum(
                    value.get("state") == "completed"
                    and set(value.get("received_members", []))
                    == set(value.get("expected_members", []))
                    for value in aggregation_records.values()
                )
                / max(1, len(sample_ids)),
                6,
            ),
            "conflict_group_count": conflict_group_count,
            "conflict_group_rate": round(
                conflict_group_count / max(1, len(aggregation_records)), 6
            ),
            "initial_conflict_count": initial_conflict_count,
            "residual_conflict_count": residual_conflict_count,
            "conflict_resolution_success_rate": round(
                conflict_resolution_success_rate, 6
            ),
            "cloud_llm_review_count": sum(
                isinstance(record.get("cloud_llm_review"), dict)
                for record in records
            ),
            "cloud_llm_review_verdicts": _counter(
                record["cloud_llm_review"].get("verdict", "unknown")
                for record in records
                if isinstance(record.get("cloud_llm_review"), dict)
            ),
            "deadline_met_count": deadline_met_count,
            "deadline_success_rate_successful": round(
                deadline_met_count / max(1, success_count), 6
            ),
            "deadline_success_rate_all_requests": round(
                deadline_met_count / expected_request_count, 6
            ),
            "average_e2e_below_200ms": bool(
                records
                and _mean(
                    record["observed_external_full_loop_ms"] for record in records
                )
                <= 200.0
            ),
            "average_business_e2e_below_200ms": bool(
                records
                and _mean(
                    record["observed_business_completion_ms"] for record in records
                )
                <= 200.0
            ),
            "average_global_final_e2e_below_200ms": bool(
                records
                and _mean(
                    record["observed_input_to_global_final_ms"]
                    for record in records
                )
                <= 200.0
            ),
            "average_sample_business_e2e_below_200ms": bool(
                sample_business_completion_ms
                and _mean(sample_business_completion_ms) <= 200.0
            ),
            "average_sample_global_final_e2e_below_200ms": bool(
                sample_global_final_ms
                and _mean(sample_global_final_ms) <= 200.0
            ),
            "business_deadline_met_count": business_deadline_met_count,
            "business_deadline_success_rate_all_requests": round(
                business_deadline_met_count / max(1, expected_request_count), 6
            ),
            "global_final_deadline_met_count": global_final_deadline_met_count,
            "global_final_deadline_success_rate_all_requests": round(
                global_final_deadline_met_count / max(1, expected_request_count), 6
            ),
            "sample_business_deadline_met_count": (
                sample_business_deadline_met_count
            ),
            "sample_business_deadline_success_rate": round(
                sample_business_deadline_met_count
                / max(1, len(sample_business_completion_ms)),
                6,
            ),
            "sample_global_final_deadline_met_count": (
                sample_global_final_deadline_met_count
            ),
            "sample_global_final_deadline_success_rate": round(
                sample_global_final_deadline_met_count
                / max(1, len(sample_global_final_ms)),
                6,
            ),
            "local_business_completion_count": local_business_completion_count,
            "local_business_completion_rate": round(
                local_business_completion_count / max(1, success_count), 6
            ),
            "authoritative_final_required_count": (
                success_count - local_business_completion_count
            ),
            "authoritative_final_required_rate": round(
                (success_count - local_business_completion_count)
                / max(1, success_count),
                6,
            ),
            "scheduler_conservative_local_completion_count": (
                scheduler_conservative_local_completion_count
            ),
            "scheduler_conservative_local_completion_rate": round(
                scheduler_conservative_local_completion_count
                / max(1, success_count),
                6,
            ),
            "summary_delivery_required_count": sum(
                record["summary_delivery_required"] for record in records
            ),
            "summary_delivery_required_rate": round(
                sum(record["summary_delivery_required"] for record in records)
                / max(1, success_count),
                6,
            ),
            "large_evidence_requested_count": sum(
                record["large_evidence_requested"] for record in records
            ),
            "large_evidence_requested_rate": round(
                sum(record["large_evidence_requested"] for record in records)
                / max(1, success_count),
                6,
            ),
            "edge_llm_selected_count": selected_count,
            "edge_llm_accepted_count": accepted_count,
            "edge_llm_runtime_error_count": edge_llm_runtime_error_count,
            "edge_llm_sla_probe_count": sum(
                record["edge_llm_selection_reason"] == "deadline_profile_probe"
                for record in records
            ),
            "edge_llm_selection_reasons": _counter(
                record["edge_llm_selection_reason"] for record in records
            ),
            "edge_llm_selection_rate": round(
                selected_count / max(1, success_count), 6
            ),
            "edge_llm_acceptance_rate_when_selected": round(
                accepted_count / max(1, selected_count), 6
            ),
            "cloud_sync_count": cloud_sync_count,
            "cloud_sync_rate": round(cloud_sync_count / max(1, success_count), 6),
            "schedule_routes": _counter(
                record["schedule_route"] for record in records
            ),
            "scheduler_selected_routes": _counter(
                record["scheduler_selected_route"] for record in records
            ),
            "executed_routes": _counter(
                record["executed_route"] for record in records
            ),
            "edge_decision_paths": _counter(
                record["edge_decision_path"] for record in records
            ),
            "local_decisions": _counter(
                record["local_decision"] for record in records
            ),
            "final_decisions": _counter(
                record["authoritative_final_decision"] for record in records
            ),
            "regional_risk_levels": _counter(
                record["region_risk_level"] for record in records
            ),
            "max_node_risk_levels": _counter(
                record["max_node_risk_level"] for record in records
            ),
            "upload_required_rate": round(
                sum(record["upload_required"] for record in records)
                / max(1, success_count),
                6,
            ),
            "measurement_wall_seconds": round(measured_seconds, 6),
            "throughput_requests_per_second": round(
                expected_request_count / max(measured_seconds, 1e-9), 6
            ),
        },
        "latency": {
            "model_forward": summarize(
                row["model_forward_ms"] for row in timestamp_records
            ),
            "perception": summarize(row["perception_ms"] for row in timestamp_records),
            "client_edge_http_wall": summarize(
                row["client_edge_http_wall_ms"] for row in records
            ),
            "observed_external_full_loop": summarize(
                row["observed_external_full_loop_ms"] for row in records
            ),
            "observed_business_completion": summarize(
                row["observed_business_completion_ms"] for row in records
            ),
            "observed_scheduler_conservative_completion": summarize(
                row["observed_scheduler_conservative_completion_ms"]
                for row in records
            ),
            "observed_input_to_global_final": summarize(
                row["observed_input_to_global_final_ms"] for row in records
            ),
            "sample_business_completion": summarize(
                sample_business_completion_ms
            ),
            "sample_scheduler_conservative_completion": summarize(
                sample_scheduler_conservative_completion_ms
            ),
            "sample_global_final": summarize(sample_global_final_ms),
            "framework_accounted_closed_loop": summarize(
                row["framework_accounted_closed_loop_ms"] for row in records
            ),
            "edge_service_wall": summarize(
                row["edge_service_wall_ms"] for row in records
            ),
            "edge_pipeline_stages": {
                name: summarize(row["pipeline_stage_ms"][name] for row in records)
                for name in (
                    "normalization",
                    "edge_decision",
                    "data_plane_preparation",
                    "scheduling",
                    "route_execution",
                )
            },
            "edge_qwen_selected": summarize(
                row["edge_llm_latency_ms"]
                for row in records
                if row["edge_llm_selected"]
            ),
            "cloud_http_round_trip": summarize(
                row["cloud_http_round_trip_ms"]
                for row in records
                if row["cloud_http_round_trip_ms"] > 0.0
            ),
            "timestamp_group_wall": summarize(
                row["timestamp_group_wall_ms"] for row in timestamp_records
            ),
            "edge_request_to_final": summarize(
                row["eventual_completion_ms"] for row in records
            ),
            "provisional_to_final_legacy_mislabeled": summarize(
                row["eventual_completion_ms"] for row in records
            ),
            "cloud_llm_review": summarize(
                row["cloud_llm_review"].get("latency_ms", 0.0)
                for row in records
                if isinstance(row.get("cloud_llm_review"), dict)
            ),
            "final_wait_wall_ms": final_wait_ms,
        },
        "communication": {
            "mean_ingress_request_bytes": _mean(
                row["ingress_request_bytes"] for row in records
            ),
            "mean_ingress_response_bytes": _mean(
                row["ingress_response_bytes"] for row in records
            ),
            "mean_legacy_cloud_request_bytes": _mean(
                row["legacy_full_request_bytes"] for row in records
            ),
            "mean_selected_cloud_request_bytes": _mean(
                row["selected_request_bytes"] for row in records
            ),
            "mean_request_reduction_ratio": _mean(
                row["request_reduction_ratio"] for row in records
            ),
        },
        "memory": {
            "processes": memory_result,
            "edge_llm_single_inference": _edge_llm_release_evidence(edge_health),
            "system_ram_baseline_mb": system_memory.baseline_mb,
            "system_ram_peak_mb": system_memory.peak_mb,
            "system_ram_peak_delta_mb": system_memory.peak_delta_mb,
            "note": (
                "On Jetson, system RAM includes unified GPU allocations. On discrete-GPU WSL, "
                "this report does not add separate VRAM to process PSS."
            ),
        },
        "framework_metrics": {
            "edge": edge_metrics,
            "cloud": cloud_metrics,
        },
        "timestamp_records": timestamp_records,
        "review_lifecycle": {
            "all_completed": bool(records)
            and all(
                record.get("review_state") == "completed"
                for record in records
            ),
            "records": review_records,
        },
        "cloud_aggregations": aggregation_records,
        "failures": failures,
        "records": records,
        "limitations": [
            "Localhost cloud transport is a deployment benchmark, not a public-WAN test.",
            "The online cloud coordinator is ExtraTrees; Qwen 9B review must be measured separately.",
            "Decision fidelity and SUMO physical control benefit are separate evaluations.",
        ],
    }
    _write_json(output_path, result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_build_report(result), encoding="utf-8")
    return result


def main() -> None:
    args = _parse_args()
    result = run(args)
    summary = {
        "run_id": result["run_id"],
        "sample_count": result["sample_count"],
        "execution": result["execution"],
        "latency": result["latency"],
        "memory": result["memory"],
        "artifacts_dir": result["artifacts_dir"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if result["execution"]["success_rate"] < 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
