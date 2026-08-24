"""用途：原子发布、查询和回滚通过门禁的边缘大模型版本，并保留完整审计记录。"""

import argparse
import fcntl
import json
import os
import re
import stat
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from edge_llm_factory.adapter_package import MANIFEST_NAME, validate_adapter_package
from edge_llm_factory.contracts import (
    ManifestError,
    base_fingerprint,
    canonical_sha256,
    read_json_object,
    sha256_file,
    validate_base_manifest,
)


RELEASE_STORE_SCHEMA = "edge-llm-release-store/v1"
RELEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _directory_digest(path: Path) -> Dict[str, Any]:
    if not path.is_dir():
        raise ManifestError("适配器包目录不存在: {}".format(path))
    records = []
    total = 0
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        if item.is_symlink():
            raise ManifestError("发布目录禁止符号链接: {}".format(item))
        size = item.stat().st_size
        total += size
        records.append(
            {
                "path": str(item.relative_to(path)),
                "bytes": size,
                "sha256": sha256_file(item),
            }
        )
    return {
        "bytes": total,
        "file_count": len(records),
        "sha256": canonical_sha256(records),
    }


def _regular_file(path: Path, label: str) -> None:
    """Reject missing files, non-regular files and the symlink itself."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ManifestError("{}不存在: {}".format(label, path)) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ManifestError("{}必须是普通文件且不能是符号链接: {}".format(label, path))


def _validate_runtime_adapter_records(
    value: Any,
    *,
    verify_files: bool,
) -> List[Dict[str, Any]]:
    """Validate the immutable, id-ordered runtime LoRA binding."""
    if not isinstance(value, list):
        raise ManifestError("runtime_adapters 必须是数组")
    validated: List[Dict[str, Any]] = []
    for expected_id, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ManifestError("runtime_adapters 条目必须是对象")
        adapter_id = raw.get("id")
        if isinstance(adapter_id, bool) or not isinstance(adapter_id, int):
            raise ManifestError("runtime adapter id 必须是整数")
        if adapter_id != expected_id:
            raise ManifestError("runtime adapter id 必须从 0 连续递增")
        default_scale = raw.get("default_scale")
        if (
            isinstance(default_scale, bool)
            or not isinstance(default_scale, (int, float))
            or default_scale not in (0, 1)
        ):
            raise ManifestError("runtime adapter default_scale 必须严格为 0 或 1")
        path_text = raw.get("path")
        if not isinstance(path_text, str) or not path_text.strip():
            raise ManifestError("runtime adapter path 必须是非空字符串")
        byte_count = raw.get("bytes")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ManifestError("runtime adapter bytes 无效")
        digest = raw.get("sha256")
        if not isinstance(digest, str) or not SHA256_HEX.fullmatch(digest):
            raise ManifestError("runtime adapter sha256 无效")
        path = Path(path_text)
        if verify_files:
            _regular_file(path, "runtime adapter")
            if path.stat().st_size != byte_count:
                raise ManifestError(
                    "runtime adapter 大小已变化: id={}".format(adapter_id)
                )
            if sha256_file(path) != digest:
                raise ManifestError(
                    "runtime adapter SHA256 已变化: id={}".format(adapter_id)
                )
        validated.append(
            {
                "id": adapter_id,
                "path": path_text,
                "bytes": byte_count,
                "sha256": digest,
                "default_scale": int(default_scale),
            }
        )
    if any(item["default_scale"] == 1 for item in validated) and len(validated) != 1:
        raise ManifestError("default_scale=1 仅允许单个常驻联合 Adapter")
    return validated


def _capture_runtime_adapters(value: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    """Resolve and hash runtime adapters supplied during promotion."""
    if value is None:
        return []
    if isinstance(value, (str, bytes, Path)) or not isinstance(value, Sequence):
        raise ManifestError("runtime_adapters 必须是数组")
    records: List[Dict[str, Any]] = []
    for expected_id, raw in enumerate(value):
        if isinstance(raw, Mapping):
            adapter_id = raw.get("id", expected_id)
            path_value = raw.get("path")
            default_scale = raw.get("default_scale", 0)
            declared_bytes = raw.get("bytes")
            declared_sha = raw.get("sha256")
        else:
            adapter_id = expected_id
            path_value = raw
            default_scale = 0
            declared_bytes = None
            declared_sha = None
        if isinstance(adapter_id, bool) or not isinstance(adapter_id, int):
            raise ManifestError("runtime adapter id 必须是整数")
        if adapter_id != expected_id:
            raise ManifestError("runtime adapter id 必须从 0 连续递增")
        if (
            isinstance(default_scale, bool)
            or not isinstance(default_scale, (int, float))
            or default_scale not in (0, 1)
        ):
            raise ManifestError("runtime adapter default_scale 必须严格为 0 或 1")
        if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
            raise ManifestError("runtime adapter path 必须是非空路径")
        unresolved = Path(path_value).expanduser()
        _regular_file(unresolved, "runtime adapter")
        path = unresolved.resolve()
        _regular_file(path, "runtime adapter")
        byte_count = path.stat().st_size
        digest = sha256_file(path)
        if declared_bytes is not None and declared_bytes != byte_count:
            raise ManifestError(
                "runtime adapter 声明大小与文件不一致: id={}".format(adapter_id)
            )
        if declared_sha is not None and declared_sha != digest:
            raise ManifestError(
                "runtime adapter 声明 SHA256 与文件不一致: id={}".format(adapter_id)
            )
        records.append(
            {
                "id": adapter_id,
                "path": str(path),
                "bytes": byte_count,
                "sha256": digest,
                "default_scale": int(default_scale),
            }
        )
    if any(item["default_scale"] == 1 for item in records) and len(records) != 1:
        raise ManifestError("default_scale=1 仅允许单个常驻联合 Adapter")
    return records


def _binding_from_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    if record.get("deployment_mode") == "base_only":
        return {
            "deployment_mode": "base_only",
            "base_manifest_sha256": record["base_manifest"]["sha256"],
            "base_fingerprint": record["base_manifest"]["fingerprint"],
            "deployment_sha256": record["deployment_artifact"]["sha256"],
            "runtime_adapters": [],
        }
    binding = {
        "base_manifest_sha256": record["base_manifest"]["sha256"],
        "base_fingerprint": record["base_manifest"]["fingerprint"],
        "adapter_package_sha256": record["adapter_package"]["sha256"],
        "adapter_sha256": record["adapter"]["adapter_sha256"],
        "deployment_sha256": record["deployment_artifact"]["sha256"],
        "adapter_id": record["adapter"]["adapter_id"],
        "adapter_version": record["adapter"]["version"],
    }
    runtime_adapters = record.get("runtime_adapters", [])
    if runtime_adapters:
        binding["runtime_adapters"] = [
            {
                "id": item["id"],
                "sha256": item["sha256"],
                "bytes": item["bytes"],
                "default_scale": item["default_scale"],
            }
            for item in runtime_adapters
        ]
    return binding


def _runtime_binding_audit(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "deployment_mode": record.get("deployment_mode", "adapter_package"),
        "binding_fingerprint": record.get("binding_fingerprint"),
        "deployment_sha256": record.get("deployment_artifact", {}).get("sha256"),
        "runtime_adapters": [
            {
                "id": item.get("id"),
                "sha256": item.get("sha256"),
                "default_scale": item.get("default_scale"),
            }
            for item in record.get("runtime_adapters", [])
            if isinstance(item, dict)
        ],
    }


def _empty_state() -> Dict[str, Any]:
    return {
        "schema_version": RELEASE_STORE_SCHEMA,
        "revision": 0,
        "active_release_id": None,
        "release_order": [],
        "releases": {},
        "history": [],
    }


def _validate_state(value: Mapping[str, Any]) -> Dict[str, Any]:
    state = dict(value)
    if state.get("schema_version") != RELEASE_STORE_SCHEMA:
        raise ManifestError("release store schema_version 无效")
    if not isinstance(state.get("revision"), int) or state["revision"] < 0:
        raise ManifestError("release store revision 无效")
    if not isinstance(state.get("releases"), dict):
        raise ManifestError("release store releases 必须是对象")
    if not isinstance(state.get("release_order"), list):
        raise ManifestError("release store release_order 必须是数组")
    release_order = state["release_order"]
    if not all(
        isinstance(value, str) and RELEASE_ID.fullmatch(value)
        for value in release_order
    ):
        raise ManifestError("release store release_order 含非法版本 ID")
    if not all(isinstance(key, str) for key in state["releases"]):
        raise ManifestError("release store releases 键必须是字符串")
    if len(state["release_order"]) != len(set(state["release_order"])):
        raise ManifestError("release store release_order 含重复版本")
    if set(state["release_order"]) != set(state["releases"]):
        raise ManifestError("release_order 与 releases 不一致")
    for release_id, record in state["releases"].items():
        if not isinstance(record, dict):
            raise ManifestError("release store release 条目必须是对象")
        if record.get("release_id") != release_id:
            raise ManifestError("release store release_id 与键不一致")
        deployment_mode = record.get("deployment_mode")
        if deployment_mode is not None and deployment_mode != "base_only":
            raise ManifestError("release store deployment_mode 无效")
        if deployment_mode == "base_only":
            if "adapter" in record or "adapter_package" in record:
                raise ManifestError("base-only release 禁止包含 adapter/package")
            if record.get("runtime_adapters") != []:
                raise ManifestError("base-only release 必须固定 runtime_adapters=[]")
            for field in ("base_manifest", "deployment_artifact"):
                if not isinstance(record.get(field), dict):
                    raise ManifestError(
                        "base-only release 缺少 {}".format(field)
                    )
        if "runtime_adapters" in record:
            _validate_runtime_adapter_records(
                record["runtime_adapters"], verify_files=False
            )
    if not isinstance(state.get("history"), list):
        raise ManifestError("release store history 必须是数组")
    active = state.get("active_release_id")
    history = state["history"]
    if state["revision"] != len(history):
        raise ManifestError("release store revision 与 history 长度不一致")
    for index, entry in enumerate(history, start=1):
        if not isinstance(entry, dict):
            raise ManifestError("release store history 条目必须是对象")
        if entry.get("sequence") != index:
            raise ManifestError("release store history sequence 不连续")
        if entry.get("action") not in {"promote", "rollback"}:
            raise ManifestError("release store history action 无效")
        source = entry.get("from_release_id")
        target = entry.get("to_release_id")
        if source is not None and source not in state["releases"]:
            raise ManifestError("release store history 引用了未知来源版本")
        if target not in state["releases"]:
            raise ManifestError("release store history 引用了未知目标版本")
    if active is not None and active not in state["releases"]:
        raise ManifestError("active_release_id 引用了未知版本")
    return state


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    payload = json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as file_obj:
            file_obj.write(payload)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(str(temporary), str(path))
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


class ReleaseStore:
    """Persistent release pointer guarded by a process-safe file lock."""

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = Path(registry_path).resolve()
        self.lock_path = self.registry_path.with_name(self.registry_path.name + ".lock")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read(self) -> Dict[str, Any]:
        if not self.registry_path.exists():
            return _empty_state()
        return _validate_state(read_json_object(self.registry_path))

    @staticmethod
    def _release_record(
        release_id: str,
        base_manifest_path: Path,
        adapter_package: Path,
        deployment_artifact: Path,
        runtime_adapters: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        if not RELEASE_ID.fullmatch(release_id):
            raise ManifestError("release_id 格式无效: {}".format(release_id))
        base_path = base_manifest_path.resolve()
        package_path = adapter_package.resolve()
        artifact_path = deployment_artifact.resolve()
        validation = validate_adapter_package(
            package_path, base_path, require_gates=True
        )
        manifest = read_json_object(package_path / MANIFEST_NAME)
        deployment = manifest.get("deployment")
        if not isinstance(deployment, dict):
            raise ManifestError("适配器 manifest 缺少 deployment")
        if not artifact_path.is_file() or artifact_path.is_symlink():
            raise ManifestError("部署 GGUF 不存在或是符号链接: {}".format(artifact_path))
        artifact_sha = sha256_file(artifact_path)
        artifact_bytes = artifact_path.stat().st_size
        if artifact_sha != deployment.get("artifact_sha256"):
            raise ManifestError("部署 GGUF SHA256 与适配器 manifest 不一致")
        if artifact_bytes != deployment.get("artifact_bytes"):
            raise ManifestError("部署 GGUF 大小与适配器 manifest 不一致")
        package_digest = _directory_digest(package_path)
        runtime_adapter_records = _capture_runtime_adapters(runtime_adapters)
        binding = {
            "base_manifest_sha256": sha256_file(base_path),
            "base_fingerprint": validation["base_fingerprint"],
            "adapter_package_sha256": package_digest["sha256"],
            "adapter_sha256": validation["adapter_sha256"],
            "deployment_sha256": artifact_sha,
            "adapter_id": validation["adapter_id"],
            "adapter_version": validation["version"],
        }
        if runtime_adapter_records:
            binding["runtime_adapters"] = [
                {
                    "id": item["id"],
                    "sha256": item["sha256"],
                    "bytes": item["bytes"],
                    "default_scale": item["default_scale"],
                }
                for item in runtime_adapter_records
            ]
        return {
            "release_id": release_id,
            "created_at_utc": _now_utc(),
            "binding_fingerprint": canonical_sha256(binding),
            "base_manifest": {
                "path": str(base_path),
                "sha256": binding["base_manifest_sha256"],
                "fingerprint": binding["base_fingerprint"],
            },
            "adapter_package": {"path": str(package_path), **package_digest},
            "deployment_artifact": {
                "path": str(artifact_path),
                "sha256": artifact_sha,
                "bytes": artifact_bytes,
                "format": deployment.get("format"),
                "quantization": deployment.get("quantization"),
            },
            "adapter": {
                "adapter_id": validation["adapter_id"],
                "scene": validation["scene"],
                "version": validation["version"],
                "adapter_sha256": validation["adapter_sha256"],
                "metrics": validation["metrics"],
                "gate_results": validation["gate_results"],
            },
            "runtime_adapters": runtime_adapter_records,
        }

    @staticmethod
    def _base_only_release_record(
        release_id: str,
        base_manifest_path: Path,
        deployment_artifact: Path,
    ) -> Dict[str, Any]:
        """Capture one immutable official-base GGUF without adapter metadata."""
        if not RELEASE_ID.fullmatch(release_id):
            raise ManifestError("release_id 格式无效: {}".format(release_id))
        unresolved_base = Path(base_manifest_path).expanduser()
        _regular_file(unresolved_base, "基座 manifest")
        base_path = unresolved_base.resolve()
        _regular_file(base_path, "基座 manifest")
        base = validate_base_manifest(read_json_object(base_path))

        unresolved_artifact = Path(deployment_artifact).expanduser()
        _regular_file(unresolved_artifact, "部署 GGUF")
        artifact_path = unresolved_artifact.resolve()
        _regular_file(artifact_path, "部署 GGUF")
        if artifact_path.suffix.lower() != ".gguf":
            raise ManifestError("base-only 部署产物必须是 GGUF")

        base_sha = sha256_file(base_path)
        artifact_sha = sha256_file(artifact_path)
        binding = {
            "deployment_mode": "base_only",
            "base_manifest_sha256": base_sha,
            "base_fingerprint": base_fingerprint(base),
            "deployment_sha256": artifact_sha,
            "runtime_adapters": [],
        }
        return {
            "release_id": release_id,
            "created_at_utc": _now_utc(),
            "deployment_mode": "base_only",
            "binding_fingerprint": canonical_sha256(binding),
            "base_manifest": {
                "path": str(base_path),
                "sha256": base_sha,
                "fingerprint": binding["base_fingerprint"],
            },
            "deployment_artifact": {
                "path": str(artifact_path),
                "sha256": artifact_sha,
                "bytes": artifact_path.stat().st_size,
                "format": "gguf",
            },
            "runtime_adapters": [],
        }

    @staticmethod
    def _verify_release(record: Mapping[str, Any]) -> Dict[str, Any]:
        base = Path(record["base_manifest"]["path"])
        artifact = Path(record["deployment_artifact"]["path"])
        _regular_file(base, "发布版本的基座 manifest")
        if sha256_file(base) != record["base_manifest"]["sha256"]:
            raise ManifestError("发布版本的基座 manifest 已变化")
        base_data = validate_base_manifest(read_json_object(base))
        if base_fingerprint(base_data) != record["base_manifest"]["fingerprint"]:
            raise ManifestError("发布版本的基座指纹已变化")
        if not artifact.is_file() or artifact.is_symlink():
            raise ManifestError("发布版本的部署 GGUF 不存在或是符号链接")
        if artifact.stat().st_size != record["deployment_artifact"]["bytes"]:
            raise ManifestError("发布版本的部署 GGUF 大小已变化")
        if sha256_file(artifact) != record["deployment_artifact"]["sha256"]:
            raise ManifestError("发布版本的部署 GGUF SHA256 已变化")
        if record.get("deployment_mode") == "base_only":
            if "adapter" in record or "adapter_package" in record:
                raise ManifestError("base-only release 禁止包含 adapter/package")
            if record.get("runtime_adapters") != []:
                raise ManifestError("base-only release 必须固定 runtime_adapters=[]")
            if canonical_sha256(_binding_from_record(record)) != record.get(
                "binding_fingerprint"
            ):
                raise ManifestError("发布版本的绑定指纹已变化")
            return {
                "status": "verified",
                "release_id": record["release_id"],
                "binding_fingerprint": record["binding_fingerprint"],
                "deployment_mode": "base_only",
            }

        package = Path(record["adapter_package"]["path"])
        package_digest = _directory_digest(package)
        for field in ("sha256", "bytes", "file_count"):
            if package_digest[field] != record["adapter_package"][field]:
                raise ManifestError("发布版本的适配器包已变化: {}".format(field))
        validation = validate_adapter_package(package, base, require_gates=True)
        if validation["base_fingerprint"] != record["base_manifest"]["fingerprint"]:
            raise ManifestError("发布版本的基座指纹已变化")
        runtime_adapters = _validate_runtime_adapter_records(
            record.get("runtime_adapters", []), verify_files=True
        )
        binding_record = dict(record)
        binding_record["runtime_adapters"] = runtime_adapters
        if canonical_sha256(_binding_from_record(binding_record)) != record.get(
            "binding_fingerprint"
        ):
            raise ManifestError("发布版本的绑定指纹已变化")
        return {
            "status": "verified",
            "release_id": record["release_id"],
            "binding_fingerprint": record["binding_fingerprint"],
        }

    @staticmethod
    def _activate(
        state: Dict[str, Any],
        release_id: str,
        action: str,
        audit: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        previous = state["active_release_id"]
        state["active_release_id"] = release_id
        state["revision"] += 1
        entry = {
            "sequence": state["revision"],
            "action": action,
            "from_release_id": previous,
            "to_release_id": release_id,
            "at_utc": _now_utc(),
        }
        if audit is not None:
            entry["audit"] = dict(audit)
        state["history"].append(entry)
        return state

    def promote(
        self,
        release_id: str,
        base_manifest_path: Path,
        adapter_package: Path,
        deployment_artifact: Path,
        runtime_adapters: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        candidate = self._release_record(
            release_id,
            base_manifest_path,
            adapter_package,
            deployment_artifact,
            runtime_adapters=runtime_adapters,
        )
        with self._locked():
            state = self._read()
            existing = state["releases"].get(release_id)
            if existing is not None:
                if existing.get("binding_fingerprint") != candidate["binding_fingerprint"]:
                    raise ManifestError("同一 release_id 已绑定不同产物")
                self._verify_release(existing)
            else:
                state["releases"][release_id] = candidate
                state["release_order"].append(release_id)
            if state["active_release_id"] == release_id:
                return {
                    "status": "already_active",
                    "registry": str(self.registry_path),
                    "active_release_id": release_id,
                    "revision": state["revision"],
                    "release": state["releases"][release_id],
                }
            self._activate(state, release_id, "promote")
            _atomic_write(self.registry_path, state)
            return {
                "status": "promoted",
                "registry": str(self.registry_path),
                "active_release_id": release_id,
                "revision": state["revision"],
                "release": state["releases"][release_id],
            }

    def promote_base_only(
        self,
        release_id: str,
        base_manifest_path: Path,
        deployment_artifact: Path,
    ) -> Dict[str, Any]:
        """Atomically promote an immutable base GGUF with no package or LoRA."""
        candidate = self._base_only_release_record(
            release_id,
            base_manifest_path,
            deployment_artifact,
        )
        with self._locked():
            state = self._read()
            existing = state["releases"].get(release_id)
            if existing is not None:
                if existing.get("binding_fingerprint") != candidate["binding_fingerprint"]:
                    raise ManifestError("同一 release_id 已绑定不同产物")
                self._verify_release(existing)
            else:
                state["releases"][release_id] = candidate
                state["release_order"].append(release_id)
            if state["active_release_id"] == release_id:
                return {
                    "status": "already_active",
                    "registry": str(self.registry_path),
                    "active_release_id": release_id,
                    "revision": state["revision"],
                    "release": state["releases"][release_id],
                }
            self._activate(state, release_id, "promote")
            _atomic_write(self.registry_path, state)
            return {
                "status": "promoted",
                "registry": str(self.registry_path),
                "active_release_id": release_id,
                "revision": state["revision"],
                "release": state["releases"][release_id],
            }

    def rollback(self, release_id: Optional[str] = None) -> Dict[str, Any]:
        with self._locked():
            state = self._read()
            current = state["active_release_id"]
            if current is None:
                raise ManifestError("当前没有可回滚的活动版本")
            target = release_id
            if target is None:
                current_index = state["release_order"].index(current)
                if current_index == 0:
                    raise ManifestError("当前版本之前没有可回滚版本")
                target = state["release_order"][current_index - 1]
            if target == current:
                raise ManifestError("回滚目标不能是当前活动版本")
            if target not in state["releases"]:
                raise ManifestError("未知回滚版本: {}".format(target))
            verification = self._verify_release(state["releases"][target])
            audit = {
                "trigger": "explicit_rollback",
                "restored_release": _runtime_binding_audit(
                    state["releases"][target]
                ),
            }
            self._activate(state, target, "rollback", audit=audit)
            _atomic_write(self.registry_path, state)
            return {
                "status": "rolled_back",
                "registry": str(self.registry_path),
                "active_release_id": target,
                "revision": state["revision"],
                "verification": verification,
                "release": state["releases"][target],
            }

    def rollback_if_active(
        self,
        expected_release_id: str,
        expected_revision: int,
        release_id: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Rollback only if the failed candidate is still the active revision.

        The compare-and-swap guard prevents a slow supervisor from reverting a
        newer promotion that arrived while the candidate process was starting.
        The failure reason is retained in release history so a successful
        runtime failback does not erase why the candidate was rejected.
        """
        if not RELEASE_ID.fullmatch(str(expected_release_id)):
            raise ManifestError(
                "expected_release_id 格式无效: {}".format(expected_release_id)
            )
        if isinstance(expected_revision, bool) or not isinstance(
            expected_revision, int
        ):
            raise ManifestError("expected_revision 必须是整数")
        if expected_revision < 0:
            raise ManifestError("expected_revision 不能为负数")
        if not RELEASE_ID.fullmatch(str(release_id)):
            raise ManifestError("release_id 格式无效: {}".format(release_id))
        failure_reason = str(reason).strip() or "candidate apply failed"
        # Keep the registry bounded even when a runtime returns a very long error.
        failure_reason = failure_reason[:2048]

        with self._locked():
            state = self._read()
            current = state["active_release_id"]
            revision = int(state["revision"])
            if current != expected_release_id or revision != expected_revision:
                return {
                    "status": "rollback_skipped",
                    "reason": "active_release_changed",
                    "registry": str(self.registry_path),
                    "expected_release_id": expected_release_id,
                    "expected_revision": expected_revision,
                    "active_release_id": current,
                    "revision": revision,
                }
            if release_id == current:
                raise ManifestError("回滚目标不能是当前活动版本")
            if release_id not in state["releases"]:
                raise ManifestError("未知回滚版本: {}".format(release_id))
            verification = self._verify_release(state["releases"][release_id])
            audit = {
                "trigger": "candidate_apply_failure",
                "failed_release_id": expected_release_id,
                "failed_revision": expected_revision,
                "error": failure_reason,
                "failed_release": _runtime_binding_audit(
                    state["releases"][expected_release_id]
                ),
                "restored_release": _runtime_binding_audit(
                    state["releases"][release_id]
                ),
            }
            self._activate(state, release_id, "rollback", audit=audit)
            _atomic_write(self.registry_path, state)
            return {
                "status": "rolled_back",
                "reason": "candidate_apply_failure",
                "registry": str(self.registry_path),
                "active_release_id": release_id,
                "revision": state["revision"],
                "verification": verification,
                "audit": audit,
                "release": state["releases"][release_id],
            }

    def status(self, verify_active: bool = True) -> Dict[str, Any]:
        with self._locked():
            state = self._read()
            result: Dict[str, Any] = {
                "status": "ok",
                "registry": str(self.registry_path),
                **state,
            }
            active = state["active_release_id"]
            result["active_integrity"] = (
                self._verify_release(state["releases"][active])
                if verify_active and active is not None
                else None
            )
            return result


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="发布、查询或回滚边缘大模型版本。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--registry", required=True)
    promote.add_argument("--release-id", "--release_id", required=True)
    promote.add_argument("--base", required=True)
    promote.add_argument("--package", required=True)
    promote.add_argument("--deployment-artifact", "--deployment_artifact", required=True)
    promote.add_argument(
        "--runtime-adapter",
        action="append",
        default=[],
        help=(
            "随此 release 预载的 llama.cpp LoRA GGUF；可重复，id 按顺序从 0 "
            "连续分配。"
        ),
    )
    promote.add_argument(
        "--runtime-adapter-default-scale",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "runtime adapter 的进程默认 scale，默认0（请求级选择）；"
            "1仅允许单个常驻联合 Adapter。"
        ),
    )

    promote_base = subparsers.add_parser("promote-base-only")
    promote_base.add_argument("--registry", required=True)
    promote_base.add_argument("--release-id", "--release_id", required=True)
    promote_base.add_argument("--base", required=True)
    promote_base.add_argument(
        "--deployment-artifact", "--deployment_artifact", required=True
    )

    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("--registry", required=True)
    rollback.add_argument("--release-id", "--release_id", default=None)

    status = subparsers.add_parser("status")
    status.add_argument("--registry", required=True)
    status.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)

    store = ReleaseStore(Path(args.registry))
    if args.command == "promote":
        runtime_adapters = [
            {
                "id": adapter_id,
                "path": Path(path),
                "default_scale": args.runtime_adapter_default_scale,
            }
            for adapter_id, path in enumerate(args.runtime_adapter)
        ]
        result = store.promote(
            args.release_id,
            Path(args.base),
            Path(args.package),
            Path(args.deployment_artifact),
            runtime_adapters=runtime_adapters,
        )
    elif args.command == "promote-base-only":
        result = store.promote_base_only(
            args.release_id,
            Path(args.base),
            Path(args.deployment_artifact),
        )
    elif args.command == "rollback":
        result = store.rollback(args.release_id)
    else:
        result = store.status(verify_active=not args.no_verify)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
