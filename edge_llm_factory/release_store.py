"""用途：原子发布、查询和回滚通过门禁的边缘大模型版本，并保留完整审计记录。"""

import argparse
import fcntl
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

from edge_llm_factory.adapter_package import MANIFEST_NAME, validate_adapter_package
from edge_llm_factory.contracts import (
    ManifestError,
    canonical_sha256,
    read_json_object,
    sha256_file,
)


RELEASE_STORE_SCHEMA = "edge-llm-release-store/v1"
RELEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}$")


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
        binding = {
            "base_manifest_sha256": sha256_file(base_path),
            "base_fingerprint": validation["base_fingerprint"],
            "adapter_package_sha256": package_digest["sha256"],
            "adapter_sha256": validation["adapter_sha256"],
            "deployment_sha256": artifact_sha,
            "adapter_id": validation["adapter_id"],
            "adapter_version": validation["version"],
        }
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
        }

    @staticmethod
    def _verify_release(record: Mapping[str, Any]) -> Dict[str, Any]:
        base = Path(record["base_manifest"]["path"])
        package = Path(record["adapter_package"]["path"])
        artifact = Path(record["deployment_artifact"]["path"])
        if not base.is_file() or sha256_file(base) != record["base_manifest"]["sha256"]:
            raise ManifestError("发布版本的基座 manifest 已变化")
        package_digest = _directory_digest(package)
        for field in ("sha256", "bytes", "file_count"):
            if package_digest[field] != record["adapter_package"][field]:
                raise ManifestError("发布版本的适配器包已变化: {}".format(field))
        if not artifact.is_file() or artifact.is_symlink():
            raise ManifestError("发布版本的部署 GGUF 不存在或是符号链接")
        if artifact.stat().st_size != record["deployment_artifact"]["bytes"]:
            raise ManifestError("发布版本的部署 GGUF 大小已变化")
        if sha256_file(artifact) != record["deployment_artifact"]["sha256"]:
            raise ManifestError("发布版本的部署 GGUF SHA256 已变化")
        validation = validate_adapter_package(package, base, require_gates=True)
        if validation["base_fingerprint"] != record["base_manifest"]["fingerprint"]:
            raise ManifestError("发布版本的基座指纹已变化")
        return {
            "status": "verified",
            "release_id": record["release_id"],
            "binding_fingerprint": record["binding_fingerprint"],
        }

    @staticmethod
    def _activate(
        state: Dict[str, Any], release_id: str, action: str
    ) -> Dict[str, Any]:
        previous = state["active_release_id"]
        state["active_release_id"] = release_id
        state["revision"] += 1
        state["history"].append(
            {
                "sequence": state["revision"],
                "action": action,
                "from_release_id": previous,
                "to_release_id": release_id,
                "at_utc": _now_utc(),
            }
        )
        return state

    def promote(
        self,
        release_id: str,
        base_manifest_path: Path,
        adapter_package: Path,
        deployment_artifact: Path,
    ) -> Dict[str, Any]:
        candidate = self._release_record(
            release_id, base_manifest_path, adapter_package, deployment_artifact
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
            self._activate(state, target, "rollback")
            _atomic_write(self.registry_path, state)
            return {
                "status": "rolled_back",
                "registry": str(self.registry_path),
                "active_release_id": target,
                "revision": state["revision"],
                "verification": verification,
                "release": state["releases"][target],
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

    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("--registry", required=True)
    rollback.add_argument("--release-id", "--release_id", default=None)

    status = subparsers.add_parser("status")
    status.add_argument("--registry", required=True)
    status.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)

    store = ReleaseStore(Path(args.registry))
    if args.command == "promote":
        result = store.promote(
            args.release_id,
            Path(args.base),
            Path(args.package),
            Path(args.deployment_artifact),
        )
    elif args.command == "rollback":
        result = store.rollback(args.release_id)
    else:
        result = store.status(verify_active=not args.no_verify)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
