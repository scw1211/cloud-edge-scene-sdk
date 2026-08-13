"""用途：真实验证云端发布到边缘应用、回滚和确认回传的完整模型更新链路。

云端通过 HTTP 发布不可变清单和模型文件；边缘逐文件下载并校验清单中的
SHA-256/字节数，在同一文件系统内原子发布下载目录，然后调用
``update_loop_evidence.run`` 作为唯一正式边缘执行器。候选模型完成真实健康检查
和动作推理后触发既有受控故障与 CAS 回滚。边缘最后把完整下载清单和本地正式
证据摘要通过带 HMAC 的真实 HTTP 请求回传云端，云端持久化后返回幂等确认。

本模块不接触任何场景插件，更不会改写工业场景代码。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlparse
import urllib.request

from edge_llm_factory.contracts import ManifestError, read_json_object, safe_relative_path, sha256_file
from edge_llm_factory.update_loop_evidence import (
    EVIDENCE_SCHEMA,
    FAULT_MODE,
    EvidenceReleaseLlamaServer,
    _git_identity,
    _package_identity,
    run as run_local_update_loop,
)


PUBLICATION_SCHEMA = "edge-llm-cloud-publication/v1"
RECEIPT_SCHEMA = "edge-llm-edge-receipt/v1"
ACK_SCHEMA = "edge-llm-cloud-ack/v1"
DISTRIBUTED_EVIDENCE_SCHEMA = "edge-llm-distributed-update-evidence/v1"
CACHE_MARKER_SCHEMA = "edge-llm-download-cache/v1"
RECEIPT_STORE_SCHEMA = "edge-llm-cloud-receipt-store/v1"
API_PREFIX = "/api/v1/model-updates"
MAX_RECEIPT_BYTES = 2 * 1024 * 1024


class ReceiptConflict(ManifestError):
    """The same edge/run identity tried to submit a different receipt."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hmac_sha256(secret: bytes, value: bytes) -> str:
    return hmac.new(secret, value, hashlib.sha256).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp.{}.{}".format(path.name, os.getpid(), secrets.token_hex(4)))
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_bytes(value))


def _read_secret(path: Path, *, formal: bool) -> bytes:
    resolved = path.resolve()
    if not resolved.is_file() or (formal and path.is_symlink()):
        raise ManifestError("共享密钥文件不存在或正式模式禁止符号链接: {}".format(path))
    secret = resolved.read_bytes().rstrip(b"\r\n")
    if len(secret) < 32:
        raise ManifestError("共享密钥至少需要 32 字节")
    return secret


def _validate_http_base_url(value: str) -> str:
    parsed = urlparse(str(value).strip())
    if parsed.scheme != "http" or not parsed.hostname or parsed.query or parsed.fragment:
        raise ManifestError("cloud-url 必须是明确的 http://host:port 地址")
    if parsed.path not in {"", "/"}:
        raise ManifestError("cloud-url 不能包含业务路径")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ManifestError("cloud-url 端口无效") from exc
    if port is None or port <= 0 or port > 65535:
        raise ManifestError("cloud-url 必须显式包含有效端口")
    return "http://{}:{}".format(parsed.hostname, port)


def _validate_identifier(value: Any, field: str) -> str:
    text = str(value).strip()
    if not text or len(text) > 128:
        raise ManifestError("{} 必须是 1-128 字符的非空标识".format(field))
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._@-")
    if any(character not in allowed for character in text):
        raise ManifestError("{} 含有不允许的字符".format(field))
    return text


def _require_outside_repository(path: Path, repository_root: Path, field: str) -> None:
    resolved = path.resolve()
    root = repository_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return
    raise ManifestError("正式 {} 必须位于 Git 仓库之外".format(field))


def _distributed_git_identity(repository_root: Path, expected_commit: str) -> Dict[str, Any]:
    """Bind formal evidence to this runner in addition to the local executor."""
    identity = _git_identity(repository_root, expected_commit)
    root = repository_root.resolve()
    source_sha256 = dict(identity.get("source_sha256", {}))
    for path in (
        Path(__file__).resolve(),
        (root / "scripts" / "measure_distributed_model_update_loop.py").resolve(),
    ):
        relative = str(path.relative_to(root))
        completed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        if completed.returncode != 0:
            raise ManifestError("正式云边取证关键源码未纳入 Git: {}".format(relative))
        committed = subprocess.run(
            ["git", "show", "HEAD:{}".format(relative)],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        actual = path.read_bytes()
        if committed.returncode != 0 or committed.stdout != actual:
            raise ManifestError("正式云边取证源码与 Git 提交不一致: {}".format(relative))
        source_sha256[relative] = hashlib.sha256(actual).hexdigest()
    return {**identity, "source_sha256": source_sha256}


def _public_identity(identity: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "adapter_id": identity["adapter_id"],
        "adapter_version": identity["adapter_version"],
        "scene": identity["scene"],
        "base_fingerprint": identity["base_fingerprint"],
        "base_sha256": identity["base"]["sha256"],
        "package_sha256": identity["package"]["sha256"],
        "artifact_sha256": identity["artifact"]["sha256"],
        "artifact_bytes": identity["artifact"]["bytes"],
        "decision_contract": identity["decision_contract"],
        "gate_results": identity["gate_results"],
    }


def _assert_compatible_releases(old: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    if old["artifact"]["sha256"] == candidate["artifact"]["sha256"]:
        raise ManifestError("旧模型与候选模型的 GGUF SHA-256 必须不同")
    if (old["adapter_id"], old["adapter_version"]) == (
        candidate["adapter_id"],
        candidate["adapter_version"],
    ):
        raise ManifestError("旧版本与候选版本适配器 ID/版本必须不同")
    for field in ("scene", "base_fingerprint"):
        if old[field] != candidate[field]:
            raise ManifestError("旧版本与候选版本 {} 不兼容".format(field))
    if old["decision_contract"] != candidate["decision_contract"]:
        raise ManifestError("旧版本与候选版本决策契约不兼容")
    if not old["gate_results"] or not candidate["gate_results"]:
        raise ManifestError("旧版本与候选版本都必须包含发布门槛")
    if not all(row.get("passed") is True for row in old["gate_results"]):
        raise ManifestError("旧版本存在未通过的发布门槛")
    if not all(row.get("passed") is True for row in candidate["gate_results"]):
        raise ManifestError("候选版本存在未通过的发布门槛")


def _file_record(
    role: str,
    kind: str,
    relative_path: str,
    source: Path,
    index: int,
) -> Dict[str, Any]:
    safe_relative_path(relative_path, "publication file target_path")
    if not source.is_file() or source.is_symlink():
        raise ManifestError("发布文件不存在或是符号链接: {}".format(source))
    file_id = "{}-{}-{:04d}".format(role, kind.replace("_", "-"), index)
    return {
        "file_id": file_id,
        "role": role,
        "kind": kind,
        "target_path": relative_path,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "download_path": "{}/files/{}".format("__RUN_PATH__", file_id),
    }


def _release_files(role: str, base: Path, package: Path, artifact: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Path]]:
    records: List[Dict[str, Any]] = []
    sources: Dict[str, Path] = {}

    base_record = _file_record(role, "base_manifest", "{}/base/{}".format(role, base.name), base, 0)
    records.append(base_record)
    sources[base_record["file_id"]] = base

    for index, path in enumerate(sorted(item for item in package.rglob("*") if item.is_file()), start=1):
        if path.is_symlink():
            raise ManifestError("发布包禁止符号链接: {}".format(path))
        relative = str(path.relative_to(package))
        record = _file_record(
            role,
            "package_file",
            "{}/package/{}".format(role, relative),
            path,
            index,
        )
        records.append(record)
        sources[record["file_id"]] = path

    artifact_record = _file_record(
        role,
        "gguf",
        "{}/artifact/{}".format(role, artifact.name),
        artifact,
        len(records) + 1,
    )
    records.append(artifact_record)
    sources[artifact_record["file_id"]] = artifact
    return records, sources


def build_cloud_publication(config: Mapping[str, Any], *, check_git: bool) -> Tuple[Dict[str, Any], Dict[str, Path], Dict[str, Any]]:
    run_id = _validate_identifier(config["run_id"], "run_id")
    old_id = _validate_identifier(config["old_release_id"], "old_release_id")
    candidate_id = _validate_identifier(config["candidate_release_id"], "candidate_release_id")
    expected_edge_id = _validate_identifier(config["expected_edge_id"], "expected_edge_id")
    expected_hardware_id = _validate_identifier(
        config["expected_hardware_id"], "expected_hardware_id"
    )
    expected_dataset_id = _validate_identifier(
        config["expected_dataset_id"], "expected_dataset_id"
    )
    if old_id == candidate_id:
        raise ManifestError("旧版本与候选版本 release_id 必须不同")

    old_base = Path(config["old_base"]).resolve()
    old_package = Path(config["old_package"]).resolve()
    old_artifact = Path(config["old_artifact"]).resolve()
    candidate_base = Path(config["candidate_base"]).resolve()
    candidate_package = Path(config["candidate_package"]).resolve()
    candidate_artifact = Path(config["candidate_artifact"]).resolve()
    old_identity = _package_identity(old_base, old_package, old_artifact)
    candidate_identity = _package_identity(candidate_base, candidate_package, candidate_artifact)
    _assert_compatible_releases(old_identity, candidate_identity)

    git_identity = (
        _distributed_git_identity(
            Path(config["repository_root"]), str(config["git_commit"])
        )
        if check_git
        else {
            "repository_root": str(Path(config["repository_root"]).resolve()),
            "git_commit": str(config["git_commit"]),
            "worktree_clean": None,
            "source_sha256": {},
        }
    )
    nonce = str(config.get("publication_nonce") or secrets.token_hex(32))
    if len(nonce) < 32:
        raise ManifestError("publication_nonce 长度不足")

    old_files, old_sources = _release_files("old", old_base, old_package, old_artifact)
    candidate_files, candidate_sources = _release_files(
        "candidate", candidate_base, candidate_package, candidate_artifact
    )
    run_path = "{}/{}".format(API_PREFIX, quote(run_id, safe=""))
    files = old_files + candidate_files
    for row in files:
        row["download_path"] = row["download_path"].replace("__RUN_PATH__", run_path)
    publication = {
        "schema_version": PUBLICATION_SCHEMA,
        "status": "published",
        "execution_mode": "real_http" if check_git else "test_http",
        "run_id": run_id,
        "publication_nonce": nonce,
        "expected_edge_id": expected_edge_id,
        "expected_hardware_id": expected_hardware_id,
        "expected_dataset_id": expected_dataset_id,
        "published_at_utc": str(config.get("published_at_utc") or _now_utc()),
        "publisher": {
            "git_commit": git_identity["git_commit"],
            "source_sha256": git_identity.get("source_sha256", {}),
        },
        "release_order": [old_id, candidate_id, old_id],
        "releases": {
            "old": {"release_id": old_id, **_public_identity(old_identity)},
            "candidate": {
                "release_id": candidate_id,
                **_public_identity(candidate_identity),
            },
        },
        "files": files,
    }
    return publication, {**old_sources, **candidate_sources}, git_identity


def _validate_publication(publication: Mapping[str, Any], expected_run_id: str) -> None:
    if publication.get("schema_version") != PUBLICATION_SCHEMA:
        raise ManifestError("云端发布清单 schema_version 无效")
    if publication.get("status") != "published" or publication.get("run_id") != expected_run_id:
        raise ManifestError("云端发布清单状态或 run_id 不匹配")
    _validate_identifier(publication.get("expected_edge_id"), "expected_edge_id")
    _validate_identifier(publication.get("expected_hardware_id"), "expected_hardware_id")
    _validate_identifier(publication.get("expected_dataset_id"), "expected_dataset_id")
    if not isinstance(publication.get("publication_nonce"), str) or len(publication["publication_nonce"]) < 32:
        raise ManifestError("云端发布清单 nonce 无效")
    releases = publication.get("releases")
    if not isinstance(releases, Mapping) or set(releases) != {"old", "candidate"}:
        raise ManifestError("云端发布清单必须包含 old/candidate")
    old_id = _validate_identifier(releases["old"].get("release_id"), "old release_id")
    candidate_id = _validate_identifier(
        releases["candidate"].get("release_id"), "candidate release_id"
    )
    if old_id == candidate_id or publication.get("release_order") != [old_id, candidate_id, old_id]:
        raise ManifestError("云端发布顺序无效")
    files = publication.get("files")
    if not isinstance(files, list) or not files:
        raise ManifestError("云端发布清单 files 必须是非空数组")
    ids = set()
    targets = set()
    roles = {"old": {"base_manifest": 0, "package_file": 0, "gguf": 0}, "candidate": {"base_manifest": 0, "package_file": 0, "gguf": 0}}
    for index, raw in enumerate(files):
        if not isinstance(raw, Mapping):
            raise ManifestError("publication.files[{}] 必须是对象".format(index))
        file_id = _validate_identifier(raw.get("file_id"), "publication file_id")
        target = str(safe_relative_path(raw.get("target_path"), "publication target_path"))
        if file_id in ids or target in targets:
            raise ManifestError("云端发布清单存在重复文件 ID 或目标路径")
        ids.add(file_id)
        targets.add(target)
        role = raw.get("role")
        kind = raw.get("kind")
        if role not in roles or kind not in roles[role]:
            raise ManifestError("云端发布文件 role/kind 无效")
        roles[role][kind] += 1
        if not isinstance(raw.get("bytes"), int) or raw["bytes"] <= 0:
            raise ManifestError("云端发布文件 bytes 无效")
        digest = str(raw.get("sha256", ""))
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ManifestError("云端发布文件 SHA-256 无效")
        expected_path = "{}/{}/files/{}".format(API_PREFIX, quote(expected_run_id, safe=""), file_id)
        if raw.get("download_path") != expected_path:
            raise ManifestError("云端发布文件 download_path 无效")
    for role, counts in roles.items():
        if counts["base_manifest"] != 1 or counts["gguf"] != 1 or counts["package_file"] < 1:
            raise ManifestError("{} 发布文件集合不完整".format(role))


class CloudUpdateService:
    """Durable cloud publication and idempotent edge-receipt store."""

    def __init__(
        self,
        config: Mapping[str, Any],
        secret: bytes,
        *,
        check_git: bool,
    ) -> None:
        self.formal_http = bool(check_git and type(self) is CloudUpdateService)
        self.secret = bytes(secret)
        if len(self.secret) < 32:
            raise ManifestError("共享密钥至少需要 32 字节")
        self.storage_dir = Path(config["storage_dir"]).resolve()
        repository_root = Path(config["repository_root"]).resolve()
        if check_git:
            _require_outside_repository(self.storage_dir, repository_root, "storage-dir")
        self.publication_path = self.storage_dir / "cloud_publication.json"
        effective_config = dict(config)
        if self.publication_path.exists():
            previous = read_json_object(self.publication_path)
            effective_config["publication_nonce"] = previous.get("publication_nonce")
            effective_config["published_at_utc"] = previous.get("published_at_utc")
        self.publication, self.sources, self.git_identity = build_cloud_publication(
            effective_config, check_git=check_git
        )
        _validate_publication(self.publication, str(self.publication["run_id"]))
        self.publication_body = _canonical_bytes(self.publication)
        self.publication_sha256 = _sha256_bytes(self.publication_body)
        self.key_id = hashlib.sha256(self.secret).hexdigest()[:16]
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        if self.publication_path.exists():
            if self.publication_path.read_bytes() != self.publication_body:
                raise ManifestError("storage-dir 已存在不同的云端发布清单")
        else:
            _atomic_bytes(self.publication_path, self.publication_body)
        self.receipt_store_path = self.storage_dir / "edge_receipts.json"
        if not self.receipt_store_path.exists():
            _atomic_json(
                self.receipt_store_path,
                {
                    "schema_version": RECEIPT_STORE_SCHEMA,
                    "run_id": self.publication["run_id"],
                    "publication_sha256": self.publication_sha256,
                    "revision": 0,
                    "receipts": {},
                },
            )
        if check_git:
            post_git = _distributed_git_identity(repository_root, str(config["git_commit"]))
            if post_git != self.git_identity:
                raise ManifestError("云端发布前后 Git 身份不一致")
        self.lock = threading.Lock()
        self.file_download_count = 0

    def manifest_response(self) -> Tuple[bytes, Dict[str, str]]:
        return self.publication_body, {
            "X-Content-SHA256": self.publication_sha256,
            "X-Manifest-Signature": _hmac_sha256(self.secret, self.publication_body),
            "X-Key-Id": self.key_id,
        }

    def file_source(self, file_id: str) -> Tuple[Path, Mapping[str, Any]]:
        record = next(
            (row for row in self.publication["files"] if row["file_id"] == file_id),
            None,
        )
        if record is None or file_id not in self.sources:
            raise KeyError(file_id)
        source = self.sources[file_id]
        if source.stat().st_size != record["bytes"] or sha256_file(source) != record["sha256"]:
            raise ManifestError("云端源文件在发布后发生变化: {}".format(file_id))
        self.file_download_count += 1
        return source, record

    def _load_store(self) -> Dict[str, Any]:
        store = read_json_object(self.receipt_store_path)
        if (
            store.get("schema_version") != RECEIPT_STORE_SCHEMA
            or store.get("run_id") != self.publication["run_id"]
            or store.get("publication_sha256") != self.publication_sha256
            or not isinstance(store.get("receipts"), dict)
        ):
            raise ManifestError("云端 edge receipt store 损坏或与发布不匹配")
        return store

    def _validate_receipt(self, receipt: Mapping[str, Any]) -> None:
        if receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("status") != "completed":
            raise ManifestError("边缘回执 schema/status 无效")
        if receipt.get("execution_mode") != "real_cloud_edge_http":
            raise ManifestError("测试替身回执不得作为正式回执")
        if receipt.get("run_id") != self.publication["run_id"]:
            raise ManifestError("边缘回执 run_id 不匹配")
        if receipt.get("edge_id") != self.publication["expected_edge_id"]:
            raise ManifestError("边缘回执 edge_id 不匹配")
        if receipt.get("hardware_id") != self.publication["expected_hardware_id"]:
            raise ManifestError("边缘回执 hardware_id 不匹配")
        if receipt.get("dataset_id") != self.publication["expected_dataset_id"]:
            raise ManifestError("边缘回执 dataset_id 不匹配")
        if receipt.get("git_commit") != self.publication["publisher"]["git_commit"]:
            raise ManifestError("边缘回执 git commit 与云端发布不匹配")
        if receipt.get("publication_nonce") != self.publication["publication_nonce"]:
            raise ManifestError("边缘回执 nonce 不匹配")
        if receipt.get("publication_sha256") != self.publication_sha256:
            raise ManifestError("边缘回执 publication SHA-256 不匹配")
        core = dict(receipt)
        receipt_id = str(core.pop("receipt_id", ""))
        if receipt_id != _sha256_bytes(_canonical_bytes(core)):
            raise ManifestError("边缘回执 receipt_id 与内容不一致")

        expected_files = {
            row["file_id"]: (row["sha256"], row["bytes"], row["target_path"])
            for row in self.publication["files"]
        }
        downloaded = receipt.get("downloaded_files")
        if not isinstance(downloaded, list) or len(downloaded) != len(expected_files):
            raise ManifestError("边缘回执下载文件集合不完整")
        observed = {}
        for row in downloaded:
            if not isinstance(row, Mapping):
                raise ManifestError("边缘下载文件回执必须是对象")
            file_id = str(row.get("file_id", ""))
            if file_id in observed:
                raise ManifestError("边缘下载文件回执存在重复 file_id")
            observed[file_id] = (row.get("sha256"), row.get("bytes"), row.get("target_path"))
        if observed != expected_files:
            raise ManifestError("边缘回执的下载文件 SHA/大小/路径与发布清单不一致")

        execution = receipt.get("local_execution")
        releases = self.publication["releases"]
        if not isinstance(execution, Mapping):
            raise ManifestError("边缘回执缺少本地执行证据摘要")
        required = {
            "execution_mode": "real_llama_server",
            "rollback_verified": True,
            "model_id_before": releases["old"]["release_id"],
            "model_id_applied": releases["candidate"]["release_id"],
            "model_id_after_rollback": releases["old"]["release_id"],
            "candidate_process_stopped": True,
            "runtime_cleanup_passed": True,
        }
        for field, expected in required.items():
            if execution.get(field) != expected:
                raise ManifestError("边缘本地执行摘要 {} 不满足正式要求".format(field))
        provenance = execution.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ManifestError("边缘本地执行摘要缺少 provenance")
        expected_provenance = {
            "git_commit": self.publication["publisher"]["git_commit"],
            "dataset_id": self.publication["expected_dataset_id"],
            "hardware_id": self.publication["expected_hardware_id"],
            "run_id": self.publication["run_id"],
        }
        for field, expected in expected_provenance.items():
            if provenance.get(field) != expected:
                raise ManifestError("边缘本地执行 provenance.{} 不匹配".format(field))
        if execution.get("transition_release_ids") != self.publication["release_order"]:
            raise ManifestError("边缘本地模型启动顺序与云端发布不一致")
        if execution.get("transition_revisions") != [1, 2, 3]:
            raise ManifestError("边缘本地隔离 registry revision 顺序无效")
        if execution.get("release_actions") != ["promote", "promote", "rollback"]:
            raise ManifestError("边缘本地 release history 不完整")
        if receipt.get("download_atomic_publish") is not True:
            raise ManifestError("边缘没有确认下载目录原子发布")
        for field in ("main_evidence_sha256", "completion_marker_sha256"):
            digest = str(execution.get(field, ""))
            if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
                raise ManifestError("边缘本地执行摘要 {} 无效".format(field))
        stage_hashes = execution.get("stage_sha256")
        if not isinstance(stage_hashes, Mapping) or set(stage_hashes) != {
            "package",
            "release",
            "apply",
            "rollback",
        }:
            raise ManifestError("边缘本地四阶段证据 SHA 集合不完整")
        for stage, digest in stage_hashes.items():
            text = str(digest)
            if len(text) != 64 or any(value not in "0123456789abcdef" for value in text):
                raise ManifestError("边缘本地 {} 阶段 SHA-256 无效".format(stage))

    def accept_receipt(self, body: bytes, signature: str) -> Tuple[bytes, Dict[str, str]]:
        expected_signature = _hmac_sha256(self.secret, body)
        if not hmac.compare_digest(str(signature).lower(), expected_signature):
            raise PermissionError("边缘回执 HMAC 校验失败")
        try:
            receipt = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestError("边缘回执不是合法 JSON") from exc
        if not isinstance(receipt, Mapping):
            raise ManifestError("边缘回执顶层必须是对象")
        self._validate_receipt(receipt)
        edge_id = str(receipt["edge_id"])
        receipt_sha = _sha256_bytes(body)
        with self.lock:
            store = self._load_store()
            existing = store["receipts"].get(edge_id)
            if existing is not None:
                if existing.get("receipt_sha256") != receipt_sha:
                    raise ReceiptConflict("同一 edge/run 已持久化不同回执")
                ack = existing["ack"]
            else:
                revision = int(store["revision"]) + 1
                ack = {
                    "schema_version": ACK_SCHEMA,
                    "status": "accepted",
                    "execution_mode": "real_http" if self.formal_http else "test_http",
                    "run_id": self.publication["run_id"],
                    "edge_id": edge_id,
                    "receipt_id": receipt["receipt_id"],
                    "receipt_sha256": receipt_sha,
                    "publication_sha256": self.publication_sha256,
                    "store_revision": revision,
                    "received_at_utc": _now_utc(),
                }
                store["revision"] = revision
                store["receipts"][edge_id] = {
                    "receipt_sha256": receipt_sha,
                    "receipt": dict(receipt),
                    "ack": ack,
                }
                _atomic_json(self.receipt_store_path, store)
                _atomic_json(self.storage_dir / "acks" / (edge_id + ".json"), ack)
        ack_body = _canonical_bytes(ack)
        return ack_body, {
            "X-Ack-Signature": _hmac_sha256(self.secret, ack_body),
            "X-Key-Id": self.key_id,
        }

    def status_response(self) -> Dict[str, Any]:
        with self.lock:
            store = self._load_store()
            return {
                "schema_version": DISTRIBUTED_EVIDENCE_SCHEMA,
                "status": "waiting" if not store["receipts"] else "edge_confirmed",
                "execution_mode": "real_http" if self.formal_http else "test_http",
                "run_id": self.publication["run_id"],
                "publication_sha256": self.publication_sha256,
                "receipt_count": len(store["receipts"]),
                "confirmed_edge_ids": sorted(store["receipts"]),
                "receipt_store_sha256": sha256_file(self.receipt_store_path),
            }


class _UpdateHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def create_cloud_http_server(service: CloudUpdateService, host: str, port: int) -> ThreadingHTTPServer:
    run_id = str(service.publication["run_id"])
    run_root = "{}/{}".format(API_PREFIX, quote(run_id, safe=""))

    class Handler(BaseHTTPRequestHandler):
        server_version = "CloudEdgeModelUpdate/1.0"

        def log_message(self, _format: str, *args: Any) -> None:
            del args

        def _send_bytes(
            self,
            status: int,
            payload: bytes,
            content_type: str,
            headers: Optional[Mapping[str, str]] = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            for name, value in dict(headers or {}).items():
                self.send_header(name, str(value))
            self.end_headers()
            self.wfile.write(payload)

        def _send_error_json(self, status: int, message: str) -> None:
            self._send_bytes(
                status,
                _canonical_bytes({"status": "error", "error": message}),
                "application/json; charset=utf-8",
            )

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                if path == run_root + "/manifest":
                    payload, headers = service.manifest_response()
                    self._send_bytes(HTTPStatus.OK, payload, "application/json", headers)
                    return
                if path == run_root + "/status":
                    payload = _canonical_bytes(service.status_response())
                    self._send_bytes(
                        HTTPStatus.OK,
                        payload,
                        "application/json",
                        {"X-Status-Signature": _hmac_sha256(service.secret, payload)},
                    )
                    return
                prefix = run_root + "/files/"
                if path.startswith(prefix):
                    file_id = unquote(path[len(prefix) :])
                    source, record = service.file_source(file_id)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(record["bytes"]))
                    self.send_header("X-Content-SHA256", record["sha256"])
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    with source.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            self.wfile.write(block)
                    return
                self._send_error_json(HTTPStatus.NOT_FOUND, "endpoint not found")
            except KeyError:
                self._send_error_json(HTTPStatus.NOT_FOUND, "file not found")
            except ManifestError as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path != run_root + "/receipts":
                self._send_error_json(HTTPStatus.NOT_FOUND, "endpoint not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
                return
            if length <= 0 or length > MAX_RECEIPT_BYTES:
                self._send_error_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "receipt size invalid")
                return
            body = self.rfile.read(length)
            try:
                ack, headers = service.accept_receipt(
                    body, self.headers.get("X-Receipt-Signature", "")
                )
                self._send_bytes(HTTPStatus.OK, ack, "application/json", headers)
            except PermissionError as exc:
                self._send_error_json(HTTPStatus.FORBIDDEN, str(exc))
            except ReceiptConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
            except ManifestError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    return _UpdateHTTPServer((host, int(port)), Handler)


def _manifest_url(cloud_url: str, run_id: str) -> str:
    return "{}{}/{}/manifest".format(
        _validate_http_base_url(cloud_url), API_PREFIX, quote(run_id, safe="")
    )


def _fetch_publication(cloud_url: str, run_id: str, secret: bytes, timeout_seconds: float) -> Tuple[Dict[str, Any], bytes, str]:
    url = _manifest_url(cloud_url, run_id)
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = response.read()
        status = int(response.status)
        content_sha = str(response.headers.get("X-Content-SHA256", "")).lower()
        signature = str(response.headers.get("X-Manifest-Signature", "")).lower()
    if status != 200 or content_sha != _sha256_bytes(body):
        raise ManifestError("云端发布清单 HTTP 状态或 SHA-256 无效")
    if not hmac.compare_digest(signature, _hmac_sha256(secret, body)):
        raise ManifestError("云端发布清单 HMAC 校验失败")
    try:
        publication = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError("云端发布清单不是合法 JSON") from exc
    if not isinstance(publication, Mapping):
        raise ManifestError("云端发布清单顶层必须是对象")
    _validate_publication(publication, run_id)
    return dict(publication), body, url


def _verify_download_tree(root: Path, publication: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records = []
    for row in publication["files"]:
        relative = safe_relative_path(row["target_path"], "download target_path")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ManifestError("边缘下载文件缺失或为符号链接: {}".format(relative))
        observed_bytes = path.stat().st_size
        observed_sha = sha256_file(path)
        if observed_bytes != row["bytes"] or observed_sha != row["sha256"]:
            raise ManifestError("边缘下载文件 SHA/大小不匹配: {}".format(relative))
        records.append(
            {
                "file_id": row["file_id"],
                "target_path": row["target_path"],
                "bytes": observed_bytes,
                "sha256": observed_sha,
            }
        )
    return records


def _download_release_paths(root: Path, publication: Mapping[str, Any], role: str) -> Dict[str, Path]:
    role_files = [row for row in publication["files"] if row["role"] == role]
    base_rows = [row for row in role_files if row["kind"] == "base_manifest"]
    artifact_rows = [row for row in role_files if row["kind"] == "gguf"]
    if len(base_rows) != 1 or len(artifact_rows) != 1:
        raise ManifestError("下载后的 {} 文件集合不完整".format(role))
    package = root / role / "package"
    return {
        "base": root / safe_relative_path(base_rows[0]["target_path"], "base target"),
        "package": package,
        "artifact": root / safe_relative_path(artifact_rows[0]["target_path"], "artifact target"),
    }


def _validate_downloaded_packages(root: Path, publication: Mapping[str, Any]) -> Dict[str, Dict[str, Path]]:
    result = {}
    for role in ("old", "candidate"):
        paths = _download_release_paths(root, publication, role)
        identity = _package_identity(paths["base"], paths["package"], paths["artifact"])
        expected = publication["releases"][role]
        observed = _public_identity(identity)
        for field, value in observed.items():
            if expected.get(field) != value:
                raise ManifestError("下载后的 {} package 身份字段 {} 不匹配".format(role, field))
        result[role] = paths
    return result


def download_publication(
    cloud_url: str,
    run_id: str,
    secret: bytes,
    cache_root: Path,
    *,
    timeout_seconds: float,
) -> Dict[str, Any]:
    publication, publication_body, manifest_url = _fetch_publication(
        cloud_url, run_id, secret, timeout_seconds
    )
    publication_sha = _sha256_bytes(publication_body)
    final_dir = cache_root.resolve() / publication_sha
    marker_path = final_dir / "DOWNLOAD_COMPLETE.json"
    if final_dir.exists():
        if not marker_path.is_file():
            raise ManifestError("边缘下载缓存目录已存在但缺少完成标记")
        marker = read_json_object(marker_path)
        if (
            marker.get("schema_version") != CACHE_MARKER_SCHEMA
            or marker.get("publication_sha256") != publication_sha
        ):
            raise ManifestError("边缘下载缓存完成标记与发布不匹配")
        downloaded_files = _verify_download_tree(final_dir, publication)
        release_paths = _validate_downloaded_packages(final_dir, publication)
        return {
            "publication": publication,
            "publication_sha256": publication_sha,
            "manifest_url": manifest_url,
            "cache_dir": str(final_dir),
            "cache_reused": True,
            "downloaded_files": downloaded_files,
            "release_paths": release_paths,
        }

    cache_root.resolve().mkdir(parents=True, exist_ok=True)
    staging = cache_root.resolve() / ".{}.in_progress.{}.{}".format(
        publication_sha, os.getpid(), secrets.token_hex(4)
    )
    staging.mkdir()
    downloaded_files = []
    try:
        _atomic_bytes(staging / "cloud_publication.json", publication_body)
        base_url = _validate_http_base_url(cloud_url)
        for row in publication["files"]:
            relative = safe_relative_path(row["target_path"], "download target_path")
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".part")
            request = urllib.request.Request(base_url + row["download_path"], method="GET")
            digest = hashlib.sha256()
            written = 0
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                if int(response.status) != 200:
                    raise ManifestError("模型文件下载 HTTP 状态不是 200")
                header_sha = str(response.headers.get("X-Content-SHA256", "")).lower()
                if header_sha != row["sha256"]:
                    raise ManifestError("模型文件下载响应 SHA header 与清单不一致")
                with temporary.open("xb") as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        written += len(block)
                        if written > row["bytes"]:
                            raise ManifestError("模型文件下载超过发布清单字节数")
                        digest.update(block)
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
            if written != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ManifestError("模型文件下载 SHA/字节数与发布清单不一致")
            os.replace(str(temporary), str(destination))
            _fsync_directory(destination.parent)
            downloaded_files.append(
                {
                    "file_id": row["file_id"],
                    "target_path": row["target_path"],
                    "bytes": written,
                    "sha256": digest.hexdigest(),
                }
            )
        release_paths = _validate_downloaded_packages(staging, publication)
        _atomic_json(
            staging / "DOWNLOAD_COMPLETE.json",
            {
                "schema_version": CACHE_MARKER_SCHEMA,
                "status": "completed",
                "publication_sha256": publication_sha,
                "file_count": len(downloaded_files),
                "completed_at_utc": _now_utc(),
            },
        )
        os.replace(str(staging), str(final_dir))
        _fsync_directory(final_dir.parent)
        release_paths = _validate_downloaded_packages(final_dir, publication)
        return {
            "publication": publication,
            "publication_sha256": publication_sha,
            "manifest_url": manifest_url,
            "cache_dir": str(final_dir),
            "cache_reused": False,
            "downloaded_files": downloaded_files,
            "release_paths": release_paths,
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _local_execution_summary(local_result: Mapping[str, Any]) -> Dict[str, Any]:
    if local_result.get("status") != "passed_evidence_generated":
        raise ManifestError("本地更新执行器没有生成正式通过证据")
    main_path = Path(str(local_result.get("main_evidence", ""))).resolve()
    completion_path = Path(str(local_result.get("completion_marker", ""))).resolve()
    if not main_path.is_file() or not completion_path.is_file():
        raise ManifestError("本地更新执行器主证据或完成标记不存在")
    main = read_json_object(main_path)
    completion = read_json_object(completion_path)
    if (
        main.get("schema_version") != EVIDENCE_SCHEMA
        or main.get("execution_mode") != "real_llama_server"
        or main.get("rollback_verified") is not True
        or completion.get("status") != "completed"
        or completion.get("execution_mode") != "real_llama_server"
    ):
        raise ManifestError("测试替身或不完整本地证据不得生成云边正式回执")
    provenance = main.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ManifestError("本地更新主证据缺少 provenance")
    main_sha = sha256_file(main_path)
    if main_sha != local_result.get("main_evidence_sha256") or completion.get("main_evidence_sha256") != main_sha:
        raise ManifestError("本地更新主证据 SHA-256 交叉校验失败")
    stage_refs = local_result.get("stage_evidence")
    if not isinstance(stage_refs, Mapping) or set(stage_refs) != {"package", "release", "apply", "rollback"}:
        raise ManifestError("本地更新四阶段证据引用不完整")
    stages = {}
    stage_hashes = {}
    for stage in ("package", "release", "apply", "rollback"):
        ref = stage_refs[stage]
        if not isinstance(ref, Mapping):
            raise ManifestError("本地更新 {} 阶段引用必须是对象".format(stage))
        path = Path(str(ref.get("path", ""))).resolve()
        if not path.is_file() or sha256_file(path) != ref.get("sha256"):
            raise ManifestError("本地更新 {} 阶段 SHA-256 无效".format(stage))
        value = read_json_object(path)
        if value.get("schema_version") != EVIDENCE_SCHEMA or value.get("stage") != stage or value.get("status") != "completed":
            raise ManifestError("本地更新 {} 阶段结构无效".format(stage))
        stages[stage] = value
        stage_hashes[stage] = ref["sha256"]
    apply = stages["apply"]
    rollback = stages["rollback"]
    release = stages["release"]
    transitions = list(apply.get("transitions", [])) + [rollback.get("rollback_transition")]
    if len(transitions) != 3 or not all(isinstance(row, Mapping) for row in transitions):
        raise ManifestError("本地更新真实模型启动证据必须恰好三次")
    if apply.get("candidate_health_and_inference_verified") is not True:
        raise ManifestError("候选模型未完成健康和推理验证")
    if rollback.get("candidate_process_stopped") is not True or rollback.get("rollback_verified") is not True:
        raise ManifestError("候选模型退出或回滚未验证")
    cleanup = rollback.get("runtime_cleanup")
    if not isinstance(cleanup, Mapping) or cleanup.get("passed") is not True:
        raise ManifestError("本地隔离运行时没有完整清理")
    actions = [row.get("action") for row in release.get("release_history", [])]
    return {
        "execution_mode": "real_llama_server",
        "rollback_verified": True,
        "provenance": {
            "git_commit": provenance.get("git_commit"),
            "dataset_id": provenance.get("dataset_id"),
            "hardware_id": provenance.get("hardware_id"),
            "run_id": provenance.get("run_id"),
        },
        "model_id_before": main.get("model_id_before"),
        "model_id_applied": main.get("model_id_applied"),
        "model_id_after_rollback": main.get("model_id_after_rollback"),
        "transition_release_ids": [row.get("release_id") for row in transitions],
        "transition_revisions": [row.get("revision") for row in transitions],
        "release_actions": actions,
        "candidate_process_stopped": True,
        "runtime_cleanup_passed": True,
        "main_evidence_sha256": main_sha,
        "stage_sha256": stage_hashes,
        "completion_marker_sha256": sha256_file(completion_path),
    }


def _build_receipt(
    publication: Mapping[str, Any],
    publication_sha256: str,
    edge_id: str,
    hardware_id: str,
    dataset_id: str,
    git_commit: str,
    downloaded_files: Sequence[Mapping[str, Any]],
    local_execution: Mapping[str, Any],
) -> Dict[str, Any]:
    if local_execution.get("execution_mode") != "real_llama_server":
        raise ManifestError("测试替身不得生成正式云边回执")
    core = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "completed",
        "execution_mode": "real_cloud_edge_http",
        "run_id": publication["run_id"],
        "edge_id": edge_id,
        "hardware_id": hardware_id,
        "dataset_id": dataset_id,
        "git_commit": git_commit,
        "publication_nonce": publication["publication_nonce"],
        "publication_sha256": publication_sha256,
        "downloaded_files": [dict(row) for row in downloaded_files],
        "download_atomic_publish": True,
        "local_execution": dict(local_execution),
        "confirmed_at_utc": _now_utc(),
    }
    return {**core, "receipt_id": _sha256_bytes(_canonical_bytes(core))}


def post_edge_receipt(
    cloud_url: str,
    receipt: Mapping[str, Any],
    secret: bytes,
    *,
    timeout_seconds: float,
    attempts: int = 3,
    require_formal_cloud: bool = False,
) -> Dict[str, Any]:
    body = _canonical_bytes(receipt)
    url = "{}{}/{}/receipts".format(
        _validate_http_base_url(cloud_url), API_PREFIX, quote(str(receipt["run_id"]), safe="")
    )
    last_error: Optional[Exception] = None
    for _index in range(max(1, int(attempts))):
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Receipt-Signature": _hmac_sha256(secret, body),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                ack_body = response.read()
                status = int(response.status)
                signature = str(response.headers.get("X-Ack-Signature", "")).lower()
            if status != 200 or not hmac.compare_digest(signature, _hmac_sha256(secret, ack_body)):
                raise ManifestError("云端确认 HTTP 状态或 HMAC 无效")
            ack = json.loads(ack_body.decode("utf-8"))
            if not isinstance(ack, Mapping):
                raise ManifestError("云端确认顶层必须是对象")
            if (
                ack.get("schema_version") != ACK_SCHEMA
                or ack.get("status") != "accepted"
                or (require_formal_cloud and ack.get("execution_mode") != "real_http")
                or ack.get("receipt_id") != receipt["receipt_id"]
                or ack.get("receipt_sha256") != _sha256_bytes(body)
                or ack.get("publication_sha256") != receipt["publication_sha256"]
            ):
                raise ManifestError("云端确认内容与边缘回执不一致")
            return {
                "url": url,
                "http_status": status,
                "request_sha256": _sha256_bytes(body),
                "ack_sha256": _sha256_bytes(ack_body),
                "ack": dict(ack),
                "hmac_verified": True,
            }
        except (OSError, HTTPError, json.JSONDecodeError, ManifestError) as exc:
            last_error = exc
    raise ManifestError("边缘回执发送失败: {}".format(last_error))


def run_edge(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Execute the complete formal edge half.  No injectable test executor exists."""
    repository_root = Path(config["repository_root"]).resolve()
    git_identity = _distributed_git_identity(repository_root, str(config["git_commit"]))
    output_dir = Path(config["output_dir"]).resolve()
    if output_dir.exists():
        raise ManifestError("正式 output-dir 必须不存在")
    _require_outside_repository(output_dir, repository_root, "output-dir")
    _require_outside_repository(
        Path(config["shared_secret_file"]), repository_root, "shared-secret-file"
    )
    cloud_url = _validate_http_base_url(str(config["cloud_url"]))
    hostname = str(urlparse(cloud_url).hostname or "").lower()
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        raise ManifestError("正式分布式取证要求 cloud-url 指向另一台机器，禁止回环地址")
    staging = output_dir.with_name(".{}.in_progress.{}.{}".format(output_dir.name, os.getpid(), secrets.token_hex(4)))
    staging.mkdir(parents=True)
    secret = _read_secret(Path(config["shared_secret_file"]), formal=True)
    try:
        transfer = download_publication(
            cloud_url,
            str(config["run_id"]),
            secret,
            staging / "downloads",
            timeout_seconds=float(config["http_timeout_seconds"]),
        )
        publication = transfer["publication"]
        if publication.get("execution_mode") != "real_http":
            raise ManifestError("测试云端发布不得生成正式云边证据")
        edge_id = _validate_identifier(config["edge_id"], "edge_id")
        if edge_id != publication["expected_edge_id"]:
            raise ManifestError("当前 edge_id 不是云端发布清单指定的边缘")
        if str(config["hardware_id"]) != publication["expected_hardware_id"]:
            raise ManifestError("当前 hardware_id 与云端发布清单不一致")
        if str(config["dataset_id"]) != publication["expected_dataset_id"]:
            raise ManifestError("当前 dataset_id 与云端发布清单不一致")
        if str(config["git_commit"]) != publication["publisher"]["git_commit"]:
            raise ManifestError("边缘 Git commit 与云端发布清单不一致")
        paths = transfer["release_paths"]
        local_config = {
            "repository_root": str(repository_root),
            "git_commit": str(config["git_commit"]),
            "old_base": str(paths["old"]["base"]),
            "old_package": str(paths["old"]["package"]),
            "old_artifact": str(paths["old"]["artifact"]),
            "candidate_base": str(paths["candidate"]["base"]),
            "candidate_package": str(paths["candidate"]["package"]),
            "candidate_artifact": str(paths["candidate"]["artifact"]),
            "old_release_id": publication["releases"]["old"]["release_id"],
            "candidate_release_id": publication["releases"]["candidate"]["release_id"],
            "binary": str(Path(config["binary"]).resolve()),
            "output_dir": str(staging / "local_execution"),
            "dataset_id": str(config["dataset_id"]),
            "hardware_id": str(config["hardware_id"]),
            "run_id": str(config["run_id"]),
            "probe_prompt": str(config["probe_prompt"]),
            "fault_mode": FAULT_MODE,
            "host": "127.0.0.1",
            "port": int(config["llama_port"]),
            "context_tokens": int(config["context_tokens"]),
            "threads": int(config["threads"]),
            "gpu_layers": int(config["gpu_layers"]),
            "startup_timeout_seconds": float(config["startup_timeout_seconds"]),
            "min_available_memory_mb": float(config["min_available_memory_mb"]),
        }
        local_result = run_local_update_loop(local_config)
        local_summary = _local_execution_summary(local_result)
        receipt = _build_receipt(
            publication,
            transfer["publication_sha256"],
            edge_id,
            str(config["hardware_id"]),
            str(config["dataset_id"]),
            str(config["git_commit"]),
            transfer["downloaded_files"],
            local_summary,
        )
        _atomic_json(staging / "edge_receipt.json", receipt)
        cloud_ack = post_edge_receipt(
            cloud_url,
            receipt,
            secret,
            timeout_seconds=float(config["http_timeout_seconds"]),
            require_formal_cloud=True,
        )
        _atomic_json(staging / "cloud_ack.json", cloud_ack)
        after_git = _distributed_git_identity(repository_root, str(config["git_commit"]))
        if after_git != git_identity:
            raise ManifestError("云边更新取证前后 Git 身份不一致")
        evidence = {
            "schema_version": DISTRIBUTED_EVIDENCE_SCHEMA,
            "status": "completed",
            "execution_mode": "real_cloud_edge_http",
            "provenance": {
                "git_commit": str(config["git_commit"]),
                "run_id": str(config["run_id"]),
                "edge_id": edge_id,
                "hardware_id": str(config["hardware_id"]),
                "dataset_id": str(config["dataset_id"]),
                "generated_at": _now_utc(),
            },
            "cloud_publication": {
                "manifest_url": transfer["manifest_url"],
                "publication_sha256": transfer["publication_sha256"],
                "hmac_key_id": hashlib.sha256(secret).hexdigest()[:16],
                "manifest_hmac_verified": True,
                "publication": publication,
            },
            "transfer": {
                "protocol": "http",
                "real_http": True,
                "atomic_download_publish": True,
                "cache_reused": transfer["cache_reused"],
                "downloaded_files": transfer["downloaded_files"],
            },
            "local_execution": local_summary,
            "edge_receipt": receipt,
            "cloud_ack": cloud_ack,
            "cloud_received_edge_confirmation": True,
        }
        evidence_path = staging / "distributed_model_update_loop.json"
        _atomic_json(evidence_path, evidence)
        completion = {
            "schema_version": DISTRIBUTED_EVIDENCE_SCHEMA,
            "status": "completed",
            "execution_mode": "real_cloud_edge_http",
            "run_id": str(config["run_id"]),
            "distributed_evidence_sha256": sha256_file(evidence_path),
            "cloud_ack_sha256": sha256_file(staging / "cloud_ack.json"),
            "completed_at_utc": _now_utc(),
        }
        _atomic_json(staging / "FORMAL_DISTRIBUTED_EVIDENCE_COMPLETE.json", completion)
        local_fragment_document = read_json_object(Path(local_result["manifest_fragment"]))
        local_target = local_fragment_document.get("model_update_loop")
        if not isinstance(local_target, Mapping):
            raise ManifestError("本地更新 manifest fragment 结构无效")
        fragment = {
            "model_update_loop": {
                "evidence": {
                    "path": "local_execution/model_update_loop.json",
                    "sha256": local_result["main_evidence_sha256"],
                },
                "stage_evidence": {
                    stage: {
                        "path": "local_execution/{}".format(
                            Path(str(local_result["stage_evidence"][stage]["path"])).name
                        ),
                        "sha256": local_result["stage_evidence"][stage]["sha256"],
                    }
                    for stage in ("package", "release", "apply", "rollback")
                },
                "completion_marker": {
                    "path": "local_execution/FORMAL_EVIDENCE_COMPLETE.json",
                    "sha256": local_result["completion_marker_sha256"],
                },
                "distributed_evidence": {
                    "path": "distributed_model_update_loop.json",
                    "sha256": completion["distributed_evidence_sha256"],
                },
                "distributed_completion_marker": {
                    "path": "FORMAL_DISTRIBUTED_EVIDENCE_COMPLETE.json",
                    "sha256": sha256_file(
                        staging / "FORMAL_DISTRIBUTED_EVIDENCE_COMPLETE.json"
                    ),
                },
                "expected": dict(local_target.get("expected", {})),
                "criteria": {
                    **dict(local_target.get("criteria", {})),
                    "real_cloud_edge_http_required": True,
                    "cloud_receipt_ack_required": True,
                },
            }
        }
        fragment_path = staging / "distributed_model_update_manifest_fragment.json"
        _atomic_json(fragment_path, fragment)
        os.replace(str(staging), str(output_dir))
        _fsync_directory(output_dir.parent)
        return {
            "status": "passed_distributed_evidence_generated",
            "evidence": str(output_dir / "distributed_model_update_loop.json"),
            "evidence_sha256": completion["distributed_evidence_sha256"],
            "completion_marker": str(output_dir / "FORMAL_DISTRIBUTED_EVIDENCE_COMPLETE.json"),
            "manifest_fragment": str(
                output_dir / "distributed_model_update_manifest_fragment.json"
            ),
            "cloud_acknowledged": True,
        }
    except Exception as exc:
        if staging.exists():
            _atomic_json(
                staging / "FAILED_NOT_FORMAL_DISTRIBUTED_EVIDENCE.json",
                {
                    "status": "failed",
                    "formal_evidence_generated": False,
                    "error": "{}: {}".format(type(exc).__name__, exc),
                    "at_utc": _now_utc(),
                },
            )
        raise


def _cloud_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("cloud", help="在云端发布模型并持久接收边缘确认")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--old-base", required=True)
    parser.add_argument("--old-package", required=True)
    parser.add_argument("--old-artifact", required=True)
    parser.add_argument("--candidate-base", required=True)
    parser.add_argument("--candidate-package", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--old-release-id", required=True)
    parser.add_argument("--candidate-release-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-edge-id", required=True)
    parser.add_argument("--expected-hardware-id", required=True)
    parser.add_argument("--expected-dataset-id", required=True)
    parser.add_argument("--storage-dir", required=True)
    parser.add_argument("--shared-secret-file", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19300)


def _edge_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("edge", help="边缘拉取、校验、应用、回滚并回传确认")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--cloud-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--edge-id", required=True)
    parser.add_argument("--shared-secret-file", required=True)
    parser.add_argument("--llama-server", dest="binary", required=True)
    parser.add_argument("--llama-port", type=int, default=19290)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--probe-prompt", required=True)
    parser.add_argument("--context-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--http-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--min-available-memory-mb", type=float, default=2048.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="真实云端发布→边缘拉取/apply/回滚→云端确认取证")
    subparsers = parser.add_subparsers(dest="role", required=True)
    _cloud_parser(subparsers)
    _edge_parser(subparsers)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    config = vars(args)
    if args.role == "cloud":
        repository_root = Path(args.repository_root).resolve()
        secret_path = Path(args.shared_secret_file)
        _require_outside_repository(
            secret_path, repository_root, "shared-secret-file"
        )
        secret = _read_secret(secret_path, formal=True)
        service = CloudUpdateService(config, secret, check_git=True)
        server = create_cloud_http_server(service, args.host, args.port)
        previous_handlers = {}

        def stop_server(_signum: int, _frame: Any) -> None:
            threading.Thread(target=server.shutdown, daemon=True).start()

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, stop_server)
        print(
            json.dumps(
                {
                    "status": "serving",
                    "run_id": service.publication["run_id"],
                    "publication_sha256": service.publication_sha256,
                    "manifest_path": "{}/{}/manifest".format(API_PREFIX, service.publication["run_id"]),
                    "listen": {"host": args.host, "port": server.server_address[1]},
                    "storage_dir": str(service.storage_dir),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            server.serve_forever(poll_interval=0.2)
        finally:
            server.server_close()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        return 0
    result = run_edge(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
