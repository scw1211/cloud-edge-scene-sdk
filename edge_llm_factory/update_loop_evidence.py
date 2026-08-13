"""用途：用真实 llama-server 取证模型 package→release→apply→rollback 闭环。

本模块的正式入口不提供 fake/dry-run 冒充实测的通道。正式运行会在隔离目录和
loopback 端口中实际启动旧、候选、回滚后三个模型进程，并在候选完成 health 和
一次推理后注入无损门禁故障。候选资产不会被修改，监督器必须自动恢复旧版本。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import urllib.request

from edge_llm_factory.adapter_package import validate_adapter_package
from edge_llm_factory.contracts import (
    ManifestError,
    base_fingerprint,
    read_json_object,
    sha256_file,
    validate_action_mapping,
    validate_base_manifest,
)
from edge_llm_factory.release_store import ReleaseStore
from edge_llm_factory.serve_release import ActiveReleaseLlamaServer


METRIC_SEMANTICS = "package_release_apply_rollback"
EVIDENCE_SCHEMA = "edge-llm-update-loop-evidence/v1"
FAULT_MODE = "post_health_inference_gate_failure"
STAGE_FILES = {
    "package": "01_package.json",
    "release": "02_release.json",
    "apply": "03_apply.json",
    "rollback": "04_rollback.json",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    payload = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError("{} 必须是非空字符串".format(field))
    return value.strip()


def _directory_sha256(root: Path) -> Tuple[str, int, int]:
    records = []
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise ManifestError("证据输入目录禁止符号链接: {}".format(path))
        size = path.stat().st_size
        total_bytes += size
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    encoded = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), len(records), total_bytes


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _available_memory_mb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError) as exc:
        raise ManifestError("无法读取 /proc/meminfo 的 MemAvailable") from exc
    raise ManifestError("/proc/meminfo 缺少 MemAvailable")


def _existing_llama_servers() -> Sequence[Dict[str, Any]]:
    found = []
    proc = Path("/proc")
    if not proc.is_dir():
        raise ManifestError("正式取证要求 Linux /proc")
    for item in proc.iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            command = (item / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
            executable = str((item / "exe").resolve())
        except (OSError, RuntimeError):
            continue
        if "llama-server" in Path(executable).name or "llama-server" in command:
            found.append(
                {"pid": int(item.name), "executable": executable, "command": command[:4096]}
            )
    return found


def _decision_contract(base_path: Path, package_path: Path) -> Dict[str, Any]:
    base = validate_base_manifest(read_json_object(base_path))
    manifest = read_json_object(package_path / "scene_adapter_manifest.json")
    mapping_path = package_path / str(manifest["action_mapping"]["path"])
    mapping = validate_action_mapping(read_json_object(mapping_path), base)
    slots = {
        str(row["slot"]): str(row["token"])
        for row in base["decision_protocol"]["slots"]
    }
    accepted_slots = {str(row["slot"]) for row in mapping["entries"]}
    accepted_tokens = sorted(slots[slot] for slot in accepted_slots)
    return {
        "scene": str(manifest["scene"]),
        "base_id": str(base["base_id"]),
        "base_fingerprint": base_fingerprint(base),
        "protocol": str(base["decision_protocol"]["name"]),
        "max_input_tokens": int(base["decision_protocol"]["max_input_tokens"]),
        "max_output_tokens": int(base["decision_protocol"]["max_output_tokens"]),
        "action_mapping_sha256": sha256_file(mapping_path),
        "accepted_tokens": accepted_tokens,
    }


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _binary_identity(binary: Path) -> Dict[str, Any]:
    if not binary.is_file() or not os.access(str(binary), os.X_OK):
        raise ManifestError("llama-server 不存在或不可执行: {}".format(binary))
    completed = subprocess.run(
        [str(binary), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30.0,
        check=False,
    )
    version = completed.stdout.strip()
    if completed.returncode != 0 or not version:
        raise ManifestError("llama-server --version 失败或没有输出")
    return {
        "path": str(binary.resolve()),
        "sha256": sha256_file(binary),
        "bytes": binary.stat().st_size,
        "version_output": version[:4096],
    }


def _git_identity(repository_root: Path, expected_commit: str) -> Dict[str, Any]:
    root = repository_root.resolve()
    imported_root = Path(__file__).resolve().parents[1]
    if root != imported_root:
        raise ManifestError(
            "repository-root 必须是实际导入 update_loop_evidence.py 的仓库根目录"
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30.0,
        check=False,
    )
    if head.returncode != 0:
        raise ManifestError("正式取证必须从 Git 工作区运行")
    observed = head.stdout.strip()
    if observed != expected_commit:
        raise ManifestError(
            "git commit 不一致：期望 {}，当前 {}".format(expected_commit, observed)
        )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30.0,
        check=False,
    )
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise ManifestError("正式取证要求干净 Git 工作区")
    source_files = (
        Path(__file__).resolve(),
        (root / "edge_llm_factory" / "serve_release.py").resolve(),
        (root / "edge_llm_factory" / "release_store.py").resolve(),
        (root / "scripts" / "evaluate_competition_targets.py").resolve(),
    )
    source_sha256 = {}
    for path in source_files:
        relative = str(path.relative_to(root))
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30.0,
            check=False,
        )
        if tracked.returncode != 0:
            raise ManifestError("正式取证关键源码未纳入 Git: {}".format(relative))
        committed = subprocess.run(
            ["git", "show", "HEAD:{}".format(relative)],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30.0,
            check=False,
        )
        if committed.returncode != 0:
            raise ManifestError("无法读取已提交关键源码: {}".format(relative))
        actual = path.read_bytes()
        if committed.stdout != actual:
            raise ManifestError("关键源码与当前 Git 提交不一致: {}".format(relative))
        source_sha256[relative] = hashlib.sha256(actual).hexdigest()
    return {
        "repository_root": str(root),
        "git_commit": observed,
        "worktree_clean": True,
        "source_sha256": source_sha256,
    }


def _package_identity(base: Path, package: Path, artifact: Path) -> Dict[str, Any]:
    validation = validate_adapter_package(package, base, require_gates=True)
    manifest = read_json_object(package / "scene_adapter_manifest.json")
    deployment = manifest["deployment"]
    if not artifact.is_file() or artifact.is_symlink():
        raise ManifestError("部署 GGUF 不存在或是符号链接: {}".format(artifact))
    observed_sha = sha256_file(artifact)
    observed_bytes = artifact.stat().st_size
    if observed_sha != deployment.get("artifact_sha256"):
        raise ManifestError("部署 GGUF SHA-256 与 package manifest 不一致")
    if observed_bytes != deployment.get("artifact_bytes"):
        raise ManifestError("部署 GGUF 字节数与 package manifest 不一致")
    package_sha, package_files, package_bytes = _directory_sha256(package)
    contract = _decision_contract(base, package)
    return {
        "base": {
            "path": str(base.resolve()),
            "sha256": sha256_file(base),
            "bytes": base.stat().st_size,
        },
        "package": {
            "path": str(package.resolve()),
            "sha256": package_sha,
            "file_count": package_files,
            "bytes": package_bytes,
        },
        "artifact": {
            "path": str(artifact.resolve()),
            "sha256": observed_sha,
            "bytes": observed_bytes,
        },
        "adapter_id": validation["adapter_id"],
        "adapter_version": validation["version"],
        "scene": validation["scene"],
        "base_fingerprint": validation["base_fingerprint"],
        "gate_results": validation["gate_results"],
        "decision_contract": contract,
    }


def _runtime_config(path: Path, endpoint: str, model: str, timeout_seconds: float) -> None:
    _atomic_json(
        path,
        {
            "schema_version": "edge-llm-runtime/v1",
            "provider": "llama_cpp",
            "endpoint": endpoint,
            "model": model,
            "timeout_seconds": timeout_seconds,
            "generation": {
                "max_input_tokens": 16,
                "max_output_tokens": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
                "thinking": False,
                "keep_alive": "0",
            },
            "authentication": {"api_key_env": ""},
        },
    )


def _completion_probe(
    endpoint: str,
    prompt: str,
    timeout_seconds: float,
    *,
    max_input_tokens: int,
    accepted_tokens: Sequence[str],
) -> Dict[str, Any]:
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "cache_prompt": False,
            "stream": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint + "/completion",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response_body = response.read()
        status = int(response.status)
    latency_ms = (time.perf_counter() - started) * 1000.0
    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("llama-server 推理响应不是 JSON") from exc
    if status != 200 or not isinstance(payload, Mapping):
        raise RuntimeError("llama-server 推理探针失败，HTTP {}".format(status))
    timings = payload.get("timings") if isinstance(payload.get("timings"), Mapping) else {}
    prompt_tokens = payload.get("tokens_evaluated", timings.get("prompt_n"))
    output_tokens = payload.get("tokens_predicted", timings.get("predicted_n"))
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        raise RuntimeError("llama-server 推理探针没有实际处理输入 token")
    if prompt_tokens > int(max_input_tokens):
        raise RuntimeError("llama-server 推理探针超过 package 输入 token 上限")
    if not isinstance(output_tokens, int) or output_tokens != 1:
        raise RuntimeError("llama-server 推理探针必须恰好生成 1 token")
    content = payload.get("content", "")
    if not isinstance(content, str):
        raise RuntimeError("llama-server 推理探针 content 类型无效")
    action_token = content.strip()
    if action_token not in set(str(value) for value in accepted_tokens):
        raise RuntimeError("llama-server 推理输出不属于当前 action mapping")
    return {
        "http_status": status,
        "latency_ms": round(latency_ms, 6),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content_preview": content[:32],
        "action_token": action_token,
        "action_mapping_accepted": True,
    }


class ControlledCandidateGateFailure(RuntimeError):
    """Expected, non-destructive failure injected after a candidate probe."""


class EvidenceReleaseLlamaServer(ActiveReleaseLlamaServer):
    """Supervisor that records real transitions and injects one controlled fault."""

    def __init__(
        self,
        *args: Any,
        probe_prompt: str,
        min_available_memory_mb: float,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.probe_prompt = probe_prompt
        self.min_available_memory_mb = float(min_available_memory_mb)
        self.fault_release_id: Optional[str] = None
        self.transitions = []
        self.failed_candidate_process: Optional[subprocess.Popen] = None

    def arm_fault(self, release_id: str) -> None:
        self.fault_release_id = str(release_id)

    def _before_start_record(
        self, release_id: str, revision: int, record: Mapping[str, Any]
    ) -> None:
        del release_id, revision, record
        available = _available_memory_mb()
        if available < self.min_available_memory_mb:
            raise RuntimeError(
                "MemAvailable {:.1f} MiB 低于正式更新门槛 {:.1f} MiB".format(
                    available, self.min_available_memory_mb
                )
            )

    def _start_record(
        self, release_id: str, revision: int, record: Mapping[str, Any]
    ) -> Dict[str, Any]:
        result = super()._start_record(release_id, revision, record)
        if self.process is None:
            raise RuntimeError("llama-server 启动后没有进程句柄")
        pid = int(self.process.pid)
        proc_exe = Path("/proc/{}/exe".format(pid))
        proc_cmdline = Path("/proc/{}/cmdline".format(pid))
        if not proc_exe.exists() or not proc_cmdline.exists():
            raise RuntimeError("正式取证要求 Linux /proc 进程证据")
        executable = str(proc_exe.resolve())
        if not os.path.samefile(executable, self.binary):
            raise RuntimeError("实际进程可执行文件与指定 llama-server 不一致")
        command = [
            value.decode("utf-8", errors="replace")
            for value in proc_cmdline.read_bytes().split(b"\0")
            if value
        ]
        artifact = Path(record["deployment_artifact"]["path"])
        if str(artifact) not in command:
            raise RuntimeError("实际进程命令行没有绑定预期 GGUF")
        package = Path(record["adapter_package"]["path"])
        base = Path(record["base_manifest"]["path"])
        contract = _decision_contract(base, package)
        observed_artifact_sha = sha256_file(artifact)
        if observed_artifact_sha != record["deployment_artifact"]["sha256"]:
            raise RuntimeError("实际进程加载的 GGUF 与 release record SHA 不一致")
        probe = _completion_probe(
            self.endpoint,
            self.probe_prompt,
            self.startup_timeout_seconds,
            max_input_tokens=int(contract["max_input_tokens"]),
            accepted_tokens=contract["accepted_tokens"],
        )
        transition = {
            **result,
            "process_executable": executable,
            "process_command": command,
            "artifact_sha256": observed_artifact_sha,
            "binding_fingerprint": record["binding_fingerprint"],
            "package_sha256": record["adapter_package"]["sha256"],
            "decision_contract": contract,
            "health_verified": True,
            "inference_probe": probe,
        }
        self.transitions.append(transition)
        if self.fault_release_id == release_id:
            self.fault_release_id = None
            self.failed_candidate_process = self.process
            raise ControlledCandidateGateFailure(
                "controlled post-health inference gate failure for {}".format(release_id)
            )
        return result


def _compact_promotion(value: Mapping[str, Any]) -> Dict[str, Any]:
    release = value["release"]
    return {
        "status": value["status"],
        "active_release_id": value["active_release_id"],
        "revision": value["revision"],
        "binding_fingerprint": release["binding_fingerprint"],
        "artifact_sha256": release["deployment_artifact"]["sha256"],
        "package_sha256": release["adapter_package"]["sha256"],
    }


def _preflight(config: Mapping[str, Any], *, check_git: bool) -> Dict[str, Any]:
    if config.get("fault_mode") != FAULT_MODE:
        raise ManifestError(
            "正式取证必须显式选择无损故障模式 {}".format(FAULT_MODE)
        )
    host = _text(config["host"], "host")
    if host not in {"127.0.0.1", "localhost"}:
        raise ManifestError("正式更新取证只允许绑定 loopback")
    port = int(config["port"])
    if port <= 0 or port > 65535:
        raise ManifestError("port 必须位于 [1, 65535]")
    if not _port_available(host, port):
        raise ManifestError("隔离端口已被占用: {}:{}".format(host, port))
    old = _package_identity(
        Path(config["old_base"]).resolve(),
        Path(config["old_package"]).resolve(),
        Path(config["old_artifact"]).resolve(),
    )
    candidate = _package_identity(
        Path(config["candidate_base"]).resolve(),
        Path(config["candidate_package"]).resolve(),
        Path(config["candidate_artifact"]).resolve(),
    )
    old_id = _text(config["old_release_id"], "old_release_id")
    candidate_id = _text(config["candidate_release_id"], "candidate_release_id")
    if old_id == candidate_id:
        raise ManifestError("旧版本与候选版本 release_id 必须不同")
    if old["artifact"]["sha256"] == candidate["artifact"]["sha256"]:
        raise ManifestError("旧版本与候选版本 GGUF SHA-256 必须不同")
    if (old["adapter_id"], old["adapter_version"]) == (
        candidate["adapter_id"],
        candidate["adapter_version"],
    ):
        raise ManifestError("旧版本与候选版本适配器 ID/版本必须可区分")
    for field in ("scene", "base_fingerprint"):
        if old[field] != candidate[field]:
            raise ManifestError("旧版本与候选版本 {} 不兼容".format(field))
    contract_fields = (
        "base_id",
        "base_fingerprint",
        "protocol",
        "max_input_tokens",
        "max_output_tokens",
        "action_mapping_sha256",
        "accepted_tokens",
    )
    for field in contract_fields:
        if old["decision_contract"][field] != candidate["decision_contract"][field]:
            raise ManifestError("旧版本与候选版本决策契约不兼容: {}".format(field))
    if not old["gate_results"] or not candidate["gate_results"]:
        raise ManifestError("旧版本与候选版本必须声明发布门槛")
    if not all(row.get("passed") is True for row in old["gate_results"]):
        raise ManifestError("旧版本存在未通过的发布门槛")
    if not all(row.get("passed") is True for row in candidate["gate_results"]):
        raise ManifestError("候选版本存在未通过的发布门槛")
    binary = _binary_identity(Path(config["binary"]).resolve())
    git_identity = (
        _git_identity(Path(config["repository_root"]), str(config["git_commit"]))
        if check_git
        else {
            "repository_root": str(Path(config["repository_root"]).resolve()),
            "git_commit": str(config["git_commit"]),
            "worktree_clean": None,
        }
    )
    min_available_memory_mb = float(config.get("min_available_memory_mb", 2048.0))
    if min_available_memory_mb <= 0:
        raise ManifestError("min_available_memory_mb 必须大于 0")
    resource_gate: Dict[str, Any] = {
        "min_available_memory_mb": min_available_memory_mb,
        "available_memory_mb": None,
        "other_llama_servers": [],
        "passed": None,
    }
    if check_git:
        repository_root = Path(config["repository_root"]).resolve()
        output_dir = Path(config["output_dir"]).resolve()
        if _path_within(output_dir, repository_root):
            raise ManifestError("正式 output-dir 必须位于 Git 仓库之外")
        existing = list(_existing_llama_servers())
        available = _available_memory_mb()
        resource_gate.update(
            {
                "available_memory_mb": round(available, 3),
                "other_llama_servers": existing,
                "passed": not existing and available >= min_available_memory_mb,
            }
        )
        if existing:
            raise ManifestError("正式更新取证前必须停止其他 llama-server")
        if available < min_available_memory_mb:
            raise ManifestError(
                "MemAvailable {:.1f} MiB 低于门槛 {:.1f} MiB".format(
                    available, min_available_memory_mb
                )
            )
    return {
        "status": "validated_not_executed",
        "formal_runtime": False,
        "fault_mode": FAULT_MODE,
        "host": host,
        "port": port,
        "old_release_id": old_id,
        "candidate_release_id": candidate_id,
        "old": old,
        "candidate": candidate,
        "llama_server": binary,
        "git": git_identity,
        "resource_gate": resource_gate,
    }


def _run_real(
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    supervisor_type: Any = EvidenceReleaseLlamaServer,
) -> Dict[str, Any]:
    formal_runtime = supervisor_type is EvidenceReleaseLlamaServer
    execution_mode = "real_llama_server" if formal_runtime else "test_double"
    output_dir = Path(config["output_dir"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ManifestError("output-dir 必须不存在或为空，防止混入旧证据")
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = output_dir / "isolated_release_store.json"
    runtime_config = output_dir / "isolated_runtime.json"
    endpoint = "http://{}:{}".format(preflight["host"], preflight["port"])
    _runtime_config(
        runtime_config,
        endpoint,
        Path(config["old_artifact"]).resolve().name,
        float(config["startup_timeout_seconds"]),
    )
    store = ReleaseStore(registry)
    server = supervisor_type(
        registry_path=registry,
        runtime_config_path=runtime_config,
        binary=Path(config["binary"]),
        host=str(preflight["host"]),
        port=int(preflight["port"]),
        context_tokens=int(config["context_tokens"]),
        threads=int(config["threads"]),
        parallel=1,
        gpu_layers=int(config["gpu_layers"]),
        poll_seconds=0.25,
        startup_timeout_seconds=float(config["startup_timeout_seconds"]),
        probe_prompt=str(config["probe_prompt"]),
        min_available_memory_mb=float(config.get("min_available_memory_mb", 2048.0)),
    )
    old_id = str(preflight["old_release_id"])
    candidate_id = str(preflight["candidate_release_id"])
    started_at = _now_utc()
    result: Optional[Dict[str, Any]] = None
    try:
        old_promotion = store.promote(
            old_id,
            Path(config["old_base"]),
            Path(config["old_package"]),
            Path(config["old_artifact"]),
        )
        old_apply = server.apply_current(force=True)
        if old_apply.get("status") != "active" or old_apply.get("release_id") != old_id:
            raise RuntimeError("旧模型未成功成为初始活动版本")

        candidate_promotion = store.promote(
            candidate_id,
            Path(config["candidate_base"]),
            Path(config["candidate_package"]),
            Path(config["candidate_artifact"]),
        )
        server.arm_fault(candidate_id)
        expected_failure = ""
        try:
            server.apply_current()
        except ControlledCandidateGateFailure as exc:
            expected_failure = str(exc)
        if not expected_failure:
            raise RuntimeError("受控候选故障没有触发")

        state = store.status(verify_active=True)
        supervisor_status = server.status()
        history = state.get("history", [])
        if [entry.get("action") for entry in history] != [
            "promote",
            "promote",
            "rollback",
        ]:
            raise RuntimeError("release store 生命周期序列不是 promote/promote/rollback")
        audit = history[-1].get("audit", {})
        if (
            state.get("active_release_id") != old_id
            or supervisor_status.get("status") != "recovered"
            or supervisor_status.get("applied_release_id") != old_id
            or supervisor_status.get("registry_revision")
            != supervisor_status.get("applied_revision")
            or audit.get("trigger") != "candidate_apply_failure"
            or audit.get("failed_release_id") != candidate_id
        ):
            raise RuntimeError("自动回滚后 registry 与真实运行时没有恢复一致")
        if len(server.transitions) != 3:
            raise RuntimeError("预期旧模型、候选模型、恢复旧模型共三次真实启动")
        if [row["release_id"] for row in server.transitions] != [
            old_id,
            candidate_id,
            old_id,
        ]:
            raise RuntimeError("真实进程切换顺序不正确")
        failed_process = server.failed_candidate_process
        if failed_process is None or failed_process.poll() is None:
            raise RuntimeError("候选进程没有在回滚时退出")
        restored_config = read_json_object(runtime_config)
        if restored_config.get("model") != str(Path(config["old_artifact"]).resolve()):
            raise RuntimeError("回滚后 runtime config 没有恢复旧 GGUF")

        result = {
            "execution_mode": execution_mode,
            "formal_runtime": formal_runtime,
            "supervisor_class": {
                "module": supervisor_type.__module__,
                "name": supervisor_type.__name__,
                "exact_formal_class": formal_runtime,
            },
            "started_at": started_at,
            "finished_at": _now_utc(),
            "fault_mode": FAULT_MODE,
            "preflight": dict(preflight),
            "old_promotion": _compact_promotion(old_promotion),
            "candidate_promotion": _compact_promotion(candidate_promotion),
            "transitions": list(server.transitions),
            "expected_failure": expected_failure,
            "registry": {
                "path": str(registry),
                "sha256": sha256_file(registry),
                "revision": state["revision"],
                "active_release_id": state["active_release_id"],
                "history": history,
            },
            "runtime_config_after_rollback": {
                "path": str(runtime_config),
                "sha256": sha256_file(runtime_config),
                "model": restored_config["model"],
            },
            "supervisor_after_rollback": supervisor_status,
            "candidate_process_stopped": True,
            "rollback_verified": True,
        }
    finally:
        observed_pids = [
            int(row["pid"])
            for row in getattr(server, "transitions", [])
            if isinstance(row, Mapping) and isinstance(row.get("pid"), int)
        ]
        server.stop()
    if result is None:
        raise RuntimeError("模型更新闭环没有生成结果")
    cleanup = {
        "tracked_pids": observed_pids,
        "all_tracked_pids_exited": all(
            not Path("/proc/{}".format(pid)).exists() for pid in observed_pids
        ),
        "port_released": _port_available(str(preflight["host"]), int(preflight["port"])),
    }
    cleanup["passed"] = (
        cleanup["all_tracked_pids_exited"] and cleanup["port_released"]
    )
    if formal_runtime and not cleanup["passed"]:
        raise RuntimeError("隔离 llama-server 未被完整清理")
    result["runtime_cleanup"] = cleanup
    result["isolated_runtime_stopped_after_evidence"] = cleanup["passed"]
    return result


def _write_formal_evidence(
    output_dir: Path, result: Mapping[str, Any], provenance: Mapping[str, Any]
) -> Dict[str, Any]:
    if result.get("execution_mode") != "real_llama_server" or result.get("formal_runtime") is not True:
        raise ManifestError("测试替身或 dry-run 不得生成正式通过证据")
    supervisor_class = result.get("supervisor_class")
    if supervisor_class != {
        "module": EvidenceReleaseLlamaServer.__module__,
        "name": EvidenceReleaseLlamaServer.__name__,
        "exact_formal_class": True,
    }:
        raise ManifestError("正式证据必须由精确 EvidenceReleaseLlamaServer 生成")
    if result.get("rollback_verified") is not True:
        raise ManifestError("回滚未验证，不得生成正式通过证据")
    preflight = result["preflight"]
    old_id = str(preflight["old_release_id"])
    candidate_id = str(preflight["candidate_release_id"])
    stage_provenance = {
        "git_commit": provenance["git_commit"],
        "run_id": provenance["run_id"],
        "dataset_id": provenance["dataset_id"],
        "hardware_id": provenance["hardware_id"],
    }
    stage_documents = {
        "package": {
            "schema_version": EVIDENCE_SCHEMA,
            "stage": "package",
            "status": "completed",
            "execution_mode": "real_llama_server",
            "provenance": stage_provenance,
            "old_release_id": old_id,
            "candidate_release_id": candidate_id,
            "old": preflight["old"],
            "candidate": preflight["candidate"],
        },
        "release": {
            "schema_version": EVIDENCE_SCHEMA,
            "stage": "release",
            "status": "completed",
            "provenance": stage_provenance,
            "old_release_id": old_id,
            "candidate_release_id": candidate_id,
            "old_promotion": result["old_promotion"],
            "candidate_promotion": result["candidate_promotion"],
            "release_history": result["registry"]["history"],
        },
        "apply": {
            "schema_version": EVIDENCE_SCHEMA,
            "stage": "apply",
            "status": "completed",
            "execution_mode": "real_llama_server",
            "provenance": stage_provenance,
            "old_release_id": old_id,
            "candidate_release_id": candidate_id,
            "llama_server": preflight["llama_server"],
            "transitions": result["transitions"][:2],
            "candidate_health_and_inference_verified": True,
            "fault_mode": result["fault_mode"],
        },
        "rollback": {
            "schema_version": EVIDENCE_SCHEMA,
            "stage": "rollback",
            "status": "completed",
            "execution_mode": "real_llama_server",
            "provenance": stage_provenance,
            "old_release_id": old_id,
            "candidate_release_id": candidate_id,
            "fault_mode": result["fault_mode"],
            "expected_failure": result["expected_failure"],
            "rollback_transition": result["transitions"][2],
            "candidate_process_stopped": result["candidate_process_stopped"],
            "registry": result["registry"],
            "runtime_config_after_rollback": result["runtime_config_after_rollback"],
            "supervisor_after_rollback": result["supervisor_after_rollback"],
            "rollback_verified": True,
            "runtime_cleanup": result["runtime_cleanup"],
            "isolated_runtime_stopped_after_evidence": result[
                "isolated_runtime_stopped_after_evidence"
            ],
        },
    }
    stage_hashes: Dict[str, str] = {}
    for stage, filename in STAGE_FILES.items():
        path = output_dir / filename
        _atomic_json(path, stage_documents[stage])
        stage_hashes[stage] = sha256_file(path)

    main = {
        "schema_version": EVIDENCE_SCHEMA,
        "provenance": dict(provenance),
        "stages": {
            stage: {"status": "completed", "evidence_sha256": stage_hashes[stage]}
            for stage in STAGE_FILES
        },
        "rollback_verified": True,
        "model_id_before": old_id,
        "model_id_applied": candidate_id,
        "model_id_after_rollback": old_id,
        "fault_mode": result["fault_mode"],
        "execution_mode": "real_llama_server",
    }
    main_path = output_dir / "model_update_loop.json"
    _atomic_json(main_path, main)
    fragment = {
        "model_update_loop": {
            "evidence": {
                "path": main_path.name,
                "sha256": sha256_file(main_path),
            },
            "stage_evidence": {
                stage: {"path": STAGE_FILES[stage], "sha256": stage_hashes[stage]}
                for stage in STAGE_FILES
            },
            "completion_marker": {
                "path": "FORMAL_EVIDENCE_COMPLETE.json",
                "sha256": "__written_after_fragment__",
            },
            "expected": {
                "git_commit": provenance["git_commit"],
                "dataset_id": provenance["dataset_id"],
                "model_ids": provenance["model_ids"],
                "hardware_id": provenance["hardware_id"],
                "metric_semantics": METRIC_SEMANTICS,
                "llama_server_sha256": preflight["llama_server"]["sha256"],
            },
            "criteria": {
                "required_stages": list(STAGE_FILES),
                "rollback_must_restore_previous_model": True,
            },
        }
    }
    fragment_path = output_dir / "model_update_loop_manifest_fragment.json"
    _atomic_json(fragment_path, fragment)
    completion_path = output_dir / "FORMAL_EVIDENCE_COMPLETE.json"
    _atomic_json(
        completion_path,
        {
            "schema_version": EVIDENCE_SCHEMA,
            "status": "completed",
            "execution_mode": "real_llama_server",
            "git_commit": provenance["git_commit"],
            "run_id": provenance["run_id"],
            "main_evidence_sha256": sha256_file(main_path),
            "stage_sha256": stage_hashes,
            "completed_at_utc": _now_utc(),
        },
    )
    # The completion marker is deliberately last.  Update the manifest fragment
    # only after the marker exists, then publish the whole staging directory by
    # one atomic rename in run().
    fragment["model_update_loop"]["completion_marker"]["sha256"] = sha256_file(
        completion_path
    )
    _atomic_json(fragment_path, fragment)
    return {
        "status": "passed_evidence_generated",
        "main_evidence": str(main_path),
        "main_evidence_sha256": sha256_file(main_path),
        "stage_evidence": {
            stage: {"path": str(output_dir / filename), "sha256": stage_hashes[stage]}
            for stage, filename in STAGE_FILES.items()
        },
        "manifest_fragment": str(fragment_path),
        "completion_marker": str(completion_path),
        "completion_marker_sha256": sha256_file(completion_path),
    }


def run(config: Mapping[str, Any], *, dry_run: bool = False) -> Dict[str, Any]:
    preflight = _preflight(config, check_git=True)
    if dry_run:
        return {
            **preflight,
            "status": "dry_run_validated_not_executed",
            "formal_evidence_generated": False,
        }
    output_dir = Path(config["output_dir"]).resolve()
    if output_dir.exists():
        raise ManifestError("正式 output-dir 必须不存在，保证整目录原子发布")
    staging_dir = output_dir.with_name(
        ".{}.in_progress.{}".format(output_dir.name, os.getpid())
    )
    if staging_dir.exists():
        raise ManifestError("取证临时目录已存在: {}".format(staging_dir))
    run_config = dict(config)
    run_config["output_dir"] = str(staging_dir)
    try:
        result = _run_real(run_config, preflight)
        post_git = _git_identity(
            Path(config["repository_root"]), str(config["git_commit"])
        )
        if post_git != preflight["git"]:
            raise ManifestError("取证运行前后 Git 身份不一致")
        provenance = {
            "git_commit": str(config["git_commit"]),
            "dataset_id": str(config["dataset_id"]),
            "model_ids": {
                "before": str(config["old_release_id"]),
                "applied": str(config["candidate_release_id"]),
            },
            "hardware_id": str(config["hardware_id"]),
            "metric_semantics": METRIC_SEMANTICS,
            "run_id": str(config["run_id"]),
            "generated_at": _now_utc(),
        }
        written = _write_formal_evidence(staging_dir, result, provenance)
        os.replace(str(staging_dir), str(output_dir))
        return {
            **written,
            "main_evidence": str(output_dir / "model_update_loop.json"),
            "stage_evidence": {
                stage: {
                    "path": str(output_dir / filename),
                    "sha256": written["stage_evidence"][stage]["sha256"],
                }
                for stage, filename in STAGE_FILES.items()
            },
            "manifest_fragment": str(
                output_dir / "model_update_loop_manifest_fragment.json"
            ),
            "completion_marker": str(output_dir / "FORMAL_EVIDENCE_COMPLETE.json"),
        }
    except Exception as exc:
        if staging_dir.exists():
            _atomic_json(
                staging_dir / "FAILED_NOT_FORMAL_EVIDENCE.json",
                {
                    "status": "failed",
                    "formal_evidence_generated": False,
                    "error": "{}: {}".format(type(exc).__name__, exc),
                    "at_utc": _now_utc(),
                },
            )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用真实 llama-server 验证模型发布、应用、候选故障自动回滚闭环。"
    )
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
    parser.add_argument("--llama-server", dest="binary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--probe-prompt", required=True)
    parser.add_argument(
        "--fault-mode",
        required=True,
        choices=[FAULT_MODE],
        help="显式授权候选通过 health/推理后注入一次无损测试故障。",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19290)
    parser.add_argument("--context-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=90.0)
    parser.add_argument(
        "--min-available-memory-mb",
        type=float,
        default=2048.0,
        help="每次加载模型前要求的 MemAvailable；正式取证还要求没有其他 llama-server。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验输入；不会启动模型，也不会创建任何 passed evidence。",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    config = vars(args)
    # The prompt is not copied into evidence; only its SHA-256 is recorded by probes.
    for field in ("dataset_id", "hardware_id", "run_id", "probe_prompt", "git_commit"):
        _text(config[field], field)
    previous_handlers = {}

    def _interrupt(signum: int, _frame: Any) -> None:
        raise InterruptedError("received signal {}".format(signum))

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _interrupt)
    try:
        result = run(config, dry_run=bool(args.dry_run))
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
