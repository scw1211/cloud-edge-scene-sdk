"""用途：监督 llama-server，使其始终加载 release store 当前活动的 GGUF。"""

import argparse
import base64
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from edge_llm_factory.action_constraints import action_token_constraint_evidence
from edge_llm_factory.contracts import ManifestError, read_json_object
from edge_llm_factory.providers import (
    validate_disabled_runtime_config,
    validate_runtime_config,
)
from edge_llm_factory.release_store import ReleaseStore


STARTUP_PROBE_SCHEMA = "edge-llm-startup-probes/v1"
RUNTIME_OUTPUT_SCHEMA = "edge-llm-runtime-output/v1"
RUNTIME_TRANSACTION_SCHEMA = "edge-llm-runtime-output-transaction/v1"
RUNTIME_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as file_obj:
            json.dump(dict(value), file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        with temporary.open("wb") as file_obj:
            file_obj.write(payload)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _regular_config_file(path: Path, label: str) -> Path:
    unresolved = Path(path).expanduser()
    try:
        mode = unresolved.lstat().st_mode
    except FileNotFoundError as exc:
        raise ManifestError("{} does not exist: {}".format(label, unresolved)) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ManifestError("{} must be a regular non-symlink file".format(label))
    return unresolved.resolve()


def load_runtime_output_descriptor(path: Path) -> Dict[str, Any]:
    """Load one scene runtime declaration and resolve its paths immutably."""
    source = _regular_config_file(Path(path), "runtime output descriptor")
    value = read_json_object(source)
    _unknown_fields(
        value,
        (
            "schema_version",
            "name",
            "template",
            "output",
            "adapter_id",
            "adapter_mode",
            "static_deployment_sha256",
            "on_missing_adapter",
        ),
        "runtime output descriptor",
    )
    if value.get("schema_version") != RUNTIME_OUTPUT_SCHEMA:
        raise ManifestError(
            "runtime output descriptor schema_version must be {}".format(
                RUNTIME_OUTPUT_SCHEMA
            )
        )
    name = value.get("name")
    if not isinstance(name, str) or RUNTIME_OUTPUT_NAME.fullmatch(name) is None:
        raise ManifestError("runtime output descriptor name is invalid")
    adapter_mode = value.get("adapter_mode", "request")
    if adapter_mode not in {"request", "process_default", "static"}:
        raise ManifestError(
            "runtime output descriptor adapter_mode must be request, "
            "process_default or static"
        )
    adapter_id = value.get("adapter_id")
    if adapter_mode == "static":
        if adapter_id is not None:
            raise ManifestError(
                "static runtime output descriptor must omit adapter_id"
            )
    elif (
        isinstance(adapter_id, bool)
        or not isinstance(adapter_id, int)
        or not 0 <= adapter_id <= 65535
    ):
        raise ManifestError("runtime output descriptor adapter_id is invalid")
    static_deployment_sha = value.get("static_deployment_sha256")
    if adapter_mode == "static":
        if (
            not isinstance(static_deployment_sha, str)
            or len(static_deployment_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in static_deployment_sha
            )
        ):
            raise ManifestError(
                "static runtime output descriptor deployment SHA256 is invalid"
            )
    elif static_deployment_sha is not None:
        raise ManifestError(
            "static_deployment_sha256 is only valid for static runtime outputs"
        )
    policy = value.get("on_missing_adapter")
    if policy not in {"base", "disable"}:
        raise ManifestError(
            "runtime output descriptor on_missing_adapter must be base or disable"
        )
    template_raw = value.get("template")
    output_raw = value.get("output")
    if not isinstance(template_raw, str) or not template_raw.strip():
        raise ManifestError("runtime output descriptor template is invalid")
    if not isinstance(output_raw, str) or not output_raw.strip():
        raise ManifestError("runtime output descriptor output is invalid")
    root = source.parent
    template_unresolved = Path(template_raw).expanduser()
    if not template_unresolved.is_absolute():
        template_unresolved = root / template_unresolved
    template = _regular_config_file(template_unresolved, "runtime output template")
    # Validate the immutable template now, before llama-server is started.
    template_config = validate_runtime_config(read_json_object(template))
    if template_config["provider"] != "llama_cpp":
        raise ManifestError("release-managed runtime output must use llama_cpp")
    configured_lora = template_config.get("lora_adapter")
    if adapter_mode in {"process_default", "static"} and configured_lora is not None:
        raise ManifestError(
            "process_default/static runtime template must omit request-level "
            "lora_adapter"
        )
    if (
        adapter_mode == "request"
        and configured_lora is not None
        and configured_lora["id"] != adapter_id
    ):
        raise ManifestError(
            "runtime output template lora_adapter.id differs from descriptor"
        )
    output_unresolved = Path(output_raw).expanduser()
    if not output_unresolved.is_absolute():
        output_unresolved = root / output_unresolved
    output = output_unresolved.resolve()
    if output == template:
        raise ManifestError("runtime output must not overwrite its immutable template")
    if output.exists():
        try:
            mode = output.lstat().st_mode
        except FileNotFoundError:
            mode = 0
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ManifestError(
                "runtime output must be absent or a regular non-symlink file"
            )
    return {
        "schema_version": RUNTIME_OUTPUT_SCHEMA,
        "name": name,
        "template": template,
        "template_config": template_config,
        "template_evidence": {
            "path": str(template),
            "sha256": _sha256(template),
        },
        "output": output,
        "adapter_id": adapter_id,
        "adapter_mode": adapter_mode,
        "static_deployment_sha256": static_deployment_sha,
        "on_missing_adapter": policy,
        "source": {"path": str(source), "sha256": _sha256(source)},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unknown_fields(value: Mapping[str, Any], allowed: Sequence[str], field: str) -> None:
    unknown = sorted(str(key) for key in value if key not in set(allowed))
    if unknown:
        raise ManifestError("{} contains unknown fields: {}".format(field, ", ".join(unknown)))


def validate_startup_probe_config(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate scene-owned probes without interpreting their business meaning."""
    if not isinstance(value, Mapping):
        raise ManifestError("startup probe config must be an object")
    _unknown_fields(
        value,
        (
            "schema_version",
            "require_for_runtime_adapters",
            "require_for_static_fusion",
            "releases",
        ),
        "startup probe config",
    )
    if value.get("schema_version") != STARTUP_PROBE_SCHEMA:
        raise ManifestError("startup probe schema_version must be {}".format(STARTUP_PROBE_SCHEMA))
    required = value.get("require_for_runtime_adapters", True)
    if not isinstance(required, bool):
        raise ManifestError("require_for_runtime_adapters must be boolean")
    static_required = value.get("require_for_static_fusion", True)
    if not isinstance(static_required, bool):
        raise ManifestError("require_for_static_fusion must be boolean")
    raw_releases = value.get("releases")
    if not isinstance(raw_releases, dict) or not raw_releases:
        raise ManifestError("startup probe releases must be a non-empty object")

    def validate_probe(
        raw_probe: Any,
        *,
        field: str,
        adapter: bool,
    ) -> Dict[str, Any]:
        if not isinstance(raw_probe, dict):
            raise ManifestError("{} must be an object".format(field))
        common_fields = (
            "prompt",
            "allowed_tokens",
            "expected_token",
            "expected_prompt_tokens",
            "timeout_seconds",
        )
        allowed_fields = (("id", "adapter_sha256") + common_fields) if adapter else common_fields
        _unknown_fields(raw_probe, allowed_fields, field)
        validated: Dict[str, Any] = {}
        if adapter:
            adapter_id = raw_probe.get("id")
            if (
                isinstance(adapter_id, bool)
                or not isinstance(adapter_id, int)
                or not 0 <= adapter_id <= 65535
            ):
                raise ManifestError("startup adapter probe id is invalid")
            adapter_sha = raw_probe.get("adapter_sha256")
            if (
                not isinstance(adapter_sha, str)
                or len(adapter_sha) != 64
                or any(character not in "0123456789abcdef" for character in adapter_sha)
            ):
                raise ManifestError("startup probe adapter_sha256 is invalid")
            validated.update({"id": adapter_id, "adapter_sha256": adapter_sha})
        prompt = raw_probe.get("prompt")
        if (
            not isinstance(prompt, str)
            or not prompt
            or len(prompt) > 4096
            or len(prompt.encode("utf-8")) > 8192
        ):
            raise ManifestError("startup probe prompt must be bounded non-empty text")
        constraint = action_token_constraint_evidence(raw_probe.get("allowed_tokens"))
        expected_token = raw_probe.get("expected_token")
        if expected_token not in constraint["allowed_tokens"]:
            raise ManifestError("startup probe expected_token is not allowed")
        expected_prompt_tokens = raw_probe.get("expected_prompt_tokens")
        if expected_prompt_tokens is not None and (
            isinstance(expected_prompt_tokens, bool)
            or not isinstance(expected_prompt_tokens, int)
            or not 1 <= expected_prompt_tokens <= 262144
        ):
            raise ManifestError(
                "startup probe expected_prompt_tokens must be in [1, 262144]"
            )
        timeout = raw_probe.get("timeout_seconds", 2.0)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or not 0.01 <= float(timeout) <= 30.0
        ):
            raise ManifestError("startup probe timeout_seconds must be in [0.01, 30]")
        validated.update(
            {
                "prompt": prompt,
                "allowed_tokens": list(constraint["allowed_tokens"]),
                "expected_token": expected_token,
                "expected_prompt_tokens": expected_prompt_tokens,
                "timeout_seconds": float(timeout),
                "constraint": constraint,
            }
        )
        return validated

    releases: Dict[str, Any] = {}
    for release_id, raw_release in raw_releases.items():
        if not isinstance(release_id, str) or not release_id.strip():
            raise ManifestError("startup probe release id must be non-empty")
        if not isinstance(raw_release, dict):
            raise ManifestError("startup probe release must be an object")
        _unknown_fields(
            raw_release,
            ("deployment_sha256", "adapters", "static"),
            "startup probe release {}".format(release_id),
        )
        deployment_sha = raw_release.get("deployment_sha256")
        if (
            not isinstance(deployment_sha, str)
            or len(deployment_sha) != 64
            or any(character not in "0123456789abcdef" for character in deployment_sha)
        ):
            raise ManifestError("startup probe deployment_sha256 is invalid")
        raw_adapters = raw_release.get("adapters")
        if not isinstance(raw_adapters, list):
            raise ManifestError("startup probe adapters must be an array")
        raw_static = raw_release.get("static", [])
        if not isinstance(raw_static, list):
            raise ManifestError("startup probe static must be an array")
        if raw_adapters and raw_static:
            raise ManifestError(
                "startup probe release cannot mix adapter and static probes"
            )
        adapters: List[Dict[str, Any]] = []
        observed_adapter_ids = []
        for raw_probe in raw_adapters:
            probe = validate_probe(
                raw_probe,
                field="startup adapter probe",
                adapter=True,
            )
            adapter_id = probe["id"]
            if observed_adapter_ids and adapter_id < observed_adapter_ids[-1]:
                raise ManifestError("startup adapter probe ids must be ordered")
            observed_adapter_ids.append(adapter_id)
            adapters.append(probe)
        unique_adapter_ids = sorted(set(observed_adapter_ids))
        if unique_adapter_ids and unique_adapter_ids != list(
            range(unique_adapter_ids[-1] + 1)
        ):
            raise ManifestError(
                "startup adapter probe ids must cover a contiguous range from 0"
            )
        releases[release_id] = {
            "deployment_sha256": deployment_sha,
            "adapters": adapters,
            "static": [
                validate_probe(
                    raw_probe,
                    field="startup static probe",
                    adapter=False,
                )
                for raw_probe in raw_static
            ],
        }
    return {
        "schema_version": STARTUP_PROBE_SCHEMA,
        "require_for_runtime_adapters": required,
        "require_for_static_fusion": static_required,
        "releases": releases,
    }


def load_startup_probe_config(path: Path) -> Dict[str, Any]:
    unresolved = Path(path).expanduser()
    try:
        mode = unresolved.lstat().st_mode
    except FileNotFoundError as exc:
        raise ManifestError("startup probe config does not exist: {}".format(unresolved)) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ManifestError("startup probe config must be a regular non-symlink file")
    resolved = unresolved.resolve()
    config = validate_startup_probe_config(read_json_object(resolved))
    config["source"] = {"path": str(resolved), "sha256": _sha256(resolved)}
    return config


class ActiveReleaseLlamaServer:
    def __init__(
        self,
        registry_path: Path,
        runtime_config_path: Optional[Path],
        binary: Path,
        host: str,
        port: int,
        context_tokens: int,
        threads: int,
        parallel: int,
        gpu_layers: int,
        poll_seconds: float,
        startup_timeout_seconds: float,
        batch_size: int = 16,
        ubatch_size: int = 16,
        no_mmap: bool = False,
        lora_adapters: Optional[list] = None,
        startup_probe_config_path: Optional[Path] = None,
        startup_gate_receipt_path: Optional[Path] = None,
        runtime_output_descriptor_paths: Optional[Sequence[Path]] = None,
    ) -> None:
        self.store = ReleaseStore(registry_path)
        self.runtime_config_path = (
            Path(runtime_config_path).expanduser().resolve()
            if runtime_config_path is not None
            else None
        )
        self.runtime_outputs = [
            load_runtime_output_descriptor(Path(path))
            for path in (runtime_output_descriptor_paths or [])
        ]
        if (self.runtime_config_path is None) == (not self.runtime_outputs):
            raise ManifestError(
                "runtime_config_path 与 runtime_output_descriptor_paths 必须且只能配置一种"
            )
        output_names = [item["name"] for item in self.runtime_outputs]
        output_paths = [item["output"] for item in self.runtime_outputs]
        if len(output_names) != len(set(output_names)):
            raise ManifestError("runtime output descriptor name 重复")
        if len(output_paths) != len(set(output_paths)):
            raise ManifestError("runtime output descriptor output 路径重复")
        self.runtime_transaction_journal_path = self.store.registry_path.with_name(
            self.store.registry_path.name + ".runtime-output-transaction.json"
        )
        self.binary = Path(binary).resolve()
        if not self.binary.is_file() or not os.access(str(self.binary), os.X_OK):
            raise ManifestError("llama-server 不存在或不可执行: {}".format(self.binary))
        self.host = str(host)
        self.port = int(port)
        self.context_tokens = int(context_tokens)
        self.threads = int(threads)
        self.parallel = int(parallel)
        self.batch_size = int(batch_size)
        self.ubatch_size = int(ubatch_size)
        self.gpu_layers = int(gpu_layers)
        self.no_mmap = bool(no_mmap)
        self.poll_seconds = float(poll_seconds)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.startup_probe_config = (
            load_startup_probe_config(startup_probe_config_path)
            if startup_probe_config_path is not None
            else None
        )
        self.startup_gate_receipt_path = (
            Path(startup_gate_receipt_path).expanduser().resolve()
            if startup_gate_receipt_path is not None
            else None
        )
        self.lora_adapters = []
        for path in lora_adapters or []:
            unresolved = Path(path).expanduser()
            try:
                mode = unresolved.lstat().st_mode
            except FileNotFoundError as exc:
                raise ManifestError(
                    "llama.cpp LoRA 不存在: {}".format(unresolved)
                ) from exc
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ManifestError(
                    "llama.cpp LoRA 必须是普通文件且不能是符号链接: {}".format(
                        unresolved
                    )
                )
            self.lora_adapters.append(unresolved.resolve())
        self._runtime_adapter_default_scales = [0] * len(self.lora_adapters)
        if self.port <= 0 or self.port > 65535:
            raise ValueError("port must be in [1, 65535]")
        if (
            min(
                self.context_tokens,
                self.threads,
                self.parallel,
                self.batch_size,
                self.ubatch_size,
            ) <= 0
            or self.gpu_layers < 0
        ):
            raise ValueError(
                "context_tokens/threads/parallel/batch_size/ubatch_size must "
                "be positive and gpu_layers non-negative"
            )
        if min(self.poll_seconds, self.startup_timeout_seconds) <= 0:
            raise ValueError("poll and startup timeout must be positive")
        self.process: Optional[subprocess.Popen] = None
        self.applied_revision = -1
        self.active_release_id: Optional[str] = None
        self.active_record: Optional[Dict[str, Any]] = None
        self.failed_revision: Optional[int] = None
        self.failed_error: Optional[str] = None
        self.last_failure: Optional[Dict[str, Any]] = None
        self.last_apply_result: Optional[Dict[str, Any]] = None
        self.last_startup_gate: Optional[Dict[str, Any]] = None
        self.last_runtime_publication: Optional[Dict[str, Any]] = None
        self.stopping = False
        if self.runtime_outputs:
            self._recover_runtime_output_transaction()

    @property
    def endpoint(self) -> str:
        return "http://{}:{}".format(self.host, self.port)

    def command(
        self,
        artifact: Path,
        runtime_adapters: Optional[list] = None,
    ) -> list:
        command = [
            str(self.binary),
            "--model",
            str(artifact),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context_tokens),
            "--threads",
            str(self.threads),
            "--threads-batch",
            str(self.threads),
            "--batch-size",
            str(self.batch_size),
            "--ubatch-size",
            str(self.ubatch_size),
            "--parallel",
            str(self.parallel),
            "--gpu-layers",
            str(self.gpu_layers),
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
            "--cache-ram",
            "0",
            "--ctx-checkpoints",
            "0",
            "--no-cache-prompt",
            "--no-cache-idle-slots",
            "--poll",
            "100",
            "--poll-batch",
            "1",
            "--no-webui",
        ]
        if self.no_mmap:
            command.append("--no-mmap")
        adapters = self.lora_adapters if runtime_adapters is None else runtime_adapters
        if adapters:
            # Current llama.cpp accepts multiple adapters as one comma-separated
            # value; repeating the flag is deprecated and only keeps the last.
            # Multi-adapter releases keep every global scale at 0 and select one
            # per request.  The only scale-1 case is one explicitly declared
            # process-default joint adapter, emitted through ``--lora``.
            scales = list(self._runtime_adapter_default_scales)
            if len(scales) != len(adapters):
                raise ManifestError("runtime adapter path/default scale count mismatch")
            if len(adapters) == 1 and scales == [1]:
                command.extend(["--lora", str(adapters[0])])
            else:
                command.extend(
                    [
                        "--lora-scaled",
                        ",".join(
                            "{}:{}".format(adapter, scale)
                            for adapter, scale in zip(adapters, scales)
                        ),
                    ]
                )
        return command

    def _runtime_adapters_for_record(self, record: Mapping[str, Any]) -> list:
        """Resolve the release-specific adapter list in stable request-id order.

        Registries created before adapters became release metadata may still
        use the legacy ``--lora-adapter`` CLI.  An explicit record member,
        including an empty list, is authoritative and cannot be mixed with
        process-global CLI state.
        """
        if "runtime_adapters" not in record:
            self._runtime_adapter_default_scales = [0] * len(self.lora_adapters)
            return list(self.lora_adapters)
        if self.lora_adapters:
            raise ManifestError(
                "--lora-adapter 不能与 release record 的 runtime_adapters 混用"
            )
        raw_adapters = record["runtime_adapters"]
        if not isinstance(raw_adapters, list):
            raise ManifestError("release runtime_adapters 必须是数组")
        adapters = []
        default_scales = []
        for expected_id, item in enumerate(raw_adapters):
            if not isinstance(item, dict) or item.get("id") != expected_id:
                raise ManifestError("release runtime adapter id 必须从 0 连续递增")
            default_scale = item.get("default_scale")
            if (
                isinstance(default_scale, bool)
                or not isinstance(default_scale, (int, float))
                or default_scale not in (0, 1)
            ):
                raise ManifestError("release runtime adapter default_scale 必须严格为 0 或 1")
            path_value = item.get("path")
            if not isinstance(path_value, str) or not path_value.strip():
                raise ManifestError("release runtime adapter path 无效")
            path = Path(path_value)
            try:
                mode = path.lstat().st_mode
            except FileNotFoundError as exc:
                raise ManifestError(
                    "release runtime adapter 不存在: {}".format(path)
                ) from exc
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ManifestError(
                    "release runtime adapter 必须是普通文件且不能是符号链接: {}".format(
                        path
                    )
                )
            adapters.append(path.resolve())
            default_scales.append(int(default_scale))
        if any(scale == 1 for scale in default_scales) and len(adapters) != 1:
            raise ManifestError("default_scale=1 仅允许单个常驻联合 Adapter")
        self._runtime_adapter_default_scales = default_scales
        return adapters

    @staticmethod
    def _release_binding(
        release_id: str,
        revision: int,
        record: Mapping[str, Any],
        runtime_output: str,
        adapter_record: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        deployment = record.get("deployment_artifact")
        if not isinstance(deployment, dict):
            raise ManifestError("active release 缺少 deployment_artifact")
        binding_fingerprint = record.get("binding_fingerprint")
        deployment_sha = deployment.get("sha256")
        if not isinstance(binding_fingerprint, str) or len(binding_fingerprint) != 64:
            raise ManifestError("active release binding_fingerprint 无效")
        if not isinstance(deployment_sha, str) or len(deployment_sha) != 64:
            raise ManifestError("active release deployment SHA256 无效")
        adapter_id = None
        adapter_sha = None
        if adapter_record is not None:
            adapter_id = adapter_record.get("id")
            adapter_sha = adapter_record.get("sha256")
        return {
            "release_id": str(release_id),
            "revision": int(revision),
            "binding_fingerprint": binding_fingerprint,
            "deployment_sha256": deployment_sha,
            "runtime_output": str(runtime_output),
            "adapter_id": adapter_id,
            "adapter_sha256": adapter_sha,
        }

    def _legacy_runtime_config(
        self,
        release_id: str,
        revision: int,
        record: Mapping[str, Any],
        artifact: Path,
    ) -> Dict[str, Any]:
        if self.runtime_config_path is None:
            raise ManifestError("legacy runtime config path is not configured")
        config = validate_runtime_config(read_json_object(self.runtime_config_path))
        if config["provider"] != "llama_cpp":
            raise ManifestError("serve-release 只管理 llama_cpp runtime")
        raw_adapters = record.get("runtime_adapters")
        adapter_record = None
        configured_lora = config.get("lora_adapter")
        if raw_adapters is not None:
            if not isinstance(raw_adapters, list):
                raise ManifestError("release runtime_adapters 必须是数组")
            if raw_adapters:
                process_default = (
                    len(raw_adapters) == 1
                    and raw_adapters[0].get("default_scale") == 1
                )
                if process_default:
                    if configured_lora is not None:
                        raise ManifestError(
                            "process-default Adapter runtime 必须省略请求级 lora_adapter"
                        )
                else:
                    if configured_lora is None:
                        raise ManifestError(
                            "release 含 runtime adapters 时单 runtime 必须显式声明 lora_adapter；"
                            "多场景请改用 --runtime-output"
                        )
                    adapter_id = configured_lora["id"]
                    if adapter_id >= len(raw_adapters):
                        raise ManifestError(
                            "single runtime lora_adapter.id 超出 active release"
                        )
                    adapter_record = raw_adapters[adapter_id]
            else:
                # The legacy path uses the same mutable file as template and
                # output.  A rollback from an adapter release therefore sees
                # the previous generated lora field here; remove it instead of
                # stranding the rollback on stale request-level state.
                config.pop("lora_adapter", None)
        config["endpoint"] = self.endpoint
        # Bind the runtime to the full immutable artifact location.  A basename
        # is ambiguous when old and candidate packages both ship model.gguf.
        config["model"] = str(artifact.resolve())
        config["release_binding"] = self._release_binding(
            release_id, revision, record, "legacy", adapter_record
        )
        return validate_runtime_config(config)

    def _build_runtime_outputs(
        self,
        release_id: str,
        revision: int,
        record: Mapping[str, Any],
        artifact: Path,
    ) -> List[Dict[str, Any]]:
        raw_adapters = record.get("runtime_adapters")
        if not isinstance(raw_adapters, list):
            raise ManifestError(
                "multi-runtime output requires explicit release runtime_adapters metadata"
            )
        deployment = record.get("deployment_artifact")
        if not isinstance(deployment, dict):
            raise ManifestError("active release lacks deployment_artifact")
        deployment_sha = deployment.get("sha256")
        outputs: List[Dict[str, Any]] = []
        for declaration in self.runtime_outputs:
            adapter_id = declaration["adapter_id"]
            adapter_mode = declaration["adapter_mode"]
            adapter_record = None
            static_active = (
                adapter_mode == "static"
                and declaration["static_deployment_sha256"] == deployment_sha
            )
            if static_active and raw_adapters:
                raise ManifestError(
                    "static-fusion runtime must have runtime_adapters=[]"
                )
            if adapter_mode != "static" and raw_adapters:
                if adapter_id >= len(raw_adapters):
                    raise ManifestError(
                        "runtime output {} requires missing adapter id {}".format(
                            declaration["name"], adapter_id
                        )
                    )
                adapter_record = raw_adapters[adapter_id]
                if not isinstance(adapter_record, dict) or adapter_record.get("id") != adapter_id:
                    raise ManifestError("release runtime adapter id/order is invalid")
                is_process_default = adapter_record.get("default_scale") == 1
                if adapter_mode == "process_default" and not is_process_default:
                    raise ManifestError(
                        "runtime output {} requires process-default adapter id {}"
                        .format(declaration["name"], adapter_id)
                    )
                if adapter_mode == "request" and is_process_default:
                    raise ManifestError(
                        "runtime output {} cannot request-select a process-default adapter"
                        .format(declaration["name"])
                    )

            binding = self._release_binding(
                release_id,
                revision,
                record,
                declaration["name"],
                (
                    adapter_record
                    if adapter_mode == "request"
                    else None
                ),
            )
            missing_required_runtime = (
                (adapter_mode == "static" and not static_active)
                or (adapter_mode != "static" and not raw_adapters)
            )
            if (
                missing_required_runtime
                and declaration["on_missing_adapter"] == "disable"
            ):
                disabled = {
                    "schema_version": "edge-llm-runtime-disabled/v1",
                    "enabled": False,
                    "release_binding": binding,
                }
                if adapter_mode == "static":
                    disabled.update(
                        {
                            "reason": "release_missing_static_deployment",
                            "required_deployment_sha256": declaration[
                                "static_deployment_sha256"
                            ],
                        }
                    )
                else:
                    disabled.update(
                        {
                            "reason": "release_missing_runtime_adapter",
                            "required_adapter_id": adapter_id,
                        }
                    )
                value = validate_disabled_runtime_config(disabled)
                state = "disabled"
            else:
                config = validate_runtime_config(declaration["template_config"])
                if config["provider"] != "llama_cpp":
                    raise ManifestError("release-managed runtime must use llama_cpp")
                config["endpoint"] = self.endpoint
                config["model"] = str(artifact.resolve())
                if adapter_record is None:
                    config.pop("lora_adapter", None)
                elif adapter_mode == "process_default":
                    config.pop("lora_adapter", None)
                else:
                    config["lora_adapter"] = {"id": adapter_id, "scale": 1.0}
                config["release_binding"] = binding
                value = validate_runtime_config(config)
                if static_active:
                    state = "static_fusion"
                elif adapter_record is None:
                    state = "base"
                elif adapter_mode == "process_default":
                    state = "process_default_adapter"
                else:
                    state = "adapter"
            outputs.append(
                {
                    "name": declaration["name"],
                    "path": declaration["output"],
                    "value": value,
                    "state": state,
                    "release_binding": binding,
                    "descriptor": dict(declaration["source"]),
                    "template": dict(declaration["template_evidence"]),
                }
            )
        bindings = {
            (
                row["release_binding"]["release_id"],
                row["release_binding"]["revision"],
                row["release_binding"]["binding_fingerprint"],
                row["release_binding"]["deployment_sha256"],
            )
            for row in outputs
        }
        if len(bindings) != 1:
            raise ManifestError("runtime outputs are not bound to one active release")
        return outputs

    def _replace_runtime_output(self, path: Path, payload: bytes) -> None:
        """Fault-injection seam for the journaled multi-file transaction."""
        _atomic_write_bytes(path, payload)

    def _remove_runtime_transaction_journal(self) -> None:
        if self.runtime_transaction_journal_path.exists():
            self.runtime_transaction_journal_path.unlink()
            _fsync_directory(self.runtime_transaction_journal_path.parent)

    def _recover_runtime_output_transaction(self) -> Optional[Dict[str, Any]]:
        """Rollback a prepared transaction left by an exception or hard crash."""
        journal_path = self.runtime_transaction_journal_path
        if not journal_path.exists():
            return None
        journal = read_json_object(journal_path)
        _unknown_fields(
            journal,
            ("schema_version", "status", "release_id", "revision", "entries"),
            "runtime output transaction",
        )
        if journal.get("schema_version") != RUNTIME_TRANSACTION_SCHEMA:
            raise ManifestError("runtime output transaction schema is invalid")
        if journal.get("status") != "prepared":
            raise ManifestError("runtime output transaction status is invalid")
        entries = journal.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ManifestError("runtime output transaction entries are invalid")
        expected_paths = {str(item["output"]) for item in self.runtime_outputs}
        observed_paths = {
            str(entry.get("path")) for entry in entries if isinstance(entry, dict)
        }
        if observed_paths != expected_paths or len(entries) != len(expected_paths):
            raise ManifestError(
                "runtime output transaction paths differ from configured outputs"
            )
        restored = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ManifestError("runtime output transaction entry is invalid")
            _unknown_fields(
                entry,
                ("name", "path", "old_exists", "old_base64", "old_sha256"),
                "runtime output transaction entry",
            )
            path = Path(str(entry["path"]))
            old_exists = entry.get("old_exists")
            if not isinstance(old_exists, bool):
                raise ManifestError("runtime transaction old_exists is invalid")
            if old_exists:
                encoded = entry.get("old_base64")
                old_sha = entry.get("old_sha256")
                if not isinstance(encoded, str) or not isinstance(old_sha, str):
                    raise ManifestError("runtime transaction old payload is invalid")
                try:
                    payload = base64.b64decode(encoded.encode("ascii"), validate=True)
                except Exception as exc:  # noqa: BLE001
                    raise ManifestError("runtime transaction old payload is invalid") from exc
                if hashlib.sha256(payload).hexdigest() != old_sha:
                    raise ManifestError("runtime transaction old payload SHA256 mismatch")
                _atomic_write_bytes(path, payload)
                state = "restored"
            else:
                if entry.get("old_base64") is not None or entry.get("old_sha256") is not None:
                    raise ManifestError("missing old runtime must not carry payload")
                if path.exists():
                    if path.is_symlink() or not path.is_file():
                        raise ManifestError("runtime output recovery target is unsafe")
                    path.unlink()
                    _fsync_directory(path.parent)
                state = "removed"
            restored.append({"name": entry.get("name"), "path": str(path), "state": state})
        self._remove_runtime_transaction_journal()
        return {
            "status": "recovered_old_generation",
            "release_id": journal.get("release_id"),
            "revision": journal.get("revision"),
            "outputs": restored,
        }

    def _publish_runtime_outputs(
        self,
        release_id: str,
        revision: int,
        outputs: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        # Any previous interrupted generation is restored before taking a new
        # snapshot.  Thus a later retry always starts from a coherent old set.
        recovered = self._recover_runtime_output_transaction()
        prepared = []
        journal_entries = []
        for row in outputs:
            path = Path(row["path"])
            payload = _json_bytes(row["value"])
            # Validate every new payload before the first rename.
            parsed = json.loads(payload.decode("utf-8"))
            if row["state"] == "disabled":
                validate_disabled_runtime_config(parsed)
            else:
                validate_runtime_config(parsed)
            if path.exists():
                if path.is_symlink() or not path.is_file():
                    raise ManifestError("runtime output target is unsafe: {}".format(path))
                old_payload = path.read_bytes()
                old_exists = True
                old_base64 = base64.b64encode(old_payload).decode("ascii")
                old_sha = hashlib.sha256(old_payload).hexdigest()
            else:
                old_exists = False
                old_base64 = None
                old_sha = None
            prepared.append((row, path, payload))
            journal_entries.append(
                {
                    "name": row["name"],
                    "path": str(path),
                    "old_exists": old_exists,
                    "old_base64": old_base64,
                    "old_sha256": old_sha,
                }
            )
        _atomic_write(
            self.runtime_transaction_journal_path,
            {
                "schema_version": RUNTIME_TRANSACTION_SCHEMA,
                "status": "prepared",
                "release_id": str(release_id),
                "revision": int(revision),
                "entries": journal_entries,
            },
        )
        _fsync_directory(self.runtime_transaction_journal_path.parent)
        try:
            published = []
            for row, path, payload in prepared:
                self._replace_runtime_output(path, payload)
                if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(payload).digest():
                    raise RuntimeError("runtime output verification failed: {}".format(path))
                published.append(
                    {
                        "name": row["name"],
                        "path": str(path),
                        "state": row["state"],
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "release_binding": dict(row["release_binding"]),
                        "descriptor": dict(row["descriptor"]),
                        "template": dict(row["template"]),
                    }
                )
        except Exception as exc:
            try:
                rollback = self._recover_runtime_output_transaction()
            except Exception as recovery_exc:  # noqa: BLE001
                raise RuntimeError(
                    "runtime output transaction failed and recovery failed: {}; {}".format(
                        exc, recovery_exc
                    )
                ) from exc
            raise RuntimeError(
                "runtime output transaction failed and restored old generation: {} ({})".format(
                    exc, rollback
                )
            ) from exc
        self._remove_runtime_transaction_journal()
        return {
            "status": "published",
            "release_id": str(release_id),
            "revision": int(revision),
            "recovered_before_publish": recovered,
            "outputs": published,
        }

    def _publish_runtime_configuration(
        self,
        release_id: str,
        revision: int,
        record: Mapping[str, Any],
        artifact: Path,
    ) -> Dict[str, Any]:
        if self.runtime_outputs:
            return self._publish_runtime_outputs(
                release_id,
                revision,
                self._build_runtime_outputs(release_id, revision, record, artifact),
            )
        if self.runtime_config_path is None:
            raise ManifestError("runtime config path is not configured")
        config = self._legacy_runtime_config(
            release_id, revision, record, artifact
        )
        _atomic_write(self.runtime_config_path, config)
        return {
            "status": "published_legacy_single_runtime",
            "release_id": str(release_id),
            "revision": int(revision),
            "outputs": [
                {
                    "name": "legacy",
                    "path": str(self.runtime_config_path),
                    "state": (
                        "adapter" if config.get("lora_adapter") is not None else "base"
                    ),
                    "sha256": _sha256(self.runtime_config_path),
                    "release_binding": dict(config["release_binding"]),
                }
            ],
        }

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        last_error = "server did not answer"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError("llama-server exited with code {}".format(self.process.returncode))
            try:
                with urllib.request.urlopen(self.endpoint + "/health", timeout=0.5) as response:
                    if response.status == 200:
                        return
            except Exception as exc:  # noqa: BLE001
                last_error = "{}: {}".format(type(exc).__name__, exc)
            time.sleep(0.1)
        raise TimeoutError("llama-server startup timed out: {}".format(last_error))

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Optional[Any] = None,
        timeout_seconds: float = 1.0,
    ) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.endpoint + path,
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
            if response.status != 200:
                raise RuntimeError("{} returned HTTP {}".format(path, response.status))
            try:
                return json.loads(response.read().decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError("{} returned invalid JSON".format(path)) from exc

    @staticmethod
    def _verify_lora_layout(
        response: Any,
        runtime_adapters: Sequence[Path],
        default_scales: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(response, list):
            raise RuntimeError("/lora-adapters did not return an array")
        if len(response) != len(runtime_adapters):
            raise RuntimeError(
                "/lora-adapters count mismatch: expected {}, got {}".format(
                    len(runtime_adapters), len(response)
                )
            )
        expected_scales = (
            [0] * len(runtime_adapters)
            if default_scales is None
            else list(default_scales)
        )
        if len(expected_scales) != len(runtime_adapters):
            raise RuntimeError("runtime adapter path/default scale count mismatch")
        verified = []
        for expected_id, (raw, expected_path, expected_scale) in enumerate(
            zip(response, runtime_adapters, expected_scales)
        ):
            if not isinstance(raw, dict):
                raise RuntimeError("/lora-adapters item is not an object")
            adapter_id = raw.get("id")
            if isinstance(adapter_id, bool) or adapter_id != expected_id:
                raise RuntimeError("/lora-adapters id/order mismatch")
            actual_path = raw.get("path")
            if actual_path != str(expected_path.resolve()):
                raise RuntimeError(
                    "/lora-adapters path mismatch for id {}".format(expected_id)
                )
            scale = raw.get("scale")
            if (
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not math.isfinite(float(scale))
                or float(scale) != float(expected_scale)
            ):
                raise RuntimeError(
                    "/lora-adapters default scale for id {} is not {}".format(
                        expected_id, expected_scale
                    )
                )
            verified.append(
                {"id": expected_id, "path": actual_path, "scale": float(scale)}
            )
        return verified

    def _release_probes(
        self,
        release_id: str,
        record: Mapping[str, Any],
        runtime_adapters: Sequence[Path],
    ) -> List[Dict[str, Any]]:
        deployment = record.get("deployment_artifact")
        deployment_sha = (
            deployment.get("sha256") if isinstance(deployment, dict) else None
        )
        static_release = any(
            declaration["adapter_mode"] == "static"
            and declaration["static_deployment_sha256"] == deployment_sha
            for declaration in self.runtime_outputs
        )
        if self.startup_probe_config is None:
            if static_release:
                raise RuntimeError(
                    "static-fusion release requires startup probes"
                )
            return []
        release_probe = self.startup_probe_config["releases"].get(release_id)
        if release_probe is None:
            if runtime_adapters and self.startup_probe_config[
                "require_for_runtime_adapters"
            ]:
                raise RuntimeError(
                    "active release {} has runtime adapters but no startup probes".format(
                        release_id
                    )
                )
            if static_release and self.startup_probe_config[
                "require_for_static_fusion"
            ]:
                raise RuntimeError(
                    "active static-fusion release {} has no startup probes".format(
                        release_id
                    )
                )
            return []
        if not isinstance(deployment, dict):
            raise RuntimeError("active release has no deployment artifact")
        if release_probe["deployment_sha256"] != deployment.get("sha256"):
            raise RuntimeError("startup probes are bound to a different deployment")
        probes = release_probe["adapters"]
        static_probes = release_probe["static"]
        if static_probes:
            if not static_release:
                raise RuntimeError(
                    "static startup probes are bound to a non-static release"
                )
            if runtime_adapters:
                raise RuntimeError(
                    "static startup probes require runtime_adapters=[]"
                )
            raw_records = record.get("runtime_adapters")
            if raw_records != []:
                raise RuntimeError(
                    "static startup probes require explicit release "
                    "runtime_adapters=[]"
                )
            return [
                dict(
                    probe,
                    id=None,
                    adapter_sha256=None,
                    activation="static_fusion",
                )
                for probe in static_probes
            ]
        if static_release and self.startup_probe_config[
            "require_for_static_fusion"
        ]:
            raise RuntimeError(
                "static-fusion release has no static startup probes"
            )
        covered_ids = {probe["id"] for probe in probes}
        if covered_ids != set(range(len(runtime_adapters))):
            raise RuntimeError(
                "startup probes must cover every runtime adapter at least once by id"
            )
        raw_records = record.get("runtime_adapters")
        for probe in probes:
            adapter_id = probe["id"]
            adapter_path = runtime_adapters[adapter_id]
            expected_sha = None
            if isinstance(raw_records, list) and adapter_id < len(raw_records):
                raw_record = raw_records[adapter_id]
                if isinstance(raw_record, dict):
                    expected_sha = raw_record.get("sha256")
            if expected_sha is None:
                expected_sha = _sha256(adapter_path)
            if probe["adapter_sha256"] != expected_sha:
                raise RuntimeError(
                    "startup probe is bound to a different adapter id {}".format(
                        adapter_id
                    )
                )
        return [dict(probe, activation="adapter") for probe in probes]

    def _verify_startup_gate(
        self,
        release_id: str,
        record: Mapping[str, Any],
        artifact: Path,
        runtime_adapters: Sequence[Path],
    ) -> Dict[str, Any]:
        default_scales = list(self._runtime_adapter_default_scales)
        if len(default_scales) != len(runtime_adapters):
            # Direct gate callers and old records predate explicit scale
            # metadata; their established contract is request-level scale 0.
            default_scales = [0] * len(runtime_adapters)
        props = self._request_json("/props", timeout_seconds=1.0)
        if not isinstance(props, dict):
            raise RuntimeError("/props did not return an object")
        expected_model_path = str(artifact.resolve())
        if props.get("model_path") != expected_model_path:
            raise RuntimeError(
                "/props model_path mismatch: expected {!r}, got {!r}".format(
                    expected_model_path, props.get("model_path")
                )
            )
        lora_layout = self._verify_lora_layout(
            self._request_json("/lora-adapters", timeout_seconds=1.0),
            runtime_adapters,
            default_scales,
        )
        probes = self._release_probes(
            release_id, record, runtime_adapters
        )
        warmups = []
        for probe in probes:
            probe_id = probe["id"]
            activation = probe.get("activation", "adapter")
            payload = {
                "prompt": probe["prompt"],
                "temperature": 0,
                "top_p": 1,
                "seed": 42,
                "n_predict": 1,
                "stream": False,
                "cache_prompt": False,
                "grammar": probe["constraint"]["grammar"],
            }
            if activation == "adapter" and default_scales[probe_id] == 0:
                payload["lora"] = [{"id": probe_id, "scale": 1.0}]
            started = time.perf_counter()
            response = self._request_json(
                "/completion",
                method="POST",
                payload=payload,
                timeout_seconds=probe["timeout_seconds"],
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            if not isinstance(response, dict):
                raise RuntimeError("/completion did not return an object")
            output = str(response.get("content", "")).strip()
            if output != probe["expected_token"]:
                raise RuntimeError(
                    "startup probe for {} returned {!r}, expected {!r}".format(
                        (
                            "static fusion"
                            if activation == "static_fusion"
                            else "adapter {}".format(probe_id)
                        ),
                        output,
                        probe["expected_token"],
                    )
                )
            timings = response.get("timings")
            predicted_n = timings.get("predicted_n") if isinstance(timings, dict) else None
            prompt_n = timings.get("prompt_n") if isinstance(timings, dict) else None
            if predicted_n is not None and predicted_n != 1:
                raise RuntimeError(
                    "startup probe for {} did not produce one token".format(
                        (
                            "static fusion"
                            if activation == "static_fusion"
                            else "adapter {}".format(probe_id)
                        )
                    )
                )
            expected_prompt_tokens = probe.get("expected_prompt_tokens")
            if (
                expected_prompt_tokens is not None
                and prompt_n != expected_prompt_tokens
            ):
                raise RuntimeError(
                    "startup probe for {} used {!r} prompt tokens, expected {}"
                    .format(
                        (
                            "static fusion"
                            if activation == "static_fusion"
                            else "adapter {}".format(probe_id)
                        ),
                        prompt_n,
                        expected_prompt_tokens,
                    )
                )
            warmups.append(
                {
                    "adapter_id": probe_id,
                    "adapter_sha256": probe["adapter_sha256"],
                    "prompt_sha256": hashlib.sha256(
                        probe["prompt"].encode("utf-8")
                    ).hexdigest(),
                    "allowed_tokens": list(probe["allowed_tokens"]),
                    "expected_token": probe["expected_token"],
                    "output_token": output,
                    "prompt_tokens": prompt_n,
                    "expected_prompt_tokens": expected_prompt_tokens,
                    "predicted_tokens": predicted_n,
                    "latency_ms": round(latency_ms, 6),
                    "constraint": {
                        "backend": probe["constraint"]["backend"],
                        "grammar_sha256": probe["constraint"]["grammar_sha256"],
                        "applies_before_sampling": True,
                    },
                    "adapter_activation": (
                        "static_fusion"
                        if activation == "static_fusion"
                        else (
                            "process_default"
                            if default_scales[probe_id] == 1
                            else "request_level"
                        )
                    ),
                }
            )
        post_warmup_layout = self._verify_lora_layout(
            self._request_json("/lora-adapters", timeout_seconds=1.0),
            runtime_adapters,
            default_scales,
        )
        return {
            "status": "passed",
            "model": {"path": expected_model_path},
            "runtime_adapters": lora_layout,
            "post_warmup_runtime_adapters": post_warmup_layout,
            "probe_policy": (
                "configured_release_probes" if self.startup_probe_config else "not_configured"
            ),
            "probe_config_source": (
                dict(self.startup_probe_config["source"])
                if self.startup_probe_config is not None
                else None
            ),
            "warmups": warmups,
        }

    def _clear_startup_gate_receipt(self) -> None:
        if (
            self.startup_gate_receipt_path is not None
            and self.startup_gate_receipt_path.exists()
        ):
            self.startup_gate_receipt_path.unlink()

    def _publish_startup_gate_receipt(
        self,
        release_id: str,
        revision: int,
        gate: Mapping[str, Any],
    ) -> None:
        if self.startup_gate_receipt_path is None:
            return
        _atomic_write(
            self.startup_gate_receipt_path,
            {
                "schema_version": "edge-llm-startup-gate-receipt/v1",
                "status": "passed",
                "supervisor_pid": os.getpid(),
                "llama_pid": self.process.pid if self.process is not None else None,
                "release_id": release_id,
                "revision": int(revision),
                "completed_at_unix": time.time(),
                "gate": dict(gate),
            },
        )

    def _is_process_healthy(self) -> bool:
        """Return true only when the tracked process is alive and serves health."""
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(self.endpoint + "/health", timeout=0.5) as response:
                return response.status == 200
        except Exception:  # noqa: BLE001
            return False

    def stop_process(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        # llama-server may create helper threads/processes.  Real processes are
        # launched in their own session below so shutdown covers the complete
        # process group; lightweight unit-test doubles keep the direct fallback.
        if isinstance(process, subprocess.Popen):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
        else:
            process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            if isinstance(process, subprocess.Popen):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
            else:
                process.kill()
            process.wait(timeout=5.0)

    def _before_start_record(
        self, release_id: str, revision: int, record: Mapping[str, Any]
    ) -> None:
        """Hook for evidence runners to enforce resource gates after shutdown."""
        del release_id, revision, record

    def _start_record(
        self,
        release_id: str,
        revision: int,
        record: Mapping[str, Any],
    ) -> Dict[str, Any]:
        artifact = Path(record["deployment_artifact"]["path"])
        runtime_adapters = self._runtime_adapters_for_record(record)
        self.stop_process()
        self._clear_startup_gate_receipt()
        self._before_start_record(release_id, revision, record)
        self.process = subprocess.Popen(
            self.command(artifact, runtime_adapters), start_new_session=True
        )
        try:
            self._wait_ready()
            startup_gate = self._verify_startup_gate(
                release_id,
                record,
                artifact,
                runtime_adapters,
            )
            runtime_publication = self._publish_runtime_configuration(
                release_id,
                revision,
                record,
                artifact,
            )
            self._publish_startup_gate_receipt(
                release_id,
                revision,
                {**startup_gate, "runtime_publication": runtime_publication},
            )
        except Exception:
            self.stop_process()
            raise
        self.active_release_id = release_id
        self.applied_revision = revision
        self.active_record = dict(record)
        self.last_startup_gate = startup_gate
        self.last_runtime_publication = runtime_publication
        return {
            "status": "active",
            "release_id": release_id,
            "revision": revision,
            "artifact": str(artifact),
            "endpoint": self.endpoint,
            "pid": self.process.pid,
            "runtime_adapters": [str(path) for path in runtime_adapters],
            "startup_gate": startup_gate,
            "runtime_publication": runtime_publication,
        }

    def apply_current(self, force: bool = False) -> Dict[str, Any]:
        quick = self.store.status(verify_active=False)
        revision = int(quick["revision"])
        release_id = quick.get("active_release_id")
        if not release_id:
            raise ManifestError("release store 没有 active release")
        if (
            not force
            and revision == self.applied_revision
            and release_id == self.active_release_id
            and self._is_process_healthy()
        ):
            result = {
                "status": "unchanged",
                "release_id": release_id,
                "revision": revision,
                "pid": self.process.pid,
            }
            self.last_apply_result = result
            return result
        previous_id = self.active_release_id
        previous_revision = self.applied_revision
        previous_record = self.active_record
        try:
            verified = self.store.status(verify_active=True)
            if (
                int(verified.get("revision", -1)) != revision
                or verified.get("active_release_id") != release_id
            ):
                raise RuntimeError("release store changed while verifying candidate")
            candidate = verified["releases"][release_id]
            result = self._start_record(str(release_id), revision, candidate)
            after_start = self.store.status(verify_active=True)
            if (
                int(after_start.get("revision", -1)) != revision
                or after_start.get("active_release_id") != release_id
            ):
                raise RuntimeError("release store changed while starting candidate")
            self.failed_revision = None
            self.failed_error = None
            self.last_apply_result = result
            return result
        except Exception as exc:
            candidate_error = "{}: {}".format(type(exc).__name__, exc)
            rollback_result: Optional[Dict[str, Any]] = None
            rollback_error: Optional[str] = None
            fallback_result: Optional[Dict[str, Any]] = None
            fallback_error: Optional[str] = None
            if previous_id is not None and previous_record is not None:
                try:
                    rollback_result = self.store.rollback_if_active(
                        expected_release_id=str(release_id),
                        expected_revision=revision,
                        release_id=previous_id,
                        reason=candidate_error,
                    )
                except Exception as rollback_exc:  # noqa: BLE001
                    rollback_error = "{}: {}".format(
                        type(rollback_exc).__name__, rollback_exc
                    )
                fallback_id = previous_id
                fallback_revision = previous_revision
                fallback_record = previous_record
                if rollback_result is not None:
                    if rollback_result.get("status") == "rolled_back":
                        fallback_id = str(rollback_result["active_release_id"])
                        fallback_revision = int(rollback_result["revision"])
                        fallback_record = rollback_result["release"]
                    elif rollback_result.get("status") == "rollback_skipped":
                        # A newer promotion won the CAS race.  Converge to that
                        # authoritative release instead of reviving stale state.
                        latest = self.store.status(verify_active=True)
                        fallback_id = str(latest["active_release_id"])
                        fallback_revision = int(latest["revision"])
                        fallback_record = latest["releases"][fallback_id]
                try:
                    fallback_result = self._start_record(
                        fallback_id, fallback_revision, fallback_record
                    )
                    converged = self.store.status(verify_active=True)
                    if (
                        converged.get("active_release_id") != fallback_id
                        or int(converged.get("revision", -1)) != fallback_revision
                        or not self._is_process_healthy()
                    ):
                        raise RuntimeError(
                            "fallback runtime did not converge with release store"
                        )
                except Exception as fallback_exc:  # noqa: BLE001
                    fallback_error = "{}: {}".format(
                        type(fallback_exc).__name__, fallback_exc
                    )
            self.failed_revision = revision
            self.failed_error = candidate_error
            self.last_failure = {
                "failed_release_id": release_id,
                "failed_revision": revision,
                "error": candidate_error,
                "rollback": self._compact_transition(rollback_result),
                "rollback_error": rollback_error,
                "fallback": self._compact_transition(fallback_result),
                "fallback_error": fallback_error,
                "at_unix": time.time(),
            }
            self.last_apply_result = {
                "status": "candidate_rejected",
                **self.last_failure,
            }
            if fallback_error is not None:
                raise RuntimeError(
                    "candidate apply failed and previous release could not be restored: {}; {}".format(
                        candidate_error, fallback_error
                    )
                ) from exc
            raise

    @staticmethod
    def _compact_transition(
        result: Optional[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if result is None:
            return None
        fields = (
            "status",
            "reason",
            "active_release_id",
            "release_id",
            "revision",
            "expected_release_id",
            "expected_revision",
        )
        return {field: result[field] for field in fields if field in result}

    def status(self) -> Dict[str, Any]:
        registry = self.store.status(verify_active=False)
        process_running = self.process is not None and self.process.poll() is None
        process_healthy = self._is_process_healthy() if process_running else False
        registry_revision = int(registry.get("revision", 0))
        converged = (
            process_running
            and process_healthy
            and registry.get("active_release_id") == self.active_release_id
            and registry_revision == self.applied_revision
        )
        if self.failed_error is not None and converged:
            # The candidate failed, but the guarded registry rollback and the
            # previous runtime both succeeded.  Keep last_failure for audit
            # without advertising an unavailable/degraded serving path.
            lifecycle_status = "recovered"
        elif not converged:
            # This also covers an unexpected runtime exit or a registry/runtime
            # revision mismatch even when no candidate exception was recorded.
            lifecycle_status = "degraded"
        else:
            lifecycle_status = "ok"
        return {
            "status": lifecycle_status,
            "registry_active_release_id": registry.get("active_release_id"),
            "registry_revision": registry_revision,
            "applied_release_id": self.active_release_id,
            "applied_revision": self.applied_revision,
            "process_running": process_running,
            "process_healthy": process_healthy,
            "pid": self.process.pid if process_running else None,
            "failed_revision": self.failed_revision,
            "failed_error": self.failed_error,
            "last_failure": self.last_failure,
            "last_apply_result": self.last_apply_result,
            "last_startup_gate": self.last_startup_gate,
            "last_runtime_publication": self.last_runtime_publication,
        }

    def run(self) -> None:
        print(json.dumps(self.apply_current(force=True), ensure_ascii=False))
        while not self.stopping:
            time.sleep(self.poll_seconds)
            if self.stopping:
                break
            try:
                result = self.apply_current()
                if result["status"] != "unchanged":
                    print(json.dumps(result, ensure_ascii=False))
            except Exception as exc:  # noqa: BLE001
                print(
                    json.dumps(
                        {
                            "status": "error",
                            "error": "{}: {}".format(type(exc).__name__, exc),
                            "supervisor": self.status(),
                        },
                        ensure_ascii=False,
                    )
                )

    def stop(self) -> None:
        self.stopping = True
        self.stop_process()


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Run llama-server for the active Edge LLM release.")
    parser.add_argument("--registry", required=True)
    runtime_group = parser.add_mutually_exclusive_group(required=True)
    runtime_group.add_argument(
        "--runtime-config",
        help="Legacy single runtime template/output path.",
    )
    runtime_group.add_argument(
        "--runtime-output",
        action="append",
        default=[],
        help=(
            "Release-managed scene runtime output descriptor JSON; repeat once "
            "per scene. All declarations publish as one journaled generation."
        ),
    )
    parser.add_argument("--binary", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18190)
    parser.add_argument("--context-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ubatch-size", type=int, default=16)
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="llama-server 并发槽数量；每块板负责两个分区时建议设为 2",
    )
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument(
        "--no-mmap",
        action="store_true",
        help="将 --no-mmap 传给 llama-server；默认保持 mmap。",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--startup-probe-config",
        help=(
            "声明式、按 release_id 绑定的逐 LoRA 安全 probe JSON；通用发布层"
            "只执行受约束单 token 请求，不解释场景语义。"
        ),
    )
    parser.add_argument(
        "--startup-gate-receipt",
        help="全部 health/props/LoRA/probe 门禁通过后原子发布的回执路径。",
    )
    parser.add_argument(
        "--lora-adapter",
        action="append",
        default=[],
        help=(
            "预载一个 llama.cpp LoRA GGUF；可重复，id 按参数顺序从 0 开始。"
            "启用后全部 Adapter 默认 scale=0，由每个场景请求显式选择。"
        ),
    )
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args(argv)
    supervisor = ActiveReleaseLlamaServer(
        registry_path=Path(args.registry),
        runtime_config_path=(
            Path(args.runtime_config) if args.runtime_config is not None else None
        ),
        binary=Path(args.binary),
        host=args.host,
        port=args.port,
        context_tokens=args.context_tokens,
        threads=args.threads,
        parallel=args.parallel,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        gpu_layers=args.gpu_layers,
        no_mmap=args.no_mmap,
        poll_seconds=args.poll_seconds,
        startup_timeout_seconds=args.startup_timeout_seconds,
        lora_adapters=[Path(path) for path in args.lora_adapter],
        startup_probe_config_path=(
            Path(args.startup_probe_config) if args.startup_probe_config else None
        ),
        startup_gate_receipt_path=(
            Path(args.startup_gate_receipt) if args.startup_gate_receipt else None
        ),
        runtime_output_descriptor_paths=[
            Path(path) for path in args.runtime_output
        ],
    )
    if args.print_command:
        status = supervisor.store.status(verify_active=True)
        record = status["releases"][status["active_release_id"]]
        adapters = supervisor._runtime_adapters_for_record(record)
        print(
            json.dumps(
                supervisor.command(
                    Path(record["deployment_artifact"]["path"]), adapters
                ),
                ensure_ascii=False,
            )
        )
        return

    def stop(_signum: int, _frame: Any) -> None:
        supervisor.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        supervisor.run()
    finally:
        supervisor.stop()


if __name__ == "__main__":
    main()
