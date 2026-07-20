"""用途：从锁定的 Qwen 多模态上游快照导出并验证可共享的纯文本语言基座。"""

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from edge_llm_factory.base_snapshot import verify_base_snapshot
from edge_llm_factory.contracts import (
    ManifestError,
    base_fingerprint,
    read_json_object,
    safe_relative_path,
    sha256_file,
    validate_base_manifest,
    write_json_object,
)


TEXT_SNAPSHOT_SCHEMA = "edge-llm-text-snapshot/v1"
MULTIMODAL_MARKERS = ("vision", "visual", "image", "video", "audio", "speech")
MULTIMODAL_ROLE_KEYS = {
    "audio_bos_token",
    "audio_eos_token",
    "audio_token",
    "image_token",
    "video_token",
    "vision_bos_token",
    "vision_eos_token",
    "model_specific_special_tokens",
}
TEXT_ONLY_CHAT_TEMPLATE = """{%- for message in messages %}
{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}
"""


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError("{} 必须是非空字符串".format(field))
    return value.strip()


def _require_integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError("{} 必须是大于等于 {} 的整数".format(field, minimum))
    return value


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError("{} 必须是布尔值".format(field))
    return value


def _contains_multimodal_marker(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in MULTIMODAL_MARKERS)


def _product(shape: Sequence[int]) -> int:
    return math.prod(int(dimension) for dimension in shape)


def _tensor_report(snapshot: Path, tensor_files: Iterable[str]) -> Dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ManifestError("验证文本基座需要 safetensors") from exc

    names: List[str] = []
    multimodal: List[str] = []
    parameter_count = 0
    multimodal_parameter_count = 0
    for relative_name in sorted(set(tensor_files)):
        path = snapshot / relative_name
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                shape = list(handle.get_slice(name).get_shape())
                count = _product(shape)
                names.append(name)
                parameter_count += count
                if _contains_multimodal_marker(name):
                    multimodal.append(name)
                    multimodal_parameter_count += count
    return {
        "tensor_count": len(names),
        "parameter_count": parameter_count,
        "multimodal_parameter_count": multimodal_parameter_count,
        "multimodal_parameter_names": multimodal,
    }


def _multimodal_config_paths(value: Any, prefix: str = "") -> List[str]:
    paths: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = "{}.{}".format(prefix, key) if prefix else str(key)
            if _contains_multimodal_marker(str(key)):
                paths.append(path)
            paths.extend(_multimodal_config_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_multimodal_config_paths(child, "{}[{}]".format(prefix, index)))
    return paths


def _file_records(snapshot: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        relative = str(path.relative_to(snapshot))
        if relative == "text_snapshot_manifest.json":
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _sanitize_tokenizer_export(output: Path) -> List[str]:
    config_path = output / "tokenizer_config.json"
    config = read_json_object(config_path)
    removed = []
    for key in sorted(MULTIMODAL_ROLE_KEYS):
        if key in config:
            config.pop(key)
            removed.append(key)
    config["chat_template"] = TEXT_ONLY_CHAT_TEMPLATE
    write_json_object(config_path, config)
    (output / "chat_template.jinja").write_text(
        TEXT_ONLY_CHAT_TEMPLATE,
        encoding="utf-8",
    )
    return removed


def validate_text_snapshot_manifest(
    value: Mapping[str, Any], base: Mapping[str, Any]
) -> Dict[str, Any]:
    manifest = dict(value)
    base_data = validate_base_manifest(base)
    if manifest.get("schema_version") != TEXT_SNAPSHOT_SCHEMA:
        raise ManifestError("text snapshot schema_version 必须是 {}".format(TEXT_SNAPSHOT_SCHEMA))
    _require_text(manifest.get("snapshot_id"), "snapshot_id")
    if manifest.get("base_id") != base_data["base_id"]:
        raise ManifestError("文本快照 base_id 与基座清单不一致")
    if manifest.get("base_fingerprint") != base_fingerprint(base_data):
        raise ManifestError("文本快照 base_fingerprint 与基座清单不一致")
    safe_relative_path(manifest.get("relative_path"), "relative_path")

    upstream = manifest.get("upstream")
    if not isinstance(upstream, dict):
        raise ManifestError("upstream 必须是对象")
    if upstream.get("model_id") != base_data["source"]["model_id"]:
        raise ManifestError("文本快照上游 model_id 不一致")
    if upstream.get("revision") != base_data["source"]["revision"]:
        raise ManifestError("文本快照上游 revision 不一致")

    derivation = manifest.get("derivation")
    if not isinstance(derivation, dict):
        raise ManifestError("derivation 必须是对象")
    if derivation.get("method") != "transformers.AutoModelForCausalLM.from_pretrained":
        raise ManifestError("只允许通过 AutoModelForCausalLM 提取语言主干")
    _require_text(derivation.get("source_architecture"), "derivation.source_architecture")
    if derivation.get("output_architecture") != base_data["model"]["architecture"]:
        raise ManifestError("纯文本输出架构与基座清单不一致")
    if derivation.get("dtype") not in {"float16", "bfloat16", "float32"}:
        raise ManifestError("derivation.dtype 必须是 float16/bfloat16/float32")
    _require_bool(
        derivation.get("source_contains_multimodal_parameters"),
        "derivation.source_contains_multimodal_parameters",
    )
    if not derivation["source_contains_multimodal_parameters"]:
        raise ManifestError("当前派生流程必须如实声明上游含多模态参数")
    _require_bool(
        derivation.get("output_contains_multimodal_parameters"),
        "derivation.output_contains_multimodal_parameters",
    )
    if derivation["output_contains_multimodal_parameters"]:
        raise ManifestError("文本快照不能包含多模态参数")
    prefixes = derivation.get("excluded_parameter_prefixes")
    if not isinstance(prefixes, list) or not prefixes:
        raise ManifestError("derivation.excluded_parameter_prefixes 必须是非空数组")
    for index, prefix in enumerate(prefixes):
        _require_text(prefix, "derivation.excluded_parameter_prefixes[{}]".format(index))
    source_parameters = _require_integer(
        derivation.get("source_parameter_count"), "derivation.source_parameter_count", 1
    )
    source_multimodal = _require_integer(
        derivation.get("source_multimodal_parameter_count"),
        "derivation.source_multimodal_parameter_count",
        1,
    )
    source_auxiliary = _require_integer(
        derivation.get("source_auxiliary_parameter_count"),
        "derivation.source_auxiliary_parameter_count",
        0,
    )
    removed_parameters = _require_integer(
        derivation.get("removed_parameter_count"),
        "derivation.removed_parameter_count",
        1,
    )
    if removed_parameters != source_multimodal + source_auxiliary:
        raise ManifestError("移除参数总数与视觉、辅助参数统计不一致")
    if source_parameters - removed_parameters != int(base_data["model"]["parameter_count"]):
        raise ManifestError("上游参数量与纯文本派生参数量无法闭合")

    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ManifestError("model 必须是对象")
    if model.get("architecture") != base_data["model"]["architecture"]:
        raise ManifestError("文本快照 architecture 不一致")
    if model.get("model_type") != base_data["model"]["model_type"]:
        raise ManifestError("文本快照 model_type 不一致")
    if model.get("modality") != "text_only":
        raise ManifestError("文本快照 modality 必须是 text_only")
    if _require_integer(model.get("parameter_count"), "model.parameter_count", 1) != int(
        base_data["model"]["parameter_count"]
    ):
        raise ManifestError("文本快照参数量与基座清单不一致")
    if _require_integer(
        model.get("multimodal_parameter_count"), "model.multimodal_parameter_count", 0
    ) != 0:
        raise ManifestError("文本快照多模态参数量必须为 0")

    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ManifestError("tokenizer 必须是对象")
    if tokenizer.get("chat_template") != "text_only_no_thinking/v1":
        raise ManifestError("tokenizer.chat_template 必须是 text_only_no_thinking/v1")
    _require_bool(tokenizer.get("vocabulary_ids_preserved"), "tokenizer.vocabulary_ids_preserved")
    removed = tokenizer.get("removed_multimodal_role_keys")
    if not isinstance(removed, list):
        raise ManifestError("tokenizer.removed_multimodal_role_keys 必须是数组")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ManifestError("files 必须是非空数组")
    seen = set()
    for index, row in enumerate(files):
        if not isinstance(row, dict):
            raise ManifestError("files[{}] 必须是对象".format(index))
        relative = str(safe_relative_path(row.get("path"), "files[{}].path".format(index)))
        if relative in seen:
            raise ManifestError("文本快照文件路径不能重复")
        seen.add(relative)
        _require_integer(row.get("bytes"), "files[{}].bytes".format(index), 1)
        digest = _require_text(row.get("sha256"), "files[{}].sha256".format(index))
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ManifestError("files[{}].sha256 格式无效".format(index))
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
    if not required.issubset(seen):
        raise ManifestError("文本快照缺少必要配置文件: {}".format(sorted(required - seen)))
    if not any(path.endswith(".safetensors") for path in seen):
        raise ManifestError("文本快照缺少 safetensors 权重")
    return manifest


def verify_text_snapshot(
    base: Mapping[str, Any],
    snapshot_manifest: Mapping[str, Any],
    snapshot_dir: Path,
    verify_tokenizer: bool = True,
) -> Dict[str, Any]:
    base_data = validate_base_manifest(base)
    manifest = validate_text_snapshot_manifest(snapshot_manifest, base_data)
    snapshot = snapshot_dir.resolve()
    if not snapshot.is_dir():
        raise ManifestError("文本基座快照目录不存在: {}".format(snapshot))

    checked = []
    tensor_files = []
    for row in manifest["files"]:
        path = snapshot / row["path"]
        if not path.is_file():
            raise ManifestError("文本基座缺少文件: {}".format(row["path"]))
        if path.stat().st_size != int(row["bytes"]):
            raise ManifestError("文本基座文件大小不匹配: {}".format(row["path"]))
        if sha256_file(path) != row["sha256"]:
            raise ManifestError("文本基座文件哈希不匹配: {}".format(row["path"]))
        checked.append(row["path"])
        if row["path"].endswith(".safetensors"):
            tensor_files.append(row["path"])

    actual_tensor_files = sorted(str(path.relative_to(snapshot)) for path in snapshot.rglob("*.safetensors"))
    if actual_tensor_files != sorted(tensor_files):
        raise ManifestError("文本基座存在未纳入清单的 safetensors 文件")
    tensor_report = _tensor_report(snapshot, tensor_files)
    if tensor_report["multimodal_parameter_names"]:
        raise ManifestError(
            "文本基座仍包含多模态参数: {}".format(
                ", ".join(tensor_report["multimodal_parameter_names"][:5])
            )
        )
    expected_parameters = int(base_data["model"]["parameter_count"])
    if tensor_report["parameter_count"] != expected_parameters:
        raise ManifestError(
            "文本基座权重参数量不一致: {} != {}".format(
                tensor_report["parameter_count"], expected_parameters
            )
        )

    config = read_json_object(snapshot / "config.json")
    forbidden_config = _multimodal_config_paths(config)
    if forbidden_config:
        raise ManifestError("文本 config 仍包含多模态字段: {}".format(forbidden_config[:5]))
    if config.get("architectures") != [base_data["model"]["architecture"]]:
        raise ManifestError("文本 config architecture 不一致")
    if config.get("model_type") != base_data["model"]["model_type"]:
        raise ManifestError("文本 config model_type 不一致")

    tokenizer_config = read_json_object(snapshot / "tokenizer_config.json")
    leaked_roles = sorted(MULTIMODAL_ROLE_KEYS.intersection(tokenizer_config))
    if leaked_roles:
        raise ManifestError("tokenizer 仍声明多模态角色: {}".format(leaked_roles))
    if tokenizer_config.get("chat_template") != TEXT_ONLY_CHAT_TEMPLATE:
        raise ManifestError("tokenizer 未使用锁定的纯文本 no-thinking 模板")

    slot_report = None
    if verify_tokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ManifestError("验证动作 token 需要 transformers") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, use_fast=True, trust_remote_code=False
        )
        slot_report = {}
        for slot in base_data["decision_protocol"]["slots"]:
            encoded = tokenizer.encode(slot["token"], add_special_tokens=False)
            if encoded != [slot["token_id"]]:
                raise ManifestError("动作槽 {} 的 tokenizer ID 不匹配".format(slot["slot"]))
            slot_report[slot["slot"]] = encoded[0]

    return {
        "status": "valid",
        "snapshot_id": manifest["snapshot_id"],
        "base_id": base_data["base_id"],
        "snapshot": str(snapshot),
        "checked_files": checked,
        "tensor_report": tensor_report,
        "action_token_ids": slot_report,
    }



def load_text_snapshot_smoke(snapshot_dir: Path, expected_parameter_count: int) -> Dict[str, Any]:
    """Load the exported snapshot and execute one real forward pass on CPU."""
    try:
        import resource
        import time
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ManifestError("文本基座加载冒烟需要 torch 和 transformers") from exc

    snapshot = snapshot_dir.resolve()
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, use_fast=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_ms = (time.perf_counter() - load_started) * 1000.0
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    if parameter_count != int(expected_parameter_count):
        raise ManifestError("加载后的文本模型参数量与快照清单不一致")
    encoded = tokenizer("A", return_tensors="pt", add_special_tokens=False)
    forward_started = time.perf_counter()
    with torch.inference_mode():
        output = model(**encoded, use_cache=False)
    forward_ms = (time.perf_counter() - forward_started) * 1000.0
    logits_shape = list(output.logits.shape)
    if logits_shape[:2] != [1, 1]:
        raise ManifestError("文本基座前向输出形状异常: {}".format(logits_shape))
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "status": "load_and_forward_passed",
        "model_class": type(model).__name__,
        "parameter_count": parameter_count,
        "load_ms": round(load_ms, 6),
        "forward_ms": round(forward_ms, 6),
        "logits_shape": logits_shape,
        "peak_process_rss_mb": round(peak_rss_mb, 6),
    }


def export_text_snapshot(
    base: Mapping[str, Any],
    source_snapshot: Path,
    output_dir: Path,
    relative_path: str,
    dtype_name: str = "bfloat16",
    manifest_output: Optional[Path] = None,
) -> Dict[str, Any]:
    base_data = validate_base_manifest(base)
    source = source_snapshot.resolve()
    source_verification = verify_base_snapshot(base_data, source, verify_tokenizer=True)
    output = output_dir.resolve()
    if output.exists():
        raise ManifestError("文本基座输出已存在，拒绝覆盖: {}".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    safe_relative_path(relative_path, "relative_path")

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ManifestError("导出文本基座需要 torch 和 transformers") from exc
    dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_by_name:
        raise ManifestError("dtype 必须是 float16/bfloat16/float32")

    raw_config = read_json_object(source / "config.json")
    source_architectures = raw_config.get("architectures", [])
    if not isinstance(source_architectures, list) or len(source_architectures) != 1:
        raise ManifestError("上游 config 必须声明唯一 architecture")
    source_tensor_files = [
        row["path"] for row in base_data["source"]["artifacts"]
        if str(row["path"]).endswith(".safetensors")
    ]
    source_tensors = _tensor_report(source, source_tensor_files)
    if not source_tensors["multimodal_parameter_names"]:
        raise ManifestError("上游快照未检测到多模态参数，派生记录与事实不符")

    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=str(output.parent)))
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(source), local_files_only=True, use_fast=True, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(source),
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype_by_name[dtype_name],
            low_cpu_mem_usage=True,
        )
        loaded_parameter_count = 0
        forbidden_loaded = []
        for name, parameter in model.named_parameters():
            loaded_parameter_count += int(parameter.numel())
            if _contains_multimodal_marker(name):
                forbidden_loaded.append(name)
        if forbidden_loaded:
            raise ManifestError("AutoModelForCausalLM 仍加载多模态参数: {}".format(forbidden_loaded[:5]))
        if loaded_parameter_count != int(base_data["model"]["parameter_count"]):
            raise ManifestError(
                "语言主干参数量不一致: {} != {}".format(
                    loaded_parameter_count, base_data["model"]["parameter_count"]
                )
            )
        model.save_pretrained(
            str(temporary),
            safe_serialization=True,
            max_shard_size="4GB",
        )
        tokenizer.chat_template = TEXT_ONLY_CHAT_TEMPLATE
        tokenizer.save_pretrained(str(temporary))
        removed_role_keys = _sanitize_tokenizer_export(temporary)
        removed_parameter_count = source_tensors["parameter_count"] - loaded_parameter_count
        source_auxiliary_parameter_count = (
            removed_parameter_count - source_tensors["multimodal_parameter_count"]
        )
        if source_auxiliary_parameter_count < 0:
            raise ManifestError("上游参数裁剪统计出现负数")
        for exported_path in temporary.rglob("*"):
            if exported_path.is_file():
                exported_path.chmod(0o644)

        files = _file_records(temporary)
        manifest = {
            "schema_version": TEXT_SNAPSHOT_SCHEMA,
            "snapshot_id": "qwen35-0.8b-text-{}@{}".format(
                dtype_name,
                base_data["source"]["revision"][:8],
            ),
            "base_id": base_data["base_id"],
            "base_fingerprint": base_fingerprint(base_data),
            "relative_path": relative_path,
            "upstream": {
                "model_id": base_data["source"]["model_id"],
                "revision": base_data["source"]["revision"],
                "source_config_sha256": base_data["source"]["config_sha256"],
            },
            "derivation": {
                "method": "transformers.AutoModelForCausalLM.from_pretrained",
                "source_architecture": source_architectures[0],
                "output_architecture": base_data["model"]["architecture"],
                "dtype": dtype_name,
                "transformers_version": str(transformers.__version__),
                "source_contains_multimodal_parameters": True,
                "output_contains_multimodal_parameters": False,
                "excluded_parameter_prefixes": ["model.visual.", "mtp."],
                "source_parameter_count": source_tensors["parameter_count"],
                "source_multimodal_parameter_count": source_tensors[
                    "multimodal_parameter_count"
                ],
                "source_auxiliary_parameter_count": source_auxiliary_parameter_count,
                "removed_parameter_count": removed_parameter_count,
            },
            "model": {
                "architecture": base_data["model"]["architecture"],
                "model_type": base_data["model"]["model_type"],
                "modality": "text_only",
                "parameter_count": loaded_parameter_count,
                "multimodal_parameter_count": 0,
            },
            "tokenizer": {
                "chat_template": "text_only_no_thinking/v1",
                "vocabulary_ids_preserved": True,
                "removed_multimodal_role_keys": removed_role_keys,
            },
            "files": files,
        }
        write_json_object(temporary / "text_snapshot_manifest.json", manifest)
        (temporary / "text_snapshot_manifest.json").chmod(0o644)
        verification = verify_text_snapshot(
            base_data,
            manifest,
            temporary,
            verify_tokenizer=True,
        )
        os.replace(str(temporary), str(output))
        verification["snapshot"] = str(output)
        if manifest_output is not None:
            write_json_object(manifest_output.resolve(), manifest)
        return {
            "status": "exported",
            "output": str(output),
            "manifest": str(output / "text_snapshot_manifest.json"),
            "external_manifest": str(manifest_output.resolve()) if manifest_output else None,
            "source_verification": source_verification,
            "verification": verification,
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出或验证共享 Qwen 纯文本基座。")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--base", required=True)
    export.add_argument("--source-snapshot", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--relative-path", required=True)
    export.add_argument("--manifest-output", default="")
    export.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    export.add_argument("--report", default="")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--base", required=True)
    verify.add_argument("--snapshot-manifest", required=True)
    verify.add_argument("--snapshot", required=True)
    verify.add_argument("--skip-tokenizer", action="store_true")
    verify.add_argument("--load-smoke", action="store_true")
    verify.add_argument("--report", default="")
    return parser


def main(argv: Optional[list] = None) -> None:
    args = _parser().parse_args(argv)
    base = read_json_object(Path(args.base))
    if args.command == "export":
        result = export_text_snapshot(
            base=base,
            source_snapshot=Path(args.source_snapshot),
            output_dir=Path(args.output),
            relative_path=args.relative_path,
            dtype_name=args.dtype,
            manifest_output=Path(args.manifest_output) if args.manifest_output else None,
        )
    else:
        result = verify_text_snapshot(
            base,
            read_json_object(Path(args.snapshot_manifest)),
            Path(args.snapshot),
            verify_tokenizer=not args.skip_tokenizer,
        )
    if args.command == "verify" and args.load_smoke:
        base_data = validate_base_manifest(base)
        result["load_smoke"] = load_text_snapshot_smoke(
            Path(args.snapshot), int(base_data["model"]["parameter_count"])
        )
    if args.report:
        write_json_object(Path(args.report).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
