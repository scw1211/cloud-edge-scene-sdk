"""用途：将已验证 LoRA 合并到锁定文本基座，生成可追溯的纯文本 HF 目录。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from edge_llm_factory.adapter_package import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_WEIGHTS_NAME,
    MANIFEST_NAME,
    inspect_safetensors,
    validate_adapter_package,
)
from edge_llm_factory.text_base import verify_text_snapshot
from edge_llm_factory.contracts import (
    ManifestError,
    read_json_object,
    sha256_file,
    validate_base_manifest,
    write_json_object,
)


MULTIMODAL_TOKENIZER_KEYS = {
    "audio_bos_token",
    "audio_eos_token",
    "audio_token",
    "image_token",
    "model_specific_special_tokens",
    "video_token",
    "vision_bos_token",
    "vision_eos_token",
}
TEXT_CHAT_TEMPLATE = """{%- for message in messages %}
{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}
"""


def _text_only_report(model: Any) -> Dict[str, Any]:
    markers = ("vision", "visual", "image", "video", "audio", "speech")
    total = 0
    forbidden = []
    for name, parameter in model.named_parameters():
        total += int(parameter.numel())
        if any(marker in name.lower() for marker in markers):
            forbidden.append(name)
    if forbidden:
        raise ManifestError("合并模型包含多模态参数: {}".format(", ".join(forbidden[:5])))
    return {"parameter_count": total, "modality": "text_only"}


def _sanitize_tokenizer(output: Path) -> None:
    config_path = output / "tokenizer_config.json"
    config = read_json_object(config_path)
    for key in MULTIMODAL_TOKENIZER_KEYS:
        config.pop(key, None)
    config["chat_template"] = TEXT_CHAT_TEMPLATE
    write_json_object(config_path, config)
    (output / "chat_template.jinja").write_text(TEXT_CHAT_TEMPLATE, encoding="utf-8")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="合并标准 LoRA 与锁定文本基座。")
    parser.add_argument("--base", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--snapshot_manifest", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_path = Path(args.base).resolve()
    base = validate_base_manifest(read_json_object(base_path))
    snapshot = Path(args.snapshot).resolve()
    snapshot_report = verify_text_snapshot(
        base,
        read_json_object(Path(args.snapshot_manifest)),
        snapshot,
        verify_tokenizer=True,
    )
    adapter = Path(args.adapter).resolve()
    if (adapter / MANIFEST_NAME).is_file():
        validate_adapter_package(adapter, base_path, require_gates=True)
    else:
        config = read_json_object(adapter / ADAPTER_CONFIG_NAME)
        if config.get("base_model_name_or_path") != base["source"]["model_id"]:
            raise ManifestError("LoRA 训练目录与锁定基座不兼容")
        inspect_safetensors(adapter / ADAPTER_WEIGHTS_NAME)

    output = Path(args.output).resolve()
    if output.exists():
        if not output.is_dir():
            raise ManifestError("合并输出路径不是目录: {}".format(output))
        if any(output.iterdir()):
            raise ManifestError("拒绝覆盖非空合并目录: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, use_fast=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtypes[args.dtype],
        device_map="cpu",
    )
    _text_only_report(model)
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    report = _text_only_report(merged)
    if report["parameter_count"] != base["model"]["parameter_count"]:
        raise ManifestError("合并模型参数量与基座清单不一致")
    merged.save_pretrained(str(output), safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(str(output))
    _sanitize_tokenizer(output)
    model_files = sorted(output.glob("*.safetensors"))
    if not model_files:
        raise ManifestError("合并目录未生成 safetensors 权重")
    training_summary_path = adapter / "train_metrics.json"
    training_summary = (
        read_json_object(training_summary_path) if training_summary_path.is_file() else {}
    )
    is_general_kd = training_summary.get("candidate_base_only") is True
    summary = {
        "task": (
            "merge_edge_llm_general_kd_candidate"
            if is_general_kd
            else "merge_edge_llm_scene_lora"
        ),
        "artifact_role": "general_kd_candidate_base" if is_general_kd else "scene_adapter",
        "base_id": base["base_id"],
        "snapshot_validation": snapshot_report,
        "adapter": str(adapter),
        "dtype": args.dtype,
        "model": report,
        "artifacts": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in model_files
        ],
    }
    write_json_object(output / "merge_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
