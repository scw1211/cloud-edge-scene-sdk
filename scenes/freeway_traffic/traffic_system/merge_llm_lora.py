"""用途：将交通决策 LoRA 合并进 Qwen 基座模型，供后续 GGUF 量化。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from traffic_system.model_modality import require_text_only_model, sanitize_tokenizer_export
from traffic_system.train_llm_sft_lora import PROJECT_ROOT, hide_project_datasets_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a traffic LoRA adapter into a base Qwen student model.")
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--adapter_dir", default="experiments/llm_sft/qwen35_0_8b_freeway_action_token_lora")
    parser.add_argument("--output_dir", default="experiments/llm_sft/qwen35_0_8b_freeway_action_token_merged_hf")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return torch.float16


def main() -> None:
    args = parse_args()
    hide_project_datasets_dir()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype_from_name(args.dtype),
        trust_remote_code=args.trust_remote_code,
        device_map="cpu",
    )
    require_text_only_model(base_model)
    model = PeftModel.from_pretrained(base_model, str(resolve_path(args.adapter_dir)))
    merged = model.merge_and_unload()
    modality_report = require_text_only_model(merged)
    merged.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(str(output_dir))
    removed_tokenizer_keys = sanitize_tokenizer_export(output_dir)

    summary: Dict[str, Any] = {
        "task": "merge_qwen_lora_student",
        "base_model": args.base_model,
        "adapter_dir": str(resolve_path(args.adapter_dir).relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "dtype": args.dtype,
        "model_modality": modality_report,
        "removed_multimodal_tokenizer_keys": list(removed_tokenizer_keys),
    }
    (output_dir / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
