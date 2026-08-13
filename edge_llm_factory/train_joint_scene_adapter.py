"""Thin, locked training entry for the joint traffic+industrial adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from edge_llm_factory import train_sft
from edge_llm_factory.build_joint_scene_dataset import PREFIXES, SCHEMA_VERSION
from edge_llm_factory.contracts import ManifestError, read_json_object, write_json_object


LOCKED_RECIPE = {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
    "epochs": 3.0,
    "batch_size": 8,
    "gradient_accumulation": 2,
    "learning_rate": 0.0001,
    "precision": "bf16",
    "max_length": 18,
    "seed": 20260813,
}
TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def audit_tokenizer_contract(snapshot: Path) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, use_fast=True, trust_remote_code=False
    )
    report: Dict[str, Any] = {"prefixes": {}, "contract": "17-to-1"}
    probes = ("0080000347123187", "2100299119910806")
    for scene, prefix in PREFIXES.items():
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        if len(prefix_ids) != 1:
            raise ManifestError("{} prefix 不是单 token".format(scene))
        lengths = [
            len(tokenizer(prefix + prompt, add_special_tokens=False)["input_ids"])
            for prompt in probes
        ]
        if lengths != [17, 17]:
            raise ManifestError("{} prefix 与 decimal16 未保持 17 tokens".format(scene))
        report["prefixes"][scene] = {
            "text": prefix,
            "token_id": int(prefix_ids[0]),
            "probe_token_lengths": lengths,
        }
    if report["prefixes"]["traffic"]["token_id"] == report["prefixes"]["industrial"]["token_id"]:
        raise ManifestError("traffic/industrial prefix token 冲突")
    return report


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("joint dataset manifest schema_version 不匹配")
    if manifest.get("formal_test_content_loaded") is not False:
        raise ManifestError("formal test content 必须保持未加载")
    if manifest.get("formal_test_used_for_training_or_selection") is not False:
        raise ManifestError("formal test 不得用于训练或选配方")
    if manifest.get("recipe") != {
        **LOCKED_RECIPE,
        "initialization": "clean_qwen35_0.8b_text",
        "target_modules": TARGET_MODULES.split(","),
    }:
        raise ManifestError("joint dataset recipe 不是冻结配方")
    for scene, count in (("traffic", 2400), ("industrial", 960)):
        source = manifest.get("sources", {}).get(scene, {})
        test = source.get("formal_test", {}) if isinstance(source, dict) else {}
        if (
            test.get("rows") != count
            or test.get("content_loaded") is not False
            or test.get("used_for_training") is not False
            or test.get("access_mode") != "sha256_and_stat_only"
        ):
            raise ManifestError("{} formal test isolation 无效".format(scene))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train the locked joint scene LoRA.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--snapshot_manifest", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir).resolve()
    manifest = read_json_object(dataset_dir / "manifest.json")
    validate_manifest(manifest)
    tokenizer_audit = audit_tokenizer_contract(Path(args.snapshot).resolve())
    train_sft.main(
        [
            "--base", args.base,
            "--snapshot", args.snapshot,
            "--snapshot_manifest", args.snapshot_manifest,
            "--train_jsonl", str(dataset_dir / "train.jsonl"),
            "--val_jsonl", str(dataset_dir / "validation.jsonl"),
            "--output", args.output,
            "--prompt_format", "raw_task",
            "--max_length", "18",
            "--epochs", "3.0",
            "--batch_size", "8",
            "--eval_batch_size", "8",
            "--gradient_accumulation", "2",
            "--learning_rate", "0.0001",
            "--weight_decay", "0.01",
            "--warmup_ratio", "0.05",
            "--rank", "16",
            "--alpha", "32",
            "--dropout", "0.05",
            "--target_modules", TARGET_MODULES,
            "--bf16",
            "--seed", "20260813",
        ]
    )
    summary_path = Path(args.output).resolve() / "train_metrics.json"
    summary = read_json_object(summary_path)
    summary.update(
        {
            "task": "joint_traffic_industrial_single_adapter_sft",
            "joint_dataset_manifest": str((dataset_dir / "manifest.json").resolve()),
            "joint_dataset_schema_version": SCHEMA_VERSION,
            "tokenizer_contract_audit": tokenizer_audit,
            "formal_test_content_loaded": False,
            "formal_test_used_for_training_or_selection": False,
            "runtime_contract": {
                "one_clean_base": True,
                "one_resident_adapter": True,
                "request_level_adapter_switching": False,
                "input": "T|I + decimal16",
                "input_tokens": 17,
                "output_tokens": 1,
            },
        }
    )
    write_json_object(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
