"""用途：把通用 Teacher 行为蒸馏到共享纯文本 Qwen 的多 token LoRA。"""

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from edge_llm_factory.contracts import (
    ManifestError,
    base_fingerprint,
    read_json_object,
    sha256_file,
    validate_base_manifest,
    write_json_object,
)
from edge_llm_factory.text_base import verify_text_snapshot


GENERAL_CATEGORIES = {"code", "math", "natural_language_reasoning"}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError("{}:{} JSON 无效".format(path, line_number)) from exc
            if not isinstance(value, dict):
                raise ManifestError("{}:{} 必须是 JSON object".format(path, line_number))
            rows.append(value)
    if not rows:
        raise ManifestError("通用蒸馏数据为空: {}".format(path))
    return rows


def _messages(row: Mapping[str, Any]) -> List[Dict[str, str]]:
    raw = row.get("messages")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ManifestError("通用蒸馏样本必须包含输入消息和 assistant target")
    messages = []
    for message in raw:
        if not isinstance(message, dict):
            raise ManifestError("messages 元素必须为 object")
        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            raise ManifestError("messages role/content 无效")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "assistant":
        raise ManifestError("通用蒸馏 target 必须是最后一条 assistant 消息")
    return messages


def validate_rows(rows: Sequence[Mapping[str, Any]], teacher_model: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    fingerprints = set()
    for row in rows:
        category = str(row.get("category", ""))
        if category not in GENERAL_CATEGORIES:
            raise ManifestError("通用蒸馏数据混入非通用类别: {}".format(category))
        if row.get("teacher_verified") is not True:
            raise ManifestError("通用蒸馏样本未经 Teacher 正确性验收")
        if str(row.get("teacher_model", "")) != teacher_model:
            raise ManifestError("样本 Teacher 与 manifest 不一致")
        fingerprint = str(row.get("prompt_fingerprint", ""))
        if not fingerprint or fingerprint in fingerprints:
            raise ManifestError("prompt fingerprint 缺失或重复")
        fingerprints.add(fingerprint)
        _messages(row)
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def tokenize_rows(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any, max_seq_length: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tokenized = []
    skipped_too_long = 0
    target_lengths = []
    sequence_lengths = []
    for row in rows:
        messages = _messages(row)
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise ManifestError("chat template 的 assistant 前缀不是完整样本前缀")
        if len(input_ids) > max_seq_length:
            skipped_too_long += 1
            continue
        target_length = len(input_ids) - len(prompt_ids)
        if target_length <= 0:
            raise ManifestError("assistant target 没有产生监督 token")
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "_category": str(row["category"]),
            }
        )
        target_lengths.append(target_length)
        sequence_lengths.append(len(input_ids))
    if not tokenized:
        raise ManifestError("所有通用蒸馏样本都超过 max_seq_length")
    stats = {
        "input_rows": len(rows),
        "tokenized_rows": len(tokenized),
        "skipped_too_long": skipped_too_long,
        "max_sequence_tokens": max(sequence_lengths),
        "mean_sequence_tokens": round(sum(sequence_lengths) / len(sequence_lengths), 3),
        "max_target_tokens": max(target_lengths),
        "mean_target_tokens": round(sum(target_lengths) / len(target_lengths), 3),
    }
    return tokenized, stats


def build_sampling_indices(
    rows: Sequence[Mapping[str, Any]], mode: str, seed: int
) -> Tuple[List[int], Dict[str, Any]]:
    if mode not in {"natural", "balanced"}:
        raise ManifestError("category_sampling 只支持 natural 或 balanced")
    groups: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        category = str(row.get("_category", ""))
        if category not in GENERAL_CATEGORIES:
            raise ManifestError("token 化样本缺少有效类别")
        groups.setdefault(category, []).append(index)
    unique_counts = {category: len(indices) for category, indices in sorted(groups.items())}
    if mode == "natural":
        indices = list(range(len(rows)))
        effective_counts = dict(unique_counts)
        replacement = False
    else:
        target = max(unique_counts.values())
        rng = random.Random(seed)
        indices = []
        for category in sorted(groups):
            chosen: List[int] = []
            while len(chosen) < target:
                cycle = list(groups[category])
                rng.shuffle(cycle)
                chosen.extend(cycle[: target - len(chosen)])
            indices.extend(chosen)
        rng.shuffle(indices)
        effective_counts = {category: target for category in sorted(groups)}
        replacement = any(count < target for count in unique_counts.values())
    report = {
        "mode": mode,
        "unique_category_counts": unique_counts,
        "effective_category_counts": effective_counts,
        "unique_rows": len(rows),
        "effective_rows_per_epoch": len(indices),
        "oversampled_rows_per_epoch": len(indices) - len(rows),
        "sampling_with_replacement": replacement,
    }
    return indices, report


class TokenizedDataset:
    def __init__(
        self, rows: Sequence[Dict[str, Any]], indices: Optional[Sequence[int]] = None
    ) -> None:
        self.rows = list(rows)
        self.indices = list(indices) if indices is not None else list(range(len(self.rows)))
        if any(index < 0 or index >= len(self.rows) for index in self.indices):
            raise ManifestError("采样索引越界")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        row = self.rows[self.indices[index]]
        return {key: value for key, value in row.items() if key != "_category"}


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, rows: Sequence[Dict[str, List[int]]]) -> Dict[str, Any]:
        import torch

        maximum = max(len(row["input_ids"]) for row in rows)
        input_ids = []
        attention = []
        labels = []
        for row in rows:
            padding = maximum - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            attention.append(row["attention_mask"] + [0] * padding)
            labels.append(row["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _training_arguments(args: argparse.Namespace, output: Path) -> Dict[str, Any]:
    from transformers import TrainingArguments

    values: Dict[str, Any] = {
        "output_dir": str(output),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "report_to": "none",
        "remove_unused_columns": False,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": args.seed,
        "dataloader_num_workers": 0,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    values["eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"] = (
        "epoch"
    )
    if args.bf16:
        values["bf16"] = True
    elif args.fp16:
        values["fp16"] = True
    return values


def _model_report(model: Any) -> Dict[str, Any]:
    markers = ("vision", "visual", "image", "video", "audio", "speech")
    parameter_count = 0
    forbidden = []
    for name, parameter in model.named_parameters():
        parameter_count += int(parameter.numel())
        if any(marker in name.lower() for marker in markers):
            forbidden.append(name)
    if forbidden:
        raise ManifestError("通用蒸馏基座包含多模态参数: {}".format(forbidden[:5]))
    return {"parameter_count": parameter_count, "modality": "text_only"}


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="训练场景无关的多 token 通用知识蒸馏 LoRA。")
    parser.add_argument("--base", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--snapshot_manifest", required=True)
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument("--category_sampling", choices=("natural", "balanced"), default="natural")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args(argv)

    if args.bf16 and args.fp16:
        raise ManifestError("bf16 与 fp16 不能同时启用")
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.gradient_accumulation) <= 0:
        raise ManifestError("训练轮数和 batch 参数必须为正数")
    if args.max_seq_length <= 0:
        raise ManifestError("max_seq_length 必须为正数")

    base = validate_base_manifest(read_json_object(Path(args.base)))
    snapshot = Path(args.snapshot).resolve()
    snapshot_report = verify_text_snapshot(
        base,
        read_json_object(Path(args.snapshot_manifest)),
        snapshot,
        verify_tokenizer=True,
    )
    manifest_path = Path(args.dataset_manifest).resolve()
    manifest = read_json_object(manifest_path)
    if manifest.get("schema_version") not in {"edge-llm-general-kd/v1", "edge-llm-general-kd/v2"}:
        raise ManifestError("通用蒸馏数据 manifest schema 不受支持")
    if manifest.get("scene_specific_samples") != 0:
        raise ManifestError("通用蒸馏数据禁止包含场景样本")
    if manifest.get("evaluation_prompt_overlap") != 0:
        raise ManifestError("通用蒸馏数据与冻结测试集存在重叠")
    if manifest.get("evaluation_set_used_for_training") is not False:
        raise ManifestError("冻结测试集使用声明无效")
    teacher_model = str(manifest.get("teacher_model", ""))
    if not teacher_model:
        raise ManifestError("通用蒸馏数据未声明 Teacher")

    train_path = Path(args.train_jsonl).resolve()
    val_path = Path(args.val_jsonl).resolve()
    if train_path == val_path:
        raise ManifestError("训练集和验证集不能相同")
    artifacts = manifest.get("artifacts", {})
    if artifacts.get("train", {}).get("sha256") != sha256_file(train_path):
        raise ManifestError("训练数据哈希与 manifest 不一致")
    if artifacts.get("validation", {}).get("sha256") != sha256_file(val_path):
        raise ManifestError("验证数据哈希与 manifest 不一致")
    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)
    train_counts = validate_rows(train_rows, teacher_model)
    val_counts = validate_rows(val_rows, teacher_model)
    train_fingerprints = {str(row["prompt_fingerprint"]) for row in train_rows}
    val_fingerprints = {str(row["prompt_fingerprint"]) for row in val_rows}
    if train_fingerprints & val_fingerprints:
        raise ManifestError("通用蒸馏训练集和验证集 prompt 重叠")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, use_fast=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_tokens, train_token_stats = tokenize_rows(
        train_rows, tokenizer, args.max_seq_length
    )
    val_tokens, val_token_stats = tokenize_rows(val_rows, tokenizer, args.max_seq_length)
    train_indices, sampling_report = build_sampling_indices(
        train_tokens, args.category_sampling, args.seed
    )

    model_kwargs: Dict[str, Any] = {"local_files_only": True, "trust_remote_code": False}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["dtype"] = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), **model_kwargs)
    model_report = _model_report(model)
    if model_report["parameter_count"] != int(base["model"]["parameter_count"]):
        raise ManifestError("通用蒸馏基座参数量与 manifest 不一致")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False

    target_modules = (
        [part.strip() for part in args.target_modules.split(",") if part.strip()]
        if args.target_modules
        else list(base["lora_policy"]["allowed_target_modules"])
    )
    if not set(target_modules).issubset(set(base["lora_policy"]["allowed_target_modules"])):
        raise ManifestError("target_modules 包含未授权模块")
    if args.rank > int(base["lora_policy"]["max_rank"]):
        raise ManifestError("LoRA rank 超过基座策略上限")
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ManifestError("拒绝覆盖非空通用蒸馏训练目录")
    output.mkdir(parents=True, exist_ok=True)
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": TrainingArguments(**_training_arguments(args, output)),
        "train_dataset": TokenizedDataset(train_tokens, train_indices),
        "eval_dataset": TokenizedDataset(val_tokens),
        "data_collator": CausalCollator(tokenizer.pad_token_id),
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))

    adapter_config_path = output / "adapter_config.json"
    adapter_config = read_json_object(adapter_config_path)
    adapter_config["base_model_name_or_path"] = base["source"]["model_id"]
    adapter_config["revision"] = base["source"]["revision"]
    write_json_object(adapter_config_path, adapter_config)
    cuda_memory = None
    if torch.cuda.is_available():
        cuda_memory = {
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 3),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 3),
        }
    summary = {
        "task": "scene_independent_general_behavior_distillation_lora",
        "base_id": base["base_id"],
        "base_fingerprint": base_fingerprint(base),
        "snapshot_validation": snapshot_report,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "teacher_model": teacher_model,
        "teacher_no_thinking": True,
        "scene_specific_samples": 0,
        "evaluation_prompt_overlap": 0,
        "evaluation_set_used_for_training": False,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_category_counts": train_counts,
        "validation_category_counts": val_counts,
        "train_tokenization": train_token_stats,
        "category_sampling": sampling_report,
        "validation_tokenization": val_token_stats,
        "max_seq_length": args.max_seq_length,
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target_modules": target_modules,
        },
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "bf16": bool(args.bf16),
            "fp16": bool(args.fp16),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
        },
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "cuda_memory": cuda_memory,
        "adapter_artifact": {
            "path": "adapter_model.safetensors",
            "sha256": sha256_file(output / "adapter_model.safetensors"),
        },
        "candidate_base_only": True,
        "not_a_scene_decision_adapter": True,
    }
    write_json_object(output / "train_metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
