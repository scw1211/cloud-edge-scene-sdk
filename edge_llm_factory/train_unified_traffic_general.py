"""在同一个 Qwen3.5-0.8B LoRA 中训练交通、数学与中文逻辑能力。

该入口独立于 ``train_general_kd``：后者继续保持“禁止场景样本”的硬门禁。
统一入口只接受 ``edge-llm-unified-traffic-general/v1`` 清单，并逐条验证交通
16 位 raw-task/A-F 单 token 契约。默认从现有交通 LoRA 继续训练，以降低灾难性
遗忘风险；训练数据按显式任务权重确定性重采样。
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
import random
import re
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


SCHEMA_VERSION = "edge-llm-unified-traffic-general/v1"
CATEGORIES = ("traffic_action", "math", "natural_language_reasoning")
TRAFFIC_PATTERN = re.compile(r"^[0-9]{16}$")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_number} JSON 无效") from exc
            if not isinstance(value, dict):
                raise ManifestError(f"{path}:{line_number} 必须是 JSON object")
            rows.append(value)
    if not rows:
        raise ManifestError(f"统一训练数据为空: {path}")
    return rows


def _messages(row: Mapping[str, Any], location: str) -> List[Dict[str, str]]:
    raw = row.get("messages")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ManifestError(f"{location} 缺少输入消息和 assistant target")
    messages: List[Dict[str, str]] = []
    for index, message in enumerate(raw):
        if not isinstance(message, Mapping):
            raise ManifestError(f"{location}.messages[{index}] 必须是 object")
        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            raise ManifestError(f"{location}.messages[{index}] role/content 无效")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "assistant":
        raise ManifestError(f"{location} 最后一条消息必须是 assistant target")
    return messages


def validate_rows(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    event_ids = set()
    general_fingerprints = set()
    for index, row in enumerate(rows, start=1):
        location = f"{split}[{index}]"
        category = str(row.get("category", ""))
        if category not in CATEGORIES:
            raise ManifestError(f"{location} 类别不受支持: {category}")
        event_id = str(row.get("event_id", "")).strip()
        fingerprint = str(row.get("prompt_fingerprint", "")).strip()
        if not event_id or event_id in event_ids:
            raise ManifestError(f"{location} event_id 缺失或重复")
        if not fingerprint:
            raise ManifestError(f"{location} prompt_fingerprint 缺失")
        event_ids.add(event_id)
        messages = _messages(row, location)
        prompt_format = str(row.get("prompt_format", ""))
        if category == "traffic_action":
            if prompt_format != "raw_task":
                raise ManifestError(f"{location} 交通样本必须使用 raw_task")
            if (
                len(messages) != 2
                or messages[0]["role"] != "user"
                or TRAFFIC_PATTERN.fullmatch(messages[0]["content"]) is None
                or messages[1]["content"] not in set("ABCDEF")
            ):
                raise ManifestError(f"{location} 违反交通 16 位/A-F 契约")
        else:
            if fingerprint in general_fingerprints:
                raise ManifestError(f"{location} 通用 prompt_fingerprint 重复")
            general_fingerprints.add(fingerprint)
            if prompt_format != "tokenizer_chat":
                raise ManifestError(f"{location} 通用样本必须使用 tokenizer_chat")
            if not any(message["role"] == "user" for message in messages[:-1]):
                raise ManifestError(f"{location} 通用样本缺少 user 消息")
            if category == "natural_language_reasoning" and messages[-1]["content"] not in set(
                "ABCD"
            ):
                raise ManifestError(f"{location} 中文逻辑 target 必须是 A-D")
        counts[category] = counts.get(category, 0) + 1
    missing = set(CATEGORIES) - set(counts)
    if missing:
        raise ManifestError(f"{split} 缺少统一任务类别: {sorted(missing)}")
    return dict(sorted(counts.items()))


def _format_texts(
    row: Mapping[str, Any], tokenizer: Any
) -> Tuple[str, str, str]:
    category = str(row["category"])
    messages = _messages(row, str(row.get("event_id", "row")))
    if category == "traffic_action":
        prompt = messages[0]["content"]
        return prompt, prompt + messages[-1]["content"], "raw_task"
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return prompt, full, "tokenizer_chat"


def tokenize_rows(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any, max_seq_length: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tokenized: List[Dict[str, Any]] = []
    sequence_lengths: List[int] = []
    target_lengths: List[int] = []
    category_lengths: Dict[str, List[int]] = {category: [] for category in CATEGORIES}
    for row in rows:
        prompt_text, full_text, prompt_format = _format_texts(row, tokenizer)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise ManifestError(f"chat/raw assistant 前缀不一致: {row['event_id']}")
        if len(input_ids) > max_seq_length:
            raise ManifestError(
                f"样本 {row['event_id']} 长度 {len(input_ids)} 超过 {max_seq_length}；拒绝截断"
            )
        target_length = len(input_ids) - len(prompt_ids)
        if target_length <= 0:
            raise ManifestError(f"样本 {row['event_id']} 没有监督 token")
        category = str(row["category"])
        if category == "traffic_action":
            if len(prompt_ids) != 16:
                raise ManifestError(
                    f"交通样本 {row['event_id']} tokenizer 输入不是 16 token: {len(prompt_ids)}"
                )
            if target_length != 1:
                raise ManifestError(
                    f"交通样本 {row['event_id']} 输出不是单 token: {target_length}"
                )
        if category == "natural_language_reasoning" and target_length > 3:
            # Chat template may append an end token; the answer letter itself is one token.
            raise ManifestError(
                f"中文逻辑样本 {row['event_id']} target token 异常: {target_length}"
            )
        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :],
                "_category": category,
                "_prompt_format": prompt_format,
            }
        )
        sequence_lengths.append(len(input_ids))
        target_lengths.append(target_length)
        category_lengths[category].append(len(input_ids))
    stats = {
        "rows": len(tokenized),
        "max_sequence_tokens": max(sequence_lengths),
        "mean_sequence_tokens": round(sum(sequence_lengths) / len(sequence_lengths), 3),
        "max_target_tokens": max(target_lengths),
        "mean_target_tokens": round(sum(target_lengths) / len(target_lengths), 3),
        "category_sequence_tokens": {
            category: {
                "rows": len(lengths),
                "mean": round(sum(lengths) / len(lengths), 3),
                "max": max(lengths),
            }
            for category, lengths in category_lengths.items()
        },
    }
    return tokenized, stats


def parse_task_weights(value: str) -> Dict[str, int]:
    weights: Dict[str, int] = {}
    for part in value.split(","):
        if not part.strip() or "=" not in part:
            raise ManifestError("task_weights 格式应为 category=positive_integer,...")
        category, raw_weight = part.split("=", 1)
        category = category.strip()
        if category not in CATEGORIES or category in weights:
            raise ManifestError(f"task_weights 类别未知或重复: {category}")
        try:
            weight = int(raw_weight)
        except ValueError as exc:
            raise ManifestError(f"task_weights 权重不是整数: {part}") from exc
        if weight <= 0:
            raise ManifestError("task_weights 必须是正整数")
        weights[category] = weight
    if set(weights) != set(CATEGORIES):
        raise ManifestError(f"task_weights 必须完整声明 {list(CATEGORIES)}")
    return weights


def build_sampling_indices(
    rows: Sequence[Mapping[str, Any]], weights: Mapping[str, int], seed: int
) -> Tuple[List[int], Dict[str, Any]]:
    groups: Dict[str, List[int]] = {category: [] for category in CATEGORIES}
    for index, row in enumerate(rows):
        groups[str(row["_category"])].append(index)
    if any(not values for values in groups.values()):
        raise ManifestError("统一训练 token 中缺少任务类别")
    unit = max(
        math.ceil(len(groups[category]) / int(weights[category]))
        for category in CATEGORIES
    )
    rng = random.Random(seed)
    output: List[int] = []
    effective: Dict[str, int] = {}
    for category in CATEGORIES:
        target = unit * int(weights[category])
        chosen: List[int] = []
        while len(chosen) < target:
            cycle = list(groups[category])
            rng.shuffle(cycle)
            chosen.extend(cycle[: target - len(chosen)])
        output.extend(chosen)
        effective[category] = target
    rng.shuffle(output)
    return output, {
        "weights": dict(weights),
        "unique_category_counts": {
            category: len(groups[category]) for category in CATEGORIES
        },
        "effective_category_counts": effective,
        "effective_rows_per_epoch": len(output),
        "unique_rows": len(rows),
        "oversampled_rows_per_epoch": len(output) - len(rows),
        "effective_ratios": {
            category: round(effective[category] / len(output), 6)
            for category in CATEGORIES
        },
    }


class TokenizedDataset:
    def __init__(
        self, rows: Sequence[Dict[str, Any]], indices: Optional[Sequence[int]] = None
    ) -> None:
        self.rows = list(rows)
        self.indices = list(indices) if indices is not None else list(range(len(rows)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        row = self.rows[self.indices[index]]
        return {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, rows: Sequence[Dict[str, List[int]]]) -> Dict[str, Any]:
        import torch

        maximum = max(len(row["input_ids"]) for row in rows)
        return {
            "input_ids": torch.tensor(
                [
                    row["input_ids"]
                    + [self.pad_token_id] * (maximum - len(row["input_ids"]))
                    for row in rows
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [
                    row["attention_mask"]
                    + [0] * (maximum - len(row["attention_mask"]))
                    for row in rows
                ],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [
                    row["labels"] + [-100] * (maximum - len(row["labels"]))
                    for row in rows
                ],
                dtype=torch.long,
            ),
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
    # 交通只有 17 token，而数学/逻辑通常约 200 token。按长度组批可避免
    # 交通样本被长文本 padding 拖慢，同时不改变任何样本或监督权重。
    if "train_sampling_strategy" in signature.parameters:
        values["train_sampling_strategy"] = "group_by_length"
    elif "group_by_length" in signature.parameters:
        values["group_by_length"] = True
    values[
        "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    ] = "epoch"
    if args.bf16:
        values["bf16"] = True
    elif args.fp16:
        values["fp16"] = True
    return values


def _model_report(model: Any) -> Dict[str, Any]:
    markers = ("vision", "visual", "image", "video", "audio", "speech")
    total = 0
    forbidden = []
    for name, parameter in model.named_parameters():
        total += int(parameter.numel())
        if any(marker in name.lower() for marker in markers):
            forbidden.append(name)
    if forbidden:
        raise ManifestError(f"统一模型包含多模态参数: {forbidden[:5]}")
    return {"parameter_count": total, "modality": "text_only"}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--snapshot_manifest", required=True)
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--resume_adapter", required=True)
    parser.add_argument(
        "--expected_resume_adapter_sha256",
        required=True,
        help="预注册的生产交通 LoRA 权重 SHA-256，防止从错误适配器继续训练",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument(
        "--task_weights",
        default="traffic_action=2,math=1,natural_language_reasoning=1",
    )
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args(argv)
    if args.bf16 and args.fp16:
        raise ManifestError("bf16 与 fp16 不能同时启用")
    if min(
        args.max_seq_length,
        args.epochs,
        args.batch_size,
        args.eval_batch_size,
        args.gradient_accumulation,
    ) <= 0:
        raise ManifestError("训练长度、轮数和 batch 参数必须为正数")
    weights = parse_task_weights(args.task_weights)

    base_path = Path(args.base).resolve()
    base = validate_base_manifest(read_json_object(base_path))
    snapshot = Path(args.snapshot).resolve()
    snapshot_report = verify_text_snapshot(
        base,
        read_json_object(Path(args.snapshot_manifest)),
        snapshot,
        verify_tokenizer=True,
    )
    manifest_path = Path(args.dataset_manifest).resolve()
    manifest = read_json_object(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("统一数据 manifest schema 不受支持")
    for field in (
        "traffic_test_used_for_training",
        "gsm8k_test_loaded",
        "logiqa_test_loaded",
    ):
        if manifest.get(field) is not False:
            raise ManifestError(f"统一数据测试隔离字段无效: {field}")
    if manifest.get("general_train_validation_prompt_overlap") != 0:
        raise ManifestError("统一通用训练和验证 prompt 存在重叠")

    train_path = Path(args.train_jsonl).resolve()
    val_path = Path(args.val_jsonl).resolve()
    if train_path == val_path:
        raise ManifestError("统一训练集和验证集不能相同")
    artifacts = manifest.get("artifacts", {})
    if artifacts.get("train", {}).get("sha256") != sha256_file(train_path):
        raise ManifestError("统一训练数据 SHA-256 与 manifest 不一致")
    if artifacts.get("validation", {}).get("sha256") != sha256_file(val_path):
        raise ManifestError("统一验证数据 SHA-256 与 manifest 不一致")
    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)
    train_counts = validate_rows(train_rows, "train")
    val_counts = validate_rows(val_rows, "validation")
    if train_counts != manifest.get("train_category_counts"):
        raise ManifestError("统一训练类别计数与 manifest 不一致")
    if val_counts != manifest.get("validation_category_counts"):
        raise ManifestError("统一验证类别计数与 manifest 不一致")
    train_fingerprints = {
        str(row["prompt_fingerprint"])
        for row in train_rows
        if row["category"] != "traffic_action"
    }
    val_fingerprints = {
        str(row["prompt_fingerprint"])
        for row in val_rows
        if row["category"] != "traffic_action"
    }
    if train_fingerprints & val_fingerprints:
        raise ManifestError("统一通用训练集和验证集 prompt fingerprint 重叠")

    resume_adapter = Path(args.resume_adapter).resolve()
    for required in ("adapter_config.json", "adapter_model.safetensors"):
        if not (resume_adapter / required).is_file():
            raise ManifestError(f"交通起始 LoRA 缺少 {required}")
    resume_weights_sha256 = sha256_file(resume_adapter / "adapter_model.safetensors")
    if resume_weights_sha256 != args.expected_resume_adapter_sha256:
        raise ManifestError("交通起始 LoRA 权重不等于预注册生产 SHA-256")
    adapter_config = read_json_object(resume_adapter / "adapter_config.json")
    if adapter_config.get("base_model_name_or_path") != base["source"]["model_id"]:
        raise ManifestError("交通起始 LoRA 与锁定基座不兼容")

    import torch
    from peft import PeftModel
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
    val_tokens, val_token_stats = tokenize_rows(
        val_rows, tokenizer, args.max_seq_length
    )
    train_indices, sampling_report = build_sampling_indices(
        train_tokens, weights, args.seed
    )

    model_kwargs: Dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["dtype"] = torch.bfloat16 if args.bf16 else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(str(snapshot), **model_kwargs)
    model_report = _model_report(base_model)
    if model_report["parameter_count"] != int(base["model"]["parameter_count"]):
        raise ManifestError("统一模型基座参数量与清单不一致")
    model = PeftModel.from_pretrained(base_model, str(resume_adapter), is_trainable=True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ManifestError("拒绝覆盖非空统一模型训练目录")
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

    output_adapter_config_path = output / "adapter_config.json"
    output_adapter_config = read_json_object(output_adapter_config_path)
    output_adapter_config["base_model_name_or_path"] = base["source"]["model_id"]
    output_adapter_config["revision"] = base["source"]["revision"]
    write_json_object(output_adapter_config_path, output_adapter_config)
    cuda_memory = None
    if torch.cuda.is_available():
        cuda_memory = {
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 3),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 3),
        }
    summary = {
        "task": "single_model_traffic_math_chinese_logic_lora",
        "candidate_id": "unified-v1-U1",
        "base_id": base["base_id"],
        "base_fingerprint": base_fingerprint(base),
        "snapshot_validation": snapshot_report,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "source_traffic_adapter": {
            "path": str(resume_adapter),
            "weights_sha256": resume_weights_sha256,
            "config_sha256": sha256_file(resume_adapter / "adapter_config.json"),
        },
        "unified_model_candidate": True,
        "traffic_scene_samples": train_counts["traffic_action"],
        "general_samples": train_counts["math"]
        + train_counts["natural_language_reasoning"],
        "formal_evaluation_used_for_training": False,
        "traffic_test_used_for_training": False,
        "gsm8k_test_loaded": False,
        "logiqa_test_loaded": False,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_category_counts": train_counts,
        "validation_category_counts": val_counts,
        "train_tokenization": train_token_stats,
        "validation_tokenization": val_token_stats,
        "task_sampling": sampling_report,
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
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
        "promotion_status": "unvalidated",
        "production_traffic_release_modified": False,
    }
    write_json_object(output / "train_metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
