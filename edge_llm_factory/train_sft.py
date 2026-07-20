"""用途：用标准动作 token 数据对锁定基座执行通用 LoRA/QLoRA 监督蒸馏。"""

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from edge_llm_factory.text_base import verify_text_snapshot
from edge_llm_factory.contracts import (
    ManifestError,
    base_fingerprint,
    read_json_object,
    sha256_file,
    validate_base_manifest,
    write_json_object,
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError("{} 第 {} 行不是合法 JSON".format(path, line_number)) from exc
            if not isinstance(row, dict):
                raise ManifestError("{} 第 {} 行必须是对象".format(path, line_number))
            rows.append(row)
    if not rows:
        raise ManifestError("蒸馏数据集为空: {}".format(path))
    return rows


def _messages(row: Mapping[str, Any]) -> List[Dict[str, str]]:
    raw = row.get("messages")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ManifestError("每条 SFT 数据必须包含输入消息和 assistant 动作 token")
    messages = []
    for message in raw:
        if not isinstance(message, dict):
            raise ManifestError("messages 中的元素必须是对象")
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role not in {"system", "user", "assistant"} or not content:
            raise ManifestError("messages role/content 无效")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "assistant":
        raise ManifestError("最后一条消息必须是 assistant target")
    return messages


def _validate_targets(
    rows: Sequence[Mapping[str, Any]], base: Mapping[str, Any]
) -> Dict[str, int]:
    allowed = {
        str(slot["token"])
        for slot in base["decision_protocol"]["slots"]
    }
    counts: Dict[str, int] = {}
    for row in rows:
        target = _messages(row)[-1]["content"].strip()
        if target not in allowed:
            raise ManifestError("SFT target 不是基座授权的单 token 动作: {!r}".format(target))
        counts[target] = counts.get(target, 0) + 1
    return dict(sorted(counts.items()))


def _format_texts(
    messages: Sequence[Mapping[str, str]], tokenizer: Any, prompt_format: str
) -> Tuple[str, str]:
    if prompt_format == "raw_task":
        user_messages = [row for row in messages[:-1] if row["role"] == "user"]
        if len(user_messages) != 1:
            raise ManifestError("raw_task 格式要求恰好一条 user 消息")
        prompt = user_messages[0]["content"]
        return prompt, prompt + messages[-1]["content"]
    prompt = tokenizer.apply_chat_template(
        list(messages[:-1]), tokenize=False, add_generation_prompt=True
    )
    # Deployment stops after one action token, so chat terminators are not targets.
    return prompt, prompt + messages[-1]["content"]


def _tokenize_rows(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_length: int,
    prompt_format: str,
) -> List[Dict[str, List[int]]]:
    output = []
    for row in rows:
        prompt, full = _format_texts(_messages(row), tokenizer, prompt_format)
        prompt_ids = tokenizer(
            prompt, add_special_tokens=False, truncation=True, max_length=max_length
        )["input_ids"]
        input_ids = tokenizer(
            full, add_special_tokens=False, truncation=True, max_length=max_length
        )["input_ids"]
        labels = [-100] * min(len(prompt_ids), len(input_ids))
        labels.extend(input_ids[len(labels) :])
        supervised = [value for value in labels if value != -100]
        if len(supervised) != 1:
            raise ManifestError(
                "单 token 训练样本应恰好保留 1 个监督 token，实际为 {}".format(len(supervised))
            )
        output.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
    return output


class _Dataset:
    def __init__(self, rows: Sequence[Dict[str, List[int]]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.rows[index]


class _Collator:
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


def _model_is_text_only(model: Any) -> Dict[str, Any]:
    markers = ("vision", "visual", "image", "video", "audio", "speech")
    total = 0
    forbidden = []
    for name, parameter in model.named_parameters():
        total += int(parameter.numel())
        if any(marker in name.lower() for marker in markers):
            forbidden.append(name)
    if forbidden:
        raise ManifestError("基座加载了多模态参数: {}".format(", ".join(forbidden[:5])))
    return {"parameter_count": total, "modality": "text_only"}


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
        "save_strategy": "no",
        "seed": args.seed,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    values["eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"] = "epoch"
    if args.bf16:
        values["bf16"] = True
    elif args.fp16:
        values["fp16"] = True
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="标准单 token 决策 LoRA/QLoRA SFT。")
    parser.add_argument("--base", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--snapshot_manifest", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt_format", choices=["tokenizer_chat", "raw_task"], default="raw_task")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[list] = None) -> None:
    args = _parser().parse_args(argv)
    if args.bf16 and args.fp16:
        raise ManifestError("bf16 与 fp16 不能同时启用")
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.gradient_accumulation) <= 0:
        raise ManifestError("训练轮数和 batch 参数必须为正数")
    base = validate_base_manifest(read_json_object(Path(args.base)))
    snapshot = Path(args.snapshot)
    snapshot_report = verify_text_snapshot(
        base,
        read_json_object(Path(args.snapshot_manifest)),
        snapshot,
        verify_tokenizer=True,
    )
    if args.max_length > int(base["decision_protocol"]["max_input_tokens"]):
        raise ManifestError("max_length 超过基座动作协议上限")
    if args.rank > int(base["lora_policy"]["max_rank"]):
        raise ManifestError("LoRA rank 超过基座策略上限")
    train_path = Path(args.train_jsonl).resolve()
    val_path = Path(args.val_jsonl).resolve()
    if train_path == val_path:
        raise ManifestError("训练集和验证集不能是同一个文件")
    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)
    train_targets = _validate_targets(train_rows, base)
    val_targets = _validate_targets(val_rows, base)

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, use_fast=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: Dict[str, Any] = {"local_files_only": True, "trust_remote_code": False}
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise ManifestError("QLoRA 4bit 训练要求可用 CUDA")
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), **model_kwargs)
    modality = _model_is_text_only(model)
    if modality["parameter_count"] != int(base["model"]["parameter_count"]):
        raise ManifestError("加载后的文本模型参数量与 base manifest 不一致")
    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    target_modules = (
        [part.strip() for part in args.target_modules.split(",") if part.strip()]
        if args.target_modules
        else list(base["lora_policy"]["allowed_target_modules"])
    )
    if not set(target_modules).issubset(set(base["lora_policy"]["allowed_target_modules"])):
        raise ManifestError("target_modules 包含基座未授权模块")
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
    train_data = _Dataset(_tokenize_rows(train_rows, tokenizer, args.max_length, args.prompt_format))
    val_data = _Dataset(_tokenize_rows(val_rows, tokenizer, args.max_length, args.prompt_format))
    output = Path(args.output).resolve()
    if output.exists():
        if not output.is_dir():
            raise ManifestError("训练输出路径不是目录: {}".format(output))
        if any(output.iterdir()):
            raise ManifestError("拒绝覆盖非空训练目录: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(**_training_arguments(args, output))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=_Collator(tokenizer.pad_token_id),
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    model.save_pretrained(str(output), safe_serialization=True)
    adapter_config_path = output / "adapter_config.json"
    adapter_config = read_json_object(adapter_config_path)
    adapter_config["base_model_name_or_path"] = base["source"]["model_id"]
    adapter_config["revision"] = base["source"]["revision"]
    write_json_object(adapter_config_path, adapter_config)
    summary = {
        "task": "edge_llm_single_token_lora_sft",
        "base_id": base["base_id"],
        "base_fingerprint": base_fingerprint(base),
        "snapshot_validation": snapshot_report,
        "train_jsonl_sha256": sha256_file(train_path),
        "val_jsonl_sha256": sha256_file(val_path),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_targets": train_targets,
        "val_targets": val_targets,
        "prompt_format": args.prompt_format,
        "max_length": args.max_length,
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target_modules": target_modules,
            "load_in_4bit": bool(args.load_in_4bit),
        },
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "test_set_used_for_training": False,
    }
    write_json_object(output / "train_metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
