"""用途：通过 LoRA SFT 将云端 Teacher 的交通决策知识蒸馏到 Qwen Student。"""

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset as TorchDataset

from traffic_system.decision_utils import read_jsonl
from traffic_system.model_modality import require_text_only_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-1 LoRA SFT for traffic decision distillation into a Qwen student."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--train_jsonl", default="datasets/llm_sft_freeway_action_token/train.jsonl")
    parser.add_argument("--val_jsonl", default="datasets/llm_sft_freeway_action_token/val.jsonl")
    parser.add_argument("--output_dir", default="experiments/llm_sft/qwen35_0_8b_freeway_action_token_lora")
    parser.add_argument("--resume_adapter", default="")
    parser.add_argument(
        "--prompt_format",
        choices=["tokenizer_chat", "manual_user_chat", "raw_task", "mixed_by_row"],
        default="tokenizer_chat",
    )
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_true")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def require_packages() -> None:
    missing = []
    for package in ("transformers", "accelerate"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise RuntimeError(
            "Missing packages: {}. Install in traffic env with: "
            "pip install transformers accelerate peft".format(", ".join(missing))
        )


def check_trainable_model_path(model_name_or_path: str) -> None:
    if model_name_or_path.lower().endswith(".gguf"):
        raise ValueError(
            "GGUF is an inference/quantization format and cannot be LoRA-trained directly. "
            "Use the original Hugging Face text model, then export/quantize after training."
        )


def hide_project_datasets_dir() -> None:
    """Prevent the repo's datasets/ directory from shadowing Hugging Face datasets."""
    root = PROJECT_ROOT.resolve()
    cleaned = []
    for path in sys.path:
        if not path:
            candidate = Path.cwd().resolve()
        else:
            try:
                candidate = Path(path).resolve()
            except OSError:
                cleaned.append(path)
                continue
        if candidate == root:
            continue
        cleaned.append(path)
    sys.path[:] = cleaned

    module = sys.modules.get("datasets")
    module_file = getattr(module, "__file__", None) if module is not None else None
    if module_file:
        try:
            if path_is_relative_to(Path(module_file).resolve(), root):
                del sys.modules["datasets"]
                return
        except OSError:
            pass
    module_paths = getattr(module, "__path__", []) if module is not None else []
    for module_path in module_paths:
        try:
            if path_is_relative_to(Path(str(module_path)).resolve(), root):
                del sys.modules["datasets"]
                return
        except OSError:
            continue


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("No SFT records found: {}".format(path))
    for index, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("Row {} must contain user and assistant chat messages.".format(index))
    return rows


def assistant_prefix_messages(messages: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return [dict(message) for message in messages[:-1]]


def format_training_texts(
    messages: Sequence[Dict[str, str]], tokenizer: Any, prompt_format: str
) -> Tuple[str, str]:
    if prompt_format in {"manual_user_chat", "raw_task"}:
        if len(messages) != 2 or messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            raise ValueError("Compact prompt formats require exactly one user and one assistant message.")
        user = str(messages[0].get("content", ""))
        assistant = str(messages[1].get("content", ""))
        if prompt_format == "raw_task":
            return user, user + assistant
        prompt = "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
        return prompt, prompt + assistant + "<|im_end|>\n"
    prompt = tokenizer.apply_chat_template(
        assistant_prefix_messages(messages), tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return prompt, full


def tokenize_rows(
    rows: Sequence[Dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
    prompt_format: str = "tokenizer_chat",
) -> List[Dict[str, List[int]]]:
    tokenized = []
    for row in rows:
        messages = row["messages"]
        row_prompt_format = (
            str(row.get("prompt_format", "tokenizer_chat"))
            if prompt_format == "mixed_by_row"
            else prompt_format
        )
        if row_prompt_format not in {"tokenizer_chat", "manual_user_chat", "raw_task"}:
            raise ValueError("Unsupported row prompt format: {}".format(row_prompt_format))
        prompt_text, full_text = format_training_texts(messages, tokenizer, row_prompt_format)
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )["input_ids"]
        full = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = full["input_ids"]
        labels = [-100] * min(len(prompt_ids), len(input_ids))
        labels.extend(input_ids[len(labels):])
        if not any(label != -100 for label in labels):
            continue
        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
    if not tokenized:
        raise ValueError("All rows were truncated before assistant targets.")
    return tokenized


class CausalDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids = []
        batch_attention = []
        batch_labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            batch_attention.append(feature["attention_mask"] + [0] * pad_len)
            batch_labels.append(feature["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


class TokenizedSFTDataset(TorchDataset):
    def __init__(self, rows: Sequence[Dict[str, List[int]]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.rows[index]


def training_arguments_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    from transformers import TrainingArguments

    kwargs: Dict[str, Any] = {
        "output_dir": str(resolve_path(args.output_dir)),
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "seed": args.seed,
        "report_to": "none",
        "remove_unused_columns": False,
        "save_total_limit": 2,
    }
    if args.bf16:
        kwargs["bf16"] = True
    if args.fp16:
        kwargs["fp16"] = True
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    kwargs["eval_steps"] = args.eval_steps
    return kwargs


def main() -> None:
    args = parse_args()
    require_packages()
    check_trainable_model_path(args.model_name_or_path)
    hide_project_datasets_dir()

    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("load_in_4bit requires bitsandbytes-compatible transformers install.") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    modality_report = require_text_only_model(model)
    model.config.use_cache = False

    if args.resume_adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Continuing LoRA training requires peft.") from exc
        model = PeftModel.from_pretrained(model, str(resolve_path(args.resume_adapter)), is_trainable=True)
        model.print_trainable_parameters()
    elif not args.no_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError("LoRA training requires peft. Install with: pip install peft") from exc
        target_modules = [part.strip() for part in args.target_modules.split(",") if part.strip()]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_rows = load_rows(resolve_path(args.train_jsonl))
    val_rows = load_rows(resolve_path(args.val_jsonl))
    train_dataset = TokenizedSFTDataset(
        tokenize_rows(train_rows, tokenizer, args.max_seq_length, args.prompt_format)
    )
    val_dataset = TokenizedSFTDataset(
        tokenize_rows(val_rows, tokenizer, args.max_seq_length, args.prompt_format)
    )

    training_args = TrainingArguments(**training_arguments_kwargs(args))
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": CausalDataCollator(tokenizer.pad_token_id),
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(resolve_path(args.output_dir)))
    tokenizer.save_pretrained(str(resolve_path(args.output_dir)))

    metrics = {
        "task": "phase2_correction_lora_sft" if args.resume_adapter else "phase1_qwen_lora_sft",
        "model_name_or_path": args.model_name_or_path,
        "train_jsonl": str(resolve_path(args.train_jsonl).relative_to(PROJECT_ROOT)),
        "val_jsonl": str(resolve_path(args.val_jsonl).relative_to(PROJECT_ROOT)),
        "output_dir": str(resolve_path(args.output_dir).relative_to(PROJECT_ROOT)),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "lora": not args.no_lora,
        "resume_adapter": args.resume_adapter or None,
        "load_in_4bit": bool(args.load_in_4bit),
        "max_seq_length": args.max_seq_length,
        "prompt_format": args.prompt_format,
        "model_modality": modality_report,
    }
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
