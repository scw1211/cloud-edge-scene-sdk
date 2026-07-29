"""用途：对 Qwen 交通 LoRA 执行原生 DPO 偏好蒸馏，无需额外 TRL 依赖。"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from traffic_system.decision_utils import read_jsonl
from traffic_system.model_modality import require_text_only_model
from traffic_system.train_llm_sft_lora import PROJECT_ROOT, hide_project_datasets_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO preference distillation for action-token Qwen.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--adapter_dir",
        default="experiments/llm_sft/qwen35_0_8b_freeway_action_token_v9_lora",
    )
    parser.add_argument(
        "--preference_jsonl",
        default="datasets/llm_sft_freeway_action_token_v9_phase2_preference/preference_train.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/llm_sft/qwen35_0_8b_freeway_action_token_v9_dpo_lora",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--sft_weight", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=16)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def tokenize_sequence(
    tokenizer: Any,
    prompt: str,
    completion: str,
    max_seq_length: int,
) -> Dict[str, List[int]]:
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )["input_ids"]
    full_ids = tokenizer(
        prompt + completion,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("Completion was truncated or merged before a supervised token remained.")
    completion_mask = [0] * len(full_ids)
    for index in range(len(prompt_ids), len(full_ids)):
        completion_mask[index] = 1
    return {"input_ids": full_ids, "completion_mask": completion_mask}


def tokenize_pairs(
    rows: Sequence[Dict[str, Any]], tokenizer: Any, max_seq_length: int
) -> List[Dict[str, Any]]:
    output = []
    for row in rows:
        prompt = str(row.get("prompt", ""))
        chosen = str(row.get("chosen", ""))
        rejected = str(row.get("rejected", ""))
        if not prompt or not chosen or not rejected or chosen == rejected:
            raise ValueError("Invalid preference row: {}".format(row.get("event_id")))
        output.append(
            {
                "event_id": str(row.get("event_id")),
                "chosen": tokenize_sequence(tokenizer, prompt, chosen, max_seq_length),
                "rejected": tokenize_sequence(tokenizer, prompt, rejected, max_seq_length),
            }
        )
    return output


def collate_sequences(
    sequences: Sequence[Dict[str, List[int]]], pad_token_id: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(len(row["input_ids"]) for row in sequences)
    input_ids = []
    attention_mask = []
    completion_mask = []
    for row in sequences:
        padding = max_length - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * padding)
        attention_mask.append([1] * len(row["input_ids"]) + [0] * padding)
        completion_mask.append(row["completion_mask"] + [0] * padding)
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
        torch.tensor(completion_mask, dtype=torch.bool, device=device),
    )


def completion_log_probabilities(
    model: Any,
    sequences: Sequence[Dict[str, List[int]]],
    pad_token_id: int,
) -> torch.Tensor:
    input_ids, attention_mask, completion_mask = collate_sequences(
        sequences, pad_token_id, model.device
    )
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    mask = completion_mask[:, 1:]
    token_log_probs = torch.gather(
        F.log_softmax(logits, dim=-1),
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1)


def load_model(args: argparse.Namespace, trainable: bool) -> Tuple[Any, Any]:
    hide_project_datasets_dir()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: Dict[str, Any] = {}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16
    base = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    require_text_only_model(base)
    model = PeftModel.from_pretrained(
        base,
        str(resolve_path(args.adapter_dir)),
        is_trainable=trainable,
    )
    model.config.use_cache = False
    if trainable:
        model.train()
    else:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
    return tokenizer, model


def batch_indices(length: int, batch_size: int) -> List[List[int]]:
    return [list(range(start, min(length, start + batch_size))) for start in range(0, length, batch_size)]


def precompute_reference(
    model: Any,
    pairs: Sequence[Dict[str, Any]],
    pad_token_id: int,
    batch_size: int,
) -> Dict[str, Tuple[float, float]]:
    result = {}
    with torch.inference_mode():
        for indices in batch_indices(len(pairs), batch_size):
            chosen = [pairs[index]["chosen"] for index in indices]
            rejected = [pairs[index]["rejected"] for index in indices]
            chosen_logps = completion_log_probabilities(model, chosen, pad_token_id)
            rejected_logps = completion_log_probabilities(model, rejected, pad_token_id)
            for index, chosen_logp, rejected_logp in zip(indices, chosen_logps, rejected_logps):
                result[pairs[index]["event_id"]] = (
                    float(chosen_logp.cpu()),
                    float(rejected_logp.cpu()),
                )
    return result


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.beta <= 0:
        raise ValueError("epochs, batch_size and beta must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_jsonl(resolve_path(args.preference_jsonl))
    if not rows:
        raise ValueError("Preference dataset is empty.")

    tokenizer, reference = load_model(args, trainable=False)
    pairs = tokenize_pairs(rows, tokenizer, args.max_seq_length)
    reference_logps = precompute_reference(
        reference, pairs, tokenizer.pad_token_id, args.batch_size
    )
    del reference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer, policy = load_model(args, trainable=True)
    trainable_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=0.0)
    metrics = []
    for epoch in range(args.epochs):
        order = list(range(len(pairs)))
        random.Random(args.seed + epoch).shuffle(order)
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            chosen = [pairs[index]["chosen"] for index in indices]
            rejected = [pairs[index]["rejected"] for index in indices]
            policy_chosen = completion_log_probabilities(policy, chosen, tokenizer.pad_token_id)
            policy_rejected = completion_log_probabilities(policy, rejected, tokenizer.pad_token_id)
            reference_chosen = torch.tensor(
                [reference_logps[pairs[index]["event_id"]][0] for index in indices],
                dtype=policy_chosen.dtype,
                device=policy_chosen.device,
            )
            reference_rejected = torch.tensor(
                [reference_logps[pairs[index]["event_id"]][1] for index in indices],
                dtype=policy_chosen.dtype,
                device=policy_chosen.device,
            )
            policy_margin = policy_chosen - policy_rejected
            reference_margin = reference_chosen - reference_rejected
            dpo_logits = args.beta * (policy_margin - reference_margin)
            dpo_loss = -F.logsigmoid(dpo_logits).mean()
            sft_loss = -policy_chosen.mean()
            loss = dpo_loss + args.sft_weight * sft_loss
            if not torch.isfinite(loss):
                raise RuntimeError("DPO produced a non-finite loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, args.max_grad_norm
            )
            if not math.isfinite(float(gradient_norm)):
                raise RuntimeError("DPO produced a non-finite gradient norm.")
            optimizer.step()
            metrics.append(
                {
                    "epoch": epoch + 1,
                    "step": len(metrics) + 1,
                    "loss": float(loss.detach().cpu()),
                    "dpo_loss": float(dpo_loss.detach().cpu()),
                    "sft_loss": float(sft_loss.detach().cpu()),
                    "policy_margin": float(policy_margin.mean().detach().cpu()),
                    "reference_margin": float(reference_margin.mean().detach().cpu()),
                    "preference_accuracy": float((policy_margin > 0).float().mean().detach().cpu()),
                    "gradient_norm": float(gradient_norm),
                }
            )
            if len(metrics) % 10 == 0 or len(metrics) == 1:
                print(json.dumps(metrics[-1], ensure_ascii=False), flush=True)

    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    summary = {
        "task": "qwen_action_token_dpo_preference_distillation",
        "base_adapter": args.adapter_dir,
        "preference_jsonl": args.preference_jsonl,
        "preference_pairs": len(pairs),
        "epochs": args.epochs,
        "steps": len(metrics),
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "sft_weight": args.sft_weight,
        "initial": metrics[0],
        "final": metrics[-1],
        "test_set_used_for_training": False,
        "model_modality": "text_only",
    }
    (output / "dpo_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
