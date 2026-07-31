"""用途：评估 SFT 后 Qwen Student 的动作准确率、风险准确率和节点 F1。"""

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from traffic_system.decision_utils import read_jsonl
from traffic_system.model_modality import require_text_only_model
from traffic_system.train_llm_sft_lora import PROJECT_ROOT, hide_project_datasets_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a phase-1 SFT Qwen traffic student.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--adapter_dir", default="experiments/llm_sft/qwen35_0_8b_freeway_action_token_lora")
    parser.add_argument("--test_jsonl", default="datasets/llm_sft_freeway_action_token/test.jsonl")
    parser.add_argument("--output_json", default="results/llm/llm_action_token_hf_eval.json")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--include_class_scores",
        action="store_true",
        help="Record A-F next-token probabilities and the highest-scoring wrong class.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--unique_event_ids",
        action="store_true",
        help="Evaluate only the first row for each event_id (useful for balanced-repeat training sets).",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--prompt_format",
        choices=["tokenizer_chat", "manual_user_chat", "raw_task"],
        default="tokenizer_chat",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def prompt_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = row.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Invalid SFT row messages.")
    return [dict(message) for message in messages[:-1]]


def target_from_row(row: Dict[str, Any]) -> Any:
    target = row.get("target")
    if isinstance(target, (dict, str)):
        return target
    messages = row.get("messages", [])
    if isinstance(messages, list) and messages:
        content = messages[-1].get("content")
        if isinstance(content, str):
            parsed = extract_json_object(content)
            if parsed is not None:
                return parsed
            if content.strip():
                return content.strip()
    raise ValueError("SFT row missing target.")


def as_set(values: Any) -> set:
    if not isinstance(values, list):
        return set()
    result = set()
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            result.add(str(value))
    return result


def node_f1(predicted: Any, target: Any) -> float:
    pred = as_set(predicted)
    gold = as_set(target)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_student(args: argparse.Namespace) -> Any:
    hide_project_datasets_dir()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    require_text_only_model(model)
    model = PeftModel.from_pretrained(model, str(resolve_path(args.adapter_dir)))
    model.eval()
    return tokenizer, model


def generate_one(row: Dict[str, Any], tokenizer: Any, model: Any, args: argparse.Namespace) -> Dict[str, Any]:
    messages = prompt_messages(row)
    if args.prompt_format in {"manual_user_chat", "raw_task"}:
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError("Compact prompt evaluation requires exactly one user message.")
        user = str(messages[0].get("content", ""))
        if args.prompt_format == "raw_task":
            prompt_text = user
        else:
            prompt_text = (
                "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
            )
    else:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_length,
        add_special_tokens=False,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0.0:
        generate_kwargs["temperature"] = args.temperature
    start = time.perf_counter()
    if args.include_class_scores:
        generate_kwargs["return_dict_in_generate"] = True
        generate_kwargs["output_scores"] = True
    with torch.inference_mode():
        generated_output = model.generate(**inputs, **generate_kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if args.include_class_scores:
        output_ids = generated_output.sequences
        first_token_logits = generated_output.scores[0][0].float()
    else:
        output_ids = generated_output
        first_token_logits = None
    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    target = target_from_row(row)
    if isinstance(target, str):
        match = re.search(r"[A-F]", text.upper())
        predicted_token = match.group(0) if match else None
        result = {
            "event_id": row.get("event_id"),
            "latency_ms": round(elapsed_ms, 3),
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "raw_output": text,
            "parsed": predicted_token,
            "target": target,
            "json_valid": predicted_token is not None,
            "decision_match": predicted_token == target,
            "risk_match": None,
            "node_f1": None,
        }
        if first_token_logits is not None:
            token_ids = {}
            for label in "ABCDEF":
                encoded = tokenizer(label, add_special_tokens=False)["input_ids"]
                if len(encoded) != 1:
                    raise ValueError("Action label {} is not one tokenizer token.".format(label))
                token_ids[label] = int(encoded[0])
            selected_logits = torch.stack(
                [first_token_logits[token_ids[label]] for label in "ABCDEF"]
            )
            probabilities = torch.softmax(selected_logits, dim=0).cpu().tolist()
            result["class_probabilities"] = {
                label: round(float(probability), 8)
                for label, probability in zip("ABCDEF", probabilities)
            }
            result["hard_negative"] = max(
                (label for label in "ABCDEF" if label != target),
                key=lambda label: result["class_probabilities"][label],
            )
        return result

    parsed = extract_json_object(text)
    return {
        "event_id": row.get("event_id"),
        "latency_ms": round(elapsed_ms, 3),
        "raw_output": text,
        "parsed": parsed,
        "target": target,
        "json_valid": parsed is not None,
        "decision_match": bool(parsed and parsed.get("decision") == target.get("decision")),
        "risk_match": bool(parsed and parsed.get("global_risk_level") == target.get("global_risk_level")),
        "node_f1": round(node_f1(parsed.get("affected_nodes") if parsed else [], target.get("affected_nodes")), 4),
    }


def average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def unique_event_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    seen = set()
    for row in rows:
        event_id = str(row.get("event_id"))
        if event_id in seen:
            continue
        seen.add(event_id)
        output.append(row)
    return output


def token_classification_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    labels = sorted(
        set("ABCDEF")
        & {
            str(value)
            for row in rows
            for value in (row.get("target"), row.get("parsed"))
            if value is not None
        }
    )
    per_class: Dict[str, Dict[str, float]] = {}
    total = len(rows)
    weighted_f1 = 0.0
    for label in labels:
        true_positive = sum(
            1 for row in rows if row.get("target") == label and row.get("parsed") == label
        )
        false_positive = sum(
            1 for row in rows if row.get("target") != label and row.get("parsed") == label
        )
        false_negative = sum(
            1 for row in rows if row.get("target") == label and row.get("parsed") != label
        )
        support = sum(1 for row in rows if row.get("target") == label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        weighted_f1 += f1 * support
    return {
        "labels": labels,
        "accuracy": round(
            sum(1 for row in rows if row.get("target") == row.get("parsed")) / max(1, total),
            4,
        ),
        "macro_precision": round(average([row["precision"] for row in per_class.values()]), 4),
        "macro_recall": round(average([row["recall"] for row in per_class.values()]), 4),
        "macro_f1": round(average([row["f1"] for row in per_class.values()]), 4),
        "weighted_f1": round(weighted_f1 / max(1, total), 4),
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(resolve_path(args.test_jsonl))
    source_count = len(rows)
    if args.unique_event_ids:
        rows = unique_event_rows(rows)
    if args.limit > 0:
        rows = rows[: args.limit]
    tokenizer, model = load_student(args)

    for _ in range(max(0, args.warmup)):
        generate_one(rows[0], tokenizer, model, args)

    examples = [generate_one(row, tokenizer, model, args) for row in rows]
    count = len(examples)
    risk_values = [item["risk_match"] for item in examples if item["risk_match"] is not None]
    node_values = [item["node_f1"] for item in examples if item["node_f1"] is not None]
    token_rows = [item for item in examples if isinstance(item.get("target"), str)]
    per_class = {}
    for label in sorted({str(item["target"]) for item in token_rows}):
        class_rows = [item for item in token_rows if item["target"] == label]
        per_class[label] = {
            "total": len(class_rows),
            "correct": sum(1 for item in class_rows if item["decision_match"]),
            "accuracy": round(average([1.0 if item["decision_match"] else 0.0 for item in class_rows]), 4),
        }
    confusion = Counter(
        "{}->{}".format(item["target"], item.get("parsed") or "invalid")
        for item in token_rows
    )
    summary = {
        "task": "phase1_qwen_sft_generation_eval",
        "model_name_or_path": args.model_name_or_path,
        "adapter_dir": str(resolve_path(args.adapter_dir).relative_to(PROJECT_ROOT)),
        "test_jsonl": str(resolve_path(args.test_jsonl).relative_to(PROJECT_ROOT)),
        "count": count,
        "source_count": source_count,
        "unique_event_ids": bool(args.unique_event_ids),
        "class_scores_included": bool(args.include_class_scores),
        "warmup_runs": max(0, args.warmup),
        "prompt_format": args.prompt_format,
        "json_valid_rate": round(average([1.0 if item["json_valid"] else 0.0 for item in examples]), 4),
        "decision_accuracy": round(average([1.0 if item["decision_match"] else 0.0 for item in examples]), 4),
        "risk_accuracy": round(average([1.0 if value else 0.0 for value in risk_values]), 4) if risk_values else None,
        "affected_node_f1": round(average([float(value) for value in node_values]), 4) if node_values else None,
        "average_generation_latency_ms": round(average([float(item["latency_ms"]) for item in examples]), 3),
        "max_generation_latency_ms": round(max([float(item["latency_ms"]) for item in examples]) if examples else 0.0, 3),
        "per_class": per_class,
        "token_classification": token_classification_metrics(token_rows) if token_rows else None,
        "confusion": dict(sorted(confusion.items())),
        "examples": examples,
    }
    output_path = resolve_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
