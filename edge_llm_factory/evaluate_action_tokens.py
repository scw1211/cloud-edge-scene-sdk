"""用途：在未见 JSONL 上评估场景 LoRA 的单 token 准确率、F1、有效率和生成时延。"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from edge_llm_factory.adapter_package import MANIFEST_NAME, validate_adapter_package
from edge_llm_factory.text_base import verify_text_snapshot
from edge_llm_factory.contracts import (
    ManifestError,
    read_json_object,
    sha256_file,
    validate_base_manifest,
    write_json_object,
)


def _rows(path: Path) -> List[Dict[str, Any]]:
    output = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError("测试集第 {} 行不是合法 JSON".format(line_number)) from exc
            if not isinstance(row, dict):
                raise ManifestError("测试集每行必须是对象")
            output.append(row)
    if not output:
        raise ManifestError("测试集为空")
    return output


def _messages(row: Mapping[str, Any]) -> List[Dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ManifestError("测试样本缺少 messages")
    result = [
        {"role": str(item.get("role")), "content": str(item.get("content"))}
        for item in messages
        if isinstance(item, dict)
    ]
    if len(result) != len(messages) or result[-1]["role"] != "assistant":
        raise ManifestError("测试样本最后一条消息必须是 assistant target")
    return result


def _prompt(row: Mapping[str, Any], tokenizer: Any, prompt_format: str) -> str:
    messages = _messages(row)
    if prompt_format == "raw_task":
        users = [message for message in messages[:-1] if message["role"] == "user"]
        if len(users) != 1:
            raise ManifestError("raw_task 测试样本必须恰好有一条 user 消息")
        return users[0]["content"]
    return tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )


def _classification(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> Dict[str, Any]:
    per_class = {}
    weighted_f1 = 0.0
    for label in labels:
        tp = sum(row["target"] == label and row["prediction"] == label for row in rows)
        fp = sum(row["target"] != label and row["prediction"] == label for row in rows)
        fn = sum(row["target"] == label and row["prediction"] != label for row in rows)
        support = sum(row["target"] == label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        weighted_f1 += support * f1
    count = len(rows)
    return {
        "accuracy": round(sum(row["correct"] for row in rows) / count, 6),
        "macro_f1": round(statistics.fmean(item["f1"] for item in per_class.values()), 6),
        "weighted_f1": round(weighted_f1 / count, 6),
        "per_class": per_class,
    }


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="评估标准单 token 场景 LoRA。")
    parser.add_argument("--base", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--snapshot_manifest", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--test_dataset_id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt_format", choices=["tokenizer_chat", "raw_task"], default="raw_task")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
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
        validate_adapter_package(adapter, base_path, require_gates=False)
    train_summary_path = adapter / "train_metrics.json"
    test_path = Path(args.test_jsonl).resolve()
    test_sha = sha256_file(test_path)
    if train_summary_path.is_file():
        training = read_json_object(train_summary_path)
        if test_sha in {training.get("train_jsonl_sha256"), training.get("val_jsonl_sha256")}:
            raise ManifestError("测试集与训练集或调参验证集完全相同")
    rows = _rows(test_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    slots = {str(row["token"]) for row in base["decision_protocol"]["slots"]}
    targets = []
    for row in rows:
        target = _messages(row)[-1]["content"].strip()
        if target not in slots:
            raise ManifestError("测试 target 不在基座动作槽中: {}".format(target))
        targets.append(target)

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, use_fast=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: Dict[str, Any] = {"local_files_only": True, "trust_remote_code": False}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), **model_kwargs)
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()

    def generate(row: Mapping[str, Any]) -> Dict[str, Any]:
        prompt = _prompt(row, tokenizer, args.prompt_format)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )
        encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        token = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        return {
            "prediction": token if token in slots else None,
            "raw_output": token,
            "latency_ms": round(latency_ms, 4),
            "prompt_tokens": int(encoded["input_ids"].shape[1]),
        }

    for index in range(max(0, args.warmup)):
        generate(rows[index % len(rows)])
    samples = []
    for row, target in zip(rows, targets):
        result = generate(row)
        result.update(
            {
                "event_id": row.get("event_id"),
                "target": target,
                "valid": result["prediction"] is not None,
                "correct": result["prediction"] == target,
            }
        )
        samples.append(result)
    labels = sorted(set(targets))
    classification = _classification(samples, labels)
    summary = {
        "task": "edge_llm_single_token_heldout_evaluation",
        "base_id": base["base_id"],
        "snapshot_validation": snapshot_report,
        "adapter": str(adapter),
        "test_dataset_id": args.test_dataset_id,
        "test_jsonl_sha256": test_sha,
        "test_set_used_for_training": False,
        "count": len(samples),
        "prompt_format": args.prompt_format,
        "valid_output_rate": round(sum(row["valid"] for row in samples) / len(samples), 6),
        "decision_accuracy": classification["accuracy"],
        "macro_f1": classification["macro_f1"],
        "weighted_f1": classification["weighted_f1"],
        "per_class": classification["per_class"],
        "average_generation_latency_ms": round(
            statistics.fmean(row["latency_ms"] for row in samples), 4
        ),
        "samples": samples,
    }
    write_json_object(Path(args.output).resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
