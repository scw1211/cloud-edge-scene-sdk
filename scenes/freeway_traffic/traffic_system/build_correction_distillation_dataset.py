"""用途：采集 Student 在线输出，并用缓存 Teacher 标签构建第二阶段纠错蒸馏数据。"""

import argparse
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from traffic_system.decision_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build on-policy correction distillation data.")
    parser.add_argument("--student_model", default="qwen35-freeway-action-general-eval:latest")
    parser.add_argument("--ollama_url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--phase1_dir", default="datasets/llm_sft_freeway_action_token")
    parser.add_argument("--output_dir", default="datasets/llm_sft_freeway_action_token_phase2")
    parser.add_argument("--correction_repeats", type=int, default=4)
    parser.add_argument("--rollout_splits", default="train,val")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def unique_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    seen = set()
    for row in rows:
        key = str(row.get("event_id"))
        if key not in seen:
            seen.add(key)
            output.append(row)
    return output


def student_predict(args: argparse.Namespace, row: Dict[str, Any]) -> Dict[str, Any]:
    messages = row.get("messages", [])
    payload = {
        "model": args.student_model,
        "messages": messages[:-1],
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "top_p": 1, "num_ctx": 128, "num_predict": 1},
    }
    request = urllib.request.Request(
        args.ollama_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    text = str(result.get("message", {}).get("content", "")).strip()
    match = re.search(r"[A-F]", text.upper())
    prediction = match.group(0) if match else None
    return {
        "event_id": row.get("event_id"),
        "target": str(row.get("target")),
        "prediction": prediction,
        "valid": prediction is not None,
        "correct": prediction == str(row.get("target")),
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 4),
        "raw_output": text,
    }


def main() -> None:
    args = parse_args()
    phase1 = Path(args.phase1_dir)
    output = Path(args.output_dir)
    train = read_jsonl(phase1 / "train.jsonl")
    val = read_jsonl(phase1 / "val.jsonl")
    rollout_source = []
    split_rows = {"train": train, "val": val}
    for split in [part.strip() for part in args.rollout_splits.split(",") if part.strip()]:
        if split not in split_rows:
            raise ValueError("Unsupported rollout split: {}".format(split))
        for row in unique_rows(split_rows[split]):
            annotated = dict(row)
            annotated["rollout_source_split"] = split
            rollout_source.append(annotated)
    if args.limit > 0:
        rollout_source = rollout_source[: args.limit]
    rollouts = []
    source_by_id = {str(row.get("event_id")): row for row in rollout_source}
    for index, row in enumerate(rollout_source, start=1):
        result = student_predict(args, row)
        rollouts.append(result)
        print(
            "[{}/{}] {} target={} pred={} correct={}".format(
                index, len(rollout_source), result["event_id"], result["target"],
                result["prediction"], result["correct"]
            ),
            flush=True,
        )

    errors = [row for row in rollouts if not row["correct"]]
    corrections = []
    for error in errors:
        source = source_by_id[str(error["event_id"])]
        corrected = dict(source)
        corrected["distillation_stage"] = "phase2_on_policy_correction"
        corrected["student_wrong_answer"] = error["prediction"]
        corrected["teacher_correction"] = error["target"]
        corrected["correction_source_split"] = source.get("rollout_source_split")
        corrections.append(corrected)
    phase2_train = list(train)
    for _ in range(max(1, args.correction_repeats)):
        phase2_train.extend(copy for copy in corrections)

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(phase2_train, output / "train.jsonl")
    write_jsonl(val, output / "val.jsonl")
    write_jsonl(rollouts, output / "student_rollouts.jsonl")
    summary = {
        "task": "phase2_on_policy_correction_distillation_dataset",
        "student_model": args.student_model,
        "teacher_correction_source": "cached_qwen3.5_9b_safety_constrained_labels",
        "test_set_used_for_training": False,
        "rollout_count": len(rollouts),
        "rollout_splits": [part.strip() for part in args.rollout_splits.split(",") if part.strip()],
        "valid_rate": round(sum(row["valid"] for row in rollouts) / len(rollouts), 6),
        "student_accuracy": round(sum(row["correct"] for row in rollouts) / len(rollouts), 6),
        "error_count": len(errors),
        "error_transitions": dict(sorted(Counter(
            "{}->{}".format(row["target"], row["prediction"] or "invalid") for row in errors
        ).items())),
        "correction_repeats": args.correction_repeats,
        "phase1_train_rows": len(train),
        "phase2_train_rows": len(phase2_train),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
