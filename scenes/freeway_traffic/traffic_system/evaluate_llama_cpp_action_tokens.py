"""在 llama.cpp 服务上验收单 token 交通决策模型及量化一致性。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import time
from typing import Any, Dict, List, Mapping, Optional
import urllib.request

from traffic_system.decision_utils import read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_TOKEN = re.compile(r"^[A-F]$")


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--host", default="http://127.0.0.1:18190")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--reference_json", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout_seconds", type=float, default=2.0)
    return parser.parse_args()


def _prompt(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("row.messages must be a list")
    users = [
        str(item.get("content", ""))
        for item in messages[:-1]
        if isinstance(item, dict) and item.get("role") == "user"
    ]
    if len(users) != 1 or len(users[0]) != 16:
        raise ValueError("routing-context-v2 row must contain one 16-char user prompt")
    return users[0]


def _target(row: Mapping[str, Any]) -> str:
    value = str(row.get("target", "")).strip().upper()
    if not VALID_TOKEN.fullmatch(value):
        raise ValueError("row.target must be one action token A-F")
    return value


def request_token(host: str, prompt: str, timeout_seconds: float) -> Dict[str, Any]:
    request = urllib.request.Request(
        host.rstrip("/") + "/completion",
        data=json.dumps(
            {
                "prompt": prompt,
                "temperature": 0.0,
                "top_p": 1.0,
                "n_predict": 1,
                "stream": False,
                "cache_prompt": False,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    raw = str(payload.get("content", "")).strip().upper()
    token = raw if VALID_TOKEN.fullmatch(raw) else None
    timings = payload.get("timings", {})
    return {
        "parsed": token,
        "raw_output": raw,
        "latency_ms": round(elapsed_ms, 4),
        "prompt_tokens": int(timings.get("prompt_n", 0) or 0),
        "predicted_tokens": int(timings.get("predicted_n", 0) or 0),
    }


def load_reference(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples", [])
    if not isinstance(examples, list):
        raise ValueError("reference_json.examples must be a list")
    return {
        str(item["event_id"]): str(item.get("parsed", ""))
        for item in examples
        if isinstance(item, dict) and item.get("event_id")
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(resolve_path(args.test_jsonl))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("test_jsonl is empty")
    reference = load_reference(
        resolve_path(args.reference_json) if args.reference_json else None
    )
    for index in range(max(0, args.warmup)):
        request_token(args.host, _prompt(rows[index % len(rows)]), args.timeout_seconds)

    examples: List[Dict[str, Any]] = []
    for row in rows:
        result = request_token(args.host, _prompt(row), args.timeout_seconds)
        event_id = str(row["event_id"])
        target = _target(row)
        examples.append(
            {
                "event_id": event_id,
                "target": target,
                **result,
                "valid": result["parsed"] is not None,
                "correct": result["parsed"] == target,
                "reference_token": reference.get(event_id),
                "reference_match": (
                    result["parsed"] == reference[event_id]
                    if event_id in reference
                    else None
                ),
            }
        )

    latencies = [float(item["latency_ms"]) for item in examples]
    comparable = [item for item in examples if item["reference_match"] is not None]
    summary = {
        "task": "llama_cpp_routing_context_v2_action_token_eval",
        "host": args.host,
        "test_jsonl": str(resolve_path(args.test_jsonl)),
        "count": len(examples),
        "valid_output_rate": round(
            sum(bool(item["valid"]) for item in examples) / len(examples), 6
        ),
        "decision_accuracy": round(
            sum(bool(item["correct"]) for item in examples) / len(examples), 6
        ),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 4),
            "p50": round(float(sorted(latencies)[len(latencies) // 2]), 4),
            "p95": round(
                float(sorted(latencies)[min(len(latencies) - 1, int(0.95 * len(latencies)))]),
                4,
            ),
            "max": round(max(latencies), 4),
        },
        "prompt_token_counts": dict(
            sorted(Counter(int(item["prompt_tokens"]) for item in examples).items())
        ),
        "max_prompt_tokens": max(int(item["prompt_tokens"]) for item in examples),
        "reference": {
            "path": args.reference_json or None,
            "comparable": len(comparable),
            "exact_match_rate": (
                round(
                    sum(bool(item["reference_match"]) for item in comparable)
                    / len(comparable),
                    6,
                )
                if comparable
                else None
            ),
        },
        "examples": examples,
    }
    output = resolve_path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    printable = {key: value for key, value in summary.items() if key != "examples"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
