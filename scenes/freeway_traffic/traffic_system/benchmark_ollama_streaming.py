"""用途：使用 Ollama 流式接口测量交通决策模型的 TTFT、准确率和内存。"""

import argparse
import json
import re
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traffic_system.benchmark_utils import ollama_rss_mb
from traffic_system.decision_utils import read_jsonl, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure streaming Ollama TTFT and output accuracy.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--mode", choices=["action_token", "json"], default="action_token")
    parser.add_argument("--num_ctx", type=int, default=128)
    parser.add_argument("--num_predict", type=int, default=1)
    parser.add_argument("--num_gpu", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep_alive", default="30m")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sample_interval_ms", type=float, default=10.0)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * ratio))
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def latency_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"average_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "average_ms": round(statistics.fmean(values), 4),
        "p50_ms": round(percentile(values, 0.50), 4),
        "p95_ms": round(percentile(values, 0.95), 4),
        "max_ms": round(max(values), 4),
    }


class MemorySampler:
    def __init__(self, interval_ms: float) -> None:
        self.interval = max(0.005, interval_ms / 1000.0)
        self.samples: List[float] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.samples.append(ollama_rss_mb())
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.samples.append(ollama_rss_mb())
        self.thread.start()

    def stop(self) -> None:
        self.samples.append(ollama_rss_mb())
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return round(max(self.samples), 4) if self.samples else 0.0


def prompt_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = row.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Invalid messages in {}".format(row.get("event_id")))
    return [dict(message) for message in messages[:-1]]


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def evaluate_output(row: Dict[str, Any], text: str, mode: str) -> Dict[str, Any]:
    target = row.get("target")
    if mode == "action_token":
        match = re.search(r"[A-F]", text.upper())
        prediction = match.group(0) if match else None
        return {
            "prediction": prediction,
            "target": target,
            "valid": prediction is not None,
            "correct": prediction == target,
        }
    parsed = extract_json(text)
    target_decision = target.get("decision") if isinstance(target, dict) else None
    prediction = parsed.get("decision") if parsed else None
    return {
        "prediction": prediction,
        "target": target_decision,
        "valid": parsed is not None,
        "correct": prediction == target_decision,
    }


def run_one(args: argparse.Namespace, row: Dict[str, Any], measured: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": args.model,
        "messages": prompt_messages(row),
        "stream": True,
        "think": False,
        "keep_alive": args.keep_alive,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
        },
    }
    if args.num_gpu is not None:
        payload["options"]["num_gpu"] = args.num_gpu
    if args.mode == "json":
        payload["format"] = "json"
    request = urllib.request.Request(
        args.host.rstrip("/") + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    sampler = MemorySampler(args.sample_interval_ms)
    sampler.start()
    started = time.perf_counter()
    first_content_at: Optional[float] = None
    content_parts = []
    final_chunk: Dict[str, Any] = {}
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                chunk = json.loads(raw_line.decode("utf-8"))
                content = str(chunk.get("message", {}).get("content", ""))
                if content and first_content_at is None:
                    first_content_at = time.perf_counter()
                if content:
                    content_parts.append(content)
                if chunk.get("done"):
                    final_chunk = chunk
    finally:
        sampler.stop()
    finished = time.perf_counter()
    text = "".join(content_parts).strip()
    evaluation = evaluate_output(row, text, args.mode)
    return {
        "event_id": row.get("event_id"),
        "measured": measured,
        "ttft_ms": round(((first_content_at or finished) - started) * 1000.0, 4),
        "total_latency_ms": round((finished - started) * 1000.0, 4),
        "prompt_eval_count": int(final_chunk.get("prompt_eval_count", 0)),
        "eval_count": int(final_chunk.get("eval_count", 0)),
        "load_duration_ms": round(final_chunk.get("load_duration", 0) / 1_000_000.0, 4),
        "prompt_eval_duration_ms": round(final_chunk.get("prompt_eval_duration", 0) / 1_000_000.0, 4),
        "eval_duration_ms": round(final_chunk.get("eval_duration", 0) / 1_000_000.0, 4),
        "peak_ollama_rss_mb": sampler.peak_mb,
        "raw_output": text,
        **evaluation,
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(resolve_path(args.test_jsonl))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No evaluation rows.")

    warmups = [run_one(args, rows[index % len(rows)], measured=False) for index in range(args.warmup)]
    samples = [run_one(args, row, measured=True) for row in rows]
    valid_count = sum(1 for sample in samples if sample["valid"])
    correct_count = sum(1 for sample in samples if sample["correct"])
    per_class = {}
    for label in sorted({str(sample["target"]) for sample in samples}):
        class_rows = [sample for sample in samples if str(sample["target"]) == label]
        per_class[label] = {
            "total": len(class_rows),
            "correct": sum(1 for sample in class_rows if sample["correct"]),
        }
    result = {
        "task": "ollama_streaming_ttft_benchmark",
        "model": args.model,
        "mode": args.mode,
        "test_jsonl": args.test_jsonl,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "num_gpu": args.num_gpu,
        "warmup_runs": args.warmup,
        "measured_runs": len(samples),
        "valid_rate": round(valid_count / len(samples), 6),
        "accuracy": round(correct_count / len(samples), 6),
        "ttft": latency_summary([sample["ttft_ms"] for sample in samples]),
        "total_latency": latency_summary([sample["total_latency_ms"] for sample in samples]),
        "prompt_tokens": {
            "average": round(statistics.fmean(sample["prompt_eval_count"] for sample in samples), 3),
            "min": min(sample["prompt_eval_count"] for sample in samples),
            "max": max(sample["prompt_eval_count"] for sample in samples),
        },
        "peak_ollama_rss_mb": max(sample["peak_ollama_rss_mb"] for sample in samples),
        "per_class": per_class,
        "warmups": warmups,
        "samples": samples,
    }
    save_json(result, resolve_path(args.output_json))
    print(json.dumps({key: value for key, value in result.items() if key not in {"warmups", "samples"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
