"""用途：使用 llama.cpp 测量单 token 交通决策的 TTFT、准确率和内存。"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traffic_system.benchmark_utils import (  # noqa: E402
    ProcessMemorySampler,
    SystemMemorySampler,
    stop_server,
    wait_until_ready,
)
from traffic_system.decision_utils import read_jsonl, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one-token action inference on llama.cpp.")
    parser.add_argument("--llama_server", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18190)
    parser.add_argument("--ctx_size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--threads_batch", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ubatch_size", type=int, default=64)
    parser.add_argument("--cache_reuse", type=int, default=1)
    parser.add_argument("--gpu_layers", type=int, default=0)
    parser.add_argument("--poll", type=int, default=50)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--threads_http", type=int, default=0)
    parser.add_argument("--no_cont_batching", action="store_true")
    parser.add_argument("--cache_prompt", dest="no_cache_prompt", action="store_false")
    parser.add_argument("--no_cache_prompt", dest="no_cache_prompt", action="store_true")
    parser.set_defaults(no_cache_prompt=True)
    parser.add_argument("--non_stream", action="store_true")
    parser.add_argument("--mlock", action="store_true")
    parser.add_argument("--mode", choices=["action_token", "ttft_only"], default="action_token")
    parser.add_argument("--prefill_no_think", action="store_true")
    parser.add_argument(
        "--prompt_format",
        choices=["full_chat", "user_chat", "raw_user"],
        default="full_chat",
    )
    parser.add_argument(
        "--warmup_jsonl",
        default="",
        help="Optional independent rows used only for warmup to avoid test-set KV cache hits.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--startup_timeout", type=int, default=60)
    parser.add_argument("--request_timeout", type=int, default=60)
    parser.add_argument("--sample_interval_ms", type=float, default=10.0)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def percentile(values: List[float], ratio: float) -> float:
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * ratio))
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def summarize_latency(values: List[float]) -> Dict[str, float]:
    return {
        "average_ms": round(statistics.fmean(values), 4),
        "p50_ms": round(percentile(values, 0.50), 4),
        "p95_ms": round(percentile(values, 0.95), 4),
        "max_ms": round(max(values), 4),
    }


def build_prompt(row: Dict[str, Any], prefill_no_think: bool, prompt_format: str) -> str:
    messages = row.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Invalid messages for {}".format(row.get("event_id")))
    user_message = messages[-2]
    user = str(user_message.get("content", ""))
    if prompt_format == "raw_user":
        prompt = user
    elif prompt_format == "user_chat":
        prompt = "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
    else:
        system = str(messages[0].get("content", "只答A-F。"))
        prompt = (
            "<|im_start|>system\n"
            + system
            + "<|im_end|>\n<|im_start|>user\n"
            + user
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
    if prefill_no_think:
        prompt += "<think>\n\n</think>\n\n"
    return prompt


def run_one(
    base_url: str,
    row: Dict[str, Any],
    timeout: int,
    measured: bool,
    mode: str,
    prefill_no_think: bool,
    prompt_format: str,
    stream: bool = True,
) -> Dict[str, Any]:
    payload = {
        "prompt": build_prompt(row, prefill_no_think, prompt_format),
        "temperature": 0,
        "top_p": 1,
        "n_predict": 1,
        "stream": stream,
        "cache_prompt": False,
    }
    request = urllib.request.Request(
        base_url + "/completion",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_content_at: Optional[float] = None
    parts: List[str] = []
    final_chunk: Dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not stream:
            final_chunk = json.loads(response.read().decode("utf-8"))
            content = str(final_chunk.get("content", ""))
            if content:
                parts.append(content)
        else:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data_text = line[5:].strip()
                if not data_text or data_text == "[DONE]":
                    continue
                chunk = json.loads(data_text)
                content = str(chunk.get("content", ""))
                if content and first_content_at is None:
                    first_content_at = time.perf_counter()
                if content:
                    parts.append(content)
                if chunk.get("stop"):
                    final_chunk = chunk
    finished = time.perf_counter()
    output = "".join(parts).strip()
    match = re.search(r"[A-F]", output.upper()) if mode == "action_token" else None
    prediction = match.group(0) if match else None
    target = str(row.get("target", "")) if mode == "action_token" else None
    timings = final_chunk.get("timings", {})
    return {
        "event_id": row.get("event_id"),
        "measured": measured,
        "ttft_ms": round(((first_content_at or finished) - started) * 1000.0, 4),
        "total_latency_ms": round((finished - started) * 1000.0, 4),
        "prompt_tokens": timings.get("prompt_n"),
        "predicted_tokens": timings.get("predicted_n"),
        "prediction": prediction,
        "target": target,
        "valid": prediction is not None if mode == "action_token" else None,
        "correct": prediction == target if mode == "action_token" else None,
        "raw_output": output,
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(resolve_path(args.test_jsonl))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No evaluation rows.")
    warmup_rows = (
        read_jsonl(resolve_path(args.warmup_jsonl))
        if args.warmup_jsonl
        else rows
    )
    if not warmup_rows:
        raise ValueError("No warmup rows.")

    command = [
        args.llama_server,
        "-m",
        str(resolve_path(args.model_path)),
        "--ctx-size",
        str(args.ctx_size),
        "--threads",
        str(args.threads),
        "--threads-batch",
        str(args.threads_batch),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--parallel",
        "1",
        "--no-cache-prompt" if args.no_cache_prompt else "--cache-prompt",
        "--cache-reuse",
        str(args.cache_reuse),
        "--cache-ram",
        "0",
        "--ctx-checkpoints",
        "0",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--no-warmup",
        "--poll",
        str(args.poll),
        "--poll-batch",
        "1" if args.poll > 0 else "0",
        "--prio",
        str(args.priority),
        "--prio-batch",
        str(args.priority),
    ]
    if args.mlock:
        command.append("--mlock")
    if args.threads_http > 0:
        command.extend(["--threads-http", str(args.threads_http)])
    if args.no_cont_batching:
        command.append("--no-cont-batching")
    if args.gpu_layers > 0:
        command[1:1] = ["--gpu-layers", str(args.gpu_layers)]
    base_url = "http://{}:{}".format(args.host, args.port)
    samples: List[Dict[str, Any]] = []
    warmups: List[Dict[str, Any]] = []
    error: Optional[str] = None
    startup_ms = 0.0
    log_tail = ""
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", errors="replace") as server_log:
        startup_system_sampler = SystemMemorySampler(args.sample_interval_ms / 1000.0)
        proc = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        startup_sampler = ProcessMemorySampler(proc.pid, args.sample_interval_ms / 1000.0)
        inference_sampler: Optional[ProcessMemorySampler] = None
        inference_system_sampler: Optional[SystemMemorySampler] = None
        startup_sampling = True
        startup_sampler.start()
        startup_system_sampler.start()
        try:
            startup_ms = wait_until_ready(base_url, proc, args.startup_timeout)
            warmups = [
                run_one(
                    base_url,
                    warmup_rows[index % len(warmup_rows)],
                    args.request_timeout,
                    False,
                    args.mode,
                    args.prefill_no_think,
                    args.prompt_format,
                    not args.non_stream,
                )
                for index in range(args.warmup)
            ]
            startup_sampler.stop()
            startup_system_sampler.stop()
            startup_sampling = False
            inference_sampler = ProcessMemorySampler(proc.pid, args.sample_interval_ms / 1000.0)
            inference_system_sampler = SystemMemorySampler(args.sample_interval_ms / 1000.0)
            inference_sampler.start()
            inference_system_sampler.start()
            samples = [
                run_one(
                    base_url,
                    row,
                    args.request_timeout,
                    True,
                    args.mode,
                    args.prefill_no_think,
                    args.prompt_format,
                    not args.non_stream,
                )
                for row in rows
            ]
        except Exception as exc:  # noqa: BLE001
            error = "{}: {}".format(type(exc).__name__, exc)
        finally:
            if startup_sampling:
                startup_sampler.stop()
                startup_system_sampler.stop()
            if inference_sampler is not None:
                inference_sampler.stop()
            if inference_system_sampler is not None:
                inference_system_sampler.stop()
            stop_server(proc)
            server_log.seek(0)
            log_tail = "\n".join(server_log.read().splitlines()[-40:])

    if not samples:
        raise RuntimeError(error or "No measured samples.")
    per_class = {}
    if args.mode == "action_token":
        for label in sorted({sample["target"] for sample in samples}):
            class_rows = [sample for sample in samples if sample["target"] == label]
            per_class[label] = {
                "total": len(class_rows),
                "correct": sum(1 for sample in class_rows if sample["correct"]),
            }
    measured_process_sampler = inference_sampler or startup_sampler
    measured_system_sampler = inference_system_sampler or startup_system_sampler
    steady_system_footprint_mb = round(
        max(0.0, measured_system_sampler.peak_mb - startup_system_sampler.baseline_mb),
        4,
    )
    result = {
        "task": "llama_cpp_action_token_streaming_benchmark",
        "runtime": "llama.cpp_gpu" if args.gpu_layers > 0 else "llama.cpp_cpu_mmap",
        "mode": args.mode,
        "model_path": args.model_path,
        "test_jsonl": args.test_jsonl,
        "warmup_jsonl": args.warmup_jsonl or args.test_jsonl,
        "prompt_format": args.prompt_format,
        "stream": not args.non_stream,
        "command": command,
        "startup_ms": startup_ms,
        "warmup_runs": args.warmup,
        "measured_runs": len(samples),
        "valid_rate": (
            round(sum(sample["valid"] for sample in samples) / len(samples), 6)
            if args.mode == "action_token"
            else None
        ),
        "accuracy": (
            round(sum(sample["correct"] for sample in samples) / len(samples), 6)
            if args.mode == "action_token"
            else None
        ),
        "ttft": summarize_latency([sample["ttft_ms"] for sample in samples]),
        "total_latency": summarize_latency([sample["total_latency_ms"] for sample in samples]),
        "peak_server_rss_mb": measured_process_sampler.peak_mb,
        "peak_server_pss_mb": measured_process_sampler.peak_pss_mb,
        "system_ram_baseline_mb": startup_system_sampler.baseline_mb,
        "system_ram_peak_mb": measured_system_sampler.peak_mb,
        "system_ram_peak_delta_mb": steady_system_footprint_mb,
        "inference_dynamic_ram_delta_mb": measured_system_sampler.peak_delta_mb,
        "startup_peak_server_rss_mb": startup_sampler.peak_mb,
        "startup_peak_server_pss_mb": startup_sampler.peak_pss_mb,
        "startup_system_ram_peak_delta_mb": startup_system_sampler.peak_delta_mb,
        "per_class": per_class,
        "warmups": warmups,
        "samples": samples,
        "server_log_tail": log_tail,
    }
    if error:
        result["error"] = error
    save_json(result, resolve_path(args.output_json))
    print(json.dumps({key: result[key] for key in (
        "runtime",
        "measured_runs",
        "valid_rate",
        "accuracy",
        "ttft",
        "total_latency",
        "peak_server_rss_mb",
        "peak_server_pss_mb",
        "per_class",
    )}, ensure_ascii=False, indent=2))
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
