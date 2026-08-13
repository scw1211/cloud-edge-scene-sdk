#!/usr/bin/env python3
"""Benchmark the deployed industrial single-token action adapter.

This program evaluates the frozen industrial held-out JSONL through the same
llama.cpp request contract used by the edge plugin.  It also measures true
streaming TTFT, end-to-end HTTP latency, concurrency, and process memory.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import urllib.request


GRAMMAR = 'root ::= "A" | "B" | "C"\n'
LABELS = ("A", "B", "C")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Iterable[float]) -> Dict[str, float]:
    items = [float(value) for value in values]
    if not items:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(items),
        "mean": round(statistics.fmean(items), 6),
        "p50": round(_percentile(items, 50.0), 6),
        "p95": round(_percentile(items, 95.0), 6),
        "max": round(max(items), 6),
    }


def _identity(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _load_dataset(path: Path) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 2:
                raise ValueError("invalid messages at line {}".format(line_number))
            prompt = str(messages[0].get("content", ""))
            target = str(messages[1].get("content", ""))
            if len(prompt) != 16 or not prompt.isdigit() or target not in LABELS:
                raise ValueError("invalid 16-to-1 contract at line {}".format(line_number))
            rows.append((str(row.get("event_id", line_number)), prompt, target))
    if not rows:
        raise ValueError("dataset is empty")
    return rows


def _post_stream(
    endpoint: str,
    prompt: str,
    lora_id: int,
    timeout_seconds: float,
) -> Dict[str, Any]:
    payload = {
        "prompt": prompt,
        "n_predict": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "cache_prompt": False,
        "grammar": GRAMMAR,
        "lora": [{"id": int(lora_id), "scale": 1.0}],
        "stream": True,
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/completion",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_ms: Optional[float] = None
    token = ""
    final: Dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            item = json.loads(line[6:])
            content = str(item.get("content", ""))
            if content and first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000.0
            token += content
            if bool(item.get("stop", False)):
                final = item
    wall_ms = (time.perf_counter() - started) * 1000.0
    timings = final.get("timings", {}) if isinstance(final, dict) else {}
    if not isinstance(timings, dict):
        timings = {}
    return {
        "token": token.strip(),
        "ttft_ms": round(first_token_ms if first_token_ms is not None else wall_ms, 6),
        "wall_ms": round(wall_ms, 6),
        "prompt_n": timings.get("prompt_n", final.get("tokens_evaluated")),
        "predicted_n": timings.get("predicted_n", final.get("tokens_predicted")),
        "server_prompt_ms": timings.get("prompt_ms"),
        "server_predicted_ms": timings.get("predicted_ms"),
    }


def _process_memory(pid: int) -> Dict[str, Optional[int]]:
    values: Dict[str, Optional[int]] = {
        "vm_rss_kib": None,
        "vm_hwm_kib": None,
        "vm_swap_kib": None,
        "gpu_memory_mib": None,
    }
    status = Path("/proc/{}/status".format(pid))
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            key, separator, raw_value = line.partition(":")
            if not separator:
                continue
            field = {
                "VmRSS": "vm_rss_kib",
                "VmHWM": "vm_hwm_kib",
                "VmSwap": "vm_swap_kib",
            }.get(key)
            if field is not None:
                values[field] = int(raw_value.strip().split()[0])
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        for line in output.splitlines():
            raw_pid, separator, raw_memory = line.partition(",")
            if separator and int(raw_pid.strip()) == pid:
                values["gpu_memory_mib"] = int(raw_memory.strip())
                break
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return values


def _classification(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    matrix = {
        reference: {prediction: 0 for prediction in LABELS}
        for reference in LABELS
    }
    invalid = 0
    for record in records:
        target = str(record["target"])
        prediction = str(record["token"])
        if prediction in LABELS:
            matrix[target][prediction] += 1
        else:
            invalid += 1
    per_class: Dict[str, Any] = {}
    f1_values = []
    for label in LABELS:
        true_positive = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[other][label] for other in LABELS)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    correct = sum(record["target"] == record["token"] for record in records)
    return {
        "count": len(records),
        "correct": correct,
        "accuracy": round(correct / len(records), 6),
        "macro_f1": round(statistics.fmean(f1_values), 6),
        "valid_count": len(records) - invalid,
        "valid_rate": round((len(records) - invalid) / len(records), 6),
        "matrix": matrix,
        "per_class": per_class,
    }


def _run_concurrency(
    endpoint: str,
    rows: Sequence[Tuple[str, str, str]],
    lora_id: int,
    concurrency: int,
    count: int,
    timeout_seconds: float,
) -> Dict[str, Any]:
    selected = [rows[index % len(rows)] for index in range(count)]
    started = time.perf_counter()
    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {
            executor.submit(
                _post_stream, endpoint, prompt, lora_id, timeout_seconds
            ): (event_id, target)
            for event_id, prompt, target in selected
        }
        for future in as_completed(future_map):
            event_id, target = future_map[future]
            record = future.result()
            records.append({"event_id": event_id, "target": target, **record})
    elapsed = time.perf_counter() - started
    return {
        "concurrency": concurrency,
        "requests": count,
        "elapsed_seconds": round(elapsed, 6),
        "throughput_requests_per_second": round(count / elapsed, 6),
        "accuracy": _classification(records),
        "ttft_ms": _summary(record["ttft_ms"] for record in records),
        "wall_ms": _summary(record["wall_ms"] for record in records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18590")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--server-binary", required=True)
    parser.add_argument("--base-gguf", required=True)
    parser.add_argument("--lora-gguf", required=True)
    parser.add_argument("--lora-id", type=int, default=1)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--concurrency-requests", type=int, default=120)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--environment-note",
        default=(
            "Measurements are from the local x86 llama.cpp CPU build "
            "(libggml-cpu-zen4); memory and latency are not GPU or Jetson "
            "Nano results."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    dataset = Path(args.dataset).resolve()
    rows = _load_dataset(dataset)
    initial_memory = _process_memory(args.server_pid)
    records: List[Dict[str, Any]] = []
    memory_checkpoints = [{"completed": 0, **initial_memory}]
    for index, (event_id, prompt, target) in enumerate(rows, start=1):
        result = _post_stream(
            args.endpoint, prompt, args.lora_id, args.timeout_seconds
        )
        records.append(
            {
                "event_id": event_id,
                "prompt": prompt,
                "target": target,
                **result,
            }
        )
        if index in {20, 100, 300, 500, len(rows)}:
            memory_checkpoints.append(
                {"completed": index, **_process_memory(args.server_pid)}
            )

    concurrency_results = []
    for raw_value in str(args.concurrency).split(","):
        concurrency_results.append(
            _run_concurrency(
                args.endpoint,
                rows,
                args.lora_id,
                int(raw_value),
                args.concurrency_requests,
                args.timeout_seconds,
            )
        )
        memory_checkpoints.append(
            {
                "completed": "concurrency_{}".format(raw_value),
                **_process_memory(args.server_pid),
            }
        )

    report = {
        "schema_version": "industrial-action-runtime-benchmark/v1",
        "created_at_epoch_ms": int(time.time() * 1000),
        "runtime": {
            "endpoint": args.endpoint,
            "pid": args.server_pid,
            "lora_id": args.lora_id,
            "server_binary": _identity(Path(args.server_binary)),
            "base_gguf": _identity(Path(args.base_gguf)),
            "lora_gguf": _identity(Path(args.lora_gguf)),
            "grammar": GRAMMAR,
            "grammar_sha256": hashlib.sha256(GRAMMAR.encode()).hexdigest(),
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
                "max_output_tokens": 1,
                "stream": True,
                "post_hoc_remapping": False,
            },
        },
        "dataset": _identity(dataset),
        "heldout": {
            "classification": _classification(records),
            "ttft_ms": _summary(record["ttft_ms"] for record in records),
            "wall_ms": _summary(record["wall_ms"] for record in records),
            "server_prompt_ms": _summary(
                record["server_prompt_ms"]
                for record in records
                if isinstance(record.get("server_prompt_ms"), (int, float))
            ),
            "prompt_token_contract_passed": all(
                record.get("prompt_n") == 16 for record in records
            ),
            "output_token_contract_passed": all(
                record.get("predicted_n") == 1 for record in records
            ),
        },
        "memory_checkpoints": memory_checkpoints,
        "concurrency": concurrency_results,
        "records": records,
        "environment_note": str(args.environment_note),
    }
    with output.open("x", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    print(json.dumps({
        "output": str(output),
        "heldout": report["heldout"],
        "memory_checkpoints": memory_checkpoints,
        "concurrency": concurrency_results,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
