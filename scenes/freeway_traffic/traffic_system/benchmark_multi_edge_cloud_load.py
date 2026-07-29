"""用途：测量多个交通边缘区域并发访问云端协调服务时的吞吐与稳定性。"""

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from traffic_system.benchmark_utils import build_payload, post_json, safe_float, summarize
from traffic_system.decision_utils import DECISION_CLASSES, load_json, read_jsonl, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark concurrent edge-to-cloud traffic requests.")
    parser.add_argument(
        "--labels_jsonl",
        default="datasets/freeway_teacher_labels_qwen9b_safe_joint_metis4_expanded.jsonl",
    )
    parser.add_argument("--url", default="http://192.168.31.135:18080/api/v1/traffic/decision")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--requests_per_level", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--output_json",
        default="results/edge/multi_edge_cloud_load_1_2_4_8.json",
    )
    return parser.parse_args()


def load_cases(path: Path) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    cases = []
    for row in read_jsonl(path):
        event_path = Path(str(row.get("event_path", "")))
        teacher = row.get("teacher_decision")
        if event_path.is_file() and isinstance(teacher, dict):
            cases.append((load_json(event_path), teacher))
    if not cases:
        raise ValueError("No event and Teacher pairs found in {}".format(path))
    return cases


def run_request(
    url: str,
    timeout: float,
    request_index: int,
    event: Dict[str, Any],
    teacher: Dict[str, Any],
) -> Dict[str, Any]:
    body = build_payload(
        "multi-edge-load-{:06d}".format(request_index),
        event,
        0.0,
        0.0,
        compact_event=True,
    )
    started = time.perf_counter()
    try:
        response = post_json(url, body, timeout)
        wall_ms = (time.perf_counter() - started) * 1000.0
        decision = response.get("decision", {})
        cloud_metrics = response.get("cloud_metrics", {})
        protocol_valid = bool(
            decision.get("safe")
            and decision.get("decision") in DECISION_CLASSES
            and isinstance(decision.get("actions"), list)
        )
        return {
            "request_index": request_index,
            "event_id": event.get("event_id"),
            "edge_id": event.get("edge_id"),
            "success": True,
            "wall_time_ms": round(wall_ms, 6),
            "cloud_decision_latency_ms": safe_float(
                cloud_metrics.get("cloud_decision_latency_ms")
            ),
            "decision": decision.get("decision"),
            "reference_decision": teacher.get("decision"),
            "protocol_valid": protocol_valid,
            "legacy_teacher_match": decision.get("decision") == teacher.get("decision"),
            "safe": bool(decision.get("safe")),
            "payload_bytes": len(body),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "request_index": request_index,
            "event_id": event.get("event_id"),
            "edge_id": event.get("edge_id"),
            "success": False,
            "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 6),
            "cloud_decision_latency_ms": 0.0,
            "decision": None,
            "reference_decision": teacher.get("decision"),
            "protocol_valid": False,
            "legacy_teacher_match": False,
            "safe": False,
            "payload_bytes": len(body),
            "error": "{}: {}".format(type(exc).__name__, exc),
        }


def run_level(
    url: str,
    timeout: float,
    concurrency: int,
    request_count: int,
    cases: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
) -> Dict[str, Any]:
    selected = [cases[index % len(cases)] for index in range(request_count)]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_request, url, timeout, index, event, teacher)
            for index, (event, teacher) in enumerate(selected)
        ]
        records = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - started
    successes = [record for record in records if record["success"]]
    edge_counts: Dict[str, int] = {}
    for record in records:
        edge_id = str(record.get("edge_id"))
        edge_counts[edge_id] = edge_counts.get(edge_id, 0) + 1
    return {
        "concurrency": concurrency,
        "requests": request_count,
        "success_count": len(successes),
        "success_rate": round(len(successes) / request_count, 6),
        "safe_response_rate": round(
            sum(record["safe"] for record in records) / request_count, 6
        ),
        "protocol_valid_response_rate": round(
            sum(record["protocol_valid"] for record in records) / request_count, 6
        ),
        "legacy_teacher_match_rate_diagnostic": round(
            sum(record["legacy_teacher_match"] for record in records) / request_count, 6
        ),
        "batch_wall_seconds": round(wall_seconds, 6),
        "throughput_requests_per_second": round(len(successes) / wall_seconds, 6),
        "request_latency": summarize(record["wall_time_ms"] for record in successes),
        "cloud_decision_latency": summarize(
            record["cloud_decision_latency_ms"] for record in successes
        ),
        "average_payload_bytes": (
            round(statistics.fmean(record["payload_bytes"] for record in records), 3)
            if records
            else 0.0
        ),
        "edge_request_counts": edge_counts,
        "failures": [record for record in records if not record["success"]],
        "records": records,
    }


def main() -> None:
    args = parse_args()
    if args.requests_per_level <= 0 or args.warmup < 0:
        raise ValueError("requests_per_level must be positive and warmup non-negative")
    levels = [int(value.strip()) for value in args.concurrency.split(",") if value.strip()]
    if not levels or min(levels) <= 0:
        raise ValueError("concurrency levels must be positive")
    cases = load_cases(Path(args.labels_jsonl))
    for index in range(args.warmup):
        event, teacher = cases[index % len(cases)]
        warmup = run_request(args.url, args.timeout, -index - 1, event, teacher)
        if not warmup["success"]:
            raise RuntimeError("Warmup failed: {}".format(warmup["error"]))

    levels_result = []
    for level in levels:
        result = run_level(
            args.url,
            args.timeout,
            level,
            args.requests_per_level,
            cases,
        )
        levels_result.append(result)
        print(
            "concurrency={} success={:.2%} throughput={:.2f} req/s p95={:.2f} ms".format(
                level,
                result["success_rate"],
                result["throughput_requests_per_second"],
                result["request_latency"]["p95_ms"],
            ),
            flush=True,
        )
    baseline_throughput = levels_result[0]["throughput_requests_per_second"]
    output = {
        "task": "multi_edge_cloud_coordination_load_benchmark",
        "scope": (
            "Concurrent HTTP and cloud coordination only; ASTGCN and edge Qwen latency are measured in the "
            "unified closed-loop benchmark. Four METIS regions are emulated on one physical Jetson02. "
            "Legacy Qwen-Teacher agreement is diagnostic and is not reported as decision accuracy."
        ),
        "device": "Jetson02",
        "cloud_url": args.url,
        "labels_jsonl": args.labels_jsonl,
        "available_cases": len(cases),
        "requests_per_level": args.requests_per_level,
        "levels": levels_result,
        "throughput_scale_vs_concurrency_1": {
            str(result["concurrency"]): round(
                result["throughput_requests_per_second"] / baseline_throughput, 6
            )
            for result in levels_result
        },
        "all_levels_successful": all(result["success_rate"] == 1.0 for result in levels_result),
        "all_levels_safe": all(result["safe_response_rate"] == 1.0 for result in levels_result),
        "all_levels_protocol_valid": all(
            result["protocol_valid_response_rate"] == 1.0 for result in levels_result
        ),
    }
    save_json(output, Path(args.output_json))
    print(json.dumps({key: output[key] for key in (
        "throughput_scale_vs_concurrency_1", "all_levels_successful", "all_levels_safe",
        "all_levels_protocol_valid"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
