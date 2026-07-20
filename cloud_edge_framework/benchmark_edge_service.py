"""用途：通过正式边缘 HTTP 服务重复测量 Edge-Qwen 与云端复核的完整在线闭环。"""

import argparse
import json
import statistics
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List


DECIDE_PATH = "/api/v1/collaboration/decide"


def _percentile(values: List[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty measurement set")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: List[float]) -> Dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 6),
        "p50": round(_percentile(values, 50.0), 6),
        "p95": round(_percentile(values, 95.0), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _counts(values: List[str]) -> Dict[str, int]:
    return {value: values.count(value) for value in sorted(set(values))}


def _post_event(
    base_url: str,
    event: Dict[str, Any],
    request_id: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    body = json.dumps({"event": event}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + DECIDE_PATH,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Idempotency-Key": request_id,
            "X-Trace-Id": "trace_{}".format(request_id),
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("edge service response must be an object")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the deployed edge service with unique, non-replayed events."
    )
    parser.add_argument(
        "--event",
        default="examples/cloud_edge_framework/traffic_event.json",
    )
    parser.add_argument("--edge-base-url", default="http://127.0.0.1:18101")
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument(
        "--output",
        default="results/framework/edge_qwen_online_http_benchmark.json",
    )
    parser.add_argument("--require-edge-llm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs <= 0 or args.warmup < 0 or args.warmup >= args.runs:
        raise ValueError("runs must be positive and warmup must be within [0, runs)")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")
    event_path = Path(args.event).resolve()
    with event_path.open("r", encoding="utf-8") as file_obj:
        base_event = json.load(file_obj)
    if not isinstance(base_event, dict):
        raise ValueError("event file must contain an object")

    session_id = uuid.uuid4().hex[:12]
    base_event_id = str(base_event.get("id", "benchmark-event"))
    records: List[Dict[str, Any]] = []
    for run_index in range(args.runs):
        request_id = "edge-qwen-{}-{:04d}".format(session_id, run_index)
        event = dict(base_event)
        event["id"] = "{}-{}-{:04d}".format(base_event_id, session_id, run_index)
        event["traceid"] = "trace_{}".format(request_id)
        started = time.perf_counter()
        result = _post_event(
            args.edge_base_url,
            event,
            request_id,
            args.timeout_seconds,
        )
        client_wall_ms = (time.perf_counter() - started) * 1000.0
        local = result["local_decision"]
        local_metadata = local.get("metadata", {})
        final = result["final_decision"]
        transport = final.get("metadata", {}).get("transport", {})
        records.append(
            {
                "run": run_index + 1,
                "event_id": event["id"],
                "client_wall_ms": round(client_wall_ms, 6),
                "edge_service_wall_ms": float(result["edge_service_wall_ms"]),
                "accounted_closed_loop_ms": float(
                    result["closed_loop_accounting"]["accounted_closed_loop_ms"]
                ),
                "deadline_ms": float(result["schedule"]["deadline_ms"]),
                "schedule_route": str(result["schedule"]["route"]),
                "executed_route": str(final["route"]),
                "local_decision": str(local["decision"]),
                "final_decision": str(final["decision"]),
                "local_source": str(local_metadata.get("source", "unknown")),
                "edge_decision_path": str(
                    local_metadata.get("edge_decision_path", "unknown")
                ),
                "edge_llm_selected": bool(
                    local_metadata.get("edge_llm_selected", False)
                ),
                "edge_llm_token": local_metadata.get("edge_llm_token"),
                "edge_llm_latency_ms": float(
                    local_metadata.get("edge_llm_latency_ms", 0.0) or 0.0
                ),
                "edge_llm_runtime_error": local_metadata.get(
                    "edge_llm_runtime_error"
                ),
                "cloud_http_round_trip_ms": float(
                    transport.get("http_round_trip_ms", 0.0) or 0.0
                ),
                "request_bytes": int(transport.get("request_bytes", 0) or 0),
                "response_bytes": int(transport.get("response_bytes", 0) or 0),
            }
        )

    steady = records[args.warmup :]
    accepted = sum(
        record["local_source"] == "edge_qwen_single_token" for record in steady
    )
    deadline_met = sum(
        record["accounted_closed_loop_ms"] <= record["deadline_ms"]
        for record in steady
    )
    output: Dict[str, Any] = {
        "schema_version": 1,
        "task": "edge_service_edge_qwen_real_http_closed_loop",
        "session_id": session_id,
        "event": str(event_path),
        "edge_base_url": args.edge_base_url,
        "runs": args.runs,
        "warmup_runs_excluded": args.warmup,
        "steady_state": {
            "client_wall_ms": _summary(
                [record["client_wall_ms"] for record in steady]
            ),
            "edge_service_wall_ms": _summary(
                [record["edge_service_wall_ms"] for record in steady]
            ),
            "accounted_closed_loop_ms": _summary(
                [record["accounted_closed_loop_ms"] for record in steady]
            ),
            "edge_llm_latency_ms": _summary(
                [record["edge_llm_latency_ms"] for record in steady]
            ),
            "cloud_http_round_trip_ms": _summary(
                [record["cloud_http_round_trip_ms"] for record in steady]
            ),
            "edge_qwen_accepted": accepted,
            "edge_qwen_acceptance_rate": round(accepted / len(steady), 6),
            "deadline_met": deadline_met,
            "deadline_met_rate": round(deadline_met / len(steady), 6),
            "decision_paths": _counts(
                [record["edge_decision_path"] for record in steady]
            ),
            "schedule_routes": _counts(
                [record["schedule_route"] for record in steady]
            ),
            "executed_routes": _counts(
                [record["executed_route"] for record in steady]
            ),
            "local_decisions": _counts(
                [record["local_decision"] for record in steady]
            ),
            "final_decisions": _counts(
                [record["final_decision"] for record in steady]
            ),
        },
        "records": records,
        "measurement_note": (
            "client_wall_ms measures the full HTTP request to the edge service; "
            "accounted_closed_loop_ms additionally includes scene-reported preprocessing "
            "and ASTGCN inference exactly once"
        ),
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(output, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.require_edge_llm and accepted != len(steady):
        raise RuntimeError(
            "Edge-Qwen was accepted for only {}/{} steady runs".format(
                accepted, len(steady)
            )
        )


if __name__ == "__main__":
    main()
