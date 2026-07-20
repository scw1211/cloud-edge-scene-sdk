"""用途：通过独立边缘服务测量完整 HTTP 闭环，避免绕过服务层。"""

import argparse
import copy
import json
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List
from urllib.request import Request, urlopen


def _percentile(values: List[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile / 100.0
    low = int(position)
    high = min(len(ordered) - 1, low + 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6),
        "p50": round(_percentile(values, 50.0), 6),
        "p95": round(_percentile(values, 95.0), 6),
        "max": round(max(values), 6),
    }


def _request(base_url: str, path: str, payload: Dict[str, Any] = None):
    body = None
    method = "GET"
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = Request(base_url.rstrip("/") + path, data=body, headers=headers, method=method)
    with urlopen(request, timeout=5.0) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the role-separated services.")
    parser.add_argument("--event", required=True)
    parser.add_argument(
        "--edge-base-url",
        "--edge_base_url",
        dest="edge_base_url",
        default="http://127.0.0.1:18101",
    )
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--output",
        default="results/framework/framework_service_benchmark.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs <= 0 or args.warmup < 0 or args.warmup >= args.runs:
        raise ValueError("runs must be positive and warmup must be within [0, runs)")
    with Path(args.event).open("r", encoding="utf-8") as file_obj:
        template = json.load(file_obj)
    if not isinstance(template, dict) or not template.get("id"):
        raise ValueError("event must be a SceneEventEnvelope with id")
    base_id = str(template["id"])
    records = []
    for run_index in range(args.runs):
        event = copy.deepcopy(template)
        event["id"] = "{}-benchmark-{:04d}".format(base_id, run_index + 1)
        event["traceid"] = "benchmark-{:04d}".format(run_index + 1)
        started = time.perf_counter()
        result = _request(
            args.edge_base_url,
            "/api/v1/collaboration/decide",
            {"event": event},
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        transport = result["final_decision"].get("metadata", {}).get("transport", {})
        records.append(
            {
                "run": run_index + 1,
                "event_id": event["id"],
                "trace_id": result["trace_id"],
                "wall_ms": wall_ms,
                "edge_service_wall_ms": result.get("edge_service_wall_ms", 0.0),
                "framework_runtime_ms": result["framework_runtime_ms"],
                "accounted_closed_loop_ms": result["closed_loop_accounting"][
                    "accounted_closed_loop_ms"
                ],
                "http_round_trip_ms": float(transport.get("http_round_trip_ms", 0.0)),
                "request_bytes": int(transport.get("request_bytes", 0)),
                "response_bytes": int(transport.get("response_bytes", 0)),
                "route": result["final_decision"]["route"],
                "idempotency_replay": bool(result["idempotency_replay"]),
            }
        )
    steady = records[args.warmup :]
    output = {
        "task": "role_separated_edge_cloud_http_benchmark",
        "edge_base_url": args.edge_base_url,
        "event_template": str(Path(args.event).resolve()),
        "runs": args.runs,
        "warmup_runs_excluded": args.warmup,
        "all_unique_event_ids": len({item["event_id"] for item in records}) == len(records),
        "idempotency_replay_count": sum(item["idempotency_replay"] for item in records),
        "steady_state": {
            "wall_ms": _summary([item["wall_ms"] for item in steady]),
            "edge_service_wall_ms": _summary(
                [item["edge_service_wall_ms"] for item in steady]
            ),
            "framework_runtime_ms": _summary(
                [item["framework_runtime_ms"] for item in steady]
            ),
            "accounted_closed_loop_ms": _summary(
                [item["accounted_closed_loop_ms"] for item in steady]
            ),
            "http_round_trip_ms": _summary(
                [item["http_round_trip_ms"] for item in steady]
            ),
            "request_bytes": _summary(
                [float(item["request_bytes"]) for item in steady]
            ),
            "response_bytes": _summary(
                [float(item["response_bytes"]) for item in steady]
            ),
            "route_counts": {
                route: sum(item["route"] == route for item in steady)
                for route in sorted({item["route"] for item in steady})
            },
        },
        "edge_metrics": _request(
            args.edge_base_url, "/api/v1/framework/metrics"
        ),
        "records": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
