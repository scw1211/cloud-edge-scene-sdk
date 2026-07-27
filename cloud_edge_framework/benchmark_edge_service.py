"""用途：通过正式边缘 HTTP 服务测量多样事件、Edge-Qwen 与云端复核闭环。"""

import argparse
import copy
import importlib
import json
import statistics
import time
import urllib.request
import urllib.error
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence


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


def _optional_summary(values: Sequence[float]) -> Optional[Dict[str, float]]:
    measurements = [float(value) for value in values]
    return _summary(measurements) if measurements else None


def _counts(values: List[str]) -> Dict[str, int]:
    return {value: values.count(value) for value in sorted(set(values))}


def _classification_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
) -> Dict[str, Any]:
    labels = sorted(set(references) | set(predictions))
    matrix = [
        [
            sum(
                reference == true_label and prediction == predicted_label
                for reference, prediction in zip(references, predictions)
            )
            for predicted_label in labels
        ]
        for true_label in labels
    ]
    f1_values = []
    per_class: Dict[str, Any] = {}
    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if support:
            f1_values.append(f1)
        per_class[label] = {
            "support": support,
            "precision": round(precision, 6) if predicted else None,
            "recall": round(recall, 6) if support else None,
            "f1": round(f1, 6) if support or predicted else None,
        }
    return {
        "accuracy": round(
            sum(
                reference == prediction
                for reference, prediction in zip(references, predictions)
            )
            / len(references),
            6,
        ),
        "macro_f1_present_classes": round(
            statistics.fmean(f1_values) if f1_values else 0.0,
            6,
        ),
        "class_names": labels,
        "matrix": matrix,
        "per_class": per_class,
    }


def _load_event_adapter(
    spec: Optional[str],
) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:
    if not spec:
        return None
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("event adapter must use module:function syntax")
    adapter = getattr(importlib.import_module(module_name), attribute)
    if not callable(adapter):
        raise TypeError("event adapter is not callable: {}".format(spec))
    return adapter


def _event_case(
    value: Any,
    source: str,
    event_adapter: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("event case {} must be an object".format(source))
    if isinstance(value.get("event"), dict):
        event = dict(value["event"])
        reference = value.get("reference", {})
    else:
        event = dict(value)
        reference = {}
    if not isinstance(reference, dict):
        raise ValueError("event case {} reference must be an object".format(source))
    input_format = "scene_event_envelope"
    if "specversion" not in event:
        if event_adapter is None:
            raise ValueError(
                "event case {} is not a public scene envelope; "
                "provide --event-adapter module:function".format(source)
            )
        event = event_adapter(event)
        input_format = "adapted_scene_event"
    if not isinstance(event, dict) or "specversion" not in event:
        raise ValueError("event adapter must return a public scene envelope")
    event_id = str(event.get("id", "")).strip()
    if not event_id:
        raise ValueError("event case {} has no event id".format(source))
    return {
        "event": event,
        "reference": dict(reference),
        "source": source,
        "input_format": input_format,
    }


def _load_event_cases(
    event_path: Optional[Path] = None,
    events_directory: Optional[Path] = None,
    events_jsonl: Optional[Path] = None,
    event_adapter: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    selected = [
        value
        for value in (event_path, events_directory, events_jsonl)
        if value is not None
    ]
    if len(selected) > 1:
        raise ValueError("use only one of --event, --events-dir and --events-jsonl")
    if not selected:
        event_path = Path("scene_plugin_template/sample_event.json")
    cases: List[Dict[str, Any]] = []
    if event_path is not None:
        resolved = event_path.resolve()
        with resolved.open("r", encoding="utf-8") as file_obj:
            cases.append(
                _event_case(json.load(file_obj), str(resolved), event_adapter)
            )
    elif events_directory is not None:
        resolved = events_directory.resolve()
        paths = sorted(path for path in resolved.glob("*.json") if path.is_file())
        if not paths:
            raise ValueError("events directory contains no JSON files: {}".format(resolved))
        for path in paths:
            with path.open("r", encoding="utf-8") as file_obj:
                cases.append(
                    _event_case(json.load(file_obj), str(path), event_adapter)
                )
    else:
        assert events_jsonl is not None
        resolved = events_jsonl.resolve()
        with resolved.open("r", encoding="utf-8") as file_obj:
            for line_number, raw_line in enumerate(file_obj, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                cases.append(
                    _event_case(
                        json.loads(line),
                        "{}:{}".format(resolved, line_number),
                        event_adapter,
                    )
                )
        if not cases:
            raise ValueError("events JSONL contains no records: {}".format(resolved))
    return cases


def _risk_level(event: Dict[str, Any]) -> str:
    data = event.get("data", {})
    if isinstance(data, dict) and data.get("risk_level") is not None:
        return str(data["risk_level"])
    return "unknown"


def _subject(event: Dict[str, Any]) -> str:
    return str(event.get("subject", "unknown"))


def _reference_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    labeled = [
        record
        for record in records
        if record.get("reference_decision") not in {None, ""}
    ]
    if not labeled:
        return {
            "labeled_count": 0,
            "local_accuracy": None,
            "final_accuracy": None,
            "local_macro_f1": None,
            "final_macro_f1": None,
            "critical_count": 0,
            "critical_local_recall": None,
            "critical_final_recall": None,
        }
    references = [str(record["reference_decision"]) for record in labeled]
    local_predictions = [str(record["local_decision"]) for record in labeled]
    final_predictions = [str(record["final_decision"]) for record in labeled]
    local_metrics = _classification_metrics(references, local_predictions)
    final_metrics = _classification_metrics(references, final_predictions)
    critical = [record for record in labeled if bool(record.get("reference_critical"))]

    def critical_prediction(record: Dict[str, Any], field: str) -> bool:
        allowed = record.get("reference_critical_decisions")
        if not isinstance(allowed, list) or not allowed:
            allowed = [record["reference_decision"]]
        return str(record[field]) in {str(value) for value in allowed}

    critical_local = sum(
        critical_prediction(record, "local_decision") for record in critical
    )
    critical_final = sum(
        critical_prediction(record, "final_decision") for record in critical
    )
    local_right_final_wrong = sum(
        record["local_decision"] == record["reference_decision"]
        and record["final_decision"] != record["reference_decision"]
        for record in labeled
    )
    local_wrong_final_right = sum(
        record["local_decision"] != record["reference_decision"]
        and record["final_decision"] == record["reference_decision"]
        for record in labeled
    )
    return {
        "labeled_count": len(labeled),
        "local_accuracy": local_metrics["accuracy"],
        "final_accuracy": final_metrics["accuracy"],
        "local_macro_f1": local_metrics["macro_f1_present_classes"],
        "final_macro_f1": final_metrics["macro_f1_present_classes"],
        "critical_count": len(critical),
        "critical_local_recall": (
            round(critical_local / len(critical), 6) if critical else None
        ),
        "critical_final_recall": (
            round(critical_final / len(critical), 6) if critical else None
        ),
        "cloud_beneficial_corrections": local_wrong_final_right,
        "cloud_harmful_corrections": local_right_final_wrong,
        "local_classification": local_metrics,
        "final_classification": final_metrics,
    }


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
        default=None,
        help="One public scene-envelope JSON file.",
    )
    parser.add_argument(
        "--events-dir",
        default=None,
        help="Directory of diverse public scene-envelope JSON files.",
    )
    parser.add_argument(
        "--events-jsonl",
        default=None,
        help="JSONL containing events or {'event': ..., 'reference': ...} cases.",
    )
    parser.add_argument(
        "--event-adapter",
        default=None,
        help="Optional module:function converting scene-native objects to envelopes.",
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
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--require-all-success", action="store_true")
    parser.add_argument(
        "--force-cloud-review",
        action="store_true",
        help="Force every event through cloud review as a comparison baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs <= 0 or args.warmup < 0 or args.warmup >= args.runs:
        raise ValueError("runs must be positive and warmup must be within [0, runs)")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")
    event_cases = _load_event_cases(
        event_path=Path(args.event) if args.event else None,
        events_directory=Path(args.events_dir) if args.events_dir else None,
        events_jsonl=Path(args.events_jsonl) if args.events_jsonl else None,
        event_adapter=_load_event_adapter(args.event_adapter),
    )

    session_id = uuid.uuid4().hex[:12]
    records: List[Dict[str, Any]] = []
    for run_index in range(args.runs):
        event_case = event_cases[run_index % len(event_cases)]
        base_event = event_case["event"]
        reference = event_case["reference"]
        base_event_id = str(base_event.get("id", "benchmark-event"))
        request_id = "edge-qwen-{}-{:04d}".format(session_id, run_index)
        event = copy.deepcopy(base_event)
        event["id"] = "{}-{}-{:04d}".format(base_event_id, session_id, run_index)
        event["traceid"] = "trace_{}".format(request_id)
        if args.force_cloud_review:
            data = event.get("data")
            if not isinstance(data, dict):
                raise ValueError("force-cloud-review requires object event.data")
            data["cloud_review_requested"] = True
        started = time.perf_counter()
        try:
            result = _post_event(
                args.edge_base_url,
                event,
                request_id,
                args.timeout_seconds,
            )
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            client_wall_ms = (time.perf_counter() - started) * 1000.0
            records.append(
                {
                    "run": run_index + 1,
                    "success": False,
                    "source_event_id": base_event_id,
                    "event_id": event["id"],
                    "source": event_case["source"],
                    "input_format": event_case["input_format"],
                    "subject": _subject(event),
                    "risk_level": _risk_level(event),
                    "client_wall_ms": round(client_wall_ms, 6),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if not args.continue_on_error:
                raise
            continue
        client_wall_ms = (time.perf_counter() - started) * 1000.0
        local = result["local_decision"]
        local_metadata = local.get("metadata", {})
        final = result["final_decision"]
        transport = final.get("metadata", {}).get("transport", {})
        records.append(
            {
                "run": run_index + 1,
                "success": True,
                "source_event_id": base_event_id,
                "event_id": event["id"],
                "source": event_case["source"],
                "input_format": event_case["input_format"],
                "subject": _subject(event),
                "risk_level": _risk_level(event),
                "reference_decision": (
                    str(reference["decision"])
                    if reference.get("decision") is not None
                    else None
                ),
                "reference_critical": bool(reference.get("critical", False)),
                "reference_critical_decisions": (
                    list(reference.get("critical_decisions", []))
                    if isinstance(reference.get("critical_decisions", []), list)
                    else []
                ),
                "reference_source": reference.get("source"),
                "reference_status": reference.get("status"),
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
    successful = [record for record in steady if record["success"]]
    if not successful:
        raise RuntimeError("no successful steady-state requests were recorded")
    accepted = sum(
        record["local_source"] == "edge_qwen_single_token" for record in successful
    )
    deadline_met = sum(
        record["accounted_closed_loop_ms"] <= record["deadline_ms"]
        for record in successful
    )
    corrections = sum(
        record["local_decision"] != record["final_decision"] for record in successful
    )
    edge_llm_measurements = [
        record["edge_llm_latency_ms"]
        for record in successful
        if record["edge_llm_latency_ms"] > 0
    ]
    request_bytes = [record["request_bytes"] for record in successful]
    response_bytes = [record["response_bytes"] for record in successful]
    output: Dict[str, Any] = {
        "schema_version": 1,
        "task": "edge_service_diverse_real_http_closed_loop",
        "session_id": session_id,
        "event_sources": {
            "single_event": args.event,
            "events_directory": args.events_dir,
            "events_jsonl": args.events_jsonl,
            "loaded_case_count": len(event_cases),
            "input_formats": _counts(
                [str(case["input_format"]) for case in event_cases]
            ),
        },
        "edge_base_url": args.edge_base_url,
        "force_cloud_review": bool(args.force_cloud_review),
        "runs": args.runs,
        "warmup_runs_excluded": args.warmup,
        "steady_state": {
            "attempted": len(steady),
            "successful": len(successful),
            "failed": len(steady) - len(successful),
            "success_rate": round(len(successful) / len(steady), 6),
            "unique_source_events": len(
                {record["source_event_id"] for record in successful}
            ),
            "subjects": _counts([record["subject"] for record in successful]),
            "risk_levels": _counts([record["risk_level"] for record in successful]),
            "client_wall_ms": _summary(
                [record["client_wall_ms"] for record in successful]
            ),
            "edge_service_wall_ms": _summary(
                [record["edge_service_wall_ms"] for record in successful]
            ),
            "accounted_closed_loop_ms": _summary(
                [record["accounted_closed_loop_ms"] for record in successful]
            ),
            "edge_llm_latency_ms": _optional_summary(edge_llm_measurements),
            "cloud_http_round_trip_ms": _summary(
                [record["cloud_http_round_trip_ms"] for record in successful]
            ),
            "edge_qwen_accepted": accepted,
            "edge_qwen_acceptance_rate": round(accepted / len(successful), 6),
            "deadline_met": deadline_met,
            "deadline_met_rate": round(deadline_met / len(successful), 6),
            "cloud_corrections": corrections,
            "cloud_correction_rate": round(corrections / len(successful), 6),
            "transport": {
                "request_bytes": _summary(request_bytes),
                "response_bytes": _summary(response_bytes),
                "total_request_bytes": sum(request_bytes),
                "total_response_bytes": sum(response_bytes),
            },
            "reference_metrics": _reference_metrics(successful),
            "decision_paths": _counts(
                [record["edge_decision_path"] for record in successful]
            ),
            "schedule_routes": _counts(
                [record["schedule_route"] for record in successful]
            ),
            "executed_routes": _counts(
                [record["executed_route"] for record in successful]
            ),
            "local_decisions": _counts(
                [record["local_decision"] for record in successful]
            ),
            "final_decisions": _counts(
                [record["final_decision"] for record in successful]
            ),
        },
        "records": records,
        "measurement_note": (
            "client_wall_ms measures the full HTTP request to the edge service; "
            "accounted_closed_loop_ms additionally includes scene-reported preprocessing "
            "and scene-model inference exactly once; scene-native inputs require an "
            "explicit adapter that converts them to the public scene envelope"
        ),
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(output, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.require_all_success and len(successful) != len(steady):
        raise RuntimeError(
            "Only {}/{} steady requests succeeded".format(
                len(successful), len(steady)
            )
        )
    if args.require_edge_llm and accepted != len(successful):
        raise RuntimeError(
            "Edge-Qwen was accepted for only {}/{} steady runs".format(
                accepted, len(successful)
            )
        )


if __name__ == "__main__":
    main()
