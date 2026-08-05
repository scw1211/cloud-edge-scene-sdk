#!/usr/bin/env python3
"""Benchmark PEMS08 current-state input through deployed edge/cloud finality.

The steady-state clock starts immediately before ``infer_sample``.  One sample
contains four METIS partitions, which are submitted concurrently.  Timed HTTP
requests use the production compact response; full local/Qwen diagnostics are
collected afterwards from the review lifecycle and do not inflate the local
response latency.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import uuid

DECIDE_PATH = "/api/v1/collaboration/decide"
REVIEW_PATH = "/api/v1/collaboration/reviews/{}"
AGGREGATION_PATH = "/api/v1/collaboration/aggregations/{}"
METRICS_PATH = "/api/v1/framework/metrics"
NON_AUTHORITATIVE_REVIEW_STAGES = {"partial_final", "local_only_timeout"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a resident PEMS08 window through current-state perception, "
            "the deployed compact /decide path, and authoritative cloud finality."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--data-npz",
        default=(
            "scenes/freeway_traffic/assets/downloads/"
            "PEMS08_r1_d0_w0_astcgn_multitask.npz"
        ),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sample-start", type=int, default=100)
    parser.add_argument("--sample-stop", type=int, default=200)
    parser.add_argument("--warmup-samples", default="100,118,125")
    parser.add_argument("--edge-url", default="http://127.0.0.1:19101")
    parser.add_argument("--cloud-url", default="http://127.0.0.1:19100")
    parser.add_argument("--request-timeout-seconds", type=float, default=3.0)
    parser.add_argument("--final-wait-seconds", type=float, default=20.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.025)
    parser.add_argument("--aggregation-timeout-ms", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--require-qwen-selected", action="store_true")
    parser.add_argument("--require-qwen-accepted", action="store_true")
    parser.add_argument("--min-qwen-selected", type=int, default=0)
    parser.add_argument("--min-qwen-accepted", type=int, default=0)
    parser.add_argument("--require-risk-coverage", action="store_true")
    parser.add_argument("--require-complete-final", action="store_true")
    parser.add_argument(
        "--output",
        default="results/framework/pems08_current_state_e2e_latest.json",
    )
    return parser.parse_args()


def _project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _parse_ids(value: str) -> List[int]:
    result: List[int] = []
    for item in str(value).split(","):
        stripped = item.strip()
        if not stripped:
            continue
        sample_id = int(stripped)
        if sample_id not in result:
            result.append(sample_id)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _get_json(base_url: str, path: str, timeout: float) -> Dict[str, Any]:
    request = Request(
        base_url.rstrip("/") + path,
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("GET {} did not return an object".format(path))
    return value


def _post_compact(
    edge_url: str,
    envelope: Mapping[str, Any],
    body: bytes,
    timeout: float,
    sample_t0: float,
) -> Dict[str, Any]:
    event_id = str(envelope["id"])
    request = Request(
        edge_url.rstrip("/") + DECIDE_PATH,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Idempotency-Key": event_id,
            "X-Trace-Id": "trace_" + event_id,
            "Prefer": "return=minimal",
        },
        method="POST",
    )
    dispatch_ms = (time.perf_counter() - sample_t0) * 1000.0
    request_started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "POST {} failed with HTTP {}: {}".format(event_id, exc.code, detail)
        ) from exc
    value = json.loads(response_body.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("edge response must be an object")
    return {
        "event_id": event_id,
        "dispatch_ms": round(dispatch_ms, 6),
        "http_wall_ms": round(
            (time.perf_counter() - request_started) * 1000.0, 6
        ),
        "response_at_ms": round((time.perf_counter() - sample_t0) * 1000.0, 6),
        "request_bytes": len(body),
        "response_bytes": len(response_body),
        "response": value,
    }


def _wait_reviews(
    edge_url: str,
    event_ids: Sequence[str],
    timeout_seconds: float,
    poll_interval_seconds: float,
    request_timeout_seconds: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    deadline = time.monotonic() + timeout_seconds
    pending = set(event_ids)
    completed: Dict[str, Dict[str, Any]] = {}
    observed_at: Dict[str, float] = {}
    while pending and time.monotonic() < deadline:
        for event_id in list(pending):
            try:
                review = _get_json(
                    edge_url,
                    REVIEW_PATH.format(quote(event_id, safe="")),
                    request_timeout_seconds,
                )
            except (HTTPError, URLError, TimeoutError, ValueError):
                continue
            if str(review.get("state", "")) == "completed":
                completed[event_id] = review
                observed_at[event_id] = time.perf_counter()
                pending.remove(event_id)
        if pending:
            time.sleep(poll_interval_seconds)
    return completed, observed_at


def _authoritative_review(review: Mapping[str, Any]) -> bool:
    return bool(
        str(review.get("state", "")) == "completed"
        and str(review.get("completion_stage", ""))
        not in NON_AUTHORITATIVE_REVIEW_STAGES
        and isinstance(review.get("final_decision"), dict)
    )


def _edge_llm_health(health: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = health.get("runtime", {})
    runtime = dict(runtime) if isinstance(runtime, dict) else {}
    plugins = runtime.get("plugins", [])
    if not isinstance(plugins, list):
        return {}
    for plugin in plugins:
        if not isinstance(plugin, dict) or str(plugin.get("scene", "")) != "traffic":
            continue
        plugin_health = plugin.get("health", {})
        plugin_health = (
            dict(plugin_health) if isinstance(plugin_health, dict) else {}
        )
        value = plugin_health.get("edge_llm", {})
        return dict(value) if isinstance(value, dict) else {}
    return {}


def _metric_total(snapshot: Mapping[str, Any], name: str) -> float:
    distributions = snapshot.get("distributions", snapshot.get("samples", {}))
    distributions = (
        dict(distributions) if isinstance(distributions, dict) else {}
    )
    value = distributions.get(name, {})
    value = dict(value) if isinstance(value, dict) else {}
    return float(value.get("count", 0) or 0) * float(value.get("mean", 0.0) or 0.0)


def _metric_count(snapshot: Mapping[str, Any], name: str) -> int:
    distributions = snapshot.get("distributions", snapshot.get("samples", {}))
    distributions = (
        dict(distributions) if isinstance(distributions, dict) else {}
    )
    value = distributions.get(name, {})
    value = dict(value) if isinstance(value, dict) else {}
    return int(value.get("count", 0) or 0)


def _counter(snapshot: Mapping[str, Any], name: str) -> int:
    counters = snapshot.get("counters", {})
    counters = dict(counters) if isinstance(counters, dict) else {}
    return int(counters.get(name, 0) or 0)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summary(values: Iterable[float]) -> Dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(data),
        "mean": round(statistics.fmean(data), 6),
        "p50": round(_percentile(data, 50), 6),
        "p95": round(_percentile(data, 95), 6),
        "p99": round(_percentile(data, 99), 6),
        "max": round(max(data), 6),
    }


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _action_authorization(decision: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = decision.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    value = metadata.get("action_authorization", {})
    return dict(value) if isinstance(value, dict) else {}


def _aggregation_from_review(review: Mapping[str, Any]) -> Dict[str, Any]:
    decision = review.get("final_decision", {})
    decision = dict(decision) if isinstance(decision, dict) else {}
    metadata = decision.get("metadata", {})
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    value = metadata.get("aggregation", {})
    return dict(value) if isinstance(value, dict) else {}


def _record_event(
    native: Mapping[str, Any],
    post: Mapping[str, Any],
    review: Optional[Mapping[str, Any]],
    sample_t0_epoch_ms: float,
    review_observed_at_ms: Optional[float],
) -> Dict[str, Any]:
    response = dict(post["response"])
    compact_final = response.get("final_decision", {})
    compact_final = dict(compact_final) if isinstance(compact_final, dict) else {}
    review = dict(review) if isinstance(review, dict) else {}
    local = review.get("local_decision", {})
    local = dict(local) if isinstance(local, dict) else {}
    final = review.get("final_decision", compact_final)
    final = dict(final) if isinstance(final, dict) else compact_final
    authoritative = _authoritative_review(review)
    local_metadata = local.get("metadata", {})
    local_metadata = dict(local_metadata) if isinstance(local_metadata, dict) else {}
    model_uncertainty = local_metadata.get("model_uncertainty", {})
    model_uncertainty = (
        dict(model_uncertainty) if isinstance(model_uncertainty, dict) else {}
    )
    local_auth = _action_authorization(local)
    deferred = local_auth.get("deferred_action_types", [])
    deferred = list(deferred) if isinstance(deferred, list) else []
    completed_at_ms = review.get("completed_at_ms") if authoritative else None
    final_exact_ms: Optional[float]
    if not authoritative:
        final_exact_ms = None
    else:
        try:
            final_exact_ms = max(0.0, float(completed_at_ms) - sample_t0_epoch_ms)
        except (TypeError, ValueError):
            final_exact_ms = review_observed_at_ms
    response_at_ms = float(post["response_at_ms"])
    final_status = str(compact_final.get("status", ""))
    final_auth = _action_authorization(compact_final)
    cloud_confirmed_in_response = bool(final_auth.get("cloud_confirmed", False))
    if deferred and not (final_status == "final" and cloud_confirmed_in_response):
        business_completion_ms = (
            max(response_at_ms, float(final_exact_ms))
            if final_exact_ms is not None
            else None
        )
    else:
        business_completion_ms = response_at_ms

    edge_llm_selected = bool(local_metadata.get("edge_llm_selected", False))
    edge_llm_accepted = (
        str(local_metadata.get("source", "")) == "edge_qwen_single_token"
        or str(local_metadata.get("edge_decision_path", "")) == "edge_qwen"
    )
    edge_llm_requires_cloud = bool(
        local_metadata.get("edge_llm_requires_cloud", False)
    )
    edge_llm_safety_fallback = bool(
        local_metadata.get("edge_llm_safety_fallback", False)
    )
    if edge_llm_accepted and edge_llm_requires_cloud:
        decision_stratum = "qwen_accepted_requires_cloud"
    elif edge_llm_accepted:
        decision_stratum = "qwen_accepted_local"
    elif edge_llm_selected:
        decision_stratum = "qwen_selected_fallback"
    else:
        decision_stratum = "student"
    schedule = response.get("schedule", {})
    schedule = dict(schedule) if isinstance(schedule, dict) else {}
    summary_delivery = response.get("summary_delivery", {})
    summary_delivery = (
        dict(summary_delivery) if isinstance(summary_delivery, dict) else {}
    )
    data_plane = response.get("data_plane", {})
    data_plane = dict(data_plane) if isinstance(data_plane, dict) else {}
    accounting = response.get("closed_loop_accounting", {})
    accounting = dict(accounting) if isinstance(accounting, dict) else {}
    pipeline = accounting.get("pipeline_stage_ms", {})
    pipeline = dict(pipeline) if isinstance(pipeline, dict) else {}
    region = native.get("region_summary", {})
    region = dict(region) if isinstance(region, dict) else {}
    safety = local_metadata.get("operational_safety_risk", {})
    safety = dict(safety) if isinstance(safety, dict) else {}
    priority = {"low": 0, "medium": 1, "high": 2, "severe": 3}
    regional_congestion = str(region.get("region_risk_level", "low"))
    max_node_congestion = str(
        region.get("max_node_risk_level", regional_congestion)
    )
    legacy_congestion = str(
        max(
            (regional_congestion, max_node_congestion),
            key=priority.get,
        )
    )
    operational_safety = str(safety.get("level", "low"))
    return {
        "event_id": str(post["event_id"]),
        "sample_id": int(native["sample_id"]),
        "partition_id": int(native["partition_id"]),
        "regional_risk_level": regional_congestion,
        "regional_congestion_level": regional_congestion,
        "max_node_congestion_level": max_node_congestion,
        "legacy_congestion_level": legacy_congestion,
        "operational_safety_level": operational_safety,
        "operational_risk_level": operational_safety,
        "operational_safety_source": str(safety.get("source", "unknown")),
        "upload_required": bool(native.get("upload_required", False)),
        "upload_level": str(native.get("upload_level", "summary")),
        "dispatch_ms": float(post["dispatch_ms"]),
        "http_wall_ms": float(post["http_wall_ms"]),
        "response_at_ms": response_at_ms,
        "business_completion_ms": (
            round(float(business_completion_ms), 6)
            if business_completion_ms is not None
            else None
        ),
        "global_final_ms": (
            round(float(final_exact_ms), 6) if final_exact_ms is not None else None
        ),
        "review_observed_at_ms": review_observed_at_ms,
        "review_state": str(review.get("state", "missing")),
        "review_authoritative": authoritative,
        "review_completion_mode": str(review.get("completion_mode", "")),
        "review_completion_stage": str(review.get("completion_stage", "")),
        "schedule_route": str(schedule.get("route", "unknown")),
        "schedule_reason": str(schedule.get("reason", "")),
        "schedule_waits_for_cloud": bool(schedule.get("waits_for_cloud", False)),
        "schedule_critical": bool(schedule.get("critical", False)),
        "schedule_uncertain": bool(schedule.get("uncertain", False)),
        "local_requires_review": bool(
            model_uncertainty.get("requires_review", False)
        ),
        "local_requires_synchronous_review": bool(
            model_uncertainty.get("requires_synchronous_review", False)
        ),
        "synchronous_review_reasons": list(
            model_uncertainty.get("synchronous_review_reasons", [])
        )
        if isinstance(
            model_uncertainty.get("synchronous_review_reasons", []), list
        )
        else [],
        "synchronous_review_resolution": str(
            model_uncertainty.get("synchronous_review_resolution", "")
        ),
        "model_uncertainty_score": float(
            model_uncertainty.get("score", 0.0) or 0.0
        ),
        "perception_confidence": float(
            model_uncertainty.get("perception_confidence", 0.0) or 0.0
        ),
        "student_confidence": model_uncertainty.get("student_confidence"),
        "student_low_confidence": bool(
            model_uncertainty.get("student_low_confidence", False)
        ),
        "student_rule_disagreement": bool(
            model_uncertainty.get("student_rule_disagreement", False)
        ),
        "prediction_set_size": int(
            model_uncertainty.get("prediction_set_size", 1) or 1
        ),
        "executed_route": str(compact_final.get("route", "unknown")),
        "response_status": final_status,
        "local_decision": str(local.get("decision", "unknown")),
        "final_decision": str(final.get("decision", compact_final.get("decision", "unknown"))),
        "local_source": str(local_metadata.get("source", "unknown")),
        "edge_decision_path": str(
            local_metadata.get(
                "edge_decision_path",
                compact_final.get("metadata", {}).get("edge_decision_path", "unknown")
                if isinstance(compact_final.get("metadata"), dict)
                else "unknown",
            )
        ),
        "decision_stratum": decision_stratum,
        "edge_llm_selected": edge_llm_selected,
        "edge_llm_accepted": edge_llm_accepted,
        "edge_llm_requires_cloud": edge_llm_requires_cloud,
        "edge_llm_model_disagreement": bool(
            local_metadata.get("edge_llm_model_disagreement", False)
        ),
        "edge_llm_safety_fallback": edge_llm_safety_fallback,
        "edge_llm_fallback_reason": local_metadata.get("edge_llm_fallback_reason"),
        "edge_llm_selection_reason": str(
            local_metadata.get(
                "edge_llm_selection_reason",
                compact_final.get("metadata", {}).get(
                    "edge_llm_selection_reason", "unknown"
                )
                if isinstance(compact_final.get("metadata"), dict)
                else "unknown",
            )
        ),
        "edge_llm_latency_ms": float(
            local_metadata.get("edge_llm_latency_ms", 0.0) or 0.0
        ),
        "edge_llm_runtime_error": local_metadata.get("edge_llm_runtime_error"),
        "deferred_action_types": [str(value) for value in deferred],
        "cloud_confirmed_in_response": cloud_confirmed_in_response,
        "summary_delivery_mode": str(summary_delivery.get("mode", "unknown")),
        "summary_persistence_stage": str(
            summary_delivery.get("persistence_stage", "unknown")
        ),
        "ordinary_summary_fast_path": bool(summary_delivery.get("fast_path", False)),
        "decision_delivery_path": (
            "local_decision_async_summary"
            if bool(summary_delivery.get("fast_path", False))
            else (
                "synchronous_cloud_review"
                if bool(schedule.get("waits_for_cloud", False))
                else "asynchronous_cloud_delivery"
            )
        ),
        "ingress_request_bytes": int(post["request_bytes"]),
        "ingress_response_bytes": int(post["response_bytes"]),
        "selected_cloud_request_bytes": int(
            data_plane.get("selected_request_bytes", 0) or 0
        ),
        "actual_cloud_json_request_bytes": int(
            data_plane.get("actual_json_request_bytes", 0) or 0
        ),
        "actual_cloud_artifact_request_bytes": int(
            data_plane.get("actual_artifact_request_bytes", 0) or 0
        ),
        "actual_cloud_transport_request_bytes": int(
            data_plane.get("actual_transport_request_bytes", 0) or 0
        ),
        "request_reduction_ratio": float(
            data_plane.get("request_reduction_ratio", 0.0) or 0.0
        ),
        "edge_service_wall_ms": float(response.get("edge_service_wall_ms", 0.0) or 0.0),
        "framework_runtime_ms": float(response.get("framework_runtime_ms", 0.0) or 0.0),
        "pipeline_stage_ms": {
            name: float(pipeline.get(name, 0.0) or 0.0)
            for name in (
                "normalization",
                "edge_decision",
                "data_plane_preparation",
                "scheduling",
                "route_execution",
            )
        },
        "aggregation": _aggregation_from_review(review),
    }


def _stratum_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    local_values = [
        float(row["response_at_ms"])
        for row in rows
        if row.get("response_at_ms") is not None
    ]
    completed_business = [
        float(row["business_completion_ms"])
        for row in rows
        if row.get("business_completion_ms") is not None
    ]
    completed_final = [
        float(row["global_final_ms"])
        for row in rows
        if row.get("global_final_ms") is not None
    ]
    pipeline_names = (
        "normalization",
        "edge_decision",
        "data_plane_preparation",
        "scheduling",
        "route_execution",
    )
    return {
        "events": len(rows),
        "local_actionable_ms": _summary(local_values),
        "business_completion_ms": _summary(completed_business),
        "global_final_ms": _summary(completed_final),
        "under_200ms_rate": round(
            sum(value <= 200.0 for value in completed_business) / max(1, len(rows)),
            6,
        ),
        "success_rate": round(len(completed_business) / max(1, len(rows)), 6),
        "qwen_selected": sum(bool(row.get("edge_llm_selected")) for row in rows),
        "qwen_accepted": sum(bool(row.get("edge_llm_accepted")) for row in rows),
        "schedule_routes": _counts(row.get("schedule_route") for row in rows),
        "executed_routes": _counts(row.get("executed_route") for row in rows),
        "decision_delivery_paths": _counts(
            row.get("decision_delivery_path") for row in rows
        ),
        "operational_risk_levels": _counts(
            row.get("operational_risk_level") for row in rows
        ),
        "event_http_wall_ms": _summary(
            float(row.get("http_wall_ms", 0.0)) for row in rows
        ),
        "framework_runtime_ms": _summary(
            float(row.get("framework_runtime_ms", 0.0)) for row in rows
        ),
        "pipeline_stage_ms": {
            name: _summary(
                float(row.get("pipeline_stage_ms", {}).get(name, 0.0))
                for row in rows
            )
            for name in pipeline_names
        },
    }


def _sample_level(
    rows: Sequence[Mapping[str, Any]], field: str, default: str = "low"
) -> str:
    priority = {"low": 0, "medium": 1, "high": 2, "severe": 3}
    return max(
        (str(row.get(field, default)) for row in rows),
        key=priority.get,
        default=default,
    )


def _sample_stratum_summary(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "samples": len(samples),
        "local_actionable_ms": _summary(
            float(sample["local_actionable_ms"])
            for sample in samples
            if sample.get("local_actionable_ms") is not None
        ),
        "business_completion_ms": _summary(
            float(sample["business_completion_ms"])
            for sample in samples
            if sample.get("business_completion_ms") is not None
        ),
        "global_authoritative_final_ms": _summary(
            float(sample["global_authoritative_final_ms"])
            for sample in samples
            if sample.get("global_authoritative_final_ms") is not None
        ),
    }


def _submit_sample(
    runtime: Any,
    traffic_event_from_output: Any,
    sample_id: int,
    experiment_id: str,
    edge_url: str,
    request_timeout_seconds: float,
    aggregation_timeout_ms: int,
) -> Dict[str, Any]:
    sample_t0_epoch_ms = time.time() * 1000.0
    sample_t0 = time.perf_counter()
    perception = runtime.infer_sample(sample_id)
    perception_done_ms = (time.perf_counter() - sample_t0) * 1000.0
    prepared: List[Tuple[Dict[str, Any], Dict[str, Any], bytes]] = []
    for native in perception.events:
        measured = copy.deepcopy(native)
        measured["sample_split"] = "{}_{}".format(runtime.split, experiment_id)
        measured["event_id"] = "{}_{}".format(measured["event_id"], experiment_id)
        measured["aggregation_timeout_ms"] = aggregation_timeout_ms
        envelope = traffic_event_from_output(measured)
        prepared.append((measured, envelope, _json_bytes({"event": envelope})))
    encode_done_ms = (time.perf_counter() - sample_t0) * 1000.0

    posts: Dict[str, Dict[str, Any]] = {}
    errors: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(prepared)) as executor:
        futures = {
            executor.submit(
                _post_compact,
                edge_url,
                envelope,
                body,
                request_timeout_seconds,
                sample_t0,
            ): (native, envelope)
            for native, envelope, body in prepared
        }
        for future in as_completed(futures):
            native, envelope = futures[future]
            try:
                posts[str(envelope["id"])] = future.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "event_id": str(envelope["id"]),
                        "partition_id": int(native["partition_id"]),
                        "error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
    partition_count = len(prepared)
    local_group_ms = (
        max(float(value["response_at_ms"]) for value in posts.values())
        if len(posts) == partition_count
        else None
    )
    return {
        "sample_id": sample_id,
        "sample_t0_epoch_ms": sample_t0_epoch_ms,
        "sample_t0_perf": sample_t0,
        "perception": perception,
        "perception_observed_ms": perception_done_ms,
        "event_encode_done_ms": encode_done_ms,
        "prepared": prepared,
        "posts": posts,
        "errors": errors,
        "partition_count": partition_count,
        "local_actionable_ms": local_group_ms,
    }


def _finalize_sample(
    submission: Mapping[str, Any],
    reviews: Mapping[str, Mapping[str, Any]],
    review_observed_perf: Mapping[str, float],
    cloud_url: str,
    request_timeout_seconds: float,
) -> Dict[str, Any]:
    prepared = list(submission["prepared"])
    posts = dict(submission["posts"])
    perception = submission["perception"]
    sample_t0_perf = float(submission["sample_t0_perf"])
    sample_reviews: Dict[str, Mapping[str, Any]] = {}
    rows = []
    for native, envelope, _ in prepared:
        event_id = str(envelope["id"])
        if event_id not in posts:
            continue
        review = reviews.get(event_id)
        if isinstance(review, Mapping):
            sample_reviews[event_id] = review
        observed_perf = review_observed_perf.get(event_id)
        rows.append(
            _record_event(
                native,
                posts[event_id],
                review,
                float(submission["sample_t0_epoch_ms"]),
                round((float(observed_perf) - sample_t0_perf) * 1000.0, 6)
                if observed_perf is not None
                else None,
            )
        )
    rows.sort(key=lambda value: int(value["partition_id"]))
    group_ids = sorted(
        {
            str(row["aggregation"].get("group_id"))
            for row in rows
            if row.get("aggregation", {}).get("group_id")
        }
    )
    aggregations = []
    for group_id in group_ids:
        try:
            aggregations.append(
                _get_json(
                    cloud_url,
                    AGGREGATION_PATH.format(quote(group_id, safe="")),
                    request_timeout_seconds,
                )
            )
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            aggregations.append(
                {
                    "group_id": group_id,
                    "state": "query_failed",
                    "error": "{}: {}".format(type(exc).__name__, exc),
                }
            )
    business_values = [
        float(row["business_completion_ms"])
        for row in rows
        if row.get("business_completion_ms") is not None
    ]
    final_values = [
        float(row["global_final_ms"])
        for row in rows
        if row.get("global_final_ms") is not None
    ]
    partition_count = int(submission["partition_count"])
    expected_members = {
        str(native.get("edge_id", "edge_node_{}".format(native["partition_id"])))
        for native, _, _ in prepared
    }
    authoritative_count = sum(
        _authoritative_review(review) for review in sample_reviews.values()
    )
    aggregations_complete = bool(
        len(group_ids) == 1
        and len(aggregations) == 1
        and all(
            str(row.get("aggregation", {}).get("group_id", "")) == group_ids[0]
            for row in rows
        )
        and aggregations[0].get("state") == "completed"
        and aggregations[0].get("completion_reason") == "all_expected_members"
        and aggregations[0].get("evidence_complete") is True
        and aggregations[0].get("finality") == "final"
        and aggregations[0].get("global_confirmation") is True
        and set(aggregations[0].get("expected_members", [])) == expected_members
        and set(aggregations[0].get("received_members", [])) == expected_members
        and len(expected_members) == partition_count
    )
    return {
        "sample_id": int(submission["sample_id"]),
        "perception_model_forward_ms": float(perception.model_forward_ms),
        "perception_reported_ms": float(perception.perception_ms),
        "perception_observed_ms": round(
            float(submission["perception_observed_ms"]), 6
        ),
        "event_encode_done_ms": round(
            float(submission["event_encode_done_ms"]), 6
        ),
        "local_actionable_ms": (
            round(float(submission["local_actionable_ms"]), 6)
            if submission.get("local_actionable_ms") is not None
            else None
        ),
        "business_completion_ms": (
            round(max(business_values), 6)
            if len(business_values) == partition_count
            else None
        ),
        "global_authoritative_final_ms": (
            round(max(final_values), 6)
            if len(final_values) == partition_count
            else None
        ),
        "operational_safety_level": _sample_level(
            rows, "operational_safety_level"
        ),
        "legacy_congestion_level": _sample_level(rows, "legacy_congestion_level"),
        "all_events_responded": len(posts) == partition_count,
        "all_reviews_terminal": len(sample_reviews) == partition_count,
        "all_reviews_authoritative": authoritative_count == partition_count,
        "all_reviews_completed": authoritative_count == partition_count,
        "aggregation_group_count": len(group_ids),
        "aggregations_complete": aggregations_complete,
        "errors": list(submission["errors"]),
        "events": rows,
        "aggregations": aggregations,
    }


def _run_sample(
    runtime: Any,
    traffic_event_from_output: Any,
    sample_id: int,
    experiment_id: str,
    edge_url: str,
    cloud_url: str,
    request_timeout_seconds: float,
    final_wait_seconds: float,
    poll_interval_seconds: float,
    aggregation_timeout_ms: int,
) -> Dict[str, Any]:
    submission = _submit_sample(
        runtime,
        traffic_event_from_output,
        sample_id,
        experiment_id,
        edge_url,
        request_timeout_seconds,
        aggregation_timeout_ms,
    )
    event_ids = [
        str(envelope["id"]) for _, envelope, _ in submission["prepared"]
    ]
    reviews, observed = _wait_reviews(
        edge_url,
        event_ids,
        final_wait_seconds,
        poll_interval_seconds,
        request_timeout_seconds,
    )
    return _finalize_sample(
        submission,
        reviews,
        observed,
        cloud_url,
        request_timeout_seconds,
    )


def main() -> None:
    args = parse_args()
    if args.sample_start < 0 or args.sample_stop <= args.sample_start:
        raise ValueError("sample range is invalid")
    if min(
        args.request_timeout_seconds,
        args.final_wait_seconds,
        args.poll_interval_seconds,
    ) <= 0:
        raise ValueError("timeouts and poll interval must be positive")
    if args.aggregation_timeout_ms <= 0:
        raise ValueError("aggregation timeout must be positive")
    if args.min_qwen_selected < 0 or args.min_qwen_accepted < 0:
        raise ValueError("Qwen minimum counts must not be negative")

    project_root = Path(args.project_root).resolve()
    scene_root = project_root / "scenes" / "freeway_traffic"
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(scene_root))
    from traffic_system.current_state_perception_runtime import (
        CurrentStateTrafficPerceptionRuntime,
    )
    from traffic_system.scene_event import traffic_event_from_output

    data_path = _project_path(project_root, args.data_npz)
    runtime = CurrentStateTrafficPerceptionRuntime(
        data_path=data_path,
        rule_config_path=(
            scene_root / "assets" / "models" / "current_state_perception_v1.json"
        ),
        topology_path=(
            scene_root / "assets" / "models" / "traffic_region_topology_metis4.json"
        ),
        split=args.split,
        top_k=args.top_k,
    )
    sample_ids = list(range(args.sample_start, args.sample_stop))
    runtime.validate_sample_ids(sample_ids)
    warmup_ids = _parse_ids(args.warmup_samples)
    runtime.validate_sample_ids(warmup_ids)
    experiment_id = "pems-e2e-{}-{}".format(
        time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8]
    )

    initial_edge_health = _get_json(
        args.edge_url, "/health", args.request_timeout_seconds
    )
    initial_cloud_health = _get_json(
        args.cloud_url, "/health", args.request_timeout_seconds
    )
    initial_llm_health = _edge_llm_health(initial_edge_health)
    if args.require_qwen_selected or args.require_qwen_accepted:
        if not (
            initial_llm_health.get("mode") == "selective"
            and initial_llm_health.get("loaded") is True
            and initial_llm_health.get("gain_profile_loaded") is True
        ):
            raise RuntimeError(
                "Edge-Qwen must be loaded in selective mode with its gain profile"
            )
    warmups = []
    for index, sample_id in enumerate(warmup_ids):
        warmups.append(
            _run_sample(
                runtime,
                traffic_event_from_output,
                sample_id,
                "{}-warmup-{}".format(experiment_id, index),
                args.edge_url,
                args.cloud_url,
                args.request_timeout_seconds,
                args.final_wait_seconds,
                args.poll_interval_seconds,
                args.aggregation_timeout_ms,
            )
        )

    measured_initial_edge_health = _get_json(
        args.edge_url, "/health", args.request_timeout_seconds
    )
    measured_initial_cloud_health = _get_json(
        args.cloud_url, "/health", args.request_timeout_seconds
    )
    initial_edge_metrics = _get_json(
        args.edge_url, METRICS_PATH, args.request_timeout_seconds
    )
    initial_cloud_metrics = _get_json(
        args.cloud_url, METRICS_PATH, args.request_timeout_seconds
    )
    measured_started = time.perf_counter()
    submissions = []
    for index, sample_id in enumerate(sample_ids, start=1):
        submission = _submit_sample(
            runtime,
            traffic_event_from_output,
            sample_id,
            experiment_id,
            args.edge_url,
            args.request_timeout_seconds,
            args.aggregation_timeout_ms,
        )
        submissions.append(submission)
        print(
            "[{}/{}] submitted sample={} local={}ms".format(
                index,
                len(sample_ids),
                sample_id,
                submission.get("local_actionable_ms"),
            ),
            flush=True,
        )
    measured_event_ids = [
        str(envelope["id"])
        for submission in submissions
        for _, envelope, _ in submission["prepared"]
    ]
    measured_reviews, measured_review_observed = _wait_reviews(
        args.edge_url,
        measured_event_ids,
        args.final_wait_seconds,
        args.poll_interval_seconds,
        args.request_timeout_seconds,
    )
    samples = [
        _finalize_sample(
            submission,
            measured_reviews,
            measured_review_observed,
            args.cloud_url,
            args.request_timeout_seconds,
        )
        for submission in submissions
    ]
    measurement_wall_seconds = time.perf_counter() - measured_started
    final_edge_health = _get_json(
        args.edge_url, "/health", args.request_timeout_seconds
    )
    final_cloud_health = _get_json(
        args.cloud_url, "/health", args.request_timeout_seconds
    )
    final_edge_metrics = _get_json(
        args.edge_url, METRICS_PATH, args.request_timeout_seconds
    )
    final_cloud_metrics = _get_json(
        args.cloud_url, METRICS_PATH, args.request_timeout_seconds
    )

    rows = [row for sample in samples for row in sample["events"]]
    strata: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        route = "sync" if row["schedule_waits_for_cloud"] else "async"
        strata["{}_{}".format(row["decision_stratum"], route)].append(row)
    sample_business = [
        float(sample["business_completion_ms"])
        for sample in samples
        if sample.get("business_completion_ms") is not None
    ]
    sample_local = [
        float(sample["local_actionable_ms"])
        for sample in samples
        if sample.get("local_actionable_ms") is not None
    ]
    sample_final = [
        float(sample["global_authoritative_final_ms"])
        for sample in samples
        if sample.get("global_authoritative_final_ms") is not None
    ]
    qwen_rows = [row for row in rows if row["edge_llm_selected"]]
    qwen_accepted_rows = [row for row in rows if row["edge_llm_accepted"]]
    raw_window_bytes = int(170 * 3 * 12 * 4)
    event_operational_safety: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    event_congestion: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    event_upload: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    event_cloud_wait: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    sample_congestion: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    sample_operational_safety: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        event_operational_safety[str(row["operational_safety_level"])].append(row)
        event_congestion[str(row["legacy_congestion_level"])].append(row)
        event_upload[str(row["upload_level"])].append(row)
        event_cloud_wait[
            "sync" if row["schedule_waits_for_cloud"] else "async"
        ].append(row)
    for sample in samples:
        sample_congestion[str(sample["legacy_congestion_level"])].append(sample)
        sample_operational_safety[
            str(sample["operational_safety_level"])
        ].append(sample)

    measured_initial_llm = _edge_llm_health(measured_initial_edge_health)
    final_llm = _edge_llm_health(final_edge_health)
    qwen_health_delta = {
        name: int(final_llm.get(name, 0) or 0)
        - int(measured_initial_llm.get(name, 0) or 0)
        for name in ("invocations", "accepted", "fallbacks")
    }
    async_request_bytes = max(
        0.0,
        _metric_total(final_edge_metrics, "async_http_request_bytes")
        - _metric_total(initial_edge_metrics, "async_http_request_bytes"),
    )
    async_response_bytes = max(
        0.0,
        _metric_total(final_edge_metrics, "async_http_response_bytes")
        - _metric_total(initial_edge_metrics, "async_http_response_bytes"),
    )
    async_delivery_count = max(
        0,
        _metric_count(final_edge_metrics, "async_http_request_bytes")
        - _metric_count(initial_edge_metrics, "async_http_request_bytes"),
    )
    aggregation_results = [
        aggregation.get("result", {})
        for sample in samples
        for aggregation in sample.get("aggregations", [])
        if isinstance(aggregation, dict) and isinstance(aggregation.get("result"), dict)
    ]
    initial_conflicts = sum(
        int(value.get("initial_conflict_count", 0) or 0)
        for value in aggregation_results
    )
    residual_conflicts = sum(
        int(value.get("residual_conflict_count", 0) or 0)
        for value in aggregation_results
    )
    coordinated_events = sum(
        int(value.get("event_count", 0) or 0) for value in aggregation_results
    )
    result = {
        "schema_version": 1,
        "task": "pems08_current_state_deployed_e2e",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "measurement_scope": {
            "start": (
                "resident/prewarmed runtime, immediately before selecting and "
                "processing one already-windowed PEMS08 NPZ sample"
            ),
            "local_actionable": (
                "sample T0 to all four compact /decide responses; each response is "
                "durable at handoff/outbox boundary"
            ),
            "business_completion": (
                "local response for locally authorized actions; authoritative final "
                "for actions deferred pending cloud confirmation"
            ),
            "global_authoritative_final": (
                "sample T0 to all four authoritative edge review lifecycle records; "
                "partial_final/local_only_timeout do not qualify"
            ),
            "sample_aggregation": "maximum of the four METIS partition events",
            "stream_replay": (
                "contiguous windows are submitted in order; the next window starts "
                "after local responses without waiting for prior async cloud finality"
            ),
            "excluded": [
                "NPZ/model/config cold load",
                "warmup groups",
                "raw CSV ingestion and live 12-step buffer construction",
                "result file serialization",
            ],
        },
        "platform": {
            "python": sys.version.split()[0],
            "edge_url": args.edge_url,
            "cloud_url": args.cloud_url,
        },
        "assets": {
            "data_path": str(data_path),
            "data_sha256": _sha256(data_path),
            "data_bytes": data_path.stat().st_size,
            "split_shape": list(runtime.split_x.shape),
            "raw_window_shape": [170, 3, 12],
            "nominal_raw_window_float32_bytes": raw_window_bytes,
            "partition_sizes": [len(value) for value in runtime.partitions],
        },
        "sample_selection": {
            "split": args.split,
            "contiguous_range": [args.sample_start, args.sample_stop],
            "sample_count": len(sample_ids),
            "warmup_samples": warmup_ids,
            "selection_note": (
                "fixed contiguous test segment; no primary/forced Qwen routing"
            ),
        },
        "services": {
            "initial_edge_health": initial_edge_health,
            "initial_cloud_health": initial_cloud_health,
            "measured_initial_edge_health": measured_initial_edge_health,
            "measured_initial_cloud_health": measured_initial_cloud_health,
            "final_edge_health": final_edge_health,
            "final_cloud_health": final_cloud_health,
            "initial_edge_metrics": initial_edge_metrics,
            "initial_cloud_metrics": initial_cloud_metrics,
            "final_edge_metrics": final_edge_metrics,
            "final_cloud_metrics": final_cloud_metrics,
        },
        "execution": {
            "sample_count": len(samples),
            "event_count": len(rows),
            "all_samples_responded": sum(
                bool(sample["all_events_responded"]) for sample in samples
            ),
            "all_samples_final": sum(
                bool(sample["all_reviews_completed"]) for sample in samples
            ),
            "complete_aggregation_samples": sum(
                bool(sample["aggregations_complete"]) for sample in samples
            ),
            "event_operational_safety_levels": _counts(
                row["operational_safety_level"] for row in rows
            ),
            "event_legacy_congestion_levels": _counts(
                row["legacy_congestion_level"] for row in rows
            ),
            "event_upload_levels": _counts(row["upload_level"] for row in rows),
            "schedule_routes": _counts(row["schedule_route"] for row in rows),
            "executed_routes": _counts(row["executed_route"] for row in rows),
            "decision_strata": _counts(row["decision_stratum"] for row in rows),
            "qwen_selected_count": len(qwen_rows),
            "qwen_selected_rate": round(len(qwen_rows) / max(1, len(rows)), 6),
            "qwen_accepted_count": len(qwen_accepted_rows),
            "qwen_acceptance_rate_when_selected": round(
                len(qwen_accepted_rows) / max(1, len(qwen_rows)), 6
            ),
            "qwen_selection_reasons": _counts(
                row["edge_llm_selection_reason"] for row in qwen_rows
            ),
            "all_qwen_selection_reasons": _counts(
                row["edge_llm_selection_reason"] for row in rows
            ),
            "qwen_fallback_reasons": _counts(
                row["edge_llm_fallback_reason"]
                for row in qwen_rows
                if row.get("edge_llm_fallback_reason")
            ),
            "qwen_health_counter_delta": qwen_health_delta,
            "qwen_runtime_error_count": sum(
                bool(row.get("edge_llm_runtime_error")) for row in rows
            ),
            "local_requires_review_count": sum(
                bool(row.get("local_requires_review")) for row in rows
            ),
            "local_requires_review_without_sync_count": sum(
                bool(row.get("local_requires_review"))
                and not bool(row.get("schedule_waits_for_cloud"))
                for row in rows
            ),
            "local_requires_synchronous_review_count": sum(
                bool(row.get("local_requires_synchronous_review"))
                for row in rows
            ),
            "local_requires_synchronous_review_without_sync_count": sum(
                bool(row.get("local_requires_synchronous_review"))
                and not bool(row.get("schedule_waits_for_cloud"))
                for row in rows
            ),
            "synchronous_review_reason_counts": _counts(
                reason
                for row in rows
                for reason in row.get("synchronous_review_reasons", [])
            ),
            "student_rule_disagreement_async_count": sum(
                bool(row.get("student_rule_disagreement"))
                and not bool(row.get("schedule_waits_for_cloud"))
                for row in rows
            ),
            "qwen_corroborated_async_count": sum(
                row.get("synchronous_review_resolution")
                == "edge_qwen_corroborated_student"
                and not bool(row.get("schedule_waits_for_cloud"))
                for row in rows
            ),
            "scheduled_sync_executed_routes": _counts(
                row["executed_route"]
                for row in rows
                if bool(row.get("schedule_waits_for_cloud"))
            ),
            "measurement_wall_seconds": round(measurement_wall_seconds, 6),
        },
        "latency_ms": {
            "sample_local_actionable": _summary(sample_local),
            "sample_business_completion": _summary(sample_business),
            "sample_global_authoritative_final": _summary(sample_final),
            "sample_perception": _summary(
                sample["perception_observed_ms"] for sample in samples
            ),
            "event_http_wall": _summary(row["http_wall_ms"] for row in rows),
            "event_framework_runtime": _summary(
                row["framework_runtime_ms"] for row in rows
            ),
            "pipeline_stage": {
                name: _summary(
                    row.get("pipeline_stage_ms", {}).get(name, 0.0)
                    for row in rows
                )
                for name in (
                    "normalization",
                    "edge_decision",
                    "data_plane_preparation",
                    "scheduling",
                    "route_execution",
                )
            },
            "edge_qwen_selected": _summary(
                row["edge_llm_latency_ms"]
                for row in qwen_rows
                if row["edge_llm_latency_ms"] > 0.0
            ),
        },
        "sla": {
            "sample_business_mean_under_200ms": bool(
                sample_business and statistics.fmean(sample_business) < 200.0
            ),
            "sample_business_under_200ms_rate": round(
                sum(value < 200.0 for value in sample_business)
                / max(1, len(samples)),
                6,
            ),
            "sample_local_under_200ms_rate": round(
                sum(value < 200.0 for value in sample_local) / max(1, len(samples)),
                6,
            ),
        },
        "communication": {
            "nominal_raw_input_bytes_per_sample": raw_window_bytes,
            "edge_ingress_request_bytes_per_sample": _summary(
                sum(row["ingress_request_bytes"] for row in sample["events"])
                for sample in samples
            ),
            "edge_ingress_response_bytes_per_sample": _summary(
                sum(row["ingress_response_bytes"] for row in sample["events"])
                for sample in samples
            ),
            "selected_cloud_request_bytes_per_event": _summary(
                row["selected_cloud_request_bytes"] for row in rows
            ),
            "response_time_cloud_transport_request_bytes_per_event": _summary(
                row["actual_cloud_transport_request_bytes"] for row in rows
            ),
            "measured_async_http_delivery_count": async_delivery_count,
            "measured_async_http_request_bytes_total": round(
                async_request_bytes, 3
            ),
            "measured_async_http_response_bytes_total": round(
                async_response_bytes, 3
            ),
            "measured_async_http_request_bytes_per_event": round(
                async_request_bytes / max(1, len(rows)), 3
            ),
            "measured_async_http_request_bytes_per_delivery": round(
                async_request_bytes / max(1, async_delivery_count), 3
            ),
            "measured_async_http_transport_bytes_total": round(
                async_request_bytes + async_response_bytes, 3
            ),
        },
        "consistency": {
            "aggregation_result_count": len(aggregation_results),
            "coordinated_event_count": coordinated_events,
            "initial_conflict_count": initial_conflicts,
            "residual_conflict_count": residual_conflicts,
            "conflict_rate": round(
                initial_conflicts / max(1, coordinated_events), 6
            ),
            "conflict_resolution_success_rate": round(
                (initial_conflicts - residual_conflicts) / max(1, initial_conflicts),
                6,
            )
            if initial_conflicts
            else 1.0,
            "note": (
                "natural PEMS aggregation conflicts only; explicit "
                "conflict_suspected/policy-force branches require separate regression"
            ),
        },
        "strata": {
            name: _stratum_summary(values) for name, values in sorted(strata.items())
        },
        "event_by_operational_safety": {
            name: _stratum_summary(values)
            for name, values in sorted(event_operational_safety.items())
        },
        "event_by_legacy_congestion": {
            name: _stratum_summary(values)
            for name, values in sorted(event_congestion.items())
        },
        "event_by_upload_level": {
            name: _stratum_summary(values)
            for name, values in sorted(event_upload.items())
        },
        "event_by_cloud_wait": {
            name: _stratum_summary(values)
            for name, values in sorted(event_cloud_wait.items())
        },
        "sample_by_legacy_congestion": {
            name: _sample_stratum_summary(values)
            for name, values in sorted(sample_congestion.items())
        },
        "sample_by_operational_safety": {
            name: _sample_stratum_summary(values)
            for name, values in sorted(sample_operational_safety.items())
        },
        "warmups": warmups,
        "samples": samples,
    }
    output_path = _project_path(project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "execution": result["execution"],
                "latency_ms": result["latency_ms"],
                "sla": result["sla"],
                "strata": result["strata"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.require_qwen_selected and not qwen_rows:
        raise RuntimeError("no natural measured event selected Edge-Qwen")
    if args.require_qwen_accepted and not qwen_accepted_rows:
        raise RuntimeError("no natural measured Edge-Qwen result was accepted")
    if len(qwen_rows) < args.min_qwen_selected:
        raise RuntimeError(
            "natural Qwen selected count {} is below {}".format(
                len(qwen_rows), args.min_qwen_selected
            )
        )
    if len(qwen_accepted_rows) < args.min_qwen_accepted:
        raise RuntimeError(
            "Qwen accepted count {} is below {}".format(
                len(qwen_accepted_rows), args.min_qwen_accepted
            )
        )
    if args.require_risk_coverage:
        observed = {str(row["legacy_congestion_level"]) for row in rows}
        missing = {"low", "medium", "high", "severe"} - observed
        if missing:
            raise RuntimeError(
                "PEMS segment is missing congestion levels: {}".format(
                    sorted(missing)
                )
            )
    if args.require_complete_final:
        incomplete = [
            sample["sample_id"]
            for sample in samples
            if not sample["all_events_responded"]
            or not sample["all_reviews_authoritative"]
            or not sample["aggregations_complete"]
        ]
        if incomplete:
            raise RuntimeError("incomplete sample groups: {}".format(incomplete))


if __name__ == "__main__":
    main()
