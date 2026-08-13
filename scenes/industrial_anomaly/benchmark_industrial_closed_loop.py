#!/usr/bin/env python3
"""Run an industrial benchmark aligned with the traffic evidence categories."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


STATES = ("normal", "review", "anomaly")
MODALITIES = ("rgb", "infrared")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _request_json(
    url: str,
    payload: Optional[Mapping[str, Any]] = None,
    *,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = 10.0,
) -> Tuple[Dict[str, Any], int, int, float]:
    body = None
    method = "GET"
    request_headers = {"Accept": "application/json", **dict(headers or {})}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        request_headers["Content-Type"] = "application/json"
        method = "POST"
    request = Request(url, data=body, headers=request_headers, method=method)
    started = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        raw_response = response.read()
    wall_ms = (time.perf_counter() - started) * 1000.0
    result = json.loads(raw_response.decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("{} returned non-object JSON".format(url))
    return result, len(body or b""), len(raw_response), wall_ms


def _set_profile(proxy_url: str, profile: str) -> Dict[str, Any]:
    result, _, _, _ = _request_json(
        proxy_url.rstrip("/") + "/__fault__/profile",
        {"profile": profile},
    )
    time.sleep(0.7)
    return result


def _proxy_status(proxy_url: str) -> Dict[str, Any]:
    result, _, _, _ = _request_json(
        proxy_url.rstrip("/") + "/__fault__/status"
    )
    return result


def _score(bands: Mapping[str, Any], modality: str, product: str, state: str) -> float:
    band = bands["modalities"][modality][product]
    low = float(band["review_low"])
    high = float(band["review_high"])
    width = high - low
    if state == "normal":
        return low - width * 0.25
    if state == "review":
        return (low + high) / 2.0
    return high + width * 0.25


def _event(
    template: Mapping[str, Any],
    bands: Mapping[str, Any],
    run_id: str,
    group_index: int,
    modality: str,
    state: str,
    *,
    experiment: str,
) -> Dict[str, Any]:
    product = "capsule"
    sample_id = "{}-{:04d}".format(experiment, group_index)
    value = json.loads(json.dumps(template))
    value["id"] = "{}-{}".format(sample_id, modality)
    value["edgeid"] = "industrial-{}-edge".format(modality)
    value["source"] = "urn:edge:industrial-{}:benchmark".format(modality)
    value["subject"] = sample_id
    value["time"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    data = value["data"]
    data["sample_id"] = sample_id
    data["product"] = product
    data["modality"] = modality
    data["score"] = _score(bands, modality, product, state)
    data["inference_ms"] = 0.0
    data["preprocessing_latency_ms"] = 0.0
    data["deadline_ms"] = 200.0
    data["model"] = {
        "name": "precomputed-industrial-score",
        "version": "benchmark-input-v1",
    }
    data["raw_uri"] = "file:///benchmark/{}/{}/{}.png".format(
        product, modality, sample_id
    )
    data["heatmap_uri"] = "https://evidence.invalid/{}/{}/{}.f32".format(
        product, modality, sample_id
    )
    value["traceid"] = "trace-{}-{}-{}".format(run_id, group_index, modality)
    return value


def _post_event(
    edge_url: str,
    event: Mapping[str, Any],
    bands: Mapping[str, Any],
    *,
    async_mode: bool,
    force_sync: bool,
    timeout: float,
) -> Dict[str, Any]:
    headers = {
        "Idempotency-Key": "benchmark-{}".format(event["id"]),
        "X-Trace-ID": str(event.get("traceid", "")),
        "X-Response-Detail": "compact",
    }
    if async_mode:
        headers["Prefer"] = "respond-async"
    response, request_bytes, response_bytes, wall_ms = _request_json(
        edge_url.rstrip("/") + "/api/v1/collaboration/decide",
        {
            "event": event,
            "conflict_suspected": bool(force_sync),
        },
        headers=headers,
        timeout=timeout,
    )
    local = response.get("local_decision") or response.get("final_decision") or {}
    final = response.get("final_decision") or {}
    accounting = response.get("closed_loop_accounting", {})
    stages = accounting.get("pipeline_stage_ms", {})
    metadata = local.get("metadata", {}) if isinstance(local, dict) else {}
    data_plane = response.get("data_plane", {})
    return {
        "event_id": str(event["id"]),
        "modality": str(event["data"]["modality"]),
        "reference": (
            "normal"
            if float(event["data"]["score"])
            < float(
                bands["modalities"][str(event["data"]["modality"])][
                    str(event["data"]["product"])
                ]["review_low"]
            )
            else (
                "review"
                if float(event["data"]["score"])
                < float(
                    bands["modalities"][str(event["data"]["modality"])][
                        str(event["data"]["product"])
                    ]["review_high"]
                )
                else "anomaly"
            )
        ),
        "response_detail": str(response.get("response_detail", "")),
        "client_request_bytes": request_bytes,
        "client_response_bytes": response_bytes,
        "client_wall_ms": round(wall_ms, 6),
        "edge_service_wall_ms": float(response.get("edge_service_wall_ms", 0.0)),
        "framework_runtime_ms": float(response.get("framework_runtime_ms", 0.0)),
        "edge_preliminary_decision_ms": float(
            accounting.get("edge_preliminary_decision_ms", 0.0)
        ),
        "accounted_closed_loop_ms": float(
            accounting.get("accounted_closed_loop_ms", 0.0)
        ),
        "pipeline_stage_ms": {
            key: float(value) for key, value in stages.items()
        },
        "local_decision": str(local.get("decision", "")),
        "response_decision": str(final.get("decision", "")),
        "response_status": str(final.get("status", "")),
        "response_route": str(final.get("route", "")),
        "schedule_route": str(response.get("schedule", {}).get("route", "")),
        "qwen_selected": bool(metadata.get("edge_qwen_selected", False)),
        "qwen_selection_reason": metadata.get("edge_qwen_selection_reason"),
        "qwen_prediction": metadata.get("edge_qwen_prediction"),
        "qwen_token": metadata.get("edge_qwen_token"),
        "qwen_latency_ms": (
            float(metadata["edge_qwen_latency_ms"])
            if metadata.get("edge_qwen_latency_ms") is not None
            else None
        ),
        "qwen_prompt_tokens": metadata.get("edge_qwen_prompt_tokens"),
        "qwen_output_tokens": metadata.get("edge_qwen_output_tokens"),
        "qwen_rule_agreement": metadata.get("edge_qwen_rule_agreement"),
        "selected_request_bytes": int(data_plane.get("selected_request_bytes", 0)),
        "request_reduction_ratio": float(data_plane.get("request_reduction_ratio", 0.0)),
    }


def _send_group(
    edge_url: str,
    rgb_event: Mapping[str, Any],
    infrared_event: Mapping[str, Any],
    bands: Mapping[str, Any],
    *,
    async_mode: bool,
    force_sync: bool,
    timeout: float,
) -> Dict[str, Any]:
    started = time.perf_counter()
    records = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _post_event,
                edge_url,
                event,
                bands,
                async_mode=async_mode,
                force_sync=force_sync,
                timeout=timeout,
            )
            for event in (rgb_event, infrared_event)
        ]
        for future in as_completed(futures):
            records.append(future.result())
    return {
        "sample_id": str(rgb_event["data"]["sample_id"]),
        "client_group_wall_ms": round((time.perf_counter() - started) * 1000.0, 6),
        "events": sorted(records, key=lambda item: item["modality"]),
    }


def _wait_reviews(
    edge_url: str,
    event_ids: Sequence[str],
    timeout: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    remaining = set(event_ids)
    results: Dict[str, Dict[str, Any]] = {}
    deadline = time.monotonic() + timeout
    while remaining and time.monotonic() < deadline:
        for event_id in list(remaining):
            try:
                review, _, _, poll_ms = _request_json(
                    edge_url.rstrip("/")
                    + "/api/v1/collaboration/reviews/"
                    + quote(event_id, safe=""),
                    timeout=3.0,
                )
            except (HTTPError, URLError, TimeoutError, OSError):
                continue
            final = review.get("final_decision")
            aggregation = (
                final.get("metadata", {}).get("aggregation", {})
                if isinstance(final, dict)
                else {}
            )
            if (
                review.get("state") == "completed"
                and isinstance(final, dict)
                and aggregation.get("finality") == "final"
                and bool(aggregation.get("global_confirmation"))
            ):
                results[event_id] = {
                    "event_id": event_id,
                    "state": review.get("state"),
                    "attempts": review.get("attempts"),
                    "preliminary_latency_ms": review.get("preliminary_latency_ms"),
                    "cloud_receipt_latency_ms": review.get("cloud_receipt_latency_ms"),
                    "eventual_completion_ms": review.get("eventual_completion_ms"),
                    "requested_at_ms": review.get("requested_at_ms"),
                    "queued_at_ms": review.get("queued_at_ms"),
                    "cloud_received_at_ms": review.get("cloud_received_at_ms"),
                    "completed_at_ms": review.get("completed_at_ms"),
                    "planned_request_bytes": review.get("planned_request_bytes"),
                    "completion_stage": review.get("completion_stage"),
                    "decision_changed": review.get("decision_changed"),
                    "final_decision": final.get("decision"),
                    "final_status": final.get("status"),
                    "aggregation": aggregation,
                    "result_poll_ms": round(poll_ms, 6),
                    "result_transport": final.get("metadata", {}).get(
                        "transport", {}
                    ),
                }
                remaining.remove(event_id)
        if remaining:
            time.sleep(0.03)
    return [results[event_id] for event_id in event_ids if event_id in results], sorted(remaining)


def _event_records(groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [event for group in groups for event in group["events"]]


def _profile_summary(
    groups: Sequence[Dict[str, Any]],
    reviews: Sequence[Dict[str, Any]],
    missing: Sequence[str],
) -> Dict[str, Any]:
    events = _event_records(groups)
    stage_names = sorted(
        {name for event in events for name in event["pipeline_stage_ms"]}
    )
    local_correct = sum(
        event["local_decision"] == event["reference"] for event in events
    )
    qwen_selected = [event for event in events if event["qwen_selected"]]
    qwen_completed = [
        event for event in qwen_selected
        if event["qwen_rule_agreement"] is not None
    ]
    qwen_agreements = sum(
        event["qwen_rule_agreement"] is True for event in qwen_completed
    )
    return {
        "group_count": len(groups),
        "event_count": len(events),
        "local_success_rate": round(len(events) / max(1, len(groups) * 2), 6),
        "local_accuracy": round(local_correct / max(1, len(events)), 6),
        "qwen_selection_rate": round(
            len(qwen_selected) / max(1, len(events)), 6
        ),
        "qwen_selected_count": len(qwen_selected),
        "qwen_completed_count": len(qwen_completed),
        "qwen_completion_rate": round(
            len(qwen_completed) / max(1, len(qwen_selected)), 6
        ),
        "qwen_rule_agreement_rate": round(
            qwen_agreements / max(1, len(qwen_completed)), 6
        ),
        "qwen_routing_contract_satisfied": all(
            event["qwen_selected"] == (event["reference"] == "review")
            for event in events
        ),
        "authoritative_final_rate": round(
            len(reviews) / max(1, len(events)), 6
        ),
        "missing_final_event_ids": list(missing),
        "client_input_to_provisional_ms": _summary(
            event["client_wall_ms"] for event in events
        ),
        "edge_service_wall_ms": _summary(
            event["edge_service_wall_ms"] for event in events
        ),
        "edge_preliminary_decision_ms": _summary(
            event["edge_preliminary_decision_ms"] for event in events
        ),
        "group_input_to_all_provisional_ms": _summary(
            group["client_group_wall_ms"] for group in groups
        ),
        "qwen_latency_ms": _summary(
            event["qwen_latency_ms"]
            for event in qwen_selected
            if event["qwen_latency_ms"] is not None
        ),
        "qwen_amortized_latency_ms": _summary(
            event["qwen_latency_ms"] or 0.0 for event in events
        ),
        "pipeline_stage_ms": {
            name: _summary(
                event["pipeline_stage_ms"].get(name, 0.0) for event in events
            )
            for name in stage_names
        },
        "input_to_authoritative_final_ms": _summary(
            float(review["eventual_completion_ms"]) for review in reviews
        ),
        "cloud_receipt_latency_ms": _summary(
            float(review["cloud_receipt_latency_ms"])
            for review in reviews
            if review.get("cloud_receipt_latency_ms") is not None
        ),
        "cloud_received_to_edge_backfill_ms": _summary(
            max(
                0.0,
                float(review["completed_at_ms"])
                - float(review["cloud_received_at_ms"]),
            )
            for review in reviews
            if review.get("cloud_received_at_ms") is not None
            and review.get("completed_at_ms") is not None
        ),
        "selected_request_bytes": _summary(
            event["selected_request_bytes"] for event in events
        ),
        "planned_request_bytes": _summary(
            float(review["planned_request_bytes"])
            for review in reviews
            if review.get("planned_request_bytes") is not None
        ),
        "client_request_bytes": _summary(
            event["client_request_bytes"] for event in events
        ),
        "client_response_bytes": _summary(
            event["client_response_bytes"] for event in events
        ),
        "records": {"groups": list(groups), "reviews": list(reviews)},
    }


def _run_profile(
    name: str,
    count: int,
    state_pairs: Sequence[Tuple[str, str]],
    templates: Mapping[str, Any],
    bands: Mapping[str, Any],
    run_id: str,
    edge_url: str,
    *,
    async_mode: bool,
    force_sync: bool = False,
    request_timeout: float,
    final_timeout: float,
    group_concurrency: int = 1,
) -> Dict[str, Any]:
    definitions = []
    for index in range(count):
        rgb_state, infrared_state = state_pairs[index % len(state_pairs)]
        definitions.append(
            (
                _event(
                    templates["rgb"], bands, run_id, index, "rgb", rgb_state,
                    experiment="{}-{}".format(run_id, name),
                ),
                _event(
                    templates["infrared"], bands, run_id, index, "infrared",
                    infrared_state,
                    experiment="{}-{}".format(run_id, name),
                ),
            )
        )
    groups: List[Dict[str, Any]] = []
    if group_concurrency == 1:
        for rgb_event, infrared_event in definitions:
            groups.append(
                _send_group(
                    edge_url,
                    rgb_event,
                    infrared_event,
                    bands,
                    async_mode=async_mode,
                    force_sync=force_sync,
                    timeout=request_timeout,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=group_concurrency) as executor:
            futures = [
                executor.submit(
                    _send_group,
                    edge_url,
                    rgb_event,
                    infrared_event,
                    bands,
                    async_mode=async_mode,
                    force_sync=force_sync,
                    timeout=request_timeout,
                )
                for rgb_event, infrared_event in definitions
            ]
            for future in as_completed(futures):
                groups.append(future.result())
    event_ids = [event["event_id"] for event in _event_records(groups)]
    reviews, missing = _wait_reviews(edge_url, event_ids, final_timeout)
    result = _profile_summary(groups, reviews, missing)
    result["profile"] = name
    result["async_mode"] = async_mode
    result["force_sync"] = force_sync
    result["group_concurrency"] = group_concurrency
    return result


def _conflict_summary(profile: Mapping[str, Any]) -> Dict[str, Any]:
    groups = profile["records"]["groups"]
    reviews = profile["records"]["reviews"]
    review_by_id = {review["event_id"]: review for review in reviews}
    disagreements = 0
    resolved = 0
    for group in groups:
        decisions = {event["local_decision"] for event in group["events"]}
        if len(decisions) <= 1:
            continue
        disagreements += 1
        group_reviews = [
            review_by_id.get(event["event_id"]) for event in group["events"]
        ]
        if all(
            review is not None
            and review["final_decision"] == "review"
            and review["aggregation"].get("global_confirmation") is True
            for review in group_reviews
        ):
            resolved += 1
    return {
        "controlled_disagreement_groups": disagreements,
        "safe_review_resolutions": resolved,
        "resolution_success_rate": round(resolved / max(1, disagreements), 6),
        "residual_semantic_conflicts": disagreements - resolved,
        "note": (
            "These are controlled RGB/infrared decision disagreements. They are "
            "reported separately from framework actuator-conflict counters."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-url", default="http://127.0.0.1:19501")
    parser.add_argument("--cloud-url", default="http://127.0.0.1:19500")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:19510")
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--rgb-template", required=True)
    parser.add_argument("--infrared-template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="run only the frozen 90-group/180-event normal_async primary profile",
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    bands_path = Path(args.thresholds).resolve()
    bands = json.loads(bands_path.read_text(encoding="utf-8"))
    templates = {
        "rgb": json.loads(Path(args.rgb_template).read_text(encoding="utf-8")),
        "infrared": json.loads(
            Path(args.infrared_template).read_text(encoding="utf-8")
        ),
    }
    run_id = "industrial-full-{}".format(int(time.time()))
    report: Dict[str, Any] = {
        "schema_version": "industrial-closed-loop-benchmark/v1",
        "run_id": run_id,
        "endpoints": {
            "edge": args.edge_url,
            "cloud": args.cloud_url,
            "fault_proxy": args.proxy_url,
        },
        "inputs": {
            "thresholds": _identity(bands_path),
            "rgb_template": _identity(Path(args.rgb_template)),
            "infrared_template": _identity(Path(args.infrared_template)),
            "perception_model_executed": False,
            "perception_accuracy_available": False,
            "perception_note": (
                "The provided industrial repository contains event contracts and "
                "review bands but no runnable RGB/infrared perception weights or "
                "labeled visual test set. Scores are controlled inputs."
            ),
        },
        "profiles": {},
    }

    _set_profile(args.proxy_url, "normal")
    report["profiles"]["normal_async"] = _run_profile(
        "normal-async",
        90,
        (("normal", "normal"), ("review", "review"), ("anomaly", "anomaly")),
        templates,
        bands,
        run_id,
        args.edge_url,
        async_mode=True,
        request_timeout=5.0,
        final_timeout=15.0,
    )
    report["profiles"]["normal_async"]["proxy"] = _proxy_status(args.proxy_url)

    if args.primary_only:
        normal_events = _event_records(
            report["profiles"]["normal_async"]["records"]["groups"]
        )
        report["scope_notes"] = {
            "physical_edge_nodes": 1,
            "aggregation_members": ["rgb", "infrared"],
            "primary_only": True,
            "request_event_count": len(normal_events),
            "cloud_9b_invocations": 0,
            "cloud_9b_note": (
                "Industrial cloud final uses deterministic cross-modal policy."
            ),
        }
        with output.open("x", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
        print(
            json.dumps(
                {
                    "output": str(output),
                    "primary": {
                        key: report["profiles"]["normal_async"][key]
                        for key in (
                            "event_count",
                            "local_accuracy",
                            "client_input_to_provisional_ms",
                            "qwen_selection_rate",
                            "qwen_selected_count",
                            "qwen_completion_rate",
                            "qwen_rule_agreement_rate",
                            "qwen_routing_contract_satisfied",
                            "qwen_latency_ms",
                            "qwen_amortized_latency_ms",
                        )
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    report["profiles"]["normal_sync"] = _run_profile(
        "normal-sync",
        30,
        (("normal", "normal"), ("review", "review"), ("anomaly", "anomaly")),
        templates,
        bands,
        run_id,
        args.edge_url,
        async_mode=False,
        force_sync=True,
        request_timeout=5.0,
        final_timeout=15.0,
    )

    report["profiles"]["controlled_conflict"] = _run_profile(
        "controlled-conflict",
        30,
        (("normal", "anomaly"), ("anomaly", "normal"), ("review", "anomaly")),
        templates,
        bands,
        run_id,
        args.edge_url,
        async_mode=True,
        request_timeout=5.0,
        final_timeout=15.0,
    )
    report["conflicts"] = _conflict_summary(
        report["profiles"]["controlled_conflict"]
    )

    for profile_name in ("mild", "severe"):
        _set_profile(args.proxy_url, profile_name)
        result = _run_profile(
            profile_name,
            20,
            (("normal", "normal"), ("review", "review"), ("anomaly", "anomaly")),
            templates,
            bands,
            run_id,
            args.edge_url,
            async_mode=True,
            request_timeout=5.0,
            final_timeout=30.0,
        )
        result["proxy"] = _proxy_status(args.proxy_url)
        report["profiles"][profile_name] = result

    _set_profile(args.proxy_url, "outage")
    outage = _run_profile(
        "outage",
        10,
        (("normal", "normal"), ("review", "review"), ("anomaly", "anomaly")),
        templates,
        bands,
        run_id,
        args.edge_url,
        async_mode=True,
        request_timeout=5.0,
        final_timeout=1.0,
    )
    outage["proxy_during_outage"] = _proxy_status(args.proxy_url)
    outbox_during, _, _, _ = _request_json(
        args.edge_url.rstrip("/") + "/api/v1/framework/outbox"
    )
    outage["outbox_during_outage"] = outbox_during
    _set_profile(args.proxy_url, "normal")
    outage_event_ids = [
        event["event_id"] for event in _event_records(outage["records"]["groups"])
    ]
    recovery_started = time.perf_counter()
    recovered, still_missing = _wait_reviews(
        args.edge_url, outage_event_ids, 30.0
    )
    outage["recovery"] = {
        "recovery_wall_ms": round(
            (time.perf_counter() - recovery_started) * 1000.0, 6
        ),
        "authoritative_final_count": len(recovered),
        "authoritative_final_rate": round(
            len(recovered) / max(1, len(outage_event_ids)), 6
        ),
        "missing_event_ids": still_missing,
    }
    outage["proxy_after_recovery"] = _proxy_status(args.proxy_url)
    report["profiles"]["outage_and_recovery"] = outage

    report["full_endpoint_concurrency"] = []
    for concurrency in (1, 2, 4, 8):
        result = _run_profile(
            "concurrency-{}".format(concurrency),
            20,
            (("normal", "normal"), ("review", "review"), ("anomaly", "anomaly")),
            templates,
            bands,
            run_id,
            args.edge_url,
            async_mode=True,
            request_timeout=10.0,
            final_timeout=20.0,
            group_concurrency=concurrency,
        )
        report["full_endpoint_concurrency"].append(
            {
                key: value
                for key, value in result.items()
                if key != "records"
            }
        )

    edge_metrics, _, _, _ = _request_json(
        args.edge_url.rstrip("/") + "/api/v1/framework/metrics"
    )
    cloud_metrics, _, _, _ = _request_json(
        args.cloud_url.rstrip("/") + "/api/v1/framework/metrics"
    )
    edge_health, _, _, _ = _request_json(
        args.edge_url.rstrip("/") + "/health"
    )
    cloud_health, _, _, _ = _request_json(
        args.cloud_url.rstrip("/") + "/health"
    )
    final_outbox, _, _, _ = _request_json(
        args.edge_url.rstrip("/") + "/api/v1/framework/outbox"
    )
    report["service_evidence"] = {
        "edge_metrics": edge_metrics,
        "cloud_metrics": cloud_metrics,
        "edge_health": edge_health,
        "cloud_health": cloud_health,
        "final_outbox": final_outbox,
        "fault_proxy_final": _proxy_status(args.proxy_url),
    }

    raw_heatmap_bytes = 160 * 160 * 4
    normal_events = _event_records(
        report["profiles"]["normal_async"]["records"]["groups"]
    )
    selected = [event["selected_request_bytes"] for event in normal_events]
    report["communication_efficiency"] = {
        "raw_heatmap_bytes_per_event": raw_heatmap_bytes,
        "selected_cloud_payload_bytes_per_event": _summary(selected),
        "mean_reduction_vs_raw_heatmap": round(
            1.0 - statistics.fmean(selected) / raw_heatmap_bytes, 6
        ),
        "note": (
            "The edge sends a semantic summary and evidence reference. The actual "
            "160x160 float heatmap body is not uploaded in this benchmark."
        ),
    }
    report["scope_notes"] = {
        "physical_edge_nodes": 1,
        "aggregation_members": ["rgb", "infrared"],
        "natural_conflict_rate_available": False,
        "controlled_conflict_rate_reported": True,
        "cloud_9b_invocations": 0,
        "cloud_9b_note": "Industrial cloud final uses deterministic cross-modal policy.",
    }
    with output.open("x", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "profiles": {
                    name: {
                        key: value
                        for key, value in result.items()
                        if key
                        in {
                            "event_count",
                            "local_accuracy",
                            "authoritative_final_rate",
                            "client_input_to_provisional_ms",
                            "input_to_authoritative_final_ms",
                            "qwen_latency_ms",
                            "missing_final_event_ids",
                        }
                    }
                    for name, result in report["profiles"].items()
                },
                "conflicts": report["conflicts"],
                "communication_efficiency": report["communication_efficiency"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
