#!/usr/bin/env python3
"""用真实 HTTP 故障和四个独立逻辑边测量弱网业务保持率。

本脚本不注入调度结果，也不手工刷新 Outbox。它只切换一个位于边缘与
既有云服务之间的故障代理，并以现有边缘服务、主动网络探测、持久 Outbox
和后台重放完成 normal -> mild -> severe -> outage -> recovery 闭环。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from http.server import ThreadingHTTPServer
import json
import multiprocessing
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import uuid

from jsonschema import Draft202012Validator


SCENE_ROOT = Path(__file__).resolve().parent
SDK_ROOT = SCENE_ROOT.parents[1]
for _import_root in (SDK_ROOT, SCENE_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from benchmark_real_current_state_e2e import (  # noqa: E402
    AGGREGATION_PATH,
    REVIEW_PATH,
    _authoritative_review,
    _get_json,
)
from prepare_metis_partition_data import prepare_partition_data  # noqa: E402
from run_partitioned_current_state_edges import (  # noqa: E402
    _collect_messages,
    _edge_worker,
    _launch_isolated_edge_services,
    _stop_edge_services,
    _validate_cloud_evidence_pull_allowlist,
)
from traffic_system.network_fault_proxy import (  # noqa: E402
    CONTROL_PROFILE_PATH,
    FaultState,
    PROFILES,
    build_handler,
)


PROFILE_ORDER = ("normal", "mild", "severe", "outage")
WEAK_PROFILES = ("mild", "severe", "outage")
EXPECTED_MEMBERS = {"edge_node_{}".format(value) for value in range(4)}
OUTBOX_PATH = "/api/v1/framework/outbox"
HEALTH_PATH = "/health"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _load_experiment_plan(path: Path) -> Tuple[Dict[str, Any], str]:
    """Load and validate the immutable plan before any measured request starts."""
    plan_path = Path(path).resolve()
    schema_path = SDK_ROOT / "schemas" / "weak_network_experiment_plan.schema.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(plan),
        key=lambda value: list(value.absolute_path),
    )
    if errors:
        detail = "; ".join(
            "{}: {}".format(
                "/".join(str(item) for item in error.absolute_path) or "$",
                error.message,
            )
            for error in errors
        )
        raise ValueError("weak-network experiment plan is invalid: {}".format(detail))
    flattened = [
        int(sample_id)
        for profile in PROFILE_ORDER
        if profile != "recovery"
        for sample_id in plan["sample_ids_by_profile"][profile]
    ]
    if len(flattened) != len(set(flattened)):
        raise ValueError("sample_ids_by_profile must be disjoint across profiles")
    expected_profiles = {name: dict(PROFILES[name]) for name in PROFILE_ORDER}
    if _canonical(plan["fault_profiles"]) != _canonical(expected_profiles):
        raise ValueError("fault_profiles differ from the committed proxy profiles")
    try:
        created_at = datetime.fromisoformat(
            str(plan["created_at"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("experiment plan created_at is not RFC3339") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("experiment plan created_at must include a timezone")
    return dict(plan), _sha256(plan_path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _post_json(
    base_url: str,
    path: str,
    value: Mapping[str, Any],
    timeout_seconds: float,
    headers: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    body = json.dumps(value, separators=(",", ":")).encode("utf-8")
    request_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    request_headers.update(dict(headers or {}))
    request = Request(
        base_url.rstrip("/") + path,
        data=body,
        headers=request_headers,
        method="POST",
    )
    with urlopen(request, timeout=float(timeout_seconds)) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("POST {} did not return an object".format(path))
    return result


def _git_identity(project_root: Path) -> Tuple[str, List[str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(project_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(project_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return commit, dirty


def _hardware_id(value: str) -> str:
    if value.strip():
        return value.strip()
    return "{}|four_logical_edge_processes".format(socket.gethostname())


def _model_ids(scene_root: Path) -> Dict[str, str]:
    paths = {
        "current_state_perception": (
            scene_root / "assets" / "models" / "current_state_perception_v1.json"
        ),
        "edge_student": (
            scene_root
            / "assets"
            / "models"
            / "edge_student_freeway_current_state_future_v1.json"
        ),
        "edge_feature_codec": (
            scene_root
            / "assets"
            / "models"
            / "traffic_tree_feature_codec_current_state_v1.npz"
        ),
        "edge_qwen_gain_router": (
            scene_root
            / "assets"
            / "models"
            / "edge_qwen_gain_router_current_state_v1.json"
        ),
        "cloud_coordinator": (
            scene_root
            / "assets"
            / "models"
            / "cloud_coordinator_current_state_future_v1.joblib"
        ),
        "traffic_topology": (
            scene_root / "assets" / "models" / "traffic_region_topology_metis4.json"
        ),
        "edge_plugin_config": (
            scene_root / "deployment" / "full" / "scene_plugins_edge.json"
        ),
        "cloud_plugin_config": (
            scene_root / "deployment" / "full" / "scene_plugins_cloud.json"
        ),
        "edge_llm_runtime_config": (
            scene_root / "deployment" / "full" / "edge_llm_runtime.json"
        ),
        "edge_llm_release_registry": (
            scene_root / "runtime" / "edge_llm_release_store.json"
        ),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _set_proxy_profile(
    proxy_url: str,
    profile: str,
    token: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError("unknown proxy profile: {}".format(profile))
    return _post_json(
        proxy_url,
        CONTROL_PROFILE_PATH,
        {"profile": profile},
        timeout_seconds,
        {"X-Fault-Control-Token": token},
    )


def _network_snapshot(health: Mapping[str, Any]) -> Dict[str, Any]:
    network = health.get("network", {})
    network = dict(network) if isinstance(network, Mapping) else {}
    snapshot = network.get("snapshot", {})
    snapshot = dict(snapshot) if isinstance(snapshot, Mapping) else {}
    latest = network.get("latest", {})
    latest = dict(latest) if isinstance(latest, Mapping) else {}
    return {
        "available": bool(snapshot.get("available", False)),
        "rtt_ms": float(snapshot.get("rtt_ms", 0.0) or 0.0),
        "jitter_ms": float(snapshot.get("jitter_ms", 0.0) or 0.0),
        "loss_rate": float(snapshot.get("loss_rate", 1.0) or 0.0),
        "consecutive_failures": int(network.get("consecutive_failures", 0) or 0),
        "latest_observed_at_ms": int(latest.get("observed_at_ms", 0) or 0),
        "latest_success": bool(latest.get("success", False)),
        "measurement": str(network.get("measurement", "")),
    }


def _wait_network_state(
    edge_urls: Sequence[str],
    available: bool,
    switched_at_ms: int,
    required_fresh_samples: int,
    timeout_seconds: float,
    request_timeout_seconds: float,
    poll_seconds: float,
) -> List[Dict[str, Any]]:
    deadline = time.monotonic() + float(timeout_seconds)
    last: List[Dict[str, Any]] = []
    observed_by_edge = {str(endpoint): set() for endpoint in edge_urls}
    while time.monotonic() < deadline:
        snapshots = []
        for endpoint in edge_urls:
            try:
                health = _get_json(endpoint, HEALTH_PATH, request_timeout_seconds)
                snapshots.append(
                    {"edge_url": endpoint, **_network_snapshot(health)}
                )
            except (HTTPError, URLError, TimeoutError, socket.timeout, ValueError):
                snapshots.append(
                    {
                        "edge_url": endpoint,
                        "available": not available,
                        "latest_observed_at_ms": 0,
                        "consecutive_failures": 0,
                        "measurement": "query_failed",
                    }
                )
        last = snapshots
        for row in snapshots:
            observed_at_ms = int(row.get("latest_observed_at_ms", 0))
            if observed_at_ms >= int(switched_at_ms):
                observed_by_edge[str(row["edge_url"])].add(observed_at_ms)
            row["fresh_probe_samples_observed"] = len(
                observed_by_edge[str(row["edge_url"])]
            )
        fresh = all(
            int(row.get("latest_observed_at_ms", 0)) >= int(switched_at_ms)
            for row in snapshots
        )
        window_replaced = all(
            len(observed_by_edge[str(endpoint)]) >= int(required_fresh_samples)
            for endpoint in edge_urls
        )
        matches = all(bool(row.get("available")) is available for row in snapshots)
        real_probe = all(
            row.get("measurement") == "application_http_health_probe"
            for row in snapshots
        )
        failures_ready = available or all(
            int(row.get("consecutive_failures", 0)) >= 2 for row in snapshots
        )
        if fresh and window_replaced and matches and real_probe and failures_ready:
            return snapshots
        time.sleep(float(poll_seconds))
    raise TimeoutError(
        "edge network probes did not replace the measurement window and converge "
        "to available={}: {}".format(available, last)
    )


def _get_review(
    edge_url: str,
    event_id: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> Dict[str, Any]:
    deadline = time.monotonic() + float(timeout_seconds)
    last_error = "review not found"
    while time.monotonic() < deadline:
        try:
            return _get_json(
                edge_url,
                REVIEW_PATH.format(quote(event_id, safe="")),
                min(timeout_seconds, 2.0),
            )
        except (
            HTTPError,
            URLError,
            TimeoutError,
            socket.timeout,
            ValueError,
        ) as exc:
            last_error = "{}: {}".format(type(exc).__name__, exc)
            time.sleep(float(poll_seconds))
    raise TimeoutError("review {} unavailable: {}".format(event_id, last_error))


def _wait_outboxes_idle(
    edge_urls: Sequence[str],
    timeout_seconds: float,
    request_timeout_seconds: float,
    poll_seconds: float,
) -> List[Dict[str, Any]]:
    deadline = time.monotonic() + float(timeout_seconds)
    last: List[Dict[str, Any]] = []
    while time.monotonic() < deadline:
        snapshots = []
        try:
            snapshots = [
                _edge_delivery_snapshot(endpoint, request_timeout_seconds)
                for endpoint in edge_urls
            ]
        except (HTTPError, URLError, TimeoutError, socket.timeout, ValueError):
            time.sleep(float(poll_seconds))
            continue
        last = snapshots
        if all(
            int(value.get("active", -1)) == 0
            and int(
                dict(value.get("reconciliation", {})).get("active", -1)
            )
            == 0
            and int(dict(value.get("durable_handoff", {})).get("pending", -1))
            == 0
            and int(
                dict(value.get("durable_handoff", {})).get(
                    "durable_pending_count", -1
                )
            )
            == 0
            for value in snapshots
        ):
            return snapshots
        time.sleep(float(poll_seconds))
    raise TimeoutError("outboxes did not drain before profile switch: {}".format(last))


def _edge_delivery_snapshot(
    edge_url: str, request_timeout_seconds: float
) -> Dict[str, Any]:
    """Join Outbox and durable-handoff state from the same edge service."""
    outbox = _get_json(edge_url, OUTBOX_PATH, request_timeout_seconds)
    health = _get_json(edge_url, HEALTH_PATH, request_timeout_seconds)
    runtime = health.get("runtime", {})
    runtime = dict(runtime) if isinstance(runtime, Mapping) else {}
    handoff = runtime.get("durable_handoff", {})
    handoff = dict(handoff) if isinstance(handoff, Mapping) else {}
    return {**dict(outbox), "durable_handoff": handoff}


def _wait_authoritative_reviews(
    edge_url: str,
    event_ids: Sequence[str],
    timeout_seconds: float,
    poll_seconds: float,
    request_timeout_seconds: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """Poll through partial finals until every requested final is authoritative."""
    pending = {str(event_id) for event_id in event_ids}
    completed: Dict[str, Dict[str, Any]] = {}
    observed: Dict[str, float] = {}
    deadline = time.monotonic() + float(timeout_seconds)
    while pending and time.monotonic() < deadline:
        for event_id in list(pending):
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0.0:
                break
            try:
                review = _get_json(
                    edge_url,
                    REVIEW_PATH.format(quote(event_id, safe="")),
                    min(float(request_timeout_seconds), remaining_seconds),
                )
            except (HTTPError, URLError, TimeoutError, socket.timeout, ValueError):
                continue
            if _authoritative_review(review):
                completed[event_id] = review
                observed[event_id] = time.perf_counter()
                pending.remove(event_id)
        if pending:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds > 0.0:
                time.sleep(min(float(poll_seconds), remaining_seconds))
    return completed, observed


def _wait_all_profile_reviews_authoritative(
    events: Sequence[Mapping[str, Any]],
    timeout_seconds: float,
    poll_seconds: float,
    request_timeout_seconds: float,
) -> Dict[str, Dict[str, Any]]:
    """Drain every online summary before changing the injected fault profile."""
    by_edge: Dict[str, List[str]] = {}
    for event in events:
        by_edge.setdefault(str(event["edge_url"]), []).append(str(event["event_id"]))
    expected_ids = {
        str(event_id) for event_ids in by_edge.values() for event_id in event_ids
    }
    completed: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(by_edge)) as executor:
        futures = [
            executor.submit(
                _wait_authoritative_reviews,
                edge_url,
                event_ids,
                timeout_seconds,
                poll_seconds,
                request_timeout_seconds,
            )
            for edge_url, event_ids in by_edge.items()
        ]
        for future in as_completed(futures):
            reviews, _ = future.result()
            completed.update(reviews)
    missing = expected_ids - set(completed)
    if missing:
        raise TimeoutError(
            "online profile summaries did not reach authoritative final: {}".format(
                sorted(missing)
            )
        )
    return completed


def _review_group_id(review: Mapping[str, Any]) -> str:
    final = review.get("final_decision", {})
    final = dict(final) if isinstance(final, Mapping) else {}
    metadata = final.get("metadata", {})
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    aggregation = metadata.get("aggregation", {})
    aggregation = (
        dict(aggregation) if isinstance(aggregation, Mapping) else {}
    )
    return str(aggregation.get("group_id", ""))


def _authorization_safe(event: Mapping[str, Any]) -> bool:
    deferred_raw = event.get("deferred_action_types", [])
    immediate_raw = event.get("immediate_action_types", [])
    if not isinstance(deferred_raw, list) or not isinstance(immediate_raw, list):
        return False
    deferred = [str(value).strip() for value in deferred_raw]
    immediate = [str(value).strip() for value in immediate_raw]
    if (
        any(not value for value in deferred + immediate)
        or len(set(deferred)) != len(deferred)
        or len(set(immediate)) != len(immediate)
        or set(deferred).intersection(immediate)
    ):
        return False
    if deferred and bool(event.get("cloud_confirmed")):
        return False
    # The runtime constructs ``immediate_action_types`` from actions that are
    # authorized at the current phase. ``all_actions_authorized`` becomes false
    # when a different action is deliberately deferred; that must not invalidate
    # the safe immediate subset. With no deferred action, the aggregate marker
    # must still prove complete authorization.
    if not deferred and immediate and event.get("local_actions_authorized") is not True:
        return False
    return True


def evaluate_business_event(
    profile: str,
    event: Mapping[str, Any],
    review: Optional[Mapping[str, Any]],
    authoritative_observed_ms: Optional[float],
    deadline_ms: float,
) -> Dict[str, Any]:
    """Apply the preregistered business endpoint without fabricating a final."""
    reasons: List[str] = []
    response_ms = event.get("input_to_response_ms")
    try:
        response_ms = float(response_ms)
    except (TypeError, ValueError):
        response_ms = None
        reasons.append("missing_real_edge_response_latency")
    if str(event.get("status", "")) not in {"provisional", "final"}:
        reasons.append("invalid_edge_response_status")
    if not _authorization_safe(event):
        reasons.append("unsafe_action_authorization")

    review_value = dict(review) if isinstance(review, Mapping) else {}
    if profile == "outage":
        endpoint = "offline_local_autonomy"
        completion_ms = response_ms
        if str(event.get("route", "")) != "local_autonomy":
            reasons.append("route_is_not_local_autonomy")
        if str(event.get("policy_route", "")) != "local_autonomy":
            reasons.append("policy_route_is_not_local_autonomy")
        if event.get("local_autonomy") is not True:
            reasons.append("local_autonomy_marker_missing")
        if event.get("cloud_confirmed") is True:
            reasons.append("offline_response_claimed_cloud_confirmation")
        if event.get("policy_waits_for_cloud") is True:
            reasons.append("offline_business_waited_for_cloud")
        if event.get("summary_delivery_required") is not True:
            reasons.append("summary_delivery_was_not_required")
        elif str(event.get("summary_persistence_stage", "")) not in {
            "handoff_durable",
            "outbox_durable",
        }:
            reasons.append("summary_was_not_durably_queued")
        if str(review_value.get("requested_route", "")) != "local_autonomy":
            reasons.append("lifecycle_route_is_not_local_autonomy")
        if str(review_value.get("state", "")) not in {"queued", "inflight"}:
            reasons.append("offline_review_not_pending_for_replay")
    else:
        policy_route = str(event.get("policy_route", ""))
        if policy_route not in {
            "edge_only",
            "local_autonomy",
            "cloud_async",
            "cloud_sync",
        }:
            reasons.append("unknown_policy_route")
        requires_final = policy_route == "cloud_sync"
        if requires_final:
            endpoint = "authoritative_final"
            completion_ms = authoritative_observed_ms
            if not _authoritative_review(review_value):
                reasons.append("authoritative_final_missing")
        else:
            endpoint = "local_response"
            completion_ms = response_ms

    if completion_ms is None:
        reasons.append("business_endpoint_not_observed")
    elif float(completion_ms) > float(deadline_ms):
        reasons.append("business_deadline_exceeded")
    return {
        "success": not reasons,
        "business_endpoint": endpoint,
        "completion_ms": (
            round(float(completion_ms), 6) if completion_ms is not None else None
        ),
        "deadline_ms": float(deadline_ms),
        "reasons": reasons,
    }


def evaluate_recovery(
    outboxes: Sequence[Mapping[str, Any]],
    reviews: Mapping[str, Mapping[str, Any]],
    aggregations: Sequence[Mapping[str, Any]],
    expected_event_ids: Sequence[str],
    expected_sample_count: int,
    expected_sample_by_event: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """Fail closed unless automatic replay restores one complete 4/4 group/sample."""
    reasons: List[str] = []
    expected_ids = [str(value) for value in expected_event_ids]
    if len(expected_ids) != len(set(expected_ids)):
        reasons.append("duplicate_expected_event_ids")
    if len(reviews) != len(expected_ids) or set(reviews) != set(expected_ids):
        reasons.append("recovered_review_set_mismatch")
    if not all(_authoritative_review(value) for value in reviews.values()):
        reasons.append("not_all_reviews_are_authoritative")
    review_groups: Dict[str, List[str]] = {}
    for event_id, review in reviews.items():
        group_id = _review_group_id(review)
        if not group_id:
            reasons.append("review_missing_aggregation_group")
            continue
        review_groups.setdefault(group_id, []).append(str(event_id))
    if len(review_groups) != int(expected_sample_count):
        reasons.append("review_aggregation_group_count_mismatch")
    if expected_sample_by_event is not None:
        for event_ids in review_groups.values():
            sample_ids = {
                int(expected_sample_by_event[event_id])
                for event_id in event_ids
                if event_id in expected_sample_by_event
            }
            if len(event_ids) != 4 or len(sample_ids) != 1:
                reasons.append("cross_sample_or_incomplete_review_group")
    for index, snapshot in enumerate(outboxes):
        reconciliation = snapshot.get("reconciliation", {})
        reconciliation = (
            dict(reconciliation) if isinstance(reconciliation, Mapping) else {}
        )
        if int(snapshot.get("active", -1)) != 0:
            reasons.append("edge_{}_outbox_not_drained".format(index))
        if int(reconciliation.get("active", -1)) != 0:
            reasons.append("edge_{}_reconciliation_not_drained".format(index))
        handoff = snapshot.get("durable_handoff", {})
        handoff = dict(handoff) if isinstance(handoff, Mapping) else {}
        if not handoff:
            reasons.append("edge_{}_durable_handoff_state_missing".format(index))
        elif int(handoff.get("pending", -1)) != 0 or int(
            handoff.get("durable_pending_count", -1)
        ) != 0:
            reasons.append("edge_{}_durable_handoff_not_drained".format(index))

    if len(aggregations) != int(expected_sample_count):
        reasons.append("aggregation_group_count_mismatch")
    group_ids = [str(value.get("group_id", "")) for value in aggregations]
    if len(group_ids) != len(set(group_ids)):
        reasons.append("duplicate_aggregation_group")
    if set(group_ids) != set(review_groups):
        reasons.append("cloud_and_review_group_sets_differ")
    for aggregation in aggregations:
        if str(aggregation.get("state", "")) != "completed":
            reasons.append("aggregation_not_completed")
        if str(aggregation.get("completion_reason", "")) != "all_expected_members":
            reasons.append("aggregation_missing_members")
        if aggregation.get("evidence_complete") is not True:
            reasons.append("aggregation_evidence_incomplete")
        if str(aggregation.get("finality", "")) != "final":
            reasons.append("aggregation_not_final")
        if aggregation.get("global_confirmation") is not True:
            reasons.append("aggregation_not_globally_confirmed")
        if set(aggregation.get("expected_members", [])) != EXPECTED_MEMBERS:
            reasons.append("aggregation_expected_members_mismatch")
        if set(aggregation.get("received_members", [])) != EXPECTED_MEMBERS:
            reasons.append("aggregation_received_members_mismatch")
        if aggregation.get("missing_members") not in ([], None):
            reasons.append("aggregation_reports_missing_members")
    reasons = list(dict.fromkeys(reasons))
    return {
        "passed": not reasons,
        # Retain the raw recovery observations.  Derived counters and
        # ``passed`` remain convenient diagnostics, but the unified evaluator
        # recomputes recovery from these records and never trusts them alone.
        "outbox_snapshots": [dict(value) for value in outboxes],
        "review_records": {
            str(event_id): dict(review) for event_id, review in reviews.items()
        },
        "aggregation_records": [dict(value) for value in aggregations],
        "expected_event_ids": expected_ids,
        "expected_sample_by_event": {
            str(event_id): int(sample_id)
            for event_id, sample_id in dict(expected_sample_by_event or {}).items()
        },
        "authoritative_review_count": sum(
            _authoritative_review(value) for value in reviews.values()
        ),
        "expected_event_count": len(expected_ids),
        "aggregation_count": len(aggregations),
        "expected_sample_count": int(expected_sample_count),
        "review_group_count": len(review_groups),
        "reasons": reasons,
    }


def _wait_profile_finals(
    events: Sequence[Mapping[str, Any]],
    start_at_ns: int,
    deadline_ms: float,
    poll_seconds: float,
    request_timeout_seconds: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    required: Dict[str, List[str]] = {}
    for event in events:
        if str(event.get("policy_route", "")) == "cloud_sync":
            required.setdefault(str(event["edge_url"]), []).append(
                str(event["event_id"])
            )
    if not required:
        return {}, {}
    remaining = max(
        0.001,
        (start_at_ns / 1_000_000_000.0 + deadline_ms / 1000.0)
        - time.monotonic(),
    )
    reviews: Dict[str, Dict[str, Any]] = {}
    observed: Dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=len(required)) as executor:
        futures = [
            executor.submit(
                _wait_authoritative_reviews,
                endpoint,
                event_ids,
                remaining,
                poll_seconds,
                request_timeout_seconds,
            )
            for endpoint, event_ids in required.items()
        ]
        for future in as_completed(futures):
            edge_reviews, edge_observed = future.result()
            reviews.update(edge_reviews)
            observed.update(edge_observed)
    return reviews, observed


def _run_profile(
    profile: str,
    sample_ids: Sequence[int],
    experiment_id: str,
    command_queues: Sequence[Any],
    result_queue: Any,
    worker_timeout_seconds: float,
    aggregation_timeout_ms: int,
    dispatch_lead_ms: float,
    business_deadline_ms: float,
    poll_seconds: float,
    request_timeout_seconds: float,
    progress: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    for sample_id in sample_ids:
        start_at_ns = time.monotonic_ns() + int(dispatch_lead_ms * 1_000_000.0)
        command = {
            "kind": "sample",
            "sample_id": int(sample_id),
            "start_at_ns": start_at_ns,
            "start_at_epoch_ms": time.time() * 1000.0 + dispatch_lead_ms,
            "experiment_id": experiment_id,
            "aggregation_timeout_ms": int(aggregation_timeout_ms),
        }
        for queue in command_queues:
            queue.put(command)
            if progress is not None:
                progress["attempted_events"] = int(
                    progress.get("attempted_events", 0)
                ) + 1
        events = _collect_messages(
            result_queue,
            "sample",
            len(command_queues),
            worker_timeout_seconds,
        )
        events.sort(key=lambda value: int(value["partition_id"]))
        if [int(value["partition_id"]) for value in events] != list(range(4)):
            raise RuntimeError("profile sample returned missing/duplicate partitions")

        reviews: Dict[str, Dict[str, Any]] = {}
        observed: Dict[str, float] = {}
        if profile == "outage":
            for event in events:
                # The fsynced handoff journal is the offline acceptance
                # boundary. Its SQLite lifecycle row is materialized by a
                # background worker and is not required to exist before the
                # local business response returns.
                reviews[str(event["event_id"])] = {
                    "state": str(event.get("review_state", "")),
                    "requested_route": str(
                        event.get("review_requested_route", "")
                    ),
                    "persistence_stage": str(
                        event.get("summary_persistence_stage", "")
                    ),
                }
        else:
            reviews, observed = _wait_profile_finals(
                events,
                start_at_ns,
                business_deadline_ms,
                poll_seconds,
                request_timeout_seconds,
            )

        sample_rows = []
        for event in events:
            event_id = str(event["event_id"])
            observed_ms = None
            if event_id in observed:
                observed_ms = (
                    float(observed[event_id])
                    - start_at_ns / 1_000_000_000.0
                ) * 1000.0
            decision = evaluate_business_event(
                profile,
                event,
                reviews.get(event_id),
                observed_ms,
                business_deadline_ms,
            )
            # Preserve the raw observations used to derive ``success``.  The
            # unified competition gate ignores that producer boolean and
            # independently recomputes the business endpoint from this record.
            response_observation = {
                name: event.get(name)
                for name in (
                    "status",
                    "route",
                    "policy_route",
                    "policy_waits_for_cloud",
                    "input_to_response_ms",
                    "local_actions_authorized",
                    "action_authorization_present",
                    "all_actions_authorized",
                    "immediate_action_types",
                    "deferred_action_types",
                    "cloud_confirmed",
                    "local_autonomy",
                    "summary_delivery_required",
                    "summary_persistence_stage",
                    "review_state",
                    "review_requested_route",
                )
            }
            row = {
                **dict(event),
                **decision,
                "profile": profile,
                "business_observation": {
                    "response": response_observation,
                    "review": dict(reviews.get(event_id, {})),
                    "authoritative_observed_ms": (
                        round(float(observed_ms), 6)
                        if observed_ms is not None
                        else None
                    ),
                    "deadline_ms": float(business_deadline_ms),
                },
            }
            rows.append(row)
            sample_rows.append(row)
        samples.append(
            {
                "sample_id": int(sample_id),
                "business_attempts": len(sample_rows),
                "business_successes": sum(row["success"] for row in sample_rows),
                "event_ids": [str(row["event_id"]) for row in sample_rows],
            }
        )
    return {
        "profile_id": profile,
        "fault_parameters": dict(PROFILES[profile]),
        "business_attempts": len(rows),
        "business_successes": sum(row["success"] for row in rows),
        "retention_rate": (
            sum(row["success"] for row in rows) / float(len(rows)) if rows else 0.0
        ),
        "samples": samples,
        "events": rows,
    }


def _wait_recovery(
    edge_urls: Sequence[str],
    outage_events: Sequence[Mapping[str, Any]],
    cloud_url: str,
    timeout_seconds: float,
    poll_seconds: float,
    request_timeout_seconds: float,
) -> Dict[str, Any]:
    event_by_id = {
        str(event["event_id"]): dict(event) for event in outage_events
    }
    deadline = time.monotonic() + float(timeout_seconds)
    last_outboxes: List[Dict[str, Any]] = []
    last_reviews: Dict[str, Dict[str, Any]] = {}
    while time.monotonic() < deadline:
        outboxes = []
        reviews: Dict[str, Dict[str, Any]] = {}
        query_failed = False
        for endpoint in edge_urls:
            try:
                outboxes.append(
                    _edge_delivery_snapshot(endpoint, request_timeout_seconds)
                )
            except (HTTPError, URLError, TimeoutError, socket.timeout, ValueError):
                query_failed = True
                break
        if not query_failed:
            for event_id, event in event_by_id.items():
                try:
                    reviews[event_id] = _get_json(
                        str(event["edge_url"]),
                        REVIEW_PATH.format(quote(event_id, safe="")),
                        request_timeout_seconds,
                    )
                except (
                    HTTPError,
                    URLError,
                    TimeoutError,
                    socket.timeout,
                    ValueError,
                ):
                    query_failed = True
                    break
        last_outboxes = outboxes
        last_reviews = reviews
        outbox_idle = bool(
            len(outboxes) == len(edge_urls)
            and all(int(value.get("active", -1)) == 0 for value in outboxes)
            and all(
                int(dict(value.get("reconciliation", {})).get("active", -1)) == 0
                for value in outboxes
            )
        )
        if (
            not query_failed
            and outbox_idle
            and len(reviews) == len(event_by_id)
            and all(_authoritative_review(value) for value in reviews.values())
        ):
            break
        time.sleep(float(poll_seconds))

    group_ids = sorted(
        {_review_group_id(review) for review in last_reviews.values()} - {""}
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
        except (
            HTTPError,
            URLError,
            TimeoutError,
            socket.timeout,
            ValueError,
        ) as exc:
            aggregations.append(
                {
                    "group_id": group_id,
                    "state": "query_failed",
                    "error": "{}: {}".format(type(exc).__name__, exc),
                }
            )
    return evaluate_recovery(
        last_outboxes,
        last_reviews,
        aggregations,
        list(event_by_id),
        len(event_by_id) // 4,
        {
            event_id: int(event["sample_id"])
            for event_id, event in event_by_id.items()
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="真实 normal/mild/severe/outage/recovery 四逻辑边弱网验收"
    )
    parser.add_argument("--project-root", default=str(SDK_ROOT))
    parser.add_argument(
        "--experiment-plan",
        required=True,
        help="实验开始前冻结的弱网计划 JSON；正式运行不接受临时生成样本",
    )
    parser.add_argument("--cloud-url", default="http://127.0.0.1:18100")
    parser.add_argument(
        "--manifest",
        default=str(
            SCENE_ROOT / "runtime" / "pems08_metis4_partitions" / "manifest.json"
        ),
    )
    parser.add_argument(
        "--source",
        default=str(
            SCENE_ROOT
            / "assets"
            / "downloads"
            / "PEMS08_r1_d0_w0_astcgn_multitask.npz"
        ),
    )
    parser.add_argument(
        "--edge-service-config-template",
        default=str(SCENE_ROOT / "deployment" / "full" / "edge_service.json"),
    )
    parser.add_argument("--edge-port-base", type=int, default=19301)
    parser.add_argument("--samples-per-profile", type=int, default=None)
    parser.add_argument("--sample-start", type=int, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--business-deadline-ms", type=float, default=None)
    parser.add_argument("--aggregation-timeout-ms", type=int, default=None)
    parser.add_argument("--dispatch-lead-ms", type=float, default=None)
    parser.add_argument("--worker-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--profile-transition-seconds", type=float, default=15.0)
    parser.add_argument("--profile-drain-seconds", type=float, default=30.0)
    parser.add_argument("--recovery-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.05)
    parser.add_argument("--proxy-drop-hold-seconds", type=float, default=0.55)
    parser.add_argument("--proxy-backend-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hardware-id", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--skip-sha256", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-auto-prepare-partitions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = Path(args.experiment_plan).resolve()
    plan, plan_sha256 = _load_experiment_plan(plan_path)
    plan_loaded_at = _utc_now()
    created_at = datetime.fromisoformat(
        str(plan["created_at"]).replace("Z", "+00:00")
    )
    loaded_at = datetime.fromisoformat(plan_loaded_at.replace("Z", "+00:00"))
    if created_at > loaded_at:
        raise ValueError("experiment plan created_at cannot be after plan load")

    # The plan is authoritative.  Legacy CLI flags may only repeat it; they
    # cannot silently alter a formal run after the samples were selected.
    scalar_bindings = {
        "split": str(plan["split"]),
        "top_k": int(plan["top_k"]),
        "business_deadline_ms": float(plan["business_deadline_ms"]),
        "aggregation_timeout_ms": int(plan["aggregation_timeout_ms"]),
        "dispatch_lead_ms": float(plan["dispatch_lead_ms"]),
        "seed": int(plan["seed"]),
    }
    for name, planned in scalar_bindings.items():
        supplied = getattr(args, name)
        if supplied is not None and supplied != planned:
            raise ValueError(
                "--{}={} differs from preregistered {}".format(
                    name.replace("_", "-"), supplied, planned
                )
            )
        setattr(args, name, planned)
    plan_samples = {
        name: [int(value) for value in plan["sample_ids_by_profile"][name]]
        for name in PROFILE_ORDER
    }
    sample_counts = {len(values) for values in plan_samples.values()}
    if args.samples_per_profile is not None and (
        len(sample_counts) != 1 or args.samples_per_profile not in sample_counts
    ):
        raise ValueError("--samples-per-profile differs from preregistered samples")
    flattened_samples = [
        sample_id for name in PROFILE_ORDER for sample_id in plan_samples[name]
    ]
    if args.sample_start is not None:
        expected_contiguous = list(
            range(args.sample_start, args.sample_start + len(flattened_samples))
        )
        if flattened_samples != expected_contiguous:
            raise ValueError("--sample-start differs from preregistered sample IDs")

    run_id = "weaknet-{}-{}".format(
        time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8]
    )
    output = (
        Path(args.output).resolve()
        if args.output
        else (SCENE_ROOT / "evidence" / "{}.json".format(run_id)).resolve()
    )
    fault_log = output.with_suffix(".proxy.jsonl")
    if output.exists() or fault_log.exists():
        raise FileExistsError(
            "formal weak-network evidence paths must not pre-exist: {}, {}".format(
                output, fault_log
            )
        )
    evidence: Dict[str, Any] = {
        "measurement_status": "not_measured",
        "status": "not_measured",
        "run_id": run_id,
        "generated_at": plan_loaded_at,
        "profiles": [],
        "errors": [],
    }
    progress = {"attempted_events": 0}
    proxy_server: Optional[ThreadingHTTPServer] = None
    proxy_thread: Optional[threading.Thread] = None
    edge_services: List[Dict[str, Any]] = []
    worker_processes: List[Any] = []
    command_queues: List[Any] = []
    try:
        if args.business_deadline_ms <= 0.0:
            raise ValueError("business deadline must be positive")
        project_root = Path(args.project_root).resolve()
        scene_root = project_root / "scenes" / "freeway_traffic"
        edge_service_template_path = Path(
            args.edge_service_config_template
        ).resolve()
        edge_service_template = json.loads(
            edge_service_template_path.read_text(encoding="utf-8")
        )
        network_probe = edge_service_template.get("network_probe", {})
        network_probe = (
            dict(network_probe) if isinstance(network_probe, Mapping) else {}
        )
        network_window_samples = int(network_probe.get("window_size", 0))
        if network_window_samples <= 0:
            raise ValueError("edge network_probe.window_size must be positive")
        # One probe may already be in flight when the proxy profile changes.
        # Requiring window_size + 1 fresh completions guarantees that this
        # boundary probe has also rolled out of the monitor's fixed window.
        network_transition_samples = network_window_samples + 1
        commit, dirty = _git_identity(project_root)
        if dirty and not args.allow_dirty:
            raise RuntimeError(
                "working tree is dirty; use --allow-dirty only for an explicitly "
                "labelled development measurement"
            )
        if commit != str(plan["git_commit"]):
            raise RuntimeError("current Git commit differs from experiment plan")
        manifest_path = Path(args.manifest).resolve()
        if not manifest_path.is_file():
            if args.no_auto_prepare_partitions:
                raise FileNotFoundError("partition manifest not found")
            prepare_partition_data(
                Path(args.source),
                scene_root / "assets" / "models" / "current_state_perception_v1.json",
                scene_root / "assets" / "models" / "traffic_region_topology_metis4.json",
                manifest_path.parent,
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sorted(int(row["partition_id"]) for row in manifest["partitions"]) != list(
            range(4)
        ):
            raise ValueError("weak-network run requires four METIS partitions")
        _get_json(args.cloud_url, HEALTH_PATH, args.request_timeout_seconds)
        evidence_pull = edge_service_template.get("evidence_pull", {})
        if isinstance(evidence_pull, Mapping) and evidence_pull.get("enabled") is True:
            if args.edge_port_base < 1 or args.edge_port_base + 3 > 65535:
                raise ValueError("edge port range is invalid")
            _validate_cloud_evidence_pull_allowlist(
                args.cloud_url,
                [
                    "http://127.0.0.1:{}".format(
                        args.edge_port_base + partition_id
                    )
                    for partition_id in range(4)
                ],
                timeout_seconds=min(
                    2.0, max(0.1, args.request_timeout_seconds)
                ),
            )

        dataset_id = "pems08_metis4:{}".format(_sha256(manifest_path))
        model_ids = _model_ids(scene_root)
        hardware_id = _hardware_id(args.hardware_id)
        if dataset_id != str(plan["dataset_id"]):
            raise RuntimeError("partition dataset hash differs from experiment plan")
        if _canonical(model_ids) != _canonical(plan["model_ids"]):
            raise RuntimeError("model/config asset hashes differ from experiment plan")
        if hardware_id != str(plan["hardware_id"]):
            raise RuntimeError("hardware_id differs from experiment plan")
        provenance = {
            "git_commit": commit,
            "dataset_id": dataset_id,
            "model_ids": model_ids,
            "hardware_id": hardware_id,
            "metric_semantics": "basic_business_function_retention",
            "run_id": run_id,
            "generated_at": evidence["generated_at"],
            "run_started_at": plan_loaded_at,
        }
        evidence.update(
            {
                "evidence_schema_version": "2.0",
                "provenance": provenance,
                "experiment_plan": {
                    "plan_id": str(plan["plan_id"]),
                    "path": str(plan_path),
                    "sha256": plan_sha256,
                    "created_at": str(plan["created_at"]),
                    "loaded_at": plan_loaded_at,
                },
                "formal_eligible": not bool(dirty),
                "configuration": {
                    "profile_order": list(PROFILE_ORDER) + ["recovery"],
                    "fault_profiles": {
                        name: dict(PROFILES[name]) for name in PROFILE_ORDER
                    },
                    "sample_ids_by_profile": plan_samples,
                    "business_deadline_ms": args.business_deadline_ms,
                    "aggregation_timeout_ms": args.aggregation_timeout_ms,
                    "dispatch_lead_ms": args.dispatch_lead_ms,
                    "split": args.split,
                    "top_k": args.top_k,
                    "seed": args.seed,
                    "cloud_url": args.cloud_url,
                    "dirty_worktree_allowed": bool(args.allow_dirty),
                    "dirty_paths": dirty,
                    "perception_runtime": "current_state_without_astgcn_or_torch",
                    "manual_outbox_flush_used": False,
                    "network_window_samples": network_window_samples,
                    "network_transition_fresh_samples": network_transition_samples,
                },
                "measurement_contract": {
                    "attempt_unit": "one real edge event",
                    "normal_and_weak_online": (
                        "edge_only/cloud_async local response completes basic work; "
                        "cloud_sync requires an observed authoritative final"
                    ),
                    "outage": (
                        "safe local_autonomy response and durable summary queue "
                        "complete basic business; no final is required while offline"
                    ),
                    "recovery": (
                        "background replay only; all outage events must become one "
                        "authoritative 4/4 final per sample with empty active queues"
                    ),
                },
            }
        )

        state = FaultState("normal", args.seed, fault_log)
        control_token = uuid.uuid4().hex
        handler = build_handler(
            state,
            args.cloud_url,
            args.proxy_drop_hold_seconds,
            args.proxy_backend_timeout_seconds,
            control_token,
        )
        proxy_server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        proxy_url = "http://127.0.0.1:{}".format(proxy_server.server_port)
        proxy_thread = threading.Thread(
            target=proxy_server.serve_forever,
            name="weak-network-fault-proxy",
            daemon=True,
        )
        proxy_thread.start()
        edge_services = _launch_isolated_edge_services(
            project_root,
            scene_root,
            run_id,
            edge_service_template_path,
            proxy_url,
            args.edge_port_base,
            args.profile_transition_seconds,
        )
        edge_urls = [str(value["endpoint"]) for value in edge_services]

        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        command_queues = [context.Queue() for _ in range(4)]
        base = {
            "manifest_path": str(manifest_path),
            "rule_config_path": str(
                scene_root / "assets" / "models" / "current_state_perception_v1.json"
            ),
            "topology_path": str(
                scene_root
                / "assets"
                / "models"
                / "traffic_region_topology_metis4.json"
            ),
            "split": args.split,
            "top_k": args.top_k,
            "verify_sha256": not args.skip_sha256,
            "request_timeout_seconds": args.request_timeout_seconds,
        }
        for partition_id, command_queue in enumerate(command_queues):
            process = context.Process(
                target=_edge_worker,
                args=(
                    {
                        **base,
                        "partition_id": partition_id,
                        "edge_url": edge_urls[partition_id],
                    },
                    command_queue,
                    result_queue,
                ),
                name="weaknet-edge-node-{}".format(partition_id),
            )
            process.start()
            worker_processes.append(process)
        ready = _collect_messages(
            result_queue, "ready", 4, args.worker_timeout_seconds
        )
        ready.sort(key=lambda value: int(value["partition_id"]))
        worker_pids = [int(value["pid"]) for value in ready]
        if len(set(worker_pids)) != 4:
            raise RuntimeError("each METIS edge must run in a distinct process")
        assigned_nodes = sorted(
            int(node) for value in ready for node in value["managed_node_ids"]
        )
        if assigned_nodes != list(range(170)):
            raise RuntimeError("four worker assignments must cover 170 nodes once")
        sample_count = min(int(value["sample_count"]) for value in ready)
        out_of_range = [
            sample_id
            for sample_id in flattened_samples
            if sample_id < 0 or sample_id >= sample_count
        ]
        if out_of_range:
            raise ValueError(
                "preregistered sample IDs exceed split size {}: {}".format(
                    sample_count, out_of_range
                )
            )
        evidence["architecture"] = {
            "logical_edge_count": 4,
            "edge_service_pids": [int(value["process"].pid) for value in edge_services],
            "worker_pids": worker_pids,
            "managed_nodes_by_edge": {
                str(value["edge_id"]): list(value["managed_node_ids"])
                for value in ready
            },
            "independent_edge_endpoints": edge_urls,
            "independent_edge_storage": [str(value["state_root"]) for value in edge_services],
            "fault_proxy_url": proxy_url,
            "fault_proxy_log": str(fault_log),
        }

        profile_results: Dict[str, Dict[str, Any]] = {}
        profile_network: Dict[str, List[Dict[str, Any]]] = {}
        for profile in PROFILE_ORDER:
            switch = _set_proxy_profile(
                proxy_url, profile, control_token, args.request_timeout_seconds
            )
            switched_at_ms = int(time.time() * 1000)
            profile_network[profile] = _wait_network_state(
                edge_urls,
                profile != "outage",
                switched_at_ms,
                network_transition_samples,
                args.profile_transition_seconds,
                args.request_timeout_seconds,
                args.poll_interval_seconds,
            )
            outbox_before_profile = [
                _edge_delivery_snapshot(endpoint, args.request_timeout_seconds)
                for endpoint in edge_urls
            ]
            sample_ids = list(plan_samples[profile])
            result = _run_profile(
                profile,
                sample_ids,
                "{}-{}".format(run_id, profile),
                command_queues,
                result_queue,
                args.worker_timeout_seconds,
                args.aggregation_timeout_ms,
                args.dispatch_lead_ms,
                args.business_deadline_ms,
                args.poll_interval_seconds,
                args.request_timeout_seconds,
                progress,
            )
            result["proxy_state_before_profile"] = switch
            result["edge_network_snapshots"] = profile_network[profile]
            result["outbox_before_profile"] = outbox_before_profile
            if profile != "outage":
                # Do not let an online profile's background summary traffic
                # spill into the next fault profile and corrupt attribution.
                result["authoritative_reviews_after_profile"] = len(
                    _wait_all_profile_reviews_authoritative(
                        result["events"],
                        args.profile_drain_seconds,
                        args.poll_interval_seconds,
                        args.request_timeout_seconds,
                    )
                )
                result["outbox_after_profile"] = _wait_outboxes_idle(
                    edge_urls,
                    args.profile_drain_seconds,
                    args.request_timeout_seconds,
                    args.poll_interval_seconds,
                )
            proxy_after_profile = state.snapshot()
            if int(proxy_after_profile.get("profile_request_count", 0)) <= 0:
                raise RuntimeError(
                    "profile {} did not carry any real proxy request".format(profile)
                )
            request_classes = dict(
                proxy_after_profile.get("profile_request_class_counts", {})
            )
            result["proxy_summary_data_plane_requests"] = int(
                request_classes.get("summary_data_plane", 0)
            )
            if profile != "outage" and int(
                request_classes.get("summary_data_plane", 0)
            ) <= 0:
                raise RuntimeError(
                    "online profile {} carried no summary data-plane request".format(
                        profile
                    )
                )
            result["proxy_state_after_profile"] = proxy_after_profile
            profile_results[profile] = result

        outage_events = profile_results["outage"]["events"]
        backlog_before_recovery = [
            _edge_delivery_snapshot(endpoint, args.request_timeout_seconds)
            for endpoint in edge_urls
        ]
        recovery_switch = _set_proxy_profile(
            proxy_url, "normal", control_token, args.request_timeout_seconds
        )
        recovery_switched_at_ms = int(time.time() * 1000)
        recovery_network = _wait_network_state(
            edge_urls,
            True,
            recovery_switched_at_ms,
            network_transition_samples,
            args.profile_transition_seconds,
            args.request_timeout_seconds,
            args.poll_interval_seconds,
        )
        recovery = _wait_recovery(
            edge_urls,
            outage_events,
            args.cloud_url,
            args.recovery_timeout_seconds,
            args.poll_interval_seconds,
            args.request_timeout_seconds,
        )
        recovery.update(
            {
                "proxy_state_after_switch": recovery_switch,
                "edge_network_snapshots": recovery_network,
                "outbox_before_recovery": backlog_before_recovery,
                "manual_flush_used": False,
            }
        )
        weak_results = [profile_results[name] for name in WEAK_PROFILES]
        weak_attempts = sum(int(value["business_attempts"]) for value in weak_results)
        weak_successes = sum(int(value["business_successes"]) for value in weak_results)
        retention = weak_successes / float(weak_attempts)
        normal = profile_results["normal"]
        passed = bool(
            normal["business_successes"] == normal["business_attempts"]
            and retention >= 0.90
            and recovery["passed"]
        )
        evidence.update(
            {
                "measurement_status": "measured",
                "status": "passed" if passed else "failed",
                "normal_baseline": normal,
                "profiles": weak_results,
                "aggregate": {
                    "business_attempts": weak_attempts,
                    "business_successes": weak_successes,
                    "retention_rate": retention,
                    "threshold": 0.90,
                },
                "recovery": recovery,
                "passed": passed,
            }
        )
    except BaseException as exc:  # noqa: BLE001
        evidence["measurement_status"] = (
            "incomplete" if progress["attempted_events"] else "not_measured"
        )
        evidence["status"] = evidence["measurement_status"]
        evidence["errors"].append(
            {"type": type(exc).__name__, "message": str(exc)}
        )
    finally:
        for queue in command_queues:
            try:
                queue.put({"kind": "stop"})
            except Exception:  # noqa: BLE001
                pass
        for process in worker_processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        _stop_edge_services(edge_services)
        if proxy_server is not None:
            proxy_server.shutdown()
            proxy_server.server_close()
        if proxy_thread is not None:
            proxy_thread.join(timeout=2.0)
        evidence["finished_at"] = _utc_now()
        provenance_value = evidence.get("provenance")
        if isinstance(provenance_value, dict):
            provenance_value["generated_at"] = evidence["finished_at"]
        evidence["attempted_real_event_count"] = progress["attempted_events"]
        if fault_log.is_file():
            evidence["proxy_log"] = {
                "path": str(fault_log),
                "sha256": _sha256(fault_log),
                "line_count": sum(
                    1
                    for line in fault_log.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ),
            }
        _write_json(output, evidence)
        print(json.dumps({**evidence, "output": str(output)}, ensure_ascii=False, indent=2))
    if evidence.get("status") != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
