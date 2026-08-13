#!/usr/bin/env python3
"""Evaluate seven non-industrial competition targets from explicit evidence.

The evaluator deliberately does not discover result files.  Every input must be
listed in a manifest together with its SHA-256 and expected provenance.  This
prevents an old experiment, a different model, or an incompatible latency
definition from silently becoming evidence for the current release.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

from jsonschema import Draft202012Validator


TARGET_ORDER = (
    "capability_retention",
    "ttft_reduction",
    "single_inference_memory",
    "weak_network_retention",
    "weighted_business_e2e",
    "traffic_global_objective",
    "model_update_loop",
)

TARGET_NAMES_ZH = {
    "capability_retention": "边缘通用能力保持率",
    "ttft_reduction": "TTFT 降幅",
    "single_inference_memory": "单次推理内存",
    "weak_network_retention": "弱网基本业务保持率",
    "weighted_business_e2e": "四路径固定权重业务端到端时延",
    "traffic_global_objective": "交通有限候选全局目标",
    "model_update_loop": "模型更新闭环",
}

EXPECTED_SEMANTICS = {
    "capability_retention": "macro_capability_retention",
    "ttft_reduction": "same_protocol_ttft_reduction",
    "single_inference_memory": "single_inference_peak_memory_mb",
    "weak_network_retention": "basic_business_function_retention",
    "weighted_business_e2e": "business_actionable_end_to_end_latency_ms",
    "traffic_global_objective": "finite_candidate_joint_plan_utility",
    "model_update_loop": "package_release_apply_rollback",
}

EXPECTED_MEMORY_SEMANTICS = (
    "peak process-tree VmRSS during measured inference window"
)
EXPECTED_MEMORY_SCOPE = "jetson_edge_single_inference"
REQUIRED_CAPABILITY_CATEGORIES = frozenset(
    {"math", "code", "natural_language_reasoning"}
)

REQUIRED_ROUTES = (
    "edge_only",
    "local_autonomy",
    "cloud_async",
    "cloud_sync",
)
REQUIRED_ROUTE_ENDPOINTS = {
    "edge_only": "provisional_response_and_durable_summary",
    "local_autonomy": "provisional_response_and_durable_summary",
    "cloud_async": "provisional_response_and_durable_summary",
    "cloud_sync": "authoritative_final",
}

STATUS_ZH = {
    "passed": "通过",
    "failed": "未通过",
    "not_measured": "未测量",
    "invalid_evidence": "证据无效",
}


class EvidenceError(Exception):
    """Base class for evidence errors."""


class MissingEvidence(EvidenceError):
    """An explicitly referenced evidence file is absent."""


class InvalidEvidence(EvidenceError):
    """Evidence exists but its integrity or contract is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(manifest_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _read_ref(
    ref: Mapping[str, Any], manifest_dir: Path, *, require_json: bool = True
) -> Tuple[Path, Any]:
    path = _resolve_path(manifest_dir, str(ref.get("path", "")))
    if not path.is_file():
        raise MissingEvidence("证据文件不存在：{}".format(path))
    expected_sha = str(ref.get("sha256", "")).lower()
    actual_sha = _sha256(path)
    if not expected_sha or actual_sha != expected_sha:
        raise InvalidEvidence(
            "SHA-256 不匹配：{}（期望 {}，实测 {}）".format(
                path, expected_sha or "<空>", actual_sha
            )
        )
    if not require_json:
        return path, None
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidEvidence("无法读取 JSON 证据 {}：{}".format(path, exc))


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256((_canonical(value) + "\n").encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidEvidence("{} 必须是对象".format(name))
    return value


def _validate_provenance(
    target: str, evidence: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    provenance = evidence.get("provenance")
    if not isinstance(provenance, Mapping):
        raise InvalidEvidence("{} 缺少 provenance".format(target))
    required = (
        "git_commit",
        "dataset_id",
        "model_ids",
        "hardware_id",
        "metric_semantics",
        "run_id",
        "generated_at",
    )
    missing = [name for name in required if name not in provenance]
    if missing:
        raise InvalidEvidence(
            "{} provenance 缺少字段：{}".format(target, ", ".join(missing))
        )
    for field in ("git_commit", "dataset_id", "model_ids", "hardware_id"):
        if field not in expected:
            raise InvalidEvidence("manifest expected 缺少 {}".format(field))
        if _canonical(provenance[field]) != _canonical(expected[field]):
            raise InvalidEvidence(
                "{} 的 {} 不兼容（期望 {}，实测 {}）".format(
                    target,
                    field,
                    _canonical(expected[field]),
                    _canonical(provenance[field]),
                )
            )
    expected_semantics = EXPECTED_SEMANTICS[target]
    manifest_semantics = expected.get("metric_semantics")
    actual_semantics = provenance.get("metric_semantics")
    if manifest_semantics != expected_semantics:
        raise InvalidEvidence(
            "{} 的 manifest 口径必须是 {}".format(target, expected_semantics)
        )
    if actual_semantics != expected_semantics:
        raise InvalidEvidence(
            "{} 的证据口径不兼容（要求 {}，实测 {}）".format(
                target, expected_semantics, actual_semantics
            )
        )


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidEvidence("{} 必须是数值".format(name))
    value = float(value)
    if not math.isfinite(value):
        raise InvalidEvidence("{} 必须是有限数值".format(name))
    return value


def _positive_samples(values: Any, name: str) -> List[float]:
    if not isinstance(values, list) or not values:
        raise InvalidEvidence("{} 必须是非空数组".format(name))
    result = [_number(value, name) for value in values]
    if any(value < 0.0 for value in result):
        raise InvalidEvidence("{} 不能包含负数".format(name))
    return result


def _count(value: Any, name: str) -> int:
    numeric = _number(value, name)
    if not numeric.is_integer():
        raise InvalidEvidence("{} 必须是整数".format(name))
    return int(numeric)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values))


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _base_result(target: str) -> Dict[str, Any]:
    return {
        "target": target,
        "name_zh": TARGET_NAMES_ZH[target],
        "status": "not_measured",
        "criteria": {},
        "metrics": {},
        "reasons": [],
        "evidence": [],
    }


def _evaluate_capability(evidence: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
    teacher = _number(evidence.get("teacher_macro_score"), "teacher_macro_score")
    edge = _number(evidence.get("edge_macro_score"), "edge_macro_score")
    if not 0.0 < teacher <= 1.0 or not 0.0 <= edge <= 1.0:
        raise InvalidEvidence("teacher/edge macro score 必须是 [0, 1] 内的比例，且 teacher 大于 0")
    category_scores = _object(evidence.get("category_scores"), "category_scores")
    if set(category_scores) != REQUIRED_CAPABILITY_CATEGORIES:
        raise InvalidEvidence(
            "category_scores 必须且只能包含 math、code、natural_language_reasoning"
        )
    category_counts = _object(
        evidence.get("category_sample_counts"), "category_sample_counts"
    )
    if set(category_counts) != REQUIRED_CAPABILITY_CATEGORIES:
        raise InvalidEvidence(
            "category_sample_counts 必须且只能包含三类能力任务"
        )
    teacher_categories: List[float] = []
    edge_categories: List[float] = []
    total_samples = 0
    for category in sorted(REQUIRED_CAPABILITY_CATEGORIES):
        values = _object(category_scores[category], "category_scores.{}".format(category))
        teacher_score = _number(values.get("teacher"), "{}.teacher".format(category))
        edge_score = _number(values.get("edge"), "{}.edge".format(category))
        if not 0.0 <= teacher_score <= 1.0 or not 0.0 <= edge_score <= 1.0:
            raise InvalidEvidence("各类别 teacher/edge score 必须位于 [0, 1]")
        count = _count(category_counts[category], "{}.sample_count".format(category))
        embedded_count = _count(
            values.get("sample_count"), "category_scores.{}.sample_count".format(category)
        )
        if count <= 0 or embedded_count != count:
            raise InvalidEvidence("三类能力任务必须各有样本且类别计数一致")
        teacher_categories.append(teacher_score)
        edge_categories.append(edge_score)
        total_samples += count
    sample_count = _count(evidence.get("sample_count"), "sample_count")
    if sample_count != total_samples:
        raise InvalidEvidence("sample_count 与三类样本数之和不一致")
    if abs(teacher - _mean(teacher_categories)) > 1e-6:
        raise InvalidEvidence("teacher_macro_score 与三类分数宏平均不一致")
    if abs(edge - _mean(edge_categories)) > 1e-6:
        raise InvalidEvidence("edge_macro_score 与三类分数宏平均不一致")
    retention = edge / teacher
    metrics = {
        "teacher_macro_score": teacher,
        "edge_macro_score": edge,
        "retention_rate": retention,
        "category_scores": dict(category_scores),
        "category_sample_counts": dict(category_counts),
        "sample_count": sample_count,
    }
    passed = retention >= 0.80
    reasons = [] if passed else ["能力保持率低于 80%"]
    return passed, metrics, reasons


def _evaluate_ttft(evidence: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
    baseline = _positive_samples(evidence.get("baseline_ttft_ms"), "baseline_ttft_ms")
    edge = _positive_samples(evidence.get("edge_ttft_ms"), "edge_ttft_ms")
    if len(baseline) != len(edge):
        raise InvalidEvidence("同一提示词集的 baseline/edge TTFT 样本数必须一致")
    baseline_mean = _mean(baseline)
    edge_mean = _mean(edge)
    if baseline_mean <= 0.0:
        raise InvalidEvidence("baseline TTFT 均值必须大于 0")
    reduction = 1.0 - edge_mean / baseline_mean
    metrics = {
        "baseline_count": len(baseline),
        "edge_count": len(edge),
        "baseline_ttft_mean_ms": baseline_mean,
        "edge_ttft_mean_ms": edge_mean,
        "ttft_reduction_rate": reduction,
    }
    passed = reduction >= 0.75
    reasons = [] if passed else ["TTFT 降幅低于 75%"]
    return passed, metrics, reasons


def _evaluate_memory(evidence: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
    peak = _number(evidence.get("peak_memory_mb"), "peak_memory_mb")
    if peak <= 0.0:
        raise InvalidEvidence("peak_memory_mb 必须大于 0")
    memory_semantics = evidence.get("memory_semantics")
    if memory_semantics != EXPECTED_MEMORY_SEMANTICS:
        raise InvalidEvidence(
            "memory_semantics 必须精确为 {}".format(
                EXPECTED_MEMORY_SEMANTICS
            )
        )
    if evidence.get("measurement_scope") != EXPECTED_MEMORY_SCOPE:
        raise InvalidEvidence(
            "measurement_scope 必须是 Jetson 边缘单次推理实测"
        )
    provenance = _object(evidence.get("provenance"), "provenance")
    hardware = _object(provenance.get("hardware_id"), "provenance.hardware_id")
    measurement_host = _object(evidence.get("measurement_host"), "measurement_host")
    if _canonical(hardware) != _canonical(measurement_host):
        raise InvalidEvidence("measurement_host 必须与 provenance.hardware_id 完全一致")
    if hardware.get("role") != "edge_device":
        raise InvalidEvidence("单次推理内存必须在边缘设备本机实测")
    if hardware.get("platform") != "nvidia_jetson":
        raise InvalidEvidence("单次推理内存必须由 NVIDIA Jetson 本机取证")
    if hardware.get("machine") != "aarch64":
        raise InvalidEvidence("Jetson 内存证据的架构必须是 aarch64")
    if "jetson" not in str(hardware.get("device_model", "")).lower():
        raise InvalidEvidence("Jetson 内存证据缺少内核读取的设备型号")
    if not str(hardware.get("hostname", "")).strip():
        raise InvalidEvidence("Jetson 内存证据缺少主机名")
    window = _object(evidence.get("measurement_window"), "measurement_window")
    if _count(
        window.get("measured_requests_per_window"),
        "measurement_window.measured_requests_per_window",
    ) != 1:
        raise InvalidEvidence("每个内存观察窗必须且只能包含一次推理请求")
    window_count = _count(window.get("window_count"), "measurement_window.window_count")
    sample_count = _count(evidence.get("sample_count"), "sample_count")
    if window_count <= 0 or sample_count != window_count:
        raise InvalidEvidence("内存观察窗数量必须为正且与 sample_count 一致")
    if _number(
        window.get("sampler_interval_ms"), "measurement_window.sampler_interval_ms"
    ) <= 0.0:
        raise InvalidEvidence("内存采样间隔必须大于 0")
    metrics = {
        "peak_memory_mb": peak,
        "memory_semantics": memory_semantics,
        "measurement_scope": evidence["measurement_scope"],
        "measurement_host": dict(measurement_host),
        "sample_count": sample_count,
    }
    passed = peak <= 1536.0
    reasons = [] if passed else ["单次推理峰值内存超过 1536 MB"]
    return passed, metrics, reasons


_WEAK_AUTHORITATIVE_STAGES = {
    "lightweight_final",
    "large_model_review",
    "large_model_correction",
}
_WEAK_DURABLE_STAGES = {"handoff_durable", "outbox_durable"}
_WEAK_EXPECTED_MEMBERS = {
    "edge_node_{}".format(partition_id) for partition_id in range(4)
}


def _weak_bool(value: Mapping[str, Any], field: str, prefix: str) -> bool:
    result = value.get(field)
    if not isinstance(result, bool):
        raise InvalidEvidence("{}.{} 必须是布尔值".format(prefix, field))
    return result


def _weak_string_list(value: Any, name: str) -> List[str]:
    if not isinstance(value, list):
        raise InvalidEvidence("{} 必须是数组".format(name))
    result = [str(item).strip() for item in value]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise InvalidEvidence("{} 必须是非空且不重复的动作名数组".format(name))
    return result


def _weak_review_authoritative(review: Mapping[str, Any]) -> bool:
    final = review.get("final_decision", {})
    final = dict(final) if isinstance(final, Mapping) else {}
    metadata = final.get("metadata", {})
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    authorization = metadata.get("action_authorization", {})
    authorization = (
        dict(authorization) if isinstance(authorization, Mapping) else {}
    )
    return bool(
        str(review.get("state", "")) == "completed"
        and str(review.get("completion_stage", ""))
        in _WEAK_AUTHORITATIVE_STAGES
        and str(final.get("status", "")) == "final"
        and authorization.get("cloud_confirmed") is True
    )


def _recompute_weak_event(
    profile_id: str,
    event: Mapping[str, Any],
    planned_deadline_ms: float,
) -> Dict[str, Any]:
    """Recompute one business result solely from raw response/review timing."""
    observation = _object(
        event.get("business_observation"), "event.business_observation"
    )
    response = _object(observation.get("response"), "business_observation.response")
    review_raw = observation.get("review", {})
    review = _object(review_raw, "business_observation.review")
    observed_deadline = _number(
        observation.get("deadline_ms"), "business_observation.deadline_ms"
    )
    if not math.isclose(
        observed_deadline, planned_deadline_ms, rel_tol=0.0, abs_tol=1e-9
    ):
        raise InvalidEvidence("逐事件 deadline_ms 偏离预注册计划")

    reasons: List[str] = []
    response_ms = _number(
        response.get("input_to_response_ms"),
        "business_observation.response.input_to_response_ms",
    )
    if response_ms < 0.0:
        raise InvalidEvidence("逐事件响应时延不能为负")
    if str(response.get("status", "")) not in {"provisional", "final"}:
        reasons.append("invalid_edge_response_status")

    immediate = _weak_string_list(
        response.get("immediate_action_types"), "response.immediate_action_types"
    )
    deferred = _weak_string_list(
        response.get("deferred_action_types"), "response.deferred_action_types"
    )
    if set(immediate).intersection(deferred):
        reasons.append("overlapping_action_stages")
    local_authorized = _weak_bool(
        response, "local_actions_authorized", "business_observation.response"
    )
    cloud_confirmed = _weak_bool(
        response, "cloud_confirmed", "business_observation.response"
    )
    waits_for_cloud = _weak_bool(
        response, "policy_waits_for_cloud", "business_observation.response"
    )
    if immediate and not local_authorized:
        reasons.append("unsafe_action_authorization")
    if deferred and cloud_confirmed:
        reasons.append("provisional_deferred_action_claimed_cloud_confirmation")
    if _weak_bool(
        response, "summary_delivery_required", "business_observation.response"
    ) is not True:
        reasons.append("summary_delivery_was_not_required")
    if str(response.get("summary_persistence_stage", "")) not in _WEAK_DURABLE_STAGES:
        reasons.append("summary_was_not_durably_queued")

    observed_ms_raw = observation.get("authoritative_observed_ms")
    if profile_id == "outage":
        endpoint = "offline_local_autonomy"
        completion_ms: Optional[float] = response_ms
        if str(response.get("route", "")) != "local_autonomy":
            reasons.append("route_is_not_local_autonomy")
        if str(response.get("policy_route", "")) != "local_autonomy":
            reasons.append("policy_route_is_not_local_autonomy")
        if _weak_bool(
            response, "local_autonomy", "business_observation.response"
        ) is not True:
            reasons.append("local_autonomy_marker_missing")
        if cloud_confirmed:
            reasons.append("offline_response_claimed_cloud_confirmation")
        if waits_for_cloud:
            reasons.append("offline_business_waited_for_cloud")
        if str(review.get("requested_route", "")) != "local_autonomy":
            reasons.append("lifecycle_route_is_not_local_autonomy")
        if str(review.get("state", "")) not in {"queued", "inflight"}:
            reasons.append("offline_review_not_pending_for_replay")
        if observed_ms_raw is not None:
            reasons.append("offline_event_claimed_authoritative_observation")
    else:
        requires_final = bool(waits_for_cloud or deferred)
        if requires_final:
            endpoint = "authoritative_final"
            if observed_ms_raw is None:
                completion_ms = None
                reasons.append("authoritative_final_not_observed")
            else:
                completion_ms = _number(
                    observed_ms_raw,
                    "business_observation.authoritative_observed_ms",
                )
                if completion_ms < 0.0:
                    raise InvalidEvidence("权威 final 观察时延不能为负")
            if not _weak_review_authoritative(review):
                reasons.append("authoritative_final_missing")
            if str(review.get("requested_route", "")) != str(
                response.get("policy_route", "")
            ):
                reasons.append("review_route_mismatch")
        else:
            endpoint = "local_response"
            completion_ms = response_ms
            if observed_ms_raw is not None:
                _number(
                    observed_ms_raw,
                    "business_observation.authoritative_observed_ms",
                )

    if completion_ms is None:
        reasons.append("business_endpoint_not_observed")
    elif completion_ms > planned_deadline_ms:
        reasons.append("business_deadline_exceeded")
    return {
        "success": not reasons,
        "business_endpoint": endpoint,
        "completion_ms": completion_ms,
        "reasons": list(dict.fromkeys(reasons)),
    }


def _weak_outbox_total(snapshot: Mapping[str, Any], state: str) -> int:
    states = _object(snapshot.get("states"), "outbox.states")
    return _count(states.get(state), "outbox.states.{}".format(state))


def _validate_weak_outbox_snapshot(
    raw: Any, name: str, *, require_idle: bool
) -> List[Mapping[str, Any]]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise InvalidEvidence("{} 必须包含四个边缘 Outbox 快照".format(name))
    snapshots = [_object(value, "{}[]".format(name)) for value in raw]
    for index, snapshot in enumerate(snapshots):
        active = _count(snapshot.get("active"), "{}.active".format(name))
        reconciliation = _object(
            snapshot.get("reconciliation"), "{}.reconciliation".format(name)
        )
        reconciliation_active = _count(
            reconciliation.get("active"), "{}.reconciliation.active".format(name)
        )
        handoff = _object(
            snapshot.get("durable_handoff"), "{}.durable_handoff".format(name)
        )
        handoff_pending = _count(
            handoff.get("pending"), "{}.durable_handoff.pending".format(name)
        )
        durable_pending = _count(
            handoff.get("durable_pending_count"),
            "{}.durable_handoff.durable_pending_count".format(name),
        )
        if min(active, reconciliation_active, handoff_pending, durable_pending) < 0:
            raise InvalidEvidence("{} 的 Outbox 计数不能为负".format(name))
        for state in ("pending", "inflight", "completed"):
            _weak_outbox_total(snapshot, state)
        if require_idle and any(
            (active, reconciliation_active, handoff_pending, durable_pending)
        ):
            raise InvalidEvidence(
                "{} 的边缘 {} 尚有未完成 Outbox/回填记录".format(name, index)
            )
    return snapshots


def _validate_weak_profile_delivery(
    profile: Mapping[str, Any], profile_id: str, event_count: int
) -> None:
    before = _validate_weak_outbox_snapshot(
        profile.get("outbox_before_profile"),
        "{}.outbox_before_profile".format(profile_id),
        require_idle=True,
    )
    if profile_id == "outage":
        return
    after = _validate_weak_outbox_snapshot(
        profile.get("outbox_after_profile"),
        "{}.outbox_after_profile".format(profile_id),
        require_idle=True,
    )
    completed_delta = sum(
        _weak_outbox_total(after[index], "completed")
        - _weak_outbox_total(before[index], "completed")
        for index in range(4)
    )
    if completed_delta != event_count:
        raise InvalidEvidence(
            "{} 的 Outbox completed 增量 {} 与事件数 {} 不一致".format(
                profile_id, completed_delta, event_count
            )
        )
    if _count(
        profile.get("authoritative_reviews_after_profile"),
        "{}.authoritative_reviews_after_profile".format(profile_id),
    ) != event_count:
        raise InvalidEvidence("{} 后台权威回填数与事件数不一致".format(profile_id))


def _recompute_weak_recovery(
    recovery: Mapping[str, Any], outage_events: Sequence[Mapping[str, Any]]
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    expected_event_ids = [str(event.get("event_id", "")) for event in outage_events]
    expected_sample_by_event = {
        str(event.get("event_id", "")): _count(
            event.get("sample_id"), "outage event.sample_id"
        )
        for event in outage_events
    }
    recorded_ids = recovery.get("expected_event_ids")
    if not isinstance(recorded_ids, list) or [str(value) for value in recorded_ids] != expected_event_ids:
        raise InvalidEvidence("recovery.expected_event_ids 与断网事件不一致")
    recorded_samples = _object(
        recovery.get("expected_sample_by_event"),
        "recovery.expected_sample_by_event",
    )
    if {str(key): _count(value, "recovery sample_id") for key, value in recorded_samples.items()} != expected_sample_by_event:
        raise InvalidEvidence("recovery.expected_sample_by_event 与断网事件不一致")

    before = _validate_weak_outbox_snapshot(
        recovery.get("outbox_before_recovery"),
        "recovery.outbox_before_recovery",
        require_idle=False,
    )
    pending_before = sum(
        _count(snapshot.get("active"), "outbox.active")
        + _count(
            _object(snapshot.get("durable_handoff"), "durable_handoff").get(
                "durable_pending_count"
            ),
            "durable_handoff.durable_pending_count",
        )
        for snapshot in before
    )
    if pending_before < len(expected_event_ids):
        reasons.append("outage_events_not_all_pending_before_recovery")
    _validate_weak_outbox_snapshot(
        recovery.get("outbox_snapshots"),
        "recovery.outbox_snapshots",
        require_idle=True,
    )
    if recovery.get("manual_flush_used") is not False:
        reasons.append("manual_outbox_flush_used")

    reviews = _object(recovery.get("review_records"), "recovery.review_records")
    if set(str(key) for key in reviews) != set(expected_event_ids):
        reasons.append("recovered_review_set_mismatch")
    review_groups: Dict[str, List[str]] = {}
    for event_id, raw_review in reviews.items():
        review = _object(raw_review, "recovery.review_records[]")
        if not _weak_review_authoritative(review):
            reasons.append("not_all_reviews_are_authoritative")
        final = _object(review.get("final_decision"), "recovery.final_decision")
        metadata = _object(final.get("metadata"), "recovery.final_decision.metadata")
        aggregation = _object(metadata.get("aggregation"), "recovery.review.aggregation")
        group_id = str(aggregation.get("group_id", ""))
        if not group_id:
            reasons.append("review_missing_aggregation_group")
        else:
            review_groups.setdefault(group_id, []).append(str(event_id))
    expected_sample_count = len(set(expected_sample_by_event.values()))
    if len(review_groups) != expected_sample_count:
        reasons.append("review_aggregation_group_count_mismatch")
    for event_ids in review_groups.values():
        sample_ids = {expected_sample_by_event[event_id] for event_id in event_ids}
        if len(event_ids) != 4 or len(sample_ids) != 1:
            reasons.append("cross_sample_or_incomplete_review_group")

    aggregations_raw = recovery.get("aggregation_records")
    if not isinstance(aggregations_raw, list):
        raise InvalidEvidence("recovery.aggregation_records 必须是数组")
    aggregations = [
        _object(value, "recovery.aggregation_records[]")
        for value in aggregations_raw
    ]
    if len(aggregations) != expected_sample_count:
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
        if set(aggregation.get("expected_members", [])) != _WEAK_EXPECTED_MEMBERS:
            reasons.append("aggregation_expected_members_mismatch")
        if set(aggregation.get("received_members", [])) != _WEAK_EXPECTED_MEMBERS:
            reasons.append("aggregation_received_members_mismatch")
        if aggregation.get("missing_members") not in ([], None):
            reasons.append("aggregation_reports_missing_members")
    return not reasons, list(dict.fromkeys(reasons))


def _evaluate_weak_network(
    evidence: Mapping[str, Any],
    experiment_plan: Mapping[str, Any],
    experiment_plan_sha256: str,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    early_measurement_status = str(evidence.get("measurement_status", ""))
    if early_measurement_status in {"not_measured", "incomplete"}:
        raise MissingEvidence(
            "弱网实验状态为 {}，未形成完整可判定证据".format(
                early_measurement_status
            )
        )
    evidence_schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "weak_network_evidence.schema.json"
    )
    evidence_schema = json.loads(evidence_schema_path.read_text(encoding="utf-8"))
    evidence_errors = sorted(
        Draft202012Validator(evidence_schema).iter_errors(evidence),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if evidence_errors:
        raise InvalidEvidence(
            "弱网证据不符合 schema：{}".format(
                "; ".join(error.message for error in evidence_errors[:5])
            )
        )
    plan_schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "weak_network_experiment_plan.schema.json"
    )
    plan_schema = json.loads(plan_schema_path.read_text(encoding="utf-8"))
    plan_errors = sorted(
        Draft202012Validator(plan_schema).iter_errors(experiment_plan),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if plan_errors:
        raise InvalidEvidence(
            "弱网实验计划不符合 schema：{}".format(
                "; ".join(error.message for error in plan_errors[:5])
            )
        )
    required_plan_fields = (
        "schema_version",
        "plan_id",
        "created_at",
        "scene",
        "git_commit",
        "dataset_id",
        "model_ids",
        "hardware_id",
        "seed",
        "profile_order",
        "fault_profiles",
        "sample_ids_by_profile",
        "business_deadline_ms",
        "aggregation_timeout_ms",
        "dispatch_lead_ms",
        "split",
        "top_k",
    )
    missing_plan = [
        name for name in required_plan_fields if name not in experiment_plan
    ]
    if missing_plan:
        raise InvalidEvidence(
            "弱网实验计划缺少字段：{}".format(", ".join(missing_plan))
        )
    if experiment_plan.get("schema_version") != "1.0":
        raise InvalidEvidence("弱网实验计划 schema_version 必须是 1.0")
    if experiment_plan.get("scene") != "freeway_traffic":
        raise InvalidEvidence("弱网实验计划 scene 必须是 freeway_traffic")
    expected_order = ["normal", "mild", "severe", "outage", "recovery"]
    if experiment_plan.get("profile_order") != expected_order:
        raise InvalidEvidence("弱网实验计划 profile_order 不兼容")
    plan_profiles = _object(
        experiment_plan.get("fault_profiles"), "experiment_plan.fault_profiles"
    )
    plan_samples = _object(
        experiment_plan.get("sample_ids_by_profile"),
        "experiment_plan.sample_ids_by_profile",
    )
    if set(plan_profiles) != {"normal", "mild", "severe", "outage"}:
        raise InvalidEvidence("弱网实验计划必须定义四个固定网络 profile")
    if set(plan_samples) != {"normal", "mild", "severe", "outage"}:
        raise InvalidEvidence("弱网实验计划必须为四个 profile 固定样本")
    flattened_plan_samples: List[int] = []
    normalized_plan_samples: Dict[str, List[int]] = {}
    for profile_id in ("normal", "mild", "severe", "outage"):
        raw_ids = plan_samples[profile_id]
        if not isinstance(raw_ids, list) or not raw_ids:
            raise InvalidEvidence("弱网实验计划每个 profile 都必须有样本")
        ids = [_count(value, "sample_id") for value in raw_ids]
        if any(value < 0 for value in ids) or len(ids) != len(set(ids)):
            raise InvalidEvidence("弱网实验计划 sample_id 必须非负且组内唯一")
        normalized_plan_samples[profile_id] = ids
        flattened_plan_samples.extend(ids)
    if len(flattened_plan_samples) != len(set(flattened_plan_samples)):
        raise InvalidEvidence("弱网实验计划不同 profile 的 sample_id 不得重叠")
    plan_ref = _object(evidence.get("experiment_plan"), "experiment_plan binding")
    if (
        str(plan_ref.get("plan_id", "")) != str(experiment_plan.get("plan_id", ""))
        or str(plan_ref.get("sha256", "")).lower()
        != str(experiment_plan_sha256).lower()
        or str(plan_ref.get("created_at", ""))
        != str(experiment_plan.get("created_at", ""))
    ):
        raise InvalidEvidence("弱网证据没有绑定预注册实验计划 ID/SHA/时间")
    provenance = _object(evidence.get("provenance"), "provenance")
    for field in ("git_commit", "dataset_id", "model_ids", "hardware_id"):
        if _canonical(provenance.get(field)) != _canonical(experiment_plan.get(field)):
            raise InvalidEvidence("弱网实验计划的 {} 与实测 provenance 不一致".format(field))
    created_at = _parse_time(experiment_plan.get("created_at"), "plan.created_at")
    loaded_at = _parse_time(plan_ref.get("loaded_at"), "plan.loaded_at")
    run_started_at = _parse_time(
        provenance.get("run_started_at"), "provenance.run_started_at"
    )
    generated_at = _parse_time(
        provenance.get("generated_at"), "provenance.generated_at"
    )
    if not created_at <= loaded_at <= run_started_at <= generated_at:
        raise InvalidEvidence("弱网计划必须先于装载和正式运行生成")
    if evidence.get("formal_eligible") is not True:
        raise InvalidEvidence("弱网证据来自脏工作区或开发运行，不能进入正式门禁")
    configuration = _object(evidence.get("configuration"), "configuration")
    configuration_bindings = {
        "profile_order": expected_order,
        "fault_profiles": plan_profiles,
        "sample_ids_by_profile": normalized_plan_samples,
        "business_deadline_ms": experiment_plan["business_deadline_ms"],
        "aggregation_timeout_ms": experiment_plan["aggregation_timeout_ms"],
        "dispatch_lead_ms": experiment_plan["dispatch_lead_ms"],
        "split": experiment_plan["split"],
        "top_k": experiment_plan["top_k"],
        "seed": experiment_plan["seed"],
    }
    for field, planned in configuration_bindings.items():
        if _canonical(configuration.get(field)) != _canonical(planned):
            raise InvalidEvidence("弱网实际配置 {} 偏离预注册计划".format(field))

    measurement_status = str(evidence.get("measurement_status", ""))
    if measurement_status in {"not_measured", "incomplete"}:
        raise MissingEvidence(
            "弱网实验状态为 {}，未形成完整可判定证据".format(measurement_status)
        )
    if measurement_status != "measured":
        raise InvalidEvidence("measurement_status 必须是 measured")
    profiles = evidence.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise InvalidEvidence("profiles 必须是非空数组")
    successes = 0
    attempts = 0
    profile_metrics = []
    profile_ids = []
    all_event_ids: List[str] = []
    for index, profile in enumerate(profiles):
        if not isinstance(profile, Mapping):
            raise InvalidEvidence("profiles[{}] 必须是对象".format(index))
        attempted = _count(profile.get("business_attempts"), "business_attempts")
        succeeded = _count(profile.get("business_successes"), "business_successes")
        if attempted <= 0 or succeeded < 0 or succeeded > attempted:
            raise InvalidEvidence("弱网 profile 的成功数/尝试数无效")
        attempts += attempted
        profile_id = str(profile.get("profile_id", ""))
        if not profile_id:
            raise InvalidEvidence("弱网 profile_id 不能为空")
        if profile_id not in normalized_plan_samples:
            raise InvalidEvidence("弱网证据包含未预注册 profile：{}".format(profile_id))
        if _canonical(profile.get("fault_parameters")) != _canonical(
            plan_profiles[profile_id]
        ):
            raise InvalidEvidence("{} 的故障参数偏离预注册计划".format(profile_id))
        samples = profile.get("samples")
        if not isinstance(samples, list):
            raise InvalidEvidence("{} 缺少逐 sample 记录".format(profile_id))
        actual_sample_ids = [
            _count(_object(row, "sample").get("sample_id"), "sample_id")
            for row in samples
        ]
        if actual_sample_ids != normalized_plan_samples[profile_id]:
            raise InvalidEvidence("{} 实测 sample_id 偏离预注册计划".format(profile_id))
        expected_attempts = len(actual_sample_ids) * 4
        if attempted != expected_attempts:
            raise InvalidEvidence("{} 必须为每个样本记录四个边缘事件".format(profile_id))
        events = profile.get("events")
        if not isinstance(events, list) or len(events) != attempted:
            raise InvalidEvidence("{} 缺少完整逐事件记录".format(profile_id))
        event_ids: List[str] = []
        event_sample_counts = {sample_id: 0 for sample_id in actual_sample_ids}
        producer_event_successes = 0
        recomputed_event_successes = 0
        recomputed_by_sample = {sample_id: 0 for sample_id in actual_sample_ids}
        for event_index, raw_event in enumerate(events):
            event = _object(raw_event, "events[{}]".format(event_index))
            event_id = str(event.get("event_id", ""))
            event_sample_id = _count(event.get("sample_id"), "event.sample_id")
            if not event_id or event_sample_id not in event_sample_counts:
                raise InvalidEvidence("{} 逐事件记录不属于预注册样本".format(profile_id))
            if str(event.get("profile", "")) != profile_id:
                raise InvalidEvidence("{} 逐事件 profile 标记不一致".format(profile_id))
            if not isinstance(event.get("success"), bool):
                raise InvalidEvidence("{} 逐事件 success 必须是布尔值".format(profile_id))
            recomputed = _recompute_weak_event(
                profile_id,
                event,
                float(experiment_plan["business_deadline_ms"]),
            )
            event_ids.append(event_id)
            event_sample_counts[event_sample_id] += 1
            producer_event_successes += int(event["success"])
            recomputed_event_successes += int(recomputed["success"])
            recomputed_by_sample[event_sample_id] += int(recomputed["success"])
        if len(event_ids) != len(set(event_ids)):
            raise InvalidEvidence("{} 逐事件 event_id 重复".format(profile_id))
        if any(count != 4 for count in event_sample_counts.values()):
            raise InvalidEvidence("{} 每个 sample 必须正好对应四个边缘事件".format(profile_id))
        if producer_event_successes != succeeded:
            raise InvalidEvidence("{} 逐事件成功数与汇总不一致".format(profile_id))
        if recomputed_event_successes != producer_event_successes:
            raise InvalidEvidence(
                "{} 自报 success 与原始响应/回填复算不一致".format(profile_id)
            )
        for sample in samples:
            sample_value = _object(sample, "{}.samples[]".format(profile_id))
            sample_id = _count(sample_value.get("sample_id"), "sample.sample_id")
            if _count(
                sample_value.get("business_attempts"), "sample.business_attempts"
            ) != 4 or _count(
                sample_value.get("business_successes"), "sample.business_successes"
            ) != recomputed_by_sample[sample_id]:
                raise InvalidEvidence(
                    "{} 的逐 sample 汇总与原始记录复算不一致".format(profile_id)
                )
        _validate_weak_profile_delivery(profile, profile_id, attempted)
        successes += recomputed_event_successes
        all_event_ids.extend(event_ids)
        profile_ids.append(profile_id)
        profile_metrics.append(
            {
                "profile_id": profile_id,
                "business_attempts": attempted,
                "business_successes": recomputed_event_successes,
                "retention_rate": recomputed_event_successes / float(attempted),
            }
        )
    expected_profiles = {"mild", "severe", "outage"}
    if len(profile_ids) != len(set(profile_ids)) or set(profile_ids) != expected_profiles:
        raise InvalidEvidence("弱网 profiles 必须且只能包含 mild、severe、outage")

    normal = evidence.get("normal_baseline")
    if not isinstance(normal, Mapping):
        raise InvalidEvidence("弱网证据缺少 normal_baseline")
    normal_attempts = _count(normal.get("business_attempts"), "normal business_attempts")
    normal_successes = _count(
        normal.get("business_successes"), "normal business_successes"
    )
    if normal_attempts <= 0 or not 0 <= normal_successes <= normal_attempts:
        raise InvalidEvidence("normal_baseline 的成功数/尝试数无效")
    if str(normal.get("profile_id", "")) != "normal":
        raise InvalidEvidence("normal_baseline.profile_id 必须是 normal")
    if _canonical(normal.get("fault_parameters")) != _canonical(
        plan_profiles["normal"]
    ):
        raise InvalidEvidence("normal 故障参数偏离预注册计划")
    normal_samples = normal.get("samples")
    normal_events = normal.get("events")
    if not isinstance(normal_samples, list) or not isinstance(normal_events, list):
        raise InvalidEvidence("normal_baseline 缺少逐样本/逐事件记录")
    normal_ids = [
        _count(_object(row, "normal sample").get("sample_id"), "normal sample_id")
        for row in normal_samples
    ]
    if normal_ids != normalized_plan_samples["normal"]:
        raise InvalidEvidence("normal 实测 sample_id 偏离预注册计划")
    if normal_attempts != len(normal_ids) * 4 or len(normal_events) != normal_attempts:
        raise InvalidEvidence("normal 必须为每个样本记录四个边缘事件")
    normal_event_ids: List[str] = []
    normal_counts = {sample_id: 0 for sample_id in normal_ids}
    normal_producer_successes = 0
    normal_recomputed_successes = 0
    normal_recomputed_by_sample = {sample_id: 0 for sample_id in normal_ids}
    for index, raw_event in enumerate(normal_events):
        event = _object(raw_event, "normal events[{}]".format(index))
        event_id = str(event.get("event_id", ""))
        sample_id = _count(event.get("sample_id"), "normal event.sample_id")
        if not event_id or sample_id not in normal_counts:
            raise InvalidEvidence("normal 逐事件记录不属于预注册样本")
        if str(event.get("profile", "")) != "normal":
            raise InvalidEvidence("normal 逐事件 profile 标记不一致")
        if not isinstance(event.get("success"), bool):
            raise InvalidEvidence("normal 逐事件 success 必须是布尔值")
        recomputed = _recompute_weak_event(
            "normal",
            event,
            float(experiment_plan["business_deadline_ms"]),
        )
        normal_event_ids.append(event_id)
        normal_counts[sample_id] += 1
        normal_producer_successes += int(event["success"])
        normal_recomputed_successes += int(recomputed["success"])
        normal_recomputed_by_sample[sample_id] += int(recomputed["success"])
    if len(normal_event_ids) != len(set(normal_event_ids)):
        raise InvalidEvidence("normal 逐事件 event_id 重复")
    if any(count != 4 for count in normal_counts.values()):
        raise InvalidEvidence("normal 每个 sample 必须正好对应四个边缘事件")
    if normal_producer_successes != normal_successes:
        raise InvalidEvidence("normal 逐事件成功数与汇总不一致")
    if normal_recomputed_successes != normal_producer_successes:
        raise InvalidEvidence("normal 自报 success 与原始响应/回填复算不一致")
    for sample in normal_samples:
        sample_value = _object(sample, "normal.samples[]")
        sample_id = _count(sample_value.get("sample_id"), "normal sample_id")
        if _count(
            sample_value.get("business_attempts"), "normal sample.business_attempts"
        ) != 4 or _count(
            sample_value.get("business_successes"), "normal sample.business_successes"
        ) != normal_recomputed_by_sample[sample_id]:
            raise InvalidEvidence("normal 逐 sample 汇总与原始记录复算不一致")
    _validate_weak_profile_delivery(normal, "normal", normal_attempts)
    all_event_ids.extend(normal_event_ids)
    if len(all_event_ids) != len(set(all_event_ids)):
        raise InvalidEvidence("不同 profile 之间出现重复 event_id")

    recovery = evidence.get("recovery")
    if not isinstance(recovery, Mapping) or not isinstance(recovery.get("passed"), bool):
        raise InvalidEvidence("弱网证据缺少可判定的 recovery.passed")
    outage_profile = next(
        profile for profile in profiles if profile.get("profile_id") == "outage"
    )
    recovery_recomputed, recovery_reasons = _recompute_weak_recovery(
        recovery,
        [
            _object(value, "outage.events[]")
            for value in outage_profile.get("events", [])
        ],
    )
    if recovery.get("passed") is not recovery_recomputed:
        raise InvalidEvidence("recovery.passed 与 Outbox/回填/汇聚原始记录复算不一致")
    retention = successes / float(attempts)
    total_attempts = normal_attempts + attempts
    attempted_real = _count(
        evidence.get("attempted_real_event_count"), "attempted_real_event_count"
    )
    if attempted_real != total_attempts:
        raise InvalidEvidence("attempted_real_event_count 与 normal/弱网尝试数不一致")

    aggregate = evidence.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise InvalidEvidence("弱网证据缺少 aggregate")
    if (
        _count(aggregate.get("business_attempts"), "aggregate business_attempts")
        != attempts
        or _count(
            aggregate.get("business_successes"), "aggregate business_successes"
        )
        != successes
    ):
        raise InvalidEvidence("aggregate 与逐 profile 计数不一致")
    metrics = {
        "business_attempts": attempts,
        "business_successes": successes,
        "retention_rate": retention,
        "profiles": profile_metrics,
        "normal_business_attempts": normal_attempts,
        "normal_business_successes": normal_recomputed_successes,
        "recovery_passed": recovery_recomputed,
        "recovery_reasons": recovery_reasons,
        "measurement_status": measurement_status,
    }
    reasons = []
    if normal_recomputed_successes != normal_attempts:
        reasons.append("正常网络基线未全部成功")
    if retention < 0.90:
        reasons.append("弱网基本业务保持率低于 90%")
    if recovery_recomputed is not True:
        reasons.append("断网恢复与补传闭环未通过")
    passed = not reasons
    producer_status = str(evidence.get("status", ""))
    expected_status = "passed" if passed else "failed"
    if producer_status != expected_status:
        raise InvalidEvidence("弱网证据 status 与计数/恢复判定不一致")
    if "passed" in evidence and evidence.get("passed") is not passed:
        raise InvalidEvidence("弱网证据 passed 与计数/恢复判定不一致")
    return passed, metrics, reasons


def _parse_time(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise InvalidEvidence("{} 必须是 RFC3339 时间字符串".format(name))
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidEvidence("{} 不是合法时间：{}".format(name, exc))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InvalidEvidence("{} 必须包含时区".format(name))
    return parsed


def _nonnegative_unique_ints(value: Any, name: str) -> List[int]:
    if not isinstance(value, list) or not value:
        raise InvalidEvidence("{} 必须是非空数组".format(name))
    normalized = [_count(item, name) for item in value]
    if any(item < 0 for item in normalized):
        raise InvalidEvidence("{} 不能包含负数".format(name))
    if len(set(normalized)) != len(normalized):
        raise InvalidEvidence("{} 不能包含重复值".format(name))
    return normalized


def _planned_route_population(
    evidence: Mapping[str, Any],
) -> Dict[str, set]:
    sources = evidence.get("source_runs")
    if not isinstance(sources, list) or not sources:
        raise InvalidEvidence("四路径证据必须包含预注册 source_runs")
    expected: Dict[str, set] = {name: set() for name in REQUIRED_ROUTES}
    seen_runs = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise InvalidEvidence("source_runs[{}] 必须是对象".format(index))
        run_id = str(source.get("run_id", "")).strip()
        route = str(source.get("policy_route", ""))
        if not run_id or run_id in seen_runs:
            raise InvalidEvidence("source_runs 的 run_id 必须非空且唯一")
        seen_runs.add(run_id)
        if route not in REQUIRED_ROUTES:
            raise InvalidEvidence("source_runs 包含未知 policy_route")
        if source.get("integrity_valid") is not True:
            raise InvalidEvidence("source_runs 必须全部通过完整性校验")
        source_errors = source.get("errors")
        if not isinstance(source_errors, list) or source_errors:
            raise InvalidEvidence("source_runs.errors 必须是空数组")
        sample_ids = _nonnegative_unique_ints(
            source.get("planned_sample_ids"),
            "source_runs[{}].planned_sample_ids".format(index),
        )
        partition_ids = _nonnegative_unique_ints(
            source.get("planned_partition_ids"),
            "source_runs[{}].planned_partition_ids".format(index),
        )
        expected[route].update(
            (run_id, sample_id, partition_id)
            for sample_id in sample_ids
            for partition_id in partition_ids
        )
    if any(not expected[name] for name in REQUIRED_ROUTES):
        raise InvalidEvidence("四条路径都必须具有预注册尝试总体")
    return expected


def _route_group_means(
    route: Mapping[str, Any],
    route_name: str,
    expected_keys: set,
) -> Tuple[List[float], Dict[str, int]]:
    samples = route.get("samples")
    if not isinstance(samples, list) or not samples:
        raise InvalidEvidence("{} 路径缺少 samples".format(route_name))
    failures = route.get("failures")
    if not isinstance(failures, list):
        raise InvalidEvidence("{} 路径缺少 failures".format(route_name))
    grouped: MutableMapping[Tuple[str, int], List[float]] = {}
    successful_keys = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise InvalidEvidence("{} samples[{}] 必须是对象".format(route_name, index))
        run_id = str(sample.get("run_id", "")).strip()
        sample_id = _count(sample.get("sample_id"), "sample_id")
        partition_id = _count(sample.get("partition_id"), "partition_id")
        if not run_id or sample_id < 0 or partition_id < 0:
            raise InvalidEvidence("{} samples 的运行/样本/分区标识无效".format(route_name))
        key = (run_id, sample_id, partition_id)
        if key in successful_keys or key not in expected_keys:
            raise InvalidEvidence("{} samples 偏离预注册总体或包含重复项".format(route_name))
        successful_keys.add(key)
        expected_group_id = "{}:{}".format(run_id, sample_id)
        if str(sample.get("group_id", "")) != expected_group_id:
            raise InvalidEvidence("{} group_id 必须由 run_id:sample_id 派生".format(route_name))
        if str(sample.get("policy_route", "")) != route_name:
            raise InvalidEvidence("{} 样本 policy_route 不一致".format(route_name))
        if str(sample.get("business_endpoint", "")) != REQUIRED_ROUTE_ENDPOINTS[route_name]:
            raise InvalidEvidence("{} 样本业务终点语义不一致".format(route_name))
        latency = _number(sample.get("latency_ms"), "latency_ms")
        if latency < 0.0:
            raise InvalidEvidence("latency_ms 不能为负数")
        grouped.setdefault((run_id, sample_id), []).append(latency)
    failed_keys = set()
    for index, failure in enumerate(failures):
        if not isinstance(failure, Mapping):
            raise InvalidEvidence("{} failures[{}] 必须是对象".format(route_name, index))
        key = (
            str(failure.get("run_id", "")).strip(),
            _count(failure.get("sample_id"), "failure.sample_id"),
            _count(failure.get("partition_id"), "failure.partition_id"),
        )
        if (
            not key[0]
            or key[1] < 0
            or key[2] < 0
            or key in failed_keys
            or key in successful_keys
            or key not in expected_keys
        ):
            raise InvalidEvidence("{} failures 偏离预注册总体或重复计数".format(route_name))
        if not str(failure.get("reason", "")).strip():
            raise InvalidEvidence("{} failures 必须声明原因".format(route_name))
        failed_keys.add(key)
    if successful_keys | failed_keys != expected_keys:
        raise InvalidEvidence("{} 未逐项覆盖预注册 sample_id × partition_id 总体".format(route_name))
    return (
        [_mean(values) for _, values in sorted(grouped.items())],
        {
            "attempt_count": len(expected_keys),
            "success_count": len(successful_keys),
            "failure_count": len(failed_keys),
        },
    )


def _evaluate_latency(
    evidence: Mapping[str, Any],
    weight_plan: Mapping[str, Any],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    weight_plan_sha256: str = "",
) -> Tuple[bool, Dict[str, Any], List[str]]:
    if bootstrap_iterations < 200:
        raise InvalidEvidence("bootstrap_iterations 至少为 200")
    if not isinstance(weight_plan, Mapping):
        raise InvalidEvidence("固定权重计划根节点必须是对象")
    if weight_plan.get("schema_version") != 1:
        raise InvalidEvidence("固定权重计划 schema_version 必须为 1")
    if evidence.get("integrity_valid") is not True:
        raise InvalidEvidence("四路径证据 integrity_valid 必须为 true")
    integrity_errors = evidence.get("integrity_errors")
    if not isinstance(integrity_errors, list):
        raise InvalidEvidence("四路径证据 integrity_errors 必须是数组")
    if integrity_errors:
        raise InvalidEvidence("四路径证据 integrity_errors 必须为空")
    producer_errors = evidence.get("errors", [])
    if not isinstance(producer_errors, list) or producer_errors:
        raise InvalidEvidence("四路径证据 errors 必须是空数组")
    run_started = _parse_time(
        evidence.get("provenance", {}).get("run_started_at"), "run_started_at"
    )
    locked_at = _parse_time(weight_plan.get("locked_at"), "weight_plan.locked_at")
    if locked_at > run_started:
        raise InvalidEvidence("固定权重计划必须在测量开始前锁定")
    plan_id = str(weight_plan.get("weight_plan_id", ""))
    if not plan_id:
        raise InvalidEvidence("weight_plan_id 不能为空")
    evidence_plan_id = str(evidence.get("weight_plan_id", ""))
    if not evidence_plan_id:
        raise InvalidEvidence("四路径证据 weight_plan_id 不能为空")
    if evidence_plan_id != plan_id:
        raise InvalidEvidence("证据与固定权重计划的 weight_plan_id 不一致")
    plan_sha = str(evidence.get("plan_sha256", "")).lower()
    if len(plan_sha) != 64 or any(
        character not in "0123456789abcdef" for character in plan_sha
    ):
        raise InvalidEvidence("四路径证据 plan_sha256 必须是有效 SHA-256")
    declared_plan_sha = str(weight_plan.get("plan_sha256", "")).lower()
    if not _is_sha256(declared_plan_sha) or declared_plan_sha != plan_sha:
        raise InvalidEvidence("证据与固定权重计划的 plan_sha256 不一致")
    declared_weight_plan_sha = str(
        evidence.get("weight_plan_sha256", "")
    ).lower()
    if declared_weight_plan_sha:
        if not weight_plan_sha256 or declared_weight_plan_sha != weight_plan_sha256:
            raise InvalidEvidence("证据声明的 weight_plan_sha256 与实物不一致")
    weights = weight_plan.get("routes")
    routes = evidence.get("routes")
    if not isinstance(weights, Mapping) or not isinstance(routes, Mapping):
        raise InvalidEvidence("权重计划和时延证据都必须包含 routes")
    if set(weights) != set(REQUIRED_ROUTES) or set(routes) != set(REQUIRED_ROUTES):
        raise InvalidEvidence(
            "四路径必须恰好是 {}".format(", ".join(REQUIRED_ROUTES))
        )
    numeric_weights = {name: _number(weights[name], name) for name in REQUIRED_ROUTES}
    if any(value <= 0.0 for value in numeric_weights.values()):
        raise InvalidEvidence("四条路径的固定权重都必须大于 0")
    if not math.isclose(sum(numeric_weights.values()), 1.0, abs_tol=1e-9):
        raise InvalidEvidence("四路径权重之和必须为 1")
    expected_population = _planned_route_population(evidence)
    route_evaluations = {
        name: _route_group_means(routes[name], name, expected_population[name])
        for name in REQUIRED_ROUTES
    }
    group_means = {name: route_evaluations[name][0] for name in REQUIRED_ROUTES}
    route_counts: Dict[str, Dict[str, int]] = {}
    for name in REQUIRED_ROUTES:
        route = routes[name]
        attempted = _count(
            route.get("attempt_count"), "{}.attempt_count".format(name)
        )
        succeeded = _count(
            route.get("success_count"), "{}.success_count".format(name)
        )
        samples = route.get("samples", [])
        if attempted <= 0 or succeeded < 0 or succeeded > attempted:
            raise InvalidEvidence("{} 路径的尝试数/成功数无效".format(name))
        if not isinstance(samples, list) or len(samples) != succeeded:
            raise InvalidEvidence(
                "{} 路径的成功数必须等于带时延的样本数".format(name)
            )
        declared_failed = _count(
            route.get("failure_count"), "{}.failure_count".format(name)
        )
        derived_counts = route_evaluations[name][1]
        if (
            attempted != derived_counts["attempt_count"]
            or succeeded != derived_counts["success_count"]
            or declared_failed != derived_counts["failure_count"]
            or declared_failed != attempted - succeeded
        ):
            raise InvalidEvidence("{} 路径计数与预注册总体不一致".format(name))
        route_counts[name] = derived_counts
    route_metrics = {
        name: {
            "weight": numeric_weights[name],
            "group_count": len(group_means[name]),
            "mean_ms": _mean(group_means[name]),
            **route_counts[name],
        }
        for name in REQUIRED_ROUTES
    }
    weighted_mean = sum(
        numeric_weights[name] * route_metrics[name]["mean_ms"]
        for name in REQUIRED_ROUTES
    )
    rng = random.Random(bootstrap_seed)
    bootstrapped = []
    for _ in range(bootstrap_iterations):
        sample_mean = 0.0
        for name in REQUIRED_ROUTES:
            values = group_means[name]
            resampled = [values[rng.randrange(len(values))] for _ in values]
            sample_mean += numeric_weights[name] * _mean(resampled)
        bootstrapped.append(sample_mean)
    ci_low = _percentile(bootstrapped, 0.025)
    ci_high = _percentile(bootstrapped, 0.975)
    metrics = {
        "weight_plan_id": plan_id,
        "plan_sha256": plan_sha,
        "weight_plan_file_sha256": weight_plan_sha256,
        "integrity_valid": True,
        "weighted_mean_ms": weighted_mean,
        "bootstrap_95_ci_ms": [ci_low, ci_high],
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "resampling_unit": "sample_id_within_run_and_policy_route",
        "routes": route_metrics,
    }
    reasons = []
    failed_routes = [
        name
        for name in REQUIRED_ROUTES
        if route_counts[name]["failure_count"] > 0
    ]
    if failed_routes:
        reasons.append(
            "存在未到达业务终点的路径事件：{}".format(", ".join(failed_routes))
        )
    if weighted_mean >= 200.0:
        reasons.append("固定权重业务端到端均值未低于 200 ms")
    if ci_high >= 200.0:
        reasons.append("分组 bootstrap 95% CI 上界未低于 200 ms")
    return not reasons, metrics, reasons


def _objective_members(value: Any, name: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise InvalidEvidence("{} 必须是非空数组".format(name))
    members: List[str] = []
    for index, member in enumerate(value):
        if not isinstance(member, str) or not member.strip():
            raise InvalidEvidence(
                "{}[{}] 必须是非空字符串".format(name, index)
            )
        if member in members:
            raise InvalidEvidence("{} 不能包含重复成员".format(name))
        members.append(member)
    return members


def _objective_sample_ids(value: Any, name: str) -> List[int]:
    if not isinstance(value, list) or not value:
        raise InvalidEvidence("{} 必须是非空数组".format(name))
    sample_ids = [_count(sample_id, name) for sample_id in value]
    if any(sample_id < 0 for sample_id in sample_ids):
        raise InvalidEvidence("{} 不能包含负数".format(name))
    if len(sample_ids) != len(set(sample_ids)):
        raise InvalidEvidence("{} 不能包含重复 sample_id".format(name))
    return sample_ids


def _objective_plan_contract(
    evidence: Mapping[str, Any],
    sample_plan: Mapping[str, Any],
    sample_plan_sha256: str,
    *,
    objective_id: str,
    definition_sha: str,
) -> Tuple[str, List[int], int, float]:
    if sample_plan.get("schema_version") != "1.0":
        raise InvalidEvidence("全局目标样本计划 schema_version 必须是 1.0")
    if sample_plan.get("scene") != "freeway_traffic":
        raise InvalidEvidence("全局目标样本计划 scene 必须是 freeway_traffic")
    plan_id = str(sample_plan.get("plan_id", "")).strip()
    if not plan_id:
        raise InvalidEvidence("全局目标样本计划 plan_id 不能为空")
    plan_sample_ids = _objective_sample_ids(
        sample_plan.get("sample_ids"), "sample_plan.sample_ids"
    )
    expected_count = _count(
        sample_plan.get("expected_sample_count"),
        "sample_plan.expected_sample_count",
    )
    if expected_count <= 0 or expected_count != len(plan_sample_ids):
        raise InvalidEvidence("预期样本数必须等于预注册 sample_id 数量")
    slow_threshold_ms = _number(
        sample_plan.get("slow_threshold_ms"), "sample_plan.slow_threshold_ms"
    )
    if slow_threshold_ms <= 0.0:
        raise InvalidEvidence("slow_threshold_ms 必须大于 0")

    provenance = _object(evidence.get("provenance"), "provenance")
    plan_expected = _object(sample_plan.get("expected"), "sample_plan.expected")
    expected_bindings = {
        "git_commit": provenance.get("git_commit"),
        "dataset_id": provenance.get("dataset_id"),
        "model_ids": provenance.get("model_ids"),
        "hardware_id": provenance.get("hardware_id"),
        "objective_id": objective_id,
        "objective_definition_sha256": definition_sha,
    }
    for field, actual in expected_bindings.items():
        planned = plan_expected.get(field)
        if field == "objective_definition_sha256":
            planned = str(planned or "").lower()
        if _canonical(planned) != _canonical(actual):
            raise InvalidEvidence(
                "全局目标样本计划 expected.{} 与实测不一致".format(field)
            )

    binding = _object(evidence.get("sample_plan"), "sample_plan binding")
    if (
        str(binding.get("plan_id", "")) != plan_id
        or str(binding.get("sha256", "")).lower() != sample_plan_sha256
        or str(binding.get("locked_at", ""))
        != str(sample_plan.get("locked_at", ""))
        or _count(binding.get("expected_sample_count"), "binding.expected_sample_count")
        != expected_count
        or _objective_sample_ids(binding.get("sample_ids"), "binding.sample_ids")
        != plan_sample_ids
        or not math.isclose(
            _number(binding.get("slow_threshold_ms"), "binding.slow_threshold_ms"),
            slow_threshold_ms,
            abs_tol=1e-12,
        )
    ):
        raise InvalidEvidence("全局目标证据没有精确绑定预注册样本计划")
    locked_at = _parse_time(sample_plan.get("locked_at"), "sample_plan.locked_at")
    run_started_at = _parse_time(
        provenance.get("run_started_at"), "provenance.run_started_at"
    )
    generated_at = _parse_time(
        provenance.get("generated_at"), "provenance.generated_at"
    )
    if not locked_at < run_started_at <= generated_at:
        raise InvalidEvidence("全局目标样本计划必须在测量开始前锁定")
    return plan_id, plan_sample_ids, expected_count, slow_threshold_ms


def _evaluate_objective(
    evidence: Mapping[str, Any],
    definition_sha: str,
    definition: Mapping[str, Any],
    sample_plan: Mapping[str, Any],
    sample_plan_sha256: str,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    objective_id = str(evidence.get("objective_id", ""))
    reported_sha = str(evidence.get("objective_definition_sha256", "")).lower()
    records = evidence.get("records")
    if not objective_id:
        raise InvalidEvidence("objective_id 不能为空")
    if reported_sha != definition_sha:
        raise InvalidEvidence("objective definition SHA 与实物不一致")
    if str(definition.get("objective_id", "")) != objective_id:
        raise InvalidEvidence("objective_id 与目标定义文件不一致")
    definition_members = _objective_members(
        definition.get("expected_members"),
        "objective definition.expected_members",
    )
    plan_id, plan_sample_ids, expected_sample_count, slow_threshold_ms = (
        _objective_plan_contract(
            evidence,
            sample_plan,
            sample_plan_sha256,
            objective_id=objective_id,
            definition_sha=definition_sha,
        )
    )
    if not isinstance(records, list):
        raise InvalidEvidence("全局目标证据 records 必须是数组")
    attempts = evidence.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != expected_sample_count:
        raise InvalidEvidence("全局目标 attempts 必须覆盖预注册样本总数")
    attempt_by_sample: Dict[int, Mapping[str, Any]] = {}
    derived_outcome_counts = {
        "authoritative_optimized": 0,
        "partial": 0,
        "failed": 0,
    }
    derived_slow_count = 0
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            raise InvalidEvidence("objective attempts[{}] 必须是对象".format(index))
        sample_id = _count(
            attempt.get("sample_id"), "objective attempts[{}].sample_id".format(index)
        )
        if sample_id < 0 or sample_id in attempt_by_sample:
            raise InvalidEvidence("objective attempts 包含负数或重复 sample_id")
        source_experiment_id = str(attempt.get("source_experiment_id", "")).strip()
        if not source_experiment_id:
            raise InvalidEvidence("objective attempts 必须声明 source_experiment_id")
        outcome = str(attempt.get("outcome", ""))
        if outcome not in derived_outcome_counts:
            raise InvalidEvidence("objective attempts 包含未知 outcome")
        group_ids = attempt.get("aggregation_group_ids")
        if not isinstance(group_ids, list):
            raise InvalidEvidence("objective attempts.aggregation_group_ids 必须是数组")
        normalized_group_ids = [str(group_id).strip() for group_id in group_ids]
        if (
            any(not group_id for group_id in normalized_group_ids)
            or len(normalized_group_ids) != len(set(normalized_group_ids))
        ):
            raise InvalidEvidence("objective attempts 的 aggregation_group_ids 无效或重复")
        errors = attempt.get("errors")
        if not isinstance(errors, list):
            raise InvalidEvidence("objective attempts.errors 必须是数组")
        latency_value = attempt.get("global_authoritative_final_ms")
        latency = None
        if latency_value is not None:
            latency = _number(latency_value, "global_authoritative_final_ms")
            if latency < 0.0:
                raise InvalidEvidence("global_authoritative_final_ms 不能为负数")
        slow = attempt.get("slow")
        if not isinstance(slow, bool) or slow is not (
            latency is not None and latency >= slow_threshold_ms
        ):
            raise InvalidEvidence("objective attempts.slow 与预注册阈值不一致")
        if outcome == "authoritative_optimized" and latency is None:
            raise InvalidEvidence("完整权威样本必须保留 global final 时延")
        derived_outcome_counts[outcome] += 1
        derived_slow_count += int(slow)
        attempt_by_sample[sample_id] = attempt
    if set(attempt_by_sample) != set(plan_sample_ids):
        raise InvalidEvidence("objective attempts 未逐项覆盖预注册 sample_id 总体")
    declared_outcomes = _object(evidence.get("outcome_counts"), "outcome_counts")
    if set(declared_outcomes) != set(derived_outcome_counts) or any(
        _count(declared_outcomes.get(name), "outcome_counts.{}".format(name))
        != count
        for name, count in derived_outcome_counts.items()
    ):
        raise InvalidEvidence("outcome_counts 与逐样本 attempts 不一致")
    if _count(evidence.get("slow_sample_count"), "slow_sample_count") != derived_slow_count:
        raise InvalidEvidence("slow_sample_count 与逐样本 attempts 不一致")

    nondecreasing = True
    exact_search = True
    candidate_set_complete = True
    constraints_satisfied = True
    active_and_applied = True
    authoritative_input = True
    enumeration_complete = True
    definition_linked = True
    selected_plan_identified = True
    improvements = []
    declared_members: Optional[List[str]] = None
    seen_group_ids = set()
    seen_record_samples = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise InvalidEvidence("objective records[{}] 必须是对象".format(index))
        sample_id = _count(
            record.get("sample_id"), "objective records[{}].sample_id".format(index)
        )
        if sample_id in seen_record_samples or sample_id not in attempt_by_sample:
            raise InvalidEvidence("objective records 包含重复或计划外 sample_id")
        attempt = attempt_by_sample[sample_id]
        if attempt.get("outcome") != "authoritative_optimized":
            raise InvalidEvidence("partial/failed 样本不能伪装成优化成功记录")
        if str(record.get("source_experiment_id", "")) != str(
            attempt.get("source_experiment_id", "")
        ):
            raise InvalidEvidence("objective record 与 attempt 的来源实验不一致")
        seen_record_samples.add(sample_id)
        group_id = record.get("group_id")
        if not isinstance(group_id, str) or not group_id.strip():
            raise InvalidEvidence(
                "objective records[{}].group_id 不能为空".format(index)
            )
        if group_id in seen_group_ids:
            raise InvalidEvidence("objective records 包含重复 group_id")
        if [group_id] != list(attempt.get("aggregation_group_ids", [])):
            raise InvalidEvidence("objective record 的 group_id 与 attempt 不一致")
        seen_group_ids.add(group_id)
        expected_members = _objective_members(
            record.get("expected_members"),
            "objective records[{}].expected_members".format(index),
        )
        observed_members = _objective_members(
            record.get("observed_members"),
            "objective records[{}].observed_members".format(index),
        )
        if observed_members != expected_members:
            raise InvalidEvidence(
                "objective records[{}] expected_members/observed_members "
                "必须精确相等".format(index)
            )
        expected_member_count = _count(
            record.get("expected_member_count"),
            "objective records[{}].expected_member_count".format(index),
        )
        observed_member_count = _count(
            record.get("observed_member_count"),
            "objective records[{}].observed_member_count".format(index),
        )
        if expected_member_count != len(expected_members):
            raise InvalidEvidence(
                "objective records[{}] expected_member_count 与成员声明不一致".format(
                    index
                )
            )
        if observed_member_count != len(observed_members):
            raise InvalidEvidence(
                "objective records[{}] observed_member_count 与成员声明不一致".format(
                    index
                )
            )
        if declared_members is None:
            declared_members = expected_members
        elif expected_members != declared_members:
            raise InvalidEvidence("objective records 的 expected_members 声明不一致")
        if expected_members != definition_members:
            raise InvalidEvidence(
                "objective records[{}] expected_members 与目标定义不一致".format(
                    index
                )
            )
        exact_search = exact_search and record.get("exact_search") is True
        candidate_set_complete = (
            candidate_set_complete and record.get("candidate_set_complete") is True
        )
        constraints_satisfied = constraints_satisfied and record.get("constraints_satisfied") is True
        active_and_applied = active_and_applied and (
            record.get("mode") == "active" and record.get("applied") is True
        )
        authoritative_input = (
            authoritative_input and record.get("authoritative_input") is True
        )
        definition_linked = definition_linked and (
            str(record.get("objective_definition_sha256", "")).lower()
            == definition_sha
        )
        candidate_space = _count(
            record.get("candidate_space_size"), "candidate_space_size"
        )
        evaluated_count = _count(
            record.get("evaluated_candidate_count"), "evaluated_candidate_count"
        )
        enumeration_complete = enumeration_complete and (
            candidate_space > 0 and evaluated_count == candidate_space
        )
        plan_sha = str(record.get("selected_plan_sha256", "")).lower()
        selected_plan_identified = selected_plan_identified and (
            len(plan_sha) == 64
            and all(character in "0123456789abcdef" for character in plan_sha)
        )
        baseline = _number(record.get("baseline_utility"), "baseline_utility")
        selected = _number(record.get("selected_utility"), "selected_utility")
        improvement = selected - baseline
        improvements.append(improvement)
        if improvement < -1e-12:
            nondecreasing = False
    successful_samples = {
        sample_id
        for sample_id, attempt in attempt_by_sample.items()
        if attempt.get("outcome") == "authoritative_optimized"
    }
    if seen_record_samples != successful_samples:
        raise InvalidEvidence("每个权威成功 attempt 必须且只能对应一条 objective record")
    metrics = {
        "objective_id": objective_id,
        "objective_definition_sha256": definition_sha,
        "sample_plan_id": plan_id,
        "sample_plan_sha256": sample_plan_sha256,
        "expected_sample_count": expected_sample_count,
        "attempt_count": len(attempts),
        "outcome_counts": derived_outcome_counts,
        "slow_threshold_ms": slow_threshold_ms,
        "slow_sample_count": derived_slow_count,
        "full_denominator_covered": True,
        "record_count": len(records),
        "expected_members": declared_members or definition_members,
        "expected_member_count": len(definition_members),
        "unique_group_count": len(seen_group_ids),
        "exact_search_all": exact_search,
        "candidate_set_complete_all": candidate_set_complete,
        "constraints_satisfied_all": constraints_satisfied,
        "active_and_applied_all": active_and_applied,
        "authoritative_input_all": authoritative_input,
        "objective_definition_linked_all": definition_linked,
        "enumeration_complete_all": enumeration_complete,
        "selected_plan_identified_all": selected_plan_identified,
        "utility_nondecreasing_all": nondecreasing,
        "mean_utility_improvement": _mean(improvements) if improvements else None,
        "minimum_utility_improvement": min(improvements) if improvements else None,
    }
    reasons = []
    if derived_outcome_counts["partial"]:
        reasons.append("预注册样本中存在 partial 汇聚")
    if derived_outcome_counts["failed"]:
        reasons.append("预注册样本中存在失败样本")
    if not records:
        reasons.append("没有样本形成完整权威联合优化记录")
    if not exact_search:
        reasons.append("不是精确搜索")
    if not candidate_set_complete:
        reasons.append("不是完整有限候选集上的精确搜索")
    if not constraints_satisfied:
        reasons.append("存在不满足约束的选中方案")
    if not active_and_applied:
        reasons.append("存在未以 active 模式实际应用的联合方案")
    if not authoritative_input:
        reasons.append("存在并非基于完整权威汇聚的联合方案")
    if not definition_linked:
        reasons.append("实测记录未绑定当前全局目标定义")
    if not enumeration_complete:
        reasons.append("有限候选集合未被完整枚举")
    if not selected_plan_identified:
        reasons.append("选中联合方案缺少有效 SHA-256 标识")
    if not nondecreasing:
        reasons.append("选中方案效用低于 baseline")
    return not reasons, metrics, reasons


def _evaluate_update_loop(
    evidence: Mapping[str, Any],
    stage_hashes: Mapping[str, str],
    stage_documents: Mapping[str, Mapping[str, Any]],
    completion: Mapping[str, Any],
    completion_sha256: str,
    expected: Mapping[str, Any],
    main_sha256: str,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Validate the real local apply/rollback lifecycle, not self-reported flags."""
    schema = "edge-llm-update-loop-evidence/v1"
    required = ("package", "release", "apply", "rollback")
    stages = _object(evidence.get("stages"), "模型更新证据 stages")
    if evidence.get("schema_version") != schema:
        raise InvalidEvidence("模型更新主证据 schema_version 不兼容")
    if evidence.get("execution_mode") != "real_llama_server":
        raise InvalidEvidence("模型更新主证据不是 real_llama_server")
    if evidence.get("fault_mode") != "post_health_inference_gate_failure":
        raise InvalidEvidence("模型更新主证据故障模式不兼容")
    provenance = _object(evidence.get("provenance"), "模型更新 provenance")
    stage_provenance = {
        key: provenance[key]
        for key in ("git_commit", "run_id", "dataset_id", "hardware_id")
    }
    for stage in required:
        declared = _object(stages.get(stage), "主证据 {} stage".format(stage))
        document = _object(stage_documents.get(stage), "{} 阶段证据".format(stage))
        if declared.get("status") != "completed":
            raise InvalidEvidence("{} 阶段主证据状态不是 completed".format(stage))
        if str(declared.get("evidence_sha256", "")).lower() != stage_hashes[stage]:
            raise InvalidEvidence("{} 阶段主证据 SHA 与文件不一致".format(stage))
        if (
            document.get("schema_version") != schema
            or document.get("stage") != stage
            or document.get("status") != "completed"
        ):
            raise InvalidEvidence("{} 阶段 JSON 合同不兼容".format(stage))
        if _canonical(document.get("provenance")) != _canonical(stage_provenance):
            raise InvalidEvidence("{} 阶段 provenance 与主证据不一致".format(stage))

    before_model = str(evidence.get("model_id_before", ""))
    applied_model = str(evidence.get("model_id_applied", ""))
    after_rollback = str(evidence.get("model_id_after_rollback", ""))
    if not before_model or not applied_model or before_model == applied_model:
        raise InvalidEvidence("缺少可区分的更新前/更新后模型 ID")
    if after_rollback != before_model:
        raise InvalidEvidence("回滚后模型 ID 未恢复")
    expected_models = expected.get("model_ids")
    if expected_models != {"before": before_model, "applied": applied_model}:
        raise InvalidEvidence("模型更新 ID 与 manifest 预登记不一致")

    package = stage_documents["package"]
    if (
        package.get("execution_mode") != "real_llama_server"
        or package.get("old_release_id") != before_model
        or package.get("candidate_release_id") != applied_model
    ):
        raise InvalidEvidence("package 阶段 release ID 或执行模式不一致")
    identities = {}
    for label in ("old", "candidate"):
        identity = _object(package.get(label), "package.{}".format(label))
        for section in ("base", "package", "artifact"):
            record = _object(identity.get(section), "package.{}.{}".format(label, section))
            if not _is_sha256(record.get("sha256")) or _count(
                record.get("bytes"), "package.{}.{}.bytes".format(label, section)
            ) <= 0:
                raise InvalidEvidence("package {} {} 资产身份无效".format(label, section))
        if _count(identity["package"].get("file_count"), "package file_count") <= 0:
            raise InvalidEvidence("package 文件数必须大于 0")
        gates = identity.get("gate_results")
        if not isinstance(gates, list) or not gates or not all(
            isinstance(row, Mapping) and row.get("passed") is True for row in gates
        ):
            raise InvalidEvidence("package {} 发布门槛未全部通过".format(label))
        contract = _object(identity.get("decision_contract"), "decision_contract")
        if (
            contract.get("max_output_tokens") != 1
            or not isinstance(contract.get("accepted_tokens"), list)
            or not contract.get("accepted_tokens")
            or not _is_sha256(contract.get("action_mapping_sha256"))
        ):
            raise InvalidEvidence("package {} 决策契约无效".format(label))
        identities[label] = identity
    if identities["old"]["artifact"]["sha256"] == identities["candidate"]["artifact"]["sha256"]:
        raise InvalidEvidence("旧版与候选 GGUF SHA 必须不同")
    for field in ("scene", "base_fingerprint"):
        if identities["old"].get(field) != identities["candidate"].get(field):
            raise InvalidEvidence("旧版与候选 {} 不兼容".format(field))
    if _canonical(identities["old"]["decision_contract"]) != _canonical(
        identities["candidate"]["decision_contract"]
    ):
        raise InvalidEvidence("旧版与候选决策契约不兼容")
    if (
        identities["old"].get("adapter_id"),
        identities["old"].get("adapter_version"),
    ) == (
        identities["candidate"].get("adapter_id"),
        identities["candidate"].get("adapter_version"),
    ):
        raise InvalidEvidence("旧版与候选 adapter ID/version 不可区分")

    release = stage_documents["release"]
    if (
        release.get("old_release_id") != before_model
        or release.get("candidate_release_id") != applied_model
    ):
        raise InvalidEvidence("release 阶段 model ID 不一致")
    old_promotion = _object(release.get("old_promotion"), "old_promotion")
    candidate_promotion = _object(release.get("candidate_promotion"), "candidate_promotion")
    for label, promotion, model_id, revision in (
        ("old", old_promotion, before_model, 1),
        ("candidate", candidate_promotion, applied_model, 2),
    ):
        if (
            promotion.get("status") != "promoted"
            or promotion.get("active_release_id") != model_id
            or promotion.get("revision") != revision
            or promotion.get("artifact_sha256") != identities[label]["artifact"]["sha256"]
            or promotion.get("package_sha256") != identities[label]["package"]["sha256"]
            or not _is_sha256(promotion.get("binding_fingerprint"))
        ):
            raise InvalidEvidence("{} promotion 与 package 阶段不一致".format(label))
    history = release.get("release_history")
    expected_history = (
        (1, "promote", None, before_model),
        (2, "promote", before_model, applied_model),
        (3, "rollback", applied_model, before_model),
    )
    if not isinstance(history, list) or len(history) != 3:
        raise InvalidEvidence("release history 必须恰好包含三次状态迁移")
    for row, expected_row in zip(history, expected_history):
        if (
            row.get("sequence"),
            row.get("action"),
            row.get("from_release_id"),
            row.get("to_release_id"),
        ) != expected_row:
            raise InvalidEvidence("release history 顺序或 revision 不正确")
    audit = _object(history[-1].get("audit"), "rollback audit")
    if (
        audit.get("trigger") != "candidate_apply_failure"
        or audit.get("failed_release_id") != applied_model
        or audit.get("failed_revision") != 2
    ):
        raise InvalidEvidence("rollback audit 与候选故障不一致")

    apply_stage = stage_documents["apply"]
    rollback_stage = stage_documents["rollback"]
    llama = _object(apply_stage.get("llama_server"), "llama_server")
    if not _is_sha256(llama.get("sha256")) or _count(llama.get("bytes"), "llama bytes") <= 0:
        raise InvalidEvidence("llama-server 资产身份无效")
    expected_binary = str(expected.get("llama_server_sha256", "")).lower()
    if not _is_sha256(expected_binary) or llama.get("sha256") != expected_binary:
        raise InvalidEvidence("llama-server SHA 未预登记或不一致")

    transitions = apply_stage.get("transitions")
    if not isinstance(transitions, list) or len(transitions) != 2:
        raise InvalidEvidence("apply 阶段必须恰好包含旧版和候选两次启动")
    rollback_transition = _object(
        rollback_stage.get("rollback_transition"), "rollback_transition"
    )
    all_transitions = list(transitions) + [rollback_transition]
    prompt_sha = None
    pids = []
    for index, (transition, label, model_id, revision) in enumerate(
        zip(
            all_transitions,
            ("old", "candidate", "old"),
            (before_model, applied_model, before_model),
            (1, 2, 3),
        )
    ):
        row = _object(transition, "transition[{}]".format(index))
        pid = _count(row.get("pid"), "transition pid")
        if pid <= 0 or pid in pids:
            raise InvalidEvidence("三次模型启动必须有不同的正 PID")
        pids.append(pid)
        identity = identities[label]
        contract = identity["decision_contract"]
        endpoint = str(row.get("endpoint", ""))
        if not (endpoint.startswith("http://127.0.0.1:") or endpoint.startswith("http://localhost:")):
            raise InvalidEvidence("模型更新探针只能使用 loopback endpoint")
        if (
            row.get("status") != "active"
            or row.get("release_id") != model_id
            or row.get("revision") != revision
            or row.get("artifact_sha256") != identity["artifact"]["sha256"]
            or row.get("package_sha256") != identity["package"]["sha256"]
            or row.get("binding_fingerprint")
            != (old_promotion if label == "old" else candidate_promotion).get(
                "binding_fingerprint"
            )
            or row.get("health_verified") is not True
            or str(row.get("process_executable", "")) != str(llama.get("path", ""))
        ):
            raise InvalidEvidence("模型启动 transition 与资产/release 不一致")
        probe = _object(row.get("inference_probe"), "inference_probe")
        current_prompt_sha = str(probe.get("prompt_sha256", "")).lower()
        if prompt_sha is None:
            prompt_sha = current_prompt_sha
        if (
            not _is_sha256(current_prompt_sha)
            or current_prompt_sha != prompt_sha
            or probe.get("http_status") != 200
            or _count(probe.get("prompt_tokens"), "prompt_tokens") <= 0
            or _count(probe.get("prompt_tokens"), "prompt_tokens")
            > _count(contract.get("max_input_tokens"), "max_input_tokens")
            or probe.get("output_tokens") != 1
            or probe.get("action_mapping_accepted") is not True
            or probe.get("action_token") not in contract["accepted_tokens"]
        ):
            raise InvalidEvidence("模型推理探针未满足单动作 token 契约")

    if (
        apply_stage.get("execution_mode") != "real_llama_server"
        or apply_stage.get("candidate_health_and_inference_verified") is not True
        or apply_stage.get("fault_mode") != "post_health_inference_gate_failure"
    ):
        raise InvalidEvidence("apply 阶段执行模式或故障注入顺序无效")
    if (
        rollback_stage.get("execution_mode") != "real_llama_server"
        or rollback_stage.get("fault_mode") != "post_health_inference_gate_failure"
        or rollback_stage.get("candidate_process_stopped") is not True
        or rollback_stage.get("rollback_verified") is not True
        or rollback_stage.get("isolated_runtime_stopped_after_evidence") is not True
    ):
        raise InvalidEvidence("rollback 阶段未证明候选退出与自动恢复")
    registry = _object(rollback_stage.get("registry"), "rollback registry")
    if (
        registry.get("active_release_id") != before_model
        or registry.get("revision") != 3
        or _canonical(registry.get("history")) != _canonical(history)
    ):
        raise InvalidEvidence("rollback registry 未恢复 old@revision3")
    runtime_config = _object(
        rollback_stage.get("runtime_config_after_rollback"), "rollback runtime config"
    )
    if runtime_config.get("model") != identities["old"]["artifact"]["path"]:
        raise InvalidEvidence("rollback runtime config 未绑定旧 GGUF 绝对路径")
    supervisor = _object(
        rollback_stage.get("supervisor_after_rollback"), "rollback supervisor"
    )
    last_failure = _object(supervisor.get("last_failure"), "last_failure")
    rollback_result = _object(last_failure.get("rollback"), "last_failure.rollback")
    if (
        supervisor.get("status") != "recovered"
        or supervisor.get("process_running") is not True
        or supervisor.get("process_healthy") is not True
        or supervisor.get("registry_active_release_id") != before_model
        or supervisor.get("applied_release_id") != before_model
        or supervisor.get("registry_revision") != 3
        or supervisor.get("applied_revision") != 3
        or rollback_result.get("status") != "rolled_back"
    ):
        raise InvalidEvidence("rollback 后 supervisor/registry/runtime 未收敛")
    cleanup = _object(rollback_stage.get("runtime_cleanup"), "runtime_cleanup")
    if not (
        cleanup.get("passed") is True
        and cleanup.get("all_tracked_pids_exited") is True
        and cleanup.get("port_released") is True
        and cleanup.get("tracked_pids") == pids
    ):
        raise InvalidEvidence("取证进程或端口未完成清理")

    if (
        completion.get("schema_version") != schema
        or completion.get("status") != "completed"
        or completion.get("execution_mode") != "real_llama_server"
        or completion.get("git_commit") != provenance.get("git_commit")
        or completion.get("run_id") != provenance.get("run_id")
        or completion.get("main_evidence_sha256") != main_sha256
        or completion.get("stage_sha256") != dict(stage_hashes)
        or not _is_sha256(completion_sha256)
    ):
        raise InvalidEvidence("模型更新完成标记与主/阶段证据不一致")
    if evidence.get("rollback_verified") is not True:
        raise InvalidEvidence("模型更新主证据未声明回滚验证完成")
    metrics = {
        "stages": {
            stage: {"status": "completed", "evidence_sha256": stage_hashes[stage]}
            for stage in required
        },
        "completion_marker_sha256": completion_sha256,
        "rollback_verified": True,
        "model_id_before": before_model,
        "model_id_applied": applied_model,
        "model_id_after_rollback": after_rollback,
        "transition_pids": pids,
        "runtime_cleanup_passed": True,
    }
    return True, metrics, []


def _evaluate_distributed_update_loop(
    evidence: Mapping[str, Any],
    completion: Mapping[str, Any],
    completion_sha256: str,
    expected: Mapping[str, Any],
    local_evidence: Mapping[str, Any],
    local_main_sha256: str,
    local_stage_sha256: Mapping[str, str],
    local_stage_documents: Mapping[str, Mapping[str, Any]],
    local_completion_sha256: str,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Validate cloud publication, edge transfer, apply/rollback and cloud receipt links."""
    distributed_schema = "edge-llm-distributed-update-evidence/v1"
    publication_schema = "edge-llm-cloud-publication/v1"
    receipt_schema = "edge-llm-edge-receipt/v1"
    ack_schema = "edge-llm-cloud-ack/v1"
    if (
        evidence.get("schema_version") != distributed_schema
        or evidence.get("status") != "completed"
        or evidence.get("execution_mode") != "real_cloud_edge_http"
        or evidence.get("cloud_received_edge_confirmation") is not True
    ):
        raise InvalidEvidence("分布式模型更新主证据不是正式完成状态")

    provenance = _object(evidence.get("provenance"), "分布式模型更新 provenance")
    local_provenance = _object(local_evidence.get("provenance"), "本地模型更新 provenance")
    for field in ("git_commit", "dataset_id", "hardware_id"):
        if provenance.get(field) != expected.get(field):
            raise InvalidEvidence("分布式模型更新 provenance.{} 与预登记不一致".format(field))
        if provenance.get(field) != local_provenance.get(field):
            raise InvalidEvidence("分布式与本地模型更新 provenance.{} 不一致".format(field))
    run_id = str(provenance.get("run_id", ""))
    edge_id = str(provenance.get("edge_id", ""))
    if not run_id or not edge_id or run_id != str(local_provenance.get("run_id", "")):
        raise InvalidEvidence("分布式模型更新 run_id/edge_id 无效")

    cloud = _object(evidence.get("cloud_publication"), "cloud_publication")
    publication = _object(cloud.get("publication"), "cloud_publication.publication")
    publication_sha = str(cloud.get("publication_sha256", "")).lower()
    if (
        publication.get("schema_version") != publication_schema
        or publication.get("status") != "published"
        or publication.get("execution_mode") != "real_http"
        or publication.get("run_id") != run_id
        or _canonical_json_sha256(publication) != publication_sha
        or cloud.get("manifest_hmac_verified") is not True
        or not _is_sha256(publication_sha)
    ):
        raise InvalidEvidence("云端发布清单的结构、SHA 或运行时 HMAC 结论无效")
    manifest_url = urlparse(str(cloud.get("manifest_url", "")))
    if (
        manifest_url.scheme != "http"
        or not manifest_url.hostname
        or manifest_url.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
        or not manifest_url.path.endswith("/{}/manifest".format(run_id))
    ):
        raise InvalidEvidence("正式云端发布必须来自非回环 HTTP 地址")
    key_id = str(cloud.get("hmac_key_id", "")).lower()
    if len(key_id) != 16 or any(character not in "0123456789abcdef" for character in key_id):
        raise InvalidEvidence("云端发布 HMAC key id 无效")

    if (
        publication.get("expected_edge_id") != edge_id
        or publication.get("expected_hardware_id") != expected.get("hardware_id")
        or publication.get("expected_dataset_id") != expected.get("dataset_id")
    ):
        raise InvalidEvidence("云端发布清单指定的边缘/硬件/数据集不匹配")
    publisher = _object(publication.get("publisher"), "publication.publisher")
    if publisher.get("git_commit") != expected.get("git_commit"):
        raise InvalidEvidence("云端发布清单 Git commit 不匹配")
    source_hashes = _object(publisher.get("source_sha256"), "publisher.source_sha256")
    required_sources = {
        "edge_llm_factory/distributed_update_evidence.py",
        "scripts/measure_distributed_model_update_loop.py",
    }
    if not required_sources.issubset(source_hashes) or not all(
        _is_sha256(source_hashes[name]) for name in required_sources
    ):
        raise InvalidEvidence("云端发布没有绑定分布式 runner 的 Git 源码 SHA")

    before_model = str(local_evidence.get("model_id_before", ""))
    applied_model = str(local_evidence.get("model_id_applied", ""))
    releases = _object(publication.get("releases"), "publication.releases")
    old_release = _object(releases.get("old"), "publication.releases.old")
    candidate_release = _object(
        releases.get("candidate"), "publication.releases.candidate"
    )
    if (
        publication.get("release_order") != [before_model, applied_model, before_model]
        or old_release.get("release_id") != before_model
        or candidate_release.get("release_id") != applied_model
    ):
        raise InvalidEvidence("云端发布顺序与本地 apply/rollback 模型不一致")
    package_stage = _object(
        local_stage_documents.get("package"), "本地 package 阶段证据"
    )
    for role, published, local_label in (
        ("old", old_release, "old"),
        ("candidate", candidate_release, "candidate"),
    ):
        local_identity = _object(
            package_stage.get(local_label), "本地 package.{}".format(local_label)
        )
        expected_release_identity = {
            "adapter_id": local_identity.get("adapter_id"),
            "adapter_version": local_identity.get("adapter_version"),
            "scene": local_identity.get("scene"),
            "base_fingerprint": local_identity.get("base_fingerprint"),
            "base_sha256": _object(local_identity.get("base"), "base").get("sha256"),
            "package_sha256": _object(local_identity.get("package"), "package").get(
                "sha256"
            ),
            "artifact_sha256": _object(local_identity.get("artifact"), "artifact").get(
                "sha256"
            ),
            "artifact_bytes": _object(local_identity.get("artifact"), "artifact").get(
                "bytes"
            ),
            "decision_contract": local_identity.get("decision_contract"),
            "gate_results": local_identity.get("gate_results"),
        }
        for field, expected_value in expected_release_identity.items():
            if _canonical(published.get(field)) != _canonical(expected_value):
                raise InvalidEvidence(
                    "云端 {} 发布身份字段 {} 与本地 package 不一致".format(
                        role, field
                    )
                )

    files = publication.get("files")
    if not isinstance(files, list) or not files:
        raise InvalidEvidence("云端发布清单 files 不能为空")
    expected_files: Dict[str, Tuple[Any, Any, Any]] = {}
    target_paths = set()
    role_kinds = {
        "old": {"base_manifest": 0, "package_file": 0, "gguf": 0},
        "candidate": {"base_manifest": 0, "package_file": 0, "gguf": 0},
    }
    total_bytes = 0
    for index, raw in enumerate(files):
        row = _object(raw, "publication.files[{}]".format(index))
        file_id = str(row.get("file_id", ""))
        target_path = str(row.get("target_path", ""))
        role = str(row.get("role", ""))
        kind = str(row.get("kind", ""))
        size = _count(row.get("bytes"), "publication file bytes")
        digest = str(row.get("sha256", "")).lower()
        if (
            not file_id
            or file_id in expected_files
            or not target_path
            or target_path.startswith("/")
            or ".." in Path(target_path).parts
            or target_path in target_paths
            or role not in role_kinds
            or kind not in role_kinds[role]
            or size <= 0
            or not _is_sha256(digest)
            or row.get("download_path")
            != "/api/v1/model-updates/{}/files/{}".format(run_id, file_id)
        ):
            raise InvalidEvidence("云端发布文件 {} 的身份或下载路径无效".format(index))
        expected_files[file_id] = (digest, size, target_path)
        target_paths.add(target_path)
        role_kinds[role][kind] += 1
        total_bytes += size
    for role, counts in role_kinds.items():
        if counts["base_manifest"] != 1 or counts["gguf"] != 1 or counts["package_file"] < 1:
            raise InvalidEvidence("云端 {} 发布文件集合不完整".format(role))

    transfer = _object(evidence.get("transfer"), "distributed transfer")
    downloaded = transfer.get("downloaded_files")
    if (
        transfer.get("protocol") != "http"
        or transfer.get("real_http") is not True
        or transfer.get("atomic_download_publish") is not True
        or not isinstance(downloaded, list)
        or len(downloaded) != len(expected_files)
    ):
        raise InvalidEvidence("边缘下载没有证明真实 HTTP、完整文件与原子发布")
    observed_files = {}
    for index, raw in enumerate(downloaded):
        row = _object(raw, "downloaded_files[{}]".format(index))
        file_id = str(row.get("file_id", ""))
        if file_id in observed_files:
            raise InvalidEvidence("边缘下载回执存在重复 file_id")
        observed_files[file_id] = (
            str(row.get("sha256", "")).lower(),
            row.get("bytes"),
            row.get("target_path"),
        )
    if observed_files != expected_files:
        raise InvalidEvidence("边缘逐文件 SHA/字节数/目标路径与云端发布不一致")

    local_summary = _object(evidence.get("local_execution"), "local_execution")
    if (
        local_summary.get("execution_mode") != "real_llama_server"
        or local_summary.get("rollback_verified") is not True
        or local_summary.get("model_id_before") != before_model
        or local_summary.get("model_id_applied") != applied_model
        or local_summary.get("model_id_after_rollback") != before_model
        or local_summary.get("transition_release_ids")
        != [before_model, applied_model, before_model]
        or local_summary.get("transition_revisions") != [1, 2, 3]
        or local_summary.get("release_actions") != ["promote", "promote", "rollback"]
        or local_summary.get("candidate_process_stopped") is not True
        or local_summary.get("runtime_cleanup_passed") is not True
        or local_summary.get("main_evidence_sha256") != local_main_sha256
        or local_summary.get("stage_sha256") != dict(local_stage_sha256)
        or local_summary.get("completion_marker_sha256") != local_completion_sha256
    ):
        raise InvalidEvidence("分布式证据中的本地 apply/rollback 摘要与本地原件不一致")
    summary_provenance = _object(
        local_summary.get("provenance"), "local_execution.provenance"
    )
    for field in ("git_commit", "dataset_id", "hardware_id", "run_id"):
        if summary_provenance.get(field) != local_provenance.get(field):
            raise InvalidEvidence("本地执行摘要 provenance.{} 不一致".format(field))

    receipt = _object(evidence.get("edge_receipt"), "edge_receipt")
    receipt_core = dict(receipt)
    receipt_id = str(receipt_core.pop("receipt_id", ""))
    if (
        receipt.get("schema_version") != receipt_schema
        or receipt.get("status") != "completed"
        or receipt.get("execution_mode") != "real_cloud_edge_http"
        or receipt.get("run_id") != run_id
        or receipt.get("edge_id") != edge_id
        or receipt.get("hardware_id") != expected.get("hardware_id")
        or receipt.get("dataset_id") != expected.get("dataset_id")
        or receipt.get("git_commit") != expected.get("git_commit")
        or receipt.get("publication_nonce") != publication.get("publication_nonce")
        or receipt.get("publication_sha256") != publication_sha
        or receipt.get("download_atomic_publish") is not True
        or _canonical(receipt.get("downloaded_files")) != _canonical(downloaded)
        or _canonical(receipt.get("local_execution")) != _canonical(local_summary)
        or receipt_id != _canonical_json_sha256(receipt_core)
    ):
        raise InvalidEvidence("边缘正式回执与发布、下载或本地执行证据不一致")

    cloud_ack = _object(evidence.get("cloud_ack"), "cloud_ack")
    ack = _object(cloud_ack.get("ack"), "cloud_ack.ack")
    ack_url = urlparse(str(cloud_ack.get("url", "")))
    receipt_sha = _canonical_json_sha256(receipt)
    ack_sha = _canonical_json_sha256(ack)
    if (
        cloud_ack.get("http_status") != 200
        or cloud_ack.get("hmac_verified") is not True
        or cloud_ack.get("request_sha256") != receipt_sha
        or cloud_ack.get("ack_sha256") != ack_sha
        or ack_url.scheme != "http"
        or not ack_url.hostname
        or ack_url.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
        or not ack_url.path.endswith("/{}/receipts".format(run_id))
        or ack.get("schema_version") != ack_schema
        or ack.get("status") != "accepted"
        or ack.get("execution_mode") != "real_http"
        or ack.get("run_id") != run_id
        or ack.get("edge_id") != edge_id
        or ack.get("receipt_id") != receipt_id
        or ack.get("receipt_sha256") != receipt_sha
        or ack.get("publication_sha256") != publication_sha
        or _count(ack.get("store_revision"), "cloud ack store_revision") <= 0
    ):
        raise InvalidEvidence("云端持久接收确认与边缘回执不一致")

    if (
        completion.get("schema_version") != distributed_schema
        or completion.get("status") != "completed"
        or completion.get("execution_mode") != "real_cloud_edge_http"
        or completion.get("run_id") != run_id
        or completion.get("distributed_evidence_sha256")
        != _canonical_json_sha256(evidence)
        or completion.get("cloud_ack_sha256") != _canonical_json_sha256(cloud_ack)
        or not _is_sha256(completion_sha256)
    ):
        raise InvalidEvidence("分布式模型更新完成标记与证据或云端确认不一致")
    metrics = {
        "distributed_completion_marker_sha256": completion_sha256,
        "publication_sha256": publication_sha,
        "edge_id": edge_id,
        "download_file_count": len(expected_files),
        "download_bytes": total_bytes,
        "download_atomic_publish": True,
        "manifest_hmac_verified_during_run": True,
        "receipt_hmac_verified_during_run": True,
        "receipt_id": receipt_id,
        "cloud_store_revision": ack["store_revision"],
        "cloud_received_edge_confirmation": True,
    }
    return True, metrics, []


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "schemas" / "competition_targets_manifest.schema.json"


def load_manifest(path: Path) -> Dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidEvidence("无法读取 manifest：{}".format(exc))
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(manifest),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        details = "; ".join(error.message for error in errors[:5])
        raise InvalidEvidence("manifest 不符合 schema：{}".format(details))
    return manifest


def evaluate_manifest(
    manifest_path: Path,
    *,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 20260808,
) -> Dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    manifest_dir = manifest_path.parent
    results = []
    for target in TARGET_ORDER:
        result = _base_result(target)
        spec = manifest["targets"][target]
        result["criteria"] = dict(spec.get("criteria", {}))
        try:
            evidence_path, evidence = _read_ref(spec["evidence"], manifest_dir)
            if not isinstance(evidence, Mapping):
                raise InvalidEvidence("{} 证据根节点必须是对象".format(target))
            result["evidence"].append(
                {
                    "path": str(evidence_path),
                    "sha256": spec["evidence"]["sha256"].lower(),
                }
            )
            _validate_provenance(target, evidence, spec["expected"])
            if target == "capability_retention":
                passed, metrics, reasons = _evaluate_capability(evidence)
            elif target == "ttft_reduction":
                passed, metrics, reasons = _evaluate_ttft(evidence)
            elif target == "single_inference_memory":
                passed, metrics, reasons = _evaluate_memory(evidence)
            elif target == "weak_network_retention":
                plan_path, experiment_plan = _read_ref(
                    spec["experiment_plan"], manifest_dir
                )
                if not isinstance(experiment_plan, Mapping):
                    raise InvalidEvidence("weak-network experiment plan 根节点必须是对象")
                plan_sha = spec["experiment_plan"]["sha256"].lower()
                result["evidence"].append(
                    {"path": str(plan_path), "sha256": plan_sha}
                )
                passed, metrics, reasons = _evaluate_weak_network(
                    evidence, experiment_plan, plan_sha
                )
            elif target == "weighted_business_e2e":
                weight_path, weight_plan = _read_ref(spec["weight_plan"], manifest_dir)
                result["evidence"].append(
                    {
                        "path": str(weight_path),
                        "sha256": spec["weight_plan"]["sha256"].lower(),
                    }
                )
                passed, metrics, reasons = _evaluate_latency(
                    evidence,
                    weight_plan,
                    bootstrap_iterations=bootstrap_iterations,
                    bootstrap_seed=bootstrap_seed,
                    weight_plan_sha256=spec["weight_plan"]["sha256"].lower(),
                )
            elif target == "traffic_global_objective":
                definition_path, definition = _read_ref(
                    spec["objective_definition"], manifest_dir
                )
                if not isinstance(definition, Mapping):
                    raise InvalidEvidence("objective definition 根节点必须是对象")
                definition_sha = spec["objective_definition"]["sha256"].lower()
                result["evidence"].append(
                    {"path": str(definition_path), "sha256": definition_sha}
                )
                sample_plan_path, sample_plan = _read_ref(
                    spec["sample_plan"], manifest_dir
                )
                if not isinstance(sample_plan, Mapping):
                    raise InvalidEvidence("global-objective sample plan 根节点必须是对象")
                sample_plan_sha = spec["sample_plan"]["sha256"].lower()
                result["evidence"].append(
                    {"path": str(sample_plan_path), "sha256": sample_plan_sha}
                )
                passed, metrics, reasons = _evaluate_objective(
                    evidence,
                    definition_sha,
                    definition,
                    sample_plan,
                    sample_plan_sha,
                )
            elif target == "model_update_loop":
                stage_hashes = {}
                stage_documents = {}
                for stage in ("package", "release", "apply", "rollback"):
                    stage_path, stage_document = _read_ref(
                        spec["stage_evidence"][stage], manifest_dir
                    )
                    if not isinstance(stage_document, Mapping):
                        raise InvalidEvidence("{} 阶段证据根节点必须是对象".format(stage))
                    stage_sha = spec["stage_evidence"][stage]["sha256"].lower()
                    stage_hashes[stage] = stage_sha
                    stage_documents[stage] = stage_document
                    result["evidence"].append(
                        {"path": str(stage_path), "sha256": stage_sha, "stage": stage}
                    )
                completion_path, completion = _read_ref(
                    spec["completion_marker"], manifest_dir
                )
                if not isinstance(completion, Mapping):
                    raise InvalidEvidence("模型更新完成标记根节点必须是对象")
                completion_sha = spec["completion_marker"]["sha256"].lower()
                result["evidence"].append(
                    {
                        "path": str(completion_path),
                        "sha256": completion_sha,
                        "stage": "completion_marker",
                    }
                )
                distributed_path, distributed = _read_ref(
                    spec["distributed_evidence"], manifest_dir
                )
                if not isinstance(distributed, Mapping):
                    raise InvalidEvidence("分布式模型更新证据根节点必须是对象")
                distributed_sha = spec["distributed_evidence"]["sha256"].lower()
                result["evidence"].append(
                    {
                        "path": str(distributed_path),
                        "sha256": distributed_sha,
                        "stage": "distributed_cloud_edge_loop",
                    }
                )
                distributed_completion_path, distributed_completion = _read_ref(
                    spec["distributed_completion_marker"], manifest_dir
                )
                if not isinstance(distributed_completion, Mapping):
                    raise InvalidEvidence("分布式模型更新完成标记根节点必须是对象")
                distributed_completion_sha = spec["distributed_completion_marker"][
                    "sha256"
                ].lower()
                result["evidence"].append(
                    {
                        "path": str(distributed_completion_path),
                        "sha256": distributed_completion_sha,
                        "stage": "distributed_completion_marker",
                    }
                )
                if (evidence_path.parent / "FAILED_NOT_FORMAL_EVIDENCE.json").exists():
                    raise InvalidEvidence("模型更新证据目录包含失败标记")
                if (
                    distributed_path.parent
                    / "FAILED_NOT_FORMAL_DISTRIBUTED_EVIDENCE.json"
                ).exists():
                    raise InvalidEvidence("分布式模型更新证据目录包含失败标记")
                local_passed, local_metrics, local_reasons = _evaluate_update_loop(
                    evidence,
                    stage_hashes,
                    stage_documents,
                    completion,
                    completion_sha,
                    spec["expected"],
                    spec["evidence"]["sha256"].lower(),
                )
                distributed_passed, distributed_metrics, distributed_reasons = (
                    _evaluate_distributed_update_loop(
                        distributed,
                        distributed_completion,
                        distributed_completion_sha,
                        spec["expected"],
                        evidence,
                        spec["evidence"]["sha256"].lower(),
                        stage_hashes,
                        stage_documents,
                        completion_sha,
                    )
                )
                if distributed_completion.get("distributed_evidence_sha256") != distributed_sha:
                    raise InvalidEvidence("分布式完成标记未绑定 manifest 指定主证据 SHA")
                passed = local_passed and distributed_passed
                metrics = {
                    **local_metrics,
                    "distributed_cloud_edge": distributed_metrics,
                }
                reasons = local_reasons + distributed_reasons
            else:  # pragma: no cover - guarded by TARGET_ORDER
                raise InvalidEvidence("未知 target：{}".format(target))
            result["status"] = "passed" if passed else "failed"
            result["metrics"] = metrics
            result["reasons"] = reasons
        except MissingEvidence as exc:
            result["status"] = "not_measured"
            result["reasons"] = [str(exc)]
        except InvalidEvidence as exc:
            result["status"] = "invalid_evidence"
            result["reasons"] = [str(exc)]
        results.append(result)
    status_counts = {
        status: sum(item["status"] == status for item in results)
        for status in STATUS_ZH
    }
    all_passed = status_counts["passed"] == len(TARGET_ORDER)
    return {
        "schema_version": "1.0",
        "evaluation_id": manifest["evaluation_id"],
        "scene": "freeway_traffic",
        "industrial_excluded": True,
        "manifest_path": str(manifest_path),
        "gate_status": "passed" if all_passed else "incomplete_or_failed",
        "status_counts": status_counts,
        "targets": results,
        "policy": {
            "explicit_evidence_only": True,
            "automatic_result_discovery": False,
            "missing_evidence_status": "not_measured",
            "required_target_count": len(TARGET_ORDER),
        },
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "{:.6f}".format(value)
    if isinstance(value, (dict, list)):
        return "`{}`".format(_canonical(value))
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# 七项统一证据门禁",
        "",
        "- 评估编号：`{}`".format(report["evaluation_id"]),
        "- 场景：交通（工业明确排除）",
        "- 总状态：**{}**".format(
            "全部通过" if report["gate_status"] == "passed" else "未全部通过"
        ),
        "- 取证原则：只读取 manifest 显式列出的文件并核验 SHA-256；不自动发现旧结果。",
        "",
        "| 序号 | 指标 | 状态 | 核心结果 |",
        "|---:|---|---|---|",
    ]
    for index, item in enumerate(report["targets"], start=1):
        metrics = item.get("metrics", {})
        if metrics:
            summary = "; ".join(
                "{}={}".format(key, _fmt(value))
                for key, value in list(metrics.items())[:4]
            )
        else:
            summary = "；".join(item.get("reasons", [])) or "无"
        lines.append(
            "| {} | {} | {} | {} |".format(
                index,
                item["name_zh"],
                STATUS_ZH[item["status"]],
                summary.replace("|", "\\|"),
            )
        )
    lines.extend(["", "## 逐项判定", ""])
    for index, item in enumerate(report["targets"], start=1):
        lines.extend(
            [
                "### {}. {}".format(index, item["name_zh"]),
                "",
                "状态：**{}**".format(STATUS_ZH[item["status"]]),
                "",
            ]
        )
        if item.get("reasons"):
            lines.append("原因：")
            lines.append("")
            for reason in item["reasons"]:
                lines.append("- {}".format(reason))
            lines.append("")
        if item.get("metrics"):
            lines.append("测量值：")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(item["metrics"], ensure_ascii=False, indent=2, sort_keys=True))
            lines.append("```")
            lines.append("")
        if item.get("evidence"):
            lines.append("证据：")
            lines.append("")
            for evidence in item["evidence"]:
                lines.append(
                    "- `{}`（SHA-256 `{}`）".format(
                        evidence["path"], evidence["sha256"]
                    )
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="交通场景七项统一证据门禁")
    parser.add_argument("--manifest", required=True, type=Path, help="证据 manifest")
    parser.add_argument("--output-json", required=True, type=Path, help="JSON 输出")
    parser.add_argument("--output-markdown", required=True, type=Path, help="中文 Markdown 输出")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = evaluate_manifest(
            args.manifest,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
        )
    except InvalidEvidence as exc:
        raise SystemExit("manifest 无效：{}".format(exc))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate_status": report["gate_status"],
                "output_json": str(args.output_json),
                "output_markdown": str(args.output_markdown),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["gate_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
