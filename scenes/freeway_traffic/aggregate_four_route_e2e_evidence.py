#!/usr/bin/env python3
"""Merge preregistered four-route benchmark runs into one gate evidence file.

The normal online workload is not expected to naturally exercise an outage
route.  This tool therefore combines independent, preregistered experiments
without hiding failed attempts or selecting samples after measurements exist.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ROUTES = ("edge_only", "local_autonomy", "cloud_async", "cloud_sync")
DURABLE_LOCAL_STAGES = {"handoff_durable", "outbox_durable"}
AUTHORITATIVE_FINAL_STAGES = {
    "lightweight_final",
    "large_model_review",
    "large_model_correction",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PlanError(ValueError):
    """Raised when a preregistered plan itself is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_time(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise PlanError("{} 必须是带时区的 RFC3339 时间".format(field))
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PlanError("{} 不是合法时间：{}".format(field, exc))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PlanError("{} 必须包含时区".format(field))
    return parsed


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON 根节点必须是对象")
    return value


def _resolve(base: Path, value: Any) -> Path:
    raw = Path(str(value)).expanduser()
    return (base / raw).resolve() if not raw.is_absolute() else raw.resolve()


def _same(actual: Any, expected: Any) -> bool:
    return _canonical(actual) == _canonical(expected)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0.0 else None


def _unique_ints(value: Any, field: str) -> List[int]:
    if not isinstance(value, list) or not value:
        raise PlanError("{} 必须是非空整数数组".format(field))
    result: List[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise PlanError("{} 必须只包含非负整数".format(field))
        if item in result:
            raise PlanError("{} 不能包含重复值".format(field))
        result.append(item)
    return result


def _validate_plan(plan: Mapping[str, Any]) -> Dict[str, Any]:
    if plan.get("schema_version") != "1.0":
        raise PlanError("schema_version 必须是 1.0")
    aggregation_id = str(plan.get("aggregation_id", "")).strip()
    weight_plan_id = str(plan.get("weight_plan_id", "")).strip()
    if not aggregation_id or not weight_plan_id:
        raise PlanError("aggregation_id 和 weight_plan_id 不能为空")
    locked_at_raw = plan.get("locked_at")
    locked_at = _parse_time(locked_at_raw, "locked_at")
    expected = plan.get("expected")
    if not isinstance(expected, Mapping):
        raise PlanError("expected 必须是对象")
    for field in ("git_commit", "dataset_id", "model_ids", "hardware_id"):
        if field not in expected or expected[field] in (None, "", [], {}):
            raise PlanError("expected.{} 不能为空".format(field))
    data_sha = str(expected.get("data_sha256", "")).lower()
    if not SHA256_RE.fullmatch(data_sha):
        raise PlanError("expected.data_sha256 必须是 64 位 SHA-256")
    routes = plan.get("routes")
    if not isinstance(routes, Mapping) or set(routes) != set(ROUTES):
        raise PlanError("routes 必须恰好包含 {}".format(", ".join(ROUTES)))
    weights: Dict[str, float] = {}
    normalized_routes: Dict[str, List[Dict[str, Any]]] = {}
    run_ids = set()
    for route in ROUTES:
        route_plan = routes[route]
        if not isinstance(route_plan, Mapping):
            raise PlanError("routes.{} 必须是对象".format(route))
        weight = route_plan.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise PlanError("{}.weight 必须是数值".format(route))
        weight = float(weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise PlanError("四条路径的权重都必须大于 0")
        weights[route] = weight
        experiments = route_plan.get("experiments")
        if not isinstance(experiments, list) or not experiments:
            raise PlanError("{}.experiments 必须是非空数组".format(route))
        normalized = []
        for index, raw in enumerate(experiments):
            if not isinstance(raw, Mapping):
                raise PlanError("{}.experiments[{}] 必须是对象".format(route, index))
            run_id = str(raw.get("run_id", "")).strip()
            if not run_id or run_id in run_ids:
                raise PlanError("run_id 不能为空且必须全局唯一：{}".format(run_id))
            run_ids.add(run_id)
            network_profile = raw.get("network_profile")
            if not isinstance(network_profile, Mapping) or not str(
                network_profile.get("profile_id", "")
            ).strip():
                raise PlanError("{} 缺少固定 network_profile.profile_id".format(run_id))
            result_path = str(raw.get("result_path", "")).strip()
            attestation_path = str(raw.get("attestation_path", "")).strip()
            if not result_path or not attestation_path:
                raise PlanError("{} 缺少 result_path/attestation_path".format(run_id))
            normalized.append(
                {
                    "run_id": run_id,
                    "result_path": result_path,
                    "attestation_path": attestation_path,
                    "sample_ids": _unique_ints(
                        raw.get("sample_ids"), "{}.sample_ids".format(run_id)
                    ),
                    "partition_ids": _unique_ints(
                        raw.get("partition_ids"), "{}.partition_ids".format(run_id)
                    ),
                    "network_profile": dict(network_profile),
                }
            )
        normalized_routes[route] = normalized
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise PlanError("四路径权重之和必须为 1")
    return {
        "aggregation_id": aggregation_id,
        "weight_plan_id": weight_plan_id,
        "locked_at": str(locked_at_raw),
        "locked_at_value": locked_at,
        "expected": dict(expected),
        "weights": weights,
        "routes": normalized_routes,
    }


def _load_source(
    base: Path,
    experiment: Mapping[str, Any],
    expected: Mapping[str, Any],
    locked_at: datetime,
    plan_sha256: str,
) -> Tuple[Optional[Mapping[str, Any]], Dict[str, Any], List[str]]:
    result_path = _resolve(base, experiment["result_path"])
    attestation_path = _resolve(base, experiment["attestation_path"])
    record: Dict[str, Any] = {
        "run_id": experiment["run_id"],
        "benchmark_path": str(result_path),
        "attestation_path": str(attestation_path),
        "network_profile": experiment["network_profile"],
        "planned_sample_ids": list(experiment["sample_ids"]),
        "planned_partition_ids": list(experiment["partition_ids"]),
    }
    errors: List[str] = []
    benchmark: Optional[Mapping[str, Any]] = None
    if not result_path.is_file():
        errors.append("benchmark_missing")
    else:
        record["benchmark_sha256"] = _sha256(result_path)
        try:
            benchmark = _read_json(result_path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append("benchmark_invalid:{}".format(exc))
    attestation: Optional[Mapping[str, Any]] = None
    if not attestation_path.is_file():
        errors.append("attestation_missing")
    else:
        record["attestation_sha256"] = _sha256(attestation_path)
        try:
            attestation = _read_json(attestation_path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append("attestation_invalid:{}".format(exc))
    if benchmark is None or attestation is None:
        record["integrity_valid"] = False
        record["errors"] = errors
        return None, record, errors

    if attestation.get("schema_version") != "1.0":
        errors.append("attestation_schema_version_mismatch")
    if str(attestation.get("run_id", "")) != experiment["run_id"]:
        errors.append("run_id_mismatch")
    if str(attestation.get("experiment_plan_sha256", "")).lower() != plan_sha256:
        errors.append("experiment_plan_sha256_mismatch")
    if str(attestation.get("benchmark_sha256", "")).lower() != record[
        "benchmark_sha256"
    ]:
        errors.append("benchmark_sha256_mismatch")
    for field in ("git_commit", "dataset_id", "model_ids", "hardware_id"):
        if not _same(attestation.get(field), expected[field]):
            errors.append("{}_mismatch".format(field))
    expected_data_sha = str(expected["data_sha256"]).lower()
    if str(attestation.get("data_sha256", "")).lower() != expected_data_sha:
        errors.append("attestation_data_sha256_mismatch")
    if not _same(attestation.get("network_profile"), experiment["network_profile"]):
        errors.append("network_profile_mismatch")
    if str(benchmark.get("task", "")) != "pems08_current_state_deployed_e2e":
        errors.append("benchmark_task_mismatch")
    assets = benchmark.get("assets", {})
    assets = assets if isinstance(assets, Mapping) else {}
    if str(assets.get("data_sha256", "")).lower() != expected_data_sha:
        errors.append("benchmark_data_sha256_mismatch")
    benchmark_started = benchmark.get("measurement_started_at_utc")
    attested_started = attestation.get("run_started_at")
    if benchmark_started != attested_started:
        errors.append("run_started_at_mismatch")
    try:
        started = _parse_time(benchmark_started, "measurement_started_at_utc")
        record["run_started_at"] = str(benchmark_started)
        if started < locked_at:
            errors.append("run_started_before_plan_lock")
    except PlanError as exc:
        errors.append("run_started_at_invalid:{}".format(exc))

    samples = benchmark.get("samples")
    if not isinstance(samples, list):
        errors.append("benchmark_samples_missing")
    else:
        observed_ids = []
        for sample in samples:
            if isinstance(sample, Mapping) and isinstance(sample.get("sample_id"), int):
                observed_ids.append(int(sample["sample_id"]))
        if sorted(observed_ids) != sorted(experiment["sample_ids"]):
            errors.append("sample_population_mismatch")
    record["integrity_valid"] = not errors
    record["errors"] = errors
    return (benchmark if not errors else None), record, errors


def _event_rows(benchmark: Mapping[str, Any]) -> Tuple[Dict[Tuple[int, int], Mapping[str, Any]], List[str]]:
    rows: Dict[Tuple[int, int], Mapping[str, Any]] = {}
    errors: List[str] = []
    samples = benchmark.get("samples", [])
    for sample in samples if isinstance(samples, list) else []:
        if not isinstance(sample, Mapping) or not isinstance(sample.get("sample_id"), int):
            errors.append("malformed_sample")
            continue
        sample_id = int(sample["sample_id"])
        events = sample.get("events")
        if not isinstance(events, list):
            errors.append("sample_{}_events_missing".format(sample_id))
            continue
        for event in events:
            if not isinstance(event, Mapping) or not isinstance(
                event.get("partition_id"), int
            ):
                errors.append("sample_{}_malformed_event".format(sample_id))
                continue
            key = (sample_id, int(event["partition_id"]))
            if key in rows:
                errors.append("duplicate_event_{}_{}".format(*key))
            else:
                rows[key] = event
    return rows, errors


def _business_endpoint(route: str, row: Mapping[str, Any]) -> Tuple[Optional[float], str, Optional[str]]:
    response = _number(row.get("response_at_ms"))
    deferred_raw = row.get("deferred_action_types", [])
    immediate_raw = row.get("immediate_action_types", [])
    if response is None:
        return None, "missing_local_response", "local_response_missing"
    if not isinstance(deferred_raw, list) or not isinstance(immediate_raw, list):
        return None, "invalid_action_stage", "action_stage_invalid"
    deferred = [str(value) for value in deferred_raw if str(value)]
    immediate = [str(value) for value in immediate_raw if str(value)]

    if route == "cloud_sync":
        final_ms = _number(row.get("global_final_ms"))
        stage = str(row.get("review_completion_stage", ""))
        authoritative = bool(
            row.get("policy_wait") is True
            and row.get("review_authoritative") is True
            and str(row.get("review_state", "")) == "completed"
            and stage in AUTHORITATIVE_FINAL_STAGES
            and str(row.get("review_final_status", "")) == "final"
            and row.get("review_cloud_confirmed") is True
        )
        if not authoritative or final_ms is None:
            return None, "missing_authoritative_final", "authoritative_final_missing"
        return max(response, final_ms), "authoritative_final", None

    # The local/async business contract ends at a safe provisional response.
    # Summary upload is still mandatory and must already be crash-recoverable;
    # policy_route therefore never means "do not upload to cloud".
    if row.get("policy_wait") is not False:
        return None, "invalid_provisional_policy", "provisional_policy_wait_invalid"
    if str(row.get("response_status", "")) != "provisional":
        return None, "missing_provisional_response", "provisional_response_missing"
    if row.get("action_authorization_present") is not True:
        return None, "missing_action_authorization", "action_authorization_missing"
    if len(deferred) != len(deferred_raw) or len(immediate) != len(immediate_raw):
        return None, "invalid_action_stage", "action_stage_invalid"
    if len(set(deferred)) != len(deferred) or len(set(immediate)) != len(immediate):
        return None, "invalid_action_stage", "action_stage_invalid"
    if set(deferred).intersection(immediate):
        return None, "invalid_action_stage", "action_stage_overlap"
    response_actions = row.get("response_actions")
    if not isinstance(response_actions, list):
        return None, "missing_action_contract", "response_action_contract_missing"
    action_types = []
    for action in response_actions:
        if not isinstance(action, Mapping):
            return None, "invalid_action_contract", "response_action_contract_invalid"
        action_type = str(action.get("action_type", "")).strip()
        requires_cloud = action.get("requires_cloud_confirmation")
        if not action_type or not isinstance(requires_cloud, bool):
            return None, "invalid_action_contract", "response_action_contract_invalid"
        action_types.append(action_type)
        expected_stage = deferred if requires_cloud else immediate
        unexpected_stage = immediate if requires_cloud else deferred
        if action_type not in expected_stage or action_type in unexpected_stage:
            return None, "invalid_action_stage", "action_authorization_stage_mismatch"
    if len(set(action_types)) != len(action_types):
        return None, "invalid_action_contract", "duplicate_response_action_type"
    if set(action_types) != set(deferred).union(immediate):
        return None, "invalid_action_stage", "action_authorization_coverage_mismatch"
    if route == "local_autonomy" and row.get("local_autonomy") is not True:
        return None, "missing_local_autonomy", "local_autonomy_marker_missing"
    if row.get("summary_delivery_required") is not True:
        return None, "summary_not_required", "summary_delivery_not_required"
    stage = str(row.get("summary_persistence_stage", ""))
    if stage not in DURABLE_LOCAL_STAGES:
        return None, "missing_durable_queue", "local_summary_not_durable"
    if row.get("cloud_confirmed_in_response") is True:
        return None, "invalid_cloud_confirmation", "provisional_claimed_cloud_confirmation"
    if deferred and row.get("local_actions_authorized") is True:
        return None, "invalid_deferred_authorization", "deferred_action_claimed_authorized"
    if not deferred and immediate and row.get("local_actions_authorized") is not True:
        return None, "unsafe_local_action", "local_action_not_authorized"
    return response, "provisional_response_and_durable_summary", None


def aggregate_plan(plan_path: Path, generated_at: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    plan_path = Path(plan_path).resolve()
    plan_raw = _read_json(plan_path)
    plan = _validate_plan(plan_raw)
    plan_sha256 = _sha256(plan_path)
    base = plan_path.parent
    route_output: Dict[str, Any] = {}
    source_runs = []
    integrity_errors = []
    observed_starts: List[str] = []
    for route in ROUTES:
        attempts = 0
        samples_out = []
        failures = []
        for experiment in plan["routes"][route]:
            planned_keys = [
                (sample_id, partition_id)
                for sample_id in experiment["sample_ids"]
                for partition_id in experiment["partition_ids"]
            ]
            attempts += len(planned_keys)
            benchmark, source_record, source_errors = _load_source(
                base,
                experiment,
                plan["expected"],
                plan["locked_at_value"],
                plan_sha256,
            )
            source_record["policy_route"] = route
            source_runs.append(source_record)
            if source_record.get("run_started_at"):
                observed_starts.append(str(source_record["run_started_at"]))
            if source_errors or benchmark is None:
                integrity_errors.extend(
                    "{}:{}".format(experiment["run_id"], value)
                    for value in source_errors
                )
                for sample_id, partition_id in planned_keys:
                    failures.append(
                        {
                            "run_id": experiment["run_id"],
                            "sample_id": sample_id,
                            "partition_id": partition_id,
                            "reason": "source_integrity_failure",
                        }
                    )
                continue
            rows, row_errors = _event_rows(benchmark)
            planned_set = set(planned_keys)
            unexpected = sorted(set(rows) - planned_set)
            if row_errors or unexpected:
                errors = list(row_errors)
                if unexpected:
                    errors.append("unregistered_events:{}".format(unexpected))
                source_record["integrity_valid"] = False
                source_record["errors"].extend(errors)
                integrity_errors.extend(
                    "{}:{}".format(experiment["run_id"], value) for value in errors
                )
                for sample_id, partition_id in planned_keys:
                    failures.append(
                        {
                            "run_id": experiment["run_id"],
                            "sample_id": sample_id,
                            "partition_id": partition_id,
                            "reason": "source_population_invalid",
                        }
                    )
                continue
            for sample_id, partition_id in planned_keys:
                row = rows.get((sample_id, partition_id))
                base_failure = {
                    "run_id": experiment["run_id"],
                    "sample_id": sample_id,
                    "partition_id": partition_id,
                }
                if row is None:
                    failures.append(dict(base_failure, reason="event_missing"))
                    continue
                observed_route = str(row.get("policy_route", "unknown"))
                if observed_route != route:
                    failures.append(
                        dict(
                            base_failure,
                            reason="policy_route_mismatch",
                            observed_policy_route=observed_route,
                            observed_delivery_route=str(row.get("delivery_route", "unknown")),
                        )
                    )
                    continue
                latency, endpoint, reason = _business_endpoint(route, row)
                if latency is None or reason:
                    failures.append(dict(base_failure, reason=reason or "business_failed"))
                    continue
                samples_out.append(
                    {
                        "group_id": "{}:{}".format(experiment["run_id"], sample_id),
                        "latency_ms": round(float(latency), 6),
                        "run_id": experiment["run_id"],
                        "sample_id": sample_id,
                        "partition_id": partition_id,
                        "network_profile_id": str(
                            experiment["network_profile"]["profile_id"]
                        ),
                        "policy_route": observed_route,
                        "delivery_route": str(row.get("delivery_route", "unknown")),
                        "business_endpoint": endpoint,
                    }
                )
        route_output[route] = {
            "attempt_count": attempts,
            "success_count": len(samples_out),
            "failure_count": attempts - len(samples_out),
            "samples": samples_out,
            "failures": failures,
        }
    generated = generated_at or datetime.now(timezone.utc).isoformat()
    _parse_time(generated, "generated_at")
    earliest_start = min(observed_starts, key=lambda value: _parse_time(value, "run_started_at")) if observed_starts else plan["locked_at"]
    expected = plan["expected"]
    evidence = {
        "schema_version": "1.0",
        "provenance": {
            "git_commit": expected["git_commit"],
            "dataset_id": expected["dataset_id"],
            "model_ids": expected["model_ids"],
            "hardware_id": expected["hardware_id"],
            "metric_semantics": "business_actionable_end_to_end_latency_ms",
            "run_id": plan["aggregation_id"],
            "generated_at": generated,
            "run_started_at": earliest_start,
        },
        "weight_plan_id": plan["weight_plan_id"],
        "plan_sha256": plan_sha256,
        "integrity_valid": not integrity_errors,
        "integrity_errors": integrity_errors,
        "errors": [],
        "route_semantics": {
            "stratification_field": "policy_route",
            "delivery_route_role": "audit_only",
            "edge_only": "safe provisional response plus durable online summary queue",
            "local_autonomy": "safe provisional response plus durable summary queue",
            "cloud_async": "safe provisional response plus durable online summary queue",
            "cloud_sync": "authoritative final",
        },
        "source_runs": source_runs,
        "routes": route_output,
    }
    weight_plan = {
        "schema_version": 1,
        "weight_plan_id": plan["weight_plan_id"],
        "locked_at": plan["locked_at"],
        "plan_sha256": plan_sha256,
        "routes": plan["weights"],
    }
    return evidence, weight_plan


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="汇总预注册的四路径业务端到端证据")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-evidence", required=True, type=Path)
    parser.add_argument("--output-weight-plan", required=True, type=Path)
    args = parser.parse_args(argv)
    evidence_path = args.output_evidence.resolve()
    weight_path = args.output_weight_plan.resolve()
    if evidence_path == weight_path:
        raise ValueError("evidence and weight-plan outputs must be different files")
    existing = [path for path in (evidence_path, weight_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite formal evidence: {}".format(
                ", ".join(str(path) for path in existing)
            )
        )
    evidence, weight_plan = aggregate_plan(args.plan)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    weight_path.write_text(
        json.dumps(weight_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "evidence": str(evidence_path),
                "weight_plan": str(weight_path),
                "integrity_valid": evidence["integrity_valid"],
                "routes": {
                    route: {
                        key: evidence["routes"][route][key]
                        for key in ("attempt_count", "success_count", "failure_count")
                    }
                    for route in ROUTES
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if evidence["integrity_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
