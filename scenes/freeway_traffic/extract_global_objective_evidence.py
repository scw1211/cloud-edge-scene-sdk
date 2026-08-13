#!/usr/bin/env python3
"""Extract gate-ready traffic global-objective evidence from real E2E runs.

The extractor is intentionally strict about the denominator.  A sample plan is
locked before measurement and every planned sample is emitted exactly once,
including failed, partial and slow samples.  Only complete authoritative
aggregations may contribute an optimization record; negative outcomes remain in
``attempts`` so that neither this extractor nor a later report can silently
drop them.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read {} {}: {}".format(label, path, exc))
    if not isinstance(value, dict):
        raise ValueError("{} {} must contain a JSON object".format(label, path))
    return value


def _git_state(project_root: Path) -> Tuple[str, bool]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(project_root),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=str(project_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    )
    return completed.stdout.strip(), dirty


def _member_list(value: Any, label: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("{} must be a non-empty list".format(label))
    members: List[str] = []
    for index, member in enumerate(value):
        if not isinstance(member, str) or not member.strip():
            raise ValueError("{}[{}] must be a non-empty string".format(label, index))
        if member in members:
            raise ValueError("{} contains duplicate member {}".format(label, member))
        members.append(member)
    return members


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sample_ids(value: Any, label: str) -> List[int]:
    if not isinstance(value, list) or not value:
        raise ValueError("{} must be a non-empty list".format(label))
    sample_ids: List[int] = []
    for index, sample_id in enumerate(value):
        if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id < 0:
            raise ValueError("{}[{}] must be a non-negative integer".format(label, index))
        if sample_id in sample_ids:
            raise ValueError("{} contains duplicate sample {}".format(label, sample_id))
        sample_ids.append(sample_id)
    return sample_ids


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty ISO timestamp".format(label))
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("{} is not an ISO timestamp".format(label)) from exc
    if parsed.tzinfo is None:
        raise ValueError("{} must include a timezone".format(label))
    return parsed


def _validate_sample_plan(
    plan: Mapping[str, Any],
    *,
    objective_id: str,
    objective_definition_sha256: str,
    git_commit: str,
    dataset_id: str,
    model_ids: Sequence[str],
    hardware_id: str,
) -> Tuple[str, str, List[int], int, float, datetime]:
    if plan.get("schema_version") != "1.0" or plan.get("scene") != "freeway_traffic":
        raise ValueError("sample plan schema_version/scene is invalid")
    plan_id = str(plan.get("plan_id", "")).strip()
    if not plan_id:
        raise ValueError("sample plan plan_id must not be empty")
    locked_at_text = str(plan.get("locked_at", "")).strip()
    locked_at = _timestamp(locked_at_text, "sample plan locked_at")
    sample_ids = _sample_ids(plan.get("sample_ids"), "sample plan sample_ids")
    expected_count = plan.get("expected_sample_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != len(sample_ids)
    ):
        raise ValueError(
            "sample plan expected_sample_count must equal the unique sample list"
        )
    slow_threshold = plan.get("slow_threshold_ms")
    if (
        isinstance(slow_threshold, bool)
        or not isinstance(slow_threshold, (int, float))
        or float(slow_threshold) <= 0.0
    ):
        raise ValueError("sample plan slow_threshold_ms must be positive")
    expected = plan.get("expected")
    if not isinstance(expected, Mapping):
        raise ValueError("sample plan expected must be an object")
    comparisons = {
        "git_commit": git_commit,
        "dataset_id": dataset_id,
        "model_ids": list(model_ids),
        "hardware_id": hardware_id,
        "objective_id": objective_id,
        "objective_definition_sha256": objective_definition_sha256.lower(),
    }
    for field, actual in comparisons.items():
        planned = expected.get(field)
        if field == "objective_definition_sha256":
            planned = str(planned or "").lower()
        if _canonical(planned) != _canonical(actual):
            raise ValueError("sample plan expected.{} does not match this run".format(field))
    return (
        plan_id,
        locked_at_text,
        sample_ids,
        expected_count,
        float(slow_threshold),
        locked_at,
    )


def _one_optimization(
    sample: Mapping[str, Any],
    objective_id: str,
    objective_definition_sha256: str,
    objective_expected_members: Sequence[str],
    source: Path,
) -> Dict[str, Any]:
    sample_id = sample.get("sample_id")
    if sample.get("aggregations_complete") is not True:
        raise ValueError(
            "{} sample {} is not a complete authoritative aggregation".format(
                source, sample_id
            )
        )
    aggregations = sample.get("aggregations")
    if not isinstance(aggregations, list) or len(aggregations) != 1:
        raise ValueError(
            "{} sample {} must contain exactly one aggregation".format(
                source, sample_id
            )
        )
    aggregation = aggregations[0]
    if not isinstance(aggregation, Mapping):
        raise ValueError("{} sample {} aggregation is invalid".format(source, sample_id))
    if not (
        aggregation.get("state") == "completed"
        and aggregation.get("completion_reason") == "all_expected_members"
        and aggregation.get("evidence_complete") is True
        and aggregation.get("finality") == "final"
        and aggregation.get("global_confirmation") is True
    ):
        raise ValueError(
            "{} sample {} aggregation is not authoritative final".format(
                source, sample_id
            )
        )
    expected_members = _member_list(
        aggregation.get("expected_members"),
        "{} sample {} expected_members".format(source, sample_id),
    )
    observed_members = _member_list(
        aggregation.get("received_members"),
        "{} sample {} received_members".format(source, sample_id),
    )
    if observed_members != expected_members:
        raise ValueError(
            "{} sample {} expected_members and received_members must match exactly".format(
                source, sample_id
            )
        )
    if expected_members != list(objective_expected_members):
        raise ValueError(
            "{} sample {} expected_members do not match the objective definition".format(
                source, sample_id
            )
        )
    if aggregation.get("missing_members") != []:
        raise ValueError("{} sample {} reports missing members".format(source, sample_id))
    result = aggregation.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("{} sample {} has no result object".format(source, sample_id))
    values = result.get("global_optimizations")
    if not isinstance(values, list):
        raise ValueError(
            "{} sample {} has no global_optimizations".format(source, sample_id)
        )
    matches = [
        value
        for value in values
        if isinstance(value, Mapping) and value.get("objective_id") == objective_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "{} sample {} expected one {} optimization, found {}".format(
                source, sample_id, objective_id, len(matches)
            )
        )
    value = matches[0]
    required = (
        "exact_search",
        "candidate_set_complete",
        "constraints_satisfied",
        "mode",
        "applied",
        "authoritative_input",
        "baseline_utility",
        "selected_utility",
        "objective_definition_sha256",
        "candidate_space_size",
        "evaluated_candidate_count",
        "expected_member_count",
        "observed_member_count",
        "expected_members",
        "observed_members",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(
            "{} sample {} optimization misses {}".format(
                source, sample_id, ", ".join(missing)
            )
        )
    if str(value.get("objective_definition_sha256", "")).lower() != str(
        objective_definition_sha256
    ).lower():
        raise ValueError(
            "{} sample {} objective definition SHA does not match the measured "
            "optimizer".format(source, sample_id)
        )
    candidate_space_size = int(value.get("candidate_space_size", 0))
    evaluated_candidate_count = int(value.get("evaluated_candidate_count", 0))
    if candidate_space_size <= 0 or evaluated_candidate_count != candidate_space_size:
        raise ValueError(
            "{} sample {} did not enumerate the complete finite candidate set".format(
                source, sample_id
            )
        )
    expected_member_count = int(value.get("expected_member_count", 0))
    observed_member_count = int(value.get("observed_member_count", 0))
    if expected_member_count != len(expected_members):
        raise ValueError(
            "{} sample {} optimizer expected_member_count disagrees with aggregation".format(
                source, sample_id
            )
        )
    if observed_member_count != len(observed_members):
        raise ValueError(
            "{} sample {} optimizer observed_member_count disagrees with aggregation".format(
                source, sample_id
            )
        )
    optimizer_expected_members = _member_list(
        value.get("expected_members"),
        "{} sample {} optimizer expected_members".format(source, sample_id),
    )
    optimizer_observed_members = _member_list(
        value.get("observed_members"),
        "{} sample {} optimizer observed_members".format(source, sample_id),
    )
    if (
        optimizer_expected_members != expected_members
        or optimizer_observed_members != observed_members
    ):
        raise ValueError(
            "{} sample {} optimizer member lists disagree with aggregation".format(
                source, sample_id
            )
        )
    group_id = aggregation.get("group_id")
    if not group_id:
        raise ValueError("{} sample {} has no group_id".format(source, sample_id))
    return {
        "group_id": str(group_id),
        "sample_id": sample_id,
        "expected_members": expected_members,
        "observed_members": observed_members,
        "expected_member_count": expected_member_count,
        "observed_member_count": observed_member_count,
        "objective_definition_sha256": str(objective_definition_sha256),
        "exact_search": value.get("exact_search") is True,
        "candidate_set_complete": value.get("candidate_set_complete") is True,
        "constraints_satisfied": value.get("constraints_satisfied") is True,
        "mode": str(value.get("mode")),
        "applied": value.get("applied") is True,
        "authoritative_input": value.get("authoritative_input") is True,
        "baseline_utility": float(value.get("baseline_utility")),
        "selected_utility": float(value.get("selected_utility")),
        "candidate_space_size": candidate_space_size,
        "evaluated_candidate_count": evaluated_candidate_count,
        "selected_plan_sha256": str(
            value.get("selected", {}).get("plan_sha256", "")
            if isinstance(value.get("selected"), Mapping)
            else ""
        ),
    }


def _sample_attempt(
    sample: Mapping[str, Any],
    *,
    objective_id: str,
    objective_definition_sha256: str,
    objective_expected_members: Sequence[str],
    source: Path,
    source_experiment_id: str,
    slow_threshold_ms: float,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    sample_id = sample.get("sample_id")
    if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id < 0:
        raise ValueError("{} contains an invalid sample_id".format(source))
    raw_latency = sample.get("global_authoritative_final_ms")
    latency: Optional[float]
    if raw_latency is None:
        latency = None
    elif isinstance(raw_latency, bool) or not isinstance(raw_latency, (int, float)):
        raise ValueError("{} sample {} has invalid final latency".format(source, sample_id))
    else:
        latency = float(raw_latency)
        if latency < 0.0:
            raise ValueError("{} sample {} has negative final latency".format(source, sample_id))
    aggregations = sample.get("aggregations")
    aggregation_rows = aggregations if isinstance(aggregations, list) else []
    group_ids = []
    for aggregation in aggregation_rows:
        if isinstance(aggregation, Mapping) and aggregation.get("group_id"):
            group_id = str(aggregation["group_id"])
            if group_id in group_ids:
                raise ValueError(
                    "{} sample {} contains duplicate aggregation group {}".format(
                        source, sample_id, group_id
                    )
                )
            group_ids.append(group_id)

    record: Optional[Dict[str, Any]] = None
    if sample.get("aggregations_complete") is True:
        record = _one_optimization(
            sample,
            objective_id,
            objective_definition_sha256,
            objective_expected_members,
            source,
        )
        record["source_experiment_id"] = source_experiment_id
        outcome = "authoritative_optimized"
    else:
        partial = any(
            isinstance(aggregation, Mapping)
            and (
                aggregation.get("finality") == "partial_final"
                or "partial" in str(aggregation.get("completion_reason", ""))
                or bool(aggregation.get("missing_members"))
            )
            for aggregation in aggregation_rows
        )
        outcome = "partial" if partial else "failed"

    errors = sample.get("errors", [])
    if not isinstance(errors, list):
        raise ValueError("{} sample {} errors must be a list".format(source, sample_id))
    attempt = {
        "sample_id": sample_id,
        "source_experiment_id": source_experiment_id,
        "outcome": outcome,
        "aggregation_group_ids": group_ids,
        "global_authoritative_final_ms": latency,
        "slow": latency is not None and latency >= slow_threshold_ms,
        "errors": [str(error) for error in errors],
    }
    return attempt, record


def extract(
    benchmark_paths: Sequence[Path],
    objective_path: Path,
    sample_plan_path: Path,
    *,
    git_commit: str,
    dataset_id: str,
    model_ids: Sequence[str],
    hardware_id: str,
    run_id: str,
) -> Dict[str, Any]:
    objective = _read_object(objective_path, "objective definition")
    objective_id = str(objective.get("objective_id", "")).strip()
    if not objective_id:
        raise ValueError("objective definition has no objective_id")
    objective_expected_members = _member_list(
        objective.get("expected_members"), "objective definition expected_members"
    )
    objective_sha256 = _sha256(objective_path)
    sample_plan = _read_object(sample_plan_path, "sample plan")
    (
        plan_id,
        plan_locked_at,
        planned_sample_ids,
        expected_sample_count,
        slow_threshold_ms,
        locked_at,
    ) = _validate_sample_plan(
        sample_plan,
        objective_id=objective_id,
        objective_definition_sha256=objective_sha256,
        git_commit=git_commit,
        dataset_id=dataset_id,
        model_ids=model_ids,
        hardware_id=hardware_id,
    )
    sample_plan_sha256 = _sha256(sample_plan_path)
    records: List[Dict[str, Any]] = []
    attempts: List[Dict[str, Any]] = []
    source_files = []
    source_started_at: List[datetime] = []
    seen_groups = set()
    seen_samples = set()
    for raw_path in benchmark_paths:
        path = raw_path.resolve()
        benchmark = _read_object(path, "benchmark")
        measured_at = _timestamp(
            benchmark.get("measurement_started_at_utc"),
            "benchmark {} measurement_started_at_utc".format(path),
        )
        if measured_at <= locked_at:
            raise ValueError(
                "benchmark {} started before the sample plan was locked".format(path)
            )
        source_started_at.append(measured_at)
        source_experiment_id = str(benchmark.get("experiment_id", "")).strip()
        if not source_experiment_id:
            raise ValueError("benchmark {} has no experiment_id".format(path))
        samples = benchmark.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("benchmark {} has no samples".format(path))
        source_files.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "experiment_id": source_experiment_id,
                "measurement_started_at_utc": benchmark["measurement_started_at_utc"],
            }
        )
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise ValueError("benchmark {} contains a non-object sample".format(path))
            attempt, record = _sample_attempt(
                sample,
                objective_id=objective_id,
                objective_definition_sha256=objective_sha256,
                objective_expected_members=objective_expected_members,
                source=path,
                source_experiment_id=source_experiment_id,
                slow_threshold_ms=slow_threshold_ms,
            )
            sample_id = attempt["sample_id"]
            if sample_id in seen_samples:
                raise ValueError("duplicate benchmark sample {}".format(sample_id))
            seen_samples.add(sample_id)
            for group_id in attempt["aggregation_group_ids"]:
                if group_id in seen_groups:
                    raise ValueError("duplicate aggregation group {}".format(group_id))
                seen_groups.add(group_id)
            attempts.append(attempt)
            if record is not None:
                records.append(record)
    planned_set = set(planned_sample_ids)
    if seen_samples != planned_set:
        missing = sorted(planned_set - seen_samples)
        unexpected = sorted(seen_samples - planned_set)
        raise ValueError(
            "benchmark samples do not match the preregistered plan; missing={}, "
            "unexpected={}".format(missing, unexpected)
        )
    if len(attempts) != expected_sample_count:
        raise ValueError("benchmark attempt count does not match expected_sample_count")
    attempts.sort(key=lambda row: int(row["sample_id"]))
    records.sort(key=lambda row: int(row["sample_id"]))
    outcome_counts = {
        outcome: sum(attempt["outcome"] == outcome for attempt in attempts)
        for outcome in ("authoritative_optimized", "partial", "failed")
    }
    return {
        "schema_version": "1.0",
        "provenance": {
            "git_commit": git_commit,
            "dataset_id": dataset_id,
            "model_ids": list(model_ids),
            "hardware_id": hardware_id,
            "metric_semantics": "finite_candidate_joint_plan_utility",
            "run_id": run_id,
            "run_started_at": min(source_started_at).isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_files": source_files,
        },
        "sample_plan": {
            "plan_id": plan_id,
            "sha256": sample_plan_sha256,
            "locked_at": plan_locked_at,
            "expected_sample_count": expected_sample_count,
            "sample_ids": planned_sample_ids,
            "slow_threshold_ms": slow_threshold_ms,
        },
        "objective_id": objective_id,
        "objective_definition_sha256": objective_sha256,
        "attempts": attempts,
        "outcome_counts": outcome_counts,
        "slow_sample_count": sum(bool(attempt["slow"]) for attempt in attempts),
        "records": records,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从真实交通端到端结果提取全局联合目标证据"
    )
    parser.add_argument("--benchmark-json", action="append", required=True)
    parser.add_argument("--objective-definition", required=True)
    parser.add_argument("--sample-plan", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--git-commit")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--model-id", action="append", required=True)
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite existing evidence: {}".format(output))
    current_commit, dirty = _git_state(project_root)
    if dirty:
        raise ValueError("formal objective evidence requires a clean Git worktree")
    if args.git_commit and str(args.git_commit) != current_commit:
        raise ValueError("--git-commit does not match the checked-out HEAD")
    value = extract(
        [Path(path) for path in args.benchmark_json],
        Path(args.objective_definition).resolve(),
        Path(args.sample_plan).resolve(),
        git_commit=current_commit,
        dataset_id=str(args.dataset_id),
        model_ids=[str(value) for value in args.model_id],
        hardware_id=str(args.hardware_id),
        run_id=str(args.run_id),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "records": len(value["records"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
