#!/usr/bin/env python3
"""Verify the isolated industrial controlled threshold-optimization demo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "results/industrial_controlled_threshold_optimization_demo_v1"
DEFAULT_INSTALL = REPO_ROOT / "scenes/industrial_anomaly/deployment/controlled_threshold_demo_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def state(score: float, review_low: float, review_high: float) -> str:
    if not review_low < review_high:
        raise AssertionError("review_low must be less than review_high")
    if score < review_low:
        return "normal"
    if score < review_high:
        return "review"
    return "anomaly"


def binary_f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2.0 * tp) / denominator


def metrics(rows: list[dict[str, str]], prefix: str) -> dict[str, Any]:
    hard_conflicts = 0
    any_review = 0
    exact_agreement = 0
    resolver_calls = 0
    tn = fp = fn = tp = 0
    for row in rows:
        rgb_state = state(
            float(row["rgb_score"]),
            float(row[f"{prefix}_rgb_review_low"]),
            float(row[f"{prefix}_rgb_review_high"]),
        )
        infrared_state = state(
            float(row["infrared_score"]),
            float(row[f"{prefix}_infrared_review_low"]),
            float(row[f"{prefix}_infrared_review_high"]),
        )
        assert rgb_state == row[f"{prefix}_rgb_state"]
        assert infrared_state == row[f"{prefix}_infrared_state"]
        conflict = {rgb_state, infrared_state} == {"normal", "anomaly"}
        review = "review" in (rgb_state, infrared_state)
        assert conflict == (row[f"{prefix}_hard_action_conflict"] == "true")
        assert review == (row[f"{prefix}_any_review"] == "true")
        hard_conflicts += int(conflict)
        any_review += int(review)
        exact_agreement += int(rgb_state == infrared_state)

        if rgb_state == infrared_state == "normal":
            prediction = "normal"
        elif rgb_state == infrared_state == "anomaly":
            prediction = "defect"
        else:
            prediction = row["et10_oof_prediction"]
            resolver_calls += 1
        assert prediction == row[f"{prefix}_final_prediction"]
        truth = row["truth"]
        tn += int(truth == "normal" and prediction == "normal")
        fp += int(truth == "normal" and prediction == "defect")
        fn += int(truth == "defect" and prediction == "normal")
        tp += int(truth == "defect" and prediction == "defect")

    count = len(rows)
    return {
        "pairs": count,
        "truth_normal": tn + fp,
        "truth_defect": fn + tp,
        "hard_action_conflict_count": hard_conflicts,
        "hard_action_conflict_rate": hard_conflicts / count,
        "any_review_count": any_review,
        "any_review_rate": any_review / count,
        "exact_state_agreement_count": exact_agreement,
        "exact_state_agreement_rate": exact_agreement / count,
        "et10_resolver_calls": resolver_calls,
        "accuracy": (tn + tp) / count,
        "macro_f1": (binary_f1(tn, fn, fp) + binary_f1(tp, fp, fn)) / 2.0,
        "anomaly_recall": tp / (tp + fn),
        "false_release": fn,
        "false_quarantine": fp,
        "confusion_matrix_truth_normal_defect__prediction_normal_defect": [[tn, fp], [fn, tp]],
    }


def close(left: Any, right: Any, path: str = "root") -> None:
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), path
        for key in left:
            close(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right), path
        for index, value in enumerate(left):
            close(value, right[index], f"{path}[{index}]")
    elif isinstance(left, float):
        assert abs(left - float(right)) <= 1e-12, f"{path}: {left} != {right}"
    else:
        assert left == right, f"{path}: {left} != {right}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--install-dir", type=Path, default=DEFAULT_INSTALL)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    install = args.install_dir.resolve()
    checks: dict[str, bool] = {}

    protocol = load_json(output / "protocol.json")
    input_manifest = load_json(output / "input_manifest.json")
    recorded_metrics = load_json(output / "metrics.json")
    initial = load_json(output / "controlled_initial_thresholds.json")
    candidate = load_json(output / "optimized_candidate_thresholds.json")
    registry = load_json(install / "registry.json")
    receipt = load_json(output / "install_receipt.json")
    with (output / "paired_before_after.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    checks["status_development_demo_only"] = (
        protocol["status"] == "DEVELOPMENT_DEMO_ONLY"
        and candidate["status"] == "DEVELOPMENT_DEMO_ONLY"
    )
    checks["identity_authority"] = (
        protocol["identity_authority"] == "USER_DECLARED_CANONICAL_BINDING"
        and input_manifest["identity_authority"] == "USER_DECLARED_CANONICAL_BINDING"
    )
    checks["not_production_history"] = protocol["production_history_claimed"] is False
    checks["full_data_disclosure"] = (
        protocol["full_443_labels_used_for_selection"] is True
        and protocol["independent_test"] is False
    )
    checks["rows_443_unique"] = len(rows) == 443 and len({row["sample_id"] for row in rows}) == 443
    checks["ten_products"] = len({row["product"] for row in rows}) == 10
    for config, label in ((initial, "initial"), (candidate, "candidate")):
        scalar_count = 0
        for modality in ("rgb", "infrared"):
            assert len(config["modalities"][modality]) == 10
            for band in config["modalities"][modality].values():
                assert float(band["review_low"]) < float(band["review_high"])
                scalar_count += 2
        checks[f"{label}_40_scalar_thresholds"] = scalar_count == 40

    recomputed_before = metrics(rows, "before")
    recomputed_after = metrics(rows, "after")
    close(recomputed_before, recorded_metrics["before"], "metrics.before")
    close(recomputed_after, recorded_metrics["after"], "metrics.after")
    checks["metrics_exactly_recomputed"] = True
    checks["candidate_exactly_18_conflicts"] = recomputed_after["hard_action_conflict_count"] == 18
    checks["candidate_conflict_rate_below_5_percent"] = recomputed_after["hard_action_conflict_rate"] < 0.05
    checks["all_metric_gates_pass"] = all(recorded_metrics["gates"].values())

    demo_ids = load_json(output / "demo_candidate_hard_conflict_ids.json")
    checks["demo_conflict_ids_18"] = demo_ids["count"] == 18 and len(demo_ids["ids"]) == 18
    checks["demo_ids_not_legacy_claim"] = (
        input_manifest["row_level_legacy_18_conflict_list_available"] is False
        and input_manifest["demo_conflict_ids_are_legacy_ids"] is False
    )

    installed_config = install / "optimized_candidate_thresholds.json"
    checks["installed_config_sha"] = (
        sha256_file(installed_config) == receipt["config_sha256"]
        == registry["versions"][registry["active_demo_version"]]["config_sha256"]
    )
    checks["isolated_registry"] = (
        registry["registry_scope"] == "ISOLATED_DEVELOPMENT_DEMO_ONLY"
        and registry["production_registry_modified"] is False
        and receipt["production_mutation"] is False
        and receipt["production_endpoint_18100_touched"] is False
    )
    checks["rollback_available"] = registry["rollback_target"] == "NO_ACTIVE_DEMO_VERSION"
    checks["production_guard_unchanged"] = (
        receipt["production_guard_sha256_before"] == receipt["production_guard_sha256_after"]
        and all(
            sha256_file(REPO_ROOT / path) == digest
            for path, digest in receipt["production_guard_sha256_after"].items()
        )
    )

    checksum_ok = True
    for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, path_text = line.split("  ", 1)
        checksum_ok &= (REPO_ROOT / path_text).is_file()
        checksum_ok &= sha256_file(REPO_ROOT / path_text) == digest
    checks["sha256sums"] = checksum_ok
    passed = all(checks.values())
    result = {
        "schema_version": "industrial-controlled-threshold-verification/v1",
        "version": protocol["version"],
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "recomputed_before": recomputed_before,
        "recomputed_after": recomputed_after,
    }
    (output / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
