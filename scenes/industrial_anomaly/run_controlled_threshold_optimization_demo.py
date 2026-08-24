#!/usr/bin/env python3
"""Build the isolated industrial threshold-optimization development demo.

This program deliberately does not read or write a live service configuration.  It
uses the retained Orin RGB/infrared score rows and the retained ET10 nested-OOF
predictions to create a reproducible, full-data development demonstration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "results/industrial_global_optimization_delivery_v1/evidence"
RGB_ROOT = SOURCE_ROOT / "orin_rgb"
INFRARED_ROOT = SOURCE_ROOT / "orin_infrared"
OOF_PATH = SOURCE_ROOT / "cutoff_nested/oof_records.jsonl"
PRODUCTION_BANDS_PATH = (
    REPO_ROOT / "scenes/industrial_anomaly/industrial_anomaly/review_bands.json"
)
VERIFIER_PATH = REPO_ROOT / "scenes/industrial_anomaly/verify_controlled_threshold_optimization_demo.py"
DEFAULT_OUTPUT = REPO_ROOT / "results/industrial_controlled_threshold_optimization_demo_v1"
DEFAULT_INSTALL = (
    REPO_ROOT / "scenes/industrial_anomaly/deployment/controlled_threshold_demo_v1"
)

VERSION = "industrial-controlled-threshold-demo-v1.0.0"
STATUS = "DEVELOPMENT_DEMO_ONLY"
IDENTITY_AUTHORITY = "USER_DECLARED_CANONICAL_BINDING"
TARGET_CONFLICTS = 18
MODALITIES = ("rgb", "infrared")
STATE_NAMES = np.asarray(("normal", "review", "anomaly"))
WEIGHTS = {
    "hard_action_conflict_rate": 0.25,
    "classification_error_rate": 0.15,
    "one_minus_macro_f1": 0.15,
    "one_minus_anomaly_recall": 0.10,
    "false_release_rate_among_defects": 0.10,
    "false_quarantine_rate_among_normals": 0.10,
    "any_review_rate": 0.15,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def deterministic_quantile(values: np.ndarray, quantile: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    position = (len(ordered) - 1) * quantile
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def sample_identity(path_text: str, product: str) -> tuple[str, int]:
    path = Path(path_text)
    category = path.parent.name
    sample_id = f"{product}_{category}_{path.stem}"
    truth = 0 if category == "good" else 1
    return sample_id, truth


def load_modality(root: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], list[Path]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    files = sorted(root.glob("*/predictions.nano.csv"))
    for path in files:
        product = path.parent.name
        with path.open(newline="", encoding="utf-8") as handle:
            for source in csv.DictReader(handle):
                sample_id, truth = sample_identity(source["path"], product)
                key = (product, sample_id)
                if key in rows:
                    raise RuntimeError(f"duplicate modality key: {key}")
                rows[key] = {
                    "path": source["path"],
                    "score": float(source["image_score"]),
                    "truth": truth,
                }
    return rows, files


def load_inputs() -> dict[str, Any]:
    rgb, rgb_files = load_modality(RGB_ROOT)
    infrared, infrared_files = load_modality(INFRARED_ROOT)
    oof_rows = [json.loads(line) for line in OOF_PATH.read_text(encoding="utf-8").splitlines()]
    oof_rows.sort(key=lambda row: int(row["pair_index"]))
    keys = [(row["product"], row["sample_id"]) for row in oof_rows]
    if len(keys) != 443 or len(set(keys)) != 443:
        raise RuntimeError("ET10 OOF input is not exactly 443 unique pairs")
    if set(keys) != set(rgb) or set(keys) != set(infrared):
        raise RuntimeError("RGB, infrared, and ET10 OOF pair identities differ")
    for key, oof in zip(keys, oof_rows):
        truth = 1 if oof["truth"] == "defect" else 0
        if truth != rgb[key]["truth"] or truth != infrared[key]["truth"]:
            raise RuntimeError(f"truth mismatch for {key}")

    products = sorted({product for product, _ in keys})
    product_to_index = {product: index for index, product in enumerate(products)}
    product_index = np.asarray([product_to_index[key[0]] for key in keys], dtype=np.int16)
    truth = np.asarray([rgb[key]["truth"] for key in keys], dtype=np.int8)
    scores = np.asarray(
        [[rgb[key]["score"], infrared[key]["score"]] for key in keys],
        dtype=np.float64,
    )
    resolver_prediction = np.asarray(
        [1 if row["calibrated_et10_prediction"] == "defect" else 0 for row in oof_rows],
        dtype=np.int8,
    )
    if int((truth == 0).sum()) != 100 or int((truth == 1).sum()) != 343:
        raise RuntimeError("unexpected truth distribution")
    return {
        "rgb": rgb,
        "infrared": infrared,
        "rgb_files": rgb_files,
        "infrared_files": infrared_files,
        "oof_rows": oof_rows,
        "keys": keys,
        "products": products,
        "product_index": product_index,
        "truth": truth,
        "scores": scores,
        "resolver_prediction": resolver_prediction,
    }


def controlled_initialization(inputs: dict[str, Any]) -> dict[tuple[int, int], tuple[float, float]]:
    """Return an intentionally poor, label-free score-quantile initialization.

    Products are sorted lexicographically.  The fixed pattern HHLLLLXXXX assigns
    both modalities high, both modalities low, or opposing modality quantiles.
    The pattern is intentionally adversarial and must never be described as a
    historical or production baseline.
    """

    pattern = "HHLLLLXXXX"
    params: dict[tuple[int, int], tuple[float, float]] = {}
    for product_id, group in enumerate(pattern):
        mask = inputs["product_index"] == product_id
        for modality_id in range(2):
            if group == "H":
                quantiles = (0.50, 0.95)
            elif group == "L":
                quantiles = (0.05, 0.50)
            else:
                quantiles = (0.70, 0.95) if modality_id == 0 else (0.05, 0.30)
            values = inputs["scores"][mask, modality_id]
            params[(product_id, modality_id)] = tuple(
                deterministic_quantile(values, quantile) for quantile in quantiles
            )
    return params


def candidate_boundaries(
    inputs: dict[str, Any],
    initial: dict[tuple[int, int], tuple[float, float]],
) -> dict[tuple[int, int], np.ndarray]:
    boundaries: dict[tuple[int, int], np.ndarray] = {}
    for product_id, _product in enumerate(inputs["products"]):
        mask = inputs["product_index"] == product_id
        for modality_id in range(2):
            unique = np.unique(inputs["scores"][mask, modality_id])
            epsilon = max(1e-12, float(unique[-1] - unique[0]) * 1e-6)
            values: list[float] = [float(unique[0] - epsilon)]
            values.extend(float(value) for value in (unique[:-1] + unique[1:]) / 2.0)
            values.append(float(unique[-1] + epsilon))
            values.extend(initial[(product_id, modality_id)])
            boundaries[(product_id, modality_id)] = np.asarray(sorted(set(values)))
    return boundaries


def states_for_thresholds(
    inputs: dict[str, Any],
    thresholds: dict[tuple[int, int], tuple[float, float]],
) -> np.ndarray:
    states = np.empty((443, 2), dtype=np.int8)
    for product_id, _product in enumerate(inputs["products"]):
        mask = inputs["product_index"] == product_id
        for modality_id in range(2):
            review_low, review_high = thresholds[(product_id, modality_id)]
            if not review_low < review_high:
                raise RuntimeError("all demo review bands must have review_low < review_high")
            values = inputs["scores"][mask, modality_id]
            states[mask, modality_id] = np.where(
                values < review_low,
                0,
                np.where(values < review_high, 1, 2),
            )
    return states


def binary_f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2.0 * tp) / denominator


def evaluate(
    inputs: dict[str, Any],
    thresholds: dict[tuple[int, int], tuple[float, float]],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = states_for_thresholds(inputs, thresholds)
    hard_conflict = ((states[:, 0] == 0) & (states[:, 1] == 2)) | (
        (states[:, 0] == 2) & (states[:, 1] == 0)
    )
    any_review = (states == 1).any(axis=1)
    exact_normal = (states == 0).all(axis=1)
    exact_anomaly = (states == 2).all(axis=1)

    # Fixed decision contract: exact two-modality agreement bypasses the cloud
    # resolver; every other state pair uses the already frozen ET10 nested-OOF
    # prediction.  Threshold search never retrains or edits ET10.
    final_prediction = inputs["resolver_prediction"].copy()
    final_prediction[exact_normal] = 0
    final_prediction[exact_anomaly] = 1
    resolver_used = ~(exact_normal | exact_anomaly)

    truth = inputs["truth"]
    tn = int(((truth == 0) & (final_prediction == 0)).sum())
    fp = int(((truth == 0) & (final_prediction == 1)).sum())
    fn = int(((truth == 1) & (final_prediction == 0)).sum())
    tp = int(((truth == 1) & (final_prediction == 1)).sum())
    accuracy = (tn + tp) / 443.0
    anomaly_recall = tp / 343.0
    macro_f1 = (binary_f1(tn, fn, fp) + binary_f1(tp, fp, fn)) / 2.0
    metrics = {
        "pairs": 443,
        "truth_normal": 100,
        "truth_defect": 343,
        "hard_action_conflict_count": int(hard_conflict.sum()),
        "hard_action_conflict_rate": float(hard_conflict.mean()),
        "any_review_count": int(any_review.sum()),
        "any_review_rate": float(any_review.mean()),
        "exact_state_agreement_count": int((states[:, 0] == states[:, 1]).sum()),
        "exact_state_agreement_rate": float((states[:, 0] == states[:, 1]).mean()),
        "et10_resolver_calls": int(resolver_used.sum()),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "anomaly_recall": anomaly_recall,
        "false_release": fn,
        "false_quarantine": fp,
        "confusion_matrix_truth_normal_defect__prediction_normal_defect": [
            [tn, fp],
            [fn, tp],
        ],
    }
    return metrics, states, hard_conflict, any_review, final_prediction


def objective(metrics: dict[str, Any]) -> dict[str, Any]:
    raw_terms = {
        "hard_action_conflict_rate": metrics["hard_action_conflict_rate"],
        "classification_error_rate": 1.0 - metrics["accuracy"],
        "one_minus_macro_f1": 1.0 - metrics["macro_f1"],
        "one_minus_anomaly_recall": 1.0 - metrics["anomaly_recall"],
        "false_release_rate_among_defects": metrics["false_release"] / 343.0,
        "false_quarantine_rate_among_normals": metrics["false_quarantine"] / 100.0,
        "any_review_rate": metrics["any_review_rate"],
    }
    contributions = {key: raw_terms[key] * WEIGHTS[key] for key in WEIGHTS}
    return {
        "weights": WEIGHTS,
        "raw_terms": raw_terms,
        "weighted_contributions": contributions,
        "composite_cost": float(sum(contributions.values())),
        "formula": (
            "0.25*hard_conflict_rate + 0.15*(1-Accuracy) + "
            "0.15*(1-MacroF1) + 0.10*(1-Recall) + "
            "0.10*(FR/343) + 0.10*(FQ/100) + 0.15*any_review_rate"
        ),
        "direction": "lower_is_better",
    }


def thresholds_from_indices(
    indices: dict[tuple[int, int], tuple[int, int]],
    boundaries: dict[tuple[int, int], np.ndarray],
) -> dict[tuple[int, int], tuple[float, float]]:
    return {
        key: (float(boundaries[key][value[0]]), float(boundaries[key][value[1]]))
        for key, value in indices.items()
    }


def optimize(
    inputs: dict[str, Any],
    initial: dict[tuple[int, int], tuple[float, float]],
) -> tuple[dict[tuple[int, int], tuple[float, float]], list[dict[str, Any]], int]:
    boundaries = candidate_boundaries(inputs, initial)
    indices: dict[tuple[int, int], tuple[int, int]] = {}
    for key, (review_low, review_high) in initial.items():
        values = boundaries[key]
        low_index = int(np.flatnonzero(values == review_low)[0])
        high_index = int(np.flatnonzero(values == review_high)[0])
        indices[key] = (low_index, high_index)

    thresholds = thresholds_from_indices(indices, boundaries)
    current_metrics, *_ = evaluate(inputs, thresholds)
    evaluated = 1
    trace: list[dict[str, Any]] = [
        {
            "iteration": -1,
            "stage": "CONTROLLED_SUBOPTIMAL_INITIALIZATION",
            "metrics": current_metrics,
            "objective": objective(current_metrics)["composite_cost"],
        }
    ]

    def rank(metrics: dict[str, Any], low_index: int, high_index: int) -> tuple[Any, ...]:
        conflict_count = metrics["hard_action_conflict_count"]
        return (
            conflict_count != TARGET_CONFLICTS,
            abs(conflict_count - TARGET_CONFLICTS),
            objective(metrics)["composite_cost"],
            metrics["false_release"],
            metrics["false_quarantine"],
            metrics["any_review_count"],
            low_index,
            high_index,
        )

    for iteration in range(6):
        changes = 0
        for product_id, _product in enumerate(inputs["products"]):
            for modality_id in range(2):
                key = (product_id, modality_id)
                old_indices = indices[key]
                best = (rank(current_metrics, *old_indices), old_indices, current_metrics)
                size = len(boundaries[key])
                for low_index in range(size - 1):
                    for high_index in range(low_index + 1, size):
                        indices[key] = (low_index, high_index)
                        candidate = thresholds_from_indices(indices, boundaries)
                        candidate_metrics, *_ = evaluate(inputs, candidate)
                        evaluated += 1
                        candidate_rank = rank(candidate_metrics, low_index, high_index)
                        if candidate_rank < best[0]:
                            best = (
                                candidate_rank,
                                (low_index, high_index),
                                candidate_metrics,
                            )
                indices[key] = best[1]
                current_metrics = best[2]
                changes += int(best[1] != old_indices)
        trace.append(
            {
                "iteration": iteration,
                "stage": "COORDINATE_DESCENT",
                "coordinate_changes": changes,
                "metrics": current_metrics,
                "objective": objective(current_metrics)["composite_cost"],
            }
        )
        if changes == 0:
            break

    candidate = thresholds_from_indices(indices, boundaries)
    candidate_metrics, *_ = evaluate(inputs, candidate)
    if candidate_metrics["hard_action_conflict_count"] != TARGET_CONFLICTS:
        raise RuntimeError("deterministic search did not reach exactly 18 hard conflicts")
    return candidate, trace, evaluated


def serialized_thresholds(
    inputs: dict[str, Any],
    thresholds: dict[tuple[int, int], tuple[float, float]],
    policy_name: str,
) -> dict[str, Any]:
    modalities: dict[str, dict[str, dict[str, float]]] = {name: {} for name in MODALITIES}
    for product_id, product in enumerate(inputs["products"]):
        for modality_id, modality in enumerate(MODALITIES):
            low, high = thresholds[(product_id, modality_id)]
            modalities[modality][product] = {
                "review_low": low,
                "review_high": high,
            }
    return {
        "schema_version": "industrial-controlled-thresholds/v1",
        "version": VERSION,
        "status": STATUS,
        "identity_authority": IDENTITY_AUTHORITY,
        "policy_name": policy_name,
        "state_rule": (
            "score < review_low => normal; review_low <= score < review_high => review; "
            "score >= review_high => anomaly"
        ),
        "scalar_threshold_count": 40,
        "products": inputs["products"],
        "modalities": modalities,
    }


def production_guard_paths() -> list[Path]:
    paths = [
        PRODUCTION_BANDS_PATH,
        REPO_ROOT / "scenes/industrial_anomaly/industrial_anomaly/plugin.py",
        REPO_ROOT / "scenes/industrial_anomaly/deployment/cloud_service_traffic_industrial_formal.json",
    ]
    return [path for path in paths if path.exists()]


def artifact_checksums(output: Path, install: Path) -> list[tuple[str, str]]:
    artifact_paths = sorted(
        path
        for path in list(output.rglob("*")) + list(install.rglob("*"))
        if path.is_file()
        and path.name not in {"SHA256SUMS", "verification.json"}
    )
    artifact_paths.extend([Path(__file__).resolve(), VERIFIER_PATH.resolve()])
    unique = sorted(set(artifact_paths), key=lambda path: relative(path))
    return [(sha256_file(path), relative(path)) for path in unique]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--install-dir", type=Path, default=DEFAULT_INSTALL)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    install = args.install_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    install.mkdir(parents=True, exist_ok=True)

    production_before = {relative(path): sha256_file(path) for path in production_guard_paths()}
    inputs = load_inputs()
    initial = controlled_initialization(inputs)
    candidate, trace, evaluations = optimize(inputs, initial)
    initial_metrics, initial_states, initial_conflicts, initial_reviews, initial_predictions = evaluate(
        inputs, initial
    )
    candidate_metrics, candidate_states, candidate_conflicts, candidate_reviews, candidate_predictions = evaluate(
        inputs, candidate
    )
    initial_objective = objective(initial_metrics)
    candidate_objective = objective(candidate_metrics)

    non_regression = {
        "hard_action_conflict_count_not_higher": (
            candidate_metrics["hard_action_conflict_count"]
            <= initial_metrics["hard_action_conflict_count"]
        ),
        "accuracy_not_lower": candidate_metrics["accuracy"] >= initial_metrics["accuracy"],
        "macro_f1_not_lower": candidate_metrics["macro_f1"] >= initial_metrics["macro_f1"],
        "anomaly_recall_not_lower": (
            candidate_metrics["anomaly_recall"] >= initial_metrics["anomaly_recall"]
        ),
        "false_release_not_higher": (
            candidate_metrics["false_release"] <= initial_metrics["false_release"]
        ),
        "false_quarantine_not_higher": (
            candidate_metrics["false_quarantine"] <= initial_metrics["false_quarantine"]
        ),
        "any_review_not_higher": (
            candidate_metrics["any_review_count"] <= initial_metrics["any_review_count"]
        ),
        "composite_cost_lower": (
            candidate_objective["composite_cost"] < initial_objective["composite_cost"]
        ),
    }
    if not all(non_regression.values()):
        raise RuntimeError(f"candidate failed controlled non-regression gates: {non_regression}")

    initial_config = serialized_thresholds(
        inputs, initial, "CONTROLLED_SUBOPTIMAL_INITIALIZATION"
    )
    candidate_config = serialized_thresholds(
        inputs, candidate, "MULTI_OBJECTIVE_OPTIMIZED_CANDIDATE"
    )
    write_json(output / "controlled_initial_thresholds.json", initial_config)
    write_json(output / "optimized_candidate_thresholds.json", candidate_config)

    pair_manifest = []
    for index, key in enumerate(inputs["keys"]):
        product, sample_id = key
        pair_manifest.append(
            {
                "pair_index": index,
                "product": product,
                "sample_id": sample_id,
                "truth": "defect" if inputs["truth"][index] else "normal",
                "rgb_path": inputs["rgb"][key]["path"],
                "infrared_path": inputs["infrared"][key]["path"],
                "rgb_score": float(inputs["scores"][index, 0]),
                "infrared_score": float(inputs["scores"][index, 1]),
            }
        )
    pair_manifest_sha = canonical_sha256(pair_manifest)

    input_files = inputs["rgb_files"] + inputs["infrared_files"] + [
        OOF_PATH,
        PRODUCTION_BANDS_PATH,
    ]
    input_manifest = {
        "schema_version": "industrial-controlled-threshold-input-manifest/v1",
        "version": VERSION,
        "status": STATUS,
        "identity_authority": IDENTITY_AUTHORITY,
        "identity_statement": (
            "The user declares the selected 443 RGB/infrared rows and ET10 rows to be "
            "the same canonical input scope."
        ),
        "row_level_legacy_18_conflict_list_available": False,
        "demo_conflict_ids_are_legacy_ids": False,
        "pairs": 443,
        "truth_normal": 100,
        "truth_defect": 343,
        "retained_et10_pair_identity_sha256": (
            "72d808e7f0da4da3899cd0008992128bd6ad64ee5522a3fceffc64fbe72bfdd0"
        ),
        "controlled_pair_manifest_sha256": pair_manifest_sha,
        "files": [
            {
                "path": relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(input_files, key=relative)
        ],
    }
    write_json(output / "input_manifest.json", input_manifest)

    metrics_document = {
        "schema_version": "industrial-controlled-threshold-metrics/v1",
        "version": VERSION,
        "status": STATUS,
        "identity_authority": IDENTITY_AUTHORITY,
        "decision_contract": {
            "pre_fusion_hard_conflict": (
                "RGB normal + infrared anomaly, or RGB anomaly + infrared normal; "
                "review is excluded"
            ),
            "final_binary_decision": (
                "both normal => normal; both anomaly => defect; every other state pair => "
                "fixed calibrated ET10 nested-OOF prediction"
            ),
            "et10_retrained": False,
            "accuracy_f1_recall_scope": "443 final binary workpiece decisions under this fixed hybrid contract",
            "not_the_cloud_et_metric": (
                "These before/after values are not the separate 85.33%->93.23% ET-only comparison."
            ),
        },
        "before": initial_metrics,
        "after": candidate_metrics,
        "delta_after_minus_before": {
            "hard_action_conflict_count": (
                candidate_metrics["hard_action_conflict_count"]
                - initial_metrics["hard_action_conflict_count"]
            ),
            "hard_action_conflict_rate": (
                candidate_metrics["hard_action_conflict_rate"]
                - initial_metrics["hard_action_conflict_rate"]
            ),
            "any_review_count": candidate_metrics["any_review_count"] - initial_metrics["any_review_count"],
            "accuracy_percentage_points": 100.0 * (
                candidate_metrics["accuracy"] - initial_metrics["accuracy"]
            ),
            "macro_f1_percentage_points": 100.0 * (
                candidate_metrics["macro_f1"] - initial_metrics["macro_f1"]
            ),
            "anomaly_recall_percentage_points": 100.0 * (
                candidate_metrics["anomaly_recall"] - initial_metrics["anomaly_recall"]
            ),
            "false_release": candidate_metrics["false_release"] - initial_metrics["false_release"],
            "false_quarantine": (
                candidate_metrics["false_quarantine"] - initial_metrics["false_quarantine"]
            ),
        },
        "gates": {
            **non_regression,
            "after_hard_action_conflict_exactly_18": (
                candidate_metrics["hard_action_conflict_count"] == TARGET_CONFLICTS
            ),
            "after_hard_action_conflict_rate_below_5_percent": (
                candidate_metrics["hard_action_conflict_rate"] < 0.05
            ),
        },
        "decision": "PASS_CONTROLLED_DEVELOPMENT_DEMO",
        "production_activation_allowed": False,
    }
    write_json(output / "metrics.json", metrics_document)

    objective_document = {
        "schema_version": "industrial-controlled-threshold-objective/v1",
        "version": VERSION,
        "status": STATUS,
        "selection_constraint": "hard_action_conflict_count == 18",
        "selection_method": (
            "deterministic lexicographic coordinate descent over observed-score transition "
            "midpoints; exact conflict target first, then minimum composite cost"
        ),
        "full_443_labels_used": True,
        "evaluated_coordinate_candidates": evaluations,
        "before": initial_objective,
        "after": candidate_objective,
        "absolute_cost_reduction": (
            initial_objective["composite_cost"] - candidate_objective["composite_cost"]
        ),
        "relative_cost_reduction": (
            (initial_objective["composite_cost"] - candidate_objective["composite_cost"])
            / initial_objective["composite_cost"]
        ),
        "trace": trace,
    }
    write_json(output / "objective_breakdown.json", objective_document)

    csv_path = output / "paired_before_after.csv"
    fieldnames = [
        "pair_index",
        "product",
        "sample_id",
        "truth",
        "rgb_path",
        "infrared_path",
        "rgb_score",
        "infrared_score",
        "et10_oof_prediction",
        "before_rgb_review_low",
        "before_rgb_review_high",
        "before_rgb_state",
        "before_infrared_review_low",
        "before_infrared_review_high",
        "before_infrared_state",
        "before_hard_action_conflict",
        "before_any_review",
        "before_final_prediction",
        "before_correct",
        "after_rgb_review_low",
        "after_rgb_review_high",
        "after_rgb_state",
        "after_infrared_review_low",
        "after_infrared_review_high",
        "after_infrared_state",
        "after_hard_action_conflict",
        "after_any_review",
        "after_final_prediction",
        "after_correct",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, item in enumerate(pair_manifest):
            product_id = int(inputs["product_index"][index])
            before_rgb = initial[(product_id, 0)]
            before_infrared = initial[(product_id, 1)]
            after_rgb = candidate[(product_id, 0)]
            after_infrared = candidate[(product_id, 1)]
            truth = int(inputs["truth"][index])
            writer.writerow(
                {
                    **item,
                    "rgb_score": format(item["rgb_score"], ".17g"),
                    "infrared_score": format(item["infrared_score"], ".17g"),
                    "et10_oof_prediction": (
                        "defect" if inputs["resolver_prediction"][index] else "normal"
                    ),
                    "before_rgb_review_low": format(before_rgb[0], ".17g"),
                    "before_rgb_review_high": format(before_rgb[1], ".17g"),
                    "before_rgb_state": STATE_NAMES[initial_states[index, 0]],
                    "before_infrared_review_low": format(before_infrared[0], ".17g"),
                    "before_infrared_review_high": format(before_infrared[1], ".17g"),
                    "before_infrared_state": STATE_NAMES[initial_states[index, 1]],
                    "before_hard_action_conflict": str(bool(initial_conflicts[index])).lower(),
                    "before_any_review": str(bool(initial_reviews[index])).lower(),
                    "before_final_prediction": "defect" if initial_predictions[index] else "normal",
                    "before_correct": str(bool(initial_predictions[index] == truth)).lower(),
                    "after_rgb_review_low": format(after_rgb[0], ".17g"),
                    "after_rgb_review_high": format(after_rgb[1], ".17g"),
                    "after_rgb_state": STATE_NAMES[candidate_states[index, 0]],
                    "after_infrared_review_low": format(after_infrared[0], ".17g"),
                    "after_infrared_review_high": format(after_infrared[1], ".17g"),
                    "after_infrared_state": STATE_NAMES[candidate_states[index, 1]],
                    "after_hard_action_conflict": str(bool(candidate_conflicts[index])).lower(),
                    "after_any_review": str(bool(candidate_reviews[index])).lower(),
                    "after_final_prediction": "defect" if candidate_predictions[index] else "normal",
                    "after_correct": str(bool(candidate_predictions[index] == truth)).lower(),
                }
            )

    demo_conflict_ids = [
        {
            "pair_index": int(index),
            "product": inputs["keys"][int(index)][0],
            "sample_id": inputs["keys"][int(index)][1],
        }
        for index in np.flatnonzero(candidate_conflicts)
    ]
    write_json(
        output / "demo_candidate_hard_conflict_ids.json",
        {
            "schema_version": "industrial-demo-conflict-ids/v1",
            "version": VERSION,
            "status": STATUS,
            "identity_authority": IDENTITY_AUTHORITY,
            "count": len(demo_conflict_ids),
            "ids": demo_conflict_ids,
            "warning": (
                "These IDs are computed by this controlled candidate. They are not a "
                "reconstruction of the unavailable legacy canonical 18-ID list."
            ),
        },
    )

    protocol = {
        "schema_version": "industrial-controlled-threshold-protocol/v1",
        "version": VERSION,
        "status": STATUS,
        "experiment_name": "CONTROLLED_GLOBAL_THRESHOLD_OPTIMIZATION_DEMO",
        "initialization_name": "CONTROLLED_SUBOPTIMAL_INITIALIZATION",
        "identity_authority": IDENTITY_AUTHORITY,
        "canonical_binding_scope": (
            "user-declared same 443 RGB/infrared score rows and ET10 fusion rows"
        ),
        "legacy_18_ids_available": False,
        "production_history_claimed": False,
        "full_443_labels_used_for_selection": True,
        "independent_test": False,
        "production_thresholds_modified": False,
        "production_service_modified": False,
        "production_endpoint_18100_touched": False,
        "candidate_conflict_target": "18/443",
        "controlled_initialization": {
            "product_order": inputs["products"],
            "pattern": "HHLLLLXXXX",
            "H": "both modalities q50/q95",
            "L": "both modalities q05/q50",
            "X": "RGB q70/q95 and infrared q05/q30",
            "labels_used": False,
            "purpose": "intentionally suboptimal demonstration baseline",
        },
        "limitations": [
            "All 443 labels participate in threshold selection.",
            "The result is a development demo and cannot be claimed as held-out or production evidence.",
            "The row-level legacy canonical 18-conflict ID list is unavailable.",
            "The 18 demo candidate IDs are newly computed and are not the unavailable legacy IDs.",
        ],
    }
    write_json(output / "protocol.json", protocol)

    production_after = {relative(path): sha256_file(path) for path in production_guard_paths()}
    if production_after != production_before:
        raise RuntimeError("production guard file changed during demo generation")

    install_config = install / "optimized_candidate_thresholds.json"
    shutil.copyfile(output / "optimized_candidate_thresholds.json", install_config)
    candidate_sha = sha256_file(install_config)
    registry = {
        "schema_version": "industrial-controlled-demo-registry/v1",
        "registry_scope": "ISOLATED_DEVELOPMENT_DEMO_ONLY",
        "active_demo_version": VERSION,
        "production_active_version": None,
        "production_registry_modified": False,
        "rollback_target": "NO_ACTIVE_DEMO_VERSION",
        "versions": {
            VERSION: {
                "status": STATUS,
                "config_path": relative(install_config),
                "config_sha256": candidate_sha,
                "activation_scope": "offline_demo_registry_only",
            }
        },
    }
    write_json(install / "registry.json", registry)
    install_receipt = {
        "schema_version": "industrial-controlled-demo-install-receipt/v1",
        "version": VERSION,
        "status": STATUS,
        "installed": True,
        "installed_scope": "isolated offline demo registry",
        "config_path": relative(install_config),
        "config_sha256": candidate_sha,
        "registry_path": relative(install / "registry.json"),
        "rollback": {
            "previous_active_demo_version": None,
            "target": "NO_ACTIVE_DEMO_VERSION",
            "action": "clear active_demo_version in the isolated demo registry only",
        },
        "production_guard_sha256_before": production_before,
        "production_guard_sha256_after": production_after,
        "production_mutation": False,
        "production_endpoint_18100_touched": False,
    }
    write_json(output / "install_receipt.json", install_receipt)

    report = f"""# 工业全局阈值优化受控演示

状态：`{STATUS}`。该实验不是production v0，不覆盖模型1.0.0、canonical 18/443证据或18100服务。

身份口径：`{IDENTITY_AUTHORITY}`。按用户裁定，当前RGB/红外443组与ET10逐件结果绑定为同一canonical输入范围；旧canonical 18个样本ID清单仍不可得。本实验输出的18个ID是新候选现场计算结果，不冒充旧清单。

## 固定决策合同

阈值先产生RGB/红外三态。只有`normal`与`anomaly`互斥才计硬冲突，`review`不计。两模态都为`normal`时最终判normal，两模态都为`anomaly`时最终判defect；其余组合固定使用已有ET10 nested-OOF逐件预测。ET不重训、不改cutoff。因此下表的Accuracy/F1不是另一条85.33%→93.23%的ET-only指标。

| 指标 | CONTROLLED_SUBOPTIMAL_INITIALIZATION | 多目标候选 | 变化 |
|---|---:|---:|---:|
| 硬冲突 | {initial_metrics['hard_action_conflict_count']}/443 ({initial_metrics['hard_action_conflict_rate']:.6%}) | {candidate_metrics['hard_action_conflict_count']}/443 ({candidate_metrics['hard_action_conflict_rate']:.6%}) | {candidate_metrics['hard_action_conflict_count']-initial_metrics['hard_action_conflict_count']} |
| any-review | {initial_metrics['any_review_count']}/443 ({initial_metrics['any_review_rate']:.6%}) | {candidate_metrics['any_review_count']}/443 ({candidate_metrics['any_review_rate']:.6%}) | {candidate_metrics['any_review_count']-initial_metrics['any_review_count']} |
| Accuracy | {initial_metrics['accuracy']:.6%} | {candidate_metrics['accuracy']:.6%} | {(candidate_metrics['accuracy']-initial_metrics['accuracy'])*100:+.6f} pp |
| Macro-F1 | {initial_metrics['macro_f1']:.6%} | {candidate_metrics['macro_f1']:.6%} | {(candidate_metrics['macro_f1']-initial_metrics['macro_f1'])*100:+.6f} pp |
| 异常Recall | {initial_metrics['anomaly_recall']:.6%} | {candidate_metrics['anomaly_recall']:.6%} | {(candidate_metrics['anomaly_recall']-initial_metrics['anomaly_recall'])*100:+.6f} pp |
| 误放行FR | {initial_metrics['false_release']} | {candidate_metrics['false_release']} | {candidate_metrics['false_release']-initial_metrics['false_release']} |
| 误隔离FQ | {initial_metrics['false_quarantine']} | {candidate_metrics['false_quarantine']} | {candidate_metrics['false_quarantine']-initial_metrics['false_quarantine']} |
| 综合代价 | {initial_objective['composite_cost']:.12f} | {candidate_objective['composite_cost']:.12f} | {candidate_objective['composite_cost']-initial_objective['composite_cost']:+.12f} |

综合代价公式：`{initial_objective['formula']}`。候选先满足硬冲突精确18/443，再在可行配置中最小化该代价；七个组成指标相对受控初始值全部不退。

候选使用全部443标签选择，故只能标记`{STATUS}`。它已安装到独立离线demo registry，可按install receipt回滚为无active demo；production配置和服务SHA在安装前后相同。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")

    checksums = artifact_checksums(output, install)
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest}  {path}\n" for digest, path in checksums),
        encoding="utf-8",
    )

    summary = {
        "status": STATUS,
        "version": VERSION,
        "identity_authority": IDENTITY_AUTHORITY,
        "before": initial_metrics,
        "after": candidate_metrics,
        "before_composite_cost": initial_objective["composite_cost"],
        "after_composite_cost": candidate_objective["composite_cost"],
        "candidate_config_sha256": candidate_sha,
        "output_dir": relative(output),
        "install_dir": relative(install),
        "verification_command": (
            f"/home/scw/miniconda3/envs/traffic/bin/python {relative(VERIFIER_PATH)} "
            f"--output-dir {relative(output)} --install-dir {relative(install)}"
        ),
    }
    write_json(output / "summary.json", summary)
    # summary is written last; refresh checksums once so it is covered.
    checksums = artifact_checksums(output, install)
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest}  {path}\n" for digest, path in checksums),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
