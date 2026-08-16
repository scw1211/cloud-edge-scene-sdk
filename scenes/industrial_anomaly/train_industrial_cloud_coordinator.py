#!/usr/bin/env python3
"""Train a leak-resistant industrial RGB/infrared ExtraTrees coordinator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
import sys

for value in (str(PROJECT_ROOT), str(ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from industrial_anomaly.cloud_coordinator import (  # noqa: E402
    FEATURE_NAMES,
    SCHEMA_VERSION,
    feature_vector,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_scores(path: Path) -> Dict[Tuple[str, str], float]:
    result: Dict[Tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            image = PurePosixPath(str(row["path"]))
            key = (image.parent.name, image.stem)
            if key in result:
                raise ValueError("duplicate industrial prediction key: {}".format(key))
            result[key] = float(row["image_score"])
    if not result:
        raise ValueError("industrial predictions CSV is empty")
    return result


def _state(score: float, low: float, high: float) -> str:
    if score < low:
        return "normal"
    if score < high:
        return "review"
    return "anomaly"


def _records(
    key: Tuple[str, str],
    scores: Mapping[str, Mapping[Tuple[str, str], float]],
    bands: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    result = {}
    for modality in ("rgb", "infrared"):
        score = float(scores[modality][key])
        band = bands[modality]["capsule"]
        low = float(band["review_low"])
        high = float(band["review_high"])
        result[modality] = {
            "score": score,
            "review_low": low,
            "review_high": high,
            "state": _state(score, low, high),
        }
    return result


def _metrics(y_true: Sequence[int], y_pred: Sequence[int], probabilities: Any) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return {
        "samples": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 9),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_true, y_pred)), 9
        ),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 9),
        "anomaly_precision": round(float(precision_score(y_true, y_pred)), 9),
        "anomaly_recall": round(float(recall_score(y_true, y_pred)), 9),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 9),
        "confusion_matrix_normal_anomaly": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-predictions", required=True)
    parser.add_argument("--infrared-predictions", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout-fraction", type=float, default=0.31)
    parser.add_argument("--trees", type=int, default=512)
    args = parser.parse_args()

    output_model = Path(args.output_model).resolve()
    output_metrics = Path(args.output_metrics).resolve()
    if output_model.exists() or output_metrics.exists():
        raise FileExistsError("industrial cloud coordinator outputs must not exist")
    rgb_path = Path(args.rgb_predictions).resolve()
    infrared_path = Path(args.infrared_predictions).resolve()
    thresholds_path = Path(args.thresholds).resolve()
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    bands = thresholds["modalities"]
    scores = {
        "rgb": _read_scores(rgb_path),
        "infrared": _read_scores(infrared_path),
    }
    keys = sorted(set(scores["rgb"]) & set(scores["infrared"]))
    if set(scores["rgb"]) != set(scores["infrared"]):
        raise ValueError("RGB and infrared prediction identities do not match")
    if len(keys) < 20:
        raise ValueError("industrial cloud coordinator needs at least 20 paired samples")

    x = []
    y = []
    identities = []
    rule_states = []
    for key in keys:
        pair = _records(key, scores, bands)
        vector, context = feature_vector(pair)
        x.append(vector)
        y.append(0 if key[0] == "good" else 1)
        identities.append("{}/{}".format(*key))
        states = [context[modality]["state"] for modality in ("rgb", "infrared")]
        rule_states.append(
            states[0] if states[0] == states[1] and states[0] != "review" else "review"
        )

    import joblib
    import numpy as np
    import sklearn
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(y), dtype=np.int64)
    train_indices, heldout_indices = train_test_split(
        indices,
        test_size=float(args.heldout_fraction),
        random_state=int(args.seed),
        stratify=np.asarray(y, dtype=np.int64),
    )
    model = ExtraTreesClassifier(
        n_estimators=int(args.trees),
        min_samples_leaf=2,
        class_weight="balanced",
        max_features=None,
        random_state=int(args.seed),
        n_jobs=1,
    )
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.int64)
    model.fit(x_array[train_indices], y_array[train_indices])
    heldout_prediction = model.predict(x_array[heldout_indices])
    heldout_probability = model.predict_proba(x_array[heldout_indices])[:, 1]
    heldout_metrics = _metrics(
        y_array[heldout_indices], heldout_prediction, heldout_probability
    )

    rule_covered = [
        int(index)
        for index in heldout_indices
        if rule_states[int(index)] in {"normal", "anomaly"}
    ]
    rule_correct = sum(
        rule_states[index] == ("anomaly" if y[index] else "normal")
        for index in rule_covered
    )
    training = {
        "dataset": "MulSen_AD/capsule paired Nano TensorRT scores",
        "split_policy": "stratified fixed heldout; heldout rows never fitted",
        "seed": int(args.seed),
        "paired_samples": len(keys),
        "train_samples": len(train_indices),
        "heldout_samples": len(heldout_indices),
        "train_identity_sha256": hashlib.sha256(
            "\n".join(identities[int(index)] for index in sorted(train_indices)).encode()
        ).hexdigest(),
        "heldout_identity_sha256": hashlib.sha256(
            "\n".join(identities[int(index)] for index in sorted(heldout_indices)).encode()
        ).hexdigest(),
        "rgb_predictions": {
            "bytes": rgb_path.stat().st_size,
            "sha256": _sha256(rgb_path),
        },
        "infrared_predictions": {
            "bytes": infrared_path.stat().st_size,
            "sha256": _sha256(infrared_path),
        },
        "thresholds": {
            "bytes": thresholds_path.stat().st_size,
            "sha256": _sha256(thresholds_path),
        },
        "sklearn_version": sklearn.__version__,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_id": "industrial-capsule-cross-modal-extratrees-v1",
        "model_type": "sklearn_extra_trees",
        "feature_names": list(FEATURE_NAMES),
        "decision_classes": {0: "normal", 1: "anomaly"},
        "supported_products": ["capsule"],
        "model": model,
        "training": training,
    }
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_model, compress=3)
    metrics = {
        "schema_version": "industrial-cloud-extratrees-evaluation/v1",
        "model_id": payload["model_id"],
        "artifact": {
            "path": str(output_model),
            "bytes": output_model.stat().st_size,
            "sha256": _sha256(output_model),
        },
        "training": training,
        "heldout": heldout_metrics,
        "deterministic_unanimous_baseline": {
            "heldout_samples": len(heldout_indices),
            "covered_samples": len(rule_covered),
            "coverage": round(len(rule_covered) / len(heldout_indices), 9),
            "covered_accuracy": round(
                rule_correct / max(1, len(rule_covered)), 9
            ),
            "unresolved_review_samples": len(heldout_indices) - len(rule_covered),
        },
        "gates": {
            "heldout_not_fitted": True,
            "heldout_accuracy_at_least_0_90": heldout_metrics["accuracy"] >= 0.90,
            "heldout_balanced_accuracy_at_least_0_90": (
                heldout_metrics["balanced_accuracy"] >= 0.90
            ),
            "heldout_anomaly_recall_at_least_0_95": (
                heldout_metrics["anomaly_recall"] >= 0.95
            ),
        },
    }
    metrics["all_gates_passed"] = all(metrics["gates"].values())
    output_metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
