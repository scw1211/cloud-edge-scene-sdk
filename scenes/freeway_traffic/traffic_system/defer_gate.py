"""用途：加载选择性协同门控器，在规则、边缘 Student 与云端复核间选择。"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

from traffic_system.decision_utils import DECISION_CLASSES


GATE_CLASSES = ["local_rule", "edge_student", "defer_cloud"]


def build_gate_features(
    base_features: np.ndarray,
    rule_predictions: np.ndarray,
    student_predictions: np.ndarray,
    student_confidences: np.ndarray,
) -> np.ndarray:
    base = np.asarray(base_features, dtype=np.float64)
    rule = np.asarray(rule_predictions, dtype=np.int64).reshape(-1)
    student = np.asarray(student_predictions, dtype=np.int64).reshape(-1)
    confidence = np.asarray(student_confidences, dtype=np.float64).reshape(-1)
    if base.ndim != 2:
        raise ValueError("base_features must be a two-dimensional array")
    count = len(base)
    if len(rule) != count or len(student) != count or len(confidence) != count:
        raise ValueError("gate feature arrays have different row counts")
    class_count = len(DECISION_CLASSES)
    if np.any(rule < 0) or np.any(rule >= class_count):
        raise ValueError("rule prediction is outside decision classes")
    if np.any(student < 0) or np.any(student >= class_count):
        raise ValueError("student prediction is outside decision classes")
    if not np.isfinite(base).all() or not np.isfinite(confidence).all():
        raise ValueError("gate features contain NaN or Inf")
    rule_one_hot = np.eye(class_count, dtype=np.float64)[rule]
    student_one_hot = np.eye(class_count, dtype=np.float64)[student]
    agreement = (rule == student).astype(np.float64).reshape(-1, 1)
    return np.concatenate(
        [base, rule_one_hot, student_one_hot, confidence.reshape(-1, 1), agreement],
        axis=1,
    )


def preferred_gate_targets(
    references: np.ndarray,
    rule_predictions: np.ndarray,
    student_predictions: np.ndarray,
) -> np.ndarray:
    truth = np.asarray(references, dtype=np.int64).reshape(-1)
    rule = np.asarray(rule_predictions, dtype=np.int64).reshape(-1)
    student = np.asarray(student_predictions, dtype=np.int64).reshape(-1)
    if len(truth) != len(rule) or len(truth) != len(student):
        raise ValueError("gate target arrays have different row counts")
    rule_correct = rule == truth
    student_correct = student == truth
    targets = np.full(len(truth), GATE_CLASSES.index("defer_cloud"), dtype=np.int64)
    targets[student_correct & ~rule_correct] = GATE_CLASSES.index("edge_student")
    targets[rule_correct] = GATE_CLASSES.index("local_rule")
    return targets


def load_defer_gate(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as artifact:
            required = {
                "schema_version",
                "feature_dim",
                "confidence_threshold",
                "gate_classes",
                "decision_classes",
                "base_feature_names",
                "tree_offsets",
                "children_left",
                "children_right",
                "features",
                "thresholds",
                "leaf_probabilities",
            }
            missing = sorted(required.difference(artifact.files))
            if missing:
                raise ValueError("Portable defer gate is missing: {}".format(", ".join(missing)))
            payload = {
                "format": "portable_extra_trees_npz",
                "schema_version": int(artifact["schema_version"]),
                "feature_dim": int(artifact["feature_dim"]),
                "gate_classes": artifact["gate_classes"].astype(str).tolist(),
                "decision_classes": artifact["decision_classes"].astype(str).tolist(),
                "base_feature_names": artifact["base_feature_names"].astype(str).tolist(),
                "tree_offsets": artifact["tree_offsets"].astype(np.int32),
                "children_left": artifact["children_left"].astype(np.int32),
                "children_right": artifact["children_right"].astype(np.int32),
                "features": artifact["features"].astype(np.int16),
                "thresholds": artifact["thresholds"].astype(np.float32),
                "leaf_probabilities": artifact["leaf_probabilities"].astype(np.float32),
                "metadata": {
                    "confidence_threshold": float(artifact["confidence_threshold"]),
                },
            }
        if payload["schema_version"] != 1:
            raise ValueError("Unsupported portable defer gate schema")
    else:
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError("Invalid defer gate: {}".format(path))
        payload["format"] = "sklearn_joblib"
        if hasattr(payload["model"], "n_jobs"):
            payload["model"].n_jobs = 1
    if list(payload.get("gate_classes", [])) != GATE_CLASSES:
        raise ValueError("Defer gate class order does not match runtime")
    if list(payload.get("decision_classes", [])) != DECISION_CLASSES:
        raise ValueError("Defer gate decision class order does not match runtime")
    return payload


def export_portable_defer_gate(payload: Dict[str, Any], path: Path) -> None:
    model = payload.get("model")
    if model is None or not hasattr(model, "estimators_"):
        raise ValueError("A fitted sklearn tree ensemble is required for export")
    offsets: List[int] = [0]
    children_left: List[np.ndarray] = []
    children_right: List[np.ndarray] = []
    features: List[np.ndarray] = []
    thresholds: List[np.ndarray] = []
    leaf_probabilities: List[np.ndarray] = []
    node_offset = 0
    for estimator in model.estimators_:
        tree = estimator.tree_
        left = tree.children_left.astype(np.int32)
        right = tree.children_right.astype(np.int32)
        left = np.where(left >= 0, left + node_offset, -1).astype(np.int32)
        right = np.where(right >= 0, right + node_offset, -1).astype(np.int32)
        raw_values = np.asarray(tree.value[:, 0, :], dtype=np.float64)
        full_values = np.zeros((tree.node_count, len(GATE_CLASSES)), dtype=np.float64)
        for local_column, class_id in enumerate(estimator.classes_.astype(int).tolist()):
            full_values[:, class_id] = raw_values[:, local_column]
        totals = np.sum(full_values, axis=1, keepdims=True)
        probabilities = np.divide(
            full_values,
            totals,
            out=np.zeros_like(full_values),
            where=totals > 0.0,
        )
        children_left.append(left)
        children_right.append(right)
        features.append(tree.feature.astype(np.int16))
        thresholds.append(tree.threshold.astype(np.float32))
        leaf_probabilities.append(probabilities.astype(np.float32))
        node_offset += int(tree.node_count)
        offsets.append(node_offset)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int16),
        feature_dim=np.asarray(payload["feature_dim"], dtype=np.int32),
        confidence_threshold=np.asarray(
            payload["metadata"]["confidence_threshold"], dtype=np.float32
        ),
        gate_classes=np.asarray(GATE_CLASSES),
        decision_classes=np.asarray(DECISION_CLASSES),
        base_feature_names=np.asarray(payload["base_feature_names"]),
        tree_offsets=np.asarray(offsets, dtype=np.int32),
        children_left=np.concatenate(children_left),
        children_right=np.concatenate(children_right),
        features=np.concatenate(features),
        thresholds=np.concatenate(thresholds),
        leaf_probabilities=np.concatenate(leaf_probabilities, axis=0),
    )


def portable_predict_proba(values: np.ndarray, payload: Dict[str, Any]) -> np.ndarray:
    probabilities = np.zeros((len(values), len(GATE_CLASSES)), dtype=np.float64)
    offsets = payload["tree_offsets"]
    left = payload["children_left"]
    right = payload["children_right"]
    features = payload["features"]
    thresholds = payload["thresholds"]
    leaf_probabilities = payload["leaf_probabilities"]
    tree_count = len(offsets) - 1
    if tree_count <= 0:
        raise ValueError("Portable defer gate contains no trees")
    for row_index, row in enumerate(values):
        for tree_index in range(tree_count):
            node = int(offsets[tree_index])
            while left[node] >= 0:
                node = int(
                    left[node]
                    if row[int(features[node])] <= float(thresholds[node])
                    else right[node]
                )
            probabilities[row_index] += leaf_probabilities[node]
    return probabilities / float(tree_count)


def predict_defer_gate(
    gate_features: np.ndarray,
    payload: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(gate_features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != int(payload["feature_dim"]):
        raise ValueError("Defer gate feature schema mismatch")
    if payload.get("format") == "portable_extra_trees_npz":
        full_probabilities = portable_predict_proba(values, payload)
    else:
        model = payload["model"]
        probabilities = model.predict_proba(values)
        columns = {int(class_id): index for index, class_id in enumerate(model.classes_)}
        full_probabilities = np.zeros((len(values), len(GATE_CLASSES)), dtype=np.float64)
        for class_id in range(len(GATE_CLASSES)):
            if class_id in columns:
                full_probabilities[:, class_id] = probabilities[:, columns[class_id]]
    choices = np.argmax(full_probabilities, axis=1).astype(np.int64)
    confidences = np.max(full_probabilities, axis=1)
    threshold = float(payload["metadata"]["confidence_threshold"])
    choices[confidences < threshold] = GATE_CLASSES.index("defer_cloud")
    return choices, confidences
