"""用途：加载云端专用协调分类器，并将分类结果转换为安全控制决策。"""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np

from traffic_system.decision_utils import build_decision_from_student_class, extract_feature_vector


def load_cloud_model(path: Path) -> Dict[str, Any]:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "model" not in payload or "decision_classes" not in payload:
        raise ValueError("Invalid cloud coordinator model: {}".format(path))
    if hasattr(payload["model"], "n_jobs"):
        payload["model"].n_jobs = 1
    return payload


def predict_cloud_batch(
    events: Sequence[Dict[str, Any]], payload: Dict[str, Any]
) -> List[Tuple[str, float]]:
    if not events:
        return []
    vectors = []
    for event in events:
        vector, names = extract_feature_vector(event)
        if list(names) != list(payload["feature_names"]):
            raise ValueError("Cloud coordinator feature schema mismatch.")
        vectors.append(vector)
    model = payload["model"]
    x = np.asarray(vectors, dtype=np.float64)
    class_ids = model.predict(x)
    probabilities = model.predict_proba(x)
    class_columns = {int(value): index for index, value in enumerate(model.classes_)}
    predictions = []
    for row_index, raw_class_id in enumerate(class_ids):
        class_id = int(raw_class_id)
        confidence = float(probabilities[row_index][class_columns[class_id]])
        predictions.append((str(payload["decision_classes"][class_id]), confidence))
    return predictions


def predict_cloud(event: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[str, float]:
    return predict_cloud_batch([event], payload)[0]


def cloud_decisions(
    events: Sequence[Dict[str, Any]], payload: Dict[str, Any], policy_version: str
) -> List[Dict[str, Any]]:
    decisions = []
    for event, (decision_class, confidence) in zip(events, predict_cloud_batch(events, payload)):
        decision = build_decision_from_student_class(
            event, decision_class, confidence, decision_source="cloud_extratrees_coordinator"
        )
        decision["policy_version"] = policy_version
        decisions.append(decision)
    return decisions


def cloud_decision(event: Dict[str, Any], payload: Dict[str, Any], policy_version: str) -> Dict[str, Any]:
    return cloud_decisions([event], payload, policy_version)[0]
