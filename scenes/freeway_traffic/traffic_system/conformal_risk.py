"""用途：校准区域风险概率，并生成用于云边调度的 conformal 风险候选集合。"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from traffic_system.risk_labels import RISK_CLASSES


SCHEMA_VERSION = 1
DEFAULT_METHOD = "marginal_aps"


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(RISK_CLASSES):
        raise ValueError(
            "{} must have shape [N, {}], got {}.".format(name, len(RISK_CLASSES), array.shape)
        )
    if not np.isfinite(array).all():
        raise ValueError("{} contains NaN or Inf.".format(name))
    return array


def _validate_labels(labels: np.ndarray, count: int) -> np.ndarray:
    array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(array) != count:
        raise ValueError("labels length {} does not match predictions {}.".format(len(array), count))
    if np.any(array < 0) or np.any(array >= len(RISK_CLASSES)):
        raise ValueError("labels contain a class outside the configured risk classes.")
    return array


def temperature_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = _as_2d(logits, "logits")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive.")
    scaled = values / float(temperature)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp_values = np.exp(scaled)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def recalibrate_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    values = _as_2d(probabilities, "probabilities")
    if np.any(values < 0.0):
        raise ValueError("probabilities contain negative values.")
    row_sums = np.sum(values, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    normalized = values / row_sums
    logits = np.log(np.clip(normalized, 1e-12, 1.0))
    return temperature_softmax(logits, temperature)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probabilities = temperature_softmax(logits, temperature)
    target = _validate_labels(labels, len(probabilities))
    return float(-np.mean(np.log(np.clip(probabilities[np.arange(len(target)), target], 1e-12, 1.0))))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    minimum: float = 0.05,
    maximum: float = 20.0,
    iterations: int = 96,
) -> float:
    values = _as_2d(logits, "logits")
    target = _validate_labels(labels, len(values))
    if minimum <= 0.0 or maximum <= minimum or iterations <= 0:
        raise ValueError("invalid temperature search range or iteration count.")

    left = math.log(minimum)
    right = math.log(maximum)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    lower = right - ratio * (right - left)
    upper = left + ratio * (right - left)
    lower_loss = negative_log_likelihood(values, target, math.exp(lower))
    upper_loss = negative_log_likelihood(values, target, math.exp(upper))
    for _ in range(iterations):
        if lower_loss <= upper_loss:
            right = upper
            upper = lower
            upper_loss = lower_loss
            lower = right - ratio * (right - left)
            lower_loss = negative_log_likelihood(values, target, math.exp(lower))
        else:
            left = lower
            lower = upper
            lower_loss = upper_loss
            upper = left + ratio * (right - left)
            upper_loss = negative_log_likelihood(values, target, math.exp(upper))
    return float(math.exp((left + right) / 2.0))


def aps_candidate_scores(probabilities: np.ndarray) -> np.ndarray:
    """Return the probability mass ranked strictly above each candidate class.

    The top-ranked class always receives score zero. This deterministic APS form is
    conservative and avoids random prediction sets in the real-time control path.
    """

    values = _as_2d(probabilities, "probabilities")
    if np.any(values < 0.0):
        raise ValueError("probabilities contain negative values.")
    normalized = values / np.sum(values, axis=1, keepdims=True)
    higher = normalized[:, :, None] > normalized[:, None, :]
    return np.sum(normalized[:, :, None] * higher, axis=1)


def true_aps_scores(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    candidate_scores = aps_candidate_scores(probabilities)
    target = _validate_labels(labels, len(candidate_scores))
    return candidate_scores[np.arange(len(target)), target]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("conformal scores must be non-empty and finite.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    rank = min(len(values), max(1, int(math.ceil((len(values) + 1) * (1.0 - alpha)))))
    return float(np.partition(values, rank - 1)[rank - 1])


def class_conditional_quantiles(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = _validate_labels(labels, len(values))
    quantiles: Dict[str, float] = {}
    for class_id, class_name in enumerate(RISK_CLASSES):
        selected = values[target == class_id]
        if not len(selected):
            raise ValueError("calibration split has no examples for class {}.".format(class_name))
        quantiles[class_name] = conformal_quantile(selected, alpha)
    return quantiles


def prediction_mask(
    probabilities: np.ndarray,
    threshold: Union[float, Sequence[float], np.ndarray],
) -> np.ndarray:
    scores = aps_candidate_scores(probabilities)
    limits = np.asarray(threshold, dtype=np.float64)
    if limits.ndim == 0:
        limits = np.full((1, len(RISK_CLASSES)), float(limits), dtype=np.float64)
    elif limits.shape == (len(RISK_CLASSES),):
        limits = limits.reshape(1, -1)
    else:
        raise ValueError("threshold must be scalar or one value per risk class.")
    if not np.isfinite(limits).all():
        raise ValueError("threshold contains NaN or Inf.")
    return scores <= limits + 1e-12


def thresholds_for_method(
    artifact: Mapping[str, Any], method: str
) -> Union[float, np.ndarray]:
    thresholds = artifact.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("risk calibrator has no thresholds object.")
    if method == "class_conditional_aps":
        class_values = thresholds.get(method)
        if not isinstance(class_values, Mapping):
            raise ValueError("risk calibrator has no class-conditional thresholds.")
        return np.asarray([float(class_values[name]) for name in RISK_CLASSES], dtype=np.float64)
    if method not in {"marginal_aps", "simultaneous_window_aps"}:
        raise ValueError("unsupported conformal method: {}.".format(method))
    return float(thresholds[method])


def validate_calibrator(artifact: Mapping[str, Any]) -> None:
    if int(artifact.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported risk calibrator schema version.")
    if list(artifact.get("risk_classes", [])) != list(RISK_CLASSES):
        raise ValueError("risk calibrator class order does not match the model.")
    temperature = float(artifact.get("temperature", 0.0))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("risk calibrator temperature is invalid.")
    method = str(artifact.get("deployment_method", DEFAULT_METHOD))
    thresholds_for_method(artifact, method)


def load_risk_calibrator(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Risk calibrator not found: {}".format(path))
    with path.open("r", encoding="utf-8") as file_obj:
        artifact = json.load(file_obj)
    if not isinstance(artifact, dict):
        raise ValueError("risk calibrator must contain a JSON object.")
    validate_calibrator(artifact)
    return artifact


def calibrated_risk_set(
    probabilities: Union[Sequence[float], np.ndarray],
    artifact: Mapping[str, Any],
    method: Optional[str] = None,
) -> Dict[str, Any]:
    validate_calibrator(artifact)
    values = np.asarray(probabilities, dtype=np.float64).reshape(1, -1)
    calibrated = recalibrate_probabilities(values, float(artifact["temperature"]))
    selected_method = method or str(artifact.get("deployment_method", DEFAULT_METHOD))
    limits = thresholds_for_method(artifact, selected_method)
    mask = prediction_mask(calibrated, limits)[0]
    class_ids = [int(index) for index in np.flatnonzero(mask)]
    if not class_ids:
        raise RuntimeError("conformal calibrator emitted an empty risk set.")
    class_names: List[str] = [RISK_CLASSES[index] for index in class_ids]
    point_id = int(np.argmax(calibrated[0]))
    target_coverage = float(artifact.get("target_coverage", 0.0))
    return {
        "method": selected_method,
        "target_coverage": round(target_coverage, 6),
        "prediction_set": class_names,
        "set_size": len(class_names),
        "ambiguous": len(class_names) > 1,
        "contains_high_or_severe": any(name in {"high", "severe"} for name in class_names),
        "lower_risk_level": class_names[0],
        "upper_risk_level": class_names[-1],
        "point_prediction": RISK_CLASSES[point_id],
        "calibrated_confidence": round(float(calibrated[0, point_id]), 6),
        "calibrated_probabilities": {
            name: round(float(calibrated[0, class_id]), 6)
            for class_id, name in enumerate(RISK_CLASSES)
        },
    }
