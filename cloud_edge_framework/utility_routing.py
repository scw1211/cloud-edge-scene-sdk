"""用途：加载轻量学习模型估计云复核收益，并在安全规则之外提出额外云请求。"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


SCHEMA_VERSION = "utility-router/v1"
FEATURE_NAMES = [
    "risk_priority",
    "risk_score",
    "prediction_confidence",
    "uncertainty_confidence",
    "prediction_set_size",
    "deadline_ms",
    "edge_work_ms",
    "network_available",
    "network_rtt_ms",
    "network_jitter_ms",
    "network_loss_rate",
    "uplink_mbps",
    "downlink_mbps",
    "planned_request_kb",
    "measured_cloud_path_ms",
    "conflict_suspected",
    "model_disagreement",
    "monitoring_force_cloud_review",
    "evidence_feature",
    "evidence_raw",
]


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        value = 1 if value else 0
    if not isinstance(value, (int, float)):
        raise ValueError("utility router {} must be numeric".format(field))
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("utility router {} must be finite".format(field))
    return result


def _sigmoid(value: float) -> float:
    if value >= 0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


def vectorize(features: Mapping[str, Any]) -> List[float]:
    evidence_level = str(features.get("evidence_level", "summary"))
    measured = features.get("measured_cloud_path_ms")
    if measured is None:
        measured = (
            _finite(features.get("network_rtt_ms", 0.0), "network_rtt_ms")
            + 1.645
            * _finite(features.get("network_jitter_ms", 0.0), "network_jitter_ms")
        )
    values = {
        "risk_priority": features.get("risk_priority", 0.0),
        "risk_score": features.get("risk_score", 0.0),
        "prediction_confidence": features.get("prediction_confidence", 0.0),
        "uncertainty_confidence": features.get("uncertainty_confidence", 0.0),
        "prediction_set_size": features.get("prediction_set_size", 1.0),
        "deadline_ms": features.get("deadline_ms", 200.0),
        "edge_work_ms": features.get("edge_work_ms", 0.0),
        "network_available": features.get("network_available", 0.0),
        "network_rtt_ms": features.get("network_rtt_ms", 0.0),
        "network_jitter_ms": features.get("network_jitter_ms", 0.0),
        "network_loss_rate": features.get("network_loss_rate", 0.0),
        "uplink_mbps": features.get("uplink_mbps", 0.0),
        "downlink_mbps": features.get("downlink_mbps", 0.0),
        "planned_request_kb": _finite(
            features.get("planned_request_bytes", 0.0), "planned_request_bytes"
        )
        / 1024.0,
        "measured_cloud_path_ms": measured,
        "conflict_suspected": features.get("conflict_suspected", 0.0),
        "model_disagreement": features.get("model_disagreement", 0.0),
        "monitoring_force_cloud_review": features.get(
            "monitoring_force_cloud_review", 0.0
        ),
        "evidence_feature": 1.0 if evidence_level == "feature" else 0.0,
        "evidence_raw": 1.0 if evidence_level == "raw" else 0.0,
    }
    return [_finite(values[name], name) for name in FEATURE_NAMES]


@dataclass(frozen=True)
class UtilityRoutePrediction:
    request_cloud: bool
    correction_probability: float
    utility: float
    threshold: float
    model_id: str
    mode: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LearnedUtilityRouter:
    """A dependency-free logistic utility model guarded by the rule scheduler."""

    def __init__(self, artifact: Mapping[str, Any], mode: str = "shadow") -> None:
        if str(artifact.get("schema_version", "")) != SCHEMA_VERSION:
            raise ValueError("unsupported utility router schema_version")
        self.mode = str(mode)
        if self.mode not in {"shadow", "active"}:
            raise ValueError("utility router mode must be shadow or active")
        feature_names = list(artifact.get("feature_names", []))
        if feature_names != FEATURE_NAMES:
            raise ValueError("utility router feature schema does not match runtime")
        model = artifact.get("model")
        utility = artifact.get("utility")
        if not isinstance(model, dict) or not isinstance(utility, dict):
            raise ValueError("utility router artifact must contain model and utility")
        self.model_id = str(artifact.get("model_id", "")).strip()
        if not self.model_id:
            raise ValueError("utility router model_id must not be empty")
        self.weights = [_finite(value, "model.weights") for value in model.get("weights", [])]
        self.means = [_finite(value, "model.means") for value in model.get("means", [])]
        self.scales = [_finite(value, "model.scales") for value in model.get("scales", [])]
        if not (
            len(self.weights) == len(self.means) == len(self.scales) == len(FEATURE_NAMES)
        ):
            raise ValueError("utility router model vectors have invalid length")
        if any(scale <= 0.0 for scale in self.scales):
            raise ValueError("utility router scales must be positive")
        self.bias = _finite(model.get("bias", 0.0), "model.bias")
        self.correction_value = _finite(
            utility.get("correction_value", 1.0), "utility.correction_value"
        )
        self.latency_cost_per_ms = _finite(
            utility.get("latency_cost_per_ms", 0.0), "utility.latency_cost_per_ms"
        )
        self.byte_cost_per_kb = _finite(
            utility.get("byte_cost_per_kb", 0.0), "utility.byte_cost_per_kb"
        )
        self.threshold = _finite(utility.get("threshold", 0.0), "utility.threshold")

    @classmethod
    def load(cls, path: Path, mode: str = "shadow") -> "LearnedUtilityRouter":
        source = Path(path).resolve()
        with source.open("r", encoding="utf-8") as file_obj:
            artifact = json.load(file_obj)
        if not isinstance(artifact, dict):
            raise ValueError("utility router artifact must be an object")
        return cls(artifact, mode=mode)

    def predict(self, features: Mapping[str, Any]) -> UtilityRoutePrediction:
        raw = vectorize(features)
        normalized = [
            (value - mean) / scale
            for value, mean, scale in zip(raw, self.means, self.scales)
        ]
        logit = self.bias + sum(
            weight * value for weight, value in zip(self.weights, normalized)
        )
        probability = _sigmoid(logit)
        cloud_path_ms = raw[FEATURE_NAMES.index("measured_cloud_path_ms")]
        request_kb = raw[FEATURE_NAMES.index("planned_request_kb")]
        utility = (
            self.correction_value * probability
            - self.latency_cost_per_ms * cloud_path_ms
            - self.byte_cost_per_kb * request_kb
        )
        network_available = bool(raw[FEATURE_NAMES.index("network_available")])
        return UtilityRoutePrediction(
            request_cloud=network_available and utility >= self.threshold,
            correction_probability=round(probability, 6),
            utility=round(utility, 6),
            threshold=self.threshold,
            model_id=self.model_id,
            mode=self.mode,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "mode": self.mode,
            "model_id": self.model_id,
            "schema_version": SCHEMA_VERSION,
            "feature_count": len(FEATURE_NAMES),
            "safety_boundary": "may request cloud but cannot suppress rule-required review",
        }


def build_artifact(
    model_id: str,
    weights: Sequence[float],
    bias: float,
    means: Sequence[float],
    scales: Sequence[float],
    correction_value: float = 1.0,
    latency_cost_per_ms: float = 0.0,
    byte_cost_per_kb: float = 0.0,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """Build a validated artifact from an offline trainer without importing ML libraries."""
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "model_id": str(model_id),
        "feature_names": list(FEATURE_NAMES),
        "model": {
            "type": "standardized_logistic_regression",
            "weights": list(weights),
            "bias": float(bias),
            "means": list(means),
            "scales": list(scales),
        },
        "utility": {
            "correction_value": float(correction_value),
            "latency_cost_per_ms": float(latency_cost_per_ms),
            "byte_cost_per_kb": float(byte_cost_per_kb),
            "threshold": float(threshold),
        },
    }
    LearnedUtilityRouter(artifact)
    return artifact
