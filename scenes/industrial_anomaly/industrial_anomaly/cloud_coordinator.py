"""Cloud-side ExtraTrees coordinator for paired industrial score summaries.

The coordinator deliberately consumes compact RGB/infrared score summaries,
not images or heatmaps.  This keeps the online cloud path symmetric with the
traffic scene: a small scene-specific tree ensemble owns the baseline decision
and an optional full LLM may review only selected uncertain results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


SCHEMA_VERSION = "industrial-cloud-extratrees/v1"
FEATURE_NAMES = (
    "rgb_score",
    "rgb_review_low",
    "rgb_review_high",
    "rgb_relative_position",
    "rgb_state_code",
    "infrared_score",
    "infrared_review_low",
    "infrared_review_high",
    "infrared_relative_position",
    "infrared_state_code",
    "relative_position_min",
    "relative_position_max",
    "relative_position_mean",
    "relative_position_gap",
    "state_agreement",
)
STATE_CODES = {"normal": -1.0, "review": 0.0, "anomaly": 1.0}


def _finite(value: Any, name: str) -> float:
    import math

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("industrial cloud feature {} must be finite".format(name))
    return number


def feature_vector(
    records: Mapping[str, Mapping[str, Any]],
) -> Tuple[Sequence[float], Dict[str, Any]]:
    """Encode one complete RGB/infrared group into the frozen feature order."""

    if set(records) != {"rgb", "infrared"}:
        raise ValueError("industrial ExtraTrees requires one RGB and one infrared record")
    per_modality: Dict[str, Dict[str, Any]] = {}
    values = []
    for modality in ("rgb", "infrared"):
        record = records[modality]
        score = _finite(record["score"], modality + ".score")
        low = _finite(record["review_low"], modality + ".review_low")
        high = _finite(record["review_high"], modality + ".review_high")
        if low < 0.0 or high <= low:
            raise ValueError("industrial ExtraTrees received an invalid review band")
        state = str(record["state"])
        if state not in STATE_CODES:
            raise ValueError("industrial ExtraTrees received an invalid local state")
        relative = (score - low) / (high - low)
        per_modality[modality] = {
            "score": score,
            "review_low": low,
            "review_high": high,
            "relative_position": relative,
            "state": state,
        }
        values.extend((score, low, high, relative, STATE_CODES[state]))

    rgb_relative = per_modality["rgb"]["relative_position"]
    infrared_relative = per_modality["infrared"]["relative_position"]
    values.extend(
        (
            min(rgb_relative, infrared_relative),
            max(rgb_relative, infrared_relative),
            (rgb_relative + infrared_relative) / 2.0,
            abs(rgb_relative - infrared_relative),
            float(
                per_modality["rgb"]["state"]
                == per_modality["infrared"]["state"]
            ),
        )
    )
    if len(values) != len(FEATURE_NAMES):
        raise RuntimeError("industrial ExtraTrees feature width changed")
    return values, per_modality


@dataclass(frozen=True)
class IndustrialCloudPrediction:
    decision: str
    confidence: float
    probabilities: Dict[str, float]
    feature_context: Dict[str, Any]


class IndustrialCloudCoordinator:
    """Validated wrapper around the serialized sklearn ExtraTrees payload."""

    def __init__(self, payload: Mapping[str, Any], path: Path) -> None:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported industrial cloud coordinator schema")
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("industrial cloud coordinator feature schema mismatch")
        model = payload.get("model")
        if model is None or not hasattr(model, "predict_proba"):
            raise ValueError("industrial cloud coordinator model is invalid")
        raw_classes = payload.get("decision_classes")
        if raw_classes != {0: "normal", 1: "anomaly"}:
            raise ValueError("industrial cloud coordinator decision classes are invalid")
        products = tuple(str(value) for value in payload.get("supported_products", ()))
        if not products:
            raise ValueError("industrial cloud coordinator has no supported products")
        self.payload = dict(payload)
        self.model = model
        self.path = Path(path).resolve()
        self.supported_products = products

    @classmethod
    def load(cls, path: Path) -> "IndustrialCloudCoordinator":
        import joblib

        resolved = Path(path).resolve()
        payload = joblib.load(resolved)
        if not isinstance(payload, dict):
            raise ValueError("industrial cloud coordinator artifact must be an object")
        if hasattr(payload.get("model"), "n_jobs"):
            payload["model"].n_jobs = 1
        return cls(payload, resolved)

    def describe(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "path": str(self.path),
            "model_id": str(self.payload.get("model_id", "")),
            "supported_products": list(self.supported_products),
            "feature_count": len(FEATURE_NAMES),
            "training": dict(self.payload.get("training", {})),
        }

    def predict(
        self,
        product: str,
        records: Mapping[str, Mapping[str, Any]],
    ) -> IndustrialCloudPrediction:
        if str(product) not in self.supported_products:
            raise ValueError(
                "industrial cloud coordinator does not support product {}".format(
                    product
                )
            )
        vector, context = feature_vector(records)
        predicted = int(self.model.predict([vector])[0])
        raw_probabilities = self.model.predict_proba([vector])[0]
        class_columns = {
            int(value): index for index, value in enumerate(self.model.classes_)
        }
        probabilities = {
            decision: float(raw_probabilities[class_columns[class_id]])
            for class_id, decision in ((0, "normal"), (1, "anomaly"))
        }
        decision = self.payload["decision_classes"][predicted]
        return IndustrialCloudPrediction(
            decision=decision,
            confidence=probabilities[decision],
            probabilities=probabilities,
            feature_context=context,
        )
