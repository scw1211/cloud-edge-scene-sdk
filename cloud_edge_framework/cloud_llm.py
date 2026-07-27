"""用途：调用可配置云端全量大模型，对专用云模型的决策进行结构化复核。"""

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Dict

from cloud_edge_framework.contracts import DecisionEnvelope, SemanticEvent


RISK_PRIORITY = {"low": 0, "medium": 1, "high": 2, "severe": 3}


@dataclass(frozen=True)
class CloudLLMReview:
    verdict: str
    recommended_decision: str
    confidence: float
    reason: str
    provider: str
    model: str
    latency_ms: float
    prompt_tokens: Any
    output_tokens: Any

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CloudLLMReviewer:
    """Produces bounded review metadata; scene plugins retain action authority."""

    def __init__(self, provider: Any, min_risk_level: str = "high") -> None:
        if min_risk_level not in RISK_PRIORITY:
            raise ValueError("cloud LLM min_risk_level is invalid")
        self.provider = provider
        self.min_risk_level = min_risk_level

    def describe(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "min_risk_level": self.min_risk_level,
            "runtime": self.provider.describe(),
        }

    def should_review(self, event: SemanticEvent) -> bool:
        if bool(event.metadata.get("cloud_llm_review_requested", False)):
            return True
        return (
            RISK_PRIORITY[event.risk.level]
            >= RISK_PRIORITY[self.min_risk_level]
        )

    @staticmethod
    def _prompt(event: SemanticEvent, baseline: DecisionEnvelope) -> str:
        payload = {
            "scene": event.scene,
            "task": event.task,
            "scope": {
                "entity_id": event.scope.entity_id,
                "subsystem": event.scope.subsystem,
                "state_variable": event.scope.state_variable,
                "region_id": event.scope.region_id,
                "shared_resources": event.scope.shared_resources,
            },
            "prediction": {
                "label": event.prediction.label,
                "confidence": event.prediction.confidence,
                "values": event.prediction.values,
            },
            "risk": {"level": event.risk.level, "score": event.risk.score},
            "uncertainty": {
                "confidence": event.uncertainty.confidence,
                "prediction_set": event.uncertainty.prediction_set,
                "calibrated": event.uncertainty.calibrated,
            },
            "authorized_action_types": sorted(
                {action.action_type for action in event.candidate_actions}
            ),
            "baseline": {
                "decision": baseline.decision,
                "actions": [action.to_dict() for action in baseline.actions],
                "reason": baseline.reason,
            },
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _parse(text: str) -> Dict[str, Any]:
        try:
            value = json.loads(str(text).strip())
        except json.JSONDecodeError as exc:
            raise ValueError("cloud LLM must return one JSON object") from exc
        if not isinstance(value, dict):
            raise ValueError("cloud LLM review must be an object")
        allowed = {"verdict", "recommended_decision", "confidence", "reason"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError("cloud LLM review has unknown fields: {}".format(unknown))
        verdict = str(value.get("verdict", "")).strip()
        if verdict not in {"accept", "challenge"}:
            raise ValueError("cloud LLM verdict must be accept or challenge")
        recommended = str(value.get("recommended_decision", "")).strip()
        if not recommended:
            raise ValueError("cloud LLM recommended_decision must not be empty")
        confidence = value.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("cloud LLM confidence must be numeric")
        confidence = float(confidence)
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            raise ValueError("cloud LLM confidence must be within 0 and 1")
        reason = str(value.get("reason", "")).strip()
        if not reason or len(reason) > 500:
            raise ValueError("cloud LLM reason must contain 1 to 500 characters")
        return {
            "verdict": verdict,
            "recommended_decision": recommended,
            "confidence": confidence,
            "reason": reason,
        }

    def review(
        self, event: SemanticEvent, baseline: DecisionEnvelope
    ) -> CloudLLMReview:
        system_prompt = (
            "You are a cloud safety reviewer. Review the baseline decision using only "
            "the supplied structured event. Return exactly one JSON object with keys "
            "verdict, recommended_decision, confidence, reason. verdict must be accept "
            "or challenge. Do not return markdown or additional text."
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "verdict",
                "recommended_decision",
                "confidence",
                "reason",
            ],
            "properties": {
                "verdict": {"type": "string", "enum": ["accept", "challenge"]},
                "recommended_decision": {"type": "string", "minLength": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string", "minLength": 1, "maxLength": 500},
            },
        }
        generator = getattr(self.provider, "generate_structured", None)
        if callable(generator):
            generated = generator(
                self._prompt(event, baseline),
                schema=schema,
                system_prompt=system_prompt,
            )
        else:
            generated = self.provider.generate(
                self._prompt(event, baseline), system_prompt=system_prompt
            )
        parsed = self._parse(generated.text)
        return CloudLLMReview(
            verdict=parsed["verdict"],
            recommended_decision=parsed["recommended_decision"],
            confidence=parsed["confidence"],
            reason=parsed["reason"],
            provider=str(generated.provider),
            model=str(generated.model),
            latency_ms=float(generated.latency_ms),
            prompt_tokens=generated.prompt_tokens,
            output_tokens=generated.output_tokens,
        )
