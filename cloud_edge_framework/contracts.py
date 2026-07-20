"""用途：定义场景插件与公共云边运行时之间的语义事件和决策协议。"""

import hashlib
import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


SCHEMA_VERSION = "1.0"
RISK_LEVELS = ("low", "medium", "high", "severe")
EVIDENCE_LEVELS = ("summary", "feature", "raw")
ROUTES = ("edge_only", "cloud_sync", "cloud_async", "local_autonomy")
DECISION_STATUSES = ("final", "provisional", "queued")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """Raised when an event or decision violates the shared contract."""


def _object(value: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("{} must be an object".format(field_name))
    return dict(value)


def _text(value: Any, field_name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ContractError("{} must be a string".format(field_name))
    result = value.strip()
    if not result and not allow_empty:
        raise ContractError("{} must not be empty".format(field_name))
    return result


def _number(value: Any, field_name: str, low: Optional[float] = None, high: Optional[float] = None) -> float:
    if isinstance(value, bool):
        raise ContractError("{} must be numeric".format(field_name))
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError("{} must be numeric".format(field_name)) from exc
    if not math.isfinite(result):
        raise ContractError("{} must be finite".format(field_name))
    if low is not None and result < low:
        raise ContractError("{} must be >= {}".format(field_name, low))
    if high is not None and result > high:
        raise ContractError("{} must be <= {}".format(field_name, high))
    return result


def _integer(value: Any, field_name: str, low: Optional[int] = None) -> int:
    if isinstance(value, bool):
        raise ContractError("{} must be an integer".format(field_name))
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError("{} must be an integer".format(field_name)) from exc
    if low is not None and result < low:
        raise ContractError("{} must be >= {}".format(field_name, low))
    return result


def _text_list(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ContractError("{} must be a list".format(field_name))
    result: List[str] = []
    for index, item in enumerate(value):
        text = _text(item, "{}[{}]".format(field_name, index))
        if text not in result:
            result.append(text)
    return result


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def stable_id(prefix: str, *parts: Any) -> str:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return "{}_{}".format(prefix, hashlib.sha256(material).hexdigest()[:20])


def random_id(prefix: str) -> str:
    return "{}_{}".format(prefix, uuid.uuid4().hex)


@dataclass(frozen=True)
class EventScope:
    entity_id: str
    subsystem: str
    state_variable: str
    region_id: str
    shared_resources: List[str] = field(default_factory=list)
    correlation_keys: List[str] = field(default_factory=list)
    window_start_ms: int = 0
    window_end_ms: int = 0

    def __post_init__(self) -> None:
        _text(self.entity_id, "scope.entity_id")
        _text(self.subsystem, "scope.subsystem")
        _text(self.state_variable, "scope.state_variable")
        _text(self.region_id, "scope.region_id")
        _text_list(self.shared_resources, "scope.shared_resources")
        _text_list(self.correlation_keys, "scope.correlation_keys")
        start = _integer(self.window_start_ms, "scope.window_start_ms", 0)
        end = _integer(self.window_end_ms, "scope.window_end_ms", 0)
        if end < start:
            raise ContractError("scope.window_end_ms must be >= scope.window_start_ms")

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "EventScope":
        data = _object(value, "scope")
        start = _integer(data.get("window_start_ms", 0), "scope.window_start_ms", 0)
        end = _integer(data.get("window_end_ms", start), "scope.window_end_ms", 0)
        if end < start:
            raise ContractError("scope.window_end_ms must be >= scope.window_start_ms")
        return cls(
            entity_id=_text(data.get("entity_id"), "scope.entity_id"),
            subsystem=_text(data.get("subsystem"), "scope.subsystem"),
            state_variable=_text(data.get("state_variable"), "scope.state_variable"),
            region_id=_text(data.get("region_id", "global"), "scope.region_id"),
            shared_resources=_text_list(data.get("shared_resources", []), "scope.shared_resources"),
            correlation_keys=_text_list(data.get("correlation_keys", []), "scope.correlation_keys"),
            window_start_ms=start,
            window_end_ms=end,
        )


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    values: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.label, "prediction.label")
        _number(self.confidence, "prediction.confidence", 0.0, 1.0)
        if not isinstance(self.probabilities, dict):
            raise ContractError("prediction.probabilities must be an object")
        total = 0.0
        for name, probability in self.probabilities.items():
            _text(str(name), "prediction probability name")
            total += _number(
                probability, "prediction.probabilities.{}".format(name), 0.0, 1.0
            )
        if self.probabilities and total > 1.001:
            raise ContractError("prediction probabilities must sum to at most 1")
        if not isinstance(self.values, dict):
            raise ContractError("prediction.values must be an object")

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Prediction":
        data = _object(value, "prediction")
        probabilities_raw = _object(data.get("probabilities", {}), "prediction.probabilities")
        probabilities = {
            _text(str(name), "prediction probability name"): _number(
                probability, "prediction.probabilities.{}".format(name), 0.0, 1.0
            )
            for name, probability in probabilities_raw.items()
        }
        if probabilities and sum(probabilities.values()) > 1.001:
            raise ContractError("prediction probabilities must sum to at most 1")
        return cls(
            label=_text(data.get("label"), "prediction.label"),
            confidence=_number(data.get("confidence"), "prediction.confidence", 0.0, 1.0),
            probabilities=probabilities,
            values=_object(data.get("values", {}), "prediction.values"),
        )


@dataclass(frozen=True)
class Risk:
    level: str
    score: float

    def __post_init__(self) -> None:
        if self.level not in RISK_LEVELS:
            raise ContractError("risk.level must be one of {}".format(", ".join(RISK_LEVELS)))
        _number(self.score, "risk.score", 0.0, 1.0)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Risk":
        data = _object(value, "risk")
        level = _text(data.get("level"), "risk.level")
        if level not in RISK_LEVELS:
            raise ContractError("risk.level must be one of {}".format(", ".join(RISK_LEVELS)))
        return cls(level=level, score=_number(data.get("score"), "risk.score", 0.0, 1.0))


@dataclass(frozen=True)
class Uncertainty:
    confidence: float
    calibrated: bool = False
    prediction_set: List[str] = field(default_factory=list)
    method: str = "raw_model_confidence"

    def __post_init__(self) -> None:
        _number(self.confidence, "uncertainty.confidence", 0.0, 1.0)
        _text(self.method, "uncertainty.method")
        prediction_set = _text_list(self.prediction_set, "uncertainty.prediction_set")
        unknown = [name for name in prediction_set if name not in RISK_LEVELS]
        if unknown:
            raise ContractError("uncertainty.prediction_set has unknown risk levels: {}".format(unknown))

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Uncertainty":
        data = _object(value, "uncertainty")
        prediction_set = _text_list(data.get("prediction_set", []), "uncertainty.prediction_set")
        unknown = [name for name in prediction_set if name not in RISK_LEVELS]
        if unknown:
            raise ContractError("uncertainty.prediction_set has unknown risk levels: {}".format(unknown))
        return cls(
            confidence=_number(data.get("confidence"), "uncertainty.confidence", 0.0, 1.0),
            calibrated=bool(data.get("calibrated", False)),
            prediction_set=prediction_set,
            method=_text(data.get("method", "raw_model_confidence"), "uncertainty.method"),
        )


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    level: str
    modality: str
    encoding: str
    uri: Optional[str] = None
    inline: Any = None
    shape: List[int] = field(default_factory=list)
    size_bytes: int = 0
    sha256: Optional[str] = None
    content_type: str = "application/octet-stream"
    codec: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.evidence_id, "evidence.evidence_id")
        if self.level not in EVIDENCE_LEVELS:
            raise ContractError("evidence.level must be one of {}".format(", ".join(EVIDENCE_LEVELS)))
        _text(self.modality, "evidence.modality")
        _text(self.encoding, "evidence.encoding")
        if self.uri is None and self.inline is None:
            raise ContractError("evidence must provide uri or inline content")
        if self.uri is not None:
            _text(self.uri, "evidence.uri")
        if not isinstance(self.shape, list):
            raise ContractError("evidence.shape must be a list")
        for dimension in self.shape:
            _integer(dimension, "evidence.shape", 0)
        _integer(self.size_bytes, "evidence.size_bytes", 0)
        if self.sha256 is not None and not _SHA256_RE.match(self.sha256.lower()):
            raise ContractError("evidence.sha256 must contain 64 hexadecimal characters")
        _text(self.content_type, "evidence.content_type")
        if not isinstance(self.codec, dict):
            raise ContractError("evidence.codec must be an object")

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Evidence":
        data = _object(value, "evidence item")
        level = _text(data.get("level"), "evidence.level")
        if level not in EVIDENCE_LEVELS:
            raise ContractError("evidence.level must be one of {}".format(", ".join(EVIDENCE_LEVELS)))
        uri = data.get("uri")
        if uri is not None:
            uri = _text(uri, "evidence.uri")
        inline = data.get("inline")
        if uri is None and inline is None:
            raise ContractError("evidence must provide uri or inline content")
        shape_raw = data.get("shape", [])
        if not isinstance(shape_raw, list):
            raise ContractError("evidence.shape must be a list")
        shape = [_integer(item, "evidence.shape", 0) for item in shape_raw]
        size_default = _json_size(inline) if inline is not None else 0
        sha256 = data.get("sha256")
        if sha256 is not None:
            sha256 = _text(sha256, "evidence.sha256").lower()
            if not _SHA256_RE.match(sha256):
                raise ContractError("evidence.sha256 must contain 64 lowercase hexadecimal characters")
        return cls(
            evidence_id=_text(data.get("evidence_id"), "evidence.evidence_id"),
            level=level,
            modality=_text(data.get("modality"), "evidence.modality"),
            encoding=_text(data.get("encoding"), "evidence.encoding"),
            uri=uri,
            inline=inline,
            shape=shape,
            size_bytes=_integer(data.get("size_bytes", size_default), "evidence.size_bytes", 0),
            sha256=sha256,
            content_type=_text(data.get("content_type", "application/octet-stream"), "evidence.content_type"),
            codec=_object(data.get("codec", {}), "evidence.codec"),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.uri is None:
            result.pop("uri")
        if self.inline is None:
            result.pop("inline")
        if self.sha256 is None:
            result.pop("sha256")
        return result


@dataclass(frozen=True)
class Action:
    action_type: str
    target_ids: List[str]
    resource_ids: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    priority: int = 50

    def __post_init__(self) -> None:
        _text(self.action_type, "action.action_type")
        _text_list(self.target_ids, "action.target_ids")
        _text_list(self.resource_ids, "action.resource_ids")
        if not isinstance(self.parameters, dict):
            raise ContractError("action.parameters must be an object")
        _text(self.reason, "action.reason", allow_empty=True)
        priority = _integer(self.priority, "action.priority", 0)
        if priority > 100:
            raise ContractError("action.priority must be <= 100")

    @classmethod
    def from_dict(cls, value: Dict[str, Any], field_name: str = "action") -> "Action":
        data = _object(value, field_name)
        priority = _integer(data.get("priority", 50), "{}.priority".format(field_name), 0)
        if priority > 100:
            raise ContractError("{}.priority must be <= 100".format(field_name))
        return cls(
            action_type=_text(data.get("action_type"), "{}.action_type".format(field_name)),
            target_ids=_text_list(data.get("target_ids", []), "{}.target_ids".format(field_name)),
            resource_ids=_text_list(data.get("resource_ids", []), "{}.resource_ids".format(field_name)),
            parameters=_object(data.get("parameters", {}), "{}.parameters".format(field_name)),
            reason=_text(data.get("reason", ""), "{}.reason".format(field_name), allow_empty=True),
            priority=priority,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Timing:
    deadline_ms: float = 200.0
    preprocessing_ms: float = 0.0
    edge_inference_ms: float = 0.0

    def __post_init__(self) -> None:
        _number(self.deadline_ms, "timing.deadline_ms", 0.001)
        _number(self.preprocessing_ms, "timing.preprocessing_ms", 0.0)
        _number(self.edge_inference_ms, "timing.edge_inference_ms", 0.0)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Timing":
        data = _object(value, "timing")
        return cls(
            deadline_ms=_number(data.get("deadline_ms", 200.0), "timing.deadline_ms", 0.001),
            preprocessing_ms=_number(data.get("preprocessing_ms", 0.0), "timing.preprocessing_ms", 0.0),
            edge_inference_ms=_number(data.get("edge_inference_ms", 0.0), "timing.edge_inference_ms", 0.0),
        )


@dataclass(frozen=True)
class SemanticEvent:
    event_id: str
    scene: str
    task: str
    edge_id: str
    occurred_at_ms: int
    scope: EventScope
    prediction: Prediction
    risk: Risk
    uncertainty: Uncertainty
    timing: Timing
    evidence: List[Evidence]
    candidate_actions: List[Action]
    model: Dict[str, Any] = field(default_factory=dict)
    scene_payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError("unsupported schema_version: {}".format(self.schema_version))
        _text(self.event_id, "event_id")
        _text(self.scene, "scene")
        _text(self.task, "task")
        _text(self.edge_id, "edge_id")
        _integer(self.occurred_at_ms, "occurred_at_ms", 0)
        if not self.evidence:
            raise ContractError("event.evidence must contain at least one item")
        if not all(isinstance(item, Evidence) for item in self.evidence):
            raise ContractError("event.evidence contains an invalid item")
        if not all(isinstance(item, Action) for item in self.candidate_actions):
            raise ContractError("event.candidate_actions contains an invalid item")
        for field_name, value in (
            ("model", self.model),
            ("scene_payload", self.scene_payload),
            ("metadata", self.metadata),
        ):
            if not isinstance(value, dict):
                raise ContractError("{} must be an object".format(field_name))

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SemanticEvent":
        data = _object(value, "event")
        version = _text(data.get("schema_version", SCHEMA_VERSION), "schema_version")
        if version != SCHEMA_VERSION:
            raise ContractError("unsupported schema_version: {}".format(version))
        evidence_raw = data.get("evidence", [])
        if not isinstance(evidence_raw, list) or not evidence_raw:
            raise ContractError("event.evidence must contain at least one item")
        actions_raw = data.get("candidate_actions", [])
        if not isinstance(actions_raw, list):
            raise ContractError("event.candidate_actions must be a list")
        return cls(
            event_id=_text(data.get("event_id"), "event_id"),
            scene=_text(data.get("scene"), "scene"),
            task=_text(data.get("task"), "task"),
            edge_id=_text(data.get("edge_id"), "edge_id"),
            occurred_at_ms=_integer(data.get("occurred_at_ms"), "occurred_at_ms", 0),
            scope=EventScope.from_dict(data.get("scope")),
            prediction=Prediction.from_dict(data.get("prediction")),
            risk=Risk.from_dict(data.get("risk")),
            uncertainty=Uncertainty.from_dict(data.get("uncertainty")),
            timing=Timing.from_dict(data.get("timing", {})),
            evidence=[Evidence.from_dict(item) for item in evidence_raw],
            candidate_actions=[
                Action.from_dict(item, "candidate_actions[{}]".format(index))
                for index, item in enumerate(actions_raw)
            ],
            model=_object(data.get("model", {}), "model"),
            scene_payload=_object(data.get("scene_payload", {}), "scene_payload"),
            metadata=_object(data.get("metadata", {}), "metadata"),
            schema_version=version,
        )

    def to_dict(
        self,
        evidence_levels: Optional[Sequence[str]] = None,
        include_scene_payload: bool = True,
    ) -> Dict[str, Any]:
        allowed_levels = set(evidence_levels) if evidence_levels is not None else None
        result = {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "scene": self.scene,
            "task": self.task,
            "edge_id": self.edge_id,
            "occurred_at_ms": self.occurred_at_ms,
            "scope": asdict(self.scope),
            "prediction": asdict(self.prediction),
            "risk": asdict(self.risk),
            "uncertainty": asdict(self.uncertainty),
            "timing": asdict(self.timing),
            "evidence": [
                item.to_dict()
                for item in self.evidence
                if allowed_levels is None or item.level in allowed_levels
            ],
            "candidate_actions": [item.to_dict() for item in self.candidate_actions],
            "model": dict(self.model),
            "metadata": dict(self.metadata),
        }
        if include_scene_payload:
            result["scene_payload"] = dict(self.scene_payload)
        return result


@dataclass(frozen=True)
class DecisionEnvelope:
    decision_id: str
    event_ids: List[str]
    scene: str
    decision: str
    risk_level: str
    confidence: float
    route: str
    status: str
    actions: List[Action]
    reason: str
    policy_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.decision_id, "decision_id")
        if not self.event_ids:
            raise ContractError("decision.event_ids must not be empty")
        _text(self.scene, "decision.scene")
        _text(self.decision, "decision.decision")
        if self.risk_level not in RISK_LEVELS:
            raise ContractError("decision.risk_level is invalid")
        _number(self.confidence, "decision.confidence", 0.0, 1.0)
        if self.route not in ROUTES:
            raise ContractError("decision.route is invalid")
        if self.status not in DECISION_STATUSES:
            raise ContractError("decision.status is invalid")

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "DecisionEnvelope":
        data = _object(value, "decision")
        version = _text(data.get("schema_version", SCHEMA_VERSION), "decision.schema_version")
        if version != SCHEMA_VERSION:
            raise ContractError("unsupported decision schema_version: {}".format(version))
        event_ids = _text_list(data.get("event_ids"), "decision.event_ids")
        actions_raw = data.get("actions", [])
        if not isinstance(actions_raw, list):
            raise ContractError("decision.actions must be a list")
        return cls(
            decision_id=_text(data.get("decision_id"), "decision.decision_id"),
            event_ids=event_ids,
            scene=_text(data.get("scene"), "decision.scene"),
            decision=_text(data.get("decision"), "decision.decision"),
            risk_level=_text(data.get("risk_level"), "decision.risk_level"),
            confidence=_number(data.get("confidence"), "decision.confidence", 0.0, 1.0),
            route=_text(data.get("route"), "decision.route"),
            status=_text(data.get("status"), "decision.status"),
            actions=[
                Action.from_dict(item, "decision.actions[{}]".format(index))
                for index, item in enumerate(actions_raw)
            ],
            reason=_text(data.get("reason", ""), "decision.reason", allow_empty=True),
            policy_version=_text(data.get("policy_version"), "decision.policy_version"),
            metadata=_object(data.get("metadata", {}), "decision.metadata"),
            schema_version=version,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "event_ids": list(self.event_ids),
            "scene": self.scene,
            "decision": self.decision,
            "risk_level": self.risk_level,
            "confidence": round(float(self.confidence), 6),
            "route": self.route,
            "status": self.status,
            "actions": [action.to_dict() for action in self.actions],
            "reason": self.reason,
            "policy_version": self.policy_version,
            "metadata": dict(self.metadata),
        }


def build_decision(
    event: SemanticEvent,
    decision: str,
    actions: Sequence[Action],
    confidence: float,
    reason: str,
    source: str,
    policy_version: str,
    route: str = "edge_only",
    status: str = "final",
) -> DecisionEnvelope:
    return DecisionEnvelope(
        decision_id=stable_id("decision", event.event_id, source, decision),
        event_ids=[event.event_id],
        scene=event.scene,
        decision=decision,
        risk_level=event.risk.level,
        confidence=max(0.0, min(1.0, float(confidence))),
        route=route,
        status=status,
        actions=list(actions),
        reason=reason,
        policy_version=policy_version,
        metadata={"source": source},
    )
