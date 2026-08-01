"""用途：定义可独立装卸的场景组件生命周期、事件适配和决策接口。"""

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, Dict, Optional, Sequence, Tuple

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from cloud_edge_framework.contracts import (
    Action,
    ContractError,
    DecisionEnvelope,
    SemanticEvent,
    build_decision,
)
from cloud_edge_framework.event_envelope import SceneEventEnvelope


RISK_PRIORITY = {"low": 0, "medium": 1, "high": 2, "severe": 3}


class ScenePlugin(ABC):
    scene = ""
    aliases: Sequence[str] = ()
    event_types: Sequence[str] = ()
    policy_version = "framework-1.0.0"

    def warmup(self) -> None:
        """Load optional model artifacts before the service starts accepting requests."""
        return

    def health(self) -> Dict[str, Any]:
        """Return plugin-owned runtime state without exposing model internals."""
        return {"status": "ok"}

    def close(self) -> None:
        """Release plugin-owned resources after a runtime snapshot is retired."""
        return

    @abstractmethod
    def payload_schema(self) -> Dict[str, Any]:
        """Return the plugin-owned JSON Schema for the native model output."""

    def contract_descriptor(self) -> Dict[str, Any]:
        """Validate and describe the plugin's external event contract."""
        cached = getattr(self, "_contract_descriptor_cache", None)
        if cached is not None:
            return dict(cached)
        scene = str(self.scene).strip()
        if not scene:
            raise ContractError("plugin scene must be non-empty")

        event_types = tuple(str(value).strip() for value in self.event_types)
        if not event_types or any(not value for value in event_types):
            raise ContractError(f"plugin {scene!r} must declare event_types")
        if len(set(event_types)) != len(event_types):
            raise ContractError(f"plugin {scene!r} declares duplicate event_types")

        schema = self.payload_schema()
        if not isinstance(schema, dict):
            raise ContractError(f"plugin {scene!r} payload_schema must return an object")
        schema_id = schema.get("$id")
        if not isinstance(schema_id, str) or not schema_id.strip():
            raise ContractError(f"plugin {scene!r} payload schema must declare $id")
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ContractError(
                f"plugin {scene!r} payload schema is invalid: {exc.message}"
            ) from exc

        descriptor = {
            "scene": scene,
            "event_types": list(event_types),
            "data_schema": schema_id,
        }
        self._contract_descriptor_cache = descriptor
        self._payload_validator_cache = Draft202012Validator(schema)
        return dict(descriptor)

    def validate_envelope(self, envelope: SceneEventEnvelope) -> SceneEventEnvelope:
        """Validate routing metadata and the plugin-owned payload."""
        descriptor = self.contract_descriptor()
        accepted_scenes = {descriptor["scene"], *self.aliases}
        if envelope.scene not in accepted_scenes:
            raise ContractError(
                f"event scene {envelope.scene!r} is not handled by plugin {self.scene!r}"
            )
        if envelope.event_type not in descriptor["event_types"]:
            raise ContractError(
                f"event type {envelope.event_type!r} is not handled by plugin {self.scene!r}"
            )
        if envelope.dataschema != descriptor["data_schema"]:
            raise ContractError(
                f"event dataschema {envelope.dataschema!r} does not match "
                f"{descriptor['data_schema']!r}"
            )

        validator = getattr(self, "_payload_validator_cache", None)
        if validator is None:
            self.contract_descriptor()
            validator = self._payload_validator_cache
        errors = sorted(
            validator.iter_errors(envelope.payload_for_validation()),
            key=lambda error: list(error.path),
        )
        if errors:
            error = errors[0]
            location = ".".join(str(part) for part in error.absolute_path) or "data"
            raise ContractError(f"scene payload {location}: {error.message}")
        return envelope

    @abstractmethod
    def normalize(self, envelope: SceneEventEnvelope) -> SemanticEvent:
        """Map plugin-owned data into the internal scheduling contract."""

    @abstractmethod
    def edge_decide(self, event: SemanticEvent) -> DecisionEnvelope:
        """Return the immediate scene decision available at the edge."""

    @abstractmethod
    def cloud_decide(self, event: SemanticEvent) -> DecisionEnvelope:
        """Return a cloud expert decision for one normalized event."""

    def apply_cloud_llm_review(
        self,
        event: SemanticEvent,
        baseline: DecisionEnvelope,
        review: Dict[str, Any],
    ) -> DecisionEnvelope:
        """Attach a review safely; scene plugins may validate and adopt a recommendation."""
        del event
        metadata = dict(baseline.metadata)
        metadata["cloud_llm_review"] = dict(review)
        metadata["cloud_llm_challenged"] = review.get("verdict") == "challenge"
        return replace(baseline, metadata=metadata)

    def prepare_cloud_event(
        self,
        event: SemanticEvent,
        evidence_level: str,
    ) -> SemanticEvent:
        """Remove edge-only state and build the scene-owned cloud data-plane payload."""
        return event

    def monitoring_signals(self, event: SemanticEvent) -> Dict[str, float]:
        """Expose scene-specific normalized signals in addition to common confidence fields."""
        value = event.metadata.get("monitoring_signals", {})
        if not isinstance(value, dict):
            raise ContractError("event metadata monitoring_signals must be an object")
        return dict(value)

    def routing_advice(
        self,
        event: SemanticEvent,
        local_decision: DecisionEnvelope,
    ) -> Dict[str, Any]:
        """Return optional scene-owned hints without making a Student a framework requirement."""
        del event, local_decision
        return {}

    def evidence_advice(
        self,
        event: SemanticEvent,
        local_decision: DecisionEnvelope,
        conflict_suspected: bool,
    ) -> Dict[str, Any]:
        """Return scene-owned evidence intent without overloading common risk fields."""
        del event, local_decision, conflict_suspected
        return {}

    def cloud_submission_metadata(
        self,
        event: SemanticEvent,
        local_decision: DecisionEnvelope,
    ) -> Dict[str, Any]:
        """Return local-decision semantics that must accompany cloud delivery."""
        del event, local_decision
        return {}

    def aggregation_spec(
        self, event: SemanticEvent
    ) -> Optional[Dict[str, Any]]:
        """Return scene-owned join metadata, or None when an event is independent."""
        value = event.metadata.get("aggregation")
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ContractError("event metadata aggregation must be an object")
        return dict(value)

    def fuse_cloud_context(
        self,
        events: Sequence[SemanticEvent],
    ) -> Sequence[SemanticEvent]:
        """Attach scene-owned cross-edge context before cloud inference."""
        return list(events)

    def action_conflict(self, left: Action, right: Action) -> Tuple[bool, str]:
        """Return whether two actions on a shared resource are incompatible."""
        if left.action_type != right.action_type:
            return True, "incompatible_action_types"
        if left.parameters != right.parameters:
            return True, "inconsistent_action_parameters"
        return False, ""

    def resolve_action_conflict(
        self,
        left: Action,
        right: Action,
        left_event: SemanticEvent,
        right_event: SemanticEvent,
    ) -> Tuple[Action, Action, str]:
        """Resolve one action pair; default keeps the higher-risk proposal on both sides."""
        left_weight = RISK_PRIORITY[left_event.risk.level] * 10 + left.priority
        right_weight = RISK_PRIORITY[right_event.risk.level] * 10 + right.priority
        selected = left if left_weight >= right_weight else right
        reason = "cloud selected the higher-risk, higher-priority action"
        coordinated = replace(selected, reason=reason)
        return coordinated, coordinated, reason

    def decision_from_candidates(
        self,
        event: SemanticEvent,
        source: str,
        confidence: float,
    ) -> DecisionEnvelope:
        selected = []
        for action in event.candidate_actions:
            min_level = str(action.parameters.get("min_risk_level", "medium"))
            if min_level not in RISK_PRIORITY:
                raise ValueError("Unknown min_risk_level in action: {}".format(min_level))
            if RISK_PRIORITY[event.risk.level] >= RISK_PRIORITY[min_level]:
                parameters = dict(action.parameters)
                parameters.pop("min_risk_level", None)
                selected.append(replace(action, parameters=parameters))
        if event.risk.level == "low":
            decision = "no_action"
            selected = []
            reason = "no operational risk detected"
        elif selected:
            decision = max(selected, key=lambda action: action.priority).action_type
            reason = "{} selected {} validated scene actions".format(source, len(selected))
        else:
            decision = "monitor"
            reason = "risk detected but no authorized actuator action was supplied"
        return build_decision(
            event=event,
            decision=decision,
            actions=selected,
            confidence=confidence,
            reason=reason,
            source=source,
            policy_version=self.policy_version,
        )
