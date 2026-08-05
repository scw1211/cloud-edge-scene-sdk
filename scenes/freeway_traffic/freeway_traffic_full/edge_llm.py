"""用途：把已发布的交通 Edge-Qwen 接入插件，并保留确定性的 Student 安全路径。"""

from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional, Tuple

from cloud_edge_framework.contracts import (
    Action,
    DecisionEnvelope,
    SemanticEvent,
    build_decision,
)
from edge_llm_factory.release_runtime import ActiveEdgeLLM, load_active_edge_llm
from traffic_system.edge_qwen_action_infer import build_action_prompt


RISK_PRIORITY = {"low": 0, "medium": 1, "high": 2, "severe": 3}
EDGE_LLM_MODES = {"disabled", "shadow", "selective", "primary"}
TRAFFIC_ADAPTER_SCENE = "freeway_traffic_management"
TRAFFIC_CONTEXT_ENCODER = "freeway-bitpacked-decimal@v1"
TRAFFIC_CONTEXT_ENCODER_V2 = "freeway-routing-context-decimal@v2"
TRAFFIC_CONTEXT_ENCODERS = {
    TRAFFIC_CONTEXT_ENCODER,
    TRAFFIC_CONTEXT_ENCODER_V2,
}


def _with_metadata(
    decision: DecisionEnvelope,
    values: Dict[str, Any],
) -> DecisionEnvelope:
    metadata = dict(decision.metadata)
    metadata.update(values)
    return replace(decision, metadata=metadata)


def _action_semantics(value: Any) -> str:
    payload = value.to_dict() if isinstance(value, Action) else dict(value)
    return json.dumps(
        {
            "action_type": payload.get("action_type"),
            "target_ids": sorted(
                str(item) for item in payload.get("target_ids", [])
            ),
            "resource_ids": sorted(
                str(item) for item in payload.get("resource_ids", [])
            ),
            "parameters": payload.get("parameters", {}),
            "reason": payload.get("reason"),
            "priority": payload.get("priority"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _decision_disagrees(
    decoded: Dict[str, Any],
    student: DecisionEnvelope,
) -> bool:
    decoded_actions = decoded.get("actions", [])
    decoded_actions = decoded_actions if isinstance(decoded_actions, list) else []
    return bool(
        str(decoded.get("decision", "")) != student.decision
        or sorted(_action_semantics(value) for value in decoded_actions)
        != sorted(_action_semantics(value) for value in student.actions)
    )


class TrafficEdgeLLMController:
    def __init__(
        self,
        release_registry_path: Optional[Path],
        runtime_config_path: Optional[Path],
        mode: str = "disabled",
        min_risk_level: str = "medium",
        student_confidence_threshold: float = 0.75,
        deadline_margin_ms: float = 15.0,
        deadline_probe_interval: int = 0,
        runtime_failure_cooldown_seconds: float = 5.0,
        min_expected_gain: float = 0.05,
        gain_profile_path: Optional[Path] = None,
    ) -> None:
        self.release_registry_path = (
            Path(release_registry_path) if release_registry_path is not None else None
        )
        self.runtime_config_path = (
            Path(runtime_config_path) if runtime_config_path is not None else None
        )
        self.mode = str(mode)
        self.min_risk_level = str(min_risk_level)
        self.student_confidence_threshold = float(student_confidence_threshold)
        self.min_expected_gain = float(min_expected_gain)
        self.gain_profile_path = (
            Path(gain_profile_path) if gain_profile_path is not None else None
        )
        self.gain_profile: Optional[Dict[str, Any]] = None
        self.gain_profile_inactive_reason: Optional[str] = None
        self.deadline_margin_ms = float(deadline_margin_ms)
        self.deadline_probe_interval = int(deadline_probe_interval)
        self.runtime_failure_cooldown_seconds = float(runtime_failure_cooldown_seconds)
        if self.mode not in EDGE_LLM_MODES:
            raise ValueError("edge_llm_mode must be one of {}".format(sorted(EDGE_LLM_MODES)))
        if self.min_risk_level not in RISK_PRIORITY:
            raise ValueError("edge_llm_min_risk_level is invalid")
        if not 0.0 <= self.student_confidence_threshold <= 1.0:
            raise ValueError("edge_llm_student_confidence_threshold must be in [0, 1]")
        if not -1.0 <= self.min_expected_gain <= 1.0:
            raise ValueError("edge_llm_min_expected_gain must be in [-1, 1]")
        if self.deadline_margin_ms < 0:
            raise ValueError("edge_llm_deadline_margin_ms must not be negative")
        if self.deadline_probe_interval < 0:
            raise ValueError("edge_llm_deadline_probe_interval must not be negative")
        if self.runtime_failure_cooldown_seconds < 0:
            raise ValueError("edge_llm_runtime_failure_cooldown_seconds must not be negative")
        configured_paths = self.release_registry_path is not None and self.runtime_config_path is not None
        if self.mode != "disabled" and not configured_paths:
            raise ValueError("enabled Edge LLM requires release registry and runtime config")
        if (self.release_registry_path is None) != (self.runtime_config_path is None):
            raise ValueError("Edge LLM release registry and runtime config must be configured together")
        self.active: Optional[ActiveEdgeLLM] = None
        self.context_encoder: Optional[str] = None
        self.last_error: Optional[str] = None
        self._circuit_open_until = 0.0
        self.invocations = 0
        self.accepted = 0
        self.fallbacks = 0
        self.deadline_limited_candidates = 0
        self._selection_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    def warmup(self) -> None:
        if not self.enabled:
            return
        active = load_active_edge_llm(
            self.release_registry_path,
            self.runtime_config_path,
            expected_scene=TRAFFIC_ADAPTER_SCENE,
        )
        contract = active.model.describe()["input_contract"]
        context_encoder = str(contract.get("context_encoder", ""))
        if context_encoder not in TRAFFIC_CONTEXT_ENCODERS:
            raise ValueError("traffic Edge LLM uses an unsupported context encoder")
        if int(contract.get("max_input_tokens", 0)) > 16:
            raise ValueError("traffic Edge LLM input contract exceeds the 16-token budget")
        self.active = active
        self.context_encoder = context_encoder
        self.gain_profile = None
        self.gain_profile_inactive_reason = None
        if (
            self.gain_profile_path is not None
            and context_encoder == TRAFFIC_CONTEXT_ENCODER_V2
        ):
            profile = json.loads(
                self.gain_profile_path.read_text(encoding="utf-8")
            )
            if int(profile.get("schema_version", 0)) != 1:
                raise ValueError("traffic Edge-Qwen gain profile schema is unsupported")
            if str(profile.get("baseline", "")) != "current_state_student":
                raise ValueError("traffic Edge-Qwen gain profile baseline is invalid")
            if not isinstance(profile.get("accepted_strata"), dict):
                raise ValueError("traffic Edge-Qwen gain profile has no accepted strata")
            self.gain_profile = profile
        elif self.gain_profile_path is not None:
            self.gain_profile_inactive_reason = (
                "active_release_does_not_use_routing_context_v2"
            )
        self.last_error = None
        self._circuit_open_until = 0.0

    @staticmethod
    def _structured_metadata(value: Any) -> Dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    def _selection_signals(
        self,
        event: SemanticEvent,
        student: DecisionEnvelope,
    ) -> Tuple[str, ...]:
        """Return explicit model-routing signals; traffic severity is intentionally absent."""
        event_uncertainty = self._structured_metadata(
            event.metadata.get("model_uncertainty")
        )
        decision_uncertainty = self._structured_metadata(
            student.metadata.get("model_uncertainty")
        )
        uncertainty = {**event_uncertainty, **decision_uncertainty}

        raw_student_confidence = uncertainty.get(
            "student_confidence",
            student.metadata.get(
                "traffic_student_candidate_confidence", student.confidence
            ),
        )
        try:
            student_confidence = float(raw_student_confidence)
        except (TypeError, ValueError):
            student_confidence = float(student.confidence)
        student_available = uncertainty.get("student_available", True) is not False

        raw_prediction_set = uncertainty.get(
            "prediction_set", event.uncertainty.prediction_set
        )
        prediction_set = (
            [str(value) for value in raw_prediction_set]
            if isinstance(raw_prediction_set, list)
            else list(event.uncertainty.prediction_set)
        )
        raw_disagreement = uncertainty.get("student_rule_disagreement")
        if raw_disagreement is None:
            raw_disagreement = student.metadata.get(
                "traffic_student_rule_disagreement", False
            )
        disagreement = bool(raw_disagreement)
        defer_recommended = bool(
            student.metadata.get("traffic_defer_recommended", False)
        )

        signals = []
        if student_available and student_confidence < self.student_confidence_threshold:
            signals.append("student_low_confidence")
        if len(prediction_set) > 1:
            signals.append("prediction_set_ambiguous")
        if disagreement:
            signals.append("student_rule_disagreement")
        if defer_recommended:
            signals.append("defer_gate_recommends_escalation")
        return tuple(signals)

    def _prompt_routing_context(
        self,
        event: SemanticEvent,
        student: DecisionEnvelope,
    ) -> Dict[str, Any]:
        uncertainty = {
            **self._structured_metadata(event.metadata.get("model_uncertainty")),
            **self._structured_metadata(student.metadata.get("model_uncertainty")),
        }
        raw_confidence = uncertainty.get(
            "student_confidence",
            student.metadata.get(
                "traffic_student_candidate_confidence", student.confidence
            ),
        )
        try:
            student_confidence = float(raw_confidence)
        except (TypeError, ValueError):
            student_confidence = float(student.confidence)
        raw_prediction_set = uncertainty.get(
            "prediction_set", event.uncertainty.prediction_set
        )
        prediction_set_size = (
            len(raw_prediction_set)
            if isinstance(raw_prediction_set, list)
            else len(event.uncertainty.prediction_set)
        )
        student_decision = str(
            student.metadata.get(
                "traffic_student_candidate_decision", student.decision
            )
        )
        rule_decision = str(
            student.metadata.get(
                "traffic_rule_candidate_decision",
                event.metadata.get("reference_edge_decision", student_decision),
            )
        )
        network_available = bool(event.metadata["edge_runtime_network_available"])
        network_status = str(
            event.metadata.get(
                "edge_runtime_network_status",
                "normal" if network_available else "offline",
            )
        ).lower()
        if not network_available:
            network_status = "offline"
        elif network_status not in {"normal", "weak"}:
            network_status = "normal"
        return {
            "student_decision": student_decision,
            "rule_decision": rule_decision,
            "student_confidence": student_confidence,
            "prediction_set_size": max(1, prediction_set_size),
            "network_status": network_status,
        }

    def _expected_gain(
        self,
        event: SemanticEvent,
        student: DecisionEnvelope,
    ) -> Tuple[float, str, bool]:
        gain = self._structured_metadata(
            student.metadata.get(
                "escalation_expected_gain",
                event.metadata.get("escalation_expected_gain"),
            )
        )
        try:
            edge_qwen_gain = float(gain.get("edge_qwen", 0.0))
        except (TypeError, ValueError):
            edge_qwen_gain = 0.0
        source = str(gain.get("source", "not_estimated"))
        current_state_contract = bool(
            str(event.scene_payload.get("perception_mode", "")).lower()
            == "current_state"
            or str(event.scene_payload.get("output_type", "")).lower()
            == "current_state_risk"
        )
        if (
            source == "not_estimated"
            and self.gain_profile is not None
            and current_state_contract
        ):
            context = self._prompt_routing_context(event, student)
            confidence_bucket = min(
                3,
                max(0, int(float(context["student_confidence"]) * 4.0)),
            )
            key = "|".join(
                [
                    str(context["network_status"]),
                    str(context["student_decision"]),
                    str(context["rule_decision"]),
                    str(confidence_bucket),
                    "1" if int(context["prediction_set_size"]) > 1 else "0",
                ]
            )
            entry = self.gain_profile["accepted_strata"].get(key)
            if isinstance(entry, dict):
                edge_qwen_gain = float(entry.get("validation_gain", 0.0))
                source = "validated_current_state_gain_profile"
            else:
                edge_qwen_gain = 0.0
                source = "current_state_gain_profile_not_selected"
        qualified = (
            source != "not_estimated"
            and edge_qwen_gain >= self.min_expected_gain
        )
        return edge_qwen_gain, source, qualified

    def _decoder_event(
        self,
        event: SemanticEvent,
        student: DecisionEnvelope,
    ) -> Tuple[Dict[str, Any], bool]:
        """Build a decode-only whitelist for proactive current-state advisories.

        The v2 current-state model predicts a future action from the current
        window.  A Student-produced traffic advisory is already locally safe,
        but it is not part of the rule teacher's current-state candidate list.
        Only that non-executing advisory may be added here; control actions and
        the original SemanticEvent remain unchanged.
        """
        value = event.to_dict(include_scene_payload=False)
        current_state = bool(
            self.context_encoder == TRAFFIC_CONTEXT_ENCODER_V2
            and (
                str(event.scene_payload.get("perception_mode", "")).lower()
                == "current_state"
                or str(event.scene_payload.get("output_type", "")).lower()
                == "current_state_risk"
            )
        )
        if not current_state:
            return value, False
        raw_candidates = value.get("candidate_actions", [])
        candidates = (
            [dict(action) for action in raw_candidates if isinstance(action, dict)]
            if isinstance(raw_candidates, list)
            else []
        )
        if any(
            str(action.get("action_type", action.get("type", "")))
            == "traffic_advisory"
            for action in candidates
        ):
            return value, False
        advisories = [
            action.to_dict()
            for action in student.actions
            if action.action_type == "traffic_advisory"
        ]
        if not advisories:
            return value, False
        value["candidate_actions"] = candidates + advisories
        return value, True

    def _selection(self, event: SemanticEvent, student: DecisionEnvelope) -> Tuple[bool, str]:
        if not self.enabled:
            return False, "disabled"
        if self.active is None:
            return False, "release_not_loaded"
        if time.monotonic() < self._circuit_open_until:
            return False, "runtime_circuit_open"
        if "edge_runtime_network_available" not in event.metadata:
            return False, "outside_edge_runtime"
        if self.mode == "primary":
            return True, "primary"
        signals = self._selection_signals(event, student)
        expected_gain, gain_source, gain_qualified = self._expected_gain(
            event, student
        )
        if self.mode == "selective":
            if not signals:
                return False, "no_model_escalation_signal"
            if gain_source == "not_estimated":
                return False, "edge_qwen_expected_gain_not_estimated"
            if not gain_qualified:
                return False, "edge_qwen_expected_gain_below_threshold"
        expected_ttft = float(
            self.active.model.validation.get("metrics", {}).get("average_ttft_ms", 100.0)
        )
        remaining = (
            event.timing.deadline_ms
            - event.timing.preprocessing_ms
            - event.timing.edge_inference_ms
        )
        if remaining < expected_ttft + self.deadline_margin_ms:
            with self._selection_lock:
                self.deadline_limited_candidates += 1
                run_probe = (
                    self.deadline_probe_interval > 0
                    and self.deadline_limited_candidates % self.deadline_probe_interval == 0
                )
            if run_probe:
                return True, "deadline_profile_probe"
            return False, "deadline_budget_insufficient"
        if self.mode == "shadow":
            return True, "shadow"
        return True, "selective:{}+expected_gain={:.6f}".format(
            "+".join(signals), expected_gain
        )

    def decide(
        self,
        event: SemanticEvent,
        student: DecisionEnvelope,
        policy_version: str,
    ) -> DecisionEnvelope:
        selected, reason = self._selection(event, student)
        if not selected:
            return _with_metadata(
                student,
                {
                    "edge_llm_configured": self.enabled,
                    "edge_llm_selected": False,
                    "edge_llm_selection_reason": reason,
                    "edge_decision_path": "student",
                },
            )
        assert self.active is not None
        self.invocations += 1
        if self.context_encoder == TRAFFIC_CONTEXT_ENCODER_V2:
            prompt = build_action_prompt(
                event.scene_payload,
                "routing_context_v2",
                routing_context=self._prompt_routing_context(event, student),
            )
        else:
            prompt = build_action_prompt(event.scene_payload, "bitpacked_decimal")
        network_available = bool(event.metadata["edge_runtime_network_available"])
        try:
            decoder_event, student_advisory_whitelisted = self._decoder_event(
                event, student
            )
            result = self.active.model.decide(
                prompt,
                decoder_event,
                network_available,
            )
            inference = dict(result["inference"])
            self._circuit_open_until = 0.0
            self.last_error = None
            decoded = dict(result["decision"])
            disagreement = _decision_disagrees(decoded, student)
            common = {
                "edge_llm_configured": True,
                "edge_llm_selected": True,
                "edge_llm_selection_reason": reason,
                "edge_llm_release_id": self.active.release_id,
                "edge_llm_release_revision": self.active.revision,
                "edge_llm_slot": inference.get("slot"),
                "edge_llm_token": inference.get("token"),
                "edge_llm_latency_ms": inference.get("latency_ms"),
                "edge_llm_prompt_tokens": inference.get("prompt_tokens"),
                "edge_llm_output_tokens": inference.get("output_tokens"),
                "edge_llm_safety_fallback": bool(decoded["safety_fallback"]),
                "edge_llm_fallback_reason": decoded.get("fallback_reason"),
                "edge_llm_requires_cloud": bool(decoded["requires_cloud"]),
                "edge_llm_model_disagreement": disagreement,
                "edge_llm_prompt_chars": len(prompt),
                "edge_llm_context_encoder": self.context_encoder,
                "edge_llm_student_advisory_whitelisted": (
                    student_advisory_whitelisted
                ),
            }
            if self.mode == "shadow":
                return _with_metadata(
                    student,
                    {
                        **common,
                        "edge_llm_shadow_decision": decoded["decision"],
                        "edge_decision_path": "student_with_llm_shadow",
                    },
                )
            if decoded["safety_fallback"] or decoded["decision"] in {
                "abstain",
                "request_cloud",
            }:
                self.fallbacks += 1
                return _with_metadata(
                    student,
                    {**common, "edge_decision_path": "student_safety_fallback"},
                )
            actions = [
                Action.from_dict(value, "edge_llm.actions")
                for value in decoded.get("actions", [])
            ]
            quality = float(
                self.active.model.validation.get("metrics", {}).get(
                    "decision_accuracy", student.confidence
                )
            )
            decision = build_decision(
                event=event,
                decision=str(decoded["decision"]),
                actions=actions,
                confidence=max(0.0, min(1.0, quality)),
                reason="validated Edge-Qwen single-token action",
                source="edge_qwen_single_token",
                policy_version=policy_version,
            )
            self.accepted += 1
            self.last_error = None
            return _with_metadata(
                decision,
                {**common, "edge_decision_path": "edge_qwen"},
            )
        except Exception as exc:  # noqa: BLE001
            self.fallbacks += 1
            self._circuit_open_until = (
                time.monotonic() + self.runtime_failure_cooldown_seconds
            )
            self.last_error = "{}: {}".format(type(exc).__name__, exc)
            return _with_metadata(
                student,
                {
                    "edge_llm_configured": True,
                    "edge_llm_selected": True,
                    "edge_llm_selection_reason": reason,
                    "edge_llm_release_id": self.active.release_id,
                    "edge_llm_runtime_error": self.last_error,
                    "edge_llm_requires_cloud": bool(
                        student.metadata.get("traffic_defer_recommended", False)
                        or self._structured_metadata(
                            student.metadata.get(
                                "model_uncertainty",
                                event.metadata.get("model_uncertainty"),
                            )
                        ).get("requires_review", False)
                    ),
                    "edge_llm_model_disagreement": True,
                    "edge_decision_path": "student_runtime_fallback",
                },
            )

    def health(self) -> Dict[str, Any]:
        active = self.active.describe() if self.active is not None else None
        retry_after = max(0.0, self._circuit_open_until - time.monotonic())
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "loaded": self.active is not None,
            "active": active,
            "context_encoder": self.context_encoder,
            "invocations": self.invocations,
            "accepted": self.accepted,
            "fallbacks": self.fallbacks,
            "deadline_probe_interval": self.deadline_probe_interval,
            "deadline_limited_candidates": self.deadline_limited_candidates,
            "student_confidence_threshold": self.student_confidence_threshold,
            "min_expected_gain": self.min_expected_gain,
            "gain_profile_path": str(self.gain_profile_path)
            if self.gain_profile_path is not None
            else None,
            "gain_profile_loaded": self.gain_profile is not None,
            "gain_profile_inactive_reason": self.gain_profile_inactive_reason,
            "gain_profile_accepted_strata": len(
                self.gain_profile.get("accepted_strata", {})
            )
            if self.gain_profile is not None
            else 0,
            "legacy_min_risk_level": self.min_risk_level,
            "risk_level_trigger_enabled": False,
            "last_error": self.last_error,
            "runtime_failure_cooldown_seconds": self.runtime_failure_cooldown_seconds,
            "runtime_circuit_open": retry_after > 0.0,
            "runtime_retry_after_seconds": round(retry_after, 6),
        }
