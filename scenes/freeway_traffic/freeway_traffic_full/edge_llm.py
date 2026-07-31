"""用途：把已发布的交通 Edge-Qwen 接入插件，并保留确定性的 Student 安全路径。"""

from dataclasses import replace
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


def _with_metadata(
    decision: DecisionEnvelope,
    values: Dict[str, Any],
) -> DecisionEnvelope:
    metadata = dict(decision.metadata)
    metadata.update(values)
    return replace(decision, metadata=metadata)


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
        self.deadline_margin_ms = float(deadline_margin_ms)
        self.deadline_probe_interval = int(deadline_probe_interval)
        self.runtime_failure_cooldown_seconds = float(runtime_failure_cooldown_seconds)
        if self.mode not in EDGE_LLM_MODES:
            raise ValueError("edge_llm_mode must be one of {}".format(sorted(EDGE_LLM_MODES)))
        if self.min_risk_level not in RISK_PRIORITY:
            raise ValueError("edge_llm_min_risk_level is invalid")
        if not 0.0 <= self.student_confidence_threshold <= 1.0:
            raise ValueError("edge_llm_student_confidence_threshold must be in [0, 1]")
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
        if contract.get("context_encoder") != TRAFFIC_CONTEXT_ENCODER:
            raise ValueError("traffic Edge LLM uses an unsupported context encoder")
        if int(contract.get("max_input_tokens", 0)) > 16:
            raise ValueError("traffic Edge LLM input contract exceeds the 16-token budget")
        self.active = active
        self.last_error = None
        self._circuit_open_until = 0.0

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
        uncertain = (
            student.confidence < self.student_confidence_threshold
            or len(event.uncertainty.prediction_set) > 1
        )
        risk_selected = (
            RISK_PRIORITY[event.risk.level] >= RISK_PRIORITY[self.min_risk_level]
        )
        if self.mode == "selective" and not (uncertain or risk_selected):
            return False, "low_risk_confident_student"
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
        return True, "shadow" if self.mode == "shadow" else "selective"

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
        prompt = build_action_prompt(event.scene_payload, "bitpacked_decimal")
        network_available = bool(event.metadata["edge_runtime_network_available"])
        try:
            result = self.active.model.decide(
                prompt,
                event.to_dict(include_scene_payload=False),
                network_available,
            )
            inference = dict(result["inference"])
            self._circuit_open_until = 0.0
            self.last_error = None
            decoded = dict(result["decision"])
            disagreement = decoded["decision"] != student.decision
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
                    "edge_llm_requires_cloud": event.risk.level in {"high", "severe"},
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
            "invocations": self.invocations,
            "accepted": self.accepted,
            "fallbacks": self.fallbacks,
            "deadline_probe_interval": self.deadline_probe_interval,
            "deadline_limited_candidates": self.deadline_limited_candidates,
            "last_error": self.last_error,
            "runtime_failure_cooldown_seconds": self.runtime_failure_cooldown_seconds,
            "runtime_circuit_open": retry_after > 0.0,
            "runtime_retry_after_seconds": round(retry_after, 6),
        }
