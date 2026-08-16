"""Industrial RGB/infrared anomaly plugin for the v0.13.1 runtime.

The plugin owns business semantics only.  Durable delivery, retries, persistent
multi-member aggregation and final-result backfill remain framework concerns.
"""

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlsplit

from cloud_edge_framework.contracts import (
    Action,
    DecisionEnvelope,
    Evidence,
    EventScope,
    Prediction,
    Risk,
    SemanticEvent,
    Timing,
    Uncertainty,
    build_decision,
)
from cloud_edge_framework.event_envelope import SceneEventEnvelope
from cloud_edge_framework.plugins.base import ScenePlugin
from edge_llm_factory.contracts import read_json_object
from edge_llm_factory.providers import (
    DISABLED_RUNTIME_SCHEMA,
    validate_disabled_runtime_config,
)
from edge_llm_factory.runtime import ConfiguredActionClient
from .action_codec import prompt_from_values
from .cloud_coordinator import IndustrialCloudCoordinator


PRODUCTS = (
    "button_cell",
    "capsule",
    "cotton",
    "cube",
    "piggy",
    "plastic_cylinder",
    "screw",
    "solar_panel",
    "toothbrush",
    "zipper",
)
EXPECTED_MODALITIES = ("rgb", "infrared")
STATE_RISK = {"normal": "low", "review": "medium", "anomaly": "high"}
INDUSTRIAL_EDGE_LLM_MODES = {"disabled", "shadow", "corroborate", "selective"}
INDUSTRIAL_ACTION_TOKENS = {"normal": "A", "review": "B", "anomaly": "C"}
INDUSTRIAL_SELECTIVE_TIMEOUT_SECONDS = 0.18
# A single-slot edge runtime serializes concurrent RGB/infrared review calls.
# Keep the low default, but allow a bounded per-deployment limit that covers
# both calls while the population-mean end-to-end SLA remains below 200 ms.
INDUSTRIAL_MAX_CONFIGURABLE_SELECTIVE_TIMEOUT_SECONDS = 0.5
INDUSTRIAL_CLOUD_LLM_MIN_EXPECTED_GAIN = 0.20
INDUSTRIAL_CLOUD_REVIEW_CONFIDENCE = 0.85


def _json_size(value: Any) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _canonical_modality(value: Any) -> str:
    text = str(value).strip().lower()
    if text == "rgb":
        return "rgb"
    if text in {"infra", "infrared"}:
        return "infrared"
    raise ValueError("industrial modality must be RGB or infrared")


def _product_from_payload(payload: Dict[str, Any]) -> str:
    explicit = str(payload.get("product", "")).strip().lower()
    if explicit:
        if explicit not in PRODUCTS:
            raise ValueError("unsupported industrial product: {}".format(explicit))
        return explicit
    raw_uri = str(payload.get("raw_uri", ""))
    path = unquote(urlsplit(raw_uri).path)
    parts = [part.lower() for part in Path(path).parts]
    matches = [product for product in PRODUCTS if product in parts]
    if len(matches) != 1:
        raise ValueError(
            "industrial product is required unless raw_uri contains exactly one "
            "supported product directory"
        )
    return matches[0]


def _confidence(score: float, low: float, high: float, state: str) -> float:
    width = max(high - low, high * 0.05, 1e-12)
    if state == "normal":
        distance = (low - score) / max(low, width)
    elif state == "review":
        center = (low + high) / 2.0
        distance = abs(score - center) / max(width / 2.0, 1e-12)
        return max(0.5, min(0.75, 0.75 - 0.25 * distance))
    else:
        distance = (score - high) / max(high, width)
    return max(0.75, min(0.99, 0.75 + 0.24 * max(0.0, distance)))


class IndustrialAnomalyPlugin(ScenePlugin):
    """Threshold edge decision plus durable RGB/infrared cloud reconciliation."""

    scene = "industrial_anomaly"
    aliases = ("industrial",)
    event_types = ("com.example.industrial.anomaly-map.v1",)
    policy_version = "industrial-1.0.0"

    def __init__(
        self,
        thresholds_path: Optional[Path] = None,
        policy_version: str = "industrial-1.0.0",
        aggregation_timeout_ms: int = 150,
        edge_llm_runtime_config_path: Optional[Path] = None,
        edge_llm_mode: str = "disabled",
        edge_llm_prompt_prefix: Optional[str] = None,
        edge_llm_selective_timeout_limit_seconds: float = (
            INDUSTRIAL_SELECTIVE_TIMEOUT_SECONDS
        ),
        cloud_model_path: Optional[Path] = None,
        cloud_llm_min_expected_gain: float = (
            INDUSTRIAL_CLOUD_LLM_MIN_EXPECTED_GAIN
        ),
        cloud_llm_review_confidence_threshold: float = (
            INDUSTRIAL_CLOUD_REVIEW_CONFIDENCE
        ),
    ) -> None:
        self.policy_version = str(policy_version)
        self.aggregation_timeout_ms = int(aggregation_timeout_ms)
        if self.aggregation_timeout_ms <= 0:
            raise ValueError("aggregation_timeout_ms must be positive")
        self.edge_llm_mode = str(edge_llm_mode).strip().lower()
        if self.edge_llm_mode not in INDUSTRIAL_EDGE_LLM_MODES:
            raise ValueError(
                "industrial edge_llm_mode must be one of {}".format(
                    sorted(INDUSTRIAL_EDGE_LLM_MODES)
                )
            )
        self.edge_llm_runtime_config_path = (
            Path(edge_llm_runtime_config_path).resolve()
            if edge_llm_runtime_config_path is not None
            else None
        )
        if self.edge_llm_mode != "disabled" and self.edge_llm_runtime_config_path is None:
            raise ValueError("enabled industrial Edge LLM requires a runtime config")
        if edge_llm_prompt_prefix not in (None, "", "I"):
            raise ValueError("industrial edge_llm_prompt_prefix must be I or empty")
        self.edge_llm_prompt_prefix = edge_llm_prompt_prefix
        self.edge_llm_selective_timeout_limit_seconds = float(
            edge_llm_selective_timeout_limit_seconds
        )
        if not (
            0.0 < self.edge_llm_selective_timeout_limit_seconds
            <= INDUSTRIAL_MAX_CONFIGURABLE_SELECTIVE_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "industrial edge_llm_selective_timeout_limit_seconds must be "
                "in (0, {:.2f}]".format(
                    INDUSTRIAL_MAX_CONFIGURABLE_SELECTIVE_TIMEOUT_SECONDS
                )
            )
        self._edge_llm_client: Optional[ConfiguredActionClient] = None
        self._edge_llm_release_disabled = False
        self._edge_llm_release_disabled_reason: Optional[str] = None
        self._edge_llm_release_binding: Optional[Dict[str, Any]] = None
        self._edge_llm_last_error: Optional[str] = None
        self._edge_llm_invocations = 0
        self._edge_llm_agreements = 0
        self._edge_llm_fallbacks = 0
        self.cloud_model_path = (
            Path(cloud_model_path).resolve()
            if cloud_model_path is not None
            else None
        )
        self.cloud_llm_min_expected_gain = float(cloud_llm_min_expected_gain)
        if not 0.0 <= self.cloud_llm_min_expected_gain <= 1.0:
            raise ValueError("industrial cloud_llm_min_expected_gain must be in [0, 1]")
        self.cloud_llm_review_confidence_threshold = float(
            cloud_llm_review_confidence_threshold
        )
        if not 0.5 <= self.cloud_llm_review_confidence_threshold <= 1.0:
            raise ValueError(
                "industrial cloud_llm_review_confidence_threshold must be in [0.5, 1]"
            )
        self._cloud_coordinator: Optional[IndustrialCloudCoordinator] = None
        self._cloud_model_last_error: Optional[str] = None
        self.thresholds_path = Path(
            thresholds_path or Path(__file__).with_name("review_bands.json")
        ).resolve()
        with self.thresholds_path.open("r", encoding="utf-8") as file_obj:
            thresholds = json.load(file_obj)
        if int(thresholds.get("schema_version", 0)) != 1:
            raise ValueError("industrial review bands schema_version must be 1")
        self._bands = dict(thresholds.get("modalities", {}))
        for modality in EXPECTED_MODALITIES:
            products = self._bands.get(modality)
            if not isinstance(products, dict) or set(products) != set(PRODUCTS):
                raise ValueError(
                    "industrial review bands must cover all products for {}".format(
                        modality
                    )
                )
            for product, band in products.items():
                low = float(band["review_low"])
                high = float(band["review_high"])
                if low < 0 or high <= low:
                    raise ValueError(
                        "invalid industrial review band for {}/{}".format(
                            modality, product
                        )
                    )
        self._payload_schema: Optional[Dict[str, Any]] = None

    def payload_schema(self) -> Dict[str, Any]:
        if self._payload_schema is None:
            with Path(__file__).with_name("data_schema.json").open(
                "r", encoding="utf-8"
            ) as file_obj:
                self._payload_schema = json.load(file_obj)
        return dict(self._payload_schema)

    def health(self) -> Dict[str, Any]:
        edge_qwen_effectively_enabled = (
            self.edge_llm_mode != "disabled"
            and not self._edge_llm_release_disabled
        )
        return {
            "status": "ok",
            "policy_version": self.policy_version,
            "decision_engine": (
                "industrial_selective_edge_qwen_with_deterministic_fast_path"
                if self.edge_llm_mode == "selective" and edge_qwen_effectively_enabled
                else (
                    "industrial_edge_qwen_with_deterministic_corroboration"
                    if edge_qwen_effectively_enabled
                    else "deterministic_review_bands"
                )
            ),
            "edge_qwen_available": self._edge_llm_client is not None,
            "edge_qwen_mode": self.edge_llm_mode,
            "edge_qwen_release_disabled": self._edge_llm_release_disabled,
            "edge_qwen_release_disabled_reason": self._edge_llm_release_disabled_reason,
            "edge_qwen_release_binding": (
                dict(self._edge_llm_release_binding)
                if self._edge_llm_release_binding is not None
                else None
            ),
            "edge_qwen_runtime_config_path": (
                str(self.edge_llm_runtime_config_path)
                if self.edge_llm_runtime_config_path is not None
                else None
            ),
            "edge_qwen_prompt_prefix": self.edge_llm_prompt_prefix,
            "edge_qwen_selective_timeout_limit_seconds": (
                self.edge_llm_selective_timeout_limit_seconds
            ),
            "edge_qwen_joint_prompt_contract": (
                self.edge_llm_prompt_prefix is not None
            ),
            "edge_qwen_invocations": self._edge_llm_invocations,
            "edge_qwen_agreements": self._edge_llm_agreements,
            "edge_qwen_fallbacks": self._edge_llm_fallbacks,
            "edge_qwen_last_error": self._edge_llm_last_error,
            "modalities": list(EXPECTED_MODALITIES),
            "product_count": len(PRODUCTS),
            "thresholds_path": str(self.thresholds_path),
            "cloud_decision_engine": (
                "industrial_extratrees_with_selective_qwen9b_review"
                if self._cloud_coordinator is not None
                else "deterministic_cross_modal_policy"
            ),
            "cloud_model_path": (
                str(self.cloud_model_path) if self.cloud_model_path is not None else None
            ),
            "cloud_model": (
                self._cloud_coordinator.describe()
                if self._cloud_coordinator is not None
                else None
            ),
            "cloud_model_last_error": self._cloud_model_last_error,
            "cloud_llm_min_expected_gain": self.cloud_llm_min_expected_gain,
            "cloud_llm_review_confidence_threshold": (
                self.cloud_llm_review_confidence_threshold
            ),
        }

    def warmup(self) -> None:
        if self.cloud_model_path is not None and self._cloud_coordinator is None:
            self._cloud_coordinator = IndustrialCloudCoordinator.load(
                self.cloud_model_path
            )
            self._cloud_model_last_error = None
        if self.edge_llm_mode == "disabled":
            return
        assert self.edge_llm_runtime_config_path is not None
        raw_runtime = read_json_object(self.edge_llm_runtime_config_path)
        if raw_runtime.get("schema_version") == DISABLED_RUNTIME_SCHEMA:
            disabled = validate_disabled_runtime_config(raw_runtime)
            self._edge_llm_client = None
            self._edge_llm_release_disabled = True
            self._edge_llm_release_disabled_reason = disabled["reason"]
            self._edge_llm_release_binding = dict(disabled["release_binding"])
            self._edge_llm_last_error = None
            return
        client = ConfiguredActionClient.from_path(self.edge_llm_runtime_config_path)
        runtime = client.describe()
        generation = runtime.get("generation", {})
        max_input_tokens = int(generation.get("max_input_tokens", 0))
        if self.edge_llm_prompt_prefix is not None:
            expected_tokens = 17 if self.edge_llm_prompt_prefix == "I" else 16
            if max_input_tokens != expected_tokens:
                raise ValueError(
                    "joint industrial Edge LLM requires exactly {} input tokens".format(
                        expected_tokens
                    )
                )
            if runtime.get("lora_adapter") is not None:
                raise ValueError(
                    "joint industrial Edge LLM runtime must omit request-level LoRA"
                )
        else:
            if max_input_tokens != 16:
                raise ValueError("industrial Edge LLM requires exactly 16 input tokens")
            if runtime.get("lora_adapter") is None:
                raise ValueError(
                    "industrial Edge LLM runtime must select a request-level LoRA"
                )
        if (
            self.edge_llm_mode == "selective"
            and float(runtime.get("timeout_seconds", 0.0))
            > self.edge_llm_selective_timeout_limit_seconds
        ):
            raise ValueError(
                "industrial selective Edge LLM timeout_seconds must be <= {:.2f}".format(
                    self.edge_llm_selective_timeout_limit_seconds
                )
            )
        self._edge_llm_client = client
        self._edge_llm_release_disabled = False
        self._edge_llm_release_disabled_reason = None
        binding = runtime.get("release_binding")
        self._edge_llm_release_binding = (
            dict(binding) if isinstance(binding, dict) else None
        )
        self._edge_llm_last_error = None

    def _edge_llm_prompt(self, event: SemanticEvent) -> str:
        modality = str(event.metadata["modality"])
        if modality not in EXPECTED_MODALITIES:
            raise ValueError("industrial Edge LLM modality is unsupported")
        product = str(event.metadata["product"])
        try:
            product_index = PRODUCTS.index(product)
        except ValueError as exc:
            raise ValueError("industrial Edge LLM product is unsupported") from exc
        prompt = prompt_from_values(
            modality,
            product_index,
            float(event.metadata["score"]),
            float(event.metadata["review_low"]),
            float(event.metadata["review_high"]),
        )
        if len(prompt) != 16 or not prompt.isdigit():
            raise ValueError("industrial Edge LLM prompt is not decimal16")
        if self.edge_llm_prompt_prefix is not None:
            prompt = self.edge_llm_prompt_prefix + prompt
        return prompt

    def _band(self, modality: str, product: str) -> Tuple[float, float]:
        raw = self._bands[modality][product]
        return float(raw["review_low"]), float(raw["review_high"])

    @staticmethod
    def _state(score: float, low: float, high: float) -> str:
        if score < low:
            return "normal"
        if score < high:
            return "review"
        return "anomaly"

    @staticmethod
    def _action(
        state: str,
        sample_id: str,
        resource_id: str,
        *,
        cloud_final: bool = False,
    ) -> List[Action]:
        if state == "normal":
            return []
        if state == "review":
            return [
                Action(
                    action_type="hold_for_review",
                    target_ids=[sample_id],
                    resource_ids=[resource_id],
                    parameters={
                        "reversible": True,
                        "requires_cloud_confirmation": False,
                        "evidence_required": True,
                    },
                    reason=(
                        "cross-modal evidence requires review"
                        if cloud_final
                        else "edge score is inside the calibrated review band"
                    ),
                    priority=70,
                )
            ]
        return [
            Action(
                action_type="quarantine_item",
                target_ids=[sample_id],
                resource_ids=[resource_id],
                parameters={
                    "reversible": True,
                    "requires_cloud_confirmation": False,
                    "evidence_required": bool(cloud_final),
                },
                reason="both modalities confirm an industrial anomaly"
                if cloud_final
                else "edge anomaly score exceeds the calibrated high threshold",
                priority=90,
            )
        ]

    def normalize(self, envelope: SceneEventEnvelope) -> SemanticEvent:
        self.validate_envelope(envelope)
        if not isinstance(envelope.data, dict):
            raise ValueError("industrial anomaly data must be an object")
        payload = dict(envelope.data)
        modality = _canonical_modality(payload["modality"])
        product = _product_from_payload(payload)
        sample_id = str(payload["sample_id"]).strip()
        score = float(payload["score"])
        low, high = self._band(modality, product)
        state = self._state(score, low, high)
        confidence = float(
            payload.get("confidence", _confidence(score, low, high, state))
        )
        confidence = max(0.0, min(1.0, confidence))
        risk_level = STATE_RISK[state]
        normalized_risk = max(0.0, min(1.0, score / max(high, 1e-12)))
        region_id = str(payload.get("region_id", "inspection_line_1"))
        resource_id = "industrial_item:{}:{}".format(product, sample_id)
        summary = {
            "sample_id": sample_id,
            "product": product,
            "modality": modality,
            "state": state,
            "score": score,
            "review_low": low,
            "review_high": high,
            "confidence": confidence,
        }
        evidence = [
            Evidence(
                evidence_id=envelope.event_id + "_summary",
                level="summary",
                modality="industrial_anomaly_summary",
                encoding="json",
                inline=summary,
                size_bytes=_json_size(summary),
                content_type="application/json",
            ),
            Evidence(
                evidence_id=envelope.event_id + "_heatmap",
                level="feature",
                modality="industrial_anomaly_heatmap",
                encoding="f32-map",
                uri=str(payload["heatmap_uri"]),
                shape=[160, 160],
                size_bytes=int(payload.get("heatmap_size_bytes", 160 * 160 * 4)),
                sha256=(
                    str(payload["heatmap_sha256"]).lower()
                    if payload.get("heatmap_sha256")
                    else None
                ),
                content_type="application/octet-stream",
                codec={"dtype": "float32", "layout": "HW", "version": 1},
            ),
        ]
        model = dict(payload.get("model", {}))
        model.setdefault("name", "industrial_patchcore_tensorrt")
        model.setdefault("version", "unversioned")
        return SemanticEvent(
            event_id=envelope.event_id,
            scene=self.scene,
            task="industrial_anomaly_assessment",
            edge_id=envelope.edge_id,
            occurred_at_ms=envelope.occurred_at_ms,
            scope=EventScope(
                entity_id=sample_id,
                subsystem=product,
                state_variable="anomaly_state",
                region_id=region_id,
                shared_resources=[resource_id],
                correlation_keys=[
                    "industrial_sample:{}:{}".format(product, sample_id),
                    resource_id,
                ],
                window_start_ms=envelope.occurred_at_ms,
                window_end_ms=envelope.occurred_at_ms,
            ),
            prediction=Prediction(
                label=state,
                confidence=confidence,
                probabilities={state: confidence},
                values={
                    "score": score,
                    "review_low": low,
                    "review_high": high,
                },
            ),
            risk=Risk(level=risk_level, score=normalized_risk),
            uncertainty=Uncertainty(
                confidence=confidence,
                calibrated=True,
                prediction_set=[risk_level],
                method="bootstrap_review_band",
            ),
            timing=Timing(
                deadline_ms=float(payload.get("deadline_ms", 200.0)),
                preprocessing_ms=float(payload.get("preprocessing_latency_ms", 0.0)),
                edge_inference_ms=float(payload.get("inference_ms", 0.0)),
            ),
            evidence=evidence,
            candidate_actions=self._action(state, sample_id, resource_id),
            model=model,
            scene_payload=payload,
            metadata={
                "adapter": "industrial_rgb_infra_v1",
                "ingress_type": envelope.event_type,
                "ingress_dataschema": envelope.dataschema,
                "sample_id": sample_id,
                "product": product,
                "modality": modality,
                "local_state": state,
                "score": score,
                "review_low": low,
                "review_high": high,
                "raw_uri": str(payload["raw_uri"]),
                "heatmap_uri": str(payload["heatmap_uri"]),
                "minimum_evidence_level": "summary",
                "transport_include_scene_payload": False,
                "model_uncertainty": {
                    "confidence": confidence,
                    "requires_review": state == "review",
                    "requires_synchronous_review": False,
                },
                "edge_qwen": {
                    "available": (
                        self.edge_llm_mode != "disabled"
                        and not self._edge_llm_release_disabled
                    ),
                    "selected": False,
                    "reason": (
                        "active_release_has_no_industrial_adapter"
                        if self._edge_llm_release_disabled
                        else (
                        "industrial_action_adapter_configured"
                        if self.edge_llm_mode != "disabled"
                        else "industrial_action_adapter_disabled"
                        )
                    ),
                },
                "aggregation": {
                    "key": "industrial:{}:{}".format(product, sample_id),
                    "member": modality,
                    "expected_members": list(EXPECTED_MODALITIES),
                    # One member can yield a useful but explicitly non-authoritative
                    # partial result after timeout; only both members become final.
                    "minimum_members": 1,
                    "timeout_ms": self.aggregation_timeout_ms,
                },
            },
        )

    def edge_decide(self, event: SemanticEvent) -> DecisionEnvelope:
        state = str(event.metadata["local_state"])
        rule_decision = build_decision(
            event=event,
            decision=state,
            actions=self._action(
                state,
                event.scope.entity_id,
                event.scope.shared_resources[0],
            ),
            confidence=event.prediction.confidence,
            reason="industrial edge review-band decision: {}".format(state),
            source="industrial_edge_review_bands",
            policy_version=self.policy_version,
        )
        metadata = dict(rule_decision.metadata)
        metadata.update(
            {
                "product": event.metadata["product"],
                "modality": event.metadata["modality"],
                "edge_qwen_selected": False,
                "edge_qwen_mode": self.edge_llm_mode,
                "decision_semantics": "normal_review_anomaly",
            }
        )
        rule_decision = replace(rule_decision, metadata=metadata)
        if self.edge_llm_mode == "disabled" or self._edge_llm_release_disabled:
            if self._edge_llm_release_disabled:
                disabled_metadata = dict(rule_decision.metadata)
                disabled_metadata.update(
                    {
                        "edge_decision_path": "industrial_rule_release_fallback",
                        "edge_qwen_selection_reason": (
                            "active_release_has_no_industrial_adapter"
                        ),
                        "edge_qwen_release_disabled": True,
                    }
                )
                return replace(rule_decision, metadata=disabled_metadata)
            return rule_decision
        if self.edge_llm_mode == "selective" and state != "review":
            fast_path = dict(rule_decision.metadata)
            fast_path.update(
                {
                    "edge_decision_path": "industrial_rule_fast_path",
                    "edge_qwen_selection_reason": (
                        "deterministic_normal_or_anomaly"
                    ),
                }
            )
            return replace(rule_decision, metadata=fast_path)
        if self._edge_llm_client is None:
            fallback = dict(rule_decision.metadata)
            fallback.update(
                {
                    "edge_qwen_selected": False,
                    "edge_qwen_fallback": True,
                    "edge_qwen_fallback_reason": "runtime_not_loaded",
                    "edge_qwen_selection_reason": "review_requires_corroboration",
                }
            )
            return replace(rule_decision, metadata=fallback)

        self._edge_llm_invocations += 1
        try:
            inference = self._edge_llm_client.predict(
                self._edge_llm_prompt(event), INDUSTRIAL_ACTION_TOKENS
            )
            predicted_state = str(inference["slot"])
            agrees = predicted_state == state
            if agrees:
                self._edge_llm_agreements += 1
            common = {
                "edge_qwen_selected": True,
                "edge_qwen_selection_reason": "review_requires_corroboration",
                "edge_qwen_prediction": predicted_state,
                "edge_qwen_token": inference.get("token"),
                "edge_qwen_latency_ms": inference.get("latency_ms"),
                "edge_qwen_prompt_tokens": inference.get("prompt_tokens"),
                "edge_qwen_output_tokens": inference.get("output_tokens"),
                "edge_qwen_rule_agreement": agrees,
                "edge_qwen_decoding_constraint": inference.get(
                    "decoding_constraint"
                ),
            }
            self._edge_llm_last_error = None
            if self.edge_llm_mode == "shadow":
                shadow = dict(rule_decision.metadata)
                shadow.update({**common, "edge_decision_path": "rule_with_llm_shadow"})
                return replace(rule_decision, metadata=shadow)
            if not agrees:
                self._edge_llm_fallbacks += 1
                fallback = dict(rule_decision.metadata)
                fallback.update(
                    {
                        **common,
                        "edge_qwen_fallback": True,
                        "edge_qwen_fallback_reason": "model_rule_disagreement",
                        "edge_decision_path": "rule_safety_fallback",
                    }
                )
                return replace(rule_decision, metadata=fallback)
            learned = build_decision(
                event=event,
                decision=state,
                actions=self._action(
                    state,
                    event.scope.entity_id,
                    event.scope.shared_resources[0],
                ),
                confidence=event.prediction.confidence,
                reason="validated industrial Edge-Qwen corroborates review-band policy",
                source="industrial_edge_qwen_corroborated",
                policy_version=self.policy_version,
            )
            learned_metadata = dict(learned.metadata)
            learned_metadata.update(
                {
                    "product": event.metadata["product"],
                    "modality": event.metadata["modality"],
                    "decision_semantics": "normal_review_anomaly",
                    "edge_qwen_mode": self.edge_llm_mode,
                    "edge_decision_path": "industrial_edge_qwen",
                    **common,
                }
            )
            return replace(learned, metadata=learned_metadata)
        except Exception as exc:  # noqa: BLE001
            self._edge_llm_fallbacks += 1
            self._edge_llm_last_error = "{}: {}".format(type(exc).__name__, exc)
            fallback = dict(rule_decision.metadata)
            fallback.update(
                {
                    "edge_qwen_selected": True,
                    "edge_qwen_fallback": True,
                    "edge_qwen_fallback_reason": "runtime_error",
                    "edge_qwen_runtime_error": self._edge_llm_last_error,
                    "edge_decision_path": "rule_runtime_fallback",
                }
            )
            return replace(rule_decision, metadata=fallback)

    def cloud_submission_metadata(
        self,
        event: SemanticEvent,
        local_decision: DecisionEnvelope,
    ) -> Dict[str, Any]:
        del event
        return {
            "edge_decision": local_decision.decision,
            "edge_decision_confidence": local_decision.confidence,
        }

    def evidence_advice(
        self,
        event: SemanticEvent,
        local_decision: DecisionEnvelope,
        conflict_suspected: bool,
    ) -> Dict[str, Any]:
        del event
        if conflict_suspected or local_decision.decision != "normal":
            return {
                "required_level": "feature",
                "reason": "industrial review or anomaly includes the heatmap evidence",
            }
        return {
            "required_level": "summary",
            "reason": "normal industrial delivery uses the compact decision summary",
        }

    def prepare_cloud_event(
        self,
        event: SemanticEvent,
        evidence_level: str,
    ) -> SemanticEvent:
        metadata = dict(event.metadata)
        metadata["selected_evidence_level"] = str(evidence_level)
        metadata["transport_include_scene_payload"] = False
        return replace(event, scene_payload={}, metadata=metadata)

    def fuse_cloud_context(
        self,
        events: Sequence[SemanticEvent],
    ) -> Sequence[SemanticEvent]:
        fused: List[SemanticEvent] = []
        for event in events:
            peers = [peer for peer in events if peer.event_id != event.event_id]
            peer_states = {
                str(peer.metadata.get("modality")): str(
                    peer.metadata.get("edge_decision", peer.metadata.get("local_state"))
                )
                for peer in peers
            }
            metadata = dict(event.metadata)
            metadata.update(
                {
                    "topology_fusion": "industrial_sample_cross_modal",
                    "fused_neighbor_count": len(peers),
                    "fused_peer_states": peer_states,
                }
            )
            fused.append(replace(event, metadata=metadata))
        return fused

    @staticmethod
    def _joint_state(events: Sequence[SemanticEvent]) -> Tuple[str, Dict[str, str]]:
        states = {
            str(event.metadata.get("modality")): str(
                event.metadata.get("edge_decision", event.metadata.get("local_state"))
            )
            for event in events
        }
        complete = set(states) == set(EXPECTED_MODALITIES)
        values = list(states.values())
        if complete and values and all(value == "normal" for value in values):
            return "normal", states
        if complete and values and all(value == "anomaly" for value in values):
            return "anomaly", states
        return "review", states

    @staticmethod
    def _cloud_records(events: Sequence[SemanticEvent]) -> Dict[str, Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        for event in events:
            modality = str(event.metadata.get("modality"))
            if modality in records:
                raise ValueError(
                    "industrial cloud group contains a duplicate {} member".format(
                        modality
                    )
                )
            records[modality] = {
                "score": float(event.metadata["score"]),
                "review_low": float(event.metadata["review_low"]),
                "review_high": float(event.metadata["review_high"]),
                "state": str(
                    event.metadata.get(
                        "edge_decision", event.metadata.get("local_state")
                    )
                ),
            }
        return records

    def _cloud_group_decisions(
        self,
        events: Sequence[SemanticEvent],
    ) -> List[DecisionEnvelope]:
        state, modality_states = self._joint_state(events)
        complete = set(modality_states) == set(EXPECTED_MODALITIES)
        product = str(events[0].metadata["product"])
        confidence = min(event.prediction.confidence for event in events)
        source = "industrial_cloud_cross_modal"
        reason = (
            "RGB and infrared decisions agree on {}".format(state)
            if state in {"normal", "anomaly"} and complete
            else "cross-modal disagreement or incomplete evidence requires review"
        )
        model_name = "deterministic_cross_modal_policy"
        model_metadata: Dict[str, Any] = {}
        review_policy = {
            "eligible": False,
            "reason": "industrial_cloud_extratrees_not_configured",
            "expected_gain": 0.0,
            "minimum_expected_gain": self.cloud_llm_min_expected_gain,
            "legacy_risk_trigger_used": False,
        }

        if complete and self._cloud_coordinator is not None:
            try:
                prediction = self._cloud_coordinator.predict(
                    product, self._cloud_records(events)
                )
                state = prediction.decision
                confidence = prediction.confidence
                source = "industrial_cloud_extratrees_coordinator"
                model_name = "industrial_extratrees"
                reason = (
                    "industrial ExtraTrees fused RGB and infrared score margins "
                    "into {}".format(state)
                )
                contains_review = "review" in modality_states.values()
                state_conflict = len(set(modality_states.values())) > 1
                uncertainty = max(0.0, 1.0 - confidence)
                expected_gain = min(
                    1.0,
                    uncertainty + (0.05 if state_conflict else 0.0),
                )
                review_candidate = (
                    contains_review
                    or state_conflict
                    or confidence < self.cloud_llm_review_confidence_threshold
                )
                eligible = (
                    review_candidate
                    and expected_gain >= self.cloud_llm_min_expected_gain
                )
                review_policy = {
                    "eligible": eligible,
                    "reason": (
                        "industrial_extratrees_uncertainty_with_expected_gain"
                        if eligible
                        else (
                            "industrial_extratrees_expected_gain_below_threshold"
                            if review_candidate
                            else "industrial_extratrees_high_confidence_consensus"
                        )
                    ),
                    "explicit_requested": False,
                    "model_uncertainty_requires_review": review_candidate,
                    "model_uncertainty_requires_synchronous_review": review_candidate,
                    "expected_gain": round(expected_gain, 6),
                    "expected_gain_source": "industrial_extratrees_confidence",
                    "minimum_expected_gain": self.cloud_llm_min_expected_gain,
                    "legacy_risk_trigger_used": False,
                }
                model_metadata = {
                    "cloud_model_id": self._cloud_coordinator.describe()["model_id"],
                    "cloud_llm_review_group_key": "industrial:{}:{}".format(
                        product, events[0].metadata["sample_id"]
                    ),
                    "cloud_model_confidence": round(confidence, 9),
                    "cloud_model_probabilities": {
                        key: round(value, 9)
                        for key, value in prediction.probabilities.items()
                    },
                    "cloud_review_context": {
                        "product": product,
                        "modality_states": dict(modality_states),
                        "modality_scores": {
                            modality: round(
                                float(record["score"]), 12
                            )
                            for modality, record in prediction.feature_context.items()
                        },
                        "extratrees_decision": state,
                        "extratrees_confidence": round(confidence, 9),
                    },
                }
                self._cloud_model_last_error = None
            except ValueError as exc:
                # The v1 artifact is deliberately capsule-only.  Unsupported
                # products retain the deterministic safe policy instead of
                # pretending the model generalizes beyond its frozen data.
                if "does not support product" not in str(exc):
                    self._cloud_model_last_error = "{}: {}".format(
                        type(exc).__name__, exc
                    )
                    state = "review"
                    confidence = max(0.5, min(0.85, confidence))
                    reason = "industrial ExtraTrees failed closed to review"
                    model_name = "industrial_extratrees_runtime_fallback"
                else:
                    model_metadata["cloud_model_fallback_reason"] = (
                        "unsupported_product"
                    )
                    review_policy["reason"] = (
                        "industrial_extratrees_product_not_supported"
                    )
            except Exception as exc:  # noqa: BLE001
                self._cloud_model_last_error = "{}: {}".format(
                    type(exc).__name__, exc
                )
                state = "review"
                confidence = max(0.5, min(0.85, confidence))
                reason = "industrial ExtraTrees failed closed to review"
                model_name = "industrial_extratrees_runtime_fallback"

        if state == "review":
            confidence = max(0.5, min(0.85, confidence))
        results: List[DecisionEnvelope] = []
        for event in events:
            decision = build_decision(
                event=event,
                decision=state,
                actions=self._action(
                    state,
                    event.scope.entity_id,
                    event.scope.shared_resources[0],
                    cloud_final=True,
                ),
                confidence=confidence,
                reason=reason,
                source=source,
                policy_version=self.policy_version,
            )
            metadata = dict(decision.metadata)
            metadata.update(
                {
                    "product": event.metadata["product"],
                    "sample_id": event.metadata["sample_id"],
                    "modality_states": dict(modality_states),
                    "cross_modal_complete": complete,
                    "evidence_escalation_required": state == "review",
                    "cloud_model": model_name,
                    "cloud_llm_review_policy": dict(review_policy),
                    **model_metadata,
                }
            )
            results.append(replace(decision, metadata=metadata))
        return results

    def cloud_decide_batch(
        self,
        events: Sequence[SemanticEvent],
    ) -> Sequence[DecisionEnvelope]:
        if not events:
            return []
        grouped: Dict[Tuple[str, str], List[int]] = {}
        for index, event in enumerate(events):
            key = (
                str(event.metadata.get("product")),
                str(event.metadata.get("sample_id")),
            )
            grouped.setdefault(key, []).append(index)
        decisions: List[Optional[DecisionEnvelope]] = [None] * len(events)
        for indices in grouped.values():
            group_decisions = self._cloud_group_decisions(
                [events[index] for index in indices]
            )
            for index, decision in zip(indices, group_decisions):
                decisions[index] = decision
        if any(decision is None for decision in decisions):
            raise RuntimeError("industrial cloud batch left an event undecided")
        return [decision for decision in decisions if decision is not None]

    def cloud_decide(self, event: SemanticEvent) -> DecisionEnvelope:
        return list(self.cloud_decide_batch([event]))[0]

    def apply_cloud_llm_review(
        self,
        event: SemanticEvent,
        baseline: DecisionEnvelope,
        review: Dict[str, Any],
    ) -> DecisionEnvelope:
        """Record Qwen's advisory review without replacing ExtraTrees authority.

        This intentionally matches the traffic scene: the scene-specific tree
        ensemble owns the online decision, while the full cloud model provides
        a structured audit signal for monitoring and future model updates.
        """

        del event
        metadata = dict(baseline.metadata)
        metadata["cloud_llm_review"] = dict(review)
        challenged = review.get("verdict") == "challenge"
        metadata["cloud_llm_challenged"] = challenged
        metadata["cloud_llm_baseline_preserved"] = True
        metadata["cloud_llm_review_role"] = "advisory_non_authoritative"
        if challenged:
            metadata["cloud_llm_advisory_recommendation"] = review.get(
                "recommended_decision"
            )
        return replace(baseline, metadata=metadata)

    def action_conflict(self, left: Action, right: Action) -> Tuple[bool, str]:
        if left.action_type == right.action_type and left.resource_ids == right.resource_ids:
            return False, ""
        return True, "industrial_modalities_proposed_different_actions"

    def resolve_action_conflict(
        self,
        left: Action,
        right: Action,
        left_event: SemanticEvent,
        right_event: SemanticEvent,
    ) -> Tuple[Action, Action, str]:
        del left, right
        sample_id = left_event.scope.entity_id
        resources = list(
            dict.fromkeys(left_event.scope.shared_resources + right_event.scope.shared_resources)
        )
        action = Action(
            action_type="hold_for_review",
            target_ids=[sample_id],
            resource_ids=resources,
            parameters={
                "reversible": True,
                "requires_cloud_confirmation": False,
                "evidence_required": True,
            },
            reason="conflicting industrial actions resolve to a safe review hold",
            priority=80,
        )
        reason = "industrial conflict resolved by reversible hold-for-review"
        return action, action, reason
