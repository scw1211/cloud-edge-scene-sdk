"""用途：演示场景原生模型输出如何通过插件接入统一云边运行时。"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Sequence

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
)
from cloud_edge_framework.event_envelope import SceneEventEnvelope
from cloud_edge_framework.plugins.base import ScenePlugin


class ExampleScenePlugin(ScenePlugin):
    """可运行的异常检测接入模板，不代表任何真实场景模型效果。"""

    scene = "industrial_anomaly"
    aliases = ("example",)
    event_types = ("com.example.industrial.anomaly-map.v1",)

    def __init__(
        self,
        policy_version: str = "example-0.2.0",
        template_mode: bool = True,
    ) -> None:
        self.policy_version = str(policy_version)
        self.template_mode = bool(template_mode)
        self._payload_schema = None

    def payload_schema(self) -> Dict[str, Any]:
        if self._payload_schema is None:
            schema_path = Path(__file__).with_name("data_schema.json")
            with schema_path.open("r", encoding="utf-8") as file_obj:
                self._payload_schema = json.load(file_obj)
        return dict(self._payload_schema)

    @staticmethod
    def _risk_level(score: float, threshold: float) -> str:
        margin = score - threshold
        if margin < 0:
            return "low"
        if margin < 0.10:
            return "medium"
        if margin < 0.25:
            return "high"
        return "severe"

    def normalize(self, envelope: SceneEventEnvelope) -> SemanticEvent:
        """把异常分数和热力图引用映射为内部调度语义。"""
        self.validate_envelope(envelope)
        if not isinstance(envelope.data, dict):
            raise ValueError("industrial anomaly data must be an object")
        payload = dict(envelope.data)
        score = float(payload["anomaly_score"])
        threshold = float(payload["threshold"])
        confidence = float(payload.get("confidence", score))
        risk_level = self._risk_level(score, threshold)
        asset_id = str(payload["asset_id"])
        resource_id = str(payload["shared_resource"])
        heatmap = dict(payload["heatmap"])
        window_ms = int(payload.get("window_ms", 5000))

        actions = [
            Action(
                action_type="set_operating_limit",
                target_ids=[asset_id],
                resource_ids=[resource_id],
                parameters={
                    "min_risk_level": "high",
                    "limit_percent": int(payload["proposed_limit_percent"]),
                },
                reason="hold throughput while a human reviews the anomaly evidence",
                priority=80,
            )
        ]
        evidence = [
            Evidence(
                evidence_id=envelope.event_id + "_summary",
                level="summary",
                modality="anomaly_summary",
                encoding="json",
                inline={
                    "anomaly_score": score,
                    "threshold": threshold,
                    "confidence": confidence,
                },
                size_bytes=64,
                content_type="application/json",
            ),
            Evidence(
                evidence_id=envelope.event_id + "_heatmap",
                level="feature",
                modality="anomaly_heatmap",
                encoding=str(heatmap["encoding"]),
                uri=str(heatmap["uri"]),
                shape=[int(value) for value in heatmap["shape"]],
                size_bytes=int(heatmap["size_bytes"]),
                content_type=str(heatmap.get("content_type", "application/octet-stream")),
                codec={"name": str(heatmap["encoding"]), "version": 1},
            ),
        ]
        return SemanticEvent(
            event_id=envelope.event_id,
            scene=self.scene,
            task="industrial_anomaly_review",
            edge_id=envelope.edge_id,
            occurred_at_ms=envelope.occurred_at_ms,
            scope=EventScope(
                entity_id=asset_id,
                subsystem=str(payload["subsystem"]),
                state_variable="anomaly_state",
                region_id=str(payload["region_id"]),
                shared_resources=[resource_id],
                correlation_keys=[asset_id + ":anomaly", resource_id],
                window_start_ms=envelope.occurred_at_ms - window_ms,
                window_end_ms=envelope.occurred_at_ms,
            ),
            prediction=Prediction(
                label="anomaly" if score >= threshold else "normal",
                confidence=confidence,
                values={"anomaly_score": score, "threshold": threshold},
            ),
            risk=Risk(level=risk_level, score=score),
            uncertainty=Uncertainty(
                confidence=confidence,
                calibrated=bool(payload.get("calibrated", False)),
                prediction_set=[risk_level],
                method=str(payload.get("confidence_method", "model_score")),
            ),
            timing=Timing(
                deadline_ms=float(payload.get("deadline_ms", 200)),
                preprocessing_ms=float(payload.get("preprocessing_latency_ms", 0)),
                edge_inference_ms=float(payload.get("inference_latency_ms", 0)),
            ),
            evidence=evidence,
            candidate_actions=actions,
            model=dict(payload.get("model", {})),
            scene_payload=payload,
            metadata={
                "adapter": "industrial_anomaly_map_v1",
                "ingress_type": envelope.event_type,
                "ingress_dataschema": envelope.dataschema,
                "transport_include_scene_payload": False,
            },
        )

    def edge_decide(self, event: SemanticEvent) -> DecisionEnvelope:
        """正式接入时可替换为场景 LoRA；模板只验证动作边界。"""
        return self.decision_from_candidates(
            event,
            source="example_edge_placeholder",
            confidence=event.prediction.confidence,
        )

    def cloud_decide(self, event: SemanticEvent) -> DecisionEnvelope:
        """正式接入时替换为云端专家模型或全局优化器。"""
        return self.decision_from_candidates(
            event,
            source="example_cloud_placeholder",
            confidence=event.uncertainty.confidence,
        )

    def prepare_cloud_event(
        self,
        event: SemanticEvent,
        evidence_level: str,
    ) -> SemanticEvent:
        metadata = dict(event.metadata)
        metadata.update(
            {
                "transport_include_scene_payload": False,
                "selected_evidence_level": evidence_level,
            }
        )
        return replace(event, scene_payload={}, metadata=metadata)

    def fuse_cloud_context(
        self,
        events: Sequence[SemanticEvent],
    ) -> Sequence[SemanticEvent]:
        return list(events)

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "template_mode": self.template_mode,
            "policy_version": self.policy_version,
        }
