"""用途：把交通模型的区域摘要接入公共云边汇聚与冲突协调链路。"""

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


_RISK_PRIORITY = {"low": 0, "medium": 1, "high": 2, "severe": 3}


class FreewayTrafficPlugin(ScenePlugin):
    """无权重参考插件；模型结果通过 data 输入，插件只做协议和安全决策。"""

    scene = "traffic"
    aliases = ("freeway_traffic_management",)
    event_types = ("com.cloudedge.traffic.edge-event.v1",)

    def __init__(
        self,
        policy_version: str = "traffic-portable-0.1.0",
        portable_demo_mode: bool = True,
    ) -> None:
        self.policy_version = str(policy_version)
        self.portable_demo_mode = bool(portable_demo_mode)
        self._payload_schema = None

    def payload_schema(self) -> Dict[str, Any]:
        if self._payload_schema is None:
            path = Path(__file__).with_name("data_schema.json")
            self._payload_schema = json.loads(path.read_text(encoding="utf-8"))
        return dict(self._payload_schema)

    @staticmethod
    def _candidate_actions(
        region_id: str,
        resource_id: str,
        nodes: Sequence[int],
        risk_level: str,
        risk_score: float,
        capabilities: Dict[str, Any],
    ):
        if risk_level == "low":
            return []
        if risk_level == "medium":
            return [
                Action(
                    action_type="congestion_warning",
                    target_ids=[region_id],
                    resource_ids=[resource_id],
                    parameters={"min_risk_level": "medium"},
                    reason="区域风险进入关注区间",
                    priority=50,
                )
            ]

        controlled = [
            int(node)
            for node in capabilities.get("variable_speed_limit_nodes", [])
            if int(node) in set(nodes)
        ]
        targets = [str(node) for node in (controlled or nodes[:3])]
        limit = 50 if risk_level == "severe" or risk_score >= 0.90 else 60
        actions = [
            Action(
                action_type="variable_speed_limit",
                target_ids=targets or [region_id],
                resource_ids=[resource_id],
                parameters={
                    "min_risk_level": "high",
                    "limit_kmh": limit,
                    "requires_cloud_confirmation": True,
                },
                reason="降低拥堵波向相邻区域传播的速度",
                priority=90 if risk_level == "severe" else 75,
            )
        ]
        ramp_nodes = [
            int(node)
            for node in capabilities.get("ramp_meter_nodes", [])
            if int(node) in set(nodes)
        ]
        if risk_level == "severe" and ramp_nodes:
            actions.append(
                Action(
                    action_type="ramp_metering",
                    target_ids=[str(node) for node in ramp_nodes],
                    resource_ids=[resource_id],
                    parameters={
                        "min_risk_level": "severe",
                        "rate_percent": 55,
                        "requires_cloud_confirmation": True,
                    },
                    reason="严重风险下限制匝道流入",
                    priority=95,
                )
            )
        return actions

    def normalize(self, envelope: SceneEventEnvelope) -> SemanticEvent:
        self.validate_envelope(envelope)
        payload = dict(envelope.data)
        summary = dict(payload["region_summary"])
        region_id = str(payload["region_id"])
        sample_id = str(payload["sample_id"])
        risk_level = str(summary["region_risk_level"])
        risk_score = float(summary["region_risk_score"])
        confidence = float(summary["region_risk_confidence"])
        nodes = [int(value) for value in payload["managed_node_ids"]]
        resource_id = str(
            payload.get("shared_resource", "corridor:" + region_id)
        )
        probabilities = {
            str(key): float(value)
            for key, value in dict(
                summary.get("region_risk_probabilities", {})
            ).items()
        }
        if not probabilities:
            probabilities = {risk_level: confidence}
        calibration = dict(summary.get("region_risk_calibration", {}))
        prediction_set = [
            str(value)
            for value in calibration.get("prediction_set", [risk_level])
        ]
        top_nodes = list(payload["top_k_risk_nodes"])
        actions = self._candidate_actions(
            region_id,
            resource_id,
            nodes,
            risk_level,
            risk_score,
            dict(payload.get("control_capabilities", {})),
        )
        metadata: Dict[str, Any] = {
            "adapter": "portable_traffic_summary_v1",
            "ingress_type": envelope.event_type,
            "ingress_dataschema": envelope.dataschema,
            "transport_include_scene_payload": False,
            "sample_id": sample_id,
            "dataset": str(payload.get("dataset", "portable_demo")),
            "sample_split": str(payload.get("sample_split", "demo")),
            "top_risk_nodes": top_nodes,
        }
        raw_aggregation = payload.get("aggregation")
        if isinstance(raw_aggregation, dict):
            expected = [
                str(value) for value in raw_aggregation["expected_members"]
            ]
            metadata["aggregation"] = {
                "key": "{}:{}:{}".format(
                    metadata["dataset"],
                    metadata["sample_split"],
                    sample_id,
                ),
                "member": str(raw_aggregation["member"]),
                "expected_members": expected,
                "minimum_members": int(
                    raw_aggregation.get("minimum_members", len(expected))
                ),
                "timeout_ms": int(raw_aggregation.get("timeout_ms", 200)),
            }

        return SemanticEvent(
            event_id=envelope.event_id,
            scene=self.scene,
            task="traffic_risk_assessment",
            edge_id=envelope.edge_id,
            occurred_at_ms=envelope.occurred_at_ms,
            scope=EventScope(
                entity_id=region_id,
                subsystem="freeway_corridor",
                state_variable="congestion_risk",
                region_id=region_id,
                shared_resources=[resource_id],
                correlation_keys=[
                    "traffic:" + sample_id,
                    resource_id,
                ],
                window_start_ms=envelope.occurred_at_ms - 300000,
                window_end_ms=envelope.occurred_at_ms,
            ),
            prediction=Prediction(
                label=risk_level,
                confidence=confidence,
                probabilities=probabilities,
                values={
                    "risk_score": risk_score,
                    "horizon_minutes": float(
                        payload["prediction_horizon_minutes"]
                    ),
                },
            ),
            risk=Risk(level=risk_level, score=risk_score),
            uncertainty=Uncertainty(
                confidence=float(
                    calibration.get("calibrated_confidence", confidence)
                ),
                calibrated=bool(calibration),
                prediction_set=prediction_set,
                method=str(calibration.get("method", "model_confidence")),
            ),
            timing=Timing(
                deadline_ms=float(payload.get("deadline_ms", 200)),
                preprocessing_ms=float(
                    payload.get("preprocessing_latency_ms", 0)
                ),
                edge_inference_ms=float(payload.get("inference_latency_ms", 0)),
            ),
            evidence=[
                Evidence(
                    evidence_id=envelope.event_id + "_summary",
                    level="summary",
                    modality="traffic_timeseries_summary",
                    encoding="json",
                    inline={
                        "region_id": region_id,
                        "risk_level": risk_level,
                        "risk_score": risk_score,
                        "confidence": confidence,
                        "top_nodes": top_nodes[:3],
                    },
                    size_bytes=len(
                        json.dumps(
                            {
                                "risk": risk_level,
                                "score": risk_score,
                                "confidence": confidence,
                            },
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ),
                    content_type="application/json",
                )
            ],
            candidate_actions=actions,
            model=dict(
                payload.get(
                    "model",
                    {"name": "external_traffic_model", "version": "unknown"},
                )
            ),
            scene_payload=payload,
            metadata=metadata,
        )

    def edge_decide(self, event: SemanticEvent) -> DecisionEnvelope:
        return self.decision_from_candidates(
            event,
            source="portable_traffic_edge_rule",
            confidence=event.prediction.confidence,
        )

    def cloud_decide(self, event: SemanticEvent) -> DecisionEnvelope:
        neighbor_level = str(
            event.metadata.get("fused_max_neighbor_risk", event.risk.level)
        )
        confidence = event.uncertainty.confidence
        if _RISK_PRIORITY.get(neighbor_level, 0) > _RISK_PRIORITY[event.risk.level]:
            confidence = min(confidence, 0.85)
        return self.decision_from_candidates(
            event,
            source="portable_traffic_cloud_rule",
            confidence=confidence,
        )

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
        fused = []
        for event in events:
            peers = [
                peer
                for peer in events
                if peer.event_id != event.event_id
                and set(peer.scope.correlation_keys)
                & set(event.scope.correlation_keys)
            ]
            metadata = dict(event.metadata)
            metadata["topology_fusion"] = "shared_correlation_key"
            metadata["fused_neighbor_count"] = len(peers)
            if peers:
                metadata["fused_max_neighbor_risk"] = max(
                    (peer.risk.level for peer in peers),
                    key=lambda value: _RISK_PRIORITY[value],
                )
            fused.append(replace(event, metadata=metadata))
        return fused

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "portable_demo_mode": self.portable_demo_mode,
            "policy_version": self.policy_version,
            "model_weights_included": False,
        }
