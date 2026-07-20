"""用途：串联场景适配、边缘决策、动态调度、云端复核和冲突协调。"""

import json
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

from cloud_edge_framework.conflicts import ConflictCoordinator, correlation_groups
from cloud_edge_framework.contracts import DecisionEnvelope, SemanticEvent, stable_id
from cloud_edge_framework.event_envelope import SceneEventEnvelope
from cloud_edge_framework.evidence import EvidencePlanner
from cloud_edge_framework.feedback import DecisionFeedbackStore
from cloud_edge_framework.performance import PerformanceProfileStore, network_class
from cloud_edge_framework.registry import SceneRegistry, build_default_registry
from cloud_edge_framework.review_queue import PendingReviewStore
from cloud_edge_framework.scheduling import (
    CollaborationScheduler,
    NetworkSnapshot,
)


class CloudRuntime:
    def __init__(self, registry: Optional[SceneRegistry] = None) -> None:
        self.registry = registry or build_default_registry()
        self.coordinator = ConflictCoordinator(self.registry)

    def warmup(self) -> None:
        self.registry.warmup()

    def normalize(self, payload: Dict[str, Any]) -> SemanticEvent:
        envelope = SceneEventEnvelope.from_dict(payload)
        plugin = self.registry.for_envelope(envelope)
        return plugin.normalize(envelope)

    def decide(self, event: SemanticEvent) -> DecisionEnvelope:
        plugin = self.registry.get(event.scene)
        decision = plugin.cloud_decide(event)
        metadata = dict(decision.metadata)
        metadata["cloud_verified"] = True
        if event.metadata.get("trace_id"):
            metadata["trace_id"] = event.metadata["trace_id"]
        return replace(
            decision,
            route="cloud_sync",
            status="final",
            metadata=metadata,
        )

    def decide_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        event = SemanticEvent.from_dict(payload)
        decision = self.decide(event)
        return {
            "event_id": event.event_id,
            "scene": event.scene,
            "decision": decision.to_dict(),
            "cloud_runtime_ms": round((time.perf_counter() - started) * 1000.0, 6),
        }

    def coordinate(self, events: Sequence[SemanticEvent]) -> Dict[str, Any]:
        started = time.perf_counter()
        fused_events = list(events)
        for scene in sorted({event.scene for event in events}):
            indices = [index for index, event in enumerate(events) if event.scene == scene]
            plugin = self.registry.get(scene)
            scene_events = plugin.fuse_cloud_context([events[index] for index in indices])
            if len(scene_events) != len(indices):
                raise ValueError("scene context fusion changed event count")
            for index, fused_event in zip(indices, scene_events):
                if fused_event.event_id != events[index].event_id:
                    raise ValueError("scene context fusion changed event identity")
                fused_events[index] = fused_event
        decisions = [self.decide(event) for event in fused_events]
        coordinated = self.coordinator.coordinate(fused_events, decisions)
        groups = correlation_groups(fused_events)
        result = coordinated.to_dict()
        result.update(
            {
                "event_count": len(events),
                "scenes": sorted({event.scene for event in events}),
                "correlation_groups": [
                    [fused_events[index].event_id for index in group] for group in groups
                ],
                "fusion": [
                    {
                        "event_id": event.event_id,
                        "method": event.metadata.get("topology_fusion"),
                        "neighbor_count": event.metadata.get("fused_neighbor_count", 0),
                    }
                    for event in fused_events
                ],
                "cloud_runtime_ms": round((time.perf_counter() - started) * 1000.0, 6),
            }
        )
        return result

    def coordinate_payloads(self, payloads: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not payloads:
            raise ValueError("events must not be empty")
        return self.coordinate([SemanticEvent.from_dict(payload) for payload in payloads])


class EdgeRuntime:
    def __init__(
        self,
        registry: Optional[SceneRegistry] = None,
        cloud: Optional[Any] = None,
        scheduler: Optional[CollaborationScheduler] = None,
        evidence_planner: Optional[EvidencePlanner] = None,
        review_store: Optional[PendingReviewStore] = None,
        performance_store: Optional[PerformanceProfileStore] = None,
        feedback_store: Optional[DecisionFeedbackStore] = None,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.cloud = cloud or CloudRuntime(self.registry)
        self.scheduler = scheduler or CollaborationScheduler()
        self.evidence_planner = evidence_planner or EvidencePlanner()
        self.review_store = review_store or PendingReviewStore()
        self.performance_store = performance_store or PerformanceProfileStore()
        self.feedback_store = feedback_store or DecisionFeedbackStore()

    @property
    def pending_reviews(self) -> List[SemanticEvent]:
        return self.review_store.events()

    def process(
        self,
        payload: Dict[str, Any],
        network: Optional[NetworkSnapshot] = None,
        conflict_suspected: bool = False,
        model_disagreement: bool = False,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        envelope = SceneEventEnvelope.from_dict(payload)
        plugin = self.registry.for_envelope(envelope)
        event = plugin.normalize(envelope)
        snapshot = network or NetworkSnapshot()
        trace_id = str(envelope.extensions.get("traceid", "")).strip() or stable_id(
            "trace", event.event_id
        )
        event_metadata = dict(event.metadata)
        event_metadata.update(
            {
                "trace_id": trace_id,
                "edge_runtime_network_available": snapshot.available,
                "edge_runtime_network_rtt_ms": snapshot.rtt_ms,
                "edge_runtime_network_loss_rate": snapshot.loss_rate,
            }
        )
        event = replace(event, metadata=event_metadata)
        scene_edge_inference_ms = event.timing.edge_inference_ms
        edge_decision_started = time.perf_counter()
        local = plugin.edge_decide(event)
        local_metadata = dict(local.metadata)
        local_metadata["trace_id"] = trace_id
        local = replace(local, metadata=local_metadata)
        edge_decision_runtime_ms = (time.perf_counter() - edge_decision_started) * 1000.0
        event = replace(
            event,
            timing=replace(
                event.timing,
                edge_inference_ms=scene_edge_inference_ms + edge_decision_runtime_ms,
            ),
        )
        evidence_plan = self.evidence_planner.plan(event, conflict_suspected)
        selected_evidence_ids = set(evidence_plan.selected_evidence_ids)
        selected_event = replace(
            event,
            evidence=[
                item for item in event.evidence if item.evidence_id in selected_evidence_ids
            ],
        )
        legacy_request_bytes = len(
            json.dumps(
                {"event": selected_event.to_dict(include_scene_payload=True)},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        cloud_event = plugin.prepare_cloud_event(
            selected_event, evidence_plan.required_level
        )
        include_scene_payload = bool(
            cloud_event.metadata.get("transport_include_scene_payload", False)
        )
        cloud_request_bytes = len(
            json.dumps(
                {
                    "event": cloud_event.to_dict(
                        include_scene_payload=include_scene_payload
                    )
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        profile = self.performance_store.estimate(
            event.scene, evidence_plan.required_level, snapshot
        )
        edge_llm_escalation = bool(local.metadata.get("edge_llm_requires_cloud", False))
        edge_llm_disagreement = bool(
            local.metadata.get("edge_llm_model_disagreement", False)
        )
        schedule = self.scheduler.schedule(
            event,
            snapshot,
            conflict_suspected=conflict_suspected,
            model_disagreement=(
                model_disagreement or edge_llm_escalation or edge_llm_disagreement
            ),
            upload_bytes=cloud_request_bytes,
            evidence_level=evidence_plan.required_level,
            measured_cloud_path_ms=profile.cloud_path_ms if profile is not None else None,
        )
        warning = ""
        if schedule.cloud_requested and not evidence_plan.complete:
            warning = "required {} evidence is unavailable".format(evidence_plan.missing_level)

        if schedule.route == "cloud_sync":
            cloud_started = time.perf_counter()
            try:
                final = self.cloud.decide(cloud_event)
                cloud_elapsed_ms = (time.perf_counter() - cloud_started) * 1000.0
                transport = final.metadata.get("transport", {})
                if not isinstance(transport, dict):
                    transport = {}
                if transport or hasattr(self.cloud, "base_url"):
                    self.performance_store.record(
                        event.scene,
                        evidence_plan.required_level,
                        snapshot,
                        True,
                        float(transport.get("http_round_trip_ms", cloud_elapsed_ms)),
                        int(transport.get("request_bytes", cloud_request_bytes)),
                        int(transport.get("response_bytes", 0)),
                    )
                feedback_request_bytes = int(
                    transport.get("request_bytes", cloud_request_bytes)
                )
                self.feedback_store.enqueue(
                    cloud_event,
                    local,
                    final,
                    evidence_plan.required_level,
                    network_class(snapshot),
                    feedback_request_bytes,
                )
                if hasattr(self.cloud, "submit_feedback"):
                    try:
                        self.cloud.submit_feedback(
                            cloud_event,
                            local,
                            final,
                            evidence_plan.required_level,
                            network_class(snapshot),
                            feedback_request_bytes,
                        )
                    except Exception as feedback_exc:  # noqa: BLE001
                        metadata = dict(final.metadata)
                        metadata["feedback_sync_error"] = "{}: {}".format(
                            type(feedback_exc).__name__, feedback_exc
                        )
                        final = replace(final, metadata=metadata)
            except Exception as exc:  # noqa: BLE001
                cloud_elapsed_ms = (time.perf_counter() - cloud_started) * 1000.0
                if hasattr(self.cloud, "base_url"):
                    self.performance_store.record(
                        event.scene,
                        evidence_plan.required_level,
                        snapshot,
                        False,
                        cloud_elapsed_ms,
                        cloud_request_bytes,
                        0,
                    )
                self.review_store.append(cloud_event)
                metadata = dict(local.metadata)
                metadata.update(
                    {
                        "local_autonomy": True,
                        "cloud_review_queued": True,
                        "cloud_error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
                final = replace(
                    local,
                    route="local_autonomy",
                    status="final",
                    metadata=metadata,
                )
        elif schedule.route == "cloud_async":
            self.review_store.append(cloud_event)
            metadata = dict(local.metadata)
            metadata.update({"cloud_review_queued": True, "evidence_warning": warning})
            final = replace(
                local,
                route="cloud_async",
                status="queued",
                metadata=metadata,
            )
        elif schedule.route == "local_autonomy":
            critical_review_queued = event.risk.level in {"high", "severe"}
            if critical_review_queued:
                self.review_store.append(cloud_event)
            metadata = dict(local.metadata)
            metadata.update(
                {
                    "local_autonomy": True,
                    "cloud_review_queued": critical_review_queued,
                }
            )
            final = replace(
                local,
                route="local_autonomy",
                status="final",
                metadata=metadata,
            )
        else:
            final = replace(local, route="edge_only", status="final")

        runtime_ms = (time.perf_counter() - started) * 1000.0
        accounted_closed_loop_ms = (
            event.timing.preprocessing_ms + scene_edge_inference_ms + runtime_ms
        )
        return {
            "trace_id": trace_id,
            "event": event.to_dict(include_scene_payload=False),
            "schedule": schedule.to_dict(),
            "evidence_plan": evidence_plan.to_dict(),
            "data_plane": {
                "legacy_full_request_bytes": legacy_request_bytes,
                "selected_request_bytes": cloud_request_bytes,
                "request_reduction_ratio": round(
                    1.0 - cloud_request_bytes / max(1, legacy_request_bytes), 6
                ),
                "inline_encoded_evidence_bytes": evidence_plan.inline_encoded_bytes,
                "referenced_source_bytes": evidence_plan.referenced_source_bytes,
                "uncompressed_source_bytes": evidence_plan.uncompressed_source_bytes,
                "performance_profile": profile.to_dict() if profile is not None else None,
            },
            "evidence_warning": warning,
            "local_decision": local.to_dict(),
            "final_decision": final.to_dict(),
            "pending_review_count": self.review_store.count(),
            "feedback_count": self.feedback_store.count(),
            "framework_runtime_ms": round(runtime_ms, 6),
            "closed_loop_accounting": {
                "edge_preprocessing_ms": event.timing.preprocessing_ms,
                "edge_inference_ms": event.timing.edge_inference_ms,
                "scene_edge_model_reported_ms": scene_edge_inference_ms,
                "edge_decision_runtime_ms": round(edge_decision_runtime_ms, 6),
                "post_model_framework_ms": round(runtime_ms, 6),
                "accounted_closed_loop_ms": round(accounted_closed_loop_ms, 6),
                "note": "reported scene inference is added once; measured edge decision and cloud transport are already inside framework time",
            },
        }

    def flush_pending(
        self,
        batch_size: int = 64,
        lease_seconds: float = 30.0,
        max_backoff_seconds: float = 60.0,
    ) -> Dict[str, Any]:
        leases = None
        if hasattr(self.review_store, "claim"):
            leases = self.review_store.claim(batch_size, lease_seconds)
            pending = [lease.event for lease in leases]
        else:
            pending = self.review_store.events()
        if not pending:
            return {
                "attempted": 0,
                "completed": 0,
                "coordination": None,
                "remaining": self.review_store.count(),
            }
        event_ids = [event.event_id for event in pending]
        try:
            coordination = self.cloud.coordinate(pending)
        except Exception as exc:  # noqa: BLE001
            error = "{}: {}".format(type(exc).__name__, exc)
            if leases is not None:
                self.review_store.release(
                    event_ids,
                    error,
                    max_backoff_seconds,
                )
            return {
                "attempted": len(pending),
                "completed": 0,
                "coordination": None,
                "error": error,
                "remaining": self.review_store.count(),
            }
        if leases is not None:
            self.review_store.acknowledge(event_ids)
        else:
            self.review_store.clear()
        return {
            "attempted": len(pending),
            "completed": len(pending),
            "coordination": coordination,
            "remaining": self.review_store.count(),
        }
