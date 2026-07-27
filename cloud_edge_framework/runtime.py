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
from cloud_edge_framework.review_tracking import ReviewLifecycleStore
from cloud_edge_framework.monitoring import CalibrationDriftMonitor
from cloud_edge_framework.scheduling import (
    CollaborationScheduler,
    NetworkSnapshot,
)


_REVIEW_CONTEXT_KEY = "_edge_review_context"
_RISK_PRIORITY = {"low": 0, "medium": 1, "high": 2, "severe": 3}


def _requires_cloud_confirmation(decision: DecisionEnvelope) -> bool:
    return any(
        action.parameters.get("requires_cloud_confirmation") is True
        for action in decision.actions
    )


def _with_action_authorization(
    decision: DecisionEnvelope,
    cloud_confirmed: bool,
) -> DecisionEnvelope:
    """Expose which proposed actions may execute at the current decision phase."""
    immediate = []
    deferred = []
    for action in decision.actions:
        if (
            action.parameters.get("requires_cloud_confirmation") is True
            and not cloud_confirmed
        ):
            deferred.append(action.action_type)
        else:
            immediate.append(action.action_type)
    metadata = dict(decision.metadata)
    metadata["action_authorization"] = {
        "cloud_confirmed": bool(cloud_confirmed),
        "immediate_action_types": immediate,
        "deferred_action_types": deferred,
        "all_actions_authorized": not deferred,
    }
    return replace(decision, metadata=metadata)


class CloudRuntime:
    def __init__(
        self,
        registry: Optional[SceneRegistry] = None,
        reviewer: Optional[Any] = None,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.reviewer = reviewer
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
        if self.reviewer is not None and self.reviewer.should_review(event):
            try:
                review = self.reviewer.review(event, decision)
                decision = plugin.apply_cloud_llm_review(
                    event, decision, review.to_dict()
                )
            except Exception as exc:  # noqa: BLE001
                decision = replace(
                    decision,
                    metadata={
                        **decision.metadata,
                        "cloud_llm_review_error": "{}: {}".format(
                            type(exc).__name__, exc
                        ),
                    },
                )
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
        review_tracker: Optional[ReviewLifecycleStore] = None,
        calibration_monitor: Optional[CalibrationDriftMonitor] = None,
        utility_router: Optional[Any] = None,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.cloud = cloud or CloudRuntime(self.registry)
        self.scheduler = scheduler or CollaborationScheduler()
        self.evidence_planner = evidence_planner or EvidencePlanner()
        self.review_store = review_store or PendingReviewStore()
        self.performance_store = performance_store or PerformanceProfileStore()
        self.feedback_store = feedback_store or DecisionFeedbackStore()
        self.review_tracker = review_tracker or ReviewLifecycleStore()
        self.calibration_monitor = calibration_monitor
        self.utility_router = utility_router

    @property
    def pending_reviews(self) -> List[SemanticEvent]:
        return self.review_store.events()

    @staticmethod
    def _pending_review_event(
        event: SemanticEvent,
        local: DecisionEnvelope,
        evidence_level: str,
        snapshot: NetworkSnapshot,
        request_bytes: int,
    ) -> SemanticEvent:
        metadata = dict(event.metadata)
        metadata[_REVIEW_CONTEXT_KEY] = {
            "schema_version": 1,
            "local_decision": local.to_dict(),
            "evidence_level": evidence_level,
            "network_class": network_class(snapshot),
            "request_bytes": max(0, int(request_bytes)),
        }
        return replace(event, metadata=metadata)

    @staticmethod
    def _clean_pending_event(event: SemanticEvent) -> SemanticEvent:
        if _REVIEW_CONTEXT_KEY not in event.metadata:
            return event
        metadata = dict(event.metadata)
        metadata.pop(_REVIEW_CONTEXT_KEY, None)
        return replace(event, metadata=metadata)


    @staticmethod
    def _routing_features(
        event: SemanticEvent,
        snapshot: NetworkSnapshot,
        evidence_level: str,
        planned_request_bytes: int,
        measured_cloud_path_ms: Optional[float],
        conflict_suspected: bool,
        model_disagreement: bool,
        monitoring_force_cloud_review: bool,
    ) -> Dict[str, Any]:
        return {
            "risk_priority": _RISK_PRIORITY[event.risk.level],
            "risk_score": float(event.risk.score),
            "prediction_confidence": float(event.prediction.confidence),
            "uncertainty_confidence": float(event.uncertainty.confidence),
            "prediction_set_size": len(
                event.uncertainty.prediction_set or [event.risk.level]
            ),
            "deadline_ms": float(event.timing.deadline_ms),
            "edge_work_ms": float(
                event.timing.preprocessing_ms + event.timing.edge_inference_ms
            ),
            "network_available": 1 if snapshot.available else 0,
            "network_rtt_ms": float(snapshot.rtt_ms),
            "network_jitter_ms": float(snapshot.jitter_ms),
            "network_loss_rate": float(snapshot.loss_rate),
            "uplink_mbps": float(snapshot.uplink_mbps),
            "downlink_mbps": float(snapshot.downlink_mbps),
            "planned_request_bytes": max(0, int(planned_request_bytes)),
            "evidence_level": str(evidence_level),
            "measured_cloud_path_ms": (
                None
                if measured_cloud_path_ms is None
                else float(measured_cloud_path_ms)
            ),
            "conflict_suspected": 1 if conflict_suspected else 0,
            "model_disagreement": 1 if model_disagreement else 0,
            "monitoring_force_cloud_review": (
                1 if monitoring_force_cloud_review else 0
            ),
        }

    def _record_replayed_feedback(
        self,
        stored_events: Sequence[SemanticEvent],
        cloud_events: Sequence[SemanticEvent],
        coordination: Dict[str, Any],
    ) -> Dict[str, Any]:
        decisions_by_event_id: Dict[str, DecisionEnvelope] = {}
        for raw_decision in coordination.get("decisions", []):
            if not isinstance(raw_decision, dict):
                continue
            decision = DecisionEnvelope.from_dict(raw_decision)
            for event_id in decision.event_ids:
                decisions_by_event_id[event_id] = decision

        local_records = 0
        cloud_submissions = 0
        legacy_events_skipped = 0
        cloud_submission_errors: List[str] = []
        for stored_event, cloud_event in zip(stored_events, cloud_events):
            context = stored_event.metadata.get(_REVIEW_CONTEXT_KEY)
            if not isinstance(context, dict):
                legacy_events_skipped += 1
                continue
            cloud_decision = decisions_by_event_id.get(cloud_event.event_id)
            if cloud_decision is None:
                raise ValueError(
                    "cloud coordination omitted decision for {}".format(
                        cloud_event.event_id
                    )
                )
            local_decision = DecisionEnvelope.from_dict(context["local_decision"])
            evidence_level = str(context.get("evidence_level", "summary"))
            network_label = str(context.get("network_class", "unknown"))
            request_bytes = int(context.get("request_bytes", 0))
            if self.feedback_store.append(
                cloud_event,
                local_decision,
                cloud_decision,
                evidence_level,
                network_label,
                request_bytes,
            ):
                local_records += 1
            if hasattr(self.cloud, "submit_feedback"):
                try:
                    if self.cloud.submit_feedback(
                        cloud_event,
                        local_decision,
                        cloud_decision,
                        evidence_level,
                        network_label,
                        request_bytes,
                    ):
                        cloud_submissions += 1
                except Exception as exc:  # noqa: BLE001
                    cloud_submission_errors.append(
                        "{}: {}".format(type(exc).__name__, exc)
                    )
        return {
            "local_records": local_records,
            "cloud_submissions": cloud_submissions,
            "legacy_events_skipped": legacy_events_skipped,
            "cloud_submission_errors": cloud_submission_errors,
        }

    def process(
        self,
        payload: Dict[str, Any],
        network: Optional[NetworkSnapshot] = None,
        conflict_suspected: bool = False,
        model_disagreement: bool = False,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        requested_at_ms = int(time.time() * 1000)
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
        monitoring_status = None
        if self.calibration_monitor is not None:
            monitoring_status = self.calibration_monitor.observe(
                event, plugin.monitoring_signals(event)
            )
            if monitoring_status["force_cloud_review"]:
                monitored_metadata = dict(event.metadata)
                monitored_metadata.update(
                    {
                        "cloud_review_requested": True,
                        "monitoring_force_cloud_review": True,
                        "monitoring_reasons": list(monitoring_status["reasons"]),
                    }
                )
                event = replace(event, metadata=monitored_metadata)
        normalization_done = time.perf_counter()
        scene_edge_inference_ms = event.timing.edge_inference_ms
        edge_decision_started = time.perf_counter()
        local = plugin.edge_decide(event)
        local_metadata = dict(local.metadata)
        local_metadata["trace_id"] = trace_id
        local = _with_action_authorization(
            replace(
                local,
                status="provisional",
                metadata=local_metadata,
            ),
            cloud_confirmed=False,
        )
        routing_advice = plugin.routing_advice(event, local)
        if not isinstance(routing_advice, dict):
            raise ValueError("scene routing_advice must return an object")
        edge_decision_runtime_ms = (time.perf_counter() - edge_decision_started) * 1000.0
        event = replace(
            event,
            timing=replace(
                event.timing,
                edge_inference_ms=scene_edge_inference_ms + edge_decision_runtime_ms,
            ),
        )
        edge_decision_done = time.perf_counter()
        edge_preliminary_decision_ms = (
            event.timing.preprocessing_ms
            + scene_edge_inference_ms
            + (edge_decision_done - started) * 1000.0
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
        planned_artifact_bytes = sum(
            item.size_bytes
            for item in cloud_event.evidence
            if item.uri is not None and item.uri.startswith("file://")
        )
        planned_upload_bytes = cloud_request_bytes + planned_artifact_bytes
        data_plane_done = time.perf_counter()
        profile = self.performance_store.estimate(
            event.scene, evidence_plan.required_level, snapshot
        )
        edge_llm_escalation = bool(local.metadata.get("edge_llm_requires_cloud", False))
        edge_llm_disagreement = bool(
            local.metadata.get("edge_llm_model_disagreement", False)
        )
        cloud_review_requested = bool(
            event.metadata.get("cloud_review_requested", False)
        )
        if _requires_cloud_confirmation(local):
            cloud_review_requested = True
        if bool(routing_advice.get("cloud_review_requested", False)):
            cloud_review_requested = True
        sla_probe_requested = (
            local.metadata.get("edge_llm_selection_reason")
            == "deadline_profile_probe"
        )
        routing_model_disagreement = (
            model_disagreement or edge_llm_escalation or edge_llm_disagreement
        )
        routing_features = self._routing_features(
            event,
            snapshot,
            evidence_plan.required_level,
            planned_upload_bytes,
            profile.cloud_path_ms if profile is not None else None,
            conflict_suspected,
            routing_model_disagreement,
            bool(event.metadata.get("monitoring_force_cloud_review", False)),
        )
        utility_route_prediction = None
        if self.utility_router is not None:
            utility_route_prediction = self.utility_router.predict(routing_features)
            if (
                utility_route_prediction.mode == "active"
                and utility_route_prediction.request_cloud
            ):
                cloud_review_requested = True
        schedule = self.scheduler.schedule(
            event,
            snapshot,
            conflict_suspected=conflict_suspected,
            model_disagreement=(
                model_disagreement or edge_llm_escalation or edge_llm_disagreement
            ),
            cloud_review_requested=cloud_review_requested,
            sla_probe_requested=sla_probe_requested,
            upload_bytes=planned_upload_bytes,
            evidence_level=evidence_plan.required_level,
            measured_cloud_path_ms=profile.cloud_path_ms if profile is not None else None,
            selective_defer=bool(
                routing_advice.get("selective_defer", False)
            ),
            defer_recommended=bool(
                routing_advice.get("defer_recommended", False)
            ),
            routing_risk_level=(
                str(routing_advice["routing_risk_level"])
                if routing_advice.get("routing_risk_level") is not None
                else None
            ),
        )
        warning = ""
        if schedule.cloud_requested and not evidence_plan.complete:
            warning = "required {} evidence is unavailable".format(evidence_plan.missing_level)
        scheduling_done = time.perf_counter()
        review_id = None

        if schedule.route == "cloud_sync":
            review_id = self.review_tracker.queue(
                event,
                local,
                "cloud_sync",
                evidence_plan.required_level,
                requested_at_ms,
                edge_preliminary_decision_ms,
                planned_upload_bytes,
                routing_features,
            )
            self.review_tracker.start([event.event_id], "sync")
            cloud_started = time.perf_counter()
            try:
                final = _with_action_authorization(
                    self.cloud.decide(cloud_event),
                    cloud_confirmed=True,
                )
                review_record = self.review_tracker.complete(
                    event.event_id, final, "sync"
                )
                final = replace(
                    final,
                    metadata={
                        **final.metadata,
                        "review_id": review_id,
                        "review_state": review_record["state"],
                        "decision_changed": review_record["decision_changed"],
                        "eventual_completion_ms": review_record[
                            "eventual_completion_ms"
                        ],
                    },
                )
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
                        int(transport.get("request_bytes", planned_upload_bytes)),
                        int(transport.get("response_bytes", 0)),
                    )
                feedback_request_bytes = int(
                    transport.get("request_bytes", planned_upload_bytes)
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
                self.review_tracker.retry(
                    [event.event_id], "{}: {}".format(type(exc).__name__, exc)
                )
                cloud_elapsed_ms = (time.perf_counter() - cloud_started) * 1000.0
                if hasattr(self.cloud, "base_url"):
                    self.performance_store.record(
                        event.scene,
                        evidence_plan.required_level,
                        snapshot,
                        False,
                        cloud_elapsed_ms,
                        planned_upload_bytes,
                        0,
                    )
                self.review_store.append(
                    self._pending_review_event(
                        cloud_event,
                        local,
                        evidence_plan.required_level,
                        snapshot,
                        planned_upload_bytes,
                    )
                )
                metadata = dict(local.metadata)
                metadata.update(
                    {
                        "local_autonomy": True,
                        "cloud_review_queued": True,
                        "review_id": review_id,
                        "review_state": "queued",
                        "cloud_error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
                final = replace(
                    local,
                    route="local_autonomy",
                    status="queued",
                    metadata=metadata,
                )
                final = _with_action_authorization(final, cloud_confirmed=False)
        elif schedule.route == "cloud_async":
            review_id = self.review_tracker.queue(
                event,
                local,
                "cloud_async",
                evidence_plan.required_level,
                requested_at_ms,
                edge_preliminary_decision_ms,
                planned_upload_bytes,
                routing_features,
            )
            self.review_store.append(
                self._pending_review_event(
                    cloud_event,
                    local,
                    evidence_plan.required_level,
                    snapshot,
                    planned_upload_bytes,
                )
            )
            metadata = dict(local.metadata)
            metadata.update(
                {
                    "cloud_review_queued": True,
                    "evidence_warning": warning,
                    "review_id": review_id,
                    "review_state": "queued",
                }
            )
            final = replace(
                local,
                route="cloud_async",
                status="queued",
                metadata=metadata,
            )
            final = _with_action_authorization(final, cloud_confirmed=False)
        elif schedule.route == "local_autonomy":
            review_queued = (
                event.risk.level in {"high", "severe"} or cloud_review_requested
            )
            if review_queued:
                review_id = self.review_tracker.queue(
                    event,
                    local,
                    "local_autonomy",
                    evidence_plan.required_level,
                    requested_at_ms,
                    edge_preliminary_decision_ms,
                    planned_upload_bytes,
                    routing_features,
                )
                self.review_store.append(
                    self._pending_review_event(
                        cloud_event,
                        local,
                        evidence_plan.required_level,
                        snapshot,
                        planned_upload_bytes,
                    )
                )
            metadata = dict(local.metadata)
            metadata.update(
                {
                    "local_autonomy": True,
                    "cloud_review_queued": review_queued,
                    "review_id": review_id,
                    "review_state": "queued" if review_queued else "not_requested",
                }
            )
            final = replace(
                local,
                route="local_autonomy",
                status="queued" if review_queued else "final",
                metadata=metadata,
            )
            final = _with_action_authorization(final, cloud_confirmed=False)
        else:
            final = _with_action_authorization(
                replace(local, route="edge_only", status="final"),
                cloud_confirmed=False,
            )

        final_transport = final.metadata.get("transport", {})
        if not isinstance(final_transport, dict):
            final_transport = {}
        actual_json_request_bytes = int(
            final_transport.get("json_request_bytes", cloud_request_bytes)
            if final_transport else 0
        )
        actual_artifact_request_bytes = int(
            final_transport.get("artifact_request_bytes", 0)
            if final_transport else 0
        )
        actual_transport_request_bytes = int(
            final_transport.get("request_bytes", 0) if final_transport else 0
        )
        route_done = time.perf_counter()
        runtime_ms = (route_done - started) * 1000.0
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
                "planned_artifact_request_bytes": planned_artifact_bytes,
                "planned_transport_request_bytes": planned_upload_bytes,
                "actual_json_request_bytes": actual_json_request_bytes,
                "actual_artifact_request_bytes": actual_artifact_request_bytes,
                "actual_transport_request_bytes": actual_transport_request_bytes,
                "transport_measurement": (
                    "measured" if final_transport else "not_transmitted_on_this_route"
                ),
                "request_reduction_ratio": round(
                    1.0 - cloud_request_bytes / max(1, legacy_request_bytes), 6
                ),
                "inline_encoded_evidence_bytes": evidence_plan.inline_encoded_bytes,
                "referenced_source_bytes": evidence_plan.referenced_source_bytes,
                "uncompressed_source_bytes": evidence_plan.uncompressed_source_bytes,
                "performance_profile": profile.to_dict() if profile is not None else None,
            },
            "evidence_warning": warning,
            "monitoring": monitoring_status,
            "utility_routing": (
                utility_route_prediction.to_dict()
                if utility_route_prediction is not None
                else {"enabled": False}
            ),
            "local_decision": local.to_dict(),
            "final_decision": final.to_dict(),
            "review": self.review_tracker.get(review_id) if review_id else None,
            "pending_review_count": self.review_store.count(),
            "feedback_count": self.feedback_store.count(),
            "framework_runtime_ms": round(runtime_ms, 6),
            "closed_loop_accounting": {
                "edge_preliminary_decision_ms": round(
                    edge_preliminary_decision_ms, 6
                ),
                "edge_preprocessing_ms": event.timing.preprocessing_ms,
                "edge_inference_ms": event.timing.edge_inference_ms,
                "scene_edge_model_reported_ms": scene_edge_inference_ms,
                "edge_decision_runtime_ms": round(edge_decision_runtime_ms, 6),
                "pipeline_stage_ms": {
                    "normalization": round((normalization_done - started) * 1000.0, 6),
                    "edge_decision": round(
                        (edge_decision_done - normalization_done) * 1000.0, 6
                    ),
                    "data_plane_preparation": round(
                        (data_plane_done - edge_decision_done) * 1000.0, 6
                    ),
                    "scheduling": round(
                        (scheduling_done - data_plane_done) * 1000.0, 6
                    ),
                    "route_execution": round(
                        (route_done - scheduling_done) * 1000.0, 6
                    ),
                },
                "post_model_framework_ms": round(runtime_ms, 6),
                "accounted_closed_loop_ms": round(accounted_closed_loop_ms, 6),
                "synchronous_cloud_closed_loop_ms": (
                    round(accounted_closed_loop_ms, 6)
                    if final.route == "cloud_sync"
                    else None
                ),
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
        self.review_tracker.start(event_ids, "replay")
        cloud_events = [self._clean_pending_event(event) for event in pending]
        try:
            coordination = self.cloud.coordinate(cloud_events)
            feedback = self._record_replayed_feedback(
                pending, cloud_events, coordination
            )
            review_completions = []
            pending_ids = set(event_ids)
            for raw_decision in coordination.get("decisions", []):
                if not isinstance(raw_decision, dict):
                    continue
                decision = DecisionEnvelope.from_dict(raw_decision)
                for decision_event_id in decision.event_ids:
                    if decision_event_id not in pending_ids:
                        continue
                    try:
                        review_completions.append(
                            self.review_tracker.complete(
                                decision_event_id, decision, "replay"
                            )
                        )
                    except KeyError:
                        continue
        except Exception as exc:  # noqa: BLE001
            error = "{}: {}".format(type(exc).__name__, exc)
            self.review_tracker.retry(event_ids, error)
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
            "feedback": feedback,
            "review_completions": review_completions,
            "review_lifecycle": self.review_tracker.snapshot(),
            "remaining": self.review_store.count(),
        }
