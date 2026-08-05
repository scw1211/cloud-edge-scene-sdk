"""用途：串联场景适配、边缘决策、动态调度、云端复核和冲突协调。"""

import hashlib
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
_SOURCE_ENVELOPE_SHA256_KEY = "_source_envelope_sha256"
_SOURCE_BUSINESS_CONTEXT_KEY = "_source_business_control_context"
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
        plugin = self.registry.for_envelope(envelope, validate=False)
        return plugin.normalize_envelope(envelope)

    def _finalize_decision(
        self,
        event: SemanticEvent,
        plugin: Any,
        decision: DecisionEnvelope,
    ) -> DecisionEnvelope:
        if self.reviewer is not None:
            review_stage = "eligibility"
            try:
                if self.reviewer.should_review(event):
                    review_stage = "inference"
                    review = self.reviewer.review(event, decision)
                    review_stage = "application"
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
                        "cloud_llm_review_error_stage": review_stage,
                        "cloud_llm_baseline_preserved": True,
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

    def decide(self, event: SemanticEvent) -> DecisionEnvelope:
        plugin = self.registry.get(event.scene)
        return self._finalize_decision(
            event,
            plugin,
            plugin.cloud_decide(event),
        )

    def decide_batch(
        self,
        events: Sequence[SemanticEvent],
    ) -> List[DecisionEnvelope]:
        normalized = list(events)
        if not normalized:
            return []
        plugin = self.registry.get(normalized[0].scene)
        if any(self.registry.get(event.scene) is not plugin for event in normalized[1:]):
            raise ValueError("cloud decision batch must contain one scene plugin")
        baselines = list(plugin.cloud_decide_batch(normalized))
        if len(baselines) != len(normalized):
            raise ValueError("scene cloud_decide_batch changed decision count")
        if any(not isinstance(decision, DecisionEnvelope) for decision in baselines):
            raise TypeError("scene cloud_decide_batch must return DecisionEnvelope values")
        return [
            self._finalize_decision(event, plugin, decision)
            for event, decision in zip(normalized, baselines)
        ]

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

    def coordinate_groups(
        self,
        event_groups: Sequence[Sequence[SemanticEvent]],
    ) -> List[Dict[str, Any]]:
        """Batch model inference while keeping every sample's logic isolated.

        A model tensor may contain ready events from several ``sample_id``
        groups, but topology fusion and conflict resolution are business
        operations and must never cross a sample boundary merely because two
        samples happened to share an inference batch.
        """
        started = time.perf_counter()
        groups = [list(events) for events in event_groups]
        if not groups or any(not events for events in groups):
            raise ValueError("cloud coordination groups must not be empty")

        fused_groups: List[List[SemanticEvent]] = []
        for events in groups:
            fused_events = list(events)
            for scene in sorted({event.scene for event in events}):
                indices = [
                    index
                    for index, event in enumerate(events)
                    if event.scene == scene
                ]
                plugin = self.registry.get(scene)
                scene_events = plugin.fuse_cloud_context(
                    [events[index] for index in indices]
                )
                if len(scene_events) != len(indices):
                    raise ValueError("scene context fusion changed event count")
                for index, fused_event in zip(indices, scene_events):
                    if fused_event.event_id != events[index].event_id:
                        raise ValueError(
                            "scene context fusion changed event identity"
                        )
                    fused_events[index] = fused_event
            fused_groups.append(fused_events)

        decision_groups: List[List[Optional[DecisionEnvelope]]] = [
            [None] * len(events) for events in fused_groups
        ]
        scenes = sorted(
            {event.scene for events in fused_groups for event in events}
        )
        for scene in scenes:
            references = [
                (group_index, event_index)
                for group_index, events in enumerate(fused_groups)
                for event_index, event in enumerate(events)
                if event.scene == scene
            ]
            scene_decisions = self.decide_batch(
                [
                    fused_groups[group_index][event_index]
                    for group_index, event_index in references
                ]
            )
            for (group_index, event_index), decision in zip(
                references, scene_decisions
            ):
                decision_groups[group_index][event_index] = decision

        results: List[Dict[str, Any]] = []
        for fused_events, raw_decisions in zip(fused_groups, decision_groups):
            if any(decision is None for decision in raw_decisions):
                raise RuntimeError("cloud decision batch left an event undecided")
            completed_decisions = [
                decision for decision in raw_decisions if decision is not None
            ]
            coordinated = self.coordinator.coordinate(
                fused_events, completed_decisions
            )
            correlation = correlation_groups(fused_events)
            result = coordinated.to_dict()
            result.update(
                {
                    "event_count": len(fused_events),
                    "scenes": sorted(
                        {event.scene for event in fused_events}
                    ),
                    "correlation_groups": [
                        [fused_events[index].event_id for index in group]
                        for group in correlation
                    ],
                    "fusion": [
                        {
                            "event_id": event.event_id,
                            "method": event.metadata.get("topology_fusion"),
                            "neighbor_count": event.metadata.get(
                                "fused_neighbor_count", 0
                            ),
                        }
                        for event in fused_events
                    ],
                    "coordination_semantics": (
                        "global_information_fusion_and_consistency_coordination"
                    ),
                    "global_optimality_claimed": False,
                    "cloud_batch_group_count": len(groups),
                    "cloud_runtime_ms": round(
                        (time.perf_counter() - started) * 1000.0, 6
                    ),
                }
            )
            results.append(result)
        return results

    def coordinate(self, events: Sequence[SemanticEvent]) -> Dict[str, Any]:
        return self.coordinate_groups([events])[0]

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
        durable_handoff: Optional[Any] = None,
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
        self.durable_handoff = durable_handoff

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
        delivery_operation: str,
        requested_route: str,
        requested_at_ms: int,
        preliminary_latency_ms: float,
        routing_features: Dict[str, Any],
    ) -> SemanticEvent:
        operation = str(delivery_operation).strip()
        if operation not in {"aggregate", "coordinate"}:
            raise ValueError("pending review delivery_operation is invalid")
        metadata = dict(event.metadata)
        metadata[_REVIEW_CONTEXT_KEY] = {
            "schema_version": 3,
            # Freeze the operation when the event is accepted.  A later plugin
            # reload must not silently turn a multi-edge summary into a regular
            # coordination request (or the reverse).
            "delivery_operation": operation,
            "local_decision": local.to_dict(),
            "requested_route": str(requested_route),
            "requested_at_ms": int(requested_at_ms),
            "preliminary_latency_ms": max(0.0, float(preliminary_latency_ms)),
            "evidence_level": evidence_level,
            "network_class": network_class(snapshot),
            "network_snapshot": {
                "available": snapshot.available,
                "rtt_ms": snapshot.rtt_ms,
                "jitter_ms": snapshot.jitter_ms,
                "loss_rate": snapshot.loss_rate,
                "cloud_queue_ms": snapshot.cloud_queue_ms,
                "cloud_compute_ms": snapshot.cloud_compute_ms,
                "uplink_mbps": snapshot.uplink_mbps,
                "downlink_mbps": snapshot.downlink_mbps,
                "expected_response_bytes": snapshot.expected_response_bytes,
            },
            "request_bytes": max(0, int(request_bytes)),
            "routing_features": dict(routing_features),
        }
        return replace(event, metadata=metadata)

    def _recover_pending_review_lifecycle(self, event: SemanticEvent) -> str:
        """Rebuild a missing auxiliary lifecycle row from the durable Outbox.

        Outbox persistence intentionally precedes lifecycle persistence.  This
        method closes that crash window before any replay can be delivered and
        acknowledged by using the frozen context stored with the event.
        """
        try:
            existing_review_id = str(
                self.review_tracker.get(event.event_id)["review_id"]
            )
        except KeyError:
            existing_review_id = None
        context = event.metadata.get(_REVIEW_CONTEXT_KEY)
        if not isinstance(context, dict):
            # Version-1 rows can rely on an already-persisted lifecycle record,
            # but a missing row cannot be reconstructed without the frozen
            # local decision and timing inputs introduced in version 2/3.
            if existing_review_id is not None:
                return existing_review_id
            raise ValueError(
                "pending Outbox event is missing durable review context: {}".format(
                    event.event_id
                )
            )
        raw_local = context.get("local_decision")
        if not isinstance(raw_local, dict):
            raise ValueError(
                "pending Outbox event is missing its local decision: {}".format(
                    event.event_id
                )
            )
        local = DecisionEnvelope.from_dict(raw_local)
        now_ms = int(time.time() * 1000)
        requested_at_ms = int(context.get("requested_at_ms", now_ms))
        preliminary_latency_ms = float(context.get("preliminary_latency_ms", 0.0))
        routing_features = context.get("routing_features", {})
        if not isinstance(routing_features, dict):
            raise ValueError("pending review routing_features must be an object")
        requested_route = str(context.get("requested_route", "cloud_async"))
        evidence_level = str(context.get("evidence_level", "summary"))
        request_bytes = max(0, int(context.get("request_bytes", 0)))
        # Queue is deliberately called even when a row already exists: besides
        # being idempotent, it validates that the durable Outbox event carries
        # the same immutable source/business identity as the lifecycle row.
        return self.review_tracker.queue(
            event,
            local,
            requested_route,
            evidence_level,
            requested_at_ms,
            preliminary_latency_ms,
            request_bytes,
            routing_features,
        )

    @staticmethod
    def _pending_delivery_operation(
        stored_event: SemanticEvent,
        cloud_event: SemanticEvent,
        plugin: Any,
    ) -> str:
        context = stored_event.metadata.get(_REVIEW_CONTEXT_KEY)
        if isinstance(context, dict):
            operation = str(context.get("delivery_operation", "")).strip()
            if operation in {"aggregate", "coordinate"}:
                return operation
        # Compatibility for version-1 Outbox records written before the
        # operation was frozen in the durable context.
        return (
            "aggregate"
            if plugin.aggregation_spec(cloud_event) is not None
            else "coordinate"
        )

    @staticmethod
    def _clean_pending_event(event: SemanticEvent) -> SemanticEvent:
        if _REVIEW_CONTEXT_KEY not in event.metadata:
            return event
        metadata = dict(event.metadata)
        metadata.pop(_REVIEW_CONTEXT_KEY, None)
        return replace(event, metadata=metadata)

    @staticmethod
    def _pending_network_snapshot(event: SemanticEvent) -> NetworkSnapshot:
        context = event.metadata.get(_REVIEW_CONTEXT_KEY)
        if isinstance(context, dict) and isinstance(
            context.get("network_snapshot"), dict
        ):
            return NetworkSnapshot.from_dict(context["network_snapshot"])
        # Legacy Outbox rows did not retain the full observation.  Use the
        # neutral default rather than fabricating precision from a class label.
        return NetworkSnapshot()

    @staticmethod
    def _pending_evidence_level(event: SemanticEvent) -> str:
        context = event.metadata.get(_REVIEW_CONTEXT_KEY)
        if not isinstance(context, dict):
            return "summary"
        return str(context.get("evidence_level", "summary"))

    @staticmethod
    def _coordination_decision(
        coordination: Any,
        event_id: str,
    ) -> Optional[DecisionEnvelope]:
        if not isinstance(coordination, dict):
            return None
        for raw_decision in coordination.get("decisions", []):
            if not isinstance(raw_decision, dict):
                continue
            decision = DecisionEnvelope.from_dict(raw_decision)
            if event_id in decision.event_ids:
                return decision
        return None

    @classmethod
    def _aggregation_decision(
        cls,
        response: Dict[str, Any],
        event_id: str,
    ) -> Optional[DecisionEnvelope]:
        decision = cls._coordination_decision(
            response.get("coordination"),
            event_id,
        )
        if decision is None:
            return None
        metadata = dict(decision.metadata)
        aggregation = response.get("aggregation")
        if isinstance(aggregation, dict):
            metadata["aggregation"] = {
                "group_id": aggregation.get("group_id"),
                "state": aggregation.get("state"),
                "completion_reason": aggregation.get("completion_reason"),
                "finality": aggregation.get("finality", "pending"),
                "evidence_complete": cls._aggregation_evidence_complete(response),
                "global_confirmation": cls._aggregation_globally_confirmed(response),
                "result_revision": int(
                    aggregation.get("result_revision", 0) or 0
                ),
                "received_members": list(
                    aggregation.get("received_members", [])
                ),
                "missing_members": list(
                    aggregation.get("missing_members", [])
                ),
            }
        transport = response.get("transport")
        if isinstance(transport, dict):
            metadata["transport"] = dict(transport)
        decision = replace(decision, metadata=metadata)
        return _with_action_authorization(
            decision,
            cloud_confirmed=cls._aggregation_globally_confirmed(response),
        )

    @staticmethod
    def _aggregation_globally_confirmed(response: Dict[str, Any]) -> bool:
        if not EdgeRuntime._aggregation_evidence_complete(response):
            return False
        aggregation = response.get("aggregation")
        coordination = response.get("coordination")
        if not isinstance(aggregation, dict) or not isinstance(coordination, dict):
            return False
        if "global_confirmation" in aggregation:
            confirmation = aggregation.get("global_confirmation") is True
        elif "cloud_confirmed" in aggregation:
            confirmation = aggregation.get("cloud_confirmed") is True
        else:
            # v1 cloud responses predate the explicit finality fields.  A
            # fully completed join plus a globally-consistent coordination is
            # the old protocol's equivalent of global confirmation.
            confirmation = True
        return bool(
            confirmation
            and coordination.get("globally_consistent", False) is True
        )

    @staticmethod
    def _aggregation_evidence_complete(response: Dict[str, Any]) -> bool:
        aggregation = response.get("aggregation")
        if not isinstance(aggregation, dict):
            return False
        structurally_complete = bool(
            aggregation.get("state") == "completed"
            and aggregation.get("completion_reason") == "all_expected_members"
            and not list(aggregation.get("missing_members", []))
        )
        if not structurally_complete:
            return False
        if "evidence_complete" not in aggregation:
            # Backward compatibility with the original aggregation response.
            return True
        return aggregation.get("evidence_complete") is True

    def _wait_for_aggregation_result(
        self,
        event: SemanticEvent,
        initial_response: Dict[str, Any],
        max_wait_seconds: float,
        poll_seconds: float = 0.01,
    ) -> Dict[str, Any]:
        """Poll a submitted high-risk aggregation within its business deadline.

        Cloud aggregation ingress deliberately acknowledges durable receipt
        without holding the HTTP request open. A high-risk edge request still
        needs synchronous review semantics, so the edge polls the result-only
        endpoint until a complete result appears or its remaining deadline is
        exhausted. Older/injected cloud clients without that endpoint retain the
        existing durable-Outbox fallback.
        """
        response = dict(initial_response)
        if self._aggregation_evidence_complete(response):
            return response
        if not hasattr(self.cloud, "aggregation_results_batch"):
            return response
        aggregation = response.get("aggregation", {})
        if not isinstance(aggregation, dict):
            return response
        group_id = str(aggregation.get("group_id", "")).strip()
        if not group_id or max_wait_seconds <= 0.0:
            return response
        deadline = time.monotonic() + max(0.0, float(max_wait_seconds))
        total_transport = self._response_transport(response)
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(max(0.001, float(poll_seconds)), remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            if getattr(self.cloud, "supports_request_timeout", False):
                batch = self.cloud.aggregation_results_batch(
                    [event],
                    {event.event_id: group_id},
                    timeout_seconds=remaining,
                )
            else:
                batch = self.cloud.aggregation_results_batch(
                    [event], {event.event_id: group_id}
                )
            if time.monotonic() > deadline:
                # Ignore a complete result that crossed the business deadline.
                # Returning the last on-time (typically waiting) response keeps
                # the already-accepted summary on result-only reconciliation
                # without authorizing a stale synchronous action.
                break
            if not isinstance(batch, dict):
                raise ValueError("aggregation result batch must be an object")
            items = batch.get("items", [])
            if not isinstance(items, list) or not items:
                raise ValueError("aggregation result batch is missing its item")
            candidate = items[0]
            if not isinstance(candidate, dict):
                raise ValueError("aggregation result item must be an object")
            response = dict(candidate)
            batch_transport = self._response_transport(batch)
            for name in ("request_bytes", "response_bytes", "http_round_trip_ms"):
                total_transport[name] = round(
                    float(total_transport.get(name, 0.0))
                    + float(batch_transport.get(name, 0.0)),
                    6,
                )
            if total_transport:
                response["transport"] = dict(total_transport)
            state = response.get("aggregation", {})
            state = state if isinstance(state, dict) else {}
            if self._aggregation_evidence_complete(response) or state.get(
                "state"
            ) == "completed":
                break
        return response

    @staticmethod
    def _response_transport(response: Any) -> Dict[str, Any]:
        if not isinstance(response, dict):
            return {}
        transport = response.get("transport", {})
        return dict(transport) if isinstance(transport, dict) else {}

    @staticmethod
    def _compact_decision(decision: DecisionEnvelope) -> Dict[str, Any]:
        """Project an executable decision without diagnostic metadata copies."""
        value = decision.to_dict()
        metadata = value.get("metadata", {})
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        retained = {
            name: metadata[name]
            for name in (
                "trace_id",
                "action_authorization",
                "edge_decision_path",
                "cloud_review_queued",
                "review_id",
                "review_state",
                "aggregation",
                "local_autonomy",
                "cloud_error",
                "edge_llm_selection_reason",
                "transport",
            )
            if name in metadata
        }
        value["metadata"] = retained
        return value

    @staticmethod
    def _compact_review(review: Dict[str, Any]) -> Dict[str, Any]:
        """Project lifecycle state without embedding either full decision."""
        return {
            name: review[name]
            for name in (
                "review_id",
                "event_id",
                "state",
                "requested_route",
                "requested_at_ms",
                "cloud_received_at_ms",
                "completed_at_ms",
                "cloud_receipt_latency_ms",
                "eventual_completion_ms",
                "decision_changed",
                "completion_mode",
                "completion_stage",
                "attempts",
                "last_error",
                "persistence_stage",
            )
            if name in review
        }

    @staticmethod
    def _summary_delivery_mode(persistence_stage: str) -> str:
        if persistence_stage in {"sync_cloud_review", "cloud_review_completed"}:
            return "sync_cloud_review"
        if persistence_stage == "handoff_durable":
            return "background_handoff"
        if persistence_stage == "outbox_durable":
            return "background_outbox"
        return "scheduler_selected"

    @staticmethod
    def _cloud_accepted_at_ms(response: Any) -> Optional[int]:
        """Read a cloud application-ingress timestamp without inventing one.

        A timestamp observed after the HTTP response would include coordination
        and optional LLM work, so it must never be labelled as cloud receipt.
        """
        value: Any = None
        if isinstance(response, DecisionEnvelope):
            value = response.metadata.get("cloud_accepted_at_ms")
        elif isinstance(response, dict):
            value = response.get("cloud_accepted_at_ms")
        if value is None or isinstance(value, bool):
            return None
        try:
            accepted_at_ms = int(value)
        except (TypeError, ValueError):
            return None
        return accepted_at_ms if accepted_at_ms >= 0 else None

    @staticmethod
    def _complete_review(
        tracker: Any,
        event_id: str,
        decision: DecisionEnvelope,
        completion_mode: str,
        completion_stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call the v2 lifecycle API without breaking injected v1 trackers."""
        if completion_stage is None:
            return tracker.complete(event_id, decision, completion_mode)
        try:
            return tracker.complete(
                event_id,
                decision,
                completion_mode,
                completion_stage=completion_stage,
            )
        except TypeError as exc:
            if "completion_stage" not in str(exc):
                raise
            return tracker.complete(event_id, decision, completion_mode)

    @staticmethod
    def _plan_evidence(
        planner: Any,
        event: SemanticEvent,
        conflict_suspected: bool,
        scene_policy: Dict[str, Any],
    ) -> Any:
        """Use scene-aware evidence planning while accepting the v1 extension API."""
        try:
            return planner.plan(
                event,
                conflict_suspected,
                scene_policy=scene_policy,
            )
        except TypeError as exc:
            # Only retry the exact old-signature failure; an implementation's
            # own TypeError must still surface.
            if "scene_policy" not in str(exc):
                raise
            return planner.plan(event, conflict_suspected)


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
        safety = event.metadata.get("operational_safety_risk", {})
        safety = dict(safety) if isinstance(safety, dict) else {}
        safety_level = str(safety.get("level", event.risk.level))
        if safety_level not in _RISK_PRIORITY:
            safety_level = event.risk.level
        safety_score = float(safety.get("score", event.risk.score))
        regional = event.metadata.get("regional_state", {})
        regional = dict(regional) if isinstance(regional, dict) else {}
        expected_gain = event.metadata.get("escalation_expected_gain", {})
        expected_gain = (
            dict(expected_gain) if isinstance(expected_gain, dict) else {}
        )
        gain_values: List[float] = []
        for key in ("edge_qwen", "cloud", "score", "expected_gain"):
            try:
                gain_values.append(float(expected_gain.get(key, 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        expected_gain_score = max(gain_values, default=0.0)
        evidence_state = event.metadata.get("evidence_completeness", {})
        evidence_state = (
            dict(evidence_state) if isinstance(evidence_state, dict) else {}
        )
        return {
            # v1 model field names are retained, but their traffic meaning is
            # now the consequence of executing the proposed action, not the
            # maximum congestion score of any observed node.
            "risk_priority": _RISK_PRIORITY[safety_level],
            "risk_score": safety_score,
            "operational_safety_level": safety_level,
            "regional_state_level": str(
                regional.get("level", event.prediction.label)
            ),
            "escalation_expected_gain": expected_gain_score,
            "evidence_complete": 1 if evidence_state.get("complete", True) else 0,
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
        response_detail: str = "full",
    ) -> Dict[str, Any]:
        response_detail = str(response_detail).strip().lower()
        if response_detail not in {"full", "compact"}:
            raise ValueError("response_detail must be full or compact")
        started = time.perf_counter()
        requested_at_ms = int(time.time() * 1000)
        envelope = SceneEventEnvelope.from_dict(payload)
        source_envelope_sha256 = hashlib.sha256(
            json.dumps(
                envelope.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        source_business_context = {
            "conflict_suspected": bool(conflict_suspected),
            "model_disagreement": bool(model_disagreement),
        }
        plugin = self.registry.for_envelope(envelope, validate=False)
        event = plugin.normalize_envelope(envelope)
        snapshot = network or NetworkSnapshot()
        trace_id = str(envelope.extensions.get("traceid", "")).strip() or stable_id(
            "trace", event.event_id
        )
        event_metadata = dict(event.metadata)
        measured_network_class = network_class(snapshot)
        edge_network_status = {
            "outage": "offline",
            "degraded": "weak",
            "good": "normal",
            "normal": "normal",
        }[measured_network_class]
        event_metadata.update(
            {
                "trace_id": trace_id,
                "edge_runtime_network_available": snapshot.available,
                "edge_runtime_network_status": edge_network_status,
                "edge_runtime_network_class": measured_network_class,
                "edge_runtime_network_rtt_ms": snapshot.rtt_ms,
                "edge_runtime_network_loss_rate": snapshot.loss_rate,
                _SOURCE_ENVELOPE_SHA256_KEY: source_envelope_sha256,
                _SOURCE_BUSINESS_CONTEXT_KEY: source_business_context,
            }
        )
        event = replace(event, metadata=event_metadata)
        monitoring_status = None
        if self.calibration_monitor is not None:
            observe_monitoring = getattr(
                self.calibration_monitor, "observe_deferred", None
            )
            if observe_monitoring is None:
                observe_monitoring = self.calibration_monitor.observe
            monitoring_status = observe_monitoring(
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
        cloud_submission_metadata = plugin.cloud_submission_metadata(event, local)
        if not isinstance(cloud_submission_metadata, dict):
            raise ValueError("scene cloud_submission_metadata must return an object")
        if (
            _SOURCE_ENVELOPE_SHA256_KEY in cloud_submission_metadata
            or _SOURCE_BUSINESS_CONTEXT_KEY in cloud_submission_metadata
        ):
            raise ValueError(
                "scene cloud_submission_metadata cannot replace source identity"
            )
        if cloud_submission_metadata:
            event = replace(
                event,
                metadata={**event.metadata, **cloud_submission_metadata},
            )
        routing_advice = plugin.routing_advice(event, local)
        if not isinstance(routing_advice, dict):
            raise ValueError("scene routing_advice must return an object")
        evidence_advice = plugin.evidence_advice(
            event, local, conflict_suspected
        )
        if not isinstance(evidence_advice, dict):
            raise ValueError("scene evidence_advice must return an object")
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
        evidence_plan = self._plan_evidence(
            self.evidence_planner,
            event,
            conflict_suspected,
            evidence_advice,
        )
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
        cloud_event = replace(
            cloud_event,
            metadata={
                **cloud_event.metadata,
                _SOURCE_ENVELOPE_SHA256_KEY: source_envelope_sha256,
                _SOURCE_BUSINESS_CONTEXT_KEY: source_business_context,
            },
        )
        aggregation_spec = plugin.aggregation_spec(cloud_event)
        if aggregation_spec is not None and not isinstance(aggregation_spec, dict):
            raise ValueError("scene aggregation_spec must return an object or None")
        delivery_operation = (
            "aggregate" if aggregation_spec is not None else "coordinate"
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
        local_model_uncertainty = local.metadata.get("model_uncertainty", {})
        local_model_uncertainty = (
            dict(local_model_uncertainty)
            if isinstance(local_model_uncertainty, dict)
            else {}
        )
        # ``requires_review`` is deliberately broad: scene plugins also use it
        # to request richer evidence and an eventual asynchronous correction.
        # Only an explicit synchronous uncertainty signal may block the
        # business response here.
        if "requires_synchronous_review" in local_model_uncertainty:
            decision_uncertain: Optional[bool] = bool(
                local_model_uncertainty["requires_synchronous_review"]
            )
        elif "requires_sync_review" in local_model_uncertainty:
            decision_uncertain = bool(
                local_model_uncertainty["requires_sync_review"]
            )
        elif "requires_review" in local_model_uncertainty:
            # Compatibility for plugins that implemented the original
            # one-level uncertainty contract. New scene plugins should emit
            # the explicit synchronous field, including ``False``.
            decision_uncertain = (
                True if bool(local_model_uncertainty["requires_review"]) else None
            )
        else:
            decision_uncertain = None
        cloud_review_requested = bool(
            event.metadata.get("cloud_review_requested", False)
        )
        cloud_llm_review_policy = event.metadata.get(
            "cloud_llm_review_policy", {}
        )
        if (
            isinstance(cloud_llm_review_policy, dict)
            and cloud_llm_review_policy.get("eligible") is True
        ):
            cloud_review_requested = True
        # Multi-edge summaries are delivered for global visibility, but delivery
        # alone is not a request to block the business path for cloud review.
        # Whether the caller waits is decided independently below.
        summary_delivery_required = aggregation_spec is not None
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
        explicit_routing_risk_level = routing_advice.get("routing_risk_level")
        if explicit_routing_risk_level is None:
            operational_safety_risk = event.metadata.get(
                "operational_safety_risk"
            )
            if (
                isinstance(operational_safety_risk, dict)
                and operational_safety_risk.get("level") is not None
            ):
                explicit_routing_risk_level = operational_safety_risk["level"]
        schedule = self.scheduler.schedule(
            event,
            snapshot,
            conflict_suspected=conflict_suspected,
            model_disagreement=(
                model_disagreement or edge_llm_escalation or edge_llm_disagreement
            ),
            decision_uncertain=decision_uncertain,
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
                str(explicit_routing_risk_level)
                if explicit_routing_risk_level is not None
                else None
            ),
        )
        scheduler_selected_route = schedule.route
        scheduler_selected_wait = bool(schedule.waits_for_cloud)
        if (
            summary_delivery_required
            and snapshot.available
            and snapshot.loss_rate < 0.95
            and schedule.route == "edge_only"
        ):
            schedule = replace(
                schedule,
                route="cloud_async",
                reason=(
                    "return the provisional decision immediately, durably upload "
                    "the lightweight summary, and observe the independent cloud "
                    "result channel"
                ),
                cloud_requested=True,
                waits_for_cloud=False,
            )
        warning = ""
        if schedule.cloud_requested and not evidence_plan.complete:
            warning = "required {} evidence is unavailable".format(evidence_plan.missing_level)
        scheduling_done = time.perf_counter()
        review_id = None
        review_response: Optional[Dict[str, Any]] = None
        persistence_stage = "not_required"
        ordinary_summary_fast_path = bool(
            summary_delivery_required
            and scheduler_selected_route == "edge_only"
            and schedule.route == "cloud_async"
            and self.durable_handoff is not None
        )

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
            persistence_stage = "sync_cloud_review"
            self.review_tracker.start([event.event_id], "sync")
            cloud_started = time.perf_counter()
            try:
                aggregation_response = None
                pre_cloud_closed_loop_ms = (
                    event.timing.preprocessing_ms
                    + scene_edge_inference_ms
                    + (time.perf_counter() - started) * 1000.0
                )
                cloud_timeout_seconds = max(
                    0.0,
                    (
                        float(schedule.deadline_ms)
                        - float(pre_cloud_closed_loop_ms)
                    )
                    / 1000.0,
                )
                if cloud_timeout_seconds <= 0.0:
                    raise TimeoutError(
                        "synchronous cloud review budget was exhausted"
                    )
                if aggregation_spec is not None and hasattr(self.cloud, "aggregate"):
                    if getattr(self.cloud, "supports_request_timeout", False):
                        aggregation_response = self.cloud.aggregate(
                            cloud_event,
                            timeout_seconds=cloud_timeout_seconds,
                        )
                    else:
                        aggregation_response = self.cloud.aggregate(cloud_event)
                    # Recompute after the submission returns.  Reusing the
                    # pre-submission estimate would grant result polling a new
                    # full window and let the synchronous path overrun the
                    # event's business deadline by the first HTTP round trip.
                    closed_loop_elapsed_ms = (
                        event.timing.preprocessing_ms
                        + scene_edge_inference_ms
                        + (time.perf_counter() - started) * 1000.0
                    )
                    remaining_deadline_seconds = max(
                        0.0,
                        (
                            float(schedule.deadline_ms)
                            - float(closed_loop_elapsed_ms)
                        )
                        / 1000.0,
                    )
                    if remaining_deadline_seconds > 0.0:
                        aggregation_response = self._wait_for_aggregation_result(
                            cloud_event,
                            aggregation_response,
                            remaining_deadline_seconds,
                        )
                        final = self._aggregation_decision(
                            aggregation_response, event.event_id
                        )
                    else:
                        # The cloud did durably accept this summary, so retain
                        # result-only reconciliation state. A result that arrived
                        # after the business deadline must not authorize the
                        # synchronous action even when it is already complete.
                        final = None
                else:
                    if getattr(self.cloud, "supports_request_timeout", False):
                        final = self.cloud.decide(
                            cloud_event,
                            timeout_seconds=cloud_timeout_seconds,
                        )
                    else:
                        final = self.cloud.decide(cloud_event)
                    closed_loop_elapsed_ms = (
                        event.timing.preprocessing_ms
                        + scene_edge_inference_ms
                        + (time.perf_counter() - started) * 1000.0
                    )
                    if closed_loop_elapsed_ms > float(schedule.deadline_ms):
                        raise TimeoutError(
                            "synchronous cloud review budget was exhausted"
                        )
                cloud_elapsed_ms = (time.perf_counter() - cloud_started) * 1000.0
                transport = (
                    aggregation_response.get("transport", {})
                    if isinstance(aggregation_response, dict)
                    else final.metadata.get("transport", {})
                )
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
                accepted_at_ms = self._cloud_accepted_at_ms(
                    aggregation_response if aggregation_response is not None else final
                )
                if accepted_at_ms is not None and hasattr(
                    self.review_tracker, "received"
                ):
                    self.review_tracker.received(
                        [event.event_id], received_at_ms=accepted_at_ms
                    )
                aggregation_complete = (
                    aggregation_response is None
                    or self._aggregation_evidence_complete(aggregation_response)
                )
                if final is None or not aggregation_complete:
                    waiting = (
                        aggregation_response.get("aggregation", {})
                        if isinstance(aggregation_response, dict)
                        else {}
                    )
                    self.review_tracker.retry(
                        [event.event_id],
                        "aggregation is waiting for peer summaries",
                    )
                    self.review_store.append(
                        self._pending_review_event(
                            cloud_event,
                            local,
                            evidence_plan.required_level,
                            snapshot,
                            int(
                                transport.get(
                                    "request_bytes", planned_upload_bytes
                                )
                            ),
                            delivery_operation,
                            "cloud_sync",
                            requested_at_ms,
                            edge_preliminary_decision_ms,
                            routing_features,
                        )
                    )
                    persistence_stage = "outbox_durable"
                    if hasattr(
                        self.review_store, "mark_aggregation_submitted"
                    ):
                        self.review_store.mark_aggregation_submitted(
                            [event.event_id]
                        )
                    metadata = dict(local.metadata)
                    if final is not None:
                        metadata.update(final.metadata)
                    aggregation_metadata = metadata.get("aggregation", {})
                    aggregation_metadata = (
                        dict(aggregation_metadata)
                        if isinstance(aggregation_metadata, dict)
                        else {}
                    )
                    aggregation_metadata.update(
                        {
                            "group_id": waiting.get("group_id"),
                            "state": waiting.get("state", "waiting"),
                            "completion_reason": waiting.get(
                                "completion_reason", ""
                            ),
                            "finality": waiting.get("finality", "pending"),
                            "evidence_complete": bool(
                                waiting.get("evidence_complete", False)
                            ),
                            "global_confirmation": bool(
                                waiting.get("global_confirmation", False)
                            ),
                            "result_revision": int(
                                waiting.get("result_revision", 0) or 0
                            ),
                            "received_members": list(
                                waiting.get("received_members", [])
                            ),
                            "missing_members": list(
                                waiting.get("missing_members", [])
                            ),
                        }
                    )
                    metadata.update(
                        {
                            "cloud_review_queued": True,
                            "review_id": review_id,
                            "review_state": "queued",
                            "aggregation": aggregation_metadata,
                            "transport": dict(transport),
                        }
                    )
                    final = _with_action_authorization(
                        replace(
                            final if final is not None else local,
                            route="cloud_async",
                            status="provisional",
                            metadata=metadata,
                        ),
                        cloud_confirmed=False,
                    )
                else:
                    final = _with_action_authorization(
                        final,
                        cloud_confirmed=(
                            aggregation_response is None
                            or self._aggregation_globally_confirmed(
                                aggregation_response
                            )
                        ),
                    )
                    review_record = self.review_tracker.complete(
                        event.event_id, final, "sync"
                    )
                    review_response = review_record
                    persistence_stage = "cloud_review_completed"
                    final = replace(
                        final,
                        metadata={
                            **final.metadata,
                            "review_id": review_id,
                            "review_state": review_record["state"],
                            "decision_changed": review_record[
                                "decision_changed"
                            ],
                            "eventual_completion_ms": review_record[
                                "eventual_completion_ms"
                            ],
                        },
                    )
                feedback_request_bytes = int(
                    transport.get("request_bytes", planned_upload_bytes)
                )
                if final.status == "final":
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
                        delivery_operation,
                        "cloud_sync",
                        requested_at_ms,
                        edge_preliminary_decision_ms,
                        routing_features,
                    )
                )
                persistence_stage = "outbox_durable"
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
                    status="provisional",
                    metadata=metadata,
                )
                final = _with_action_authorization(final, cloud_confirmed=False)
        elif schedule.route == "cloud_async":
            pending_event = self._pending_review_event(
                cloud_event,
                local,
                evidence_plan.required_level,
                snapshot,
                planned_upload_bytes,
                delivery_operation,
                "cloud_async",
                requested_at_ms,
                edge_preliminary_decision_ms,
                routing_features,
            )
            if ordinary_summary_fast_path:
                # The handoff journal is fsynced before submit returns, but the
                # larger SQLite Outbox transaction and lifecycle materialization
                # happen on its retrying background worker. This preserves a
                # crash-recoverable acceptance boundary without charging the
                # ordinary local-decision path for the full Outbox transaction.
                try:
                    self.durable_handoff.submit(
                        pending_event,
                        timeout_seconds=min(
                            0.05,
                            max(0.005, float(schedule.deadline_ms) / 1000.0),
                        ),
                    )
                    review_id = stable_id(
                        "review", event.event_id, local.decision_id
                    )
                    persistence_stage = "handoff_durable"
                    review_response = {
                        "review_id": review_id,
                        "event_id": event.event_id,
                        "requested_route": "cloud_async",
                        "state": "queued",
                        "persistence_stage": persistence_stage,
                    }
                except Exception:
                    # Saturation or a journal failure fails closed to the original
                    # synchronous durable path. Low latency is optional; accepted
                    # cloud intent durability is not.
                    ordinary_summary_fast_path = False
            if not ordinary_summary_fast_path:
                # The durable Outbox is the source of truth. Persist it before the
                # auxiliary lifecycle row so a process crash can never lose an
                # accepted summary merely because the two stores are not atomic.
                self.review_store.append(pending_event)
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
                persistence_stage = "outbox_durable"
            metadata = dict(local.metadata)
            metadata.update(
                {
                    "cloud_review_queued": True,
                    "evidence_warning": warning,
                    "review_id": review_id,
                    "review_state": "queued",
                    "summary_persistence_stage": persistence_stage,
                }
            )
            final = replace(
                local,
                route="cloud_async",
                status="provisional",
                metadata=metadata,
            )
            final = _with_action_authorization(final, cloud_confirmed=False)
        elif schedule.route == "local_autonomy":
            review_queued = (
                summary_delivery_required
                or event.risk.level in {"high", "severe"}
                or cloud_review_requested
            )
            if review_queued:
                self.review_store.append(
                    self._pending_review_event(
                        cloud_event,
                        local,
                        evidence_plan.required_level,
                        snapshot,
                        planned_upload_bytes,
                        delivery_operation,
                        "local_autonomy",
                        requested_at_ms,
                        edge_preliminary_decision_ms,
                        routing_features,
                    )
                )
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
                persistence_stage = "outbox_durable"
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
                status="provisional" if review_queued else "final",
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
        pipeline_stage_ms = {
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
        }
        if response_detail == "compact":
            return {
                "response_detail": "compact",
                "trace_id": trace_id,
                "event_id": event.event_id,
                "scene": event.scene,
                "schedule": {
                    "route": schedule.route,
                    "reason": schedule.reason,
                    "waits_for_cloud": schedule.waits_for_cloud,
                    "critical": schedule.critical,
                    "uncertain": schedule.uncertain,
                },
                "final_decision": self._compact_decision(final),
                "review": self._compact_review(review_response)
                if review_response is not None
                else self._compact_review(
                    {
                        "review_id": review_id,
                        "event_id": event.event_id,
                        "state": "queued" if review_id else "not_requested",
                        "persistence_stage": persistence_stage,
                    }
                ),
                "summary_delivery": {
                    "required": summary_delivery_required,
                    "mode": self._summary_delivery_mode(persistence_stage),
                    "persistence_stage": persistence_stage,
                    "fast_path": ordinary_summary_fast_path,
                },
                "data_plane": {
                    "selected_request_bytes": cloud_request_bytes,
                    "actual_json_request_bytes": actual_json_request_bytes,
                    "actual_artifact_request_bytes": actual_artifact_request_bytes,
                    "actual_transport_request_bytes": actual_transport_request_bytes,
                    "request_reduction_ratio": round(
                        1.0 - cloud_request_bytes / max(1, legacy_request_bytes), 6
                    ),
                },
                "framework_runtime_ms": round(runtime_ms, 6),
                "closed_loop_accounting": {
                    "edge_preliminary_decision_ms": round(
                        edge_preliminary_decision_ms, 6
                    ),
                    "accounted_closed_loop_ms": round(
                        accounted_closed_loop_ms, 6
                    ),
                    "synchronous_cloud_closed_loop_ms": (
                        round(accounted_closed_loop_ms, 6)
                        if final.route == "cloud_sync"
                        else None
                    ),
                    "pipeline_stage_ms": pipeline_stage_ms,
                },
            }

        if review_response is None and review_id:
            review_response = self.review_tracker.get(review_id)
        pending_review_count = self.review_store.count()
        if self.durable_handoff is not None:
            handoff_snapshot = self.durable_handoff.snapshot()
            pending_review_count += int(
                handoff_snapshot.get(
                    "pending", handoff_snapshot.get("durable_pending_count", 0)
                )
            )
        return {
            "trace_id": trace_id,
            "event": event.to_dict(include_scene_payload=False),
            "schedule": schedule.to_dict(),
            "evidence_plan": evidence_plan.to_dict(),
            "data_plane": {
                "summary_delivery_required": summary_delivery_required,
                "summary_delivery_mode": (
                    self._summary_delivery_mode(persistence_stage)
                ),
                "summary_persistence_stage": persistence_stage,
                "ordinary_summary_fast_path": ordinary_summary_fast_path,
                "scheduler_selected_route": scheduler_selected_route,
                "scheduler_selected_wait": scheduler_selected_wait,
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
            "review": review_response,
            "pending_review_count": pending_review_count,
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
                "pipeline_stage_ms": pipeline_stage_ms,
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
        waiting_poll_seconds: float = 0.025,
        partial_poll_seconds: float = 1.0,
        aggregation_max_wait_seconds: float = 10.0,
        reconciliation_poll_seconds: float = 5.0,
        reconciliation_max_wait_seconds: float = 60.0,
        aggregation_batch_wait_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        # Kept in the call signature for rolling compatibility with 0.13.1.
        # Cloud ingress no longer waits for peers or for a model batch.
        del aggregation_batch_wait_seconds
        leases = None
        reconciliation_ids = set()
        aggregation_submitted_ids = set()
        reconciliation_swept_expired_ids: List[str] = []
        if hasattr(self.review_store, "claim"):
            if hasattr(self.review_store, "sweep_expired_reconciliation"):
                reconciliation_swept_expired_ids = (
                    self.review_store.sweep_expired_reconciliation()
                )
            reconciliation_leases = []
            if hasattr(self.review_store, "claim_reconciliation"):
                # Reserve a small share so a continuous stream of ordinary
                # summaries cannot starve already-low-frequency correction
                # polls until their bounded deadline expires.
                reconciliation_capacity = max(1, int(batch_size) // 4)
                reconciliation_leases = self.review_store.claim_reconciliation(
                    reconciliation_capacity,
                    lease_seconds,
                )
                reconciliation_ids = {
                    lease.event.event_id for lease in reconciliation_leases
                }
            ordinary_capacity = max(
                0, int(batch_size) - len(reconciliation_leases)
            )
            leases = (
                self.review_store.claim(ordinary_capacity, lease_seconds)
                if ordinary_capacity
                else []
            )
            # Deliver new members before reconciliation polls in the same
            # batch, so a late B can complete the group before A re-reads it.
            leases.extend(reconciliation_leases)
            aggregation_submitted_ids = {
                lease.event.event_id
                for lease in leases
                if bool(getattr(lease, "aggregation_submitted", False))
                or bool(getattr(lease, "reconciliation", False))
            }
            pending = [lease.event for lease in leases]
        else:
            pending = self.review_store.events()
        if not pending:
            return {
                "attempted": 0,
                "completed": 0,
                "terminal": 0,
                "waiting": 0,
                "aggregation_waiting": 0,
                "retrying": 0,
                "reconciliation_waiting": 0,
                "reconciliation_completed": 0,
                "reconciliation_expired": len(
                    reconciliation_swept_expired_ids
                ),
                "coordination": None,
                "remaining": (
                    self.review_store.work_count()
                    if hasattr(self.review_store, "work_count")
                    else self.review_store.count()
                ),
            }
        event_ids = [event.event_id for event in pending]
        # The Outbox and lifecycle stores cannot share one SQLite transaction.
        # Recreate any auxiliary row lost by a crash after durable Outbox append
        # before the event is allowed to leave the edge.
        for stored_event in pending:
            self._recover_pending_review_lifecycle(stored_event)
        self.review_tracker.start(event_ids, "replay")
        cloud_events = [self._clean_pending_event(event) for event in pending]
        aggregation_items = []
        regular_items = []
        for stored_event, cloud_event in zip(pending, cloud_events):
            plugin = self.registry.get(cloud_event.scene)
            operation = self._pending_delivery_operation(
                stored_event, cloud_event, plugin
            )
            if operation == "aggregate" and hasattr(self.cloud, "aggregate"):
                aggregation_items.append((stored_event, cloud_event))
            else:
                regular_items.append((stored_event, cloud_event))

        completed_ids: List[str] = []
        retry_ids: List[str] = []
        waiting_ids: List[str] = []
        partial_waiting_ids: List[str] = []
        reconciliation_waiting_ids: List[str] = []
        partial_decisions: Dict[str, DecisionEnvelope] = {}
        reconciliation_completed_ids: List[str] = []
        reconciliation_retry_ids: List[str] = []
        errors: List[Dict[str, str]] = []
        review_completions = []
        feedback_totals = {
            "local_records": 0,
            "cloud_submissions": 0,
            "legacy_events_skipped": 0,
            "cloud_submission_errors": [],
        }
        coordination_results = []
        deliveries: List[Dict[str, Any]] = []
        partial_results = 0

        aggregation_submit_items = [
            item
            for item in aggregation_items
            if item[1].event_id not in aggregation_submitted_ids
        ]
        aggregation_result_items = [
            item
            for item in aggregation_items
            if item[1].event_id in aggregation_submitted_ids
        ]
        batched_aggregation_responses: Dict[str, Dict[str, Any]] = {}
        aggregation_batch_errors: Dict[str, Exception] = {}
        aggregation_batch_elapsed_ms: Dict[str, float] = {}
        aggregation_batch_transport: Dict[str, Dict[str, Any]] = {}
        aggregation_batch_attempted_ids = set()

        def record_batch_response(
            operation: str,
            items: List[Any],
            batch_response: Dict[str, Any],
            elapsed_ms: float,
        ) -> None:
            raw_items = batch_response.get("items")
            if not isinstance(raw_items, list):
                raise ValueError(
                    "{} response omitted items".format(operation)
                )
            responses: Dict[str, Dict[str, Any]] = {}
            for response in raw_items:
                if not isinstance(response, dict):
                    raise ValueError(
                        "{} response items must be objects".format(operation)
                    )
                event_id = str(response.get("event_id", "")).strip()
                if not event_id or event_id in responses:
                    raise ValueError(
                        "{} response has invalid event identity".format(
                            operation
                        )
                    )
                responses[event_id] = dict(response)
            expected_ids = {item[1].event_id for item in items}
            if set(responses) != expected_ids:
                raise ValueError(
                    "{} response event identities do not match request".format(
                        operation
                    )
                )
            batch_transport = self._response_transport(batch_response)
            item_count = len(items)
            for index, (_, cloud_event) in enumerate(items):
                event_id = cloud_event.event_id
                item_transport = dict(batch_transport)
                for metric_name in ("request_bytes", "response_bytes"):
                    total = int(batch_transport.get(metric_name, 0))
                    quotient, remainder = divmod(total, item_count)
                    item_transport[metric_name] = quotient + int(
                        index < remainder
                    )
                responses[event_id]["transport"] = item_transport
                batched_aggregation_responses[event_id] = responses[event_id]
                aggregation_batch_elapsed_ms[event_id] = elapsed_ms
                aggregation_batch_transport[event_id] = item_transport
            deliveries.append(
                {
                    "operation": operation,
                    "event_count": item_count,
                    "success": True,
                    "http_round_trip_ms": float(
                        batch_transport.get("http_round_trip_ms", elapsed_ms)
                    ),
                    "request_bytes": int(
                        batch_transport.get("request_bytes", 0)
                    ),
                    "response_bytes": int(
                        batch_transport.get("response_bytes", 0)
                    ),
                }
            )

        def run_batch(operation: str, items: List[Any]) -> None:
            if not items:
                return
            event_ids = {item[1].event_id for item in items}
            aggregation_batch_attempted_ids.update(event_ids)
            batch_started = time.perf_counter()
            try:
                if operation == "aggregate_batch":
                    batch_response = self.cloud.aggregate_batch(
                        [item[1] for item in items]
                    )
                else:
                    event_group_ids = {}
                    for _, cloud_event in items:
                        plugin = self.registry.get(cloud_event.scene)
                        raw_spec = plugin.aggregation_spec(cloud_event)
                        if not isinstance(raw_spec, dict):
                            raise ValueError(
                                "result lookup requires an aggregation spec"
                            )
                        key = str(raw_spec.get("key", "")).strip()
                        if not key:
                            raise ValueError(
                                "result lookup aggregation key is empty"
                            )
                        event_group_ids[cloud_event.event_id] = stable_id(
                            "aggregation", cloud_event.scene, key
                        )
                    batch_response = self.cloud.aggregation_results_batch(
                        [item[1] for item in items], event_group_ids
                    )
                elapsed_ms = (
                    time.perf_counter() - batch_started
                ) * 1000.0
                record_batch_response(
                    operation, items, batch_response, elapsed_ms
                )
            except Exception as exc:  # noqa: BLE001
                elapsed_ms = (
                    time.perf_counter() - batch_started
                ) * 1000.0
                for event_id in event_ids:
                    aggregation_batch_errors[event_id] = exc
                    aggregation_batch_elapsed_ms[event_id] = elapsed_ms
                    aggregation_batch_transport[event_id] = {}
                deliveries.append(
                    {
                        "operation": operation,
                        "event_count": len(items),
                        "success": False,
                        "http_round_trip_ms": round(
                            elapsed_ms, 6
                        ),
                        "request_bytes": 0,
                        "response_bytes": 0,
                    }
                )

        if aggregation_submit_items and hasattr(self.cloud, "aggregate_batch"):
            run_batch("aggregate_batch", aggregation_submit_items)
        if aggregation_result_items and hasattr(
            self.cloud, "aggregation_results_batch"
        ):
            run_batch("aggregate_results_batch", aggregation_result_items)

        for stored_event, cloud_event in aggregation_items:
            is_reconciliation = cloud_event.event_id in reconciliation_ids
            batch_attempted = (
                cloud_event.event_id in aggregation_batch_attempted_ids
            )
            delivery_started = time.perf_counter()
            delivery_recorded = False
            try:
                batch_error = aggregation_batch_errors.get(
                    cloud_event.event_id
                )
                if batch_error is not None:
                    raise batch_error
                if cloud_event.event_id in batched_aggregation_responses:
                    response = batched_aggregation_responses[cloud_event.event_id]
                    delivery_elapsed_ms = aggregation_batch_elapsed_ms[
                        cloud_event.event_id
                    ]
                else:
                    response = self.cloud.aggregate(cloud_event)
                    delivery_elapsed_ms = (
                        time.perf_counter() - delivery_started
                    ) * 1000.0
                transport = self._response_transport(response)
                if not batch_attempted:
                    deliveries.append(
                        {
                            "operation": "aggregate",
                            "event_count": 1,
                            "success": True,
                            "http_round_trip_ms": float(
                                transport.get(
                                    "http_round_trip_ms", delivery_elapsed_ms
                                )
                            ),
                            "request_bytes": int(
                                transport.get("request_bytes", 0)
                            ),
                            "response_bytes": int(
                                transport.get("response_bytes", 0)
                            ),
                        }
                    )
                delivery_recorded = True
                self.performance_store.record(
                    cloud_event.scene,
                    self._pending_evidence_level(stored_event),
                    self._pending_network_snapshot(stored_event),
                    True,
                    float(transport.get("http_round_trip_ms", delivery_elapsed_ms)),
                    int(transport.get("request_bytes", 0)),
                    int(transport.get("response_bytes", 0)),
                )
                accepted_at_ms = self._cloud_accepted_at_ms(response)
                if accepted_at_ms is not None and hasattr(
                    self.review_tracker, "received"
                ):
                    self.review_tracker.received(
                        [cloud_event.event_id], received_at_ms=accepted_at_ms
                    )
                coordination = response.get("coordination")
                decision = self._aggregation_decision(
                    response, cloud_event.event_id
                )
                evidence_complete = self._aggregation_evidence_complete(response)
                if (
                    decision is None
                    or not isinstance(coordination, dict)
                    or not evidence_complete
                ):
                    if is_reconciliation:
                        reconciliation_waiting_ids.append(cloud_event.event_id)
                        if decision is not None:
                            coordination_results.append(response)
                    elif decision is not None:
                        partial_waiting_ids.append(cloud_event.event_id)
                        partial_decisions[cloud_event.event_id] = decision
                        partial_results += 1
                        if hasattr(self.review_tracker, "record_partial"):
                            self.review_tracker.record_partial(
                                cloud_event.event_id, decision
                            )
                        coordination_results.append(response)
                    elif not is_reconciliation:
                        waiting_ids.append(cloud_event.event_id)
                    waiting_reason = (
                        "partial aggregation is waiting for all expected members"
                        if decision is not None
                        else "aggregation is waiting for peer summaries"
                    )
                    if is_reconciliation:
                        # Its lifecycle is already terminal at partial/local
                        # timeout.  Keep that visible until a richer, complete
                        # revision arrives; only the bounded poll queue moves.
                        pass
                    elif hasattr(self.review_tracker, "waiting"):
                        self.review_tracker.waiting(
                            [cloud_event.event_id], waiting_reason
                        )
                    else:
                        self.review_tracker.retry(
                            [cloud_event.event_id], waiting_reason
                        )
                    continue
                decision = _with_action_authorization(
                    replace(decision, route="cloud_async", status="final"),
                    cloud_confirmed=self._aggregation_globally_confirmed(response),
                )
                feedback = self._record_replayed_feedback(
                    [stored_event], [cloud_event], coordination
                )
                for name in (
                    "local_records",
                    "cloud_submissions",
                    "legacy_events_skipped",
                ):
                    feedback_totals[name] += int(feedback[name])
                feedback_totals["cloud_submission_errors"].extend(
                    feedback["cloud_submission_errors"]
                )
                try:
                    review_completions.append(
                        self.review_tracker.complete(
                            cloud_event.event_id,
                            decision,
                            "reconciliation" if is_reconciliation else "replay",
                        )
                    )
                except KeyError:
                    pass
                if is_reconciliation:
                    reconciliation_completed_ids.append(cloud_event.event_id)
                else:
                    completed_ids.append(cloud_event.event_id)
                coordination_results.append(response)
            except Exception as exc:  # noqa: BLE001
                error = "{}: {}".format(type(exc).__name__, exc)
                if not delivery_recorded and not batch_attempted:
                    deliveries.append(
                        {
                            "operation": "aggregate",
                            "event_count": 1,
                            "success": False,
                            "http_round_trip_ms": round(
                                (time.perf_counter() - delivery_started) * 1000.0,
                                6,
                            ),
                            "request_bytes": 0,
                            "response_bytes": 0,
                        }
                    )
                    self.performance_store.record(
                        cloud_event.scene,
                        self._pending_evidence_level(stored_event),
                        self._pending_network_snapshot(stored_event),
                        False,
                        (time.perf_counter() - delivery_started) * 1000.0,
                        0,
                        0,
                    )
                if batch_attempted:
                    item_transport = aggregation_batch_transport.get(
                        cloud_event.event_id, {}
                    )
                    self.performance_store.record(
                        cloud_event.scene,
                        self._pending_evidence_level(stored_event),
                        self._pending_network_snapshot(stored_event),
                        False,
                        aggregation_batch_elapsed_ms.get(
                            cloud_event.event_id, 0.0
                        ),
                        int(item_transport.get("request_bytes", 0)),
                        int(item_transport.get("response_bytes", 0)),
                    )
                if is_reconciliation:
                    reconciliation_retry_ids.append(cloud_event.event_id)
                else:
                    retry_ids.append(cloud_event.event_id)
                errors.append(
                    {"event_id": cloud_event.event_id, "error": error}
                )
                if not is_reconciliation:
                    self.review_tracker.retry([cloud_event.event_id], error)

        if regular_items:
            regular_stored = [item[0] for item in regular_items]
            regular_cloud = [item[1] for item in regular_items]
            regular_ids = [event.event_id for event in regular_cloud]
            delivery_started = time.perf_counter()
            delivery_recorded = False
            try:
                coordination = self.cloud.coordinate(regular_cloud)
                delivery_elapsed_ms = (
                    time.perf_counter() - delivery_started
                ) * 1000.0
                transport = self._response_transport(coordination)
                deliveries.append(
                    {
                        "operation": "coordinate",
                        "event_count": len(regular_cloud),
                        "success": True,
                        "http_round_trip_ms": float(
                            transport.get("http_round_trip_ms", delivery_elapsed_ms)
                        ),
                        "request_bytes": int(transport.get("request_bytes", 0)),
                        "response_bytes": int(transport.get("response_bytes", 0)),
                    }
                )
                delivery_recorded = True
                item_count = max(1, len(regular_items))
                for stored_event, cloud_event in regular_items:
                    self.performance_store.record(
                        cloud_event.scene,
                        self._pending_evidence_level(stored_event),
                        self._pending_network_snapshot(stored_event),
                        True,
                        float(
                            transport.get(
                                "http_round_trip_ms", delivery_elapsed_ms
                            )
                        ),
                        int(transport.get("request_bytes", 0)) // item_count,
                        int(transport.get("response_bytes", 0)) // item_count,
                    )
                accepted_at_ms = self._cloud_accepted_at_ms(coordination)
                if accepted_at_ms is not None and hasattr(
                    self.review_tracker, "received"
                ):
                    self.review_tracker.received(
                        regular_ids, received_at_ms=accepted_at_ms
                    )
                feedback = self._record_replayed_feedback(
                    regular_stored, regular_cloud, coordination
                )
                for name in (
                    "local_records",
                    "cloud_submissions",
                    "legacy_events_skipped",
                ):
                    feedback_totals[name] += int(feedback[name])
                feedback_totals["cloud_submission_errors"].extend(
                    feedback["cloud_submission_errors"]
                )
                for cloud_event in regular_cloud:
                    decision = self._coordination_decision(
                        coordination, cloud_event.event_id
                    )
                    if decision is None:
                        raise ValueError(
                            "cloud coordination omitted decision for {}".format(
                                cloud_event.event_id
                            )
                        )
                    try:
                        review_completions.append(
                            self.review_tracker.complete(
                                cloud_event.event_id,
                                replace(
                                    decision,
                                    route="cloud_async",
                                    status="final",
                                ),
                                "replay",
                            )
                        )
                    except KeyError:
                        pass
                completed_ids.extend(regular_ids)
                coordination_results.append(coordination)
            except Exception as exc:  # noqa: BLE001
                error = "{}: {}".format(type(exc).__name__, exc)
                if not delivery_recorded:
                    deliveries.append(
                        {
                            "operation": "coordinate",
                            "event_count": len(regular_cloud),
                            "success": False,
                            "http_round_trip_ms": round(
                                (time.perf_counter() - delivery_started) * 1000.0,
                                6,
                            ),
                            "request_bytes": 0,
                            "response_bytes": 0,
                        }
                    )
                    for stored_event, cloud_event in regular_items:
                        self.performance_store.record(
                            cloud_event.scene,
                            self._pending_evidence_level(stored_event),
                            self._pending_network_snapshot(stored_event),
                            False,
                            (time.perf_counter() - delivery_started) * 1000.0,
                            0,
                            0,
                        )
                retry_ids.extend(regular_ids)
                errors.extend(
                    {"event_id": event_id, "error": error}
                    for event_id in regular_ids
                )
                self.review_tracker.retry(regular_ids, error)

        partial_expired_ids: List[str] = []
        local_timeout_ids: List[str] = []
        reconciliation_expired_ids: List[str] = list(
            reconciliation_swept_expired_ids
        )
        if leases is not None:
            if completed_ids:
                self.review_store.acknowledge(completed_ids)
            if reconciliation_completed_ids and hasattr(
                self.review_store, "acknowledge_reconciliation"
            ):
                self.review_store.acknowledge_reconciliation(
                    reconciliation_completed_ids
                )
            if retry_ids:
                self.review_store.release(
                    retry_ids,
                    errors[0]["error"] if errors else "aggregation waiting",
                    max_backoff_seconds,
                )
            if waiting_ids:
                if hasattr(self.review_store, "defer_aggregation_wait"):
                    local_timeout_ids = self.review_store.defer_aggregation_wait(
                        waiting_ids,
                        waiting_poll_seconds,
                        aggregation_max_wait_seconds,
                        reconciliation_poll_seconds,
                        reconciliation_max_wait_seconds,
                    )
                elif hasattr(self.review_store, "defer_waiting"):
                    self.review_store.defer_waiting(waiting_ids, waiting_poll_seconds)
                else:
                    self.review_store.release(
                        waiting_ids,
                        "aggregation waiting",
                        waiting_poll_seconds,
                    )
            if partial_waiting_ids:
                if hasattr(self.review_store, "defer_aggregation_wait"):
                    partial_expired_ids = self.review_store.defer_aggregation_wait(
                        partial_waiting_ids,
                        partial_poll_seconds,
                        aggregation_max_wait_seconds,
                        reconciliation_poll_seconds,
                        reconciliation_max_wait_seconds,
                    )
                elif hasattr(self.review_store, "defer_waiting"):
                    self.review_store.defer_waiting(
                        partial_waiting_ids, partial_poll_seconds
                    )
                else:
                    self.review_store.release(
                        partial_waiting_ids,
                        "partial aggregation waiting",
                        partial_poll_seconds,
                    )
            if reconciliation_waiting_ids and hasattr(
                self.review_store, "defer_reconciliation"
            ):
                reconciliation_expired_ids.extend(
                    self.review_store.defer_reconciliation(
                        reconciliation_waiting_ids,
                        reconciliation_poll_seconds,
                    )
                )
            if reconciliation_retry_ids and hasattr(
                self.review_store, "release_reconciliation"
            ):
                reconciliation_expired_ids.extend(
                    self.review_store.release_reconciliation(
                        reconciliation_retry_ids,
                        errors[0]["error"] if errors else "reconciliation failed",
                        max_backoff_seconds,
                    )
                )
        elif not retry_ids and not waiting_ids and not partial_waiting_ids:
            self.review_store.clear()

        for event_id in partial_expired_ids:
            partial = partial_decisions[event_id]
            metadata = dict(partial.metadata)
            metadata["aggregation_wait_expired"] = {
                "max_wait_seconds": float(aggregation_max_wait_seconds),
                "outcome": "partial_final",
                "evidence_complete": False,
            }
            try:
                review_completions.append(
                    self._complete_review(
                        self.review_tracker,
                        event_id,
                        replace(
                            partial,
                            route="cloud_async",
                            status="provisional",
                            metadata=metadata,
                        ),
                        "partial_timeout",
                        "partial_final",
                    )
                )
            except KeyError:
                pass

        for event_id in local_timeout_ids:
            try:
                review = self.review_tracker.get(event_id)
                local = DecisionEnvelope.from_dict(review["local_decision"])
                metadata = dict(local.metadata)
                metadata["aggregation_wait_expired"] = {
                    "max_wait_seconds": float(aggregation_max_wait_seconds),
                    "outcome": "local_only_timeout",
                    "evidence_complete": False,
                }
                review_completions.append(
                    self._complete_review(
                        self.review_tracker,
                        event_id,
                        replace(
                            local,
                            route="local_autonomy",
                            status="provisional",
                            metadata=metadata,
                        ),
                        "aggregation_timeout",
                        "local_only_timeout",
                    )
                )
            except KeyError:
                pass

        expired_ids = set(partial_expired_ids + local_timeout_ids)
        active_waiting_ids = [
            event_id
            for event_id in waiting_ids + partial_waiting_ids
            if event_id not in expired_ids
        ]
        active_partial_ids = [
            event_id
            for event_id in partial_waiting_ids
            if event_id not in expired_ids
        ]
        reconciliation_expired_set = set(reconciliation_expired_ids)
        active_reconciliation_ids = [
            event_id
            for event_id in reconciliation_waiting_ids
            + reconciliation_retry_ids
            if event_id not in reconciliation_expired_set
        ]

        remaining = (
            self.review_store.work_count()
            if hasattr(self.review_store, "work_count")
            else self.review_store.count()
        )

        return {
            "attempted": len(pending),
            "completed": len(completed_ids) + len(reconciliation_completed_ids),
            "terminal": len(completed_ids) + len(expired_ids),
            # v1 compatibility: waiting meant every outstanding item,
            # including delivery failures awaiting retry.
            "waiting": len(active_waiting_ids)
            + len(retry_ids)
            + len(active_reconciliation_ids),
            "aggregation_waiting": len(active_waiting_ids)
            + len(active_reconciliation_ids),
            "partial_waiting": len(active_partial_ids),
            "retrying": len(retry_ids) + len(reconciliation_retry_ids),
            "reconciliation_waiting": len(active_reconciliation_ids),
            "reconciliation_completed": len(reconciliation_completed_ids),
            "reconciliation_expired": len(reconciliation_expired_set),
            "partial": partial_results,
            "aggregation_expired": len(expired_ids),
            "partial_expired": len(partial_expired_ids),
            "local_timeout_expired": len(local_timeout_ids),
            "coordination": (
                coordination_results[0]
                if len(coordination_results) == 1
                else coordination_results
            ),
            "feedback": feedback_totals,
            "errors": errors,
            "deliveries": deliveries,
            "review_completions": review_completions,
            "review_lifecycle": self.review_tracker.snapshot(),
            "remaining": remaining,
        }
