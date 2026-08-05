"""Targeted invariants for durable, non-blocking multi-edge summary delivery."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from typing import Any, Dict, Optional

from cloud_edge_framework.contracts import DecisionEnvelope, SemanticEvent, build_decision
from cloud_edge_framework.cloud_llm import CloudLLMReviewer
from cloud_edge_framework.event_envelope import SceneEventEnvelope
from cloud_edge_framework.handoff import DurableOutboxHandoff
from cloud_edge_framework.metrics import FrameworkMetrics
from cloud_edge_framework.networking import StaticNetworkMonitor
from cloud_edge_framework.plugins.base import ScenePlugin
from cloud_edge_framework.registry import SceneRegistry
from cloud_edge_framework.reliability import (
    IdempotencyConflictError,
    SQLiteOutbox,
)
from cloud_edge_framework.replay import OutboxReplayWorker
from cloud_edge_framework.review_tracking import ReviewLifecycleStore
from cloud_edge_framework.runtime import CloudRuntime, EdgeRuntime
from cloud_edge_framework.scheduling import CollaborationScheduler, NetworkSnapshot
from cloud_edge_framework.service_config import ReplayConfig


class _AggregationPlugin(ScenePlugin):
    scene = "async_summary_fixture"
    aliases = ()
    event_types = ("org.example.async-summary.v1",)
    data_schema_id = "https://example.org/schemas/async-summary-v1.json"
    policy_version = "async-summary-1.0.0"

    def __init__(
        self,
        aggregation_enabled: bool = True,
        requires_cloud_confirmation: bool = True,
        risk_level: str = "high",
        deadline_ms: float = 500.0,
    ) -> None:
        self.aggregation_enabled = bool(aggregation_enabled)
        self.requires_cloud_confirmation = bool(requires_cloud_confirmation)
        self.risk_level = str(risk_level)
        self.deadline_ms = float(deadline_ms)

    def payload_schema(self) -> Dict[str, Any]:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": self.data_schema_id,
            "type": "object",
            "required": ["sample_id", "member", "value"],
            "properties": {
                "sample_id": {"type": "string", "minLength": 1},
                "member": {"type": "string", "minLength": 1},
                "value": {"type": "number"},
            },
            "additionalProperties": False,
        }

    def normalize(self, envelope: SceneEventEnvelope) -> SemanticEvent:
        self.validate_envelope(envelope)
        data = dict(envelope.data)
        return SemanticEvent.from_dict(
            {
                "schema_version": "1.0",
                "event_id": envelope.event_id,
                "scene": self.scene,
                "task": "multi_edge_summary",
                "edge_id": envelope.edge_id,
                "occurred_at_ms": envelope.occurred_at_ms,
                "scope": {
                    "entity_id": data["sample_id"],
                    "subsystem": "fixture",
                    "state_variable": "fixture_state",
                    "region_id": "fixture_region",
                    "shared_resources": ["fixture_resource"],
                    "correlation_keys": ["sample:" + data["sample_id"]],
                    "window_start_ms": envelope.occurred_at_ms - 100,
                    "window_end_ms": envelope.occurred_at_ms,
                },
                "prediction": {
                    "label": self.risk_level,
                    "confidence": 0.92,
                    "probabilities": {self.risk_level: 0.92},
                    "values": {"value": data["value"]},
                },
                "risk": {
                    "level": self.risk_level,
                    "score": 0.20 if self.risk_level == "low" else 0.90,
                },
                "uncertainty": {
                    "confidence": 0.91,
                    "calibrated": True,
                    "prediction_set": [self.risk_level],
                    "method": "fixture",
                },
                "timing": {
                    "deadline_ms": self.deadline_ms,
                    "preprocessing_ms": 1.0,
                    "edge_inference_ms": 2.0,
                },
                "evidence": [
                    {
                        "evidence_id": envelope.event_id + "_summary",
                        "level": "summary",
                        "modality": "fixture",
                        "encoding": "json",
                        "inline": {"value": data["value"]},
                        "size_bytes": 16,
                        "content_type": "application/json",
                    },
                    {
                        "evidence_id": envelope.event_id + "_feature",
                        "level": "feature",
                        "modality": "fixture",
                        "encoding": "json",
                        "inline": [data["value"]],
                        "size_bytes": 8,
                        "content_type": "application/json",
                    },
                ],
                "candidate_actions": [
                    {
                        "action_type": "hold_fixture",
                        "target_ids": [data["sample_id"]],
                        "resource_ids": ["fixture_resource"],
                        "parameters": {
                            "min_risk_level": self.risk_level,
                            "requires_cloud_confirmation": (
                                self.requires_cloud_confirmation
                            ),
                        },
                        "reason": "fixture action requires complete aggregation",
                        "priority": 90,
                    }
                ],
                "model": {"name": "fixture_model", "version": "1"},
                "scene_payload": data,
                "metadata": {
                    "transport_include_scene_payload": True,
                    "aggregation": {
                        "key": "sample:" + data["sample_id"],
                        "member": data["member"],
                        "expected_members": ["edge-a", "edge-b"],
                        "minimum_members": 2,
                        "timeout_ms": 100,
                    },
                },
            }
        )

    def aggregation_spec(
        self, event: SemanticEvent
    ) -> Optional[Dict[str, Any]]:
        if not self.aggregation_enabled:
            return None
        return super().aggregation_spec(event)

    def edge_decide(self, event: SemanticEvent):
        return self.decision_from_candidates(event, "fixture_edge", 0.92)

    def cloud_decide(self, event: SemanticEvent):
        return build_decision(
            event=event,
            decision="monitor",
            actions=[],
            confidence=0.98,
            reason="fixture cloud decision",
            source="fixture_cloud",
            policy_version=self.policy_version,
        )


class _ForbiddenSlowCloud:
    def __init__(self) -> None:
        self.aggregate_calls = 0
        self.coordinate_calls = 0

    def aggregate(self, event: SemanticEvent) -> Dict[str, Any]:
        del event
        self.aggregate_calls += 1
        time.sleep(0.05)
        raise AssertionError("process() must not call the slow cloud path")

    def coordinate(self, events):
        del events
        self.coordinate_calls += 1
        time.sleep(0.05)
        raise AssertionError("process() must not call the slow cloud path")

    def decide(self, event: SemanticEvent):
        del event
        raise AssertionError("process() must not call the slow cloud path")


class _RejectingHandoff:
    def __init__(self) -> None:
        self.timeout_seconds = None

    def submit(self, event, timeout_seconds: float):
        del event
        self.timeout_seconds = float(timeout_seconds)
        raise OSError("journal unavailable")

    def snapshot(self):
        return {"pending": 0}


class _CoalescingWorkerProbe(OutboxReplayWorker):
    def __init__(self, config: ReplayConfig) -> None:
        super().__init__(
            manager=None,
            outbox=None,
            network_monitor=None,
            config=config,
            metrics=None,
        )
        self.run_started_at = None

    def _work_count(self) -> int:
        return 1

    def run_once(self) -> Dict[str, Any]:
        self.run_started_at = time.monotonic()
        self._stop_event.set()
        return {"status": "completed", "attempted": 1, "completed": 1}


class _ToggleAggregationCloud:
    def __init__(self, complete: bool = False, partial: bool = False) -> None:
        self.complete = bool(complete)
        self.partial = bool(partial)
        self.aggregate_calls = 0
        self.coordinate_calls = 0
        self.accepted_at_ms = []

    def aggregate(self, event: SemanticEvent) -> Dict[str, Any]:
        self.aggregate_calls += 1
        accepted_at_ms = int(time.time() * 1000)
        self.accepted_at_ms.append(accepted_at_ms)
        if not self.complete and not self.partial:
            return {
                "aggregation": {
                    "group_id": "sample:" + event.scope.entity_id,
                    "state": "waiting",
                    "completion_reason": "",
                    "received_members": [event.edge_id],
                    "missing_members": ["edge-b"],
                    "evidence_complete": False,
                    "global_confirmation": False,
                    "finality": "pending",
                    "result_revision": 0,
                },
                "coordination": None,
                "cloud_accepted_at_ms": accepted_at_ms,
                "transport": {
                    "request_bytes": 100,
                    "response_bytes": 80,
                    "http_round_trip_ms": 1.0,
                },
            }
        decision = build_decision(
            event=event,
            decision="monitor",
            actions=[],
            confidence=0.99,
            reason="all expected summaries were fused",
            source="fixture_cloud",
            policy_version=_AggregationPlugin.policy_version,
        )
        if self.partial:
            return {
                "aggregation": {
                    "group_id": "sample:" + event.scope.entity_id,
                    "state": "completed",
                    "completion_reason": "timeout_with_partial_members",
                    "received_members": [event.edge_id],
                    "missing_members": ["edge-b"],
                    "evidence_complete": False,
                    "global_confirmation": False,
                    "finality": "partial_final",
                    "result_revision": 1,
                },
                "coordination": {
                    "decisions": [decision.to_dict()],
                    "globally_consistent": False,
                    "initial_conflict_count": 0,
                    "residual_conflict_count": 0,
                },
                "cloud_accepted_at_ms": accepted_at_ms,
                "transport": {
                    "request_bytes": 100,
                    "response_bytes": 120,
                    "http_round_trip_ms": 1.0,
                },
            }
        return {
            "aggregation": {
                "group_id": "sample:" + event.scope.entity_id,
                "state": "completed",
                "completion_reason": "all_expected_members",
                "received_members": ["edge-a", "edge-b"],
                "missing_members": [],
                "evidence_complete": True,
                "global_confirmation": True,
                "finality": "final",
                "result_revision": 1,
            },
            "coordination": {
                "decisions": [decision.to_dict()],
                "globally_consistent": True,
                "initial_conflict_count": 0,
                "residual_conflict_count": 0,
            },
            "cloud_accepted_at_ms": accepted_at_ms,
            "transport": {
                "request_bytes": 100,
                "response_bytes": 120,
                "http_round_trip_ms": 1.0,
            },
        }

    def coordinate(self, events):
        del events
        self.coordinate_calls += 1
        raise AssertionError("frozen aggregate delivery must not become coordinate")


class _ResultPollingCloud(_ToggleAggregationCloud):
    def __init__(self, aggregate_delay_seconds: float = 0.0) -> None:
        super().__init__(complete=False)
        self.aggregate_delay_seconds = float(aggregate_delay_seconds)
        self.result_calls = 0

    def aggregate(self, event: SemanticEvent) -> Dict[str, Any]:
        if self.aggregate_delay_seconds > 0.0:
            time.sleep(self.aggregate_delay_seconds)
        return super().aggregate(event)

    def aggregation_results_batch(self, events, event_group_ids):
        self.result_calls += 1
        items = []
        for event in events:
            completed = _ToggleAggregationCloud(complete=True).aggregate(event)
            completed["event_id"] = event.event_id
            completed["group_id"] = event_group_ids[event.event_id]
            completed["aggregation"]["group_id"] = event_group_ids[event.event_id]
            items.append(completed)
        return {
            "items": items,
            "transport": {
                "request_bytes": 32,
                "response_bytes": 96,
                "http_round_trip_ms": 1.0,
            },
        }


class _DeadlineAwareDecisionCloud:
    supports_request_timeout = True

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = float(delay_seconds)
        self.timeout_seconds = None
        self.calls = 0

    def decide(
        self,
        event: SemanticEvent,
        timeout_seconds: Optional[float] = None,
    ) -> DecisionEnvelope:
        self.calls += 1
        self.timeout_seconds = timeout_seconds
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return replace(
            build_decision(
                event=event,
                decision="monitor",
                actions=[],
                confidence=0.99,
                reason="bounded non-aggregation cloud review",
                source="fixture_cloud",
                policy_version=_AggregationPlugin.policy_version,
            ),
            route="cloud_sync",
            status="final",
        )


class _LateMemberAggregationCloud:
    """Small stateful cloud fixture: B makes A's group authoritative."""

    def __init__(self, partial_before_full: bool = False) -> None:
        self.partial_before_full = bool(partial_before_full)
        self.events: Dict[str, SemanticEvent] = {}
        self.aggregate_calls = 0

    @staticmethod
    def _decision(event: SemanticEvent) -> DecisionEnvelope:
        return build_decision(
            event=event,
            decision="monitor",
            actions=[],
            confidence=0.99,
            reason="late member completed the expected aggregation",
            source="fixture_cloud",
            policy_version=_AggregationPlugin.policy_version,
        )

    def aggregate(self, event: SemanticEvent) -> Dict[str, Any]:
        self.aggregate_calls += 1
        self.events[event.edge_id] = event
        accepted_at_ms = int(time.time() * 1000)
        complete = {"edge-a", "edge-b"}.issubset(self.events)
        if complete:
            decisions = [
                self._decision(self.events[member]).to_dict()
                for member in ("edge-a", "edge-b")
            ]
            return {
                "aggregation": {
                    "group_id": "aggregation-late-member",
                    "state": "completed",
                    "completion_reason": "all_expected_members",
                    "received_members": ["edge-a", "edge-b"],
                    "missing_members": [],
                    "evidence_complete": True,
                    "global_confirmation": True,
                    "finality": "final",
                    "result_revision": 2,
                },
                "coordination": {
                    "decisions": decisions,
                    "globally_consistent": True,
                    "initial_conflict_count": 0,
                    "residual_conflict_count": 0,
                },
                "cloud_accepted_at_ms": accepted_at_ms,
            }
        if self.partial_before_full:
            return {
                "aggregation": {
                    "group_id": "aggregation-late-member",
                    "state": "completed",
                    "completion_reason": "timeout_with_partial_members",
                    "received_members": ["edge-a"],
                    "missing_members": ["edge-b"],
                    "evidence_complete": False,
                    "global_confirmation": False,
                    "finality": "partial_final",
                    "result_revision": 1,
                },
                "coordination": {
                    "decisions": [self._decision(event).to_dict()],
                    "globally_consistent": False,
                    "initial_conflict_count": 0,
                    "residual_conflict_count": 0,
                },
                "cloud_accepted_at_ms": accepted_at_ms,
            }
        return {
            "aggregation": {
                "group_id": "aggregation-late-member",
                "state": "waiting",
                "completion_reason": "",
                "received_members": ["edge-a"],
                "missing_members": ["edge-b"],
                "evidence_complete": False,
                "global_confirmation": False,
                "finality": "pending",
                "result_revision": 0,
            },
            "coordination": None,
            "cloud_accepted_at_ms": accepted_at_ms,
        }

    def coordinate(self, events):
        del events
        raise AssertionError("aggregation reconciliation must remain aggregate")


class _BatchAggregationCloud:
    def __init__(self) -> None:
        self.aggregate_calls = 0
        self.aggregate_batch_calls = 0
        self.batch_event_ids = []
        self.wait_seconds = None

    @staticmethod
    def _decision(event: SemanticEvent) -> DecisionEnvelope:
        return build_decision(
            event=event,
            decision="monitor",
            actions=[],
            confidence=0.99,
            reason="one batch completed all expected summaries",
            source="fixture_cloud",
            policy_version=_AggregationPlugin.policy_version,
        )

    def aggregate(self, event: SemanticEvent) -> Dict[str, Any]:
        del event
        self.aggregate_calls += 1
        raise AssertionError("batch-capable clients must not use per-event aggregate")

    def aggregate_batch(
        self,
        events,
        wait_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        self.aggregate_batch_calls += 1
        self.batch_event_ids = [event.event_id for event in events]
        self.wait_seconds = float(wait_seconds)
        decisions = [self._decision(event).to_dict() for event in events]
        coordination = {
            "decisions": decisions,
            "globally_consistent": True,
            "initial_conflict_count": 0,
            "residual_conflict_count": 0,
        }
        accepted_at_ms = int(time.time() * 1000)
        aggregation = {
            "group_id": "aggregation-batch",
            "state": "completed",
            "completion_reason": "all_expected_members",
            "received_members": ["edge-a", "edge-b"],
            "missing_members": [],
            "evidence_complete": True,
            "global_confirmation": True,
            "finality": "final",
            "result_revision": 1,
        }
        return {
            "items": [
                {
                    "event_id": event.event_id,
                    "aggregation": dict(aggregation),
                    "coordination": coordination,
                    "cloud_accepted_at_ms": accepted_at_ms,
                }
                for event in events
            ],
            "transport": {
                "request_bytes": 240,
                "response_bytes": 320,
                "http_round_trip_ms": 5.0,
            },
        }

    def coordinate(self, events):
        del events
        raise AssertionError("aggregation batch must not become coordinate")


class _ResultChannelCloud:
    def __init__(self) -> None:
        self.submission_calls = 0
        self.result_calls = 0
        self.events: Dict[str, SemanticEvent] = {}

    @staticmethod
    def _decision(event: SemanticEvent) -> DecisionEnvelope:
        return build_decision(
            event=event,
            decision="monitor",
            actions=[],
            confidence=0.99,
            reason="result-only poll observed the completed group",
            source="fixture_cloud",
            policy_version=_AggregationPlugin.policy_version,
        )

    def aggregate_batch(self, events, wait_seconds: float = 0.0):
        self.submission_calls += 1
        self.assert_zero_wait = float(wait_seconds)
        self.events.update({event.event_id: event for event in events})
        return {
            "items": [
                {
                    "event_id": event.event_id,
                    "group_id": "aggregation-result-channel",
                    "aggregation": {
                        "group_id": "aggregation-result-channel",
                        "state": "waiting",
                        "completion_reason": "",
                        "received_members": [event.edge_id],
                        "missing_members": ["edge-b"],
                        "evidence_complete": False,
                        "global_confirmation": False,
                        "finality": "pending",
                        "result_revision": 0,
                    },
                    "coordination": None,
                    "cloud_accepted_at_ms": int(time.time() * 1000),
                }
                for event in events
            ],
            "transport": {
                "request_bytes": 100,
                "response_bytes": 80,
                "http_round_trip_ms": 1.0,
            },
        }

    def aggregation_results_batch(self, events, event_group_ids):
        self.result_calls += 1
        self.last_group_ids = dict(event_group_ids)
        decisions = [self._decision(event).to_dict() for event in events]
        coordination = {
            "decisions": decisions,
            "globally_consistent": True,
            "initial_conflict_count": 0,
            "residual_conflict_count": 0,
        }
        return {
            "items": [
                {
                    "event_id": event.event_id,
                    "group_id": event_group_ids[event.event_id],
                    "aggregation": {
                        "group_id": event_group_ids[event.event_id],
                        "state": "completed",
                        "completion_reason": "all_expected_members",
                        "received_members": ["edge-a", "edge-b"],
                        "missing_members": [],
                        "evidence_complete": True,
                        "global_confirmation": True,
                        "finality": "final",
                        "result_revision": 1,
                    },
                    "coordination": coordination,
                    "cloud_accepted_at_ms": int(time.time() * 1000),
                }
                for event in events
            ],
            "transport": {
                "request_bytes": 40,
                "response_bytes": 120,
                "http_round_trip_ms": 1.0,
            },
        }

    def aggregate(self, event):
        del event
        raise AssertionError("result polling must not resubmit a summary")


class _RuntimeSnapshot:
    def __init__(self, runtime: EdgeRuntime) -> None:
        self.runtime = runtime

    def require_edge(self) -> EdgeRuntime:
        return self.runtime


class _RuntimeManager:
    def __init__(self, runtime: EdgeRuntime) -> None:
        self.runtime = runtime

    @contextmanager
    def lease(self):
        yield _RuntimeSnapshot(self.runtime)


class _CrashAfterOutboxTracker(ReviewLifecycleStore):
    """Inject the exact crash window between Outbox and lifecycle persistence."""

    def queue(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected crash after durable Outbox append")


class _AlwaysAsyncScheduler(CollaborationScheduler):
    def schedule(self, *args, **kwargs):
        scheduled = super().schedule(*args, **kwargs)
        return replace(
            scheduled,
            route="cloud_async",
            reason="test fixture requests durable background delivery",
            cloud_requested=True,
            waits_for_cloud=False,
        )


def _payload(
    event_id: str = "summary-1",
    sample_id: str = "sample-1",
    member: str = "edge-a",
    value: float = 1.0,
) -> Dict[str, Any]:
    event_time = datetime.now(timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")
    return {
        "specversion": "1.0",
        "id": event_id,
        "source": "urn:edge:{}:fixture".format(member),
        "type": _AggregationPlugin.event_types[0],
        "scene": _AggregationPlugin.scene,
        "edgeid": member,
        "subject": sample_id,
        "time": event_time,
        "datacontenttype": "application/json",
        "dataschema": _AggregationPlugin.data_schema_id,
        "data": {
            "sample_id": sample_id,
            "member": member,
            "value": value,
        },
    }


def _network() -> NetworkSnapshot:
    return NetworkSnapshot(
        available=True,
        rtt_ms=1.0,
        jitter_ms=0.0,
        loss_rate=0.0,
        cloud_queue_ms=1.0,
        cloud_compute_ms=1.0,
        uplink_mbps=100.0,
        downlink_mbps=100.0,
    )


class AsyncSummaryDeliveryTest(unittest.TestCase):
    def test_model_batch_keeps_sample_coordination_isolated(self) -> None:
        registry = SceneRegistry([_AggregationPlugin(True)])
        try:
            plugin = registry.get(_AggregationPlugin.scene)
            groups = []
            for sample_index in (1, 2):
                groups.append(
                    [
                        plugin.normalize(
                            SceneEventEnvelope.from_dict(
                                _payload(
                                    event_id="sample-{}-{}".format(
                                        sample_index, member
                                    ),
                                    sample_id="sample-{}".format(sample_index),
                                    member=member,
                                )
                            )
                        )
                        for member in ("edge-a", "edge-b")
                    ]
                )

            results = CloudRuntime(registry).coordinate_groups(groups)

            self.assertEqual(len(results), 2)
            for index, result in enumerate(results, start=1):
                self.assertEqual(result["event_count"], 2)
                self.assertEqual(result["cloud_batch_group_count"], 2)
                expected_ids = {
                    "sample-{}-edge-a".format(index),
                    "sample-{}-edge-b".format(index),
                }
                observed_ids = {
                    event_id
                    for decision in result["decisions"]
                    for event_id in decision["event_ids"]
                }
                self.assertEqual(observed_ids, expected_ids)
        finally:
            registry.close()

    def test_worker_coalesces_only_the_immediate_wakeup_batch(self) -> None:
        worker = _CoalescingWorkerProbe(
            ReplayConfig(
                interval_seconds=1.0,
                batch_size=8,
                lease_seconds=1.0,
                max_backoff_seconds=1.0,
                batch_coalesce_seconds=0.03,
            )
        )
        started = time.monotonic()
        worker.start()
        worker.notify()
        worker._thread.join(timeout=1.0)

        self.assertIsNotNone(worker.run_started_at)
        self.assertFalse(worker._thread.is_alive())
        self.assertGreaterEqual(worker.run_started_at - started, 0.02)

    def test_ready_members_use_one_batch_request_and_complete_together(self) -> None:
        with tempfile.TemporaryDirectory(prefix="async-summary-batch-") as directory:
            root = Path(directory)
            registry = SceneRegistry([_AggregationPlugin(True)])
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            cloud = _BatchAggregationCloud()
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=cloud,
                    scheduler=_AlwaysAsyncScheduler(),
                    review_store=outbox,
                    review_tracker=tracker,
                )
                runtime.process(
                    _payload("summary-a", "sample-1", "edge-a", 1.0),
                    network=_network(),
                )
                runtime.process(
                    _payload("summary-b", "sample-1", "edge-b", 2.0),
                    network=_network(),
                )

                flushed = runtime.flush_pending(
                    batch_size=8,
                    aggregation_batch_wait_seconds=0.12,
                )

                self.assertEqual(flushed["attempted"], 2)
                self.assertEqual(flushed["completed"], 2)
                self.assertEqual(flushed["remaining"], 0)
                self.assertEqual(flushed["errors"], [])
                self.assertEqual(cloud.aggregate_batch_calls, 1)
                self.assertEqual(cloud.aggregate_calls, 0)
                self.assertEqual(
                    cloud.batch_event_ids, ["summary-a", "summary-b"]
                )
                self.assertEqual(cloud.wait_seconds, 0.0)
                self.assertEqual(
                    {item["operation"] for item in flushed["deliveries"]},
                    {"aggregate_batch"},
                )
                self.assertEqual(len(flushed["deliveries"]), 1)
                self.assertEqual(flushed["deliveries"][0]["event_count"], 2)
                self.assertEqual(
                    sum(item["request_bytes"] for item in flushed["deliveries"]),
                    240,
                )
                self.assertEqual(
                    sum(item["response_bytes"] for item in flushed["deliveries"]),
                    320,
                )
                for event_id in ("summary-a", "summary-b"):
                    review = tracker.get(event_id)
                    self.assertEqual(review["state"], "completed")
                    self.assertEqual(
                        review["completion_stage"], "lightweight_final"
                    )
                    self.assertEqual(
                        review["final_decision"]["status"], "final"
                    )
            finally:
                tracker.close()
                registry.close()

    def test_waiting_result_poll_does_not_resubmit_summary_payload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="async-result-channel-") as directory:
            root = Path(directory)
            registry = SceneRegistry([_AggregationPlugin(True)])
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            cloud = _ResultChannelCloud()
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=cloud,
                    scheduler=_AlwaysAsyncScheduler(),
                    review_store=outbox,
                    review_tracker=tracker,
                )
                runtime.process(_payload(), network=_network())

                submitted = runtime.flush_pending(
                    waiting_poll_seconds=0.001
                )
                self.assertEqual(submitted["waiting"], 1)
                self.assertEqual(cloud.submission_calls, 1)
                self.assertEqual(cloud.result_calls, 0)
                self.assertEqual(cloud.assert_zero_wait, 0.0)

                time.sleep(0.01)
                completed = runtime.flush_pending(
                    waiting_poll_seconds=0.001
                )
                self.assertEqual(completed["completed"], 1)
                self.assertEqual(completed["remaining"], 0)
                self.assertEqual(cloud.submission_calls, 1)
                self.assertEqual(cloud.result_calls, 1)
                self.assertEqual(
                    completed["deliveries"][0]["operation"],
                    "aggregate_results_batch",
                )
            finally:
                tracker.close()
                registry.close()

    def test_cloud_llm_prompt_keeps_v1_risk_key_and_adds_split_semantics(self) -> None:
        plugin = _AggregationPlugin(False)
        event = plugin.normalize(SceneEventEnvelope.from_dict(_payload()))
        event = replace(
            event,
            metadata={
                **event.metadata,
                "regional_state": {"level": "congested"},
                "operational_safety_risk": {"level": "low"},
                "model_uncertainty": {"score": 0.2},
                "escalation_expected_gain": {"score": 0.1},
            },
        )

        prompt = json.loads(
            CloudLLMReviewer._prompt(event, plugin.cloud_decide(event))
        )

        self.assertEqual(prompt["risk"], {"level": "high", "score": 0.9})
        self.assertEqual(prompt["risk_semantics"], "legacy_mixed")
        self.assertEqual(prompt["regional_state"]["level"], "congested")
        self.assertEqual(prompt["operational_safety_risk"]["level"], "low")
        self.assertEqual(prompt["model_uncertainty"]["score"], 0.2)
        self.assertEqual(prompt["escalation_expected_gain"]["score"], 0.1)

    def test_cloud_llm_eligibility_error_preserves_expert_baseline(self) -> None:
        registry = SceneRegistry([_AggregationPlugin(False)])
        try:
            event = registry.get("async_summary_fixture").normalize(
                SceneEventEnvelope.from_dict(_payload())
            )
            event = replace(
                event,
                metadata={
                    **event.metadata,
                    "cloud_llm_review_policy": "invalid-policy",
                },
            )
            runtime = CloudRuntime(
                registry,
                reviewer=CloudLLMReviewer(provider=object()),
            )

            decision = runtime.decide(event)

            self.assertEqual(decision.decision, "monitor")
            self.assertEqual(decision.status, "final")
            self.assertEqual(decision.metadata["cloud_llm_review_error_stage"], "eligibility")
            self.assertTrue(decision.metadata["cloud_llm_baseline_preserved"])
            self.assertIn(
                "cloud_llm_review_policy must be an object",
                decision.metadata["cloud_llm_review_error"],
            )
        finally:
            registry.close()

    def test_required_cloud_confirmation_submits_synchronously_then_falls_back(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="async-summary-process-") as directory:
            root = Path(directory)
            plugin = _AggregationPlugin()
            registry = SceneRegistry([plugin])
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            cloud = _ToggleAggregationCloud(complete=False)
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=cloud,
                    review_store=outbox,
                    review_tracker=tracker,
                )
                result = runtime.process(_payload(), network=_network())

                self.assertEqual(
                    result["data_plane"]["scheduler_selected_route"],
                    "cloud_sync",
                )
                self.assertTrue(
                    result["data_plane"]["scheduler_selected_wait"]
                )
                self.assertEqual(result["schedule"]["route"], "cloud_sync")
                self.assertTrue(result["schedule"]["waits_for_cloud"])
                self.assertEqual(result["local_decision"]["status"], "provisional")
                # This fixture emulates an older cloud client without the
                # result-only polling endpoint.  The edge still makes the
                # synchronous submission, then durably falls back to replay
                # because the peer summary is not complete yet.
                self.assertEqual(result["final_decision"]["route"], "cloud_async")
                self.assertEqual(result["final_decision"]["status"], "provisional")
                self.assertEqual(result["pending_review_count"], 1)
                self.assertEqual(outbox.count(), 1)
                self.assertEqual(cloud.aggregate_calls, 1)
                self.assertEqual(cloud.coordinate_calls, 0)
            finally:
                tracker.close()
                registry.close()

    def test_required_cloud_confirmation_polls_result_before_returning(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sync-summary-result-") as directory:
            root = Path(directory)
            registry = SceneRegistry([_AggregationPlugin()])
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            cloud = _ResultPollingCloud()
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=cloud,
                    review_store=outbox,
                    review_tracker=tracker,
                )
                result = runtime.process(_payload(), network=_network())

                self.assertEqual(result["schedule"]["route"], "cloud_sync")
                self.assertTrue(result["schedule"]["waits_for_cloud"])
                self.assertEqual(result["final_decision"]["status"], "final")
                self.assertTrue(
                    result["final_decision"]["metadata"]["aggregation"][
                        "global_confirmation"
                    ]
                )
                self.assertEqual(cloud.aggregate_calls, 1)
                self.assertEqual(cloud.result_calls, 1)
                self.assertEqual(outbox.count(), 0)
            finally:
                tracker.close()
                registry.close()

    def test_initial_cloud_round_trip_is_charged_to_sync_deadline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sync-summary-deadline-") as directory:
            root = Path(directory)
            registry = SceneRegistry([_AggregationPlugin(deadline_ms=30.0)])
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            cloud = _ResultPollingCloud(aggregate_delay_seconds=0.04)
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=cloud,
                    review_store=outbox,
                    review_tracker=tracker,
                )
                result = runtime.process(_payload(), network=_network())

                # The submission itself consumed the business budget.  The
                # edge must durably fall back instead of granting polling a new
                # post-submission deadline window.
                self.assertEqual(cloud.aggregate_calls, 1)
                self.assertEqual(cloud.result_calls, 0)
                self.assertEqual(result["final_decision"]["status"], "provisional")
                self.assertEqual(result["final_decision"]["route"], "cloud_async")
                self.assertEqual(outbox.count(), 1)
            finally:
                tracker.close()
                registry.close()

    def test_compact_sync_response_projects_review_and_preserves_metrics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="compact-sync-summary-") as directory:
            root = Path(directory)
            registry = SceneRegistry([_AggregationPlugin()])
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            cloud = _ToggleAggregationCloud(complete=True)
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=cloud,
                    review_store=outbox,
                    review_tracker=tracker,
                )
                full = runtime.process(
                    _payload(event_id="summary-full"),
                    network=_network(),
                )
                compact = runtime.process(
                    _payload(event_id="summary-compact"),
                    network=_network(),
                    response_detail="compact",
                )

                self.assertEqual(compact["response_detail"], "compact")
                self.assertEqual(
                    full["data_plane"]["summary_delivery_mode"],
                    "sync_cloud_review",
                )
                self.assertEqual(
                    compact["summary_delivery"]["mode"], "sync_cloud_review"
                )
                self.assertEqual(
                    compact["summary_delivery"]["persistence_stage"],
                    "cloud_review_completed",
                )
                self.assertNotIn("local_decision", compact["review"])
                self.assertNotIn("final_decision", compact["review"])
                self.assertNotIn("routing_features", compact["review"])
                transport = compact["final_decision"]["metadata"]["transport"]
                self.assertGreater(transport["request_bytes"], 0)
                self.assertIn(
                    "actual_artifact_request_bytes", compact["data_plane"]
                )
                compact_bytes = len(
                    json.dumps(compact, separators=(",", ":")).encode("utf-8")
                )
                full_bytes = len(
                    json.dumps(full, separators=(",", ":")).encode("utf-8")
                )
                self.assertLess(compact_bytes, full_bytes * 0.6)
            finally:
                tracker.close()
                registry.close()

    def test_non_aggregation_sync_review_receives_remaining_deadline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sync-decision-deadline-") as directory:
            root = Path(directory)
            registry = SceneRegistry(
                [_AggregationPlugin(aggregation_enabled=False, deadline_ms=100.0)]
            )
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            cloud = _DeadlineAwareDecisionCloud()
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=cloud,
                    review_store=outbox,
                    review_tracker=tracker,
                )
                result = runtime.process(_payload(), network=_network())

                self.assertEqual(cloud.calls, 1)
                self.assertIsNotNone(cloud.timeout_seconds)
                self.assertGreater(cloud.timeout_seconds, 0.0)
                self.assertLessEqual(cloud.timeout_seconds, 0.1)
                self.assertEqual(result["final_decision"]["status"], "final")
                self.assertEqual(outbox.count(), 0)
            finally:
                tracker.close()
                outbox.close()
                registry.close()

    def test_late_non_aggregation_result_fails_closed_to_outbox(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sync-decision-late-") as directory:
            root = Path(directory)
            registry = SceneRegistry(
                [_AggregationPlugin(aggregation_enabled=False, deadline_ms=30.0)]
            )
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            cloud = _DeadlineAwareDecisionCloud(delay_seconds=0.04)
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=cloud,
                    review_store=outbox,
                    review_tracker=tracker,
                )
                result = runtime.process(_payload(), network=_network())

                self.assertEqual(cloud.calls, 1)
                self.assertEqual(result["final_decision"]["status"], "provisional")
                self.assertEqual(result["final_decision"]["route"], "local_autonomy")
                self.assertFalse(
                    result["final_decision"]["metadata"]["action_authorization"][
                        "all_actions_authorized"
                    ]
                )
                self.assertEqual(outbox.count(), 1)
            finally:
                tracker.close()
                outbox.close()
                registry.close()

    def test_ordinary_summary_is_delivered_through_background_outbox(self) -> None:
        with tempfile.TemporaryDirectory(prefix="async-ordinary-summary-") as directory:
            root = Path(directory)
            plugin = _AggregationPlugin(
                requires_cloud_confirmation=False,
                risk_level="low",
            )
            registry = SceneRegistry([plugin])
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            cloud = _ForbiddenSlowCloud()
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=cloud,
                    review_store=outbox,
                    review_tracker=tracker,
                )
                result = runtime.process(_payload(), network=_network())

                self.assertEqual(
                    result["data_plane"]["scheduler_selected_route"],
                    "edge_only",
                )
                self.assertFalse(
                    result["data_plane"]["scheduler_selected_wait"]
                )
                self.assertEqual(result["schedule"]["route"], "cloud_async")
                self.assertFalse(result["schedule"]["waits_for_cloud"])
                self.assertEqual(result["final_decision"]["route"], "cloud_async")
                self.assertEqual(result["final_decision"]["status"], "provisional")
                self.assertEqual(result["pending_review_count"], 1)
                self.assertEqual(outbox.count(), 1)
                self.assertEqual(cloud.aggregate_calls, 0)
                self.assertEqual(cloud.coordinate_calls, 0)
            finally:
                tracker.close()
                registry.close()

    def test_ordinary_summary_returns_after_journal_before_outbox_append(self) -> None:
        with tempfile.TemporaryDirectory(prefix="async-handoff-runtime-") as directory:
            root = Path(directory)
            registry = SceneRegistry(
                [
                    _AggregationPlugin(
                        requires_cloud_confirmation=False,
                        risk_level="low",
                    )
                ]
            )
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            append_started = threading.Event()
            append_release = threading.Event()
            original_append = outbox.append

            def blocked_append(event):
                append_started.set()
                append_release.wait(timeout=2.0)
                return original_append(event)

            outbox.append = blocked_append
            handoff = DurableOutboxHandoff(
                outbox,
                root / "handoff.jsonl",
            )
            cloud = _ToggleAggregationCloud(complete=True)
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=cloud,
                    review_store=outbox,
                    review_tracker=tracker,
                    durable_handoff=handoff,
                )
                started = time.monotonic()
                result = runtime.process(
                    _payload(),
                    network=_network(),
                    response_detail="compact",
                )
                elapsed = time.monotonic() - started

                self.assertTrue(append_started.wait(timeout=0.5))
                self.assertLess(elapsed, 0.2)
                self.assertEqual(
                    result["summary_delivery"]["mode"], "background_handoff"
                )
                self.assertEqual(
                    result["summary_delivery"]["persistence_stage"],
                    "handoff_durable",
                )
                self.assertTrue(result["summary_delivery"]["fast_path"])
                self.assertEqual(outbox.count(), 0)

                append_release.set()
                deadline = time.monotonic() + 1.0
                while outbox.count() == 0 and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertEqual(outbox.count(), 1)

                flushed = runtime.flush_pending(
                    waiting_poll_seconds=0.001,
                    max_backoff_seconds=0.001,
                )
                self.assertEqual(flushed["completed"], 1)
                self.assertEqual(tracker.get("summary-1")["state"], "completed")
            finally:
                append_release.set()
                handoff.close()
                tracker.close()
                outbox.close()
                registry.close()

    def test_handoff_failure_falls_back_to_synchronous_outbox_durability(self) -> None:
        with tempfile.TemporaryDirectory(prefix="async-handoff-fallback-") as directory:
            root = Path(directory)
            registry = SceneRegistry(
                [
                    _AggregationPlugin(
                        requires_cloud_confirmation=False,
                        risk_level="low",
                    )
                ]
            )
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            handoff = _RejectingHandoff()
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=_ForbiddenSlowCloud(),
                    review_store=outbox,
                    review_tracker=tracker,
                    durable_handoff=handoff,
                )
                result = runtime.process(
                    _payload(),
                    network=_network(),
                    response_detail="compact",
                )

                self.assertLessEqual(handoff.timeout_seconds, 0.05)
                self.assertFalse(result["summary_delivery"]["fast_path"])
                self.assertEqual(
                    result["summary_delivery"]["mode"], "background_outbox"
                )
                self.assertEqual(
                    result["summary_delivery"]["persistence_stage"],
                    "outbox_durable",
                )
                self.assertEqual(outbox.count(), 1)
            finally:
                tracker.close()
                outbox.close()
                registry.close()

    def test_context_v3_freezes_delivery_operation_across_plugin_reload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="async-summary-freeze-") as directory:
            root = Path(directory)
            original_registry = SceneRegistry([_AggregationPlugin(True)])
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            try:
                original = EdgeRuntime(
                    registry=original_registry,
                    cloud=_ForbiddenSlowCloud(),
                    review_store=outbox,
                    review_tracker=tracker,
                )
                original.process(_payload(), network=_network())
                stored = outbox.events()[0]
                context = stored.metadata["_edge_review_context"]
                self.assertEqual(context["schema_version"], 3)
                self.assertEqual(context["delivery_operation"], "aggregate")
                self.assertEqual(context["requested_route"], "cloud_sync")
                self.assertGreater(context["requested_at_ms"], 0)
                self.assertGreater(context["preliminary_latency_ms"], 0.0)
                self.assertIsInstance(context["routing_features"], dict)

                updated_registry = SceneRegistry([_AggregationPlugin(False)])
                cloud = _ToggleAggregationCloud(complete=True)
                try:
                    updated = EdgeRuntime(
                        registry=updated_registry,
                        cloud=cloud,
                        review_store=outbox,
                        review_tracker=tracker,
                    )
                    flushed = updated.flush_pending(max_backoff_seconds=0.001)
                finally:
                    updated_registry.close()

                self.assertEqual(flushed["attempted"], 1)
                self.assertEqual(flushed["completed"], 1)
                self.assertEqual(flushed["errors"], [])
                self.assertEqual(outbox.count(), 0)
                self.assertEqual(cloud.aggregate_calls, 1)
                self.assertEqual(cloud.coordinate_calls, 0)
                self.assertEqual(
                    tracker.get("summary-1")["cloud_received_at_ms"],
                    cloud.accepted_at_ms[0],
                )
            finally:
                tracker.close()
                original_registry.close()

    def test_waiting_summary_is_not_failure_or_ack_then_retry_completes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="async-summary-wait-") as directory:
            root = Path(directory)
            registry = SceneRegistry(
                [
                    _AggregationPlugin(
                        True,
                        requires_cloud_confirmation=False,
                        risk_level="low",
                    )
                ]
            )
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            cloud = _ToggleAggregationCloud(complete=False)
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=cloud,
                    scheduler=_AlwaysAsyncScheduler(),
                    review_store=outbox,
                    review_tracker=tracker,
                )
                runtime.process(_payload(), network=_network())
                metrics = FrameworkMetrics("test-edge")
                worker = OutboxReplayWorker(
                    manager=_RuntimeManager(runtime),
                    outbox=outbox,
                    network_monitor=StaticNetworkMonitor(_network()),
                    config=ReplayConfig(
                        interval_seconds=1.0,
                        batch_size=1,
                        lease_seconds=1.0,
                        max_backoff_seconds=10.0,
                        waiting_poll_seconds=0.001,
                    ),
                    metrics=metrics,
                )

                waiting = worker.run_once()
                self.assertEqual(waiting["status"], "waiting")
                self.assertEqual(waiting["attempted"], 1)
                self.assertEqual(waiting["completed"], 0)
                self.assertEqual(waiting["waiting"], 1)
                self.assertEqual(waiting["errors"], [])
                self.assertEqual(outbox.count(), 1)
                first_snapshot = outbox.snapshot()
                self.assertEqual(first_snapshot["states"]["pending"], 1)
                self.assertEqual(first_snapshot["states"]["completed"], 0)
                self.assertEqual(first_snapshot["delivery_attempts"], 1)
                self.assertEqual(tracker.get("summary-1")["state"], "queued")
                first_counters = metrics.snapshot()["counters"]
                self.assertEqual(first_counters["outbox_replay_waiting_total"], 1)
                self.assertEqual(first_counters["outbox_replay_errors_total"], 0)
                self.assertEqual(
                    first_counters.get("outbox_replay_failures_total", 0), 0
                )

                time.sleep(0.01)
                cloud.complete = True
                completed = worker.run_once()
                self.assertEqual(completed["status"], "completed")
                self.assertEqual(completed["attempted"], 1)
                self.assertEqual(completed["completed"], 1)
                self.assertEqual(completed["waiting"], 0)
                self.assertEqual(completed["errors"], [])
                self.assertEqual(outbox.count(), 0)
                second_snapshot = outbox.snapshot()
                self.assertEqual(second_snapshot["states"]["completed"], 1)
                review = tracker.get("summary-1")
                self.assertEqual(review["state"], "completed")
                self.assertEqual(review["final_decision"]["status"], "final")
                self.assertEqual(cloud.aggregate_calls, 2)
                self.assertEqual(cloud.coordinate_calls, 0)
                # Receipt is the first cloud application-ingress timestamp, not
                # the later response/completion timestamp from the second poll.
                self.assertEqual(
                    review["cloud_received_at_ms"], cloud.accepted_at_ms[0]
                )
            finally:
                tracker.close()
                registry.close()

    def test_outbox_rejects_same_event_id_with_different_payload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="async-summary-conflict-") as directory:
            registry = SceneRegistry([_AggregationPlugin(True)])
            outbox = SQLiteOutbox(Path(directory) / "outbox.sqlite3")
            try:
                plugin = registry.get(_AggregationPlugin.scene)
                source = _payload(value=1.0)
                first = plugin.normalize(SceneEventEnvelope.from_dict(source))
                same = plugin.normalize(SceneEventEnvelope.from_dict(source))
                conflicting = plugin.normalize(
                    SceneEventEnvelope.from_dict(_payload(value=2.0))
                )

                self.assertTrue(outbox.append(first))
                self.assertFalse(outbox.append(same))
                with self.assertRaises(IdempotencyConflictError):
                    outbox.append(conflicting)
                self.assertEqual(outbox.count(), 1)
            finally:
                registry.close()

    def test_partial_result_expires_without_becoming_a_global_final(self) -> None:
        with tempfile.TemporaryDirectory(prefix="async-summary-partial-") as directory:
            root = Path(directory)
            registry = SceneRegistry([_AggregationPlugin(True)])
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            cloud = _ToggleAggregationCloud(partial=True)
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=cloud,
                    review_store=outbox,
                    review_tracker=tracker,
                )
                runtime.process(_payload(), network=_network())

                first = runtime.flush_pending(
                    waiting_poll_seconds=0.001,
                    partial_poll_seconds=0.001,
                    aggregation_max_wait_seconds=0.05,
                )
                self.assertEqual(first["partial_waiting"], 1)
                self.assertEqual(first["partial_expired"], 0)
                self.assertEqual(outbox.count(), 1)

                time.sleep(0.06)
                second = runtime.flush_pending(
                    waiting_poll_seconds=0.001,
                    partial_poll_seconds=0.001,
                    aggregation_max_wait_seconds=0.05,
                )
                self.assertEqual(second["completed"], 0)
                self.assertEqual(second["terminal"], 1)
                self.assertEqual(second["partial_expired"], 1)
                self.assertEqual(second["waiting"], 0)
                self.assertEqual(second["errors"], [])
                self.assertEqual(outbox.count(), 0)

                review = tracker.get("summary-1")
                self.assertEqual(review["state"], "completed")
                self.assertEqual(review["completion_stage"], "partial_final")
                self.assertEqual(review["completion_mode"], "partial_timeout")
                self.assertEqual(review["final_decision"]["status"], "provisional")
                self.assertFalse(
                    review["final_decision"]["metadata"][
                        "aggregation_wait_expired"
                    ]["evidence_complete"]
                )
                lifecycle = tracker.snapshot()
                self.assertEqual(
                    lifecycle["completion_stages"]["partial_final"], 1
                )
                self.assertEqual(
                    lifecycle["latency_ms"]["lightweight_final"]["count"], 0
                )
            finally:
                tracker.close()
                registry.close()

    def test_missing_minimum_members_expires_to_local_only_timeout(self) -> None:
        with tempfile.TemporaryDirectory(prefix="async-summary-local-timeout-") as directory:
            root = Path(directory)
            registry = SceneRegistry([_AggregationPlugin(True)])
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            cloud = _ToggleAggregationCloud(complete=False)
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=cloud,
                    review_store=outbox,
                    review_tracker=tracker,
                )
                runtime.process(_payload(), network=_network())
                first = runtime.flush_pending(
                    waiting_poll_seconds=0.001,
                    aggregation_max_wait_seconds=0.05,
                )
                self.assertEqual(first["waiting"], 1)
                self.assertEqual(first["local_timeout_expired"], 0)

                time.sleep(0.06)
                second = runtime.flush_pending(
                    waiting_poll_seconds=0.001,
                    aggregation_max_wait_seconds=0.05,
                )
                self.assertEqual(second["completed"], 0)
                self.assertEqual(second["terminal"], 1)
                self.assertEqual(second["local_timeout_expired"], 1)
                self.assertEqual(second["errors"], [])
                self.assertEqual(outbox.count(), 0)

                review = tracker.get("summary-1")
                self.assertEqual(review["completion_stage"], "local_only_timeout")
                self.assertEqual(review["completion_mode"], "aggregation_timeout")
                self.assertEqual(
                    review["final_decision"]["route"], "local_autonomy"
                )
                self.assertEqual(review["final_decision"]["status"], "provisional")
            finally:
                tracker.close()
                registry.close()

    def test_offline_backlog_gets_full_wait_window_after_cloud_acceptance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="async-summary-offline-backlog-") as directory:
            root = Path(directory)
            registry = SceneRegistry(
                [
                    _AggregationPlugin(
                        True,
                        requires_cloud_confirmation=False,
                        risk_level="low",
                    )
                ]
            )
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            cloud = _ToggleAggregationCloud(complete=False)
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=cloud,
                    review_store=outbox,
                    review_tracker=tracker,
                )
                runtime.process(_payload(), network=_network())

                # The event is older than the aggregation TTL before its first
                # successful cloud delivery, as it could be after an outage.
                time.sleep(0.06)
                first = runtime.flush_pending(
                    waiting_poll_seconds=0.001,
                    aggregation_max_wait_seconds=0.05,
                )
                self.assertEqual(first["waiting"], 1)
                self.assertEqual(first["local_timeout_expired"], 0)
                self.assertEqual(outbox.count(), 1)

                # The TTL applies only after that first accepted response.
                time.sleep(0.06)
                second = runtime.flush_pending(
                    waiting_poll_seconds=0.001,
                    aggregation_max_wait_seconds=0.05,
                )
                self.assertEqual(second["waiting"], 0)
                self.assertEqual(second["local_timeout_expired"], 1)
                self.assertEqual(second["terminal"], 1)
                self.assertEqual(outbox.count(), 0)
            finally:
                tracker.close()
                registry.close()

    def test_late_member_upgrades_timed_out_lifecycle_to_authoritative_final(
        self,
    ) -> None:
        for partial_before_full, timed_out_stage in (
            (False, "local_only_timeout"),
            (True, "partial_final"),
        ):
            with self.subTest(timed_out_stage=timed_out_stage), tempfile.TemporaryDirectory(
                prefix="async-summary-reconciliation-"
            ) as directory:
                root = Path(directory)
                registry = SceneRegistry([_AggregationPlugin(True)])
                outbox = SQLiteOutbox(root / "outbox.sqlite3")
                tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
                cloud = _LateMemberAggregationCloud(partial_before_full)
                try:
                    runtime = EdgeRuntime(
                        registry=registry,
                        cloud=cloud,
                        scheduler=_AlwaysAsyncScheduler(),
                        review_store=outbox,
                        review_tracker=tracker,
                    )
                    runtime.process(
                        _payload(event_id="summary-a", member="edge-a"),
                        network=_network(),
                    )
                    first = runtime.flush_pending(
                        waiting_poll_seconds=0.001,
                        partial_poll_seconds=0.001,
                        aggregation_max_wait_seconds=0.01,
                        reconciliation_poll_seconds=0.20,
                        reconciliation_max_wait_seconds=0.80,
                    )
                    self.assertEqual(first["aggregation_waiting"], 1)

                    time.sleep(0.02)
                    timed_out = runtime.flush_pending(
                        waiting_poll_seconds=0.001,
                        partial_poll_seconds=0.001,
                        aggregation_max_wait_seconds=0.01,
                        reconciliation_poll_seconds=0.20,
                        reconciliation_max_wait_seconds=0.80,
                    )
                    self.assertEqual(timed_out["aggregation_expired"], 1)
                    self.assertEqual(outbox.count(), 0)
                    self.assertEqual(outbox.reconciliation_count(), 1)
                    self.assertEqual(
                        tracker.get("summary-a")["completion_stage"],
                        timed_out_stage,
                    )

                    # B arrives after A's ordinary Outbox work and lifecycle
                    # have already terminated.  Its submission completes the
                    # cloud group, but A remains visibly timed out until its
                    # bounded low-frequency poll observes revision 2.
                    runtime.process(
                        _payload(event_id="summary-b", member="edge-b"),
                        network=_network(),
                    )
                    b_result = runtime.flush_pending(
                        waiting_poll_seconds=0.001,
                        partial_poll_seconds=0.001,
                        aggregation_max_wait_seconds=0.01,
                        reconciliation_poll_seconds=0.20,
                        reconciliation_max_wait_seconds=0.80,
                    )
                    self.assertEqual(b_result["completed"], 1)
                    self.assertEqual(
                        tracker.get("summary-a")["completion_stage"],
                        timed_out_stage,
                    )

                    time.sleep(0.21)
                    reconciled = runtime.flush_pending(
                        waiting_poll_seconds=0.001,
                        partial_poll_seconds=0.001,
                        aggregation_max_wait_seconds=0.01,
                        reconciliation_poll_seconds=0.20,
                        reconciliation_max_wait_seconds=0.80,
                    )
                    self.assertEqual(reconciled["reconciliation_completed"], 1)
                    self.assertEqual(reconciled["reconciliation_expired"], 0)
                    self.assertEqual(outbox.reconciliation_count(), 0)
                    final_review = tracker.get("summary-a")
                    self.assertEqual(final_review["state"], "completed")
                    self.assertEqual(
                        final_review["completion_stage"], "lightweight_final"
                    )
                    self.assertEqual(
                        final_review["completion_mode"], "reconciliation"
                    )
                    self.assertEqual(final_review["final_decision"]["status"], "final")
                    self.assertTrue(
                        final_review["final_decision"]["metadata"]["aggregation"][
                            "evidence_complete"
                        ]
                    )
                    self.assertTrue(
                        final_review["final_decision"]["metadata"]["aggregation"][
                            "global_confirmation"
                        ]
                    )
                finally:
                    tracker.close()
                    registry.close()

    def test_worker_waits_for_durable_next_available_time_and_sweeps_deadline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="outbox-next-due-") as directory:
            root = Path(directory)
            registry = SceneRegistry([_AggregationPlugin(True)])
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=_ToggleAggregationCloud(complete=False),
                    scheduler=_AlwaysAsyncScheduler(),
                    review_store=outbox,
                    review_tracker=tracker,
                )
                runtime.process(_payload(), network=_network())
                runtime.flush_pending(
                    waiting_poll_seconds=0.08,
                    aggregation_max_wait_seconds=0.01,
                    reconciliation_poll_seconds=0.02,
                    reconciliation_max_wait_seconds=0.03,
                )
                delay = outbox.next_available_delay()
                self.assertIsNotNone(delay)
                self.assertGreater(delay, 0.01)
                self.assertLessEqual(delay, 0.08)

                worker = OutboxReplayWorker(
                    manager=_RuntimeManager(runtime),
                    outbox=outbox,
                    network_monitor=StaticNetworkMonitor(_network()),
                    config=ReplayConfig(
                        interval_seconds=1.0,
                        batch_size=1,
                        lease_seconds=1.0,
                        max_backoff_seconds=1.0,
                        waiting_poll_seconds=0.08,
                        aggregation_max_wait_seconds=0.01,
                        reconciliation_poll_seconds=0.02,
                        reconciliation_max_wait_seconds=0.03,
                    ),
                    metrics=FrameworkMetrics("test-edge"),
                )
                selected_wait = worker._next_wait_seconds({"status": "scheduled"})
                self.assertLess(selected_wait, 0.20)
                self.assertNotEqual(selected_wait, 1.0)

                # Move through the ordinary timeout, then let the bounded
                # reconciliation deadline expire without a late member.
                time.sleep(0.09)
                timed_out = worker.run_once()
                self.assertEqual(timed_out["aggregation_expired"], 1)
                self.assertEqual(outbox.reconciliation_count(), 1)
                time.sleep(0.04)
                swept = worker.run_once()
                self.assertEqual(swept["status"], "reconciliation_expired")
                self.assertEqual(swept["reconciliation_expired"], 1)
                self.assertEqual(outbox.work_count(), 0)
            finally:
                tracker.close()
                registry.close()

    def test_runtime_retry_uses_immutable_source_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="async-source-identity-") as directory:
            root = Path(directory)
            registry = SceneRegistry([_AggregationPlugin(True)])
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=_ForbiddenSlowCloud(),
                    review_store=outbox,
                    review_tracker=tracker,
                )
                source = _payload(value=1.0)
                runtime.process(source, network=_network())
                runtime.process(source, network=_network())
                self.assertEqual(outbox.count(), 1)

                changed = _payload(value=2.0)
                with self.assertRaises(IdempotencyConflictError):
                    runtime.process(changed, network=_network())
                self.assertEqual(outbox.count(), 1)
            finally:
                tracker.close()
                registry.close()

    def test_business_control_flags_are_part_of_outbox_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="async-business-controls-") as directory:
            root = Path(directory)
            registry = SceneRegistry([_AggregationPlugin(True)])
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=_ForbiddenSlowCloud(),
                    review_store=outbox,
                    review_tracker=tracker,
                )
                conflict_payload = _payload(event_id="summary-conflict-control")
                runtime.process(conflict_payload, network=_network())
                with self.assertRaises(IdempotencyConflictError):
                    runtime.process(
                        conflict_payload,
                        network=_network(),
                        conflict_suspected=True,
                    )

                disagreement_payload = _payload(
                    event_id="summary-model-disagreement"
                )
                runtime.process(disagreement_payload, network=_network())
                with self.assertRaises(IdempotencyConflictError):
                    runtime.process(
                        disagreement_payload,
                        network=_network(),
                        model_disagreement=True,
                    )
                self.assertEqual(outbox.count(), 2)
            finally:
                tracker.close()
                registry.close()

    def test_review_lifecycle_rejects_business_control_context_change(self) -> None:
        with tempfile.TemporaryDirectory(prefix="review-business-controls-") as directory:
            root = Path(directory)
            registry = SceneRegistry([_AggregationPlugin(True)])
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=_ForbiddenSlowCloud(),
                    review_store=outbox,
                    review_tracker=tracker,
                )
                runtime.process(_payload(), network=_network())
                stored = outbox.events()[0]
                context = stored.metadata["_edge_review_context"]
                local = DecisionEnvelope.from_dict(context["local_decision"])
                exact_review_id = tracker.queue(
                    stored,
                    local,
                    context["requested_route"],
                    context["evidence_level"],
                    context["requested_at_ms"],
                    context["preliminary_latency_ms"],
                    context["request_bytes"],
                    context["routing_features"],
                )
                self.assertEqual(exact_review_id, tracker.get("summary-1")["review_id"])

                metadata = dict(stored.metadata)
                controls = dict(metadata["_source_business_control_context"])
                controls["conflict_suspected"] = True
                metadata["_source_business_control_context"] = controls
                changed = replace(stored, metadata=metadata)
                with self.assertRaises(IdempotencyConflictError):
                    tracker.queue(
                        changed,
                        local,
                        context["requested_route"],
                        context["evidence_level"],
                        context["requested_at_ms"],
                        context["preliminary_latency_ms"],
                        context["request_bytes"],
                        context["routing_features"],
                    )
            finally:
                tracker.close()
                registry.close()

    def test_replay_recovers_lifecycle_after_crash_between_two_stores(self) -> None:
        with tempfile.TemporaryDirectory(prefix="async-crash-recovery-") as directory:
            root = Path(directory)
            registry = SceneRegistry([_AggregationPlugin(True)])
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            review_path = root / "reviews.sqlite3"
            crashing_tracker = _CrashAfterOutboxTracker(review_path)
            try:
                crashing_runtime = EdgeRuntime(
                    registry=registry,
                    cloud=_ForbiddenSlowCloud(),
                    scheduler=_AlwaysAsyncScheduler(),
                    review_store=outbox,
                    review_tracker=crashing_tracker,
                )
                with self.assertRaisesRegex(RuntimeError, "injected crash"):
                    crashing_runtime.process(_payload(), network=_network())
                self.assertEqual(outbox.count(), 1)
            finally:
                crashing_tracker.close()

            stored = outbox.events()[0]
            context = dict(stored.metadata["_edge_review_context"])
            tracker = ReviewLifecycleStore(review_path)
            cloud = _ToggleAggregationCloud(complete=True)
            try:
                recovered_runtime = EdgeRuntime(
                    registry=registry,
                    cloud=cloud,
                    review_store=outbox,
                    review_tracker=tracker,
                )
                flushed = recovered_runtime.flush_pending(
                    waiting_poll_seconds=0.001,
                    max_backoff_seconds=0.001,
                )
                self.assertEqual(flushed["completed"], 1)
                self.assertEqual(flushed["errors"], [])
                self.assertEqual(outbox.count(), 0)

                review = tracker.get("summary-1")
                self.assertEqual(review["state"], "completed")
                self.assertEqual(review["requested_route"], "cloud_async")
                self.assertEqual(
                    review["requested_at_ms"], context["requested_at_ms"]
                )
                self.assertAlmostEqual(
                    review["preliminary_latency_ms"],
                    context["preliminary_latency_ms"],
                    places=6,
                )
                self.assertEqual(
                    review["evidence_level"], context["evidence_level"]
                )
                self.assertEqual(
                    review["planned_request_bytes"], context["request_bytes"]
                )
                self.assertEqual(
                    review["routing_features"], context["routing_features"]
                )
                self.assertEqual(review["attempts"], 1)
                self.assertEqual(cloud.aggregate_calls, 1)
            finally:
                tracker.close()
                registry.close()


if __name__ == "__main__":
    unittest.main()
