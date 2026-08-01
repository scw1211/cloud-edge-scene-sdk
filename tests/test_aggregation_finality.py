from concurrent.futures import ThreadPoolExecutor
import sqlite3
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from cloud_edge_framework.aggregation import (
    AggregationLease,
    AggregationSpec,
    MultiEdgeEventAggregator,
)
from cloud_edge_framework.cloud_service import CloudApiService
from cloud_edge_framework.contracts import (
    DECISION_STATUSES,
    DecisionEnvelope,
    SemanticEvent,
    build_decision,
)
from cloud_edge_framework.reliability import IdempotencyConflictError


EXPECTED_MEMBERS = ["edge-0", "edge-1", "edge-2", "edge-3"]


def _event(index: int) -> SemanticEvent:
    member = EXPECTED_MEMBERS[index]
    return SemanticEvent.from_dict(
        {
            "schema_version": "1.0",
            "event_id": "event-{}".format(index),
            "scene": "aggregation_test",
            "task": "test",
            "edge_id": member,
            "occurred_at_ms": 1_000 + index,
            "scope": {
                "entity_id": "sample-1",
                "subsystem": "test",
                "state_variable": "risk",
                "region_id": "region-1",
                "shared_resources": ["resource-1"],
                "correlation_keys": ["sample-1"],
                "window_start_ms": 1_000,
                "window_end_ms": 2_000,
            },
            "prediction": {
                "label": "medium",
                "confidence": 0.8,
                "probabilities": {"medium": 0.8},
                "values": {},
            },
            "risk": {"level": "medium", "score": 0.6},
            "uncertainty": {
                "confidence": 0.8,
                "calibrated": True,
                "prediction_set": ["medium"],
                "method": "test",
            },
            "timing": {
                "deadline_ms": 200,
                "preprocessing_ms": 1,
                "edge_inference_ms": 2,
            },
            "evidence": [
                {
                    "evidence_id": "evidence-{}".format(index),
                    "level": "summary",
                    "modality": "test",
                    "encoding": "json",
                    "inline": {"member": member},
                    "size_bytes": 16,
                    "content_type": "application/json",
                }
            ],
            "candidate_actions": [],
            "model": {"name": "test", "version": "1"},
            "scene_payload": {},
            "metadata": {},
        }
    )


def _spec(index: int) -> AggregationSpec:
    return AggregationSpec(
        key="sample-1",
        member=EXPECTED_MEMBERS[index],
        expected_members=list(EXPECTED_MEMBERS),
        minimum_members=2,
        timeout_ms=1,
    )


def _coordination(event: SemanticEvent) -> dict:
    decision = build_decision(
        event=event,
        decision="monitor",
        actions=[],
        confidence=0.9,
        reason="test cloud decision",
        source="test_cloud",
        policy_version="test-1",
    )
    return {
        "decisions": [decision.to_dict()],
        "event_count": 1,
        "initial_conflict_count": 0,
        "residual_conflict_count": 0,
        "resolution_success_rate": 1.0,
        "globally_consistent": True,
    }


def _create_legacy_aggregation_database(path: Path) -> None:
    with sqlite3.connect(str(path)) as connection:
        connection.execute(
            """
            CREATE TABLE aggregation_groups (
                group_id TEXT PRIMARY KEY,
                scene TEXT NOT NULL,
                group_key TEXT NOT NULL,
                expected_members_json TEXT NOT NULL,
                minimum_members INTEGER NOT NULL,
                timeout_ms INTEGER NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('waiting','inflight','completed')),
                first_received_at_ms INTEGER NOT NULL,
                deadline_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                completion_reason TEXT NOT NULL DEFAULT '',
                result_json TEXT,
                last_error TEXT NOT NULL DEFAULT '',
                UNIQUE(scene, group_key)
            )
            """
        )


class AggregationFinalityTest(unittest.TestCase):
    def test_exact_member_event_retry_is_idempotent(self) -> None:
        aggregator = MultiEdgeEventAggregator()
        try:
            event = _event(0)
            first = aggregator.submit(event, _spec(0))
            time.sleep(0.002)
            retried = aggregator.submit(event, _spec(0))

            self.assertEqual(retried, first)
            self.assertGreater(first["submitted_event_received_at_ms"], 0)
            self.assertEqual(retried["received_members"], ["edge-0"])
            count = aggregator._connection.execute(
                "SELECT COUNT(*) AS count FROM aggregation_events"
            ).fetchone()["count"]
            self.assertEqual(int(count), 1)
        finally:
            aggregator.close()

    def test_same_member_event_payload_change_is_idempotency_conflict(self) -> None:
        aggregator = MultiEdgeEventAggregator()
        try:
            event = _event(0)
            aggregator.submit(event, _spec(0))
            changed = replace(
                event,
                risk=replace(event.risk, score=0.95),
            )

            with self.assertRaises(IdempotencyConflictError):
                aggregator.submit(changed, _spec(0))
            self.assertEqual(
                aggregator.get(aggregator.submit(event, _spec(0))["group_id"])[
                    "received_members"
                ],
                ["edge-0"],
            )
        finally:
            aggregator.close()

    def test_completed_group_still_rejects_changed_retry_payload(self) -> None:
        aggregator = MultiEdgeEventAggregator()
        try:
            group_id = ""
            original = _event(0)
            for index in range(4):
                snapshot = aggregator.submit(_event(index), _spec(index))
                group_id = str(snapshot["group_id"])
            lease = aggregator.claim(group_id)
            self.assertIsNotNone(lease)
            aggregator.complete(group_id, {"global_confirmation": True})

            exact_retry = aggregator.submit(original, _spec(0))
            self.assertEqual(exact_retry["state"], "completed")

            changed = replace(
                original,
                prediction=replace(original.prediction, confidence=0.61),
            )
            with self.assertRaises(IdempotencyConflictError):
                aggregator.submit(changed, _spec(0))
            self.assertEqual(aggregator.get(group_id)["state"], "completed")
        finally:
            aggregator.close()

    def test_result_revision_migrates_an_existing_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregation.sqlite3"
            _create_legacy_aggregation_database(path)
            aggregator = MultiEdgeEventAggregator(path)
            try:
                columns = {
                    str(row["name"])
                    for row in aggregator._connection.execute(
                        "PRAGMA table_info(aggregation_groups)"
                    ).fetchall()
                }
                self.assertIn("result_revision", columns)
                self.assertIn("claimed_member_count", columns)
                self.assertIn("attempts", columns)
                self.assertIn("next_attempt_at_ms", columns)
            finally:
                aggregator.close()

    def test_schema_migration_is_safe_across_concurrent_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregation.sqlite3"
            _create_legacy_aggregation_database(path)
            worker_count = 6
            barrier = threading.Barrier(worker_count)

            def open_aggregator() -> MultiEdgeEventAggregator:
                barrier.wait()
                return MultiEdgeEventAggregator(path)

            aggregators = []
            try:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [
                        executor.submit(open_aggregator)
                        for _ in range(worker_count)
                    ]
                    aggregators = [future.result() for future in futures]
                columns = {
                    str(row["name"])
                    for row in aggregators[0]._connection.execute(
                        "PRAGMA table_info(aggregation_groups)"
                    ).fetchall()
                }
                self.assertIn("result_revision", columns)
                self.assertIn("claimed_member_count", columns)
                self.assertIn("attempts", columns)
                self.assertIn("next_attempt_at_ms", columns)
            finally:
                for aggregator in aggregators:
                    aggregator.close()

    def test_failed_coordination_uses_persistent_bounded_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregation.sqlite3"
            aggregator = MultiEdgeEventAggregator(
                path,
                retry_base_seconds=1.0,
                retry_max_seconds=1.5,
            )
            group_id = ""
            try:
                for index in range(4):
                    group_id = str(
                        aggregator.submit(_event(index), _spec(index))["group_id"]
                    )
                lease = aggregator.claim(group_id)
                self.assertIsNotNone(lease)
                aggregator.release(group_id, "first coordination failure")

                first = aggregator.get(group_id)
                self.assertEqual(first["state"], "waiting")
                self.assertEqual(first["attempts"], 1)
                self.assertGreater(first["next_attempt_at_ms"], int(time.time() * 1000))
                self.assertEqual(first["last_error"], "first coordination failure")
                self.assertIsNone(aggregator.claim(group_id))
                self.assertEqual(aggregator.claim_due(), [])
            finally:
                aggregator.close()

            # Retry state must survive a service restart.
            aggregator = MultiEdgeEventAggregator(
                path,
                retry_base_seconds=1.0,
                retry_max_seconds=1.5,
            )
            try:
                persisted = aggregator.get(group_id)
                self.assertEqual(persisted["attempts"], 1)
                self.assertIsNone(aggregator.claim(group_id))

                # Make the persisted retry due without sleeping for a second.
                with aggregator._lock, aggregator._connection:
                    aggregator._connection.execute(
                        "UPDATE aggregation_groups SET next_attempt_at_ms=0 "
                        "WHERE group_id=?",
                        (group_id,),
                    )
                retry = aggregator.claim_due()
                self.assertEqual(len(retry), 1)
                aggregator.release(group_id, "second coordination failure")

                second = aggregator.get(group_id)
                self.assertEqual(second["attempts"], 2)
                remaining_ms = second["next_attempt_at_ms"] - int(
                    time.time() * 1000
                )
                self.assertGreaterEqual(remaining_ms, 1_400)
                self.assertLessEqual(remaining_ms, 1_500)

                with aggregator._lock, aggregator._connection:
                    aggregator._connection.execute(
                        "UPDATE aggregation_groups SET next_attempt_at_ms=0 "
                        "WHERE group_id=?",
                        (group_id,),
                    )
                final_lease = aggregator.claim(group_id)
                self.assertIsNotNone(final_lease)
                aggregator.complete(group_id, {"global_confirmation": True})
                completed = aggregator.get(group_id)
                self.assertEqual(completed["attempts"], 0)
                self.assertEqual(completed["next_attempt_at_ms"], 0)
                self.assertEqual(completed["last_error"], "")
            finally:
                aggregator.close()

    def test_new_member_clears_failed_coordination_backoff(self) -> None:
        aggregator = MultiEdgeEventAggregator(
            retry_base_seconds=10.0,
            retry_max_seconds=10.0,
        )
        try:
            group_id = str(aggregator.submit(_event(0), _spec(0))["group_id"])
            aggregator.submit(_event(1), _spec(1))
            time.sleep(0.01)
            lease = aggregator.claim_due()
            self.assertEqual(len(lease), 1)
            aggregator.release(group_id, "partial coordination failure")
            self.assertEqual(aggregator.get(group_id)["attempts"], 1)
            self.assertIsNone(aggregator.claim(group_id))

            aggregator.submit(_event(2), _spec(2))
            reset = aggregator.get(group_id)
            self.assertEqual(reset["attempts"], 0)
            self.assertEqual(reset["next_attempt_at_ms"], 0)
            self.assertEqual(reset["last_error"], "")
            self.assertIsNotNone(aggregator.claim(group_id))
        finally:
            aggregator.close()

    def test_member_arriving_during_failed_lease_retries_richer_revision_now(self) -> None:
        aggregator = MultiEdgeEventAggregator(
            retry_base_seconds=10.0,
            retry_max_seconds=10.0,
        )
        try:
            group_id = str(aggregator.submit(_event(0), _spec(0))["group_id"])
            aggregator.submit(_event(1), _spec(1))
            time.sleep(0.01)
            stale_lease = aggregator.claim(group_id)
            self.assertIsNotNone(stale_lease)
            self.assertEqual(len(stale_lease.events), 2)

            aggregator.submit(_event(2), _spec(2))
            aggregator.release(group_id, "stale two-member coordination failed")
            reset = aggregator.get(group_id)
            self.assertEqual(reset["attempts"], 0)
            self.assertEqual(reset["next_attempt_at_ms"], 0)

            richer_lease = aggregator.claim(group_id)
            self.assertIsNotNone(richer_lease)
            self.assertEqual(len(richer_lease.events), 3)
        finally:
            aggregator.close()

    def test_partial_revisions_upgrade_to_a_complete_final(self) -> None:
        aggregator = MultiEdgeEventAggregator()
        try:
            first = aggregator.submit(_event(0), _spec(0))
            group_id = str(first["group_id"])
            aggregator.submit(_event(1), _spec(1))
            time.sleep(0.01)

            leases = aggregator.claim_due()
            self.assertEqual(len(leases), 1)
            first_partial = leases[0]
            self.assertEqual(
                first_partial.completion_reason, "timeout_with_partial_members"
            )
            self.assertEqual(first_partial.result_revision, 1)
            aggregator.complete(group_id, {"revision": 1})
            snapshot = aggregator.get(group_id)
            self.assertEqual(snapshot["finality"], "partial_final")
            self.assertFalse(snapshot["evidence_complete"])
            self.assertEqual(snapshot["result_revision"], 1)

            aggregator.submit(_event(2), _spec(2))
            second_partial = aggregator.claim(group_id)
            self.assertIsNotNone(second_partial)
            self.assertEqual(
                second_partial.completion_reason, "timeout_with_partial_members"
            )
            self.assertEqual(second_partial.result_revision, 2)
            aggregator.complete(group_id, {"revision": 2})
            snapshot = aggregator.get(group_id)
            self.assertEqual(snapshot["finality"], "partial_final")
            self.assertFalse(snapshot["evidence_complete"])
            self.assertEqual(snapshot["result_revision"], 2)

            before_coordination = aggregator.submit(_event(3), _spec(3))
            self.assertEqual(before_coordination["state"], "waiting")
            self.assertEqual(before_coordination["missing_members"], [])
            self.assertFalse(before_coordination["evidence_complete"])
            self.assertEqual(before_coordination["finality"], "pending")

            complete = aggregator.claim(group_id)
            self.assertIsNotNone(complete)
            self.assertEqual(complete.completion_reason, "all_expected_members")
            self.assertEqual(complete.result_revision, 3)
            aggregator.complete(group_id, {"revision": 3})
            snapshot = aggregator.get(group_id)
            self.assertEqual(snapshot["finality"], "final")
            self.assertTrue(snapshot["evidence_complete"])
            self.assertEqual(snapshot["result_revision"], 3)
            self.assertEqual(snapshot["received_members"], EXPECTED_MEMBERS)
        finally:
            aggregator.close()

    def test_members_arriving_during_partial_coordination_force_recompute(self) -> None:
        aggregator = MultiEdgeEventAggregator()
        try:
            first = aggregator.submit(_event(0), _spec(0))
            group_id = str(first["group_id"])
            aggregator.submit(_event(1), _spec(1))
            time.sleep(0.01)

            stale_partial = aggregator.claim(group_id)
            self.assertIsNotNone(stale_partial)
            self.assertEqual(len(stale_partial.received_members), 2)

            # These members arrive while cloud coordination for the 2/4 lease
            # is still in flight.  Completing that stale lease must not freeze
            # the now-complete group as a partial result.
            aggregator.submit(_event(2), _spec(2))
            aggregator.submit(_event(3), _spec(3))
            aggregator.complete(group_id, {"revision": 1, "members": 2})

            pending = aggregator.get(group_id)
            self.assertEqual(pending["state"], "waiting")
            self.assertEqual(pending["finality"], "pending")
            self.assertEqual(pending["received_members"], EXPECTED_MEMBERS)

            complete = aggregator.claim(group_id)
            self.assertIsNotNone(complete)
            self.assertEqual(complete.completion_reason, "all_expected_members")
            self.assertEqual(len(complete.events), 4)
            aggregator.complete(group_id, {"revision": 2, "members": 4})

            final = aggregator.get(group_id)
            self.assertEqual(final["state"], "completed")
            self.assertTrue(final["evidence_complete"])
            self.assertEqual(final["finality"], "final")
            self.assertEqual(final["result_revision"], 2)
        finally:
            aggregator.close()

    def test_cloud_marks_partial_without_extending_the_status_enum(self) -> None:
        self.assertEqual(
            set(DECISION_STATUSES), {"final", "provisional", "queued"}
        )
        event = _event(0)
        lease = AggregationLease(
            group_id="aggregation-1",
            scene=event.scene,
            group_key="sample-1",
            completion_reason="timeout_with_partial_members",
            expected_members=list(EXPECTED_MEMBERS),
            received_members=EXPECTED_MEMBERS[:2],
            missing_members=EXPECTED_MEMBERS[2:],
            result_revision=1,
            events=[event, _event(1)],
        )
        marked = CloudApiService._mark_aggregation_finality(
            _coordination(event), lease
        )
        decision = DecisionEnvelope.from_dict(marked["decisions"][0])
        aggregation = decision.metadata["aggregation"]
        self.assertEqual(decision.status, "provisional")
        self.assertEqual(decision.route, "cloud_async")
        self.assertEqual(aggregation["finality"], "partial_final")
        self.assertFalse(aggregation["evidence_complete"])
        self.assertFalse(aggregation["cloud_confirmed"])
        self.assertEqual(aggregation["result_revision"], 1)
        self.assertTrue(marked["observed_members_consistent"])
        self.assertFalse(marked["globally_consistent"])

    def test_cloud_marks_only_complete_consistent_result_confirmed(self) -> None:
        event = _event(0)
        lease = AggregationLease(
            group_id="aggregation-1",
            scene=event.scene,
            group_key="sample-1",
            completion_reason="all_expected_members",
            expected_members=list(EXPECTED_MEMBERS),
            received_members=list(EXPECTED_MEMBERS),
            missing_members=[],
            result_revision=2,
            events=[_event(index) for index in range(4)],
        )
        marked = CloudApiService._mark_aggregation_finality(
            _coordination(event), lease
        )
        decision = DecisionEnvelope.from_dict(marked["decisions"][0])
        aggregation = decision.metadata["aggregation"]
        self.assertEqual(decision.status, "final")
        self.assertEqual(decision.route, "cloud_sync")
        self.assertEqual(aggregation["finality"], "final")
        self.assertTrue(aggregation["evidence_complete"])
        self.assertTrue(aggregation["cloud_confirmed"])
        self.assertEqual(aggregation["result_revision"], 2)
        self.assertTrue(marked["globally_consistent"])

        inconsistent = _coordination(event)
        inconsistent["globally_consistent"] = False
        marked = CloudApiService._mark_aggregation_finality(inconsistent, lease)
        decision = DecisionEnvelope.from_dict(marked["decisions"][0])
        self.assertEqual(decision.status, "final")
        self.assertFalse(
            decision.metadata["aggregation"]["cloud_confirmed"]
        )
        self.assertFalse(marked["globally_consistent"])


if __name__ == "__main__":
    unittest.main()
