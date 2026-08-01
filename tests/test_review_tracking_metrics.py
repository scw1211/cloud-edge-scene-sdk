from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from cloud_edge_framework.contracts import DecisionEnvelope, SemanticEvent
from cloud_edge_framework.metrics import FrameworkMetrics
from cloud_edge_framework.reliability import SQLiteOutbox
from cloud_edge_framework.review_tracking import ReviewLifecycleStore


def _event(event_id: str = "event-1") -> SemanticEvent:
    return SemanticEvent.from_dict(
        {
            "schema_version": "1.0",
            "event_id": event_id,
            "scene": "test_scene",
            "task": "test_task",
            "edge_id": "edge-1",
            "occurred_at_ms": 1_000,
            "scope": {
                "entity_id": "entity-1",
                "subsystem": "test",
                "state_variable": "state",
                "region_id": "region-1",
                "shared_resources": [],
                "correlation_keys": [],
                "window_start_ms": 900,
                "window_end_ms": 1_000,
            },
            "prediction": {"label": "low", "confidence": 0.9},
            "risk": {"level": "low", "score": 0.1},
            "uncertainty": {
                "confidence": 0.9,
                "calibrated": True,
                "prediction_set": ["low"],
                "method": "test",
            },
            "timing": {"deadline_ms": 200.0},
            "evidence": [
                {
                    "evidence_id": "evidence-1",
                    "level": "summary",
                    "modality": "test",
                    "encoding": "json",
                    "inline": {"score": 0.1},
                    "size_bytes": 16,
                    "content_type": "application/json",
                }
            ],
            "candidate_actions": [],
        }
    )


def _decision(
    event_id: str,
    large_model: bool = False,
    decision: str = "monitor",
) -> DecisionEnvelope:
    metadata = {"source": "test"}
    if large_model:
        metadata["cloud_llm_review"] = {
            "verdict": "challenge",
            "recommended_decision": "monitor",
        }
    return DecisionEnvelope(
        decision_id="decision-{}-{}".format(event_id, int(large_model)),
        event_ids=[event_id],
        scene="test_scene",
        decision=decision,
        risk_level="low",
        confidence=0.95,
        route="cloud_sync",
        status="final",
        actions=[],
        reason="test",
        policy_version="test-1",
        metadata=metadata,
    )


def _create_legacy_database(path: Path) -> None:
    connection = sqlite3.connect(str(path))
    connection.execute(
        """
        CREATE TABLE review_lifecycle (
            review_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL UNIQUE,
            trace_id TEXT NOT NULL,
            scene TEXT NOT NULL,
            requested_route TEXT NOT NULL,
            state TEXT NOT NULL,
            requested_at_ms INTEGER NOT NULL,
            queued_at_ms INTEGER NOT NULL,
            started_at_ms INTEGER,
            completed_at_ms INTEGER,
            preliminary_latency_ms REAL NOT NULL,
            eventual_completion_ms REAL,
            evidence_level TEXT NOT NULL,
            planned_request_bytes INTEGER NOT NULL,
            routing_features_json TEXT NOT NULL DEFAULT '{}',
            local_decision_json TEXT NOT NULL,
            final_decision_json TEXT,
            decision_changed INTEGER,
            completion_mode TEXT NOT NULL DEFAULT '',
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NOT NULL DEFAULT ''
        )
        """
    )
    connection.commit()
    connection.close()


def _insert_legacy_completed(
    path: Path,
    event_id: str,
    final: DecisionEnvelope,
    decision_changed: bool,
) -> None:
    local = _decision(event_id)
    with sqlite3.connect(str(path)) as connection:
        connection.execute(
            """
            INSERT INTO review_lifecycle(
                review_id, event_id, trace_id, scene, requested_route, state,
                requested_at_ms, queued_at_ms, started_at_ms, completed_at_ms,
                preliminary_latency_ms, eventual_completion_ms, evidence_level,
                planned_request_bytes, routing_features_json, local_decision_json,
                final_decision_json, decision_changed, completion_mode, attempts,
                last_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "review-{}".format(event_id),
                event_id,
                "trace-{}".format(event_id),
                "test_scene",
                "cloud_async",
                "completed",
                1_000,
                1_001,
                1_002,
                1_100,
                5.0,
                100.0,
                "summary",
                128,
                "{}",
                json.dumps(local.to_dict(), sort_keys=True),
                json.dumps(final.to_dict(), sort_keys=True),
                1 if decision_changed else 0,
                "replay",
                1,
                "",
            ),
        )


def _create_legacy_outbox_database(path: Path) -> None:
    with sqlite3.connect(str(path)) as connection:
        connection.execute(
            """
            CREATE TABLE outbox_events (
                event_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('pending','inflight','completed')),
                attempts INTEGER NOT NULL DEFAULT 0,
                available_at_ms INTEGER NOT NULL,
                lease_until_ms INTEGER,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                completed_at_ms INTEGER,
                last_error TEXT NOT NULL DEFAULT ''
            )
            """
        )


class ReviewLifecycleStoreTest(unittest.TestCase):
    def test_schema_migration_is_safe_across_concurrent_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.sqlite3"
            _create_legacy_database(path)
            worker_count = 6
            barrier = threading.Barrier(worker_count)

            def open_store() -> ReviewLifecycleStore:
                barrier.wait()
                return ReviewLifecycleStore(path)

            stores = []
            try:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [executor.submit(open_store) for _ in range(worker_count)]
                    stores = [future.result() for future in futures]
                columns = {
                    str(row["name"])
                    for row in stores[0]._connection.execute(
                        "PRAGMA table_info(review_lifecycle)"
                    ).fetchall()
                }
                self.assertIn("source_identity", columns)
                self.assertIn("cloud_received_at_ms", columns)
                self.assertIn("cloud_receipt_latency_ms", columns)
                self.assertIn("completion_stage", columns)
            finally:
                for store in stores:
                    store.close()

    def test_migration_backfills_legacy_large_model_completion_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.sqlite3"
            _create_legacy_database(path)
            _insert_legacy_completed(
                path,
                "legacy-lightweight",
                _decision("legacy-lightweight"),
                decision_changed=False,
            )
            _insert_legacy_completed(
                path,
                "legacy-review",
                _decision("legacy-review", large_model=True),
                decision_changed=False,
            )
            _insert_legacy_completed(
                path,
                "legacy-correction",
                _decision(
                    "legacy-correction",
                    large_model=True,
                    decision="hold",
                ),
                decision_changed=True,
            )

            store = ReviewLifecycleStore(path)
            try:
                self.assertEqual(
                    store.get("legacy-lightweight")["completion_stage"],
                    "lightweight_final",
                )
                self.assertEqual(
                    store.get("legacy-review")["completion_stage"],
                    "large_model_review",
                )
                self.assertEqual(
                    store.get("legacy-correction")["completion_stage"],
                    "large_model_correction",
                )
                snapshot = store.snapshot()
                self.assertEqual(snapshot["terminal_completed"], 3)
                self.assertEqual(snapshot["authoritative_completed"], 3)
                self.assertEqual(snapshot["non_authoritative_completed"], 0)
                self.assertEqual(snapshot["completion_stages"], {
                    "large_model_correction": 1,
                    "large_model_review": 1,
                    "lightweight_final": 1,
                    "local_only_timeout": 0,
                    "partial_final": 0,
                })
            finally:
                store.close()

    def test_migrates_legacy_database_and_records_first_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.sqlite3"
            _create_legacy_database(path)
            store = ReviewLifecycleStore(path)
            event = _event()
            local = _decision(event.event_id)
            store.queue(
                event,
                local,
                "cloud_async",
                "summary",
                requested_at_ms=1_000,
                preliminary_latency_ms=12.5,
                planned_request_bytes=128,
            )
            store.received([event.event_id], received_at_ms=1_040)
            store.received([event.event_id], received_at_ms=1_090)

            record = store.get(event.event_id)
            self.assertEqual(record["cloud_received_at_ms"], 1_040)
            self.assertEqual(record["cloud_receipt_latency_ms"], 40.0)
            self.assertEqual(record["completion_stage"], "")
            store.close()

    def test_snapshot_separates_lightweight_and_large_model_completion(self) -> None:
        store = ReviewLifecycleStore()
        cases = (
            (False, "monitor"),
            (True, "monitor"),
            (True, "hold"),
        )
        for index, (large_model, final_decision) in enumerate(cases, start=1):
            event = _event("event-{}".format(index))
            local = _decision(event.event_id)
            store.queue(
                event,
                local,
                "cloud_async",
                "summary",
                requested_at_ms=1_000,
                preliminary_latency_ms=float(index * 10),
                planned_request_bytes=128,
            )
            store.received([event.event_id], received_at_ms=1_020 + index)
            store.complete(
                event.event_id,
                _decision(
                    event.event_id,
                    large_model=large_model,
                    decision=final_decision,
                ),
                "replay",
                completed_at_ms=1_100 + index * 10,
            )

        snapshot = store.snapshot()
        latency = snapshot["latency_ms"]
        self.assertEqual(snapshot["completion_stages"]["lightweight_final"], 1)
        self.assertEqual(snapshot["completion_stages"]["large_model_review"], 1)
        self.assertEqual(snapshot["completion_stages"]["large_model_correction"], 1)
        self.assertEqual(latency["edge_provisional"]["count"], 3)
        self.assertEqual(latency["cloud_receipt"]["count"], 3)
        self.assertEqual(latency["lightweight_final"]["count"], 1)
        self.assertEqual(latency["large_model_review"]["count"], 1)
        self.assertEqual(latency["large_model_correction"]["count"], 1)
        self.assertEqual(latency["edge_preliminary"], latency["edge_provisional"])
        self.assertEqual(latency["asynchronous_cloud_eventual"]["count"], 3)
        self.assertEqual(snapshot["corrections"], 1)
        store.close()

    def test_non_authoritative_stages_are_not_final_or_routing_samples(self) -> None:
        store = ReviewLifecycleStore()
        for index, stage in enumerate(
            ("lightweight_final", "partial_final", "local_only_timeout"), start=1
        ):
            event = _event("terminal-{}".format(index))
            local = _decision(event.event_id)
            store.queue(
                event,
                local,
                "cloud_async",
                "summary",
                requested_at_ms=1_000,
                preliminary_latency_ms=10.0,
                planned_request_bytes=128,
                routing_features={"score": float(index)},
            )
            store.complete(
                event.event_id,
                _decision(event.event_id, decision="hold"),
                "replay",
                completed_at_ms=1_100,
                completion_stage=stage,
            )

        snapshot = store.snapshot()
        self.assertEqual(snapshot["completed"], 3)
        self.assertEqual(snapshot["terminal_completed"], 3)
        self.assertEqual(snapshot["authoritative_completed"], 1)
        self.assertEqual(snapshot["non_authoritative_completed"], 2)
        self.assertEqual(
            snapshot["completion_count_semantics"][
                "cloud_correction_rate_denominator"
            ],
            "authoritative_completed",
        )
        self.assertEqual(snapshot["latency_ms"]["lightweight_final"]["count"], 1)
        self.assertEqual(snapshot["latency_ms"]["partial_final"]["count"], 1)
        self.assertEqual(snapshot["latency_ms"]["local_only_timeout"]["count"], 1)
        self.assertEqual(snapshot["corrections"], 1)
        self.assertEqual(snapshot["cloud_correction_rate"], 1.0)
        dataset = store.routing_dataset()
        self.assertEqual(dataset["schema_version"], 2)
        self.assertEqual(dataset["sample_count"], 1)
        self.assertEqual(dataset["samples"][0]["event_id"], "terminal-1")
        store.close()


class SQLiteOutboxMigrationTest(unittest.TestCase):
    def test_schema_migration_is_safe_across_concurrent_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outbox.sqlite3"
            _create_legacy_outbox_database(path)
            worker_count = 6
            barrier = threading.Barrier(worker_count)

            def open_outbox() -> SQLiteOutbox:
                barrier.wait()
                return SQLiteOutbox(path)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(open_outbox) for _ in range(worker_count)]
                outboxes = [future.result() for future in futures]
            self.assertEqual(len(outboxes), worker_count)
            with sqlite3.connect(str(path)) as connection:
                columns = {
                    str(row[1])
                    for row in connection.execute(
                        "PRAGMA table_info(outbox_events)"
                    ).fetchall()
                }
            self.assertIn("aggregation_wait_started_at_ms", columns)


class FrameworkMetricsTest(unittest.TestCase):
    def test_partial_revision_is_separate_from_authoritative_consistency(self) -> None:
        metrics = FrameworkMetrics("cloud")
        metrics.record_coordination_result(
            {
                "aggregation_finality": "partial_final",
                "event_count": 2,
                "initial_conflict_count": 1,
                "residual_conflict_count": 0,
                "resolution_success_rate": 1.0,
            },
            replayed=False,
        )
        partial_snapshot = metrics.snapshot()
        partial_counters = partial_snapshot["counters"]
        self.assertEqual(partial_counters["coordination_partial_revisions_total"], 1)
        self.assertEqual(partial_counters["coordination_partial_events_total"], 2)
        self.assertNotIn("coordination_events_total", partial_counters)
        self.assertNotIn("coordination_conflicts_initial_total", partial_counters)

        metrics.record_coordination_result(
            {
                "aggregation_finality": "final",
                "event_count": 4,
                "initial_conflict_count": 1,
                "residual_conflict_count": 0,
                "resolution_success_rate": 1.0,
            },
            replayed=False,
        )
        counters = metrics.snapshot()["counters"]
        self.assertEqual(counters["coordination_events_total"], 4)
        self.assertEqual(counters["coordination_conflicts_initial_total"], 1)
        self.assertEqual(
            counters["coordination_conflict_resolution_successes_total"], 1
        )

    def test_waiting_replay_is_not_a_failure(self) -> None:
        metrics = FrameworkMetrics("edge")
        metrics.record_replay(attempted=4, completed=0, waiting=4, errors=0)
        counters = metrics.snapshot()["counters"]
        self.assertEqual(counters["outbox_replay_waiting_total"], 4)
        self.assertNotIn("outbox_replay_failures_total", counters)

    def test_legacy_partial_replay_call_keeps_v1_failure_semantics(self) -> None:
        metrics = FrameworkMetrics("edge")
        metrics.record_replay(4, 1)
        counters = metrics.snapshot()["counters"]
        self.assertNotIn("outbox_replay_failures_total", counters)

    def test_replay_errors_and_async_delivery_are_recorded(self) -> None:
        metrics = FrameworkMetrics("edge")
        metrics.record_replay(attempted=4, completed=1, waiting=2, errors=1)
        metrics.record_async_delivery(8.5, 120, 64)
        snapshot = metrics.snapshot()
        counters = snapshot["counters"]
        self.assertEqual(counters["outbox_replay_errors_total"], 1)
        self.assertEqual(counters["outbox_replay_failures_total"], 1)
        self.assertEqual(counters["async_cloud_delivery_successes_total"], 1)
        self.assertEqual(
            snapshot["distributions"]["async_cloud_delivery_ms"]["mean"], 8.5
        )


if __name__ == "__main__":
    unittest.main()
