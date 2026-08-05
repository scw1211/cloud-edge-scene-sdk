"""Regression tests for SQLite durability and idempotency boundaries."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from cloud_edge_framework.contracts import SemanticEvent
import cloud_edge_framework.reliability as reliability
from cloud_edge_framework.reliability import SQLiteIdempotencyStore, SQLiteOutbox


def _event(event_id: str, label: str = "normal") -> SemanticEvent:
    return SemanticEvent.from_dict(
        {
            "event_id": event_id,
            "scene": "reliability_fixture",
            "task": "durability",
            "edge_id": "edge-a",
            "occurred_at_ms": 1,
            "scope": {
                "entity_id": "entity-a",
                "subsystem": "test",
                "state_variable": "state",
                "region_id": "region-a",
            },
            "prediction": {"label": label, "confidence": 0.9},
            "risk": {"level": "low", "score": 0.1},
            "uncertainty": {
                "confidence": 0.9,
                "prediction_set": ["low"],
                "method": "fixture",
            },
            "timing": {},
            "evidence": [
                {
                    "evidence_id": "summary-a",
                    "level": "summary",
                    "modality": "fixture",
                    "encoding": "json",
                    "inline": {"label": label},
                }
            ],
            "candidate_actions": [],
        }
    )


class _ExecuteFailureConnection:
    """Connection proxy that fails one selected Outbox INSERT."""

    def __init__(self, connection: sqlite3.Connection, failure: BaseException) -> None:
        self.connection = connection
        self.failure = failure
        self.failed = False
        self.rollback_calls = 0

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def execute(self, sql, parameters=()):
        if (
            not self.failed
            and "INSERT OR IGNORE INTO outbox_events" in str(sql)
            and parameters
            and str(parameters[0]) == "event-fail"
        ):
            self.failed = True
            raise self.failure
        return self.connection.execute(sql, parameters)

    def rollback(self):
        self.rollback_calls += 1
        return self.connection.rollback()


class _CommitFailureConnection:
    """Connection proxy that rolls back one idempotency completion commit."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.failed = False
        self._completing_response = False

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def __enter__(self):
        self.connection.__enter__()
        self._completing_response = False
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None and self._completing_response and not self.failed:
            self.failed = True
            self.connection.rollback()
            raise sqlite3.OperationalError("injected completion commit failure")
        return self.connection.__exit__(exc_type, exc_value, traceback)

    def execute(self, sql, parameters=()):
        normalized = " ".join(str(sql).split())
        if (
            "UPDATE idempotency_responses" in normalized
            and "SET response_json=" in normalized
        ):
            self._completing_response = True
        return self.connection.execute(sql, parameters)


def _idempotency_counts(path: Path):
    with sqlite3.connect(str(path)) as connection:
        return connection.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM idempotency_responses "
            " WHERE state='completed'), "
            "(SELECT value FROM idempotency_metadata "
            " WHERE name='completed_count'), "
            "(SELECT COUNT(*) FROM idempotency_responses "
            " WHERE state='inflight')"
        ).fetchone()


class SQLiteIdempotencySemanticsTest(unittest.TestCase):
    def test_request_lock_cleanup_does_not_wait_for_database_lock(self) -> None:
        with tempfile.TemporaryDirectory(prefix="idempotency-lock-split-") as directory:
            store = SQLiteIdempotencyStore(
                Path(directory) / "idempotency.sqlite3", 60.0, 100
            )
            entered = threading.Event()
            release_request = threading.Event()
            exited = threading.Event()

            def use_request_lock() -> None:
                with store._request_lock("independent-key"):
                    entered.set()
                    if not release_request.wait(2.0):
                        raise TimeoutError("request lock was not released")
                exited.set()

            thread = threading.Thread(target=use_request_lock)
            thread.start()
            try:
                self.assertTrue(entered.wait(1.0))
                # Model another request holding the SQLite connection lock during
                # its completion transaction. Registry cleanup for this unrelated
                # key must still finish instead of joining that database convoy.
                with store._lock:
                    release_request.set()
                    completed_without_database_lock = exited.wait(0.5)
                self.assertTrue(completed_without_database_lock)
            finally:
                release_request.set()
                thread.join(timeout=2.0)
                store.close()
            self.assertFalse(thread.is_alive())

    def test_capacity_is_strict_across_ten_store_instances(self) -> None:
        with tempfile.TemporaryDirectory(prefix="idempotency-shared-cap-") as directory:
            path = Path(directory) / "idempotency.sqlite3"
            stores = [
                SQLiteIdempotencyStore(path, 60.0, max_entries=3)
                for _ in range(10)
            ]

            def execute(index: int):
                value, replayed = stores[index].execute(
                    "shared-cap-key-{}".format(index),
                    {"index": index},
                    lambda: {"index": index},
                )
                return value, replayed, _idempotency_counts(path)

            try:
                with ThreadPoolExecutor(max_workers=10) as executor:
                    results = list(executor.map(execute, range(10)))

                for index, (value, replayed, counts) in enumerate(results):
                    self.assertEqual(value, {"index": index})
                    self.assertFalse(replayed)
                    actual, tracked, _inflight = counts
                    self.assertEqual(actual, tracked)
                    self.assertLessEqual(actual, 3)

                self.assertEqual(_idempotency_counts(path), (3, 3, 0))
            finally:
                for store in stores:
                    store.close()

    def test_capacity_count_survives_eviction_expiry_and_restart(self) -> None:
        with tempfile.TemporaryDirectory(prefix="idempotency-cap-lifecycle-") as directory:
            path = Path(directory) / "idempotency.sqlite3"
            now = [1_000.0]
            store = SQLiteIdempotencyStore(path, 1.0, max_entries=3)
            traced_statements = []
            store._connection.set_trace_callback(traced_statements.append)
            try:
                with patch(
                    "cloud_edge_framework.reliability.time.time",
                    side_effect=lambda: now[0],
                ):
                    for index in range(4):
                        value, replayed = store.execute(
                            "key-{}".format(index),
                            {"index": index},
                            lambda index=index: {"index": index},
                        )
                        self.assertEqual(value, {"index": index})
                        self.assertFalse(replayed)
                        now[0] += 0.01

                self.assertEqual(_idempotency_counts(path), (3, 3, 0))
                with sqlite3.connect(str(path)) as connection:
                    retained = {
                        row[0]
                        for row in connection.execute(
                            "SELECT request_key FROM idempotency_responses "
                            "WHERE state='completed'"
                        ).fetchall()
                    }
                self.assertEqual(retained, {"key-1", "key-2", "key-3"})
                self.assertFalse(
                    any(
                        "COUNT(*) FROM IDEMPOTENCY_RESPONSES" in statement.upper()
                        for statement in traced_statements
                    ),
                    traced_statements,
                )
            finally:
                store.close()

            # Startup reconciliation repairs metadata written by an older build
            # or an interrupted external migration before enforcing the cap.
            with sqlite3.connect(str(path)) as connection:
                connection.execute(
                    "UPDATE idempotency_metadata SET value=99 "
                    "WHERE name='completed_count'"
                )

            with patch(
                "cloud_edge_framework.reliability.time.time",
                side_effect=lambda: now[0],
            ):
                reopened = SQLiteIdempotencyStore(path, 1.0, max_entries=3)
                try:
                    self.assertEqual(_idempotency_counts(path), (3, 3, 0))
                    now[0] += 1.1
                    value, replayed = reopened.execute(
                        "fresh-key",
                        {"index": "fresh"},
                        lambda: {"index": "fresh"},
                    )
                    self.assertEqual(value, {"index": "fresh"})
                    self.assertFalse(replayed)
                    self.assertEqual(_idempotency_counts(path), (1, 1, 0))
                finally:
                    reopened.close()

            final_store = SQLiteIdempotencyStore(path, 1.0, max_entries=3)
            try:
                self.assertEqual(_idempotency_counts(path), (1, 1, 0))
            finally:
                final_store.close()

    def test_completion_commit_failure_rolls_back_row_and_counter(self) -> None:
        with tempfile.TemporaryDirectory(prefix="idempotency-commit-fail-") as directory:
            path = Path(directory) / "idempotency.sqlite3"
            store = SQLiteIdempotencyStore(path, 60.0, max_entries=3)
            proxy = _CommitFailureConnection(store._connection)
            store._connection = proxy
            calls = 0

            def operation():
                nonlocal calls
                calls += 1
                return {"decision": "allow"}

            try:
                with self.assertRaisesRegex(
                    sqlite3.OperationalError,
                    "injected completion commit failure",
                ):
                    store.execute("commit-failure", {"event_id": "a"}, operation)

                self.assertTrue(proxy.failed)
                self.assertEqual(calls, 1)
                self.assertEqual(_idempotency_counts(path), (0, 0, 1))
            finally:
                store.close()

    def test_two_store_instances_execute_same_key_once_and_replay(self) -> None:
        with tempfile.TemporaryDirectory(prefix="idempotency-two-store-") as directory:
            path = Path(directory) / "idempotency.sqlite3"
            first_store = SQLiteIdempotencyStore(path, 60.0, 100)
            second_store = SQLiteIdempotencyStore(path, 60.0, 100)
            operation_started = threading.Event()
            release_operation = threading.Event()
            calls = 0
            calls_lock = threading.Lock()

            def operation():
                nonlocal calls
                with calls_lock:
                    calls += 1
                operation_started.set()
                if not release_operation.wait(2.0):
                    raise TimeoutError("test operation was not released")
                return {"decision": "allow", "sequence": 1}

            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    first = executor.submit(
                        first_store.execute,
                        "shared-key",
                        {"event_id": "event-a"},
                        operation,
                    )
                    self.assertTrue(operation_started.wait(1.0))
                    second = executor.submit(
                        second_store.execute,
                        "shared-key",
                        {"event_id": "event-a"},
                        operation,
                    )
                    time.sleep(0.05)
                    with calls_lock:
                        self.assertEqual(calls, 1)
                    release_operation.set()
                    first_value, first_replayed = first.result(timeout=3.0)
                    second_value, second_replayed = second.result(timeout=3.0)

                self.assertEqual(first_value, second_value)
                self.assertEqual(
                    sorted([first_replayed, second_replayed]),
                    [False, True],
                )
                with calls_lock:
                    self.assertEqual(calls, 1)
            finally:
                release_operation.set()
                first_store.close()
                second_store.close()

    def test_expired_response_is_reexecuted_without_waiting_for_cleanup(self) -> None:
        with tempfile.TemporaryDirectory(prefix="idempotency-strict-ttl-") as directory:
            store = SQLiteIdempotencyStore(
                Path(directory) / "idempotency.sqlite3",
                ttl_seconds=0.05,
                max_entries=100,
            )
            now = [1_000.0]
            calls = []

            def operation():
                generation = len(calls) + 1
                calls.append(generation)
                return {"generation": generation}

            try:
                with patch(
                    "cloud_edge_framework.reliability.time.time",
                    side_effect=lambda: now[0],
                ):
                    first, first_replayed = store.execute(
                        "ttl-key", {"value": 1}, operation
                    )
                    now[0] += 0.05
                    second, second_replayed = store.execute(
                        "ttl-key", {"value": 1}, operation
                    )
                    third, third_replayed = store.execute(
                        "ttl-key", {"value": 1}, operation
                    )

                self.assertEqual(first, {"generation": 1})
                self.assertFalse(first_replayed)
                self.assertEqual(second, {"generation": 2})
                self.assertFalse(second_replayed)
                self.assertEqual(third, second)
                self.assertTrue(third_replayed)
                self.assertEqual(calls, [1, 2])
            finally:
                store.close()

    def test_compressed_response_replays_exactly_after_store_restart(self) -> None:
        with tempfile.TemporaryDirectory(prefix="idempotency-compressed-") as directory:
            path = Path(directory) / "idempotency.sqlite3"
            response = {
                "decision": "allow",
                "explanation": "路况摘要" * 4_000,
                "metadata": {"partition_ids": list(range(64)), "stable": True},
            }
            first_store = SQLiteIdempotencyStore(path, 60.0, 100)
            first_value, first_replayed = first_store.execute(
                "compressed-key",
                {"event_id": "event-compressed"},
                lambda: response,
            )
            first_store.close()

            with sqlite3.connect(str(path)) as connection:
                storage_type, stored_bytes = connection.execute(
                    "SELECT typeof(response_json), length(response_json) "
                    "FROM idempotency_responses WHERE request_key=?",
                    ("compressed-key",),
                ).fetchone()
            uncompressed_bytes = len(
                json.dumps(
                    response,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            self.assertEqual(storage_type, "blob")
            self.assertLess(stored_bytes, uncompressed_bytes)

            calls = 0

            def should_not_run():
                nonlocal calls
                calls += 1
                return {"decision": "different"}

            second_store = SQLiteIdempotencyStore(path, 60.0, 100)
            try:
                replayed_value, replayed = second_store.execute(
                    "compressed-key",
                    {"event_id": "event-compressed"},
                    should_not_run,
                )
            finally:
                second_store.close()

            self.assertFalse(first_replayed)
            self.assertEqual(first_value, response)
            self.assertTrue(replayed)
            self.assertEqual(replayed_value, response)
            self.assertEqual(calls, 0)


class SQLiteOutboxBatchFailureTest(unittest.TestCase):
    def test_bounded_batch_always_commits_callers_own_ticket(self) -> None:
        with tempfile.TemporaryDirectory(prefix="outbox-bounded-batch-") as directory:
            path = Path(directory) / "outbox.sqlite3"
            outbox = SQLiteOutbox(path)
            now_ms = int(time.time() * 1000)
            queued_events = [_event("queued-a"), _event("queued-b")]
            with outbox._append_lock:
                outbox._append_queue.extend(
                    reliability._AppendTicket(
                        event.event_id,
                        outbox._serialized(event),
                        now_ms,
                    )
                    for event in queued_events
                )
            try:
                with patch(
                    "cloud_edge_framework.reliability._OUTBOX_APPEND_MAX_BATCH",
                    2,
                ):
                    self.assertTrue(outbox.append(_event("callers-own-ticket")))

                with sqlite3.connect(str(path)) as connection:
                    event_ids = {
                        str(row[0])
                        for row in connection.execute(
                            "SELECT event_id FROM outbox_events"
                        ).fetchall()
                    }
                self.assertIn("callers-own-ticket", event_ids)
                self.assertEqual(len(event_ids), 2)
            finally:
                outbox.close()

    def _assert_batch_failure_rolls_back(self, failure: BaseException) -> None:
        with tempfile.TemporaryDirectory(prefix="outbox-batch-failure-") as directory:
            path = Path(directory) / "outbox.sqlite3"
            outbox = SQLiteOutbox(path)
            proxy = _ExecuteFailureConnection(outbox._connection, failure)
            outbox._connection = proxy
            outcomes = []
            outcome_lock = threading.Lock()

            def append(event):
                try:
                    value = outbox.append(event)
                    outcome = ("result", value)
                except BaseException as exc:  # noqa: BLE001 - asserted below
                    outcome = (type(exc), str(exc))
                with outcome_lock:
                    outcomes.append(outcome)

            # Queue the tickets in a known order while no request can drain them.
            # The first INSERT is staged before the selected second INSERT fails,
            # proving that rollback removes already-applied peer work too.
            outbox._lock.acquire()
            threads = []
            try:
                for event in (
                    _event("event-good-before"),
                    _event("event-fail"),
                    _event("event-good-after"),
                ):
                    thread = threading.Thread(target=append, args=(event,))
                    thread.start()
                    threads.append(thread)
                    deadline = time.monotonic() + 1.0
                    while time.monotonic() < deadline:
                        with outbox._append_lock:
                            if len(outbox._append_queue) == len(threads):
                                break
                        time.sleep(0.001)
                    else:
                        self.fail("append ticket did not reach the group-commit queue")
            finally:
                outbox._lock.release()

            for thread in threads:
                thread.join(timeout=2.0)
                self.assertFalse(thread.is_alive())

            self.assertEqual(len(outcomes), 3)
            self.assertTrue(
                all(outcome[0] is type(failure) for outcome in outcomes), outcomes
            )
            self.assertGreaterEqual(proxy.rollback_calls, 1)
            with sqlite3.connect(str(path)) as connection:
                failed_batch_rows = connection.execute(
                    "SELECT COUNT(*) FROM outbox_events"
                ).fetchone()[0]
            self.assertEqual(failed_batch_rows, 0)

            # The rolled-back tickets were removed from the handoff queue, and a
            # later independent append must not sweep them into its successful
            # transaction.
            self.assertTrue(outbox.append(_event("event-recovery")))
            with sqlite3.connect(str(path)) as connection:
                event_ids = [
                    str(row[0])
                    for row in connection.execute(
                        "SELECT event_id FROM outbox_events ORDER BY event_id"
                    ).fetchall()
                ]
            self.assertEqual(event_ids, ["event-recovery"])
            outbox.close()

    def test_sql_failure_rolls_back_every_ticket_in_batch(self) -> None:
        self._assert_batch_failure_rolls_back(
            sqlite3.OperationalError("injected SQL failure")
        )

    def test_unknown_failure_rolls_back_every_ticket_in_batch(self) -> None:
        self._assert_batch_failure_rolls_back(
            RuntimeError("injected unknown failure")
        )


if __name__ == "__main__":
    unittest.main()
