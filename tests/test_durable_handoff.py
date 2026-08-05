"""Focused durability and latency invariants for DurableOutboxHandoff."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from cloud_edge_framework.contracts import SemanticEvent
from cloud_edge_framework.handoff import DurableOutboxHandoff
from cloud_edge_framework.reliability import (
    IdempotencyConflictError,
    SQLiteOutbox,
    source_submission_identity,
)


def _event(event_id: str = "event-1", source_byte: str = "a") -> SemanticEvent:
    source_hash = source_byte * 64
    return SemanticEvent.from_dict(
        {
            "schema_version": "1.0",
            "event_id": event_id,
            "scene": "handoff-test",
            "task": "summary",
            "edge_id": "edge-a",
            "occurred_at_ms": 1000,
            "scope": {
                "entity_id": "entity-1",
                "subsystem": "fixture",
                "state_variable": "state",
                "region_id": "region-1",
                "window_start_ms": 900,
                "window_end_ms": 1000,
            },
            "prediction": {
                "label": "low",
                "confidence": 0.95,
                "probabilities": {"low": 0.95},
            },
            "risk": {"level": "low", "score": 0.1},
            "uncertainty": {
                "confidence": 0.95,
                "calibrated": True,
                "prediction_set": ["low"],
                "method": "fixture",
            },
            "timing": {"deadline_ms": 200.0},
            "evidence": [
                {
                    "evidence_id": "evidence-1",
                    "level": "summary",
                    "modality": "fixture",
                    "encoding": "json",
                    "inline": {"value": 1},
                    "size_bytes": 1,
                    "content_type": "application/json",
                }
            ],
            "candidate_actions": [],
            "metadata": {
                "_source_envelope_sha256": source_hash,
                "_source_business_control_context": {
                    "conflict_suspected": False,
                    "model_disagreement": False,
                },
            },
        }
    )


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


class _IdentityAwareOutbox:
    def submission_identity(self, event_id: str):
        for event in reversed(getattr(self, "events", [])):
            if event.event_id == event_id:
                return source_submission_identity(event.metadata)
        return None


class _BlockingOutbox(_IdentityAwareOutbox):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.events = []

    def append(self, event: SemanticEvent) -> bool:
        self.started.set()
        self.release.wait(timeout=5.0)
        self.events.append(event)
        return True


class _FlakyOutbox(_IdentityAwareOutbox):
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0
        self.events = []

    def append(self, event: SemanticEvent) -> bool:
        self.calls += 1
        if self.calls <= self.failures:
            raise OSError("temporary failure")
        self.events.append(event)
        return True


class _AlwaysFailingOutbox(_IdentityAwareOutbox):
    def __init__(self) -> None:
        self.calls = 0

    def append(self, event: SemanticEvent) -> bool:
        del event
        self.calls += 1
        raise OSError("offline")


class _RecordingOutbox(_IdentityAwareOutbox):
    def __init__(self) -> None:
        self.events = []

    def append(self, event: SemanticEvent) -> bool:
        self.events.append(event)
        return True


class DurableOutboxHandoffTest(unittest.TestCase):
    def test_submit_does_not_wait_for_blocked_outbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outbox = _BlockingOutbox()
            handoff = DurableOutboxHandoff(
                outbox,
                Path(directory) / "handoff.jsonl",
            )
            started = time.monotonic()
            self.assertTrue(handoff.submit(_event()))
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 0.5)
            self.assertTrue(outbox.started.wait(timeout=1.0))
            self.assertEqual(handoff.snapshot()["durable_pending_count"], 1)
            # The delivery worker is blocked in the first append, but the
            # independent journal writer must still accept another request.
            second_started = time.monotonic()
            self.assertTrue(handoff.submit(_event(event_id="event-2")))
            self.assertLess(time.monotonic() - second_started, 0.5)
            self.assertEqual(handoff.snapshot()["durable_pending_count"], 2)
            outbox.release.set()
            _wait_until(lambda: handoff.snapshot()["pending"] == 0)
            handoff.close()

    def test_failed_outbox_append_retries_with_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outbox = _FlakyOutbox(failures=2)
            handoff = DurableOutboxHandoff(
                outbox,
                Path(directory) / "handoff.jsonl",
                retry_backoff_seconds=0.005,
                max_retry_backoff_seconds=0.02,
            )
            handoff.submit(_event())
            _wait_until(lambda: handoff.snapshot()["pending"] == 0)
            self.assertEqual(outbox.calls, 3)
            self.assertGreaterEqual(handoff.snapshot()["retry_count"], 2)
            self.assertIsNone(handoff.snapshot()["last_error"])
            handoff.close()

    def test_restart_recovers_put_without_ack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "handoff.jsonl"
            first_outbox = _AlwaysFailingOutbox()
            first = DurableOutboxHandoff(
                first_outbox,
                journal,
                retry_backoff_seconds=0.05,
                max_retry_backoff_seconds=0.05,
            )
            first.submit(_event())
            _wait_until(lambda: first.snapshot()["retry_count"] > 0)
            first.close(timeout_seconds=0.1)
            self.assertGreater(journal.stat().st_size, 0)

            second_outbox = _RecordingOutbox()
            second = DurableOutboxHandoff(second_outbox, journal)
            self.assertEqual(second.snapshot()["recovered"], 1)
            _wait_until(lambda: second.snapshot()["pending"] == 0)
            self.assertEqual([item.event_id for item in second_outbox.events], ["event-1"])
            second.close()

    def test_pending_duplicate_and_conflict_are_synchronous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outbox = _BlockingOutbox()
            handoff = DurableOutboxHandoff(
                outbox,
                Path(directory) / "handoff.jsonl",
            )
            original = _event()
            self.assertTrue(handoff.submit(original))
            self.assertTrue(outbox.started.wait(timeout=1.0))
            # Runtime-only measurements differ, but the stable source identity
            # makes this the same cloud submission.
            exact_retry = replace(original, occurred_at_ms=1001)
            self.assertFalse(handoff.submit(exact_retry))
            with self.assertRaises(IdempotencyConflictError):
                handoff.submit(_event(source_byte="b"))
            outbox.release.set()
            _wait_until(lambda: handoff.snapshot()["pending"] == 0)
            handoff.close()

    def test_completed_outbox_conflict_is_rejected_before_journal_acceptance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = root / "handoff.jsonl"
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            handoff = DurableOutboxHandoff(outbox, journal)
            try:
                self.assertTrue(handoff.submit(_event(source_byte="a")))
                _wait_until(lambda: handoff.snapshot()["pending"] == 0)
                self.assertEqual(outbox.count(), 1)
                self.assertEqual(journal.read_bytes(), b"")

                self.assertFalse(handoff.submit(_event(source_byte="a")))
                with self.assertRaises(IdempotencyConflictError):
                    handoff.submit(_event(source_byte="b"))

                self.assertEqual(handoff.snapshot()["pending"], 0)
                self.assertEqual(handoff.snapshot()["retry_count"], 0)
                self.assertEqual(journal.read_bytes(), b"")
            finally:
                handoff.close()
                outbox.close()

    def test_historical_identity_lookup_does_not_wait_for_outbox_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            handoff = DurableOutboxHandoff(outbox, root / "handoff.jsonl")
            outbox._lock.acquire()
            try:
                started = time.monotonic()
                self.assertTrue(handoff.submit(_event()))
                self.assertLess(time.monotonic() - started, 0.5)
                self.assertEqual(handoff.snapshot()["durable_pending_count"], 1)
            finally:
                outbox._lock.release()
            try:
                _wait_until(lambda: handoff.snapshot()["pending"] == 0)
            finally:
                handoff.close()
                outbox.close()

    def test_second_instance_cannot_share_or_compact_owned_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "handoff.jsonl"
            blocking_outbox = _BlockingOutbox()
            first = DurableOutboxHandoff(blocking_outbox, journal)
            try:
                self.assertTrue(first.submit(_event()))
                self.assertTrue(blocking_outbox.started.wait(timeout=1.0))
                with self.assertRaisesRegex(RuntimeError, "already owned"):
                    DurableOutboxHandoff(_RecordingOutbox(), journal)

                # The accepted put remains present while the sole owner is
                # blocked before Outbox persistence; no second instance can
                # observe local emptiness and truncate it.
                self.assertGreater(journal.stat().st_size, 0)
                self.assertEqual(first.snapshot()["pending"], 1)
            finally:
                blocking_outbox.release.set()
                _wait_until(lambda: first.snapshot()["pending"] == 0)
                first.close()

            replacement = DurableOutboxHandoff(_RecordingOutbox(), journal)
            replacement.close()

    def test_callback_runs_only_after_outbox_and_ack_are_durable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "handoff.jsonl"
            outbox = _BlockingOutbox()
            callbacks = []
            handoff = DurableOutboxHandoff(
                outbox,
                journal,
                persisted_callback=lambda event, inserted: callbacks.append(
                    (event.event_id, inserted, journal.read_bytes())
                ),
            )
            handoff.submit(_event())
            self.assertTrue(outbox.started.wait(timeout=1.0))
            self.assertEqual(callbacks, [])
            outbox.release.set()
            _wait_until(lambda: len(callbacks) == 1)
            self.assertEqual(callbacks[0][:2], ("event-1", True))
            # Steady-state compaction occurs before callback; either way there
            # can be no unacknowledged put when observers are notified.
            self.assertEqual(callbacks[0][2], b"")
            handoff.close()

    def test_incomplete_tail_is_truncated_and_prior_put_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "handoff.jsonl"
            failing = _AlwaysFailingOutbox()
            first = DurableOutboxHandoff(
                failing,
                journal,
                retry_backoff_seconds=0.05,
                max_retry_backoff_seconds=0.05,
            )
            first.submit(_event())
            _wait_until(lambda: first.snapshot()["retry_count"] > 0)
            first.close(timeout_seconds=0.1)
            with journal.open("ab") as stream:
                stream.write(b'{"body":{"op":"put"')
                stream.flush()

            outbox = _RecordingOutbox()
            recovered = DurableOutboxHandoff(outbox, journal)
            self.assertEqual(recovered.snapshot()["recovered"], 1)
            _wait_until(lambda: recovered.snapshot()["pending"] == 0)
            self.assertEqual(len(outbox.events), 1)
            recovered.close()

    def test_complete_checksum_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "handoff.jsonl"
            journal.write_text(
                json.dumps({"body": {"op": "ack"}, "checksum": "bad"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                DurableOutboxHandoff(_RecordingOutbox(), journal)


if __name__ == "__main__":
    unittest.main()
