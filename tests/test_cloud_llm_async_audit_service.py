"""Concurrency invariants for the optional asynchronous cloud LLM audit."""

from dataclasses import replace
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from cloud_edge_framework.cloud_service import CloudApiService
from cloud_edge_framework.contracts import DecisionEnvelope
from cloud_edge_framework.feedback import DecisionFeedbackStore
from tests.test_aggregation_finality import _coordination, _event


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _BlockingReviewer:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def review(self, event, decision):
        del event, decision
        self.started.set()
        if not self.release.wait(5.0):
            raise TimeoutError("test reviewer was not released")
        return SimpleNamespace(
            to_dict=lambda: {
                "recommendation": "keep_business_result",
                "latency_ms": 5_000.0,
            }
        )


def _audit_coordination(index: int):
    event = _event(index)
    coordination = _coordination(event)
    decision = DecisionEnvelope.from_dict(coordination["decisions"][0])
    decision = replace(
        decision,
        metadata={
            **decision.metadata,
            "cloud_llm_audit_requested": True,
            "cloud_llm_review_group_key": "group-{}".format(index),
        },
    )
    return event, {**coordination, "decisions": [decision.to_dict()]}


def _bare_service(maxsize: int = 2) -> CloudApiService:
    service = object.__new__(CloudApiService)
    service.cloud_reviewer = object()
    service._cloud_audit_queue = queue.Queue(maxsize=maxsize)
    service._cloud_audit_pending = set()
    service._cloud_audit_completed_ids = set()
    service._cloud_audit_lock = threading.RLock()
    service._cloud_audit_stop = threading.Event()
    service._cloud_audit_accepting = True
    service._cloud_audit_state = {
        "running": True,
        "queued": 0,
        "completed": 0,
        "failed": 0,
        "dropped": 0,
        "shutdown_incomplete": False,
        "errors": [],
        "business_result_blocking": False,
    }
    return service


class CloudLlmAsyncAuditServiceTest(unittest.TestCase):
    def test_full_queue_does_not_increment_successful_queue_count(self) -> None:
        service = _bare_service(maxsize=1)
        service._cloud_audit_queue.put_nowait({"occupied": True})
        event, coordination = _audit_coordination(0)

        service._enqueue_cloud_audits(
            ["group-0"], [[event]], [coordination]
        )

        self.assertEqual(service._cloud_audit_state["queued"], 0)
        self.assertEqual(service._cloud_audit_state["failed"], 1)
        self.assertEqual(service._cloud_audit_state["dropped"], 1)
        self.assertEqual(service._cloud_audit_pending, set())
        service._cloud_audit_queue.get_nowait()
        service._cloud_audit_queue.task_done()

    def test_duplicate_group_is_enqueued_once(self) -> None:
        service = _bare_service()
        event, coordination = _audit_coordination(0)

        for _ in range(2):
            service._enqueue_cloud_audits(
                ["group-0"], [[event]], [coordination]
            )

        self.assertEqual(service._cloud_audit_state["queued"], 1)
        self.assertEqual(service._cloud_audit_queue.qsize(), 1)
        self.assertEqual(len(service._cloud_audit_pending), 1)
        service._cloud_audit_queue.get_nowait()
        service._cloud_audit_queue.task_done()

    def test_slow_review_does_not_block_enqueue_and_close_is_truthful(self) -> None:
        service = _bare_service(maxsize=2)
        reviewer = _BlockingReviewer()
        service.cloud_reviewer = reviewer
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        service.cloud_audit_store = DecisionFeedbackStore(
            Path(temporary.name) / "audits.jsonl"
        )
        service._cloud_audit_worker = threading.Thread(
            target=service._cloud_audit_loop,
            name="test-cloud-llm-async-audit",
            daemon=True,
        )
        service._aggregation_stop = threading.Event()
        service._aggregation_wakeup = threading.Event()
        service._aggregation_worker_state = {"running": True}
        service._aggregation_worker = threading.Thread(target=lambda: None)
        service._aggregation_worker.start()
        service.manager = _Closable()
        service.aggregator = _Closable()
        service.idempotency = _Closable()
        service._cloud_audit_worker.start()

        first_event, first_coordination = _audit_coordination(0)
        enqueue_started = time.perf_counter()
        service._enqueue_cloud_audits(
            ["group-0"], [[first_event]], [first_coordination]
        )
        enqueue_elapsed = time.perf_counter() - enqueue_started
        self.assertLess(enqueue_elapsed, 0.1)
        self.assertTrue(reviewer.started.wait(1.0))

        second_event, second_coordination = _audit_coordination(1)
        service._enqueue_cloud_audits(
            ["group-1"], [[second_event]], [second_coordination]
        )
        try:
            close_started = time.perf_counter()
            service.close()
            close_elapsed = time.perf_counter() - close_started

            self.assertLess(close_elapsed, 1.5)
            self.assertTrue(service._cloud_audit_worker.is_alive())
            self.assertTrue(service._cloud_audit_state["running"])
            self.assertTrue(service._cloud_audit_state["shutdown_incomplete"])
            self.assertEqual(service._cloud_audit_state["dropped"], 1)
            self.assertEqual(len(service._cloud_audit_pending), 1)
        finally:
            reviewer.release.set()

        service._cloud_audit_worker.join(timeout=2.0)
        self.assertFalse(service._cloud_audit_worker.is_alive())
        self.assertFalse(service._cloud_audit_state["running"])
        self.assertFalse(service._cloud_audit_state["shutdown_incomplete"])
        self.assertEqual(service._cloud_audit_state["completed"], 1)
        self.assertEqual(service._cloud_audit_pending, set())
        records = service.cloud_audit_store.records()
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["business_result_changed"])


if __name__ == "__main__":
    unittest.main()
