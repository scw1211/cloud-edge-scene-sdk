"""Safety and durability tests for deferred request-path monitoring."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import time
import unittest

from cloud_edge_framework.contracts import SemanticEvent
from cloud_edge_framework.monitoring import CalibrationDriftMonitor, MonitoringPolicy


def _event(index: int, scene: str = "deferred-monitoring") -> SemanticEvent:
    return SemanticEvent.from_dict(
        {
            "schema_version": "1.0",
            "event_id": "event-{}".format(index),
            "scene": scene,
            "task": "classification",
            "edge_id": "edge-a",
            "occurred_at_ms": 1000 + index,
            "scope": {
                "entity_id": "entity-{}".format(index),
                "subsystem": "fixture",
                "state_variable": "state",
                "region_id": "region-a",
                "window_start_ms": 900 + index,
                "window_end_ms": 1000 + index,
            },
            "prediction": {
                "label": "low",
                "confidence": 0.9,
                "probabilities": {"low": 0.9},
            },
            "risk": {"level": "low", "score": 0.1},
            "uncertainty": {
                "confidence": 0.9,
                "calibrated": True,
                "prediction_set": ["low"],
                "method": "fixture",
            },
            "timing": {"deadline_ms": 200.0},
            "evidence": [
                {
                    "evidence_id": "evidence-{}".format(index),
                    "level": "summary",
                    "modality": "fixture",
                    "encoding": "json",
                    "inline": {"value": index},
                    "size_bytes": 1,
                    "content_type": "application/json",
                }
            ],
            "candidate_actions": [],
        }
    )


def _policy() -> MonitoringPolicy:
    return MonitoringPolicy(
        window_size=100,
        bins=5,
        min_labeled_samples=2,
        min_drift_samples=2,
        bootstrap_reference_size=10,
        evaluation_interval_events=4,
        evaluation_max_staleness_ms=100,
    )


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


class DeferredMonitoringTest(unittest.TestCase):
    def test_concurrent_observations_persist_and_survive_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "monitoring.sqlite3"
            monitor = CalibrationDriftMonitor(
                path,
                _policy(),
                deferred_queue_size=128,
            )
            with ThreadPoolExecutor(max_workers=16) as executor:
                results = list(
                    executor.map(
                        lambda index: monitor.observe_deferred(
                            _event(index), {"scene_signal": 0.25}
                        ),
                        range(40),
                    )
                )
            self.assertTrue(all(item["deferred"]["accepted"] for item in results))
            self.assertTrue(all(not item["force_cloud_review"] for item in results))
            _wait_until(lambda: monitor.deferred_snapshot()["pending"] == 0)
            self.assertEqual(monitor.deferred_snapshot()["processed"], 40)
            monitor.close()

            reopened = CalibrationDriftMonitor(path, _policy())
            self.assertEqual(
                reopened.scene_snapshot("deferred-monitoring")["observed_count"],
                40,
            )
            reopened.close()

    def test_persisted_known_degraded_state_forces_review_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "monitoring.sqlite3"
            policy = MonitoringPolicy(
                window_size=20,
                bins=5,
                min_labeled_samples=2,
                min_drift_samples=2,
                bootstrap_reference_size=10,
                max_psi=0.01,
                evaluation_interval_events=1,
                evaluation_max_staleness_ms=100,
            )
            monitor = CalibrationDriftMonitor(path, policy)
            monitor.set_reference(
                "deferred-monitoring", "scene_signal", [0.01, 0.02]
            )
            monitor.observe(_event(1), {"scene_signal": 0.99})
            degraded = monitor.observe(_event(2), {"scene_signal": 0.98})
            self.assertEqual(degraded["status"], "degraded")
            monitor.close()

            reopened = CalibrationDriftMonitor(path, policy)
            started = time.monotonic()
            result = reopened.observe_deferred(
                _event(3), {"scene_signal": 0.97}
            )
            self.assertLess(time.monotonic() - started, 0.1)
            self.assertTrue(result["deferred"]["accepted"])
            self.assertTrue(result["force_cloud_review"])
            self.assertIn("psi_exceeded:scene_signal", result["reasons"])
            reopened.close()

    def test_worker_failure_is_visible_and_subsequent_requests_fail_closed(self) -> None:
        monitor = CalibrationDriftMonitor(policy=_policy())

        def fail(_observation) -> None:
            raise OSError("forced persistence failure")

        monitor._process_deferred_observation = fail
        first = monitor.observe_deferred(_event(1))
        self.assertTrue(first["deferred"]["accepted"])
        _wait_until(lambda: monitor.deferred_snapshot()["worker_failed"])
        second = monitor.observe_deferred(_event(2))
        self.assertFalse(second["deferred"]["accepted"])
        self.assertTrue(second["force_cloud_review"])
        self.assertIn("monitoring_deferred_worker_failed", second["reasons"])
        status = monitor.deferred_snapshot()
        self.assertEqual(status["failed"], 1)
        self.assertIn("forced persistence failure", status["last_error"])
        monitor.close()

    def test_queue_saturation_fails_closed_without_blocking(self) -> None:
        monitor = CalibrationDriftMonitor(
            policy=_policy(), deferred_queue_size=1
        )
        original = monitor._process_deferred_observation
        entered = threading.Event()
        release = threading.Event()

        def block(observation) -> None:
            entered.set()
            release.wait(timeout=2.0)
            original(observation)

        monitor._process_deferred_observation = block
        self.assertTrue(
            monitor.observe_deferred(_event(1))["deferred"]["accepted"]
        )
        self.assertTrue(entered.wait(timeout=1.0))
        self.assertTrue(
            monitor.observe_deferred(_event(2))["deferred"]["accepted"]
        )
        started = time.monotonic()
        saturated = monitor.observe_deferred(_event(3))
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertFalse(saturated["deferred"]["accepted"])
        self.assertTrue(saturated["force_cloud_review"])
        self.assertIn(
            "monitoring_deferred_queue_saturated", saturated["reasons"]
        )
        release.set()
        _wait_until(lambda: monitor.deferred_snapshot()["pending"] == 0)
        monitor.close()

    def test_close_is_bounded_and_leaves_unprocessed_work_visible(self) -> None:
        monitor = CalibrationDriftMonitor(
            policy=_policy(), deferred_queue_size=4
        )
        original = monitor._process_deferred_observation
        entered = threading.Event()
        release = threading.Event()

        def block(observation) -> None:
            entered.set()
            release.wait(timeout=2.0)
            original(observation)

        monitor._process_deferred_observation = block
        monitor.observe_deferred(_event(1))
        self.assertTrue(entered.wait(timeout=1.0))
        monitor.observe_deferred(_event(2))
        started = time.monotonic()
        monitor.close(timeout_seconds=0.03)
        self.assertLess(time.monotonic() - started, 0.2)
        status = monitor.deferred_snapshot()
        self.assertTrue(status["close_timed_out"])
        self.assertGreaterEqual(status["unprocessed_visible"], 1)
        after_close = monitor.observe_deferred(_event(3))
        self.assertTrue(after_close["force_cloud_review"])
        self.assertIn("monitoring_deferred_closed", after_close["reasons"])

        release.set()
        _wait_until(lambda: not monitor.deferred_snapshot()["worker_running"])

    def test_invalid_signal_is_rejected_before_enqueue(self) -> None:
        monitor = CalibrationDriftMonitor(policy=_policy())
        with self.assertRaises(ValueError):
            monitor.observe_deferred(_event(1), {"bad": 1.5})
        self.assertEqual(monitor.deferred_snapshot()["accepted"], 0)
        monitor.close()


if __name__ == "__main__":
    unittest.main()
