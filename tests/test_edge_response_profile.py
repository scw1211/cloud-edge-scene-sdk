"""Response projection and idempotency behavior at edge HTTP ingress."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest

from cloud_edge_framework.edge_service import EdgeApiService
from cloud_edge_framework.metrics import FrameworkMetrics
from cloud_edge_framework.reliability import (
    IdempotencyConflictError,
    SQLiteIdempotencyStore,
)
from cloud_edge_framework.scheduling import NetworkSnapshot


class _Runtime:
    def __init__(self) -> None:
        self.calls = []

    def process(self, event, network, **controls):
        del event, network
        detail = controls["response_detail"]
        self.calls.append(dict(controls))
        return {
            "response_detail": detail,
            "schedule": {"route": "cloud_sync"},
            "final_decision": {
                "route": "cloud_sync",
                "metadata": {
                    "transport": {
                        "http_round_trip_ms": 3.5,
                        "request_bytes": 100,
                        "response_bytes": 80,
                        "attempts": 1,
                    }
                },
            },
            "data_plane": {
                "selected_request_bytes": 100,
                "actual_transport_request_bytes": 100,
                "actual_artifact_request_bytes": 0,
            },
            "closed_loop_accounting": {
                "edge_preliminary_decision_ms": 2.0,
                "accounted_closed_loop_ms": 6.0,
                "synchronous_cloud_closed_loop_ms": 6.0,
            },
            "framework_runtime_ms": 4.0,
        }


class _Snapshot:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime

    def require_edge(self):
        return self.runtime


class _Manager:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime

    @contextmanager
    def lease(self):
        yield _Snapshot(self.runtime)


class _Network:
    def snapshot(self):
        return NetworkSnapshot()


class _Replay:
    def __init__(self) -> None:
        self.notifications = 0

    def notify(self):
        self.notifications += 1


def _service(path: Path):
    service = object.__new__(EdgeApiService)
    runtime = _Runtime()
    service.manager = _Manager(runtime)
    service.network_monitor = _Network()
    service.replay_worker = _Replay()
    service.metrics = FrameworkMetrics("edge-test")
    service.idempotency = SQLiteIdempotencyStore(
        path,
        ttl_seconds=60.0,
        max_entries=100,
    )
    return service, runtime


class EdgeResponseProfileTest(unittest.TestCase):
    def test_minimal_profile_replays_and_profile_change_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, runtime = _service(Path(directory) / "idempotency.sqlite3")
            payload = {"event": {"id": "event-1"}}
            headers = {
                "idempotency-key": "request-1",
                "prefer": "return=minimal",
            }
            try:
                first = service.decide(payload, headers)
                replay = service.decide(payload, headers)

                self.assertEqual(first["response_detail"], "compact")
                self.assertFalse(first["idempotency_replay"])
                self.assertTrue(replay["idempotency_replay"])
                self.assertEqual(len(runtime.calls), 1)
                self.assertEqual(runtime.calls[0]["response_detail"], "compact")

                with self.assertRaises(IdempotencyConflictError):
                    service.decide(
                        payload,
                        {
                            "idempotency-key": "request-1",
                            "x-response-detail": "full",
                        },
                    )
            finally:
                service.idempotency.close()

    def test_compact_result_still_records_transport_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, _runtime = _service(Path(directory) / "idempotency.sqlite3")
            try:
                service.decide(
                    {"event": {"id": "event-2"}},
                    {
                        "idempotency-key": "request-2",
                        "x-response-detail": "compact",
                    },
                )
                metrics = service.metrics.snapshot()
                distributions = metrics["distributions"]
                self.assertEqual(distributions["http_round_trip_ms"]["count"], 1)
                self.assertEqual(distributions["http_response_bytes"]["mean"], 80.0)
            finally:
                service.idempotency.close()


if __name__ == "__main__":
    unittest.main()
