"""Compatibility contracts for rolling cloud/edge framework upgrades."""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import time
import unittest

from cloud_edge_framework.cloud_service import CloudApiService
from cloud_edge_framework.contracts import DecisionEnvelope
from cloud_edge_framework.reliability import SQLiteIdempotencyStore
from cloud_edge_framework.reliable_transport import ReliableHttpCloudClient
from cloud_edge_framework.runtime import EdgeRuntime
from cloud_edge_framework.transport import CloudTransportError, HttpCloudClient


def _decision() -> DecisionEnvelope:
    return DecisionEnvelope(
        decision_id="compat-decision",
        event_ids=["compat-event"],
        scene="compat",
        decision="monitor",
        risk_level="low",
        confidence=0.9,
        route="cloud_async",
        status="final",
        actions=[],
        reason="compatibility test",
        policy_version="compat-1",
        metadata={},
    )


class _LegacyEvidencePlanner:
    def __init__(self) -> None:
        self.calls = 0

    def plan(self, event, conflict_suspected=False):
        del event, conflict_suspected
        self.calls += 1
        return "legacy-plan"


class _LegacyReviewTracker:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, event_id, decision, completion_mode):
        self.calls.append((event_id, decision.decision_id, completion_mode))
        return {"event_id": event_id}


class _CloudRuntimeStub:
    def __init__(self) -> None:
        self.decide_calls = 0
        self.coordinate_calls = 0

    def decide_payload(self, event):
        del event
        self.decide_calls += 1
        return {"decision": _decision().to_dict()}

    def coordinate_payloads(self, events):
        del events
        self.coordinate_calls += 1
        return {"decisions": [_decision().to_dict()], "globally_consistent": True}


class _CloudSnapshot:
    def __init__(self, runtime):
        self.runtime = runtime

    def require_cloud(self):
        return self.runtime


class _CloudManager:
    def __init__(self, runtime):
        self.runtime = runtime

    @contextmanager
    def lease(self):
        yield _CloudSnapshot(self.runtime)


class _MetricsStub:
    def record_cloud_request(self, *args):
        del args

    def record_coordination_result(self, *args):
        del args


class _TransportEvent:
    def __init__(self, event_id: str) -> None:
        self.event_id = event_id
        self.metadata = {"trace_id": "trace-" + event_id}

    def to_dict(self, include_scene_payload: bool = False):
        del include_scene_payload
        return {
            "event_id": self.event_id,
            "metadata": dict(self.metadata),
        }


class _LegacyAggregateClient(HttpCloudClient):
    def __init__(self) -> None:
        self.batch_calls = 0
        self.single_calls = 0

    def _materialize_artifacts(self, event):
        return event, {
            "artifact_count": 0,
            "artifact_uploaded_count": 0,
            "artifact_request_bytes": 0,
            "artifact_response_bytes": 0,
            "artifact_upload_ms": 0.0,
        }

    def _post(self, path, payload):
        del payload
        self.batch_calls += 1
        raise CloudTransportError(
            "cloud returned HTTP 404: missing {}".format(path)
        )

    def aggregate(self, event):
        self.single_calls += 1
        return {
            "aggregation": {"state": "waiting"},
            "coordination": None,
            "transport": {
                "request_bytes": 10,
                "response_bytes": 20,
                "http_round_trip_ms": 1.5,
            },
        }


class _CompactBatchClient(_LegacyAggregateClient):
    def _post(self, path, payload):
        self.batch_calls += 1
        self.last_path = path
        self.last_payload = payload
        return {
            "items": [
                {"event_id": event["event_id"], "group_id": "group-1"}
                for event in payload["events"]
            ],
            "groups": [
                {
                    "group_id": "group-1",
                    "aggregation": {"state": "completed"},
                    "coordination": {"decisions": []},
                }
            ],
            "_transport_metrics": {
                "request_bytes": 30,
                "response_bytes": 40,
                "http_round_trip_ms": 2.0,
            },
        }


class ProtocolCompatibilityTest(unittest.TestCase):
    def test_batch_request_identity_is_order_independent(self) -> None:
        first = {
            "events": [
                {"event_id": "event-b", "metadata": {}},
                {"event_id": "event-a", "metadata": {"trace_id": "trace-a"}},
            ]
        }
        second = {"events": list(reversed(first["events"]))}

        first_key, first_trace = ReliableHttpCloudClient._request_identity(
            "/api/v1/collaboration/aggregate/batch", first
        )
        second_key, second_trace = ReliableHttpCloudClient._request_identity(
            "/api/v1/collaboration/aggregate/batch", second
        )

        self.assertEqual(first_key, second_key)
        self.assertIn(first_trace, {"", "trace-a"})
        self.assertIn(second_trace, {"", "trace-a"})

    def test_new_edge_falls_back_to_legacy_single_aggregate_endpoint(self) -> None:
        client = _LegacyAggregateClient()
        result = client.aggregate_batch(
            [_TransportEvent("event-a"), _TransportEvent("event-b")],
            wait_seconds=0.15,
        )

        self.assertTrue(result["fallback_single_event"])
        self.assertEqual(client.batch_calls, 1)
        self.assertEqual(client.single_calls, 2)
        self.assertEqual(
            [item["event_id"] for item in result["items"]],
            ["event-a", "event-b"],
        )
        self.assertEqual(result["transport"]["request_bytes"], 20)
        self.assertEqual(result["transport"]["response_bytes"], 40)

    def test_compact_group_response_is_expanded_only_inside_edge_client(self) -> None:
        client = _CompactBatchClient()
        result = client.aggregate_batch(
            [_TransportEvent("event-a"), _TransportEvent("event-b")],
            wait_seconds=0.15,
        )

        self.assertEqual(client.batch_calls, 1)
        self.assertEqual(client.last_payload["wait_ms"], 0)
        self.assertEqual(len(result["groups"]), 1)
        for item in result["items"]:
            self.assertEqual(item["aggregation"]["state"], "completed")
            self.assertEqual(item["coordination"], {"decisions": []})
        self.assertEqual(result["transport"]["request_bytes"], 30)
        self.assertEqual(result["transport"]["response_bytes"], 40)

    def test_old_complete_aggregation_response_is_still_authoritative(self) -> None:
        response = {
            "aggregation": {
                "state": "completed",
                "completion_reason": "all_expected_members",
                "missing_members": [],
            },
            "coordination": {"globally_consistent": True},
        }
        self.assertTrue(EdgeRuntime._aggregation_evidence_complete(response))
        self.assertTrue(EdgeRuntime._aggregation_globally_confirmed(response))

        response["aggregation"]["evidence_complete"] = False
        self.assertFalse(EdgeRuntime._aggregation_evidence_complete(response))
        self.assertFalse(EdgeRuntime._aggregation_globally_confirmed(response))

    def test_v1_injected_extension_signatures_remain_callable(self) -> None:
        planner = _LegacyEvidencePlanner()
        planned = EdgeRuntime._plan_evidence(
            planner,
            object(),
            False,
            {"required_level": "summary"},
        )
        self.assertEqual(planned, "legacy-plan")
        self.assertEqual(planner.calls, 1)

        tracker = _LegacyReviewTracker()
        completed = EdgeRuntime._complete_review(
            tracker,
            "compat-event",
            _decision(),
            "aggregation_timeout",
            "local_only_timeout",
        )
        self.assertEqual(completed["event_id"], "compat-event")
        self.assertEqual(len(tracker.calls), 1)

    def test_idempotent_cloud_retries_keep_first_acceptance_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = _CloudRuntimeStub()
            service = CloudApiService.__new__(CloudApiService)
            service.manager = _CloudManager(runtime)
            service.idempotency = SQLiteIdempotencyStore(
                Path(directory) / "idempotency.sqlite3",
                ttl_seconds=60.0,
                max_entries=100,
            )
            service.metrics = _MetricsStub()
            headers = {"idempotency-key": "compat-decision-request"}
            payload = {
                "event": {
                    "event_id": "compat-event",
                    "metadata": {"trace_id": "compat-trace"},
                }
            }

            first = service.cloud_decision(payload, headers)
            time.sleep(0.01)
            second = service.cloud_decision(payload, headers)

            self.assertFalse(first["idempotency_replay"])
            self.assertTrue(second["idempotency_replay"])
            self.assertEqual(
                second["cloud_accepted_at_ms"],
                first["cloud_accepted_at_ms"],
            )
            self.assertEqual(runtime.decide_calls, 1)

            coordinate_headers = {"idempotency-key": "compat-coordinate-request"}
            coordinate_payload = {"events": [payload["event"]]}
            coordinated_first = service.coordinate(
                coordinate_payload, coordinate_headers
            )
            time.sleep(0.01)
            coordinated_second = service.coordinate(
                coordinate_payload, coordinate_headers
            )
            self.assertEqual(
                coordinated_second["cloud_accepted_at_ms"],
                coordinated_first["cloud_accepted_at_ms"],
            )
            self.assertEqual(runtime.coordinate_calls, 1)


class IdempotencyConcurrencyTest(unittest.TestCase):
    def test_completed_response_survives_store_recreation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "idempotency.sqlite3"
            first_store = SQLiteIdempotencyStore(
                path,
                ttl_seconds=60.0,
                max_entries=100,
            )
            first_value, first_replayed = first_store.execute(
                "persistent-key",
                {"value": 1},
                lambda: {"result": 7},
            )
            first_store.close()

            calls = 0

            def operation():
                nonlocal calls
                calls += 1
                return {"result": 8}

            second_store = SQLiteIdempotencyStore(
                path,
                ttl_seconds=60.0,
                max_entries=100,
            )
            second_value, second_replayed = second_store.execute(
                "persistent-key",
                {"value": 1},
                operation,
            )
            second_store.close()

            self.assertEqual(first_value, {"result": 7})
            self.assertFalse(first_replayed)
            self.assertEqual(second_value, first_value)
            self.assertTrue(second_replayed)
            self.assertEqual(calls, 0)

    def test_different_request_keys_do_not_serialize_operations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteIdempotencyStore(
                Path(directory) / "idempotency.sqlite3",
                ttl_seconds=60.0,
                max_entries=100,
            )
            state_lock = threading.Lock()
            release = threading.Event()
            both_started = threading.Event()
            active = 0

            def operation(name):
                def run():
                    nonlocal active
                    with state_lock:
                        active += 1
                        if active == 2:
                            both_started.set()
                    release.wait(2.0)
                    with state_lock:
                        active -= 1
                    return {"name": name}

                return run

            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(
                    store.execute, "key-a", {"value": "a"}, operation("a")
                )
                second = executor.submit(
                    store.execute, "key-b", {"value": "b"}, operation("b")
                )
                try:
                    self.assertTrue(both_started.wait(1.0))
                finally:
                    release.set()
                self.assertFalse(first.result(timeout=2.0)[1])
                self.assertFalse(second.result(timeout=2.0)[1])

    def test_same_request_key_still_executes_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteIdempotencyStore(
                Path(directory) / "idempotency.sqlite3",
                ttl_seconds=60.0,
                max_entries=100,
            )
            entered = threading.Event()
            release = threading.Event()
            calls = 0

            def operation():
                nonlocal calls
                calls += 1
                entered.set()
                release.wait(2.0)
                return {"value": 7}

            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(
                    store.execute, "same-key", {"value": 1}, operation
                )
                self.assertTrue(entered.wait(1.0))
                second = executor.submit(
                    store.execute, "same-key", {"value": 1}, operation
                )
                time.sleep(0.05)
                self.assertEqual(calls, 1)
                release.set()
                first_value, first_replayed = first.result(timeout=2.0)
                second_value, second_replayed = second.result(timeout=2.0)

            self.assertEqual(first_value, {"value": 7})
            self.assertEqual(second_value, first_value)
            self.assertFalse(first_replayed)
            self.assertTrue(second_replayed)
            self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
