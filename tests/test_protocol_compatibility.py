"""Compatibility contracts for rolling cloud/edge framework upgrades."""

from contextlib import contextmanager
from pathlib import Path
import tempfile
import time
import unittest

from cloud_edge_framework.cloud_service import CloudApiService
from cloud_edge_framework.contracts import DecisionEnvelope
from cloud_edge_framework.reliability import SQLiteIdempotencyStore
from cloud_edge_framework.runtime import EdgeRuntime


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


class ProtocolCompatibilityTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
