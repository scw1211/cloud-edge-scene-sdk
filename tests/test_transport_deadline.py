"""Per-call cloud transport deadlines and retry-budget invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from cloud_edge_framework.contracts import SemanticEvent, build_decision
from cloud_edge_framework.reliable_transport import ReliableHttpCloudClient
from cloud_edge_framework.transport import CloudTransportError, HttpCloudClient


class _Response:
    def __init__(self, payload):
        self.body = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        return False

    def read(self):
        return self.body


class _LegacyPostOverrideClient(HttpCloudClient):
    """Models rolling-compatible clients that override the old `_post` API."""

    def __init__(self):
        super().__init__("http://cloud.invalid")
        self.calls = 0

    def _post(self, path, payload):
        del path, payload
        self.calls += 1
        return {
            **_aggregation_response(),
            "_transport_metrics": {
                "request_bytes": 1,
                "response_bytes": 1,
                "http_round_trip_ms": 1.0,
            },
        }


def _event() -> SemanticEvent:
    return SemanticEvent.from_dict(
        {
            "schema_version": "1.0",
            "event_id": "deadline-event-1",
            "scene": "deadline-test",
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
                "label": "high",
                "confidence": 0.95,
                "probabilities": {"high": 0.95},
            },
            "risk": {"level": "high", "score": 0.9},
            "uncertainty": {
                "confidence": 0.95,
                "calibrated": True,
                "prediction_set": ["high"],
                "method": "fixture",
            },
            "timing": {"deadline_ms": 200.0},
            "evidence": [
                {
                    "evidence_id": "deadline-summary-1",
                    "level": "summary",
                    "modality": "fixture",
                    "encoding": "json",
                    "inline": {"value": 1},
                    "size_bytes": 1,
                    "content_type": "application/json",
                }
            ],
            "candidate_actions": [],
            "metadata": {},
        }
    )


def _artifact_event(directory: str) -> SemanticEvent:
    path = Path(directory) / "summary.json"
    data = b'{"value":1}'
    path.write_bytes(data)
    payload = _event().to_dict()
    evidence = payload["evidence"][0]
    evidence.pop("inline", None)
    evidence["uri"] = path.as_uri()
    evidence["size_bytes"] = len(data)
    evidence["sha256"] = hashlib.sha256(data).hexdigest()
    return SemanticEvent.from_dict(payload)


def _decision_response(event: SemanticEvent):
    decision = build_decision(
        event=event,
        decision="monitor",
        actions=[],
        confidence=0.99,
        reason="deadline transport fixture",
        source="fixture_cloud",
        policy_version="deadline-1",
    )
    return {
        "decision": decision.to_dict(),
        "cloud_runtime_ms": 1.0,
        "cloud_accepted_at_ms": 1001,
    }


def _aggregation_response():
    return {
        "aggregation": {
            "group_id": "group-1",
            "state": "waiting",
            "missing_members": ["edge-b"],
        },
        "coordination": None,
        "cloud_accepted_at_ms": 1001,
    }


def _results_response():
    return {
        "items": [
            {
                "event_id": "deadline-event-1",
                "group_id": "group-1",
                "aggregation": _aggregation_response()["aggregation"],
                "coordination": None,
            }
        ]
    }


class HttpCloudClientDeadlineTest(unittest.TestCase):
    def test_decide_artifact_upload_and_post_share_one_budget(self) -> None:
        clock = [5.0]
        calls = []

        def fake_urlopen(request, timeout):
            method = request.get_method()
            calls.append((method, timeout))
            if method == "PUT":
                clock[0] += 0.04
                return _Response({"status": "stored"})
            clock[0] += 0.01
            return _Response(_decision_response(event))

        client = HttpCloudClient("http://cloud.invalid", timeout_seconds=1.0)
        with tempfile.TemporaryDirectory() as directory:
            event = _artifact_event(directory)
            with patch(
                "cloud_edge_framework.transport.time.monotonic",
                side_effect=lambda: clock[0],
            ), patch(
                "cloud_edge_framework.transport.urlopen",
                side_effect=fake_urlopen,
            ):
                result = client.decide(event, timeout_seconds=0.1)

        self.assertEqual(result.decision, "monitor")
        self.assertEqual([method for method, _ in calls], ["PUT", "POST"])
        self.assertAlmostEqual(calls[0][1], 0.1, places=7)
        self.assertAlmostEqual(calls[1][1], 0.06, places=7)

    def test_aggregate_artifact_upload_consumes_post_budget(self) -> None:
        clock = [8.0]
        calls = []

        def fake_urlopen(request, timeout):
            method = request.get_method()
            calls.append((method, timeout))
            if method == "PUT":
                clock[0] += 0.03
                return _Response({"status": "stored"})
            return _Response(_aggregation_response())

        client = HttpCloudClient("http://cloud.invalid", timeout_seconds=1.0)
        with tempfile.TemporaryDirectory() as directory:
            event = _artifact_event(directory)
            with patch(
                "cloud_edge_framework.transport.time.monotonic",
                side_effect=lambda: clock[0],
            ), patch(
                "cloud_edge_framework.transport.urlopen",
                side_effect=fake_urlopen,
            ):
                client.aggregate(event, timeout_seconds=0.1)

        self.assertEqual([method for method, _ in calls], ["PUT", "POST"])
        self.assertAlmostEqual(calls[0][1], 0.1, places=7)
        self.assertAlmostEqual(calls[1][1], 0.07, places=7)

    def test_aggregate_forwards_explicit_timeout_to_urlopen(self) -> None:
        timeouts = []

        def fake_urlopen(request, timeout):
            del request
            timeouts.append(timeout)
            return _Response(_aggregation_response())

        client = HttpCloudClient("http://cloud.invalid", timeout_seconds=1.5)
        with patch("cloud_edge_framework.transport.urlopen", side_effect=fake_urlopen):
            result = client.aggregate(_event(), timeout_seconds=0.125)

        self.assertEqual(result["aggregation"]["group_id"], "group-1")
        self.assertEqual(len(timeouts), 1)
        self.assertGreater(timeouts[0], 0.0)
        self.assertLessEqual(timeouts[0], 0.125)

    def test_aggregation_results_batch_forwards_remaining_budget(self) -> None:
        timeouts = []

        def fake_urlopen(request, timeout):
            del request
            timeouts.append(timeout)
            return _Response(_results_response())

        client = HttpCloudClient("http://cloud.invalid", timeout_seconds=1.5)
        event = _event()
        with patch("cloud_edge_framework.transport.urlopen", side_effect=fake_urlopen):
            result = client.aggregation_results_batch(
                [event],
                {event.event_id: "group-1"},
                timeout_seconds=0.2,
            )

        self.assertEqual(result["items"][0]["event_id"], event.event_id)
        self.assertEqual(len(timeouts), 1)
        self.assertGreater(timeouts[0], 0.0)
        self.assertLessEqual(timeouts[0], 0.2)

    def test_omitted_timeout_preserves_configured_default(self) -> None:
        timeouts = []

        def fake_urlopen(request, timeout):
            del request
            timeouts.append(timeout)
            return _Response(_aggregation_response())

        client = HttpCloudClient("http://cloud.invalid", timeout_seconds=0.75)
        with patch("cloud_edge_framework.transport.urlopen", side_effect=fake_urlopen):
            client.aggregate(_event())

        self.assertEqual(timeouts, [0.75])

    def test_omitted_timeout_supports_legacy_post_override(self) -> None:
        client = _LegacyPostOverrideClient()

        result = client.aggregate(_event())

        self.assertEqual(result["aggregation"]["group_id"], "group-1")
        self.assertEqual(client.calls, 1)

    def test_base_client_rejects_a_response_after_explicit_budget(self) -> None:
        clock = [5.0]

        def fake_urlopen(request, timeout):
            del request, timeout
            clock[0] += 0.06
            return _Response({"ok": True})

        client = HttpCloudClient("http://cloud.invalid", timeout_seconds=1.0)
        with patch(
            "cloud_edge_framework.transport.time.monotonic",
            side_effect=lambda: clock[0],
        ), patch(
            "cloud_edge_framework.transport.urlopen",
            side_effect=fake_urlopen,
        ):
            with self.assertRaisesRegex(CloudTransportError, "timeout budget"):
                client._post("/deadline", {}, timeout_seconds=0.05)


class ReliableHttpCloudClientDeadlineTest(unittest.TestCase):
    def test_artifact_retries_and_backoff_share_explicit_budget(self) -> None:
        clock = [4.0]
        timeouts = []
        attempts = [0]

        def fake_urlopen(request, timeout):
            del request
            attempts[0] += 1
            timeouts.append(timeout)
            if attempts[0] == 1:
                clock[0] += 0.04
                raise URLError("temporary artifact outage")
            clock[0] += 0.01
            return _Response({"status": "stored"})

        client = ReliableHttpCloudClient(
            "http://cloud.invalid",
            timeout_seconds=0.5,
            max_attempts=3,
            retry_backoff_seconds=0.02,
        )
        with patch(
            "cloud_edge_framework.reliable_transport.time.monotonic",
            side_effect=lambda: clock[0],
        ), patch(
            "cloud_edge_framework.transport.time.monotonic",
            side_effect=lambda: clock[0],
        ), patch(
            "cloud_edge_framework.reliable_transport.time.sleep",
            side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        ), patch(
            "cloud_edge_framework.transport.urlopen",
            side_effect=fake_urlopen,
        ):
            result = client._put_bytes(
                "/api/v1/evidence/hash",
                b"artifact",
                {"Content-Type": "application/octet-stream"},
                timeout_seconds=0.1,
            )

        self.assertEqual(result["attempts"], 2)
        self.assertAlmostEqual(timeouts[0], 0.1, places=7)
        self.assertAlmostEqual(timeouts[1], 0.04, places=7)

    def test_retries_and_backoff_share_explicit_total_budget(self) -> None:
        clock = [10.0]
        timeouts = []
        sleeps = []
        attempts = [0]

        def monotonic():
            return clock[0]

        def sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds

        def fake_urlopen(request, timeout):
            del request
            attempts[0] += 1
            timeouts.append(timeout)
            if attempts[0] == 1:
                clock[0] += 0.04
                raise URLError("temporary outage")
            clock[0] += 0.01
            return _Response({"ok": True})

        client = ReliableHttpCloudClient(
            "http://cloud.invalid",
            timeout_seconds=0.5,
            max_attempts=3,
            retry_backoff_seconds=0.02,
        )
        with patch(
            "cloud_edge_framework.reliable_transport.time.monotonic",
            side_effect=monotonic,
        ), patch(
            "cloud_edge_framework.reliable_transport.time.sleep",
            side_effect=sleep,
        ), patch(
            "cloud_edge_framework.reliable_transport.urlopen",
            side_effect=fake_urlopen,
        ):
            result = client._post("/deadline", {}, timeout_seconds=0.1)

        self.assertTrue(result["ok"])
        self.assertEqual(result["_transport_metrics"]["attempts"], 2)
        self.assertEqual(sleeps, [0.02])
        self.assertAlmostEqual(timeouts[0], 0.1, places=7)
        self.assertAlmostEqual(timeouts[1], 0.04, places=7)

    def test_retry_stops_when_backoff_cannot_fit_remaining_budget(self) -> None:
        clock = [20.0]
        timeouts = []
        sleeps = []

        def fake_urlopen(request, timeout):
            del request
            timeouts.append(timeout)
            clock[0] += 0.04
            raise URLError("temporary outage")

        client = ReliableHttpCloudClient(
            "http://cloud.invalid",
            timeout_seconds=0.5,
            max_attempts=3,
            retry_backoff_seconds=0.02,
        )
        with patch(
            "cloud_edge_framework.reliable_transport.time.monotonic",
            side_effect=lambda: clock[0],
        ), patch(
            "cloud_edge_framework.reliable_transport.time.sleep",
            side_effect=lambda seconds: sleeps.append(seconds),
        ), patch(
            "cloud_edge_framework.reliable_transport.urlopen",
            side_effect=fake_urlopen,
        ):
            with self.assertRaisesRegex(CloudTransportError, "timeout budget"):
                client._post("/deadline", {}, timeout_seconds=0.05)

        self.assertEqual(len(timeouts), 1)
        self.assertEqual(sleeps, [])

    def test_late_success_is_rejected_after_total_budget(self) -> None:
        clock = [30.0]

        def fake_urlopen(request, timeout):
            del request, timeout
            clock[0] += 0.06
            return _Response({"ok": True})

        client = ReliableHttpCloudClient(
            "http://cloud.invalid",
            timeout_seconds=0.5,
            max_attempts=2,
            retry_backoff_seconds=0.01,
        )
        with patch(
            "cloud_edge_framework.reliable_transport.time.monotonic",
            side_effect=lambda: clock[0],
        ), patch(
            "cloud_edge_framework.reliable_transport.urlopen",
            side_effect=fake_urlopen,
        ):
            with self.assertRaisesRegex(CloudTransportError, "timeout budget"):
                client._post("/deadline", {}, timeout_seconds=0.05)

    def test_omitted_timeout_keeps_per_attempt_default(self) -> None:
        timeouts = []
        attempts = [0]

        def fake_urlopen(request, timeout):
            del request
            attempts[0] += 1
            timeouts.append(timeout)
            if attempts[0] == 1:
                raise URLError("temporary outage")
            return _Response({"ok": True})

        client = ReliableHttpCloudClient(
            "http://cloud.invalid",
            timeout_seconds=0.25,
            max_attempts=2,
            retry_backoff_seconds=0.01,
        )
        with patch(
            "cloud_edge_framework.reliable_transport.time.sleep",
            return_value=None,
        ), patch(
            "cloud_edge_framework.reliable_transport.urlopen",
            side_effect=fake_urlopen,
        ):
            result = client._post("/deadline", {})

        self.assertTrue(result["ok"])
        self.assertEqual(timeouts, [0.25, 0.25])


if __name__ == "__main__":
    unittest.main()
