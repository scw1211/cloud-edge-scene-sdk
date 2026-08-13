"""真实故障代理与弱网业务保持率判据测试。"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRAFFIC_ROOT = REPOSITORY_ROOT / "scenes" / "freeway_traffic"
for import_root in (REPOSITORY_ROOT, TRAFFIC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from benchmark_weak_network_retention import (  # noqa: E402
    _wait_authoritative_reviews,
    evaluate_business_event,
    evaluate_recovery,
)
from traffic_system.network_fault_proxy import (  # noqa: E402
    CONTROL_PROFILE_PATH,
    CONTROL_STATUS_PATH,
    FaultProxyHTTPServer,
    FaultState,
    build_handler,
)


def _post_json(url, value, headers=None):
    body = json.dumps(value).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(dict(headers or {}))
    request = Request(
        url,
        data=body,
        headers=request_headers,
        method="POST",
    )
    with urlopen(request, timeout=2.0) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


class _BackendHandler(BaseHTTPRequestHandler):
    captured = []

    def log_message(self, fmt, *args):
        del fmt, args

    def do_GET(self):
        body = json.dumps({"status": "ok", "path": self.path}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        received = self.rfile.read(length)
        self.__class__.captured.append(
            {
                "headers": {name.lower(): value for name, value in self.headers.items()},
                "body": received,
            }
        )
        body = json.dumps({"accepted": True}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FaultProxyTests(unittest.TestCase):
    def test_profile_switch_is_seeded_and_counters_are_explicit(self):
        state = FaultState("severe", seed=41, log_path=None)
        first = [state.sample() for _ in range(5)]
        state.set_profile("normal")
        normal = state.sample()
        self.assertEqual("normal", normal["profile"])
        self.assertFalse(normal["dropped"])
        self.assertEqual(0.0, normal["delay_ms"])
        state.set_profile("severe")
        repeated = [state.sample() for _ in range(5)]
        self.assertEqual(
            [(row["dropped"], row["delay_ms"]) for row in first],
            [(row["dropped"], row["delay_ms"]) for row in repeated],
        )
        self.assertEqual(5, repeated[-1]["profile_request_id"])
        snapshot = state.snapshot()
        self.assertEqual(2, snapshot["switch_count"])
        self.assertEqual(5, snapshot["profile_request_count"])
        self.assertGreater(snapshot["profile_delay_ms_total"], 0.0)

    def test_control_is_out_of_band_and_required_headers_are_forwarded(self):
        _BackendHandler.captured = []
        backend = ThreadingHTTPServer(("127.0.0.1", 0), _BackendHandler)
        backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
        backend_thread.start()
        state = FaultState("normal", seed=7, log_path=None)
        proxy = FaultProxyHTTPServer(
            ("127.0.0.1", 0),
            build_handler(
                state,
                "http://127.0.0.1:{}".format(backend.server_port),
                0.01,
                1.0,
                "secret",
            ),
        )
        proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        proxy_thread.start()
        self.assertEqual(128, type(proxy).request_queue_size)
        proxy_url = "http://127.0.0.1:{}".format(proxy.server_port)
        try:
            status, result = _post_json(
                proxy_url + "/api/v1/collaboration/aggregate/batch",
                {"items": []},
                {
                    "Idempotency-Key": "id-1",
                    "X-Trace-ID": "trace-1",
                    "X-Request-ID": "request-1",
                    "Prefer": "respond-async",
                    "X-Response-Detail": "compact",
                },
            )
            self.assertEqual(200, status)
            self.assertTrue(result["accepted"])
            headers = _BackendHandler.captured[-1]["headers"]
            self.assertEqual("id-1", headers["idempotency-key"])
            self.assertEqual("trace-1", headers["x-trace-id"])
            self.assertEqual("request-1", headers["x-request-id"])
            self.assertEqual("respond-async", headers["prefer"])
            self.assertEqual("compact", headers["x-response-detail"])

            status_request = Request(
                proxy_url + CONTROL_STATUS_PATH,
                headers={"X-Fault-Control-Token": "secret"},
                method="GET",
            )
            with urlopen(status_request, timeout=2.0) as response:
                before = json.loads(response.read().decode("utf-8"))
            self.assertEqual(1, before["request_count"])
            self.assertEqual(
                1,
                before["profile_request_class_counts"]["summary_data_plane"],
            )
            status, switched = _post_json(
                proxy_url + CONTROL_PROFILE_PATH,
                {"profile": "outage"},
                {"X-Fault-Control-Token": "secret"},
            )
            self.assertEqual(200, status)
            self.assertEqual("outage", switched["profile"])
            self.assertEqual(1, switched["request_count"])

            with self.assertRaises(HTTPError) as raised:
                _post_json(
                    proxy_url + CONTROL_PROFILE_PATH,
                    {"profile": "normal"},
                    {"X-Fault-Control-Token": "wrong"},
                )
            self.assertEqual(403, raised.exception.code)
        finally:
            proxy.shutdown()
            proxy.server_close()
            backend.shutdown()
            backend.server_close()
            proxy_thread.join(timeout=2.0)
            backend_thread.join(timeout=2.0)


def _event(**overrides):
    value = {
        "status": "provisional",
        "route": "cloud_async",
        "policy_route": "cloud_async",
        "policy_waits_for_cloud": False,
        "input_to_response_ms": 80.0,
        "local_actions_authorized": True,
        "immediate_action_types": ["monitor"],
        "deferred_action_types": [],
        "cloud_confirmed": False,
        "local_autonomy": False,
        "summary_delivery_required": True,
        "summary_persistence_stage": "handoff_durable",
    }
    value.update(overrides)
    return value


def _review(**overrides):
    value = {
        "state": "completed",
        "completion_stage": "lightweight_final",
        "requested_route": "cloud_async",
        "final_decision": {
            "status": "final",
            "decision": "monitor",
            "metadata": {
                "action_authorization": {"cloud_confirmed": True}
            },
        },
    }
    value.update(overrides)
    return value


class BusinessRetentionTests(unittest.TestCase):
    def test_pure_deferred_action_is_safe_when_authoritative_final_arrives(self):
        event = _event(
            policy_route="cloud_sync",
            policy_waits_for_cloud=True,
            local_actions_authorized=False,
            immediate_action_types=[],
            deferred_action_types=["regional_coordination"],
        )
        result = evaluate_business_event(
            "severe",
            event,
            _review(requested_route="cloud_sync"),
            165.0,
            200.0,
        )
        self.assertTrue(result["success"])
        self.assertEqual("authoritative_final", result["business_endpoint"])

    def test_all_immediate_actions_require_and_accept_local_authorization(self):
        result = evaluate_business_event(
            "mild",
            _event(
                local_actions_authorized=True,
                immediate_action_types=["monitor", "traffic_advisory"],
                deferred_action_types=[],
            ),
            None,
            None,
            200.0,
        )
        self.assertTrue(result["success"])
        self.assertEqual("local_response", result["business_endpoint"])

    def test_safe_immediate_subset_is_not_invalidated_by_deferred_action(self):
        result = evaluate_business_event(
            "severe",
            _event(
                policy_route="cloud_sync",
                policy_waits_for_cloud=True,
                local_actions_authorized=False,
                immediate_action_types=["variable_speed_limit"],
                deferred_action_types=["regional_coordination"],
            ),
            _review(requested_route="cloud_sync"),
            165.0,
            200.0,
        )
        self.assertTrue(result["success"])
        self.assertEqual("authoritative_final", result["business_endpoint"])

    def test_cloud_async_does_not_wait_for_a_deferred_cloud_only_action(self):
        result = evaluate_business_event(
            "severe",
            _event(
                policy_route="cloud_async",
                policy_waits_for_cloud=False,
                local_actions_authorized=False,
                immediate_action_types=["traffic_advisory"],
                deferred_action_types=["regional_coordination"],
            ),
            None,
            None,
            200.0,
        )
        self.assertTrue(result["success"])
        self.assertEqual("local_response", result["business_endpoint"])

    def test_offline_local_autonomy_is_a_business_success_without_final(self):
        result = evaluate_business_event(
            "outage",
            _event(
                route="local_autonomy",
                policy_route="local_autonomy",
                local_autonomy=True,
            ),
            {"state": "queued", "requested_route": "local_autonomy"},
            None,
            200.0,
        )
        self.assertTrue(result["success"])
        self.assertEqual("offline_local_autonomy", result["business_endpoint"])

    def test_offline_result_fails_if_summary_is_not_durable(self):
        result = evaluate_business_event(
            "outage",
            _event(
                route="local_autonomy",
                policy_route="local_autonomy",
                local_autonomy=True,
                summary_persistence_stage="not_required",
            ),
            {"state": "queued", "requested_route": "local_autonomy"},
            None,
            200.0,
        )
        self.assertFalse(result["success"])
        self.assertIn("summary_was_not_durably_queued", result["reasons"])

    def test_online_deferred_action_requires_observed_authoritative_final(self):
        event = _event(
            policy_route="cloud_sync",
            policy_waits_for_cloud=True,
            local_actions_authorized=False,
            immediate_action_types=[],
            deferred_action_types=["regional_coordination"],
        )
        missing = evaluate_business_event("severe", event, None, None, 200.0)
        self.assertFalse(missing["success"])
        self.assertIn("authoritative_final_missing", missing["reasons"])
        completed = evaluate_business_event(
            "severe", event, _review(requested_route="cloud_sync"), 165.0, 200.0
        )
        self.assertTrue(completed["success"])
        self.assertEqual("authoritative_final", completed["business_endpoint"])

    def test_review_polling_ignores_partial_and_waits_for_authoritative_final(self):
        partial = _review(completion_stage="partial_final")
        authoritative = _review(completion_stage="lightweight_final")
        with patch(
            "benchmark_weak_network_retention._get_json",
            side_effect=[partial, authoritative],
        ) as get_json:
            reviews, observed = _wait_authoritative_reviews(
                "http://127.0.0.1:1",
                ["event-1"],
                timeout_seconds=0.1,
                poll_seconds=0.0,
                request_timeout_seconds=0.01,
            )
        self.assertEqual(2, get_json.call_count)
        self.assertEqual(authoritative, reviews["event-1"])
        self.assertIn("event-1", observed)

    def test_review_polling_treats_socket_timeout_as_missing_before_deadline(self):
        authoritative = _review(completion_stage="lightweight_final")
        with patch(
            "benchmark_weak_network_retention._get_json",
            side_effect=[socket.timeout("timed out"), authoritative],
        ) as get_json:
            reviews, observed = _wait_authoritative_reviews(
                "http://127.0.0.1:1",
                ["event-1"],
                timeout_seconds=0.1,
                poll_seconds=0.0,
                request_timeout_seconds=0.01,
            )
        self.assertEqual(2, get_json.call_count)
        self.assertEqual(authoritative, reviews["event-1"])
        self.assertIn("event-1", observed)

    def test_recovery_requires_drained_outbox_and_complete_four_member_groups(self):
        outbox = {
            "active": 0,
            "reconciliation": {"active": 0},
            "durable_handoff": {"pending": 0, "durable_pending_count": 0},
        }
        event_ids = ["event-{}".format(value) for value in range(8)]
        reviews = {
            event_id: _review(
                final_decision={
                    "status": "final",
                    "decision": "monitor",
                    "metadata": {
                        "action_authorization": {"cloud_confirmed": True},
                        "aggregation": {
                            "group_id": "group-{}".format(index // 4)
                        }
                    },
                }
            )
            for index, event_id in enumerate(event_ids)
        }
        groups = [
            {
                "group_id": "group-{}".format(value),
                "state": "completed",
                "completion_reason": "all_expected_members",
                "evidence_complete": True,
                "finality": "final",
                "global_confirmation": True,
                "expected_members": ["edge_node_{}".format(item) for item in range(4)],
                "received_members": ["edge_node_{}".format(item) for item in range(4)],
                "missing_members": [],
            }
            for value in range(2)
        ]
        result = evaluate_recovery(
            [outbox] * 4,
            reviews,
            groups,
            event_ids,
            expected_sample_count=2,
            expected_sample_by_event={
                event_id: index // 4 for index, event_id in enumerate(event_ids)
            },
        )
        self.assertTrue(result["passed"])

        partial = dict(groups[0])
        partial["global_confirmation"] = False
        failed = evaluate_recovery(
            [outbox] * 4,
            reviews,
            [partial, groups[1]],
            event_ids,
            expected_sample_count=2,
            expected_sample_by_event={
                event_id: index // 4 for index, event_id in enumerate(event_ids)
            },
        )
        self.assertFalse(failed["passed"])
        self.assertIn("aggregation_not_globally_confirmed", failed["reasons"])

        handoff_pending = {
            **outbox,
            "durable_handoff": {"pending": 1, "durable_pending_count": 1},
        }
        failed = evaluate_recovery(
            [handoff_pending, outbox, outbox, outbox],
            reviews,
            groups,
            event_ids,
            expected_sample_count=2,
            expected_sample_by_event={
                event_id: index // 4 for index, event_id in enumerate(event_ids)
            },
        )
        self.assertFalse(failed["passed"])
        self.assertIn("edge_0_durable_handoff_not_drained", failed["reasons"])


if __name__ == "__main__":
    unittest.main()
