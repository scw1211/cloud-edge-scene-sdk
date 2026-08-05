from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch

from cloud_edge_framework.http_api import build_role_handler
from scenes.freeway_traffic import benchmark_real_current_state_e2e as benchmark


def _native():
    return {
        "sample_id": 7,
        "partition_id": 0,
        "region_summary": {
            "region_risk_level": "low",
            "max_node_risk_level": "severe",
        },
        "upload_required": True,
        "upload_level": "regional_context",
    }


def _post():
    return {
        "event_id": "event-7-0",
        "dispatch_ms": 3.0,
        "http_wall_ms": 12.0,
        "response_at_ms": 20.0,
        "request_bytes": 100,
        "response_bytes": 50,
        "response": {
            "schedule": {
                "route": "cloud_async",
                "reason": "summary delivery is asynchronous",
                "waits_for_cloud": False,
                "critical": False,
                "uncertain": False,
            },
            "final_decision": {
                "status": "provisional",
                "route": "cloud_async",
                "decision": "reroute",
                "metadata": {
                    "action_authorization": {
                        "cloud_confirmed": False,
                        "deferred_action_types": ["reroute"],
                    }
                },
            },
            "summary_delivery": {
                "mode": "background_handoff",
                "persistence_stage": "handoff_durable",
                "fast_path": True,
            },
            "data_plane": {"selected_request_bytes": 80},
        },
    }


def _review(stage):
    return {
        "state": "completed",
        "completion_mode": "replay",
        "completion_stage": stage,
        "completed_at_ms": 1120,
        "local_decision": {
            "decision": "reroute",
            "metadata": {
                "source": "edge_qwen_single_token",
                "edge_decision_path": "edge_qwen",
                "edge_llm_selected": True,
                "edge_llm_requires_cloud": True,
                "edge_llm_safety_fallback": False,
                "operational_safety_risk": {
                    "level": "high",
                    "source": "candidate_action_consequence_policy",
                },
                "action_authorization": {
                    "cloud_confirmed": False,
                    "deferred_action_types": ["reroute"],
                },
            },
        },
        "final_decision": {
            "status": "final",
            "route": "cloud_async",
            "decision": "no_action",
            "metadata": {
                "action_authorization": {
                    "cloud_confirmed": True,
                    "deferred_action_types": [],
                },
                "aggregation": {
                    "group_id": "group-7",
                    "state": "completed",
                    "evidence_complete": True,
                },
            },
        },
    }


class _ConcurrentRoleService:
    role = "edge"

    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def handle_post(self, path, payload, headers):
        del headers
        with self.lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.03)
            return {"path": path, "value": payload["value"]}
        finally:
            with self.lock:
                self.active -= 1

    def handle_get(self, path, headers):
        del headers
        return {"path": path}

    def record_failure(self, method, path):
        raise AssertionError("unexpected failure: {} {}".format(method, path))


class _IdempotentRoleService:
    role = "edge"

    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0
        self.operations = 0
        self.idempotency_keys = []
        self.responses = {}

    def handle_post(self, path, payload, headers):
        key = str(headers.get("idempotency-key", ""))
        with self.lock:
            self.calls += 1
            self.idempotency_keys.append(key)
            if key not in self.responses:
                self.operations += 1
                self.responses[key] = {
                    "path": path,
                    "value": payload["value"],
                    "operation": self.operations,
                }
            return dict(self.responses[key])

    def handle_get(self, path, headers):
        del headers
        return {"path": path}

    def record_failure(self, method, path):
        raise AssertionError("unexpected failure: {} {}".format(method, path))


class _DropFirstRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    request_count = 0
    idempotency_keys = []
    state_lock = threading.Lock()

    def log_message(self, fmt, *args):
        del fmt, args

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        with type(self).state_lock:
            type(self).request_count += 1
            request_number = type(self).request_count
            type(self).idempotency_keys.append(self.headers.get("Idempotency-Key"))
        if request_number == 1:
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
            return
        value = json.loads(body.decode("utf-8"))
        response_body = json.dumps({"value": value["value"]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


class Pems08E2EBenchmarkTests(unittest.TestCase):
    def test_congestion_coverage_flag_keeps_legacy_cli_alias(self):
        for flag in (
            "--require-congestion-level-coverage",
            "--require-risk-coverage",
        ):
            with self.subTest(flag=flag), patch("sys.argv", ["benchmark", flag]):
                args = benchmark.parse_args()
                self.assertTrue(args.require_congestion_level_coverage)

    def test_role_handler_uses_http_11(self):
        handler = build_role_handler(_ConcurrentRoleService(), 4096, False)
        self.assertEqual(handler.protocol_version, "HTTP/1.1")

    def test_four_partition_connections_are_parallel_and_reused(self):
        service = _ConcurrentRoleService()
        base_handler = build_role_handler(service, 4096, False)

        class RecordingHandler(base_handler):
            peer_ports = []
            peer_lock = threading.Lock()

            def do_POST(self):
                with type(self).peer_lock:
                    type(self).peer_ports.append(int(self.client_address[1]))
                super().do_POST()

        server = ThreadingHTTPServer(("127.0.0.1", 0), RecordingHandler)
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        pool = benchmark._PartitionConnectionPool(
            "http://127.0.0.1:{}".format(server.server_address[1]),
            timeout_seconds=1.0,
            connection_count=4,
        )
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                first = list(
                    executor.map(
                        lambda partition_id: pool.for_partition(
                            partition_id
                        ).post_json(
                            "/decide",
                            json.dumps({"value": partition_id}).encode("utf-8"),
                            {"Content-Type": "application/json"},
                        ),
                        range(4),
                    )
                )
                second = list(
                    executor.map(
                        lambda partition_id: pool.for_partition(
                            partition_id
                        ).post_json(
                            "/decide",
                            json.dumps({"value": partition_id + 4}).encode("utf-8"),
                            {"Content-Type": "application/json"},
                        ),
                        range(4),
                    )
                )
            self.assertEqual(service.max_active, 4)
            self.assertTrue(all(not result[2] for result in first))
            self.assertTrue(all(result[2] for result in second))
            snapshot = pool.snapshot()
            self.assertEqual(snapshot["connection_count"], 4)
            self.assertEqual(snapshot["totals"]["connections_opened"], 4)
            self.assertEqual(snapshot["totals"]["reused_requests"], 4)
            # Eight requests travelled over exactly four peer sockets: one
            # resident connection per partition, reused for the second window.
            self.assertEqual(
                sorted(Counter(RecordingHandler.peer_ports).values()),
                [2, 2, 2, 2],
            )
        finally:
            pool.close()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=1.0)

    def test_idle_keep_alive_releases_handler_and_reconnects_idempotently(self):
        service = _IdempotentRoleService()
        base_handler = build_role_handler(service, 4096, False)

        class ShortIdleHandler(base_handler):
            keep_alive_idle_timeout_seconds = 0.05
            active_handlers = 0
            handler_exits = 0
            handler_lock = threading.Lock()
            handler_exited = threading.Event()

            def handle(self):
                with type(self).handler_lock:
                    type(self).active_handlers += 1
                try:
                    super().handle()
                finally:
                    with type(self).handler_lock:
                        type(self).active_handlers -= 1
                        type(self).handler_exits += 1
                    type(self).handler_exited.set()

        server = ThreadingHTTPServer(("127.0.0.1", 0), ShortIdleHandler)
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        connection = benchmark._PersistentJsonConnection(
            "http://127.0.0.1:{}".format(server.server_address[1]),
            timeout_seconds=1.0,
        )
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": "idle-event-1",
        }
        try:
            first, _, first_reused, first_attempts = connection.post_json(
                "/decide", b'{"value":11}', headers
            )
            self.assertFalse(first_reused)
            self.assertEqual(first_attempts, 1)
            self.assertTrue(ShortIdleHandler.handler_exited.wait(1.0))
            with ShortIdleHandler.handler_lock:
                self.assertEqual(ShortIdleHandler.active_handlers, 0)
                self.assertEqual(ShortIdleHandler.handler_exits, 1)

            second, _, second_reused, second_attempts = connection.post_json(
                "/decide", b'{"value":11}', headers
            )
            self.assertEqual(second, first)
            self.assertFalse(second_reused)
            self.assertEqual(second_attempts, 2)
            self.assertEqual(service.idempotency_keys, ["idle-event-1"] * 2)
            self.assertEqual(service.calls, 2)
            self.assertEqual(service.operations, 1)
            snapshot = connection.snapshot()
            self.assertEqual(snapshot["connections_opened"], 2)
            self.assertEqual(snapshot["reconnects"], 1)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=1.0)

    def test_connection_without_first_request_releases_handler(self):
        service = _ConcurrentRoleService()
        base_handler = build_role_handler(service, 4096, False)

        class ShortFirstRequestHandler(base_handler):
            keep_alive_idle_timeout_seconds = 0.05
            active_handlers = 0
            handler_exits = 0
            handler_lock = threading.Lock()
            handler_started = threading.Event()
            handler_exited = threading.Event()

            def handle(self):
                with type(self).handler_lock:
                    type(self).active_handlers += 1
                type(self).handler_started.set()
                try:
                    super().handle()
                finally:
                    with type(self).handler_lock:
                        type(self).active_handlers -= 1
                        type(self).handler_exits += 1
                    type(self).handler_exited.set()

        server = ThreadingHTTPServer(("127.0.0.1", 0), ShortFirstRequestHandler)
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        client = socket.create_connection(server.server_address, timeout=1.0)
        try:
            self.assertTrue(ShortFirstRequestHandler.handler_started.wait(1.0))
            # The client deliberately sends no request bytes. The bounded first
            # request-line wait must close the connection and release its thread.
            self.assertTrue(ShortFirstRequestHandler.handler_exited.wait(1.0))
            with ShortFirstRequestHandler.handler_lock:
                self.assertEqual(ShortFirstRequestHandler.active_handlers, 0)
                self.assertEqual(ShortFirstRequestHandler.handler_exits, 1)
            self.assertEqual(client.recv(1), b"")
            self.assertEqual(service.calls, 0)
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=1.0)

    def test_keep_alive_idle_timeout_is_not_a_request_deadline(self):
        service = _ConcurrentRoleService()
        base_handler = build_role_handler(service, 4096, False)

        class ShortIdleHandler(base_handler):
            keep_alive_idle_timeout_seconds = 0.05

        server = ThreadingHTTPServer(("127.0.0.1", 0), ShortIdleHandler)
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        client = socket.create_connection(server.server_address, timeout=1.0)
        reader = client.makefile("rb")

        def read_response():
            status_line = reader.readline()
            self.assertTrue(status_line.startswith(b"HTTP/1.1 200"), status_line)
            headers = {}
            while True:
                line = reader.readline()
                if line == b"\r\n":
                    break
                name, value = line.decode("iso-8859-1").split(":", 1)
                headers[name.lower()] = value.strip()
            body = reader.read(int(headers["content-length"]))
            return json.loads(body.decode("utf-8"))

        body = b'{"value":21}'
        request_headers = (
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            + "Content-Length: {}\r\n\r\n".format(len(body)).encode("ascii")
        )
        try:
            client.sendall(b"POST /decide HTTP/1.1\r\n" + request_headers + body)
            self.assertEqual(read_response()["value"], 21)

            # Once this complete request line arrives, the handler must restore
            # the normal request timeout. Headers/body may legitimately take
            # longer than the keep-alive *idle* bound without being disconnected.
            client.sendall(b"POST /decide HTTP/1.1\r\n")
            time.sleep(0.12)
            client.sendall(request_headers + body)
            self.assertEqual(read_response()["value"], 21)
            self.assertEqual(service.calls, 2)
        finally:
            reader.close()
            client.close()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=1.0)

    def test_broken_persistent_connection_retries_same_idempotency_key(self):
        _DropFirstRequestHandler.request_count = 0
        _DropFirstRequestHandler.idempotency_keys = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _DropFirstRequestHandler)
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        connection = benchmark._PersistentJsonConnection(
            "http://127.0.0.1:{}".format(server.server_address[1]),
            timeout_seconds=1.0,
        )
        try:
            value, _, reused, attempts = connection.post_json(
                "/decide",
                b'{"value":7}',
                {
                    "Content-Type": "application/json",
                    "Idempotency-Key": "event-7",
                },
            )
            self.assertEqual(value, {"value": 7})
            self.assertFalse(reused)
            self.assertEqual(attempts, 2)
            self.assertEqual(
                _DropFirstRequestHandler.idempotency_keys,
                ["event-7", "event-7"],
            )
            snapshot = connection.snapshot()
            self.assertEqual(snapshot["logical_requests"], 1)
            self.assertEqual(snapshot["transport_attempts"], 2)
            self.assertEqual(snapshot["connections_opened"], 2)
            self.assertEqual(snapshot["reconnects"], 1)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=1.0)

    def test_under_200_sla_requires_complete_population(self):
        complete = benchmark._under_threshold_sla([100.0, 200.0], 2)
        self.assertTrue(complete["complete"])
        self.assertTrue(complete["mean_under_threshold"])
        self.assertEqual(complete["under_threshold_rate"], 1.0)

        # A failed partition must stay in the denominator instead of allowing
        # the three surviving rows to present a complete four-partition SLA.
        incomplete = benchmark._under_threshold_sla([100.0, 120.0, 140.0], 4)
        self.assertFalse(incomplete["complete"])
        self.assertFalse(incomplete["mean_under_threshold"])
        self.assertEqual(incomplete["under_threshold_rate"], 0.75)

        report = benchmark._latency_sla_report(
            [150.0],
            [150.0],
            [250.0],
            [100.0, 120.0, 140.0],
            [100.0, 120.0, 140.0],
            [220.0, 230.0, 240.0],
            expected_sample_count=1,
            expected_event_count=4,
        )
        self.assertTrue(report["sample_business_complete"])
        self.assertFalse(report["event_local_complete"])
        self.assertFalse(report["event_business_complete"])
        self.assertFalse(report["event_global_authoritative_final_complete"])

    def test_conflict_metrics_distinguish_zero_conflicts_from_missing_data(self):
        complete_zero = benchmark._conflict_metrics(
            0, 0, 400, 100, 100, 400, True
        )
        self.assertTrue(complete_zero["complete"])
        self.assertEqual(complete_zero["conflict_rate"], 0.0)
        self.assertFalse(complete_zero["conflict_resolution_evaluated"])
        self.assertIsNone(complete_zero["conflict_resolution_success_rate"])

        observed = benchmark._conflict_metrics(
            4, 1, 400, 100, 100, 400, True
        )
        self.assertTrue(observed["complete"])
        self.assertEqual(observed["conflict_rate"], 0.01)
        self.assertTrue(observed["conflict_resolution_evaluated"])
        self.assertEqual(observed["conflict_resolution_success_rate"], 0.75)

        missing = benchmark._conflict_metrics(
            0, 0, 396, 99, 100, 400, False
        )
        self.assertFalse(missing["complete"])
        self.assertIsNone(missing["conflict_rate"])
        self.assertIsNone(missing["conflict_resolution_success_rate"])

        inconsistent = benchmark._conflict_metrics(
            1, 2, 400, 100, 100, 400, True
        )
        self.assertFalse(inconsistent["complete"])
        self.assertIsNone(inconsistent["conflict_rate"])
        for initial, residual in ((0, 1), (-1, 0)):
            with self.subTest(initial=initial, residual=residual):
                invalid = benchmark._conflict_metrics(
                    initial, residual, 400, 100, 100, 400, True
                )
                self.assertFalse(invalid["complete"])
                self.assertIsNone(invalid["conflict_rate"])

    def test_partial_final_never_counts_as_authoritative_or_business_complete(self):
        row = benchmark._record_event(
            _native(),
            _post(),
            _review("partial_final"),
            sample_t0_epoch_ms=1000,
            review_observed_at_ms=140.0,
        )

        self.assertFalse(row["review_authoritative"])
        self.assertIsNone(row["global_final_ms"])
        self.assertIsNone(row["business_completion_ms"])

    def test_missing_review_cannot_complete_a_deferred_compact_action(self):
        row = benchmark._record_event(
            _native(),
            _post(),
            None,
            sample_t0_epoch_ms=1000,
            review_observed_at_ms=None,
        )

        self.assertFalse(row["review_authoritative"])
        self.assertEqual(row["deferred_action_types"], ["reroute"])
        self.assertIsNone(row["global_final_ms"])
        self.assertIsNone(row["business_completion_ms"])

    def test_authoritative_final_uses_common_sample_t0(self):
        row = benchmark._record_event(
            _native(),
            _post(),
            _review("lightweight_final"),
            sample_t0_epoch_ms=1000,
            review_observed_at_ms=140.0,
        )

        self.assertTrue(row["review_authoritative"])
        self.assertEqual(row["global_final_ms"], 120.0)
        self.assertEqual(row["business_completion_ms"], 120.0)
        self.assertEqual(row["decision_stratum"], "qwen_accepted_requires_cloud")

    def test_congestion_and_action_safety_are_reported_separately(self):
        row = benchmark._record_event(
            _native(),
            _post(),
            _review("lightweight_final"),
            sample_t0_epoch_ms=1000,
            review_observed_at_ms=140.0,
        )

        self.assertEqual(row["regional_congestion_level"], "low")
        self.assertEqual(row["legacy_congestion_level"], "severe")
        self.assertEqual(row["operational_safety_level"], "high")
        self.assertEqual(row["decision_delivery_path"], "local_decision_async_summary")

    def test_async_metric_total_reconstructs_byte_sum(self):
        snapshot = {
            "distributions": {
                "async_http_request_bytes": {"count": 4, "mean": 123.5}
            }
        }
        self.assertEqual(
            benchmark._metric_total(snapshot, "async_http_request_bytes"),
            494.0,
        )
        self.assertEqual(
            benchmark._metric_count(snapshot, "async_http_request_bytes"), 4
        )

    def test_async_metric_total_accepts_legacy_samples_key(self):
        snapshot = {
            "samples": {
                "async_http_request_bytes": {"count": 2, "mean": 50.0}
            }
        }
        self.assertEqual(
            benchmark._metric_total(snapshot, "async_http_request_bytes"),
            100.0,
        )


if __name__ == "__main__":
    unittest.main()
