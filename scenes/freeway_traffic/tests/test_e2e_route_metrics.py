import unittest
from unittest.mock import Mock

from scenes.freeway_traffic import benchmark_real_current_state_e2e as benchmark


def _native():
    return {
        "sample_id": 12,
        "partition_id": 1,
        "region_summary": {
            "region_risk_level": "medium",
            "max_node_risk_level": "high",
        },
        "upload_required": True,
        "upload_level": "summary",
    }


def _post(policy_route, delivery_route, policy_wait, override, deferred):
    return {
        "event_id": "event-12-1",
        "dispatch_ms": 1.0,
        "http_wall_ms": 8.0,
        "response_at_ms": 20.0,
        "request_bytes": 90,
        "response_bytes": 40,
        "response": {
            "schedule": {
                "route": delivery_route,
                "waits_for_cloud": False,
                "reason": "test delivery",
            },
            "data_plane": {
                "scheduler_selected_route": policy_route,
                "scheduler_selected_wait": policy_wait,
                "provisional_first_override": override,
            },
            "final_decision": {
                "status": "provisional",
                "route": delivery_route,
                "decision": "monitor",
                "actions": [
                    {
                        "action_type": action_type,
                        "parameters": {"requires_cloud_confirmation": True},
                    }
                    for action_type in deferred
                ],
                "metadata": {
                    "action_authorization": {
                        "cloud_confirmed": False,
                        "deferred_action_types": list(deferred),
                        "immediate_action_types": [],
                        "all_actions_authorized": not deferred,
                    }
                },
            },
            "summary_delivery": {
                "required": True,
                "mode": "background_handoff",
                "persistence_stage": "handoff_durable",
                "fast_path": True,
            },
        },
    }


def _review():
    return {
        "state": "completed",
        "completion_mode": "cloud",
        "completion_stage": "lightweight_final",
        "completed_at_ms": 1080.0,
        "local_decision": {
            "decision": "monitor",
            "metadata": {
                "source": "edge_student",
                "action_authorization": {
                    "cloud_confirmed": False,
                    "deferred_action_types": [],
                },
            },
        },
        "final_decision": {
            "status": "final",
            "route": "cloud_async",
            "decision": "monitor",
            "metadata": {
                "action_authorization": {
                    "cloud_confirmed": True,
                    "deferred_action_types": [],
                }
            },
        },
    }


class E2ERouteMetricTests(unittest.TestCase):
    def test_compact_post_requests_provisional_first_delivery(self):
        connection = Mock()
        connection.post_json.return_value = ({"status": "ok"}, 13, True, 1)
        envelope = {"id": "event-12-1"}

        result = benchmark._post_compact(
            connection,
            envelope,
            b"{}",
            timeout=1.0,
            sample_t0=benchmark.time.perf_counter(),
        )

        headers = connection.post_json.call_args.args[2]
        self.assertEqual(headers["Prefer"], "return=minimal, respond-async")
        self.assertEqual(headers["Idempotency-Key"], envelope["id"])
        self.assertEqual(result["event_id"], envelope["id"])

    def test_record_keeps_policy_and_delivery_routes_separate(self):
        row = benchmark._record_event(
            _native(),
            _post("cloud_sync", "cloud_async", True, True, []),
            _review(),
            sample_t0_epoch_ms=1000.0,
            review_observed_at_ms=90.0,
        )

        self.assertEqual(row["policy_route"], "cloud_sync")
        self.assertEqual(row["delivery_route"], "cloud_async")
        self.assertEqual(row["schedule_route"], "cloud_async")
        self.assertEqual(
            row["schedule_route_semantics"],
            "delivery_route_compatibility_alias",
        )
        self.assertTrue(row["policy_wait"])
        self.assertTrue(row["provisional_first_override"])
        self.assertEqual(row["business_endpoint"], "authoritative_final")
        self.assertEqual(row["business_completion_ms"], 80.0)
        self.assertTrue(row["route_success"])

    def test_missing_required_final_is_an_explicit_route_failure(self):
        row = benchmark._record_event(
            _native(),
            _post("cloud_sync", "cloud_async", True, True, []),
            None,
            sample_t0_epoch_ms=1000.0,
            review_observed_at_ms=None,
        )

        self.assertEqual(row["business_endpoint"], "missing_authoritative_final")
        self.assertIsNone(row["business_completion_ms"])
        self.assertFalse(row["route_success"])

    def test_async_local_response_is_a_successful_business_endpoint(self):
        row = benchmark._record_event(
            _native(),
            _post("cloud_async", "cloud_async", False, False, []),
            None,
            sample_t0_epoch_ms=1000.0,
            review_observed_at_ms=None,
        )

        self.assertEqual(
            row["business_endpoint"],
            "provisional_response_and_durable_summary",
        )
        self.assertEqual(row["business_completion_ms"], 20.0)
        self.assertTrue(row["route_success"])

    def test_grouped_bootstrap_is_deterministic_and_grouped_by_sample(self):
        rows = [
            {"sample_id": 1, "latency": 10.0},
            {"sample_id": 1, "latency": 20.0},
            {"sample_id": 2, "latency": 30.0},
            {"sample_id": 2, "latency": 40.0},
            {"sample_id": 3, "latency": 50.0},
            {"sample_id": 3, "latency": 60.0},
        ]
        first = benchmark._grouped_bootstrap_ci(
            rows, "latency", iterations=300, seed=17
        )
        second = benchmark._grouped_bootstrap_ci(
            rows, "latency", iterations=300, seed=17
        )

        self.assertEqual(first, second)
        self.assertEqual(first["estimate"], 35.0)
        self.assertEqual(first["group_count"], 3)
        self.assertEqual(first["valid_replicates"], 300)
        self.assertLessEqual(first["lower"], first["estimate"])
        self.assertGreaterEqual(first["upper"], first["estimate"])

    def test_route_manifest_requires_all_positive_weights_summing_to_one(self):
        weights = benchmark._validate_route_manifest(
            {
                "schema_version": 1,
                "weight_plan_id": "traffic-route-mix-v1",
                "locked_at": "2026-08-08T09:00:00+08:00",
                "plan_sha256": "a" * 64,
                "routes": {
                    "edge_only": 0.1,
                    "local_autonomy": 0.2,
                    "cloud_async": 0.3,
                    "cloud_sync": 0.4,
                },
            }
        )
        self.assertEqual(sum(weights.values()), 1.0)

        for invalid in (
            {
                "weight_plan_id": "traffic-route-mix-v1",
                "locked_at": "2026-08-08T09:00:00+08:00",
                "plan_sha256": "a" * 64,
                "routes": {
                    "edge_only": 0.2,
                    "local_autonomy": 0.2,
                    "cloud_async": 0.2,
                }
            },
            {
                "weight_plan_id": "traffic-route-mix-v1",
                "locked_at": "2026-08-08T09:00:00+08:00",
                "plan_sha256": "a" * 64,
                "routes": {
                    "edge_only": 0.1,
                    "local_autonomy": 0.1,
                    "cloud_async": 0.1,
                    "cloud_sync": 0.1,
                }
            },
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    benchmark._validate_route_manifest(invalid)

    def test_weighted_gate_preserves_failures_in_route_denominator(self):
        weights = {
            "edge_only": 0.1,
            "local_autonomy": 0.2,
            "cloud_async": 0.3,
            "cloud_sync": 0.4,
        }
        rows = [
            {
                "sample_id": index,
                "policy_route": route,
                "delivery_route": route,
                "business_endpoint": "authoritative_final"
                if route == "cloud_sync"
                else "provisional_response_and_durable_summary",
                "business_completion_ms": latency,
                "route_success": True,
            }
            for index, (route, latency) in enumerate(
                zip(
                    benchmark.BUSINESS_POLICY_ROUTES,
                    (100.0, 200.0, 300.0, 400.0),
                )
            )
        ]
        summary = benchmark._route_business_summary(rows, iterations=100, seed=9)
        gate = benchmark._weighted_business_e2e(summary, weights)
        self.assertTrue(gate["valid"])
        self.assertEqual(gate["weighted_business_e2e_ms"], 300.0)
        self.assertFalse(gate["target_passed"])

        failed_rows = list(rows)
        failed_rows[-1] = dict(
            failed_rows[-1],
            business_endpoint="missing_authoritative_final",
            business_completion_ms=None,
            route_success=False,
        )
        failed_summary = benchmark._route_business_summary(
            failed_rows, iterations=100, seed=9
        )
        failed_gate = benchmark._weighted_business_e2e(failed_summary, weights)
        self.assertFalse(failed_gate["valid"])
        self.assertEqual(failed_summary["cloud_sync"]["event_count"], 1)
        self.assertEqual(failed_summary["cloud_sync"]["failure_count"], 1)
        self.assertEqual(failed_summary["cloud_sync"]["success_rate"], 0.0)
        self.assertIsNone(failed_gate["weighted_business_e2e_ms"])

        missing_population_gate = benchmark._weighted_business_e2e(
            summary,
            weights,
            expected_event_count=5,
            observed_event_count=4,
        )
        self.assertFalse(missing_population_gate["valid"])
        self.assertIn(
            "population:observed_events=4/5",
            missing_population_gate["invalid_reasons"],
        )

    def test_weighted_gate_bootstraps_fixed_route_mix(self):
        weights = {route: 0.25 for route in benchmark.BUSINESS_POLICY_ROUTES}
        rows = []
        for route_index, route in enumerate(benchmark.BUSINESS_POLICY_ROUTES):
            for sample_id in range(8):
                rows.append(
                    {
                        "sample_id": "{}-{}".format(route, sample_id),
                        "policy_route": route,
                        "delivery_route": route,
                        "business_endpoint": "provisional_response_and_durable_summary",
                        "business_completion_ms": 90.0 + route_index + sample_id,
                        "route_success": True,
                    }
                )
        summary = benchmark._route_business_summary(rows, iterations=100, seed=4)
        gate = benchmark._weighted_business_e2e(
            summary,
            weights,
            rows=rows,
            iterations=300,
            seed=4,
        )
        self.assertTrue(gate["valid"])
        self.assertTrue(gate["target_passed"])
        self.assertLess(gate["weighted_business_e2e_ms"], 200.0)
        self.assertLess(
            gate["weighted_business_e2e_mean_95ci"]["upper"], 200.0
        )


if __name__ == "__main__":
    unittest.main()
