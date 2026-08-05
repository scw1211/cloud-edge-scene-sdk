import unittest

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


class Pems08E2EBenchmarkTests(unittest.TestCase):
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
            "samples": {
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


if __name__ == "__main__":
    unittest.main()
