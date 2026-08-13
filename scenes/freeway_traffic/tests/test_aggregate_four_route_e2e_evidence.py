import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scenes.freeway_traffic import aggregate_four_route_e2e_evidence as aggregator
from scripts import evaluate_competition_targets as evaluator


ROUTES = aggregator.ROUTES


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FourRouteEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="four-route-evidence-")
        self.root = Path(self.temp.name)
        self.data_sha = "d" * 64
        self.expected = {
            "git_commit": "a" * 40,
            "dataset_id": "pems08-fixed-v1",
            "data_sha256": self.data_sha,
            "model_ids": {"student": "b" * 64, "fusion": "c" * 64},
            "hardware_id": {"edge": "orin", "cloud": "server"},
        }
        self.run_started = "2026-08-09T09:00:00+08:00"
        self.plan = {
            "schema_version": "1.0",
            "aggregation_id": "aggregate-test",
            "weight_plan_id": "weights-before-test",
            "locked_at": "2026-08-09T08:00:00+08:00",
            "expected": self.expected,
            "routes": {},
        }
        for index, route in enumerate(ROUTES):
            run_id = "{}-run".format(route)
            result = self.root / "{}.json".format(route)
            attestation = self.root / "{}.attestation.json".format(route)
            self._write_benchmark(result, route, index)
            self.plan["routes"][route] = {
                "weight": 0.25,
                "experiments": [
                    {
                        "run_id": run_id,
                        "result_path": result.name,
                        "attestation_path": attestation.name,
                        "sample_ids": [index],
                        "partition_ids": [0, 1],
                        "network_profile": self._profile(route),
                    }
                ],
            }
        self.plan_path = self.root / "plan.json"
        self._save_plan()
        for route in ROUTES:
            self._refresh(route)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _profile(route):
        return {
            "profile_id": "profile-{}".format(route),
            "cloud_available": route != "local_autonomy",
        }

    def _write_benchmark(self, path, route, sample_id, omit_partition=None):
        events = []
        for partition_id in (0, 1):
            if partition_id == omit_partition:
                continue
            deferred = ["regional_coordination"] if route == "local_autonomy" else []
            events.append(
                {
                    "sample_id": sample_id,
                    "partition_id": partition_id,
                    "policy_route": route,
                    # Deliberately different: the aggregator must not stratify on it.
                    "delivery_route": "cloud_async",
                    "policy_wait": route == "cloud_sync",
                    "response_at_ms": 60.0 + partition_id,
                    "response_status": "provisional",
                    "global_final_ms": 150.0 + partition_id,
                    "review_authoritative": True,
                    "review_state": "completed",
                    "review_completion_stage": "lightweight_final",
                    "review_final_status": "final",
                    "review_cloud_confirmed": True,
                    "deferred_action_types": deferred,
                    "immediate_action_types": [],
                    "response_actions": [
                        {
                            "action_type": action_type,
                            "requires_cloud_confirmation": True,
                        }
                        for action_type in deferred
                    ],
                    "action_authorization_present": True,
                    "local_actions_authorized": route != "local_autonomy",
                    "local_autonomy": route == "local_autonomy",
                    "cloud_confirmed_in_response": False,
                    "summary_delivery_required": True,
                    "summary_persistence_stage": (
                        "outbox_durable"
                        if route == "local_autonomy"
                        else "handoff_durable"
                    ),
                }
            )
        value = {
            "schema_version": 2,
            "task": "pems08_current_state_deployed_e2e",
            "measurement_started_at_utc": self.run_started,
            "assets": {"data_sha256": self.data_sha},
            "sample_selection": {"sample_count": 1},
            "samples": [{"sample_id": sample_id, "events": events}],
        }
        path.write_text(json.dumps(value), encoding="utf-8")

    def _write_attestation(self, path, benchmark, run_id, route, **overrides):
        value = {
            "schema_version": "1.0",
            "run_id": run_id,
            "experiment_plan_sha256": _sha256(self.plan_path),
            "benchmark_sha256": _sha256(benchmark),
            "git_commit": self.expected["git_commit"],
            "dataset_id": self.expected["dataset_id"],
            "data_sha256": self.expected["data_sha256"],
            "model_ids": self.expected["model_ids"],
            "hardware_id": self.expected["hardware_id"],
            "network_profile": self._profile(route),
            "run_started_at": self.run_started,
        }
        value.update(overrides)
        path.write_text(json.dumps(value), encoding="utf-8")

    def _save_plan(self):
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")

    def _refresh(self, route):
        experiment = self.plan["routes"][route]["experiments"][0]
        self._write_attestation(
            self.root / experiment["attestation_path"],
            self.root / experiment["result_path"],
            experiment["run_id"],
            route,
        )

    def test_multirun_output_is_accepted_by_unified_latency_evaluator(self):
        evidence, weights = aggregator.aggregate_plan(
            self.plan_path, generated_at="2026-08-09T10:00:00+08:00"
        )
        self.assertTrue(evidence["integrity_valid"])
        for route in ROUTES:
            self.assertEqual(evidence["routes"][route]["attempt_count"], 2)
            self.assertEqual(evidence["routes"][route]["success_count"], 2)
        autonomy = evidence["routes"]["local_autonomy"]["samples"][0]
        self.assertEqual(autonomy["latency_ms"], 60.0)
        self.assertEqual(
            autonomy["business_endpoint"],
            "provisional_response_and_durable_summary",
        )
        sync = evidence["routes"]["cloud_sync"]["samples"][0]
        self.assertEqual(sync["latency_ms"], 150.0)
        passed, metrics, reasons = evaluator._evaluate_latency(
            evidence,
            weights,
            bootstrap_iterations=300,
            bootstrap_seed=17,
        )
        self.assertTrue(passed, reasons)
        self.assertLess(metrics["weighted_mean_ms"], 200.0)
        self.assertEqual(metrics["routes"]["cloud_async"]["group_count"], 1)
        self.assertEqual(
            metrics["resampling_unit"],
            "sample_id_within_run_and_policy_route",
        )

    def test_missing_event_remains_in_attempt_denominator(self):
        route = "cloud_async"
        experiment = self.plan["routes"][route]["experiments"][0]
        result = self.root / experiment["result_path"]
        self._write_benchmark(result, route, 2, omit_partition=1)
        self._refresh(route)
        evidence, weights = aggregator.aggregate_plan(self.plan_path)
        actual = evidence["routes"][route]
        self.assertEqual(actual["attempt_count"], 2)
        self.assertEqual(actual["success_count"], 1)
        self.assertEqual(actual["failure_count"], 1)
        self.assertEqual(actual["failures"][0]["reason"], "event_missing")
        passed, _, reasons = evaluator._evaluate_latency(
            evidence, weights, bootstrap_iterations=300, bootstrap_seed=3
        )
        self.assertFalse(passed)
        self.assertIn("未到达业务终点", reasons[0])

    def test_policy_route_not_delivery_route_controls_stratification(self):
        evidence, _ = aggregator.aggregate_plan(self.plan_path)
        sample = evidence["routes"]["edge_only"]["samples"][0]
        self.assertEqual(sample["policy_route"], "edge_only")
        self.assertEqual(sample["delivery_route"], "cloud_async")
        route = "edge_only"
        experiment = self.plan["routes"][route]["experiments"][0]
        path = self.root / experiment["result_path"]
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["samples"][0]["events"][0]["policy_route"] = "cloud_async"
        path.write_text(json.dumps(raw), encoding="utf-8")
        self._refresh(route)
        evidence, _ = aggregator.aggregate_plan(self.plan_path)
        self.assertEqual(evidence["routes"][route]["success_count"], 1)
        self.assertEqual(
            evidence["routes"][route]["failures"][0]["reason"],
            "policy_route_mismatch",
        )

    def test_local_autonomy_requires_durable_queue(self):
        route = "local_autonomy"
        experiment = self.plan["routes"][route]["experiments"][0]
        path = self.root / experiment["result_path"]
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["samples"][0]["events"][0]["summary_persistence_stage"] = "not_required"
        path.write_text(json.dumps(raw), encoding="utf-8")
        self._refresh(route)
        evidence, _ = aggregator.aggregate_plan(self.plan_path)
        self.assertEqual(evidence["routes"][route]["success_count"], 1)
        self.assertEqual(
            evidence["routes"][route]["failures"][0]["reason"],
            "local_summary_not_durable",
        )

    def test_local_autonomy_requires_marker_and_safe_authorization(self):
        route = "local_autonomy"
        experiment = self.plan["routes"][route]["experiments"][0]
        path = self.root / experiment["result_path"]
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["samples"][0]["events"][0]["local_autonomy"] = False
        raw["samples"][0]["events"][1].update(
            {
                "deferred_action_types": [],
                "immediate_action_types": ["unsafe_control"],
                "response_actions": [
                    {
                        "action_type": "unsafe_control",
                        "requires_cloud_confirmation": False,
                    }
                ],
                "local_actions_authorized": False,
            }
        )
        path.write_text(json.dumps(raw), encoding="utf-8")
        self._refresh(route)
        evidence, _ = aggregator.aggregate_plan(self.plan_path)
        self.assertEqual(evidence["routes"][route]["success_count"], 0)
        self.assertEqual(
            [value["reason"] for value in evidence["routes"][route]["failures"]],
            ["local_autonomy_marker_missing", "local_action_not_authorized"],
        )

    def test_async_deferred_action_still_uses_safe_provisional_endpoint(self):
        latency, endpoint, reason = aggregator._business_endpoint(
            "cloud_async",
            {
                "response_at_ms": 55.0,
                "response_status": "provisional",
                "policy_wait": False,
                "deferred_action_types": ["regional_coordination"],
                "immediate_action_types": ["traffic_advisory"],
                "response_actions": [
                    {
                        "action_type": "regional_coordination",
                        "requires_cloud_confirmation": True,
                    },
                    {
                        "action_type": "traffic_advisory",
                        "requires_cloud_confirmation": False,
                    },
                ],
                "action_authorization_present": True,
                "local_actions_authorized": False,
                "cloud_confirmed_in_response": False,
                "summary_delivery_required": True,
                "summary_persistence_stage": "handoff_durable",
            },
        )
        self.assertEqual(latency, 55.0)
        self.assertEqual(endpoint, "provisional_response_and_durable_summary")
        self.assertIsNone(reason)

    def test_async_provisional_requires_action_authorization_contract(self):
        latency, endpoint, reason = aggregator._business_endpoint(
            "cloud_async",
            {
                "response_at_ms": 55.0,
                "response_status": "provisional",
                "policy_wait": False,
                "deferred_action_types": [],
                "immediate_action_types": ["variable_speed_limit"],
                "response_actions": [
                    {
                        "action_type": "variable_speed_limit",
                        "requires_cloud_confirmation": False,
                    }
                ],
                "action_authorization_present": False,
                "local_actions_authorized": False,
                "cloud_confirmed_in_response": False,
                "summary_delivery_required": True,
                "summary_persistence_stage": "handoff_durable",
            },
        )
        self.assertIsNone(latency)
        self.assertEqual(endpoint, "missing_action_authorization")
        self.assertEqual(reason, "action_authorization_missing")

    def test_high_risk_cloud_required_action_cannot_be_counted_as_immediate(self):
        latency, endpoint, reason = aggregator._business_endpoint(
            "cloud_async",
            {
                "response_at_ms": 12.0,
                "response_status": "provisional",
                "policy_wait": False,
                "deferred_action_types": [],
                "immediate_action_types": ["regional_coordination"],
                "response_actions": [
                    {
                        "action_type": "regional_coordination",
                        "requires_cloud_confirmation": True,
                    }
                ],
                "action_authorization_present": True,
                "local_actions_authorized": True,
                "cloud_confirmed_in_response": False,
                "summary_delivery_required": True,
                "summary_persistence_stage": "handoff_durable",
            },
        )
        self.assertIsNone(latency)
        self.assertEqual(endpoint, "invalid_action_stage")
        self.assertEqual(reason, "action_authorization_stage_mismatch")

    def test_forged_authoritative_boolean_cannot_hide_partial_stage(self):
        route = "cloud_sync"
        experiment = self.plan["routes"][route]["experiments"][0]
        path = self.root / experiment["result_path"]
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["samples"][0]["events"][0].update(
            {
                "review_authoritative": True,
                "review_completion_stage": "partial_final",
                "review_final_status": "final",
                "review_cloud_confirmed": True,
                "global_final_ms": 1.0,
            }
        )
        path.write_text(json.dumps(raw), encoding="utf-8")
        self._refresh(route)
        evidence, _ = aggregator.aggregate_plan(self.plan_path)
        self.assertEqual(evidence["routes"][route]["success_count"], 1)
        self.assertEqual(
            evidence["routes"][route]["failures"][0]["reason"],
            "authoritative_final_missing",
        )

    def test_modified_weights_break_run_attestation_plan_binding(self):
        self.plan["routes"]["edge_only"]["weight"] = 0.10
        self.plan["routes"]["cloud_async"]["weight"] = 0.40
        self._save_plan()
        evidence, _ = aggregator.aggregate_plan(self.plan_path)
        self.assertFalse(evidence["integrity_valid"])
        self.assertTrue(
            all(
                "experiment_plan_sha256_mismatch" in value
                for value in evidence["integrity_errors"]
            )
        )

    def test_gate_rejects_dropped_preregistered_member_even_if_counts_are_rewritten(self):
        evidence, weights = aggregator.aggregate_plan(self.plan_path)
        route = evidence["routes"]["cloud_async"]
        route["samples"].pop()
        route["attempt_count"] = 1
        route["success_count"] = 1
        route["failure_count"] = 0
        with self.assertRaisesRegex(evaluator.InvalidEvidence, "预注册"):
            evaluator._evaluate_latency(
                evidence,
                weights,
                bootstrap_iterations=300,
                bootstrap_seed=13,
            )

    def test_gate_rejects_self_reported_group_id_instead_of_resampling_events(self):
        evidence, weights = aggregator.aggregate_plan(self.plan_path)
        evidence["routes"]["edge_only"]["samples"][0]["group_id"] = "fast-event"
        with self.assertRaisesRegex(evaluator.InvalidEvidence, "run_id:sample_id"):
            evaluator._evaluate_latency(
                evidence,
                weights,
                bootstrap_iterations=300,
                bootstrap_seed=13,
            )

    def test_cli_refuses_to_overwrite_formal_outputs(self):
        evidence_path = self.root / "formal-evidence.json"
        weight_path = self.root / "formal-weights.json"
        evidence_path.write_text("old evidence", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            aggregator.main(
                [
                    "--plan",
                    str(self.plan_path),
                    "--output-evidence",
                    str(evidence_path),
                    "--output-weight-plan",
                    str(weight_path),
                ]
            )
        self.assertEqual(evidence_path.read_text(encoding="utf-8"), "old evidence")
        self.assertFalse(weight_path.exists())

    def test_attestation_mismatch_invalidates_every_preregistered_attempt(self):
        route = "cloud_sync"
        experiment = self.plan["routes"][route]["experiments"][0]
        self._write_attestation(
            self.root / experiment["attestation_path"],
            self.root / experiment["result_path"],
            experiment["run_id"],
            route,
            git_commit="wrong-commit",
        )
        evidence, _ = aggregator.aggregate_plan(self.plan_path)
        self.assertFalse(evidence["integrity_valid"])
        self.assertEqual(evidence["routes"][route]["attempt_count"], 2)
        self.assertEqual(evidence["routes"][route]["success_count"], 0)
        self.assertEqual(evidence["routes"][route]["failure_count"], 2)
        self.assertTrue(
            any("git_commit_mismatch" in value for value in evidence["integrity_errors"])
        )

    def test_result_with_unregistered_event_cannot_be_subset_selected(self):
        route = "edge_only"
        experiment = self.plan["routes"][route]["experiments"][0]
        path = self.root / experiment["result_path"]
        raw = json.loads(path.read_text(encoding="utf-8"))
        extra = dict(raw["samples"][0]["events"][0])
        extra["partition_id"] = 9
        raw["samples"][0]["events"].append(extra)
        path.write_text(json.dumps(raw), encoding="utf-8")
        self._refresh(route)
        evidence, _ = aggregator.aggregate_plan(self.plan_path)
        self.assertFalse(evidence["integrity_valid"])
        self.assertEqual(evidence["routes"][route]["success_count"], 0)
        self.assertTrue(
            any("unregistered_events" in value for value in evidence["integrity_errors"])
        )


if __name__ == "__main__":
    unittest.main()
