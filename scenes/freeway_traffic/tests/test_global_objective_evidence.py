import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


SCENE_ROOT = Path(__file__).resolve().parents[1]
if str(SCENE_ROOT) not in sys.path:
    sys.path.insert(0, str(SCENE_ROOT))

from extract_global_objective_evidence import extract  # noqa: E402


class GlobalObjectiveEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="global-objective-evidence-")
        self.root = Path(self.temporary.name)
        self.objective = self.root / "objective.json"
        self.objective.write_text(
            json.dumps(
                {
                    "objective_id": "objective-v1",
                    "expected_members": [
                        "edge_node_0",
                        "edge_node_1",
                        "edge_node_2",
                        "edge_node_3",
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _sample(self, mode="active", applied=True, complete=True):
        definition_sha = hashlib.sha256(self.objective.read_bytes()).hexdigest()
        expected_members = [
            "edge_node_0",
            "edge_node_1",
            "edge_node_2",
            "edge_node_3",
        ]
        received_members = list(expected_members if complete else expected_members[:2])
        return {
            "sample_id": 7,
            "aggregations_complete": complete,
            "global_authoritative_final_ms": 225.0 if complete else None,
            "errors": [] if complete else ["aggregation timeout"],
            "aggregations": [
                {
                    "group_id": "group-7",
                    "state": "completed" if complete else "timed_out",
                    "completion_reason": (
                        "all_expected_members"
                        if complete
                        else "timeout_with_partial_members"
                    ),
                    "expected_members": expected_members,
                    "received_members": received_members,
                    "missing_members": [] if complete else expected_members[2:],
                    "evidence_complete": complete,
                    "finality": "final" if complete else "partial_final",
                    "global_confirmation": complete,
                    "result": {
                        "global_optimizations": [
                            {
                                "objective_id": "objective-v1",
                                "objective_definition_sha256": definition_sha,
                                "exact_search": True,
                                "candidate_set_complete": True,
                                "constraints_satisfied": True,
                                "mode": mode,
                                "applied": applied,
                                "authoritative_input": True,
                                "baseline_utility": -1.0,
                                "selected_utility": 0.25,
                                "candidate_space_size": 16,
                                "evaluated_candidate_count": 16,
                                "expected_member_count": 4,
                                "observed_member_count": 4,
                                "expected_members": [
                                    "edge_node_0",
                                    "edge_node_1",
                                    "edge_node_2",
                                    "edge_node_3",
                                ],
                                "observed_members": [
                                    "edge_node_0",
                                    "edge_node_1",
                                    "edge_node_2",
                                    "edge_node_3",
                                ],
                                "selected": {"plan_sha256": "a" * 64},
                            }
                        ]
                    },
                }
            ],
        }

    def _write_benchmark(self, sample):
        path = self.root / "benchmark.json"
        samples = sample if isinstance(sample, list) else [sample]
        path.write_text(
            json.dumps(
                {
                    "experiment_id": "source-run-1",
                    "measurement_started_at_utc": "2026-08-08T10:00:00+00:00",
                    "samples": samples,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def _write_plan(self, sample_ids=(7,), **expected_overrides):
        definition_sha = hashlib.sha256(self.objective.read_bytes()).hexdigest()
        expected = {
            "git_commit": "a" * 40,
            "dataset_id": "fixed-test",
            "model_ids": ["cloud-v1"],
            "hardware_id": "cloud-host",
            "objective_id": "objective-v1",
            "objective_definition_sha256": definition_sha,
        }
        expected.update(expected_overrides)
        path = self.root / "sample-plan.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "plan_id": "objective-samples-v1",
                    "locked_at": "2026-08-08T09:00:00+00:00",
                    "scene": "freeway_traffic",
                    "expected": expected,
                    "expected_sample_count": len(sample_ids),
                    "sample_ids": list(sample_ids),
                    "slow_threshold_ms": 200.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def _extract(self, sample, *, sample_ids=(7,), plan_path=None):
        return extract(
            [self._write_benchmark(sample)],
            self.objective,
            plan_path or self._write_plan(sample_ids),
            git_commit="a" * 40,
            dataset_id="fixed-test",
            model_ids=["cloud-v1"],
            hardware_id="cloud-host",
            run_id="run-1",
        )

    def test_extracts_active_authoritative_record(self):
        value = self._extract(self._sample())
        self.assertEqual(value["objective_id"], "objective-v1")
        self.assertEqual(len(value["records"]), 1)
        self.assertEqual(value["records"][0]["group_id"], "group-7")
        self.assertEqual(value["records"][0]["selected_utility"], 0.25)
        self.assertTrue(value["records"][0]["applied"])
        self.assertEqual(value["attempts"][0]["outcome"], "authoritative_optimized")
        self.assertTrue(value["attempts"][0]["slow"])
        self.assertEqual(value["slow_sample_count"], 1)
        self.assertEqual(
            value["records"][0]["expected_members"],
            value["records"][0]["observed_members"],
        )

    def test_partial_aggregation_is_preserved_in_denominator(self):
        value = self._extract(self._sample(complete=False))
        self.assertEqual(value["records"], [])
        self.assertEqual(value["attempts"][0]["outcome"], "partial")
        self.assertEqual(value["outcome_counts"]["partial"], 1)

    def test_shadow_is_preserved_for_gate_to_reject(self):
        value = self._extract(self._sample(mode="shadow", applied=False))
        self.assertEqual(value["records"][0]["mode"], "shadow")
        self.assertFalse(value["records"][0]["applied"])

    def test_definition_sha_mismatch_is_rejected(self):
        sample = self._sample()
        sample["aggregations"][0]["result"]["global_optimizations"][0][
            "objective_definition_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "definition SHA"):
            self._extract(sample)

    def test_missing_member_declaration_is_rejected(self):
        sample = self._sample()
        sample["aggregations"][0].pop("expected_members")
        with self.assertRaisesRegex(ValueError, "expected_members"):
            self._extract(sample)

    def test_inconsistent_member_declaration_is_rejected(self):
        sample = self._sample()
        sample["aggregations"][0]["received_members"] = [
            "edge_node_0",
            "edge_node_1",
            "edge_node_2",
        ]
        with self.assertRaisesRegex(ValueError, "must match exactly"):
            self._extract(sample)

    def test_duplicate_members_are_rejected(self):
        sample = self._sample()
        sample["aggregations"][0]["received_members"][-1] = "edge_node_2"
        with self.assertRaisesRegex(ValueError, "duplicate member"):
            self._extract(sample)

    def test_self_reported_complete_subset_cannot_replace_definition_contract(self):
        sample = self._sample()
        subset = ["edge_node_0", "edge_node_1"]
        aggregation = sample["aggregations"][0]
        aggregation["expected_members"] = list(subset)
        aggregation["received_members"] = list(subset)
        optimization = aggregation["result"]["global_optimizations"][0]
        optimization["expected_members"] = list(subset)
        optimization["observed_members"] = list(subset)
        optimization["expected_member_count"] = 2
        optimization["observed_member_count"] = 2
        with self.assertRaisesRegex(ValueError, "objective definition"):
            self._extract(sample)

    def test_missing_planned_sample_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"missing=\[8\]"):
            self._extract(self._sample(), sample_ids=(7, 8))

    def test_duplicate_benchmark_sample_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate benchmark sample"):
            self._extract([self._sample(), self._sample()], sample_ids=(7,))

    def test_plan_context_mismatch_is_rejected(self):
        plan = self._write_plan(dataset_id="different-dataset")
        with self.assertRaisesRegex(ValueError, "expected.dataset_id"):
            self._extract(self._sample(), plan_path=plan)


if __name__ == "__main__":
    unittest.main()
