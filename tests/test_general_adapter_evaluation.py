import copy
import json
import tempfile
import unittest
from pathlib import Path

from edge_llm_factory.evaluate_general_adapter import (
    EVALUATION_SCHEMA_VERSION,
    ManifestError,
    _apply_assistant_prefill,
    _evaluate_choice,
    _evaluate_math,
    _read_jsonl,
    _select_rows,
    _selected_categories,
    _selected_sample_manifest,
    _summary,
    _teacher_scores,
    _validate_adapter_training_evidence,
    evaluate_response,
)
from edge_llm_factory.contracts import sha256_file


class GeneralAdapterEvaluationTest(unittest.TestCase):
    def _write_json(self, path: Path, value):
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def _write_jsonl(self, path: Path, rows):
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    def test_math_prefers_explicit_final_answer(self):
        result = _evaluate_math("先算出 12，修正后 FINAL: 15", "15")
        self.assertTrue(result["correct"])
        self.assertEqual(result["prediction"], "15")

    def test_math_rejects_expression_after_final_marker(self):
        result = _evaluate_math("FINAL: $90 - 20 = 70", "90")
        self.assertFalse(result["correct"])
        self.assertIsNone(result["prediction"])

    def test_math_rejects_unfinished_derivation_without_final_marker(self):
        result = _evaluate_math("先得到 12，然后还要继续计算", "12")
        self.assertFalse(result["correct"])
        self.assertIsNone(result["prediction"])

    def test_choice_ignores_empty_reasoning_wrapper(self):
        result = _evaluate_choice("<think>\n\n</think>\nC", "C")
        self.assertTrue(result["correct"])
        self.assertEqual(result["prediction"], "C")

    def test_choice_rejects_explanation_or_multiple_letters(self):
        for output in ("答案是 C", "C，因为其他选项错误", "A/C", "AB"):
            with self.subTest(output=output):
                result = _evaluate_choice(output, "C")
                self.assertFalse(result["correct"])
                self.assertIsNone(result["prediction"])

    def test_mbpp_evaluation_does_not_require_reference_answer(self):
        row = {
            "sample_id": "code-1",
            "category": "code",
            "prompt": "Write square().",
            "test_list": ["assert square(4) == 16"],
            "test_imports": [],
        }
        result = evaluate_response(
            "def square(value):\n    return value * value", row, code_timeout=1.0
        )
        self.assertTrue(result["correct"])
        self.assertEqual(result["reference"], "official MBPP tests")

    def test_jsonl_requires_category_specific_fields_and_unique_sample_ids(self):
        valid_rows = [
            {
                "sample_id": "math-1",
                "category": "math",
                "prompt": "1+1?",
                "reference_answer": "2",
            },
            {
                "sample_id": "code-1",
                "category": "code",
                "prompt": "Write identity().",
                "test_list": ["assert identity(2) == 2"],
            },
            {
                "sample_id": "nlr-1",
                "category": "natural_language_reasoning",
                "prompt": "Choose.",
                "reference_answer": "c",
            },
        ]
        invalid_rows = {
            "missing_math_reference": [
                {"sample_id": "math-1", "category": "math", "prompt": "1+1?"}
            ],
            "missing_code_tests": [
                {"sample_id": "code-1", "category": "code", "prompt": "Write f."}
            ],
            "unknown_category": [
                {"sample_id": "x", "category": "traffic", "prompt": "x"}
            ],
            "duplicate_sample_id": [valid_rows[0], {**valid_rows[1], "sample_id": "math-1"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.jsonl"
            self._write_jsonl(path, valid_rows)
            loaded = _read_jsonl(path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[2]["reference_answer"], "C")
            self.assertEqual(loaded[1]["test_imports"], [])

            for name, rows in invalid_rows.items():
                with self.subTest(name=name):
                    self._write_jsonl(path, rows)
                    with self.assertRaises(ManifestError):
                        _read_jsonl(path)

    def test_selected_sample_manifest_is_stable_and_reports_category_counts(self):
        rows = [
            {
                "sample_id": "math-2",
                "category": "math",
                "prompt": "2+2?",
                "reference_answer": "4",
            },
            {
                "sample_id": "code-2",
                "category": "code",
                "prompt": "Write f.",
                "test_list": ["assert f() == 1"],
                "test_imports": [],
            },
            {
                "sample_id": "math-1",
                "category": "math",
                "prompt": "1+1?",
                "reference_answer": "2",
            },
            {
                "sample_id": "code-1",
                "category": "code",
                "prompt": "Write g.",
                "test_list": ["assert g() == 2"],
                "test_imports": [],
            },
        ]
        categories = {"code", "math"}
        selected_a = _select_rows(rows, categories, limit_per_category=1)
        selected_b = _select_rows(list(reversed(rows)), categories, limit_per_category=1)
        manifest_a = _selected_sample_manifest(selected_a, "a" * 64)
        manifest_b = _selected_sample_manifest(selected_b, "a" * 64)

        self.assertEqual(selected_a, selected_b)
        self.assertEqual(
            [sample["sample_id"] for sample in manifest_a["samples"]],
            ["code-1", "math-1"],
        )
        self.assertEqual(manifest_a["category_sample_counts"], {"code": 1, "math": 1})
        self.assertEqual(manifest_a["sample_count"], 2)
        self.assertEqual(manifest_a, manifest_b)
        self.assertEqual(len(manifest_a["sha256"]), 64)

    def test_teacher_evidence_requires_exact_samples_protocol_and_real_counts(self):
        rows = [
            {
                "sample_id": "code-1",
                "category": "code",
                "prompt": "Write f.",
                "test_list": ["assert f() == 1"],
                "test_imports": [],
            },
            {
                "sample_id": "math-1",
                "category": "math",
                "prompt": "1+1?",
                "reference_answer": "2",
            },
        ]
        dataset_sha = "b" * 64
        manifest = _selected_sample_manifest(rows, dataset_sha)
        protocol = {
            "schema_version": "protocol/v1",
            "max_input_tokens": 1024,
            "decoding": {"do_sample": False},
        }
        report = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "dataset": {"path": "/frozen.jsonl", "sha256": dataset_sha},
            "selected_sample_manifest": manifest,
            "evaluation_protocol": protocol,
            "summary": {
                "completed_samples": 2,
                "categories": {
                    "code": {"sample_count": 1, "correct": 0, "score": 0.0},
                    "math": {"sample_count": 1, "correct": 1, "score": 1.0},
                },
            },
            "samples": [
                {"sample_id": "code-1", "category": "code", "correct": False},
                {"sample_id": "math-1", "category": "math", "correct": True},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.json"
            self._write_json(path, report)
            scores, evidence = _teacher_scores(
                path,
                "teacher",
                dataset_sha256=dataset_sha,
                selected_manifest=manifest,
                protocol=protocol,
            )
            self.assertEqual(scores, {"code": 0.0, "math": 1.0})
            self.assertEqual(evidence["sample_count"], 2)

            invalid_reports = {}
            invalid_reports["dataset"] = copy.deepcopy(report)
            invalid_reports["dataset"]["dataset"]["sha256"] = "c" * 64
            invalid_reports["protocol"] = copy.deepcopy(report)
            invalid_reports["protocol"]["evaluation_protocol"]["max_input_tokens"] = 512
            invalid_reports["sample_set"] = copy.deepcopy(report)
            invalid_reports["sample_set"]["samples"][0]["sample_id"] = "code-other"
            invalid_reports["completed_count"] = copy.deepcopy(report)
            invalid_reports["completed_count"]["summary"]["completed_samples"] = 1
            invalid_reports["fabricated_summary"] = copy.deepcopy(report)
            invalid_reports["fabricated_summary"]["summary"]["categories"]["code"].update(
                {"correct": 1, "score": 1.0}
            )
            for name, invalid in invalid_reports.items():
                with self.subTest(name=name):
                    self._write_json(path, invalid)
                    with self.assertRaises(ManifestError):
                        _teacher_scores(
                            path,
                            "teacher",
                            dataset_sha256=dataset_sha,
                            selected_manifest=manifest,
                            protocol=protocol,
                        )

    def test_adapter_evidence_binds_training_manifest_and_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            self._write_json(adapter / "adapter_config.json", {"r": 8})
            training_manifest = root / "training_manifest.json"
            isolation = {
                "evaluation_prompt_overlap": 0,
                "scene_specific_samples": 0,
                "evaluation_set_used_for_training": False,
            }
            self._write_json(training_manifest, {**isolation, "train_rows": 10})
            metrics = {
                **isolation,
                "dataset_manifest_sha256": sha256_file(training_manifest),
            }
            self._write_json(adapter / "train_metrics.json", metrics)

            evidence = _validate_adapter_training_evidence(adapter, training_manifest)
            self.assertEqual(
                evidence["dataset_manifest"]["sha256"], sha256_file(training_manifest)
            )
            self.assertFalse(evidence["data_isolation"]["evaluation_set_used_for_training"])

            self._write_json(
                adapter / "train_metrics.json",
                {**metrics, "evaluation_set_used_for_training": True},
            )
            with self.assertRaises(ManifestError):
                _validate_adapter_training_evidence(adapter, training_manifest)

            self._write_json(adapter / "train_metrics.json", metrics)
            self._write_json(training_manifest, {**isolation, "train_rows": 11})
            with self.assertRaises(ManifestError):
                _validate_adapter_training_evidence(adapter, training_manifest)

    def test_category_selection_rejects_unknown_category(self):
        with self.assertRaises(ManifestError):
            _selected_categories(["traffic"])

    def test_empty_think_prefill_matches_renderer_contract(self):
        prompt = "<|im_start|>assistant\n"
        self.assertEqual(
            _apply_assistant_prefill(prompt, "empty_think"),
            prompt + "<think>\n\n</think>\n\n",
        )
        self.assertEqual(_apply_assistant_prefill(prompt, "none"), prompt)
        with self.assertRaises(ManifestError):
            _apply_assistant_prefill(prompt, "unknown")

    def test_summary_reports_each_category_and_sample_count(self):
        summary = _summary(
            [
                {"category": "math", "correct": True, "generation_ms": 10.0},
                {"category": "math", "correct": False, "generation_ms": 20.0},
                {
                    "category": "natural_language_reasoning",
                    "correct": True,
                    "generation_ms": 5.0,
                },
            ]
        )
        self.assertEqual(summary["categories"]["math"]["score"], 0.5)
        self.assertEqual(summary["categories"]["math"]["sample_count"], 2)
        self.assertEqual(summary["categories"]["math"]["average_generation_ms"], 15.0)
        self.assertEqual(summary["overall_micro_score"], 0.666667)


if __name__ == "__main__":
    unittest.main()
