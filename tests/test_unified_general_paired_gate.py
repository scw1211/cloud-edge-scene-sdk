import json
import tempfile
import unittest
from pathlib import Path

import edge_llm_factory.evaluate_unified_general_adapter as candidate_evaluator
import edge_llm_factory.evaluate_unified_general_teacher as teacher_evaluator
from edge_llm_factory.contracts import ManifestError, sha256_file
from edge_llm_factory.evaluate_unified_general_adapter import (
    EVALUATION_SCHEMA_VERSION as CANDIDATE_SCHEMA,
    MAX_NEW_TOKENS,
)
from edge_llm_factory.evaluate_unified_general_teacher import (
    EVALUATION_SCHEMA_VERSION as TEACHER_SCHEMA,
)
from edge_llm_factory.gate_unified_general_paired import run_gate


class UnifiedGeneralPairedGateTest(unittest.TestCase):
    @staticmethod
    def _write_json(path, value):
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )

    @staticmethod
    def _write_jsonl(path, rows):
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _rows(correct_by_category, valid_override=None):
        rows = []
        valid_override = valid_override or {}
        for category, correct_values in correct_by_category.items():
            for index, correct in enumerate(correct_values):
                sample_id = f"{category}:{index}"
                valid = valid_override.get(sample_id, True)
                if not valid:
                    raw_output = "invalid"
                    prediction = None
                elif category == "math":
                    raw_output = "FINAL: 15" if correct else "FINAL: 16"
                    prediction = "15" if correct else "16"
                else:
                    raw_output = "C" if correct else "A"
                    prediction = "C" if correct else "A"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "category": category,
                        "row_sha256": f"{index + 1:064x}",
                        "reference": "15" if category == "math" else "C",
                        "raw_output": raw_output,
                        "prediction": prediction,
                        "valid": valid,
                        "correct": correct,
                    }
                )
        return rows

    @staticmethod
    def _protocol(schema):
        common = {
            "uses_each_row_system_prompt_verbatim": True,
            "uses_each_row_prompt_verbatim": True,
            "max_new_tokens": dict(MAX_NEW_TOKENS),
            "math_scoring": "strict_trailing_FINAL_numeric",
            "logic_scoring": "strict_full_output_A_to_D",
        }
        if schema == TEACHER_SCHEMA:
            return {
                **common,
                "renderer": "ollama_api_chat",
                "max_input_tokens": 2048,
                "decoding": {
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 42,
                    "thinking": False,
                },
            }
        return {
            **common,
            "renderer": "tokenizer.apply_chat_template",
            "max_input_tokens": 512,
            "decoding": {"do_sample": False},
            "precision": "bfloat16",
        }

    def _summary(self, schema, samples_path, rows):
        categories = {}
        for category in ("math", "natural_language_reasoning"):
            selected = [row for row in rows if row["category"] == category]
            correct = sum(row["correct"] for row in selected)
            valid = sum(row["valid"] for row in selected)
            categories[category] = {
                "sample_count": len(selected),
                "correct": correct,
                "valid": valid,
                "accuracy": round(correct / len(selected), 6),
                "valid_rate": round(valid / len(selected), 6),
            }
        summary = {
            "schema_version": schema,
            "development_only": True,
            "formal_blind_test_used": False,
            "dataset": {
                "sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "blind_test_content_loaded": False,
            },
            "selection": {
                "categories": ["math", "natural_language_reasoning"],
                "limit_per_category": 400,
                "sample_count": len(rows),
                "selected_rows_sha256": "c" * 64,
            },
            "evaluation_protocol": self._protocol(schema),
            "metrics": {"categories": categories},
            "model_identity": (
                {
                    "provider": "ollama",
                    "model": "qwen3.5:9b",
                    "expected_model_sha256": "e" * 64,
                    "model_sha256": "e" * 64,
                    "show_response_sha256": "f" * 64,
                    "version_response_sha256": "1" * 64,
                    "stable_across_evaluation": True,
                    "attestation_before": {
                        "provider": "ollama",
                        "endpoint": "http://localhost:11434",
                        "model": "qwen3.5:9b",
                        "expected_model_sha256": "e" * 64,
                        "model_sha256": "e" * 64,
                        "show_response_sha256": "f" * 64,
                        "version_response_sha256": "1" * 64,
                    },
                    "attestation_after": {
                        "provider": "ollama",
                        "endpoint": "http://localhost:11434",
                        "model": "qwen3.5:9b",
                        "expected_model_sha256": "e" * 64,
                        "model_sha256": "e" * 64,
                        "show_response_sha256": "f" * 64,
                        "version_response_sha256": "1" * 64,
                    },
                }
                if schema == TEACHER_SCHEMA
                else {
                    "model_sha256": "d" * 64,
                    "adapter_artifact_sha256": "2" * 64,
                }
            ),
            "artifacts": {
                "samples": {
                    "path": str(samples_path.resolve()),
                    "bytes": samples_path.stat().st_size,
                    "sha256": sha256_file(samples_path),
                },
                "evaluator": (
                    {
                        "path": str(Path(teacher_evaluator.__file__).resolve()),
                        "sha256": sha256_file(
                            Path(teacher_evaluator.__file__).resolve()
                        ),
                    }
                    if schema == TEACHER_SCHEMA
                    else {
                        "path": str(Path(candidate_evaluator.__file__).resolve()),
                        "bytes": Path(candidate_evaluator.__file__).stat().st_size,
                        "sha256": sha256_file(
                            Path(candidate_evaluator.__file__).resolve()
                        ),
                    }
                ),
            },
        }
        if schema == CANDIDATE_SCHEMA:
            adapter_model = samples_path.parent / "adapter_model.safetensors"
            summary["adapter"] = {
                "path": str(adapter_model.parent.resolve()),
                "artifact_sha256": "2" * 64,
                "weights_binding": {
                    "expected_sha256": sha256_file(adapter_model),
                    "actual_sha256": sha256_file(adapter_model),
                    "externally_anchored_and_recomputed": True,
                },
                "files": {
                    "adapter_weights": {
                        "path": str(adapter_model.resolve()),
                        "bytes": adapter_model.stat().st_size,
                        "sha256": sha256_file(adapter_model),
                    }
                },
            }
        return summary

    def _fixture(self, root, candidate_rows, teacher_rows):
        candidate_samples = root / "candidate.jsonl"
        teacher_samples = root / "teacher.jsonl"
        candidate_summary = root / "candidate_summary.json"
        teacher_summary = root / "teacher_summary.json"
        candidate_adapter_model = root / "adapter_model.safetensors"
        candidate_adapter_model.write_bytes(b"candidate-u2-weights")
        self._write_jsonl(candidate_samples, candidate_rows)
        self._write_jsonl(teacher_samples, teacher_rows)
        self._write_json(
            candidate_summary,
            self._summary(CANDIDATE_SCHEMA, candidate_samples, candidate_rows),
        )
        self._write_json(
            teacher_summary,
            self._summary(TEACHER_SCHEMA, teacher_samples, teacher_rows),
        )
        return (
            candidate_samples,
            candidate_summary,
            teacher_samples,
            teacher_summary,
            candidate_adapter_model,
        )

    @staticmethod
    def _run(paths, output):
        return run_gate(
            candidate_samples_path=paths[0],
            candidate_summary_path=paths[1],
            teacher_samples_path=paths[2],
            teacher_summary_path=paths[3],
            output_path=output,
            candidate_adapter_model_path=paths[4],
            expected_candidate_adapter_sha256=sha256_file(paths[4]),
            expected_candidate_evaluator_sha256=sha256_file(
                Path(candidate_evaluator.__file__).resolve()
            ),
            expected_teacher_evaluator_sha256=sha256_file(
                Path(teacher_evaluator.__file__).resolve()
            ),
            seed=20260810,
            bootstrap_iterations=200,
        )

    def test_passes_and_reports_paired_corrections_and_reproducible_ci(self):
        teacher = self._rows(
            {
                "math": [True] * 399 + [False],
                "natural_language_reasoning": [True] * 400,
            }
        )
        candidate = self._rows(
            {
                "math": [True] * 398 + [False, True],
                "natural_language_reasoning": [True] * 400,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root, candidate, teacher)
            first = self._run(paths, root / "gate1.json")
            second = self._run(paths, root / "gate2.json")
            self.assertTrue(first["passed"])
            math = first["categories"]["math"]
            self.assertEqual(math["retention"], 1.0)
            self.assertEqual(math["teacher_correct_set_candidate_coverage"], 0.997494)
            self.assertEqual(math["corrected_teacher_errors"], 1)
            self.assertEqual(math["regressed_teacher_correct"], 1)
            self.assertEqual(math["net_corrections"], 0)
            self.assertEqual(math["bootstrap"], second["categories"]["math"]["bootstrap"])
            self.assertEqual(first["inputs"]["candidate"]["samples"]["sha256"], sha256_file(paths[0]))
            self.assertTrue(
                first["inputs"]["candidate"]["adapter_model"][
                    "externally_anchored_and_recomputed"
                ]
            )

    def test_fails_when_category_retention_or_strict_valid_rate_misses(self):
        teacher = self._rows(
            {
                "math": [True] * 400,
                "natural_language_reasoning": [True] * 400,
            }
        )
        candidate = self._rows(
            {
                "math": [True] * 400,
                "natural_language_reasoning": [True] * 200 + [False] * 200,
            },
            valid_override={"natural_language_reasoning:399": False},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self._run(
                self._fixture(root, candidate, teacher), root / "gate.json"
            )
            self.assertFalse(report["passed"])
            logic = report["categories"]["natural_language_reasoning"]
            self.assertEqual(logic["retention"], 0.5)
            self.assertFalse(logic["checks"]["retention_at_least_0p80"])
            self.assertFalse(logic["checks"]["candidate_valid_rate_equals_1"])

    def test_rejects_pair_mismatch_and_summary_sample_tamper(self):
        correct = {
            "math": [True] * 400,
            "natural_language_reasoning": [True] * 400,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._rows(correct)
            teacher = self._rows(correct)
            teacher[0]["reference"] = "16"
            teacher[0]["raw_output"] = "FINAL: 16"
            teacher[0]["prediction"] = "16"
            paths = self._fixture(root, candidate, teacher)
            with self.assertRaisesRegex(ManifestError, "reference 不一致"):
                self._run(paths, root / "gate.json")

    def test_rejects_raw_output_tamper_even_when_claimed_flags_and_hash_are_updated(self):
        correct = {
            "math": [True] * 400,
            "natural_language_reasoning": [True] * 400,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._rows(correct)
            teacher = self._rows(correct)
            paths = self._fixture(root, candidate, teacher)
            candidate[0]["raw_output"] = "FINAL: 999"
            self._write_jsonl(paths[0], candidate)
            self._write_json(
                paths[1], self._summary(CANDIDATE_SCHEMA, paths[0], candidate)
            )
            with self.assertRaisesRegex(ManifestError, "独立重判"):
                self._run(paths, root / "gate.json")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root, self._rows(correct), self._rows(correct))
            paths[0].write_text(paths[0].read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "SHA-256"):
                self._run(paths, root / "gate.json")

    def test_refuses_overwrite_before_processing(self):
        correct = {
            "math": [True] * 400,
            "natural_language_reasoning": [True] * 400,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root, self._rows(correct), self._rows(correct))
            output = root / "gate.json"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "拒绝覆盖"):
                self._run(paths, output)

    def test_rejects_non_frozen_selection_size_limit_and_category_balance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            short = {
                "math": [True] * 399,
                "natural_language_reasoning": [True] * 399,
            }
            paths = self._fixture(root, self._rows(short), self._rows(short))
            with self.assertRaisesRegex(ManifestError, "sample_count 必须为 800"):
                self._run(paths, root / "short.json")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            correct = {
                "math": [True] * 400,
                "natural_language_reasoning": [True] * 400,
            }
            paths = self._fixture(root, self._rows(correct), self._rows(correct))
            for summary_path, schema, samples_path, rows in (
                (paths[1], CANDIDATE_SCHEMA, paths[0], self._rows(correct)),
                (paths[3], TEACHER_SCHEMA, paths[2], self._rows(correct)),
            ):
                summary = self._summary(schema, samples_path, rows)
                summary["selection"]["limit_per_category"] = 0
                self._write_json(summary_path, summary)
            with self.assertRaisesRegex(ManifestError, "limit_per_category 必须为 400"):
                self._run(paths, root / "bad_limit.json")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unbalanced = {
                "math": [True] * 399,
                "natural_language_reasoning": [True] * 401,
            }
            paths = self._fixture(
                root, self._rows(unbalanced), self._rows(unbalanced)
            )
            with self.assertRaisesRegex(ManifestError, "必须恰好有 400 条"):
                self._run(paths, root / "unbalanced.json")

    def test_rejects_candidate_protocol_drift(self):
        correct = {
            "math": [True] * 400,
            "natural_language_reasoning": [True] * 400,
        }
        mutations = (
            ("precision", lambda value: value.__setitem__("precision", "float16")),
            ("max_input_tokens", lambda value: value.__setitem__("max_input_tokens", 511)),
            (
                "max_new_tokens",
                lambda value: value["max_new_tokens"].__setitem__("math", 255),
            ),
            (
                "do_sample",
                lambda value: value["decoding"].__setitem__("do_sample", True),
            ),
            (
                "math_scoring",
                lambda value: value.__setitem__("math_scoring", "loose_numeric"),
            ),
            (
                "logic_scoring",
                lambda value: value.__setitem__("logic_scoring", "contains_A_to_D"),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_rows = self._rows(correct)
            teacher_rows = self._rows(correct)
            paths = self._fixture(root, candidate_rows, teacher_rows)
            for name, mutate in mutations:
                with self.subTest(field=name):
                    summary = self._summary(
                        CANDIDATE_SCHEMA, paths[0], candidate_rows
                    )
                    mutate(summary["evaluation_protocol"])
                    self._write_json(paths[1], summary)
                    with self.assertRaisesRegex(ManifestError, "candidate"):
                        self._run(paths, root / f"candidate_{name}.json")

    def test_rejects_teacher_protocol_drift(self):
        correct = {
            "math": [True] * 400,
            "natural_language_reasoning": [True] * 400,
        }
        mutations = (
            ("max_input_tokens", lambda value: value.__setitem__("max_input_tokens", 1024)),
            (
                "max_new_tokens",
                lambda value: value["max_new_tokens"].__setitem__(
                    "natural_language_reasoning", 31
                ),
            ),
            (
                "temperature",
                lambda value: value["decoding"].__setitem__("temperature", 0.1),
            ),
            ("top_p", lambda value: value["decoding"].__setitem__("top_p", 0.9)),
            ("seed", lambda value: value["decoding"].__setitem__("seed", 43)),
            (
                "thinking",
                lambda value: value["decoding"].__setitem__("thinking", True),
            ),
            (
                "math_scoring",
                lambda value: value.__setitem__("math_scoring", "loose_numeric"),
            ),
            (
                "logic_scoring",
                lambda value: value.__setitem__("logic_scoring", "contains_A_to_D"),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_rows = self._rows(correct)
            teacher_rows = self._rows(correct)
            paths = self._fixture(root, candidate_rows, teacher_rows)
            for name, mutate in mutations:
                with self.subTest(field=name):
                    summary = self._summary(TEACHER_SCHEMA, paths[2], teacher_rows)
                    mutate(summary["evaluation_protocol"])
                    self._write_json(paths[3], summary)
                    with self.assertRaisesRegex(ManifestError, "teacher"):
                        self._run(paths, root / f"teacher_{name}.json")

    def test_rejects_evaluator_anchor_summary_or_path_drift(self):
        correct = {
            "math": [True] * 400,
            "natural_language_reasoning": [True] * 400,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self._rows(correct)
            paths = self._fixture(root, rows, rows)
            with self.assertRaisesRegex(ManifestError, "当前评测器 SHA-256"):
                run_gate(
                    candidate_samples_path=paths[0],
                    candidate_summary_path=paths[1],
                    teacher_samples_path=paths[2],
                    teacher_summary_path=paths[3],
                    output_path=root / "bad_anchor.json",
                    candidate_adapter_model_path=paths[4],
                    expected_candidate_adapter_sha256=sha256_file(paths[4]),
                    expected_candidate_evaluator_sha256="0" * 64,
                    expected_teacher_evaluator_sha256=sha256_file(
                        Path(teacher_evaluator.__file__).resolve()
                    ),
                    seed=20260810,
                    bootstrap_iterations=10,
                )

            candidate_summary = self._summary(CANDIDATE_SCHEMA, paths[0], rows)
            candidate_summary["artifacts"]["evaluator"]["sha256"] = "0" * 64
            self._write_json(paths[1], candidate_summary)
            with self.assertRaisesRegex(ManifestError, "summary 评测器 SHA-256"):
                self._run(paths, root / "bad_candidate_summary_sha.json")

            candidate_summary = self._summary(CANDIDATE_SCHEMA, paths[0], rows)
            candidate_summary["artifacts"]["evaluator"]["path"] = str(
                root / "not_the_evaluator.py"
            )
            self._write_json(paths[1], candidate_summary)
            with self.assertRaisesRegex(ManifestError, "不是当前冻结文件"):
                self._run(paths, root / "bad_candidate_path.json")

            self._write_json(
                paths[1], self._summary(CANDIDATE_SCHEMA, paths[0], rows)
            )
            teacher_summary = self._summary(TEACHER_SCHEMA, paths[2], rows)
            teacher_summary["artifacts"]["evaluator"]["sha256"] = "0" * 64
            self._write_json(paths[3], teacher_summary)
            with self.assertRaisesRegex(ManifestError, "summary 评测器 SHA-256"):
                self._run(paths, root / "bad_teacher_summary_sha.json")

    def test_rejects_candidate_adapter_external_anchor_or_summary_tamper(self):
        correct = {
            "math": [True] * 400,
            "natural_language_reasoning": [True] * 400,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self._rows(correct)
            paths = self._fixture(root, rows, rows)
            with self.assertRaisesRegex(ManifestError, "外部锚点"):
                run_gate(
                    candidate_samples_path=paths[0],
                    candidate_summary_path=paths[1],
                    teacher_samples_path=paths[2],
                    teacher_summary_path=paths[3],
                    output_path=root / "bad_adapter_anchor.json",
                    candidate_adapter_model_path=paths[4],
                    expected_candidate_adapter_sha256="0" * 64,
                    expected_candidate_evaluator_sha256=sha256_file(
                        Path(candidate_evaluator.__file__).resolve()
                    ),
                    expected_teacher_evaluator_sha256=sha256_file(
                        Path(teacher_evaluator.__file__).resolve()
                    ),
                    seed=20260810,
                    bootstrap_iterations=10,
                )

            summary = self._summary(CANDIDATE_SCHEMA, paths[0], rows)
            summary["adapter"]["files"]["adapter_weights"]["sha256"] = "0" * 64
            self._write_json(paths[1], summary)
            with self.assertRaisesRegex(ManifestError, "summary 权重 SHA-256"):
                self._run(paths, root / "bad_adapter_summary.json")


if __name__ == "__main__":
    unittest.main()
