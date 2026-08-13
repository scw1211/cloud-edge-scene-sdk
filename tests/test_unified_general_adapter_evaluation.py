import json
import tempfile
import unittest
from pathlib import Path

from edge_llm_factory.contracts import ManifestError, sha256_file
from edge_llm_factory.evaluate_unified_general_adapter import (
    DATASET_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION_V2,
    adapter_identity,
    evaluator_identity,
    evaluate_choice,
    evaluate_math,
    evaluate_rows,
    read_evaluation_rows,
    summarize,
    validate_dataset_binding,
)


class UnifiedGeneralAdapterEvaluationTest(unittest.TestCase):
    def _write_json(self, path: Path, value):
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )

    def _write_jsonl(self, path: Path, rows):
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _rows():
        return [
            {
                "sample_id": "math:validation:1",
                "category": "math",
                "system_prompt": "[TASK:MATH]\nKeep this exact.",
                "prompt": "12 + 3 = ?",
                "reference_answer": "15",
            },
            {
                "sample_id": "logic:validation:1",
                "category": "natural_language_reasoning",
                "system_prompt": "[任务:中文逻辑选择]\n只输出选项。",
                "prompt": "请选择。\nA.甲\nB.乙\nC.丙\nD.丁",
                "reference_answer": "C",
            },
        ]

    @staticmethod
    def _v2_rows():
        rows = []
        for index in range(400):
            rows.append(
                {
                    "sample_id": f"math:promotion:{index}",
                    "category": "math",
                    "system_prompt": "[TASK:MATH]\nReturn FINAL: <number>",
                    "prompt": f"{index} + 1 = ?",
                    "reference_answer": str(index + 1),
                }
            )
            rows.append(
                {
                    "sample_id": f"logic:promotion:{index}",
                    "category": "natural_language_reasoning",
                    "system_prompt": "[任务:中文逻辑选择]\n只输出选项。",
                    "prompt": f"第 {index} 题。\nA.甲\nB.乙\nC.丙\nD.丁",
                    "reference_answer": "C",
                }
            )
        return rows

    @staticmethod
    def _v2_manifest(dataset: Path):
        return {
            "schema_version": DATASET_SCHEMA_VERSION_V2,
            "promotion_dev_rows": 800,
            "promotion_dev_category_counts": {
                "math": 400,
                "natural_language_reasoning": 400,
            },
            "general_train_validation_prompt_overlap": 0,
            "general_train_promotion_prompt_overlap": 0,
            "general_validation_promotion_prompt_overlap": 0,
            "u1_general_dev_overlap_with_train_or_promotion": 0,
            "u1_general_dev_used_as_training_validation": True,
            "u1_general_dev_used_for_promotion": False,
            "traffic_test_used_for_training": False,
            "gsm8k_test_loaded": False,
            "logiqa_test_loaded": False,
            "formal_evaluation_used_for_training": False,
            "u1_evaluation_outputs_used_for_training": False,
            "blind_evaluation_used_for_training": False,
            "promotion_dev_used_for_training": False,
            "promotion_dev_used_for_model_selection": False,
            "blind_content_policy": {
                "traffic_test": "sha256_and_stat_only",
                "gsm8k_test": "split_not_loaded",
                "logiqa_test": "sha256_and_stat_only",
                "test_content_used_for_training_or_development": False,
            },
            "artifacts": {
                "general_dev_evaluation": {
                    "path": dataset.name,
                    "rows": 800,
                    "bytes": dataset.stat().st_size,
                    "sha256": sha256_file(dataset),
                }
            },
            "sources": {
                "traffic": {
                    "test": {
                        "content_loaded": False,
                        "used_for_training": False,
                        "access_mode": "sha256_and_stat_only",
                    }
                },
                "gsm8k": {"test_split_loaded": False},
                "logiqa2": {
                    "test_split_loaded": False,
                    "test": {
                        "content_loaded": False,
                        "access_mode": "sha256_and_stat_only",
                    },
                },
            },
        }

    def test_strict_math_and_choice_scoring(self):
        self.assertTrue(evaluate_math("过程\nFINAL: 1,234.5", "1234.5")["correct"])
        for output in ("15", "FINAL: 10 + 5 = 15", "FINAL: fifteen"):
            with self.subTest(output=output):
                self.assertFalse(evaluate_math(output, "15")["valid"])
        self.assertTrue(evaluate_choice(" c ", "C")["correct"])
        for output in ("答案是 C", "C，因为", "A/C", "<think></think>C"):
            with self.subTest(output=output):
                self.assertFalse(evaluate_choice(output, "C")["valid"])

    def test_stub_generation_receives_row_prompts_verbatim_and_summarizes(self):
        rows = self._rows()
        rows[0]["system_prompt"] = "  [TASK:MATH]\nKeep this exact.\n"
        rows[0]["prompt"] = "\n12 + 3 = ?  "
        seen = []

        def stub(messages, max_new_tokens, row):
            seen.append((messages, max_new_tokens, row["sample_id"]))
            output = "work\nFINAL: 15" if row["category"] == "math" else "C"
            return {
                "raw_output": output,
                "prompt_tokens": 11,
                "output_tokens": 2,
                "generation_ms": 10.0 if row["category"] == "math" else 20.0,
            }

        samples = evaluate_rows(rows, stub)
        self.assertEqual(
            seen[0][0],
            [
                {"role": "system", "content": rows[0]["system_prompt"]},
                {"role": "user", "content": rows[0]["prompt"]},
            ],
        )
        self.assertEqual(seen[0][1], 256)
        self.assertEqual(seen[1][1], 32)
        metrics = summarize(samples)
        self.assertEqual(metrics["sample_count"], 2)
        self.assertEqual(metrics["correct"], 2)
        self.assertEqual(metrics["valid"], 2)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["latency"]["p50_ms"], 15.0)
        self.assertEqual(
            metrics["categories"]["natural_language_reasoning"]["valid_rate"],
            1.0,
        )

    def test_jsonl_contract_rejects_duplicate_unknown_and_missing_system_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "general_dev_evaluation.jsonl"
            self._write_jsonl(path, self._rows())
            loaded = read_evaluation_rows(path)
            self.assertEqual(len(loaded), 2)
            invalid = {
                "duplicate": [self._rows()[0], self._rows()[0]],
                "unknown": [{**self._rows()[0], "category": "traffic_action"}],
                "missing_system": [
                    {key: value for key, value in self._rows()[0].items() if key != "system_prompt"}
                ],
                "bad_reference": [{**self._rows()[1], "reference_answer": "E"}],
            }
            for name, rows in invalid.items():
                with self.subTest(name=name):
                    self._write_jsonl(path, rows)
                    with self.assertRaises(ManifestError):
                        read_evaluation_rows(path)

    def test_dataset_binding_proves_development_only_and_rejects_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "general_dev_evaluation.jsonl"
            manifest = root / "manifest.json"
            self._write_jsonl(dataset, self._rows())
            value = {
                "schema_version": DATASET_SCHEMA_VERSION,
                "general_train_validation_prompt_overlap": 0,
                "gsm8k_test_loaded": False,
                "logiqa_test_loaded": False,
                "artifacts": {
                    "general_dev_evaluation": {
                        "path": dataset.name,
                        "sha256": sha256_file(dataset),
                    }
                },
                "sources": {
                    "gsm8k": {"test_split_loaded": False},
                    "logiqa2": {
                        "test_split_loaded": False,
                        "test_content_loaded": False,
                    },
                },
            }
            self._write_json(manifest, value)
            report = validate_dataset_binding(dataset, manifest)
            self.assertFalse(report["blind_test_content_loaded"])
            self.assertEqual(report["training_development_prompt_overlap"], 0)

            dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "SHA-256"):
                validate_dataset_binding(dataset, manifest)

    def test_v2_binding_accepts_only_frozen_800_row_promotion_dev(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "general_dev_evaluation.jsonl"
            manifest = root / "manifest.json"
            self._write_jsonl(dataset, self._v2_rows())
            self._write_json(manifest, self._v2_manifest(dataset))

            report = validate_dataset_binding(dataset, manifest)
            self.assertEqual(report["schema_version"], DATASET_SCHEMA_VERSION_V2)
            self.assertEqual(report["dataset_role"], "promotion_development")
            self.assertEqual(report["promotion_rows"], 800)
            self.assertEqual(
                report["promotion_category_counts"],
                {"math": 400, "natural_language_reasoning": 400},
            )
            self.assertFalse(report["blind_test_content_loaded"])
            self.assertFalse(report["u1_general_dev_used_for_promotion"])

    def test_v2_binding_rejects_each_promotion_or_blind_isolation_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "general_dev_evaluation.jsonl"
            manifest = root / "manifest.json"
            self._write_jsonl(dataset, self._v2_rows())
            clean = self._v2_manifest(dataset)

            mutations = {
                "artifact_rows": lambda value: value["artifacts"][
                    "general_dev_evaluation"
                ].update(rows=799),
                "promotion_rows": lambda value: value.update(promotion_dev_rows=799),
                "promotion_counts": lambda value: value.update(
                    promotion_dev_category_counts={
                        "math": 399,
                        "natural_language_reasoning": 401,
                    }
                ),
                "train_validation_overlap": lambda value: value.update(
                    general_train_validation_prompt_overlap=1
                ),
                "train_promotion_overlap": lambda value: value.update(
                    general_train_promotion_prompt_overlap=1
                ),
                "validation_promotion_overlap": lambda value: value.update(
                    general_validation_promotion_prompt_overlap=1
                ),
                "u1_overlap": lambda value: value.update(
                    u1_general_dev_overlap_with_train_or_promotion=1
                ),
                "u1_used_for_promotion": lambda value: value.update(
                    u1_general_dev_used_for_promotion=True
                ),
                "promotion_used_for_training": lambda value: value.update(
                    promotion_dev_used_for_training=True
                ),
                "promotion_used_for_selection": lambda value: value.update(
                    promotion_dev_used_for_model_selection=True
                ),
                "blind_test_loaded": lambda value: value.update(
                    logiqa_test_loaded=True
                ),
                "source_test_loaded": lambda value: value["sources"]["logiqa2"][
                    "test"
                ].update(content_loaded=True),
                "promotion_path_escape": lambda value: value["artifacts"][
                    "general_dev_evaluation"
                ].update(path="../blind.jsonl"),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    value = json.loads(json.dumps(clean))
                    mutate(value)
                    self._write_json(manifest, value)
                    with self.assertRaises(ManifestError):
                        validate_dataset_binding(dataset, manifest)

    def test_v2_binding_checks_actual_promotion_category_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "general_dev_evaluation.jsonl"
            manifest = root / "manifest.json"
            rows = self._v2_rows()
            rows[-1] = {
                **rows[-1],
                "sample_id": "math:promotion:extra",
                "category": "math",
                "reference_answer": "1",
            }
            self._write_jsonl(dataset, rows)
            self._write_json(manifest, self._v2_manifest(dataset))
            with self.assertRaisesRegex(ManifestError, "实际内容"):
                validate_dataset_binding(dataset, manifest)

    def test_adapter_identity_hashes_model_and_binds_unified_training_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            self._write_json(
                adapter / "adapter_config.json",
                {"base_model_name_or_path": "Qwen/Qwen3.5-0.8B", "r": 16},
            )
            metrics = {
                "dataset_manifest_sha256": "a" * 64,
                "formal_evaluation_used_for_training": False,
                "traffic_test_used_for_training": False,
                "gsm8k_test_loaded": False,
                "logiqa_test_loaded": False,
            }
            self._write_json(adapter / "train_metrics.json", metrics)
            base = {"source": {"model_id": "Qwen/Qwen3.5-0.8B"}}
            first = adapter_identity(adapter, base, "a" * 64)
            second = adapter_identity(adapter, base, "a" * 64)
            self.assertEqual(first["artifact_sha256"], second["artifact_sha256"])
            self.assertEqual(
                first["training_data_binding"]["dataset_manifest_sha256"], "a" * 64
            )

            with self.assertRaisesRegex(ManifestError, "训练 manifest"):
                adapter_identity(adapter, base, "b" * 64)

            anchored = adapter_identity(
                adapter,
                base,
                "a" * 64,
                sha256_file(adapter / "adapter_model.safetensors"),
            )
            self.assertTrue(
                anchored["weights_binding"]["externally_anchored_and_recomputed"]
            )
            with self.assertRaisesRegex(ManifestError, "外部 SHA-256"):
                adapter_identity(adapter, base, "a" * 64, "0" * 64)

    def test_adapter_identity_accepts_u2_nested_receipt_and_checks_all_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"u2-weights")
            self._write_json(
                adapter / "adapter_config.json",
                {"base_model_name_or_path": "Qwen/Qwen3.5-0.8B"},
            )
            isolation = {
                field: False
                for field in (
                    "traffic_test_used_for_training",
                    "gsm8k_test_loaded",
                    "logiqa_test_loaded",
                    "formal_evaluation_used_for_training",
                    "u1_evaluation_outputs_used_for_training",
                    "blind_evaluation_used_for_training",
                    "promotion_dev_used_for_training",
                    "promotion_dev_used_for_model_selection",
                )
            }
            metrics = {
                "dataset": {
                    "schema_version": DATASET_SCHEMA_VERSION_V2,
                    "manifest_sha256": "a" * 64,
                    "promotion_dev_content_read_by_trainer": False,
                    "isolation": isolation,
                }
            }
            self._write_json(adapter / "train_metrics.json", metrics)
            base = {"source": {"model_id": "Qwen/Qwen3.5-0.8B"}}
            report = adapter_identity(adapter, base, "a" * 64)
            binding = report["training_data_binding"]
            self.assertEqual(binding["receipt_format"], "u2_nested_dataset")
            self.assertEqual(binding["dataset_manifest_sha256"], "a" * 64)
            self.assertFalse(binding["promotion_dev_content_read_by_trainer"])

            metrics["dataset"]["isolation"][
                "blind_evaluation_used_for_training"
            ] = True
            self._write_json(adapter / "train_metrics.json", metrics)
            with self.assertRaisesRegex(
                ManifestError, "blind_evaluation_used_for_training"
            ):
                adapter_identity(adapter, base, "a" * 64)

    def test_evaluator_identity_binds_current_path_size_and_sha256(self):
        identity = evaluator_identity()
        path = Path(identity["path"])
        self.assertEqual(path.resolve(), path)
        self.assertTrue(path.is_file())
        self.assertEqual(identity["bytes"], path.stat().st_size)
        self.assertEqual(identity["sha256"], sha256_file(path))


if __name__ == "__main__":
    unittest.main()
