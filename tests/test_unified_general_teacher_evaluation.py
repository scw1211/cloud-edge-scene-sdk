import json
import tempfile
import unittest
from pathlib import Path

from edge_llm_factory.contracts import ManifestError, sha256_file
from edge_llm_factory.evaluate_unified_general_adapter import (
    DATASET_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION_V2,
)
from edge_llm_factory.evaluate_unified_general_teacher import (
    OllamaTeacherGenerator,
    TEACHER_MODEL,
    attest_teacher_model,
    run_evaluation,
)


class FakeOllama:
    def __init__(self):
        self.calls = []

    def __call__(self, method, url, payload, timeout_seconds):
        self.calls.append((method, url, payload, timeout_seconds))
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": TEACHER_MODEL,
                        "model": TEACHER_MODEL,
                        "digest": "sha256:" + "a" * 64,
                    }
                ]
            }
        if url.endswith("/api/show"):
            return {"details": {"family": "qwen3"}, "capabilities": ["completion"]}
        if url.endswith("/api/version"):
            return {"version": "0.test"}
        if url.endswith("/api/chat"):
            category = "math" if payload["options"]["num_predict"] == 256 else "logic"
            return {
                "model": TEACHER_MODEL,
                "done": True,
                "message": {"content": "work\nFINAL: 15" if category == "math" else "C"},
                "prompt_eval_count": 11,
                "eval_count": 2,
            }
        raise AssertionError(url)


class UnifiedGeneralTeacherEvaluationTest(unittest.TestCase):
    @staticmethod
    def _rows():
        return [
            {
                "sample_id": "math:validation:1",
                "category": "math",
                "system_prompt": "  [TASK:MATH]\nExact system.\n",
                "prompt": "\n12 + 3 = ?  ",
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
            rows.extend(
                [
                    {
                        "sample_id": f"math:promotion:{index}",
                        "category": "math",
                        "system_prompt": "[TASK:MATH]\nReturn FINAL: <number>",
                        "prompt": f"{index} + 1 = ?",
                        "reference_answer": str(index + 1),
                    },
                    {
                        "sample_id": f"logic:promotion:{index}",
                        "category": "natural_language_reasoning",
                        "system_prompt": "[任务:中文逻辑选择]\n只输出选项。",
                        "prompt": f"第 {index} 题。\nA.甲\nB.乙\nC.丙\nD.丁",
                        "reference_answer": "C",
                    },
                ]
            )
        return rows

    @staticmethod
    def _v2_manifest(dataset):
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

    @staticmethod
    def _write_json(path, value):
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )

    def test_attestation_locks_exact_model_and_hashes_runtime_evidence(self):
        fake = FakeOllama()
        result = attest_teacher_model(
            "http://127.0.0.1:11434/",
            10,
            fake,
            expected_model_sha256="a" * 64,
        )
        self.assertEqual(result["model"], TEACHER_MODEL)
        self.assertEqual(result["model_sha256"], "a" * 64)
        self.assertEqual(result["version_response"], {"version": "0.test"})
        self.assertEqual([call[1].rsplit("/", 2)[-1] for call in fake.calls], ["tags", "show", "version"])

    def test_attestation_rejects_alias_or_invalid_digest(self):
        def no_exact(method, url, payload, timeout):
            del method, payload, timeout
            if url.endswith("/api/tags"):
                return {"models": [{"name": "qwen3.5:latest", "digest": "a" * 64}]}
            raise AssertionError(url)

        with self.assertRaisesRegex(ManifestError, "精确解析"):
            attest_teacher_model("http://localhost:11434", 10, no_exact)

        def bad_digest(method, url, payload, timeout):
            del method, payload, timeout
            if url.endswith("/api/tags"):
                return {"models": [{"name": TEACHER_MODEL, "digest": "bad"}]}
            raise AssertionError(url)

        with self.assertRaisesRegex(ManifestError, "SHA-256"):
            attest_teacher_model("http://localhost:11434", 10, bad_digest)

        with self.assertRaisesRegex(ManifestError, "预注册"):
            attest_teacher_model(
                "http://localhost:11434",
                10,
                FakeOllama(),
                expected_model_sha256="b" * 64,
            )

    def test_generator_preserves_messages_and_uses_category_token_budget(self):
        fake = FakeOllama()
        generator = OllamaTeacherGenerator(
            "http://localhost:11434", 10, transport=fake, num_ctx=2048
        )
        row = self._rows()[0]
        messages = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["prompt"]},
        ]
        result = generator(messages, 256, row)
        payload = fake.calls[-1][2]
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["model"], TEACHER_MODEL)
        self.assertEqual(payload["think"], False)
        self.assertEqual(payload["options"]["num_predict"], 256)
        self.assertEqual(payload["options"]["num_ctx"], 2048)
        self.assertEqual(result["raw_output"], "work\nFINAL: 15")

    def test_generator_rejects_incomplete_or_wrong_model_response(self):
        row = self._rows()[0]
        messages = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["prompt"]},
        ]

        def response(**changes):
            value = {
                "model": TEACHER_MODEL,
                "done": True,
                "message": {"content": "FINAL: 15"},
                "prompt_eval_count": 11,
                "eval_count": 2,
            }
            value.update(changes)
            return value

        for expected_message, value in (
            ("模型必须精确", response(model="qwen3.5:latest")),
            ("done=true", response(done=False)),
            ("prompt_eval_count", {key: item for key, item in response().items() if key != "prompt_eval_count"}),
            ("eval_count", {key: item for key, item in response().items() if key != "eval_count"}),
        ):
            with self.subTest(expected_message=expected_message):
                def transport(method, url, payload, timeout, result=value):
                    del method, url, payload, timeout
                    return result

                generator = OllamaTeacherGenerator(
                    "http://localhost:11434", 10, transport=transport, num_ctx=2048
                )
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    generator(messages, 256, row)

    def test_run_rejects_attestation_drift_without_writing_outputs(self):
        class DriftingOllama(FakeOllama):
            def __init__(self):
                super().__init__()
                self.tag_calls = 0

            def __call__(self, method, url, payload, timeout_seconds):
                if url.endswith("/api/tags"):
                    self.tag_calls += 1
                    digest = "a" * 64 if self.tag_calls == 1 else "b" * 64
                    return {
                        "models": [
                            {
                                "name": TEACHER_MODEL,
                                "model": TEACHER_MODEL,
                                "digest": digest,
                            }
                        ]
                    }
                return super().__call__(method, url, payload, timeout_seconds)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "general_dev_evaluation.jsonl"
            dataset.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in self._rows()
                ),
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            self._write_json(
                manifest,
                {
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
                },
            )
            samples = root / "teacher_samples.jsonl"
            summary = root / "teacher_summary.json"
            with self.assertRaisesRegex(ManifestError, "预注册"):
                run_evaluation(
                    dataset=dataset,
                    dataset_manifest=manifest,
                    samples_output=samples,
                    summary_output=summary,
                    endpoint="http://localhost:11434",
                    expected_model_sha256="a" * 64,
                    timeout_seconds=10,
                    categories={"math", "natural_language_reasoning"},
                    limit_per_category=0,
                    num_ctx=2048,
                    transport=DriftingOllama(),
                )
            self.assertFalse(samples.exists())
            self.assertFalse(summary.exists())

    def test_teacher_rejects_v2_promotion_training_leak_before_ollama(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "general_dev_evaluation.jsonl"
            dataset.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in self._v2_rows()
                ),
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            value = self._v2_manifest(dataset)
            value["promotion_dev_used_for_training"] = True
            self._write_json(manifest, value)
            fake = FakeOllama()
            with self.assertRaisesRegex(ManifestError, "promotion_dev_used_for_training"):
                run_evaluation(
                    dataset=dataset,
                    dataset_manifest=manifest,
                    samples_output=root / "samples.jsonl",
                    summary_output=root / "summary.json",
                    endpoint="http://localhost:11434",
                    expected_model_sha256="a" * 64,
                    timeout_seconds=10,
                    categories={"math", "natural_language_reasoning"},
                    limit_per_category=0,
                    num_ctx=2048,
                    transport=fake,
                )
            self.assertEqual(fake.calls, [])

    def test_stub_end_to_end_outputs_strict_metrics_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "general_dev_evaluation.jsonl"
            dataset.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in self._rows()
                ),
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            self._write_json(
                manifest,
                {
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
                },
            )
            samples = root / "teacher_samples.jsonl"
            summary = root / "teacher_summary.json"
            report = run_evaluation(
                dataset=dataset,
                dataset_manifest=manifest,
                samples_output=samples,
                summary_output=summary,
                endpoint="http://localhost:11434",
                expected_model_sha256="a" * 64,
                timeout_seconds=10,
                categories={"math", "natural_language_reasoning"},
                limit_per_category=0,
                num_ctx=2048,
                transport=FakeOllama(),
            )
            self.assertEqual(report["metrics"]["accuracy"], 1.0)
            self.assertEqual(report["metrics"]["valid_rate"], 1.0)
            self.assertEqual(report["selection"]["sample_count"], 2)
            self.assertEqual(report["model_identity"]["model_sha256"], "a" * 64)
            self.assertTrue(report["model_identity"]["stable_across_evaluation"])
            self.assertEqual(
                report["model_identity"]["attestation_before"]["model_sha256"],
                report["model_identity"]["attestation_after"]["model_sha256"],
            )
            self.assertEqual(report["artifacts"]["samples"]["sha256"], sha256_file(samples))
            self.assertEqual(len(report["artifacts"]["evaluator"]["sha256"]), 64)
            with self.assertRaisesRegex(ManifestError, "拒绝覆盖"):
                run_evaluation(
                    dataset=dataset,
                    dataset_manifest=manifest,
                    samples_output=samples,
                    summary_output=summary,
                    endpoint="http://localhost:11434",
                    expected_model_sha256="a" * 64,
                    timeout_seconds=10,
                    categories={"math", "natural_language_reasoning"},
                    limit_per_category=0,
                    num_ctx=2048,
                    transport=FakeOllama(),
                )


if __name__ == "__main__":
    unittest.main()
