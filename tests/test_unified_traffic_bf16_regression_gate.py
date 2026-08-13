from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAFFIC_ROOT = REPOSITORY_ROOT / "scenes" / "freeway_traffic"
if str(TRAFFIC_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAFFIC_ROOT))

from traffic_system.summarize_unified_traffic_bf16_regression import (  # noqa: E402
    RegressionGateError,
    _classification_metrics,
    build_report,
    main,
    paired_bootstrap_accuracy_delta,
    sha256_file,
)


class UnifiedTrafficBf16RegressionGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.project = self.root / "traffic"
        self.project.mkdir()
        self.base_dir = self.project / "base_model"
        self.base_dir.mkdir()
        self.base_config = self.base_dir / "config.json"
        self.base_config.write_text('{"model_type":"stub"}', encoding="utf-8")
        self.base_manifest = self.base_dir / "text_snapshot_manifest.json"
        self.base_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "test/v1",
                    "files": [
                        {
                            "path": "config.json",
                            "bytes": self.base_config.stat().st_size,
                            "sha256": sha256_file(self.base_config),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.evaluator = (
            TRAFFIC_ROOT / "traffic_system" / "eval_llm_sft_student.py"
        )
        self.test_jsonl = self.project / "datasets" / "test.jsonl"
        self.test_jsonl.parent.mkdir()
        self.targets = {
            f"event-{index:04d}": "ABCDEF"[index % 6] for index in range(2400)
        }
        self.test_jsonl.write_text(
            "".join(
                json.dumps({"event_id": event_id, "target": target}) + "\n"
                for event_id, target in self.targets.items()
            ),
            encoding="utf-8",
        )
        self.test_sha = sha256_file(self.test_jsonl)

        self.incumbent_dir = self.project / "incumbent"
        self.candidate_dir = self.project / "candidate"
        self.incumbent_dir.mkdir()
        self.candidate_dir.mkdir()
        self.incumbent_adapter = self.incumbent_dir / "adapter_model.safetensors"
        self.candidate_adapter = self.candidate_dir / "adapter_model.safetensors"
        self.incumbent_adapter.write_bytes(b"incumbent-adapter")
        self.candidate_adapter.write_bytes(b"candidate-adapter")
        self.incumbent_config = self.incumbent_dir / "adapter_config.json"
        self.candidate_config = self.candidate_dir / "adapter_config.json"
        self.incumbent_config.write_text('{"r":8}', encoding="utf-8")
        self.candidate_config.write_text('{"r":16}', encoding="utf-8")

        self.incumbent_json = self.project / "incumbent.json"
        self.candidate_json = self.project / "candidate.json"
        self._write_evaluation(
            self.incumbent_json,
            self.incumbent_dir,
            wrong_indexes={index for index in range(2400) if index % 10 == 0},
        )
        self._write_evaluation(self.candidate_json, self.candidate_dir)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _examples(self, wrong_indexes=None):
        wrong_indexes = set(wrong_indexes or [])
        examples = []
        for index, (event_id, target) in enumerate(self.targets.items()):
            parsed = (
                "ABCDEF"[("ABCDEF".index(target) + 1) % 6]
                if index in wrong_indexes
                else target
            )
            examples.append(
                {
                    "event_id": event_id,
                    "latency_ms": 1.0,
                    "prompt_tokens": 16,
                    "raw_output": parsed,
                    "parsed": parsed,
                    "target": target,
                    "json_valid": True,
                    "decision_match": parsed == target,
                }
            )
        return examples

    def _write_evaluation(self, path: Path, adapter_dir: Path, wrong_indexes=None):
        examples = self._examples(wrong_indexes)
        normalized = [
            {
                "event_id": row["event_id"],
                "target": row["target"],
                "parsed": row["parsed"],
                "valid": row["json_valid"],
                "correct": row["decision_match"],
                "latency_ms": row["latency_ms"],
            }
            for row in examples
        ]
        metrics = _classification_metrics(normalized)
        def identity(file_path: Path):
            return {
                "path": str(file_path.resolve()),
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }

        report = {
            "model_name_or_path": str(self.base_dir.relative_to(self.project)),
            "adapter_dir": str(adapter_dir.relative_to(self.project)),
            "test_jsonl": str(self.test_jsonl.relative_to(self.project)),
            "count": 2400,
            "source_count": 2400,
            "class_scores_included": True,
            "warmup_runs": 5,
            "prompt_format": "raw_task",
            "bf16": True,
            "max_seq_length": 16,
            "max_new_tokens": 1,
            "temperature": 0.0,
            "evaluation_artifacts": {
                "adapter_model": identity(adapter_dir / "adapter_model.safetensors"),
                "adapter_config": identity(adapter_dir / "adapter_config.json"),
                "base_text_snapshot_manifest": identity(self.base_manifest),
                "base_text_snapshot_files": {
                    "config.json": identity(self.base_config),
                },
                "evaluator": identity(self.evaluator),
                "verified_unchanged_during_evaluation": True,
            },
            "json_valid_rate": round(metrics["valid_output_rate"], 4),
            "decision_accuracy": round(metrics["accuracy"], 4),
            "token_classification": {
                "weighted_f1": round(metrics["weighted_f1"], 4)
            },
            "examples": examples,
        }
        path.write_text(json.dumps(report), encoding="utf-8")

    def _build(self, **overrides):
        arguments = {
            "incumbent_evaluation": self.incumbent_json,
            "candidate_evaluation": self.candidate_json,
            "test_jsonl": self.test_jsonl,
            "incumbent_adapter_model": self.incumbent_adapter,
            "expected_incumbent_adapter_sha256": sha256_file(
                self.incumbent_adapter
            ),
            "candidate_adapter_model": self.candidate_adapter,
            "expected_candidate_adapter_sha256": sha256_file(
                self.candidate_adapter
            ),
            "project_root": self.project,
            "expected_test_sha256": self.test_sha,
            "expected_evaluator_sha256": sha256_file(self.evaluator),
            "expected_gate_sha256": sha256_file(
                TRAFFIC_ROOT
                / "traffic_system"
                / "summarize_unified_traffic_bf16_regression.py"
            ),
            "bootstrap_replicates": 100,
        }
        arguments.update(overrides)
        return build_report(**arguments)

    def test_builds_paired_report_and_passes_hard_gate(self):
        report = self._build()
        gate_script = (
            TRAFFIC_ROOT
            / "traffic_system"
            / "summarize_unified_traffic_bf16_regression.py"
        ).resolve()
        self.assertEqual(report["gate_script_identity"]["path"], str(gate_script))
        self.assertEqual(
            report["gate_script_identity"]["bytes"], gate_script.stat().st_size
        )
        self.assertEqual(
            report["gate_script_identity"]["sha256"], sha256_file(gate_script)
        )
        self.assertEqual(report["frozen_test"]["sample_count"], 2400)
        self.assertEqual(report["frozen_test"]["unique_event_ids"], 2400)
        self.assertEqual(
            report["incumbent"]["adapter_model"]["sha256"],
            sha256_file(self.incumbent_adapter),
        )
        self.assertEqual(
            report["candidate"]["adapter_model"]["sha256"],
            sha256_file(self.candidate_adapter),
        )
        paired = report["paired_comparison"]
        self.assertEqual(paired["candidate_correct_incumbent_wrong"], 240)
        self.assertEqual(paired["candidate_wrong_incumbent_correct"], 0)
        self.assertEqual(paired["net_corrected"], 240)
        self.assertEqual(paired["candidate_minus_incumbent_accuracy"], 0.1)
        self.assertEqual(paired["bootstrap"]["seed"], 20260810)
        self.assertEqual(paired["bootstrap"]["replicates"], 100)
        metric_checks = report["hard_gate"]["checks"]
        self.assertTrue(
            metric_checks["candidate_accuracy_absolute_minimum"]["passed"]
        )
        self.assertTrue(
            metric_checks["candidate_accuracy_not_below_incumbent"]["passed"]
        )
        self.assertTrue(
            metric_checks["candidate_weighted_f1_absolute_minimum"]["passed"]
        )
        self.assertTrue(
            metric_checks["candidate_weighted_f1_not_below_incumbent"]["passed"]
        )
        self.assertEqual(
            report["hard_gate"]["metric_conditions"],
            {
                "candidate_accuracy_absolute_minimum": True,
                "candidate_accuracy_not_below_incumbent": True,
                "candidate_weighted_f1_absolute_minimum": True,
                "candidate_weighted_f1_not_below_incumbent": True,
            },
        )
        self.assertTrue(report["hard_gate"]["metric_conditions_all_passed"])
        self.assertTrue(report["hard_gate"]["valid_output_contract_passed"])
        self.assertTrue(report["hard_gate"]["all_passed"])

    def test_hard_gate_fails_below_candidate_thresholds(self):
        self._write_evaluation(
            self.candidate_json,
            self.candidate_dir,
            wrong_indexes={index for index in range(2400) if index % 2 == 0},
        )
        report = self._build()
        self.assertFalse(report["hard_gate"]["all_passed"])
        self.assertFalse(
            report["hard_gate"]["checks"][
                "candidate_accuracy_absolute_minimum"
            ]["passed"]
        )

    def test_hard_gate_rejects_candidate_above_absolute_floors_but_below_incumbent(self):
        self._write_evaluation(
            self.candidate_json,
            self.candidate_dir,
            wrong_indexes={index for index in range(2400) if index % 4 == 0},
        )
        report = self._build()
        checks = report["hard_gate"]["checks"]

        self.assertGreater(report["candidate"]["metrics"]["accuracy"], 0.66125)
        self.assertGreater(
            report["candidate"]["metrics"]["weighted_f1"], 0.666848
        )
        self.assertTrue(checks["candidate_accuracy_absolute_minimum"]["passed"])
        self.assertTrue(
            checks["candidate_weighted_f1_absolute_minimum"]["passed"]
        )
        self.assertFalse(
            checks["candidate_accuracy_not_below_incumbent"]["passed"]
        )
        self.assertFalse(
            checks["candidate_weighted_f1_not_below_incumbent"]["passed"]
        )
        self.assertEqual(
            report["hard_gate"]["metric_conditions"],
            {
                "candidate_accuracy_absolute_minimum": True,
                "candidate_accuracy_not_below_incumbent": False,
                "candidate_weighted_f1_absolute_minimum": True,
                "candidate_weighted_f1_not_below_incumbent": False,
            },
        )
        self.assertFalse(report["hard_gate"]["metric_conditions_all_passed"])
        self.assertTrue(checks["candidate_valid_output_rate"]["passed"])
        self.assertFalse(report["hard_gate"]["all_passed"])

    def test_rejects_event_target_misalignment(self):
        value = json.loads(self.candidate_json.read_text(encoding="utf-8"))
        row = value["examples"][0]
        row["target"] = "B" if row["target"] != "B" else "C"
        row["decision_match"] = row["parsed"] == row["target"]
        self.candidate_json.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(RegressionGateError, "目标与冻结测试集不同"):
            self._build()

    def test_rejects_wrong_bf16_inference_protocol(self):
        cases = {
            "bf16": (False, "bf16=true"),
            "max_seq_length": (32, "max_seq_length必须为16"),
            "max_new_tokens": (2, "max_new_tokens必须为1"),
            "temperature": (0.1, "temperature必须为0.0"),
        }
        original = self.candidate_json.read_text(encoding="utf-8")
        for field, (value, message) in cases.items():
            with self.subTest(field=field):
                report = json.loads(original)
                report[field] = value
                self.candidate_json.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(RegressionGateError, message):
                    self._build()
        self.candidate_json.write_text(original, encoding="utf-8")

    def test_rejects_raw_output_tampering(self):
        report = json.loads(self.candidate_json.read_text(encoding="utf-8"))
        report["examples"][0]["raw_output"] = "B"
        self.candidate_json.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(RegressionGateError, "严格单token重解析"):
            self._build()

    def test_rejects_reported_artifact_identity_tampering(self):
        original = self.candidate_json.read_text(encoding="utf-8")
        for field in (
            "adapter_model",
            "adapter_config",
            "base_text_snapshot_manifest",
            "evaluator",
        ):
            with self.subTest(field=field):
                report = json.loads(original)
                report["evaluation_artifacts"][field]["sha256"] = "0" * 64
                self.candidate_json.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(RegressionGateError, "报告身份与当前文件不一致"):
                    self._build()
        self.candidate_json.write_text(original, encoding="utf-8")

    def test_rejects_adapter_replaced_after_evaluation(self):
        self.candidate_adapter.write_bytes(b"candidate-adapter-replaced")
        with self.assertRaisesRegex(RegressionGateError, "报告身份与当前文件不一致"):
            self._build(
                expected_candidate_adapter_sha256=sha256_file(
                    self.candidate_adapter
                )
            )

    def test_rejects_duplicate_event_id(self):
        value = json.loads(self.candidate_json.read_text(encoding="utf-8"))
        value["examples"][1]["event_id"] = value["examples"][0]["event_id"]
        self.candidate_json.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(RegressionGateError, "event_id重复"):
            self._build()

    def test_rejects_test_or_adapter_hash_drift(self):
        with self.subTest("test"):
            with self.assertRaisesRegex(RegressionGateError, "测试集SHA-256不匹配"):
                self._build(expected_test_sha256="0" * 64)
        with self.subTest("adapter"):
            with self.assertRaisesRegex(RegressionGateError, "U1 LoRA权重SHA-256"):
                self._build(expected_candidate_adapter_sha256="1" * 64)

    def test_rejects_symlinked_or_misnamed_adapter(self):
        with self.subTest("symlink"):
            symlink = self.candidate_dir / "candidate-link.safetensors"
            symlink.symlink_to(self.candidate_adapter)
            with self.assertRaisesRegex(RegressionGateError, "非符号链接普通文件"):
                self._build(candidate_adapter_model=symlink)
        with self.subTest("filename"):
            renamed = self.candidate_dir / "candidate.safetensors"
            renamed.write_bytes(self.candidate_adapter.read_bytes())
            with self.assertRaisesRegex(RegressionGateError, "文件名必须是"):
                self._build(candidate_adapter_model=renamed)

    def test_bootstrap_is_seeded_and_deterministic(self):
        deltas = [1, 0, -1, 1] * 25
        first = paired_bootstrap_accuracy_delta(deltas, seed=7, replicates=50)
        second = paired_bootstrap_accuracy_delta(deltas, seed=7, replicates=50)
        self.assertEqual(first, second)

    def test_cli_refuses_to_overwrite_output(self):
        output = self.project / "already-exists.json"
        output.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(RegressionGateError, "拒绝覆盖"):
            main(
                [
                    "--incumbent-evaluation",
                    str(self.incumbent_json),
                    "--candidate-evaluation",
                    str(self.candidate_json),
                    "--test-jsonl",
                    str(self.test_jsonl),
                    "--incumbent-adapter-model",
                    str(self.incumbent_adapter),
                    "--expected-incumbent-adapter-sha256",
                    hashlib.sha256(b"incumbent-adapter").hexdigest(),
                    "--candidate-adapter-model",
                    str(self.candidate_adapter),
                    "--expected-candidate-adapter-sha256",
                    hashlib.sha256(b"candidate-adapter").hexdigest(),
                    "--project-root",
                    str(self.project),
                    "--output",
                    str(output),
                ]
            )


if __name__ == "__main__":
    unittest.main()
