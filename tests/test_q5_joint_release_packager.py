"""Fail-closed tests for the clean-Q5 plus one joint-LoRA packager."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from edge_llm_factory.contracts import ManifestError, sha256_file
from edge_llm_factory.q5_joint_release import build_q5_joint_release


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _identity(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


class Q5JointReleasePackagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="q5-joint-packager-test-")
        self.root = Path(self.temporary.name)
        self.output = self.root / "package"
        self.base = self.root / "base.Q5_K_M.gguf"
        self.lora = self.root / "joint.F16-LoRA.gguf"
        self.base.write_bytes(b"clean-q5-k-m")
        self.lora.write_bytes(b"one-joint-f16-lora")
        self.base_id = _identity(self.base)
        self.lora_id = _identity(self.lora)

        self.adapter = self.root / "peft"
        self.adapter.mkdir()
        (self.adapter / "adapter_model.safetensors").write_bytes(b"stub-safetensors")
        _write_json(self.adapter / "adapter_config.json", {"peft_type": "LORA"})
        _write_json(self.adapter / "train_metrics.json", {"status": "completed"})
        self.adapter_ids = {
            "weights": _identity(self.adapter / "adapter_model.safetensors"),
            "config": _identity(self.adapter / "adapter_config.json"),
            "train_metrics": _identity(self.adapter / "train_metrics.json"),
        }

        self.formal_traffic = self.root / "formal_traffic.json"
        self.formal_industrial = self.root / "formal_industrial.json"
        self.formal_gate = self.root / "formal_gate.json"
        self.q5_full = self.root / "q5_full.json"
        self.nano_observation = self.root / "nano_observation.json"
        self.nano_gate = self.root / "nano_gate.json"
        self.descriptor = self.root / "descriptor.json"
        _write_json(self.root / "base_manifest.json", {})
        _write_json(self.root / "action_mapping.json", {})
        self._write_valid_inputs()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _formal_samples(count: int, prediction: str) -> list:
        return [
            {
                "event_id": "sample-{}".format(index),
                "prediction": prediction,
                "generated_token_ids": [32],
                "prompt_tokens": 17,
                "valid": True,
            }
            for index in range(count)
        ]

    def _formal_report(self, scene: str) -> dict:
        traffic = scene == "traffic"
        count = 2400 if traffic else 960
        common = {
            "base_manifest_identity": {"bytes": 10, "sha256": "a" * 64},
            "snapshot_manifest_identity": {"bytes": 20, "sha256": "b" * 64},
            "dataset_manifest_identity": {"bytes": 30, "sha256": "c" * 64},
            "dataset_artifacts": {"train": {"sha256": "d" * 64}},
            "evaluator_identity": {"bytes": 40, "sha256": "e" * 64},
        }
        accuracy = 0.68 if traffic else 1.0
        return {
            **common,
            "evaluation_mode": "joint_formal",
            "joint_scene": scene,
            "prompt_format": "raw_task",
            "prompt_prefix": "T" if traffic else "I",
            "required_prompt_tokens": 17,
            "count": count,
            "valid_output_rate": 1.0,
            "test_set_used_for_training": False,
            "expected_adapter_weights_sha256": self.adapter_ids["weights"]["sha256"],
            "adapter_artifacts": self.adapter_ids,
            "precision": {
                "requested": "bfloat16",
                "effective": "bfloat16",
                "cuda_available": True,
            },
            "decoding_constraint": {
                "enabled": True,
                "allowed_tokens": list("ABCDEF" if traffic else "ABC"),
                "applies_before_sampling": True,
                "post_hoc_remapping": False,
            },
            "samples": self._formal_samples(count, "F" if traffic else "A"),
            "decision_accuracy": accuracy,
            "macro_f1": 0.67 if traffic else 1.0,
            "weighted_f1": 0.67 if traffic else 1.0,
        }

    @staticmethod
    def _q5_records(count: int) -> list:
        return [
            {
                "prompt_n": 17,
                "predicted_n": 1,
                "request_has_lora_field": False,
            }
            for _ in range(count)
        ]

    @staticmethod
    def _q5_scene(count: int, accuracy: float, macro_f1: float, weighted_f1: float) -> dict:
        return {
            "classification": {
                "count": count,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "valid_output_rate": 1.0,
            },
            "contract": {
                "input_chars_all_17_prefix_plus_16_digits": True,
                "prompt_tokens_all_17": True,
                "output_tokens_all_1": True,
                "grammar_applied_before_sampling": True,
                "post_hoc_remapping": False,
                "requests_have_no_lora_fields": True,
            },
            "records": Q5JointReleasePackagerTests._q5_records(count),
        }

    @staticmethod
    def _nano_records() -> list:
        return [
            {
                "scene": "traffic" if index % 2 == 0 else "industrial",
                "prediction": "F" if index % 2 == 0 else "A",
                "prompt_tokens": 17,
                "output_tokens": 1,
                "request_has_lora_field": False,
                "valid_output": True,
            }
            for index in range(500)
        ]

    def _write_valid_inputs(self) -> None:
        traffic = self._formal_report("traffic")
        industrial = self._formal_report("industrial")
        _write_json(self.formal_traffic, traffic)
        _write_json(self.formal_industrial, industrial)
        shared = {
            name: traffic[name]
            for name in (
                "base_manifest_identity",
                "snapshot_manifest_identity",
                "dataset_manifest_identity",
                "dataset_artifacts",
                "evaluator_identity",
            )
        }
        _write_json(
            self.formal_gate,
            {
                "schema_version": "edge-llm-joint-traffic-industrial-gate/v2",
                "passed": True,
                "one_adapter_two_isolated_tests": True,
                "adapter_artifacts": self.adapter_ids,
                **shared,
                "results": {
                    "traffic": {"passed": True, "checks": {"quality": True}},
                    "industrial": {"passed": True, "checks": {"quality": True}},
                },
            },
        )
        _write_json(
            self.q5_full,
            {
                "schema_version": "edge-llm-joint-q5-f16-lora-full-deployment-validation/v1",
                "runtime": {
                    "base": self.base_id,
                    "lora": self.lora_id,
                    "resident_lora_count": 1,
                    "resident_lora_default_scale": 1.0,
                    "request_level_lora_switching": False,
                    "reconnects": 0,
                },
                "traffic": self._q5_scene(2400, 0.68, 0.67, 0.67),
                "industrial": self._q5_scene(960, 1.0, 1.0, 1.0),
                "gates": {"all_quality_and_contract_checks": True},
                "all_contract_gates_passed": True,
            },
        )
        _write_json(
            self.nano_observation,
            {
                "schema_version": "edge-llm-joint-single-adapter-alternating-observation/v1",
                "status": "completed",
                "completed_requests": 500,
                "stop_reason": {"code": "completed"},
                "runtime": {"base": self.base_id, "adapter": self.lora_id},
                "benchmark_contract": {
                    "per_scene_requested": 250,
                    "total_requested": 500,
                    "required_prompt_tokens": 17,
                    "required_output_tokens": 1,
                    "one_resident_adapter": True,
                    "resident_adapter_default_scale": 1.0,
                    "request_level_lora_switching": False,
                    "requests_omit_lora_fields": True,
                },
                "summary": {
                    "overall": {
                        "valid_output_rate": 1.0,
                        "latency_ms": {"mean": 90.0},
                    },
                    "traffic": {"accuracy": 0.68, "weighted_f1": 0.67},
                    "industrial": {
                        "accuracy": 1.0,
                        "macro_f1": 1.0,
                        "weighted_f1": 1.0,
                    },
                },
                "records": self._nano_records(),
                "errors": [],
            },
        )
        _write_json(
            self.nano_gate,
            {
                "schema_version": "q5km-single-f16-nano-observational-memory-gate/v1",
                "gate_pass": True,
                "checks": {"all_fixed_checks": True},
                "completed_requests": 500,
                "traffic_count": 250,
                "industrial_count": 250,
                "strict_peak_bytes": 1_400_000_000,
                "threshold_bytes": 1_500_000_000,
                "command_contract": {
                    "resident_adapter_count": 1,
                    "resident_adapter_default_scale": 1.0,
                },
                "traffic_accuracy": 0.68,
                "traffic_weighted_f1": 0.67,
                "industrial_accuracy": 1.0,
                "industrial_macro_f1": 1.0,
                "industrial_weighted_f1": 1.0,
                "valid_output_rate": 1.0,
                "overall_mean_latency_ms": 90.0,
            },
        )
        _write_json(
            self.descriptor,
            {
                "schema_version": "edge-llm-q5-joint-release-input/v1",
                "adapter_id": "traffic-industrial-joint-q5",
                "version": "1.0.0",
                "base_manifest": "base_manifest.json",
                "primary_adapter_source": "peft",
                "action_mapping": "action_mapping.json",
                "deployment_artifact": self.base_id,
                "runtime_adapter": {
                    "id": 0,
                    "mode": "process_default",
                    "default_scale": 1,
                    **self.lora_id,
                },
                "evidence": {
                    "formal_traffic": str(self.formal_traffic),
                    "formal_industrial": str(self.formal_industrial),
                    "formal_gate": str(self.formal_gate),
                    "q5_full": str(self.q5_full),
                    "nano_observation": str(self.nano_observation),
                    "nano_gate": str(self.nano_gate),
                },
                "training": {
                    "teacher_model": "locked-teacher",
                    "methods": ["sft"],
                    "train_dataset_id": "joint-train-val-only",
                    "test_dataset_id": "traffic-2400-industrial-960",
                    "test_set_used_for_training": False,
                },
            },
        )

    def _mock_builder(self, **kwargs):
        spec = json.loads(Path(kwargs["spec_path"]).read_text(encoding="utf-8"))
        self.assertEqual(set(spec["evaluation"]["evidence"]), {"q5_joint_gate_summary"})
        self.assertEqual(spec["deployment"]["quantization"], "Q5_K_M")
        self.assertEqual(spec["input_contract"]["max_input_tokens"], 17)
        source = Path(spec["evaluation"]["evidence"]["q5_joint_gate_summary"])
        target = Path(kwargs["output_dir"]) / "evidence" / "q5_joint_gate_summary.json"
        target.parent.mkdir(parents=True)
        shutil.copy2(source, target)
        return {"status": "built", "package": str(kwargs["output_dir"])}

    def test_builds_one_process_default_scale_one_package(self) -> None:
        with mock.patch(
            "edge_llm_factory.q5_joint_release.build_adapter_package",
            side_effect=self._mock_builder,
        ) as builder:
            result = build_q5_joint_release(self.descriptor, self.output)
        self.assertEqual(builder.call_count, 1)
        self.assertEqual(len(result["runtime_adapters"]), 1)
        self.assertEqual(result["runtime_adapters"][0]["id"], 0)
        self.assertEqual(result["runtime_adapters"][0]["mode"], "process_default")
        self.assertEqual(result["runtime_adapters"][0]["default_scale"], 1)
        summary = json.loads(
            (self.output / "evidence" / "q5_joint_gate_summary.json").read_text()
        )
        self.assertEqual(
            summary["schema_version"], "edge-llm-q5-joint-release-gate-summary/v1"
        )
        def without_host_paths(value):
            if isinstance(value, dict):
                return {
                    key: without_host_paths(item)
                    for key, item in value.items()
                    if key != "path"
                }
            if isinstance(value, list):
                return [without_host_paths(item) for item in value]
            return value

        # TemporaryDirectory names are random and can legitimately contain
        # the two-character substring "q6".  The release evidence itself must
        # not contain Q6 semantics, but host paths are not semantic content.
        self.assertNotIn("q6", json.dumps(without_host_paths(summary)).lower())
        self.assertFalse(summary["runtime_contract"]["request_level_lora_switching"])
        self.assertEqual(summary["runtime_contract"]["input_tokens"], 17)
        self.assertEqual(summary["runtime_contract"]["output_tokens"], 1)

    def test_missing_nano_evidence_refuses_without_output(self) -> None:
        self.nano_observation.unlink()
        with mock.patch(
            "edge_llm_factory.q5_joint_release.build_adapter_package"
        ) as builder:
            with self.assertRaisesRegex(ManifestError, "nano_observation"):
                build_q5_joint_release(self.descriptor, self.output)
        builder.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_failed_nano_gate_refuses_without_output(self) -> None:
        report = json.loads(self.nano_gate.read_text())
        report["gate_pass"] = False
        _write_json(self.nano_gate, report)
        with mock.patch(
            "edge_llm_factory.q5_joint_release.build_adapter_package"
        ) as builder:
            with self.assertRaisesRegex(ManifestError, "nano_gate.gate_pass"):
                build_q5_joint_release(self.descriptor, self.output)
        builder.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_nano_strict_peak_over_decimal_1p5gb_refuses_without_output(self) -> None:
        report = json.loads(self.nano_gate.read_text())
        report["strict_peak_bytes"] = 1_500_000_001
        # Even a forged/stale positive gate cannot bypass the packager's own
        # decimal-byte hard limit.
        _write_json(self.nano_gate, report)
        with mock.patch(
            "edge_llm_factory.q5_joint_release.build_adapter_package"
        ) as builder:
            with self.assertRaisesRegex(ManifestError, "strict_peak_bytes"):
                build_q5_joint_release(self.descriptor, self.output)
        builder.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_request_level_switching_contract_refuses(self) -> None:
        report = json.loads(self.q5_full.read_text())
        report["runtime"]["request_level_lora_switching"] = True
        _write_json(self.q5_full, report)
        with self.assertRaisesRegex(ManifestError, "request_switching"):
            build_q5_joint_release(self.descriptor, self.output)
        self.assertFalse(self.output.exists())

    def test_mutated_q5_asset_refuses_without_output(self) -> None:
        self.base.write_bytes(b"mutated-q5")
        with self.assertRaisesRegex(ManifestError, "deployment_artifact.bytes mismatch"):
            build_q5_joint_release(self.descriptor, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
