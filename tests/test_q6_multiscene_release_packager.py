"""Fail-closed tests for the clean-Q6 dual-LoRA release packager."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from edge_llm_factory.contracts import ManifestError, sha256_file
from edge_llm_factory.q6_multiscene_release import build_q6_multiscene_release


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


class Q6MultisceneReleasePackagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="q6-packager-test-")
        self.root = Path(self.temporary.name)
        self.base = self.root / "base.Q6_K.gguf"
        self.traffic = self.root / "traffic.F32-LoRA.gguf"
        self.industrial = self.root / "industrial.F32-LoRA.gguf"
        self.base.write_bytes(b"clean-q6-base")
        self.traffic.write_bytes(b"traffic-lora")
        self.industrial.write_bytes(b"industrial-lora")
        self.base_id = _identity(self.base)
        self.traffic_id = _identity(self.traffic)
        self.industrial_id = _identity(self.industrial)
        self.quality_path = self.root / "quality.json"
        self.isolation_path = self.root / "isolation.json"
        self.stability_path = self.root / "stability.json"
        self.descriptor_path = self.root / "descriptor.json"
        self.output = self.root / "package"
        self._write_valid_inputs()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _classification(count: int, accuracy: float, weighted_f1: float) -> dict:
        return {
            "count": count,
            "accuracy": accuracy,
            "weighted_f1": weighted_f1,
            "valid_output_rate": 1.0,
        }

    @staticmethod
    def _contract() -> dict:
        return {
            "input_chars_all_16_digits": True,
            "prompt_tokens_all_16": True,
            "output_tokens_all_1": True,
            "grammar_applied_before_sampling": True,
            "post_hoc_remapping": False,
        }

    def _write_valid_inputs(self) -> None:
        _write_json(
            self.quality_path,
            {
                "schema_version": "q6-full-multilora-quality-gate/v1",
                "runtime": {
                    "base_q6": self.base_id,
                    "traffic_lora_id0": self.traffic_id,
                    "industrial_lora_id1": self.industrial_id,
                },
                "traffic": {
                    "classification": self._classification(2400, 0.69, 0.68),
                    "contract": self._contract(),
                },
                "industrial": {
                    "classification": self._classification(960, 1.0, 1.0),
                    "contract": self._contract(),
                },
            },
        )
        runtime = lambda adapter_id: {  # noqa: E731 - compact fixture
            "model": str(self.base.resolve()),
            "lora_adapter": {"id": adapter_id, "scale": 1.0},
        }
        scene = {
            "count": 100,
            "valid_output_rate": 1.0,
        }
        _write_json(
            self.isolation_path,
            {
                "schema_version": "multi-adapter-isolation-benchmark/v1",
                "traffic_runtime": runtime(0),
                "industrial_runtime": runtime(1),
                "sequential_alternating": {"traffic": scene, "industrial": scene},
                "concurrent_mixed_4": {"traffic": scene, "industrial": scene},
                "sequential_vs_concurrent_prediction_consistency": {
                    "traffic": {"consistency_rate": 1.0},
                    "industrial": {"consistency_rate": 1.0},
                },
                "cross_scene_adapter_contamination_detected": False,
            },
        )
        sample = lambda count, rss, swap, hwm, pswpin: {  # noqa: E731
            "request_count": count,
            "llama_vmrss_kib": rss,
            "llama_vmswap_kib": swap,
            "llama_vmhwm_kib": hwm,
            "pswpin_pages": pswpin,
        }
        _write_json(
            self.stability_path,
            {
                "schema_version": "multi-adapter-alternating-stability/v1",
                "status": "completed",
                "attempted_requests": 500,
                "completed_requests": 500,
                "stop_reason": {"code": "completed"},
                "benchmark_contract": {
                    "per_scene_requested": 250,
                    "total_requested": 500,
                    "traffic_lora_id": 0,
                    "industrial_lora_id": 1,
                },
                "asset_evidence": {
                    "base_q6": self.base_id,
                    "runtime_adapters": [
                        {"id": 0, **self.traffic_id},
                        {"id": 1, **self.industrial_id},
                    ],
                },
                "summary": {
                    "overall": {"completion_rate": 1.0},
                    "traffic": {
                        "completed": 250,
                        "valid_output_rate_on_completed": 1.0,
                        "accuracy_on_completed": 0.69,
                    },
                    "industrial": {
                        "completed": 250,
                        "valid_output_rate_on_completed": 1.0,
                        "accuracy_on_completed": 1.0,
                    },
                    "resources": {
                        "minimum_mem_available_mib": 300.0,
                        "peak_llama_rss_mib": 700.0,
                        "peak_llama_vmswap_mib": 2.0,
                    },
                },
                "resource_samples": [
                    sample(100, 700000, 1000, 800000, 10),
                    sample(500, 710000, 2000, 800000, 20),
                ],
                "errors": [],
            },
        )
        # The generic builder is mocked in these tests, so these need only be
        # explicit paths; Q6 evidence validation happens before it is invoked.
        _write_json(self.root / "base_manifest.json", {})
        (self.root / "adapter").mkdir()
        _write_json(self.root / "action_mapping.json", {})
        _write_json(
            self.descriptor_path,
            {
                "schema_version": "edge-llm-q6-multiscene-release-input/v1",
                "adapter_id": "traffic-industrial-clean-q6",
                "version": "1.0.0",
                "base_manifest": "base_manifest.json",
                "primary_adapter_source": "adapter",
                "action_mapping": "action_mapping.json",
                "deployment_artifact": self.base_id,
                "runtime_adapters": [
                    {
                        "id": 0,
                        "scene": "freeway_traffic_management",
                        "default_scale": 0,
                        **self.traffic_id,
                    },
                    {
                        "id": 1,
                        "scene": "industrial_anomaly",
                        "default_scale": 0,
                        **self.industrial_id,
                    },
                ],
                "evidence": {
                    "full_quality": str(self.quality_path),
                    "isolation": str(self.isolation_path),
                    "nano_stability": str(self.stability_path),
                },
                "training": {
                    "teacher_model": "frozen-test-teacher",
                    "methods": ["sft"],
                    "train_dataset_id": "frozen-train",
                    "test_dataset_id": "traffic-2400-industrial-960",
                    "test_set_used_for_training": False,
                },
            },
        )

    def _mock_builder(self, **kwargs):
        spec = json.loads(Path(kwargs["spec_path"]).read_text(encoding="utf-8"))
        self.assertEqual(
            set(spec["evaluation"]["evidence"]), {"q6_gate_summary"}
        )
        source = Path(spec["evaluation"]["evidence"]["q6_gate_summary"])
        target = Path(kwargs["output_dir"]) / "evidence" / "q6_gate_summary.json"
        target.parent.mkdir(parents=True)
        shutil.copy2(source, target)
        return {"status": "built", "package": str(kwargs["output_dir"])}

    def test_builds_only_after_all_three_completed_evidence_sets(self) -> None:
        with mock.patch(
            "edge_llm_factory.q6_multiscene_release.build_adapter_package",
            side_effect=self._mock_builder,
        ) as builder:
            result = build_q6_multiscene_release(self.descriptor_path, self.output)
        self.assertEqual(result["status"], "built")
        self.assertEqual([row["id"] for row in result["runtime_adapters"]], [0, 1])
        self.assertEqual(builder.call_count, 1)
        summary = json.loads(
            (self.output / "evidence" / "q6_gate_summary.json").read_text()
        )
        self.assertTrue(summary["release_authorized_by_packager"])
        self.assertEqual(summary["metrics"]["nano_completed_requests"], 500.0)
        self.assertEqual(
            summary["metrics"]["nano_peak_llama_resident_plus_swap_bytes"],
            (700 + 2) * 1024 * 1024,
        )
        self.assertEqual(
            summary["metrics"]["nano_peak_llama_vmhwm_bytes"],
            800000 * 1024,
        )
        memory_gate = next(
            row
            for row in summary["fixed_gates"]
            if row["metric"] == "nano_peak_llama_resident_plus_swap_bytes"
        )
        self.assertEqual(memory_gate["value"], 1_500_000_000)

    def test_missing_nano_evidence_refuses_without_output(self) -> None:
        self.stability_path.unlink()
        with self.assertRaisesRegex(ManifestError, "无法读取 JSON"):
            build_q6_multiscene_release(self.descriptor_path, self.output)
        self.assertFalse(self.output.exists())

    def test_stopped_nano_gate_refuses_without_output(self) -> None:
        report = json.loads(self.stability_path.read_text())
        report["status"] = "stopped"
        report["completed_requests"] = 80
        _write_json(self.stability_path, report)
        with self.assertRaisesRegex(ManifestError, "stability.status mismatch"):
            build_q6_multiscene_release(self.descriptor_path, self.output)
        self.assertFalse(self.output.exists())

    def test_nano_report_without_exact_asset_hashes_refuses(self) -> None:
        report = json.loads(self.stability_path.read_text())
        report.pop("asset_evidence")
        _write_json(self.stability_path, report)
        with self.assertRaisesRegex(ManifestError, "lacks asset_evidence"):
            build_q6_multiscene_release(self.descriptor_path, self.output)

    def test_q8_or_mutated_base_cannot_be_substituted(self) -> None:
        self.base.write_bytes(b"different-artifact")
        with self.assertRaisesRegex(ManifestError, "deployment_artifact.bytes mismatch"):
            build_q6_multiscene_release(self.descriptor_path, self.output)
        self.assertFalse(self.output.exists())

    def test_post_100_request_growth_gate_is_fail_closed(self) -> None:
        report = json.loads(self.stability_path.read_text())
        report["resource_samples"][1]["llama_vmrss_kib"] += 40 * 1024
        _write_json(self.stability_path, report)
        with self.assertRaisesRegex(ManifestError, "rss_growth_100_to_500_mib"):
            build_q6_multiscene_release(self.descriptor_path, self.output)

    def test_decimal_1p5gb_total_footprint_gate_refuses_over_limit(self) -> None:
        report = json.loads(self.stability_path.read_text())
        report["summary"]["resources"]["peak_llama_rss_mib"] = 1400.0
        report["summary"]["resources"]["peak_llama_vmswap_mib"] = 50.0
        _write_json(self.stability_path, report)
        with self.assertRaisesRegex(
            ManifestError, "peak_llama_resident_plus_swap_bytes"
        ):
            build_q6_multiscene_release(self.descriptor_path, self.output)
        self.assertFalse(self.output.exists())

    def test_raw_kib_peak_fields_are_accepted_when_summary_peaks_are_absent(self) -> None:
        report = json.loads(self.stability_path.read_text())
        report["summary"]["resources"].pop("peak_llama_rss_mib")
        report["summary"]["resources"].pop("peak_llama_vmswap_mib")
        _write_json(self.stability_path, report)
        with mock.patch(
            "edge_llm_factory.q6_multiscene_release.build_adapter_package",
            side_effect=self._mock_builder,
        ):
            result = build_q6_multiscene_release(self.descriptor_path, self.output)
        self.assertEqual(result["status"], "built")

    def test_missing_peak_memory_fields_refuses_without_output(self) -> None:
        report = json.loads(self.stability_path.read_text())
        report["summary"]["resources"].pop("peak_llama_rss_mib")
        report["summary"]["resources"].pop("peak_llama_vmswap_mib")
        for row in report["resource_samples"]:
            row.pop("llama_vmrss_kib")
            row.pop("llama_vmswap_kib")
        _write_json(self.stability_path, report)
        with self.assertRaisesRegex(ManifestError, "missing peak_llama_rss_mib"):
            build_q6_multiscene_release(self.descriptor_path, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_peak_swap_fields_refuses_without_output(self) -> None:
        report = json.loads(self.stability_path.read_text())
        report["summary"]["resources"].pop("peak_llama_vmswap_mib")
        for row in report["resource_samples"]:
            row.pop("llama_vmswap_kib")
        _write_json(self.stability_path, report)
        with self.assertRaisesRegex(ManifestError, "missing peak_llama_vmswap_mib"):
            build_q6_multiscene_release(self.descriptor_path, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
