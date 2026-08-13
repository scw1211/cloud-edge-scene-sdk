import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRAFFIC_ROOT = REPOSITORY_ROOT / "scenes" / "freeway_traffic"
for import_root in (REPOSITORY_ROOT, TRAFFIC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from traffic_system import measure_current_edge_llm_targets as target_measurement
from traffic_system import measure_current_edge_llm_memory as memory_measurement


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sample(sample_id: str, category: str, correct: bool, ttft_ms: float):
    return {
        "sample_id": sample_id,
        "benchmark": "fixture",
        "category": category,
        "correct": correct,
        "prediction": "x",
        "reference": "x",
        "execution_error": None,
        "raw_output": "x",
        "ttft_ms": ttft_ms,
        "wall_time_ms": ttft_ms + 1.0,
    }


def _raw():
    sample_ids = ["m1", "c1", "n1"]
    teacher = [
        _sample("m1", "math", True, 400.0),
        _sample("c1", "code", True, 420.0),
        _sample("n1", "natural_language_reasoning", True, 380.0),
    ]
    edge = [
        _sample("m1", "math", True, 80.0),
        _sample("c1", "code", False, 84.0),
        _sample("n1", "natural_language_reasoning", True, 76.0),
    ]
    return {
        "git_commit": "a" * 40,
        "dataset_id": "general-v1@sha256:" + "b" * 64,
        "dataset_sample_ids": sample_ids,
        "hardware_id": "server-5070ti-runtime-v1",
        "run_id": "run-1",
        "generated_at": "2026-08-09T12:00:00+08:00",
        "teacher_model_id": "ollama:qwen3.5:9b@sha256:" + "c" * 64,
        "edge_model_id": "gguf:current-v2.gguf@sha256:" + "d" * 64,
        "teacher_samples": teacher,
        "edge_samples": edge,
    }


def _memory_raw():
    return {
        "git_commit": "a" * 40,
        "dataset_id": "general-v1@sha256:" + "b" * 64,
        "hardware_id": {
            "role": "edge_device",
            "platform": "nvidia_jetson",
            "machine": "aarch64",
            "device_model": "NVIDIA Jetson Orin Nano Developer Kit",
            "hostname": "edge-138",
        },
        "run_id": "memory-run-1",
        "generated_at": "2026-08-09T12:30:00+08:00",
        "edge_model_id": "gguf:current-v2.gguf@sha256:" + "d" * 64,
        "sampler_interval_ms": 10.0,
        "warmup_requests": 2,
        "memory_samples": [
            {
                "sample_id": "m1",
                "category": "math",
                "peak_process_tree_rss_mb": 1012.5,
                "peak_process_tree_pss_mb": 900.0,
            },
            {
                "sample_id": "c1",
                "category": "code",
                "peak_process_tree_rss_mb": 1024.0,
                "peak_process_tree_pss_mb": 910.0,
            },
        ],
    }


class CurrentEdgeLlmTargetMeasurementTest(unittest.TestCase):
    def test_gate_evidence_uses_paired_current_run_samples(self):
        evidence = target_measurement.build_gate_evidence(_raw())

        capability = evidence["capability_retention"]
        self.assertEqual(capability["teacher_macro_score"], 1.0)
        self.assertAlmostEqual(capability["edge_macro_score"], 2.0 / 3.0, places=6)
        self.assertEqual(capability["sample_count"], 3)

        ttft = evidence["ttft_reduction"]
        self.assertEqual(ttft["paired_sample_ids"], ["m1", "c1", "n1"])
        self.assertEqual(ttft["baseline_ttft_ms"], [400.0, 420.0, 380.0])
        self.assertEqual(ttft["edge_ttft_ms"], [80.0, 84.0, 76.0])

        self.assertEqual(set(evidence), {"capability_retention", "ttft_reduction"})
        self.assertEqual(
            capability["category_sample_counts"],
            {"code": 1, "math": 1, "natural_language_reasoning": 1},
        )

    def test_incomplete_or_reordered_samples_are_rejected(self):
        raw = _raw()
        raw["edge_samples"] = list(reversed(raw["edge_samples"]))
        with self.assertRaisesRegex(
            target_measurement.MeasurementContractError, "incomplete, duplicated, or reordered"
        ):
            target_measurement.build_gate_evidence(raw)

    def test_nonpositive_ttft_is_rejected(self):
        raw = _raw()
        raw["teacher_samples"][0]["ttft_ms"] = 0.0
        with self.assertRaisesRegex(target_measurement.MeasurementContractError, "positive"):
            target_measurement.build_gate_evidence(raw)

    def test_server_run_never_emits_legacy_memory_value(self):
        raw = _raw()
        raw["edge_inference_peak_rss_mb"] = 123.0
        evidence = target_measurement.build_gate_evidence(raw)
        self.assertNotIn("single_inference_memory", evidence)

    def test_raw_samples_require_all_three_categories(self):
        raw = _raw()
        raw["teacher_samples"] = raw["teacher_samples"][:2]
        raw["edge_samples"] = raw["edge_samples"][:2]
        raw["dataset_sample_ids"] = raw["dataset_sample_ids"][:2]
        with self.assertRaisesRegex(
            target_measurement.MeasurementContractError,
            "math, code, and natural_language_reasoning",
        ):
            target_measurement.build_gate_evidence(raw)

    def test_raw_sample_categories_must_match_pairwise(self):
        raw = _raw()
        raw["edge_samples"][0]["category"] = "code"
        with self.assertRaisesRegex(
            target_measurement.MeasurementContractError, "category order"
        ):
            target_measurement.build_gate_evidence(raw)

    def test_dataset_rejects_missing_category_even_when_metadata_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "eval.jsonl"
            metadata = root / "metadata.json"
            rows = [
                {"sample_id": "m1", "category": "math"},
                {"sample_id": "c1", "category": "code"},
            ]
            dataset.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            metadata.write_text(
                json.dumps(
                    {"total_samples": 2, "category_counts": {"math": 1, "code": 1}}
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                target_measurement.MeasurementContractError, "exactly math"
            ):
                target_measurement._validate_dataset(dataset, metadata)

    def test_jetson_memory_evidence_is_independent(self):
        evidence = memory_measurement.build_memory_evidence(_memory_raw())
        self.assertEqual(evidence["peak_memory_mb"], 1024.0)
        self.assertEqual(evidence["measurement_scope"], "jetson_edge_single_inference")
        self.assertEqual(
            evidence["measurement_window"]["measured_requests_per_window"], 1
        )
        self.assertEqual(evidence["provenance"]["hardware_id"]["role"], "edge_device")

    def test_memory_evidence_rejects_server_identity(self):
        raw = _memory_raw()
        raw["hardware_id"] = {
            "role": "server",
            "platform": "linux_workstation",
            "machine": "x86_64",
            "device_model": "5070 Ti",
            "hostname": "server",
        }
        with self.assertRaisesRegex(
            target_measurement.MeasurementContractError, "Jetson edge host"
        ):
            memory_measurement.build_memory_evidence(raw)

    def test_detect_jetson_hardware_uses_kernel_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model_path.write_bytes(b"NVIDIA Jetson Orin Nano Developer Kit\x00")
            hardware = memory_measurement.detect_jetson_hardware(
                model_path=model_path, machine="aarch64", hostname="edge-test"
            )
            self.assertEqual(hardware["platform"], "nvidia_jetson")
            with self.assertRaisesRegex(
                target_measurement.MeasurementContractError, "aarch64"
            ):
                memory_measurement.detect_jetson_hardware(
                    model_path=model_path, machine="x86_64", hostname="server"
                )

    def test_detect_jetson_hardware_accepts_real_orin_model_string(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model_path.write_bytes(b"NVIDIA Orin Nano Developer Kit\x00")
            hardware = memory_measurement.detect_jetson_hardware(
                model_path=model_path, machine="aarch64", hostname="jetson02-desktop"
            )
            self.assertEqual(hardware["platform"], "nvidia_jetson")
            self.assertEqual(hardware["device_model"], "NVIDIA Orin Nano Developer Kit")

    def test_asset_validation_rejects_old_or_mismatched_model(self):
        edge_bytes = b"current edge"
        manifest_bytes = b"teacher manifest"
        blob_bytes = b"teacher blob"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            edge = root / "current-v2.gguf"
            manifest = root / "manifest"
            blob = root / "blob"
            edge.write_bytes(edge_bytes)
            manifest.write_bytes(manifest_bytes)
            blob.write_bytes(blob_bytes)
            contracts = (
                {"file": "assets/downloads/current-v2.gguf", "sha256": _digest(edge_bytes)},
                {
                    "ollama_manifest_sha256": _digest(manifest_bytes),
                    "model_blob_sha256": _digest(blob_bytes),
                },
            )
            with mock.patch.object(target_measurement, "_catalog_contracts", return_value=contracts):
                observed = target_measurement.validate_model_assets(edge, manifest, blob)
                self.assertEqual(observed["edge_asset_sha256"], _digest(edge_bytes))

                old = root / "old-v9.gguf"
                old.write_bytes(edge_bytes)
                with self.assertRaisesRegex(
                    target_measurement.MeasurementContractError, "current catalog asset"
                ):
                    target_measurement.validate_model_assets(old, manifest, blob)

                edge.write_bytes(b"tampered")
                with self.assertRaisesRegex(
                    target_measurement.MeasurementContractError, "model identity mismatch"
                ):
                    target_measurement.validate_model_assets(edge, manifest, blob)

    def test_manifest_fragment_has_gate_compatible_provenance_and_hashes(self):
        evidence = target_measurement.build_gate_evidence(_raw())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for target, value in evidence.items():
                path = root / (target + ".json")
                path.write_text(json.dumps(value), encoding="utf-8")
                paths[target] = path

            fragment = target_measurement._manifest_fragment(paths, evidence)
            self.assertEqual(
                set(fragment["targets"]), {"capability_retention", "ttft_reduction"}
            )
            for target, item in fragment["targets"].items():
                self.assertEqual(item["evidence"]["path"], target + ".json")
                self.assertEqual(len(item["evidence"]["sha256"]), 64)
                self.assertNotIn("run_id", item["expected"])
                self.assertNotIn("generated_at", item["expected"])
                self.assertEqual(
                    item["expected"]["metric_semantics"],
                    target_measurement.METRIC_SEMANTICS[target],
                )


if __name__ == "__main__":
    unittest.main()
