"""Focused fail-closed tests for the statically fused joint-Q5 packager."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from edge_llm_factory.contracts import ManifestError, sha256_file
from edge_llm_factory.q5_static_joint_release import (
    MODEL_MODE,
    NANO_SHA_FILES,
    PACKAGE_KIND,
    SCHEMA,
    SUMMARY_SCHEMA,
    _classification,
    _memory_summary,
    _validate_sha_manifest,
    _validate_sidecar_command,
    build_q5_static_joint_release,
    validate_q5_static_joint_package,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class Q5StaticJointReleasePackagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="q5-static-joint-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _nano_sha_bundle(self) -> Path:
        bundle = self.root / "nano"
        bundle.mkdir()
        for name in sorted(NANO_SHA_FILES):
            (bundle / name).write_text("fixture:{}\n".format(name), encoding="utf-8")
        manifest = bundle / "SHA256SUMS.txt"
        manifest.write_text(
            "".join(
                "{}  {}\n".format(sha256_file(bundle / name), name)
                for name in sorted(NANO_SHA_FILES)
            ),
            encoding="utf-8",
        )
        return manifest

    def test_nano_checksum_manifest_requires_and_verifies_all_24_files(self) -> None:
        manifest = self._nano_sha_bundle()
        records = _validate_sha_manifest(manifest)
        self.assertEqual(set(records), NANO_SHA_FILES)
        self.assertEqual(len(records), 24)

    def test_nano_checksum_manifest_fails_after_evidence_mutation(self) -> None:
        manifest = self._nano_sha_bundle()
        (manifest.parent / "observed_gate_summary.json").write_text("mutated\n")
        with self.assertRaisesRegex(ManifestError, "observed_gate_summary"):
            _validate_sha_manifest(manifest)

    @staticmethod
    def _observation() -> dict:
        return {
            "runtime": {
                "static_fused_model": {"path": "/deploy/joint.Q5_K_M.gguf"}
            }
        }

    def test_sidecar_command_accepts_static_model_and_no_lora(self) -> None:
        command = self.root / "cmdline.txt"
        command.write_text(
            "llama-server --model /deploy/joint.Q5_K_M.gguf "
            "--ctx-size 128 --batch-size 16 --ubatch-size 16 "
            "--parallel 1 --gpu-layers 99\n",
            encoding="utf-8",
        )
        _validate_sidecar_command(command, self._observation())

    def test_sidecar_command_rejects_any_runtime_lora_option(self) -> None:
        command = self.root / "cmdline.txt"
        command.write_text(
            "llama-server --model /deploy/joint.Q5_K_M.gguf "
            "--ctx-size 128 --batch-size 16 --ubatch-size 16 "
            "--parallel 1 --gpu-layers 99 --lora /deploy/forbidden.gguf\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ManifestError, "must not contain --lora"):
            _validate_sidecar_command(command, self._observation())

    def test_raw_memory_peak_includes_vmhwm_and_swap_without_subtraction(self) -> None:
        rows = [
            {
                "phase": "inference",
                "vmrss_kib": 100,
                "vmhwm_kib": 120,
                "vmswap_kib": 30,
                "mem_available_kib": 500,
                "oom_kill": 2,
                "pswpin": 10,
                "consecutive_d_samples": 0,
            },
            {
                "phase": "inference",
                "vmrss_kib": 110,
                "vmhwm_kib": 140,
                "vmswap_kib": 40,
                "mem_available_kib": 450,
                "oom_kill": 2,
                "pswpin": 13,
                "consecutive_d_samples": 1,
            },
        ]
        summary = _memory_summary(rows, "inference")
        self.assertEqual(summary["peak"], 150 * 1024)
        self.assertEqual(summary["minimum_available"], 450 * 1024)
        self.assertEqual(summary["oom_delta"], 0)
        self.assertEqual(summary["pswpin_delta"], 3)

    def test_metrics_are_recomputed_from_records(self) -> None:
        records = [
            {"target": "A", "prediction": "A", "valid_output": True},
            {"target": "B", "prediction": "A", "valid_output": True},
            {"target": "B", "prediction": "B", "valid_output": True},
        ]
        result = _classification(records, "AB")
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["valid_output_rate"], 1.0)

    def _package_fixture(self) -> tuple:
        package = self.root / "package"
        evidence = package / "evidence" / "q5_static_joint_gate_summary.json"
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "release_authorized_by_packager": True,
            "assets": {
                "static_q5_k_m": {
                    "name": "joint.Q5_K_M.gguf",
                    "bytes": 577990656,
                    "sha256": "3" * 64,
                }
            },
            "runtime_contract": {
                "runtime_adapters": [],
                "runtime_lora_count": 0,
            },
        }
        _write_json(evidence, summary)
        manifest = {
            "package_kind": PACKAGE_KIND,
            "runtime_adapters": [],
            "input_contract": {
                "context_encoder": "scene-prefixed-decimal17@v1",
                "max_input_tokens": 17,
            },
            "training_lineage": {"peft_artifact_runtime_loaded": False},
            "deployment": {
                "model_mode": MODEL_MODE,
                "runtime_adapters": [],
                "runtime_lora_count": 0,
                "request_level_lora_switching": False,
                "scene_prefixes": {"traffic": "T", "industrial": "I"},
                "input_tokens": 17,
                "output_tokens": 1,
                "artifact_bytes": 577990656,
                "artifact_sha256": "3" * 64,
            },
            "evaluation": {
                "evidence": {
                    "q5_static_joint_gate_summary": {
                        "path": "evidence/q5_static_joint_gate_summary.json",
                        "sha256": sha256_file(evidence),
                    }
                }
            },
        }
        _write_json(package / "scene_adapter_manifest.json", manifest)
        return package, manifest

    def test_dedicated_package_validator_requires_explicit_empty_runtime_adapters(self) -> None:
        package, _ = self._package_fixture()
        with mock.patch(
            "edge_llm_factory.q5_static_joint_release.validate_adapter_package",
            return_value={"status": "valid"},
        ):
            result = validate_q5_static_joint_package(package, self.root / "base.json")
        self.assertEqual(result["runtime_adapters"], [])
        self.assertEqual(result["deployment_model_mode"], MODEL_MODE)

    def test_dedicated_package_validator_rejects_hidden_runtime_adapter(self) -> None:
        package, manifest = self._package_fixture()
        manifest["runtime_adapters"] = [{"id": 0, "path": "forbidden.gguf"}]
        _write_json(package / "scene_adapter_manifest.json", manifest)
        with mock.patch(
            "edge_llm_factory.q5_static_joint_release.validate_adapter_package",
            return_value={"status": "valid"},
        ):
            with self.assertRaisesRegex(ManifestError, "runtime_adapters"):
                validate_q5_static_joint_package(package, self.root / "base.json")

    def test_dedicated_package_validator_rejects_noncanonical_context_encoder(self) -> None:
        package, manifest = self._package_fixture()
        manifest["input_contract"]["context_encoder"] = "joint-scene-prefix-decimal17@v1"
        _write_json(package / "scene_adapter_manifest.json", manifest)
        with mock.patch(
            "edge_llm_factory.q5_static_joint_release.validate_adapter_package",
            return_value={"status": "valid"},
        ):
            with self.assertRaisesRegex(ManifestError, "input_contract.context_encoder"):
                validate_q5_static_joint_package(package, self.root / "base.json")

    def test_dedicated_package_validator_rejects_noncanonical_max_input_tokens(self) -> None:
        package, manifest = self._package_fixture()
        manifest["input_contract"]["max_input_tokens"] = 18
        _write_json(package / "scene_adapter_manifest.json", manifest)
        with mock.patch(
            "edge_llm_factory.q5_static_joint_release.validate_adapter_package",
            return_value={"status": "valid"},
        ):
            with self.assertRaisesRegex(ManifestError, "input_contract.max_input_tokens"):
                validate_q5_static_joint_package(package, self.root / "base.json")

    def test_descriptor_with_runtime_adapter_fails_before_output_creation(self) -> None:
        descriptor = self.root / "descriptor.json"
        output = self.root / "output"
        _write_json(
            descriptor,
            {
                "schema_version": SCHEMA,
                "runtime_adapter": {"path": "forbidden.gguf"},
            },
        )
        with self.assertRaisesRegex(ManifestError, "must not contain runtime adapters"):
            build_q5_static_joint_release(descriptor, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
