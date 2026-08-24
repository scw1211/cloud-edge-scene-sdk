"""Release-store and llama supervisor lifecycle regression tests."""

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from edge_llm_factory.contracts import (  # noqa: E402
    ManifestError,
    base_fingerprint,
    read_json_object,
    sha256_file,
    validate_gates,
    write_json_object,
)
from edge_llm_factory.release_store import ReleaseStore  # noqa: E402
from edge_llm_factory.release_runtime import (  # noqa: E402
    _runtime_matches_release,
    load_active_edge_llm,
)
from edge_llm_factory.serve_release import ActiveReleaseLlamaServer  # noqa: E402
from edge_llm_factory.providers import (  # noqa: E402
    validate_disabled_runtime_config,
    validate_runtime_config,
)


def _base_manifest() -> dict:
    return {
        "schema_version": "edge-llm-base/v1",
        "base_id": "test-base@1",
        "source": {
            "model_id": "test/base",
            "revision": "a" * 40,
            "config_sha256": "1" * 64,
            "tokenizer_sha256": "2" * 64,
            "tokenizer_config_sha256": "3" * 64,
            "artifacts": [
                {"path": "model.safetensors", "sha256": "4" * 64, "bytes": 1}
            ],
        },
        "model": {
            "loader": "test.Loader",
            "architecture": "TestForCausalLM",
            "model_type": "test_text",
            "modality": "text_only",
            "parameter_count": 1,
            "num_hidden_layers": 1,
            "hidden_size": 1,
        },
        "lora_policy": {
            "format": "peft-safetensors",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "max_rank": 8,
            "allowed_target_modules": ["q_proj"],
            "allow_modules_to_save": False,
        },
        "decision_protocol": {
            "name": "single_token_action/v1",
            "max_input_tokens": 16,
            "max_output_tokens": 1,
            "thinking": False,
            "slots": [{"slot": "A", "token": "A", "token_id": 32}],
            "reserved_slots": {},
        },
        "deployment_gates": {
            "max_system_ram_mb": 1536,
            "max_mean_closed_loop_ms": 200,
            "min_valid_output_rate": 0.99,
        },
    }


def _write_safetensors(path: Path) -> None:
    header = json.dumps(
        {
            "test.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")


def _make_release(root: Path, version: str) -> tuple:
    base = _base_manifest()
    base_path = root / "base.json"
    if not base_path.exists():
        write_json_object(base_path, base)

    package = root / ("package-" + version)
    package.mkdir()
    artifact = root / ("model-" + version + ".gguf")
    artifact.write_bytes(("gguf-" + version).encode("utf-8"))

    action_mapping = {
        "schema_version": "edge-llm-action-map/v1",
        "scene": "test_scene",
        "protocol": "single_token_action/v1",
        "entries": [
            {
                "slot": "A",
                "decision": "no_action",
                "candidate_action_type": None,
                "min_risk_level": "low",
                "max_risk_level": "severe",
                "requires_cloud": False,
                "safe_offline": True,
            }
        ],
        "fallback_slot": "A",
    }
    action_path = package / "action_mapping.json"
    write_json_object(action_path, action_mapping)

    adapter_path = package / "adapter_model.safetensors"
    _write_safetensors(adapter_path)
    write_json_object(
        package / "adapter_config.json",
        {
            "base_model_name_or_path": "test/base",
            "bias": "none",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 1,
            "lora_alpha": 1,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj"],
            "modules_to_save": None,
        },
    )
    evidence_path = package / "evidence" / "accuracy.json"
    write_json_object(evidence_path, {"accuracy": 1.0})
    gates = [{"metric": "accuracy", "operator": ">=", "value": 0.5}]
    metrics = {"accuracy": 1.0}
    manifest = {
        "schema_version": "edge-llm-adapter/v1",
        "adapter_id": "test-adapter-" + version,
        "scene": "test_scene",
        "version": version,
        "base": {
            "base_id": base["base_id"],
            "fingerprint": base_fingerprint(base),
        },
        "adapter_artifact": {
            "path": adapter_path.name,
            "sha256": sha256_file(adapter_path),
            "bytes": adapter_path.stat().st_size,
            "format": "safetensors",
        },
        "lora": {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "rank": 1,
            "alpha": 1,
            "dropout": 0.0,
            "target_modules": ["q_proj"],
            "modules_to_save": [],
        },
        "input_contract": {
            "event_type": "test.event.v1",
            "data_schema": "https://example.test/test-event-v1.json",
            "context_encoder": "test-encoder@1",
            "llm_input_type": "compact_text_code",
            "max_input_tokens": 16,
            "direct_media_to_llm": False,
        },
        "action_mapping": {
            "path": action_path.name,
            "sha256": sha256_file(action_path),
        },
        "training": {
            "teacher_model": "test-teacher",
            "methods": ["sft"],
            "train_dataset_id": "test-train",
            "test_dataset_id": "test-heldout",
            "test_set_used_for_training": False,
        },
        "evaluation": {
            "evidence": {
                "accuracy": {
                    "path": "evidence/accuracy.json",
                    "sha256": sha256_file(evidence_path),
                }
            },
            "metrics": metrics,
            "metric_sources": {
                "accuracy": {"evidence": "accuracy", "path": "accuracy"}
            },
            "gates": gates,
            "gate_results": validate_gates(metrics, gates),
        },
        "deployment": {
            "runtime": "llama.cpp",
            "format": "gguf",
            "quantization": "TEST",
            "artifact_sha256": sha256_file(artifact),
            "artifact_bytes": artifact.stat().st_size,
            "max_input_tokens": 16,
            "max_output_tokens": 1,
            "thinking": False,
        },
    }
    write_json_object(package / "scene_adapter_manifest.json", manifest)
    return base_path, package, artifact


class _FakeProcess:
    next_pid = 1000

    def __init__(self) -> None:
        type(self).next_pid += 1
        self.pid = type(self).next_pid
        self.alive = True

    def poll(self):
        return None if self.alive else 0

    def terminate(self) -> None:
        self.alive = False

    def kill(self) -> None:
        self.alive = False

    def wait(self, timeout=None):
        self.alive = False
        return 0


class _FakeReleaseServer(ActiveReleaseLlamaServer):
    def __init__(
        self,
        root: Path,
        registry: Path,
        lora_adapters=None,
        runtime_output_descriptor_paths=None,
    ) -> None:
        binary = root / "fake-llama-server"
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(0o700)
        runtime_config = root / "runtime.json"
        runtime_config.write_text("{}\n", encoding="utf-8")
        super().__init__(
            registry_path=registry,
            runtime_config_path=(
                None if runtime_output_descriptor_paths else runtime_config
            ),
            binary=binary,
            host="127.0.0.1",
            port=18990,
            context_tokens=16,
            threads=1,
            parallel=1,
            gpu_layers=0,
            poll_seconds=0.01,
            startup_timeout_seconds=0.1,
            lora_adapters=lora_adapters,
            runtime_output_descriptor_paths=runtime_output_descriptor_paths,
        )
        self.fail_release_ids = set()
        self.transitions = []

    def _is_process_healthy(self):
        return self.process is not None and self.process.poll() is None

    def _start_record(self, release_id, revision, record):
        self.stop_process()
        if release_id in self.fail_release_ids:
            raise RuntimeError("synthetic candidate startup failure")
        self.process = _FakeProcess()
        self.active_release_id = str(release_id)
        self.applied_revision = int(revision)
        self.active_record = dict(record)
        result = {
            "status": "active",
            "release_id": str(release_id),
            "revision": int(revision),
            "artifact": record["deployment_artifact"]["path"],
            "endpoint": self.endpoint,
            "pid": self.process.pid,
        }
        self.transitions.append((str(release_id), int(revision)))
        return result


class EdgeLLMReleaseLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="edge-llm-release-")
        self.root = Path(self.temporary.name)
        self.registry = self.root / "release-store.json"
        self.store = ReleaseStore(self.registry)
        self.release_a = _make_release(self.root, "1.0.0")
        self.release_b = _make_release(self.root, "2.0.0")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _promote(self, release_id: str, release, runtime_adapters=None) -> dict:
        base, package, artifact = release
        return self.store.promote(
            release_id,
            base,
            package,
            artifact,
            runtime_adapters=runtime_adapters,
        )

    def _runtime_template(self, path: Path, adapter_id: int) -> None:
        write_json_object(
            path,
            {
                "authentication": {"api_key_env": ""},
                "endpoint": "http://127.0.0.1:18990",
                "generation": {
                    "keep_alive": "30m",
                    "max_input_tokens": 16,
                    "max_output_tokens": 1,
                    "seed": 42,
                    "temperature": 0.0,
                    "thinking": False,
                    "top_p": 1.0,
                },
                "lora_adapter": {"id": adapter_id, "scale": 1.0},
                "model": "template.gguf",
                "provider": "llama_cpp",
                "schema_version": "edge-llm-runtime/v1",
                "timeout_seconds": 0.18 if adapter_id == 1 else 0.5,
            },
        )

    def _runtime_output_descriptors(self) -> tuple:
        traffic_template = self.root / "traffic-template.json"
        industrial_template = self.root / "industrial-template.json"
        self._runtime_template(traffic_template, 0)
        self._runtime_template(industrial_template, 1)
        traffic_output = self.root / "generated" / "traffic.json"
        industrial_output = self.root / "generated" / "industrial.json"
        traffic_descriptor = self.root / "traffic-output.json"
        industrial_descriptor = self.root / "industrial-output.json"
        write_json_object(
            traffic_descriptor,
            {
                "schema_version": "edge-llm-runtime-output/v1",
                "name": "traffic",
                "template": str(traffic_template),
                "output": str(traffic_output),
                "adapter_id": 0,
                "on_missing_adapter": "base",
            },
        )
        write_json_object(
            industrial_descriptor,
            {
                "schema_version": "edge-llm-runtime-output/v1",
                "name": "industrial",
                "template": str(industrial_template),
                "output": str(industrial_output),
                "adapter_id": 1,
                "on_missing_adapter": "disable",
            },
        )
        return (
            [traffic_descriptor, industrial_descriptor],
            traffic_output,
            industrial_output,
        )

    def test_multi_runtime_outputs_share_release_binding_and_rollback_legacy_policy(
        self,
    ) -> None:
        old = self._promote("release-a", self.release_a)
        traffic_lora = self.root / "traffic-lora.gguf"
        industrial_lora = self.root / "industrial-lora.gguf"
        traffic_lora.write_bytes(b"traffic")
        industrial_lora.write_bytes(b"industrial")
        current = self._promote(
            "release-b",
            self.release_b,
            runtime_adapters=[traffic_lora, industrial_lora],
        )
        descriptors, traffic_output, industrial_output = (
            self._runtime_output_descriptors()
        )
        server = _FakeReleaseServer(
            self.root,
            self.registry,
            runtime_output_descriptor_paths=descriptors,
        )

        published = server._publish_runtime_configuration(
            "release-b",
            current["revision"],
            current["release"],
            self.release_b[2],
        )
        self.assertEqual(
            [row["state"] for row in published["outputs"]],
            ["adapter", "adapter"],
        )
        traffic = validate_runtime_config(read_json_object(traffic_output))
        industrial = validate_runtime_config(read_json_object(industrial_output))
        self.assertEqual(traffic["lora_adapter"]["id"], 0)
        self.assertEqual(industrial["lora_adapter"]["id"], 1)
        self.assertEqual(
            traffic["release_binding"]["binding_fingerprint"],
            industrial["release_binding"]["binding_fingerprint"],
        )
        self.assertEqual(traffic["release_binding"]["revision"], 2)

        rollback = self.store.rollback("release-a")
        restored = server._publish_runtime_configuration(
            "release-a",
            rollback["revision"],
            rollback["release"],
            self.release_a[2],
        )
        self.assertEqual(
            [row["state"] for row in restored["outputs"]],
            ["base", "disabled"],
        )
        traffic = validate_runtime_config(read_json_object(traffic_output))
        industrial = validate_disabled_runtime_config(
            read_json_object(industrial_output)
        )
        self.assertIsNone(traffic["lora_adapter"])
        self.assertEqual(industrial["required_adapter_id"], 1)
        self.assertEqual(
            traffic["release_binding"]["binding_fingerprint"],
            old["release"]["binding_fingerprint"],
        )
        self.assertEqual(
            self.store.status(False)["history"][-1]["audit"]["restored_release"][
                "runtime_adapters"
            ],
            [],
        )

    def test_legacy_single_runtime_rollback_removes_stale_lora(self) -> None:
        old = self._promote("release-a", self.release_a)
        adapter = self.root / "traffic-lora.gguf"
        adapter.write_bytes(b"traffic")
        current = self._promote(
            "release-b", self.release_b, runtime_adapters=[adapter]
        )
        server = _FakeReleaseServer(self.root, self.registry)
        self._runtime_template(server.runtime_config_path, 0)
        candidate_config = server._legacy_runtime_config(
            "release-b", current["revision"], current["release"], self.release_b[2]
        )
        write_json_object(server.runtime_config_path, candidate_config)
        restored_config = server._legacy_runtime_config(
            "release-a", 3, old["release"], self.release_a[2]
        )
        self.assertIsNone(restored_config["lora_adapter"])
        self.assertIsNone(restored_config["release_binding"]["adapter_id"])

    def test_runtime_release_binding_rejects_stale_revision(self) -> None:
        promoted = self._promote("release-a", self.release_a)
        server = _FakeReleaseServer(self.root, self.registry)
        self._runtime_template(server.runtime_config_path, 0)
        template = validate_runtime_config(read_json_object(server.runtime_config_path))
        template.pop("lora_adapter", None)
        template["model"] = str(self.release_a[2].resolve())
        template["release_binding"] = server._release_binding(
            "release-a", 1, promoted["release"], "traffic", None
        )
        runtime = validate_runtime_config(template)
        manifest = read_json_object(
            self.release_a[1] / "scene_adapter_manifest.json"
        )
        _runtime_matches_release(
            runtime,
            manifest,
            "release-a",
            self.release_a[2],
            revision=1,
            record=promoted["release"],
        )
        runtime["release_binding"]["revision"] = 2
        with self.assertRaisesRegex(ManifestError, "revision"):
            _runtime_matches_release(
                runtime,
                manifest,
                "release-a",
                self.release_a[2],
                revision=1,
                record=promoted["release"],
            )

    def test_multi_runtime_transaction_recovers_all_old_after_second_rename_fault(
        self,
    ) -> None:
        traffic_lora = self.root / "traffic-lora.gguf"
        industrial_lora = self.root / "industrial-lora.gguf"
        traffic_lora.write_bytes(b"traffic")
        industrial_lora.write_bytes(b"industrial")
        promoted = self._promote(
            "release-b",
            self.release_b,
            runtime_adapters=[traffic_lora, industrial_lora],
        )
        descriptors, traffic_output, industrial_output = (
            self._runtime_output_descriptors()
        )
        traffic_output.parent.mkdir(parents=True)
        traffic_output.write_bytes(b'{"old":"traffic"}\n')
        industrial_output.write_bytes(b'{"old":"industrial"}\n')
        old_traffic = traffic_output.read_bytes()
        old_industrial = industrial_output.read_bytes()
        server = _FakeReleaseServer(
            self.root,
            self.registry,
            runtime_output_descriptor_paths=descriptors,
        )
        original_replace = server._replace_runtime_output
        calls = {"value": 0}

        def fail_second(path, payload):
            calls["value"] += 1
            if calls["value"] == 2:
                raise OSError("synthetic rename fault")
            return original_replace(path, payload)

        with patch.object(server, "_replace_runtime_output", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "restored old generation"):
                server._publish_runtime_configuration(
                    "release-b",
                    promoted["revision"],
                    promoted["release"],
                    self.release_b[2],
                )
        self.assertEqual(traffic_output.read_bytes(), old_traffic)
        self.assertEqual(industrial_output.read_bytes(), old_industrial)
        self.assertFalse(server.runtime_transaction_journal_path.exists())

    def test_multi_runtime_constructor_recovers_interrupted_mixed_generation(
        self,
    ) -> None:
        self._promote("release-a", self.release_a)
        descriptors, traffic_output, industrial_output = (
            self._runtime_output_descriptors()
        )
        traffic_output.parent.mkdir(parents=True)
        traffic_output.write_bytes(b'{"generation":"old-traffic"}\n')
        industrial_output.write_bytes(b'{"generation":"old-industrial"}\n')
        server = _FakeReleaseServer(
            self.root,
            self.registry,
            runtime_output_descriptor_paths=descriptors,
        )
        old_traffic = traffic_output.read_bytes()
        old_industrial = industrial_output.read_bytes()
        entries = []
        for name, path, payload in (
            ("traffic", traffic_output, old_traffic),
            ("industrial", industrial_output, old_industrial),
        ):
            entries.append(
                {
                    "name": name,
                    "path": str(path),
                    "old_exists": True,
                    "old_base64": __import__("base64").b64encode(payload).decode(
                        "ascii"
                    ),
                    "old_sha256": __import__("hashlib").sha256(payload).hexdigest(),
                }
            )
        write_json_object(
            server.runtime_transaction_journal_path,
            {
                "schema_version": "edge-llm-runtime-output-transaction/v1",
                "status": "prepared",
                "release_id": "release-a",
                "revision": 1,
                "entries": entries,
            },
        )
        traffic_output.write_bytes(b'{"generation":"new-traffic"}\n')

        recovered = _FakeReleaseServer(
            self.root,
            self.registry,
            runtime_output_descriptor_paths=descriptors,
        )
        self.assertFalse(recovered.runtime_transaction_journal_path.exists())
        self.assertEqual(traffic_output.read_bytes(), old_traffic)
        self.assertEqual(industrial_output.read_bytes(), old_industrial)

    def test_release_record_binds_ordered_runtime_adapters_and_rollback_removes_them(
        self,
    ) -> None:
        old = self._promote("release-a", self.release_a)
        self.assertEqual(old["release"]["runtime_adapters"], [])

        traffic = self.root / "traffic-lora.gguf"
        industrial = self.root / "industrial-lora.gguf"
        traffic.write_bytes(b"traffic-lora")
        industrial.write_bytes(b"industrial-lora")
        promoted = self._promote(
            "release-b", self.release_b, runtime_adapters=[traffic, industrial]
        )
        records = promoted["release"]["runtime_adapters"]
        self.assertEqual([item["id"] for item in records], [0, 1])
        self.assertEqual([item["default_scale"] for item in records], [0, 0])
        self.assertEqual(
            [item["bytes"] for item in records],
            [traffic.stat().st_size, industrial.stat().st_size],
        )
        self.assertEqual(
            [item["sha256"] for item in records],
            [sha256_file(traffic), sha256_file(industrial)],
        )

        server = _FakeReleaseServer(self.root, self.registry)
        adapter_paths = server._runtime_adapters_for_record(promoted["release"])
        command = server.command(self.release_b[2], adapter_paths)
        self.assertEqual(command.count("--lora-scaled"), 1)
        self.assertEqual(
            command[command.index("--lora-scaled") + 1],
            "{}:0,{}:0".format(traffic.resolve(), industrial.resolve()),
        )

        rollback = self.store.rollback("release-a")
        self.assertEqual(rollback["release"]["runtime_adapters"], [])
        restored_adapters = server._runtime_adapters_for_record(rollback["release"])
        self.assertEqual(restored_adapters, [])
        self.assertNotIn(
            "--lora-scaled", server.command(self.release_a[2], restored_adapters)
        )

    def test_runtime_adapter_integrity_is_reverified(self) -> None:
        adapter = self.root / "traffic-lora.gguf"
        adapter.write_bytes(b"traffic-lora")
        promoted = self._promote(
            "release-a", self.release_a, runtime_adapters=[adapter]
        )
        self.assertEqual(
            self.store.status(verify_active=True)["active_integrity"]["status"],
            "verified",
        )
        adapter.write_bytes(b"changed")
        with self.assertRaisesRegex(ManifestError, "大小已变化|SHA256 已变化"):
            self.store.status(verify_active=True)
        self.assertEqual(promoted["release"]["runtime_adapters"][0]["id"], 0)

    def test_runtime_adapter_contract_fails_closed(self) -> None:
        adapter = self.root / "adapter.gguf"
        adapter.write_bytes(b"adapter")
        bad_values = (
            [{"id": 1, "path": adapter, "default_scale": 0}],
            [{"id": 0, "path": adapter, "default_scale": 2}],
            [
                {
                    "id": 0,
                    "path": adapter,
                    "default_scale": 0,
                    "sha256": "0" * 64,
                }
            ],
        )
        for index, value in enumerate(bad_values):
            with self.subTest(value=value):
                with self.assertRaises(ManifestError):
                    self._promote(
                        "bad-{}".format(index),
                        self.release_a,
                        runtime_adapters=value,
                    )

        link = self.root / "adapter-link.gguf"
        link.symlink_to(adapter)
        with self.assertRaisesRegex(ManifestError, "符号链接"):
            self._promote(
                "bad-symlink", self.release_a, runtime_adapters=[link]
            )

    def test_single_process_default_adapter_uses_lora_and_omits_request_binding(
        self,
    ) -> None:
        adapter = self.root / "joint-lora-f16.gguf"
        adapter.write_bytes(b"joint-lora")
        promoted = self._promote(
            "release-b",
            self.release_b,
            runtime_adapters=[
                {"id": 0, "path": adapter, "default_scale": 1}
            ],
        )
        record = promoted["release"]
        self.assertEqual(record["runtime_adapters"][0]["default_scale"], 1)

        templates = []
        descriptors = []
        outputs = []
        for scene, timeout in (("traffic", 0.5), ("industrial", 0.18)):
            template = self.root / (scene + "-joint-template.json")
            write_json_object(
                template,
                {
                    "authentication": {"api_key_env": ""},
                    "endpoint": "http://127.0.0.1:19393",
                    "generation": {
                        "keep_alive": "30m",
                        "max_input_tokens": 17,
                        "max_output_tokens": 1,
                        "seed": 42,
                        "temperature": 0.0,
                        "thinking": False,
                        "top_p": 1.0,
                    },
                    "model": "joint-candidate",
                    "provider": "llama_cpp",
                    "schema_version": "edge-llm-runtime/v1",
                    "timeout_seconds": timeout,
                },
            )
            output = self.root / "generated-joint" / (scene + ".json")
            descriptor = self.root / (scene + "-joint-output.json")
            write_json_object(
                descriptor,
                {
                    "schema_version": "edge-llm-runtime-output/v1",
                    "name": scene,
                    "template": str(template),
                    "output": str(output),
                    "adapter_id": 0,
                    "adapter_mode": "process_default",
                    "on_missing_adapter": "disable",
                },
            )
            templates.append(template)
            descriptors.append(descriptor)
            outputs.append(output)

        server = _FakeReleaseServer(
            self.root,
            self.registry,
            runtime_output_descriptor_paths=descriptors,
        )
        runtime_adapters = server._runtime_adapters_for_record(record)
        command = server.command(self.release_b[2], runtime_adapters)
        self.assertNotIn("--lora-scaled", command)
        self.assertEqual(command.count("--lora"), 1)
        self.assertEqual(command[command.index("--lora") + 1], str(adapter.resolve()))

        published = server._publish_runtime_configuration(
            "release-b",
            promoted["revision"],
            record,
            self.release_b[2],
        )
        self.assertEqual(
            [row["state"] for row in published["outputs"]],
            ["process_default_adapter", "process_default_adapter"],
        )
        for output in outputs:
            runtime = validate_runtime_config(read_json_object(output))
            self.assertIsNone(runtime["lora_adapter"])
            self.assertEqual(runtime["generation"]["max_input_tokens"], 17)
            self.assertIsNone(runtime["release_binding"]["adapter_id"])

    def test_static_fusion_outputs_enable_both_scenes_and_rollback_legacy_policy(
        self,
    ) -> None:
        old = self._promote("release-a", self.release_a)
        promoted = self._promote("release-static", self.release_b)
        deployment_sha = promoted["release"]["deployment_artifact"]["sha256"]
        descriptors = []
        outputs = []
        for scene, policy in (("traffic", "base"), ("industrial", "disable")):
            template = self.root / (scene + "-static-template.json")
            self._runtime_template(template, 0)
            value = read_json_object(template)
            value.pop("lora_adapter")
            value["generation"]["max_input_tokens"] = 17
            write_json_object(template, value)
            output = self.root / "generated-static" / (scene + ".json")
            descriptor = self.root / (scene + "-static-output.json")
            write_json_object(
                descriptor,
                {
                    "schema_version": "edge-llm-runtime-output/v1",
                    "name": scene + "-static",
                    "template": str(template),
                    "output": str(output),
                    "adapter_mode": "static",
                    "static_deployment_sha256": deployment_sha,
                    "on_missing_adapter": policy,
                },
            )
            descriptors.append(descriptor)
            outputs.append(output)

        server = _FakeReleaseServer(
            self.root,
            self.registry,
            runtime_output_descriptor_paths=descriptors,
        )
        adapters = server._runtime_adapters_for_record(promoted["release"])
        self.assertEqual(adapters, [])
        command = server.command(self.release_b[2], adapters)
        self.assertNotIn("--lora", command)
        self.assertNotIn("--lora-scaled", command)
        published = server._publish_runtime_configuration(
            "release-static",
            promoted["revision"],
            promoted["release"],
            self.release_b[2],
        )
        self.assertEqual(
            [row["state"] for row in published["outputs"]],
            ["static_fusion", "static_fusion"],
        )
        for output in outputs:
            runtime = validate_runtime_config(read_json_object(output))
            self.assertIsNone(runtime["lora_adapter"])
            self.assertIsNone(runtime["release_binding"]["adapter_id"])

        rollback = self.store.rollback("release-a")
        restored = server._publish_runtime_configuration(
            "release-a",
            rollback["revision"],
            rollback["release"],
            self.release_a[2],
        )
        self.assertEqual(
            [row["state"] for row in restored["outputs"]],
            ["base", "disabled"],
        )
        self.assertIsNone(
            validate_runtime_config(read_json_object(outputs[0]))["lora_adapter"]
        )
        disabled = validate_disabled_runtime_config(read_json_object(outputs[1]))
        self.assertEqual(disabled["reason"], "release_missing_static_deployment")
        self.assertEqual(disabled["required_deployment_sha256"], deployment_sha)
        self.assertEqual(
            validate_runtime_config(read_json_object(outputs[0]))["release_binding"][
                "binding_fingerprint"
            ],
            old["release"]["binding_fingerprint"],
        )

    def test_legacy_cli_adapter_only_applies_to_legacy_release_record(self) -> None:
        adapter = self.root / "legacy.gguf"
        adapter.write_bytes(b"legacy")
        promoted = self._promote("release-a", self.release_a)
        server = _FakeReleaseServer(
            self.root, self.registry, lora_adapters=[adapter]
        )

        legacy_record = dict(promoted["release"])
        legacy_record.pop("runtime_adapters")
        self.assertEqual(
            server._runtime_adapters_for_record(legacy_record), [adapter.resolve()]
        )
        with self.assertRaisesRegex(ManifestError, "不能.*混用"):
            server._runtime_adapters_for_record(promoted["release"])

    def test_registry_without_runtime_adapters_remains_verifiable(self) -> None:
        self._promote("release-a", self.release_a)
        state = json.loads(self.registry.read_text(encoding="utf-8"))
        state["releases"]["release-a"].pop("runtime_adapters")
        write_json_object(self.registry, state)
        verified = self.store.status(verify_active=True)
        self.assertEqual(verified["active_integrity"]["status"], "verified")

    def test_promote_apply_and_explicit_rollback_are_observable(self) -> None:
        self.assertEqual(self._promote("release-a", self.release_a)["revision"], 1)
        server = _FakeReleaseServer(self.root, self.registry)
        first = server.apply_current(force=True)
        self.assertEqual((first["release_id"], first["revision"]), ("release-a", 1))

        self.assertEqual(self._promote("release-b", self.release_b)["revision"], 2)
        applied = server.apply_current()
        self.assertEqual((applied["release_id"], applied["revision"]), ("release-b", 2))

        rolled_back = self.store.rollback("release-a")
        self.assertEqual(rolled_back["revision"], 3)
        restored = server.apply_current()
        self.assertEqual((restored["release_id"], restored["revision"]), ("release-a", 3))

        observed = server.status()
        self.assertEqual(observed["status"], "ok")
        self.assertEqual(observed["registry_active_release_id"], "release-a")
        self.assertEqual(observed["applied_release_id"], "release-a")
        self.assertEqual(observed["registry_revision"], observed["applied_revision"])
        self.assertEqual(
            [entry["action"] for entry in self.store.status(False)["history"]],
            ["promote", "promote", "rollback"],
        )
        self.assertEqual(
            server.transitions,
            [("release-a", 1), ("release-b", 2), ("release-a", 3)],
        )

    def test_failed_candidate_rolls_registry_back_and_does_not_loop(self) -> None:
        self._promote("release-a", self.release_a)
        server = _FakeReleaseServer(self.root, self.registry)
        server.apply_current(force=True)
        self._promote("release-b", self.release_b)
        server.fail_release_ids.add("release-b")

        with self.assertRaisesRegex(
            RuntimeError, "synthetic candidate startup failure"
        ):
            server.apply_current()

        stored = self.store.status(verify_active=True)
        self.assertEqual(stored["active_release_id"], "release-a")
        self.assertEqual(stored["revision"], 3)
        audit = stored["history"][-1]["audit"]
        self.assertEqual(audit["trigger"], "candidate_apply_failure")
        self.assertEqual(audit["failed_release_id"], "release-b")
        self.assertEqual(audit["failed_revision"], 2)
        self.assertIn("synthetic candidate startup failure", audit["error"])

        observed = server.status()
        self.assertEqual(observed["status"], "recovered")
        self.assertEqual(observed["registry_active_release_id"], "release-a")
        self.assertEqual(observed["applied_release_id"], "release-a")
        self.assertEqual(observed["registry_revision"], 3)
        self.assertEqual(observed["applied_revision"], 3)
        self.assertEqual(observed["last_failure"]["rollback"]["status"], "rolled_back")

        transition_count = len(server.transitions)
        unchanged = server.apply_current()
        self.assertEqual(unchanged["status"], "unchanged")
        self.assertEqual(len(server.transitions), transition_count)
        self.assertEqual(self.store.status(False)["revision"], 3)

    def test_runtime_exit_is_degraded_without_a_candidate_failure(self) -> None:
        self._promote("release-a", self.release_a)
        server = _FakeReleaseServer(self.root, self.registry)
        server.apply_current(force=True)

        server.process.alive = False

        observed = server.status()
        self.assertEqual(observed["status"], "degraded")
        self.assertFalse(observed["process_running"])
        self.assertIsNone(observed["failed_error"])

    def test_failed_candidate_and_failed_fallback_are_degraded(self) -> None:
        self._promote("release-a", self.release_a)
        server = _FakeReleaseServer(self.root, self.registry)
        server.apply_current(force=True)
        self._promote("release-b", self.release_b)
        server.fail_release_ids.update(("release-a", "release-b"))

        with self.assertRaisesRegex(
            RuntimeError, "previous release could not be restored"
        ):
            server.apply_current()

        observed = server.status()
        self.assertEqual(observed["status"], "degraded")
        self.assertFalse(observed["process_running"])
        self.assertEqual(observed["registry_active_release_id"], "release-a")
        self.assertEqual(observed["registry_revision"], 3)
        self.assertEqual(observed["applied_revision"], 1)
        self.assertIn(
            "synthetic candidate startup failure",
            observed["last_failure"]["fallback_error"],
        )

    def test_conditional_rollback_does_not_revert_a_newer_promotion(self) -> None:
        self._promote("release-a", self.release_a)
        self._promote("release-b", self.release_b)
        release_c = _make_release(self.root, "3.0.0")
        self._promote("release-c", release_c)

        result = self.store.rollback_if_active(
            expected_release_id="release-b",
            expected_revision=2,
            release_id="release-a",
            reason="late failure from release-b",
        )
        self.assertEqual(result["status"], "rollback_skipped")
        self.assertEqual(result["reason"], "active_release_changed")
        state = self.store.status(verify_active=True)
        self.assertEqual(state["active_release_id"], "release-c")
        self.assertEqual(state["revision"], 3)
        self.assertEqual(len(state["history"]), 3)

    def test_base_only_release_is_first_class_and_has_no_adapter_metadata(self) -> None:
        base = self.release_a[0]
        artifact = self.root / "official-base-q4_k_m.gguf"
        artifact.write_bytes(b"official-base-q4-k-m")

        promoted = self.store.promote_base_only(
            "official-base", base, artifact
        )
        record = promoted["release"]
        self.assertEqual(record["deployment_mode"], "base_only")
        self.assertEqual(record["runtime_adapters"], [])
        self.assertEqual(record["deployment_artifact"]["format"], "gguf")
        self.assertNotIn("adapter", record)
        self.assertNotIn("adapter_package", record)

        status = self.store.status(verify_active=True)
        self.assertEqual(status["active_integrity"]["status"], "verified")
        self.assertEqual(
            status["active_integrity"]["deployment_mode"], "base_only"
        )
        self.assertEqual(
            self.store.promote_base_only("official-base", base, artifact)["status"],
            "already_active",
        )

        artifact.write_bytes(b"tampered")
        with self.assertRaisesRegex(ManifestError, "GGUF"):
            self.store.status(verify_active=True)

    def test_serve_release_starts_and_binds_base_only_static_runtime(self) -> None:
        base = self.release_a[0]
        artifact = self.root / "official-base-q4_k_m.gguf"
        artifact.write_bytes(b"official-base-q4-k-m")
        promoted = self.store.promote_base_only(
            "official-base", base, artifact
        )
        record = promoted["release"]

        template = self.root / "base-runtime-template.json"
        write_json_object(
            template,
            {
                "authentication": {"api_key_env": ""},
                "endpoint": "http://127.0.0.1:18990",
                "generation": {
                    "keep_alive": "30m",
                    "max_input_tokens": 16,
                    "max_output_tokens": 1,
                    "seed": 42,
                    "temperature": 0.0,
                    "thinking": False,
                    "top_p": 1.0,
                },
                "model": "template.gguf",
                "provider": "llama_cpp",
                "schema_version": "edge-llm-runtime/v1",
                "timeout_seconds": 0.5,
            },
        )
        output = self.root / "generated" / "base-runtime.json"
        descriptor = self.root / "base-runtime-output.json"
        write_json_object(
            descriptor,
            {
                "schema_version": "edge-llm-runtime-output/v1",
                "name": "base",
                "template": str(template),
                "output": str(output),
                "adapter_mode": "static",
                "static_deployment_sha256": record["deployment_artifact"][
                    "sha256"
                ],
                "on_missing_adapter": "base",
            },
        )
        binary = self.root / "fake-base-llama-server"
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(0o700)
        server = ActiveReleaseLlamaServer(
            registry_path=self.registry,
            runtime_config_path=None,
            binary=binary,
            host="127.0.0.1",
            port=18990,
            context_tokens=256,
            threads=1,
            parallel=1,
            gpu_layers=99,
            poll_seconds=0.01,
            startup_timeout_seconds=0.1,
            runtime_output_descriptor_paths=[descriptor],
        )

        adapters = server._runtime_adapters_for_record(record)
        self.assertEqual(adapters, [])
        command = server.command(artifact, adapters)
        self.assertNotIn("--lora", command)
        self.assertNotIn("--lora-scaled", command)
        runtime_outputs = server._build_runtime_outputs(
            "official-base", 1, record, artifact
        )
        self.assertEqual(runtime_outputs[0]["state"], "static_fusion")
        binding = runtime_outputs[0]["release_binding"]
        self.assertEqual(binding["binding_fingerprint"], record["binding_fingerprint"])
        self.assertIsNone(binding["adapter_id"])
        self.assertIsNone(binding["adapter_sha256"])

        with patch(
            "edge_llm_factory.serve_release.subprocess.Popen",
            return_value=_FakeProcess(),
        ), patch.object(server, "_wait_ready"), patch.object(
            server,
            "_verify_startup_gate",
            return_value={"status": "passed"},
        ), patch.object(
            server,
            "_publish_runtime_configuration",
            return_value={"status": "published"},
        ):
            active = server.apply_current(force=True)
        self.assertEqual(active["status"], "active")
        self.assertEqual(active["runtime_adapters"], [])
        with patch.object(server, "_is_process_healthy", return_value=True):
            self.assertEqual(server.status()["status"], "ok")
        server.stop_process()

    def test_load_active_edge_llm_uses_base_only_scene_protocol(self) -> None:
        artifact = self.root / "official-base-q4_k_m.gguf"
        artifact.write_bytes(b"official-base-q4-k-m")
        action_mapping = self.root / "base-only-action-mapping.json"
        write_json_object(
            action_mapping,
            {
                "schema_version": "edge-llm-action-map/v1",
                "scene": "test_scene",
                "protocol": "single_token_action/v1",
                "entries": [
                    {
                        "slot": "A",
                        "decision": "no_action",
                        "candidate_action_type": None,
                        "min_risk_level": "low",
                        "max_risk_level": "severe",
                        "requires_cloud": False,
                        "safe_offline": True,
                    }
                ],
                "fallback_slot": "A",
            },
        )
        base = _base_manifest()
        base["base_only_scene_protocols"] = {
            "test_scene": {
                "version": "official-base-v1",
                "action_mapping": {
                    "path": action_mapping.name,
                    "sha256": sha256_file(action_mapping),
                },
                "input_contract": {
                    "event_type": "test.event.v1",
                    "data_schema": "https://example.test/test-event-v1.json",
                    "context_encoder": "test-encoder@1",
                    "llm_input_type": "compact_text_code",
                    "max_input_tokens": 16,
                    "direct_media_to_llm": False,
                },
                "deployment": {
                    "runtime": "llama.cpp",
                    "format": "gguf",
                    "quantization": "Q4_K_M",
                    "artifact_sha256": sha256_file(artifact),
                    "artifact_bytes": artifact.stat().st_size,
                    "max_input_tokens": 16,
                    "max_output_tokens": 1,
                    "thinking": False,
                },
            }
        }
        base_path = self.root / "base-only-manifest.json"
        write_json_object(base_path, base)
        promoted = self.store.promote_base_only(
            "official-base", base_path, artifact
        )
        record = promoted["release"]
        runtime_path = self.root / "base-only-runtime.json"
        binding = ActiveReleaseLlamaServer._release_binding(
            "official-base", 1, record, "test_scene", None
        )
        write_json_object(
            runtime_path,
            {
                "authentication": {"api_key_env": ""},
                "endpoint": "http://127.0.0.1:18990",
                "generation": {
                    "keep_alive": "30m",
                    "max_input_tokens": 16,
                    "max_output_tokens": 1,
                    "seed": 42,
                    "temperature": 0.0,
                    "thinking": False,
                    "top_p": 1.0,
                },
                "model": str(artifact.resolve()),
                "provider": "llama_cpp",
                "release_binding": binding,
                "schema_version": "edge-llm-runtime/v1",
                "timeout_seconds": 0.5,
            },
        )

        active = load_active_edge_llm(
            self.registry, runtime_path, expected_scene="test_scene"
        )
        description = active.model.describe()
        self.assertEqual(description["deployment_mode"], "base_only")
        self.assertIsNone(description["adapter_id"])
        self.assertEqual(description["metrics"], {})
        self.assertEqual(description["scene"], "test_scene")
        self.assertEqual(
            description["runtime"]["release_binding"]["binding_fingerprint"],
            record["binding_fingerprint"],
        )


if __name__ == "__main__":
    unittest.main()
