import hashlib
import json
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from edge_llm_factory.build_joint_scene_dataset import RAW16_CONTRACT, build
from edge_llm_factory.contracts import ManifestError
from edge_llm_factory.evaluate_action_tokens import main as evaluate_main
from edge_llm_factory.gate_joint_scene_adapter import gate, main as gate_main
from edge_llm_factory.providers import validate_runtime_config
from edge_llm_factory.serve_release import (
    load_runtime_output_descriptor,
    load_startup_probe_config,
)
from edge_llm_factory.train_joint_scene_adapter import (
    audit_tokenizer_contract,
    main as train_main,
)


def _write_rows(path: Path, scene: str, count: int, split: str) -> None:
    allowed = "ABCDEF" if scene == "traffic" else "ABC"
    with path.open("w", encoding="utf-8") as file_obj:
        for index in range(count):
            prompt = ("0" if scene == "traffic" else "2") + str(
                index % 10**15
            ).zfill(15)
            row = {
                "event_id": "{}:{}:{}".format(scene, split, index),
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": allowed[index % len(allowed)]},
                ],
                "prompt_format": "raw_task",
            }
            file_obj.write(json.dumps(row) + "\n")


class _Tokenizer:
    ids = {"T": 51, "I": 40}

    def __call__(self, text, add_special_tokens=False):
        if text in self.ids:
            return {"input_ids": [self.ids[text]]}
        return {"input_ids": list(range(len(text)))}


class JointSceneAdapterTests(unittest.TestCase):
    def test_joint_candidate_configs_bind_one_process_default_runtime(self):
        repository = Path(__file__).resolve().parents[1]
        candidate = (
            repository
            / "scenes"
            / "industrial_anomaly"
            / "deployment"
            / "joint_candidate"
        )
        runtimes = {}
        expected_base_path = (
            "/home/jetson02/cloud-edge-deploy-industrial-multilora-v1/"
            "deployment/joint_single_adapter/base.Q5_K_M.gguf"
        )
        for scene in ("traffic", "industrial"):
            runtime_path = candidate / (
                "edge_llm_runtime_{}_template.json".format(scene)
            )
            runtime = validate_runtime_config(
                json.loads(runtime_path.read_text(encoding="utf-8"))
            )
            self.assertEqual(runtime["endpoint"], "http://127.0.0.1:19393")
            self.assertEqual(runtime["generation"]["max_input_tokens"], 17)
            self.assertEqual(runtime["generation"]["max_output_tokens"], 1)
            self.assertIsNone(runtime["lora_adapter"])
            self.assertEqual(runtime["model"], expected_base_path)
            descriptor = load_runtime_output_descriptor(
                candidate / "runtime_output_{}.json".format(scene)
            )
            self.assertEqual(descriptor["adapter_id"], 0)
            self.assertEqual(descriptor["adapter_mode"], "process_default")
            self.assertEqual(descriptor["on_missing_adapter"], "disable")
            runtimes[scene] = str(descriptor["output"])

        plugins = json.loads(
            (candidate / "scene_plugins_edge.json").read_text(encoding="utf-8")
        )["plugins"]
        traffic_options = plugins[0]["options"]
        industrial_options = plugins[1]["options"]
        self.assertEqual(traffic_options["edge_llm_prompt_prefix"], "T")
        self.assertEqual(industrial_options["edge_llm_prompt_prefix"], "I")
        self.assertEqual(
            str((repository / traffic_options["edge_llm_runtime_config_path"]).resolve()),
            runtimes["traffic"],
        )
        self.assertEqual(
            str((repository / industrial_options["edge_llm_runtime_config_path"]).resolve()),
            runtimes["industrial"],
        )
        service = json.loads(
            (candidate / "edge_service.json").read_text(encoding="utf-8")
        )
        self.assertEqual(service["role"], "edge")
        self.assertEqual(service["listen"]["port"], 19391)

        identity = json.loads(
            (candidate / "candidate_identity.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            identity["release_id"],
            "traffic-industrial-joint-v1-clean-q5km-f16",
        )
        self.assertEqual(identity["base"]["path"], expected_base_path)
        self.assertEqual(identity["base"]["bytes"], 577990656)
        self.assertEqual(
            identity["base"]["sha256"],
            "3b24fe1e89760dbfccbcbf6420fdc3ca7a6dd07cb2dc5c1578ce62a0252461de",
        )
        self.assertEqual(identity["runtime_adapter"]["bytes"], 12793312)
        self.assertEqual(
            identity["runtime_adapter"]["path"],
            "/home/jetson02/cloud-edge-deploy-industrial-multilora-v1/"
            "deployment/joint_single_adapter/"
            "joint_traffic_industrial.F16-LoRA.gguf",
        )
        self.assertEqual(
            identity["runtime_adapter"]["sha256"],
            "e97da30341514bff9285c389e7d53797037925d253aa2d6578dbf8cbf8ba86f1",
        )
        self.assertEqual(identity["runtime_adapter"]["default_scale"], 1)
        self.assertEqual(identity["status"], "full_passed_nano_pending")
        self.assertEqual(identity["adapter_package"]["status"], "pending_nano_gate")
        self.assertIsNone(identity["adapter_package"]["sha256"])
        self.assertEqual(identity["full_validation"]["status"], "passed")
        self.assertEqual(identity["full_validation"]["result"]["bytes"], 2484871)
        self.assertEqual(
            identity["full_validation"]["result"]["sha256"],
            "e7a1405c2e9546bea71dfb9a3a63a8ece0a69b709142ece95c09098ed2300636",
        )
        probes_path = candidate / "startup_probes.json"
        probes = load_startup_probe_config(probes_path)
        release_probes = probes["releases"][identity["release_id"]]
        self.assertEqual(
            release_probes["deployment_sha256"], identity["base"]["sha256"]
        )
        self.assertEqual([row["id"] for row in release_probes["adapters"]], [0, 0])
        self.assertEqual(
            [row["prompt"][0] for row in release_probes["adapters"]],
            ["T", "I"],
        )
        self.assertTrue(
            all(len(row["prompt"]) == 17 for row in release_probes["adapters"])
        )
        self.assertEqual(
            [row["allowed_tokens"] for row in release_probes["adapters"]],
            [list("ABCDEF"), list("ABC")],
        )
        self.assertEqual(
            [row["expected_token"] for row in release_probes["adapters"]],
            ["F", "A"],
        )
        self.assertEqual(
            [row["expected_prompt_tokens"] for row in release_probes["adapters"]],
            [17, 17],
        )
        self.assertTrue(
            all(
                row["adapter_sha256"] == identity["runtime_adapter"]["sha256"]
                for row in release_probes["adapters"]
            )
        )
        self.assertEqual(identity["startup_probes"]["bytes"], probes_path.stat().st_size)
        self.assertEqual(
            identity["startup_probes"]["sha256"],
            hashlib.sha256(probes_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            service["cloud"]["base_url"], "http://192.168.31.135:18100"
        )
        self.assertFalse(service["release_watch"]["enabled"])
        self.assertEqual(
            identity["deployment_scope"]["purpose"],
            "nano222_controlled_sidecar_candidate",
        )
        self.assertFalse(identity["deployment_scope"]["release_watch_enabled"])
        forbidden_candidate_fragments = (
            "q4km",
            "base.q4",
            "f32-lora",
            "deployment/multilora/",
            '"max_input_tokens": 16',
            '"default_scale": 0',
            '"lora_adapter"',
            "--lora-scaled",
            "127.0.0.1:18190",
            "127.0.0.1:19100",
        )
        for path in candidate.glob("*.json"):
            contents = path.read_text(encoding="utf-8").lower()
            for fragment in forbidden_candidate_fragments:
                self.assertNotIn(fragment, contents, str(path))

    def _dataset(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        paths = {}
        for scene, test_count in (("traffic", 2400), ("industrial", 960)):
            for split, count in (("train", 6), ("validation", 3), ("test", test_count)):
                path = root / "{}_{}.jsonl".format(scene, split)
                _write_rows(path, scene, count, split)
                paths[(scene, split)] = path
        output = root / "joint"
        result = build(
            paths[("traffic", "train")],
            paths[("traffic", "validation")],
            paths[("traffic", "test")],
            paths[("industrial", "train")],
            paths[("industrial", "validation")],
            paths[("industrial", "test")],
            output,
        )
        return paths, output, result

    def test_builder_prefixes_only_train_and_validation_and_binds_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, output, result = self._dataset(Path(directory))
            self.assertEqual(result["artifacts"]["train"]["rows"], 12)
            self.assertEqual(result["artifacts"]["validation"]["rows"], 6)
            self.assertFalse(result["formal_test_content_loaded"])
            for scene in ("traffic", "industrial"):
                identity = result["sources"][scene]["formal_test"]
                self.assertFalse(identity["content_loaded"])
                self.assertEqual(
                    identity["sha256"],
                    hashlib.sha256(paths[(scene, "test")].read_bytes()).hexdigest(),
                )
            rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
            self.assertTrue(all(len(row["messages"][0]["content"]) == 17 for row in rows))
            self.assertEqual(
                {row["messages"][0]["content"][0] for row in rows}, {"T", "I"}
            )

    def test_tokenizer_audit_proves_distinct_single_token_prefixes(self):
        transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: _Tokenizer()
            )
        )
        with patch.dict("sys.modules", {"transformers": transformers}):
            report = audit_tokenizer_contract(Path("unused"))
        self.assertEqual(report["prefixes"]["traffic"]["token_id"], 51)
        self.assertEqual(report["prefixes"]["industrial"]["token_id"], 40)

    def test_raw16_builder_proves_disjoint_scene_domains_without_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for scene, test_count in (("traffic", 2400), ("industrial", 960)):
                for split, count in (("train", 6), ("validation", 3), ("test", test_count)):
                    path = root / "{}_{}.jsonl".format(scene, split)
                    _write_rows(path, scene, count, split)
                    paths[(scene, split)] = path
            result = build(
                paths[("traffic", "train")],
                paths[("traffic", "validation")],
                paths[("traffic", "test")],
                paths[("industrial", "train")],
                paths[("industrial", "validation")],
                paths[("industrial", "test")],
                root / "joint_raw16",
                input_contract=RAW16_CONTRACT,
            )
            self.assertEqual(result["contract"]["input_tokens"], 16)
            self.assertEqual(result["contract"]["prefixes"], {"traffic": "", "industrial": ""})
            self.assertEqual(
                result["contract"]["scene_first_characters"],
                {"traffic": ["0"], "industrial": ["2"]},
            )
            self.assertEqual(
                result["contract"]["cross_scene_prompt_overlap"],
                {"train": 0, "validation": 0},
            )
            rows = [
                json.loads(line)
                for line in (root / "joint_raw16" / "train.jsonl").read_text().splitlines()
            ]
            self.assertTrue(all(len(row["messages"][0]["content"]) == 16 for row in rows))

    def test_raw16_tokenizer_audit_requires_sixteen_tokens(self):
        transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: _Tokenizer()
            )
        )
        with patch.dict("sys.modules", {"transformers": transformers}):
            report = audit_tokenizer_contract(Path("unused"), RAW16_CONTRACT)
        self.assertEqual(report["contract"], "16-to-1")
        self.assertEqual(report["prefixes"]["traffic"]["text"], "")

    def test_trainer_wrapper_passes_only_the_locked_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output, _ = self._dataset(Path(directory))
            adapter = Path(directory) / "adapter"

            def fake_train(argv):
                self.assertIn("--bf16", argv)
                self.assertEqual(argv[argv.index("--rank") + 1], "16")
                self.assertEqual(argv[argv.index("--max_length") + 1], "18")
                adapter.mkdir()
                (adapter / "train_metrics.json").write_text("{}", encoding="utf-8")

            with patch(
                "edge_llm_factory.train_joint_scene_adapter.audit_tokenizer_contract",
                return_value={"contract": "17-to-1"},
            ), patch(
                "edge_llm_factory.train_joint_scene_adapter.train_sft.main",
                side_effect=fake_train,
            ):
                train_main(
                    [
                        "--base", "base.json",
                        "--snapshot", "snapshot",
                        "--snapshot_manifest", "snapshot.json",
                        "--dataset_dir", str(output),
                        "--output", str(adapter),
                    ]
                )
            summary = json.loads((adapter / "train_metrics.json").read_text())
            self.assertTrue(summary["runtime_contract"]["one_resident_adapter"])
            self.assertFalse(summary["runtime_contract"]["request_level_adapter_switching"])

    def test_gate_keeps_scene_metrics_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, manifest_path, traffic, industrial = self._gate_fixture(root)
            result = gate(manifest, traffic, industrial, manifest_path)
            self.assertTrue(result["passed"])

    @staticmethod
    def _identity(path):
        return {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def _gate_fixture(self, root):
        root.mkdir(parents=True, exist_ok=True)
        _, dataset, manifest = self._dataset(root)
        manifest_path = dataset / "manifest.json"
        base = root / "base.json"
        base.write_text(
            json.dumps(
                {
                    "schema_version": "edge-llm-base/v1",
                    "base_id": "test",
                    "source": {
                        "model_id": "test/model",
                        "revision": "a" * 40,
                        "config_sha256": "a" * 64,
                        "tokenizer_sha256": "b" * 64,
                        "tokenizer_config_sha256": "c" * 64,
                        "artifacts": [{"path": "x", "sha256": "d" * 64, "bytes": 1}],
                    },
                    "model": {
                        "loader": "transformers.AutoModelForCausalLM",
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
                        "max_rank": 64,
                        "allowed_target_modules": ["q_proj"],
                        "allow_modules_to_save": False,
                    },
                    "decision_protocol": {
                        "name": "single_token_action/v1",
                        "max_input_tokens": 128,
                        "max_output_tokens": 1,
                        "thinking": False,
                        "slots": [
                            {"slot": token, "token": token, "token_id": 32 + index}
                            for index, token in enumerate("ABCDEFGH")
                        ],
                        "reserved_slots": {"G": "abstain", "H": "request_cloud"},
                    },
                    "deployment_gates": {
                        "max_system_ram_mb": 1536,
                        "max_mean_closed_loop_ms": 200,
                        "min_valid_output_rate": 0.99,
                    },
                }
            ),
            encoding="utf-8",
        )
        snapshot = root / "snapshot_manifest.json"
        evaluator = root / "evaluator.py"
        config = root / "adapter_config.json"
        weights = root / "adapter_model.safetensors"
        metrics = root / "train_metrics.json"
        for path, value in ((snapshot, "{}"), (evaluator, "# evaluator\n"), (config, "{}"), (weights, "weights")):
            path.write_text(value, encoding="utf-8")
        metrics.write_text(
            json.dumps(
                {
                    "train_jsonl_sha256": manifest["artifacts"]["train"]["sha256"],
                    "train_rows": manifest["artifacts"]["train"]["rows"],
                    "val_jsonl_sha256": manifest["artifacts"]["validation"]["sha256"],
                    "val_rows": manifest["artifacts"]["validation"]["rows"],
                    "runtime_contract": {
                        "one_resident_adapter": True,
                        "request_level_adapter_switching": False,
                        "input_tokens": 17,
                        "output_tokens": 1,
                    },
                }
            ),
            encoding="utf-8",
        )
        common = {
            "evaluation_mode": "joint_formal",
            "test_set_used_for_training": False,
            "prompt_format": "raw_task",
            "required_prompt_tokens": 17,
            "dataset_manifest_identity": self._identity(manifest_path),
            "dataset_artifacts": {
                split: {
                    "path": str(Path(manifest["artifacts"][split]["path"]).resolve()),
                    "bytes": manifest["artifacts"][split]["bytes"],
                    "sha256": manifest["artifacts"][split]["sha256"],
                    "rows": manifest["artifacts"][split]["rows"],
                }
                for split in ("train", "validation")
            },
            "evaluator_identity": self._identity(evaluator),
            "base_manifest_identity": self._identity(base),
            "snapshot_manifest_identity": self._identity(snapshot),
            "adapter_artifacts": {
                "weights": self._identity(weights),
                "config": self._identity(config),
                "train_metrics": self._identity(metrics),
            },
            "expected_adapter_weights_sha256": self._identity(weights)["sha256"],
            "precision": {
                "requested": "bfloat16",
                "effective": "bfloat16",
                "floating_parameter_dtypes": {"torch.bfloat16": 1},
                "parameter_devices": {"cuda": 1},
                "cuda_available": True,
            },
        }

        def report(scene):
            allowed = "ABCDEF" if scene == "traffic" else "ABC"
            count = 2400 if scene == "traffic" else 960
            samples = []
            for index in range(count):
                token = allowed[index % len(allowed)]
                samples.append(
                    {
                        "event_id": "{}:{}".format(scene, index),
                        "target": token,
                        "prediction": token,
                        "valid": True,
                        "correct": True,
                        "prompt_tokens": 17,
                        "generated_token_ids": [32 + ord(token) - ord("A")],
                    }
                )
            per_class = {
                token: {"support": count // len(allowed), "precision": 1.0, "recall": 1.0, "f1": 1.0}
                for token in allowed
            }
            return {
                **copy.deepcopy(common),
                "joint_scene": scene,
                "count": count,
                "test_jsonl_sha256": manifest["sources"][scene]["formal_test"]["sha256"],
                "prompt_prefix": "T" if scene == "traffic" else "I",
                "valid_output_rate": 1.0,
                "decision_accuracy": 1.0,
                "macro_f1": 1.0,
                "weighted_f1": 1.0,
                "per_class": per_class,
                "decoding_constraint": {
                    "enabled": True,
                    "backend": "transformers_prefix_allowed_tokens_fn",
                    "allowed_tokens": list(allowed),
                    "allowed_token_ids": {token: 32 + ord(token) - ord("A") for token in allowed},
                    "reserved_slots_excluded_from_sampling": {"G": "G", "H": "H"},
                    "applies_before_sampling": True,
                    "post_hoc_remapping": False,
                },
                "samples": samples,
            }
        return manifest, manifest_path, report("traffic"), report("industrial")

    def test_gate_rejects_different_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, path, traffic, industrial = self._gate_fixture(root)
            other = root / "other.safetensors"
            other.write_text("other", encoding="utf-8")
            industrial["adapter_artifacts"]["weights"] = self._identity(other)
            with self.assertRaisesRegex(ManifestError, "同一组 Adapter"):
                gate(manifest, traffic, industrial, path)

    def test_gate_rejects_empty_samples_and_tampered_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, path, traffic, industrial = self._gate_fixture(Path(directory))
            traffic["samples"] = []
            with self.assertRaisesRegex(ManifestError, "samples 长度"):
                gate(manifest, traffic, industrial, path)
            manifest, path, traffic, industrial = self._gate_fixture(Path(directory) / "second")
            traffic["decision_accuracy"] = 0.5
            with self.assertRaisesRegex(ManifestError, "汇总指标"):
                gate(manifest, traffic, industrial, path)

    def test_gate_rejects_constraint_or_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, path, traffic, industrial = self._gate_fixture(Path(directory))
            traffic["decoding_constraint"]["enabled"] = False
            with self.assertRaisesRegex(ManifestError, "采样约束"):
                gate(manifest, traffic, industrial, path)
            manifest, path, traffic, industrial = self._gate_fixture(Path(directory) / "second")
            industrial["precision"]["effective"] = "float16"
            with self.assertRaisesRegex(ManifestError, "CUDA BF16"):
                gate(manifest, traffic, industrial, path)

    def test_gate_rejects_dataset_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, path, traffic, industrial = self._gate_fixture(Path(directory))
            Path(manifest["artifacts"]["train"]["path"]).write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "实际 SHA"):
                gate(manifest, traffic, industrial, path)

    def test_evaluator_and_gate_refuse_to_overwrite_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "exists.json"
            output.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "拒绝覆盖"):
                evaluate_main(
                    [
                        "--base", "unused-base.json",
                        "--snapshot", "unused-snapshot",
                        "--snapshot_manifest", "unused-snapshot.json",
                        "--adapter", "unused-adapter",
                        "--test_jsonl", "unused-test.jsonl",
                        "--test_dataset_id", "unused",
                        "--output", str(output),
                    ]
                )
            with self.assertRaisesRegex(ManifestError, "拒绝覆盖"):
                gate_main(
                    [
                        "--dataset_manifest", "unused-manifest.json",
                        "--traffic_evaluation", "unused-traffic.json",
                        "--industrial_evaluation", "unused-industrial.json",
                        "--output", str(output),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
