import argparse
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_utils_stub = types.ModuleType("torch.utils")
    torch_data_stub = types.ModuleType("torch.utils.data")
    torch_data_stub.Dataset = object
    torch_utils_stub.data = torch_data_stub
    torch_stub.utils = torch_utils_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.utils"] = torch_utils_stub
    sys.modules["torch.utils.data"] = torch_data_stub


SCENE_ROOT = Path(__file__).resolve().parents[1] / "scenes" / "freeway_traffic"
if str(SCENE_ROOT) not in sys.path:
    sys.path.insert(0, str(SCENE_ROOT))

from traffic_system import eval_llm_sft_student as evaluator  # noqa: E402
from traffic_system import summarize_unified_traffic_bf16_regression as gate  # noqa: E402


class TrafficBf16ExternalPathContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_reports_external_relative_adapter_as_canonical_absolute_path(self):
        project_root = self.root / "scene"
        project_root.mkdir()
        external_adapter = self.root / "adapter"
        external_adapter.mkdir()
        with mock.patch.object(evaluator, "PROJECT_ROOT", project_root):
            self.assertEqual(
                evaluator.report_path("../adapter"), str(external_adapter.resolve())
            )
            self.assertEqual(
                evaluator.report_path(str(external_adapter)),
                str(external_adapter.resolve()),
            )

    def test_rejects_existing_output_before_artifact_or_model_work(self):
        output = self.root / "already-present.json"
        output.write_text("do not replace", encoding="utf-8")
        with mock.patch.object(
            evaluator,
            "parse_args",
            return_value=argparse.Namespace(output_json=str(output)),
        ), mock.patch.object(
            evaluator,
            "capture_evaluation_artifacts",
            side_effect=AssertionError("artifact capture must not start"),
        ), mock.patch.object(
            evaluator,
            "load_student",
            side_effect=AssertionError("model loading must not start"),
        ):
            with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
                evaluator.main()
        self.assertEqual(output.read_text(encoding="utf-8"), "do not replace")

    def test_captures_adapter_config_and_verifies_snapshot_files(self):
        adapter = self.root / "adapter"
        snapshot = self.root / "snapshot"
        adapter.mkdir()
        snapshot.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        config = snapshot / "config.json"
        config.write_text('{"model_type":"stub"}', encoding="utf-8")
        manifest = {
            "files": [
                {
                    "path": "config.json",
                    "bytes": config.stat().st_size,
                    "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                }
            ]
        }
        (snapshot / "text_snapshot_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        args = argparse.Namespace(
            adapter_dir=str(adapter), model_name_or_path=str(snapshot)
        )

        artifacts = evaluator.capture_evaluation_artifacts(args)

        self.assertEqual(
            artifacts["adapter_config"]["path"],
            str((adapter / "adapter_config.json").resolve()),
        )
        self.assertEqual(set(artifacts["base_text_snapshot_files"]), {"config.json"})
        config.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "与清单不一致"):
            evaluator.capture_evaluation_artifacts(args)

    def test_gate_resolves_absolute_and_project_relative_external_paths(self):
        project_root = self.root / "scene"
        project_root.mkdir()
        external = self.root / "adapter" / "adapter_model.safetensors"

        self.assertEqual(
            gate._resolve_declared_path(
                "../adapter/adapter_model.safetensors", project_root, "adapter"
            ),
            external.resolve(),
        )
        self.assertEqual(
            gate._resolve_declared_path(str(external), project_root, "adapter"),
            external.resolve(),
        )

    def test_gate_requires_evaluator_and_gate_external_anchors_as_a_pair(self):
        with self.assertRaisesRegex(gate.RegressionGateError, "必须同时提供"):
            gate.build_report(
                incumbent_evaluation=self.root / "incumbent.json",
                candidate_evaluation=self.root / "candidate.json",
                test_jsonl=self.root / "test.jsonl",
                incumbent_adapter_model=self.root / "incumbent.safetensors",
                expected_incumbent_adapter_sha256="0" * 64,
                candidate_adapter_model=self.root / "candidate.safetensors",
                expected_candidate_adapter_sha256="1" * 64,
                project_root=self.root,
                expected_evaluator_sha256="2" * 64,
            )

    def test_trust_remote_code_is_rejected_before_loading(self):
        fake_peft = types.ModuleType("peft")
        fake_transformers = types.ModuleType("transformers")
        fake_peft.PeftModel = object
        fake_transformers.AutoModelForCausalLM = object
        fake_transformers.AutoTokenizer = object
        args = argparse.Namespace(trust_remote_code=True)

        with mock.patch.dict(
            sys.modules,
            {"peft": fake_peft, "transformers": fake_transformers},
        ):
            with self.assertRaisesRegex(ValueError, "禁止 trust_remote_code"):
                evaluator.load_student(args)

    def test_model_and_tokenizer_load_only_from_local_files(self):
        calls = {}

        class FakeTokenizer:
            pad_token_id = 0
            eos_token = "</s>"

        class FakeModel:
            device = "cpu"

            def eval(self):
                calls["eval"] = True
                return self

        tokenizer = FakeTokenizer()
        model = FakeModel()

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls["tokenizer"] = (path, kwargs)
                return tokenizer

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls["model"] = (path, kwargs)
                return model

        class FakePeftModel:
            @staticmethod
            def from_pretrained(base_model, adapter_path):
                calls["adapter"] = (base_model, adapter_path)
                return base_model

        fake_peft = types.ModuleType("peft")
        fake_transformers = types.ModuleType("transformers")
        fake_peft.PeftModel = FakePeftModel
        fake_transformers.AutoModelForCausalLM = FakeAutoModel
        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        args = argparse.Namespace(
            trust_remote_code=False,
            model_name_or_path=str(self.root / "snapshot"),
            adapter_dir=str(self.root / "adapter"),
            bf16=True,
        )

        with mock.patch.dict(
            sys.modules,
            {"peft": fake_peft, "transformers": fake_transformers},
        ), mock.patch.object(
            evaluator, "require_text_only_model", return_value={}
        ):
            loaded_tokenizer, loaded_model = evaluator.load_student(args)

        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertIs(loaded_model, model)
        self.assertTrue(calls["tokenizer"][1]["local_files_only"])
        self.assertFalse(calls["tokenizer"][1]["trust_remote_code"])
        self.assertTrue(calls["model"][1]["local_files_only"])
        self.assertFalse(calls["model"][1]["trust_remote_code"])
        self.assertTrue(calls["eval"])


if __name__ == "__main__":
    unittest.main()
