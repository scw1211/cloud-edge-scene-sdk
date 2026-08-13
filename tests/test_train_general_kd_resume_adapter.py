import json
import tempfile
import unittest
from pathlib import Path

from edge_llm_factory.contracts import ManifestError
from edge_llm_factory import train_general_kd
from edge_llm_factory.contracts import sha256_file


class TrainGeneralKdResumeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.adapter = Path(self.temp.name)
        (self.adapter / "adapter_model.safetensors").write_bytes(b"weights")
        (self.adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "Qwen/base",
                    "target_modules": ["q_proj", "v_proj"],
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                }
            ),
            encoding="utf-8",
        )
        self.base = {
            "source": {"model_id": "Qwen/base"},
            "lora_policy": {
                "allowed_target_modules": ["q_proj", "v_proj"],
                "max_rank": 64,
            },
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_accepts_exact_same_base_adapter(self):
        weights = self.adapter / "adapter_model.safetensors"
        result = train_general_kd.validate_resume_adapter(
            self.adapter, sha256_file(weights), self.base
        )
        self.assertEqual(result["rank"], 16)
        self.assertEqual(result["initialization"]["mode"], "resume_same_base_lora")

    def test_rejects_wrong_hash_or_base(self):
        with self.assertRaisesRegex(ManifestError, "SHA-256"):
            train_general_kd.validate_resume_adapter(self.adapter, "0" * 64, self.base)
        wrong = dict(self.base)
        wrong["source"] = {"model_id": "Qwen/other"}
        with self.assertRaisesRegex(ManifestError, "文本基座"):
            train_general_kd.validate_resume_adapter(
                self.adapter,
                sha256_file(self.adapter / "adapter_model.safetensors"),
                wrong,
            )


if __name__ == "__main__":
    unittest.main()
