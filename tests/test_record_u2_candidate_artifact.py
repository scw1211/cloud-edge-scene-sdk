import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from edge_llm_factory.contracts import ManifestError, canonical_sha256
from edge_llm_factory.record_u2_candidate_artifact import (
    REQUIRED_FALSE_ISOLATION_FIELDS,
    SCHEMA_VERSION,
    _parser,
    build_receipt,
    write_receipt_exclusive,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class U2CandidateArtifactReceiptStubTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.candidate = self.root / "candidate"
        self.u1 = self.root / "u1"
        self.candidate.mkdir()
        self.u1.mkdir()

        (self.candidate / "adapter_model.safetensors").write_bytes(b"u2-weights")
        (self.candidate / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "locked-base"}), encoding="utf-8"
        )
        (self.u1 / "adapter_model.safetensors").write_bytes(b"u1-weights")
        (self.u1 / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        self.dataset = self.root / "dataset-manifest.json"
        self.dataset.write_text('{"schema_version":"dataset-stub"}\n', encoding="utf-8")
        self.script = self.root / "train_u2.py"
        self.script.write_text("# frozen U2 training script\n", encoding="utf-8")
        script_identity = {
            "path": str(self.script.resolve()),
            "bytes": self.script.stat().st_size,
            "sha256": _sha(self.script),
        }
        self.preregistration = self.root / "u2-preregistration.json"
        self._write_preregistration(script_identity["sha256"])
        self.protocol = self.root / "u2-protocol.json"
        self.protocol.write_text('{"protocol":"u2"}\n', encoding="utf-8")
        self.metrics = {
            "task": "single_model_traffic_math_chinese_logic_lora_u2",
            "candidate_id": "unified-v2-U2",
            "training_implementation": {
                "script": script_identity,
                "attestation_before": script_identity,
                "attestation_after": script_identity,
                "stable_across_training": True,
            },
            "dataset": {
                "schema_version": "edge-llm-unified-traffic-general/v2",
                "manifest_path": str(self.dataset.resolve()),
                "manifest_sha256": _sha(self.dataset),
                "promotion_dev_content_read_by_trainer": False,
                "isolation": {
                    field: False for field in REQUIRED_FALSE_ISOLATION_FIELDS
                },
            },
            "source_u1_adapter": {
                "path": str(self.u1.resolve()),
                "weights_sha256": _sha(self.u1 / "adapter_model.safetensors"),
                "config_sha256": _sha(self.u1 / "adapter_config.json"),
            },
            "task_sampling": {
                "seed": 20260811,
                "weights": {
                    "traffic_action": 5,
                    "math": 2,
                    "natural_language_reasoning": 3,
                },
            },
            "optimization": {
                "epochs": 1.0,
                "batch_size": 2,
                "eval_batch_size": 2,
                "gradient_accumulation": 4,
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
                "warmup_ratio": 0.05,
                "logging_steps": 20,
                "max_seq_length": 512,
                "candidate_output_directory": str(self.candidate.resolve()),
                "bf16": True,
                "fp16": False,
                "gradient_checkpointing": False,
                "loss_weights": {
                    "standard_lm_ce": 1.0,
                    "restricted_slot_ce": 1.0,
                    "outside_allowed_mass": 0.25,
                },
            },
            "train_metrics": {
                "train_runtime": 12.5,
                "train_loss": 0.75,
                "epoch": 1.0,
            },
            "adapter_artifact": {
                "path": "adapter_model.safetensors",
                "sha256": _sha(self.candidate / "adapter_model.safetensors"),
            },
            "promotion_status": "unvalidated",
            "production_traffic_release_modified": False,
        }
        self._write_metrics()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_metrics(self) -> None:
        (self.candidate / "train_metrics.json").write_text(
            json.dumps(self.metrics, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _write_preregistration(self, trainer_sha256: str) -> None:
        self.preregistration.write_text(
            json.dumps(
                {
                    "schema_version": "future-compatible-r2-stub",
                    "repository": {
                        "implementation_sha256": {"trainer": trainer_sha256}
                    },
                    "training_recipe": {
                        "precision": "bfloat16",
                        "epochs": 1.0,
                        "batch_size": 2,
                        "eval_batch_size": 2,
                        "gradient_accumulation": 4,
                        "learning_rate": 1e-5,
                        "weight_decay": 0.01,
                        "warmup_ratio": 0.05,
                        "logging_steps": 20,
                        "gradient_checkpointing": False,
                        "loss_weights": {
                            "standard_lm_ce": 1.0,
                            "restricted_slot_ce": 1.0,
                            "outside_allowed_mass": 0.25,
                        },
                        "output_directory": str(self.candidate.resolve()),
                    },
                    "token_and_sampling_contract": {
                        "max_sequence_length": 512,
                        "seed": 20260811,
                        "task_weights": {
                            "traffic_action": 5,
                            "math": 2,
                            "natural_language_reasoning": 3,
                        },
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _build(self):
        return build_receipt(
            candidate_dir=self.candidate,
            u2_preregistration=self.preregistration,
            expected_u2_preregistration_sha256=_sha(self.preregistration),
            u2_protocol=self.protocol,
            expected_u2_protocol_sha256=_sha(self.protocol),
            expected_dataset_manifest_sha256=_sha(self.dataset),
            expected_u1_weights_sha256=_sha(
                self.u1 / "adapter_model.safetensors"
            ),
            expected_u1_config_sha256=_sha(self.u1 / "adapter_config.json"),
        )

    def test_builds_fixed_machine_readable_receipt_without_exit_code_claim(self) -> None:
        receipt = self._build()
        self.assertEqual(receipt["schema_version"], SCHEMA_VERSION)
        self.assertEqual(receipt["candidate_id"], "unified-v2-U2")
        created_at = datetime.fromisoformat(receipt["created_at"].replace("Z", "+00:00"))
        self.assertEqual(created_at.tzinfo, timezone.utc)
        self.assertEqual(
            receipt["anchors"]["u2_preregistration"]["trainer_sha256"],
            _sha(self.script),
        )
        self.assertEqual(
            set(receipt["candidate"]["files"]),
            {
                "adapter_model.safetensors",
                "adapter_config.json",
                "train_metrics.json",
            },
        )
        compact_identities = {
            name: {"bytes": value["bytes"], "sha256": value["sha256"]}
            for name, value in sorted(receipt["candidate"]["files"].items())
        }
        self.assertEqual(
            receipt["candidate"]["canonical_artifact_sha256"],
            canonical_sha256(compact_identities),
        )
        self.assertTrue(receipt["training_evidence"]["completion"]["confirmed"])
        binding = receipt["training_evidence"]["preregistered_recipe_binding"]
        self.assertTrue(binding["exact_match"])
        self.assertEqual(
            binding["training_recipe"]["output_directory"],
            str(self.candidate.resolve()),
        )
        self.assertNotIn("exit_code", json.dumps(receipt, sort_keys=True))
        for identity in receipt["candidate"]["files"].values():
            self.assertEqual(set(identity), {"path", "bytes", "sha256"})

    def test_rejects_incomplete_training_or_isolation_drift(self) -> None:
        self.metrics["train_metrics"]["epoch"] = 0.5
        self._write_metrics()
        with self.assertRaisesRegex(ManifestError, "完成 epoch"):
            self._build()

        self.metrics["train_metrics"]["epoch"] = 1.0
        self.metrics["dataset"]["isolation"]["blind_evaluation_used_for_training"] = True
        self._write_metrics()
        with self.assertRaisesRegex(ManifestError, "blind_evaluation_used_for_training"):
            self._build()

    def test_rejects_preregistered_trainer_sha_drift(self) -> None:
        self._write_preregistration("f" * 64)
        with self.assertRaisesRegex(ManifestError, "冻结预注册 trainer"):
            self._build()

    def test_rejects_every_frozen_recipe_or_sampling_drift(self) -> None:
        clean = copy.deepcopy(self.metrics)
        mutations = {
            "precision": lambda value: value["optimization"].update(bf16=False),
            "epochs": lambda value: value["optimization"].update(epochs=2.0),
            "batch_size": lambda value: value["optimization"].update(batch_size=3),
            "eval_batch_size": lambda value: value["optimization"].update(
                eval_batch_size=3
            ),
            "gradient_accumulation": lambda value: value["optimization"].update(
                gradient_accumulation=5
            ),
            "learning_rate": lambda value: value["optimization"].update(
                learning_rate=2e-5
            ),
            "weight_decay": lambda value: value["optimization"].update(
                weight_decay=0.02
            ),
            "warmup_ratio": lambda value: value["optimization"].update(
                warmup_ratio=0.1
            ),
            "logging_steps": lambda value: value["optimization"].update(
                logging_steps=21
            ),
            "max_seq_length": lambda value: value["optimization"].update(
                max_seq_length=513
            ),
            "gradient_checkpointing": lambda value: value["optimization"].update(
                gradient_checkpointing=True
            ),
            "standard_lm_ce": lambda value: value["optimization"][
                "loss_weights"
            ].update(standard_lm_ce=2.0),
            "restricted_slot_ce": lambda value: value["optimization"][
                "loss_weights"
            ].update(restricted_slot_ce=2.0),
            "outside_allowed_mass": lambda value: value["optimization"][
                "loss_weights"
            ].update(outside_allowed_mass=0.5),
            "seed": lambda value: value["task_sampling"].update(seed=20260812),
            "task_weights": lambda value: value["task_sampling"]["weights"].update(
                math=3
            ),
            "candidate_output_directory": lambda value: value["optimization"].update(
                candidate_output_directory=str(self.root / "wrong-candidate")
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                self.metrics = copy.deepcopy(clean)
                mutate(self.metrics)
                self._write_metrics()
                with self.assertRaises(ManifestError):
                    self._build()

    def test_rejects_anchor_drift_and_existing_receipt(self) -> None:
        with self.assertRaisesRegex(ManifestError, "u2_protocol"):
            build_receipt(
                candidate_dir=self.candidate,
                u2_preregistration=self.preregistration,
                expected_u2_preregistration_sha256=_sha(self.preregistration),
                u2_protocol=self.protocol,
                expected_u2_protocol_sha256="0" * 64,
                expected_dataset_manifest_sha256=_sha(self.dataset),
                expected_u1_weights_sha256=_sha(
                    self.u1 / "adapter_model.safetensors"
                ),
                expected_u1_config_sha256=_sha(self.u1 / "adapter_config.json"),
            )

        output = self.root / "receipt.json"
        receipt = self._build()
        write_receipt_exclusive(output, receipt)
        before = output.read_bytes()
        with self.assertRaisesRegex(ManifestError, "拒绝覆盖"):
            write_receipt_exclusive(output, receipt)
        self.assertEqual(output.read_bytes(), before)

    def test_cli_has_only_explicit_required_inputs(self) -> None:
        required = {
            action.dest
            for action in _parser()._actions
            if action.required
        }
        self.assertEqual(
            required,
            {
                "candidate_dir",
                "u2_preregistration",
                "expected_u2_preregistration_sha256",
                "u2_protocol",
                "expected_u2_protocol_sha256",
                "expected_dataset_manifest_sha256",
                "expected_u1_weights_sha256",
                "expected_u1_config_sha256",
                "output",
            },
        )


if __name__ == "__main__":
    unittest.main()
