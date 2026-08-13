import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import edge_llm_factory.record_u2_bf16_execution_receipt as execution_receipt_module
from edge_llm_factory.contracts import ManifestError, canonical_sha256
from edge_llm_factory.record_u2_bf16_execution_receipt import (
    CANDIDATE_RECEIPT_SCHEMA,
    CANDIDATE_ID,
    GENERAL_CANDIDATE_SCHEMA,
    GENERAL_GATE_SCHEMA,
    GENERAL_TEACHER_SCHEMA,
    SCHEMA_VERSION,
    STEP_RECEIPTS_SCHEMA,
    TRAFFIC_GATE_SCHEMA,
    U2_PROTOCOL_SCHEMA,
    build_execution_receipt,
    write_receipt_exclusive,
)


class U2ExecutionReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.candidate_dir = self.root / "candidate" / "lora"
        self.candidate_dir.mkdir(parents=True)
        self._write_bytes(
            self.candidate_dir / "adapter_model.safetensors", b"stub-u2-weights"
        )
        self._write_json(
            self.candidate_dir / "adapter_config.json", {"r": 8, "lora_alpha": 16}
        )
        self._write_json(
            self.candidate_dir / "train_metrics.json", {"task": "stub-complete-u2"}
        )
        self._create_evidence(traffic_pass=True, math_pass=True, logic_pass=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _identity(path: Path) -> dict:
        payload = path.read_bytes()
        return {
            "path": str(path.resolve()),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    @staticmethod
    def _write_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _candidate_receipt(self) -> dict:
        files = {
            name: self._identity(self.candidate_dir / name)
            for name in (
                "adapter_model.safetensors",
                "adapter_config.json",
                "train_metrics.json",
            )
        }
        canonical = canonical_sha256(
            {
                name: {"bytes": row["bytes"], "sha256": row["sha256"]}
                for name, row in sorted(files.items())
            }
        )
        return {
            "schema_version": CANDIDATE_RECEIPT_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "candidate": {
                "path": str(self.candidate_dir.resolve()),
                "files": files,
                "canonical_artifact_sha256": canonical,
            },
            "anchors": {
                "u2_protocol": {
                    **self._identity(self.protocol),
                    "expected_sha256": self._identity(self.protocol)["sha256"],
                    "verified": True,
                }
            },
            "training_evidence": {"completion": {"confirmed": True}},
        }

    def _protocol_value(self) -> dict:
        implementation_names = (
            "dataset_builder",
            "trainer",
            "candidate_artifact_recorder",
            "traffic_evaluator",
            "traffic_gate",
            "general_candidate_evaluator",
            "general_teacher_evaluator",
            "general_paired_gate",
            "action_constraint",
            "execution_receipt_recorder",
        )
        self.protocol_implementation_files = {}
        for name in implementation_names:
            if name == "execution_receipt_recorder":
                path = Path(execution_receipt_module.__file__).resolve()
            else:
                path = self.root / "protocol_files" / "implementations" / f"{name}.py"
                self._write_bytes(path, f"# stub implementation: {name}\n".encode())
            self.protocol_implementation_files[name] = path

        frozen_names = (
            "base_manifest",
            "text_snapshot_manifest",
            "u1_adapter_weights",
            "u1_adapter_config",
            "dataset_manifest",
            "training_jsonl",
            "training_validation_jsonl",
            "general_promotion_jsonl",
            "traffic_regression_jsonl",
            "production_traffic_adapter_weights",
            "production_traffic_adapter_config",
        )
        text_snapshot = self.root / "protocol_files" / "text_snapshot"
        text_snapshot.mkdir(parents=True, exist_ok=True)
        self.protocol_snapshot_files = {}
        for name, payload in (
            ("config.json", b'{"model_type":"stub"}\n'),
            ("model.safetensors", b"stub text model weights"),
        ):
            path = text_snapshot / name
            self._write_bytes(path, payload)
            self.protocol_snapshot_files[name] = path
        self.protocol_frozen_files = {}
        for name in frozen_names:
            if name == "text_snapshot_manifest":
                path = text_snapshot / "text_snapshot_manifest.json"
                self._write_json(
                    path,
                    {
                        "snapshot_id": "stub-text-snapshot",
                        "files": [
                            {
                                "path": relative,
                                "bytes": self._identity(file_path)["bytes"],
                                "sha256": self._identity(file_path)["sha256"],
                            }
                            for relative, file_path in self.protocol_snapshot_files.items()
                        ],
                    },
                )
            elif name == "production_traffic_adapter_weights":
                path = self.root / "protocol_files" / "incumbent" / "adapter_model.safetensors"
                self._write_bytes(path, b"stub incumbent traffic weights")
            elif name == "production_traffic_adapter_config":
                path = self.root / "protocol_files" / "incumbent" / "adapter_config.json"
                self._write_json(path, {"r": 8})
            else:
                path = self.root / "protocol_files" / "frozen" / f"{name}.bin"
                self._write_bytes(path, f"opaque stub frozen input: {name}\n".encode())
            self.protocol_frozen_files[name] = path
        self.teacher_digest = "e" * 64

        names = [
            "train_U2_once",
            "create_candidate_artifact_receipt_without_scoring",
            "traffic_incumbent_bf16",
            "traffic_candidate_bf16",
            "traffic_paired_gate",
            "general_teacher_fixed_800",
            "general_candidate_bf16_fixed_800",
            "general_paired_gate",
        ]
        commands = [
            ["python", "-m", "stub.train"],
            ["python", "-m", "stub.receipt", "U2_PROTOCOL_SHA256"],
            ["python", "stub_traffic.py", "incumbent"],
            ["python", "stub_traffic.py", "candidate"],
            ["python", "stub_traffic_gate.py", "CANDIDATE_ADAPTER_SHA256"],
            ["python", "-m", "stub.teacher"],
            ["python", "-m", "stub.candidate", "CANDIDATE_ADAPTER_SHA256"],
            ["python", "-m", "stub.general_gate", "CANDIDATE_ADAPTER_SHA256"],
        ]
        return {
            "schema_version": U2_PROTOCOL_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "teacher": {
                "provider": "ollama",
                "model": "qwen3.5:9b",
                "expected_model_digest_sha256": self.teacher_digest,
            },
            "implementations": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": self._identity(path)["sha256"],
                }
                for name, path in self.protocol_implementation_files.items()
            },
            "frozen_inputs": {
                "text_snapshot": {"path": str(text_snapshot.resolve())},
                **{
                    name: (
                        {
                            "path": self._identity(path)["path"],
                            "sha256": self._identity(path)["sha256"],
                        }
                        if name in {"base_manifest", "text_snapshot_manifest"}
                        else self._identity(path)
                    )
                    for name, path in self.protocol_frozen_files.items()
                },
            },
            "exact_execution": {
                "dynamic_argument_bindings": {
                    "U2_PROTOCOL_SHA256": {
                        "source_file": str(self.protocol.resolve()),
                        "literal_placeholder_must_not_be_executed": True,
                    },
                    "CANDIDATE_ADAPTER_SHA256": {
                        "receipt_path": str(self.candidate_receipt.resolve()),
                        "json_pointer": "/candidate/files/adapter_model.safetensors/sha256",
                        "must_match_file": str(
                            (
                                self.candidate_dir / "adapter_model.safetensors"
                            ).resolve()
                        ),
                        "literal_placeholder_must_not_be_executed": True,
                    },
                },
                "steps": [
                    {
                        "step": index,
                        "name": name,
                        "working_directory": str(self.root.resolve()),
                        "argv": command,
                    }
                    for index, (name, command) in enumerate(
                        zip(names, commands), start=1
                    )
                ],
            },
        }

    def _traffic_artifacts(self, *, candidate: bool) -> dict:
        model = (
            self.candidate_dir / "adapter_model.safetensors"
            if candidate
            else self.protocol_frozen_files["production_traffic_adapter_weights"]
        )
        config = (
            self.candidate_dir / "adapter_config.json"
            if candidate
            else self.protocol_frozen_files["production_traffic_adapter_config"]
        )
        return {
            "adapter_model": self._identity(model),
            "adapter_config": self._identity(config),
            "base_text_snapshot_manifest": self._identity(
                self.protocol_frozen_files["text_snapshot_manifest"]
            ),
            "base_text_snapshot_files": {
                name: self._identity(path)
                for name, path in self.protocol_snapshot_files.items()
            },
            "evaluator": self._identity(
                self.protocol_implementation_files["traffic_evaluator"]
            ),
            "verified_unchanged_during_evaluation": True,
        }

    def _traffic_eval(self, *, candidate: bool) -> dict:
        return {
            "task": "phase1_qwen_sft_generation_eval",
            "bf16": True,
            "max_seq_length": 16,
            "max_new_tokens": 1,
            "temperature": 0.0,
            "count": 2400,
            "evaluation_artifacts": self._traffic_artifacts(candidate=candidate),
        }

    def _general_summary(self, *, candidate: bool, samples: Path) -> dict:
        artifacts = {
            "samples": self._identity(samples),
            "evaluator": self._identity(
                self.protocol_implementation_files[
                    "general_candidate_evaluator"
                    if candidate
                    else "general_teacher_evaluator"
                ]
            ),
        }
        summary = {
            "schema_version": (
                GENERAL_CANDIDATE_SCHEMA if candidate else GENERAL_TEACHER_SCHEMA
            ),
            "development_only": True,
            "formal_blind_test_used": False,
            "dataset": {"blind_test_content_loaded": False},
            "artifacts": artifacts,
        }
        if candidate:
            adapter_files = {
                "adapter_weights": self._identity(
                    self.candidate_dir / "adapter_model.safetensors"
                ),
                "adapter_config": self._identity(
                    self.candidate_dir / "adapter_config.json"
                ),
                "train_metrics": self._identity(
                    self.candidate_dir / "train_metrics.json"
                ),
            }
            artifact_sha = canonical_sha256(
                {
                    name: {"bytes": row["bytes"], "sha256": row["sha256"]}
                    for name, row in sorted(adapter_files.items())
                }
            )
            model_payload = {
                "base_manifest_sha256": self._identity(
                    self.protocol_frozen_files["base_manifest"]
                )["sha256"],
                "snapshot_manifest_sha256": self._identity(
                    self.protocol_frozen_files["text_snapshot_manifest"]
                )["sha256"],
                "snapshot_id": "stub-text-snapshot",
                "adapter_artifact_sha256": artifact_sha,
            }
            summary.update(
                {
                    "snapshot_validation": {
                        "status": "valid",
                        "snapshot_id": "stub-text-snapshot",
                        "snapshot": str(
                            (self.root / "protocol_files" / "text_snapshot").resolve()
                        ),
                        "checked_files": list(self.protocol_snapshot_files),
                    },
                    "adapter": {
                        "path": str(self.candidate_dir.resolve()),
                        "artifact_sha256": artifact_sha,
                        "files": adapter_files,
                        "weights_binding": {
                            "expected_sha256": adapter_files["adapter_weights"]["sha256"],
                            "actual_sha256": adapter_files["adapter_weights"]["sha256"],
                            "externally_anchored_and_recomputed": True,
                        },
                    },
                    "model_identity": {
                        **model_payload,
                        "model_sha256": canonical_sha256(model_payload),
                    },
                }
            )
        else:
            attestation = {
                "provider": "ollama",
                "endpoint": "http://127.0.0.1:11434",
                "model": "qwen3.5:9b",
                "expected_model_sha256": self.teacher_digest,
                "model_sha256": self.teacher_digest,
                "show_response_sha256": "f" * 64,
                "version_response_sha256": "1" * 64,
            }
            summary["model_identity"] = {
                **attestation,
                "stable_across_evaluation": True,
                "attestation_before": dict(attestation),
                "attestation_after": dict(attestation),
            }
        return summary

    def _step(
        self, name: str, outputs: dict, *, exit_code: int = 0
    ) -> dict:
        protocol_step = next(
            row
            for row in self.protocol_value["exact_execution"]["steps"]
            if row["name"] == name
        )
        dynamic = {
            "U2_PROTOCOL_SHA256": self._identity(self.protocol)["sha256"],
            "CANDIDATE_ADAPTER_SHA256": self._identity(
                self.candidate_dir / "adapter_model.safetensors"
            )["sha256"],
        }
        return {
            "step": protocol_step["step"],
            "name": name,
            "command_argv": [dynamic.get(token, token) for token in protocol_step["argv"]],
            "working_directory": protocol_step["working_directory"],
            "started_at": "2026-08-11T00:00:00Z",
            "ended_at": "2026-08-11T00:00:01Z",
            "observed_exit_code": exit_code,
            "exit_code_observation": {
                "source": "subprocess.CompletedProcess.returncode",
                "observed": True,
                "inferred": False,
            },
            "outputs": outputs,
        }

    def _create_evidence(
        self, *, traffic_pass: bool, math_pass: bool, logic_pass: bool
    ) -> None:
        self.candidate_receipt = self.root / "candidate_receipt.json"
        self.protocol = self.root / "u2_protocol.json"
        self.protocol_value = self._protocol_value()
        self._write_json(self.protocol, self.protocol_value)
        self._write_json(self.candidate_receipt, self._candidate_receipt())
        self.traffic_incumbent = self.root / "traffic_incumbent.json"
        self.traffic_candidate = self.root / "traffic_candidate.json"
        self._write_json(self.traffic_incumbent, self._traffic_eval(candidate=False))
        self._write_json(self.traffic_candidate, self._traffic_eval(candidate=True))

        checks = {
            "candidate_accuracy_absolute_minimum": {"passed": traffic_pass},
            "candidate_accuracy_not_below_incumbent": {"passed": traffic_pass},
            "candidate_weighted_f1_absolute_minimum": {"passed": traffic_pass},
            "candidate_weighted_f1_not_below_incumbent": {"passed": traffic_pass},
            "candidate_valid_output_rate": {"passed": traffic_pass},
        }
        self.traffic_gate = self.root / "traffic_gate.json"
        self._write_json(
            self.traffic_gate,
            {
                "schema_version": TRAFFIC_GATE_SCHEMA,
                "gate_script_identity": self._identity(
                    self.protocol_implementation_files["traffic_gate"]
                ),
                "incumbent": {
                    "evaluation": self._identity(self.traffic_incumbent),
                    "adapter_model": self._identity(
                        self.protocol_frozen_files[
                            "production_traffic_adapter_weights"
                        ]
                    ),
                    "evaluation_artifacts": self._traffic_artifacts(candidate=False),
                },
                "candidate": {
                    "evaluation": self._identity(self.traffic_candidate),
                    "adapter_model": self._identity(
                        self.candidate_dir / "adapter_model.safetensors"
                    ),
                    "evaluation_artifacts": self._traffic_artifacts(candidate=True),
                },
                "external_sha256_anchors": {
                    "frozen_test": {
                        "expected_sha256": self._identity(
                            self.protocol_frozen_files["traffic_regression_jsonl"]
                        )["sha256"],
                        "actual_sha256": self._identity(
                            self.protocol_frozen_files["traffic_regression_jsonl"]
                        )["sha256"],
                        "matched": True,
                    },
                    "evaluator": {
                        "expected_sha256": self._identity(
                            self.protocol_implementation_files["traffic_evaluator"]
                        )["sha256"],
                        "actual_sha256": self._identity(
                            self.protocol_implementation_files["traffic_evaluator"]
                        )["sha256"],
                        "matched": True,
                    },
                    "gate": {
                        "expected_sha256": self._identity(
                            self.protocol_implementation_files["traffic_gate"]
                        )["sha256"],
                        "actual_sha256": self._identity(
                            self.protocol_implementation_files["traffic_gate"]
                        )["sha256"],
                        "matched": True,
                    },
                    "triple_anchor_complete": True,
                },
                "frozen_test": {
                    **self._identity(
                        self.protocol_frozen_files["traffic_regression_jsonl"]
                    ),
                    "expected_sha256": self._identity(
                        self.protocol_frozen_files["traffic_regression_jsonl"]
                    )["sha256"],
                    "sample_count": 2400,
                    "unique_event_ids": 2400,
                },
                "hard_gate": {"checks": checks, "all_passed": traffic_pass},
            },
        )

        self.general_candidate_samples = self.root / "candidate_samples.jsonl"
        self.general_teacher_samples = self.root / "teacher_samples.jsonl"
        self._write_bytes(self.general_candidate_samples, b'{"stub":"candidate"}\n')
        self._write_bytes(self.general_teacher_samples, b'{"stub":"teacher"}\n')
        self.general_candidate_summary = self.root / "candidate_summary.json"
        self.general_teacher_summary = self.root / "teacher_summary.json"
        self._write_json(
            self.general_candidate_summary,
            self._general_summary(
                candidate=True, samples=self.general_candidate_samples
            ),
        )
        self._write_json(
            self.general_teacher_summary,
            self._general_summary(candidate=False, samples=self.general_teacher_samples),
        )
        candidate_model_sha = json.loads(
            self.general_candidate_summary.read_text(encoding="utf-8")
        )["model_identity"]["model_sha256"]
        self.general_gate = self.root / "general_gate.json"
        self._write_json(
            self.general_gate,
            {
                "schema_version": GENERAL_GATE_SCHEMA,
                "development_only": True,
                "formal_blind_test_used": False,
                "inputs": {
                    "evaluation_implementations": {
                        "candidate": self._identity(
                            self.protocol_implementation_files[
                                "general_candidate_evaluator"
                            ]
                        ),
                        "teacher": self._identity(
                            self.protocol_implementation_files[
                                "general_teacher_evaluator"
                            ]
                        ),
                    },
                    "candidate": {
                        "adapter_model": self._identity(
                            self.candidate_dir / "adapter_model.safetensors"
                        ),
                        "samples": self._identity(self.general_candidate_samples),
                        "summary": self._identity(self.general_candidate_summary),
                    },
                    "teacher": {
                        "samples": self._identity(self.general_teacher_samples),
                        "summary": self._identity(self.general_teacher_summary),
                    },
                    "gate_script": {
                        "path": str(
                            self.protocol_implementation_files[
                                "general_paired_gate"
                            ].resolve()
                        ),
                        "sha256": self._identity(
                            self.protocol_implementation_files[
                                "general_paired_gate"
                            ]
                        )["sha256"],
                    },
                },
                "model_binding": {
                    "candidate_model_sha256": candidate_model_sha,
                    "teacher_model": "qwen3.5:9b",
                    "teacher_model_sha256": self.teacher_digest,
                },
                "categories": {
                    "math": {
                        "checks": {
                            "retention_at_least_0p80": math_pass,
                            "candidate_valid_rate_equals_1": math_pass,
                        },
                        "passed": math_pass,
                    },
                    "natural_language_reasoning": {
                        "checks": {
                            "retention_at_least_0p80": logic_pass,
                            "candidate_valid_rate_equals_1": logic_pass,
                        },
                        "passed": logic_pass,
                    },
                },
                "passed": math_pass and logic_pass,
            },
        )

        gate_exit = 0 if traffic_pass else 2
        general_exit = 0 if math_pass and logic_pass else 2
        candidate_files = {
            "adapter_model": self._identity(
                self.candidate_dir / "adapter_model.safetensors"
            ),
            "adapter_config": self._identity(
                self.candidate_dir / "adapter_config.json"
            ),
            "train_metrics": self._identity(
                self.candidate_dir / "train_metrics.json"
            ),
        }
        steps = [
            self._step("train_U2_once", candidate_files),
            self._step(
                "create_candidate_artifact_receipt_without_scoring",
                {"candidate_receipt": self._identity(self.candidate_receipt)},
            ),
            self._step(
                "traffic_incumbent_bf16",
                {"traffic_incumbent_evaluation": self._identity(self.traffic_incumbent)},
            ),
            self._step(
                "traffic_candidate_bf16",
                {"traffic_candidate_evaluation": self._identity(self.traffic_candidate)},
            ),
            self._step(
                "traffic_paired_gate",
                {"traffic_gate": self._identity(self.traffic_gate)},
                exit_code=gate_exit,
            ),
            self._step(
                "general_teacher_fixed_800",
                {
                    "general_teacher_samples": self._identity(
                        self.general_teacher_samples
                    ),
                    "general_teacher_summary": self._identity(
                        self.general_teacher_summary
                    ),
                },
            ),
            self._step(
                "general_candidate_bf16_fixed_800",
                {
                    "general_candidate_samples": self._identity(
                        self.general_candidate_samples
                    ),
                    "general_candidate_summary": self._identity(
                        self.general_candidate_summary
                    ),
                },
            ),
            self._step(
                "general_paired_gate",
                {"general_gate": self._identity(self.general_gate)},
                exit_code=general_exit,
            ),
        ]
        self.step_receipts = self.root / "step_receipts.json"
        self._write_json(
            self.step_receipts,
            {
                "schema_version": STEP_RECEIPTS_SCHEMA,
                "candidate_id": CANDIDATE_ID,
                "run_id": "u2-stub-run",
                "steps": steps,
            },
        )
        self.step_receipts_sha = self._identity(self.step_receipts)["sha256"]

    def _build(self) -> dict:
        return build_execution_receipt(
            candidate_receipt_path=self.candidate_receipt,
            traffic_incumbent_evaluation_path=self.traffic_incumbent,
            traffic_candidate_evaluation_path=self.traffic_candidate,
            traffic_gate_path=self.traffic_gate,
            general_candidate_samples_path=self.general_candidate_samples,
            general_candidate_summary_path=self.general_candidate_summary,
            general_teacher_samples_path=self.general_teacher_samples,
            general_teacher_summary_path=self.general_teacher_summary,
            general_gate_path=self.general_gate,
            step_receipts_path=self.step_receipts,
            expected_step_receipts_sha256=self.step_receipts_sha,
        )

    def _refresh_step_output(self, step_name: str, role: str, path: Path) -> None:
        receipts = json.loads(self.step_receipts.read_text(encoding="utf-8"))
        for step in receipts["steps"]:
            if step["name"] == step_name:
                step["outputs"][role] = self._identity(path)
                break
        else:  # pragma: no cover - fixture programming error
            self.fail(f"missing step receipt fixture: {step_name}")
        self._write_json(self.step_receipts, receipts)
        self.step_receipts_sha = self._identity(self.step_receipts)["sha256"]

    def test_all_three_valid_gates_pass(self) -> None:
        receipt = self._build()
        self.assertEqual(receipt["schema_version"], SCHEMA_VERSION)
        self.assertEqual(
            receipt["three_hard_gates"],
            {"traffic": True, "math": True, "natural_language_reasoning": True},
        )
        self.assertTrue(receipt["all_three_hard_gates_passed"])
        self.assertIn("eligible_to_freeze", receipt["decision"])
        self.assertFalse(receipt["automatic_merge_quantization_or_release_performed"])
        self.assertEqual(len(receipt["receipt_id"]), 64)

    def test_real_filesystem_protocol_paths_and_current_bindings_pass(self) -> None:
        receipt = self._build()
        anchored = receipt["anchored_u2_protocol"]
        self.assertEqual(anchored["path"], str(self.protocol.resolve()))
        current = anchored["current_file_bindings"]
        self.assertEqual(set(current["implementations"]), set(
            self.protocol_value["implementations"]
        ))
        self.assertTrue(
            current["implementations"]["general_paired_gate"][
                "verified_against_protocol"
            ]
        )
        self.assertEqual(
            current["implementations"]["execution_receipt_recorder"]["path"],
            str(Path(execution_receipt_module.__file__).resolve()),
        )
        self.assertTrue(
            current["implementations"]["execution_receipt_recorder"][
                "verified_against_protocol"
            ]
        )
        self.assertTrue(
            current["frozen_inputs"]["traffic_regression_jsonl"][
                "verified_via_sha_bound_traffic_gate_without_opening_test_content"
            ]
        )

    def test_valid_general_gate_false_with_observed_exit_two_is_stop_loss(self) -> None:
        self._create_evidence(traffic_pass=True, math_pass=False, logic_pass=True)
        receipt = self._build()
        self.assertFalse(receipt["three_hard_gates"]["math"])
        self.assertEqual(
            receipt["gate_exit_interpretation"]["general_paired_gate"],
            "valid_gate_failure_report_present",
        )
        self.assertEqual(
            receipt["decision"], "stop_loss_no_merge_no_quantization_no_release"
        )

    def test_exit_two_does_not_create_failure_when_gate_report_passes(self) -> None:
        log = json.loads(self.step_receipts.read_text(encoding="utf-8"))
        for step in log["steps"]:
            if step["name"] == "traffic_paired_gate":
                step["observed_exit_code"] = 2
        self._write_json(self.step_receipts, log)
        self.step_receipts_sha = self._identity(self.step_receipts)["sha256"]
        with self.assertRaisesRegex(ManifestError, "report passed"):
            self._build()

    def test_invalid_gate_schema_is_tool_error_not_valid_gate_failure(self) -> None:
        gate = json.loads(self.traffic_gate.read_text(encoding="utf-8"))
        gate["schema_version"] = "invalid/tool-output"
        self._write_json(self.traffic_gate, gate)
        log = json.loads(self.step_receipts.read_text(encoding="utf-8"))
        for step in log["steps"]:
            if step["name"] == "traffic_paired_gate":
                step["observed_exit_code"] = 2
                step["outputs"]["traffic_gate"] = self._identity(self.traffic_gate)
        self._write_json(self.step_receipts, log)
        self.step_receipts_sha = self._identity(self.step_receipts)["sha256"]
        with self.assertRaisesRegex(ManifestError, "schema_version mismatch"):
            self._build()

    def test_rejects_unobserved_or_inferred_exit_code(self) -> None:
        log = json.loads(self.step_receipts.read_text(encoding="utf-8"))
        log["steps"][0]["exit_code_observation"]["inferred"] = True
        self._write_json(self.step_receipts, log)
        self.step_receipts_sha = self._identity(self.step_receipts)["sha256"]
        with self.assertRaisesRegex(ManifestError, "explicitly observed"):
            self._build()

    def test_rejects_step_command_not_equal_to_resolved_protocol(self) -> None:
        log = json.loads(self.step_receipts.read_text(encoding="utf-8"))
        log["steps"][1]["command_argv"][-1] = "U2_PROTOCOL_SHA256"
        self._write_json(self.step_receipts, log)
        self.step_receipts_sha = self._identity(self.step_receipts)["sha256"]
        with self.assertRaisesRegex(ManifestError, "command_argv differs"):
            self._build()

    def test_rejects_protocol_drift_after_candidate_receipt(self) -> None:
        protocol = json.loads(self.protocol.read_text(encoding="utf-8"))
        protocol["exact_execution"]["steps"][0]["argv"].append("--drift")
        self._write_json(self.protocol, protocol)
        with self.assertRaisesRegex(ManifestError, "path/bytes/SHA-256"):
            self._build()

    def test_rejects_protocol_implementation_file_drift(self) -> None:
        self.protocol_implementation_files["general_paired_gate"].write_text(
            "# drifted gate implementation\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            ManifestError,
            "u2_protocol.implementations.general_paired_gate SHA-256",
        ):
            self._build()

    def test_rejects_non_test_frozen_input_drift(self) -> None:
        self.protocol_frozen_files["production_traffic_adapter_config"].write_text(
            "drifted incumbent config\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            ManifestError,
            "u2_protocol.frozen_inputs.production_traffic_adapter_config",
        ):
            self._build()

    def test_rejects_snapshot_file_drift_from_protocol_manifest(self) -> None:
        self.protocol_snapshot_files["config.json"].write_text(
            '{"model_type":"drifted"}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(
            ManifestError,
            "current file differs from the anchored manifest",
        ):
            self._build()

    def test_rejects_traffic_candidate_config_not_bound_to_receipt(self) -> None:
        report = json.loads(self.traffic_candidate.read_text(encoding="utf-8"))
        report["evaluation_artifacts"]["adapter_config"]["sha256"] = "0" * 64
        self._write_json(self.traffic_candidate, report)
        self._refresh_step_output(
            "traffic_candidate_bf16",
            "traffic_candidate_evaluation",
            self.traffic_candidate,
        )
        with self.assertRaisesRegex(
            ManifestError,
            "traffic_candidate_evaluation.evaluation_artifacts.adapter_config SHA-256",
        ):
            self._build()

    def test_rejects_traffic_gate_incumbent_config_not_bound_to_protocol(self) -> None:
        gate = json.loads(self.traffic_gate.read_text(encoding="utf-8"))
        gate["incumbent"]["evaluation_artifacts"]["adapter_config"][
            "sha256"
        ] = "0" * 64
        self._write_json(self.traffic_gate, gate)
        self._refresh_step_output(
            "traffic_paired_gate", "traffic_gate", self.traffic_gate
        )
        with self.assertRaisesRegex(
            ManifestError,
            "traffic_gate.incumbent.evaluation_artifacts.adapter_config SHA-256",
        ):
            self._build()

    def test_rejects_general_candidate_train_metrics_not_bound_to_receipt(self) -> None:
        summary = json.loads(
            self.general_candidate_summary.read_text(encoding="utf-8")
        )
        summary["adapter"]["files"]["train_metrics"]["sha256"] = "0" * 64
        self._write_json(self.general_candidate_summary, summary)
        self._refresh_step_output(
            "general_candidate_bf16_fixed_800",
            "general_candidate_summary",
            self.general_candidate_summary,
        )
        with self.assertRaisesRegex(
            ManifestError,
            "general_candidate_summary.adapter.files.train_metrics SHA-256",
        ):
            self._build()

    def test_rejects_general_candidate_base_not_bound_to_protocol(self) -> None:
        summary = json.loads(
            self.general_candidate_summary.read_text(encoding="utf-8")
        )
        summary["model_identity"]["base_manifest_sha256"] = "0" * 64
        self._write_json(self.general_candidate_summary, summary)
        self._refresh_step_output(
            "general_candidate_bf16_fixed_800",
            "general_candidate_summary",
            self.general_candidate_summary,
        )
        with self.assertRaisesRegex(
            ManifestError,
            "general candidate base manifest differs from protocol",
        ):
            self._build()

    def test_rejects_teacher_summary_digest_not_bound_to_protocol(self) -> None:
        summary = json.loads(self.general_teacher_summary.read_text(encoding="utf-8"))
        summary["model_identity"]["model_sha256"] = "0" * 64
        self._write_json(self.general_teacher_summary, summary)
        self._refresh_step_output(
            "general_teacher_fixed_800",
            "general_teacher_summary",
            self.general_teacher_summary,
        )
        with self.assertRaisesRegex(
            ManifestError,
            "general teacher digest differs from protocol",
        ):
            self._build()

    def test_rejects_general_gate_teacher_digest_not_bound_to_protocol(self) -> None:
        gate = json.loads(self.general_gate.read_text(encoding="utf-8"))
        gate["model_binding"]["teacher_model_sha256"] = "0" * 64
        self._write_json(self.general_gate, gate)
        self._refresh_step_output(
            "general_paired_gate", "general_gate", self.general_gate
        )
        with self.assertRaisesRegex(
            ManifestError,
            "general gate teacher digest differs from protocol",
        ):
            self._build()

    def test_rejects_general_gate_script_identity_not_bound_to_protocol(self) -> None:
        gate = json.loads(self.general_gate.read_text(encoding="utf-8"))
        gate["inputs"]["gate_script"]["sha256"] = "0" * 64
        self._write_json(self.general_gate, gate)
        log = json.loads(self.step_receipts.read_text(encoding="utf-8"))
        for step in log["steps"]:
            if step["name"] == "general_paired_gate":
                step["outputs"]["general_gate"] = self._identity(self.general_gate)
        self._write_json(self.step_receipts, log)
        self.step_receipts_sha = self._identity(self.step_receipts)["sha256"]
        with self.assertRaisesRegex(
            ManifestError, "general_gate.inputs.gate_script SHA-256"
        ):
            self._build()

    def test_rejects_traffic_triple_anchor_not_bound_to_protocol(self) -> None:
        gate = json.loads(self.traffic_gate.read_text(encoding="utf-8"))
        gate["external_sha256_anchors"]["evaluator"]["actual_sha256"] = "0" * 64
        self._write_json(self.traffic_gate, gate)
        log = json.loads(self.step_receipts.read_text(encoding="utf-8"))
        for step in log["steps"]:
            if step["name"] == "traffic_paired_gate":
                step["outputs"]["traffic_gate"] = self._identity(self.traffic_gate)
        self._write_json(self.step_receipts, log)
        self.step_receipts_sha = self._identity(self.step_receipts)["sha256"]
        with self.assertRaisesRegex(
            ManifestError,
            "traffic gate evaluator actual SHA-256 differs",
        ):
            self._build()

    def test_rejects_artifact_identity_drift(self) -> None:
        self.general_candidate_samples.write_text("tampered\n", encoding="utf-8")
        with self.assertRaises(ManifestError):
            self._build()

    def test_exclusive_writer_refuses_overwrite(self) -> None:
        receipt = self._build()
        output = self.root / "final_receipt.json"
        write_receipt_exclusive(output, receipt)
        with self.assertRaisesRegex(ManifestError, "refusing to overwrite"):
            write_receipt_exclusive(output, receipt)


if __name__ == "__main__":
    unittest.main()
