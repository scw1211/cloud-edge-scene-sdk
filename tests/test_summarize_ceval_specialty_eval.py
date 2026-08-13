import argparse
import hashlib
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from edge_llm_factory.contracts import ManifestError
from edge_llm_factory import summarize_ceval_specialty_eval as summary
from scenes.freeway_traffic.traffic_system import (
    eval_general_capability_retention as evaluator,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(values):
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class CevalSpecialtySummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "frozen.jsonl"
        self.manifest = self.root / "frozen.manifest.json"
        self.preregistration = self.root / "pre_registration.json"
        self.attestation = self.root / "evaluation_attestation.json"
        self.evaluation = self.root / "evaluation.json"
        self.final_selection = self.root / "final_candidate_selection.json"
        self.final_selection_anchor = self.root / "final_candidate_selection.sha256"
        self.teacher_blob = self.root / "teacher.blob"
        self.teacher_manifest = self.root / "teacher.manifest.json"
        self.candidate_gguf = self.root / "candidate.gguf"
        self.candidate_manifest = self.root / "candidate.manifest.json"
        self.candidate_release = self.root / "candidate.release.json"
        self.subjects = ("subject_a", "subject_b")
        self.rows = self._dataset_rows()
        self._write_model_assets()
        self._write_final_selection()
        self._write_dataset_stack()
        self._write_attestation()
        self._write_evaluation(
            teacher_predictions=("A", "B", "C", "A"),
            candidate_predictions=("A", "A", "D", "A"),
        )

    def tearDown(self):
        self.temp.cleanup()

    def _dataset_rows(self):
        references = ("A", "B", "D", "A")
        rows = []
        for index, reference in enumerate(references):
            subject = self.subjects[index // 2]
            rows.append(
                {
                    "benchmark": "ceval",
                    "category": "natural_language_reasoning",
                    "sample_id": "sample_{}".format(index),
                    "prompt": "question {}\nA. a\nB. b\nC. c\nD. d".format(index),
                    "prompt_fingerprint": hashlib.sha256(
                        "prompt-{}".format(index).encode("utf-8")
                    ).hexdigest(),
                    "reference_answer": reference,
                    "source": {"config": subject, "split": "test"},
                }
            )
        return rows

    def _write_json(self, path, value):
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _protocol(self):
        return {
            "backend": "Ollama",
            "endpoint": "/api/chat",
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "system_prompt": "回答中文单项选择题。只输出 A、B、C 或 D，不要解释。",
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "num_ctx": 1024,
            "num_predict": 4,
            "timeout_seconds": 180,
            "scoring": "fullmatch whitespace plus exactly one A/B/C/D token",
        }

    def _write_model_assets(self):
        self.teacher_blob.write_bytes(b"teacher-model-blob")
        self._write_json(
            self.teacher_manifest,
            {"layers": [{"digest": "sha256:" + _sha256(self.teacher_blob)}]},
        )
        self.candidate_gguf.write_bytes(b"candidate-gguf")
        self._write_json(
            self.candidate_manifest,
            {"layers": [{"digest": "sha256:" + _sha256(self.candidate_gguf)}]},
        )
        self._write_json(
            self.candidate_release,
            {
                "release_id": "candidate-release",
                "deployment_artifact": {"sha256": _sha256(self.candidate_gguf)},
            },
        )

    def _write_final_selection(self):
        self._write_json(
            self.final_selection,
            {
                "selection_id": "synthetic-final-candidate",
                "candidate_model": "candidate-model",
                "candidate_gguf_sha256": _sha256(self.candidate_gguf),
            },
        )
        self.final_selection_anchor.write_text(
            "{}  {}\n".format(
                _sha256(self.final_selection), self.final_selection.name
            ),
            encoding="utf-8",
        )

    def _write_dataset_stack(self):
        with self.dataset.open("w", encoding="utf-8") as file_obj:
            for row in self.rows:
                file_obj.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        dataset_sha = _sha256(self.dataset)
        subject_counts = {
            subject: sum(
                row["source"]["config"] == subject for row in self.rows
            )
            for subject in self.subjects
        }
        per_subject_counts = set(subject_counts.values())
        if len(per_subject_counts) != 1:
            raise AssertionError("synthetic fixture subjects must be balanced")
        samples_per_subject = next(iter(per_subject_counts))
        manifest = {
            "artifact": {
                "path": str(self.dataset),
                "rows": len(self.rows),
                "sha256": dataset_sha,
            },
            "selection": {
                "selected_rows": len(self.rows),
                "selected_subject_counts": subject_counts,
                "selected_sample_ids_sha256": _digest(
                    row["sample_id"] for row in self.rows
                ),
                "selected_prompt_fingerprints_sha256": _digest(
                    row["prompt_fingerprint"] for row in self.rows
                ),
            },
        }
        self._write_json(self.manifest, manifest)
        preregistration = {
            "scope": {"category": "natural_language_reasoning.social_science"},
            "subjects": list(self.subjects),
            "teacher": {
                "model": "teacher-model",
                "manifest_sha256": _sha256(self.teacher_manifest),
                "model_blob_sha256": _sha256(self.teacher_blob),
            },
            "evaluation_protocol": self._protocol(),
            "formal_evaluation": {
                "path": str(self.dataset),
                "sha256": dataset_sha,
                "manifest_path": str(self.manifest),
                "manifest_sha256": _sha256(self.manifest),
                "sample_count": len(self.rows),
                "subject_count": len(self.subjects),
                "samples_per_subject": samples_per_subject,
            },
            "formal_gates": {
                "teacher_minimum_accuracy": 0.7,
                "candidate_minimum_accuracy": 0.7,
                "candidate_to_teacher_retention_minimum": 0.8,
                "teacher_exact_output_rate_required": 1.0,
                "candidate_exact_output_rate_required": 1.0,
                "completed_samples_required_per_model": len(self.rows),
                "execution_errors_allowed": 0,
                "formal_runs_allowed": 1,
                "result_based_retry_allowed": False,
                "runtime_gate_runs_only_after_accuracy_gate_passes": True,
            },
            "formal_procedure": {
                "required_final_selection_record": str(self.final_selection),
                "required_final_selection_external_sha256_anchor": str(
                    self.final_selection_anchor
                ),
                "attestation_must_bind_final_selection_sha256": True,
            },
        }
        self._write_json(self.preregistration, preregistration)

    def _write_attestation(
        self, *, teacher_model="teacher-model", candidate_model="candidate-model"
    ):
        value = {
            "schema_version": "edge-llm-ceval-specialty-evaluation-attestation/v1",
            "pre_registration_sha256": _sha256(self.preregistration),
            "dataset_sha256": _sha256(self.dataset),
            "dataset_manifest_sha256": _sha256(self.manifest),
            "final_candidate_selection_sha256": _sha256(self.final_selection),
            "evaluation_protocol": self._protocol(),
            "evaluator": {
                "path": str(Path(evaluator.__file__).resolve()),
                "sha256": _sha256(Path(evaluator.__file__).resolve()),
            },
            "models": {
                "teacher": {
                    "model": teacher_model,
                    "ollama_runtime_digest": _sha256(self.teacher_manifest),
                    "ollama_manifest": {
                        "path": str(self.teacher_manifest),
                        "sha256": _sha256(self.teacher_manifest),
                    },
                    "model_blob": {
                        "path": str(self.teacher_blob),
                        "sha256": _sha256(self.teacher_blob),
                    },
                },
                "candidate": {
                    "model": candidate_model,
                    "ollama_runtime_digest": _sha256(self.candidate_manifest),
                    "ollama_manifest": {
                        "path": str(self.candidate_manifest),
                        "sha256": _sha256(self.candidate_manifest),
                    },
                    "gguf": {
                        "path": str(self.candidate_gguf),
                        "sha256": _sha256(self.candidate_gguf),
                    },
                    "release_manifest": {
                        "path": str(self.candidate_release),
                        "sha256": _sha256(self.candidate_release),
                    },
                },
            },
        }
        self._write_json(self.attestation, value)

    def _sample(self, row, prediction, *, raw_output=None, execution_error=None):
        output = prediction if raw_output is None else raw_output
        strict_prediction = summary._strict_prediction(output)
        return {
            "sample_id": row["sample_id"],
            "benchmark": row["benchmark"],
            "category": row["category"],
            "correct": strict_prediction == row["reference_answer"],
            "prediction": strict_prediction,
            "reference": row["reference_answer"],
            "execution_error": execution_error,
            "raw_output": output,
        }

    def _write_evaluation(
        self,
        teacher_predictions,
        candidate_predictions,
        *,
        teacher_model="teacher-model",
        candidate_model="candidate-model",
        dataset_jsonl=None,
    ):
        value = {
            "task": "general_capability_retention",
            "dataset_jsonl": dataset_jsonl or str(self.dataset.resolve()),
            "num_ctx": 1024,
            "no_thinking": True,
            "evaluation_protocol": self._protocol(),
            "evaluation_attestation_sha256": _sha256(self.attestation),
            "evaluator_sha256": _sha256(Path(evaluator.__file__).resolve()),
            "dataset_sha256": _sha256(self.dataset),
            "dataset_manifest_sha256": _sha256(self.manifest),
            "final_candidate_selection_path": str(self.final_selection.resolve()),
            "final_candidate_selection_sha256": _sha256(self.final_selection),
            "pre_registration_sha256": _sha256(self.preregistration),
            "sample_count": len(self.rows),
            "models": {
                "teacher": {
                    "model": teacher_model,
                    "runtime_binding": self._runtime_binding(
                        _sha256(self.teacher_manifest)
                    ),
                    "samples": [
                        self._sample(row, prediction)
                        for row, prediction in zip(self.rows, teacher_predictions)
                    ],
                },
                "candidate": {
                    "model": candidate_model,
                    "runtime_binding": self._runtime_binding(
                        _sha256(self.candidate_manifest)
                    ),
                    "samples": [
                        self._sample(row, prediction)
                        for row, prediction in zip(self.rows, candidate_predictions)
                    ],
                },
            },
        }
        self._write_json(self.evaluation, value)

    @staticmethod
    def _runtime_binding(digest):
        return {
            "expected_manifest_sha256": digest,
            "observed_before_sha256": digest,
            "observed_after_sha256": digest,
            "verified": True,
        }

    def _bind_evaluation_to_current_attestation(self):
        value = json.loads(self.evaluation.read_text(encoding="utf-8"))
        value["evaluation_attestation_sha256"] = _sha256(self.attestation)
        self._write_json(self.evaluation, value)

    def _rebind_attestation_to_current_preregistration(self):
        value = json.loads(self.attestation.read_text(encoding="utf-8"))
        value["pre_registration_sha256"] = _sha256(self.preregistration)
        value["evaluation_protocol"] = json.loads(
            self.preregistration.read_text(encoding="utf-8")
        )["evaluation_protocol"]
        self._write_json(self.attestation, value)
        self._bind_evaluation_to_current_attestation()
        evaluation = json.loads(self.evaluation.read_text(encoding="utf-8"))
        evaluation["pre_registration_sha256"] = _sha256(self.preregistration)
        self._write_json(self.evaluation, evaluation)

    def _summarize(
        self,
        iterations=500,
        *,
        expected_preregistration_sha256=None,
        expected_attestation_sha256=None,
    ):
        return summary.summarize(
            evaluation_path=self.evaluation,
            expected_evaluation_sha256=_sha256(self.evaluation),
            dataset_path=self.dataset,
            manifest_path=self.manifest,
            preregistration_path=self.preregistration,
            expected_preregistration_sha256=(
                expected_preregistration_sha256 or _sha256(self.preregistration)
            ),
            final_candidate_selection_path=self.final_selection,
            expected_final_candidate_selection_sha256=_sha256(
                self.final_selection
            ),
            evaluation_attestation_path=self.attestation,
            expected_evaluation_attestation_sha256=(
                expected_attestation_sha256 or _sha256(self.attestation)
            ),
            teacher_label="teacher",
            candidate_label="candidate",
            bootstrap_iterations=iterations,
            bootstrap_seed=17,
        )

    def test_recomputes_paired_subject_metrics_and_gate(self):
        result = self._summarize()
        self.assertEqual(result["models"]["teacher"]["overall"]["correct"], 3)
        self.assertEqual(result["models"]["candidate"]["overall"]["correct"], 3)
        self.assertEqual(
            result["models"]["teacher"]["subject_macro_accuracy"], 0.75
        )
        self.assertEqual(
            result["models"]["candidate"]["subject_macro_accuracy"], 0.75
        )
        pairwise = result["pairwise"]
        self.assertEqual(pairwise["both_correct"], 2)
        self.assertEqual(pairwise["teacher_correct_candidate_wrong"], 1)
        self.assertEqual(pairwise["teacher_wrong_candidate_correct"], 1)
        self.assertEqual(pairwise["both_wrong"], 0)
        self.assertEqual(pairwise["candidate_to_teacher_micro_retention_ratio"], 1.0)
        bootstrap = pairwise["bootstrap"]
        self.assertEqual(bootstrap["iterations"], 500)
        self.assertEqual(
            bootstrap["micro_retention_ratio_95ci"]["valid_iterations"], 500
        )
        self.assertTrue(
            result["pre_registered_gate"]["all_machine_verifiable_gates_passed"]
        )
        self.assertTrue(result["pre_registered_gate"]["runtime_gate_allowed"])
        self.assertEqual(
            result["pre_registered_gate"]["procedural_requirements"][
                "verification_status"
            ],
            "not_verifiable_from_evaluation_result_json",
        )

    def test_ordinary_scoring_remains_legacy_while_formal_is_strict(self):
        math_row = {"category": "math", "reference_answer": "12"}
        choice_row = {
            "category": "natural_language_reasoning",
            "reference_answer": "C",
        }
        self.assertTrue(
            evaluator.evaluate_response("unfinished calculation 12", math_row, 1.0)[
                "correct"
            ]
        )
        self.assertTrue(
            evaluator.evaluate_response("答案是 C", choice_row, 1.0)["correct"]
        )
        self.assertFalse(
            evaluator.evaluate_response(
                "unfinished calculation 12",
                math_row,
                1.0,
                strict_choice_protocol=True,
            )["correct"]
        )
        self.assertFalse(
            evaluator.evaluate_response(
                "答案是 C",
                choice_row,
                1.0,
                strict_choice_protocol=True,
            )["correct"]
        )

    def test_formal_model_load_request_contains_no_scored_prompt(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"{}"
        with mock.patch.object(evaluator.urllib.request, "urlopen", return_value=response) as call:
            evaluator.load_ollama_model("http://127.0.0.1:11434", "candidate", 7)
        request = call.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual(payload, {"model": "candidate", "stream": False, "keep_alive": "30m"})
        self.assertNotIn("prompt", payload)

    def test_bootstrap_is_deterministic_and_sample_paired(self):
        first = self._summarize(iterations=300)["pairwise"]["bootstrap"]
        second = self._summarize(iterations=300)["pairwise"]["bootstrap"]
        self.assertEqual(first, second)
        self.assertEqual(
            first["method"],
            "subject_stratified_sample_paired_nonparametric_bootstrap",
        )
        self.assertEqual(
            first["strata_sample_counts"], {"subject_a": 2, "subject_b": 2}
        )
        self.assertEqual(first["pairing_unit"], "sample_id")

    def test_paired_bootstrap_is_exactly_one_for_identical_sample_outcomes(self):
        predictions = ("A", "B", "C", "A")
        self._write_evaluation(predictions, predictions)
        bootstrap = self._summarize(iterations=500)["pairwise"]["bootstrap"]
        for metric in (
            "micro_retention_ratio_95ci",
            "subject_macro_retention_ratio_95ci",
        ):
            self.assertEqual(bootstrap[metric]["lower"], 1.0)
            self.assertEqual(bootstrap[metric]["upper"], 1.0)
            self.assertEqual(bootstrap[metric]["valid_iterations"], 500)

    def test_rejects_unanchored_pre_registration(self):
        with self.assertRaisesRegex(ManifestError, "外部锚点"):
            self._summarize(expected_preregistration_sha256="0" * 64)

    def test_rejects_unanchored_evaluation_attestation(self):
        with self.assertRaisesRegex(ManifestError, "评测取证清单.*外部锚点"):
            self._summarize(expected_attestation_sha256="0" * 64)

    def test_rejects_final_selection_tamper_against_external_anchor(self):
        self.final_selection.write_text(
            self.final_selection.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ManifestError, "外部锚"):
            self._summarize()

    def test_rejects_attestation_not_binding_final_selection(self):
        value = json.loads(self.attestation.read_text(encoding="utf-8"))
        value["final_candidate_selection_sha256"] = "f" * 64
        self._write_json(self.attestation, value)
        self._bind_evaluation_to_current_attestation()
        with self.assertRaisesRegex(ManifestError, "未绑定最终候选选择"):
            self._summarize()

    def test_rejects_candidate_using_teacher_model_name(self):
        self._write_attestation(candidate_model="teacher-model")
        self._write_evaluation(
            ("A", "B", "C", "A"),
            ("A", "A", "D", "A"),
            candidate_model="teacher-model",
        )
        with self.assertRaisesRegex(ManifestError, "同一模型名"):
            self._summarize()

    def test_rejects_teacher_or_candidate_asset_tamper(self):
        self.teacher_blob.write_bytes(b"tampered-teacher")
        with self.assertRaisesRegex(ManifestError, "实际 SHA-256"):
            self._summarize()

        self._write_model_assets()
        self._write_attestation()
        self._write_evaluation(
            ("A", "B", "C", "A"), ("A", "A", "D", "A")
        )
        self.candidate_gguf.write_bytes(b"tampered-candidate")
        with self.assertRaisesRegex(ManifestError, "实际 SHA-256"):
            self._summarize()

    def test_rejects_release_manifest_not_bound_to_candidate_gguf(self):
        self._write_json(
            self.candidate_release,
            {"deployment_artifact": {"sha256": "f" * 64}},
        )
        self._write_attestation()
        self._bind_evaluation_to_current_attestation()
        with self.assertRaisesRegex(ManifestError, "未引用.*GGUF"):
            self._summarize()

    def test_rejects_candidate_asset_identical_to_teacher_blob(self):
        self.candidate_gguf.write_bytes(self.teacher_blob.read_bytes())
        self._write_json(
            self.candidate_manifest,
            {"layers": [{"digest": "sha256:" + _sha256(self.candidate_gguf)}]},
        )
        self._write_json(
            self.candidate_release,
            {"deployment_artifact": {"sha256": _sha256(self.candidate_gguf)}},
        )
        self._write_attestation()
        self._bind_evaluation_to_current_attestation()
        with self.assertRaisesRegex(ManifestError, "不能与 Teacher"):
            self._summarize()

    def test_rejects_runtime_digest_not_bound_to_attested_manifest(self):
        value = json.loads(self.evaluation.read_text(encoding="utf-8"))
        value["models"]["candidate"]["runtime_binding"][
            "observed_after_sha256"
        ] = "f" * 64
        self._write_json(self.evaluation, value)
        with self.assertRaisesRegex(ManifestError, "实际运行模型"):
            self._summarize()

    def test_evaluator_formal_contract_feeds_summarizer(self):
        args = argparse.Namespace(
            dataset_jsonl=str(self.dataset),
            num_ctx=1024,
            timeout=180,
            resume=False,
            evaluation_attestation=str(self.attestation),
            expected_evaluation_attestation_sha256=_sha256(self.attestation),
            pre_registration=str(self.preregistration),
            expected_preregistration_sha256=_sha256(self.preregistration),
            dataset_manifest=str(self.manifest),
            final_candidate_selection=str(self.final_selection),
            expected_final_candidate_selection_sha256=_sha256(
                self.final_selection
            ),
        )
        formal = evaluator.load_formal_attestation(
            args,
            self.rows,
            [("teacher", "teacher-model"), ("candidate", "candidate-model")],
        )
        produced = evaluator.empty_result(args, self.rows, formal)
        existing = json.loads(self.evaluation.read_text(encoding="utf-8"))
        produced["models"] = existing["models"]
        self._write_json(self.evaluation, produced)
        result = self._summarize(iterations=100)
        self.assertTrue(result["validation"]["evaluator_source_hash_bound"])
        self.assertTrue(
            result["validation"]["candidate_runtime_model_bound_to_manifest"]
        )

    def test_rejects_any_full_protocol_mismatch(self):
        value = json.loads(self.evaluation.read_text(encoding="utf-8"))
        value["evaluation_protocol"]["seed"] = 43
        self._write_json(self.evaluation, value)
        with self.assertRaisesRegex(ManifestError, "完整协议"):
            self._summarize()

    def test_relative_dataset_reference_is_compatible(self):
        self._write_evaluation(
            ("A", "B", "C", "A"),
            ("A", "A", "D", "A"),
            dataset_jsonl=self.dataset.name,
        )
        self.assertTrue(
            self._summarize()["validation"][
                "evaluation_dataset_reference_matches"
            ]
        )

    def test_accepts_only_registered_specialty_scope_whitelist(self):
        self.subjects = summary.POLITICS_CIVICS_SUBJECTS
        self.rows = []
        for subject in self.subjects:
            for index in range(summary.POLITICS_CIVICS_SAMPLES_PER_SUBJECT):
                sample_id = "ceval_{}_test_{}".format(subject, index)
                self.rows.append(
                    {
                        "benchmark": "ceval",
                        "category": "natural_language_reasoning",
                        "sample_id": sample_id,
                        "prompt": "{} question {}\nA. a\nB. b\nC. c\nD. d".format(
                            subject, index
                        ),
                        "prompt_fingerprint": hashlib.sha256(
                            sample_id.encode("utf-8")
                        ).hexdigest(),
                        "reference_answer": "ABCD"[index % 4],
                        "source": {"config": subject, "split": "test"},
                    }
                )
        self._write_dataset_stack()
        preregistration = json.loads(
            self.preregistration.read_text(encoding="utf-8")
        )
        preregistration["scope"][
            "category"
        ] = "natural_language_reasoning.politics_civics"
        self._write_json(self.preregistration, preregistration)
        self._write_attestation()
        predictions = tuple(row["reference_answer"] for row in self.rows)
        self._write_evaluation(predictions, predictions)
        result = self._summarize(iterations=100)
        self.assertEqual(
            result["scope_category"],
            "natural_language_reasoning.politics_civics",
        )

        preregistration["scope"]["category"] = "natural_language_reasoning.other"
        self._write_json(self.preregistration, preregistration)
        self._rebind_attestation_to_current_preregistration()
        with self.assertRaisesRegex(ManifestError, "允许的 C-Eval 专项"):
            self._summarize(iterations=100)

    def test_politics_scope_rejects_category_only_relabel(self):
        preregistration = json.loads(
            self.preregistration.read_text(encoding="utf-8")
        )
        preregistration["scope"][
            "category"
        ] = "natural_language_reasoning.politics_civics"
        self._write_json(self.preregistration, preregistration)
        self._rebind_attestation_to_current_preregistration()
        with self.assertRaisesRegex(ManifestError, "politics_civics"):
            self._summarize(iterations=100)

    def test_execution_error_fails_zero_error_gate(self):
        value = json.loads(self.evaluation.read_text(encoding="utf-8"))
        value["models"]["candidate"]["samples"][0]["execution_error"] = "timeout"
        self._write_json(self.evaluation, value)
        result = self._summarize()
        self.assertEqual(
            result["models"]["candidate"]["overall"]["execution_error_count"],
            1,
        )
        self.assertFalse(
            result["pre_registered_gate"]["all_machine_verifiable_gates_passed"]
        )

    def test_gate_threshold_boundary_is_inclusive(self):
        preregistration = json.loads(
            self.preregistration.read_text(encoding="utf-8")
        )
        gates = preregistration["formal_gates"]
        gates["teacher_minimum_accuracy"] = 0.75
        gates["candidate_minimum_accuracy"] = 0.75
        gates["candidate_to_teacher_retention_minimum"] = 1.0
        self._write_json(self.preregistration, preregistration)
        self._rebind_attestation_to_current_preregistration()
        self.assertTrue(
            self._summarize()["pre_registered_gate"][
                "all_machine_verifiable_gates_passed"
            ]
        )

        preregistration["formal_gates"]["candidate_minimum_accuracy"] = 0.750001
        self._write_json(self.preregistration, preregistration)
        self._rebind_attestation_to_current_preregistration()
        self.assertFalse(
            self._summarize()["pre_registered_gate"][
                "all_machine_verifiable_gates_passed"
            ]
        )

    def test_rejects_dataset_hash_mismatch(self):
        self.dataset.write_text(
            self.dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ManifestError, "题集 SHA-256 与预注册"):
            self._summarize()

    def test_rejects_manifest_hash_mismatch(self):
        value = json.loads(self.manifest.read_text(encoding="utf-8"))
        value["extra"] = "changed"
        self._write_json(self.manifest, value)
        with self.assertRaisesRegex(ManifestError, "manifest SHA-256"):
            self._summarize()

    def test_rejects_missing_or_duplicate_model_samples(self):
        value = json.loads(self.evaluation.read_text(encoding="utf-8"))
        value["models"]["candidate"]["samples"].pop()
        self._write_json(self.evaluation, value)
        with self.assertRaisesRegex(ManifestError, "未完成冻结题集全集"):
            self._summarize()

        self._write_evaluation(
            teacher_predictions=("A", "B", "C", "A"),
            candidate_predictions=("A", "A", "D", "A"),
        )
        value = json.loads(self.evaluation.read_text(encoding="utf-8"))
        value["models"]["candidate"]["samples"][1]["sample_id"] = "sample_0"
        self._write_json(self.evaluation, value)
        with self.assertRaisesRegex(ManifestError, "sample_id 重复"):
            self._summarize()

    def test_strict_invalid_output_is_counted_and_fails_gate(self):
        value = json.loads(self.evaluation.read_text(encoding="utf-8"))
        sample = value["models"]["candidate"]["samples"][0]
        sample.update(
            {
                "raw_output": "答案是 A",
                "prediction": None,
                "correct": False,
            }
        )
        self._write_json(self.evaluation, value)
        result = self._summarize()
        candidate = result["models"]["candidate"]["overall"]
        self.assertEqual(candidate["strict_output_count"], 3)
        self.assertEqual(candidate["strict_output_rate"], 0.75)
        self.assertFalse(
            result["pre_registered_gate"]["all_machine_verifiable_gates_passed"]
        )

    def test_rejects_stored_correctness_that_disagrees_with_raw_output(self):
        value = json.loads(self.evaluation.read_text(encoding="utf-8"))
        value["models"]["candidate"]["samples"][0]["correct"] = False
        self._write_json(self.evaluation, value)
        with self.assertRaisesRegex(ManifestError, "correct 与严格重算不一致"):
            self._summarize()

    def test_cli_writes_new_summary_and_refuses_overwrite(self):
        output = self.root / "summary.json"
        args = [
            "--evaluation_json",
            str(self.evaluation),
            "--expected_evaluation_sha256",
            _sha256(self.evaluation),
            "--dataset_jsonl",
            str(self.dataset),
            "--dataset_manifest",
            str(self.manifest),
            "--pre_registration",
            str(self.preregistration),
            "--expected_preregistration_sha256",
            _sha256(self.preregistration),
            "--final_candidate_selection",
            str(self.final_selection),
            "--expected_final_candidate_selection_sha256",
            _sha256(self.final_selection),
            "--evaluation_attestation",
            str(self.attestation),
            "--expected_evaluation_attestation_sha256",
            _sha256(self.attestation),
            "--teacher_label",
            "teacher",
            "--candidate_label",
            "candidate",
            "--output_json",
            str(output),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            summary.main(args)
        saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(saved["model_inference_performed"])
        with self.assertRaisesRegex(ManifestError, "拒绝覆盖"):
            with contextlib.redirect_stdout(io.StringIO()):
                summary.main(args)


if __name__ == "__main__":
    unittest.main()
