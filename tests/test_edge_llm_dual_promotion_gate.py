import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from edge_llm_factory.dual_promotion_gate import (
    DualGateError,
    OFFICIAL_GENERAL_CATEGORIES,
    SCHEMA_VERSION,
    evaluate_dual_gate,
)


ARTIFACT_PROVENANCE_FIELDS = {
    "general_evaluation": "general_evaluation_sha256",
    "general_dataset": "general_dataset_sha256",
    "candidate_model": "candidate_model_sha256",
    "traffic_evaluation": "traffic_candidate_evaluation_sha256",
    "runtime_evidence": "runtime_candidate_evidence_sha256",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection_sha256(sample_ids):
    payload = json.dumps(
        sorted(sample_ids), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def integrity():
    return {
        "dataset_sha256_locked_before_training": True,
        "grouped_split_by_source_id": True,
        "final_test_used_for_model_selection": False,
        "train_validation_overlap_count": 0,
        "train_test_overlap_count": 0,
        "validation_test_overlap_count": 0,
        "near_duplicate_train_test_overlap_count": 0,
    }


def categories(code, math, language):
    return {
        "code": {"score": code / 20, "correct": code, "sample_count": 20},
        "math": {"score": math / 30, "correct": math, "sample_count": 30},
        "natural_language_reasoning": {
            "score": language / 30,
            "correct": language,
            "sample_count": 30,
        },
    }


def traffic_metrics():
    return {
        "decision_accuracy": 0.6613,
        "weighted_f1": 0.6668,
        "valid_output_rate": 1.0,
        "selective_cascade_accuracy": 0.609167,
        "selective_gain_over_student": 0.11125,
        "gguf_exact_match_rate": 0.9775,
        "per_class_recall": {
            "A": 0.4866,
            "B": 0.5164,
            "C": 0.5575,
            "D": 0.7314,
            "E": 0.5170,
            "F": 0.8117,
        },
    }


def dataset_rows():
    rows = []
    counts = {"code": 20, "math": 30, "natural_language_reasoning": 30}
    for category in OFFICIAL_GENERAL_CATEGORIES:
        for index in range(counts[category]):
            rows.append(
                {
                    "sample_id": "{}-{:03d}".format(category, index),
                    "category": category,
                    "prompt": "{} prompt {}".format(category, index),
                }
            )
    return rows


def general_artifact(payload, sample_ids):
    return {
        "protocol": copy.deepcopy(payload["general"]["protocol"]),
        "selection": {
            "sample_count": len(sample_ids),
            "sample_ids": list(sample_ids),
            "selection_sha256": selection_sha256(sample_ids),
        },
        "teacher": {
            "model_sha256": payload["provenance"]["teacher_model_sha256"],
            "categories": copy.deepcopy(
                payload["general"]["teacher"]["categories"]
            ),
        },
        "candidate": {
            "model_sha256": payload["provenance"]["candidate_model_sha256"],
            "categories": copy.deepcopy(
                payload["general"]["candidate"]["categories"]
            ),
        },
    }


def traffic_artifact(payload):
    traffic = payload["traffic"]
    return {
        "dataset_sha256": payload["provenance"]["traffic_test_dataset_sha256"],
        "traffic_model_sha256": traffic["artifact_isolation"][
            "candidate_traffic_model_sha256"
        ],
        "incumbent": copy.deepcopy(traffic["incumbent"]),
        "candidate": copy.deepcopy(traffic["candidate"]),
        "paired_accuracy_delta_ci95": copy.deepcopy(
            traffic["paired_accuracy_delta_ci95"]
        ),
        "critical_action_tokens": copy.deepcopy(traffic["critical_action_tokens"]),
        "action_mapping": copy.deepcopy(traffic["action_mapping"]),
        "artifact_isolation": copy.deepcopy(traffic["artifact_isolation"]),
        "protocol": copy.deepcopy(traffic["protocol"]),
    }


def runtime_artifact(payload):
    return {
        "candidate_model_sha256": payload["provenance"]["candidate_model_sha256"],
        "runtime": copy.deepcopy(payload["runtime"]),
    }


def _replace_artifact(payload, name, artifact_payload):
    path = Path(payload["artifact_files"][name]["path"])
    write_json(path, artifact_payload)
    digest = sha256(path)
    payload["artifact_files"][name]["sha256"] = digest
    payload["provenance"][ARTIFACT_PROVENANCE_FIELDS[name]] = digest


def sync_general_artifact(payload):
    rows = [json.loads(line) for line in Path(
        payload["artifact_files"]["general_dataset"]["path"]
    ).read_text(encoding="utf-8").splitlines() if line.strip()]
    _replace_artifact(
        payload,
        "general_evaluation",
        general_artifact(payload, [row["sample_id"] for row in rows]),
    )


def sync_traffic_artifact(payload):
    _replace_artifact(payload, "traffic_evaluation", traffic_artifact(payload))


def sync_runtime_artifact(payload):
    _replace_artifact(payload, "runtime_evidence", runtime_artifact(payload))


def evidence(root):
    root = Path(root)
    candidate_model_path = root / "candidate-q6.gguf"
    candidate_model_path.write_bytes(b"candidate-model-binary-v2")
    candidate_model_sha = sha256(candidate_model_path)

    rows = dataset_rows()
    dataset_path = root / "general-eval.jsonl"
    dataset_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    dataset_sha = sha256(dataset_path)

    traffic_protocol = {
        "context_encoder": "freeway-routing-context-decimal@v2",
        "prompt_format": "raw_task",
        "max_input_tokens": 16,
        "max_output_tokens": 1,
        "thinking": False,
    }
    general_protocol = {
        "dataset_sha256": dataset_sha,
        "backend_family": "ollama",
        "renderer": "qwen3.5@ollama-0.31.1",
        "think": False,
        "prompt_contract": "general-capability-eval-v2",
        "num_ctx": 1024,
        "max_output_tokens": 256,
    }
    traffic_model_sha = "c" * 64
    payload = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "teacher_model_sha256": "1" * 64,
            "incumbent_model_sha256": "2" * 64,
            "candidate_model_sha256": candidate_model_sha,
            "general_dataset_sha256": dataset_sha,
            "traffic_test_dataset_sha256": "5" * 64,
            "runtime_binary_sha256": "6" * 64,
            "general_evaluation_sha256": "0" * 64,
            "traffic_candidate_evaluation_sha256": "0" * 64,
            "runtime_candidate_evidence_sha256": "0" * 64,
            "hardware_id": "jetson-orin-nano-4gb",
        },
        "pre_registration": {
            "focus_categories": ["math", "natural_language_reasoning"],
        },
        "general": {
            "teacher": {"categories": categories(12, 21, 21)},
            # 两个专项都是 17/21=80.95%，但官方三类宏平均仍不足 80%。
            "candidate": {"categories": categories(3, 17, 17)},
            "protocol": {
                "teacher": general_protocol,
                "candidate": dict(general_protocol),
            },
            "data_integrity": integrity(),
        },
        "traffic": {
            "incumbent": traffic_metrics(),
            "candidate": traffic_metrics(),
            "paired_accuracy_delta_ci95": [-0.005, 0.006],
            "critical_action_tokens": ["E", "F"],
            "action_mapping": {
                "incumbent_sha256": "a" * 64,
                "candidate_sha256": "a" * 64,
            },
            "artifact_isolation": {
                "incumbent_traffic_model_sha256": traffic_model_sha,
                "candidate_traffic_model_sha256": traffic_model_sha,
                "general_adapter_separate_profile": True,
                "mixed_adapter_loading_disabled": True,
            },
            "protocol": {
                "incumbent": traffic_protocol,
                "candidate": dict(traffic_protocol),
            },
            "data_integrity": integrity(),
        },
        "runtime": {
            "general_candidate": {
                "teacher_ttft_mean_ms": 400.0,
                "candidate": {
                    "ttft_mean_ms": 84.0,
                    "ttft_p95_ms": 92.0,
                    "peak_rss_mb": 700.0,
                    "artifact_bytes": candidate_model_path.stat().st_size,
                    "steady_rss_growth_500_mb": 0.3,
                    "vm_swap_mb": 0.0,
                },
            },
            "traffic_non_regression": {
                "incumbent": {
                    "ttft_mean_ms": 82.0,
                    "ttft_p95_ms": 90.0,
                    "peak_rss_mb": 680.0,
                    "artifact_bytes": 630_000_000,
                },
                "candidate": {
                    "ttft_mean_ms": 84.0,
                    "ttft_p95_ms": 92.0,
                    "peak_rss_mb": 700.0,
                    "artifact_bytes": 630_000_000,
                    "steady_rss_growth_500_mb": 0.3,
                    "vm_swap_mb": 0.0,
                },
            },
        },
    }

    general_evaluation_path = root / "general-evaluation.json"
    traffic_evaluation_path = root / "traffic-evaluation.json"
    runtime_evidence_path = root / "runtime-evidence.json"
    write_json(
        general_evaluation_path,
        general_artifact(payload, [row["sample_id"] for row in rows]),
    )
    write_json(traffic_evaluation_path, traffic_artifact(payload))
    write_json(runtime_evidence_path, runtime_artifact(payload))

    payload["artifact_files"] = {
        "general_evaluation": {
            "path": str(general_evaluation_path),
            "sha256": sha256(general_evaluation_path),
        },
        "general_dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha,
        },
        "candidate_model": {
            "path": str(candidate_model_path),
            "sha256": candidate_model_sha,
        },
        "traffic_evaluation": {
            "path": str(traffic_evaluation_path),
            "sha256": sha256(traffic_evaluation_path),
        },
        "runtime_evidence": {
            "path": str(runtime_evidence_path),
            "sha256": sha256(runtime_evidence_path),
        },
    }
    for name, provenance_field in ARTIFACT_PROVENANCE_FIELDS.items():
        payload["provenance"][provenance_field] = payload["artifact_files"][name][
            "sha256"
        ]
    return payload


class DualPromotionGateTest(unittest.TestCase):
    def test_focused_profile_passes_without_claiming_official_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_dual_gate(evidence(tmp))
        self.assertTrue(report["focused_profile_allowed"])
        self.assertFalse(report["competition_general_requirement_met"])
        self.assertFalse(report["competition_claim_allowed"])
        self.assertFalse(report["promotion_allowed"])
        self.assertIn("专项通过不等于", report["general"]["official_macro"]["scope_note"])

    def test_official_three_category_requirement_allows_competition_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            payload["general"]["candidate"]["categories"]["code"] = {
                "score": 0.5,
                "correct": 10,
                "sample_count": 20,
            }
            sync_general_artifact(payload)
            report = evaluate_dual_gate(payload)
        self.assertTrue(report["focused_profile_allowed"])
        self.assertTrue(report["competition_general_requirement_met"])
        self.assertTrue(report["competition_claim_allowed"])
        self.assertTrue(report["promotion_allowed"])

    def test_high_categories_cannot_compensate_for_one_failed_official_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            # Math and language are perfect, but code retains only 3/12=25%.
            # Their macro average is high enough; the per-category gate must
            # still reject the competition-wide claim.
            payload["general"]["candidate"]["categories"] = categories(3, 30, 30)
            sync_general_artifact(payload)
            report = evaluate_dual_gate(payload)
        self.assertGreaterEqual(
            report["general"]["official_macro"]["retention"], 0.8
        )
        self.assertFalse(report["general"]["official_category_gate_passed"])
        self.assertFalse(report["competition_general_requirement_met"])
        self.assertFalse(report["competition_claim_allowed"])

    def test_real_artifact_files_are_read_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            report = evaluate_dual_gate(payload)
            files = report["artifact_files"]["files"]
            self.assertEqual(set(files), set(ARTIFACT_PROVENANCE_FIELDS))
            self.assertTrue(all(item["verified"] for item in files.values()))
            self.assertEqual(
                files["candidate_model"]["size_bytes"],
                Path(payload["artifact_files"]["candidate_model"]["path"]).stat().st_size,
            )

    def test_missing_artifact_record_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            del payload["artifact_files"]["runtime_evidence"]
            with self.assertRaisesRegex(DualGateError, "缺少必需证据"):
                evaluate_dual_gate(payload)

    def test_fake_sha_cannot_pass_without_matching_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            payload["artifact_files"]["general_evaluation"]["sha256"] = "f" * 64
            payload["provenance"]["general_evaluation_sha256"] = "f" * 64
            with self.assertRaisesRegex(DualGateError, "SHA-256 不匹配"):
                evaluate_dual_gate(payload)

    def test_tampered_file_is_detected_after_evidence_was_signed(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            path = Path(payload["artifact_files"]["traffic_evaluation"]["path"])
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(DualGateError, "SHA-256 不匹配"):
                evaluate_dual_gate(payload)

    def test_tampered_candidate_model_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            path = Path(payload["artifact_files"]["candidate_model"]["path"])
            path.write_bytes(b"different-model")
            with self.assertRaisesRegex(DualGateError, "SHA-256 不匹配"):
                evaluate_dual_gate(payload)

    def test_general_evaluation_candidate_sha_must_match_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            path = Path(payload["artifact_files"]["general_evaluation"]["path"])
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["candidate"]["model_sha256"] = "e" * 64
            _replace_artifact(payload, "general_evaluation", artifact)
            with self.assertRaisesRegex(DualGateError, "candidate_model_sha256"):
                evaluate_dual_gate(payload)

    def test_general_evaluation_selection_ids_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            path = Path(payload["artifact_files"]["general_evaluation"]["path"])
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["selection"]["sample_ids"][0] = "invented-sample"
            _replace_artifact(payload, "general_evaluation", artifact)
            with self.assertRaisesRegex(DualGateError, "sample_ids"):
                evaluate_dual_gate(payload)

    def test_general_evaluation_selection_hash_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            path = Path(payload["artifact_files"]["general_evaluation"]["path"])
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["selection"].pop("sample_ids")
            artifact["selection"]["selection_sha256"] = "d" * 64
            _replace_artifact(payload, "general_evaluation", artifact)
            with self.assertRaisesRegex(DualGateError, "selection_sha256"):
                evaluate_dual_gate(payload)

    def test_general_evaluation_scores_are_not_self_report_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            path = Path(payload["artifact_files"]["general_evaluation"]["path"])
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["candidate"]["categories"]["math"] = {
                "score": 0.6,
                "correct": 18,
                "sample_count": 30,
            }
            _replace_artifact(payload, "general_evaluation", artifact)
            with self.assertRaisesRegex(DualGateError, "分项分数"):
                evaluate_dual_gate(payload)

    def test_general_protocol_is_verified_inside_evaluation_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            payload["general"]["protocol"]["candidate"]["renderer"] = "other-renderer"
            sync_general_artifact(payload)
            report = evaluate_dual_gate(payload)
        self.assertFalse(report["focused_profile_allowed"])
        self.assertIn("general.protocol.renderer", " ".join(report["reasons"]))

    def test_focus_is_limited_to_one_or_two_official_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            payload["pre_registration"]["focus_categories"] = ["science"]
            with self.assertRaisesRegex(DualGateError, "官方类别"):
                evaluate_dual_gate(payload)

        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            payload["pre_registration"]["focus_categories"] = list(
                OFFICIAL_GENERAL_CATEGORIES
            )
            with self.assertRaisesRegex(DualGateError, "一到两个"):
                evaluate_dual_gate(payload)

    def test_official_category_set_cannot_be_redefined(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            del payload["general"]["candidate"]["categories"]["code"]
            with self.assertRaisesRegex(DualGateError, "官方三类"):
                evaluate_dual_gate(payload)

    def test_focused_category_below_eighty_percent_blocks_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            payload["general"]["candidate"]["categories"]["math"] = {
                "score": 16 / 30,
                "correct": 16,
                "sample_count": 30,
            }
            sync_general_artifact(payload)
            report = evaluate_dual_gate(payload)
        self.assertFalse(report["focused_profile_allowed"])
        self.assertIn("通用分项 math", " ".join(report["reasons"]))

    def test_general_adapter_must_not_replace_or_mix_with_traffic_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            payload["traffic"]["artifact_isolation"][
                "candidate_traffic_model_sha256"
            ] = "d" * 64
            payload["traffic"]["artifact_isolation"][
                "mixed_adapter_loading_disabled"
            ] = False
            sync_traffic_artifact(payload)
            report = evaluate_dual_gate(payload)
        self.assertFalse(report["focused_profile_allowed"])
        reasons = " ".join(report["reasons"])
        self.assertIn("traffic_model_artifact_unchanged", reasons)
        self.assertIn("mixed_adapter_loading_disabled", reasons)

    def test_traffic_regression_remains_an_independent_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            payload["traffic"]["candidate"]["decision_accuracy"] = 0.64
            sync_traffic_artifact(payload)
            report = evaluate_dual_gate(payload)
        self.assertFalse(report["traffic"]["passed"])
        self.assertFalse(report["focused_profile_allowed"])
        self.assertIn("traffic.decision_accuracy", " ".join(report["reasons"]))

    def test_general_candidate_resource_gate_is_separate_from_traffic(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            payload["runtime"]["general_candidate"]["candidate"][
                "peak_rss_mb"
            ] = 1600.0
            sync_runtime_artifact(payload)
            report = evaluate_dual_gate(payload)
        self.assertFalse(report["runtime"]["general_candidate"]["passed"])
        self.assertTrue(report["runtime"]["traffic_non_regression"]["passed"])

    def test_traffic_runtime_non_regression_is_separate_from_general_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            payload["runtime"]["traffic_non_regression"]["candidate"][
                "ttft_mean_ms"
            ] = 120.0
            sync_runtime_artifact(payload)
            report = evaluate_dual_gate(payload)
        self.assertTrue(report["runtime"]["general_candidate"]["passed"])
        self.assertFalse(report["runtime"]["traffic_non_regression"]["passed"])

    def test_general_ttft_is_not_compared_to_traffic_incumbent(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            for side in ("incumbent", "candidate"):
                payload["runtime"]["traffic_non_regression"][side][
                    "ttft_mean_ms"
                ] = 10.0
                payload["runtime"]["traffic_non_regression"][side][
                    "ttft_p95_ms"
                ] = 12.0
            sync_runtime_artifact(payload)
            report = evaluate_dual_gate(payload)
        self.assertTrue(report["runtime"]["general_candidate"]["passed"])
        self.assertTrue(report["runtime"]["traffic_non_regression"]["passed"])
        self.assertTrue(report["focused_profile_allowed"])

    def test_runtime_artifact_size_must_match_real_candidate_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            payload["runtime"]["general_candidate"]["candidate"][
                "artifact_bytes"
            ] += 1
            sync_runtime_artifact(payload)
            with self.assertRaisesRegex(DualGateError, "实际文件大小"):
                evaluate_dual_gate(payload)

    def test_leakage_or_final_set_selection_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            payload["general"]["data_integrity"]["train_test_overlap_count"] = 1
            payload["traffic"]["data_integrity"][
                "final_test_used_for_model_selection"
            ] = True
            report = evaluate_dual_gate(payload)
        self.assertFalse(report["focused_profile_allowed"])
        reasons = " ".join(report["reasons"])
        self.assertIn("train_test_overlap_count", reasons)
        self.assertIn("final_test_used_for_model_selection", reasons)

    def test_thresholds_cannot_weaken_fixed_floors(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            payload["thresholds"] = {"focused_retention": 0.79}
            with self.assertRaises(DualGateError):
                evaluate_dual_gate(payload)

    def test_legacy_v1_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            payload["schema_version"] = "edge-llm-dual-promotion-evidence/v1"
            with self.assertRaisesRegex(DualGateError, "旧格式不能用于发布门禁"):
                evaluate_dual_gate(payload)

    def test_evaluation_does_not_mutate_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = evidence(tmp)
            before = copy.deepcopy(payload)
            evaluate_dual_gate(payload)
            self.assertEqual(payload, before)


if __name__ == "__main__":
    unittest.main()
