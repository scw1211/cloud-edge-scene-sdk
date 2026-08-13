import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from edge_llm_factory import build_unified_traffic_general_dataset_v2 as builder
from edge_llm_factory.contracts import ManifestError


def _logic_row(index: int, answer: int = 0):
    return {
        "example_id": index,
        "answer": answer,
        "text": f"材料 {index}",
        "question": f"问题 {index}",
        "options": [f"A.甲{index}", f"B.乙{index}", f"C.丙{index}", f"D.丁{index}"],
    }


class U2UnifiedDatasetBuilderTests(unittest.TestCase):
    def test_traffic_splits_must_be_distinct_before_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp) / "same.jsonl"
            shared.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "三个不同文件"):
                builder._validate_traffic_input_identity(
                    {
                        "traffic_train": shared,
                        "traffic_validation": shared,
                        "traffic_test": shared,
                    }
                )

    def test_manifest_isolation_declaration_matches_u2_trainer_contract(self) -> None:
        declaration = builder._isolation_declaration()
        self.assertEqual(
            declaration,
            {field: False for field in builder.REQUIRED_FALSE_ISOLATION_FIELDS},
        )
        self.assertEqual(
            set(declaration),
            {
                "traffic_test_used_for_training",
                "gsm8k_test_loaded",
                "logiqa_test_loaded",
                "formal_evaluation_used_for_training",
                "u1_evaluation_outputs_used_for_training",
                "blind_evaluation_used_for_training",
                "promotion_dev_used_for_training",
                "promotion_dev_used_for_model_selection",
            },
        )

    def test_option_prefix_cleanup_handles_official_and_full_width_forms(self) -> None:
        cases = {
            "A.答案": "答案",
            "B．答案": "答案",
            "C、答案": "答案",
            "D) 答案": "答案",
            "（Ａ）答案": "答案",
            "(b) 答案": "答案",
            "没有前缀": "没有前缀",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                actual, _ = builder._strip_option_prefix(raw)
                self.assertEqual(actual, expected)
        self.assertEqual(builder._strip_option_prefix("A."), ("A.", False))

    def test_deterministic_option_permutation_relabels_the_same_answer(self) -> None:
        options = ["A.甲", "B.乙", "C.丙", "D.丁"]
        first = builder._deterministic_option_permutation(
            options, answer_index=2, fingerprint="f" * 64, seed=20260811
        )
        second = builder._deterministic_option_permutation(
            options, answer_index=2, fingerprint="f" * 64, seed=20260811
        )
        self.assertEqual(first, second)
        permuted, new_answer, order, stripped_count = first
        self.assertEqual(stripped_count, 4)
        self.assertEqual(set(order), {0, 1, 2, 3})
        self.assertEqual(permuted[new_answer], "丙")

    def test_spent_u1_dev_is_training_validation_not_promotion(self) -> None:
        rows = [
            {
                "event_id": "math-old-dev",
                "category": "math",
                "prompt_fingerprint": "m" * 64,
            },
            {
                "event_id": "logic-old-dev",
                "category": "natural_language_reasoning",
                "prompt_fingerprint": "l" * 64,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "val.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            loaded = builder._load_spent_u1_general_validation(
                Path(tmp),
                {
                    "math": {"m" * 64},
                    "natural_language_reasoning": {"l" * 64},
                },
            )
        self.assertEqual(len(loaded), 2)
        self.assertTrue(
            all(
                row["u2_role"]
                == "spent_u1_development_for_training_validation"
                for row in loaded
            )
        )

    def test_math_inventory_uses_old_train_plus_unused_minus_new_dev(self) -> None:
        official = {}
        for index in range(10):
            fingerprint = hashlib.sha256(f"math-{index}".encode()).hexdigest()
            official[fingerprint] = {
                "prompt_fingerprint": fingerprint,
                "source_index": index,
            }
        keys = list(official)
        old_train = set(keys[:4])
        old_dev = set(keys[4:6])
        train, dev, report = builder._select_math_inventory(
            official,
            old_train,
            old_dev,
            seed=11,
            expected_u1_train=4,
            expected_u1_dev=2,
            expected_unused=4,
            new_dev_count=1,
        )
        train_fingerprints = {row["prompt_fingerprint"] for row in train}
        dev_fingerprints = {row["prompt_fingerprint"] for row in dev}
        self.assertEqual(len(train), 7)
        self.assertEqual(len(dev), 1)
        self.assertTrue(old_train <= train_fingerprints)
        self.assertFalse(old_dev & (train_fingerprints | dev_fingerprints))
        self.assertFalse(train_fingerprints & dev_fingerprints)
        self.assertEqual(report["previously_unused_rows_added_to_train"], 3)

    def test_logic_inventory_excludes_all_u1_dev_fingerprints(self) -> None:
        clean_train = {}
        for index in range(8):
            row = _logic_row(index, answer=index % 4)
            clean_train[builder._logiqa_content_fingerprint(row)] = row
        clean_dev = {}
        for index in range(500, 901):
            row = _logic_row(index, answer=index % 4)
            clean_dev[builder._logiqa_content_fingerprint(row)] = row
        dev_prompt_fingerprints = {
            builder._sha_text(f"{builder.LOGIC_CATEGORY}\n{fingerprint}")
            for fingerprint in clean_dev
        }
        old_dev = set(sorted(dev_prompt_fingerprints)[: builder.LOGIC_U1_DEV_ROWS])
        train, dev, report = builder._select_logic_inventory(
            clean_train, clean_dev, old_dev, seed=23, new_dev_count=1
        )
        train_fingerprints = {row["prompt_fingerprint"] for row in train}
        new_dev_fingerprints = {row["prompt_fingerprint"] for row in dev}
        self.assertEqual(len(train), 8)
        self.assertEqual(len(dev), 1)
        self.assertFalse(old_dev & (train_fingerprints | new_dev_fingerprints))
        self.assertFalse(train_fingerprints & new_dev_fingerprints)
        self.assertEqual(report["u1_dev_rows_excluded"], 400)

    def test_test_file_attestation_does_not_call_any_content_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            test_path = Path(tmp) / "blind-test.jsonl"
            test_path.write_bytes(b"blind bytes are only hashed")
            expected_sha = hashlib.sha256(test_path.read_bytes()).hexdigest()
            with patch.object(
                builder,
                "_read_jsonl",
                side_effect=AssertionError("test content parser must not be called"),
            ), patch.object(
                builder,
                "_traffic_rows",
                side_effect=AssertionError("traffic parser must not be called"),
            ):
                attestation = builder._file_attestation(test_path, expected_sha)
            self.assertEqual(attestation["sha256"], expected_sha)
            self.assertEqual(attestation["bytes"], test_path.stat().st_size)
            self.assertFalse(attestation["content_loaded"])
            self.assertEqual(attestation["access_mode"], "sha256_and_stat_only")


if __name__ == "__main__":
    unittest.main()
