import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from edge_llm_factory.build_ceval_kd_source import (
    SYSTEM_PROMPT,
    build_focused_ceval_source,
    build_focused_ceval_source_from_training_splits,
    main,
)
from edge_llm_factory.build_frozen_nlr_eval import (
    exclusion_identities,
    format_ceval_prompt,
    item_fingerprint,
    prompt_fingerprint,
)
from edge_llm_factory.contracts import ManifestError, sha256_file
from edge_llm_factory.general_kd_data import prepare_source_rows


def _row(index, *, answer="A", question=None):
    return {
        "id": index,
        "question": question or "Question {}?____".format(index),
        "A": "choice A {}".format(index),
        "B": "choice B {}".format(index),
        "C": "choice C {}".format(index),
        "D": "choice D {}".format(index),
        "answer": answer,
    }


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class FocusedCevalKdSourceTests(unittest.TestCase):
    def test_combines_dev_and_val_but_keeps_frozen_validation_out_of_training(self):
        subjects = ("education_science", "marxism")
        dev_rows = {
            subject: [
                _row(index, question="{} dev {}____".format(subject, index))
                for index in range(2)
            ]
            for subject in subjects
        }
        val_rows = {
            subject: [
                _row(index, question="{} val {}____".format(subject, index))
                for index in range(4)
            ]
            for subject in subjects
        }
        frozen_validation = []
        for subject in subjects:
            prompt = format_ceval_prompt(val_rows[subject][0])
            frozen_validation.append(
                {
                    "sample_id": "ceval_{}_val_0".format(subject),
                    "prompt": prompt,
                    "prompt_fingerprint": prompt_fingerprint(prompt),
                    "reference_answer": "A",
                    "source": {"config": subject, "split": "val", "source_id": 0},
                }
            )

        train, validation, report = build_focused_ceval_source_from_training_splits(
            [
                ("dev", dev_rows, {}),
                ("val", val_rows, {}),
            ],
            frozen_validation,
            set(),
            set(),
            validation_per_subject=1,
            seed=23,
        )

        self.assertEqual(len(train), 10)
        self.assertEqual(len(validation), 2)
        self.assertEqual(report["train_rows"], 10)
        self.assertEqual(report["validation_rows"], 2)
        self.assertEqual(
            report["train_subject_counts"],
            {"education_science": 5, "marxism": 5},
        )
        self.assertEqual(
            {row["event_id"] for row in validation},
            {"ceval_education_science_val_0", "ceval_marxism_val_0"},
        )
        self.assertFalse(
            {row["prompt_fingerprint"] for row in train}
            & {row["prompt_fingerprint"] for row in validation}
        )
        self.assertFalse(
            {row["item_fingerprint"] for row in train}
            & {row["item_fingerprint"] for row in validation}
        )

    def test_deterministic_subject_strata_match_general_kd_source_contract(self):
        rows = {
            subject: [
                _row(
                    index,
                    answer="ABCD"[index % 4],
                    question="{} question {}?____".format(subject, index),
                )
                for index in range(6)
            ]
            for subject in ("education_science", "marxism")
        }
        first = build_focused_ceval_source(
            rows,
            set(),
            set(),
            validation_per_subject=2,
            train_per_subject=3,
            seed=19,
        )
        second = build_focused_ceval_source(
            rows,
            set(),
            set(),
            validation_per_subject=2,
            train_per_subject=3,
            seed=19,
        )
        self.assertEqual(first, second)
        train, validation, report = first
        self.assertEqual(
            report["train_subject_counts"],
            {"education_science": 3, "marxism": 3},
        )
        self.assertEqual(
            report["validation_subject_counts"],
            {"education_science": 2, "marxism": 2},
        )
        self.assertEqual(report["unused_rows"], 2)
        self.assertFalse(
            {row["item_fingerprint"] for row in train}
            & {row["item_fingerprint"] for row in validation}
        )
        for row in train + validation:
            self.assertEqual(row["category"], "natural_language_reasoning")
            self.assertEqual(row["prompt_format"], "tokenizer_chat")
            self.assertEqual(row["messages"][0], {"role": "system", "content": SYSTEM_PROMPT})
            self.assertEqual(row["messages"][1]["content"], row["source_prompt"])
            self.assertRegex(row["messages"][-1]["content"], r"^[A-D]$")
            self.assertEqual(
                row["prompt_fingerprint"],
                prompt_fingerprint(row["source_prompt"]),
            )
            self.assertEqual(
                row["item_fingerprint"], item_fingerprint(row["source_prompt"])
            )

        accepted_train, skipped = prepare_source_rows(
            train, [], "train", limit_per_category=0, seed=7
        )
        self.assertEqual(len(accepted_train), len(train))
        self.assertEqual(skipped, {})

    def test_strictly_excludes_frozen_prompt_and_format_independent_item(self):
        rows = {
            "logic": [
                _row(index, question="logic {}?____".format(index))
                for index in range(6)
            ]
        }
        exact = format_ceval_prompt(rows["logic"][0])
        alternate = format_ceval_prompt(rows["logic"][1]).replace("\nA. ", "\nA) ")
        excluded_prompts, excluded_items, _ = exclusion_identities(
            [("frozen", [{"prompt": exact}, {"source_prompt": alternate}])]
        )
        train, validation, report = build_focused_ceval_source(
            rows,
            excluded_prompts,
            excluded_items,
            validation_per_subject=1,
            seed=5,
        )
        selected = train + validation
        self.assertEqual(report["evaluation_excluded_rows"], 2)
        self.assertEqual(report["excluded_prompt_overlap"], 0)
        self.assertEqual(report["excluded_item_overlap"], 0)
        selected_items = {row["item_fingerprint"] for row in selected}
        self.assertNotIn(item_fingerprint(exact), selected_items)
        self.assertNotIn(item_fingerprint(alternate), selected_items)

    def test_deduplicates_same_answer_and_excludes_conflicting_answer_group(self):
        duplicate_one = _row(1, answer="A", question="same answer item____")
        duplicate_two = dict(duplicate_one, id=2)
        conflict_one = _row(3, answer="B", question="conflicting item____")
        conflict_two = dict(conflict_one, id=4, answer="C")
        unique = [
            _row(index, answer="D", question="unique {}____".format(index))
            for index in (5, 6, 7)
        ]
        train, validation, report = build_focused_ceval_source(
            {
                "logic": [
                    duplicate_one,
                    duplicate_two,
                    conflict_one,
                    conflict_two,
                ]
                + unique
            },
            set(),
            set(),
            validation_per_subject=1,
            seed=3,
        )
        self.assertEqual(len(train), 3)
        self.assertEqual(len(validation), 1)
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertEqual(report["conflicting_answer_items"], 1)
        self.assertEqual(report["conflicting_answer_rows"], 2)
        item_ids = [row["item_fingerprint"] for row in train + validation]
        self.assertEqual(len(item_ids), len(set(item_ids)))
        self.assertNotIn(item_fingerprint(format_ceval_prompt(conflict_one)), item_ids)

    def test_rejects_duplicate_event_id_and_invalid_answer(self):
        duplicate_id = [
            _row(1, question="first____"),
            _row(1, question="second____"),
            _row(2, question="third____"),
        ]
        with self.assertRaisesRegex(ManifestError, "event_id 重复"):
            build_focused_ceval_source(
                {"logic": duplicate_id},
                set(),
                set(),
                validation_per_subject=1,
            )
        with self.assertRaisesRegex(ManifestError, "answer 必须是"):
            build_focused_ceval_source(
                {
                    "logic": [
                        _row(1, answer="E"),
                        _row(2),
                    ]
                },
                set(),
                set(),
                validation_per_subject=1,
            )

    def test_requires_train_and_validation_for_every_subject(self):
        with self.assertRaisesRegex(ManifestError, "无法分出"):
            build_focused_ceval_source(
                {"logic": [_row(1), _row(2)]},
                set(),
                set(),
                validation_per_subject=2,
            )

    def test_main_writes_hash_manifest_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            cache = root / "cache"
            cache.mkdir()
            evaluation = root / "frozen.jsonl"
            evaluation.write_text(
                json.dumps({"prompt": "an unrelated frozen prompt"}) + "\n",
                encoding="utf-8",
            )
            rows = {
                subject: [
                    _row(index, question="{} {}____".format(subject, index))
                    for index in range(5)
                ]
                for subject in ("education_science", "marxism")
            }
            source_files = {
                subject: {
                    "subject": subject,
                    "relative_path": "{}/ceval-exam-val.arrow".format(subject),
                    "path": str(cache / subject / "ceval-exam-val.arrow"),
                    "sha256": str(index + 1) * 64,
                    "rows": len(rows[subject]),
                }
                for index, subject in enumerate(sorted(rows))
            }
            output = root / "output"
            argv = [
                "--ceval_cache_root",
                str(cache),
                "--dataset_split",
                "val",
                "--subject",
                "marxism",
                "--subject",
                "education_science",
                "--frozen_evaluation_jsonl",
                str(evaluation),
                "--output_dir",
                str(output),
                "--validation_per_subject",
                "1",
                "--train_per_subject",
                "2",
                "--seed",
                "11",
            ]
            with mock.patch(
                "edge_llm_factory.build_ceval_kd_source.load_ceval_rows",
                return_value=(rows, source_files),
            ), redirect_stdout(io.StringIO()):
                main(argv)

            train = _read_jsonl(output / "train.jsonl")
            validation = _read_jsonl(output / "validation.jsonl")
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(train), 4)
            self.assertEqual(len(validation), 2)
            self.assertEqual(
                manifest["dataset"]["subject_filter"],
                ["education_science", "marxism"],
            )
            self.assertEqual(manifest["evaluation_prompt_overlap"], 0)
            self.assertEqual(manifest["evaluation_item_overlap"], 0)
            self.assertIs(manifest["evaluation_set_used_for_training"], False)
            self.assertEqual(
                manifest["artifacts"]["train"]["sha256"],
                sha256_file(output / "train.jsonl"),
            )
            self.assertEqual(
                manifest["artifacts"]["validation"]["sha256"],
                sha256_file(output / "validation.jsonl"),
            )
            with self.assertRaisesRegex(ManifestError, "拒绝覆盖"):
                main(argv)

    def test_main_rejects_duplicate_subject_and_frozen_evaluation_path(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            cache = root / "cache"
            cache.mkdir()
            frozen = root / "frozen.jsonl"
            frozen.write_text(json.dumps({"prompt": "frozen"}) + "\n")
            base = [
                "--ceval_cache_root",
                str(cache),
                "--subject",
                "logic",
                "--frozen_evaluation_jsonl",
                str(frozen),
                "--output_dir",
                str(root / "first"),
            ]
            with self.assertRaisesRegex(ManifestError, "subject 不能重复声明"):
                main(base + ["--subject", "logic"])

            duplicate_evaluation = [
                "--ceval_cache_root",
                str(cache),
                "--subject",
                "logic",
                "--frozen_evaluation_jsonl",
                str(frozen),
                "--frozen_evaluation_jsonl",
                str(frozen),
                "--output_dir",
                str(root / "second"),
            ]
            with self.assertRaisesRegex(
                ManifestError, "冻结 evaluation 不能重复声明"
            ):
                main(duplicate_evaluation)


if __name__ == "__main__":
    unittest.main()
