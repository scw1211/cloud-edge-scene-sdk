import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from edge_llm_factory.build_frozen_nlr_eval import (
    build_nlr_evaluation,
    discover_split_arrow_files,
    discover_val_arrow_files,
    exclusion_identities,
    format_ceval_prompt,
    item_fingerprint,
    main,
    prompt_fingerprint,
)
from edge_llm_factory.contracts import ManifestError, sha256_file


def _row(index, *, answer="A", question=None):
    return {
        "id": index,
        "question": question or f"Question {index}?____",
        "A": f"choice A {index}",
        "B": f"choice B {index}",
        "C": f"choice C {index}",
        "D": f"choice D {index}",
        "answer": answer,
    }


class FrozenNlrEvaluationTests(unittest.TestCase):
    def test_prompt_matches_existing_ceval_evaluator_shape(self):
        rendered = format_ceval_prompt(_row(3))
        self.assertEqual(
            rendered,
            "Question 3?____\n"
            "A. choice A 3\n"
            "B. choice B 3\n"
            "C. choice C 3\n"
            "D. choice D 3",
        )
        self.assertEqual(prompt_fingerprint(rendered), prompt_fingerprint(rendered))
        self.assertEqual(item_fingerprint(rendered), item_fingerprint(rendered))

    def test_exclusion_detects_same_question_with_different_choice_markers(self):
        rendered = format_ceval_prompt(_row(4))
        alternate = rendered.replace("\nA. ", "\nA) ")
        prompts, questions, counts = exclusion_identities(
            [("train", [{"source_prompt": alternate}])]
        )
        self.assertIn(prompt_fingerprint(alternate), prompts)
        self.assertIn(item_fingerprint(rendered), questions)
        self.assertEqual(
            counts,
            {
                "train": {
                    "rows": 1,
                    "new_unique_prompts": 1,
                    "new_unique_items": 1,
                }
            },
        )

    def test_selection_is_deterministic_balanced_and_has_zero_overlap(self):
        subject_rows = {
            "alpha": [_row(index, question=f"Alpha {index}?____") for index in range(5)],
            "beta": [_row(index, question=f"Beta {index}?____") for index in range(5)],
            "gamma": [_row(index, question=f"Gamma {index}?____") for index in range(5)],
        }
        excluded_prompt = format_ceval_prompt(subject_rows["alpha"][0])
        prompts, questions, _ = exclusion_identities(
            [("old_eval", [{"prompt": excluded_prompt}])]
        )
        first, first_report = build_nlr_evaluation(
            subject_rows, prompts, questions, sample_count=6, seed=17
        )
        second, second_report = build_nlr_evaluation(
            subject_rows, prompts, questions, sample_count=6, seed=17
        )
        self.assertEqual(first, second)
        self.assertEqual(first_report, second_report)
        self.assertEqual(
            first_report["selected_subject_counts"],
            {"alpha": 2, "beta": 2, "gamma": 2},
        )
        self.assertEqual(first_report["excluded_prompt_overlap"], 0)
        self.assertEqual(first_report["excluded_item_overlap"], 0)
        self.assertEqual(first_report["excluded_rows"], 1)
        for row in first:
            self.assertEqual(row["benchmark"], "ceval")
            self.assertEqual(row["category"], "natural_language_reasoning")
            self.assertIn(row["reference_answer"], {"A", "B", "C", "D"})
            self.assertIn("source", row)
            self.assertNotEqual(
                item_fingerprint(row["prompt"]),
                item_fingerprint(excluded_prompt),
            )

    def test_same_stem_with_different_choices_remains_two_distinct_items(self):
        first = _row(1, question="下列说法正确的是____")
        second = _row(2, question="下列说法正确的是____")
        selected, report = build_nlr_evaluation(
            {"logic": [first], "college_chemistry": [second]},
            set(),
            set(),
            sample_count=2,
            seed=3,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(report["duplicate_rows"], 0)
        self.assertNotEqual(
            item_fingerprint(selected[0]["prompt"]),
            item_fingerprint(selected[1]["prompt"]),
        )

    def test_conflicting_answer_for_identical_item_excludes_entire_group(self):
        first = _row(1, answer="A", question="Duplicated item?____")
        second = dict(first)
        second["answer"] = "B"
        remaining = _row(2, answer="C", question="Unique item?____")
        selected, report = build_nlr_evaluation(
            {"logic": [first, remaining], "civil_servant": [second]},
            set(),
            set(),
            sample_count=1,
            seed=5,
        )
        self.assertEqual([row["reference_answer"] for row in selected], ["C"])
        self.assertEqual(report["conflicting_answer_items"], 1)
        self.assertEqual(report["conflicting_answer_rows"], 2)

    def test_all_exclusion_representations_are_indexed(self):
        exact = format_ceval_prompt(_row(8))
        prompts, items, _ = exclusion_identities(
            [
                (
                    "train",
                    [
                        {
                            "prompt": "Wrapper that is not the original item",
                            "source_prompt": exact,
                            "messages": [
                                {"role": "user", "content": "Another wrapper"}
                            ],
                        }
                    ],
                )
            ]
        )
        self.assertIn(prompt_fingerprint(exact), prompts)
        self.assertIn(item_fingerprint(exact), items)

    def test_unanswered_rows_are_skipped_and_invalid_answers_are_rejected(self):
        rows = {"logic": [_row(0, answer=""), _row(1, answer="d")]}
        selected, report = build_nlr_evaluation(
            rows, set(), set(), sample_count=1, seed=1
        )
        self.assertEqual(selected[0]["reference_answer"], "D")
        self.assertEqual(report["unanswered_rows"], 1)
        with self.assertRaisesRegex(ManifestError, "answer 必须是"):
            build_nlr_evaluation(
                {"logic": [_row(0, answer="E")]},
                set(),
                set(),
                sample_count=1,
                seed=1,
            )

    def test_fails_when_pool_is_too_small(self):
        with self.assertRaisesRegex(ManifestError, "少于请求"):
            build_nlr_evaluation(
                {"logic": [_row(0)]}, set(), set(), sample_count=2, seed=1
            )

    def test_discovery_only_accepts_val_and_rejects_duplicate_subject(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            val = root / "logic" / "0.0.0" / "hash1" / "ceval-exam-val.arrow"
            dev = root / "logic" / "0.0.0" / "hash1" / "ceval-exam-dev.arrow"
            val.parent.mkdir(parents=True)
            val.write_bytes(b"val")
            dev.write_bytes(b"dev")
            self.assertEqual(discover_val_arrow_files(root), {"logic": val.resolve()})

            duplicate = (
                root / "logic" / "0.0.0" / "hash2" / "ceval-exam-val.arrow"
            )
            duplicate.parent.mkdir(parents=True)
            duplicate.write_bytes(b"other")
            with self.assertRaisesRegex(ManifestError, "多个 val Arrow"):
                discover_val_arrow_files(root)

    def test_discovery_and_selection_preserve_test_split_identity(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            test = root / "logic" / "0.0.0" / "hash" / "ceval-exam-test.arrow"
            test.parent.mkdir(parents=True)
            test.write_bytes(b"test")
            self.assertEqual(
                discover_split_arrow_files(root, "test"),
                {"logic": test.resolve()},
            )
        selected, _ = build_nlr_evaluation(
            {"logic": [_row(5)]},
            set(),
            set(),
            sample_count=1,
            seed=7,
            dataset_split="test",
        )
        self.assertEqual(selected[0]["sample_id"], "ceval_logic_test_5")
        self.assertEqual(selected[0]["source"]["split"], "test")

    def test_main_writes_hash_bound_manifest_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            cache = root / "cache"
            arrow_paths = {}
            source_rows = {}
            for subject in ("logic", "civil_servant"):
                path = (
                    cache
                    / subject
                    / "0.0.0"
                    / "hash"
                    / "ceval-exam-val.arrow"
                )
                path.parent.mkdir(parents=True)
                path.write_bytes((subject + "-arrow").encode("utf-8"))
                arrow_paths[subject] = path
                source_rows[subject] = [
                    _row(0, question=f"{subject} old?____"),
                    _row(1, question=f"{subject} new?____"),
                ]

            train = root / "train.jsonl"
            train.write_text(
                json.dumps(
                    {"prompt": format_ceval_prompt(source_rows["logic"][0])},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            validation = root / "validation.jsonl"
            validation.write_text(
                json.dumps({"source_prompt": "unrelated validation prompt"}) + "\n",
                encoding="utf-8",
            )
            old_eval = root / "old_eval.jsonl"
            old_eval.write_text(
                json.dumps({"prompt": "unrelated old evaluation prompt"}) + "\n",
                encoding="utf-8",
            )
            output = root / "frozen.jsonl"
            manifest_path = root / "manifest.json"

            def fake_read(path):
                subject = path.relative_to(cache).parts[0]
                return source_rows[subject]

            argv = [
                "--ceval_cache_root",
                str(cache),
                "--train_jsonl",
                str(train),
                "--validation_jsonl",
                str(validation),
                "--old_eval_jsonl",
                str(old_eval),
                "--output_jsonl",
                str(output),
                "--manifest",
                str(manifest_path),
                "--sample_count",
                "2",
                "--seed",
                "9",
            ]
            with mock.patch(
                "edge_llm_factory.build_frozen_nlr_eval._read_arrow_rows",
                side_effect=fake_read,
            ), redirect_stdout(io.StringIO()):
                main(argv)

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 2)
            self.assertEqual(manifest["artifact"]["sha256"], sha256_file(output))
            self.assertEqual(manifest["dataset"]["subject_count"], 2)
            self.assertEqual(
                {entry["subject"] for entry in manifest["dataset"]["arrow_files"]},
                {"logic", "civil_servant"},
            )
            self.assertTrue(
                all(entry["sha256"] == sha256_file(arrow_paths[entry["subject"]])
                    for entry in manifest["dataset"]["arrow_files"])
            )
            self.assertEqual(manifest["selection"]["excluded_item_overlap"], 0)

            with self.assertRaisesRegex(ManifestError, "拒绝覆盖"):
                main(argv)

    def test_main_subject_filter_limits_source_and_manifest(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            cache = root / "cache"
            source_rows = {}
            for subject in ("logic", "civil_servant", "college_economics"):
                path = (
                    cache
                    / subject
                    / "0.0.0"
                    / "hash"
                    / "ceval-exam-val.arrow"
                )
                path.parent.mkdir(parents=True)
                path.write_bytes((subject + "-arrow").encode("utf-8"))
                source_rows[subject] = [
                    _row(index, question=f"{subject} {index}?____")
                    for index in range(3)
                ]
            exclusions = []
            for name in ("train", "validation", "old_eval"):
                path = root / f"{name}.jsonl"
                path.write_text(
                    json.dumps({"prompt": f"unrelated {name} prompt"}) + "\n",
                    encoding="utf-8",
                )
                exclusions.append(path)
            output = root / "frozen.jsonl"
            manifest_path = root / "manifest.json"

            def fake_read(path):
                return source_rows[path.relative_to(cache).parts[0]]

            argv = [
                "--ceval_cache_root",
                str(cache),
                "--train_jsonl",
                str(exclusions[0]),
                "--validation_jsonl",
                str(exclusions[1]),
                "--old_eval_jsonl",
                str(exclusions[2]),
                "--subject",
                "college_economics",
                "--subject",
                "logic",
                "--output_jsonl",
                str(output),
                "--manifest",
                str(manifest_path),
                "--sample_count",
                "4",
            ]
            with mock.patch(
                "edge_llm_factory.build_frozen_nlr_eval._read_arrow_rows",
                side_effect=fake_read,
            ), redirect_stdout(io.StringIO()):
                main(argv)

            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {row["source"]["config"] for row in rows},
                {"college_economics", "logic"},
            )
            self.assertEqual(manifest["dataset"]["subject_count"], 2)
            self.assertEqual(
                manifest["dataset"]["subject_filter"],
                ["college_economics", "logic"],
            )
            self.assertEqual(
                {entry["subject"] for entry in manifest["dataset"]["arrow_files"]},
                {"college_economics", "logic"},
            )

    def test_main_rejects_unknown_or_repeated_subject_filters(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            cache = root / "cache"
            path = cache / "logic" / "0.0.0" / "hash" / "ceval-exam-val.arrow"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"logic-arrow")
            exclusions = []
            for name in ("train", "validation", "old_eval"):
                exclusion = root / f"{name}.jsonl"
                exclusion.write_text(
                    json.dumps({"prompt": f"unrelated {name}"}) + "\n",
                    encoding="utf-8",
                )
                exclusions.append(exclusion)

            base_argv = [
                "--ceval_cache_root",
                str(cache),
                "--train_jsonl",
                str(exclusions[0]),
                "--validation_jsonl",
                str(exclusions[1]),
                "--old_eval_jsonl",
                str(exclusions[2]),
                "--output_jsonl",
                str(root / "frozen.jsonl"),
                "--manifest",
                str(root / "manifest.json"),
                "--sample_count",
                "1",
            ]
            with mock.patch(
                "edge_llm_factory.build_frozen_nlr_eval._read_arrow_rows",
                return_value=[_row(1)],
            ), self.assertRaisesRegex(ManifestError, "不存在指定 subject"):
                main(base_argv + ["--subject", "missing"])
            with mock.patch(
                "edge_llm_factory.build_frozen_nlr_eval._read_arrow_rows",
                return_value=[_row(1)],
            ), self.assertRaisesRegex(ManifestError, "不能重复声明"):
                main(base_argv + ["--subject", "logic", "--subject", "logic"])

    def test_main_requires_train_validation_and_old_eval_roles(self):
        with self.assertRaises(SystemExit):
            with redirect_stdout(io.StringIO()):
                main(
                    [
                        "--ceval_cache_root",
                        "/does/not/matter",
                        "--output_jsonl",
                        "/tmp/out.jsonl",
                        "--manifest",
                        "/tmp/manifest.json",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
