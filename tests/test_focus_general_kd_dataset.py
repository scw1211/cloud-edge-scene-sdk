import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from edge_llm_factory.contracts import ManifestError, sha256_file
from edge_llm_factory.focus_general_kd_dataset import build_focused_dataset


TEACHER = "qwen3.5:9b"


def _canonical_fingerprint(prompt):
    canonical = json.dumps(
        {"user_messages": [" ".join(prompt.split())]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _row(category, label, prompt=None):
    prompt = prompt or "question {}".format(label)
    return {
        "category": category,
        "prompt_fingerprint": _canonical_fingerprint(prompt),
        "teacher_model": TEACHER,
        "teacher_verified": True,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "A"},
        ],
    }


def _evaluation_row(category, prompt):
    return {"category": category, "prompt": prompt, "reference_answer": "unused"}


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


class FocusGeneralKdDatasetTest(unittest.TestCase):
    def _source(
        self, root, train=None, validation=None, evaluation=None, **overrides
    ):
        train = train if train is not None else [
            _row("math", "train-math"),
            _row("natural_language_reasoning", "train-nlr"),
        ]
        validation = validation if validation is not None else [
            _row("math", "val-math"),
            _row("natural_language_reasoning", "val-nlr"),
        ]
        evaluation = evaluation if evaluation is not None else [
            _evaluation_row("math", "frozen evaluation math"),
            _evaluation_row(
                "natural_language_reasoning", "frozen evaluation reasoning"
            ),
        ]
        train_path = root / "train.jsonl"
        val_path = root / "val.jsonl"
        evaluation_path = root / "evaluation.jsonl"
        _write_jsonl(train_path, train)
        _write_jsonl(val_path, validation)
        _write_jsonl(evaluation_path, evaluation)
        manifest = {
            "schema_version": "edge-llm-general-kd/v2",
            "teacher_model": TEACHER,
            "teacher_no_thinking": True,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_category_counts": self._counts(train),
            "validation_category_counts": self._counts(validation),
            "scene_specific_samples": 0,
            "evaluation_prompt_overlap": 0,
            "evaluation_set_used_for_training": False,
            "artifacts": {
                "train": {"path": "train.jsonl", "sha256": sha256_file(train_path)},
                "validation": {"path": "val.jsonl", "sha256": sha256_file(val_path)},
                "evaluation": {
                    "path": "evaluation.jsonl",
                    "sha256": sha256_file(evaluation_path),
                },
            },
        }
        manifest.update(overrides)
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    @staticmethod
    def _counts(rows):
        counts = {}
        for row in rows:
            category = row["category"]
            counts[category] = counts.get(category, 0) + 1
        return dict(sorted(counts.items()))

    def test_builds_preregistered_nlr_subset_and_recomputes_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            output = root / "focused"
            result = build_focused_dataset(
                source, output, ["natural_language_reasoning"]
            )

            self.assertEqual(
                result["pre_registered_focus_categories"],
                ["natural_language_reasoning"],
            )
            self.assertEqual(result["train_category_counts"], {"natural_language_reasoning": 1})
            self.assertEqual(
                result["validation_category_counts"],
                {"natural_language_reasoning": 1},
            )
            audit = result["focus_category_audit"]["natural_language_reasoning"]
            self.assertEqual(audit["train"]["rows"], 1)
            self.assertEqual(audit["validation"]["rows"], 1)
            self.assertRegex(audit["train"]["content_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(
                audit["validation"]["content_sha256"], r"^[0-9a-f]{64}$"
            )
            self.assertEqual(audit["evaluation_prompt_overlap"], 0)
            self.assertEqual(result["scene_specific_samples"], 0)
            self.assertEqual(result["evaluation_prompt_overlap"], 0)
            self.assertIs(result["evaluation_set_used_for_training"], False)
            self.assertEqual(
                result["artifacts"]["train"]["sha256"],
                sha256_file(output / "train.jsonl"),
            )
            rows = [
                json.loads(line)
                for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["category"] for row in rows], ["natural_language_reasoning"])
            self.assertEqual(
                rows[0]["prompt_fingerprint"],
                _canonical_fingerprint("question train-nlr"),
            )

    def test_accepts_v1_source_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root, schema_version="edge-llm-general-kd/v1")
            result = build_focused_dataset(
                source, root / "focused", ["natural_language_reasoning"]
            )
            self.assertEqual(result["schema_version"], "edge-llm-general-kd/v1")

    def test_rejects_unknown_focus_category(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ManifestError):
                build_focused_dataset(self._source(root), root / "focused", ["traffic"])

    def test_rejects_empty_focus_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ManifestError, "至少一个"):
                build_focused_dataset(self._source(root), root / "focused", [])

    def test_rejects_more_than_two_focus_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ManifestError, "最多预注册两个"):
                build_focused_dataset(
                    self._source(root),
                    root / "focused",
                    ["math", "code", "natural_language_reasoning"],
                )

    def test_rejects_empty_filtered_validation_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(
                root,
                validation=[_row("math", "val-math")],
            )
            with self.assertRaises(ManifestError):
                build_focused_dataset(
                    source, root / "focused", ["natural_language_reasoning"]
                )

    def test_each_requested_focus_requires_its_own_validation_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(
                root,
                validation=[_row("math", "val-math")],
            )
            with self.assertRaisesRegex(
                ManifestError,
                "natural_language_reasoning 在验证集中没有可用样本",
            ):
                build_focused_dataset(
                    source,
                    root / "focused",
                    ["math", "natural_language_reasoning"],
                )

    def test_rejects_train_validation_prompt_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(
                root,
                train=[_row("natural_language_reasoning", "same")],
                validation=[_row("natural_language_reasoning", "same")],
            )
            with self.assertRaisesRegex(ManifestError, "prompt 重叠"):
                build_focused_dataset(
                    source, root / "focused", ["natural_language_reasoning"]
                )

    def test_rejects_invalid_source_isolation_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root, scene_specific_samples=1)
            with self.assertRaisesRegex(ManifestError, "隔离声明"):
                build_focused_dataset(
                    source, root / "focused", ["natural_language_reasoning"]
                )

    def test_rejects_tampered_prompt_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tampered = _row("natural_language_reasoning", "train-nlr")
            tampered["prompt_fingerprint"] = "0" * 64
            source = self._source(
                root,
                train=[tampered],
                validation=[_row("natural_language_reasoning", "val-nlr")],
            )
            with self.assertRaisesRegex(
                ManifestError, "prompt_fingerprint 与实际内容不一致"
            ):
                build_focused_dataset(
                    source, root / "focused", ["natural_language_reasoning"]
                )

    def test_rejects_false_zero_evaluation_overlap_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(
                root,
                train=[
                    _row(
                        "natural_language_reasoning",
                        "train-overlap",
                        prompt="same frozen prompt",
                    )
                ],
                validation=[_row("natural_language_reasoning", "val-nlr")],
                evaluation=[
                    _evaluation_row(
                        "natural_language_reasoning", "same frozen prompt"
                    )
                ],
            )
            with self.assertRaisesRegex(
                ManifestError,
                "evaluation_prompt_overlap.*声明=0，实测=1",
            ):
                build_focused_dataset(
                    source, root / "focused", ["natural_language_reasoning"]
                )

    def test_rejects_validation_overlap_with_frozen_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(
                root,
                train=[_row("natural_language_reasoning", "train-nlr")],
                validation=[
                    _row(
                        "natural_language_reasoning",
                        "val-overlap",
                        prompt="same validation prompt",
                    )
                ],
                evaluation=[
                    _evaluation_row(
                        "natural_language_reasoning", "same validation prompt"
                    )
                ],
            )
            with self.assertRaisesRegex(
                ManifestError,
                "evaluation_prompt_overlap.*声明=0，实测=1",
            ):
                build_focused_dataset(
                    source, root / "focused", ["natural_language_reasoning"]
                )

    def test_binds_new_preregistered_evaluation_without_rewriting_source_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            additional = root / "future_math.jsonl"
            _write_jsonl(
                additional,
                [_evaluation_row("math", "a newly frozen unseen math question")],
            )
            result = build_focused_dataset(
                source,
                root / "focused",
                ["math"],
                additional_evaluation_jsonl=additional,
            )
            evidence = result["additional_frozen_evaluation"]
            self.assertEqual(evidence["sha256"], sha256_file(additional))
            self.assertEqual(evidence["rows"], 1)
            self.assertEqual(evidence["overlap"], {"train": 0, "validation": 0})
            self.assertIs(evidence["used_for_training"], False)

    def test_rejects_overlap_with_new_preregistered_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            additional = root / "future_math.jsonl"
            _write_jsonl(
                additional,
                [_evaluation_row("math", "question train-math")],
            )
            with self.assertRaisesRegex(ManifestError, "新增冻结终测集存在 prompt 重叠"):
                build_focused_dataset(
                    source,
                    root / "focused",
                    ["math"],
                    additional_evaluation_jsonl=additional,
                )

    def test_rejects_false_evaluation_used_for_training_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root, evaluation_set_used_for_training=True)
            with self.assertRaisesRegex(
                ManifestError, "evaluation_set_used_for_training"
            ):
                build_focused_dataset(
                    source, root / "focused", ["natural_language_reasoning"]
                )

    def test_accepts_content_verified_legacy_fingerprint_and_rewrites_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _row("natural_language_reasoning", "legacy")
            train["source_prompt"] = "question legacy"
            legacy = hashlib.sha256(
                b"natural_language_reasoning\nquestion legacy"
            ).hexdigest()
            train["prompt_fingerprint"] = legacy
            source = self._source(
                root,
                train=[train],
                validation=[_row("natural_language_reasoning", "val-nlr")],
            )
            build_focused_dataset(
                source, root / "focused", ["natural_language_reasoning"]
            )
            output_row = json.loads(
                (root / "focused" / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(
                output_row["prompt_fingerprint"],
                _canonical_fingerprint("question legacy"),
            )
            self.assertNotEqual(output_row["prompt_fingerprint"], legacy)

    def test_rejects_source_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            (root / "train.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "哈希"):
                build_focused_dataset(
                    source, root / "focused", ["natural_language_reasoning"]
                )

    def test_rejects_unknown_category_in_source_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(
                root,
                train=[
                    _row("natural_language_reasoning", "train-nlr"),
                    _row("traffic", "train-traffic"),
                ],
                validation=[_row("natural_language_reasoning", "val-nlr")],
            )
            with self.assertRaisesRegex(ManifestError, "非通用类别"):
                build_focused_dataset(
                    source, root / "focused", ["natural_language_reasoning"]
                )


if __name__ == "__main__":
    unittest.main()
