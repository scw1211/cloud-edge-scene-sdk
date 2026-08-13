import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from edge_llm_factory import build_unified_traffic_general_dataset as dataset_builder
from edge_llm_factory import train_unified_traffic_general as unified_trainer
from edge_llm_factory.contracts import ManifestError, sha256_file as real_sha256_file


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _traffic_row(event_id: str, prompt: str, target: str, fingerprint: str):
    return {
        "event_id": event_id,
        "category": "traffic_action",
        "prompt_format": "raw_task",
        "prompt_fingerprint": fingerprint,
        "source_prompt": prompt,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
    }


def _general_row(
    event_id: str,
    category: str,
    prompt: str,
    target: str,
    fingerprint: str,
):
    system = (
        dataset_builder.MATH_SYSTEM
        if category == "math"
        else dataset_builder.LOGIC_SYSTEM
    )
    return {
        "event_id": event_id,
        "category": category,
        "prompt_format": "tokenizer_chat",
        "prompt_fingerprint": fingerprint,
        "source_prompt": prompt,
        "reference_answer": target,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
    }


class _CharacterTokenizer:
    """Tokenizer stub that makes every character exactly one token."""

    @staticmethod
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        if tokenize:
            raise AssertionError("the trainer requests text chat templates")
        pieces = []
        for message in messages:
            role = message["role"]
            if role == "assistant":
                pieces.append("<assistant>" + message["content"])
            else:
                pieces.append(f"<{role}>" + message["content"] + f"</{role}>")
        if add_generation_prompt:
            pieces.append("<assistant>")
        return "".join(pieces)

    @staticmethod
    def __call__(text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("unified tokenization must disable special tokens")
        return {"input_ids": [ord(character) for character in text]}


class UnifiedDatasetContractTests(unittest.TestCase):
    def test_traffic_requires_exactly_16_digits_and_an_a_to_f_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.jsonl"
            _write_jsonl(
                valid,
                [
                    {
                        "event_id": "traffic-1",
                        "messages": [
                            {"role": "user", "content": "0123456789012345"},
                            {"role": "assistant", "content": "F"},
                        ],
                    }
                ],
            )
            rows = dataset_builder._traffic_rows(valid, "train")
            self.assertEqual(rows[0]["prompt_format"], "raw_task")
            self.assertEqual(rows[0]["messages"][-1]["content"], "F")

            invalid_cases = (
                ("012345678901234", "A", "16 位"),
                ("012345678901234x", "A", "16 位"),
                ("0123456789012345", "G", "A-F"),
            )
            for index, (prompt, target, expected_error) in enumerate(invalid_cases):
                with self.subTest(prompt=prompt, target=target):
                    path = root / f"invalid-{index}.jsonl"
                    _write_jsonl(
                        path,
                        [
                            {
                                "event_id": f"invalid-{index}",
                                "messages": [
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": target},
                                ],
                            }
                        ],
                    )
                    with self.assertRaisesRegex(ManifestError, expected_error):
                        dataset_builder._traffic_rows(path, "train")

    def test_logiqa_rows_with_empty_options_are_removed_not_trained(self) -> None:
        valid = {
            "text": "材料",
            "question": "哪项必然成立？",
            "options": ["甲", "乙", "丙", "丁"],
            "answer": 2,
        }
        invalid = {
            "text": "材料",
            "question": "哪项必然成立？",
            "options": ["甲", "", "丙", "丁"],
            "answer": 0,
        }
        groups, invalid_count = dataset_builder._logiqa_groups([valid, invalid])
        self.assertEqual(invalid_count, 1)
        self.assertEqual(sum(len(group) for group in groups.values()), 1)
        retained = next(iter(groups.values()))[0]
        self.assertEqual(retained["options"], valid["options"])

    def test_builder_rejects_general_train_validation_prompt_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._input_paths(root)
            output = root / "out"
            traffic_train, traffic_val, math_train, math_val, logic_train, logic_val = (
                self._minimal_builder_rows()
            )
            math_val[0]["prompt_fingerprint"] = math_train[0]["prompt_fingerprint"]
            argv = self._builder_argv(paths, output)
            with self._builder_mocks(
                paths,
                traffic_train,
                traffic_val,
                math_train,
                math_val,
                logic_train,
                logic_val,
            ):
                with self.assertRaisesRegex(ManifestError, "训练集和验证集 prompt 重叠"):
                    dataset_builder.main(argv)

    def test_formal_test_files_are_hashed_but_never_passed_to_a_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._input_paths(root)
            output = root / "out"
            rows = self._minimal_builder_rows()
            argv = self._builder_argv(paths, output)
            with self._builder_mocks(paths, *rows) as mocks:
                with redirect_stdout(io.StringIO()):
                    dataset_builder.main(argv)

            traffic_loader = mocks[0]
            logiqa_loader = mocks[2]
            self.assertEqual(
                [call.args[0] for call in traffic_loader.call_args_list],
                [paths["traffic_train"].resolve(), paths["traffic_val"].resolve()],
            )
            logiqa_loader.assert_called_once_with(
                paths["logiqa_train"].resolve(),
                paths["logiqa_dev"].resolve(),
                3600,
                400,
                20260810,
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["traffic_test_used_for_training"])
            self.assertFalse(manifest["gsm8k_test_loaded"])
            self.assertFalse(manifest["logiqa_test_loaded"])
            self.assertFalse(manifest["sources"]["traffic"]["test"]["content_loaded"])
            self.assertFalse(manifest["sources"]["logiqa2"]["test_content_loaded"])

    @staticmethod
    def _input_paths(root: Path):
        paths = {
            name: root / f"{name}.jsonl"
            for name in (
                "traffic_train",
                "traffic_val",
                "traffic_test",
                "logiqa_train",
                "logiqa_dev",
                "logiqa_test",
            )
        }
        for name, path in paths.items():
            path.write_bytes(("not parsed: " + name).encode("utf-8"))
        return paths

    @staticmethod
    def _minimal_builder_rows():
        return (
            [_traffic_row("traffic:train:1", "0123456789012345", "A", "tf-train")],
            [_traffic_row("traffic:val:1", "0123456789012345", "A", "tf-val")],
            [_general_row("math:train:1", "math", "1+1?", "FINAL: 2", "m-train")],
            [_general_row("math:val:1", "math", "2+2?", "FINAL: 4", "m-val")],
            [
                _general_row(
                    "logic:train:1",
                    "natural_language_reasoning",
                    "训练逻辑题",
                    "A",
                    "l-train",
                )
            ],
            [
                _general_row(
                    "logic:val:1",
                    "natural_language_reasoning",
                    "验证逻辑题",
                    "B",
                    "l-val",
                )
            ],
        )

    @staticmethod
    def _builder_argv(paths, output):
        return [
            "--traffic_train",
            str(paths["traffic_train"]),
            "--traffic_val",
            str(paths["traffic_val"]),
            "--traffic_test",
            str(paths["traffic_test"]),
            "--logiqa_train_zh",
            str(paths["logiqa_train"]),
            "--logiqa_dev_zh",
            str(paths["logiqa_dev"]),
            "--logiqa_test_zh",
            str(paths["logiqa_test"]),
            "--output_dir",
            str(output),
        ]

    @staticmethod
    def _builder_mocks(
        paths,
        traffic_train,
        traffic_val,
        math_train,
        math_val,
        logic_train,
        logic_val,
    ):
        def traffic_side_effect(path, split):
            return traffic_train if split == "train" else traffic_val

        def hash_side_effect(path):
            path = Path(path).resolve()
            if path == paths["logiqa_test"].resolve():
                return dataset_builder.LOGIQA_TEST_SHA256
            return real_sha256_file(path)

        traffic_patch = patch.object(
            dataset_builder, "_traffic_rows", side_effect=traffic_side_effect
        )
        gsm_patch = patch.object(
            dataset_builder,
            "_gsm8k_rows",
            return_value=(math_train, math_val, {"test_split_loaded": False}),
        )
        logiqa_patch = patch.object(
            dataset_builder,
            "_clean_logiqa",
            return_value=(logic_train, logic_val, {"test_split_loaded": False}),
        )
        hash_patch = patch.object(dataset_builder, "sha256_file", side_effect=hash_side_effect)

        class _CombinedPatches:
            def __enter__(self):
                self.entered = [
                    traffic_patch.__enter__(),
                    gsm_patch.__enter__(),
                    logiqa_patch.__enter__(),
                    hash_patch.__enter__(),
                ]
                return self.entered

            def __exit__(self, exc_type, exc_value, traceback):
                for active_patch in (hash_patch, logiqa_patch, gsm_patch, traffic_patch):
                    active_patch.__exit__(exc_type, exc_value, traceback)

        return _CombinedPatches()


class UnifiedTrainerContractTests(unittest.TestCase):
    def test_validate_rows_allows_legacy_traffic_prompt_duplicates_only(self) -> None:
        rows = [
            _traffic_row("traffic:1", "0123456789012345", "A", "same-traffic"),
            _traffic_row("traffic:2", "0123456789012345", "B", "same-traffic"),
            _general_row("math:1", "math", "1+1?", "FINAL: 2", "math-1"),
            _general_row(
                "logic:1",
                "natural_language_reasoning",
                "逻辑题",
                "A",
                "logic-1",
            ),
        ]
        counts = unified_trainer.validate_rows(rows, "train")
        self.assertEqual(counts["traffic_action"], 2)

        rows.append(
            _general_row("math:2", "math", "另一道题", "FINAL: 3", "math-1")
        )
        with self.assertRaisesRegex(ManifestError, "通用 prompt_fingerprint 重复"):
            unified_trainer.validate_rows(rows, "train")

    def test_mixed_raw_and_chat_tokenization_preserves_supervision_boundaries(self) -> None:
        rows = [
            _traffic_row("traffic:1", "0123456789012345", "C", "traffic-1"),
            _general_row("math:1", "math", "1+1?", "FINAL: 2", "math-1"),
            _general_row(
                "logic:1",
                "natural_language_reasoning",
                "逻辑题",
                "D",
                "logic-1",
            ),
        ]
        tokenized, stats = unified_trainer.tokenize_rows(
            rows, _CharacterTokenizer(), max_seq_length=1024
        )
        traffic = tokenized[0]
        self.assertEqual(len(traffic["input_ids"]), 17)
        self.assertEqual(traffic["labels"][:16], [-100] * 16)
        self.assertNotEqual(traffic["labels"][16], -100)
        for chat_row in tokenized[1:]:
            first_target = next(
                index for index, token in enumerate(chat_row["labels"]) if token != -100
            )
            self.assertGreater(first_target, 0)
            self.assertTrue(all(token == -100 for token in chat_row["labels"][:first_target]))
            self.assertTrue(all(token != -100 for token in chat_row["labels"][first_target:]))
        self.assertEqual(stats["category_sequence_tokens"]["traffic_action"]["max"], 17)

    def test_two_to_one_to_one_sampling_is_effective_and_deterministic(self) -> None:
        rows = (
            [{"_category": "traffic_action"} for _ in range(3)]
            + [{"_category": "math"}]
            + [{"_category": "natural_language_reasoning"} for _ in range(2)]
        )
        weights = unified_trainer.parse_task_weights(
            "traffic_action=2,math=1,natural_language_reasoning=1"
        )
        indices, report = unified_trainer.build_sampling_indices(rows, weights, seed=9)
        second_indices, second_report = unified_trainer.build_sampling_indices(
            rows, weights, seed=9
        )
        self.assertEqual(indices, second_indices)
        self.assertEqual(report, second_report)
        self.assertEqual(
            report["effective_category_counts"],
            {"traffic_action": 4, "math": 2, "natural_language_reasoning": 2},
        )
        self.assertEqual(
            report["effective_ratios"],
            {"traffic_action": 0.5, "math": 0.25, "natural_language_reasoning": 0.25},
        )

    def test_resume_adapter_must_match_preregistered_production_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_path = root / "base.json"
            snapshot_manifest_path = root / "snapshot.json"
            dataset_manifest_path = root / "dataset_manifest.json"
            snapshot_path = root / "snapshot"
            resume_adapter = root / "adapter"
            train_path = root / "train.jsonl"
            val_path = root / "val.jsonl"
            output_path = root / "candidate"
            snapshot_path.mkdir()
            resume_adapter.mkdir()
            base_path.write_text("{}", encoding="utf-8")
            snapshot_manifest_path.write_text("{}", encoding="utf-8")
            (resume_adapter / "adapter_model.safetensors").write_bytes(b"not-production")
            (resume_adapter / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-0.8B"}),
                encoding="utf-8",
            )
            train_rows, val_rows = self._minimal_training_rows()
            _write_jsonl(train_path, train_rows)
            _write_jsonl(val_path, val_rows)
            counts = {
                "math": 1,
                "natural_language_reasoning": 1,
                "traffic_action": 1,
            }
            manifest = {
                "schema_version": unified_trainer.SCHEMA_VERSION,
                "traffic_test_used_for_training": False,
                "gsm8k_test_loaded": False,
                "logiqa_test_loaded": False,
                "general_train_validation_prompt_overlap": 0,
                "train_category_counts": counts,
                "validation_category_counts": counts,
                "artifacts": {
                    "train": {"sha256": real_sha256_file(train_path)},
                    "validation": {"sha256": real_sha256_file(val_path)},
                },
            }
            dataset_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            fake_base = {
                "source": {"model_id": "Qwen/Qwen3.5-0.8B", "revision": "locked"},
                "model": {"parameter_count": 1},
                "base_id": "fake-base",
            }

            def read_object(path):
                path = Path(path).resolve()
                if path == base_path.resolve():
                    return fake_base
                if path == snapshot_manifest_path.resolve():
                    return {}
                if path == dataset_manifest_path.resolve():
                    return manifest
                if path == (resume_adapter / "adapter_config.json").resolve():
                    return {"base_model_name_or_path": "Qwen/Qwen3.5-0.8B"}
                raise AssertionError(f"unexpected JSON read: {path}")

            argv = [
                "--base",
                str(base_path),
                "--snapshot",
                str(snapshot_path),
                "--snapshot_manifest",
                str(snapshot_manifest_path),
                "--dataset_manifest",
                str(dataset_manifest_path),
                "--train_jsonl",
                str(train_path),
                "--val_jsonl",
                str(val_path),
                "--resume_adapter",
                str(resume_adapter),
                "--expected_resume_adapter_sha256",
                "registered-production-sha256",
                "--output",
                str(output_path),
            ]
            with patch.object(
                unified_trainer, "read_json_object", side_effect=read_object
            ), patch.object(
                unified_trainer, "validate_base_manifest", side_effect=lambda value: value
            ), patch.object(
                unified_trainer, "verify_text_snapshot", return_value={"valid": True}
            ):
                with self.assertRaisesRegex(
                    ManifestError, "不等于预注册生产 SHA-256"
                ):
                    unified_trainer.main(argv)

    @staticmethod
    def _minimal_training_rows():
        train = [
            _traffic_row("traffic:train:1", "0123456789012345", "A", "tf-common"),
            _general_row("math:train:1", "math", "1+1?", "FINAL: 2", "math-train"),
            _general_row(
                "logic:train:1",
                "natural_language_reasoning",
                "训练逻辑题",
                "A",
                "logic-train",
            ),
        ]
        validation = [
            _traffic_row("traffic:val:1", "0123456789012345", "B", "tf-common"),
            _general_row("math:val:1", "math", "2+2?", "FINAL: 4", "math-val"),
            _general_row(
                "logic:val:1",
                "natural_language_reasoning",
                "验证逻辑题",
                "B",
                "logic-val",
            ),
        ]
        return train, validation


if __name__ == "__main__":
    unittest.main()
