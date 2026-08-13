import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from edge_llm_factory.contracts import ManifestError
from edge_llm_factory import train_unified_traffic_general_u2 as u2


@unittest.skipUnless(
    torch is not None and hasattr(torch, "zeros"),
    "requires a real PyTorch installation",
)
class UnifiedU2LossTests(unittest.TestCase):
    def test_restricted_loss_uses_first_target_and_penalizes_outside_mass(self) -> None:
        logits = torch.zeros((3, 4, 10), dtype=torch.float32, requires_grad=True)
        labels = torch.tensor(
            [
                [-100, -100, 2, -100],
                [-100, -100, 6, -100],
                [-100, -100, 8, -100],
            ],
            dtype=torch.long,
        )
        categories = torch.tensor(
            [
                u2.CATEGORY_IDS["traffic_action"],
                u2.CATEGORY_IDS["natural_language_reasoning"],
                u2.CATEGORY_IDS["math"],
            ],
            dtype=torch.long,
        )
        first_targets = torch.tensor([2, 2, 2], dtype=torch.long)
        allowed = {
            u2.CATEGORY_IDS["traffic_action"]: (1, 2, 3),
            u2.CATEGORY_IDS["natural_language_reasoning"]: (5, 6),
        }

        restricted, outside, count = u2.restricted_slot_losses(
            logits, labels, categories, first_targets, allowed
        )
        self.assertEqual(count, 2)
        expected_restricted = (
            torch.log(torch.tensor(3.0)) + torch.log(torch.tensor(2.0))
        ).item() / 2.0
        self.assertAlmostEqual(
            restricted.item(), expected_restricted
        )
        self.assertGreater(outside.item(), 0.0)
        (restricted + outside).backward()
        self.assertIsNotNone(logits.grad)

        improved = logits.detach().clone()
        improved[0, 1, :] = -10.0
        improved[0, 1, 1:4] = 0.0
        improved[1, 1, :] = -10.0
        improved[1, 1, 5:7] = 0.0
        restricted_after, outside_after, _ = u2.restricted_slot_losses(
            improved, labels, categories, first_targets, allowed
        )
        self.assertAlmostEqual(restricted.item(), restricted_after.item(), places=5)
        self.assertLess(outside_after.item(), outside.item())

    def test_restricted_loss_rejects_target_outside_slot(self) -> None:
        logits = torch.zeros((1, 3, 8), dtype=torch.float32)
        labels = torch.tensor([[-100, -100, 7]], dtype=torch.long)
        categories = torch.tensor([u2.CATEGORY_IDS["traffic_action"]])
        positions = torch.tensor([2])
        with self.assertRaisesRegex(ManifestError, "不在允许槽位"):
            u2.restricted_slot_losses(
                logits,
                labels,
                categories,
                positions,
                {u2.CATEGORY_IDS["traffic_action"]: (1, 2, 3)},
            )

    def test_trainer_combines_standard_and_both_restricted_losses(self) -> None:
        class FakeTrainerBase:
            def __init__(self, *args, **kwargs):
                del args, kwargs

        class FakeModel:
            def __call__(self, **inputs):
                self.seen_keys = sorted(inputs)
                logits = torch.zeros((1, 3, 6), dtype=torch.float32)
                logits[0, 1, 1] = 1.0
                return SimpleNamespace(loss=torch.tensor(2.0), logits=logits)

        trainer_class = u2._make_u2_trainer_class(FakeTrainerBase)
        trainer = trainer_class(
            allowed_token_ids={u2.CATEGORY_IDS["traffic_action"]: (1, 2)},
            lm_ce_weight=1.0,
            restricted_ce_weight=2.0,
            outside_mass_weight=3.0,
        )
        model = FakeModel()
        inputs = {
            "input_ids": torch.tensor([[0, 0, 1]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[-100, -100, 1]]),
            "u2_category_id": torch.tensor([u2.CATEGORY_IDS["traffic_action"]]),
            "u2_first_target_index": torch.tensor([2]),
        }
        total = trainer.compute_loss(model, inputs)
        trainer_model_keys = model.seen_keys
        restricted, outside, _ = u2.restricted_slot_losses(
            model(input_ids=torch.tensor([[0, 0, 1]])).logits,
            torch.tensor([[-100, -100, 1]]),
            torch.tensor([u2.CATEGORY_IDS["traffic_action"]]),
            torch.tensor([2]),
            {u2.CATEGORY_IDS["traffic_action"]: (1, 2)},
        )
        self.assertTrue(torch.allclose(total, 2.0 + 2.0 * restricted + 3.0 * outside))
        self.assertEqual(trainer_model_keys, ["attention_mask", "input_ids", "labels"])
        self.assertIn("standard_lm_ce", trainer.u2_last_loss_components)


class UnifiedU2SamplingTests(unittest.TestCase):
    def test_default_task_ratio_and_math_hardness_are_deterministic(self) -> None:
        rows = (
            [
                {"_category": "traffic_action", "_hardness_weight": 1.0}
                for _ in range(20)
            ]
            + [
                {"_category": "math", "_hardness_weight": weight}
                for weight in (1.0, 1.0, 1.0, 12.0)
            ]
            + [
                {"_category": "natural_language_reasoning", "_hardness_weight": 1.0}
                for _ in range(6)
            ]
        )
        weights = u2.parse_task_weights(
            "traffic_action=5,math=2,natural_language_reasoning=3"
        )
        first, report = u2.build_u2_sampling_indices(rows, weights, seed=41)
        second, second_report = u2.build_u2_sampling_indices(rows, weights, seed=41)
        self.assertEqual(first, second)
        self.assertEqual(report, second_report)
        self.assertEqual(
            report["effective_ratios"],
            {
                "traffic_action": 0.5,
                "math": 0.2,
                "natural_language_reasoning": 0.3,
            },
        )
        math_start = 20
        hard_index = math_start + 3
        selected_math = [
            index for index in first if rows[index]["_category"] == "math"
        ]
        self.assertEqual(set(selected_math), set(range(math_start, math_start + 4)))
        self.assertGreater(selected_math.count(hard_index), len(selected_math) / 2)
        self.assertEqual(
            report["math_hardness_sampling"]["mode"],
            "full_coverage_then_weighted_remainder",
        )

    def test_equal_math_weights_cover_every_row_before_repeating(self) -> None:
        rows = (
            [
                {"_category": "traffic_action", "_hardness_weight": 1.0}
                for _ in range(3600)
            ]
            + [
                {"_category": "math", "_hardness_weight": 1.0}
                for _ in range(6673)
            ]
            + [
                {
                    "_category": "natural_language_reasoning",
                    "_hardness_weight": 1.0,
                }
                for _ in range(12193)
            ]
        )
        selected, report = u2.build_u2_sampling_indices(
            rows,
            u2.parse_task_weights(
                "traffic_action=5,math=2,natural_language_reasoning=3"
            ),
            seed=20260811,
        )
        math_indices = {
            index for index in selected if rows[index]["_category"] == "math"
        }
        self.assertEqual(len(math_indices), 6673)
        self.assertEqual(report["selected_unique_rows"]["math"], 6673)
        self.assertEqual(
            report["math_hardness_sampling"]["mode"],
            "uniform_cycle_equal_weights",
        )
        self.assertEqual(
            report["math_hardness_sampling"]["additional_weighted_draws"], 0
        )

    def test_non_math_hardness_is_rejected(self) -> None:
        rows = [
            {
                "event_id": "traffic:1",
                "category": "traffic_action",
                "prompt_format": "raw_task",
                "prompt_fingerprint": "traffic-1",
                "hardness_weight": 2.0,
                "messages": [
                    {"role": "user", "content": "0123456789012345"},
                    {"role": "assistant", "content": "A"},
                ],
            },
            {
                "event_id": "math:1",
                "category": "math",
                "prompt_format": "tokenizer_chat",
                "prompt_fingerprint": "math-1",
                "messages": [
                    {"role": "user", "content": "1+1?"},
                    {"role": "assistant", "content": "FINAL: 2"},
                ],
            },
            {
                "event_id": "logic:1",
                "category": "natural_language_reasoning",
                "prompt_format": "tokenizer_chat",
                "prompt_fingerprint": "logic-1",
                "messages": [
                    {"role": "user", "content": "选择"},
                    {"role": "assistant", "content": "B"},
                ],
            },
        ]
        with self.assertRaisesRegex(ManifestError, "只有 math"):
            u2.validate_v2_rows(rows, "train")


class UnifiedU2IsolationAndStartTests(unittest.TestCase):
    def test_official_training_contract_requires_bf16_and_manifest_weights(self) -> None:
        manifest = {
            "recommended_effective_task_weights": {
                "traffic_action": 5,
                "math": 2,
                "natural_language_reasoning": 3,
            }
        }
        weights = u2.parse_task_weights(
            "traffic_action=5,math=2,natural_language_reasoning=3"
        )
        u2.validate_u2_training_contract(
            bf16=True, fp16=False, weights=weights, manifest=manifest
        )
        with self.assertRaisesRegex(ManifestError, "必须且只能使用 BF16"):
            u2.validate_u2_training_contract(
                bf16=False, fp16=False, weights=weights, manifest=manifest
            )
        with self.assertRaisesRegex(ManifestError, "任务权重"):
            u2.validate_u2_training_contract(
                bf16=True,
                fp16=False,
                weights=u2.parse_task_weights(
                    "traffic_action=2,math=1,natural_language_reasoning=1"
                ),
                manifest=manifest,
            )

    def test_manifest_requires_every_isolation_field_to_be_false(self) -> None:
        manifest = {
            "schema_version": u2.SCHEMA_VERSION,
            "general_train_validation_prompt_overlap": 0,
            "general_train_promotion_prompt_overlap": 0,
            "general_validation_promotion_prompt_overlap": 0,
            "u1_general_dev_used_as_training_validation": True,
            "u1_general_dev_used_for_promotion": False,
            "u1_general_dev_overlap_with_train_or_promotion": 0,
            "u1_general_dev_training_validation_rows": 800,
            "u1_general_dev_training_validation_category_counts": {
                "math": 400,
                "natural_language_reasoning": 400,
            },
            "provenance": {"parent": "train-only"},
            **{field: False for field in u2.REQUIRED_FALSE_ISOLATION_FIELDS},
        }
        self.assertEqual(
            u2.validate_v2_manifest(manifest),
            {field: False for field in u2.REQUIRED_FALSE_ISOLATION_FIELDS},
        )
        manifest["u1_evaluation_outputs_used_for_training"] = True
        with self.assertRaisesRegex(
            ManifestError, "u1_evaluation_outputs_used_for_training"
        ):
            u2.validate_v2_manifest(manifest)

    def test_wrong_u1_starting_weights_are_rejected_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp)
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"candidate-u1")
            (adapter / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-0.8B"}),
                encoding="utf-8",
            )
            actual = hashlib.sha256(b"candidate-u1").hexdigest()
            config_sha256 = u2.sha256_file(adapter / "adapter_config.json")
            with self.assertRaisesRegex(ManifestError, "不等于预注册"):
                u2.verify_u1_adapter(
                    adapter,
                    "0" * 64,
                    config_sha256,
                    "Qwen/Qwen3.5-0.8B",
                )
            report = u2.verify_u1_adapter(
                adapter,
                actual,
                config_sha256,
                "Qwen/Qwen3.5-0.8B",
            )
            self.assertEqual(report["weights_sha256"], actual)

            with self.assertRaisesRegex(ManifestError, "配置不等于预注册"):
                u2.verify_u1_adapter(
                    adapter,
                    actual,
                    "0" * 64,
                    "Qwen/Qwen3.5-0.8B",
                )


if __name__ == "__main__":
    unittest.main()
