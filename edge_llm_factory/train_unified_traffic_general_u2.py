"""从锁定的 U1 LoRA 继续训练统一交通、数学与中文逻辑模型。

U2 与 U1 使用不同的数据清单和训练入口，避免改写已经完成取证的 U1
训练器。它保留标准 causal-LM 交叉熵，并在交通 A-F、中文逻辑 A-D 的
首个监督 token 上增加槽位内交叉熵和槽位外概率质量惩罚。数学训练样本
可以通过 ``hardness_weight`` 在数学类别内部进行确定性加权抽样；该权重
不会改变交通或中文逻辑类别的任务配比。

本入口只读取训练集和验证集。正式交通测试集、GSM8K test、LogiQA test、
U1 评测输出及盲测数据必须在 v2 manifest 中明确声明未用于训练。
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import inspect
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from edge_llm_factory.contracts import (
    ManifestError,
    base_fingerprint,
    read_json_object,
    sha256_file,
    validate_base_manifest,
    write_json_object,
)
from edge_llm_factory.text_base import verify_text_snapshot
from edge_llm_factory import train_unified_traffic_general as u1


SCHEMA_VERSION = "edge-llm-unified-traffic-general/v2"
CATEGORIES = u1.CATEGORIES
CATEGORY_IDS = {category: index for index, category in enumerate(CATEGORIES)}
RESTRICTED_LABELS = {
    "traffic_action": tuple("ABCDEF"),
    "natural_language_reasoning": tuple("ABCD"),
}
REQUIRED_FALSE_ISOLATION_FIELDS = (
    "traffic_test_used_for_training",
    "gsm8k_test_loaded",
    "logiqa_test_loaded",
    "formal_evaluation_used_for_training",
    "u1_evaluation_outputs_used_for_training",
    "blind_evaluation_used_for_training",
    "promotion_dev_used_for_training",
    "promotion_dev_used_for_model_selection",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate_v2_manifest(manifest: Mapping[str, Any]) -> Dict[str, bool]:
    """Validate the v2 schema and return the frozen isolation declaration."""

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("U2 数据 manifest schema 不受支持")
    isolation: Dict[str, bool] = {}
    for field in REQUIRED_FALSE_ISOLATION_FIELDS:
        if manifest.get(field) is not False:
            raise ManifestError(f"U2 数据测试隔离字段必须显式为 false: {field}")
        isolation[field] = False
    if manifest.get("general_train_validation_prompt_overlap") != 0:
        raise ManifestError("U2 通用训练和验证 prompt 存在重叠")
    if manifest.get("general_train_promotion_prompt_overlap") != 0:
        raise ManifestError("U2 通用训练和 promotion prompt 存在重叠")
    if manifest.get("general_validation_promotion_prompt_overlap") != 0:
        raise ManifestError("U2 通用验证和 promotion prompt 存在重叠")
    if manifest.get("u1_general_dev_used_as_training_validation") is not True:
        raise ManifestError("U2 未如实声明已使用 U1 dev 进行训练期验证")
    if manifest.get("u1_general_dev_used_for_promotion") is not False:
        raise ManifestError("U1 dev 不能进入 U2 promotion")
    if manifest.get("u1_general_dev_overlap_with_train_or_promotion") != 0:
        raise ManifestError("U1 dev 不能进入 U2 gradient train 或 promotion")
    if manifest.get("u1_general_dev_training_validation_rows") != 800:
        raise ManifestError("U2 训练期验证必须精确绑定已花费的 U1 dev 800 条")
    if manifest.get(
        "u1_general_dev_training_validation_category_counts"
    ) != {"math": 400, "natural_language_reasoning": 400}:
        raise ManifestError("U1 dev 训练期验证类别必须为 math/logic 各 400 条")
    provenance = manifest.get("provenance", manifest.get("sources"))
    if not isinstance(provenance, Mapping) or not provenance:
        raise ManifestError("U2 manifest 缺少非空 provenance/sources")
    return isolation


def validate_v2_rows(
    rows: Sequence[Mapping[str, Any]], split: str
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Reuse U1 contracts and validate the optional math hardness field."""

    counts = u1.validate_rows(rows, split)
    math_weights: List[float] = []
    explicit_math_weights = 0
    for index, row in enumerate(rows, start=1):
        category = str(row["category"])
        location = f"{split}[{index}]"
        raw_weight = row.get("hardness_weight", 1.0)
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"{location} hardness_weight 不是数值") from exc
        if not math.isfinite(weight) or weight <= 0.0:
            raise ManifestError(f"{location} hardness_weight 必须为有限正数")
        if category != "math" and "hardness_weight" in row and weight != 1.0:
            raise ManifestError(
                f"{location} 只有 math 样本可设置非 1 hardness_weight"
            )
        if category == "math":
            math_weights.append(weight)
            explicit_math_weights += int("hardness_weight" in row)
    return counts, {
        "math_rows": len(math_weights),
        "explicit_math_hardness_rows": explicit_math_weights,
        "math_hardness_min": min(math_weights),
        "math_hardness_max": max(math_weights),
        "math_hardness_mean": round(sum(math_weights) / len(math_weights), 6),
    }


def tokenize_v2_rows(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any, max_seq_length: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tokenized, stats = u1.tokenize_rows(rows, tokenizer, max_seq_length)
    for source, target in zip(rows, tokenized):
        first_target = next(
            (index for index, label in enumerate(target["labels"]) if label != -100),
            None,
        )
        if first_target is None or first_target <= 0:
            raise ManifestError(f"样本 {source['event_id']} 首个监督 token 位置无效")
        target["_first_target_index"] = int(first_target)
        target["_hardness_weight"] = float(source.get("hardness_weight", 1.0))
    return tokenized, stats


def parse_task_weights(value: str) -> Dict[str, int]:
    return u1.parse_task_weights(value)


def validate_u2_training_contract(
    *,
    bf16: bool,
    fp16: bool,
    weights: Mapping[str, int],
    manifest: Mapping[str, Any],
) -> None:
    """Keep the one allowed U2 recipe identical to the frozen manifest."""

    if not bf16 or fp16:
        raise ManifestError("U2 正式训练必须且只能使用 BF16")
    declared = manifest.get("recommended_effective_task_weights")
    if not isinstance(declared, Mapping):
        raise ManifestError("U2 manifest 缺少冻结任务权重")
    normalized = {str(key): int(value) for key, value in declared.items()}
    if dict(weights) != normalized:
        raise ManifestError("U2 任务权重与冻结 manifest 不一致")


def _weighted_draws(
    indices: Sequence[int],
    weights: Sequence[float],
    count: int,
    seed: int,
) -> List[int]:
    """Make deterministic weighted draws without relying on global RNG state."""

    if len(indices) != len(weights) or not indices:
        raise ManifestError("math hardness 加权抽样输入长度无效")
    if count < 0:
        raise ManifestError("math hardness 抽样数量不能为负")
    numeric = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) or weight <= 0.0 for weight in numeric):
        raise ManifestError("math hardness 抽样权重必须为有限正数")
    cumulative: List[float] = []
    running = 0.0
    for weight in numeric:
        running += weight
        cumulative.append(running)
    rng = random.Random(seed)
    output: List[int] = []
    for _ in range(count):
        position = rng.random() * running
        selected = bisect_left(cumulative, position)
        output.append(int(indices[min(selected, len(indices) - 1)]))
    return output


def _uniform_cycle_draws(indices: Sequence[int], count: int, seed: int) -> List[int]:
    if not indices:
        raise ManifestError("U2 类内抽样不能使用空类别")
    rng = random.Random(seed)
    output: List[int] = []
    while len(output) < count:
        cycle = list(indices)
        rng.shuffle(cycle)
        output.extend(cycle[: count - len(output)])
    return output


def _coverage_then_weighted_draws(
    indices: Sequence[int],
    weights: Sequence[float],
    count: int,
    seed: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """Cover every math row once before applying non-uniform hardness draws.

    ``build_u2_sampling_indices`` always allocates at least the number of unique
    rows in every category.  Equal math weights therefore use the same shuffled
    cycle sampler as the other categories.  When explicit hardness differs, one
    shuffled full-coverage cycle is emitted first and only the remaining quota
    is sampled with replacement.  This prevents a hardness policy from silently
    dropping part of the frozen math corpus for an entire epoch.
    """

    if len(indices) != len(weights) or not indices:
        raise ManifestError("math hardness 覆盖抽样输入长度无效")
    if count < len(indices):
        raise ManifestError("math 类别配额不足以覆盖全部冻结样本")
    numeric = [float(weight) for weight in weights]
    equal_weights = all(
        math.isclose(weight, numeric[0], rel_tol=0.0, abs_tol=1e-12)
        for weight in numeric[1:]
    )
    if equal_weights:
        selected = _uniform_cycle_draws(indices, count, seed)
        return selected, {
            "mode": "uniform_cycle_equal_weights",
            "mandatory_coverage_rows": len(indices),
            "additional_weighted_draws": 0,
            "additional_draws_with_replacement": False,
        }

    covered = _uniform_cycle_draws(indices, len(indices), seed)
    remaining = count - len(covered)
    additional = _weighted_draws(indices, numeric, remaining, seed + 1)
    return covered + additional, {
        "mode": "full_coverage_then_weighted_remainder",
        "mandatory_coverage_rows": len(indices),
        "additional_weighted_draws": remaining,
        "additional_draws_with_replacement": remaining > 0,
    }


def build_u2_sampling_indices(
    rows: Sequence[Mapping[str, Any]], weights: Mapping[str, int], seed: int
) -> Tuple[List[int], Dict[str, Any]]:
    """Apply task ratios, then math hardness only within the math class."""

    groups: Dict[str, List[int]] = {category: [] for category in CATEGORIES}
    for index, row in enumerate(rows):
        category = str(row.get("_category", ""))
        if category not in groups:
            raise ManifestError(f"U2 token 类别未知: {category}")
        groups[category].append(index)
    if any(not values for values in groups.values()):
        raise ManifestError("U2 训练 token 中缺少任务类别")
    unit = max(
        math.ceil(len(groups[category]) / int(weights[category]))
        for category in CATEGORIES
    )
    category_seeds = {
        "traffic_action": seed + 101,
        "math": seed + 211,
        "natural_language_reasoning": seed + 307,
    }
    effective = {
        category: unit * int(weights[category]) for category in CATEGORIES
    }
    selected_by_category: Dict[str, List[int]] = {}
    math_sampling_policy: Dict[str, Any] = {}
    for category in CATEGORIES:
        if category == "math":
            selected_by_category[category], math_sampling_policy = (
                _coverage_then_weighted_draws(
                groups[category],
                [float(rows[index].get("_hardness_weight", 1.0)) for index in groups[category]],
                effective[category],
                category_seeds[category],
                )
            )
        else:
            selected_by_category[category] = _uniform_cycle_draws(
                groups[category], effective[category], category_seeds[category]
            )
    output = [
        index
        for category in CATEGORIES
        for index in selected_by_category[category]
    ]
    random.Random(seed + 401).shuffle(output)
    math_unique_weights = [
        float(rows[index].get("_hardness_weight", 1.0)) for index in groups["math"]
    ]
    math_selected_weights = [
        float(rows[index].get("_hardness_weight", 1.0))
        for index in selected_by_category["math"]
    ]
    return output, {
        "algorithm": "task_ratio_then_math_full_coverage_v2",
        "seed": int(seed),
        "weights": dict(weights),
        "unique_category_counts": {
            category: len(groups[category]) for category in CATEGORIES
        },
        "effective_category_counts": effective,
        "effective_rows_per_epoch": len(output),
        "unique_rows": len(rows),
        "effective_ratios": {
            category: round(effective[category] / len(output), 6)
            for category in CATEGORIES
        },
        "selected_unique_rows": {
            category: len(set(selected_by_category[category]))
            for category in CATEGORIES
        },
        "math_hardness_sampling": {
            **math_sampling_policy,
            "unique_weight_mean": round(
                sum(math_unique_weights) / len(math_unique_weights), 6
            ),
            "selected_weight_mean": round(
                sum(math_selected_weights) / len(math_selected_weights), 6
            ),
            "unique_weight_min": min(math_unique_weights),
            "unique_weight_max": max(math_unique_weights),
        },
    }


def resolve_allowed_token_ids(tokenizer: Any) -> Dict[int, Tuple[int, ...]]:
    """Resolve and validate the single-token output slots once before training."""

    output: Dict[int, Tuple[int, ...]] = {}
    for category, labels in RESTRICTED_LABELS.items():
        token_ids: List[int] = []
        for label in labels:
            ids = tokenizer(label, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ManifestError(f"U2 槽位标签 {category}:{label} 不是单 token")
            token_ids.append(int(ids[0]))
        if len(set(token_ids)) != len(token_ids):
            raise ManifestError(f"U2 槽位标签 {category} token id 不唯一")
        output[CATEGORY_IDS[category]] = tuple(token_ids)
    return output


def restricted_slot_losses(
    logits: Any,
    labels: Any,
    category_ids: Any,
    first_target_indices: Any,
    allowed_token_ids: Mapping[int, Sequence[int]],
) -> Tuple[Any, Any, int]:
    """Return allowed-slot CE and outside-slot mass for restricted rows.

    For a causal LM, label position ``p`` is predicted by logits at ``p - 1``.
    Math rows have no restricted slot and do not enter either mean.
    """

    import torch
    import torch.nn.functional as functional

    restricted_ce: List[Any] = []
    outside_mass: List[Any] = []
    batch_size, sequence_length, _ = logits.shape
    if labels.shape[:2] != (batch_size, sequence_length):
        raise ManifestError("U2 restricted loss 的 logits/labels 形状不一致")
    for row_index in range(batch_size):
        category_id = int(category_ids[row_index].item())
        if category_id not in allowed_token_ids:
            continue
        target_position = int(first_target_indices[row_index].item())
        if target_position <= 0 or target_position >= sequence_length:
            raise ManifestError("U2 restricted loss 首监督位置越界")
        target_id = int(labels[row_index, target_position].item())
        allowed = tuple(int(value) for value in allowed_token_ids[category_id])
        if target_id not in allowed:
            raise ManifestError("U2 restricted loss 目标 token 不在允许槽位")
        slot_logits = logits[row_index, target_position - 1]
        allowed_tensor = torch.tensor(allowed, device=slot_logits.device, dtype=torch.long)
        allowed_logits = slot_logits.index_select(0, allowed_tensor)
        target_slot = allowed.index(target_id)
        restricted_ce.append(
            functional.cross_entropy(
                allowed_logits.unsqueeze(0),
                torch.tensor([target_slot], device=slot_logits.device),
            )
        )
        allowed_log_mass = torch.logsumexp(allowed_logits, dim=0)
        total_log_mass = torch.logsumexp(slot_logits, dim=0)
        allowed_mass = torch.exp(allowed_log_mass - total_log_mass)
        outside_mass.append((1.0 - allowed_mass).clamp(min=0.0, max=1.0))
    zero = logits.sum() * 0.0
    if not restricted_ce:
        return zero, zero, 0
    return torch.stack(restricted_ce).mean(), torch.stack(outside_mass).mean(), len(
        restricted_ce
    )


class U2TokenizedDataset:
    def __init__(
        self, rows: Sequence[Dict[str, Any]], indices: Optional[Sequence[int]] = None
    ) -> None:
        self.rows = list(rows)
        self.indices = list(indices) if indices is not None else list(range(len(rows)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[self.indices[index]]
        return {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": row["labels"],
            "u2_category_id": CATEGORY_IDS[str(row["_category"])],
            "u2_first_target_index": int(row["_first_target_index"]),
        }


class U2Collator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        maximum = max(len(row["input_ids"]) for row in rows)
        return {
            "input_ids": torch.tensor(
                [
                    row["input_ids"]
                    + [self.pad_token_id] * (maximum - len(row["input_ids"]))
                    for row in rows
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [
                    row["attention_mask"]
                    + [0] * (maximum - len(row["attention_mask"]))
                    for row in rows
                ],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [
                    row["labels"] + [-100] * (maximum - len(row["labels"]))
                    for row in rows
                ],
                dtype=torch.long,
            ),
            "u2_category_id": torch.tensor(
                [row["u2_category_id"] for row in rows], dtype=torch.long
            ),
            "u2_first_target_index": torch.tensor(
                [row["u2_first_target_index"] for row in rows], dtype=torch.long
            ),
        }


def verify_u1_adapter(
    adapter: Path,
    expected_weights_sha256: str,
    expected_config_sha256: str,
    expected_base_model_id: str,
) -> Dict[str, str]:
    """Reject a non-U1 or unregistered starting adapter before model loading."""

    adapter = adapter.resolve()
    if SHA256_PATTERN.fullmatch(expected_weights_sha256) is None:
        raise ManifestError("expected_u1_adapter_sha256 必须是 64 位小写 SHA-256")
    if SHA256_PATTERN.fullmatch(expected_config_sha256) is None:
        raise ManifestError(
            "expected_u1_adapter_config_sha256 必须是 64 位小写 SHA-256"
        )
    weights_path = adapter / "adapter_model.safetensors"
    config_path = adapter / "adapter_config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise ManifestError("U1 起始 LoRA 缺少 adapter 权重或配置")
    actual = sha256_file(weights_path)
    if actual != expected_weights_sha256:
        raise ManifestError("U1 起始 LoRA 权重不等于预注册 SHA-256")
    actual_config_sha256 = sha256_file(config_path)
    if actual_config_sha256 != expected_config_sha256:
        raise ManifestError("U1 起始 LoRA 配置不等于预注册 SHA-256")
    config = read_json_object(config_path)
    if config.get("base_model_name_or_path") != expected_base_model_id:
        raise ManifestError("U1 起始 LoRA 与锁定基座不兼容")
    return {
        "path": str(adapter),
        "weights_sha256": actual,
        "config_sha256": actual_config_sha256,
    }


def _training_arguments(args: argparse.Namespace, output: Path) -> Dict[str, Any]:
    from transformers import TrainingArguments

    values: Dict[str, Any] = {
        "output_dir": str(output),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "report_to": "none",
        "remove_unused_columns": False,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": args.seed,
        "dataloader_num_workers": 0,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "train_sampling_strategy" in signature.parameters:
        values["train_sampling_strategy"] = "group_by_length"
    elif "group_by_length" in signature.parameters:
        values["group_by_length"] = True
    values[
        "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    ] = "epoch"
    if args.bf16:
        values["bf16"] = True
    elif args.fp16:
        values["fp16"] = True
    return values


def _make_u2_trainer_class(trainer_base: Any) -> Any:
    class U2RestrictedTrainer(trainer_base):
        def __init__(
            self,
            *trainer_args: Any,
            allowed_token_ids: Mapping[int, Sequence[int]],
            lm_ce_weight: float,
            restricted_ce_weight: float,
            outside_mass_weight: float,
            **trainer_kwargs: Any,
        ) -> None:
            super().__init__(*trainer_args, **trainer_kwargs)
            self.u2_allowed_token_ids = dict(allowed_token_ids)
            self.u2_loss_weights = {
                "standard_lm_ce": float(lm_ce_weight),
                "restricted_slot_ce": float(restricted_ce_weight),
                "outside_allowed_mass": float(outside_mass_weight),
            }
            self.u2_last_loss_components: Dict[str, float] = {}

        def compute_loss(
            self,
            model: Any,
            inputs: Dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Optional[Any] = None,
        ) -> Any:
            del num_items_in_batch
            category_ids = inputs.pop("u2_category_id")
            first_target_indices = inputs.pop("u2_first_target_index")
            labels = inputs["labels"]
            outputs = model(**inputs)
            lm_loss = outputs.loss
            restricted_ce, outside_mass, restricted_rows = restricted_slot_losses(
                outputs.logits,
                labels,
                category_ids,
                first_target_indices,
                self.u2_allowed_token_ids,
            )
            total = (
                self.u2_loss_weights["standard_lm_ce"] * lm_loss
                + self.u2_loss_weights["restricted_slot_ce"] * restricted_ce
                + self.u2_loss_weights["outside_allowed_mass"] * outside_mass
            )
            self.u2_last_loss_components = {
                # Keep scalar tensors detached and convert only once when the final
                # receipt is written. Calling .item() on every CUDA step would add
                # an unnecessary device synchronization to the training hot path.
                "standard_lm_ce": lm_loss.detach(),
                "restricted_slot_ce": restricted_ce.detach(),
                "outside_allowed_mass": outside_mass.detach(),
                "restricted_rows": int(restricted_rows),
                "combined": total.detach(),
            }
            return (total, outputs) if return_outputs else total

    return U2RestrictedTrainer


def _model_report(model: Any) -> Dict[str, Any]:
    return u1._model_report(model)


def training_script_identity() -> Dict[str, Any]:
    path = Path(__file__).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_training_script_stability(
    before: Mapping[str, Any]
) -> Dict[str, Any]:
    after = training_script_identity()
    if dict(before) != after:
        raise ManifestError("U2 训练脚本在训练前后发生漂移")
    return {
        "script": after,
        "attestation_before": dict(before),
        "attestation_after": after,
        "stable_across_training": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    training_script_before = training_script_identity()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--snapshot_manifest", required=True)
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument("--expected_dataset_manifest_sha256", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--u1_adapter", required=True)
    parser.add_argument("--expected_u1_adapter_sha256", required=True)
    parser.add_argument("--expected_u1_adapter_config_sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument(
        "--task_weights",
        default="traffic_action=5,math=2,natural_language_reasoning=3",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--lm_ce_weight", type=float, default=1.0)
    parser.add_argument("--restricted_ce_weight", type=float, default=1.0)
    parser.add_argument("--outside_mass_weight", type=float, default=0.25)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args(argv)
    if args.bf16 and args.fp16:
        raise ManifestError("bf16 与 fp16 不能同时启用")
    positive = (
        args.max_seq_length,
        args.epochs,
        args.batch_size,
        args.eval_batch_size,
        args.gradient_accumulation,
        args.learning_rate,
        args.lm_ce_weight,
        args.restricted_ce_weight,
        args.outside_mass_weight,
    )
    if min(positive) <= 0:
        raise ManifestError("U2 训练长度、轮数、batch、学习率和 loss 权重必须为正")
    weights = parse_task_weights(args.task_weights)

    base_path = Path(args.base).resolve()
    base = validate_base_manifest(read_json_object(base_path))
    snapshot = Path(args.snapshot).resolve()
    snapshot_report = verify_text_snapshot(
        base,
        read_json_object(Path(args.snapshot_manifest)),
        snapshot,
        verify_tokenizer=True,
    )
    manifest_path = Path(args.dataset_manifest).resolve()
    if SHA256_PATTERN.fullmatch(args.expected_dataset_manifest_sha256) is None:
        raise ManifestError(
            "expected_dataset_manifest_sha256 必须是 64 位小写 SHA-256"
        )
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != args.expected_dataset_manifest_sha256:
        raise ManifestError("U2 数据 manifest 不等于预注册 SHA-256")
    manifest = read_json_object(manifest_path)
    isolation = validate_v2_manifest(manifest)
    validate_u2_training_contract(
        bf16=bool(args.bf16),
        fp16=bool(args.fp16),
        weights=weights,
        manifest=manifest,
    )
    train_path = Path(args.train_jsonl).resolve()
    val_path = Path(args.val_jsonl).resolve()
    if train_path == val_path:
        raise ManifestError("U2 训练集和验证集不能相同")
    artifacts = manifest.get("artifacts", {})
    if artifacts.get("train", {}).get("sha256") != sha256_file(train_path):
        raise ManifestError("U2 训练数据 SHA-256 与 manifest 不一致")
    if artifacts.get("validation", {}).get("sha256") != sha256_file(val_path):
        raise ManifestError("U2 验证数据 SHA-256 与 manifest 不一致")
    promotion_artifact = artifacts.get("general_dev_evaluation", {})
    promotion_relative = promotion_artifact.get("path")
    promotion_sha256 = promotion_artifact.get("sha256")
    if not isinstance(promotion_relative, str) or SHA256_PATTERN.fullmatch(
        str(promotion_sha256)
    ) is None:
        raise ManifestError("U2 manifest 缺少 promotion dev 路径或 SHA-256")
    promotion_path = (manifest_path.parent / promotion_relative).resolve()
    if promotion_path in {train_path, val_path}:
        raise ManifestError("U2 promotion dev 不能与训练/验证文件相同")
    if manifest.get("promotion_dev_rows") != 800 or manifest.get(
        "promotion_dev_category_counts"
    ) != {"math": 400, "natural_language_reasoning": 400}:
        raise ManifestError("U2 promotion dev 声明必须是独立的 math/logic 各 400 条")
    train_rows = u1._read_jsonl(train_path)
    val_rows = u1._read_jsonl(val_path)
    train_counts, train_hardness = validate_v2_rows(train_rows, "train")
    val_counts, val_hardness = validate_v2_rows(val_rows, "validation")
    if train_counts != manifest.get("train_category_counts"):
        raise ManifestError("U2 训练类别计数与 manifest 不一致")
    if val_counts != manifest.get("validation_category_counts"):
        raise ManifestError("U2 验证类别计数与 manifest 不一致")
    train_general = {
        str(row["prompt_fingerprint"])
        for row in train_rows
        if row["category"] != "traffic_action"
    }
    val_general = {
        str(row["prompt_fingerprint"])
        for row in val_rows
        if row["category"] != "traffic_action"
    }
    if train_general & val_general:
        raise ManifestError("U2 通用训练集和验证集 prompt fingerprint 重叠")

    start_report = verify_u1_adapter(
        Path(args.u1_adapter),
        args.expected_u1_adapter_sha256,
        args.expected_u1_adapter_config_sha256,
        str(base["source"]["model_id"]),
    )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, use_fast=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    allowed_token_ids = resolve_allowed_token_ids(tokenizer)
    train_tokens, train_token_stats = tokenize_v2_rows(
        train_rows, tokenizer, args.max_seq_length
    )
    val_tokens, val_token_stats = tokenize_v2_rows(
        val_rows, tokenizer, args.max_seq_length
    )
    train_indices, sampling_report = build_u2_sampling_indices(
        train_tokens, weights, args.seed
    )

    model_kwargs: Dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["dtype"] = torch.bfloat16 if args.bf16 else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(str(snapshot), **model_kwargs)
    model_report = _model_report(base_model)
    if model_report["parameter_count"] != int(base["model"]["parameter_count"]):
        raise ManifestError("U2 模型基座参数量与清单不一致")
    model = PeftModel.from_pretrained(
        base_model, start_report["path"], is_trainable=True
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ManifestError("拒绝覆盖非空 U2 模型训练目录")
    output.mkdir(parents=True, exist_ok=True)
    trainer_class = _make_u2_trainer_class(Trainer)
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": TrainingArguments(**_training_arguments(args, output)),
        "train_dataset": U2TokenizedDataset(train_tokens, train_indices),
        "eval_dataset": U2TokenizedDataset(val_tokens),
        "data_collator": U2Collator(tokenizer.pad_token_id),
        "allowed_token_ids": allowed_token_ids,
        "lm_ce_weight": args.lm_ce_weight,
        "restricted_ce_weight": args.restricted_ce_weight,
        "outside_mass_weight": args.outside_mass_weight,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = trainer_class(**trainer_kwargs)
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))

    output_config_path = output / "adapter_config.json"
    output_config = read_json_object(output_config_path)
    output_config["base_model_name_or_path"] = base["source"]["model_id"]
    output_config["revision"] = base["source"]["revision"]
    write_json_object(output_config_path, output_config)
    cuda_memory = None
    if torch.cuda.is_available():
        cuda_memory = {
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 3),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 3),
        }
    provenance = manifest.get("provenance", manifest.get("sources"))
    last_loss_components = {
        key: (
            float(value.float().item())
            if hasattr(value, "float") and hasattr(value, "item")
            else value
        )
        for key, value in trainer.u2_last_loss_components.items()
    }
    training_implementation = verify_training_script_stability(
        training_script_before
    )
    summary = {
        "task": "single_model_traffic_math_chinese_logic_lora_u2",
        "candidate_id": "unified-v2-U2",
        "base_id": base["base_id"],
        "base_fingerprint": base_fingerprint(base),
        "snapshot_validation": snapshot_report,
        "training_implementation": training_implementation,
        "dataset": {
            "schema_version": SCHEMA_VERSION,
            "manifest_path": str(manifest_path),
            "manifest_sha256": actual_manifest_sha256,
            "train_path": str(train_path),
            "train_sha256": sha256_file(train_path),
            "validation_path": str(val_path),
            "validation_sha256": sha256_file(val_path),
            "promotion_dev_declared_path": str(promotion_path),
            "promotion_dev_declared_sha256": str(promotion_sha256),
            "promotion_dev_content_read_by_trainer": False,
            "provenance": provenance,
            "isolation": isolation,
        },
        "source_u1_adapter": start_report,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_category_counts": train_counts,
        "validation_category_counts": val_counts,
        "train_hardness": train_hardness,
        "validation_hardness": val_hardness,
        "train_tokenization": train_token_stats,
        "validation_tokenization": val_token_stats,
        "task_sampling": sampling_report,
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "logging_steps": args.logging_steps,
            "max_seq_length": args.max_seq_length,
            "candidate_output_directory": str(output),
            "bf16": bool(args.bf16),
            "fp16": bool(args.fp16),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "loss_weights": trainer.u2_loss_weights,
            "restricted_slot_labels": {
                category: list(labels) for category, labels in RESTRICTED_LABELS.items()
            },
            "restricted_slot_token_ids": {
                str(category_id): list(token_ids)
                for category_id, token_ids in allowed_token_ids.items()
            },
            "last_observed_loss_components": last_loss_components,
        },
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "cuda_memory": cuda_memory,
        "adapter_artifact": {
            "path": "adapter_model.safetensors",
            "sha256": sha256_file(output / "adapter_model.safetensors"),
        },
        "promotion_status": "unvalidated",
        "production_traffic_release_modified": False,
    }
    write_json_object(output / "train_metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
