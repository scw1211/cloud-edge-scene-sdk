"""Write a hash-anchored receipt for an already completed U2 adapter.

This module is deliberately a recorder, not a training or evaluation runner.  It
only reads the candidate's three required files and the explicitly supplied
anchors.  In particular, it does not infer or report a training process exit
code because that fact is not observable from an adapter directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence

from edge_llm_factory.contracts import ManifestError, canonical_sha256


SCHEMA_VERSION = "edge-llm-u2-candidate-artifact-receipt/v1"
U2_TASK = "single_model_traffic_math_chinese_logic_lora_u2"
DATASET_SCHEMA_VERSION = "edge-llm-unified-traffic-general/v2"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_ARTIFACTS = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "train_metrics.json",
)
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
TASK_WEIGHT_CATEGORIES = {
    "traffic_action",
    "math",
    "natural_language_reasoning",
}
LOSS_WEIGHT_FIELDS = {
    "standard_lm_ce",
    "restricted_slot_ce",
    "outside_allowed_mass",
}


def _required_sha256(value: str, field: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ManifestError(f"{field} 必须是 64 位小写 SHA-256")
    return value


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{field} 必须是对象")
    return value


def _read_file(path: Path, label: str) -> tuple[bytes, Dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ManifestError(f"{label} 不存在或不是文件: {resolved}")
    try:
        content = resolved.read_bytes()
    except OSError as exc:
        raise ManifestError(f"无法读取 {label}: {resolved}: {exc}") from exc
    return content, {
        "path": str(resolved),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _read_json(path: Path, label: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    content, identity = _read_file(path, label)
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{label} 不是有效 UTF-8 JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{label} 顶层必须是对象")
    return value, identity


def _anchored_file(path: Path, expected_sha256: str, label: str) -> Dict[str, Any]:
    expected = _required_sha256(expected_sha256, f"expected_{label}_sha256")
    _, identity = _read_file(path, label)
    if identity["sha256"] != expected:
        raise ManifestError(f"{label} 与显式 SHA-256 锚点不一致")
    return {**identity, "expected_sha256": expected, "verified": True}


def _anchored_json(
    path: Path, expected_sha256: str, label: str
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    expected = _required_sha256(expected_sha256, f"expected_{label}_sha256")
    value, identity = _read_json(path, label)
    if identity["sha256"] != expected:
        raise ManifestError(f"{label} 与显式 SHA-256 锚点不一致")
    return value, {**identity, "expected_sha256": expected, "verified": True}


def _positive_finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{field} 必须是有限正数")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ManifestError(f"{field} 必须是有限正数")
    return numeric


def _nonnegative_finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{field} 必须是有限非负数")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ManifestError(f"{field} 必须是有限非负数")
    return numeric


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError(f"{field} 必须是正整数")
    return value


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestError(f"{field} 必须是非负整数")
    return value


def _same_numeric(expected: Any, actual: Any, field: str, *, positive: bool) -> float:
    validator = _positive_finite if positive else _nonnegative_finite
    expected_number = validator(expected, f"u2_preregistration.{field}")
    actual_number = validator(actual, f"train_metrics.{field}")
    if actual_number != expected_number:
        raise ManifestError(f"U2 实际训练参数与冻结预注册不一致: {field}")
    return actual_number


def _verify_training_completion(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    if metrics.get("task") != U2_TASK:
        raise ManifestError("train_metrics.json 不是 U2 统一训练记录")
    if metrics.get("candidate_id") != "unified-v2-U2":
        raise ManifestError("train_metrics.json candidate_id 不是 unified-v2-U2")
    train_metrics = _object(metrics.get("train_metrics"), "train_metrics.train_metrics")
    runtime = _positive_finite(
        train_metrics.get("train_runtime"), "train_metrics.train_metrics.train_runtime"
    )
    loss = train_metrics.get("train_loss")
    if isinstance(loss, bool):
        raise ManifestError("train_metrics.train_metrics.train_loss 必须是有限数值")
    try:
        loss_value = float(loss)
    except (TypeError, ValueError) as exc:
        raise ManifestError(
            "train_metrics.train_metrics.train_loss 必须是有限数值"
        ) from exc
    if not math.isfinite(loss_value):
        raise ManifestError("train_metrics.train_metrics.train_loss 必须是有限数值")
    completed_epoch = _positive_finite(
        train_metrics.get("epoch"), "train_metrics.train_metrics.epoch"
    )
    optimization = _object(metrics.get("optimization"), "train_metrics.optimization")
    requested_epochs = _positive_finite(
        optimization.get("epochs"), "train_metrics.optimization.epochs"
    )
    if completed_epoch + 1e-9 < requested_epochs:
        raise ManifestError("U2 训练记录的完成 epoch 小于冻结训练轮数")
    if metrics.get("promotion_status") != "unvalidated":
        raise ManifestError("U2 候选在产物收据阶段必须仍为 unvalidated")
    if metrics.get("production_traffic_release_modified") is not False:
        raise ManifestError("U2 训练不得修改生产交通发布")
    return {
        "confirmed": True,
        "observed_train_runtime_seconds": runtime,
        "observed_train_loss": loss_value,
        "completed_epoch": completed_epoch,
        "requested_epochs": requested_epochs,
    }


def _verify_isolation(metrics: Mapping[str, Any]) -> Dict[str, bool]:
    dataset = _object(metrics.get("dataset"), "train_metrics.dataset")
    if dataset.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ManifestError("train_metrics.dataset schema 不是 U2 v2")
    if dataset.get("promotion_dev_content_read_by_trainer") is not False:
        raise ManifestError("U2 trainer 读取了 promotion dev 内容")
    isolation = _object(dataset.get("isolation"), "train_metrics.dataset.isolation")
    result: Dict[str, bool] = {}
    for field in REQUIRED_FALSE_ISOLATION_FIELDS:
        if isolation.get(field) is not False:
            raise ManifestError(f"U2 训练隔离声明必须显式为 false: {field}")
        result[field] = False
    return result


def _verify_script_identity(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    implementation = _object(
        metrics.get("training_implementation"), "train_metrics.training_implementation"
    )
    if implementation.get("stable_across_training") is not True:
        raise ManifestError("U2 训练脚本缺少训练前后稳定声明")
    script = dict(_object(implementation.get("script"), "training_implementation.script"))
    before = dict(
        _object(
            implementation.get("attestation_before"),
            "training_implementation.attestation_before",
        )
    )
    after = dict(
        _object(
            implementation.get("attestation_after"),
            "training_implementation.attestation_after",
        )
    )
    if script != before or script != after:
        raise ManifestError("U2 训练脚本前后身份不一致")
    declared_sha = _required_sha256(str(script.get("sha256", "")), "training script sha256")
    declared_path = script.get("path")
    declared_bytes = script.get("bytes")
    if not isinstance(declared_path, str) or not declared_path:
        raise ManifestError("U2 训练脚本 path 无效")
    if (
        not isinstance(declared_bytes, int)
        or isinstance(declared_bytes, bool)
        or declared_bytes < 0
    ):
        raise ManifestError("U2 训练脚本 bytes 无效")
    _, actual = _read_file(Path(declared_path), "U2 training script")
    if actual["bytes"] != declared_bytes or actual["sha256"] != declared_sha:
        raise ManifestError("U2 训练脚本当前身份与训练记录不一致")
    return {**actual, "stable_across_training": True, "recomputed": True}


def _verify_preregistered_trainer(
    preregistration: Mapping[str, Any], script: Mapping[str, Any]
) -> str:
    repository = _object(
        preregistration.get("repository"), "u2_preregistration.repository"
    )
    implementation = _object(
        repository.get("implementation_sha256"),
        "u2_preregistration.repository.implementation_sha256",
    )
    trainer_sha = _required_sha256(
        str(implementation.get("trainer", "")),
        "u2_preregistration.repository.implementation_sha256.trainer",
    )
    if trainer_sha != script.get("sha256"):
        raise ManifestError("U2 训练脚本 SHA-256 与冻结预注册 trainer 不一致")
    return trainer_sha


def _verify_preregistered_training_recipe(
    preregistration: Mapping[str, Any],
    metrics: Mapping[str, Any],
    candidate: Path,
) -> Dict[str, Any]:
    recipe = _object(
        preregistration.get("training_recipe"),
        "u2_preregistration.training_recipe",
    )
    contract = _object(
        preregistration.get("token_and_sampling_contract"),
        "u2_preregistration.token_and_sampling_contract",
    )
    optimization = _object(metrics.get("optimization"), "train_metrics.optimization")
    sampling = _object(metrics.get("task_sampling"), "train_metrics.task_sampling")

    if recipe.get("precision") != "bfloat16":
        raise ManifestError("U2 冻结预注册 precision 必须是 bfloat16")
    if optimization.get("bf16") is not True or optimization.get("fp16") is not False:
        raise ManifestError("U2 实际训练 precision 与冻结 bfloat16 不一致")

    observed: Dict[str, Any] = {
        "precision": "bfloat16",
        "epochs": _same_numeric(
            recipe.get("epochs"),
            optimization.get("epochs"),
            "training_recipe.epochs",
            positive=True,
        ),
        "batch_size": _positive_integer(
            optimization.get("batch_size"), "train_metrics.optimization.batch_size"
        ),
        "eval_batch_size": _positive_integer(
            optimization.get("eval_batch_size"),
            "train_metrics.optimization.eval_batch_size",
        ),
        "gradient_accumulation": _positive_integer(
            optimization.get("gradient_accumulation"),
            "train_metrics.optimization.gradient_accumulation",
        ),
        "learning_rate": _same_numeric(
            recipe.get("learning_rate"),
            optimization.get("learning_rate"),
            "training_recipe.learning_rate",
            positive=True,
        ),
        "weight_decay": _same_numeric(
            recipe.get("weight_decay"),
            optimization.get("weight_decay"),
            "training_recipe.weight_decay",
            positive=False,
        ),
        "warmup_ratio": _same_numeric(
            recipe.get("warmup_ratio"),
            optimization.get("warmup_ratio"),
            "training_recipe.warmup_ratio",
            positive=False,
        ),
    }
    for field in ("batch_size", "eval_batch_size", "gradient_accumulation"):
        expected = _positive_integer(
            recipe.get(field), f"u2_preregistration.training_recipe.{field}"
        )
        if observed[field] != expected:
            raise ManifestError(
                f"U2 实际训练参数与冻结预注册不一致: training_recipe.{field}"
            )

    logging_steps = _positive_integer(
        optimization.get("logging_steps"),
        "train_metrics.optimization.logging_steps",
    )
    if logging_steps != _positive_integer(
        recipe.get("logging_steps"),
        "u2_preregistration.training_recipe.logging_steps",
    ):
        raise ManifestError(
            "U2 实际训练参数与冻结预注册不一致: training_recipe.logging_steps"
        )
    observed["logging_steps"] = logging_steps

    if not isinstance(recipe.get("gradient_checkpointing"), bool):
        raise ManifestError(
            "u2_preregistration.training_recipe.gradient_checkpointing 必须是布尔值"
        )
    if optimization.get("gradient_checkpointing") is not recipe.get(
        "gradient_checkpointing"
    ):
        raise ManifestError(
            "U2 实际训练参数与冻结预注册不一致: training_recipe.gradient_checkpointing"
        )
    observed["gradient_checkpointing"] = optimization["gradient_checkpointing"]

    expected_losses = _object(
        recipe.get("loss_weights"),
        "u2_preregistration.training_recipe.loss_weights",
    )
    actual_losses = _object(
        optimization.get("loss_weights"),
        "train_metrics.optimization.loss_weights",
    )
    if set(expected_losses) != LOSS_WEIGHT_FIELDS or set(actual_losses) != LOSS_WEIGHT_FIELDS:
        raise ManifestError("U2 loss_weights 字段集合与冻结合同不一致")
    observed_losses: Dict[str, float] = {}
    for field in sorted(LOSS_WEIGHT_FIELDS):
        observed_losses[field] = _same_numeric(
            expected_losses.get(field),
            actual_losses.get(field),
            f"training_recipe.loss_weights.{field}",
            positive=True,
        )
    observed["loss_weights"] = observed_losses

    output_directory = recipe.get("output_directory")
    if not isinstance(output_directory, str) or not output_directory:
        raise ManifestError("u2_preregistration.training_recipe.output_directory 无效")
    preregistered_output = Path(output_directory).expanduser().resolve()
    metric_output = optimization.get("candidate_output_directory")
    if not isinstance(metric_output, str) or not metric_output:
        raise ManifestError("train_metrics.optimization.candidate_output_directory 无效")
    if Path(metric_output).expanduser().resolve() != preregistered_output:
        raise ManifestError("U2 metrics 候选输出目录与冻结预注册不一致")
    if candidate != preregistered_output:
        raise ManifestError("U2 实际 candidate_dir 与冻结预注册输出目录不一致")
    observed["output_directory"] = str(candidate)

    max_seq_length = _positive_integer(
        optimization.get("max_seq_length"),
        "train_metrics.optimization.max_seq_length",
    )
    if max_seq_length != _positive_integer(
        contract.get("max_sequence_length"),
        "u2_preregistration.token_and_sampling_contract.max_sequence_length",
    ):
        raise ManifestError(
            "U2 实际训练参数与冻结预注册不一致: "
            "token_and_sampling_contract.max_sequence_length"
        )
    seed = _nonnegative_integer(
        sampling.get("seed"), "train_metrics.task_sampling.seed"
    )
    if seed != _nonnegative_integer(
        contract.get("seed"), "u2_preregistration.token_and_sampling_contract.seed"
    ):
        raise ManifestError(
            "U2 实际训练参数与冻结预注册不一致: token_and_sampling_contract.seed"
        )

    expected_weights = _object(
        contract.get("task_weights"),
        "u2_preregistration.token_and_sampling_contract.task_weights",
    )
    actual_weights = _object(
        sampling.get("weights"), "train_metrics.task_sampling.weights"
    )
    if (
        set(expected_weights) != TASK_WEIGHT_CATEGORIES
        or set(actual_weights) != TASK_WEIGHT_CATEGORIES
    ):
        raise ManifestError("U2 task_weights 类别集合与冻结合同不一致")
    normalized_weights: Dict[str, int] = {}
    for category in sorted(TASK_WEIGHT_CATEGORIES):
        expected_weight = _positive_integer(
            expected_weights.get(category),
            f"u2_preregistration.token_and_sampling_contract.task_weights.{category}",
        )
        actual_weight = _positive_integer(
            actual_weights.get(category),
            f"train_metrics.task_sampling.weights.{category}",
        )
        if actual_weight != expected_weight:
            raise ManifestError(
                "U2 实际 task_weights 与冻结预注册不一致: " + category
            )
        normalized_weights[category] = actual_weight

    return {
        "exact_match": True,
        "training_recipe": observed,
        "token_and_sampling_contract": {
            "max_seq_length": max_seq_length,
            "seed": seed,
            "task_weights": normalized_weights,
        },
    }


def _verify_dataset(
    metrics: Mapping[str, Any], expected_dataset_manifest_sha256: str
) -> Dict[str, Any]:
    expected = _required_sha256(
        expected_dataset_manifest_sha256, "expected_dataset_manifest_sha256"
    )
    dataset = _object(metrics.get("dataset"), "train_metrics.dataset")
    if dataset.get("manifest_sha256") != expected:
        raise ManifestError("train_metrics 中的数据 manifest SHA-256 与显式锚点不一致")
    manifest_path = dataset.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise ManifestError("train_metrics 缺少数据 manifest path")
    return _anchored_file(Path(manifest_path), expected, "dataset_manifest")


def _verify_u1_start(
    metrics: Mapping[str, Any],
    expected_weights_sha256: str,
    expected_config_sha256: str,
) -> Dict[str, Any]:
    expected_weights = _required_sha256(
        expected_weights_sha256, "expected_u1_weights_sha256"
    )
    expected_config = _required_sha256(
        expected_config_sha256, "expected_u1_config_sha256"
    )
    source = _object(metrics.get("source_u1_adapter"), "train_metrics.source_u1_adapter")
    source_path = source.get("path")
    if not isinstance(source_path, str) or not source_path:
        raise ManifestError("train_metrics 缺少 U1 adapter path")
    if source.get("weights_sha256") != expected_weights:
        raise ManifestError("train_metrics 中的 U1 权重 SHA-256 与显式锚点不一致")
    if source.get("config_sha256") != expected_config:
        raise ManifestError("train_metrics 中的 U1 配置 SHA-256 与显式锚点不一致")
    root = Path(source_path)
    return {
        "path": str(root.resolve()),
        "adapter_weights": _anchored_file(
            root / "adapter_model.safetensors", expected_weights, "u1_weights"
        ),
        "adapter_config": _anchored_file(
            root / "adapter_config.json", expected_config, "u1_config"
        ),
    }


def build_receipt(
    *,
    candidate_dir: Path,
    u2_preregistration: Path,
    expected_u2_preregistration_sha256: str,
    u2_protocol: Path,
    expected_u2_protocol_sha256: str,
    expected_dataset_manifest_sha256: str,
    expected_u1_weights_sha256: str,
    expected_u1_config_sha256: str,
) -> Dict[str, Any]:
    candidate = candidate_dir.resolve()
    if not candidate.is_dir():
        raise ManifestError(f"U2 candidate_dir 不存在或不是目录: {candidate}")

    artifact_files: Dict[str, Dict[str, Any]] = {}
    metrics: Optional[Dict[str, Any]] = None
    for name in REQUIRED_ARTIFACTS:
        path = candidate / name
        if name == "train_metrics.json":
            metrics, identity = _read_json(path, name)
        else:
            _, identity = _read_file(path, name)
        artifact_files[name] = identity
    assert metrics is not None

    declared_artifact = _object(
        metrics.get("adapter_artifact"), "train_metrics.adapter_artifact"
    )
    if declared_artifact.get("path") != "adapter_model.safetensors":
        raise ManifestError("train_metrics adapter_artifact.path 无效")
    if declared_artifact.get("sha256") != artifact_files["adapter_model.safetensors"]["sha256"]:
        raise ManifestError("train_metrics adapter_artifact SHA-256 与候选权重不一致")

    canonical_artifact_sha = canonical_sha256(
        {
            name: {"bytes": identity["bytes"], "sha256": identity["sha256"]}
            for name, identity in sorted(artifact_files.items())
        }
    )
    completion = _verify_training_completion(metrics)
    isolation = _verify_isolation(metrics)
    script = _verify_script_identity(metrics)
    preregistration, preregistration_identity = _anchored_json(
        u2_preregistration,
        expected_u2_preregistration_sha256,
        "u2_preregistration",
    )
    preregistered_trainer_sha = _verify_preregistered_trainer(
        preregistration, script
    )
    recipe_binding = _verify_preregistered_training_recipe(
        preregistration, metrics, candidate
    )
    dataset = _verify_dataset(metrics, expected_dataset_manifest_sha256)
    u1_start = _verify_u1_start(
        metrics, expected_u1_weights_sha256, expected_u1_config_sha256
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": "u2_candidate_artifact",
        "candidate_id": "unified-v2-U2",
        "created_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z"),
        "candidate": {
            "path": str(candidate),
            "files": artifact_files,
            "canonical_artifact_sha256": canonical_artifact_sha,
        },
        "anchors": {
            "u2_preregistration": {
                **preregistration_identity,
                "trainer_sha256": preregistered_trainer_sha,
            },
            "u2_protocol": _anchored_file(
                u2_protocol, expected_u2_protocol_sha256, "u2_protocol"
            ),
            "dataset_manifest": dataset,
            "source_u1_adapter": u1_start,
        },
        "training_evidence": {
            "completion": completion,
            "isolation": isolation,
            "training_script": script,
            "preregistered_recipe_binding": recipe_binding,
        },
    }


def write_receipt_exclusive(path: Path, receipt: Mapping[str, Any]) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(dict(receipt), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ManifestError(f"拒绝覆盖已有 U2 候选收据: {output}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            output.unlink()
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--u2-preregistration", required=True)
    parser.add_argument("--expected-u2-preregistration-sha256", required=True)
    parser.add_argument("--u2-protocol", required=True)
    parser.add_argument("--expected-u2-protocol-sha256", required=True)
    parser.add_argument("--expected-dataset-manifest-sha256", required=True)
    parser.add_argument("--expected-u1-weights-sha256", required=True)
    parser.add_argument("--expected-u1-config-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    receipt = build_receipt(
        candidate_dir=Path(args.candidate_dir),
        u2_preregistration=Path(args.u2_preregistration),
        expected_u2_preregistration_sha256=args.expected_u2_preregistration_sha256,
        u2_protocol=Path(args.u2_protocol),
        expected_u2_protocol_sha256=args.expected_u2_protocol_sha256,
        expected_dataset_manifest_sha256=args.expected_dataset_manifest_sha256,
        expected_u1_weights_sha256=args.expected_u1_weights_sha256,
        expected_u1_config_sha256=args.expected_u1_config_sha256,
    )
    write_receipt_exclusive(Path(args.output), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
