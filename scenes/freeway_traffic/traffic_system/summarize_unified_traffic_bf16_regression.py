"""汇总正式交通 LoRA 与统一候选 LoRA 的配对 BF16 非回归证据。

本工具不加载模型，也不执行推理。它只消费
``eval_llm_sft_student.py`` 对同一冻结交通测试集生成的两份 JSON，重新从
逐事件记录计算全部指标，并把测试集和两边 ``adapter_model.safetensors`` 的
SHA-256 锁进同一份报告。

当前合同专用于 ``current_state_future_v2`` 的 2400 条冻结测试集。测试集内容
必须与内置 SHA-256 一致，事件 ID 必须唯一，且双方必须逐事件使用相同目标。
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Sequence


SCHEMA_VERSION = "edge-unified-traffic-bf16-paired-regression/v1"
EXPECTED_TEST_SHA256 = (
    "604ebf54458874b744051c5d2b7bd340771138e6de1347a01ff91ea806ce55ac"
)
EXPECTED_SAMPLE_COUNT = 2400
EXPECTED_PROMPT_TOKENS = 16
EXPECTED_MAX_SEQUENCE_LENGTH = 16
EXPECTED_MAX_NEW_TOKENS = 1
EXPECTED_TEMPERATURE = 0.0
ACTION_TOKENS = tuple("ABCDEF")
BOOTSTRAP_SEED = 20260810
BOOTSTRAP_REPLICATES = 5000

GATE_THRESHOLDS = {
    "accuracy_minimum": 0.66125,
    "weighted_f1_minimum": 0.666848,
    "valid_output_rate_minimum": 1.0,
}


class RegressionGateError(ValueError):
    """输入证据不完整、未绑定或不符合配对合同。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_regular_file(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise RegressionGateError(f"{label} 必须是存在的非符号链接普通文件: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise RegressionGateError(f"{label} 必须是存在的非符号链接普通文件: {resolved}")
    return resolved


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegressionGateError(f"{label} 不是可读取的 UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RegressionGateError(f"{label} 必须是 JSON object")
    return value


def _require_sha256(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise RegressionGateError(f"{label} 必须是64位小写SHA-256")
    return normalized


def _optional_sha256(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, label)


def _identity(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_reported_file_identity(
    raw: Any, *, label: str, current_path: Path
) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RegressionGateError(f"{label}身份记录缺失")
    current_path = _require_regular_file(current_path, label)
    declared_path = _resolve_declared_path(raw.get("path"), current_path.parent, label)
    if declared_path != current_path:
        raise RegressionGateError(f"{label}报告路径与当前文件不同")
    declared_bytes = raw.get("bytes")
    if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int):
        raise RegressionGateError(f"{label}.bytes必须是整数")
    declared_sha256 = _require_sha256(raw.get("sha256"), f"{label}.sha256")
    current = _identity(current_path)
    if declared_bytes != current["bytes"] or declared_sha256 != current["sha256"]:
        raise RegressionGateError(f"{label}报告身份与当前文件不一致")
    return current


def _resolve_declared_path(value: Any, project_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RegressionGateError(f"{label} 必须是非空路径")
    raw = Path(value)
    return (raw if raw.is_absolute() else project_root / raw).resolve()


def _validate_snapshot_file_identities(
    raw: Any, *, manifest_path: Path, snapshot_directory: Path, label: str
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise RegressionGateError(f"{label}身份记录缺失")
    manifest = _read_json_object(manifest_path, f"{label}清单")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise RegressionGateError(f"{label}清单.files必须是非空数组")
    snapshot_directory = snapshot_directory.resolve()
    identities: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RegressionGateError(f"{label}清单.files[{index}]必须是对象")
        relative_name = row.get("path")
        if not isinstance(relative_name, str) or not relative_name.strip():
            raise RegressionGateError(f"{label}清单.files[{index}].path无效")
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RegressionGateError(f"{label}清单.files[{index}].path越界")
        normalized_name = relative.as_posix()
        if normalized_name in identities:
            raise RegressionGateError(f"{label}清单文件重复: {normalized_name}")
        current = _validate_reported_file_identity(
            raw.get(normalized_name),
            label=f"{label}.{normalized_name}",
            current_path=snapshot_directory / relative,
        )
        declared_bytes = row.get("bytes")
        if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int):
            raise RegressionGateError(f"{label}清单bytes无效: {normalized_name}")
        declared_sha256 = _require_sha256(
            row.get("sha256"), f"{label}清单.{normalized_name}.sha256"
        )
        if declared_bytes != current["bytes"] or declared_sha256 != current["sha256"]:
            raise RegressionGateError(f"{label}文件与清单身份不一致: {normalized_name}")
        identities[normalized_name] = current
    if set(raw) != set(identities):
        raise RegressionGateError(f"{label}报告文件集与清单不一致")
    return identities


def read_frozen_test(
    path: Path,
    *,
    expected_sha256: str = EXPECTED_TEST_SHA256,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
) -> Dict[str, str]:
    path = _require_regular_file(path, "冻结交通测试集")
    expected_sha256 = _require_sha256(expected_sha256, "expected_test_sha256")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RegressionGateError(
            f"冻结交通测试集SHA-256不匹配: 期望 {expected_sha256}，实测 {actual_sha256}"
        )
    targets: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RegressionGateError(
                    f"冻结交通测试集第{line_number}行JSON无效"
                ) from exc
            if not isinstance(row, dict):
                raise RegressionGateError(f"冻结交通测试集第{line_number}行必须是对象")
            event_id = row.get("event_id")
            target = row.get("target")
            if not isinstance(event_id, str) or not event_id:
                raise RegressionGateError(f"冻结交通测试集第{line_number}行event_id无效")
            if event_id in targets:
                raise RegressionGateError(f"冻结交通测试集event_id重复: {event_id}")
            if target not in ACTION_TOKENS:
                raise RegressionGateError(
                    f"冻结交通测试集第{line_number}行target必须是A-F"
                )
            targets[event_id] = str(target)
    if len(targets) != expected_count:
        raise RegressionGateError(
            f"冻结交通测试集必须有{expected_count}个唯一样本，实测{len(targets)}"
        )
    return targets


def _validate_example(raw: Any, location: str) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RegressionGateError(f"{location}必须是JSON object")
    event_id = raw.get("event_id")
    target = raw.get("target")
    parsed = raw.get("parsed")
    if not isinstance(event_id, str) or not event_id:
        raise RegressionGateError(f"{location}.event_id无效")
    if target not in ACTION_TOKENS:
        raise RegressionGateError(f"{location}.target必须是A-F")
    if parsed is not None and parsed not in ACTION_TOKENS:
        raise RegressionGateError(f"{location}.parsed只能是A-F或null")
    raw_output = raw.get("raw_output")
    if not isinstance(raw_output, str):
        raise RegressionGateError(f"{location}.raw_output必须是字符串")
    reparsed = raw_output if raw_output in ACTION_TOKENS else None
    if parsed != reparsed:
        raise RegressionGateError(
            f"{location}.parsed与raw_output严格单token重解析结果不同"
        )
    valid = raw.get("json_valid")
    match = raw.get("decision_match")
    if not isinstance(valid, bool) or not isinstance(match, bool):
        raise RegressionGateError(f"{location}缺少布尔json_valid/decision_match")
    if valid != (reparsed is not None):
        raise RegressionGateError(f"{location}.json_valid与raw_output重解析结果矛盾")
    if match != (reparsed == target):
        raise RegressionGateError(
            f"{location}.decision_match与raw_output重解析结果/target矛盾"
        )
    prompt_tokens = raw.get("prompt_tokens")
    if prompt_tokens != EXPECTED_PROMPT_TOKENS:
        raise RegressionGateError(
            f"{location}.prompt_tokens必须为{EXPECTED_PROMPT_TOKENS}，实测{prompt_tokens}"
        )
    latency = raw.get("latency_ms")
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or float(latency) < 0
    ):
        raise RegressionGateError(f"{location}.latency_ms必须是非负有限数")
    return {
        "event_id": event_id,
        "target": str(target),
        "parsed": parsed,
        "valid": valid,
        "correct": match,
        "latency_ms": float(latency),
    }


def _round4(value: float) -> float:
    return round(float(value), 4)


def _assert_reported_metric(
    report: Mapping[str, Any], field: str, actual: float, label: str
) -> None:
    declared = report.get(field)
    if isinstance(declared, bool) or not isinstance(declared, (int, float)):
        raise RegressionGateError(f"{label}.{field}缺失或不是数字")
    if abs(float(declared) - _round4(actual)) > 1e-12:
        raise RegressionGateError(
            f"{label}.{field}与逐事件重算不一致: 声明{declared}，重算{_round4(actual)}"
        )


def _classification_metrics(examples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(examples)
    correct = sum(bool(row["correct"]) for row in examples)
    valid = sum(bool(row["valid"]) for row in examples)
    weighted_f1 = 0.0
    per_class: Dict[str, Dict[str, Any]] = {}
    for token in ACTION_TOKENS:
        support = sum(row["target"] == token for row in examples)
        true_positive = sum(
            row["target"] == token and row["parsed"] == token for row in examples
        )
        false_positive = sum(
            row["target"] != token and row["parsed"] == token for row in examples
        )
        false_negative = support - true_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        weighted_f1 += f1 * support
        per_class[token] = {
            "support": support,
            "true_positive": true_positive,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    return {
        "sample_count": total,
        "correct": correct,
        "valid": valid,
        "accuracy": round(correct / total, 6),
        "valid_output_rate": round(valid / total, 6),
        "weighted_f1": round(weighted_f1 / total, 6),
        "per_class": per_class,
    }


def read_evaluation(
    path: Path,
    *,
    label: str,
    project_root: Path,
    test_path: Path,
    adapter_model_path: Path,
    frozen_targets: Mapping[str, str],
) -> Dict[str, Any]:
    path = _require_regular_file(path, f"{label}评估JSON")
    report = _read_json_object(path, f"{label}评估JSON")
    if report.get("prompt_format") != "raw_task":
        raise RegressionGateError(f"{label}评估必须使用raw_task提示协议")
    if report.get("class_scores_included") is not True:
        raise RegressionGateError(f"{label}评估必须启用include_class_scores")
    if report.get("bf16") is not True:
        raise RegressionGateError(f"{label}评估必须明确记录bf16=true")
    max_seq_length = report.get("max_seq_length")
    if (
        isinstance(max_seq_length, bool)
        or not isinstance(max_seq_length, int)
        or max_seq_length != EXPECTED_MAX_SEQUENCE_LENGTH
    ):
        raise RegressionGateError(
            f"{label}.max_seq_length必须为{EXPECTED_MAX_SEQUENCE_LENGTH}"
        )
    max_new_tokens = report.get("max_new_tokens")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens != EXPECTED_MAX_NEW_TOKENS
    ):
        raise RegressionGateError(
            f"{label}.max_new_tokens必须为{EXPECTED_MAX_NEW_TOKENS}"
        )
    temperature = report.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) != EXPECTED_TEMPERATURE
    ):
        raise RegressionGateError(
            f"{label}.temperature必须为{EXPECTED_TEMPERATURE}"
        )
    if report.get("count") != EXPECTED_SAMPLE_COUNT:
        raise RegressionGateError(f"{label}评估count必须为{EXPECTED_SAMPLE_COUNT}")
    if report.get("source_count") != EXPECTED_SAMPLE_COUNT:
        raise RegressionGateError(f"{label}评估source_count必须为{EXPECTED_SAMPLE_COUNT}")
    warmup_runs = report.get("warmup_runs")
    if isinstance(warmup_runs, bool) or not isinstance(warmup_runs, int) or warmup_runs < 1:
        raise RegressionGateError(f"{label}评估必须至少预热一次")
    model_name_or_path = report.get("model_name_or_path")
    if not isinstance(model_name_or_path, str) or not model_name_or_path.strip():
        raise RegressionGateError(f"{label}.model_name_or_path必须是非空字符串")
    declared_test = _resolve_declared_path(
        report.get("test_jsonl"), project_root, f"{label}.test_jsonl"
    )
    if declared_test != test_path:
        raise RegressionGateError(f"{label}评估声明的测试集路径与CLI绑定路径不同")
    declared_adapter = _resolve_declared_path(
        report.get("adapter_dir"), project_root, f"{label}.adapter_dir"
    )
    if declared_adapter != adapter_model_path.parent:
        raise RegressionGateError(f"{label}评估声明的LoRA目录与CLI绑定权重不同")
    artifacts = report.get("evaluation_artifacts")
    if not isinstance(artifacts, Mapping):
        raise RegressionGateError(f"{label}.evaluation_artifacts缺失")
    if artifacts.get("verified_unchanged_during_evaluation") is not True:
        raise RegressionGateError(f"{label}未证明评估期间模型和评估器身份稳定")
    declared_base = _resolve_declared_path(
        model_name_or_path, project_root, f"{label}.model_name_or_path"
    )
    artifact_identities = {
        "adapter_model": _validate_reported_file_identity(
            artifacts.get("adapter_model"),
            label=f"{label}.adapter_model",
            current_path=adapter_model_path,
        ),
        "adapter_config": _validate_reported_file_identity(
            artifacts.get("adapter_config"),
            label=f"{label}.adapter_config",
            current_path=adapter_model_path.parent / "adapter_config.json",
        ),
        "base_text_snapshot_manifest": _validate_reported_file_identity(
            artifacts.get("base_text_snapshot_manifest"),
            label=f"{label}.base_text_snapshot_manifest",
            current_path=declared_base / "text_snapshot_manifest.json",
        ),
        "evaluator": _validate_reported_file_identity(
            artifacts.get("evaluator"),
            label=f"{label}.evaluator",
            current_path=Path(__file__).with_name("eval_llm_sft_student.py"),
        ),
        "verified_unchanged_during_evaluation": True,
    }
    artifact_identities["base_text_snapshot_files"] = (
        _validate_snapshot_file_identities(
            artifacts.get("base_text_snapshot_files"),
            manifest_path=declared_base / "text_snapshot_manifest.json",
            snapshot_directory=declared_base,
            label=f"{label}.base_text_snapshot_files",
        )
    )
    raw_examples = report.get("examples")
    if not isinstance(raw_examples, list) or len(raw_examples) != EXPECTED_SAMPLE_COUNT:
        raise RegressionGateError(
            f"{label}.examples必须含{EXPECTED_SAMPLE_COUNT}条逐事件记录"
        )
    examples: Dict[str, Dict[str, Any]] = {}
    for index, raw in enumerate(raw_examples):
        row = _validate_example(raw, f"{label}.examples[{index}]")
        event_id = row["event_id"]
        if event_id in examples:
            raise RegressionGateError(f"{label}评估event_id重复: {event_id}")
        examples[event_id] = row
    if set(examples) != set(frozen_targets):
        missing = sorted(set(frozen_targets) - set(examples))[:5]
        extra = sorted(set(examples) - set(frozen_targets))[:5]
        raise RegressionGateError(
            f"{label}评估样本集合与冻结测试集不同: missing={missing}, extra={extra}"
        )
    for event_id, target in frozen_targets.items():
        if examples[event_id]["target"] != target:
            raise RegressionGateError(f"{label}评估目标与冻结测试集不同: {event_id}")
    ordered = [examples[event_id] for event_id in frozen_targets]
    metrics = _classification_metrics(ordered)
    _assert_reported_metric(report, "decision_accuracy", metrics["accuracy"], label)
    _assert_reported_metric(
        report, "json_valid_rate", metrics["valid_output_rate"], label
    )
    token_report = report.get("token_classification")
    if not isinstance(token_report, Mapping):
        raise RegressionGateError(f"{label}.token_classification缺失")
    declared_weighted_f1 = token_report.get("weighted_f1")
    if (
        isinstance(declared_weighted_f1, bool)
        or not isinstance(declared_weighted_f1, (int, float))
        or abs(float(declared_weighted_f1) - _round4(metrics["weighted_f1"]))
        > 1e-12
    ):
        raise RegressionGateError(
            f"{label}.token_classification.weighted_f1与逐事件重算不一致"
        )
    return {
        "report": report,
        "report_identity": _identity(path),
        "examples": examples,
        "metrics": metrics,
        "evaluation_artifacts": artifact_identities,
        "protocol": {
            "model_name_or_path": model_name_or_path,
            "prompt_format": report.get("prompt_format"),
            "warmup_runs": warmup_runs,
            "class_scores_included": report.get("class_scores_included"),
            "bf16": report.get("bf16"),
            "max_seq_length": max_seq_length,
            "max_new_tokens": max_new_tokens,
            "temperature": float(temperature),
        },
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise RegressionGateError("不能对空bootstrap结果计算分位数")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_accuracy_delta(
    deltas: Sequence[int],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> Dict[str, Any]:
    if not deltas or any(value not in {-1, 0, 1} for value in deltas):
        raise RegressionGateError("配对正确性差值只能由-1、0、1组成")
    if replicates <= 0:
        raise RegressionGateError("bootstrap重复数必须大于0")
    generator = random.Random(seed)
    count = len(deltas)
    estimates = []
    for _ in range(replicates):
        estimates.append(
            sum(deltas[generator.randrange(count)] for _ in range(count)) / count
        )
    return {
        "method": "event_paired_nonparametric_bootstrap",
        "unit": "event_id",
        "seed": seed,
        "replicates": replicates,
        "ci_level": 0.95,
        "candidate_minus_incumbent_ci95": [
            round(_quantile(estimates, 0.025), 6),
            round(_quantile(estimates, 0.975), 6),
        ],
    }


def build_report(
    *,
    incumbent_evaluation: Path,
    candidate_evaluation: Path,
    test_jsonl: Path,
    incumbent_adapter_model: Path,
    expected_incumbent_adapter_sha256: str,
    candidate_adapter_model: Path,
    expected_candidate_adapter_sha256: str,
    project_root: Path,
    expected_test_sha256: str = EXPECTED_TEST_SHA256,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    expected_evaluator_sha256: str | None = None,
    expected_gate_sha256: str | None = None,
) -> Dict[str, Any]:
    project_root = project_root.resolve()
    expected_test = _require_sha256(expected_test_sha256, "expected_test_sha256")
    expected_evaluator = _optional_sha256(
        expected_evaluator_sha256, "expected_evaluator_sha256"
    )
    expected_gate = _optional_sha256(expected_gate_sha256, "expected_gate_sha256")
    if (expected_evaluator is None) != (expected_gate is None):
        raise RegressionGateError(
            "expected_evaluator_sha256与expected_gate_sha256必须同时提供"
        )
    evaluator_identity = _identity(
        _require_regular_file(
            Path(__file__).with_name("eval_llm_sft_student.py"), "评估器"
        )
    )
    gate_identity = _identity(_require_regular_file(Path(__file__), "门禁脚本"))
    if (
        expected_evaluator is not None
        and evaluator_identity["sha256"] != expected_evaluator
    ):
        raise RegressionGateError("评估器SHA-256与外部锚点不一致")
    if expected_gate is not None and gate_identity["sha256"] != expected_gate:
        raise RegressionGateError("门禁脚本SHA-256与外部锚点不一致")
    test_jsonl = _require_regular_file(test_jsonl, "冻结交通测试集")
    incumbent_adapter_model = _require_regular_file(
        incumbent_adapter_model, "正式adapter_model.safetensors"
    )
    candidate_adapter_model = _require_regular_file(
        candidate_adapter_model, "U1 adapter_model.safetensors"
    )
    if incumbent_adapter_model.name != "adapter_model.safetensors":
        raise RegressionGateError("正式LoRA权重文件名必须是adapter_model.safetensors")
    if candidate_adapter_model.name != "adapter_model.safetensors":
        raise RegressionGateError("U1 LoRA权重文件名必须是adapter_model.safetensors")
    incumbent_expected = _require_sha256(
        expected_incumbent_adapter_sha256, "expected_incumbent_adapter_sha256"
    )
    candidate_expected = _require_sha256(
        expected_candidate_adapter_sha256, "expected_candidate_adapter_sha256"
    )
    incumbent_actual = sha256_file(incumbent_adapter_model)
    candidate_actual = sha256_file(candidate_adapter_model)
    if incumbent_actual != incumbent_expected:
        raise RegressionGateError("正式LoRA权重SHA-256与外部锚点不一致")
    if candidate_actual != candidate_expected:
        raise RegressionGateError("U1 LoRA权重SHA-256与外部锚点不一致")
    if incumbent_actual == candidate_actual:
        raise RegressionGateError("正式LoRA与U1 LoRA权重不能是同一文件内容")

    frozen_targets = read_frozen_test(
        test_jsonl,
        expected_sha256=expected_test,
        expected_count=expected_count,
    )
    incumbent = read_evaluation(
        incumbent_evaluation,
        label="incumbent",
        project_root=project_root,
        test_path=test_jsonl,
        adapter_model_path=incumbent_adapter_model,
        frozen_targets=frozen_targets,
    )
    candidate = read_evaluation(
        candidate_evaluation,
        label="candidate",
        project_root=project_root,
        test_path=test_jsonl,
        adapter_model_path=candidate_adapter_model,
        frozen_targets=frozen_targets,
    )
    if incumbent["protocol"] != candidate["protocol"]:
        raise RegressionGateError("正式LoRA与U1评估协议不完全一致")
    incumbent_artifacts = incumbent["evaluation_artifacts"]
    candidate_artifacts = candidate["evaluation_artifacts"]
    if (
        incumbent_artifacts["base_text_snapshot_manifest"]["sha256"]
        != candidate_artifacts["base_text_snapshot_manifest"]["sha256"]
    ):
        raise RegressionGateError("正式LoRA与U1评估使用的基座manifest SHA-256不同")
    if (
        incumbent_artifacts["evaluator"]["sha256"]
        != candidate_artifacts["evaluator"]["sha256"]
    ):
        raise RegressionGateError("正式LoRA与U1评估器SHA-256不同")
    if incumbent_artifacts["evaluator"]["sha256"] != evaluator_identity["sha256"]:
        raise RegressionGateError("评估报告与当前评估器SHA-256不同")

    deltas = []
    corrected = 0
    regressed = 0
    both_correct = 0
    both_wrong = 0
    for event_id in frozen_targets:
        incumbent_correct = bool(incumbent["examples"][event_id]["correct"])
        candidate_correct = bool(candidate["examples"][event_id]["correct"])
        deltas.append(int(candidate_correct) - int(incumbent_correct))
        if candidate_correct and not incumbent_correct:
            corrected += 1
        elif incumbent_correct and not candidate_correct:
            regressed += 1
        elif incumbent_correct:
            both_correct += 1
        else:
            both_wrong += 1

    per_class = {}
    for token in ACTION_TOKENS:
        incumbent_class = incumbent["metrics"]["per_class"][token]
        candidate_class = candidate["metrics"]["per_class"][token]
        if incumbent_class["support"] != candidate_class["support"]:
            raise RegressionGateError(f"A-F类别{token}支持数不一致")
        per_class[token] = {
            "support": incumbent_class["support"],
            "incumbent_recall": incumbent_class["recall"],
            "candidate_recall": candidate_class["recall"],
            "candidate_minus_incumbent_recall": round(
                candidate_class["recall"] - incumbent_class["recall"], 6
            ),
        }

    incumbent_metrics = incumbent["metrics"]
    candidate_metrics = candidate["metrics"]
    checks = {
        "candidate_accuracy_absolute_minimum": {
            "actual": candidate_metrics["accuracy"],
            "requirement": f">={GATE_THRESHOLDS['accuracy_minimum']}",
            "passed": candidate_metrics["accuracy"]
            >= GATE_THRESHOLDS["accuracy_minimum"],
        },
        "candidate_accuracy_not_below_incumbent": {
            "actual": candidate_metrics["accuracy"],
            "incumbent": incumbent_metrics["accuracy"],
            "requirement": ">=same_run_incumbent_accuracy",
            "passed": candidate_metrics["accuracy"]
            >= incumbent_metrics["accuracy"],
        },
        "candidate_weighted_f1_absolute_minimum": {
            "actual": candidate_metrics["weighted_f1"],
            "requirement": f">={GATE_THRESHOLDS['weighted_f1_minimum']}",
            "passed": candidate_metrics["weighted_f1"]
            >= GATE_THRESHOLDS["weighted_f1_minimum"],
        },
        "candidate_weighted_f1_not_below_incumbent": {
            "actual": candidate_metrics["weighted_f1"],
            "incumbent": incumbent_metrics["weighted_f1"],
            "requirement": ">=same_run_incumbent_weighted_f1",
            "passed": candidate_metrics["weighted_f1"]
            >= incumbent_metrics["weighted_f1"],
        },
        "candidate_valid_output_rate": {
            "actual": candidate_metrics["valid_output_rate"],
            "requirement": "==1.0",
            "passed": candidate_metrics["valid_output_rate"]
            == GATE_THRESHOLDS["valid_output_rate_minimum"],
        },
    }
    metric_condition_names = (
        "candidate_accuracy_absolute_minimum",
        "candidate_accuracy_not_below_incumbent",
        "candidate_weighted_f1_absolute_minimum",
        "candidate_weighted_f1_not_below_incumbent",
    )
    metric_conditions = {
        name: bool(checks[name]["passed"]) for name in metric_condition_names
    }
    all_passed = all(check["passed"] for check in checks.values())
    paired_delta = sum(deltas) / len(deltas)
    if evaluator_identity != _identity(
        _require_regular_file(
            Path(__file__).with_name("eval_llm_sft_student.py"), "评估器"
        )
    ):
        raise RegressionGateError("门禁运行期间评估器身份发生漂移")
    if gate_identity != _identity(_require_regular_file(Path(__file__), "门禁脚本")):
        raise RegressionGateError("门禁运行期间门禁脚本身份发生漂移")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "unified_U1_vs_production_traffic_BF16_paired_regression",
        "gate_script_identity": gate_identity,
        "external_sha256_anchors": {
            "frozen_test": {
                "expected_sha256": expected_test,
                "actual_sha256": sha256_file(test_jsonl),
                "matched": True,
            },
            "evaluator": {
                "expected_sha256": expected_evaluator,
                "actual_sha256": evaluator_identity["sha256"],
                "matched": expected_evaluator is not None,
            },
            "gate": {
                "expected_sha256": expected_gate,
                "actual_sha256": gate_identity["sha256"],
                "matched": expected_gate is not None,
            },
            "triple_anchor_complete": (
                expected_evaluator is not None and expected_gate is not None
            ),
        },
        "frozen_test": {
            **_identity(test_jsonl),
            "expected_sha256": expected_test,
            "sample_count": len(frozen_targets),
            "unique_event_ids": len(frozen_targets),
            "target_distribution": dict(sorted(Counter(frozen_targets.values()).items())),
        },
        "incumbent": {
            "evaluation": incumbent["report_identity"],
            "adapter_model": _identity(incumbent_adapter_model),
            "evaluation_artifacts": incumbent["evaluation_artifacts"],
            "metrics": incumbent["metrics"],
        },
        "candidate": {
            "evaluation": candidate["report_identity"],
            "adapter_model": _identity(candidate_adapter_model),
            "evaluation_artifacts": candidate["evaluation_artifacts"],
            "metrics": candidate["metrics"],
        },
        "protocol_binding": {
            **incumbent["protocol"],
            "expected_prompt_tokens": EXPECTED_PROMPT_TOKENS,
            "same_protocol": True,
            "same_event_ids_and_targets": True,
        },
        "paired_comparison": {
            "candidate_minus_incumbent_accuracy": round(paired_delta, 6),
            "candidate_correct_incumbent_wrong": corrected,
            "candidate_wrong_incumbent_correct": regressed,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "net_corrected": corrected - regressed,
            "per_class_recall": per_class,
            "bootstrap": paired_bootstrap_accuracy_delta(
                deltas,
                seed=bootstrap_seed,
                replicates=bootstrap_replicates,
            ),
        },
        "hard_gate": {
            "thresholds": dict(GATE_THRESHOLDS),
            "metric_conditions": metric_conditions,
            "metric_conditions_all_passed": all(metric_conditions.values()),
            "valid_output_contract_passed": checks[
                "candidate_valid_output_rate"
            ]["passed"],
            "checks": checks,
            "all_passed": all_passed,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-evaluation", required=True)
    parser.add_argument("--candidate-evaluation", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--incumbent-adapter-model", required=True)
    parser.add_argument("--expected-incumbent-adapter-sha256", required=True)
    parser.add_argument("--candidate-adapter-model", required=True)
    parser.add_argument("--expected-candidate-adapter-sha256", required=True)
    parser.add_argument("--expected-evaluator-sha256")
    parser.add_argument("--expected-gate-sha256")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="eval_llm_sft_student.py写入相对路径时使用的交通场景根目录",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise RegressionGateError(f"拒绝覆盖已有门禁报告: {output}")
    report = build_report(
        incumbent_evaluation=Path(args.incumbent_evaluation),
        candidate_evaluation=Path(args.candidate_evaluation),
        test_jsonl=Path(args.test_jsonl),
        incumbent_adapter_model=Path(args.incumbent_adapter_model),
        expected_incumbent_adapter_sha256=args.expected_incumbent_adapter_sha256,
        candidate_adapter_model=Path(args.candidate_adapter_model),
        expected_candidate_adapter_sha256=args.expected_candidate_adapter_sha256,
        expected_evaluator_sha256=args.expected_evaluator_sha256,
        expected_gate_sha256=args.expected_gate_sha256,
        project_root=Path(args.project_root),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "incumbent": report["incumbent"]["metrics"],
                "candidate": report["candidate"]["metrics"],
                "paired_comparison": report["paired_comparison"],
                "hard_gate": report["hard_gate"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if not report["hard_gate"]["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
