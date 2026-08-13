"""对统一候选与 Qwen3.5 9B Teacher 的同开发集结果做配对门禁。

本工具读取两个评测器已经产出的逐题 JSONL 与 summary JSON，并对候选
``adapter_model.safetensors`` 只做字节数与 SHA-256 现场重算；它不加载模型、
原始数据集或任何 test/blind split。逐题配对由 sample_id、category、reference
和 row_sha256 四重锁定，并重新计算所有准确率和有效率。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from edge_llm_factory.contracts import (
    HEX64,
    ManifestError,
    read_json_object,
    sha256_file,
    write_json_object,
)
from edge_llm_factory.evaluate_unified_general_adapter import (
    EVALUATION_SCHEMA_VERSION as CANDIDATE_SCHEMA,
    GENERAL_CATEGORIES,
    MAX_NEW_TOKENS,
    _percentile,
    evaluate_choice,
    evaluate_math,
)
from edge_llm_factory.evaluate_unified_general_teacher import (
    EVALUATION_SCHEMA_VERSION as TEACHER_SCHEMA,
    TEACHER_MODEL,
)


GATE_SCHEMA_VERSION = "edge-llm-unified-general-paired-gate/v1"
RETENTION_MINIMUM = 0.80
VALID_RATE_REQUIRED = 1.0
EXPECTED_SAMPLES_PER_CATEGORY = 400
EXPECTED_SAMPLE_COUNT = EXPECTED_SAMPLES_PER_CATEGORY * len(GENERAL_CATEGORIES)
EXPECTED_CANDIDATE_MAX_INPUT_TOKENS = 512
EXPECTED_TEACHER_MAX_INPUT_TOKENS = 2048
CANDIDATE_EVALUATOR_PATH = Path(__file__).with_name(
    "evaluate_unified_general_adapter.py"
)
TEACHER_EVALUATOR_PATH = Path(__file__).with_name(
    "evaluate_unified_general_teacher.py"
)


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} 必须是非空字符串")
    return value.strip()


def _required_sha(value: Any, field: str) -> str:
    text = _required_text(value, field).lower()
    if HEX64.fullmatch(text) is None:
        raise ManifestError(f"{field} 必须是 SHA-256")
    return text


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{field} 必须是对象")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{field} 必须是布尔值")
    return value


def _read_samples(path: Path, side: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_number} JSON 无效") from exc
            if not isinstance(value, Mapping):
                raise ManifestError(f"{path}:{line_number} 必须是对象")
            location = f"{side}.samples[{line_number}]"
            sample_id = _required_text(value.get("sample_id"), location + ".sample_id")
            if sample_id in rows:
                raise ManifestError(f"{side} sample_id 重复: {sample_id}")
            category = _required_text(value.get("category"), location + ".category")
            if category not in GENERAL_CATEGORIES:
                raise ManifestError(f"{location}.category 不受支持: {category}")
            reference = _required_text(value.get("reference"), location + ".reference")
            row_sha = _required_sha(value.get("row_sha256"), location + ".row_sha256")
            raw_output = value.get("raw_output")
            if not isinstance(raw_output, str):
                raise ManifestError(f"{location}.raw_output 必须是字符串")
            prediction = value.get("prediction")
            if prediction is not None and not isinstance(prediction, str):
                raise ManifestError(f"{location}.prediction 必须是字符串或 null")
            valid = _boolean(value.get("valid"), location + ".valid")
            correct = _boolean(value.get("correct"), location + ".correct")
            if correct and not valid:
                raise ManifestError(f"{location} correct=true 但 valid=false")
            rescored = (
                evaluate_math(raw_output.strip(), reference)
                if category == "math"
                else evaluate_choice(raw_output.strip(), reference)
            )
            for field, declared in (
                ("prediction", prediction),
                ("reference", reference),
                ("valid", valid),
                ("correct", correct),
            ):
                if rescored[field] != declared:
                    raise ManifestError(
                        f"{location}.{field} 与 raw_output 独立重判结果不一致"
                    )
            rows[sample_id] = {
                "sample_id": sample_id,
                "category": category,
                "reference": reference,
                "row_sha256": row_sha,
                "raw_output": raw_output,
                "prediction": prediction,
                "valid": valid,
                "correct": correct,
            }
    if not rows:
        raise ManifestError(f"{side} 逐题结果为空")
    return rows


def _verify_samples_binding(
    samples_path: Path, summary: Mapping[str, Any], side: str
) -> Dict[str, Any]:
    artifacts = _mapping(summary.get("artifacts"), f"{side}.summary.artifacts")
    declared = _mapping(artifacts.get("samples"), f"{side}.summary.artifacts.samples")
    declared_path = Path(
        _required_text(declared.get("path"), f"{side}.summary.artifacts.samples.path")
    ).resolve()
    if samples_path.resolve() != declared_path:
        raise ManifestError(f"{side} samples 路径与 summary 绑定不一致")
    actual_sha = sha256_file(samples_path)
    declared_sha = _required_sha(
        declared.get("sha256"), f"{side}.summary.artifacts.samples.sha256"
    )
    if actual_sha != declared_sha:
        raise ManifestError(f"{side} samples SHA-256 与 summary 不一致")
    declared_bytes = declared.get("bytes")
    if (
        isinstance(declared_bytes, bool)
        or not isinstance(declared_bytes, int)
        or declared_bytes != samples_path.stat().st_size
    ):
        raise ManifestError(f"{side} samples 字节数与 summary 不一致")
    return {
        "path": str(samples_path.resolve()),
        "bytes": samples_path.stat().st_size,
        "sha256": actual_sha,
    }


def _verify_development_summary(
    summary: Mapping[str, Any], side: str, expected_schema: str
) -> None:
    if summary.get("schema_version") != expected_schema:
        raise ManifestError(f"{side} summary schema_version 不匹配")
    if summary.get("development_only") is not True:
        raise ManifestError(f"{side} summary 未声明 development_only=true")
    if summary.get("formal_blind_test_used") is not False:
        raise ManifestError(f"{side} summary 未证明 formal_blind_test_used=false")
    dataset = _mapping(summary.get("dataset"), f"{side}.summary.dataset")
    if dataset.get("blind_test_content_loaded") is not False:
        raise ManifestError(f"{side} summary 未证明 blind test 未加载")


def _required_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{field} 必须是整数")
    return value


def _required_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{field} 必须是数值")
    return float(value)


def _verify_max_new_tokens(value: Any, field: str) -> None:
    declared = _mapping(value, field)
    if set(declared) != set(MAX_NEW_TOKENS):
        raise ManifestError(f"{field} 类别集合不匹配")
    for category, expected in MAX_NEW_TOKENS.items():
        actual = _required_integer(declared.get(category), f"{field}.{category}")
        if actual != expected:
            raise ManifestError(
                f"{field}.{category} 必须为 {expected}，实际为 {actual}"
            )


def _verify_candidate_protocol(summary: Mapping[str, Any]) -> None:
    protocol = _mapping(
        summary.get("evaluation_protocol"), "candidate.summary.evaluation_protocol"
    )
    expected_text = {
        "renderer": "tokenizer.apply_chat_template",
        "precision": "bfloat16",
        "math_scoring": "strict_trailing_FINAL_numeric",
        "logic_scoring": "strict_full_output_A_to_D",
    }
    for field, expected in expected_text.items():
        if protocol.get(field) != expected:
            raise ManifestError(
                f"candidate evaluation_protocol.{field} 必须为 {expected}"
            )
    for field in (
        "uses_each_row_system_prompt_verbatim",
        "uses_each_row_prompt_verbatim",
    ):
        if protocol.get(field) is not True:
            raise ManifestError(f"candidate evaluation_protocol.{field} 必须为 true")
    max_input = _required_integer(
        protocol.get("max_input_tokens"),
        "candidate.summary.evaluation_protocol.max_input_tokens",
    )
    if max_input != EXPECTED_CANDIDATE_MAX_INPUT_TOKENS:
        raise ManifestError(
            "candidate evaluation_protocol.max_input_tokens 必须为 "
            f"{EXPECTED_CANDIDATE_MAX_INPUT_TOKENS}"
        )
    _verify_max_new_tokens(
        protocol.get("max_new_tokens"),
        "candidate.summary.evaluation_protocol.max_new_tokens",
    )
    decoding = _mapping(
        protocol.get("decoding"), "candidate.summary.evaluation_protocol.decoding"
    )
    if decoding.get("do_sample") is not False:
        raise ManifestError(
            "candidate evaluation_protocol.decoding.do_sample 必须为 false"
        )


def _verify_teacher_protocol(summary: Mapping[str, Any]) -> None:
    protocol = _mapping(
        summary.get("evaluation_protocol"), "teacher.summary.evaluation_protocol"
    )
    expected_text = {
        "renderer": "ollama_api_chat",
        "math_scoring": "strict_trailing_FINAL_numeric",
        "logic_scoring": "strict_full_output_A_to_D",
    }
    for field, expected in expected_text.items():
        if protocol.get(field) != expected:
            raise ManifestError(
                f"teacher evaluation_protocol.{field} 必须为 {expected}"
            )
    for field in (
        "uses_each_row_system_prompt_verbatim",
        "uses_each_row_prompt_verbatim",
    ):
        if protocol.get(field) is not True:
            raise ManifestError(f"teacher evaluation_protocol.{field} 必须为 true")
    max_input = _required_integer(
        protocol.get("max_input_tokens"),
        "teacher.summary.evaluation_protocol.max_input_tokens",
    )
    if max_input != EXPECTED_TEACHER_MAX_INPUT_TOKENS:
        raise ManifestError(
            "teacher evaluation_protocol.max_input_tokens 必须为 "
            f"{EXPECTED_TEACHER_MAX_INPUT_TOKENS}"
        )
    _verify_max_new_tokens(
        protocol.get("max_new_tokens"),
        "teacher.summary.evaluation_protocol.max_new_tokens",
    )
    decoding = _mapping(
        protocol.get("decoding"), "teacher.summary.evaluation_protocol.decoding"
    )
    if _required_number(
        decoding.get("temperature"),
        "teacher.summary.evaluation_protocol.decoding.temperature",
    ) != 0.0:
        raise ManifestError("teacher decoding.temperature 必须为 0")
    if _required_number(
        decoding.get("top_p"), "teacher.summary.evaluation_protocol.decoding.top_p"
    ) != 1.0:
        raise ManifestError("teacher decoding.top_p 必须为 1")
    if _required_integer(
        decoding.get("seed"), "teacher.summary.evaluation_protocol.decoding.seed"
    ) != 42:
        raise ManifestError("teacher decoding.seed 必须为 42")
    if decoding.get("thinking") is not False:
        raise ManifestError("teacher decoding.thinking 必须为 false")


def _verify_evaluator_binding(
    summary: Mapping[str, Any],
    *,
    side: str,
    expected_sha256: str,
    current_path: Path,
    require_declared_bytes: bool,
) -> Dict[str, Any]:
    expected_sha = _required_sha(
        expected_sha256, f"expected_{side}_evaluator_sha256"
    )
    resolved_current = current_path.resolve()
    if not resolved_current.is_file():
        raise ManifestError(f"{side} 当前评测器文件不存在: {resolved_current}")
    actual_sha = sha256_file(resolved_current)
    if actual_sha != expected_sha:
        raise ManifestError(f"{side} 当前评测器 SHA-256 与外部锚点不一致")
    artifacts = _mapping(summary.get("artifacts"), f"{side}.summary.artifacts")
    declared = _mapping(
        artifacts.get("evaluator"), f"{side}.summary.artifacts.evaluator"
    )
    declared_path = Path(
        _required_text(
            declared.get("path"), f"{side}.summary.artifacts.evaluator.path"
        )
    ).resolve()
    if declared_path != resolved_current:
        raise ManifestError(f"{side} summary 声明的评测器路径不是当前冻结文件")
    declared_sha = _required_sha(
        declared.get("sha256"), f"{side}.summary.artifacts.evaluator.sha256"
    )
    if declared_sha != expected_sha:
        raise ManifestError(f"{side} summary 评测器 SHA-256 与外部锚点不一致")
    declared_bytes = declared.get("bytes")
    if require_declared_bytes:
        if (
            isinstance(declared_bytes, bool)
            or not isinstance(declared_bytes, int)
            or declared_bytes != resolved_current.stat().st_size
        ):
            raise ManifestError(f"{side} summary 评测器字节数与当前文件不一致")
    elif declared_bytes is not None and (
        isinstance(declared_bytes, bool)
        or not isinstance(declared_bytes, int)
        or declared_bytes != resolved_current.stat().st_size
    ):
        raise ManifestError(f"{side} summary 评测器字节数与当前文件不一致")
    return {
        "path": str(resolved_current),
        "bytes": resolved_current.stat().st_size,
        "sha256": actual_sha,
        "expected_sha256": expected_sha,
        "triple_bound": True,
    }


def _verify_candidate_adapter_binding(
    summary: Mapping[str, Any],
    *,
    adapter_model_path: Path,
    expected_sha256: str,
) -> Dict[str, Any]:
    """Bind the candidate summary to the externally preregistered weights file."""
    expected_sha = _required_sha(
        expected_sha256, "expected_candidate_adapter_sha256"
    )
    resolved = adapter_model_path.resolve()
    if resolved.name != "adapter_model.safetensors" or not resolved.is_file():
        raise ManifestError(
            "candidate_adapter_model 必须是实际存在的 adapter_model.safetensors"
        )
    actual_sha = sha256_file(resolved)
    if actual_sha != expected_sha:
        raise ManifestError("candidate adapter 权重 SHA-256 与外部锚点不一致")

    adapter = _mapping(summary.get("adapter"), "candidate.summary.adapter")
    declared_adapter_path = Path(
        _required_text(adapter.get("path"), "candidate.summary.adapter.path")
    ).resolve()
    if declared_adapter_path != resolved.parent:
        raise ManifestError("candidate summary adapter 目录与外部权重路径不一致")
    files = _mapping(adapter.get("files"), "candidate.summary.adapter.files")
    weights = _mapping(
        files.get("adapter_weights"),
        "candidate.summary.adapter.files.adapter_weights",
    )
    declared_path = Path(
        _required_text(
            weights.get("path"),
            "candidate.summary.adapter.files.adapter_weights.path",
        )
    ).resolve()
    if declared_path != resolved:
        raise ManifestError("candidate summary 权重路径与外部权重路径不一致")
    declared_sha = _required_sha(
        weights.get("sha256"),
        "candidate.summary.adapter.files.adapter_weights.sha256",
    )
    if declared_sha != expected_sha:
        raise ManifestError("candidate summary 权重 SHA-256 与外部锚点不一致")
    declared_bytes = weights.get("bytes")
    if (
        isinstance(declared_bytes, bool)
        or not isinstance(declared_bytes, int)
        or declared_bytes != resolved.stat().st_size
    ):
        raise ManifestError("candidate summary 权重字节数与实际文件不一致")

    adapter_artifact_sha = _required_sha(
        adapter.get("artifact_sha256"), "candidate.summary.adapter.artifact_sha256"
    )
    weights_binding = _mapping(
        adapter.get("weights_binding"), "candidate.summary.adapter.weights_binding"
    )
    if _required_sha(
        weights_binding.get("expected_sha256"),
        "candidate.summary.adapter.weights_binding.expected_sha256",
    ) != expected_sha:
        raise ManifestError("candidate summary 外部权重锚点与门禁外部锚点不一致")
    if _required_sha(
        weights_binding.get("actual_sha256"),
        "candidate.summary.adapter.weights_binding.actual_sha256",
    ) != actual_sha:
        raise ManifestError("candidate summary 权重重算值与门禁现场重算不一致")
    if weights_binding.get("externally_anchored_and_recomputed") is not True:
        raise ManifestError("candidate summary 未证明权重已外部锚定并现场重算")
    model = _mapping(
        summary.get("model_identity"), "candidate.summary.model_identity"
    )
    if model.get("adapter_artifact_sha256") != adapter_artifact_sha:
        raise ManifestError("candidate model_identity 未绑定 summary adapter artifact")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": actual_sha,
        "expected_sha256": expected_sha,
        "summary_declared_sha256": declared_sha,
        "externally_anchored_and_recomputed": True,
    }


def _verify_summary_metrics(
    summary: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any]], side: str
) -> None:
    metrics = _mapping(summary.get("metrics"), f"{side}.summary.metrics")
    categories = _mapping(metrics.get("categories"), f"{side}.summary.metrics.categories")
    for category in GENERAL_CATEGORIES:
        selected = [row for row in rows.values() if row["category"] == category]
        if not selected:
            raise ManifestError(f"{side} 缺少类别 {category}")
        claimed = _mapping(categories.get(category), f"{side}.summary.metrics.categories.{category}")
        total = len(selected)
        correct = sum(bool(row["correct"]) for row in selected)
        valid = sum(bool(row["valid"]) for row in selected)
        expected = {
            "sample_count": total,
            "correct": correct,
            "valid": valid,
            "accuracy": round(correct / total, 6),
            "valid_rate": round(valid / total, 6),
        }
        for field, value in expected.items():
            if claimed.get(field) != value:
                raise ManifestError(
                    f"{side} summary {category}.{field} 与逐题重算不一致"
                )


def _verify_same_selection(
    candidate_summary: Mapping[str, Any], teacher_summary: Mapping[str, Any]
) -> Dict[str, Any]:
    candidate_dataset = _mapping(candidate_summary.get("dataset"), "candidate.dataset")
    teacher_dataset = _mapping(teacher_summary.get("dataset"), "teacher.dataset")
    for field in ("sha256", "manifest_sha256"):
        candidate_value = _required_sha(candidate_dataset.get(field), f"candidate.dataset.{field}")
        teacher_value = _required_sha(teacher_dataset.get(field), f"teacher.dataset.{field}")
        if candidate_value != teacher_value:
            raise ManifestError(f"candidate/teacher dataset.{field} 不一致")
    candidate_selection = _mapping(candidate_summary.get("selection"), "candidate.selection")
    teacher_selection = _mapping(teacher_summary.get("selection"), "teacher.selection")
    for field in ("categories", "limit_per_category", "sample_count", "selected_rows_sha256"):
        if candidate_selection.get(field) != teacher_selection.get(field):
            raise ManifestError(f"candidate/teacher selection.{field} 不一致")
    selected_sha = _required_sha(
        candidate_selection.get("selected_rows_sha256"),
        "candidate.selection.selected_rows_sha256",
    )
    sample_count = candidate_selection.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count != EXPECTED_SAMPLE_COUNT
    ):
        raise ManifestError(
            f"selection.sample_count 必须为 {EXPECTED_SAMPLE_COUNT}"
        )
    limit = candidate_selection.get("limit_per_category")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit != EXPECTED_SAMPLES_PER_CATEGORY
    ):
        raise ManifestError(
            "selection.limit_per_category 必须为 "
            f"{EXPECTED_SAMPLES_PER_CATEGORY}"
        )
    categories = candidate_selection.get("categories")
    if categories != sorted(GENERAL_CATEGORIES):
        raise ManifestError("门禁要求同时包含 math 与 natural_language_reasoning")
    return {
        "dataset_sha256": candidate_dataset["sha256"],
        "dataset_manifest_sha256": candidate_dataset["manifest_sha256"],
        "selected_rows_sha256": selected_sha,
        "sample_count": sample_count,
    }


def _model_binding(
    candidate_summary: Mapping[str, Any], teacher_summary: Mapping[str, Any]
) -> Dict[str, Any]:
    candidate = _mapping(candidate_summary.get("model_identity"), "candidate.model_identity")
    teacher = _mapping(teacher_summary.get("model_identity"), "teacher.model_identity")
    candidate_sha = _required_sha(
        candidate.get("model_sha256"), "candidate.model_identity.model_sha256"
    )
    teacher_sha = _required_sha(
        teacher.get("model_sha256"), "teacher.model_identity.model_sha256"
    )
    if teacher.get("provider") != "ollama" or teacher.get("model") != TEACHER_MODEL:
        raise ManifestError(f"Teacher 必须是 Ollama {TEACHER_MODEL}")
    if teacher.get("stable_across_evaluation") is not True:
        raise ManifestError("Teacher summary 未证明评测前后模型与运行时稳定")
    before = _mapping(
        teacher.get("attestation_before"),
        "teacher.model_identity.attestation_before",
    )
    after = _mapping(
        teacher.get("attestation_after"),
        "teacher.model_identity.attestation_after",
    )
    for field in (
        "provider",
        "endpoint",
        "model",
        "expected_model_sha256",
        "model_sha256",
        "show_response_sha256",
        "version_response_sha256",
    ):
        if before.get(field) != after.get(field):
            raise ManifestError(f"Teacher 评测前后 attestation.{field} 漂移")
    if before.get("model_sha256") != teacher_sha:
        raise ManifestError("Teacher 顶层 model_sha256 与前后 attestation 不一致")
    expected_teacher_sha = _required_sha(
        teacher.get("expected_model_sha256"),
        "teacher.model_identity.expected_model_sha256",
    )
    if expected_teacher_sha != teacher_sha:
        raise ManifestError("Teacher 实际 digest 与预注册 digest 不一致")
    return {
        "candidate_model_sha256": candidate_sha,
        "teacher_model": TEACHER_MODEL,
        "teacher_model_sha256": teacher_sha,
    }


def _paired_rows(
    candidate: Mapping[str, Mapping[str, Any]],
    teacher: Mapping[str, Mapping[str, Any]],
) -> List[Tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if set(candidate) != set(teacher):
        missing_candidate = sorted(set(teacher) - set(candidate))
        missing_teacher = sorted(set(candidate) - set(teacher))
        raise ManifestError(
            "candidate/teacher sample_id 集合不一致: "
            f"candidate缺{missing_candidate[:5]} teacher缺{missing_teacher[:5]}"
        )
    pairs = []
    for sample_id in sorted(candidate):
        left = candidate[sample_id]
        right = teacher[sample_id]
        for field in ("category", "reference", "row_sha256"):
            if left[field] != right[field]:
                raise ManifestError(f"样本 {sample_id} 的 {field} 不一致")
        pairs.append((left, right))
    return pairs


def _bootstrap(
    pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
    seed: int,
    iterations: int,
) -> Dict[str, Any]:
    if iterations <= 0:
        raise ManifestError("bootstrap_iterations 必须大于 0")
    rng = random.Random(seed)
    deltas: List[float] = []
    retentions: List[float] = []
    attempts = 0
    max_attempts = iterations * 100
    size = len(pairs)
    while len(retentions) < iterations and attempts < max_attempts:
        attempts += 1
        sampled = [pairs[rng.randrange(size)] for _ in range(size)]
        candidate_accuracy = sum(bool(left["correct"]) for left, _ in sampled) / size
        teacher_accuracy = sum(bool(right["correct"]) for _, right in sampled) / size
        if teacher_accuracy <= 0:
            continue
        deltas.append(candidate_accuracy - teacher_accuracy)
        retentions.append(candidate_accuracy / teacher_accuracy)
    if len(retentions) != iterations:
        raise ManifestError("Teacher 正确样本过少，无法形成固定次数 retention bootstrap")
    return {
        "seed": seed,
        "iterations": iterations,
        "discarded_zero_teacher_accuracy_resamples": attempts - iterations,
        "accuracy_delta_95ci": [
            round(_percentile(deltas, 0.025), 6),
            round(_percentile(deltas, 0.975), 6),
        ],
        "retention_95ci": [
            round(_percentile(retentions, 0.025), 6),
            round(_percentile(retentions, 0.975), 6),
        ],
    }


def _category_metrics(
    pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
    seed: int,
    iterations: int,
) -> Dict[str, Any]:
    total = len(pairs)
    candidate_correct = sum(bool(left["correct"]) for left, _ in pairs)
    teacher_correct = sum(bool(right["correct"]) for _, right in pairs)
    candidate_valid = sum(bool(left["valid"]) for left, _ in pairs)
    teacher_valid = sum(bool(right["valid"]) for _, right in pairs)
    if teacher_correct == 0:
        raise ManifestError("Teacher 准确率为 0，retention 无定义")
    corrected = sum(
        (not bool(right["correct"])) and bool(left["correct"])
        for left, right in pairs
    )
    regressed = sum(
        bool(right["correct"]) and (not bool(left["correct"]))
        for left, right in pairs
    )
    candidate_accuracy = candidate_correct / total
    teacher_accuracy = teacher_correct / total
    retention = candidate_accuracy / teacher_accuracy
    result = {
        "sample_count": total,
        "candidate": {
            "correct": candidate_correct,
            "accuracy": round(candidate_accuracy, 6),
            "valid": candidate_valid,
            "valid_rate": round(candidate_valid / total, 6),
        },
        "teacher": {
            "correct": teacher_correct,
            "accuracy": round(teacher_accuracy, 6),
            "valid": teacher_valid,
            "valid_rate": round(teacher_valid / total, 6),
        },
        "retention": round(retention, 6),
        "teacher_correct_set_candidate_coverage": round(
            (teacher_correct - regressed) / teacher_correct, 6
        ),
        "corrected_teacher_errors": corrected,
        "regressed_teacher_correct": regressed,
        "net_corrections": corrected - regressed,
        "accuracy_delta": round(candidate_accuracy - teacher_accuracy, 6),
        "bootstrap": _bootstrap(pairs, seed, iterations),
    }
    result["checks"] = {
        "retention_at_least_0p80": retention >= RETENTION_MINIMUM,
        "candidate_valid_rate_equals_1": candidate_valid == total,
    }
    result["passed"] = all(result["checks"].values())
    return result


def run_gate(
    *,
    candidate_samples_path: Path,
    candidate_summary_path: Path,
    teacher_samples_path: Path,
    teacher_summary_path: Path,
    output_path: Path,
    candidate_adapter_model_path: Path,
    expected_candidate_adapter_sha256: str,
    expected_candidate_evaluator_sha256: str,
    expected_teacher_evaluator_sha256: str,
    seed: int,
    bootstrap_iterations: int,
) -> Dict[str, Any]:
    if output_path.exists():
        raise ManifestError(f"拒绝覆盖已有门禁结果: {output_path}")
    candidate_summary = read_json_object(candidate_summary_path)
    teacher_summary = read_json_object(teacher_summary_path)
    _verify_development_summary(candidate_summary, "candidate", CANDIDATE_SCHEMA)
    _verify_development_summary(teacher_summary, "teacher", TEACHER_SCHEMA)
    _verify_candidate_protocol(candidate_summary)
    _verify_teacher_protocol(teacher_summary)
    candidate_evaluator = _verify_evaluator_binding(
        candidate_summary,
        side="candidate",
        expected_sha256=expected_candidate_evaluator_sha256,
        current_path=CANDIDATE_EVALUATOR_PATH,
        require_declared_bytes=True,
    )
    teacher_evaluator = _verify_evaluator_binding(
        teacher_summary,
        side="teacher",
        expected_sha256=expected_teacher_evaluator_sha256,
        current_path=TEACHER_EVALUATOR_PATH,
        require_declared_bytes=False,
    )
    candidate_adapter = _verify_candidate_adapter_binding(
        candidate_summary,
        adapter_model_path=candidate_adapter_model_path,
        expected_sha256=expected_candidate_adapter_sha256,
    )
    candidate_samples_identity = _verify_samples_binding(
        candidate_samples_path, candidate_summary, "candidate"
    )
    teacher_samples_identity = _verify_samples_binding(
        teacher_samples_path, teacher_summary, "teacher"
    )
    selection = _verify_same_selection(candidate_summary, teacher_summary)
    models = _model_binding(candidate_summary, teacher_summary)
    candidate = _read_samples(candidate_samples_path, "candidate")
    teacher = _read_samples(teacher_samples_path, "teacher")
    _verify_summary_metrics(candidate_summary, candidate, "candidate")
    _verify_summary_metrics(teacher_summary, teacher, "teacher")
    pairs = _paired_rows(candidate, teacher)
    if len(pairs) != selection["sample_count"]:
        raise ManifestError("summary selection.sample_count 与逐题配对数不一致")
    categories: Dict[str, Any] = {}
    for offset, category in enumerate(GENERAL_CATEGORIES):
        selected = [pair for pair in pairs if pair[0]["category"] == category]
        if len(selected) != EXPECTED_SAMPLES_PER_CATEGORY:
            raise ManifestError(
                f"配对结果类别 {category} 必须恰好有 "
                f"{EXPECTED_SAMPLES_PER_CATEGORY} 条"
            )
        categories[category] = _category_metrics(
            selected, seed + offset, bootstrap_iterations
        )
    passed = all(value["passed"] for value in categories.values())
    report = {
        "schema_version": GATE_SCHEMA_VERSION,
        "task": "unified_general_candidate_vs_qwen3.5_9b_teacher_paired_gate",
        "development_only": True,
        "formal_blind_test_used": False,
        "thresholds": {
            "category_retention_minimum": RETENTION_MINIMUM,
            "candidate_valid_rate_required": VALID_RATE_REQUIRED,
        },
        "selection_binding": selection,
        "model_binding": models,
        "inputs": {
            "evaluation_implementations": {
                "candidate": candidate_evaluator,
                "teacher": teacher_evaluator,
            },
            "candidate": {
                "adapter_model": candidate_adapter,
                "samples": candidate_samples_identity,
                "summary": {
                    "path": str(candidate_summary_path.resolve()),
                    "bytes": candidate_summary_path.stat().st_size,
                    "sha256": sha256_file(candidate_summary_path),
                },
            },
            "teacher": {
                "samples": teacher_samples_identity,
                "summary": {
                    "path": str(teacher_summary_path.resolve()),
                    "bytes": teacher_summary_path.stat().st_size,
                    "sha256": sha256_file(teacher_summary_path),
                },
            },
            "gate_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "categories": categories,
        "passed": passed,
    }
    write_json_object(output_path, report)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_samples", required=True)
    parser.add_argument("--candidate_summary", required=True)
    parser.add_argument("--teacher_samples", required=True)
    parser.add_argument("--teacher_summary", required=True)
    parser.add_argument("--candidate_adapter_model", required=True)
    parser.add_argument("--expected_candidate_adapter_sha256", required=True)
    parser.add_argument("--expected_candidate_evaluator_sha256", required=True)
    parser.add_argument("--expected_teacher_evaluator_sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.seed < 0:
        raise ManifestError("seed 不能为负数")
    report = run_gate(
        candidate_samples_path=Path(args.candidate_samples).resolve(),
        candidate_summary_path=Path(args.candidate_summary).resolve(),
        teacher_samples_path=Path(args.teacher_samples).resolve(),
        teacher_summary_path=Path(args.teacher_summary).resolve(),
        output_path=Path(args.output).resolve(),
        candidate_adapter_model_path=Path(args.candidate_adapter_model).resolve(),
        expected_candidate_adapter_sha256=args.expected_candidate_adapter_sha256,
        expected_candidate_evaluator_sha256=args.expected_candidate_evaluator_sha256,
        expected_teacher_evaluator_sha256=args.expected_teacher_evaluator_sha256,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
