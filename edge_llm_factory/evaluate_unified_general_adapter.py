"""评估统一交通/通用 LoRA 的数学与中文逻辑开发集能力。

这个入口与 ``evaluate_general_adapter.py`` 分离。后者只允许纯通用 LoRA，
这里则明确允许从交通 LoRA 继续训练得到的统一候选。评测器只读取统一数据
构建器产出的 ``general_dev_evaluation.jsonl``；v1 将其视为训练开发集，v2
将其严格锁定为独立晋级开发集。数据 manifest 必须证明 GSM8K/LogiQA2.0
以及交通的正式测试内容没有被加载。

每道题严格使用 JSONL 自身的 ``system_prompt`` 和 ``prompt``：数学答案必须
以 ``FINAL: <number>`` 结束；中文逻辑输出必须完整地等于一个 A-D 字符。
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from edge_llm_factory.contracts import (
    HEX64,
    ManifestError,
    canonical_sha256,
    read_json_object,
    sha256_file,
    validate_base_manifest,
    write_json_object,
)
from edge_llm_factory.text_base import verify_text_snapshot


EVALUATION_SCHEMA_VERSION = "edge-llm-unified-general-dev-evaluation/v1"
DATASET_SCHEMA_VERSION_V1 = "edge-llm-unified-traffic-general/v1"
DATASET_SCHEMA_VERSION_V2 = "edge-llm-unified-traffic-general/v2"
# Backward-compatible public name used by the existing v1 tests and callers.
DATASET_SCHEMA_VERSION = DATASET_SCHEMA_VERSION_V1
SUPPORTED_DATASET_SCHEMA_VERSIONS = {
    DATASET_SCHEMA_VERSION_V1,
    DATASET_SCHEMA_VERSION_V2,
}
U2_REQUIRED_FALSE_ISOLATION_FIELDS = (
    "traffic_test_used_for_training",
    "gsm8k_test_loaded",
    "logiqa_test_loaded",
    "formal_evaluation_used_for_training",
    "u1_evaluation_outputs_used_for_training",
    "blind_evaluation_used_for_training",
    "promotion_dev_used_for_training",
    "promotion_dev_used_for_model_selection",
)
GENERAL_CATEGORIES = ("math", "natural_language_reasoning")
MAX_NEW_TOKENS = {"math": 256, "natural_language_reasoning": 32}
MATH_FINAL_PATTERN = re.compile(
    r"FINAL\s*:\s*([-+$]?[0-9][0-9,]*(?:\.[0-9]+)?)\s*$",
    re.IGNORECASE,
)
CHOICE_PATTERN = re.compile(r"\s*([A-D])\s*", re.IGNORECASE)


def _required_text(row: Mapping[str, Any], field: str, location: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{location}.{field} 必须是非空字符串")
    return value.strip()


def _required_verbatim(row: Mapping[str, Any], field: str, location: str) -> str:
    """校验文本非空，但保留其原始字节语义供提示词渲染。"""
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{location}.{field} 必须是非空字符串")
    return value


def _canonical_row_sha256(row: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_row(row: Mapping[str, Any], location: str) -> Dict[str, Any]:
    sample_id = _required_text(row, "sample_id", location)
    category = _required_text(row, "category", location)
    if category not in GENERAL_CATEGORIES:
        raise ManifestError(f"{location}.category 只允许 math 或 natural_language_reasoning")
    # system/user 内容必须逐字送入 chat template，不能在评测器里 strip 或改写。
    system_prompt = _required_verbatim(row, "system_prompt", location)
    prompt = _required_verbatim(row, "prompt", location)
    reference = _required_text(row, "reference_answer", location)
    if category == "math":
        if _normalize_number(reference) is None:
            raise ManifestError(f"{location}.reference_answer 不是有效数值")
    else:
        reference = reference.upper()
        if reference not in {"A", "B", "C", "D"}:
            raise ManifestError(f"{location}.reference_answer 必须是 A、B、C 或 D")
    return {
        "sample_id": sample_id,
        "category": category,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "reference_answer": reference,
    }


def read_evaluation_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_number} JSON 无效") from exc
            if not isinstance(value, dict):
                raise ManifestError(f"{path}:{line_number} 必须是 JSON object")
            row = _validate_row(value, f"{path}:{line_number}")
            if row["sample_id"] in seen:
                raise ManifestError(f"开发集 sample_id 重复: {row['sample_id']}")
            seen.add(row["sample_id"])
            rows.append(row)
    if not rows:
        raise ManifestError("统一模型通用开发集为空")
    return rows


def _nested_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"统一数据 manifest 缺少对象字段 {field}")
    return value


def _required_integer(row: Mapping[str, Any], field: str, location: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{location}.{field} 必须是整数")
    return value


def _require_exact_false(row: Mapping[str, Any], field: str, location: str) -> None:
    if row.get(field) is not False:
        raise ManifestError(f"{location}.{field} 必须显式为 false")


def _validate_v1_isolation(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Preserve every v1 isolation check; v2 support must not weaken it."""
    if manifest.get("general_train_validation_prompt_overlap") != 0:
        raise ManifestError("统一数据训练/开发 prompt 存在重叠")
    if manifest.get("gsm8k_test_loaded") is not False:
        raise ManifestError("统一数据未证明 GSM8K test 未加载")
    if manifest.get("logiqa_test_loaded") is not False:
        raise ManifestError("统一数据未证明 LogiQA2.0 test 未加载")
    sources = _nested_mapping(manifest.get("sources"), "sources")
    gsm8k = _nested_mapping(sources.get("gsm8k"), "sources.gsm8k")
    logiqa = _nested_mapping(sources.get("logiqa2"), "sources.logiqa2")
    if gsm8k.get("test_split_loaded") is not False:
        raise ManifestError("GSM8K 来源没有锁定为仅 train split")
    if logiqa.get("test_split_loaded") is not False:
        raise ManifestError("LogiQA2.0 来源没有锁定为 train/dev")
    if logiqa.get("test_content_loaded") is not False:
        raise ManifestError("LogiQA2.0 test 内容被标记为已加载")
    return {
        "dataset_role": "training_development",
        "training_development_prompt_overlap": 0,
    }


def _validate_v2_isolation(
    manifest: Mapping[str, Any], artifact: Mapping[str, Any], dataset: Path
) -> Dict[str, Any]:
    """Validate the frozen U2 promotion set without touching any test source."""
    if _required_integer(artifact, "rows", "artifacts.general_dev_evaluation") != 800:
        raise ManifestError("U2 promotion dev artifact 必须声明 rows=800")
    declared_bytes = _required_integer(
        artifact, "bytes", "artifacts.general_dev_evaluation"
    )
    if declared_bytes != dataset.stat().st_size:
        raise ManifestError("U2 promotion dev artifact bytes 与实际文件不一致")
    if manifest.get("promotion_dev_rows") != 800:
        raise ManifestError("U2 promotion_dev_rows 必须为 800")
    expected_counts = {"math": 400, "natural_language_reasoning": 400}
    if manifest.get("promotion_dev_category_counts") != expected_counts:
        raise ManifestError("U2 promotion dev 必须是数学/中文逻辑各 400 条")

    zero_fields = (
        "general_train_validation_prompt_overlap",
        "general_train_promotion_prompt_overlap",
        "general_validation_promotion_prompt_overlap",
        "u1_general_dev_overlap_with_train_or_promotion",
    )
    for field in zero_fields:
        if manifest.get(field) != 0:
            raise ManifestError(f"U2 隔离字段必须为 0: {field}")
    for field in (
        "traffic_test_used_for_training",
        "gsm8k_test_loaded",
        "logiqa_test_loaded",
        "formal_evaluation_used_for_training",
        "u1_evaluation_outputs_used_for_training",
        "blind_evaluation_used_for_training",
        "promotion_dev_used_for_training",
        "promotion_dev_used_for_model_selection",
        "u1_general_dev_used_for_promotion",
    ):
        _require_exact_false(manifest, field, "manifest")
    if manifest.get("u1_general_dev_used_as_training_validation") is not True:
        raise ManifestError(
            "U2 必须如实声明 U1 general dev 只作训练期 validation"
        )

    policy = _nested_mapping(
        manifest.get("blind_content_policy"), "blind_content_policy"
    )
    _require_exact_false(
        policy,
        "test_content_used_for_training_or_development",
        "blind_content_policy",
    )
    if policy.get("traffic_test") != "sha256_and_stat_only":
        raise ManifestError("U2 traffic test 只允许 SHA-256/stat 取证")
    if policy.get("gsm8k_test") != "split_not_loaded":
        raise ManifestError("U2 GSM8K test split 必须未加载")
    if policy.get("logiqa_test") != "sha256_and_stat_only":
        raise ManifestError("U2 LogiQA test 只允许 SHA-256/stat 取证")

    sources = _nested_mapping(manifest.get("sources"), "sources")
    traffic = _nested_mapping(sources.get("traffic"), "sources.traffic")
    traffic_test = _nested_mapping(
        traffic.get("test"), "sources.traffic.test"
    )
    _require_exact_false(traffic_test, "content_loaded", "sources.traffic.test")
    _require_exact_false(
        traffic_test, "used_for_training", "sources.traffic.test"
    )
    if traffic_test.get("access_mode") != "sha256_and_stat_only":
        raise ManifestError("U2 traffic test access_mode 不是哈希/stat only")

    gsm8k = _nested_mapping(sources.get("gsm8k"), "sources.gsm8k")
    _require_exact_false(gsm8k, "test_split_loaded", "sources.gsm8k")
    logiqa = _nested_mapping(sources.get("logiqa2"), "sources.logiqa2")
    _require_exact_false(logiqa, "test_split_loaded", "sources.logiqa2")
    logiqa_test = _nested_mapping(
        logiqa.get("test"), "sources.logiqa2.test"
    )
    _require_exact_false(logiqa_test, "content_loaded", "sources.logiqa2.test")
    if logiqa_test.get("access_mode") != "sha256_and_stat_only":
        raise ManifestError("U2 LogiQA test access_mode 不是哈希/stat only")

    rows = read_evaluation_rows(dataset)
    actual_counts = dict(sorted(Counter(row["category"] for row in rows).items()))
    if len(rows) != 800 or actual_counts != expected_counts:
        raise ManifestError("U2 promotion dev 实际内容必须是数学/中文逻辑各 400 条")
    return {
        "dataset_role": "promotion_development",
        "promotion_rows": 800,
        "promotion_category_counts": expected_counts,
        "training_development_prompt_overlap": 0,
        "training_promotion_prompt_overlap": 0,
        "validation_promotion_prompt_overlap": 0,
        "u1_general_dev_used_for_promotion": False,
    }


def validate_dataset_binding(
    dataset: Path, manifest_path: Path
) -> Dict[str, Any]:
    manifest = read_json_object(manifest_path)
    schema_version = manifest.get("schema_version")
    if schema_version not in SUPPORTED_DATASET_SCHEMA_VERSIONS:
        raise ManifestError("统一数据 manifest schema_version 不匹配")
    artifacts = _nested_mapping(manifest.get("artifacts"), "artifacts")
    dev = _nested_mapping(
        artifacts.get("general_dev_evaluation"), "artifacts.general_dev_evaluation"
    )
    declared_path = _required_text(dev, "path", "artifacts.general_dev_evaluation")
    declared_sha = _required_text(dev, "sha256", "artifacts.general_dev_evaluation")
    if schema_version == DATASET_SCHEMA_VERSION_V2 and Path(declared_path).parts != (
        "general_dev_evaluation.jsonl",
    ):
        raise ManifestError(
            "U2 评测器只允许读取 manifest 同目录的 general_dev_evaluation.jsonl"
        )
    expected_path = (manifest_path.parent / declared_path).resolve()
    if dataset.resolve() != expected_path:
        raise ManifestError("开发集路径不是统一数据 manifest 锁定的文件")
    actual_sha = sha256_file(dataset)
    if actual_sha != declared_sha:
        raise ManifestError("开发集 SHA-256 与统一数据 manifest 不一致")
    if schema_version == DATASET_SCHEMA_VERSION_V1:
        isolation = _validate_v1_isolation(manifest)
    else:
        isolation = _validate_v2_isolation(manifest, dev, dataset)
    return {
        "path": str(dataset.resolve()),
        "bytes": dataset.stat().st_size,
        "sha256": actual_sha,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "schema_version": schema_version,
        "blind_test_content_loaded": False,
        **isolation,
    }


def _normalize_number(value: str) -> Optional[Decimal]:
    clean = value.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return Decimal(clean)
    except InvalidOperation:
        return None


def evaluate_math(text: str, reference: str) -> Dict[str, Any]:
    match = MATH_FINAL_PATTERN.search(text)
    prediction_text = match.group(1) if match else ""
    prediction = _normalize_number(prediction_text)
    expected = _normalize_number(reference)
    valid = prediction is not None
    return {
        "prediction": prediction_text or None,
        "reference": reference,
        "valid": valid,
        "correct": bool(valid and expected is not None and prediction == expected),
    }


def evaluate_choice(text: str, reference: str) -> Dict[str, Any]:
    match = CHOICE_PATTERN.fullmatch(text)
    prediction = match.group(1).upper() if match else None
    return {
        "prediction": prediction,
        "reference": reference,
        "valid": prediction is not None,
        "correct": prediction == reference,
    }


def evaluate_response(text: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    if row["category"] == "math":
        return evaluate_math(text, str(row["reference_answer"]))
    return evaluate_choice(text, str(row["reference_answer"]))


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ManifestError("不能对空数组计算分位数")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_summary(samples: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    values = [float(sample["generation_ms"]) for sample in samples]
    return {
        "mean_ms": round(statistics.fmean(values), 4),
        "p50_ms": round(_percentile(values, 0.50), 4),
        "p95_ms": round(_percentile(values, 0.95), 4),
        "max_ms": round(max(values), 4),
    }


def summarize(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ManifestError("没有可汇总的开发集结果")

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        total = len(selected)
        correct = sum(bool(sample["correct"]) for sample in selected)
        valid = sum(bool(sample["valid"]) for sample in selected)
        return {
            "sample_count": total,
            "correct": correct,
            "valid": valid,
            "accuracy": round(correct / total, 6),
            "valid_rate": round(valid / total, 6),
            "latency": _latency_summary(selected),
        }

    categories = {
        category: aggregate(
            [sample for sample in samples if sample["category"] == category]
        )
        for category in GENERAL_CATEGORIES
        if any(sample["category"] == category for sample in samples)
    }
    result = aggregate(samples)
    result["macro_accuracy"] = round(
        statistics.fmean(value["accuracy"] for value in categories.values()), 6
    )
    result["categories"] = categories
    return result


GenerationFunction = Callable[
    [Sequence[Mapping[str, str]], int, Mapping[str, Any]], Mapping[str, Any]
]


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]], generator: GenerationFunction
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for row in rows:
        messages = [
            {"role": "system", "content": str(row["system_prompt"])},
            {"role": "user", "content": str(row["prompt"])},
        ]
        observation = dict(
            generator(messages, MAX_NEW_TOKENS[str(row["category"])], row)
        )
        text = observation.get("raw_output")
        if not isinstance(text, str):
            raise ManifestError("生成器必须返回字符串 raw_output")
        for field in ("prompt_tokens", "output_tokens"):
            value = observation.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ManifestError(f"生成器 {field} 必须是非负整数")
        generation_ms = observation.get("generation_ms")
        if (
            isinstance(generation_ms, bool)
            or not isinstance(generation_ms, (int, float))
            or float(generation_ms) < 0
        ):
            raise ManifestError("生成器 generation_ms 必须是非负数")
        evaluated = evaluate_response(text.strip(), row)
        samples.append(
            {
                "sample_id": str(row["sample_id"]),
                "category": str(row["category"]),
                "row_sha256": _canonical_row_sha256(row),
                "system_prompt_sha256": hashlib.sha256(
                    str(row["system_prompt"]).encode("utf-8")
                ).hexdigest(),
                "prompt_sha256": hashlib.sha256(
                    str(row["prompt"]).encode("utf-8")
                ).hexdigest(),
                "prompt_tokens": int(observation["prompt_tokens"]),
                "output_tokens": int(observation["output_tokens"]),
                "generation_ms": round(float(generation_ms), 4),
                "raw_output": text.strip(),
                **evaluated,
            }
        )
    return samples


def _file_identity(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def evaluator_identity() -> Dict[str, Any]:
    """返回当前候选评测器的不可变文件身份，供结果与门禁交叉核验。"""
    return _file_identity(Path(__file__).resolve())


def _training_data_binding(
    metrics: Mapping[str, Any], dataset_manifest_sha256: str
) -> Optional[Dict[str, Any]]:
    """Read both the legacy v1 receipt and the nested U2 receipt strictly."""
    nested_dataset = metrics.get("dataset")
    is_u2_receipt = metrics.get("task") == (
        "single_model_traffic_math_chinese_logic_lora_u2"
    ) or (
        isinstance(nested_dataset, Mapping)
        and (
            nested_dataset.get("schema_version") == DATASET_SCHEMA_VERSION_V2
            or "manifest_sha256" in nested_dataset
        )
    )
    if is_u2_receipt:
        nested_dataset = _nested_mapping(nested_dataset, "train_metrics.dataset")
        if nested_dataset.get("schema_version") != DATASET_SCHEMA_VERSION_V2:
            raise ManifestError("统一 LoRA U2 train_metrics 数据 schema_version 无效")
        declared_manifest = _required_text(
            nested_dataset,
            "manifest_sha256",
            "train_metrics.dataset",
        ).lower()
        if HEX64.fullmatch(declared_manifest) is None:
            raise ManifestError("统一 LoRA U2 训练 manifest SHA-256 无效")
        if declared_manifest != dataset_manifest_sha256:
            raise ManifestError("统一 LoRA 训练 manifest 与本次开发集不一致")
        isolation = _nested_mapping(
            nested_dataset.get("isolation"), "train_metrics.dataset.isolation"
        )
        for field in U2_REQUIRED_FALSE_ISOLATION_FIELDS:
            _require_exact_false(isolation, field, "train_metrics.dataset.isolation")
        _require_exact_false(
            nested_dataset,
            "promotion_dev_content_read_by_trainer",
            "train_metrics.dataset",
        )
        return {
            "receipt_format": "u2_nested_dataset",
            "dataset_schema_version": DATASET_SCHEMA_VERSION_V2,
            "dataset_manifest_sha256": declared_manifest,
            "isolation": {field: False for field in U2_REQUIRED_FALSE_ISOLATION_FIELDS},
            "promotion_dev_content_read_by_trainer": False,
            "blind_test_content_loaded": False,
        }

    # Preserve the v1 receipt layout. Older receipts which did not bind a dataset
    # remain readable; once the legacy manifest field is present all legacy
    # isolation declarations are mandatory and exact.
    declared_manifest = metrics.get("dataset_manifest_sha256")
    if declared_manifest is None:
        return None
    declared_manifest = _required_text(
        metrics, "dataset_manifest_sha256", "train_metrics"
    ).lower()
    if HEX64.fullmatch(declared_manifest) is None:
        raise ManifestError("统一 LoRA v1 训练 manifest SHA-256 无效")
    if declared_manifest != dataset_manifest_sha256:
        raise ManifestError("统一 LoRA 训练 manifest 与本次开发集不一致")
    legacy_fields = (
        "formal_evaluation_used_for_training",
        "traffic_test_used_for_training",
        "gsm8k_test_loaded",
        "logiqa_test_loaded",
    )
    for field in legacy_fields:
        _require_exact_false(metrics, field, "train_metrics")
    return {
        "receipt_format": "v1_top_level",
        "dataset_manifest_sha256": declared_manifest,
        "isolation": {field: False for field in legacy_fields},
        "formal_evaluation_used_for_training": False,
        "blind_test_content_loaded": False,
    }


def adapter_identity(
    adapter: Optional[Path],
    base: Mapping[str, Any],
    dataset_manifest_sha256: str,
    expected_weights_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if adapter is None:
        return None
    if not adapter.is_dir():
        raise ManifestError("LoRA 目录不存在")
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    for path in (config_path, weights_path):
        if not path.is_file():
            raise ManifestError(f"LoRA 缺少 {path.name}")
    config = read_json_object(config_path)
    declared_base = config.get("base_model_name_or_path")
    if declared_base and declared_base != base["source"]["model_id"]:
        raise ManifestError("LoRA adapter_config 与锁定基座不兼容")
    files = {
        "adapter_config": _file_identity(config_path),
        "adapter_weights": _file_identity(weights_path),
    }
    weights_binding: Optional[Dict[str, Any]] = None
    if expected_weights_sha256 is not None:
        expected_weights_sha = expected_weights_sha256.strip().lower()
        if HEX64.fullmatch(expected_weights_sha) is None:
            raise ManifestError("expected_adapter_sha256 必须是 SHA-256")
        if files["adapter_weights"]["sha256"] != expected_weights_sha:
            raise ManifestError("LoRA adapter_model.safetensors 与外部 SHA-256 锚点不一致")
        weights_binding = {
            "expected_sha256": expected_weights_sha,
            "actual_sha256": files["adapter_weights"]["sha256"],
            "externally_anchored_and_recomputed": True,
        }
    metrics_path = adapter / "train_metrics.json"
    training_binding: Optional[Dict[str, Any]] = None
    if metrics_path.is_file():
        files["train_metrics"] = _file_identity(metrics_path)
        metrics = read_json_object(metrics_path)
        training_binding = _training_data_binding(metrics, dataset_manifest_sha256)
    artifact_sha = canonical_sha256(
        {
            name: {"bytes": value["bytes"], "sha256": value["sha256"]}
            for name, value in sorted(files.items())
        }
    )
    return {
        "path": str(adapter.resolve()),
        "artifact_sha256": artifact_sha,
        "files": files,
        "weights_binding": weights_binding,
        "training_data_binding": training_binding,
    }


def model_identity(
    base_manifest: Path,
    snapshot_manifest: Path,
    snapshot_validation: Mapping[str, Any],
    adapter: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    payload = {
        "base_manifest_sha256": sha256_file(base_manifest),
        "snapshot_manifest_sha256": sha256_file(snapshot_manifest),
        "snapshot_id": snapshot_validation.get("snapshot_id"),
        "adapter_artifact_sha256": (
            adapter.get("artifact_sha256") if adapter is not None else None
        ),
    }
    return {**payload, "model_sha256": canonical_sha256(payload)}


def _select_rows(
    rows: Sequence[Dict[str, Any]], categories: set[str], limit_per_category: int
) -> List[Dict[str, Any]]:
    if limit_per_category < 0:
        raise ManifestError("limit_per_category 不能为负数")
    counts: Dict[str, int] = {}
    selected: List[Dict[str, Any]] = []
    for row in rows:
        category = str(row["category"])
        if category not in categories:
            continue
        if limit_per_category and counts.get(category, 0) >= limit_per_category:
            continue
        selected.append(row)
        counts[category] = counts.get(category, 0) + 1
    missing = categories - set(counts)
    if missing:
        raise ManifestError(f"开发集缺少类别: {sorted(missing)}")
    return selected


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


class TransformersGenerator:
    def __init__(self, model: Any, tokenizer: Any, torch_module: Any, max_input_tokens: int):
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.max_input_tokens = max_input_tokens
        self.warmed = False

    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        max_new_tokens: int,
        row: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del row
        prompt = self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(
            prompt, add_special_tokens=False, truncation=False, return_tensors="pt"
        )
        prompt_tokens = int(encoded["input_ids"].shape[1])
        if prompt_tokens > self.max_input_tokens:
            raise ManifestError(
                f"输入 {prompt_tokens} tokens 超过上限 {self.max_input_tokens}；拒绝截断"
            )
        try:
            device = next(self.model.parameters()).device
        except StopIteration as exc:
            raise ManifestError("模型没有可定位设备的参数") from exc
        encoded = {name: value.to(device) for name, value in encoded.items()}
        generate_args = {
            **encoded,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if not self.warmed:
            with self.torch.inference_mode():
                self.model.generate(**generate_args, max_new_tokens=1)
            self.warmed = True
        if device.type == "cuda":
            self.torch.cuda.synchronize(device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **generate_args, max_new_tokens=max_new_tokens
            )
        if device.type == "cuda":
            self.torch.cuda.synchronize(device)
        generation_ms = (time.perf_counter() - started) * 1000.0
        output_ids = generated[0, prompt_tokens:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return {
            "raw_output": text,
            "prompt_tokens": prompt_tokens,
            "output_tokens": int(output_ids.shape[0]),
            "generation_ms": generation_ms,
        }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="锁定的 edge-llm-base manifest")
    parser.add_argument("--snapshot", required=True, help="纯文本基座快照目录")
    parser.add_argument("--snapshot_manifest", required=True)
    parser.add_argument("--adapter", help="当前交通 LoRA 或统一候选 LoRA")
    parser.add_argument(
        "--expected_adapter_sha256",
        help="U2 使用 --adapter 时必填的外部预注册 adapter_model.safetensors SHA-256",
    )
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument("--samples_output", required=True)
    parser.add_argument("--summary_output", required=True)
    parser.add_argument("--model_label", default="candidate")
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--limit_per_category", type=int, default=0)
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args(argv)

    if args.max_input_tokens <= 0:
        raise ManifestError("max_input_tokens 必须大于 0")
    label = args.model_label.strip()
    if not label:
        raise ManifestError("model_label 不能为空")
    categories = {value.strip() for value in args.category if value.strip()}
    if not categories:
        categories = set(GENERAL_CATEGORIES)
    unknown = categories - set(GENERAL_CATEGORIES)
    if unknown:
        raise ManifestError(f"未知类别: {sorted(unknown)}")

    dataset = Path(args.dataset_jsonl).resolve()
    dataset_manifest = Path(args.dataset_manifest).resolve()
    if not dataset.is_file() or not dataset_manifest.is_file():
        raise ManifestError("统一开发集或其 manifest 不存在")
    data_identity = validate_dataset_binding(dataset, dataset_manifest)
    rows = _select_rows(
        read_evaluation_rows(dataset), categories, args.limit_per_category
    )

    base_path = Path(args.base).resolve()
    snapshot_path = Path(args.snapshot).resolve()
    snapshot_manifest_path = Path(args.snapshot_manifest).resolve()
    base = validate_base_manifest(read_json_object(base_path))
    snapshot_validation = verify_text_snapshot(
        base,
        read_json_object(snapshot_manifest_path),
        snapshot_path,
        verify_tokenizer=True,
    )
    adapter_path = Path(args.adapter).resolve() if args.adapter else None
    if (
        adapter_path is not None
        and data_identity["schema_version"] == DATASET_SCHEMA_VERSION_V2
        and args.expected_adapter_sha256 is None
    ):
        raise ManifestError("U2 使用 adapter 评测时必须提供 expected_adapter_sha256")
    if adapter_path is None and args.expected_adapter_sha256 is not None:
        raise ManifestError("expected_adapter_sha256 只能与 adapter 同时使用")
    adapter_report = adapter_identity(
        adapter_path,
        base,
        data_identity["manifest_sha256"],
        args.expected_adapter_sha256,
    )
    identity = model_identity(
        base_path, snapshot_manifest_path, snapshot_validation, adapter_report
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: Dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["dtype"] = torch.bfloat16 if args.bf16 else torch.float16
        precision = "bfloat16" if args.bf16 else "float16"
    else:
        precision = "float32"
    model = AutoModelForCausalLM.from_pretrained(str(snapshot_path), **model_kwargs)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    generator = TransformersGenerator(model, tokenizer, torch, args.max_input_tokens)
    samples = evaluate_rows(rows, generator)
    metrics = summarize(samples)

    samples_output = Path(args.samples_output).resolve()
    summary_output = Path(args.summary_output).resolve()
    if samples_output == summary_output:
        raise ManifestError("逐题 JSONL 与 summary JSON 不能使用同一路径")
    for path in (samples_output, summary_output):
        if path.exists():
            raise ManifestError(f"拒绝覆盖已有评测结果: {path}")
    _write_jsonl(samples_output, samples)
    samples_identity = _file_identity(samples_output)
    selected_rows_sha = canonical_sha256(
        [_canonical_row_sha256(row) for row in rows]
    )
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ManifestError("tokenizer 缺少 chat_template")
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "task": "unified_model_math_and_chinese_logic_development_evaluation",
        "model_label": label,
        "development_only": True,
        "formal_blind_test_used": False,
        "dataset": data_identity,
        "selection": {
            "categories": sorted(categories),
            "limit_per_category": args.limit_per_category,
            "sample_count": len(rows),
            "selected_rows_sha256": selected_rows_sha,
        },
        "base_id": base["base_id"],
        "snapshot_validation": snapshot_validation,
        "adapter": adapter_report,
        "model_identity": identity,
        "evaluation_protocol": {
            "renderer": "tokenizer.apply_chat_template",
            "uses_each_row_system_prompt_verbatim": True,
            "uses_each_row_prompt_verbatim": True,
            "chat_template_sha256": hashlib.sha256(
                chat_template.encode("utf-8")
            ).hexdigest(),
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": dict(MAX_NEW_TOKENS),
            "decoding": {"do_sample": False},
            "precision": precision,
            "math_scoring": "strict_trailing_FINAL_numeric",
            "logic_scoring": "strict_full_output_A_to_D",
        },
        "metrics": metrics,
        "artifacts": {
            "samples": samples_identity,
            "evaluator": evaluator_identity(),
        },
    }
    write_json_object(summary_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
