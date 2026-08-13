"""在冻结通用题集上直接评估 Hugging Face 基座或 LoRA 候选。

这个入口不依赖 Ollama 的 chat/reasoning 包装，专门用于量化前的 F16/BF16
蒸馏门禁。交通等场景的单 token 测试仍由各自的冻结测试入口负责。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from edge_llm_factory.contracts import (
    ManifestError,
    read_json_object,
    sha256_file,
    validate_base_manifest,
    write_json_object,
)
from edge_llm_factory.text_base import verify_text_snapshot


GENERAL_CATEGORIES = ("code", "math", "natural_language_reasoning")
EMPTY_THINK_ASSISTANT_PREFILL = "<think>\n\n</think>\n\n"
EVALUATION_SCHEMA_VERSION = "edge-llm-general-adapter-evaluation/v3"
SELECTED_SAMPLE_SCHEMA_VERSION = "edge-llm-general-selected-samples/v1"
PROTOCOL_SCHEMA_VERSION = "edge-llm-general-evaluation-protocol/v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_NEW_TOKENS = {
    "math": 256,
    "code": 384,
    "natural_language_reasoning": 32,
}
ALLOWED_IMPORTS = {
    "bisect",
    "collections",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "re",
    "statistics",
    "string",
    "typing",
}
DANGEROUS_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}


def _required_string(row: Mapping[str, Any], field: str, location: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{location}.{field} 必须是非空字符串")
    return value.strip()


def _string_list(
    row: Mapping[str, Any], field: str, location: str, *, required: bool
) -> List[str]:
    value = row.get(field, [])
    if not isinstance(value, list) or (required and not value):
        qualifier = "非空" if required else ""
        raise ManifestError(f"{location}.{field} 必须是{qualifier}字符串数组")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ManifestError(f"{location}.{field} 必须是字符串数组且元素不能为空")
    return [item.strip() for item in value]


def _validate_evaluation_row(
    row: Mapping[str, Any], location: str
) -> Dict[str, Any]:
    category = _required_string(row, "category", location)
    if category not in GENERAL_CATEGORIES:
        raise ManifestError(f"冻结题集含未知类别: {category}")
    sample_id = _required_string(row, "sample_id", location)
    prompt = _required_string(row, "prompt", location)
    result = dict(row)
    if category == "code":
        result["test_list"] = _string_list(row, "test_list", location, required=True)
        result["test_imports"] = _string_list(
            row, "test_imports", location, required=False
        )
    else:
        reference = _required_string(row, "reference_answer", location)
        if category == "natural_language_reasoning" and reference.upper() not in {
            "A",
            "B",
            "C",
            "D",
        }:
            raise ManifestError(f"{location}.reference_answer 必须是 A、B、C 或 D")
        result["reference_answer"] = (
            reference.upper()
            if category == "natural_language_reasoning"
            else reference
        )
    result["sample_id"] = sample_id
    result["category"] = category
    result["prompt"] = prompt
    return result


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sample_ids = set()
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_number} JSON 无效") from exc
            if not isinstance(row, dict):
                raise ManifestError(f"{path}:{line_number} 必须是 JSON object")
            location = f"{path}:{line_number}"
            validated = _validate_evaluation_row(row, location)
            sample_id = validated["sample_id"]
            if sample_id in sample_ids:
                raise ManifestError(f"冻结题集 sample_id 重复: {sample_id}")
            sample_ids.add(sample_id)
            rows.append(validated)
    if not rows:
        raise ManifestError("冻结通用题集为空")
    return rows


def _task_messages(row: Mapping[str, Any]) -> tuple[List[Dict[str, str]], int]:
    category = str(row["category"])
    if category == "math":
        system = (
            "You solve grade-school math accurately. Do not provide a long explanation. "
            "End with exactly: FINAL: <number>"
        )
        user = str(row["prompt"])
        max_new_tokens = MAX_NEW_TOKENS[category]
    elif category == "code":
        tests = "\n".join(str(value) for value in row.get("test_list", []))
        system = (
            "Write a correct Python solution. Output only Python code without Markdown "
            "fences or explanation."
        )
        user = "{}\nThe function must pass these tests:\n{}".format(row["prompt"], tests)
        max_new_tokens = MAX_NEW_TOKENS[category]
    else:
        system = "回答中文单项选择题。只输出 A、B、C 或 D，不要解释。"
        user = str(row["prompt"])
        # 旧入口只给 4 token，部分后端先产生空 think wrapper，尚未生成答案即被截断。
        max_new_tokens = MAX_NEW_TOKENS[category]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], max_new_tokens


def _normalize_number(value: str) -> Optional[Decimal]:
    clean = value.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return Decimal(clean)
    except InvalidOperation:
        return None


def _evaluate_math(text: str, reference: str) -> Dict[str, Any]:
    # The contract says the response must *end* with one numeric final answer.
    # Do not accept expressions such as ``FINAL: 90 - 20 = 70`` by silently
    # taking the first number, and do not fall back to an arbitrary number from
    # an unfinished/truncated derivation.  Both behaviours can create false
    # positives in a retention gate.
    final_match = re.search(
        r"FINAL\s*:\s*([-+$]?[0-9][0-9,]*(?:\.[0-9]+)?)\s*$",
        text,
        re.IGNORECASE,
    )
    prediction_text = final_match.group(1) if final_match else ""
    prediction = _normalize_number(prediction_text)
    expected = _normalize_number(reference)
    return {
        "prediction": prediction_text or None,
        "reference": reference,
        "correct": prediction is not None and expected is not None and prediction == expected,
    }


def _evaluate_choice(text: str, reference: str) -> Dict[str, Any]:
    # 优先读取 </think> 后的答案，兼容空 reasoning wrapper；答案本身必须
    # 严格为一个 A-D 字符，不能从解释或多个候选字母中挑一个算对。
    answer_text = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    match = re.fullmatch(r"\s*([A-D])\s*", answer_text, re.IGNORECASE)
    prediction = match.group(1) if match else None
    if prediction is not None:
        prediction = prediction.upper()
    return {
        "prediction": prediction,
        "reference": reference,
        "correct": prediction == reference,
    }


def _extract_code(text: str) -> str:
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    code = fence.group(1).strip() if fence else text.strip()
    lines = code.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"\s*(?:from\s+\w+\s+import|import\s+\w+|def\s+\w+)", line):
            return "\n".join(lines[index:]).strip()
    return code


def _validate_code_ast(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax_error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] not in ALLOWED_IMPORTS:
                    return f"blocked_import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if module not in ALLOWED_IMPORTS:
                return f"blocked_import: {node.module}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in DANGEROUS_CALLS:
                return f"blocked_call: {node.func.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"blocked_dunder_attribute: {node.attr}"
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            return f"blocked_dunder_name: {node.id}"
    return None


def _code_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def _evaluate_code(text: str, row: Mapping[str, Any], timeout: float) -> Dict[str, Any]:
    code = _extract_code(text)
    imports = "\n".join(str(value) for value in row.get("test_imports", []))
    tests = "\n".join(str(value) for value in row.get("test_list", []))
    program = f"{imports}\n{code}\n{tests}\n"
    validation_error = _validate_code_ast(program)
    if validation_error:
        return {
            "prediction": code,
            "reference": "official MBPP tests",
            "correct": False,
            "execution_error": validation_error,
        }
    try:
        with tempfile.TemporaryDirectory(prefix="general_adapter_code_") as directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-"],
                input=program,
                text=True,
                cwd=directory,
                capture_output=True,
                timeout=timeout,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
                preexec_fn=_code_limits,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return {
            "prediction": code,
            "reference": "official MBPP tests",
            "correct": False,
            "execution_error": "timeout",
        }
    return {
        "prediction": code,
        "reference": "official MBPP tests",
        "correct": completed.returncode == 0,
        "execution_error": None if completed.returncode == 0 else completed.stderr[-1000:],
    }


def evaluate_response(text: str, row: Mapping[str, Any], code_timeout: float) -> Dict[str, Any]:
    category = str(row["category"])
    if category == "code":
        return _evaluate_code(text, row, code_timeout)
    reference = str(row["reference_answer"])
    if category == "math":
        return _evaluate_math(text, reference)
    return _evaluate_choice(text, reference)


def _summary(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    categories: Dict[str, Dict[str, Any]] = {}
    for category in sorted({str(sample["category"]) for sample in samples}):
        selected = [sample for sample in samples if sample["category"] == category]
        score = sum(bool(sample["correct"]) for sample in selected) / len(selected)
        categories[category] = {
            "total": len(selected),
            "sample_count": len(selected),
            "correct": sum(bool(sample["correct"]) for sample in selected),
            "score": round(score, 6),
            "average_generation_ms": round(
                statistics.fmean(float(sample["generation_ms"]) for sample in selected), 4
            ),
        }
    return {
        "completed_samples": len(samples),
        "overall_micro_score": round(
            sum(bool(sample["correct"]) for sample in samples) / len(samples), 6
        ),
        "overall_macro_score": round(
            statistics.fmean(float(value["score"]) for value in categories.values()), 6
        ),
        "categories": categories,
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _category_sample_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {
        category: sum(str(row["category"]) == category for row in rows)
        for category in sorted({str(row["category"]) for row in rows})
    }


def _select_rows(
    rows: Sequence[Dict[str, Any]], categories: set[str], limit_per_category: int
) -> List[Dict[str, Any]]:
    if limit_per_category < 0:
        raise ManifestError("limit_per_category 不能为负数")
    selected = sorted(
        (row for row in rows if str(row["category"]) in categories),
        key=lambda row: (str(row["category"]), str(row["sample_id"])),
    )
    if limit_per_category > 0:
        kept: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        for row in selected:
            category = str(row["category"])
            if counts.get(category, 0) >= limit_per_category:
                continue
            kept.append(row)
            counts[category] = counts.get(category, 0) + 1
        selected = kept
    observed = {str(row["category"]) for row in selected}
    missing = categories - observed
    if missing:
        raise ManifestError(f"筛选后的冻结题集缺少类别: {sorted(missing)}")
    return selected


def _selected_sample_manifest(
    rows: Sequence[Mapping[str, Any]], dataset_sha256: str
) -> Dict[str, Any]:
    if SHA256_PATTERN.fullmatch(dataset_sha256) is None:
        raise ManifestError("冻结题集 SHA-256 无效")
    samples = [
        {
            "sample_id": str(row["sample_id"]),
            "category": str(row["category"]),
            "row_sha256": hashlib.sha256(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        for row in rows
    ]
    payload: Dict[str, Any] = {
        "schema_version": SELECTED_SAMPLE_SCHEMA_VERSION,
        "dataset_sha256": dataset_sha256,
        "sample_count": len(samples),
        "category_sample_counts": _category_sample_counts(rows),
        "samples": samples,
    }
    payload["sha256"] = _canonical_sha256(payload)
    return payload


def _evaluation_protocol(
    tokenizer: Any,
    *,
    assistant_prefill: str,
    max_input_tokens: int,
    precision: str,
) -> Dict[str, Any]:
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ManifestError("tokenizer 缺少可锁定的 chat_template")
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "backend_family": "transformers",
        "renderer": "tokenizer.apply_chat_template",
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
        "assistant_prefill": assistant_prefill,
        "max_input_tokens": max_input_tokens,
        "max_output_tokens_by_category": dict(sorted(MAX_NEW_TOKENS.items())),
        "decoding": {"do_sample": False},
        "precision": precision,
    }


def _manifest_without_sha(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    required = {
        "schema_version",
        "dataset_sha256",
        "sample_count",
        "category_sample_counts",
        "samples",
    }
    if not required.issubset(manifest):
        raise ManifestError("Teacher selected_sample_manifest 字段不完整")
    return {field: manifest[field] for field in sorted(required)}


def _teacher_payload(evaluation: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    models = evaluation.get("models")
    if models is None:
        return evaluation
    if not isinstance(models, Mapping) or label not in models:
        raise ManifestError(f"Teacher 评测缺少模型标签 {label}")
    payload = models[label]
    if not isinstance(payload, Mapping):
        raise ManifestError(f"Teacher 模型标签 {label} 必须是 JSON object")
    return payload


def _teacher_scores(
    path: Optional[Path],
    label: str,
    *,
    dataset_sha256: str,
    selected_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[Dict[str, float], Optional[Dict[str, Any]]]:
    if path is None:
        return {}, None
    evaluation = read_json_object(path)
    if evaluation.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ManifestError(
            "Teacher 评测必须由同版本 evaluate_general_adapter 生成"
        )
    dataset = evaluation.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("sha256") != dataset_sha256:
        raise ManifestError("Teacher 与候选没有绑定同一冻结题集 SHA-256")
    teacher_protocol = evaluation.get("evaluation_protocol")
    if teacher_protocol != protocol:
        raise ManifestError("Teacher 与候选的评测协议不一致")

    declared_manifest = evaluation.get("selected_sample_manifest")
    if not isinstance(declared_manifest, Mapping):
        raise ManifestError("Teacher 评测缺少 selected_sample_manifest")
    declared_payload = _manifest_without_sha(declared_manifest)
    declared_sha = declared_manifest.get("sha256")
    if (
        not isinstance(declared_sha, str)
        or SHA256_PATTERN.fullmatch(declared_sha) is None
        or _canonical_sha256(declared_payload) != declared_sha
    ):
        raise ManifestError("Teacher selected_sample_manifest 自身哈希无效")
    if dict(declared_manifest) != dict(selected_manifest):
        raise ManifestError("Teacher 与候选的实际选中样本清单不一致")

    payload = _teacher_payload(evaluation, label)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ManifestError("Teacher 评测缺少逐样本结果")
    observed_samples = []
    observed_correct: Dict[str, int] = {}
    observed_ids = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ManifestError(f"Teacher samples[{index}] 必须是 JSON object")
        sample_id = _required_string(sample, "sample_id", f"Teacher samples[{index}]")
        category = _required_string(sample, "category", f"Teacher samples[{index}]")
        correct = sample.get("correct")
        if not isinstance(correct, bool):
            raise ManifestError(f"Teacher samples[{index}].correct 必须是布尔值")
        if sample_id in observed_ids:
            raise ManifestError(f"Teacher 逐样本结果 sample_id 重复: {sample_id}")
        observed_ids.add(sample_id)
        observed_samples.append((sample_id, category))
        observed_correct[category] = observed_correct.get(category, 0) + int(correct)
    expected_samples = [
        (str(item["sample_id"]), str(item["category"]))
        for item in selected_manifest["samples"]
    ]
    if sorted(observed_samples) != sorted(expected_samples):
        raise ManifestError("Teacher 逐样本结果与 selected_sample_manifest 不一致")

    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ManifestError("Teacher 评测缺少 summary")
    expected_count = int(selected_manifest["sample_count"])
    if summary.get("completed_samples") != expected_count:
        raise ManifestError("Teacher completed_samples 与选中样本数不一致")
    categories = summary.get("categories")
    if not isinstance(categories, Mapping):
        raise ManifestError("Teacher 评测缺少分类汇总")
    expected_counts = dict(selected_manifest["category_sample_counts"])
    if set(categories) != set(expected_counts):
        raise ManifestError("Teacher 分类汇总与选中类别不一致")
    scores: Dict[str, float] = {}
    for category, expected_category_count in expected_counts.items():
        metrics = categories[category]
        if not isinstance(metrics, Mapping):
            raise ManifestError(f"Teacher categories.{category} 必须是 JSON object")
        count = metrics.get("sample_count", metrics.get("total"))
        if count != expected_category_count:
            raise ManifestError(f"Teacher {category} 样本数与选中清单不一致")
        correct = metrics.get("correct")
        score = metrics.get("score")
        if (
            isinstance(correct, bool)
            or not isinstance(correct, int)
            or correct < 0
            or correct > expected_category_count
            or correct != observed_correct.get(category, 0)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0.0 <= float(score) <= 1.0
            or abs(float(score) - correct / expected_category_count) > 1e-6
        ):
            raise ManifestError(f"Teacher {category} score/correct/sample_count 不一致")
        scores[category] = float(score)
    return scores, {
        "path": str(path),
        "sha256": sha256_file(path),
        "label": label,
        "dataset_sha256": dataset_sha256,
        "selected_sample_manifest_sha256": selected_manifest["sha256"],
        "evaluation_protocol_sha256": _canonical_sha256(protocol),
        "sample_count": expected_count,
        "category_sample_counts": expected_counts,
    }


def _retention(
    summary: Mapping[str, Any], teacher_scores: Mapping[str, float]
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for category, metrics in summary["categories"].items():
        teacher_score = teacher_scores.get(category)
        output[category] = {
            "student_score": metrics["score"],
            "teacher_score": teacher_score,
            "retention": (
                round(float(metrics["score"]) / teacher_score, 6)
                if teacher_score is not None and teacher_score > 0
                else None
            ),
        }
    return output


def _selected_categories(values: Iterable[str]) -> set[str]:
    categories = {value.strip() for value in values if value.strip()}
    unknown = categories - set(GENERAL_CATEGORIES)
    if unknown:
        raise ManifestError(f"未知通用类别: {sorted(unknown)}")
    return categories or set(GENERAL_CATEGORIES)


def _apply_assistant_prefill(prompt: str, mode: str) -> str:
    if mode == "none":
        return prompt
    if mode == "empty_think":
        return prompt + EMPTY_THINK_ASSISTANT_PREFILL
    raise ManifestError("未知 assistant prefill 模式: {}".format(mode))


def _validate_adapter_training_evidence(
    adapter: Path, dataset_manifest_path: Optional[Path]
) -> Dict[str, Any]:
    metrics_path = adapter / "train_metrics.json"
    weights_path = adapter / "adapter_model.safetensors"
    config_path = adapter / "adapter_config.json"
    for path, label in (
        (metrics_path, "train_metrics.json"),
        (weights_path, "adapter_model.safetensors"),
        (config_path, "adapter_config.json"),
    ):
        if not path.is_file():
            raise ManifestError(f"通用 LoRA 缺少 {label}")
    metrics = read_json_object(metrics_path)
    expected_isolation = {
        "evaluation_prompt_overlap": 0,
        "scene_specific_samples": 0,
        "evaluation_set_used_for_training": False,
    }
    for field, expected in expected_isolation.items():
        if metrics.get(field) != expected:
            raise ManifestError(f"通用 LoRA 训练隔离字段无效: {field}")
    declared_manifest_sha = metrics.get("dataset_manifest_sha256")
    if (
        not isinstance(declared_manifest_sha, str)
        or SHA256_PATTERN.fullmatch(declared_manifest_sha) is None
    ):
        raise ManifestError("通用 LoRA train_metrics 缺少有效 dataset_manifest_sha256")
    if dataset_manifest_path is None:
        raise ManifestError("评估通用 LoRA 必须提供 --adapter_dataset_manifest")
    if not dataset_manifest_path.is_file():
        raise ManifestError("通用 LoRA 训练数据 manifest 不存在")
    actual_manifest_sha = sha256_file(dataset_manifest_path)
    if actual_manifest_sha != declared_manifest_sha:
        raise ManifestError("通用 LoRA 训练数据 manifest 哈希与 train_metrics 不一致")
    manifest = read_json_object(dataset_manifest_path)
    for field, expected in expected_isolation.items():
        if manifest.get(field) != expected:
            raise ManifestError(f"通用 LoRA 数据 manifest 隔离字段无效: {field}")
    return {
        "path": str(adapter),
        "sha256": sha256_file(weights_path),
        "adapter_config_sha256": sha256_file(config_path),
        "train_metrics_sha256": sha256_file(metrics_path),
        "dataset_manifest": {
            "path": str(dataset_manifest_path),
            "sha256": actual_manifest_sha,
        },
        "data_isolation": expected_isolation,
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="直接评估 BF16/F16 通用蒸馏 LoRA。")
    parser.add_argument("--base", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--snapshot_manifest", required=True)
    parser.add_argument("--adapter")
    parser.add_argument(
        "--adapter_dataset_manifest",
        help="训练该 LoRA 时使用的 manifest；其 SHA 必须匹配 train_metrics.json",
    )
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher_evaluation")
    parser.add_argument("--teacher_label", default="teacher")
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--limit_per_category", type=int, default=0)
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument("--code_timeout", type=float, default=3.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--assistant_prefill",
        choices=("none", "empty_think"),
        default="none",
        help="empty_think 精确复现 Ollama qwen3.5 think=false 的 4-token 前缀",
    )
    args = parser.parse_args(argv)

    if args.max_input_tokens <= 0:
        raise ManifestError("max_input_tokens 必须大于 0")
    if args.code_timeout <= 0:
        raise ManifestError("code_timeout 必须大于 0")
    if args.adapter_dataset_manifest and not args.adapter:
        raise ManifestError("--adapter_dataset_manifest 只能与 --adapter 一起使用")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_path = Path(args.base).resolve()
    base = validate_base_manifest(read_json_object(base_path))
    snapshot = Path(args.snapshot).resolve()
    snapshot_validation = verify_text_snapshot(
        base,
        read_json_object(Path(args.snapshot_manifest)),
        snapshot,
        verify_tokenizer=True,
    )
    dataset = Path(args.dataset_jsonl).resolve()
    if not dataset.is_file():
        raise ManifestError("冻结通用题集不存在")
    dataset_sha256 = sha256_file(dataset)
    categories = _selected_categories(args.category)
    rows = _select_rows(_read_jsonl(dataset), categories, args.limit_per_category)
    selected_manifest = _selected_sample_manifest(rows, dataset_sha256)

    adapter = Path(args.adapter).resolve() if args.adapter else None
    adapter_manifest = (
        Path(args.adapter_dataset_manifest).resolve()
        if args.adapter_dataset_manifest
        else None
    )
    adapter_evidence = (
        _validate_adapter_training_evidence(adapter, adapter_manifest)
        if adapter is not None
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, use_fast=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    precision = (
        "bfloat16" if args.bf16 else "float16"
    ) if torch.cuda.is_available() else "float32"
    protocol = _evaluation_protocol(
        tokenizer,
        assistant_prefill=args.assistant_prefill,
        max_input_tokens=args.max_input_tokens,
        precision=precision,
    )
    # Teacher 必须是同一评测器在同一文件、同一实际选中样本和同一协议上的结果。
    teacher_path = Path(args.teacher_evaluation).resolve() if args.teacher_evaluation else None
    teacher_scores, teacher_evidence = _teacher_scores(
        teacher_path,
        args.teacher_label,
        dataset_sha256=dataset_sha256,
        selected_manifest=selected_manifest,
        protocol=protocol,
    )
    model_kwargs: Dict[str, Any] = {"local_files_only": True, "trust_remote_code": False}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["dtype"] = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), **model_kwargs)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()

    samples: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        messages, max_new_tokens = _task_messages(row)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt = _apply_assistant_prefill(prompt, args.assistant_prefill)
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
            return_tensors="pt",
        )
        prompt_tokens = int(encoded["input_ids"].shape[1])
        if prompt_tokens > args.max_input_tokens:
            raise ManifestError(
                "样本 {} 输入为 {} tokens，超过锁定上限 {}；拒绝截断题目".format(
                    row["sample_id"], prompt_tokens, args.max_input_tokens
                )
            )
        encoded = {name: value.to(model.device) for name, value in encoded.items()}
        if index == 0:
            with torch.inference_mode():
                model.generate(
                    **encoded,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generation_ms = (time.perf_counter() - started) * 1000.0
        output_ids = generated[0, encoded["input_ids"].shape[1] :]
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        evaluated = evaluate_response(text, row, args.code_timeout)
        samples.append(
            {
                "sample_id": row.get("sample_id"),
                "category": row["category"],
                "prompt_tokens": prompt_tokens,
                "output_tokens": int(output_ids.shape[0]),
                "generation_ms": round(generation_ms, 4),
                "raw_output": text,
                **evaluated,
            }
        )

    summary = _summary(samples)
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "task": "frozen_general_capability_evaluation",
        "base_id": base["base_id"],
        "snapshot_validation": snapshot_validation,
        "adapter": adapter_evidence,
        "dataset": {"path": str(dataset), "sha256": dataset_sha256},
        "selected_sample_manifest": selected_manifest,
        "evaluation_protocol": protocol,
        "teacher_evaluation": teacher_evidence,
        "categories": sorted(categories),
        "category_sample_counts": _category_sample_counts(rows),
        "limit_per_category": args.limit_per_category,
        "no_thinking_text_template": True,
        "assistant_prefill": args.assistant_prefill,
        "sample_count": len(samples),
        "summary": summary,
        "retention": _retention(summary, teacher_scores),
        "samples": samples,
    }
    write_json_object(Path(args.output).resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
