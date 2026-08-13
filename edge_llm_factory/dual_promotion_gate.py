"""边缘大模型的通用专项、竞赛声明与交通非回归门禁。

本模块只消费已经完成的离线评测证据，不训练模型，也不修改发布状态。
v2 合同要求关键证据必须落到真实文件：门禁会读取文件、计算 SHA-256，
并把评测协议、冻结样本、分项得分和候选模型哈希交叉核对。仅在 JSON 中
填写看似合法的哈希不能通过。

报告明确区分两个结论：

* ``focused_profile_allowed``：一到两个预注册专项可作为独立 profile 使用；
* ``competition_claim_allowed``：官方三类整体要求、专项目标、运行时资源和
  交通非回归均通过后，才允许作竞赛整体声明。

旧 v1 自报式证据 fail-closed，不做静默兼容。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableSequence, Sequence, Tuple


SCHEMA_VERSION = "edge-llm-dual-promotion-evidence/v2"
REPORT_SCHEMA_VERSION = "edge-llm-dual-promotion-gate/v2"

OFFICIAL_GENERAL_CATEGORIES = (
    "code",
    "math",
    "natural_language_reasoning",
)

REQUIRED_ARTIFACT_FILES = (
    "general_evaluation",
    "general_dataset",
    "candidate_model",
    "traffic_evaluation",
    "runtime_evidence",
)

ARTIFACT_PROVENANCE_FIELDS = {
    "general_evaluation": "general_evaluation_sha256",
    "general_dataset": "general_dataset_sha256",
    "candidate_model": "candidate_model_sha256",
    "traffic_evaluation": "traffic_candidate_evaluation_sha256",
    "runtime_evidence": "runtime_candidate_evidence_sha256",
}

# 这些底线不可由候选证据调低；evidence.thresholds 只能收紧。
FLOORS = {
    "focused_retention": 0.80,
    "focused_sample_count_minimum": 30.0,
    "official_category_retention": 0.80,
    "official_macro_retention": 0.80,
    "traffic_accuracy_max_drop": 0.005,
    "traffic_weighted_f1_max_drop": 0.005,
    "traffic_selective_accuracy_max_drop": 0.005,
    "traffic_selective_gain_minimum": 0.10,
    "traffic_gguf_exact_match_max_drop": 0.0025,
    "traffic_accuracy_ci_lower_minimum": -0.01,
    "traffic_per_class_recall_max_drop": 0.02,
    "traffic_critical_recall_max_drop": 0.01,
    "traffic_ttft_ratio_maximum": 1.05,
    "ttft_reduction_vs_teacher_minimum": 0.75,
    "peak_rss_hard_limit_mb": 1536.0,
    "peak_rss_growth_maximum_mb": 64.0,
    "artifact_size_ratio_maximum": 1.05,
    "steady_rss_growth_500_maximum_mb": 16.0,
}

TRAFFIC_METRICS = (
    "decision_accuracy",
    "weighted_f1",
    "valid_output_rate",
    "selective_cascade_accuracy",
    "selective_gain_over_student",
    "gguf_exact_match_rate",
)

GENERAL_PROTOCOL_FIELDS = (
    "dataset_sha256",
    "backend_family",
    "renderer",
    "think",
    "prompt_contract",
    "num_ctx",
    "max_output_tokens",
)

INTEGRITY_BOOLEAN_FIELDS = (
    "dataset_sha256_locked_before_training",
    "grouped_split_by_source_id",
    "final_test_used_for_model_selection",
)

INTEGRITY_ZERO_FIELDS = (
    "train_validation_overlap_count",
    "train_test_overlap_count",
    "validation_test_overlap_count",
    "near_duplicate_train_test_overlap_count",
)

PROVENANCE_SHA_FIELDS = (
    "teacher_model_sha256",
    "incumbent_model_sha256",
    "candidate_model_sha256",
    "general_dataset_sha256",
    "traffic_test_dataset_sha256",
    "runtime_binary_sha256",
    "general_evaluation_sha256",
    "traffic_candidate_evaluation_sha256",
    "runtime_candidate_evidence_sha256",
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DualGateError(ValueError):
    """输入证据不完整、不可读取或合同不合法。"""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DualGateError("{} 必须是 JSON object".format(path))
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise DualGateError("{} 必须是 JSON array".format(path))
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DualGateError("{} 必须是有限数字".format(path))
    result = float(value)
    if not math.isfinite(result):
        raise DualGateError("{} 必须是有限数字".format(path))
    return result


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DualGateError("{} 必须是非负整数".format(path))
    return value


def _positive_int(value: Any, path: str) -> int:
    result = _nonnegative_int(value, path)
    if result <= 0:
        raise DualGateError("{} 必须大于 0".format(path))
    return result


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DualGateError("{} 必须是非空字符串".format(path))
    return value


def _sha_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise DualGateError("{} 必须是 64 位小写 SHA-256".format(path))
    return value


def _score(value: Any, path: str) -> float:
    result = _number(value, path)
    if result < 0.0 or result > 1.0:
        raise DualGateError("{} 必须在 [0, 1]".format(path))
    return result


def _add_check(
    checks: MutableSequence[Dict[str, Any]],
    reasons: MutableSequence[str],
    name: str,
    passed: bool,
    actual: Any,
    requirement: str,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "actual": actual,
            "requirement": requirement,
        }
    )
    if not passed:
        reasons.append("{}：实际 {}，要求 {}".format(name, actual, requirement))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DualGateError("无法读取证据文件 {}: {}".format(path, exc)) from exc
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _selection_sha256(sample_ids: Sequence[str]) -> str:
    canonical = _canonical_json(sorted(sample_ids)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _thresholds(evidence: Mapping[str, Any]) -> Dict[str, float]:
    supplied = _mapping(evidence.get("thresholds", {}), "thresholds")
    result = dict(FLOORS)
    unknown = set(supplied) - set(FLOORS)
    if unknown:
        raise DualGateError("thresholds 含未知字段: {}".format(sorted(unknown)))
    for key, raw in supplied.items():
        value = _number(raw, "thresholds.{}".format(key))
        floor = FLOORS[key]
        if key.endswith("_minimum") or key in {
            "focused_retention",
            "official_category_retention",
            "official_macro_retention",
        }:
            if value < floor:
                raise DualGateError("{} 不能低于底线 {}".format(key, floor))
        elif value > floor:
            raise DualGateError("{} 不能高于底线 {}".format(key, floor))
        result[key] = value
    return result


def _validate_provenance(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    provenance = _mapping(evidence.get("provenance"), "provenance")
    report: Dict[str, Any] = {}
    for field in PROVENANCE_SHA_FIELDS:
        report[field] = _sha_string(
            provenance.get(field), "provenance.{}".format(field)
        )
    if report["candidate_model_sha256"] == report["incumbent_model_sha256"]:
        raise DualGateError("candidate_model_sha256 不能与 incumbent_model_sha256 相同")
    report["hardware_id"] = _nonempty_string(
        provenance.get("hardware_id"), "provenance.hardware_id"
    )
    return report


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DualGateError("{} 不是可读取的 UTF-8 JSON: {}".format(label, exc)) from exc


def _validate_artifact_files(
    evidence: Mapping[str, Any],
    provenance: Mapping[str, Any],
    artifact_base_dir: Path | None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Path]]:
    records = _mapping(evidence.get("artifact_files"), "artifact_files")
    missing = [name for name in REQUIRED_ARTIFACT_FILES if name not in records]
    if missing:
        raise DualGateError("artifact_files 缺少必需证据: {}".format(sorted(missing)))

    report: Dict[str, Any] = {}
    loaded: Dict[str, Any] = {}
    resolved_paths: Dict[str, Path] = {}
    required_realpaths = set()
    for name, raw_record in records.items():
        if not isinstance(name, str) or not name.strip():
            raise DualGateError("artifact_files 的证据名必须是非空字符串")
        record = _mapping(raw_record, "artifact_files.{}".format(name))
        raw_path = _nonempty_string(
            record.get("path"), "artifact_files.{}.path".format(name)
        )
        expected_sha = _sha_string(
            record.get("sha256"), "artifact_files.{}.sha256".format(name)
        )
        path = Path(raw_path).expanduser()
        if not path.is_absolute() and artifact_base_dir is not None:
            path = artifact_base_dir / path
        path = path.resolve()
        if not path.is_file():
            raise DualGateError("artifact_files.{} 指向的文件不存在: {}".format(name, path))
        if name in REQUIRED_ARTIFACT_FILES:
            if path in required_realpaths:
                raise DualGateError("必需证据文件不能复用同一路径: {}".format(path))
            required_realpaths.add(path)
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise DualGateError(
                "artifact_files.{} SHA-256 不匹配：声明 {}，实测 {}".format(
                    name, expected_sha, actual_sha
                )
            )
        provenance_field = ARTIFACT_PROVENANCE_FIELDS.get(name)
        if provenance_field is not None and expected_sha != provenance[provenance_field]:
            raise DualGateError(
                "artifact_files.{} 与 provenance.{} 不一致".format(
                    name, provenance_field
                )
            )
        resolved_paths[name] = path
        report[name] = {
            "path": str(path),
            "sha256": actual_sha,
            "size_bytes": path.stat().st_size,
            "verified": True,
        }
        if name in {"general_evaluation", "traffic_evaluation", "runtime_evidence"}:
            loaded[name] = _read_json(path, "artifact_files.{}".format(name))
    return {"passed": True, "files": report}, loaded, resolved_paths


def _read_general_dataset(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DualGateError("general_dataset 无法作为 UTF-8 读取: {}".format(exc)) from exc
    stripped = text.lstrip()
    rows: Any
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            rows = parsed
        elif isinstance(parsed, Mapping):
            rows = parsed.get("samples", parsed.get("rows"))
        else:
            rows = None
    else:
        rows = None
    if rows is None:
        rows = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DualGateError(
                    "general_dataset 第 {} 行不是 JSON: {}".format(line_number, exc)
                ) from exc
    if not isinstance(rows, list) or not rows:
        raise DualGateError("general_dataset 必须包含非空 JSON rows/samples")

    sample_ids = []
    categories = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, "general_dataset[{}]".format(index))
        sample_id = _nonempty_string(
            row.get("sample_id"), "general_dataset[{}].sample_id".format(index)
        )
        category = _nonempty_string(
            row.get("category"), "general_dataset[{}].category".format(index)
        )
        if category not in OFFICIAL_GENERAL_CATEGORIES:
            raise DualGateError(
                "general_dataset[{}].category 不是官方类别: {}".format(
                    index, category
                )
            )
        sample_ids.append(sample_id)
        categories.append(category)
    if len(set(sample_ids)) != len(sample_ids):
        raise DualGateError("general_dataset.sample_id 必须唯一")
    counts = Counter(categories)
    missing = set(OFFICIAL_GENERAL_CATEGORIES) - set(counts)
    if missing:
        raise DualGateError("general_dataset 缺少官方类别: {}".format(sorted(missing)))
    return {
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "selection_sha256": _selection_sha256(sample_ids),
        "category_counts": dict(counts),
    }


def _validate_integrity(
    payload: Mapping[str, Any], label: str, reasons: MutableSequence[str]
) -> Dict[str, Any]:
    checks: list[Dict[str, Any]] = []
    for field in INTEGRITY_BOOLEAN_FIELDS:
        if field not in payload or not isinstance(payload[field], bool):
            raise DualGateError("{}.{} 必须是布尔值".format(label, field))
        expected = False if field == "final_test_used_for_model_selection" else True
        _add_check(
            checks,
            reasons,
            "{}.{}".format(label, field),
            payload[field] is expected,
            payload[field],
            "== {}".format(str(expected).lower()),
        )
    for field in INTEGRITY_ZERO_FIELDS:
        actual = _nonnegative_int(payload.get(field), "{}.{}".format(label, field))
        _add_check(checks, reasons, "{}.{}".format(label, field), actual == 0, actual, "== 0")
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def _category_scores(model: Mapping[str, Any], path: str) -> Dict[str, Dict[str, Any]]:
    categories = _mapping(model.get("categories"), "{}.categories".format(path))
    if set(categories) != set(OFFICIAL_GENERAL_CATEGORIES):
        raise DualGateError(
            "{}.categories 必须恰好是官方三类 {}".format(
                path, list(OFFICIAL_GENERAL_CATEGORIES)
            )
        )
    result: Dict[str, Dict[str, Any]] = {}
    for name in OFFICIAL_GENERAL_CATEGORIES:
        item = _mapping(categories[name], "{}.categories.{}".format(path, name))
        score = _score(item.get("score"), "{}.categories.{}.score".format(path, name))
        sample_count = _positive_int(
            item.get("sample_count"),
            "{}.categories.{}.sample_count".format(path, name),
        )
        correct = _nonnegative_int(
            item.get("correct"), "{}.categories.{}.correct".format(path, name)
        )
        if correct > sample_count:
            raise DualGateError("{}.categories.{}.correct 超过样本数".format(path, name))
        if abs(score - correct / sample_count) > 1e-6:
            raise DualGateError(
                "{}.categories.{} 的 score 与 correct/sample_count 不一致".format(
                    path, name
                )
            )
        result[name] = {
            "score": score,
            "sample_count": sample_count,
            "correct": correct,
        }
    return result


def _general_protocol_side(payload: Mapping[str, Any], path: str) -> Dict[str, Any]:
    missing = [field for field in GENERAL_PROTOCOL_FIELDS if field not in payload]
    if missing:
        raise DualGateError("{} 缺少字段 {}".format(path, sorted(missing)))
    think = payload["think"]
    if not isinstance(think, bool):
        raise DualGateError("{}.think 必须是布尔值".format(path))
    return {
        "dataset_sha256": _sha_string(
            payload["dataset_sha256"], "{}.dataset_sha256".format(path)
        ),
        "backend_family": _nonempty_string(
            payload["backend_family"], "{}.backend_family".format(path)
        ),
        "renderer": _nonempty_string(payload["renderer"], "{}.renderer".format(path)),
        "think": think,
        "prompt_contract": _nonempty_string(
            payload["prompt_contract"], "{}.prompt_contract".format(path)
        ),
        "num_ctx": _positive_int(payload["num_ctx"], "{}.num_ctx".format(path)),
        "max_output_tokens": _positive_int(
            payload["max_output_tokens"], "{}.max_output_tokens".format(path)
        ),
    }


def _validate_general_protocol(
    general: Mapping[str, Any],
    provenance_dataset_sha256: str,
    reasons: MutableSequence[str],
) -> Dict[str, Any]:
    protocol = _mapping(general.get("protocol"), "general.protocol")
    teacher = _general_protocol_side(
        _mapping(protocol.get("teacher"), "general.protocol.teacher"),
        "general.protocol.teacher",
    )
    candidate = _general_protocol_side(
        _mapping(protocol.get("candidate"), "general.protocol.candidate"),
        "general.protocol.candidate",
    )
    checks: list[Dict[str, Any]] = []
    for field in GENERAL_PROTOCOL_FIELDS:
        _add_check(
            checks,
            reasons,
            "general.protocol.{}".format(field),
            candidate[field] == teacher[field],
            candidate[field],
            "== {!r}".format(teacher[field]),
        )
    for side, payload in (("teacher", teacher), ("candidate", candidate)):
        _add_check(
            checks,
            reasons,
            "general.protocol.{}.dataset_matches_provenance".format(side),
            payload["dataset_sha256"] == provenance_dataset_sha256,
            payload["dataset_sha256"],
            "== provenance.general_dataset_sha256 {!r}".format(
                provenance_dataset_sha256
            ),
        )
    return {
        "passed": all(item["passed"] for item in checks),
        "teacher": teacher,
        "candidate": candidate,
        "checks": checks,
    }


def _category_tables_equal(
    left: Mapping[str, Mapping[str, Any]], right: Mapping[str, Mapping[str, Any]]
) -> bool:
    if set(left) != set(right):
        return False
    for name in left:
        if left[name]["sample_count"] != right[name]["sample_count"]:
            return False
        if left[name]["correct"] != right[name]["correct"]:
            return False
        if abs(left[name]["score"] - right[name]["score"]) > 1e-9:
            return False
    return True


def _validate_general_evaluation_artifact(
    artifact: Any,
    general: Mapping[str, Any],
    protocol: Mapping[str, Any],
    teacher: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    dataset: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = _mapping(artifact, "general_evaluation")
    teacher_artifact = _mapping(payload.get("teacher"), "general_evaluation.teacher")
    candidate_artifact = _mapping(
        payload.get("candidate"), "general_evaluation.candidate"
    )
    teacher_sha = _sha_string(
        teacher_artifact.get("model_sha256"),
        "general_evaluation.teacher.model_sha256",
    )
    candidate_sha = _sha_string(
        candidate_artifact.get("model_sha256"),
        "general_evaluation.candidate.model_sha256",
    )
    if teacher_sha != provenance["teacher_model_sha256"]:
        raise DualGateError("general_evaluation 的 teacher_model_sha256 与 provenance 不一致")
    if candidate_sha != provenance["candidate_model_sha256"]:
        raise DualGateError("general_evaluation 的 candidate_model_sha256 与 provenance 不一致")

    artifact_protocol_raw = _mapping(
        payload.get("protocol"), "general_evaluation.protocol"
    )
    artifact_protocol = {
        side: _general_protocol_side(
            _mapping(artifact_protocol_raw.get(side), "general_evaluation.protocol.{}".format(side)),
            "general_evaluation.protocol.{}".format(side),
        )
        for side in ("teacher", "candidate")
    }
    for side in ("teacher", "candidate"):
        if artifact_protocol[side] != protocol[side]:
            raise DualGateError(
                "general_evaluation.protocol.{} 与 gate general.protocol 不一致".format(
                    side
                )
            )

    selection = _mapping(payload.get("selection"), "general_evaluation.selection")
    sample_count = _positive_int(
        selection.get("sample_count"), "general_evaluation.selection.sample_count"
    )
    if sample_count != dataset["sample_count"]:
        raise DualGateError("general_evaluation 样本数与 general_dataset 不一致")
    has_ids = "sample_ids" in selection
    has_selection_sha = "selection_sha256" in selection
    if not has_ids and not has_selection_sha:
        raise DualGateError("general_evaluation.selection 必须提供 sample_ids 或 selection_sha256")
    if has_ids:
        raw_ids = _sequence(selection["sample_ids"], "general_evaluation.selection.sample_ids")
        ids = [
            _nonempty_string(value, "general_evaluation.selection.sample_ids[{}]".format(index))
            for index, value in enumerate(raw_ids)
        ]
        if len(ids) != sample_count or len(set(ids)) != len(ids):
            raise DualGateError("general_evaluation.selection.sample_ids 数量错误或重复")
        if set(ids) != set(dataset["sample_ids"]):
            raise DualGateError("general_evaluation.sample_ids 与 general_dataset 不一致")
    if has_selection_sha:
        selection_sha = _sha_string(
            selection["selection_sha256"],
            "general_evaluation.selection.selection_sha256",
        )
        if selection_sha != dataset["selection_sha256"]:
            raise DualGateError("general_evaluation.selection_sha256 与 general_dataset 不一致")

    artifact_teacher = _category_scores(
        teacher_artifact, "general_evaluation.teacher"
    )
    artifact_candidate = _category_scores(
        candidate_artifact, "general_evaluation.candidate"
    )
    if not _category_tables_equal(artifact_teacher, teacher):
        raise DualGateError("general_evaluation.teacher 分项分数与 gate 不一致")
    if not _category_tables_equal(artifact_candidate, candidate):
        raise DualGateError("general_evaluation.candidate 分项分数与 gate 不一致")
    for name in OFFICIAL_GENERAL_CATEGORIES:
        expected_count = dataset["category_counts"][name]
        if artifact_teacher[name]["sample_count"] != expected_count:
            raise DualGateError(
                "general_evaluation.{} 样本数与 general_dataset 类别计数不一致".format(
                    name
                )
            )
    return {
        "passed": True,
        "sample_count": sample_count,
        "selection_sha256": dataset["selection_sha256"],
        "candidate_model_sha256": candidate_sha,
        "protocol_verified": True,
        "category_scores_verified": True,
    }


def _evaluate_general(
    evidence: Mapping[str, Any],
    thresholds: Mapping[str, float],
    reasons: MutableSequence[str],
    general_evaluation_artifact: Any,
    general_dataset: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    preregistration = _mapping(evidence.get("pre_registration"), "pre_registration")
    focus_raw = _sequence(
        preregistration.get("focus_categories"), "pre_registration.focus_categories"
    )
    focus = []
    for category in focus_raw:
        if category not in OFFICIAL_GENERAL_CATEGORIES or category in focus:
            raise DualGateError(
                "focus_categories 只能包含一到两个不重复的官方类别 {}".format(
                    list(OFFICIAL_GENERAL_CATEGORIES)
                )
            )
        focus.append(category)
    if not 1 <= len(focus) <= 2:
        raise DualGateError("focus_categories 必须预先固定一到两个官方类别")

    general = _mapping(evidence.get("general"), "general")
    protocol = _validate_general_protocol(
        general, provenance["general_dataset_sha256"], reasons
    )
    teacher = _category_scores(
        _mapping(general.get("teacher"), "general.teacher"), "general.teacher"
    )
    candidate = _category_scores(
        _mapping(general.get("candidate"), "general.candidate"),
        "general.candidate",
    )
    artifact = _validate_general_evaluation_artifact(
        general_evaluation_artifact,
        general,
        protocol,
        teacher,
        candidate,
        general_dataset,
        provenance,
    )

    category_checks = []
    for category in OFFICIAL_GENERAL_CATEGORIES:
        teacher_score = teacher[category]["score"]
        candidate_score = candidate[category]["score"]
        teacher_count = teacher[category]["sample_count"]
        candidate_count = candidate[category]["sample_count"]
        if teacher_count != candidate_count:
            raise DualGateError(
                "general.teacher 与 general.candidate 的 {} 样本数不一致".format(
                    category
                )
            )
        if teacher_score <= 0.0:
            raise DualGateError("9B 在 {} 上得分为 0，保持率无定义".format(category))
        retention = candidate_score / teacher_score
        selected = category in focus
        official_category_passed = (
            retention + 1e-12 >= thresholds["official_category_retention"]
        )
        enough_samples = candidate_count >= int(
            thresholds["focused_sample_count_minimum"]
        )
        passed = (not selected) or (
            retention + 1e-12 >= thresholds["focused_retention"] and enough_samples
        )
        if selected and retention + 1e-12 < thresholds["focused_retention"]:
            reasons.append(
                "通用分项 {} 保持率 {:.6f}，低于 {:.6f}".format(
                    category, retention, thresholds["focused_retention"]
                )
            )
        if selected and not enough_samples:
            reasons.append(
                "通用分项 {} 只有 {} 个冻结样本，少于 {}".format(
                    category,
                    candidate_count,
                    int(thresholds["focused_sample_count_minimum"]),
                )
            )
        category_checks.append(
            {
                "category": category,
                "focused": selected,
                "teacher_score": round(teacher_score, 6),
                "candidate_score": round(candidate_score, 6),
                "retention": round(retention, 6),
                "official_minimum_retention": thresholds[
                    "official_category_retention"
                ],
                "official_passed": official_category_passed,
                "minimum_retention": (
                    thresholds["focused_retention"] if selected else None
                ),
                "minimum_sample_count": (
                    int(thresholds["focused_sample_count_minimum"])
                    if selected
                    else None
                ),
                "passed": passed,
                "sample_count": teacher_count,
            }
        )

    teacher_macro = sum(teacher[name]["score"] for name in OFFICIAL_GENERAL_CATEGORIES) / 3.0
    candidate_macro = sum(candidate[name]["score"] for name in OFFICIAL_GENERAL_CATEGORIES) / 3.0
    official_retention = candidate_macro / teacher_macro
    official_macro_passed = (
        official_retention + 1e-12 >= thresholds["official_macro_retention"]
    )
    if not official_macro_passed:
        reasons.append(
            "官方三类宏平均保持率 {:.6f}，低于 {:.6f}".format(
                official_retention, thresholds["official_macro_retention"]
            )
        )
    official_categories_passed = all(
        item["official_passed"] for item in category_checks
    )
    for item in category_checks:
        if not item["official_passed"]:
            reasons.append(
                "官方分项 {} 保持率 {:.6f}，低于 {:.6f}".format(
                    item["category"],
                    item["retention"],
                    thresholds["official_category_retention"],
                )
            )

    focused_score_passed = all(
        item["passed"] for item in category_checks if item["focused"]
    )
    integrity = _validate_integrity(
        _mapping(general.get("data_integrity"), "general.data_integrity"),
        "general.data_integrity",
        reasons,
    )
    focused_general_allowed = (
        focused_score_passed and protocol["passed"] and integrity["passed"]
    )
    competition_general_requirement_met = (
        official_macro_passed
        and official_categories_passed
        and protocol["passed"]
        and integrity["passed"]
    )
    return {
        "passed": competition_general_requirement_met,
        "focused_categories": focus,
        "categories": category_checks,
        "focused_gate_passed": focused_score_passed,
        "focused_profile_general_gate_passed": focused_general_allowed,
        "competition_general_requirement_met": competition_general_requirement_met,
        "official_category_gate_passed": official_categories_passed,
        "protocol": protocol,
        "verified_evaluation": artifact,
        "official_macro": {
            "official_categories": list(OFFICIAL_GENERAL_CATEGORIES),
            "teacher_score": round(teacher_macro, 6),
            "candidate_score": round(candidate_macro, 6),
            "retention": round(official_retention, 6),
            "minimum_retention": thresholds["official_macro_retention"],
            "passed": official_macro_passed,
            "scope_note": (
                "专项通过不等于竞赛整体通过；宏平均仅作诊断，"
                "整体门禁要求三个官方分项各自达到保持率底线"
            ),
        },
        "data_integrity": integrity,
    }


def _traffic_metrics(payload: Mapping[str, Any], path: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for metric in TRAFFIC_METRICS:
        result[metric] = _score(payload.get(metric), "{}.{}".format(path, metric))
    recalls = _mapping(
        payload.get("per_class_recall"), "{}.per_class_recall".format(path)
    )
    if not recalls:
        raise DualGateError("{}.per_class_recall 不能为空".format(path))
    result["per_class_recall"] = {
        str(token): _score(score, "{}.per_class_recall.{}".format(path, token))
        for token, score in recalls.items()
    }
    return result


def _validate_traffic_evaluation_artifact(
    artifact: Any,
    traffic: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = _mapping(artifact, "traffic_evaluation")
    dataset_sha = _sha_string(
        payload.get("dataset_sha256"), "traffic_evaluation.dataset_sha256"
    )
    if dataset_sha != provenance["traffic_test_dataset_sha256"]:
        raise DualGateError("traffic_evaluation.dataset_sha256 与 provenance 不一致")
    traffic_model_sha = _sha_string(
        payload.get("traffic_model_sha256"),
        "traffic_evaluation.traffic_model_sha256",
    )
    isolation = _mapping(
        traffic.get("artifact_isolation"), "traffic.artifact_isolation"
    )
    if traffic_model_sha != isolation.get("candidate_traffic_model_sha256"):
        raise DualGateError("traffic_evaluation 的交通模型 SHA 与隔离合同不一致")
    compared_fields = (
        "incumbent",
        "candidate",
        "paired_accuracy_delta_ci95",
        "critical_action_tokens",
        "action_mapping",
        "artifact_isolation",
        "protocol",
    )
    for field in compared_fields:
        if field not in payload or _canonical_json(payload[field]) != _canonical_json(
            traffic.get(field)
        ):
            raise DualGateError(
                "traffic_evaluation.{} 与 gate traffic.{} 不一致".format(
                    field, field
                )
            )
    return {
        "passed": True,
        "dataset_sha256": dataset_sha,
        "traffic_model_sha256": traffic_model_sha,
        "metrics_verified": True,
    }


def _evaluate_traffic(
    evidence: Mapping[str, Any],
    thresholds: Mapping[str, float],
    reasons: MutableSequence[str],
    traffic_evaluation_artifact: Any,
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    traffic = _mapping(evidence.get("traffic"), "traffic")
    artifact = _validate_traffic_evaluation_artifact(
        traffic_evaluation_artifact, traffic, provenance
    )
    incumbent = _traffic_metrics(
        _mapping(traffic.get("incumbent"), "traffic.incumbent"),
        "traffic.incumbent",
    )
    candidate = _traffic_metrics(
        _mapping(traffic.get("candidate"), "traffic.candidate"),
        "traffic.candidate",
    )
    checks: list[Dict[str, Any]] = []

    drop_contract = {
        "decision_accuracy": thresholds["traffic_accuracy_max_drop"],
        "weighted_f1": thresholds["traffic_weighted_f1_max_drop"],
        "selective_cascade_accuracy": thresholds[
            "traffic_selective_accuracy_max_drop"
        ],
        "gguf_exact_match_rate": thresholds["traffic_gguf_exact_match_max_drop"],
    }
    for metric, allowed_drop in drop_contract.items():
        required = incumbent[metric] - allowed_drop
        _add_check(
            checks,
            reasons,
            "traffic.{}".format(metric),
            candidate[metric] + 1e-12 >= required,
            round(candidate[metric], 6),
            ">= {:.6f}（当前模型 {:.6f} - 容差 {:.6f}）".format(
                required, incumbent[metric], allowed_drop
            ),
        )
    _add_check(
        checks,
        reasons,
        "traffic.valid_output_rate",
        candidate["valid_output_rate"] == 1.0,
        candidate["valid_output_rate"],
        "== 1.0",
    )
    _add_check(
        checks,
        reasons,
        "traffic.selective_gain_over_student",
        candidate["selective_gain_over_student"] + 1e-12
        >= thresholds["traffic_selective_gain_minimum"],
        candidate["selective_gain_over_student"],
        ">= {:.6f}".format(thresholds["traffic_selective_gain_minimum"]),
    )

    if set(incumbent["per_class_recall"]) != set(candidate["per_class_recall"]):
        raise DualGateError("交通候选与当前模型的动作类别不一致")
    critical_raw = _sequence(
        traffic.get("critical_action_tokens", []), "traffic.critical_action_tokens"
    )
    critical = set()
    for token in critical_raw:
        if not isinstance(token, str) or token not in incumbent["per_class_recall"]:
            raise DualGateError("critical_action_tokens 含未知动作: {}".format(token))
        critical.add(token)
    recall_checks = []
    for token in sorted(incumbent["per_class_recall"]):
        allowed_drop = (
            thresholds["traffic_critical_recall_max_drop"]
            if token in critical
            else thresholds["traffic_per_class_recall_max_drop"]
        )
        required = incumbent["per_class_recall"][token] - allowed_drop
        _add_check(
            checks,
            reasons,
            "traffic.recall.{}".format(token),
            candidate["per_class_recall"][token] + 1e-12 >= required,
            candidate["per_class_recall"][token],
            ">= {:.6f}".format(required),
        )
        recall_checks.append(checks[-1])

    ci = _sequence(
        traffic.get("paired_accuracy_delta_ci95"),
        "traffic.paired_accuracy_delta_ci95",
    )
    if len(ci) != 2:
        raise DualGateError("traffic.paired_accuracy_delta_ci95 必须含 lower/upper 两项")
    ci_lower = _number(ci[0], "traffic.paired_accuracy_delta_ci95[0]")
    ci_upper = _number(ci[1], "traffic.paired_accuracy_delta_ci95[1]")
    if ci_lower > ci_upper:
        raise DualGateError("traffic.paired_accuracy_delta_ci95 下界不能大于上界")
    _add_check(
        checks,
        reasons,
        "traffic.paired_accuracy_noninferiority",
        ci_lower + 1e-12 >= thresholds["traffic_accuracy_ci_lower_minimum"],
        [ci_lower, ci_upper],
        "95% CI lower >= {:.6f}".format(
            thresholds["traffic_accuracy_ci_lower_minimum"]
        ),
    )

    mapping = _mapping(traffic.get("action_mapping"), "traffic.action_mapping")
    incumbent_sha = _sha_string(
        mapping.get("incumbent_sha256"),
        "traffic.action_mapping.incumbent_sha256",
    )
    candidate_sha = _sha_string(
        mapping.get("candidate_sha256"),
        "traffic.action_mapping.candidate_sha256",
    )
    _add_check(
        checks,
        reasons,
        "traffic.action_mapping_sha256",
        incumbent_sha == candidate_sha,
        candidate_sha,
        "== {}".format(incumbent_sha),
    )

    isolation = _mapping(
        traffic.get("artifact_isolation"), "traffic.artifact_isolation"
    )
    incumbent_model_sha = _sha_string(
        isolation.get("incumbent_traffic_model_sha256"),
        "traffic.artifact_isolation.incumbent_traffic_model_sha256",
    )
    candidate_model_sha = _sha_string(
        isolation.get("candidate_traffic_model_sha256"),
        "traffic.artifact_isolation.candidate_traffic_model_sha256",
    )
    _add_check(
        checks,
        reasons,
        "traffic.traffic_model_artifact_unchanged",
        incumbent_model_sha == candidate_model_sha,
        candidate_model_sha,
        "== {}".format(incumbent_model_sha),
    )
    for field in ("general_adapter_separate_profile", "mixed_adapter_loading_disabled"):
        value = isolation.get(field)
        if not isinstance(value, bool):
            raise DualGateError("traffic.artifact_isolation.{} 必须是布尔值".format(field))
        _add_check(
            checks,
            reasons,
            "traffic.artifact_isolation.{}".format(field),
            value,
            value,
            "== true",
        )

    protocol = _mapping(traffic.get("protocol"), "traffic.protocol")
    incumbent_protocol = _mapping(
        protocol.get("incumbent"), "traffic.protocol.incumbent"
    )
    candidate_protocol = _mapping(
        protocol.get("candidate"), "traffic.protocol.candidate"
    )
    required_protocol_fields = (
        "context_encoder",
        "prompt_format",
        "max_input_tokens",
        "max_output_tokens",
        "thinking",
    )
    for field in required_protocol_fields:
        if field not in incumbent_protocol or field not in candidate_protocol:
            raise DualGateError("traffic.protocol 缺少字段 {}".format(field))
        _add_check(
            checks,
            reasons,
            "traffic.protocol.{}".format(field),
            candidate_protocol[field] == incumbent_protocol[field],
            candidate_protocol[field],
            "== {!r}".format(incumbent_protocol[field]),
        )

    integrity = _validate_integrity(
        _mapping(traffic.get("data_integrity"), "traffic.data_integrity"),
        "traffic.data_integrity",
        reasons,
    )
    return {
        "passed": all(item["passed"] for item in checks) and integrity["passed"],
        "checks": checks,
        "recall_checks": recall_checks,
        "verified_evaluation": artifact,
        "data_integrity": integrity,
    }


def _runtime_metrics(payload: Mapping[str, Any], path: str) -> Dict[str, float]:
    names = ("ttft_mean_ms", "ttft_p95_ms", "peak_rss_mb", "artifact_bytes")
    result = {
        name: _number(payload.get(name), "{}.{}".format(path, name))
        for name in names
    }
    if any(value < 0 for value in result.values()):
        raise DualGateError("{} 的运行时数值不能为负".format(path))
    return result


def _validate_runtime_evidence_artifact(
    artifact: Any,
    runtime: Mapping[str, Any],
    provenance: Mapping[str, Any],
    candidate_model_size: int,
) -> Dict[str, Any]:
    payload = _mapping(artifact, "runtime_evidence")
    candidate_sha = _sha_string(
        payload.get("candidate_model_sha256"),
        "runtime_evidence.candidate_model_sha256",
    )
    if candidate_sha != provenance["candidate_model_sha256"]:
        raise DualGateError("runtime_evidence 的候选模型 SHA 与 provenance 不一致")
    artifact_runtime = _mapping(payload.get("runtime"), "runtime_evidence.runtime")
    if _canonical_json(artifact_runtime) != _canonical_json(runtime):
        raise DualGateError("runtime_evidence.runtime 与 gate runtime 不一致")
    general_candidate = _mapping(
        _mapping(runtime.get("general_candidate"), "runtime.general_candidate").get(
            "candidate"
        ),
        "runtime.general_candidate.candidate",
    )
    artifact_bytes = _number(
        general_candidate.get("artifact_bytes"),
        "runtime.general_candidate.candidate.artifact_bytes",
    )
    if int(artifact_bytes) != candidate_model_size or artifact_bytes != int(artifact_bytes):
        raise DualGateError(
            "runtime 通用候选 artifact_bytes 与 candidate_model 实际文件大小不一致"
        )
    return {
        "passed": True,
        "candidate_model_sha256": candidate_sha,
        "candidate_model_size_bytes": candidate_model_size,
        "runtime_fields_verified": True,
    }


def _evaluate_runtime(
    evidence: Mapping[str, Any],
    thresholds: Mapping[str, float],
    reasons: MutableSequence[str],
    runtime_artifact: Any,
    provenance: Mapping[str, Any],
    candidate_model_size: int,
) -> Dict[str, Any]:
    runtime = _mapping(evidence.get("runtime"), "runtime")
    artifact = _validate_runtime_evidence_artifact(
        runtime_artifact, runtime, provenance, candidate_model_size
    )

    general_payload = _mapping(
        runtime.get("general_candidate"), "runtime.general_candidate"
    )
    teacher_ttft = _number(
        general_payload.get("teacher_ttft_mean_ms"),
        "runtime.general_candidate.teacher_ttft_mean_ms",
    )
    if teacher_ttft <= 0:
        raise DualGateError("runtime.general_candidate.teacher_ttft_mean_ms 必须大于 0")
    candidate_payload = _mapping(
        general_payload.get("candidate"), "runtime.general_candidate.candidate"
    )
    candidate = _runtime_metrics(
        candidate_payload, "runtime.general_candidate.candidate"
    )
    general_checks: list[Dict[str, Any]] = []
    ttft_reduction = 1.0 - candidate["ttft_mean_ms"] / teacher_ttft
    _add_check(
        general_checks,
        reasons,
        "runtime.general_candidate.ttft_reduction_vs_teacher",
        ttft_reduction + 1e-12
        >= thresholds["ttft_reduction_vs_teacher_minimum"],
        round(ttft_reduction, 6),
        ">= {:.6f}".format(thresholds["ttft_reduction_vs_teacher_minimum"]),
    )
    _add_check(
        general_checks,
        reasons,
        "runtime.general_candidate.peak_rss_mb",
        candidate["peak_rss_mb"] <= thresholds["peak_rss_hard_limit_mb"] + 1e-12,
        candidate["peak_rss_mb"],
        "<= {:.1f} MB".format(thresholds["peak_rss_hard_limit_mb"]),
    )
    rss_growth = _number(
        candidate_payload.get("steady_rss_growth_500_mb"),
        "runtime.general_candidate.candidate.steady_rss_growth_500_mb",
    )
    vm_swap = _number(
        candidate_payload.get("vm_swap_mb"),
        "runtime.general_candidate.candidate.vm_swap_mb",
    )
    _add_check(
        general_checks,
        reasons,
        "runtime.general_candidate.steady_rss_growth_500_mb",
        rss_growth <= thresholds["steady_rss_growth_500_maximum_mb"] + 1e-12,
        rss_growth,
        "<= {:.6f}".format(thresholds["steady_rss_growth_500_maximum_mb"]),
    )
    _add_check(
        general_checks,
        reasons,
        "runtime.general_candidate.vm_swap_mb",
        vm_swap == 0.0,
        vm_swap,
        "== 0",
    )

    traffic_payload = _mapping(
        runtime.get("traffic_non_regression"), "runtime.traffic_non_regression"
    )
    incumbent = _runtime_metrics(
        _mapping(traffic_payload.get("incumbent"), "runtime.traffic_non_regression.incumbent"),
        "runtime.traffic_non_regression.incumbent",
    )
    traffic_candidate_payload = _mapping(
        traffic_payload.get("candidate"), "runtime.traffic_non_regression.candidate"
    )
    traffic_candidate = _runtime_metrics(
        traffic_candidate_payload, "runtime.traffic_non_regression.candidate"
    )
    traffic_checks: list[Dict[str, Any]] = []
    for metric in ("ttft_mean_ms", "ttft_p95_ms"):
        maximum = incumbent[metric] * thresholds["traffic_ttft_ratio_maximum"]
        _add_check(
            traffic_checks,
            reasons,
            "runtime.traffic_non_regression.{}".format(metric),
            traffic_candidate[metric] <= maximum + 1e-12,
            traffic_candidate[metric],
            "<= {:.6f}（交通旧值的 {:.2f} 倍）".format(
                maximum, thresholds["traffic_ttft_ratio_maximum"]
            ),
        )
    rss_maximum = min(
        thresholds["peak_rss_hard_limit_mb"],
        incumbent["peak_rss_mb"] + thresholds["peak_rss_growth_maximum_mb"],
    )
    _add_check(
        traffic_checks,
        reasons,
        "runtime.traffic_non_regression.peak_rss_mb",
        traffic_candidate["peak_rss_mb"] <= rss_maximum + 1e-12,
        traffic_candidate["peak_rss_mb"],
        "<= {:.6f}".format(rss_maximum),
    )
    artifact_maximum = (
        incumbent["artifact_bytes"] * thresholds["artifact_size_ratio_maximum"]
    )
    _add_check(
        traffic_checks,
        reasons,
        "runtime.traffic_non_regression.artifact_bytes",
        traffic_candidate["artifact_bytes"] <= artifact_maximum + 1e-12,
        int(traffic_candidate["artifact_bytes"]),
        "<= {:.0f}".format(artifact_maximum),
    )
    traffic_rss_growth = _number(
        traffic_candidate_payload.get("steady_rss_growth_500_mb"),
        "runtime.traffic_non_regression.candidate.steady_rss_growth_500_mb",
    )
    traffic_vm_swap = _number(
        traffic_candidate_payload.get("vm_swap_mb"),
        "runtime.traffic_non_regression.candidate.vm_swap_mb",
    )
    _add_check(
        traffic_checks,
        reasons,
        "runtime.traffic_non_regression.steady_rss_growth_500_mb",
        traffic_rss_growth
        <= thresholds["steady_rss_growth_500_maximum_mb"] + 1e-12,
        traffic_rss_growth,
        "<= {:.6f}".format(thresholds["steady_rss_growth_500_maximum_mb"]),
    )
    _add_check(
        traffic_checks,
        reasons,
        "runtime.traffic_non_regression.vm_swap_mb",
        traffic_vm_swap == 0.0,
        traffic_vm_swap,
        "== 0",
    )

    general_report = {
        "passed": all(item["passed"] for item in general_checks),
        "checks": general_checks,
    }
    traffic_report = {
        "passed": all(item["passed"] for item in traffic_checks),
        "checks": traffic_checks,
    }
    return {
        "passed": general_report["passed"] and traffic_report["passed"],
        "general_candidate": general_report,
        "traffic_non_regression": traffic_report,
        "verified_evidence": artifact,
    }


def evaluate_dual_gate(
    evidence: Mapping[str, Any], artifact_base_dir: Path | str | None = None
) -> Dict[str, Any]:
    """校验真实证据并返回可机读的专项/竞赛双门禁报告。"""

    if evidence.get("schema_version") != SCHEMA_VERSION:
        raise DualGateError(
            "schema_version 必须为 {}；旧格式不能用于发布门禁".format(
                SCHEMA_VERSION
            )
        )
    provenance = _validate_provenance(evidence)
    base_dir = Path(artifact_base_dir).resolve() if artifact_base_dir is not None else None
    artifact_report, loaded, artifact_paths = _validate_artifact_files(
        evidence, provenance, base_dir
    )
    general_dataset = _read_general_dataset(artifact_paths["general_dataset"])
    thresholds = _thresholds(evidence)
    reasons: list[str] = []
    general = _evaluate_general(
        evidence,
        thresholds,
        reasons,
        loaded["general_evaluation"],
        general_dataset,
        provenance,
    )
    traffic = _evaluate_traffic(
        evidence,
        thresholds,
        reasons,
        loaded["traffic_evaluation"],
        provenance,
    )
    runtime = _evaluate_runtime(
        evidence,
        thresholds,
        reasons,
        loaded["runtime_evidence"],
        provenance,
        artifact_paths["candidate_model"].stat().st_size,
    )

    common_profile_gates = (
        artifact_report["passed"]
        and traffic["passed"]
        and runtime["general_candidate"]["passed"]
        and runtime["traffic_non_regression"]["passed"]
    )
    focused_profile_allowed = (
        general["focused_profile_general_gate_passed"] and common_profile_gates
    )
    competition_general_requirement_met = general[
        "competition_general_requirement_met"
    ]
    competition_claim_allowed = (
        focused_profile_allowed and competition_general_requirement_met
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "task": "edge_llm_verified_general_and_traffic_non_regression_gate",
        "provenance": provenance,
        "artifact_files": artifact_report,
        "thresholds": thresholds,
        "general": general,
        "traffic": traffic,
        "runtime": runtime,
        "focused_profile_allowed": focused_profile_allowed,
        "competition_general_requirement_met": competition_general_requirement_met,
        "competition_claim_allowed": competition_claim_allowed,
        "passed": competition_claim_allowed,
        # 保留字段名给旧消费者，但采用最安全语义：不得把专项门禁当整体发布门禁。
        "promotion_allowed": competition_claim_allowed,
        "reasons": reasons,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="执行可核验的通用能力与交通非回归门禁")
    parser.add_argument("--evidence-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    evidence_path = Path(args.evidence_json).resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    report = evaluate_dual_gate(
        _mapping(evidence, "root"), artifact_base_dir=evidence_path.parent
    )
    report["input_artifact"] = {
        "path": str(evidence_path),
        "sha256": _sha256(evidence_path),
    }
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["competition_claim_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
