"""用途：从已验收通用蒸馏数据中生成预注册类别的隔离训练子集。"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from edge_llm_factory.contracts import (
    ManifestError,
    read_json_object,
    safe_relative_path,
    sha256_file,
    write_json_object,
)
from edge_llm_factory.train_general_kd import validate_rows


SUPPORTED_SCHEMAS = {"edge-llm-general-kd/v1", "edge-llm-general-kd/v2"}
GENERAL_CATEGORIES = ("code", "math", "natural_language_reasoning")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestError(
                        "{}:{} JSON 无效".format(path, line_number)
                    ) from exc
                if not isinstance(value, dict):
                    raise ManifestError(
                        "{}:{} 必须是 JSON object".format(path, line_number)
                    )
                rows.append(value)
    except OSError as exc:
        raise ManifestError("无法读取通用蒸馏数据 {}: {}".format(path, exc)) from exc
    if not rows:
        raise ManifestError("通用蒸馏数据为空: {}".format(path))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def _focus_categories(raw: Iterable[str]) -> Tuple[str, ...]:
    categories = tuple(str(value).strip() for value in raw if str(value).strip())
    if not categories:
        raise ManifestError("必须预注册至少一个通用能力类别")
    if len(categories) > 2:
        raise ManifestError("聚焦蒸馏最多预注册两个通用能力类别")
    if len(categories) != len(set(categories)):
        raise ManifestError("预注册通用能力类别不能重复")
    unknown = sorted(set(categories) - set(GENERAL_CATEGORIES))
    if unknown:
        raise ManifestError("未知通用能力类别: {}".format(", ".join(unknown)))
    return categories


def _normalise_prompt_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _prompt_parts(row: Mapping[str, Any]) -> List[str]:
    """Return only requester-visible prompt content, never the assistant target."""

    messages = row.get("messages")
    user_parts: List[str] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                raise ManifestError("messages 元素必须是 object")
            if message.get("role") == "user":
                content = _normalise_prompt_text(message.get("content", ""))
                if not content:
                    raise ManifestError("user message content 不能为空")
                user_parts.append(content)
    prompt = _normalise_prompt_text(row.get("prompt", ""))
    if prompt:
        return [prompt]
    source_prompt = _normalise_prompt_text(row.get("source_prompt", ""))
    if source_prompt:
        if user_parts:
            rendered_user = _normalise_prompt_text("\n".join(user_parts))
            if not (
                rendered_user == source_prompt
                or rendered_user.startswith(source_prompt + " ")
            ):
                raise ManifestError("source_prompt 与 user messages 内容不一致")
        return [source_prompt]
    if user_parts:
        return user_parts
    raise ManifestError("样本缺少可重算指纹的 user messages/prompt")


def _canonical_prompt_json(row: Mapping[str, Any]) -> str:
    payload = {"user_messages": _prompt_parts(row)}
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _canonical_prompt_fingerprint(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_prompt_json(row).encode("utf-8")).hexdigest()


def _legacy_prompt_fingerprint(row: Mapping[str, Any]) -> Optional[str]:
    """Accept old manifests only when their legacy fingerprint is content-provable."""

    source_prompt = _normalise_prompt_text(row.get("source_prompt", ""))
    category = str(row.get("category", ""))
    if not source_prompt or not category:
        return None
    return hashlib.sha256((category + "\n" + source_prompt).encode("utf-8")).hexdigest()


def _canonicalize_rows(
    rows: Sequence[Mapping[str, Any]], split: str
) -> List[Dict[str, Any]]:
    canonicalized: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        calculated = _canonical_prompt_fingerprint(row)
        declared = str(row.get("prompt_fingerprint", "")).strip().lower()
        accepted = {calculated}
        legacy = _legacy_prompt_fingerprint(row)
        if legacy:
            accepted.add(legacy)
        if not SHA256_PATTERN.fullmatch(declared) or declared not in accepted:
            raise ManifestError(
                "源 {} 第 {} 行 prompt_fingerprint 与实际内容不一致".format(
                    split, index
                )
            )
        normalized = dict(row)
        normalized["prompt_fingerprint"] = calculated
        canonicalized.append(normalized)
    return canonicalized


def _scene_specific_sample_count(rows: Sequence[Mapping[str, Any]]) -> int:
    scene_fields = ("scene", "scene_id", "scene_name", "scene_payload")
    return sum(
        row.get("scene_specific") is True
        or any(row.get(field) not in (None, "", {}, []) for field in scene_fields)
        for row in rows
    )


def _validate_isolation(
    manifest: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    evaluation_fingerprints: set,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    train_overlap = sum(
        str(row["prompt_fingerprint"]) in evaluation_fingerprints
        for row in train_rows
    )
    val_overlap = sum(
        str(row["prompt_fingerprint"]) in evaluation_fingerprints for row in val_rows
    )
    actual = {
        "scene_specific_samples": _scene_specific_sample_count(
            list(train_rows) + list(val_rows)
        ),
        "evaluation_prompt_overlap": train_overlap + val_overlap,
        "evaluation_set_used_for_training": train_overlap > 0,
    }
    for field, value in actual.items():
        if manifest.get(field) != value:
            raise ManifestError(
                "源数据隔离声明与实际数据不一致: {}（声明={!r}，实测={!r}）".format(
                    field, manifest.get(field), value
                )
            )
    expected = {
        "scene_specific_samples": 0,
        "evaluation_prompt_overlap": 0,
        "evaluation_set_used_for_training": False,
    }
    for field, value in expected.items():
        if actual[field] != value:
            raise ManifestError("源数据隔离要求未满足: {}".format(field))
    return actual, {"train": train_overlap, "validation": val_overlap}


def _artifact_path(
    manifest_path: Path, manifest: Mapping[str, Any], name: str
) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get(name), dict):
        raise ManifestError("源 manifest 缺少 {} artifact".format(name))
    artifact = artifacts[name]
    relative = safe_relative_path(artifact.get("path"), "artifacts.{}.path".format(name))
    path = (manifest_path.parent / relative).resolve()
    expected_sha = str(artifact.get("sha256", ""))
    try:
        actual_sha = sha256_file(path)
    except OSError as exc:
        raise ManifestError("无法读取源 {} 数据: {}".format(name, exc)) from exc
    if not SHA256_PATTERN.fullmatch(expected_sha) or actual_sha != expected_sha:
        raise ManifestError("源 {} 数据哈希与 manifest 不一致".format(name))
    return path


def _verified_external_artifact(
    path: Path, expected_sha: Any, label: str
) -> Tuple[Path, str]:
    expected = str(expected_sha or "").strip().lower()
    if not SHA256_PATTERN.fullmatch(expected):
        raise ManifestError("源 manifest 缺少有效的 {} SHA-256".format(label))
    path = Path(path).resolve()
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise ManifestError("无法读取源 {}: {}".format(label, exc)) from exc
    if actual != expected:
        raise ManifestError("源 {} 哈希与 manifest 不一致".format(label))
    return path, actual


def _request_manifest(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    request_path = manifest_path.parent / "request_manifest.json"
    expected = str(manifest.get("request_manifest_sha256", "")).strip().lower()
    if not expected and not request_path.exists():
        return None
    if not SHA256_PATTERN.fullmatch(expected):
        raise ManifestError("源 manifest 的 request_manifest_sha256 无效")
    try:
        actual = sha256_file(request_path)
    except OSError as exc:
        raise ManifestError("无法读取源 request manifest: {}".format(exc)) from exc
    if actual != expected:
        raise ManifestError("源 request manifest 哈希不一致")
    return read_json_object(request_path)


def _evaluation_artifact(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    explicit_path: Optional[Path],
) -> Tuple[Path, str]:
    artifacts = manifest.get("artifacts")
    evaluation_artifact = (
        artifacts.get("evaluation") if isinstance(artifacts, dict) else None
    )
    source_entry = None
    sources = manifest.get("sources")
    if isinstance(sources, dict) and isinstance(sources.get("frozen_evaluation"), dict):
        source_entry = sources["frozen_evaluation"]

    request = _request_manifest(manifest_path, manifest)
    request_sha = request.get("evaluation_sha256") if request else None
    declared_sha = manifest.get("evaluation_sha256")
    if isinstance(evaluation_artifact, dict):
        declared_sha = evaluation_artifact.get("sha256")
    elif isinstance(source_entry, dict):
        declared_sha = source_entry.get("sha256")
    elif request_sha:
        declared_sha = request_sha

    if explicit_path is not None:
        return _verified_external_artifact(
            Path(explicit_path), declared_sha, "正式评测集"
        )
    if isinstance(evaluation_artifact, dict):
        return (
            _artifact_path(manifest_path, manifest, "evaluation"),
            str(evaluation_artifact["sha256"]),
        )
    if isinstance(source_entry, dict):
        raw_path = source_entry.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ManifestError("源 manifest 的正式评测集 path 无效")
        path = Path(raw_path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        return _verified_external_artifact(
            path, source_entry.get("sha256"), "正式评测集"
        )
    if request and request_sha:
        candidates: List[Path] = []
        request_path = request.get("evaluation_path")
        if isinstance(request_path, str) and request_path.strip():
            candidate = Path(request_path)
            candidates.append(
                candidate
                if candidate.is_absolute()
                else manifest_path.parent / candidate
            )
        candidates.extend(
            [
                manifest_path.parent / "evaluation.jsonl",
                manifest_path.parent.parent / "general_capability_eval" / "eval.jsonl",
            ]
        )
        for candidate in candidates:
            if candidate.is_file() and sha256_file(candidate) == str(request_sha):
                return candidate.resolve(), str(request_sha)
        raise ManifestError("无法按 request manifest 找到已验真的正式评测集")
    raise ManifestError("源 manifest 未提供可验真的正式评测集")


def _category_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        category = str(row.get("category", ""))
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _verify_declared_counts(
    manifest: Mapping[str, Any], split: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    row_field = "train_rows" if split == "train" else "validation_rows"
    if row_field in manifest and manifest[row_field] != len(rows):
        raise ManifestError("源 manifest 的 {} 与文件不一致".format(row_field))
    count_field = "{}_category_counts".format(split)
    if count_field in manifest and manifest[count_field] != _category_counts(rows):
        raise ManifestError("源 manifest 的 {} 与文件不一致".format(count_field))


def _content_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    encoded = [
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    encoded.sort()
    return hashlib.sha256("\n".join(encoded).encode("utf-8")).hexdigest()


def _focus_audit(
    categories: Sequence[str],
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    evaluation_fingerprints: set,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for category in categories:
        category_train = [row for row in train_rows if row["category"] == category]
        category_val = [row for row in val_rows if row["category"] == category]
        if not category_train:
            raise ManifestError(
                "预注册类别 {} 在训练集中没有可用样本".format(category)
            )
        if not category_val:
            raise ManifestError(
                "预注册类别 {} 在验证集中没有可用样本".format(category)
            )
        train_overlap = sum(
            row["prompt_fingerprint"] in evaluation_fingerprints
            for row in category_train
        )
        val_overlap = sum(
            row["prompt_fingerprint"] in evaluation_fingerprints for row in category_val
        )
        report[category] = {
            "train": {
                "rows": len(category_train),
                "content_sha256": _content_sha256(category_train),
                "evaluation_prompt_overlap": train_overlap,
            },
            "validation": {
                "rows": len(category_val),
                "content_sha256": _content_sha256(category_val),
                "evaluation_prompt_overlap": val_overlap,
            },
            "evaluation_prompt_overlap": train_overlap + val_overlap,
        }
    return report


def build_focused_dataset(
    source_manifest: Path,
    output_dir: Path,
    focus_categories: Sequence[str],
    evaluation_jsonl: Optional[Path] = None,
    additional_evaluation_jsonl: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a focused pair while using the frozen evaluation set only for leakage checks."""

    source_manifest = Path(source_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    categories = _focus_categories(focus_categories)
    manifest = read_json_object(source_manifest)
    schema = str(manifest.get("schema_version", ""))
    if schema not in SUPPORTED_SCHEMAS:
        raise ManifestError("通用蒸馏数据 manifest schema 不受支持")
    teacher_model = str(manifest.get("teacher_model", "")).strip()
    if not teacher_model:
        raise ManifestError("源通用蒸馏数据未声明 Teacher")

    train_path = _artifact_path(source_manifest, manifest, "train")
    val_path = _artifact_path(source_manifest, manifest, "validation")
    if train_path == val_path:
        raise ManifestError("源训练集和验证集不能相同")
    if output_dir == source_manifest.parent:
        raise ManifestError("输出目录不能覆盖源数据目录")

    evaluation_path, evaluation_sha = _evaluation_artifact(
        source_manifest, manifest, evaluation_jsonl
    )
    if evaluation_path in {train_path, val_path}:
        raise ManifestError("正式评测集不能与训练集或验证集相同")

    train_rows = _canonicalize_rows(_read_jsonl(train_path), "train")
    val_rows = _canonicalize_rows(_read_jsonl(val_path), "validation")
    evaluation_rows = _read_jsonl(evaluation_path)
    source_evaluation_fingerprints = {
        _canonical_prompt_fingerprint(row) for row in evaluation_rows
    }
    additional_evaluation_path: Optional[Path] = None
    additional_evaluation_sha: Optional[str] = None
    additional_evaluation_rows: List[Dict[str, Any]] = []
    additional_evaluation_fingerprints: set = set()
    if additional_evaluation_jsonl is not None:
        additional_evaluation_path = Path(additional_evaluation_jsonl).resolve()
        if additional_evaluation_path in {train_path, val_path}:
            raise ManifestError("新增冻结终测集不能与训练集或验证集相同")
        if not additional_evaluation_path.is_file():
            raise ManifestError(
                "新增冻结终测集不存在: {}".format(additional_evaluation_path)
            )
        additional_evaluation_sha = sha256_file(additional_evaluation_path)
        additional_evaluation_rows = _read_jsonl(additional_evaluation_path)
        additional_evaluation_fingerprints = {
            _canonical_prompt_fingerprint(row)
            for row in additional_evaluation_rows
        }
    validate_rows(train_rows, teacher_model)
    validate_rows(val_rows, teacher_model)
    _verify_declared_counts(manifest, "train", train_rows)
    _verify_declared_counts(manifest, "validation", val_rows)

    train_fingerprints = {str(row["prompt_fingerprint"]) for row in train_rows}
    val_fingerprints = {str(row["prompt_fingerprint"]) for row in val_rows}
    overlap = sorted(train_fingerprints & val_fingerprints)
    if overlap:
        raise ManifestError("源训练集和验证集 prompt 重叠")

    isolation, evaluation_overlap = _validate_isolation(
        manifest, train_rows, val_rows, source_evaluation_fingerprints
    )

    additional_overlap = {
        "train": sum(
            str(row["prompt_fingerprint"]) in additional_evaluation_fingerprints
            for row in train_rows
        ),
        "validation": sum(
            str(row["prompt_fingerprint"]) in additional_evaluation_fingerprints
            for row in val_rows
        ),
    }
    if sum(additional_overlap.values()) > 0:
        raise ManifestError(
            "通用蒸馏数据与新增冻结终测集存在 prompt 重叠: train={}, validation={}".format(
                additional_overlap["train"], additional_overlap["validation"]
            )
        )
    all_evaluation_fingerprints = (
        source_evaluation_fingerprints | additional_evaluation_fingerprints
    )

    selected = set(categories)
    focused_train = [row for row in train_rows if str(row["category"]) in selected]
    focused_val = [row for row in val_rows if str(row["category"]) in selected]
    focus_audit = _focus_audit(
        categories, focused_train, focused_val, all_evaluation_fingerprints
    )

    output_paths = {
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "val.jsonl",
        "manifest": output_dir / "manifest.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise ManifestError("聚焦数据输出已存在，拒绝覆盖: {}".format(", ".join(existing)))

    _write_jsonl(output_paths["train"], focused_train)
    _write_jsonl(output_paths["validation"], focused_val)
    result: Dict[str, Any] = {
        "schema_version": schema,
        "task": "scene_independent_focused_behavior_distillation",
        "teacher_model": teacher_model,
        "teacher_no_thinking": manifest.get("teacher_no_thinking"),
        "train_rows": len(focused_train),
        "validation_rows": len(focused_val),
        "train_category_counts": _category_counts(focused_train),
        "validation_category_counts": _category_counts(focused_val),
        "pre_registered_focus_categories": list(categories),
        "focus_registration": {
            "selection_stage": "before_candidate_training",
            "final_evaluation_used_for_selection": False,
        },
        "focus_category_audit": focus_audit,
        **isolation,
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_schema_version": schema,
        "source_artifacts": {
            "train_sha256": sha256_file(train_path),
            "validation_sha256": sha256_file(val_path),
            "evaluation_path": str(evaluation_path),
            "evaluation_sha256": evaluation_sha,
            "evaluation_rows": len(evaluation_rows),
            "evaluation_overlap": evaluation_overlap,
        },
        "additional_frozen_evaluation": (
            {
                "path": str(additional_evaluation_path),
                "sha256": additional_evaluation_sha,
                "rows": len(additional_evaluation_rows),
                "used_for_training": False,
                "overlap": additional_overlap,
            }
            if additional_evaluation_path is not None
            else None
        ),
        "artifacts": {
            "train": {
                "path": output_paths["train"].name,
                "sha256": sha256_file(output_paths["train"]),
            },
            "validation": {
                "path": output_paths["validation"].name,
                "sha256": sha256_file(output_paths["validation"]),
            },
        },
    }
    write_json_object(output_paths["manifest"], result)
    return result


def main(argv: Sequence[str] = None) -> None:
    parser = argparse.ArgumentParser(
        description="从已验收通用蒸馏数据生成预注册能力类别子集"
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--focus-category", action="append", required=True)
    parser.add_argument(
        "--evaluation-jsonl",
        help="可选；显式指定正式评测集，仍必须匹配源 manifest 绑定的 SHA-256",
    )
    parser.add_argument(
        "--additional-evaluation-jsonl",
        help="可选；训练数据生成后新冻结的终测集，仅做真实重叠审计并绑定 SHA-256",
    )
    args = parser.parse_args(argv)
    report = build_focused_dataset(
        Path(args.source_manifest),
        Path(args.output_dir),
        args.focus_category,
        Path(args.evaluation_jsonl) if args.evaluation_jsonl else None,
        (
            Path(args.additional_evaluation_jsonl)
            if args.additional_evaluation_jsonl
            else None
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
