"""Build a frozen, leakage-checked C-Eval reasoning holdout.

Only local HuggingFace C-Eval ``dev``, ``val``, or labelled ``test`` Arrow shards are
accepted.  Training, validation, and earlier evaluation JSONL files are treated as exclusion sets;
their requester-visible prompts are recomputed instead of trusting declared
fingerprints.  Selection is deterministic and balanced across C-Eval subjects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from edge_llm_factory.contracts import ManifestError, sha256_file


SCHEMA_VERSION = "edge-llm-frozen-nlr-evaluation/v1"
DATASET_ID = "ceval/ceval-exam"
DATASET_SPLIT = "val"
DATASET_SPLITS = ("dev", "val", "test")
CATEGORY = "natural_language_reasoning"
ANSWER_CHOICES = ("A", "B", "C", "D")
SUBJECT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")
VAL_ARROW_NAME = "ceval-exam-val.arrow"
_OPTION_LINE = re.compile(
    r"(?:^|\n)\s*([ABCD])\s*[.、:：)]\s*", re.IGNORECASE
)
_OPTION_INLINE = re.compile(
    r"(?:^|\s)([ABCD])\s*[.、:：)]\s+", re.IGNORECASE
)


def normalize_prompt(value: str) -> str:
    """Normalize presentation-only differences without changing question text."""

    normalized = unicodedata.normalize("NFKC", str(value)).replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")
    return re.sub(r"\s+", " ", normalized).strip()


def prompt_fingerprint(value: str) -> str:
    normalized = normalize_prompt(value)
    if not normalized:
        raise ManifestError("C-Eval prompt 不能为空")
    return hashlib.sha256((CATEGORY + "\n" + normalized).encode("utf-8")).hexdigest()


def _item_parts(value: str) -> Optional[Tuple[str, str, str, str, str]]:
    """Parse a rendered C-Eval item independently of option-marker style."""

    rendered = unicodedata.normalize("NFKC", str(value)).replace("\r\n", "\n")
    rendered = rendered.replace("\r", "\n")
    matches = list(_OPTION_LINE.finditer(rendered))
    if [match.group(1).upper() for match in matches] != list(ANSWER_CHOICES):
        inline_matches = list(_OPTION_INLINE.finditer(rendered))
        matches = []
        for index in range(max(0, len(inline_matches) - 3)):
            candidate = inline_matches[index : index + 4]
            if [match.group(1).upper() for match in candidate] == list(
                ANSWER_CHOICES
            ):
                matches = candidate
                break
    if len(matches) != 4:
        return None
    stem = normalize_prompt(rendered[: matches[0].start()])
    options = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index < 3 else len(rendered)
        options.append(normalize_prompt(rendered[match.end() : end]))
    if not stem or any(not option for option in options):
        return None
    return (stem, options[0], options[1], options[2], options[3])


def item_fingerprint(value: str) -> str:
    """Fingerprint question stem plus all choices, but not answer or formatting."""

    parts = _item_parts(value)
    if parts is None:
        raise ManifestError("无法从 prompt 解析完整的 C-Eval 题干与 A/B/C/D 选项")
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(("ceval-item\n" + payload).encode("utf-8")).hexdigest()


def _maybe_item_fingerprint(value: str) -> Optional[str]:
    parts = _item_parts(value)
    if parts is None:
        return None
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(("ceval-item\n" + payload).encode("utf-8")).hexdigest()


def format_ceval_prompt(row: Mapping[str, Any], location: str = "C-Eval row") -> str:
    question = str(row.get("question", "")).strip()
    if not question:
        raise ManifestError(f"{location} 缺少 question")
    options = []
    for choice in ANSWER_CHOICES:
        option = str(row.get(choice, "")).strip()
        if not option:
            raise ManifestError(f"{location} 缺少选项 {choice}")
        options.append(f"{choice}. {option}")
    return question + "\n" + "\n".join(options)


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
                    raise ManifestError(f"{path}:{line_number} JSON 无效") from exc
                if not isinstance(value, dict):
                    raise ManifestError(f"{path}:{line_number} 必须是 JSON object")
                rows.append(value)
    except OSError as exc:
        raise ManifestError(f"无法读取排除集 {path}: {exc}") from exc
    if not rows:
        raise ManifestError(f"排除集为空: {path}")
    return rows


def _row_prompts(row: Mapping[str, Any], location: str) -> List[str]:
    """Return every requester-visible representation carried by an exclusion row."""

    prompts: List[str] = []
    for field in ("source_prompt", "prompt"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            prompts.append(value)
    question = row.get("question")
    if isinstance(question, str) and question.strip():
        if all(
            isinstance(row.get(choice), str) and str(row[choice]).strip()
            for choice in ANSWER_CHOICES
        ):
            prompts.append(format_ceval_prompt(row, location))
        else:
            prompts.append(question)
    messages = row.get("messages")
    if isinstance(messages, list):
        users = [
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict)
            and message.get("role") == "user"
            and str(message.get("content", "")).strip()
        ]
        prompts.extend(users)
        if len(users) > 1:
            prompts.append("\n".join(users))
    unique: List[str] = []
    seen: Set[str] = set()
    for prompt in prompts:
        normalized = normalize_prompt(prompt)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(prompt)
    if not unique:
        raise ManifestError(f"{location} 无法提取 requester-visible prompt")
    return unique


def exclusion_identities(
    sources: Sequence[Tuple[str, Sequence[Mapping[str, Any]]]]
) -> Tuple[Set[str], Set[str], Dict[str, Dict[str, int]]]:
    """Recompute prompt and complete-item identities for declared exclusions."""

    prompt_ids: Set[str] = set()
    item_ids: Set[str] = set()
    counts: Dict[str, Dict[str, int]] = {}
    for label, rows in sources:
        prompt_before = len(prompt_ids)
        item_before = len(item_ids)
        for index, row in enumerate(rows):
            for prompt in _row_prompts(row, f"{label}[{index}]"):
                prompt_ids.add(prompt_fingerprint(prompt))
                item_id = _maybe_item_fingerprint(prompt)
                if item_id is not None:
                    item_ids.add(item_id)
        counts[label] = {
            "rows": len(rows),
            "new_unique_prompts": len(prompt_ids) - prompt_before,
            "new_unique_items": len(item_ids) - item_before,
        }
    return prompt_ids, item_ids, counts


def discover_split_arrow_files(
    cache_root: Path, dataset_split: str
) -> Dict[str, Path]:
    if dataset_split not in DATASET_SPLITS:
        raise ManifestError(f"C-Eval split 不受支持: {dataset_split}")
    arrow_name = f"ceval-exam-{dataset_split}.arrow"
    root = cache_root.resolve()
    if not root.is_dir():
        raise ManifestError(f"C-Eval Arrow 缓存根目录不存在: {root}")
    discovered: Dict[str, Path] = {}
    for path in sorted(root.rglob(arrow_name)):
        if not path.is_file():
            continue
        try:
            relative = path.resolve().relative_to(root)
        except ValueError as exc:
            raise ManifestError(f"C-Eval Arrow 路径越出缓存根目录: {path}") from exc
        if len(relative.parts) < 2:
            raise ManifestError(f"无法从 Arrow 路径识别 C-Eval 配置: {relative}")
        subject = relative.parts[0]
        if not SUBJECT_PATTERN.fullmatch(subject):
            raise ManifestError(f"C-Eval 配置名无效: {subject}")
        if subject in discovered:
            raise ManifestError(
                f"C-Eval 配置 {subject} 存在多个 val Arrow，拒绝含糊选择"
            )
        discovered[subject] = path.resolve()
    if not discovered:
        raise ManifestError(f"缓存根目录中没有 {arrow_name}: {root}")
    return dict(sorted(discovered.items()))


def discover_val_arrow_files(cache_root: Path) -> Dict[str, Path]:
    """Backward-compatible val discovery used by older callers/tests."""

    return discover_split_arrow_files(cache_root, "val")


def _read_arrow_rows(path: Path) -> List[Dict[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise ManifestError("读取 HuggingFace Arrow 缓存需要安装 pyarrow") from exc
    try:
        with pa.memory_map(str(path), "r") as source:
            table = ipc.open_stream(source).read_all()
    except Exception as exc:
        raise ManifestError(f"无法读取 C-Eval Arrow {path}: {exc}") from exc
    required = {"id", "question", "A", "B", "C", "D", "answer"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ManifestError(f"C-Eval Arrow 缺少字段 {missing}: {path}")
    return [dict(row) for row in table.select(sorted(required)).to_pylist()]


def load_ceval_rows(
    cache_root: Path, dataset_split: str
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    root = cache_root.resolve()
    subject_rows: Dict[str, List[Dict[str, Any]]] = {}
    source_files: Dict[str, Dict[str, Any]] = {}
    for subject, path in discover_split_arrow_files(root, dataset_split).items():
        rows = _read_arrow_rows(path)
        relative = path.relative_to(root)
        subject_rows[subject] = rows
        source_files[subject] = {
            "subject": subject,
            "relative_path": relative.as_posix(),
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(rows),
        }
    return subject_rows, source_files


def load_ceval_val_rows(
    cache_root: Path,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    """Backward-compatible val loader used by older callers/tests."""

    return load_ceval_rows(cache_root, "val")


def _stable_key(seed: int, *parts: Any) -> str:
    payload = ":".join([str(seed)] + [str(part) for part in parts])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _digest(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def build_nlr_evaluation(
    subject_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    excluded_prompts: Set[str],
    excluded_items: Set[str],
    *,
    sample_count: int,
    seed: int,
    source_files: Optional[Mapping[str, Mapping[str, Any]]] = None,
    dataset_split: str = DATASET_SPLIT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if sample_count <= 0:
        raise ManifestError("sample_count 必须大于 0")
    if not subject_rows:
        raise ManifestError("C-Eval val 配置不能为空")
    if dataset_split not in DATASET_SPLITS:
        raise ManifestError(f"C-Eval split 不受支持: {dataset_split}")

    eligible: Dict[str, List[Dict[str, Any]]] = {
        subject: [] for subject in subject_rows
    }
    candidate_groups: Dict[str, List[Dict[str, Any]]] = {}
    sample_ids: Set[str] = set()
    subject_counters: Dict[str, Counter] = {
        subject: Counter() for subject in subject_rows
    }
    source_files = source_files or {}

    for subject in sorted(subject_rows):
        if not SUBJECT_PATTERN.fullmatch(subject):
            raise ManifestError(f"C-Eval 配置名无效: {subject}")
        stats = subject_counters[subject]
        for row_index, raw in enumerate(subject_rows[subject]):
            stats["source_rows"] += 1
            answer = str(raw.get("answer", "")).strip().upper()
            if not answer:
                stats["unanswered_rows"] += 1
                continue
            if answer not in ANSWER_CHOICES:
                raise ManifestError(
                    f"C-Eval {subject} {dataset_split} row {row_index} 的 answer 必须是 A、B、C 或 D"
                )
            stats["answered_rows"] += 1
            prompt = format_ceval_prompt(
                raw, f"C-Eval {subject} {dataset_split} row {row_index}"
            )
            prompt_id = prompt_fingerprint(prompt)
            item_id = item_fingerprint(prompt)
            raw_id = raw.get("id", row_index)
            if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
                raise ManifestError(
                    f"C-Eval {subject} {dataset_split} row {row_index} 的 id 无效"
                )
            source_id = str(raw_id).strip()
            if not source_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", source_id):
                raise ManifestError(
                    f"C-Eval {subject} {dataset_split} row {row_index} 的 id 无效"
                )
            sample_id = f"ceval_{subject}_{dataset_split}_{source_id}"
            if sample_id in sample_ids:
                raise ManifestError(f"C-Eval sample_id 重复: {sample_id}")
            sample_ids.add(sample_id)
            if prompt_id in excluded_prompts or item_id in excluded_items:
                stats["excluded_rows"] += 1
                continue

            source_info = source_files.get(subject, {})
            source = {
                "dataset_id": DATASET_ID,
                "config": subject,
                "split": dataset_split,
                "row_index": row_index,
                "source_id": raw_id,
            }
            if source_info:
                source["arrow_relative_path"] = str(source_info.get("relative_path", ""))
                source["arrow_sha256"] = str(source_info.get("sha256", ""))
            candidate_groups.setdefault(item_id, []).append(
                {
                    "benchmark": "ceval",
                    "category": CATEGORY,
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "reference_answer": answer,
                    "prompt_fingerprint": prompt_id,
                    "source": source,
                    "_item_fingerprint": item_id,
                }
            )

    conflicting_answer_items = 0
    for item_id in sorted(candidate_groups):
        group = candidate_groups[item_id]
        answers = {str(row["reference_answer"]) for row in group}
        if len(answers) > 1:
            conflicting_answer_items += 1
            for row in group:
                subject_counters[str(row["source"]["config"])][
                    "conflicting_answer_rows"
                ] += 1
            continue
        group.sort(
            key=lambda row: (
                str(row["source"]["config"]),
                str(row["sample_id"]),
            )
        )
        kept = group[0]
        eligible[str(kept["source"]["config"])].append(kept)
        for duplicate in group[1:]:
            subject_counters[str(duplicate["source"]["config"])][
                "duplicate_rows"
            ] += 1

    subject_stats: Dict[str, Dict[str, int]] = {}
    count_fields = (
        "source_rows",
        "answered_rows",
        "unanswered_rows",
        "excluded_rows",
        "duplicate_rows",
        "conflicting_answer_rows",
        "eligible_rows",
    )
    for subject in sorted(subject_rows):
        candidates = eligible[subject]
        candidates.sort(
            key=lambda row: _stable_key(
                seed, subject, row["sample_id"], row["prompt_fingerprint"]
            )
        )
        subject_counters[subject]["eligible_rows"] = len(candidates)
        subject_stats[subject] = {
            field: int(subject_counters[subject][field]) for field in count_fields
        }

    total_eligible = sum(len(rows) for rows in eligible.values())
    if total_eligible < sample_count:
        raise ManifestError(
            f"排除后仅剩 {total_eligible} 道 C-Eval val 题，少于请求的 {sample_count} 道"
        )

    subject_order = sorted(eligible, key=lambda subject: _stable_key(seed, subject))
    offsets = {subject: 0 for subject in subject_order}
    selected_internal: List[Dict[str, Any]] = []
    while len(selected_internal) < sample_count:
        progressed = False
        for subject in subject_order:
            offset = offsets[subject]
            if offset >= len(eligible[subject]):
                continue
            selected_internal.append(eligible[subject][offset])
            offsets[subject] += 1
            progressed = True
            if len(selected_internal) == sample_count:
                break
        if not progressed:
            raise ManifestError("C-Eval 分层抽样提前耗尽候选池")

    selected_prompt_ids = {str(row["prompt_fingerprint"]) for row in selected_internal}
    selected_item_ids = {str(row["_item_fingerprint"]) for row in selected_internal}
    prompt_overlap = selected_prompt_ids & excluded_prompts
    item_overlap = selected_item_ids & excluded_items
    if prompt_overlap or item_overlap:
        raise ManifestError("冻结 C-Eval 终测集与排除集存在 prompt 重叠")

    selected: List[Dict[str, Any]] = []
    for internal in selected_internal:
        row = dict(internal)
        row.pop("_item_fingerprint")
        selected.append(row)
    selected_counter = Counter(str(row["source"]["config"]) for row in selected)
    selected_subject_counts = {
        subject: int(selected_counter[subject]) for subject in sorted(subject_rows)
    }
    all_stats = Counter()
    for stats in subject_stats.values():
        all_stats.update(stats)
    report = {
        "source_subject_count": len(subject_rows),
        "source_rows": all_stats["source_rows"],
        "answered_rows": all_stats["answered_rows"],
        "unanswered_rows": all_stats["unanswered_rows"],
        "excluded_rows": all_stats["excluded_rows"],
        "duplicate_rows": all_stats["duplicate_rows"],
        "conflicting_answer_rows": all_stats["conflicting_answer_rows"],
        "conflicting_answer_items": conflicting_answer_items,
        "eligible_rows": total_eligible,
        "selected_rows": len(selected),
        "subject_counts": subject_stats,
        "eligible_subject_counts": {
            subject: len(eligible[subject]) for subject in sorted(eligible)
        },
        "selected_subject_counts": selected_subject_counts,
        "excluded_prompt_overlap": len(prompt_overlap),
        "excluded_item_overlap": len(item_overlap),
        "selected_sample_ids_sha256": _digest(
            str(row["sample_id"]) for row in selected
        ),
        "selected_prompt_fingerprints_sha256": _digest(
            str(row["prompt_fingerprint"]) for row in selected
        ),
    }
    return selected, report


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as file_obj:
            for row in rows:
                file_obj.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    except FileExistsError as exc:
        raise ManifestError(f"拒绝覆盖已有冻结证据: {path}") from exc


def _write_manifest_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as file_obj:
            json.dump(dict(value), file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            file_obj.write("\n")
    except FileExistsError as exc:
        raise ManifestError(f"拒绝覆盖已有冻结证据: {path}") from exc


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="构建与训练/验证/旧评测隔离的冻结 C-Eval 自然语言推理终测集。"
    )
    parser.add_argument("--ceval_cache_root", required=True)
    parser.add_argument(
        "--dataset_split", choices=DATASET_SPLITS, default=DATASET_SPLIT
    )
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--validation_jsonl", required=True)
    parser.add_argument("--old_eval_jsonl", required=True)
    parser.add_argument("--exclude_jsonl", action="append", default=[])
    parser.add_argument(
        "--subject",
        action="append",
        default=[],
        help=(
            "只从指定 C-Eval subject 冻结样本；可重复声明。"
            "未声明时使用缓存中的全部 subject。"
        ),
    )
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample_count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args(argv)

    cache_root = Path(args.ceval_cache_root).resolve()
    output_path = Path(args.output_jsonl).resolve()
    manifest_path = Path(args.manifest).resolve()
    if output_path == manifest_path:
        raise ManifestError("output_jsonl 与 manifest 不能是同一文件")
    for path in (output_path, manifest_path):
        if path.exists():
            raise ManifestError(f"拒绝覆盖已有冻结证据: {path}")

    exclusion_sources: List[Tuple[str, Sequence[Mapping[str, Any]]]] = []
    exclusion_artifacts: List[Dict[str, Any]] = []
    seen_exclusion_paths: Set[Path] = set()
    declared_exclusions = [
        ("train", args.train_jsonl),
        ("validation", args.validation_jsonl),
        ("old_eval", args.old_eval_jsonl),
    ] + [
        (f"additional_{index}", raw_path)
        for index, raw_path in enumerate(args.exclude_jsonl)
    ]
    for label, raw_path in declared_exclusions:
        path = Path(raw_path).resolve()
        if path in seen_exclusion_paths:
            raise ManifestError(f"排除集不能重复声明: {path}")
        seen_exclusion_paths.add(path)
        if not path.is_file():
            raise ManifestError(f"排除集不存在: {path}")
        rows = _read_jsonl(path)
        exclusion_sources.append((label, rows))
        exclusion_artifacts.append(
            {
                "label": label,
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(rows),
            }
        )
    excluded_prompts, excluded_items, exclusion_counts = exclusion_identities(
        exclusion_sources
    )
    subject_rows, source_files = load_ceval_rows(cache_root, args.dataset_split)
    requested_subjects = [str(subject).strip() for subject in args.subject]
    if any(not subject for subject in requested_subjects):
        raise ManifestError("subject 不能为空")
    if len(set(requested_subjects)) != len(requested_subjects):
        raise ManifestError("subject 不能重复声明")
    unknown_subjects = sorted(set(requested_subjects) - set(subject_rows))
    if unknown_subjects:
        raise ManifestError(f"C-Eval 缓存中不存在指定 subject: {unknown_subjects}")
    if requested_subjects:
        selected_subjects = sorted(requested_subjects)
        subject_rows = {subject: subject_rows[subject] for subject in selected_subjects}
        source_files = {
            subject: source_files[subject] for subject in selected_subjects
        }
    selected, selection = build_nlr_evaluation(
        subject_rows,
        excluded_prompts,
        excluded_items,
        sample_count=args.sample_count,
        seed=args.seed,
        source_files=source_files,
        dataset_split=args.dataset_split,
    )
    _write_jsonl(output_path, selected)

    arrow_manifest = [source_files[subject] for subject in sorted(source_files)]
    arrow_set_sha256 = hashlib.sha256(
        "\n".join(
            f"{row['subject']}:{row['relative_path']}:{row['sha256']}"
            for row in arrow_manifest
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": "preregistered_scene_independent_natural_language_reasoning_evaluation",
        "dataset": {
            "dataset_id": DATASET_ID,
            "split": args.dataset_split,
            "cache_root": str(cache_root),
            "subject_count": len(source_files),
            "arrow_files": arrow_manifest,
            "arrow_files_sha256": arrow_set_sha256,
            "subject_filter": sorted(requested_subjects),
        },
        "seed": args.seed,
        "selection_policy": (
            "subject order=sha256(seed:subject); within subject="
            "sha256(seed:subject:sample_id:prompt_fingerprint); balanced round-robin"
        ),
        "training_or_tuning_use_allowed": False,
        "scene_specific_samples": 0,
        "exclusions": exclusion_artifacts,
        "exclusion_unique_counts": exclusion_counts,
        "excluded_unique_prompt_count": len(excluded_prompts),
        "excluded_unique_item_count": len(excluded_items),
        "selection": selection,
        "artifact": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "rows": len(selected),
        },
    }
    _write_manifest_exclusive(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
