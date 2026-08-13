"""Build a focused, leakage-checked C-Eval distillation source.

The builder reads labelled C-Eval Arrow shards from local storage, removes every
prompt or complete item present in frozen evaluations, and deterministically
splits each requested subject into training and validation rows.  The produced
JSONL files are source inputs for :mod:`edge_llm_factory.general_kd_data`; they
contain gold A-D references, not Teacher rollouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from edge_llm_factory.build_frozen_nlr_eval import (
    ANSWER_CHOICES,
    CATEGORY,
    DATASET_ID,
    DATASET_SPLITS,
    SUBJECT_PATTERN,
    _read_jsonl,
    exclusion_identities,
    format_ceval_prompt,
    item_fingerprint,
    load_ceval_rows,
    prompt_fingerprint,
)
from edge_llm_factory.contracts import ManifestError, sha256_file


SCHEMA_VERSION = "edge-llm-focused-ceval-kd-source/v1"
SYSTEM_PROMPT = "回答中文单项选择题。只输出 A、B、C 或 D，不要解释。"
DEFAULT_DATASET_SPLIT = "val"
DEFAULT_SEED = 20260809


def _stable_key(seed: int, *parts: Any) -> str:
    payload = "|".join(
        [SCHEMA_VERSION, str(seed)] + [str(part) for part in parts]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _digest(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _source_id(
    raw: Mapping[str, Any], subject: str, dataset_split: str, row_index: int
) -> Tuple[Any, str, str]:
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
    event_id = f"ceval_{subject}_{dataset_split}_{source_id}"
    return raw_id, source_id, event_id


def _source_row(
    *,
    subject: str,
    dataset_split: str,
    row_index: int,
    raw_id: Any,
    event_id: str,
    prompt: str,
    prompt_id: str,
    item_id: str,
    answer: str,
    source_file: Mapping[str, Any],
) -> Dict[str, Any]:
    source: Dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "config": subject,
        "split": dataset_split,
        "row_index": row_index,
        "source_id": raw_id,
    }
    if source_file:
        source["arrow_relative_path"] = str(source_file.get("relative_path", ""))
        source["arrow_sha256"] = str(source_file.get("sha256", ""))
    return {
        "event_id": event_id,
        "category": CATEGORY,
        "prompt_format": "tokenizer_chat",
        "source_prompt": prompt,
        "prompt_fingerprint": prompt_id,
        "item_fingerprint": item_id,
        "subject": subject,
        "reference_answer": answer,
        "source": source,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def build_focused_ceval_source(
    subject_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    excluded_prompts: Set[str],
    excluded_items: Set[str],
    *,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    validation_per_subject: int = 5,
    train_per_subject: int = 0,
    seed: int = DEFAULT_SEED,
    source_files: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Return deterministic train/validation source rows and an audit report.

    ``train_per_subject=0`` means use every eligible item left after reserving the
    validation stratum.  Duplicate items with the same answer are represented
    once; every copy of an item with conflicting answers is excluded.
    """

    if dataset_split not in DATASET_SPLITS:
        raise ManifestError(f"C-Eval split 不受支持: {dataset_split}")
    if not subject_rows:
        raise ManifestError("C-Eval subject 不能为空")
    for name, value, minimum in (
        ("validation_per_subject", validation_per_subject, 1),
        ("train_per_subject", train_per_subject, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ManifestError(f"{name} 必须是大于等于 {minimum} 的整数")

    source_files = source_files or {}
    candidates_by_item: Dict[str, List[Dict[str, Any]]] = {}
    subject_counters: Dict[str, Counter] = {
        subject: Counter() for subject in subject_rows
    }
    event_ids: Set[str] = set()
    prompt_to_item: Dict[str, str] = {}

    for subject in sorted(subject_rows):
        if not SUBJECT_PATTERN.fullmatch(subject):
            raise ManifestError(f"C-Eval subject 无效: {subject}")
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
            previous_item = prompt_to_item.setdefault(prompt_id, item_id)
            if previous_item != item_id:
                raise ManifestError("C-Eval prompt 指纹对应多个不同完整题目")

            raw_id, _source_id_text, event_id = _source_id(
                raw, subject, dataset_split, row_index
            )
            if event_id in event_ids:
                raise ManifestError(f"C-Eval event_id 重复: {event_id}")
            event_ids.add(event_id)
            if prompt_id in excluded_prompts or item_id in excluded_items:
                stats["evaluation_excluded_rows"] += 1
                continue

            candidate = _source_row(
                subject=subject,
                dataset_split=dataset_split,
                row_index=row_index,
                raw_id=raw_id,
                event_id=event_id,
                prompt=prompt,
                prompt_id=prompt_id,
                item_id=item_id,
                answer=answer,
                source_file=source_files.get(subject, {}),
            )
            candidates_by_item.setdefault(item_id, []).append(candidate)

    eligible: Dict[str, List[Dict[str, Any]]] = {
        subject: [] for subject in subject_rows
    }
    conflicting_answer_items = 0
    for item_id in sorted(candidates_by_item):
        group = candidates_by_item[item_id]
        answers = {str(row["reference_answer"]) for row in group}
        if len(answers) > 1:
            conflicting_answer_items += 1
            for row in group:
                subject_counters[str(row["subject"])][
                    "conflicting_answer_rows"
                ] += 1
            continue
        group.sort(key=lambda row: (str(row["subject"]), str(row["event_id"])))
        kept = group[0]
        eligible[str(kept["subject"])].append(kept)
        for duplicate in group[1:]:
            subject_counters[str(duplicate["subject"])]["duplicate_rows"] += 1

    train_rows: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
    subject_stats: Dict[str, Dict[str, int]] = {}
    for subject in sorted(subject_rows):
        rows = eligible[subject]
        rows.sort(
            key=lambda row: _stable_key(
                seed,
                subject,
                row["event_id"],
                row["prompt_fingerprint"],
                row["item_fingerprint"],
            )
        )
        minimum_train = train_per_subject if train_per_subject > 0 else 1
        required = validation_per_subject + minimum_train
        if len(rows) < required:
            raise ManifestError(
                f"C-Eval subject {subject} 排除后仅 {len(rows)} 题，"
                f"无法分出 validation={validation_per_subject}、train>={minimum_train}"
            )
        validation = rows[:validation_per_subject]
        remaining = rows[validation_per_subject:]
        training = (
            remaining[:train_per_subject] if train_per_subject > 0 else remaining
        )
        validation_rows.extend(validation)
        train_rows.extend(training)
        stats = subject_counters[subject]
        stats["eligible_rows"] = len(rows)
        stats["validation_rows"] = len(validation)
        stats["train_rows"] = len(training)
        stats["unused_rows"] = len(remaining) - len(training)
        subject_stats[subject] = {
            field: int(stats[field])
            for field in (
                "source_rows",
                "answered_rows",
                "unanswered_rows",
                "evaluation_excluded_rows",
                "duplicate_rows",
                "conflicting_answer_rows",
                "eligible_rows",
                "train_rows",
                "validation_rows",
                "unused_rows",
            )
        }

    train_prompt_ids = {str(row["prompt_fingerprint"]) for row in train_rows}
    validation_prompt_ids = {
        str(row["prompt_fingerprint"]) for row in validation_rows
    }
    train_item_ids = {str(row["item_fingerprint"]) for row in train_rows}
    validation_item_ids = {str(row["item_fingerprint"]) for row in validation_rows}
    if train_prompt_ids & validation_prompt_ids or train_item_ids & validation_item_ids:
        raise ManifestError("C-Eval 蒸馏源 train/validation 存在题目重叠")
    selected_prompt_ids = train_prompt_ids | validation_prompt_ids
    selected_item_ids = train_item_ids | validation_item_ids
    excluded_prompt_overlap = selected_prompt_ids & excluded_prompts
    excluded_item_overlap = selected_item_ids & excluded_items
    if excluded_prompt_overlap or excluded_item_overlap:
        raise ManifestError("C-Eval 蒸馏源与冻结 evaluation 存在题目泄漏")
    selected_event_ids = [str(row["event_id"]) for row in train_rows + validation_rows]
    if len(selected_event_ids) != len(set(selected_event_ids)):
        raise ManifestError("C-Eval 蒸馏源 event_id 重复")

    total_stats = Counter()
    for stats in subject_stats.values():
        total_stats.update(stats)
    report = {
        "source_subject_count": len(subject_rows),
        "source_rows": total_stats["source_rows"],
        "answered_rows": total_stats["answered_rows"],
        "unanswered_rows": total_stats["unanswered_rows"],
        "evaluation_excluded_rows": total_stats["evaluation_excluded_rows"],
        "duplicate_rows": total_stats["duplicate_rows"],
        "conflicting_answer_rows": total_stats["conflicting_answer_rows"],
        "conflicting_answer_items": conflicting_answer_items,
        "eligible_rows": total_stats["eligible_rows"],
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "unused_rows": total_stats["unused_rows"],
        "subject_counts": subject_stats,
        "train_subject_counts": dict(
            sorted(Counter(str(row["subject"]) for row in train_rows).items())
        ),
        "validation_subject_counts": dict(
            sorted(
                Counter(str(row["subject"]) for row in validation_rows).items()
            )
        ),
        "train_validation_prompt_overlap": 0,
        "train_validation_item_overlap": 0,
        "excluded_prompt_overlap": len(excluded_prompt_overlap),
        "excluded_item_overlap": len(excluded_item_overlap),
        "train_event_ids_sha256": _digest(
            str(row["event_id"]) for row in train_rows
        ),
        "validation_event_ids_sha256": _digest(
            str(row["event_id"]) for row in validation_rows
        ),
        "train_prompt_fingerprints_sha256": _digest(train_prompt_ids),
        "validation_prompt_fingerprints_sha256": _digest(
            validation_prompt_ids
        ),
        "train_item_fingerprints_sha256": _digest(train_item_ids),
        "validation_item_fingerprints_sha256": _digest(validation_item_ids),
    }
    return train_rows, validation_rows, report


def _prepare_role_rows(
    subject_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    excluded_prompts: Set[str],
    excluded_items: Set[str],
    *,
    dataset_split: str,
    source_files: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Normalize and clean one official C-Eval split before role assignment."""

    if dataset_split not in DATASET_SPLITS:
        raise ManifestError(f"C-Eval split 不受支持: {dataset_split}")
    if not subject_rows:
        raise ManifestError("C-Eval subject 不能为空")
    source_files = source_files or {}
    counters: Dict[str, Counter] = {subject: Counter() for subject in subject_rows}
    event_ids: Set[str] = set()
    candidates_by_item: Dict[str, List[Dict[str, Any]]] = {}
    for subject in sorted(subject_rows):
        if not SUBJECT_PATTERN.fullmatch(subject):
            raise ManifestError(f"C-Eval subject 无效: {subject}")
        stats = counters[subject]
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
            raw_id, _source_id_text, event_id = _source_id(
                raw, subject, dataset_split, row_index
            )
            if event_id in event_ids:
                raise ManifestError(f"C-Eval event_id 重复: {event_id}")
            event_ids.add(event_id)
            if prompt_id in excluded_prompts or item_id in excluded_items:
                stats["evaluation_excluded_rows"] += 1
                continue
            candidates_by_item.setdefault(item_id, []).append(
                _source_row(
                    subject=subject,
                    dataset_split=dataset_split,
                    row_index=row_index,
                    raw_id=raw_id,
                    event_id=event_id,
                    prompt=prompt,
                    prompt_id=prompt_id,
                    item_id=item_id,
                    answer=answer,
                    source_file=source_files.get(subject, {}),
                )
            )

    eligible: Dict[str, List[Dict[str, Any]]] = {
        subject: [] for subject in subject_rows
    }
    conflicting_answer_items = 0
    for item_id in sorted(candidates_by_item):
        group = candidates_by_item[item_id]
        if len({str(row["reference_answer"]) for row in group}) > 1:
            conflicting_answer_items += 1
            for row in group:
                counters[str(row["subject"])]["conflicting_answer_rows"] += 1
            continue
        group.sort(key=lambda row: (str(row["subject"]), str(row["event_id"])))
        eligible[str(group[0]["subject"])].append(group[0])
        for duplicate in group[1:]:
            counters[str(duplicate["subject"])]["duplicate_rows"] += 1

    subject_counts: Dict[str, Dict[str, int]] = {}
    totals = Counter()
    for subject in sorted(subject_rows):
        counters[subject]["eligible_rows"] = len(eligible[subject])
        subject_counts[subject] = {
            field: int(counters[subject][field])
            for field in (
                "source_rows",
                "answered_rows",
                "unanswered_rows",
                "evaluation_excluded_rows",
                "duplicate_rows",
                "conflicting_answer_rows",
                "eligible_rows",
            )
        }
        totals.update(subject_counts[subject])
    return eligible, {
        "dataset_split": dataset_split,
        "source_rows": totals["source_rows"],
        "answered_rows": totals["answered_rows"],
        "unanswered_rows": totals["unanswered_rows"],
        "evaluation_excluded_rows": totals["evaluation_excluded_rows"],
        "duplicate_rows": totals["duplicate_rows"],
        "conflicting_answer_rows": totals["conflicting_answer_rows"],
        "conflicting_answer_items": conflicting_answer_items,
        "eligible_rows": totals["eligible_rows"],
        "subject_counts": subject_counts,
    }


def build_focused_ceval_source_from_roles(
    train_subject_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    validation_subject_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    excluded_prompts: Set[str],
    excluded_items: Set[str],
    *,
    train_dataset_split: str,
    validation_dataset_split: str,
    validation_per_subject: int = 5,
    train_per_subject: int = 0,
    seed: int = DEFAULT_SEED,
    train_source_files: Optional[Mapping[str, Mapping[str, Any]]] = None,
    validation_source_files: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Build train and validation from separately frozen official split roles.

    This mode supports the intended C-Eval protocol where all clean ``val`` rows
    form the training source while the five labelled ``dev`` rows per subject are
    kept as an independent validation source.
    """

    if train_dataset_split == validation_dataset_split:
        raise ManifestError("显式 train/validation 角色必须来自不同 C-Eval split")
    if set(train_subject_rows) != set(validation_subject_rows):
        raise ManifestError("train/validation C-Eval subject 集合必须完全一致")
    if (
        isinstance(validation_per_subject, bool)
        or not isinstance(validation_per_subject, int)
        or validation_per_subject <= 0
    ):
        raise ManifestError("validation_per_subject 必须是大于 0 的整数")
    if (
        isinstance(train_per_subject, bool)
        or not isinstance(train_per_subject, int)
        or train_per_subject < 0
    ):
        raise ManifestError("train_per_subject 必须是大于等于 0 的整数")

    train_pool, train_source_report = _prepare_role_rows(
        train_subject_rows,
        excluded_prompts,
        excluded_items,
        dataset_split=train_dataset_split,
        source_files=train_source_files,
    )
    validation_pool, validation_source_report = _prepare_role_rows(
        validation_subject_rows,
        excluded_prompts,
        excluded_items,
        dataset_split=validation_dataset_split,
        source_files=validation_source_files,
    )

    train_rows: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
    subject_counts: Dict[str, Dict[str, int]] = {}
    for subject in sorted(train_subject_rows):
        training = sorted(
            train_pool[subject],
            key=lambda row: _stable_key(
                seed,
                "train",
                subject,
                row["event_id"],
                row["prompt_fingerprint"],
                row["item_fingerprint"],
            ),
        )
        validation = sorted(
            validation_pool[subject],
            key=lambda row: _stable_key(
                seed,
                "validation",
                subject,
                row["event_id"],
                row["prompt_fingerprint"],
                row["item_fingerprint"],
            ),
        )
        if train_per_subject > 0:
            if len(training) < train_per_subject:
                raise ManifestError(
                    f"C-Eval train subject {subject} 仅 {len(training)} 题，"
                    f"少于请求的 {train_per_subject} 题"
                )
            selected_train = training[:train_per_subject]
        else:
            selected_train = training
        if not selected_train:
            raise ManifestError(f"C-Eval train subject {subject} 没有可用样本")
        if len(validation) < validation_per_subject:
            raise ManifestError(
                f"C-Eval validation subject {subject} 仅 {len(validation)} 题，"
                f"少于请求的 {validation_per_subject} 题"
            )
        selected_validation = validation[:validation_per_subject]
        train_rows.extend(selected_train)
        validation_rows.extend(selected_validation)
        subject_counts[subject] = {
            "train_eligible_rows": len(training),
            "train_rows": len(selected_train),
            "train_unused_rows": len(training) - len(selected_train),
            "validation_eligible_rows": len(validation),
            "validation_rows": len(selected_validation),
            "validation_unused_rows": len(validation) - len(selected_validation),
        }

    train_prompt_ids = {str(row["prompt_fingerprint"]) for row in train_rows}
    validation_prompt_ids = {
        str(row["prompt_fingerprint"]) for row in validation_rows
    }
    train_item_ids = {str(row["item_fingerprint"]) for row in train_rows}
    validation_item_ids = {str(row["item_fingerprint"]) for row in validation_rows}
    prompt_overlap = train_prompt_ids & validation_prompt_ids
    item_overlap = train_item_ids & validation_item_ids
    if prompt_overlap or item_overlap:
        raise ManifestError(
            "显式 C-Eval train/validation split 存在重复题目: prompt={} item={}".format(
                len(prompt_overlap), len(item_overlap)
            )
        )
    excluded_prompt_overlap = (
        train_prompt_ids | validation_prompt_ids
    ) & excluded_prompts
    excluded_item_overlap = (train_item_ids | validation_item_ids) & excluded_items
    if excluded_prompt_overlap or excluded_item_overlap:
        raise ManifestError("C-Eval 蒸馏源与冻结 evaluation 存在题目泄漏")

    return train_rows, validation_rows, {
        "source_subject_count": len(train_subject_rows),
        "source_rows": train_source_report["source_rows"]
        + validation_source_report["source_rows"],
        "evaluation_excluded_rows": train_source_report[
            "evaluation_excluded_rows"
        ]
        + validation_source_report["evaluation_excluded_rows"],
        "duplicate_rows": train_source_report["duplicate_rows"]
        + validation_source_report["duplicate_rows"],
        "conflicting_answer_rows": train_source_report[
            "conflicting_answer_rows"
        ]
        + validation_source_report["conflicting_answer_rows"],
        "conflicting_answer_items": train_source_report[
            "conflicting_answer_items"
        ]
        + validation_source_report["conflicting_answer_items"],
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "subject_counts": subject_counts,
        "source_roles": {
            "train": train_source_report,
            "validation": validation_source_report,
        },
        "train_subject_counts": dict(
            sorted(Counter(str(row["subject"]) for row in train_rows).items())
        ),
        "validation_subject_counts": dict(
            sorted(
                Counter(str(row["subject"]) for row in validation_rows).items()
            )
        ),
        "train_validation_prompt_overlap": 0,
        "train_validation_item_overlap": 0,
        "excluded_prompt_overlap": len(excluded_prompt_overlap),
        "excluded_item_overlap": len(excluded_item_overlap),
        "train_event_ids_sha256": _digest(
            str(row["event_id"]) for row in train_rows
        ),
        "validation_event_ids_sha256": _digest(
            str(row["event_id"]) for row in validation_rows
        ),
        "train_prompt_fingerprints_sha256": _digest(train_prompt_ids),
        "validation_prompt_fingerprints_sha256": _digest(
            validation_prompt_ids
        ),
        "train_item_fingerprints_sha256": _digest(train_item_ids),
        "validation_item_fingerprints_sha256": _digest(validation_item_ids),
    }


def _frozen_validation_source_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    subjects: Set[str],
    validation_per_subject: int,
) -> List[Dict[str, Any]]:
    """Convert an immutable frozen C-Eval evaluation into KD source rows."""

    converted: List[Dict[str, Any]] = []
    event_ids: Set[str] = set()
    prompt_ids: Set[str] = set()
    item_ids: Set[str] = set()
    counts: Counter = Counter()
    for index, raw in enumerate(rows):
        source = raw.get("source")
        if not isinstance(source, Mapping):
            raise ManifestError(f"冻结 validation 第 {index} 行缺少 source")
        subject = str(source.get("config", "")).strip()
        if subject not in subjects:
            raise ManifestError(
                f"冻结 validation 第 {index} 行 subject 不在请求集合: {subject}"
            )
        prompt = str(raw.get("prompt", "")).strip()
        if not prompt:
            raise ManifestError(f"冻结 validation 第 {index} 行缺少 prompt")
        answer = str(raw.get("reference_answer", "")).strip().upper()
        if answer not in ANSWER_CHOICES:
            raise ManifestError(
                f"冻结 validation 第 {index} 行 reference_answer 必须是 A-D"
            )
        prompt_id = prompt_fingerprint(prompt)
        declared_prompt_id = str(raw.get("prompt_fingerprint", "")).strip()
        if declared_prompt_id and declared_prompt_id != prompt_id:
            raise ManifestError(f"冻结 validation 第 {index} 行 prompt 指纹不匹配")
        item_id = item_fingerprint(prompt)
        event_id = str(raw.get("sample_id", "")).strip()
        if not event_id:
            raise ManifestError(f"冻结 validation 第 {index} 行缺少 sample_id")
        if event_id in event_ids:
            raise ManifestError(f"冻结 validation event_id 重复: {event_id}")
        if prompt_id in prompt_ids or item_id in item_ids:
            raise ManifestError("冻结 validation 内部存在重复题目")
        event_ids.add(event_id)
        prompt_ids.add(prompt_id)
        item_ids.add(item_id)
        counts[subject] += 1
        converted.append(
            {
                "event_id": event_id,
                "category": CATEGORY,
                "prompt_format": "tokenizer_chat",
                "source_prompt": prompt,
                "prompt_fingerprint": prompt_id,
                "item_fingerprint": item_id,
                "subject": subject,
                "reference_answer": answer,
                "source": dict(source),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
            }
        )
    expected_counts = {subject: validation_per_subject for subject in subjects}
    if dict(counts) != expected_counts:
        raise ManifestError(
            "冻结 validation 必须每个 subject 恰好 {} 题；实测 {}".format(
                validation_per_subject, dict(sorted(counts.items()))
            )
        )
    return sorted(converted, key=lambda row: str(row["event_id"]))


def build_focused_ceval_source_from_training_splits(
    training_sources: Sequence[
        Tuple[
            str,
            Mapping[str, Sequence[Mapping[str, Any]]],
            Mapping[str, Mapping[str, Any]],
        ]
    ],
    frozen_validation_rows: Sequence[Mapping[str, Any]],
    formal_excluded_prompts: Set[str],
    formal_excluded_items: Set[str],
    *,
    validation_per_subject: int = 5,
    train_per_subject: int = 0,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Build training from several official splits and bind frozen validation.

    The frozen validation is excluded from every training split by both prompt
    and complete-item identity.  Formal evaluation exclusions are applied by the
    same two identities to training and are also asserted disjoint from the
    frozen validation.
    """

    if not training_sources:
        raise ManifestError("至少需要一个 C-Eval training split")
    split_names = [split for split, _rows, _files in training_sources]
    if len(split_names) != len(set(split_names)):
        raise ManifestError("C-Eval training split 不能重复声明")
    subject_sets = [set(rows) for _split, rows, _files in training_sources]
    subjects = subject_sets[0]
    if not subjects or any(current != subjects for current in subject_sets[1:]):
        raise ManifestError("所有 C-Eval training split 的 subject 集合必须一致")
    if (
        isinstance(validation_per_subject, bool)
        or not isinstance(validation_per_subject, int)
        or validation_per_subject <= 0
    ):
        raise ManifestError("validation_per_subject 必须是大于 0 的整数")
    if (
        isinstance(train_per_subject, bool)
        or not isinstance(train_per_subject, int)
        or train_per_subject < 0
    ):
        raise ManifestError("train_per_subject 必须是大于等于 0 的整数")

    validation_rows = _frozen_validation_source_rows(
        frozen_validation_rows,
        subjects=subjects,
        validation_per_subject=validation_per_subject,
    )
    validation_prompt_ids = {
        str(row["prompt_fingerprint"]) for row in validation_rows
    }
    validation_item_ids = {str(row["item_fingerprint"]) for row in validation_rows}
    if validation_prompt_ids & formal_excluded_prompts:
        raise ManifestError("冻结 validation 与 formal evaluation 存在 prompt 重叠")
    if validation_item_ids & formal_excluded_items:
        raise ManifestError("冻结 validation 与 formal evaluation 存在完整题目重叠")

    training_excluded_prompts = formal_excluded_prompts | validation_prompt_ids
    training_excluded_items = formal_excluded_items | validation_item_ids
    role_reports: Dict[str, Any] = {}
    candidates_by_item: Dict[str, List[Dict[str, Any]]] = {}
    for split, split_rows, split_files in training_sources:
        eligible, source_report = _prepare_role_rows(
            split_rows,
            training_excluded_prompts,
            training_excluded_items,
            dataset_split=split,
            source_files=split_files,
        )
        role_reports[split] = source_report
        for subject_rows in eligible.values():
            for row in subject_rows:
                candidates_by_item.setdefault(
                    str(row["item_fingerprint"]), []
                ).append(row)

    combined: Dict[str, List[Dict[str, Any]]] = {subject: [] for subject in subjects}
    cross_split_duplicate_rows = 0
    cross_split_conflicting_rows = 0
    cross_split_conflicting_items = 0
    for item_id in sorted(candidates_by_item):
        group = candidates_by_item[item_id]
        if len({str(row["reference_answer"]) for row in group}) > 1:
            cross_split_conflicting_items += 1
            cross_split_conflicting_rows += len(group)
            continue
        group.sort(key=lambda row: str(row["event_id"]))
        combined[str(group[0]["subject"])].append(group[0])
        cross_split_duplicate_rows += len(group) - 1

    train_rows: List[Dict[str, Any]] = []
    subject_counts: Dict[str, Dict[str, int]] = {}
    for subject in sorted(subjects):
        ordered = sorted(
            combined[subject],
            key=lambda row: _stable_key(
                seed,
                "train",
                subject,
                row["event_id"],
                row["prompt_fingerprint"],
                row["item_fingerprint"],
            ),
        )
        if train_per_subject > 0:
            if len(ordered) < train_per_subject:
                raise ManifestError(
                    f"C-Eval train subject {subject} 仅 {len(ordered)} 题，"
                    f"少于请求的 {train_per_subject} 题"
                )
            selected = ordered[:train_per_subject]
        else:
            selected = ordered
        if not selected:
            raise ManifestError(f"C-Eval train subject {subject} 没有可用样本")
        train_rows.extend(selected)
        subject_counts[subject] = {
            "train_eligible_rows": len(ordered),
            "train_rows": len(selected),
            "train_unused_rows": len(ordered) - len(selected),
            "validation_rows": validation_per_subject,
        }

    train_prompt_ids = {str(row["prompt_fingerprint"]) for row in train_rows}
    train_item_ids = {str(row["item_fingerprint"]) for row in train_rows}
    if train_prompt_ids & validation_prompt_ids or train_item_ids & validation_item_ids:
        raise ManifestError("C-Eval train 与冻结 validation 存在题目泄漏")
    if train_prompt_ids & formal_excluded_prompts:
        raise ManifestError("C-Eval train 与 formal evaluation 存在 prompt 泄漏")
    if train_item_ids & formal_excluded_items:
        raise ManifestError("C-Eval train 与 formal evaluation 存在完整题目泄漏")
    all_event_ids = [str(row["event_id"]) for row in train_rows + validation_rows]
    if len(all_event_ids) != len(set(all_event_ids)):
        raise ManifestError("C-Eval train/validation event_id 重复")

    source_rows = sum(int(report["source_rows"]) for report in role_reports.values())
    evaluation_excluded_rows = sum(
        int(report["evaluation_excluded_rows"]) for report in role_reports.values()
    )
    return train_rows, validation_rows, {
        "source_subject_count": len(subjects),
        "source_rows": source_rows,
        "evaluation_excluded_rows": evaluation_excluded_rows,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "subject_counts": subject_counts,
        "source_roles": role_reports,
        "cross_split_duplicate_rows": cross_split_duplicate_rows,
        "cross_split_conflicting_rows": cross_split_conflicting_rows,
        "cross_split_conflicting_items": cross_split_conflicting_items,
        "train_subject_counts": dict(
            sorted(Counter(str(row["subject"]) for row in train_rows).items())
        ),
        "validation_subject_counts": dict(
            sorted(Counter(str(row["subject"]) for row in validation_rows).items())
        ),
        "train_validation_prompt_overlap": 0,
        "train_validation_item_overlap": 0,
        "formal_prompt_overlap": 0,
        "formal_item_overlap": 0,
        "train_event_ids_sha256": _digest(
            str(row["event_id"]) for row in train_rows
        ),
        "validation_event_ids_sha256": _digest(
            str(row["event_id"]) for row in validation_rows
        ),
        "train_prompt_fingerprints_sha256": _digest(train_prompt_ids),
        "validation_prompt_fingerprints_sha256": _digest(validation_prompt_ids),
        "train_item_fingerprints_sha256": _digest(train_item_ids),
        "validation_item_fingerprints_sha256": _digest(validation_item_ids),
    }


def _write_jsonl_exclusive(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> None:
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
        raise ManifestError(f"拒绝覆盖已有蒸馏源: {path}") from exc


def _write_manifest_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as file_obj:
            json.dump(dict(value), file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            file_obj.write("\n")
    except FileExistsError as exc:
        raise ManifestError(f"拒绝覆盖已有蒸馏源: {path}") from exc


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="从本地 C-Eval Arrow 构建聚焦、无冻结评测泄漏的通用蒸馏源。"
    )
    parser.add_argument("--ceval_cache_root", required=True)
    parser.add_argument(
        "--dataset_split",
        choices=DATASET_SPLITS,
        action="append",
        help=(
            "作为训练来源的 C-Eval split；可重复声明。"
            "未声明时默认 val。"
        ),
    )
    parser.add_argument(
        "--validation_dataset_split",
        choices=DATASET_SPLITS,
        help=(
            "可选：从另一个官方 split 固定构建 validation。"
            "例如 dataset_split=val、validation_dataset_split=dev。"
        ),
    )
    parser.add_argument(
        "--subject",
        action="append",
        required=True,
        help="纳入的 C-Eval subject；必须至少声明一个，可重复使用本参数。",
    )
    parser.add_argument(
        "--frozen_evaluation_jsonl",
        action="append",
        required=True,
        help="必须排除的冻结 evaluation；可重复使用本参数。",
    )
    parser.add_argument(
        "--frozen_validation_jsonl",
        help=(
            "可选：直接绑定不可变的带答案 validation JSONL。"
            "提供后，它不参与训练，并按 prompt+完整题目双重排除。"
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--validation_per_subject", type=int, default=5)
    parser.add_argument("--train_per_subject", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    dataset_splits = args.dataset_split or [DEFAULT_DATASET_SPLIT]
    if len(dataset_splits) != len(set(dataset_splits)):
        raise ManifestError("dataset_split 不能重复声明")
    if args.frozen_validation_jsonl and args.validation_dataset_split:
        raise ManifestError(
            "frozen_validation_jsonl 与 validation_dataset_split 不能同时使用"
        )
    if len(dataset_splits) > 1 and not args.frozen_validation_jsonl:
        raise ManifestError(
            "多个 training split 必须同时提供 frozen_validation_jsonl"
        )

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ManifestError(f"拒绝覆盖非空蒸馏源目录: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    manifest_path = output_dir / "manifest.json"
    for path in (train_path, validation_path, manifest_path):
        if path.exists():
            raise ManifestError(f"拒绝覆盖已有蒸馏源: {path}")

    requested_subjects = [str(subject).strip() for subject in args.subject]
    if any(not subject for subject in requested_subjects):
        raise ManifestError("subject 不能为空")
    if len(set(requested_subjects)) != len(requested_subjects):
        raise ManifestError("subject 不能重复声明")

    evaluation_sources: List[Tuple[str, Sequence[Mapping[str, Any]]]] = []
    evaluation_artifacts: List[Dict[str, Any]] = []
    seen_evaluation_paths: Set[Path] = set()
    for index, raw_path in enumerate(args.frozen_evaluation_jsonl):
        path = Path(raw_path).resolve()
        if path in seen_evaluation_paths:
            raise ManifestError(f"冻结 evaluation 不能重复声明: {path}")
        seen_evaluation_paths.add(path)
        if not path.is_file():
            raise ManifestError(f"冻结 evaluation 不存在: {path}")
        rows = _read_jsonl(path)
        label = f"frozen_evaluation_{index}"
        evaluation_sources.append((label, rows))
        evaluation_artifacts.append(
            {
                "label": label,
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(rows),
            }
        )
    excluded_prompts, excluded_items, exclusion_counts = exclusion_identities(
        evaluation_sources
    )

    cache_root = Path(args.ceval_cache_root).resolve()
    selected_subjects = sorted(requested_subjects)
    loaded_training_sources: List[
        Tuple[
            str,
            Dict[str, Sequence[Mapping[str, Any]]],
            Dict[str, Mapping[str, Any]],
        ]
    ] = []
    for dataset_split in dataset_splits:
        all_subject_rows, all_source_files = load_ceval_rows(
            cache_root, dataset_split
        )
        unknown = sorted(set(requested_subjects) - set(all_subject_rows))
        if unknown:
            raise ManifestError(
                f"C-Eval {dataset_split} 缓存中不存在指定 subject: {unknown}"
            )
        loaded_training_sources.append(
            (
                dataset_split,
                {
                    subject: all_subject_rows[subject]
                    for subject in selected_subjects
                },
                {
                    subject: all_source_files[subject]
                    for subject in selected_subjects
                },
            )
        )
    primary_split, subject_rows, source_files = loaded_training_sources[0]

    validation_source_files: Optional[Dict[str, Dict[str, Any]]] = None
    frozen_validation_artifact: Optional[Dict[str, Any]] = None
    if args.frozen_validation_jsonl:
        frozen_validation_path = Path(args.frozen_validation_jsonl).resolve()
        if not frozen_validation_path.is_file():
            raise ManifestError(
                f"冻结 validation 不存在: {frozen_validation_path}"
            )
        frozen_validation_rows = _read_jsonl(frozen_validation_path)
        train_rows, validation_rows, selection = (
            build_focused_ceval_source_from_training_splits(
                loaded_training_sources,
                frozen_validation_rows,
                excluded_prompts,
                excluded_items,
                validation_per_subject=args.validation_per_subject,
                train_per_subject=args.train_per_subject,
                seed=args.seed,
            )
        )
        frozen_validation_artifact = {
            "path": str(frozen_validation_path),
            "sha256": sha256_file(frozen_validation_path),
            "rows": len(frozen_validation_rows),
            "used_for_training": False,
        }
    elif args.validation_dataset_split:
        all_validation_rows, all_validation_files = load_ceval_rows(
            cache_root, args.validation_dataset_split
        )
        missing_validation = sorted(
            set(selected_subjects) - set(all_validation_rows)
        )
        if missing_validation:
            raise ManifestError(
                f"C-Eval validation split 缺少指定 subject: {missing_validation}"
            )
        validation_subject_rows = {
            subject: all_validation_rows[subject] for subject in selected_subjects
        }
        validation_source_files = {
            subject: all_validation_files[subject] for subject in selected_subjects
        }
        train_rows, validation_rows, selection = (
            build_focused_ceval_source_from_roles(
                subject_rows,
                validation_subject_rows,
                excluded_prompts,
                excluded_items,
                train_dataset_split=primary_split,
                validation_dataset_split=args.validation_dataset_split,
                validation_per_subject=args.validation_per_subject,
                train_per_subject=args.train_per_subject,
                seed=args.seed,
                train_source_files=source_files,
                validation_source_files=validation_source_files,
            )
        )
    else:
        train_rows, validation_rows, selection = build_focused_ceval_source(
            subject_rows,
            excluded_prompts,
            excluded_items,
            dataset_split=primary_split,
            validation_per_subject=args.validation_per_subject,
            train_per_subject=args.train_per_subject,
            seed=args.seed,
            source_files=source_files,
        )
    _write_jsonl_exclusive(train_path, train_rows)
    _write_jsonl_exclusive(validation_path, validation_rows)

    arrow_manifest: List[Dict[str, Any]] = []
    for dataset_split, _rows, files in loaded_training_sources:
        arrow_manifest.extend(
            dict(files[subject], role=f"train_source:{dataset_split}")
            for subject in selected_subjects
        )
    if validation_source_files is not None:
        arrow_manifest.extend(
            dict(validation_source_files[subject], role="validation_source")
            for subject in selected_subjects
        )
    arrow_set_sha256 = hashlib.sha256(
        "\n".join(
            f"{row['role']}:{row['subject']}:{row['relative_path']}:{row['sha256']}"
            for row in arrow_manifest
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": "scene_independent_focused_ceval_distillation_source",
        "category": CATEGORY,
        "prompt_protocol": {
            "format": "tokenizer_chat",
            "system": SYSTEM_PROMPT,
            "assistant_target": "gold_reference_exactly_one_of_A_B_C_D",
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "train_splits": dataset_splits,
            "validation_split": args.validation_dataset_split,
            "cache_root": str(cache_root),
            "subject_filter": selected_subjects,
            "subject_count": len(selected_subjects),
            "arrow_files": arrow_manifest,
            "arrow_files_sha256": arrow_set_sha256,
        },
        "seed": args.seed,
        "selection_policy": (
            "frozen validation is copied exactly and excluded from every training "
            "split by prompt and complete-item identity; training rows are ordered "
            "within each subject by sha256(schema|seed|train|subject|event_id|"
            "prompt_fingerprint|item_fingerprint)"
            if frozen_validation_artifact is not None
            else
            "within each subject order=sha256(schema|seed|subject|event_id|"
            "prompt_fingerprint|item_fingerprint); reserve first N for validation; "
            "use requested count or all remaining for train"
        ),
        "validation_per_subject": args.validation_per_subject,
        "train_per_subject": args.train_per_subject,
        "training_or_tuning_use_allowed": True,
        "scene_specific_samples": 0,
        "evaluation_prompt_overlap": 0,
        "evaluation_item_overlap": 0,
        "evaluation_set_used_for_training": False,
        "frozen_evaluations": evaluation_artifacts,
        "frozen_validation": frozen_validation_artifact,
        "exclusion_unique_counts": exclusion_counts,
        "excluded_unique_prompt_count": len(excluded_prompts),
        "excluded_unique_item_count": len(excluded_items),
        "selection": selection,
        "artifacts": {
            "train": {
                "path": train_path.name,
                "sha256": sha256_file(train_path),
                "rows": len(train_rows),
            },
            "validation": {
                "path": validation_path.name,
                "sha256": sha256_file(validation_path),
                "rows": len(validation_rows),
            },
        },
    }
    _write_manifest_exclusive(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
