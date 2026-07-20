"""用途：用官方训练分片构建无冻结测试泄漏的第二轮通用蒸馏源数据。"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from edge_llm_factory.contracts import ManifestError, sha256_file, write_json_object
from edge_llm_factory.general_kd_eval import TEST_MARKER


MBPP_DATASET_ID = "google-research-datasets/mbpp"
MBPP_CONFIG = "full"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
MBPP_SPLIT_RANGES = {
    "train": (601, 974),
    "validation": (511, 600),
    "test": (11, 510),
    "prompt": (1, 10),
}
GENERIC_CATEGORIES = ("math", "natural_language_reasoning")
CODE_SYSTEM = (
    "Write a correct Python solution. Output only Python code without Markdown fences "
    "or explanation."
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError("{}:{} JSON 无效".format(path, line_number)) from exc
            if not isinstance(value, dict):
                raise ManifestError("{}:{} 必须是 JSON object".format(path, line_number))
            rows.append(value)
    if not rows:
        raise ManifestError("数据文件为空: {}".format(path))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def normalize_prompt(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _validated_messages(row: Mapping[str, Any]) -> List[Dict[str, str]]:
    raw = row.get("messages")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ManifestError("通用样本缺少 messages")
    messages = []
    for message in raw:
        if not isinstance(message, dict):
            raise ManifestError("messages 元素必须是 object")
        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            raise ManifestError("messages role/content 无效")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "assistant":
        raise ManifestError("最后一条消息必须是 assistant reference")
    return messages


def select_generic_rows(
    rows: Sequence[Mapping[str, Any]], evaluation_prompts: Sequence[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    eval_set = {normalize_prompt(value) for value in evaluation_prompts if value.strip()}
    selected: Dict[str, Dict[str, Any]] = {}
    skipped = Counter()
    for raw in rows:
        category = str(raw.get("category", ""))
        if category not in GENERIC_CATEGORIES:
            skipped["excluded_category"] += 1
            continue
        source_prompt = str(raw.get("source_prompt", "")).strip()
        normalized = normalize_prompt(source_prompt)
        if not normalized:
            raise ManifestError("通用样本缺少 source_prompt")
        if normalized in eval_set:
            raise ManifestError("通用源数据与冻结评测 prompt 重叠")
        event_id = str(raw.get("event_id") or raw.get("source_event_id") or "").strip()
        if not event_id:
            raise ManifestError("通用样本缺少 event_id/source_event_id")
        fingerprint = hashlib.sha256(
            (category + "\n" + normalized).encode("utf-8")
        ).hexdigest()
        if fingerprint in selected:
            skipped["duplicate_prompt"] += 1
            continue
        selected[fingerprint] = {
            "event_id": event_id,
            "category": category,
            "prompt_format": "tokenizer_chat",
            "source_prompt": source_prompt,
            "messages": _validated_messages(raw),
        }
    return list(selected.values()), dict(sorted(skipped.items()))


def validate_mbpp_split(split: str, task_ids: Sequence[int]) -> None:
    if split not in {"train", "validation"}:
        raise ManifestError("只允许使用 MBPP train/validation 构建蒸馏数据")
    lower, upper = MBPP_SPLIT_RANGES[split]
    expected = set(range(lower, upper + 1))
    actual = {int(value) for value in task_ids}
    if actual != expected or len(task_ids) != len(expected):
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise ManifestError(
            "MBPP {} 分片 task_id 不完整或越界: missing={}, extra={}".format(
                split, missing, extra
            )
        )


def build_mbpp_rows(
    items: Sequence[Mapping[str, Any]], split: str, evaluation_prompts: Sequence[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    task_ids = [int(item["task_id"]) for item in items]
    validate_mbpp_split(split, task_ids)
    eval_set = {normalize_prompt(value) for value in evaluation_prompts if value.strip()}
    rows = []
    evaluation_overlap_task_ids = []
    for item in items:
        task_id = int(item["task_id"])
        source_prompt = str(item.get("text") or item.get("prompt") or "").strip()
        code = str(item.get("code", "")).strip()
        tests = [str(value).strip() for value in item.get("test_list", []) if str(value).strip()]
        setup = str(item.get("test_setup_code", "")).strip()
        if not source_prompt or not code or not tests:
            raise ManifestError("MBPP task {} 缺少题目、参考代码或测试".format(task_id))
        if normalize_prompt(source_prompt) in eval_set:
            evaluation_overlap_task_ids.append(task_id)
            continue
        executable_checks = ([setup] if setup else []) + tests
        user_prompt = source_prompt + TEST_MARKER + "\n".join(executable_checks)
        rows.append(
            {
                "event_id": "mbpp_full_{}_{}".format(split, task_id),
                "category": "code",
                "prompt_format": "tokenizer_chat",
                "source_prompt": source_prompt,
                "messages": [
                    {"role": "system", "content": CODE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": code},
                ],
            }
        )
    return rows, {
        "evaluation_overlap": len(evaluation_overlap_task_ids),
        "evaluation_overlap_task_ids": evaluation_overlap_task_ids,
        "selected_rows": len(rows),
        "official_rows": len(items),
    }


def _prompt_keys(rows: Sequence[Mapping[str, Any]]) -> set:
    return {
        (str(row["category"]), normalize_prompt(str(row["source_prompt"])))
        for row in rows
    }


def _counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(row["category"]) for row in rows).items()))


def _digest(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="构建 GSM8K/C-Eval 已验收样本与 MBPP 官方分片组成的通用蒸馏源。"
    )
    parser.add_argument("--generic_train_jsonl", required=True)
    parser.add_argument("--generic_val_jsonl", required=True)
    parser.add_argument("--evaluation_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mbpp_revision", default=MBPP_REVISION)
    args = parser.parse_args(argv)

    generic_train_path = Path(args.generic_train_jsonl).resolve()
    generic_val_path = Path(args.generic_val_jsonl).resolve()
    evaluation_path = Path(args.evaluation_jsonl).resolve()
    for path in (generic_train_path, generic_val_path, evaluation_path):
        if not path.is_file():
            raise ManifestError("输入文件不存在: {}".format(path))
    output = Path(args.output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ManifestError("拒绝覆盖非空通用蒸馏源目录")
    output.mkdir(parents=True, exist_ok=True)

    evaluation_rows = _read_jsonl(evaluation_path)
    evaluation_prompts = [str(row.get("prompt", "")) for row in evaluation_rows]
    generic_train, train_skipped = select_generic_rows(
        _read_jsonl(generic_train_path), evaluation_prompts
    )
    generic_val, val_skipped = select_generic_rows(
        _read_jsonl(generic_val_path), evaluation_prompts
    )

    from datasets import load_dataset

    mbpp_train = load_dataset(
        MBPP_DATASET_ID,
        MBPP_CONFIG,
        revision=args.mbpp_revision,
        split="train",
    )
    mbpp_val = load_dataset(
        MBPP_DATASET_ID, MBPP_CONFIG, revision=args.mbpp_revision, split="validation"
    )
    code_train, mbpp_train_filter = build_mbpp_rows(list(mbpp_train), "train", evaluation_prompts)
    code_val, mbpp_val_filter = build_mbpp_rows(list(mbpp_val), "validation", evaluation_prompts)
    train_rows = generic_train + code_train
    val_rows = generic_val + code_val
    overlap = _prompt_keys(train_rows) & _prompt_keys(val_rows)
    if overlap:
        raise ManifestError("通用蒸馏源训练集与验证集 prompt 重叠: {}".format(len(overlap)))

    train_output = output / "train.jsonl"
    val_output = output / "val.jsonl"
    _write_jsonl(train_output, train_rows)
    _write_jsonl(val_output, val_rows)
    train_ids = [str(row["event_id"]).rsplit("_", 1)[-1] for row in code_train]
    val_ids = [str(row["event_id"]).rsplit("_", 1)[-1] for row in code_val]
    manifest = {
        "schema_version": "edge-llm-general-kd-source/v1",
        "task": "scene_independent_general_kd_source",
        "sources": {
            "generic_train": {
                "path": str(generic_train_path),
                "sha256": sha256_file(generic_train_path),
            },
            "generic_validation": {
                "path": str(generic_val_path),
                "sha256": sha256_file(generic_val_path),
            },
            "mbpp": {
                "dataset_id": MBPP_DATASET_ID,
                "config": MBPP_CONFIG,
                "revision": args.mbpp_revision,
                "train_task_id_range": list(MBPP_SPLIT_RANGES["train"]),
                "validation_task_id_range": list(MBPP_SPLIT_RANGES["validation"]),
                "test_task_id_range": list(MBPP_SPLIT_RANGES["test"]),
                "test_split_loaded": False,
                "official_train_task_id_sha256": _digest(str(value) for value in range(601, 975)),
                "official_validation_task_id_sha256": _digest(str(value) for value in range(511, 601)),
                "selected_train_task_id_sha256": _digest(train_ids),
                "selected_validation_task_id_sha256": _digest(val_ids),
                "filter_report": {
                    "train": mbpp_train_filter,
                    "validation": mbpp_val_filter,
                },
            },
            "frozen_evaluation": {
                "path": str(evaluation_path),
                "sha256": sha256_file(evaluation_path),
                "used_for_training": False,
            },
        },
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_category_counts": _counts(train_rows),
        "validation_category_counts": _counts(val_rows),
        "skipped_generic_rows": {"train": train_skipped, "validation": val_skipped},
        "train_validation_prompt_overlap": 0,
        "evaluation_prompt_overlap": 0,
        "scene_specific_samples": 0,
        "artifacts": {
            "train": {"path": train_output.name, "sha256": sha256_file(train_output)},
            "validation": {"path": val_output.name, "sha256": sha256_file(val_output)},
        },
    }
    write_json_object(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
