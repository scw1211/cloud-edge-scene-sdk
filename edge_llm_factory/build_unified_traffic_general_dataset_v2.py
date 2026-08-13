"""构建 U2 统一训练数据，并保持正式测试集内容隔离。

U2 与 v1 的关键差异：

* 交通训练/验证集原样保留为 3600/1200 条；
* GSM8K 仍只加载官方 ``train``。U1 已使用的 400 条开发题只用于
  训练期验证；再从 U1 从未使用的 3473 条中冻结新的 400 条晋级题，
  训练集由 U1 原有 3600 条和余下 3073 条组成；
* LogiQA2.0 中文训练集使用清洗后的全部 12193 条。U1 已使用的 400 条
  开发题只用于训练期验证；从官方 dev 的其余题目中另冻 400 条晋级题；
* LogiQA 训练选项会先去掉自带的 ``A.``/``B.`` 等前缀，再进行确定性
  置换并同步重标答案，避免模型利用固定答案位置；
* 交通 test 和 LogiQA test 只允许执行 SHA-256 与 stat，不会交给任何
  JSON/数据集解析器。GSM8K test split 从不加载。

构建结果使用 ``edge-llm-unified-traffic-general/v2`` 清单，并为所有输入、
派生集合和输出记录 SHA-256。清单自身的 SHA 写入 ``manifest.sha256``，
避免自引用哈希。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from edge_llm_factory.contracts import ManifestError, sha256_file, write_json_object


SCHEMA_VERSION = "edge-llm-unified-traffic-general/v2"
U1_SCHEMA_VERSION = "edge-llm-unified-traffic-general/v1"
U1_MANIFEST_SHA256 = (
    "8c33ab4674ea48fb82ff947cf2030a7f36892b06c514ebf6e131d8fe0faacb7f"
)
TRAFFIC_CATEGORY = "traffic_action"
MATH_CATEGORY = "math"
LOGIC_CATEGORY = "natural_language_reasoning"
TRAFFIC_PATTERN = re.compile(r"^[0-9]{16}$")
TRAFFIC_LABELS = set("ABCDEF")
CHOICE_LABELS = "ABCD"

TRAFFIC_TRAIN_ROWS = 3600
TRAFFIC_VALIDATION_ROWS = 1200
TRAFFIC_TRAIN_SHA256 = (
    "38124e1a7ee65f80b2e5384a5f9389f818298d0f09882b418f25e0051586eaa9"
)
TRAFFIC_VALIDATION_SHA256 = (
    "8a18e17805d8dc1ec89e801fb6df9aac3d77ee1c390419ea26d6eaf7c08e1d4f"
)
TRAFFIC_TEST_SHA256 = (
    "604ebf54458874b744051c5d2b7bd340771138e6de1347a01ff91ea806ce55ac"
)
MATH_U1_TRAIN_ROWS = 3600
MATH_U1_DEV_ROWS = 400
MATH_UNUSED_ROWS = 3473
MATH_TRAIN_ROWS = 6673
MATH_DEV_ROWS = 400
LOGIC_CLEAN_TRAIN_ROWS = 12193
LOGIC_CLEAN_DEV_ROWS = 1494
LOGIC_U1_DEV_ROWS = 400
LOGIC_DEV_ROWS = 400
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

LOGIQA_TRAIN_SHA256 = (
    "d87a15811cda64cb021d43cb9bc1d282424a8dfb8e35e6a7f6d6a0b36b38a54e"
)
LOGIQA_DEV_SHA256 = (
    "a72a23160c9e12e15ea8c13e57af5032a7c37157573ebdd7e7c8e0ad34aef780"
)
LOGIQA_TEST_SHA256 = (
    "7a8db83ccb3ebdc8d5b3886fd0ad9346c7e565722d2d592987b24dd57f251853"
)
GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_TRAIN_ROWS = 7473
GSM8K_TRAIN_FINGERPRINT = "c6f812ae33c9159d"

MATH_SYSTEM = (
    "[TASK:MATH]\nSolve the grade-school math problem accurately. "
    "Keep the reasoning concise and end with exactly: FINAL: <number>"
)
LOGIC_SYSTEM = (
    "[任务:中文逻辑选择]\n根据材料进行逻辑推理。只输出 A、B、C 或 D，不要解释。"
)

_OPTION_PREFIX = re.compile(
    r"^\s*(?:(?:[A-Da-dＡ-Ｄａ-ｄ])\s*[.．、:：)）]|[（(]\s*[A-Da-dＡ-Ｄａ-ｄ]\s*[)）])\s*"
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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
    if not rows:
        raise ManifestError(f"数据文件为空: {path}")
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(
                json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def _canonical_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fingerprint_set_sha256(values: Iterable[str]) -> str:
    return _sha_text("\n".join(sorted(str(value) for value in values)) + "\n")


def _stable_order(
    rows: Iterable[Mapping[str, Any]], seed: int, namespace: str
) -> List[Dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: _sha_text(
            f"{seed}\n{namespace}\n{row.get('prompt_fingerprint', '')}"
        ),
    )


def _messages(row: Mapping[str, Any]) -> List[Dict[str, str]]:
    raw = row.get("messages")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ManifestError("样本缺少输入消息和 assistant target")
    messages: List[Dict[str, str]] = []
    for message in raw:
        if not isinstance(message, Mapping):
            raise ManifestError("messages 元素必须是 object")
        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            raise ManifestError("messages role/content 无效")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "assistant":
        raise ManifestError("最后一条消息必须是 assistant target")
    return messages


def _traffic_rows(path: Path, split: str) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen = set()
    for index, raw in enumerate(_read_jsonl(path), start=1):
        messages = _messages(raw)
        if len(messages) != 2 or messages[0]["role"] != "user":
            raise ManifestError(f"交通 {split} 第 {index} 条不是单 user/assistant")
        prompt = messages[0]["content"]
        target = messages[1]["content"]
        if TRAFFIC_PATTERN.fullmatch(prompt) is None:
            raise ManifestError(f"交通 {split} 第 {index} 条不是 16 位编码")
        if target not in TRAFFIC_LABELS:
            raise ManifestError(f"交通 {split} 第 {index} 条 target 不是 A-F")
        source_event_id = str(raw.get("event_id", "")).strip()
        if not source_event_id or source_event_id in seen:
            raise ManifestError(
                f"交通 {split} event_id 缺失或重复: {source_event_id}"
            )
        seen.add(source_event_id)
        output.append(
            {
                **dict(raw),
                "event_id": f"traffic:{split}:{source_event_id}",
                "source_event_id": source_event_id,
                "source_split": split,
                "category": TRAFFIC_CATEGORY,
                "prompt_format": "raw_task",
                "source_prompt": prompt,
                "prompt_fingerprint": _sha_text(f"{TRAFFIC_CATEGORY}\n{prompt}"),
                "supervision_source": "frozen_future_observed_traffic_policy",
                "messages": messages,
            }
        )
    expected = TRAFFIC_TRAIN_ROWS if split == "train" else TRAFFIC_VALIDATION_ROWS
    if len(output) != expected:
        raise ManifestError(f"交通 {split} 数量漂移: {len(output)} != {expected}")
    return output


def _clean_gsm8k_answer(raw_answer: str) -> Tuple[str, str]:
    if "####" not in raw_answer:
        raise ManifestError("GSM8K answer 缺少 #### 最终答案")
    reasoning, final = raw_answer.rsplit("####", 1)
    reasoning = re.sub(r"<<[^<>]*>>", "", reasoning).strip()
    final = final.strip().replace(",", "")
    if not final:
        raise ManifestError("GSM8K 最终答案为空")
    return reasoning, final


def _load_gsm8k_train() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    # 避免仓库根目录遮蔽 Hugging Face datasets 包。
    import sys

    project_root = Path(__file__).resolve().parents[1]
    sys.path[:] = [
        value
        for value in sys.path
        if not value or Path(value).resolve() != project_root.resolve()
    ]
    from datasets import load_dataset

    dataset = load_dataset(GSM8K_DATASET_ID, GSM8K_CONFIG, split="train")
    if len(dataset) != GSM8K_TRAIN_ROWS:
        raise ManifestError(f"GSM8K train 数量漂移: {len(dataset)}")
    dataset_fingerprint = str(getattr(dataset, "_fingerprint", ""))
    if dataset_fingerprint != GSM8K_TRAIN_FINGERPRINT:
        raise ManifestError(f"GSM8K train fingerprint 漂移: {dataset_fingerprint}")
    unique: Dict[str, Dict[str, Any]] = {}
    duplicates = 0
    canonical_source_rows: List[str] = []
    for index, item in enumerate(dataset):
        question = str(item["question"]).strip()
        raw_answer = str(item["answer"])
        reasoning, final = _clean_gsm8k_answer(raw_answer)
        content_fingerprint = _sha_text(_canonical_text(question))
        prompt_fingerprint = _sha_text(f"{MATH_CATEGORY}\n{content_fingerprint}")
        if prompt_fingerprint in unique:
            duplicates += 1
            continue
        unique[prompt_fingerprint] = {
            "source_index": index,
            "question": question,
            "reasoning": reasoning,
            "final": final,
            "content_fingerprint": content_fingerprint,
            "prompt_fingerprint": prompt_fingerprint,
        }
        canonical_source_rows.append(
            json.dumps(
                {"question": question, "answer": raw_answer},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    if len(unique) != GSM8K_TRAIN_ROWS:
        raise ManifestError(f"GSM8K 去重后数量漂移: {len(unique)}")
    return unique, {
        "dataset_id": GSM8K_DATASET_ID,
        "config": GSM8K_CONFIG,
        "split_loaded": "train",
        "test_split_loaded": False,
        "official_train_rows": len(dataset),
        "dataset_fingerprint": dataset_fingerprint,
        "unique_questions": len(unique),
        "duplicates_removed": duplicates,
        "canonical_source_rows_sha256": _sha_text(
            "\n".join(canonical_source_rows) + "\n"
        ),
    }


def _extract_u1_general_inventory(
    u1_dataset_dir: Path,
) -> Tuple[Dict[str, set[str]], Dict[str, set[str]], Dict[str, Any]]:
    manifest_path = u1_dataset_dir / "manifest.json"
    train_path = u1_dataset_dir / "train.jsonl"
    validation_path = u1_dataset_dir / "val.jsonl"
    for path in (manifest_path, train_path, validation_path):
        if not path.is_file():
            raise ManifestError(f"U1 数据文件不存在: {path}")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != U1_MANIFEST_SHA256:
        raise ManifestError(f"U1 manifest SHA-256 漂移: {manifest_sha}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != U1_SCHEMA_VERSION:
        raise ManifestError("U1 manifest schema 不匹配")
    expected_train_sha = str(manifest["artifacts"]["train"]["sha256"])
    expected_val_sha = str(manifest["artifacts"]["validation"]["sha256"])
    if sha256_file(train_path) != expected_train_sha:
        raise ManifestError("U1 train SHA-256 与 manifest 不一致")
    if sha256_file(validation_path) != expected_val_sha:
        raise ManifestError("U1 val SHA-256 与 manifest 不一致")

    def collect(path: Path) -> Dict[str, set[str]]:
        values = {MATH_CATEGORY: set(), LOGIC_CATEGORY: set()}
        for row in _read_jsonl(path):
            category = str(row.get("category", ""))
            if category not in values:
                continue
            fingerprint = str(row.get("prompt_fingerprint", ""))
            if len(fingerprint) != 64:
                raise ManifestError(f"U1 {category} 缺少 prompt_fingerprint")
            if fingerprint in values[category]:
                raise ManifestError(f"U1 {category} prompt_fingerprint 重复")
            values[category].add(fingerprint)
        return values

    train = collect(train_path)
    validation = collect(validation_path)
    expected_counts = {
        ("train", MATH_CATEGORY): MATH_U1_TRAIN_ROWS,
        ("train", LOGIC_CATEGORY): 3600,
        ("validation", MATH_CATEGORY): MATH_U1_DEV_ROWS,
        ("validation", LOGIC_CATEGORY): LOGIC_U1_DEV_ROWS,
    }
    for (split, category), expected in expected_counts.items():
        actual = len((train if split == "train" else validation)[category])
        if actual != expected:
            raise ManifestError(
                f"U1 {split}/{category} 数量漂移: {actual} != {expected}"
            )
    for category in (MATH_CATEGORY, LOGIC_CATEGORY):
        if train[category] & validation[category]:
            raise ManifestError(f"U1 {category} train/validation fingerprint 重叠")
    report = {
        "dataset_dir": str(u1_dataset_dir),
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "train": {
            "path": str(train_path),
            "sha256": expected_train_sha,
            "math_fingerprint_sha256": _fingerprint_set_sha256(train[MATH_CATEGORY]),
            "logic_fingerprint_sha256": _fingerprint_set_sha256(train[LOGIC_CATEGORY]),
        },
        "validation": {
            "path": str(validation_path),
            "sha256": expected_val_sha,
            "math_fingerprint_sha256": _fingerprint_set_sha256(
                validation[MATH_CATEGORY]
            ),
            "logic_fingerprint_sha256": _fingerprint_set_sha256(
                validation[LOGIC_CATEGORY]
            ),
        },
    }
    return train, validation, report


def _load_spent_u1_general_validation(
    u1_dataset_dir: Path,
    expected_fingerprints: Mapping[str, set[str]],
) -> List[Dict[str, Any]]:
    """Load the already-used U1 general dev rows for training-time validation.

    These rows are deliberately not reused as the U2 promotion set.  Keeping
    them in ``val.jsonl`` lets checkpoint selection observe all three tasks,
    while the newly frozen 400+400 promotion rows remain unseen by training.
    """

    rows = [
        dict(row)
        for row in _read_jsonl(u1_dataset_dir / "val.jsonl")
        if str(row.get("category", "")) in {MATH_CATEGORY, LOGIC_CATEGORY}
    ]
    actual = {
        category: {
            str(row.get("prompt_fingerprint", ""))
            for row in rows
            if row.get("category") == category
        }
        for category in (MATH_CATEGORY, LOGIC_CATEGORY)
    }
    if actual != {
        category: set(expected_fingerprints[category])
        for category in (MATH_CATEGORY, LOGIC_CATEGORY)
    }:
        raise ManifestError("U1 已使用通用开发集与冻结 fingerprint 不一致")
    for row in rows:
        row["u2_role"] = "spent_u1_development_for_training_validation"
    return rows


def _select_math_inventory(
    official: Mapping[str, Mapping[str, Any]],
    u1_train_fingerprints: set[str],
    u1_dev_fingerprints: set[str],
    seed: int,
    *,
    expected_u1_train: int = MATH_U1_TRAIN_ROWS,
    expected_u1_dev: int = MATH_U1_DEV_ROWS,
    expected_unused: int = MATH_UNUSED_ROWS,
    new_dev_count: int = MATH_DEV_ROWS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    official_keys = set(official)
    if len(u1_train_fingerprints) != expected_u1_train:
        raise ManifestError("U1 math train 数量不符合冻结合同")
    if len(u1_dev_fingerprints) != expected_u1_dev:
        raise ManifestError("U1 math dev 数量不符合冻结合同")
    if u1_train_fingerprints & u1_dev_fingerprints:
        raise ManifestError("U1 math train/dev 重叠")
    missing = (u1_train_fingerprints | u1_dev_fingerprints) - official_keys
    if missing:
        raise ManifestError(f"U1 math 有 {len(missing)} 条不属于锁定 GSM8K train")
    unused = official_keys - u1_train_fingerprints - u1_dev_fingerprints
    if len(unused) != expected_unused:
        raise ManifestError(f"GSM8K U1 未使用数量漂移: {len(unused)}")
    ordered_unused = sorted(
        unused, key=lambda value: _sha_text(f"{seed}\nmath-u2-dev\n{value}")
    )
    new_dev_keys = set(ordered_unused[:new_dev_count])
    new_train_keys = set(u1_train_fingerprints) | set(ordered_unused[new_dev_count:])
    if new_train_keys & new_dev_keys or u1_dev_fingerprints & (
        new_train_keys | new_dev_keys
    ):
        raise ManifestError("U2 math 集合隔离失败")
    train = _stable_order(
        (official[key] for key in new_train_keys), seed + 1, "math-u2-train"
    )
    dev = _stable_order(
        (official[key] for key in new_dev_keys), seed + 2, "math-u2-dev"
    )
    return train, dev, {
        "u1_train_rows_reused": len(u1_train_fingerprints),
        "u1_dev_rows_excluded": len(u1_dev_fingerprints),
        "previously_unused_rows": len(unused),
        "previously_unused_rows_added_to_train": len(unused) - new_dev_count,
        "new_dev_rows": len(new_dev_keys),
        "u2_train_rows": len(new_train_keys),
        "u2_train_fingerprint_sha256": _fingerprint_set_sha256(new_train_keys),
        "u2_dev_fingerprint_sha256": _fingerprint_set_sha256(new_dev_keys),
        "excluded_u1_dev_fingerprint_sha256": _fingerprint_set_sha256(
            u1_dev_fingerprints
        ),
        "train_dev_overlap": 0,
        "u1_dev_overlap": 0,
    }


def _convert_math(item: Mapping[str, Any], split: str) -> Dict[str, Any]:
    source_id = f"gsm8k_train_{item['source_index']}"
    answer = f"{item['reasoning']}\nFINAL: {item['final']}".strip()
    return {
        "event_id": f"math:{split}:{source_id}",
        "source_event_id": source_id,
        "source_split": split,
        "category": MATH_CATEGORY,
        "prompt_format": "tokenizer_chat",
        "source_prompt": str(item["question"]),
        "prompt_fingerprint": str(item["prompt_fingerprint"]),
        "content_fingerprint": str(item["content_fingerprint"]),
        "supervision_source": "official_gsm8k_human_reasoning",
        "reference_answer": str(item["final"]),
        "messages": [
            {"role": "system", "content": MATH_SYSTEM},
            {"role": "user", "content": str(item["question"])},
            {"role": "assistant", "content": answer},
        ],
    }


def _logiqa_content_fingerprint(row: Mapping[str, Any]) -> str:
    payload = {
        "text": _canonical_text(str(row.get("text", ""))),
        "question": _canonical_text(str(row.get("question", ""))),
        # 身份指纹必须与 U1 一致，因此这里使用官方原始选项；前缀清理和置换
        # 是训练表示变换，不改变样本身份。
        "options": [_canonical_text(str(value)) for value in row.get("options", [])],
    }
    if (
        (not payload["text"] and not payload["question"])
        or len(payload["options"]) != 4
        or any(not option for option in payload["options"])
    ):
        raise ManifestError("LogiQA2.0 样本缺少题干或四个有效选项")
    return _sha_text(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    )


def _logiqa_groups(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], int]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    invalid_rows = 0
    for raw in rows:
        answer = raw.get("answer")
        if isinstance(answer, bool) or not isinstance(answer, int) or answer not in range(4):
            raise ManifestError("LogiQA2.0 answer 必须是 0..3")
        try:
            fingerprint = _logiqa_content_fingerprint(raw)
        except ManifestError:
            invalid_rows += 1
            continue
        groups[fingerprint].append(dict(raw))
    return groups, invalid_rows


def _load_clean_logiqa(
    train_path: Path, dev_path: Path
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if sha256_file(train_path) != LOGIQA_TRAIN_SHA256:
        raise ManifestError("LogiQA2.0 train_zh SHA-256 不匹配")
    if sha256_file(dev_path) != LOGIQA_DEV_SHA256:
        raise ManifestError("LogiQA2.0 dev_zh SHA-256 不匹配")
    raw_train = _read_jsonl(train_path)
    raw_dev = _read_jsonl(dev_path)
    train_groups, invalid_train_rows = _logiqa_groups(raw_train)
    dev_groups, invalid_dev_rows = _logiqa_groups(raw_dev)
    clean_train: Dict[str, Dict[str, Any]] = {}
    conflicting_train_groups = 0
    for fingerprint, group in train_groups.items():
        labels = {int(row["answer"]) for row in group}
        if len(labels) != 1:
            conflicting_train_groups += 1
            continue
        clean_train[fingerprint] = group[0]
    clean_dev: Dict[str, Dict[str, Any]] = {}
    conflicting_dev_groups = 0
    train_dev_removed = 0
    for fingerprint, group in dev_groups.items():
        if fingerprint in train_groups:
            train_dev_removed += 1
            continue
        labels = {int(row["answer"]) for row in group}
        if len(labels) != 1:
            conflicting_dev_groups += 1
            continue
        clean_dev[fingerprint] = group[0]
    if set(clean_train) & set(clean_dev):
        raise ManifestError("LogiQA2.0 清洗后 train/dev 仍有内容重叠")
    if len(clean_train) != LOGIC_CLEAN_TRAIN_ROWS or len(clean_dev) != LOGIC_CLEAN_DEV_ROWS:
        raise ManifestError(
            f"LogiQA2.0 清洗数量漂移: train={len(clean_train)}, dev={len(clean_dev)}"
        )
    return clean_train, clean_dev, {
        "dataset_id": "LogiQA2.0-zh",
        "source_commit": "955e1d3df6c59d9bfb44d9913da1e1a27ec14e18",
        "license": "CC BY-NC-SA 4.0",
        "raw_train_rows": len(raw_train),
        "raw_dev_rows": len(raw_dev),
        "clean_train_rows": len(clean_train),
        "clean_dev_rows": len(clean_dev),
        "invalid_train_rows_removed": invalid_train_rows,
        "invalid_dev_rows_removed": invalid_dev_rows,
        "conflicting_train_groups_removed": conflicting_train_groups,
        "conflicting_dev_groups_removed": conflicting_dev_groups,
        "train_dev_groups_removed": train_dev_removed,
        "clean_train_fingerprint_sha256": _fingerprint_set_sha256(clean_train),
        "clean_dev_fingerprint_sha256": _fingerprint_set_sha256(clean_dev),
        "train_dev_overlap_after_cleaning": 0,
        "test_split_loaded": False,
    }


def _strip_option_prefix(value: str) -> Tuple[str, bool]:
    original = str(value).strip()
    cleaned = _OPTION_PREFIX.sub("", original, count=1).strip()
    if not cleaned:
        # A small number of official rows use a literal option such as ``A.``.
        # In that case the token is the option content, not a duplicated label;
        # preserving it is safer than deleting the row or inventing text.
        return original, False
    return cleaned, cleaned != original


def _deterministic_option_permutation(
    options: Sequence[str], answer_index: int, fingerprint: str, seed: int
) -> Tuple[List[str], int, List[int], int]:
    if len(options) != 4 or answer_index not in range(4):
        raise ManifestError("LogiQA2.0 置换要求四个选项和 0..3 答案")
    cleaned: List[str] = []
    stripped_count = 0
    for option in options:
        value, stripped = _strip_option_prefix(str(option))
        cleaned.append(value)
        stripped_count += int(stripped)
    order = sorted(
        range(4),
        key=lambda index: _sha_text(
            f"{seed}\nlogiqa-option-permutation\n{fingerprint}\n{index}"
        ),
    )
    permuted = [cleaned[index] for index in order]
    new_answer_index = order.index(answer_index)
    if permuted[new_answer_index] != cleaned[answer_index]:
        raise ManifestError("LogiQA2.0 置换后答案重标失败")
    return permuted, new_answer_index, order, stripped_count


def _select_logic_inventory(
    clean_train: Mapping[str, Mapping[str, Any]],
    clean_dev: Mapping[str, Mapping[str, Any]],
    u1_dev_fingerprints: set[str],
    seed: int,
    *,
    new_dev_count: int = LOGIC_DEV_ROWS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if len(u1_dev_fingerprints) != LOGIC_U1_DEV_ROWS:
        raise ManifestError("U1 logic dev 数量不符合冻结合同")
    clean_dev_prompt_to_content = {
        _sha_text(f"{LOGIC_CATEGORY}\n{fingerprint}"): fingerprint
        for fingerprint in clean_dev
    }
    missing = u1_dev_fingerprints - set(clean_dev_prompt_to_content)
    if missing:
        raise ManifestError(f"U1 logic dev 有 {len(missing)} 条不属于锁定官方 dev")
    remaining_prompt_fingerprints = set(clean_dev_prompt_to_content) - u1_dev_fingerprints
    ordered_remaining = sorted(
        remaining_prompt_fingerprints,
        key=lambda value: _sha_text(f"{seed}\nlogic-u2-dev\n{value}"),
    )
    new_dev_prompt_fingerprints = set(ordered_remaining[:new_dev_count])
    dev_rows = [
        dict(clean_dev[clean_dev_prompt_to_content[value]])
        for value in ordered_remaining[:new_dev_count]
    ]
    train_rows = [dict(value) for value in clean_train.values()]
    train_prompt_fingerprints = {
        _sha_text(f"{LOGIC_CATEGORY}\n{fingerprint}") for fingerprint in clean_train
    }
    if train_prompt_fingerprints & new_dev_prompt_fingerprints:
        raise ManifestError("U2 logic train/dev 重叠")
    if u1_dev_fingerprints & (
        train_prompt_fingerprints | new_dev_prompt_fingerprints
    ):
        raise ManifestError("U1 logic dev 未被完全排除")
    train_rows = _stable_order(
        (
            {
                **row,
                "prompt_fingerprint": _sha_text(
                    f"{LOGIC_CATEGORY}\n{_logiqa_content_fingerprint(row)}"
                ),
            }
            for row in train_rows
        ),
        seed + 3,
        "logic-u2-train",
    )
    dev_rows = _stable_order(
        (
            {
                **row,
                "prompt_fingerprint": _sha_text(
                    f"{LOGIC_CATEGORY}\n{_logiqa_content_fingerprint(row)}"
                ),
            }
            for row in dev_rows
        ),
        seed + 4,
        "logic-u2-dev",
    )
    return train_rows, dev_rows, {
        "u2_train_rows": len(train_rows),
        "u1_dev_rows_excluded": len(u1_dev_fingerprints),
        "remaining_official_dev_rows": len(remaining_prompt_fingerprints),
        "new_dev_rows": len(dev_rows),
        "u2_train_fingerprint_sha256": _fingerprint_set_sha256(
            train_prompt_fingerprints
        ),
        "u2_dev_fingerprint_sha256": _fingerprint_set_sha256(
            new_dev_prompt_fingerprints
        ),
        "excluded_u1_dev_fingerprint_sha256": _fingerprint_set_sha256(
            u1_dev_fingerprints
        ),
        "train_dev_overlap": 0,
        "u1_dev_overlap": 0,
    }


def _convert_logic(
    item: Mapping[str, Any], split: str, seed: int, permute: bool
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    content_fingerprint = _logiqa_content_fingerprint(item)
    prompt_fingerprint = _sha_text(f"{LOGIC_CATEGORY}\n{content_fingerprint}")
    original_answer = int(item["answer"])
    if permute:
        options, answer_index, order, stripped_count = _deterministic_option_permutation(
            [str(value) for value in item["options"]],
            original_answer,
            content_fingerprint,
            seed,
        )
    else:
        options = []
        stripped_count = 0
        for raw_option in item["options"]:
            option, stripped = _strip_option_prefix(str(raw_option))
            options.append(option)
            stripped_count += int(stripped)
        answer_index = original_answer
        order = list(range(4))
    answer = CHOICE_LABELS[answer_index]
    prompt = (
        f"材料：{str(item['text']).strip()}\n"
        f"问题：{str(item['question']).strip()}\n"
        "选项：\n"
        + "\n".join(
            f"{label}. {value}" for label, value in zip(CHOICE_LABELS, options)
        )
    )
    source_id = str(item.get("example_id", content_fingerprint[:16]))
    row = {
        "event_id": f"logic:{split}:{source_id}:{content_fingerprint[:12]}",
        "source_event_id": source_id,
        "source_split": split,
        "category": LOGIC_CATEGORY,
        "prompt_format": "tokenizer_chat",
        "source_prompt": prompt,
        "prompt_fingerprint": prompt_fingerprint,
        "content_fingerprint": content_fingerprint,
        "supervision_source": "official_logiqa2_human_verified",
        "reference_answer": answer,
        "option_permutation": order,
        "original_reference_answer": CHOICE_LABELS[original_answer],
        "messages": [
            {"role": "system", "content": LOGIC_SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }
    return row, {
        "prefixes_stripped": stripped_count,
        "original_label": CHOICE_LABELS[original_answer],
        "new_label": answer,
        "permuted": order != list(range(4)),
    }


def _file_attestation(path: Path, expected_sha256: str | None = None) -> Dict[str, Any]:
    """只做哈希和 stat；不得把 test 路径传给内容解析器。"""
    if not path.is_file():
        raise ManifestError(f"输入文件不存在: {path}")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ManifestError(f"测试文件 SHA-256 漂移: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": stat.st_size,
        "content_loaded": False,
        "access_mode": "sha256_and_stat_only",
    }


def _validate_traffic_input_identity(paths: Mapping[str, Path]) -> None:
    """Lock all three traffic splits before any training split is parsed."""

    keys = ("traffic_train", "traffic_validation", "traffic_test")
    resolved = [paths[key].resolve() for key in keys]
    if len(set(resolved)) != len(resolved):
        raise ManifestError("交通 train/validation/test 必须是三个不同文件")
    expected = {
        "traffic_train": TRAFFIC_TRAIN_SHA256,
        "traffic_validation": TRAFFIC_VALIDATION_SHA256,
        "traffic_test": TRAFFIC_TEST_SHA256,
    }
    observed: Dict[str, str] = {}
    for key in keys:
        path = paths[key]
        if not path.is_file():
            raise ManifestError(f"输入文件不存在: {path}")
        observed[key] = sha256_file(path)
        if observed[key] != expected[key]:
            raise ManifestError(f"锁定交通数据 SHA-256 漂移: {key}")
    if len(set(observed.values())) != len(observed):
        raise ManifestError("交通 train/validation/test 内容不能相同")


def _category_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(row["category"]) for row in rows).items()))


def _evaluation_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "sample_id": str(row["event_id"]),
            "category": str(row["category"]),
            "system_prompt": str(row["messages"][0]["content"]),
            "prompt": str(row["source_prompt"]),
            "reference_answer": str(row["reference_answer"]),
            "prompt_fingerprint": str(row["prompt_fingerprint"]),
        }
        for row in rows
        if row["category"] != TRAFFIC_CATEGORY
    ]


def _artifact(path: Path, rows: int) -> Dict[str, Any]:
    return {
        "path": path.name,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _isolation_declaration() -> Dict[str, bool]:
    return {field: False for field in REQUIRED_FALSE_ISOLATION_FIELDS}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic_train", required=True)
    parser.add_argument("--traffic_val", required=True)
    parser.add_argument("--traffic_test", required=True)
    parser.add_argument("--u1_dataset_dir", required=True)
    parser.add_argument("--logiqa_train_zh", required=True)
    parser.add_argument("--logiqa_dev_zh", required=True)
    parser.add_argument("--logiqa_test_zh", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args(argv)

    paths = {
        "traffic_train": Path(args.traffic_train).resolve(),
        "traffic_validation": Path(args.traffic_val).resolve(),
        "traffic_test": Path(args.traffic_test).resolve(),
        "logiqa_train": Path(args.logiqa_train_zh).resolve(),
        "logiqa_validation": Path(args.logiqa_dev_zh).resolve(),
        "logiqa_test": Path(args.logiqa_test_zh).resolve(),
    }
    # 训练/验证输入必须可解析；两个正式 test 在下面只走 attest。
    for key in (
        "traffic_train",
        "traffic_validation",
        "logiqa_train",
        "logiqa_validation",
    ):
        if not paths[key].is_file():
            raise ManifestError(f"输入文件不存在: {paths[key]}")
    _validate_traffic_input_identity(paths)
    traffic_test_attestation = _file_attestation(
        paths["traffic_test"], TRAFFIC_TEST_SHA256
    )
    logiqa_test_attestation = _file_attestation(
        paths["logiqa_test"], LOGIQA_TEST_SHA256
    )

    output = Path(args.output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ManifestError("拒绝覆盖非空统一数据目录")
    output.mkdir(parents=True, exist_ok=True)

    u1_dataset_dir = Path(args.u1_dataset_dir).resolve()
    u1_train, u1_dev, u1_report = _extract_u1_general_inventory(u1_dataset_dir)
    spent_u1_general_validation = _load_spent_u1_general_validation(
        u1_dataset_dir, u1_dev
    )
    traffic_train = _traffic_rows(paths["traffic_train"], "train")
    traffic_val = _traffic_rows(paths["traffic_validation"], "validation")

    official_math, gsm8k_report = _load_gsm8k_train()
    selected_math_train, selected_math_dev, math_selection_report = (
        _select_math_inventory(
            official_math,
            u1_train[MATH_CATEGORY],
            u1_dev[MATH_CATEGORY],
            args.seed,
        )
    )
    math_train = [_convert_math(item, "train") for item in selected_math_train]
    math_dev = [_convert_math(item, "validation") for item in selected_math_dev]
    if len(math_train) != MATH_TRAIN_ROWS or len(math_dev) != MATH_DEV_ROWS:
        raise ManifestError("U2 math 最终数量不符合 6673/400 合同")

    clean_logic_train, clean_logic_dev, logiqa_report = _load_clean_logiqa(
        paths["logiqa_train"], paths["logiqa_validation"]
    )
    selected_logic_train, selected_logic_dev, logic_selection_report = (
        _select_logic_inventory(
            clean_logic_train,
            clean_logic_dev,
            u1_dev[LOGIC_CATEGORY],
            args.seed,
        )
    )
    logic_train: List[Dict[str, Any]] = []
    logic_dev: List[Dict[str, Any]] = []
    transform_reports: List[Dict[str, Any]] = []
    for item in selected_logic_train:
        row, report = _convert_logic(item, "train", args.seed, permute=True)
        logic_train.append(row)
        transform_reports.append(report)
    for item in selected_logic_dev:
        row, report = _convert_logic(item, "validation", args.seed, permute=False)
        logic_dev.append(row)
        transform_reports.append(report)
    if len(logic_train) != LOGIC_CLEAN_TRAIN_ROWS or len(logic_dev) != LOGIC_DEV_ROWS:
        raise ManifestError("U2 logic 最终数量不符合 12193/400 合同")

    general_train = math_train + logic_train
    promotion_dev = math_dev + logic_dev
    train_fp = [str(row["prompt_fingerprint"]) for row in general_train]
    validation_fp = [
        str(row["prompt_fingerprint"]) for row in spent_u1_general_validation
    ]
    promotion_fp = [str(row["prompt_fingerprint"]) for row in promotion_dev]
    if any(
        len(set(values)) != len(values)
        for values in (train_fp, validation_fp, promotion_fp)
    ):
        raise ManifestError("U2 通用 train/validation/promotion 内部存在重复 fingerprint")
    if (
        set(train_fp) & set(validation_fp)
        or set(train_fp) & set(promotion_fp)
        or set(validation_fp) & set(promotion_fp)
    ):
        raise ManifestError("U2 通用 train/validation/promotion fingerprint 重叠")
    excluded_u1_dev = u1_dev[MATH_CATEGORY] | u1_dev[LOGIC_CATEGORY]
    if set(validation_fp) != excluded_u1_dev:
        raise ManifestError("U1 已使用通用开发题未被精确绑定为训练期 validation")
    if excluded_u1_dev & (set(train_fp) | set(promotion_fp)):
        raise ManifestError("U1 通用开发题泄漏进 U2 train/promotion")

    logic_train_labels = Counter(row["reference_answer"] for row in logic_train)
    logic_dev_labels = Counter(row["reference_answer"] for row in logic_dev)
    if set(logic_train_labels) != set(CHOICE_LABELS):
        raise ManifestError("LogiQA U2 train 的 A-D 标签分布不完整")
    if set(logic_dev_labels) != set(CHOICE_LABELS):
        raise ManifestError("LogiQA U2 dev 的 A-D 标签分布不完整")

    train_rows = traffic_train + general_train
    val_rows = traffic_val + spent_u1_general_validation
    random.Random(args.seed + 10).shuffle(train_rows)
    random.Random(args.seed + 11).shuffle(val_rows)
    train_output = output / "train.jsonl"
    val_output = output / "val.jsonl"
    dev_output = output / "general_dev_evaluation.jsonl"
    _write_jsonl(train_output, train_rows)
    _write_jsonl(val_output, val_rows)
    _write_jsonl(dev_output, _evaluation_rows(promotion_dev))

    label_before = Counter(report["original_label"] for report in transform_reports)
    label_after = Counter(report["new_label"] for report in transform_reports)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": "single_model_traffic_math_chinese_logic_u2_training_dataset",
        "seed": args.seed,
        "blind_content_policy": {
            "traffic_test": "sha256_and_stat_only",
            "gsm8k_test": "split_not_loaded",
            "logiqa_test": "sha256_and_stat_only",
            "test_content_used_for_training_or_development": False,
        },
        "prompt_formats": ["raw_task", "tokenizer_chat"],
        "traffic_contract": {
            "input_encoding": "routing_context_v2",
            "input_characters": 16,
            "output_labels": list("ABCDEF"),
            "single_token_required": True,
        },
        "general_contracts": {
            "math": {"task_prefix": "[TASK:MATH]", "output": "FINAL: <number>"},
            "natural_language_reasoning": {
                "task_prefix": "[任务:中文逻辑选择]",
                "output_labels": list(CHOICE_LABELS),
                "single_token_required": True,
                "train_option_prefixes_removed": True,
                "train_options_deterministically_permuted": True,
                "answer_relabelled_after_permutation": True,
            },
        },
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "promotion_dev_rows": len(promotion_dev),
        "train_category_counts": _category_counts(train_rows),
        "validation_category_counts": _category_counts(val_rows),
        "promotion_dev_category_counts": _category_counts(promotion_dev),
        "general_train_validation_prompt_overlap": 0,
        "general_train_promotion_prompt_overlap": 0,
        "general_validation_promotion_prompt_overlap": 0,
        "u1_general_dev_overlap_with_train_or_promotion": 0,
        "u1_general_dev_training_validation_rows": len(
            spent_u1_general_validation
        ),
        "u1_general_dev_training_validation_category_counts": _category_counts(
            spent_u1_general_validation
        ),
        "u1_general_dev_used_as_training_validation": True,
        "u1_general_dev_used_for_promotion": False,
        **_isolation_declaration(),
        "recommended_effective_task_weights": {
            TRAFFIC_CATEGORY: 5,
            MATH_CATEGORY: 2,
            LOGIC_CATEGORY: 3,
        },
        "sources": {
            "u1_dataset": u1_report,
            "traffic": {
                "train": {
                    "path": str(paths["traffic_train"]),
                    "bytes": paths["traffic_train"].stat().st_size,
                    "sha256": sha256_file(paths["traffic_train"]),
                    "rows": len(traffic_train),
                },
                "validation": {
                    "path": str(paths["traffic_validation"]),
                    "bytes": paths["traffic_validation"].stat().st_size,
                    "sha256": sha256_file(paths["traffic_validation"]),
                    "rows": len(traffic_val),
                },
                "test": {
                    **traffic_test_attestation,
                    "used_for_training": False,
                },
            },
            "gsm8k": {**gsm8k_report, "selection": math_selection_report},
            "logiqa2": {
                **logiqa_report,
                "train": {
                    "path": str(paths["logiqa_train"]),
                    "bytes": paths["logiqa_train"].stat().st_size,
                    "sha256": sha256_file(paths["logiqa_train"]),
                },
                "validation": {
                    "path": str(paths["logiqa_validation"]),
                    "bytes": paths["logiqa_validation"].stat().st_size,
                    "sha256": sha256_file(paths["logiqa_validation"]),
                },
                "test": logiqa_test_attestation,
                "selection": logic_selection_report,
            },
        },
        "logiqa_option_transformation": {
            "algorithm": "sha256_order(seed, content_fingerprint, original_index)",
            "train_rows_permuted": sum(
                int(report["permuted"])
                for report in transform_reports[: len(logic_train)]
            ),
            "option_prefixes_removed": sum(
                int(report["prefixes_stripped"]) for report in transform_reports
            ),
            "label_distribution_before": dict(sorted(label_before.items())),
            "label_distribution_after": dict(sorted(label_after.items())),
            "train_label_distribution": dict(sorted(logic_train_labels.items())),
            "dev_label_distribution": dict(sorted(logic_dev_labels.items())),
        },
        "artifacts": {
            "train": _artifact(train_output, len(train_rows)),
            "validation": _artifact(val_output, len(val_rows)),
            "general_dev_evaluation": _artifact(dev_output, len(promotion_dev)),
        },
    }
    manifest_path = output / "manifest.json"
    write_json_object(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    (output / "manifest.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "manifest_sha256_sidecar": str(output / "manifest.sha256"),
                "train_rows": len(train_rows),
                "validation_rows": len(val_rows),
                "promotion_dev_rows": len(promotion_dev),
                "status": "built",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
