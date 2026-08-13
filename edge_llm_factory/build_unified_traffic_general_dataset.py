"""构建交通、GSM8K 数学和 LogiQA2.0 中文逻辑的统一训练数据。

这个构建器刻意不读取任何正式测试题内容：

* 交通 test 只计算文件哈希，用于证明未混入训练；
* GSM8K 只加载官方 train；
* LogiQA2.0 只解析 train_zh/dev_zh，test_zh 只记录哈希和字节数。

交通保持 16 位 ``raw_task`` 输入和 A-F 单 token 输出；数学与中文逻辑
使用普通 chat 模板。三类任务最终训练在同一个 Qwen3.5-0.8B LoRA 中。
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


SCHEMA_VERSION = "edge-llm-unified-traffic-general/v1"
TRAFFIC_CATEGORY = "traffic_action"
MATH_CATEGORY = "math"
LOGIC_CATEGORY = "natural_language_reasoning"
TRAFFIC_PATTERN = re.compile(r"^[0-9]{16}$")
TRAFFIC_LABELS = set("ABCDEF")
CHOICE_LABELS = set("ABCD")
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


def _stable_order(rows: Sequence[Mapping[str, Any]], seed: int) -> List[Dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: _sha_text(
            f"{seed}\n{row.get('category', '')}\n{row.get('prompt_fingerprint', '')}"
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
        event_id = str(raw.get("event_id", "")).strip()
        if not event_id or event_id in seen:
            raise ManifestError(f"交通 {split} event_id 缺失或重复: {event_id}")
        seen.add(event_id)
        fingerprint = _sha_text(f"{TRAFFIC_CATEGORY}\n{prompt}")
        output.append(
            {
                **dict(raw),
                "event_id": f"traffic:{split}:{event_id}",
                "source_event_id": event_id,
                "source_split": split,
                "category": TRAFFIC_CATEGORY,
                "prompt_format": "raw_task",
                "source_prompt": prompt,
                "prompt_fingerprint": fingerprint,
                "supervision_source": "frozen_future_observed_traffic_policy",
                "messages": messages,
            }
        )
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


def _gsm8k_rows(
    train_count: int, validation_count: int, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    # Remove the repository root from sys.path if it shadows Hugging Face datasets.
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
    fingerprint = str(getattr(dataset, "_fingerprint", ""))
    if fingerprint != GSM8K_TRAIN_FINGERPRINT:
        raise ManifestError(f"GSM8K train fingerprint 漂移: {fingerprint}")
    unique: Dict[str, Dict[str, Any]] = {}
    duplicate_count = 0
    for index, item in enumerate(dataset):
        question = str(item["question"]).strip()
        reasoning, final = _clean_gsm8k_answer(str(item["answer"]))
        prompt_key = _canonical_text(question)
        content_fingerprint = _sha_text(prompt_key)
        if content_fingerprint in unique:
            duplicate_count += 1
            continue
        unique[content_fingerprint] = {
            "source_index": index,
            "question": question,
            "reasoning": reasoning,
            "final": final,
            "content_fingerprint": content_fingerprint,
        }
    ordered = sorted(
        unique.values(),
        key=lambda row: _sha_text(f"{seed}\nmath\n{row['content_fingerprint']}"),
    )
    required = train_count + validation_count
    if required > len(ordered):
        raise ManifestError(
            f"GSM8K 请求 {required} 条，超过去重后的 {len(ordered)} 条"
        )
    validation_items = ordered[:validation_count]
    train_items = ordered[validation_count:required]

    def convert(item: Mapping[str, Any], split: str) -> Dict[str, Any]:
        answer = f"{item['reasoning']}\nFINAL: {item['final']}".strip()
        source_id = f"gsm8k_train_{item['source_index']}"
        prompt_fingerprint = _sha_text(f"{MATH_CATEGORY}\n{item['content_fingerprint']}")
        return {
            "event_id": f"math:{split}:{source_id}",
            "source_event_id": source_id,
            "source_split": split,
            "category": MATH_CATEGORY,
            "prompt_format": "tokenizer_chat",
            "source_prompt": item["question"],
            "prompt_fingerprint": prompt_fingerprint,
            "supervision_source": "official_gsm8k_human_reasoning",
            "reference_answer": str(item["final"]),
            "messages": [
                {"role": "system", "content": MATH_SYSTEM},
                {"role": "user", "content": str(item["question"])},
                {"role": "assistant", "content": answer},
            ],
        }

    return (
        [convert(item, "train") for item in train_items],
        [convert(item, "validation") for item in validation_items],
        {
            "dataset_id": GSM8K_DATASET_ID,
            "config": GSM8K_CONFIG,
            "split_loaded": "train",
            "test_split_loaded": False,
            "official_train_rows": len(dataset),
            "dataset_fingerprint": fingerprint,
            "unique_questions": len(unique),
            "duplicates_removed": duplicate_count,
        },
    )


def _logiqa_content_fingerprint(row: Mapping[str, Any]) -> str:
    payload = {
        "text": _canonical_text(str(row.get("text", ""))),
        "question": _canonical_text(str(row.get("question", ""))),
        "options": [_canonical_text(str(value)) for value in row.get("options", [])],
    }
    # 官方中文训练集中有 2 条 ``question`` 为空，题干已完整包含在
    # ``text`` 中。这是上游数据的合法形态，不应为了通过构建而伪造问题。
    if (
        (not payload["text"] and not payload["question"])
        or len(payload["options"]) != 4
        or any(not option for option in payload["options"])
    ):
        raise ManifestError("LogiQA2.0 样本缺少题干或四个有效选项")
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


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
            # 官方文件里有少量空选项。它们无法形成可执行的四选一任务，
            # 必须显式剔除并写入 manifest，不能用空字符串参与训练。
            invalid_rows += 1
            continue
        groups[fingerprint].append(dict(raw))
    return groups, invalid_rows


def _clean_logiqa(
    train_path: Path,
    dev_path: Path,
    train_count: int,
    validation_count: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
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
    for fingerprint, group in dev_groups.items():
        if fingerprint in train_groups:
            continue
        labels = {int(row["answer"]) for row in group}
        if len(labels) != 1:
            continue
        clean_dev[fingerprint] = group[0]
    if set(clean_train) & set(clean_dev):
        raise ManifestError("LogiQA2.0 清洗后 train/dev 仍有内容重叠")
    if len(clean_train) != 12193 or len(clean_dev) != 1494:
        raise ManifestError(
            f"LogiQA2.0 清洗数量漂移: train={len(clean_train)}, dev={len(clean_dev)}"
        )
    train_ordered = sorted(
        clean_train.items(), key=lambda pair: _sha_text(f"{seed}\nlogic\n{pair[0]}")
    )
    dev_ordered = sorted(
        clean_dev.items(), key=lambda pair: _sha_text(f"{seed + 1}\nlogic\n{pair[0]}")
    )
    if train_count > len(train_ordered) or validation_count > len(dev_ordered):
        raise ManifestError("LogiQA2.0 请求样本数超过清洗后的分片")

    def convert(pair: Tuple[str, Dict[str, Any]], split: str) -> Dict[str, Any]:
        fingerprint, item = pair
        answer = "ABCD"[int(item["answer"])]
        options = [str(value).strip() for value in item["options"]]
        prompt = (
            f"材料：{str(item['text']).strip()}\n"
            f"问题：{str(item['question']).strip()}\n"
            "选项：\n"
            + "\n".join(f"{label}. {value}" for label, value in zip("ABCD", options))
        )
        source_id = str(item.get("example_id", fingerprint[:16]))
        return {
            "event_id": f"logic:{split}:{source_id}:{fingerprint[:12]}",
            "source_event_id": source_id,
            "source_split": split,
            "category": LOGIC_CATEGORY,
            "prompt_format": "tokenizer_chat",
            "source_prompt": prompt,
            "prompt_fingerprint": _sha_text(f"{LOGIC_CATEGORY}\n{fingerprint}"),
            "content_fingerprint": fingerprint,
            "supervision_source": "official_logiqa2_human_verified",
            "reference_answer": answer,
            "messages": [
                {"role": "system", "content": LOGIC_SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
        }

    return (
        [convert(pair, "train") for pair in train_ordered[:train_count]],
        [convert(pair, "validation") for pair in dev_ordered[:validation_count]],
        {
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
            "train_dev_overlap_after_cleaning": 0,
            "test_split_loaded": False,
        },
    )


def _category_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(row["category"]) for row in rows).items()))


def _evaluation_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for row in rows:
        category = str(row["category"])
        if category == TRAFFIC_CATEGORY:
            continue
        output.append(
            {
                "sample_id": str(row["event_id"]),
                "category": category,
                "system_prompt": str(row["messages"][0]["content"]),
                "prompt": str(row["source_prompt"]),
                "reference_answer": str(row["reference_answer"]),
            }
        )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic_train", required=True)
    parser.add_argument("--traffic_val", required=True)
    parser.add_argument("--traffic_test", required=True)
    parser.add_argument("--logiqa_train_zh", required=True)
    parser.add_argument("--logiqa_dev_zh", required=True)
    parser.add_argument("--logiqa_test_zh", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--math_train_count", type=int, default=3600)
    parser.add_argument("--math_val_count", type=int, default=400)
    parser.add_argument("--logic_train_count", type=int, default=3600)
    parser.add_argument("--logic_val_count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args(argv)
    if min(
        args.math_train_count,
        args.math_val_count,
        args.logic_train_count,
        args.logic_val_count,
    ) <= 0:
        raise ManifestError("各任务训练和验证样本数必须为正数")
    paths = {
        "traffic_train": Path(args.traffic_train).resolve(),
        "traffic_validation": Path(args.traffic_val).resolve(),
        "traffic_test": Path(args.traffic_test).resolve(),
        "logiqa_train": Path(args.logiqa_train_zh).resolve(),
        "logiqa_validation": Path(args.logiqa_dev_zh).resolve(),
        "logiqa_test": Path(args.logiqa_test_zh).resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise ManifestError(f"输入文件不存在: {path}")
    if sha256_file(paths["logiqa_test"]) != LOGIQA_TEST_SHA256:
        raise ManifestError("LogiQA2.0 test_zh SHA-256 不匹配")
    output = Path(args.output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ManifestError("拒绝覆盖非空统一数据目录")
    output.mkdir(parents=True, exist_ok=True)

    traffic_train = _traffic_rows(paths["traffic_train"], "train")
    traffic_val = _traffic_rows(paths["traffic_validation"], "validation")
    math_train, math_val, gsm8k_report = _gsm8k_rows(
        args.math_train_count, args.math_val_count, args.seed
    )
    logic_train, logic_val, logiqa_report = _clean_logiqa(
        paths["logiqa_train"],
        paths["logiqa_validation"],
        args.logic_train_count,
        args.logic_val_count,
        args.seed,
    )
    train_rows = traffic_train + math_train + logic_train
    val_rows = traffic_val + math_val + logic_val
    # GSM8K/LogiQA 必须严格去重、训练与开发隔离。交通正式数据是既有的
    # 离散 16 位状态码数据，同一状态码可由不同时间窗产生不同监督标签；
    # 为了不改变现有生产基线的训练分布，这部分不擅自去重，而是如实记录。
    general_train = [row for row in train_rows if row["category"] != TRAFFIC_CATEGORY]
    general_val = [row for row in val_rows if row["category"] != TRAFFIC_CATEGORY]
    general_train_fingerprints = {
        str(row["prompt_fingerprint"]) for row in general_train
    }
    general_val_fingerprints = {str(row["prompt_fingerprint"]) for row in general_val}
    if len(general_train_fingerprints) != len(general_train):
        raise ManifestError("通用训练集存在重复 prompt fingerprint")
    if len(general_val_fingerprints) != len(general_val):
        raise ManifestError("通用验证集存在重复 prompt fingerprint")
    if general_train_fingerprints & general_val_fingerprints:
        raise ManifestError("通用训练集和验证集 prompt 重叠")
    traffic_train_fingerprints = {
        str(row["prompt_fingerprint"]) for row in traffic_train
    }
    traffic_val_fingerprints = {str(row["prompt_fingerprint"]) for row in traffic_val}
    random.Random(args.seed + 2).shuffle(train_rows)
    random.Random(args.seed + 3).shuffle(val_rows)

    train_output = output / "train.jsonl"
    val_output = output / "val.jsonl"
    dev_eval_output = output / "general_dev_evaluation.jsonl"
    _write_jsonl(train_output, train_rows)
    _write_jsonl(val_output, val_rows)
    _write_jsonl(dev_eval_output, _evaluation_rows(math_val + logic_val))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": "single_model_traffic_math_chinese_logic_training_dataset",
        "seed": args.seed,
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
                "output_labels": list("ABCD"),
                "single_token_required": True,
            },
        },
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_category_counts": _category_counts(train_rows),
        "validation_category_counts": _category_counts(val_rows),
        "general_train_validation_prompt_overlap": 0,
        "traffic_legacy_prompt_statistics": {
            "train_unique_prompts": len(traffic_train_fingerprints),
            "validation_unique_prompts": len(traffic_val_fingerprints),
            "train_duplicate_rows": len(traffic_train) - len(traffic_train_fingerprints),
            "validation_duplicate_rows": len(traffic_val)
            - len(traffic_val_fingerprints),
            "train_validation_prompt_overlap": len(
                traffic_train_fingerprints & traffic_val_fingerprints
            ),
            "preserved_without_deduplication": True,
        },
        "traffic_test_used_for_training": False,
        "gsm8k_test_loaded": False,
        "logiqa_test_loaded": False,
        "recommended_effective_task_weights": {
            TRAFFIC_CATEGORY: 2,
            MATH_CATEGORY: 1,
            LOGIC_CATEGORY: 1,
        },
        "sources": {
            "traffic": {
                "train": {
                    "path": str(paths["traffic_train"]),
                    "sha256": sha256_file(paths["traffic_train"]),
                },
                "validation": {
                    "path": str(paths["traffic_validation"]),
                    "sha256": sha256_file(paths["traffic_validation"]),
                },
                "test": {
                    "path": str(paths["traffic_test"]),
                    "sha256": sha256_file(paths["traffic_test"]),
                    "content_loaded": False,
                    "used_for_training": False,
                },
            },
            "gsm8k": gsm8k_report,
            "logiqa2": {
                **logiqa_report,
                "train_sha256": sha256_file(paths["logiqa_train"]),
                "validation_sha256": sha256_file(paths["logiqa_validation"]),
                "test_sha256": sha256_file(paths["logiqa_test"]),
                "test_bytes": paths["logiqa_test"].stat().st_size,
                "test_content_loaded": False,
            },
        },
        "artifacts": {
            "train": {"path": train_output.name, "sha256": sha256_file(train_output)},
            "validation": {"path": val_output.name, "sha256": sha256_file(val_output)},
            "general_dev_evaluation": {
                "path": dev_eval_output.name,
                "sha256": sha256_file(dev_eval_output),
            },
        },
    }
    write_json_object(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
