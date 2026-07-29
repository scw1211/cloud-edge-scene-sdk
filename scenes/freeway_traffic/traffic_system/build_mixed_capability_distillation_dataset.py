"""用途：构建交通回放与独立数学、代码、中文推理样本组成的混合蒸馏数据集。"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from datasets import get_dataset_config_names, load_dataset

from traffic_system.decision_utils import read_jsonl, write_jsonl


MATH_SYSTEM = (
    "You solve grade-school math accurately. Do not provide a long explanation. "
    "End with exactly: FINAL: <number>"
)
CODE_SYSTEM = "Write a correct Python solution. Output only Python code without Markdown fences or explanation."
CHOICE_SYSTEM = "回答中文单项选择题。只输出 A、B、C 或 D，不要解释。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-checked mixed capability SFT data.")
    parser.add_argument("--traffic_dir", default="datasets/llm_sft_freeway_action_token_v9")
    parser.add_argument("--benchmark_jsonl", default="datasets/general_capability_eval/eval.jsonl")
    parser.add_argument("--output_dir", default="datasets/llm_sft_qwen_v9_mixed_capability")
    parser.add_argument("--gsm8k_count", type=int, default=600)
    parser.add_argument("--code_repeats", type=int, default=3)
    parser.add_argument("--choice_repeats", type=int, default=2)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def normalize_prompt(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def chat_row(
    sample_id: str,
    category: str,
    system: str,
    prompt: str,
    answer: str,
) -> Dict[str, Any]:
    return {
        "event_id": sample_id,
        "category": category,
        "prompt_format": "tokenizer_chat",
        "source_prompt": prompt,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def gsm8k_rows(count: int, seed: int) -> List[Dict[str, Any]]:
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    indices = random.Random(seed).sample(range(len(dataset)), min(count, len(dataset)))
    rows = []
    for index in indices:
        item = dataset[index]
        raw_answer = str(item["answer"])
        reasoning, final = raw_answer.rsplit("####", 1)
        reasoning = re.sub(r"<<[^<>]*>>", "", reasoning).strip()
        answer = (reasoning + "\nFINAL: " + final.strip()).strip()
        rows.append(
            chat_row(
                "gsm8k_train_{}".format(index),
                "math",
                MATH_SYSTEM,
                str(item["question"]),
                answer,
            )
        )
    return rows


def mbpp_rows() -> List[Dict[str, Any]]:
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    rows = []
    for item in dataset:
        tests = "\n".join(str(value) for value in item.get("test_list", []))
        source_prompt = str(item["prompt"])
        user_prompt = "{}\nThe function must pass these tests:\n{}".format(source_prompt, tests)
        row = chat_row(
            "mbpp_train_{}".format(item["task_id"]),
            "code",
            CODE_SYSTEM,
            user_prompt,
            str(item["code"]).strip(),
        )
        row["source_prompt"] = source_prompt
        rows.append(row)
    return rows


def ceval_prompt(item: Dict[str, Any]) -> str:
    return "{}\nA. {}\nB. {}\nC. {}\nD. {}".format(
        item["question"], item["A"], item["B"], item["C"], item["D"]
    )


def ceval_rows() -> List[Dict[str, Any]]:
    rows = []
    configs = [
        config
        for config in get_dataset_config_names("ceval/ceval-exam")
        if config != "default"
    ]
    for config in configs:
        dataset = load_dataset("ceval/ceval-exam", config, split="dev")
        for item in dataset:
            prompt = ceval_prompt(item)
            rows.append(
                chat_row(
                    "ceval_{}_dev_{}".format(config, item["id"]),
                    "natural_language_reasoning",
                    CHOICE_SYSTEM,
                    prompt,
                    str(item["answer"]).strip().upper(),
                )
            )
    return rows


def split_rows(
    rows: Sequence[Dict[str, Any]], fraction: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * fraction)))
    return shuffled[validation_count:], shuffled[:validation_count]


def repeat_rows(rows: Sequence[Dict[str, Any]], repeats: int) -> List[Dict[str, Any]]:
    return [dict(row) for _ in range(repeats) for row in rows]


def digest(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    traffic_dir = Path(args.traffic_dir)
    traffic_train = read_jsonl(traffic_dir / "train.jsonl")
    traffic_val = read_jsonl(traffic_dir / "val.jsonl")
    traffic_test = read_jsonl(traffic_dir / "test.jsonl")
    for row in traffic_train + traffic_val:
        row["category"] = "traffic_action"
        row["prompt_format"] = "raw_task"

    math_train, math_val = split_rows(
        gsm8k_rows(args.gsm8k_count, args.seed), args.validation_fraction, args.seed + 1
    )
    code_train, code_val = split_rows(
        mbpp_rows(), args.validation_fraction, args.seed + 2
    )
    choice_train, choice_val = split_rows(
        ceval_rows(), args.validation_fraction, args.seed + 3
    )

    benchmark_rows = read_jsonl(Path(args.benchmark_jsonl))
    benchmark_prompts = {normalize_prompt(str(row.get("prompt", ""))) for row in benchmark_rows}
    general_train = math_train + code_train + choice_train
    leaked_prompts = sorted(
        {
            normalize_prompt(str(row.get("source_prompt", "")))
            for row in general_train
        }
        & benchmark_prompts
    )
    if leaked_prompts:
        raise ValueError("General benchmark prompt leakage detected ({} rows).".format(len(leaked_prompts)))

    train = list(traffic_train)
    train.extend(math_train)
    train.extend(repeat_rows(code_train, args.code_repeats))
    train.extend(repeat_rows(choice_train, args.choice_repeats))
    validation = list(traffic_val) + math_val + code_val + choice_val
    random.Random(args.seed + 4).shuffle(train)
    random.Random(args.seed + 5).shuffle(validation)

    train_ids = {str(row["event_id"]) for row in train}
    traffic_test_ids = {str(row["event_id"]) for row in traffic_test}
    if train_ids & traffic_test_ids:
        raise ValueError("Traffic strict-test leakage detected.")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, output / "train.jsonl")
    write_jsonl(validation, output / "val.jsonl")
    category_counts = Counter(str(row["category"]) for row in train)
    validation_counts = Counter(str(row["category"]) for row in validation)
    summary = {
        "task": "mixed_traffic_and_general_capability_distillation_dataset",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_category_counts": dict(sorted(category_counts.items())),
        "validation_category_counts": dict(sorted(validation_counts.items())),
        "general_sources": {
            "math": "GSM8K train",
            "code": "MBPP sanitized train",
            "natural_language_reasoning": "C-Eval dev (all configurations)",
        },
        "general_benchmark": "GSM8K test + MBPP sanitized test + C-Eval val",
        "general_benchmark_prompt_overlap": 0,
        "traffic_test_used_for_training": False,
        "general_eval_used_for_training": False,
        "traffic_test_id_sha256": digest(traffic_test_ids),
        "general_benchmark_id_sha256": digest(
            str(row["sample_id"]) for row in benchmark_rows
        ),
        "prompt_formats": ["raw_task", "tokenizer_chat"],
        "seed": args.seed,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
