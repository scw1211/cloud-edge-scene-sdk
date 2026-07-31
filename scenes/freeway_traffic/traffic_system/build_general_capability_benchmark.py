"""用途：构建数学、代码和中文推理能力保持率所需的补充评测集。"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset  # noqa: E402

from traffic_system.decision_utils import save_json, write_jsonl  # noqa: E402


CEVAL_SUBJECTS = [
    "logic",
    "chinese_language_and_literature",
    "civil_servant",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed general-capability benchmark subset.")
    parser.add_argument("--output_jsonl", default="datasets/general_capability_eval/eval.jsonl")
    parser.add_argument("--metadata_json", default="datasets/general_capability_eval/metadata.json")
    parser.add_argument("--gsm8k_count", type=int, default=30)
    parser.add_argument("--mbpp_count", type=int, default=20)
    parser.add_argument("--ceval_per_subject", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def fixed_indices(size: int, count: int, seed: int) -> List[int]:
    if count <= 0 or count > size:
        raise ValueError("count must be in [1, {}], got {}".format(size, count))
    return sorted(random.Random(seed).sample(range(size), count))


def gsm8k_rows(count: int, seed: int) -> List[Dict[str, Any]]:
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for index in fixed_indices(len(dataset), count, seed):
        item = dataset[index]
        answer = str(item["answer"])
        final_answer = answer.rsplit("####", 1)[-1].strip()
        rows.append(
            {
                "benchmark": "gsm8k",
                "category": "math",
                "sample_id": "gsm8k_test_{}".format(index),
                "prompt": str(item["question"]),
                "reference_answer": final_answer,
            }
        )
    return rows


def mbpp_rows(count: int, seed: int) -> List[Dict[str, Any]]:
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    rows = []
    for index in fixed_indices(len(dataset), count, seed + 1):
        item = dataset[index]
        rows.append(
            {
                "benchmark": "mbpp",
                "category": "code",
                "sample_id": "mbpp_test_{}".format(item["task_id"]),
                "prompt": str(item["prompt"]),
                "test_imports": [str(value) for value in item.get("test_imports", [])],
                "test_list": [str(value) for value in item["test_list"]],
            }
        )
    return rows


def ceval_prompt(item: Dict[str, Any]) -> str:
    return "{}\nA. {}\nB. {}\nC. {}\nD. {}".format(
        item["question"],
        item["A"],
        item["B"],
        item["C"],
        item["D"],
    )


def ceval_rows(subjects: Sequence[str], count: int, seed: int) -> List[Dict[str, Any]]:
    rows = []
    for offset, subject in enumerate(subjects):
        dataset = load_dataset("ceval/ceval-exam", subject, split="val")
        for index in fixed_indices(len(dataset), count, seed + 100 + offset):
            item = dataset[index]
            rows.append(
                {
                    "benchmark": "ceval",
                    "category": "natural_language_reasoning",
                    "subject": subject,
                    "sample_id": "ceval_{}_val_{}".format(subject, item["id"]),
                    "prompt": ceval_prompt(item),
                    "reference_answer": str(item["answer"]).strip().upper(),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    rows = []
    rows.extend(gsm8k_rows(args.gsm8k_count, args.seed))
    rows.extend(mbpp_rows(args.mbpp_count, args.seed))
    rows.extend(ceval_rows(CEVAL_SUBJECTS, args.ceval_per_subject, args.seed))
    count = write_jsonl(rows, resolve_path(args.output_jsonl))
    category_counts = Counter(row["category"] for row in rows)
    metadata = {
        "name": "general_capability_retention_v1",
        "seed": args.seed,
        "total_samples": count,
        "category_counts": dict(category_counts),
        "sources": {
            "math": "openai/gsm8k main test",
            "code": "google-research-datasets/mbpp sanitized test",
            "natural_language_reasoning": {
                "dataset": "ceval/ceval-exam val",
                "subjects": CEVAL_SUBJECTS,
            },
        },
        "metrics": {
            "math": "final-answer exact match",
            "code": "pass@1 against official tests",
            "natural_language_reasoning": "multiple-choice accuracy",
        },
    }
    save_json(metadata, resolve_path(args.metadata_json))
    print(metadata)


if __name__ == "__main__":
    main()
