"""用途：由第一阶段 Teacher 数据构建无系统提示的超短交通决策 SFT 数据集。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from traffic_system.decision_utils import read_jsonl
from traffic_system.ultracompact_codec import (
    encode_bitpacked_decimal_prompt,
    encode_contextual_decimal_prompt,
    encode_positional_decimal_prompt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ultracompact action-token SFT dataset.")
    parser.add_argument("--source_dir", default="datasets/llm_sft_freeway_action_token")
    parser.add_argument("--output_dir", default="datasets/llm_sft_freeway_action_token_v8")
    parser.add_argument(
        "--encoding",
        choices=["positional_decimal", "bitpacked_decimal", "contextual_decimal"],
        default="bitpacked_decimal",
    )
    return parser.parse_args()


def transform(row: Dict[str, Any], encoding: str) -> Dict[str, Any]:
    messages = row.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("Source row must contain system, user, and assistant messages.")
    user_prompt = str(messages[-2].get("content", ""))
    target = str(row.get("target") or messages[-1].get("content", "")).strip().upper()
    if target not in set("ABCDEF"):
        raise ValueError("Invalid action target: {!r}".format(target))
    if encoding == "positional_decimal":
        code = encode_positional_decimal_prompt(user_prompt)
    elif encoding == "bitpacked_decimal":
        code = encode_bitpacked_decimal_prompt(user_prompt)
    elif encoding == "contextual_decimal":
        code = encode_contextual_decimal_prompt(user_prompt)
    else:
        raise ValueError("Unsupported encoding: {}".format(encoding))
    return {
        **{key: value for key, value in row.items() if key != "messages"},
        "messages": [
            {"role": "user", "content": code},
            {"role": "assistant", "content": target},
        ],
        "legacy_prompt": user_prompt,
        "feature_code": code,
        "target": target,
        "input_encoding": encoding,
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    source_dir = resolve_path(args.source_dir)
    output_dir = resolve_path(args.output_dir)
    counts = {}
    code_lengths = []
    for split in ("train", "val", "test"):
        rows = [transform(row, args.encoding) for row in read_jsonl(source_dir / (split + ".jsonl"))]
        write_jsonl(output_dir / (split + ".jsonl"), rows)
        counts[split] = len(rows)
        code_lengths.extend(len(row["feature_code"]) for row in rows)
    summary = {
        "task": "ultracompact_qwen_action_token_dataset",
        "source_dir": str(source_dir.relative_to(PROJECT_ROOT)),
        "input_encoding": args.encoding,
        "chat_messages": ["user", "assistant"],
        "system_prompt_removed": True,
        "counts": counts,
        "code_length_chars": {"min": min(code_lengths), "max": max(code_lengths)},
        "test_set_used_for_training": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
