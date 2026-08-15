"""Build the small, isolated traffic+industrial single-adapter SFT dataset.

Only train/validation JSONL content is parsed.  Formal test files are bound by
row count, byte count and SHA-256 only, and remain unopened as JSON records.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence

from edge_llm_factory.contracts import ManifestError, sha256_file, write_json_object


SCHEMA_VERSION = "edge-llm-joint-traffic-industrial/v1"
RAW16_SCHEMA_VERSION = "edge-llm-joint-traffic-industrial-raw16/v2"
SCHEMA_VERSIONS = frozenset((SCHEMA_VERSION, RAW16_SCHEMA_VERSION))
PREFIX17_CONTRACT = "scene-prefix1+decimal16@v1"
RAW16_CONTRACT = "scene-disjoint-decimal16@v2"
INPUT_CONTRACTS = frozenset((PREFIX17_CONTRACT, RAW16_CONTRACT))
PREFIXES = {"traffic": "T", "industrial": "I"}
TARGETS = {"traffic": frozenset("ABCDEF"), "industrial": frozenset("ABC")}
EXPECTED_TEST_ROWS = {"traffic": 2400, "industrial": 960}


def _read_rows(
    path: Path, scene: str, split: str, input_contract: str
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError("{}:{} JSON 无效".format(path, line_number)) from exc
            messages = source.get("messages") if isinstance(source, dict) else None
            if not isinstance(messages, list) or len(messages) != 2:
                raise ManifestError("{}:{} 必须包含一问一答".format(path, line_number))
            user, assistant = messages
            if not isinstance(user, dict) or not isinstance(assistant, dict):
                raise ManifestError("{}:{} messages 元素必须是对象".format(path, line_number))
            prompt = str(user.get("content", ""))
            target = str(assistant.get("content", "")).strip()
            if user.get("role") != "user" or assistant.get("role") != "assistant":
                raise ManifestError("{}:{} 消息角色无效".format(path, line_number))
            if len(prompt) != 16 or not prompt.isdigit():
                raise ManifestError("{}:{} 不是原始 decimal16 输入".format(path, line_number))
            if target not in TARGETS[scene]:
                raise ManifestError("{}:{} target 超出 {} 动作槽".format(path, line_number, scene))
            row = dict(source)
            prefix = PREFIXES[scene] if input_contract == PREFIX17_CONTRACT else ""
            row["messages"] = [
                {"role": "user", "content": prefix + prompt},
                {"role": "assistant", "content": target},
            ]
            row["prompt_format"] = "raw_task"
            metadata = dict(row.get("metadata", {}))
            metadata.update(
                {
                    "joint_scene": scene,
                    "joint_scene_prefix": prefix,
                    "joint_input_contract": input_contract,
                    "joint_split": split,
                }
            )
            row["metadata"] = metadata
            rows.append(row)
    if not rows:
        raise ManifestError("数据集为空: {}".format(path))
    return rows


def _artifact(path: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    with path.open("x", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    return {
        "path": str(path.resolve()),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _test_identity(path: Path, scene: str) -> Dict[str, Any]:
    # Deliberately do not parse JSONL.  Counting newline bytes and hashing are
    # the complete allowed interaction with the frozen formal test artifact.
    row_count = 0
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            row_count += block.count(b"\n")
    if row_count != EXPECTED_TEST_ROWS[scene]:
        raise ManifestError(
            "{} formal test 行数必须为 {}，实际 {}".format(
                scene, EXPECTED_TEST_ROWS[scene], row_count
            )
        )
    return {
        "path": str(path.resolve()),
        "rows": row_count,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "content_loaded": False,
        "used_for_training": False,
        "used_for_validation": False,
        "access_mode": "sha256_and_stat_only",
    }


def build(
    traffic_train: Path,
    traffic_val: Path,
    traffic_test: Path,
    industrial_train: Path,
    industrial_val: Path,
    industrial_test: Path,
    output_dir: Path,
    seed: int = 20260813,
    input_contract: str = PREFIX17_CONTRACT,
) -> Dict[str, Any]:
    if input_contract not in INPUT_CONTRACTS:
        raise ManifestError("unsupported joint input contract: {}".format(input_contract))
    all_inputs = [
        traffic_train,
        traffic_val,
        traffic_test,
        industrial_train,
        industrial_val,
        industrial_test,
    ]
    resolved_inputs = [Path(path).resolve() for path in all_inputs]
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise ManifestError("traffic/industrial train/validation/test paths must be distinct")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "traffic": {
            "train": traffic_train.resolve(),
            "validation": traffic_val.resolve(),
            "test": traffic_test.resolve(),
        },
        "industrial": {
            "train": industrial_train.resolve(),
            "validation": industrial_val.resolve(),
            "test": industrial_test.resolve(),
        },
    }
    split_rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": []}
    source_summary: Dict[str, Any] = {}
    for scene, paths in sources.items():
        scene_rows = {}
        for split in ("train", "validation"):
            rows = _read_rows(paths[split], scene, split, input_contract)
            scene_rows[split] = rows
            split_rows[split].extend(rows)
        source_summary[scene] = {
            "prefix": PREFIXES[scene] if input_contract == PREFIX17_CONTRACT else "",
            "train": {
                "path": str(paths["train"]),
                "rows": len(scene_rows["train"]),
                "sha256": sha256_file(paths["train"]),
                "content_loaded": True,
            },
            "validation": {
                "path": str(paths["validation"]),
                "rows": len(scene_rows["validation"]),
                "sha256": sha256_file(paths["validation"]),
                "content_loaded": True,
            },
            "formal_test": _test_identity(paths["test"], scene),
        }

    for offset, split in enumerate(("train", "validation")):
        random.Random(seed + offset).shuffle(split_rows[split])
    train_prompts = {row["messages"][0]["content"] for row in split_rows["train"]}
    validation_prompts = {
        row["messages"][0]["content"] for row in split_rows["validation"]
    }
    prompt_overlap = train_prompts & validation_prompts
    cross_scene_prompt_overlap: Dict[str, int] = {}
    scene_first_characters: Dict[str, List[str]] = {}
    if input_contract == RAW16_CONTRACT:
        for split, rows in split_rows.items():
            per_scene = {
                scene: {
                    row["messages"][0]["content"]
                    for row in rows
                    if row["metadata"]["joint_scene"] == scene
                }
                for scene in PREFIXES
            }
            overlap = per_scene["traffic"] & per_scene["industrial"]
            if overlap:
                raise ManifestError(
                    "raw16 {} traffic/industrial prompt overlap: {}".format(
                        split, len(overlap)
                    )
                )
            cross_scene_prompt_overlap[split] = 0
        for scene in PREFIXES:
            characters = sorted(
                {
                    row["messages"][0]["content"][0]
                    for rows in split_rows.values()
                    for row in rows
                    if row["metadata"]["joint_scene"] == scene
                }
            )
            scene_first_characters[scene] = characters
        if set(scene_first_characters["traffic"]) & set(
            scene_first_characters["industrial"]
        ):
            raise ManifestError("raw16 traffic/industrial first-character domains overlap")
    artifacts = {
        split: _artifact(output_dir / (split + ".jsonl"), rows)
        for split, rows in split_rows.items()
    }
    distribution: Dict[str, Any] = {}
    for split, rows in split_rows.items():
        distribution[split] = {
            scene: dict(
                sorted(
                    Counter(
                        row["messages"][-1]["content"]
                        for row in rows
                        if row["metadata"]["joint_scene"] == scene
                    ).items()
                )
            )
            for scene in PREFIXES
        }
    manifest = {
        "schema_version": (
            SCHEMA_VERSION
            if input_contract == PREFIX17_CONTRACT
            else RAW16_SCHEMA_VERSION
        ),
        "seed": seed,
        "contract": {
            "input": input_contract,
            "input_characters": 17 if input_contract == PREFIX17_CONTRACT else 16,
            "input_tokens": 17 if input_contract == PREFIX17_CONTRACT else 16,
            "output_tokens": 1,
            "prefixes": (
                PREFIXES
                if input_contract == PREFIX17_CONTRACT
                else {"traffic": "", "industrial": ""}
            ),
            "scene_first_characters": scene_first_characters,
            "cross_scene_prompt_overlap": cross_scene_prompt_overlap,
            "outputs": {"traffic": "A-F", "industrial": "A-C"},
        },
        "artifacts": artifacts,
        "sources": source_summary,
        "target_distribution": distribution,
        "formal_test_content_loaded": False,
        "formal_test_used_for_training_or_selection": False,
        "train_validation_prompt_overlap": len(prompt_overlap),
        "train_validation_prompt_overlap_note": (
            "The frozen traffic source contains legacy train/validation prompt "
            "overlap.  Validation loss is diagnostic only; promotion uses the "
            "separate frozen traffic and industrial test artifacts."
        ),
        "recipe": {
            "initialization": "clean_qwen35_0.8b_text",
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "epochs": 3.0,
            "batch_size": 8,
            "gradient_accumulation": 2,
            "learning_rate": 0.0001,
            "precision": "bf16",
            "max_length": 18 if input_contract == PREFIX17_CONTRACT else 17,
            "seed": seed,
        },
        "promotion_gates": {
            "traffic": {
                "count": 2400,
                "accuracy_min": 0.66,
                "weighted_f1_min": 0.65,
                "valid_output_rate": 1.0,
            },
            "industrial": {
                "count": 960,
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "weighted_f1": 1.0,
                "valid_output_rate": 1.0,
            },
        },
    }
    write_json_object(output_dir / "manifest.json", manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build one joint traffic+industrial LoRA dataset.")
    parser.add_argument("--traffic_train", required=True)
    parser.add_argument("--traffic_val", required=True)
    parser.add_argument("--traffic_test", required=True)
    parser.add_argument("--industrial_train", required=True)
    parser.add_argument("--industrial_val", required=True)
    parser.add_argument("--industrial_test", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--input_contract",
        choices=sorted(INPUT_CONTRACTS),
        default=PREFIX17_CONTRACT,
    )
    args = parser.parse_args(argv)
    result = build(
        Path(args.traffic_train), Path(args.traffic_val), Path(args.traffic_test),
        Path(args.industrial_train), Path(args.industrial_val), Path(args.industrial_test),
        Path(args.output_dir), args.seed, args.input_contract,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
