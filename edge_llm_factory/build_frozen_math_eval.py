"""Build a preregistered GSM8K math holdout without touching scene models.

The builder intentionally accepts only the official GSM8K ``main/test`` parquet
as the candidate pool.  Prompts already used by training, validation, an older
evaluation, or any other declared exclusion file are removed before a seeded,
deterministic selection is made.  The emitted manifest binds the source,
exclusions, selected sample IDs, and final JSONL by SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from edge_llm_factory.contracts import ManifestError, sha256_file, write_json_object


SCHEMA_VERSION = "edge-llm-frozen-math-evaluation/v1"
DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"


def normalize_prompt(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def prompt_fingerprint(value: str) -> str:
    normalized = normalize_prompt(value)
    if not normalized:
        raise ManifestError("数学题 prompt 不能为空")
    return hashlib.sha256(("math\n" + normalized).encode("utf-8")).hexdigest()


def extract_reference(answer: str) -> str:
    matches = re.findall(r"####\s*([-+$]?[0-9][0-9,]*(?:\.[0-9]+)?)", answer)
    if len(matches) != 1:
        raise ManifestError("GSM8K answer 必须恰好包含一个 #### 数值答案")
    return matches[0].replace(",", "")


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
    return rows


def _row_prompt(row: Mapping[str, Any], location: str) -> str:
    if isinstance(row.get("prompt"), str) and str(row["prompt"]).strip():
        return str(row["prompt"])
    if isinstance(row.get("source_prompt"), str) and str(row["source_prompt"]).strip():
        return str(row["source_prompt"])
    messages = row.get("messages")
    if isinstance(messages, list):
        users = [
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        if len(users) == 1 and users[0].strip():
            return users[0]
    raise ManifestError(f"{location} 无法提取 prompt")


def exclusion_fingerprints(
    sources: Sequence[Tuple[str, Sequence[Mapping[str, Any]]]]
) -> Tuple[set[str], Dict[str, int]]:
    fingerprints: set[str] = set()
    counts: Dict[str, int] = {}
    for label, rows in sources:
        before = len(fingerprints)
        for index, row in enumerate(rows):
            fingerprints.add(prompt_fingerprint(_row_prompt(row, f"{label}[{index}]")))
        counts[label] = len(fingerprints) - before
    return fingerprints, counts


def build_math_evaluation(
    gsm8k_rows: Sequence[Mapping[str, Any]],
    excluded: set[str],
    *,
    sample_count: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if sample_count <= 0:
        raise ManifestError("sample_count 必须大于 0")
    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()
    excluded_count = 0
    duplicate_count = 0
    for row_index, raw in enumerate(gsm8k_rows):
        question = str(raw.get("question", "")).strip()
        answer = str(raw.get("answer", "")).strip()
        if not question or not answer:
            raise ManifestError(f"GSM8K test row {row_index} 缺少 question/answer")
        fingerprint = prompt_fingerprint(question)
        if fingerprint in seen:
            duplicate_count += 1
            continue
        seen.add(fingerprint)
        if fingerprint in excluded:
            excluded_count += 1
            continue
        candidates.append(
            {
                "benchmark": "gsm8k",
                "category": "math",
                "sample_id": f"gsm8k_main_test_{row_index}",
                "prompt": question,
                "reference_answer": extract_reference(answer),
                "prompt_fingerprint": fingerprint,
                "source_row_index": row_index,
            }
        )
    if len(candidates) < sample_count:
        raise ManifestError(
            f"排除后仅剩 {len(candidates)} 道数学题，少于请求的 {sample_count} 道"
        )
    candidates.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['sample_id']}:{row['prompt_fingerprint']}".encode("utf-8")
        ).hexdigest()
    )
    selected = candidates[:sample_count]
    selected_ids = sorted(str(row["sample_id"]) for row in selected)
    selected_fingerprints = sorted(str(row["prompt_fingerprint"]) for row in selected)
    report = {
        "source_rows": len(gsm8k_rows),
        "excluded_prompt_count": excluded_count,
        "duplicate_prompt_count": duplicate_count,
        "eligible_rows": len(candidates),
        "selected_rows": len(selected),
        "selected_sample_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
        "selected_prompt_fingerprints_sha256": hashlib.sha256(
            "\n".join(selected_fingerprints).encode("utf-8")
        ).hexdigest(),
    }
    return selected, report


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="构建与训练数据隔离的冻结 GSM8K 数学终测集。")
    parser.add_argument("--gsm8k_parquet", required=True)
    parser.add_argument("--exclude_jsonl", action="append", default=[])
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample_count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args(argv)

    parquet_path = Path(args.gsm8k_parquet).resolve()
    output_path = Path(args.output_jsonl).resolve()
    manifest_path = Path(args.manifest).resolve()
    for path in (output_path, manifest_path):
        if path.exists():
            raise ManifestError(f"拒绝覆盖已有冻结证据: {path}")
    if not parquet_path.is_file():
        raise ManifestError(f"GSM8K parquet 不存在: {parquet_path}")

    from pyarrow import parquet

    table = parquet.read_table(parquet_path, columns=["question", "answer"])
    gsm8k_rows = table.to_pylist()
    exclusion_sources: List[Tuple[str, Sequence[Mapping[str, Any]]]] = []
    exclusion_artifacts = []
    for raw_path in args.exclude_jsonl:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ManifestError(f"排除集不存在: {path}")
        rows = _read_jsonl(path)
        label = f"exclude_{len(exclusion_sources)}"
        exclusion_sources.append((label, rows))
        exclusion_artifacts.append(
            {"label": label, "path": str(path), "sha256": sha256_file(path), "rows": len(rows)}
        )
    excluded, exclusion_counts = exclusion_fingerprints(exclusion_sources)
    selected, selection = build_math_evaluation(
        gsm8k_rows, excluded, sample_count=args.sample_count, seed=args.seed
    )
    _write_jsonl(output_path, selected)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": "preregistered_scene_independent_math_evaluation",
        "dataset": {
            "dataset_id": DATASET_ID,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "path": str(parquet_path),
            "sha256": sha256_file(parquet_path),
        },
        "seed": args.seed,
        "selection_policy": "sha256(seed:sample_id:prompt_fingerprint), ascending",
        "training_or_tuning_use_allowed": False,
        "scene_specific_samples": 0,
        "exclusions": exclusion_artifacts,
        "exclusion_unique_counts": exclusion_counts,
        "selection": selection,
        "artifact": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "rows": len(selected),
        },
    }
    write_json_object(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
