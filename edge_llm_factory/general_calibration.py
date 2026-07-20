"""用途：从非场景训练样本构建可审计的通用 GGUF 量化校准文本。"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from edge_llm_factory.contracts import ManifestError, sha256_file, write_json_object


GENERAL_CATEGORIES = (
    "math",
    "code",
    "natural_language_reasoning",
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ManifestError("{}:{} 必须是 JSON object".format(path, line_number))
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError("无法读取 JSONL {}: {}".format(path, exc)) from exc
    return rows


def _normalise_text(value: str) -> str:
    return " ".join(value.strip().split())


def source_prompt(row: Mapping[str, Any]) -> str:
    for key in ("source_prompt", "prompt"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return _normalise_text(value)
    messages = row.get("messages")
    if isinstance(messages, list):
        user_parts = [
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        if user_parts:
            return _normalise_text("\n".join(user_parts))
    raise ManifestError("校准样本缺少 source_prompt、prompt 或 user message")


def prompt_fingerprint(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(source_prompt(row).encode("utf-8")).hexdigest()


def _validated_messages(row: Mapping[str, Any]) -> List[Dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ManifestError("校准样本 messages 必须是非空数组")
    validated: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ManifestError("校准样本 message 必须是 object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ManifestError("校准样本包含不支持的 role: {}".format(role))
        if not isinstance(content, str) or not content.strip():
            raise ManifestError("校准样本 message content 不能为空")
        validated.append({"role": str(role), "content": content.strip()})
    if not any(message["role"] == "user" for message in validated):
        raise ManifestError("校准样本必须包含 user message")
    if not any(message["role"] == "assistant" for message in validated):
        raise ManifestError("校准样本必须包含 assistant message")
    return validated


def _render(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> Tuple[str, int]:
    text = tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=False
    )
    if not isinstance(text, str) or not text.strip():
        raise ManifestError("tokenizer 未能渲染校准对话")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return text.strip(), len(token_ids)


def _round_robin(groups: Mapping[str, Sequence[Dict[str, Any]]]) -> Iterable[Dict[str, Any]]:
    indexes = {category: 0 for category in groups}
    categories = sorted(groups)
    while True:
        emitted = False
        for category in categories:
            index = indexes[category]
            if index >= len(groups[category]):
                continue
            yield groups[category][index]
            indexes[category] += 1
            emitted = True
        if not emitted:
            return


def build_calibration(
    source_paths: Sequence[Path],
    exclude_paths: Sequence[Path],
    tokenizer: Any,
    categories: Sequence[str],
    target_tokens: int,
    minimum_tokens: int,
    max_sample_tokens: int,
    seed: int,
) -> Tuple[str, Dict[str, Any]]:
    if not source_paths:
        raise ManifestError("至少需要一个校准数据源")
    if target_tokens <= 0 or minimum_tokens <= 0 or minimum_tokens > target_tokens:
        raise ManifestError("token 目标必须满足 0 < minimum_tokens <= target_tokens")
    if max_sample_tokens <= 0:
        raise ManifestError("max_sample_tokens 必须大于 0")
    allowed = tuple(dict.fromkeys(categories))
    if not allowed or any(category not in GENERAL_CATEGORIES for category in allowed):
        raise ManifestError("校准类别只能来自 {}".format(", ".join(GENERAL_CATEGORIES)))

    excluded_fingerprints = {
        prompt_fingerprint(row)
        for path in exclude_paths
        for row in _read_jsonl(path)
    }
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    observed_categories: Counter = Counter()
    skipped = Counter()
    seen_fingerprints = set()
    for path in source_paths:
        for row in _read_jsonl(path):
            category = str(row.get("category", ""))
            observed_categories[category or "<missing>"] += 1
            if category not in allowed:
                skipped["non_general_category"] += 1
                continue
            fingerprint = prompt_fingerprint(row)
            if fingerprint in excluded_fingerprints:
                skipped["evaluation_overlap"] += 1
                continue
            if fingerprint in seen_fingerprints:
                skipped["duplicate_prompt"] += 1
                continue
            messages = _validated_messages(row)
            rendered, token_count = _render(tokenizer, messages)
            if token_count > max_sample_tokens:
                skipped["sample_too_long"] += 1
                continue
            seen_fingerprints.add(fingerprint)
            groups[category].append(
                {
                    "category": category,
                    "fingerprint": fingerprint,
                    "rendered": rendered,
                    "token_count": token_count,
                }
            )

    rng = random.Random(seed)
    for category in allowed:
        rng.shuffle(groups[category])
        if not groups[category]:
            raise ManifestError("通用校准类别 {} 没有可用样本".format(category))

    selected: List[Dict[str, Any]] = []
    selected_tokens = 0
    for sample in _round_robin({category: groups[category] for category in allowed}):
        if selected and selected_tokens >= target_tokens:
            break
        selected.append(sample)
        selected_tokens += int(sample["token_count"])
    if selected_tokens < minimum_tokens:
        raise ManifestError(
            "通用校准文本只有 {} tokens，低于 minimum_tokens={}".format(
                selected_tokens, minimum_tokens
            )
        )

    text = "\n\n".join(str(sample["rendered"]) for sample in selected) + "\n"
    selected_counts = Counter(str(sample["category"]) for sample in selected)
    selected_fingerprints = [str(sample["fingerprint"]) for sample in selected]
    manifest = {
        "schema_version": "edge-llm-general-calibration/v1",
        "task": "scene_independent_llm_quantization_calibration",
        "seed": seed,
        "categories": list(allowed),
        "scene_specific_samples": 0,
        "source_files": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in source_paths
        ],
        "excluded_evaluation_files": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in exclude_paths
        ],
        "observed_category_counts": dict(sorted(observed_categories.items())),
        "selected_category_counts": dict(sorted(selected_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
        "sample_count": len(selected),
        "token_count": selected_tokens,
        "target_tokens": target_tokens,
        "max_sample_tokens": max_sample_tokens,
        "evaluation_prompt_overlap": 0,
        "selected_prompt_fingerprints_sha256": hashlib.sha256(
            "\n".join(selected_fingerprints).encode("ascii")
        ).hexdigest(),
        "calibration_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return text, manifest


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="构建场景无关的 GGUF 通用校准文本。")
    parser.add_argument("--source_jsonl", action="append", required=True)
    parser.add_argument("--exclude_jsonl", action="append", default=[])
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--target_tokens", type=int, default=50000)
    parser.add_argument("--minimum_tokens", type=int, default=20000)
    parser.add_argument("--max_sample_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output_text", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)

    source_paths = [Path(value).resolve() for value in args.source_jsonl]
    exclude_paths = [Path(value).resolve() for value in args.exclude_jsonl]
    snapshot = Path(args.snapshot).resolve()
    output_text = Path(args.output_text).resolve()
    manifest_path = Path(args.manifest).resolve()
    for path in source_paths + exclude_paths:
        if not path.is_file():
            raise ManifestError("JSONL 文件不存在: {}".format(path))
    if not snapshot.is_dir():
        raise ManifestError("文本基座不存在: {}".format(snapshot))
    if output_text.exists() or manifest_path.exists():
        raise ManifestError("拒绝覆盖已有校准文本或清单")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, use_fast=True, trust_remote_code=False
    )
    text, manifest = build_calibration(
        source_paths=source_paths,
        exclude_paths=exclude_paths,
        tokenizer=tokenizer,
        categories=args.category or GENERAL_CATEGORIES,
        target_tokens=args.target_tokens,
        minimum_tokens=args.minimum_tokens,
        max_sample_tokens=args.max_sample_tokens,
        seed=args.seed,
    )
    output_text.parent.mkdir(parents=True, exist_ok=True)
    output_text.write_text(text, encoding="utf-8")
    manifest["calibration_text"] = {
        "path": str(output_text),
        "sha256": sha256_file(output_text),
        "bytes": output_text.stat().st_size,
    }
    write_json_object(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
