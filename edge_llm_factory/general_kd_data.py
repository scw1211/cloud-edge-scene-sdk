"""用途：调用云端 Teacher 构建无场景、无测试泄漏的通用行为蒸馏集。"""

import argparse
import hashlib
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from edge_llm_factory.contracts import ManifestError, sha256_file, write_json_object
from edge_llm_factory.general_kd_eval import (
    canonical_teacher_target,
    evaluate_teacher_output,
)


GENERAL_CATEGORIES = ("code", "math", "natural_language_reasoning")
NUM_PREDICT = {"code": 384, "math": 256, "natural_language_reasoning": 4}


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


def prompt_fingerprint(row: Mapping[str, Any]) -> str:
    category = str(row.get("category", ""))
    source_prompt = normalize_prompt(str(row.get("source_prompt", "")))
    if not source_prompt:
        raise ManifestError("通用样本缺少 source_prompt")
    return hashlib.sha256((category + "\n" + source_prompt).encode("utf-8")).hexdigest()


def _validated_messages(row: Mapping[str, Any]) -> List[Dict[str, str]]:
    raw = row.get("messages")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ManifestError("通用样本必须包含输入和 assistant reference")
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


def prepare_source_rows(
    rows: Sequence[Mapping[str, Any]],
    evaluation_prompts: Sequence[str],
    split: str,
    limit_per_category: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    eval_set = {normalize_prompt(value) for value in evaluation_prompts}
    unique: Dict[str, Dict[str, Any]] = {}
    skipped = Counter()
    for raw in rows:
        category = str(raw.get("category", ""))
        if category not in GENERAL_CATEGORIES:
            skipped["non_general_category"] += 1
            continue
        if str(raw.get("prompt_format", "tokenizer_chat")) != "tokenizer_chat":
            skipped["non_chat_format"] += 1
            continue
        source_prompt = normalize_prompt(str(raw.get("source_prompt", "")))
        if not source_prompt:
            skipped["missing_source_prompt"] += 1
            continue
        if source_prompt in eval_set:
            skipped["evaluation_overlap"] += 1
            continue
        messages = _validated_messages(raw)
        fingerprint = prompt_fingerprint(raw)
        if fingerprint in unique:
            skipped["duplicate_prompt"] += 1
            continue
        event_id = str(raw.get("event_id", "")).strip()
        if not event_id:
            raise ManifestError("通用样本缺少 event_id")
        unique[fingerprint] = {
            "sample_id": "{}:{}".format(split, event_id),
            "source_event_id": event_id,
            "source_split": split,
            "category": category,
            "source_prompt": str(raw.get("source_prompt", "")).strip(),
            "prompt_fingerprint": fingerprint,
            "messages": messages,
        }
    selected = list(unique.values())
    selected.sort(
        key=lambda row: hashlib.sha256(
            "{}:{}:{}".format(seed, row["category"], row["sample_id"]).encode("utf-8")
        ).hexdigest()
    )
    if limit_per_category > 0:
        counts = Counter()
        limited = []
        for row in selected:
            category = str(row["category"])
            if counts[category] >= limit_per_category:
                skipped["over_category_limit"] += 1
                continue
            counts[category] += 1
            limited.append(row)
        selected = limited
    return selected, dict(sorted(skipped.items()))


def _teacher_chat(
    host: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    category: str,
    num_ctx: int,
    timeout: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": list(messages),
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "num_ctx": num_ctx,
            "num_predict": NUM_PREDICT[category],
        },
    }
    request = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    return {
        "text": str(value.get("message", {}).get("content", "")).strip(),
        "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 4),
        "prompt_tokens": int(value.get("prompt_eval_count", 0)),
        "output_tokens": int(value.get("eval_count", 0)),
        "load_duration_ms": round(float(value.get("load_duration", 0)) / 1_000_000.0, 4),
        "prompt_eval_duration_ms": round(
            float(value.get("prompt_eval_duration", 0)) / 1_000_000.0, 4
        ),
        "eval_duration_ms": round(float(value.get("eval_duration", 0)) / 1_000_000.0, 4),
    }


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
        file_obj.flush()


def _category_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(row["category"]) for row in rows).items()))


def _request_fingerprint(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _output_row(source: Mapping[str, Any], rollout: Mapping[str, Any]) -> Dict[str, Any]:
    messages = list(source["messages"][:-1]) + [
        {"role": "assistant", "content": str(rollout["teacher_target"])}
    ]
    return {
        "sample_id": source["sample_id"],
        "source_event_id": source["source_event_id"],
        "source_split": source["source_split"],
        "category": source["category"],
        "source_prompt": source["source_prompt"],
        "prompt_fingerprint": source["prompt_fingerprint"],
        "teacher_model": rollout["teacher_model"],
        "teacher_verified": True,
        "messages": messages,
    }


def load_reused_verified_rows(
    paths: Sequence[Path], teacher_model: str
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    reused: Dict[str, Dict[str, Any]] = {}
    sources = []
    for path in paths:
        rows = _read_jsonl(path)
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(rows),
            }
        )
        for row in rows:
            if row.get("teacher_verified") is not True:
                raise ManifestError("复用样本未经 Teacher 验收: {}".format(path))
            if str(row.get("teacher_model", "")) != teacher_model:
                raise ManifestError("复用样本 Teacher 与当前请求不一致: {}".format(path))
            declared = str(row.get("prompt_fingerprint", ""))
            calculated = prompt_fingerprint(row)
            if not declared or declared != calculated:
                raise ManifestError("复用样本 prompt fingerprint 无效: {}".format(path))
            messages = _validated_messages(row)
            candidate = {
                "category": str(row.get("category", "")),
                "teacher_target": messages[-1]["content"],
                "source_path": str(path),
                "source_sha256": sources[-1]["sha256"],
            }
            previous = reused.get(declared)
            if previous and previous["teacher_target"] != candidate["teacher_target"]:
                raise ManifestError("同一 prompt 的复用 Teacher target 不一致")
            reused[declared] = candidate
    return reused, sources


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="生成无场景的 9B -> 0.8B 通用行为蒸馏集。")
    parser.add_argument("--source_train_jsonl", required=True)
    parser.add_argument("--source_val_jsonl", required=True)
    parser.add_argument("--evaluation_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_manifest")
    parser.add_argument("--reuse_verified_jsonl", action="append", default=[])
    parser.add_argument("--teacher_model", default="qwen3.5:9b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--num_ctx", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--code_timeout", type=float, default=3.0)
    parser.add_argument("--limit_train_per_category", type=int, default=0)
    parser.add_argument("--limit_val_per_category", type=int, default=0)
    parser.add_argument("--minimum_accepted_per_category", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    train_source = Path(args.source_train_jsonl).resolve()
    val_source = Path(args.source_val_jsonl).resolve()
    evaluation_path = Path(args.evaluation_jsonl).resolve()
    source_manifest_path = Path(args.source_manifest).resolve() if args.source_manifest else None
    reuse_paths = [Path(value).resolve() for value in args.reuse_verified_jsonl]
    required_paths = [train_source, val_source, evaluation_path] + reuse_paths
    if source_manifest_path:
        required_paths.append(source_manifest_path)
    for path in required_paths:
        if not path.is_file():
            raise ManifestError("输入文件不存在: {}".format(path))
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ManifestError("输出目录非空；如需恢复请使用 --resume")
    output.mkdir(parents=True, exist_ok=True)

    evaluation_rows = _read_jsonl(evaluation_path)
    evaluation_prompts = [str(row.get("prompt", "")) for row in evaluation_rows]
    train_rows, train_skipped = prepare_source_rows(
        _read_jsonl(train_source),
        evaluation_prompts,
        "train",
        args.limit_train_per_category,
        args.seed,
    )
    val_rows, val_skipped = prepare_source_rows(
        _read_jsonl(val_source),
        evaluation_prompts,
        "validation",
        args.limit_val_per_category,
        args.seed + 1,
    )
    train_fingerprints = {row["prompt_fingerprint"] for row in train_rows}
    val_fingerprints = {row["prompt_fingerprint"] for row in val_rows}
    overlap = train_fingerprints & val_fingerprints
    if overlap:
        raise ManifestError("训练集与验证集 prompt 重叠: {}".format(len(overlap)))
    if train_skipped.get("evaluation_overlap", 0) or val_skipped.get("evaluation_overlap", 0):
        raise ManifestError("检测到冻结测试 prompt 泄漏")

    reused_by_fingerprint, reuse_sources = load_reused_verified_rows(
        reuse_paths, args.teacher_model
    )
    request_config = {
        "schema_version": "edge-llm-general-kd-request/v2",
        "teacher_model": args.teacher_model,
        "source_train_sha256": sha256_file(train_source),
        "source_val_sha256": sha256_file(val_source),
        "source_manifest_sha256": sha256_file(source_manifest_path)
        if source_manifest_path
        else None,
        "reuse_verified_sources": reuse_sources,
        "evaluation_sha256": sha256_file(evaluation_path),
        "selected_train_ids": [row["sample_id"] for row in train_rows],
        "selected_val_ids": [row["sample_id"] for row in val_rows],
        "num_ctx": args.num_ctx,
        "num_predict": NUM_PREDICT,
        "no_thinking": True,
        "seed": args.seed,
    }
    request_config["fingerprint"] = _request_fingerprint(request_config)
    request_path = output / "request_manifest.json"
    if request_path.exists():
        current = json.loads(request_path.read_text(encoding="utf-8"))
        if current.get("fingerprint") != request_config["fingerprint"]:
            raise ManifestError("恢复配置与已有 request manifest 不一致")
    else:
        write_json_object(request_path, request_config)

    rollouts_path = output / "teacher_rollouts.jsonl"
    existing = _read_jsonl(rollouts_path) if rollouts_path.exists() else []
    rollout_by_id = {str(row["sample_id"]): row for row in existing}
    all_sources = train_rows + val_rows
    source_by_id = {str(row["sample_id"]): row for row in all_sources}
    if set(rollout_by_id) - set(source_by_id):
        raise ManifestError("rollout 包含当前请求中不存在的 sample_id")

    for index, source in enumerate(all_sources, start=1):
        sample_id = str(source["sample_id"])
        if sample_id in rollout_by_id:
            continue
        reused = reused_by_fingerprint.get(str(source["prompt_fingerprint"]))
        if reused:
            if reused["category"] != source["category"]:
                raise ManifestError("复用样本类别与当前源数据不一致")
            rollout = {
                "sample_id": sample_id,
                "source_split": source["source_split"],
                "category": source["category"],
                "teacher_model": args.teacher_model,
                "accepted": True,
                "teacher_target": reused["teacher_target"],
                "evaluation": {
                    "correct": True,
                    "reused_teacher_verified": True,
                },
                "response": {
                    "reused_teacher_verified": True,
                    "source_path": reused["source_path"],
                    "source_sha256": reused["source_sha256"],
                },
            }
            _append_jsonl(rollouts_path, rollout)
            rollout_by_id[sample_id] = rollout
            print("[{}/{}] {} {} REUSE".format(index, len(all_sources), source["source_split"], source["category"]), flush=True)
            continue
        response = _teacher_chat(
            args.host,
            args.teacher_model,
            source["messages"][:-1],
            str(source["category"]),
            args.num_ctx,
            args.timeout,
        )
        evaluation = evaluate_teacher_output(
            response["text"], source, code_timeout=args.code_timeout
        )
        accepted = bool(evaluation.get("correct"))
        rollout = {
            "sample_id": sample_id,
            "source_split": source["source_split"],
            "category": source["category"],
            "teacher_model": args.teacher_model,
            "accepted": accepted,
            "teacher_target": canonical_teacher_target(
                response["text"], evaluation, str(source["category"])
            )
            if accepted
            else None,
            "evaluation": evaluation,
            "response": response,
        }
        _append_jsonl(rollouts_path, rollout)
        rollout_by_id[sample_id] = rollout
        print(
            "[{}/{}] {} {} {}".format(
                index,
                len(all_sources),
                source["source_split"],
                source["category"],
                "ACCEPT" if accepted else "REJECT",
            ),
            flush=True,
        )

    accepted_train = [
        _output_row(row, rollout_by_id[str(row["sample_id"])])
        for row in train_rows
        if rollout_by_id[str(row["sample_id"])]["accepted"]
    ]
    accepted_val = [
        _output_row(row, rollout_by_id[str(row["sample_id"])])
        for row in val_rows
        if rollout_by_id[str(row["sample_id"])]["accepted"]
    ]
    rejected = [row for row in rollout_by_id.values() if not row["accepted"]]
    train_counts = _category_counts(accepted_train)
    val_counts = _category_counts(accepted_val)
    for category in GENERAL_CATEGORIES:
        if train_counts.get(category, 0) < args.minimum_accepted_per_category:
            raise ManifestError("{} 通过验收的训练样本不足".format(category))
        if val_counts.get(category, 0) < 1:
            raise ManifestError("{} 没有通过验收的验证样本".format(category))

    train_output = output / "train.jsonl"
    val_output = output / "val.jsonl"
    rejected_output = output / "rejected.jsonl"
    _write_jsonl(train_output, accepted_train)
    _write_jsonl(val_output, accepted_val)
    _write_jsonl(rejected_output, rejected)
    manifest = {
        "schema_version": "edge-llm-general-kd/v2",
        "task": "scene_independent_black_box_behavior_distillation",
        "teacher_model": args.teacher_model,
        "teacher_no_thinking": True,
        "train_rows": len(accepted_train),
        "validation_rows": len(accepted_val),
        "rejected_rows": len(rejected),
        "train_category_counts": train_counts,
        "validation_category_counts": val_counts,
        "selected_source_counts": {
            "train": _category_counts(train_rows),
            "validation": _category_counts(val_rows),
        },
        "source_manifest_sha256": sha256_file(source_manifest_path)
        if source_manifest_path
        else None,
        "reused_verified_rows": sum(
            bool(row.get("response", {}).get("reused_teacher_verified"))
            for row in rollout_by_id.values()
        ),
        "reuse_verified_sources": reuse_sources,
        "skipped_source_counts": {"train": train_skipped, "validation": val_skipped},
        "scene_specific_samples": 0,
        "evaluation_prompt_overlap": 0,
        "evaluation_set_used_for_training": False,
        "code_outputs_executed_in_restricted_subprocess": True,
        "request_manifest_sha256": sha256_file(request_path),
        "artifacts": {
            "train": {"path": train_output.name, "sha256": sha256_file(train_output)},
            "validation": {"path": val_output.name, "sha256": sha256_file(val_output)},
            "rejected": {"path": rejected_output.name, "sha256": sha256_file(rejected_output)},
            "rollouts": {"path": rollouts_path.name, "sha256": sha256_file(rollouts_path)},
        },
    }
    write_json_object(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
