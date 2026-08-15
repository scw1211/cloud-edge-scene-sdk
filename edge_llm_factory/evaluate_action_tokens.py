"""用途：在未见 JSONL 上评估场景 LoRA 的单 token 准确率、F1、有效率和生成时延。"""

import argparse
from collections import Counter
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from edge_llm_factory.action_constraints import normalize_action_tokens
from edge_llm_factory.build_joint_scene_dataset import SCHEMA_VERSIONS
from edge_llm_factory.adapter_package import MANIFEST_NAME, validate_adapter_package
from edge_llm_factory.text_base import verify_text_snapshot
from edge_llm_factory.contracts import (
    ManifestError,
    read_json_object,
    sha256_file,
    validate_base_manifest,
    write_json_object,
)


JOINT_SCENE_CONTRACTS = {
    "traffic": {
        "prefix": "T",
        "allowed_tokens": ("A", "B", "C", "D", "E", "F"),
        "count": 2400,
    },
    "industrial": {
        "prefix": "I",
        "allowed_tokens": ("A", "B", "C"),
        "count": 960,
    },
}
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def _file_identity(path: Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ManifestError("证据文件不存在: {}".format(resolved))
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _jsonl_row_count(path: Path) -> int:
    count = 0
    with Path(path).open("rb") as file_obj:
        for line in file_obj:
            if line.strip():
                count += 1
    return count


def _verified_dataset_artifact(
    manifest: Mapping[str, Any], split: str
) -> Dict[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(
        artifacts.get(split), Mapping
    ):
        raise ManifestError("joint dataset manifest 缺少 {} artifact".format(split))
    declared = artifacts[split]
    identity = _file_identity(Path(str(declared.get("path", ""))))
    rows = _jsonl_row_count(Path(identity["path"]))
    if identity["sha256"] != declared.get("sha256"):
        raise ManifestError("joint dataset {} SHA256 与 manifest 不一致".format(split))
    if identity["bytes"] != declared.get("bytes"):
        raise ManifestError("joint dataset {} bytes 与 manifest 不一致".format(split))
    if rows != declared.get("rows"):
        raise ManifestError("joint dataset {} rows 与 manifest 不一致".format(split))
    return {**identity, "rows": rows}


def _joint_evaluation_context(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    requested = any(
        value
        for value in (
            args.joint_scene,
            args.dataset_manifest,
            args.expected_adapter_weights_sha256,
        )
    )
    if not requested:
        return None
    if not (
        args.joint_scene
        and args.dataset_manifest
        and args.expected_adapter_weights_sha256
    ):
        raise ManifestError(
            "联合正式评测必须同时提供 joint_scene、dataset_manifest 和 "
            "expected_adapter_weights_sha256"
        )
    if SHA256_HEX.fullmatch(args.expected_adapter_weights_sha256) is None:
        raise ManifestError("expected_adapter_weights_sha256 必须是小写 SHA256")
    contract = JOINT_SCENE_CONTRACTS[args.joint_scene]
    if args.limit != 0:
        raise ManifestError("联合正式评测禁止 limit，必须运行完整冻结测试集")
    if not args.bf16:
        raise ManifestError("联合正式评测必须请求 BF16")
    if args.prompt_format != "raw_task":
        raise ManifestError("联合正式评测必须使用 raw_task")
    if not args.constrain_action_tokens:
        raise ManifestError("联合正式评测必须启用采样前动作白名单")
    allowed = _parse_allowed_action_tokens(args.allowed_action_tokens)
    if tuple(allowed) != tuple(contract["allowed_tokens"]):
        raise ManifestError("联合正式评测动作白名单与场景不一致")

    manifest_path = Path(args.dataset_manifest).resolve()
    manifest = read_json_object(manifest_path)
    if manifest.get("schema_version") not in SCHEMA_VERSIONS:
        raise ManifestError("joint dataset manifest schema_version 不匹配")
    manifest_contract = manifest.get("contract")
    if not isinstance(manifest_contract, Mapping):
        raise ManifestError("joint dataset manifest 缺少 contract")
    input_tokens = manifest_contract.get("input_tokens")
    prefixes = manifest_contract.get("prefixes")
    expected_prefix = (
        prefixes.get(args.joint_scene) if isinstance(prefixes, Mapping) else None
    )
    if (
        input_tokens not in (16, 17)
        or manifest_contract.get("output_tokens") != 1
        or not isinstance(expected_prefix, str)
    ):
        raise ManifestError("joint dataset manifest 的输入输出合同不匹配")
    if args.prompt_prefix != expected_prefix:
        raise ManifestError("联合正式评测 prompt_prefix 与 manifest 场景合同不一致")
    if args.required_prompt_tokens != input_tokens:
        raise ManifestError(
            "联合正式评测必须严格要求 {} input tokens".format(input_tokens)
        )
    dataset_artifacts = {
        split: _verified_dataset_artifact(manifest, split)
        for split in ("train", "validation")
    }
    sources = manifest.get("sources")
    source = sources.get(args.joint_scene) if isinstance(sources, Mapping) else None
    formal_test = source.get("formal_test") if isinstance(source, Mapping) else None
    if not isinstance(formal_test, Mapping):
        raise ManifestError("joint dataset manifest 缺少场景 formal_test")
    if (
        formal_test.get("rows") != contract["count"]
        or formal_test.get("content_loaded") is not False
        or formal_test.get("used_for_training") is not False
        or formal_test.get("used_for_validation") is not False
        or formal_test.get("access_mode") != "sha256_and_stat_only"
    ):
        raise ManifestError("joint dataset formal_test 隔离合同无效")
    return {
        "scene": args.joint_scene,
        "contract": contract,
        "manifest": manifest,
        "dataset_manifest": _file_identity(manifest_path),
        "dataset_artifacts": dataset_artifacts,
        "formal_test": dict(formal_test),
        "input_tokens": int(input_tokens),
        "prompt_prefix": expected_prefix,
    }


def _rows(path: Path) -> List[Dict[str, Any]]:
    output = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError("测试集第 {} 行不是合法 JSON".format(line_number)) from exc
            if not isinstance(row, dict):
                raise ManifestError("测试集每行必须是对象")
            output.append(row)
    if not output:
        raise ManifestError("测试集为空")
    return output


def _messages(row: Mapping[str, Any]) -> List[Dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ManifestError("测试样本缺少 messages")
    result = [
        {"role": str(item.get("role")), "content": str(item.get("content"))}
        for item in messages
        if isinstance(item, dict)
    ]
    if len(result) != len(messages) or result[-1]["role"] != "assistant":
        raise ManifestError("测试样本最后一条消息必须是 assistant target")
    return result


def _prompt(row: Mapping[str, Any], tokenizer: Any, prompt_format: str) -> str:
    messages = _messages(row)
    if prompt_format == "raw_task":
        users = [message for message in messages[:-1] if message["role"] == "user"]
        if len(users) != 1:
            raise ManifestError("raw_task 测试样本必须恰好有一条 user 消息")
        return users[0]["content"]
    return tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )


def _classification(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> Dict[str, Any]:
    per_class = {}
    weighted_f1 = 0.0
    for label in labels:
        tp = sum(row["target"] == label and row["prediction"] == label for row in rows)
        fp = sum(row["target"] != label and row["prediction"] == label for row in rows)
        fn = sum(row["target"] == label and row["prediction"] != label for row in rows)
        support = sum(row["target"] == label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        weighted_f1 += support * f1
    count = len(rows)
    return {
        "accuracy": round(sum(row["correct"] for row in rows) / count, 6),
        "macro_f1": round(statistics.fmean(item["f1"] for item in per_class.values()), 6),
        "weighted_f1": round(weighted_f1 / count, 6),
        "per_class": per_class,
    }


def _parse_allowed_action_tokens(value: str) -> tuple:
    tokens = tuple(part.strip() for part in str(value).split(",") if part.strip())
    return normalize_action_tokens(tokens)


def _hf_action_token_ids(
    tokenizer: Any,
    allowed_tokens: Sequence[str],
    declared_token_ids: Mapping[str, int],
) -> Dict[str, int]:
    """Resolve and verify the exact tokenizer ids used by the generation whitelist."""

    resolved: Dict[str, int] = {}
    for token in allowed_tokens:
        encoded = tokenizer(token, add_special_tokens=False)
        token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        if not isinstance(token_ids, list) or len(token_ids) != 1:
            raise ManifestError(
                "约束动作 {!r} 不是 tokenizer 的单 token".format(token)
            )
        token_id = int(token_ids[0])
        if token not in declared_token_ids:
            raise ManifestError("约束动作未在基座协议中声明: {!r}".format(token))
        if token_id != int(declared_token_ids[token]):
            raise ManifestError(
                "约束动作 {!r} 的 tokenizer id 与基座协议不一致".format(token)
            )
        resolved[token] = token_id
    return resolved


def _hf_constraint_kwargs(token_ids: Mapping[str, int]) -> Dict[str, Any]:
    """Build the Transformers pre-sampling whitelist callback."""

    allowed_ids = tuple(int(token_id) for token_id in token_ids.values())
    if not allowed_ids:
        raise ManifestError("动作 token id 白名单不能为空")

    def prefix_allowed_tokens_fn(_batch_id: int, _input_ids: Any) -> List[int]:
        return list(allowed_ids)

    return {"prefix_allowed_tokens_fn": prefix_allowed_tokens_fn}


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="评估标准单 token 场景 LoRA。")
    parser.add_argument("--base", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--snapshot_manifest", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--test_dataset_id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt_format", choices=["tokenizer_chat", "raw_task"], default="raw_task")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--prompt_prefix",
        default="",
        help="可选的原始任务前缀；联合场景适配器使用 T 或 I。",
    )
    parser.add_argument(
        "--required_prompt_tokens",
        type=int,
        default=0,
        help="大于 0 时要求每条输入严格等于该 token 数。",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--joint_scene",
        choices=sorted(JOINT_SCENE_CONTRACTS),
        default="",
        help="启用绑定数据清单、模型身份和 BF16 的联合正式评测。",
    )
    parser.add_argument(
        "--dataset_manifest",
        default="",
        help="联合正式训练数据 manifest；必须与 joint_scene 同时使用。",
    )
    parser.add_argument(
        "--expected_adapter_weights_sha256",
        default="",
        help="联合正式候选 adapter_model.safetensors 的冻结 SHA256。",
    )
    parser.add_argument(
        "--constrain_action_tokens",
        action="store_true",
        help="在首个生成 token 的 logits 上只允许动作白名单；默认关闭以保留历史口径。",
    )
    parser.add_argument(
        "--allowed_action_tokens",
        default="A,B,C,D,E,F",
        help="启用约束时允许的单 token 动作，逗号分隔。",
    )
    args = parser.parse_args(argv)

    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise ManifestError("拒绝覆盖已有评测输出: {}".format(output_path))
    joint = _joint_evaluation_context(args)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_path = Path(args.base).resolve()
    base = validate_base_manifest(read_json_object(base_path))
    snapshot = Path(args.snapshot).resolve()
    snapshot_manifest_path = Path(args.snapshot_manifest).resolve()
    snapshot_report = verify_text_snapshot(
        base,
        read_json_object(snapshot_manifest_path),
        snapshot,
        verify_tokenizer=True,
    )
    adapter = Path(args.adapter).resolve()
    if (adapter / MANIFEST_NAME).is_file():
        validate_adapter_package(adapter, base_path, require_gates=False)
    adapter_weights_path = adapter / "adapter_model.safetensors"
    adapter_config_path = adapter / "adapter_config.json"
    train_summary_path = adapter / "train_metrics.json"
    adapter_artifacts = {
        "weights": _file_identity(adapter_weights_path),
        "config": _file_identity(adapter_config_path),
        "train_metrics": (
            _file_identity(train_summary_path)
            if train_summary_path.is_file()
            else None
        ),
    }
    if joint is not None:
        if adapter_artifacts["train_metrics"] is None:
            raise ManifestError("联合正式候选缺少 train_metrics.json")
        if (
            adapter_artifacts["weights"]["sha256"]
            != args.expected_adapter_weights_sha256
        ):
            raise ManifestError("联合正式候选 Adapter weights SHA256 不匹配")
        training = read_json_object(train_summary_path)
        for split, metric_field, row_field in (
            ("train", "train_jsonl_sha256", "train_rows"),
            ("validation", "val_jsonl_sha256", "val_rows"),
        ):
            artifact = joint["dataset_artifacts"][split]
            if (
                training.get(metric_field) != artifact["sha256"]
                or training.get(row_field) != artifact["rows"]
            ):
                raise ManifestError(
                    "Adapter train_metrics 未绑定 joint dataset {} artifact".format(
                        split
                    )
                )
    test_path = Path(args.test_jsonl).resolve()
    test_sha = sha256_file(test_path)
    if train_summary_path.is_file():
        training = read_json_object(train_summary_path)
        if test_sha in {training.get("train_jsonl_sha256"), training.get("val_jsonl_sha256")}:
            raise ManifestError("测试集与训练集或调参验证集完全相同")
    if joint is not None:
        formal_test = joint["formal_test"]
        if test_path != Path(str(formal_test.get("path", ""))).resolve():
            raise ManifestError("联合正式 test_jsonl 路径与 dataset manifest 不一致")
        if test_sha != formal_test.get("sha256"):
            raise ManifestError("联合正式 test_jsonl SHA256 与 dataset manifest 不一致")
        if test_path.stat().st_size != formal_test.get("bytes"):
            raise ManifestError("联合正式 test_jsonl bytes 与 dataset manifest 不一致")
    rows = _rows(test_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    if joint is not None and len(rows) != joint["contract"]["count"]:
        raise ManifestError("联合正式 test_jsonl 样本数与冻结合同不一致")
    slot_rows = [dict(row) for row in base["decision_protocol"]["slots"]]
    slots = {str(row["token"]) for row in slot_rows}
    slot_tokens = {str(row["slot"]): str(row["token"]) for row in slot_rows}
    reserved_tokens = {
        slot: slot_tokens[slot]
        for slot in base["decision_protocol"].get("reserved_slots", {})
        if slot in slot_tokens
    }
    targets = []
    for row in rows:
        target = _messages(row)[-1]["content"].strip()
        if target not in slots:
            raise ManifestError("测试 target 不在基座动作槽中: {}".format(target))
        targets.append(target)

    allowed_tokens = _parse_allowed_action_tokens(args.allowed_action_tokens)
    if args.constrain_action_tokens and set(allowed_tokens) & set(
        reserved_tokens.values()
    ):
        raise ManifestError("约束动作白名单不得包含基座保留动作槽")
    if args.constrain_action_tokens and not set(targets).issubset(allowed_tokens):
        raise ManifestError("测试 target 超出约束动作白名单")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, use_fast=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    constrained_token_ids: Dict[str, int] = {}
    if args.constrain_action_tokens:
        constrained_token_ids = _hf_action_token_ids(
            tokenizer,
            allowed_tokens,
            {str(row["token"]): int(row["token_id"]) for row in slot_rows},
        )
    model_kwargs: Dict[str, Any] = {"local_files_only": True, "trust_remote_code": False}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), **model_kwargs)
    model = PeftModel.from_pretrained(
        model, str(adapter), autocast_adapter_dtype=False
    )
    if args.bf16:
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    floating_parameter_dtypes = Counter(
        str(parameter.dtype)
        for parameter in model.parameters()
        if parameter.is_floating_point()
    )
    parameter_devices = Counter(
        parameter.device.type for parameter in model.parameters()
    )
    effective_precision = (
        "bfloat16"
        if set(floating_parameter_dtypes) == {"torch.bfloat16"}
        else "mixed_or_other"
    )
    precision = {
        "requested": "bfloat16" if args.bf16 else "default",
        "effective": effective_precision,
        "floating_parameter_dtypes": dict(sorted(floating_parameter_dtypes.items())),
        "parameter_devices": dict(sorted(parameter_devices.items())),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if joint is not None and (
        effective_precision != "bfloat16"
        or not torch.cuda.is_available()
        or set(parameter_devices) != {"cuda"}
    ):
        raise ManifestError("联合正式评测没有实际运行于 CUDA BF16")

    def generate(row: Mapping[str, Any]) -> Dict[str, Any]:
        prompt = args.prompt_prefix + _prompt(row, tokenizer, args.prompt_format)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )
        if (
            args.required_prompt_tokens > 0
            and int(encoded["input_ids"].shape[1]) != args.required_prompt_tokens
        ):
            raise ManifestError(
                "评测输入不是严格 {} tokens".format(args.required_prompt_tokens)
            )
        encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}
        started = time.perf_counter()
        generation_kwargs: Dict[str, Any] = {}
        if args.constrain_action_tokens:
            generation_kwargs = _hf_constraint_kwargs(constrained_token_ids)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **generation_kwargs,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        token = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        generated_token_ids = [
            int(value)
            for value in generated[0, encoded["input_ids"].shape[1] :].tolist()
        ]
        return {
            "prediction": token if token in slots else None,
            "raw_output": token,
            "latency_ms": round(latency_ms, 4),
            "prompt_tokens": int(encoded["input_ids"].shape[1]),
            "generated_token_ids": generated_token_ids,
        }

    for index in range(max(0, args.warmup)):
        generate(rows[index % len(rows)])
    samples = []
    for row, target in zip(rows, targets):
        result = generate(row)
        result.update(
            {
                "event_id": row.get("event_id"),
                "target": target,
                "valid": result["prediction"] is not None,
                "correct": result["prediction"] == target,
            }
        )
        samples.append(result)
    labels = sorted(set(targets))
    classification = _classification(samples, labels)
    summary = {
        "task": "edge_llm_single_token_heldout_evaluation",
        "base_id": base["base_id"],
        "snapshot_validation": snapshot_report,
        "adapter": str(adapter),
        "evaluation_mode": (
            "joint_formal" if joint is not None else "standard"
        ),
        "joint_scene": joint["scene"] if joint is not None else None,
        "expected_adapter_weights_sha256": (
            args.expected_adapter_weights_sha256 if joint is not None else None
        ),
        "evaluator_identity": _file_identity(Path(__file__)),
        "base_manifest_identity": _file_identity(base_path),
        "snapshot_manifest_identity": _file_identity(snapshot_manifest_path),
        "adapter_artifacts": adapter_artifacts,
        "dataset_manifest_identity": (
            joint["dataset_manifest"] if joint is not None else None
        ),
        "dataset_artifacts": (
            joint["dataset_artifacts"] if joint is not None else None
        ),
        "precision": precision,
        "test_dataset_id": args.test_dataset_id,
        "test_jsonl_sha256": test_sha,
        "test_set_used_for_training": False,
        "count": len(samples),
        "prompt_format": args.prompt_format,
        "prompt_prefix": args.prompt_prefix,
        "required_prompt_tokens": args.required_prompt_tokens or None,
        "decoding_constraint": {
            "enabled": bool(args.constrain_action_tokens),
            "backend": (
                "transformers_prefix_allowed_tokens_fn"
                if args.constrain_action_tokens
                else "unconstrained_greedy"
            ),
            "allowed_tokens": list(allowed_tokens)
            if args.constrain_action_tokens
            else None,
            "allowed_token_ids": constrained_token_ids
            if args.constrain_action_tokens
            else None,
            "reserved_slots_excluded_from_sampling": reserved_tokens
            if args.constrain_action_tokens
            else None,
            "applies_before_sampling": bool(args.constrain_action_tokens),
            "post_hoc_remapping": False,
        },
        "valid_output_rate": round(sum(row["valid"] for row in samples) / len(samples), 6),
        "decision_accuracy": classification["accuracy"],
        "macro_f1": classification["macro_f1"],
        "weighted_f1": classification["weighted_f1"],
        "per_class": classification["per_class"],
        "average_generation_latency_ms": round(
            statistics.fmean(row["latency_ms"] for row in samples), 4
        ),
        "samples": samples,
    }
    write_json_object(output_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
