"""Strictly gate two frozen scene evaluations produced by one joint adapter."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from edge_llm_factory.build_joint_scene_dataset import SCHEMA_VERSIONS
from edge_llm_factory.contracts import (
    ManifestError,
    read_json_object,
    sha256_file,
    validate_base_manifest,
    write_json_object,
)
from edge_llm_factory.evaluate_action_tokens import JOINT_SCENE_CONTRACTS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _current_file_identity(value: Mapping[str, Any], label: str) -> Dict[str, Any]:
    _require(isinstance(value, Mapping), "{} identity 必须是对象".format(label))
    path = Path(str(value.get("path", ""))).resolve()
    _require(path.is_file(), "{} 文件不存在: {}".format(label, path))
    actual = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    _require(dict(value) == actual, "{} identity 与当前文件不一致".format(label))
    return actual


def _jsonl_rows(path: Path) -> int:
    count = 0
    with Path(path).open("rb") as file_obj:
        for line in file_obj:
            if line.strip():
                count += 1
    return count


def _verified_dataset_artifacts(
    dataset_manifest: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    declared = dataset_manifest.get("artifacts")
    _require(isinstance(declared, Mapping), "dataset manifest 缺少 artifacts")
    output: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "validation"):
        record = declared.get(split)
        _require(isinstance(record, Mapping), "dataset manifest 缺少 {}".format(split))
        path = Path(str(record.get("path", ""))).resolve()
        _require(path.is_file(), "dataset {} 文件不存在".format(split))
        actual = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": _jsonl_rows(path),
        }
        _require(
            actual
            == {
                "path": str(Path(str(record.get("path", ""))).resolve()),
                "bytes": record.get("bytes"),
                "sha256": record.get("sha256"),
                "rows": record.get("rows"),
            },
            "dataset {} 实际 SHA/bytes/rows 与 manifest 不一致".format(split),
        )
        output[split] = actual
    return output


def _classification(
    samples: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> Dict[str, Any]:
    per_class: Dict[str, Any] = {}
    weighted_f1 = 0.0
    for label in labels:
        tp = sum(
            row["target"] == label and row["prediction"] == label
            for row in samples
        )
        fp = sum(
            row["target"] != label and row["prediction"] == label
            for row in samples
        )
        fn = sum(
            row["target"] == label and row["prediction"] != label
            for row in samples
        )
        support = sum(row["target"] == label for row in samples)
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
    count = len(samples)
    return {
        "accuracy": round(
            sum(row["prediction"] == row["target"] for row in samples) / count, 6
        ),
        "macro_f1": round(
            sum(record["f1"] for record in per_class.values()) / len(per_class), 6
        ),
        "weighted_f1": round(weighted_f1 / count, 6),
        "per_class": per_class,
    }


def _strict_samples(
    scene: str,
    report: Mapping[str, Any],
    action_token_ids: Mapping[str, int],
    input_tokens: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    contract = JOINT_SCENE_CONTRACTS[scene]
    expected_count = int(contract["count"])
    allowed = tuple(contract["allowed_tokens"])
    samples = report.get("samples")
    _require(isinstance(samples, list), "{} samples 必须是数组".format(scene))
    _require(
        len(samples) == expected_count,
        "{} samples 长度必须等于 {}".format(scene, expected_count),
    )
    event_ids = []
    normalized = []
    for index, raw in enumerate(samples):
        _require(isinstance(raw, Mapping), "{} sample {} 无效".format(scene, index))
        target = raw.get("target")
        prediction = raw.get("prediction")
        _require(target in allowed, "{} sample target 超出动作槽".format(scene))
        _require(prediction in allowed, "{} sample prediction 超出动作槽".format(scene))
        _require(raw.get("valid") is True, "{} sample valid 不为 true".format(scene))
        _require(
            raw.get("correct") is (prediction == target),
            "{} sample correct 与原始预测不一致".format(scene),
        )
        _require(
            raw.get("prompt_tokens") == input_tokens,
            "{} sample 不是{}-token输入".format(scene, input_tokens),
        )
        generated = raw.get("generated_token_ids")
        _require(
            isinstance(generated, list)
            and len(generated) == 1
            and generated[0] == action_token_ids[prediction],
            "{} sample 不是严格单动作token输出".format(scene),
        )
        event_id = raw.get("event_id")
        _require(
            isinstance(event_id, str) and event_id,
            "{} sample 缺少 event_id".format(scene),
        )
        event_ids.append(event_id)
        normalized.append({"target": target, "prediction": prediction})
    _require(len(set(event_ids)) == expected_count, "{} event_id 不唯一".format(scene))
    _require(
        {row["target"] for row in normalized} == set(allowed),
        "{} 冻结测试 target 类别不完整".format(scene),
    )
    recomputed = _classification(normalized, allowed)
    checks = {
        "count": report.get("count") == expected_count,
        "samples_count": len(samples) == expected_count,
        "all_samples_{}_to_1".format(input_tokens): True,
        "valid_output_rate": report.get("valid_output_rate") == 1.0,
        "accuracy_summary_exact": report.get("decision_accuracy")
        == recomputed["accuracy"],
        "macro_f1_summary_exact": report.get("macro_f1")
        == recomputed["macro_f1"],
        "weighted_f1_summary_exact": report.get("weighted_f1")
        == recomputed["weighted_f1"],
        "per_class_summary_exact": report.get("per_class")
        == recomputed["per_class"],
    }
    _require(all(checks.values()), "{} 汇总指标与 samples 重算结果不一致".format(scene))
    return checks, recomputed


def _strict_constraint(
    scene: str,
    report: Mapping[str, Any],
    action_token_ids: Mapping[str, int],
    reserved_tokens: Mapping[str, str],
) -> Dict[str, bool]:
    allowed = tuple(JOINT_SCENE_CONTRACTS[scene]["allowed_tokens"])
    expected_ids = {token: action_token_ids[token] for token in allowed}
    constraint = report.get("decoding_constraint")
    _require(isinstance(constraint, Mapping), "{} decoding_constraint 缺失".format(scene))
    checks = {
        "enabled": constraint.get("enabled") is True,
        "backend": constraint.get("backend")
        == "transformers_prefix_allowed_tokens_fn",
        "allowed_tokens": constraint.get("allowed_tokens") == list(allowed),
        "allowed_token_ids": constraint.get("allowed_token_ids") == expected_ids,
        "reserved_slots_excluded": constraint.get(
            "reserved_slots_excluded_from_sampling"
        )
        == dict(reserved_tokens),
        "before_sampling": constraint.get("applies_before_sampling") is True,
        "no_posthoc": constraint.get("post_hoc_remapping") is False,
    }
    _require(all(checks.values()), "{} 动作采样约束不严格".format(scene))
    return checks


def _strict_precision(scene: str, report: Mapping[str, Any]) -> Dict[str, bool]:
    precision = report.get("precision")
    _require(isinstance(precision, Mapping), "{} precision 缺失".format(scene))
    dtype_counts = precision.get("floating_parameter_dtypes")
    devices = precision.get("parameter_devices")
    checks = {
        "requested_bf16": precision.get("requested") == "bfloat16",
        "effective_bf16": precision.get("effective") == "bfloat16",
        "only_bf16_parameters": isinstance(dtype_counts, Mapping)
        and set(dtype_counts) == {"torch.bfloat16"}
        and int(dtype_counts["torch.bfloat16"]) > 0,
        "cuda_available": precision.get("cuda_available") is True,
        "only_cuda_parameters": isinstance(devices, Mapping)
        and set(devices) == {"cuda"}
        and int(devices["cuda"]) > 0,
    }
    _require(all(checks.values()), "{} 未证明实际 CUDA BF16".format(scene))
    return checks


def _common_identity(
    traffic: Mapping[str, Any], industrial: Mapping[str, Any], field: str
) -> Dict[str, Any]:
    left = traffic.get(field)
    right = industrial.get(field)
    _require(
        isinstance(left, Mapping) and dict(left) == dict(right or {}),
        "两场景 {} 不一致".format(field),
    )
    return _current_file_identity(left, field)


def gate(
    dataset_manifest: Mapping[str, Any],
    traffic: Mapping[str, Any],
    industrial: Mapping[str, Any],
    dataset_manifest_path: Path,
) -> Dict[str, Any]:
    _require(
        dataset_manifest.get("schema_version") in SCHEMA_VERSIONS,
        "joint dataset schema无效",
    )
    manifest_contract = dataset_manifest.get("contract")
    _require(isinstance(manifest_contract, Mapping), "dataset manifest 缺少 contract")
    input_tokens = manifest_contract.get("input_tokens")
    prefixes = manifest_contract.get("prefixes")
    _require(input_tokens in (16, 17), "dataset input_tokens 必须为16或17")
    _require(isinstance(prefixes, Mapping), "dataset manifest 缺少 prefixes")
    dataset_manifest_identity = {
        "path": str(Path(dataset_manifest_path).resolve()),
        "bytes": Path(dataset_manifest_path).resolve().stat().st_size,
        "sha256": sha256_file(Path(dataset_manifest_path).resolve()),
    }
    _require(
        traffic.get("dataset_manifest_identity") == dataset_manifest_identity
        and industrial.get("dataset_manifest_identity") == dataset_manifest_identity,
        "两份评测未绑定当前 dataset manifest",
    )
    dataset_artifacts = _verified_dataset_artifacts(dataset_manifest)
    _require(
        traffic.get("dataset_artifacts") == dataset_artifacts
        and industrial.get("dataset_artifacts") == dataset_artifacts,
        "两份评测未绑定当前 train/validation SHA/rows",
    )

    evaluator_identity = _common_identity(traffic, industrial, "evaluator_identity")
    base_identity = _common_identity(traffic, industrial, "base_manifest_identity")
    snapshot_identity = _common_identity(
        traffic, industrial, "snapshot_manifest_identity"
    )
    base = validate_base_manifest(read_json_object(Path(base_identity["path"])))
    slots = {str(row["token"]): int(row["token_id"]) for row in base["decision_protocol"]["slots"]}
    reserved = {
        slot: next(
            str(row["token"])
            for row in base["decision_protocol"]["slots"]
            if str(row["slot"]) == slot
        )
        for slot in base["decision_protocol"].get("reserved_slots", {})
    }

    traffic_adapter = traffic.get("adapter_artifacts")
    industrial_adapter = industrial.get("adapter_artifacts")
    _require(
        isinstance(traffic_adapter, Mapping)
        and dict(traffic_adapter) == dict(industrial_adapter or {}),
        "两场景评测不是同一组 Adapter artifacts",
    )
    adapter_artifacts = {
        name: _current_file_identity(traffic_adapter.get(name), "adapter_{}".format(name))
        for name in ("weights", "config", "train_metrics")
    }
    _require(
        traffic.get("expected_adapter_weights_sha256")
        == adapter_artifacts["weights"]["sha256"]
        and industrial.get("expected_adapter_weights_sha256")
        == adapter_artifacts["weights"]["sha256"],
        "评测未绑定预注册 Adapter weights SHA256",
    )
    train_metrics = read_json_object(Path(adapter_artifacts["train_metrics"]["path"]))
    for split, sha_field, rows_field in (
        ("train", "train_jsonl_sha256", "train_rows"),
        ("validation", "val_jsonl_sha256", "val_rows"),
    ):
        artifact = dataset_artifacts[split]
        _require(
            train_metrics.get(sha_field) == artifact["sha256"]
            and train_metrics.get(rows_field) == artifact["rows"],
            "train_metrics 未绑定当前 {} artifact".format(split),
        )
    runtime_contract = train_metrics.get("runtime_contract")
    _require(
        isinstance(runtime_contract, Mapping)
        and runtime_contract.get("one_resident_adapter") is True
        and runtime_contract.get("request_level_adapter_switching") is False
        and runtime_contract.get("input_tokens") == input_tokens
        and runtime_contract.get("output_tokens") == 1,
        "train_metrics 联合单Adapter运行合同无效",
    )

    sources = dataset_manifest.get("sources")
    _require(isinstance(sources, Mapping), "dataset manifest 缺少 sources")
    report_results: Dict[str, Any] = {}
    for scene, report in (("traffic", traffic), ("industrial", industrial)):
        contract = JOINT_SCENE_CONTRACTS[scene]
        expected_prefix = prefixes.get(scene)
        _require(isinstance(expected_prefix, str), "{} prefix 合同无效".format(scene))
        formal_test = sources.get(scene, {}).get("formal_test")
        _require(isinstance(formal_test, Mapping), "{} formal_test 缺失".format(scene))
        common_checks = {
            "mode": report.get("evaluation_mode") == "joint_formal",
            "scene": report.get("joint_scene") == scene,
            "frozen_test_sha256": report.get("test_jsonl_sha256")
            == formal_test.get("sha256"),
            "test_not_training": report.get("test_set_used_for_training") is False,
            "prompt_format": report.get("prompt_format") == "raw_task",
            "prefix": report.get("prompt_prefix") == expected_prefix,
            "required_prompt_tokens": report.get("required_prompt_tokens")
            == input_tokens,
        }
        _require(all(common_checks.values()), "{} 联合评测公共合同无效".format(scene))
        constraint_checks = _strict_constraint(scene, report, slots, reserved)
        precision_checks = _strict_precision(scene, report)
        sample_checks, recomputed = _strict_samples(
            scene, report, slots, int(input_tokens)
        )
        threshold_checks = (
            {
                "accuracy": recomputed["accuracy"] >= 0.66,
                "weighted_f1": recomputed["weighted_f1"] >= 0.65,
                "valid_output_rate": report.get("valid_output_rate") == 1.0,
            }
            if scene == "traffic"
            else {
                "accuracy": recomputed["accuracy"] == 1.0,
                "macro_f1": recomputed["macro_f1"] == 1.0,
                "weighted_f1": recomputed["weighted_f1"] == 1.0,
                "valid_output_rate": report.get("valid_output_rate") == 1.0,
            }
        )
        report_results[scene] = {
            "passed": all(threshold_checks.values()),
            "checks": {
                **common_checks,
                **{"constraint_" + key: value for key, value in constraint_checks.items()},
                **{"precision_" + key: value for key, value in precision_checks.items()},
                **sample_checks,
                **{"threshold_" + key: value for key, value in threshold_checks.items()},
            },
            "recomputed_metrics": recomputed,
        }

    return {
        "schema_version": "edge-llm-joint-traffic-industrial-gate/v2",
        "passed": all(item["passed"] for item in report_results.values()),
        "one_adapter_two_isolated_tests": True,
        "dataset_manifest_identity": dataset_manifest_identity,
        "dataset_artifacts": dataset_artifacts,
        "evaluator_identity": evaluator_identity,
        "base_manifest_identity": base_identity,
        "snapshot_manifest_identity": snapshot_identity,
        "adapter_artifacts": adapter_artifacts,
        "results": report_results,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Gate one traffic+industrial adapter.")
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument("--traffic_evaluation", required=True)
    parser.add_argument("--industrial_evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise ManifestError("拒绝覆盖已有联合 gate 输出: {}".format(output_path))
    dataset_manifest_path = Path(args.dataset_manifest).resolve()
    result = gate(
        read_json_object(dataset_manifest_path),
        read_json_object(Path(args.traffic_evaluation)),
        read_json_object(Path(args.industrial_evaluation)),
        dataset_manifest_path,
    )
    write_json_object(output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
