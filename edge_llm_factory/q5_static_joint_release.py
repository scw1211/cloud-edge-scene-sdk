"""Fail-closed release packager for one statically fused joint Q5_K_M model.

The runtime artifact already contains the jointly trained traffic/industrial
LoRA delta.  Consequently this module deliberately accepts no runtime adapter
and emits ``runtime_adapters: []``.  The original PEFT files remain in the
standard package solely as auditable training/merge lineage.

All source evidence is checked before the generic package builder is called.
The generic package is built in a private staging directory, hardened with the
static-runtime binding, validated again, and only then atomically published.
This module never writes a release registry and never contacts a runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from edge_llm_factory.adapter_package import (
    MANIFEST_NAME,
    build_adapter_package,
    validate_adapter_package,
)
from edge_llm_factory.contracts import (
    ManifestError,
    read_json_object,
    sha256_file,
    write_json_object,
)


SCHEMA = "edge-llm-q5-static-joint-release-input/v1"
SUMMARY_SCHEMA = "edge-llm-q5-static-joint-release-gate-summary/v1"
PACKAGE_KIND = "joint-static-fused-model-release"
MODEL_MODE = "joint_static_fusion"
CONTEXT_ENCODER = "scene-prefixed-decimal17@v1"
MAX_INPUT_TOKENS = 17
FORMAL_GATE_SCHEMA = "edge-llm-joint-traffic-industrial-gate/v2"
STATIC_PROVENANCE_SCHEMA = "edge-llm-joint-static-fusion-provenance/v1"
STATIC_CANDIDATE_SCHEMA = "edge-llm-joint-static-fusion-candidate-summary/v1"
STATIC_FULL_SCHEMA = "edge-llm-joint-static-fusion-full-evaluation/v1"
NANO_OBSERVATION_SCHEMA = "edge-llm-joint-static-fusion-alternating-observation/v1"
NANO_GATE_SCHEMA = "q5km-static-fusion-nano-observational-memory-gate/v1"

TRAFFIC_SCENE = "freeway_traffic_management"
TRAFFIC_MIN_ACCURACY = 0.66
TRAFFIC_MIN_WEIGHTED_F1 = 0.65
MAX_NANO_STRICT_PEAK_BYTES = 1_500_000_000
MIN_NANO_MEM_AVAILABLE_BYTES = 256 * 1024 * 1024
MAX_NANO_GROWTH_BYTES = 32 * 1024 * 1024
MAX_MEAN_LATENCY_MS = 200.0

NANO_SHA_FILES = {
    "EXECUTION_PROTOCOL.json",
    "IMPLEMENTATION_SHA256SUMS.txt",
    "INPUT_SHA256SUMS.txt",
    "RUN_STATUS.json",
    "benchmark.pid",
    "candidate_after.txt",
    "execution_inputs.json",
    "formal_after.txt",
    "formal_before.txt",
    "inference_memory_samples_20ms.jsonl",
    "llama_pid.txt",
    "lora_adapters_after.json",
    "lora_adapters_before.json",
    "nano222_joint_q5_static_fusion_observed_500_v2.json",
    "observed_gate_summary.json",
    "props_after.json",
    "props_before.json",
    "props_contract_after.json",
    "props_contract_before.json",
    "q5_static_fusion_full_gate.json",
    "restore_formal.sh",
    "sidecar_cmdline.txt",
    "startup_memory_samples_20ms.jsonl",
    "startup_phase_summary.json",
}

NANO_REQUIRED_CHECKS = {
    "completed_500",
    "growth_100_to_500_le_32MiB",
    "industrial_accuracy_1",
    "industrial_macro_f1_1",
    "industrial_weighted_f1_1",
    "minimum_mem_available_ge_256MiB",
    "oom_delta_zero",
    "overall_mean_latency_lt_200ms",
    "runtime_lora_count_zero_before_after",
    "strict_17_to_1_and_no_request_lora",
    "strict_alternation_no_crosstalk",
    "strict_peak_le_1500000000",
    "target_D_consecutive_below_5_during_inference",
    "traffic_accuracy_ge_0_66",
    "traffic_weighted_f1_ge_0_65",
    "valid_outputs_1",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _path(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ManifestError("evidence is missing {}".format(dotted))
        current = current[part]
    return current


def _equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ManifestError(
            "{} mismatch: expected {!r}, got {!r}".format(field, expected, actual)
        )


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError("{} must be numeric".format(field))
    result = float(value)
    if not math.isfinite(result):
        raise ManifestError("{} must be finite".format(field))
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError("{} must be an integer".format(field))
    return int(value)


def _close(actual: Any, expected: float, field: str, tolerance: float = 1e-8) -> float:
    value = _number(actual, field)
    if abs(value - float(expected)) > tolerance:
        raise ManifestError(
            "{} mismatch: expected {}, got {}".format(field, expected, value)
        )
    return value


def _at_least(actual: Any, minimum: float, field: str) -> float:
    value = _number(actual, field)
    if value < minimum:
        raise ManifestError("{} is {} but must be >= {}".format(field, value, minimum))
    return value


def _at_most(actual: Any, maximum: float, field: str) -> float:
    value = _number(actual, field)
    if value > maximum:
        raise ManifestError("{} is {} but must be <= {}".format(field, value, maximum))
    return value


def _resolve(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError("{} must be a non-empty path".format(field))
    raw = Path(value).expanduser()
    return (raw if raw.is_absolute() else root / raw).resolve()


def _actual_identity(path: Path, field: str) -> Dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ManifestError("{} must be a regular non-symlink file: {}".format(field, path))
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _portable(identity: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "name": Path(str(identity["path"])).name,
        "bytes": identity["bytes"],
        "sha256": identity["sha256"],
    }


def _declared_identity(root: Path, value: Any, field: str) -> Tuple[Path, Dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ManifestError("{} must be an identity object".format(field))
    path = _resolve(root, value.get("path"), field + ".path")
    actual = _actual_identity(path, field)
    _equal(value.get("bytes"), actual["bytes"], field + ".bytes")
    _equal(value.get("sha256"), actual["sha256"], field + ".sha256")
    return path, actual


def _identity_equal(record: Any, expected: Mapping[str, Any], field: str) -> None:
    if not isinstance(record, Mapping):
        raise ManifestError("{} identity is missing".format(field))
    _equal(record.get("bytes"), expected["bytes"], field + ".bytes")
    _equal(record.get("sha256"), expected["sha256"], field + ".sha256")


def _read_json_value(path: Path, field: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError("{} is not valid JSON: {}".format(field, path)) from exc


def _read_jsonl(path: Path, field: str) -> list:
    rows = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ManifestError("{} line {} must be an object".format(field, line_number))
            rows.append(dict(row))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError("{} is not valid JSONL".format(field)) from exc
    _require(bool(rows), "{} must not be empty".format(field))
    return rows


def _classification(records: Sequence[Mapping[str, Any]], labels: str) -> Dict[str, float]:
    count = len(records)
    _require(count > 0, "classification records must not be empty")
    supports = {label: 0 for label in labels}
    true_positive = {label: 0 for label in labels}
    predicted = {label: 0 for label in labels}
    correct = 0
    valid = 0
    for row in records:
        target = row.get("target")
        prediction = row.get("prediction")
        _require(target in labels, "record target is outside {}".format(labels))
        _require(prediction in labels, "record prediction is outside {}".format(labels))
        supports[target] += 1
        predicted[prediction] += 1
        if prediction == target:
            correct += 1
            true_positive[target] += 1
        if row.get("valid", row.get("valid_output")) is True:
            valid += 1
    f1_values = []
    weighted = 0.0
    for label in labels:
        precision = true_positive[label] / predicted[label] if predicted[label] else 0.0
        recall = true_positive[label] / supports[label] if supports[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        weighted += f1 * supports[label]
    return {
        "count": float(count),
        "correct": float(correct),
        "accuracy": correct / count,
        "macro_f1": sum(f1_values) / len(f1_values),
        "weighted_f1": weighted / count,
        "valid_output_rate": valid / count,
    }


def _training_artifacts(adapter_source: Path) -> Dict[str, Dict[str, Any]]:
    artifacts = {
        "weights": _actual_identity(adapter_source / "adapter_model.safetensors", "training adapter weights"),
        "config": _actual_identity(adapter_source / "adapter_config.json", "training adapter config"),
        "train_metrics": _actual_identity(adapter_source / "train_metrics.json", "training adapter metrics"),
    }
    config = read_json_object(adapter_source / "adapter_config.json")
    _equal(config.get("peft_type"), "LORA", "training adapter peft_type")
    _equal(config.get("task_type"), "CAUSAL_LM", "training adapter task_type")
    metrics = read_json_object(adapter_source / "train_metrics.json")
    _equal(metrics.get("task"), "joint_traffic_industrial_single_adapter_sft", "train_metrics.task")
    _equal(metrics.get("formal_test_content_loaded"), False, "train_metrics.formal_test_content_loaded")
    _equal(metrics.get("formal_test_used_for_training_or_selection"), False, "train_metrics.formal_test_used")
    _equal(metrics.get("test_set_used_for_training"), False, "train_metrics.test_set_used")
    _equal(_path(metrics, "runtime_contract.input_tokens"), 17, "train_metrics.input_tokens")
    _equal(_path(metrics, "runtime_contract.output_tokens"), 1, "train_metrics.output_tokens")
    _equal(_path(metrics, "runtime_contract.request_level_adapter_switching"), False, "train_metrics.request_switching")
    _equal(_path(metrics, "snapshot_validation.status"), "valid", "train_metrics.snapshot_status")
    _equal(_path(metrics, "snapshot_validation.tensor_report.multimodal_parameter_count"), 0, "train_metrics.multimodal_params")
    return artifacts


def _validate_formal_report(
    scene: str, report: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]
) -> Dict[str, float]:
    traffic = scene == "traffic"
    labels = "ABCDEF" if traffic else "ABC"
    count = 2400 if traffic else 960
    _equal(report.get("evaluation_mode"), "joint_formal", scene + ".evaluation_mode")
    _equal(report.get("joint_scene"), scene, scene + ".joint_scene")
    _equal(report.get("prompt_format"), "raw_task", scene + ".prompt_format")
    _equal(report.get("prompt_prefix"), "T" if traffic else "I", scene + ".prompt_prefix")
    _equal(report.get("required_prompt_tokens"), 17, scene + ".prompt_tokens")
    _equal(report.get("count"), count, scene + ".count")
    _equal(report.get("valid_output_rate"), 1.0, scene + ".valid_output_rate")
    _equal(report.get("test_set_used_for_training"), False, scene + ".test_set_used")
    _equal(report.get("expected_adapter_weights_sha256"), artifacts["weights"]["sha256"], scene + ".adapter_sha")
    formal_artifacts = report.get("adapter_artifacts")
    _require(isinstance(formal_artifacts, Mapping), scene + " adapter_artifacts missing")
    for name, identity in artifacts.items():
        _identity_equal(formal_artifacts.get(name), identity, scene + ".adapter_artifacts." + name)
    precision = report.get("precision")
    _require(isinstance(precision, Mapping), scene + " precision missing")
    _equal(precision.get("requested"), "bfloat16", scene + ".precision.requested")
    _equal(precision.get("effective"), "bfloat16", scene + ".precision.effective")
    _equal(precision.get("cuda_available"), True, scene + ".precision.cuda")
    constraint = report.get("decoding_constraint")
    _require(isinstance(constraint, Mapping), scene + " constraint missing")
    _equal(constraint.get("enabled"), True, scene + ".constraint.enabled")
    _equal(constraint.get("allowed_tokens"), list(labels), scene + ".constraint.tokens")
    _equal(constraint.get("applies_before_sampling"), True, scene + ".constraint.before_sampling")
    _equal(constraint.get("post_hoc_remapping"), False, scene + ".constraint.post_hoc")
    samples = report.get("samples")
    _require(isinstance(samples, list) and len(samples) == count, scene + " samples count invalid")
    for index, row in enumerate(samples):
        _require(isinstance(row, Mapping), "{}.samples[{}] invalid".format(scene, index))
        _equal(row.get("prompt_tokens"), 17, "{}.samples[{}].prompt_tokens".format(scene, index))
        _equal(row.get("valid"), True, "{}.samples[{}].valid".format(scene, index))
        generated = row.get("generated_token_ids")
        _require(isinstance(generated, list) and len(generated) == 1, "{}.samples[{}] output count".format(scene, index))
        _require(row.get("prediction") in labels, "{}.samples[{}] prediction slot".format(scene, index))
    accuracy = _number(report.get("decision_accuracy"), scene + ".accuracy")
    macro_f1 = _number(report.get("macro_f1"), scene + ".macro_f1")
    weighted_f1 = _number(report.get("weighted_f1"), scene + ".weighted_f1")
    if traffic:
        _at_least(accuracy, TRAFFIC_MIN_ACCURACY, scene + ".accuracy")
        _at_least(weighted_f1, TRAFFIC_MIN_WEIGHTED_F1, scene + ".weighted_f1")
    else:
        _equal(accuracy, 1.0, scene + ".accuracy")
        _equal(macro_f1, 1.0, scene + ".macro_f1")
        _equal(weighted_f1, 1.0, scene + ".weighted_f1")
    return {
        "bf16_{}_accuracy".format(scene): accuracy,
        "bf16_{}_macro_f1".format(scene): macro_f1,
        "bf16_{}_weighted_f1".format(scene): weighted_f1,
        "bf16_{}_valid_output_rate".format(scene): 1.0,
    }


def _validate_formal(
    traffic: Mapping[str, Any],
    industrial: Mapping[str, Any],
    gate: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> Dict[str, float]:
    metrics = {}
    metrics.update(_validate_formal_report("traffic", traffic, artifacts))
    metrics.update(_validate_formal_report("industrial", industrial, artifacts))
    _equal(gate.get("schema_version"), FORMAL_GATE_SCHEMA, "formal_gate.schema_version")
    _equal(gate.get("passed"), True, "formal_gate.passed")
    _equal(gate.get("one_adapter_two_isolated_tests"), True, "formal_gate.one_adapter")
    for name, identity in artifacts.items():
        _identity_equal(_path(gate, "adapter_artifacts." + name), identity, "formal_gate.adapter." + name)
    for field in (
        "base_manifest_identity",
        "snapshot_manifest_identity",
        "dataset_manifest_identity",
        "dataset_artifacts",
        "evaluator_identity",
    ):
        _equal(traffic.get(field), industrial.get(field), "formal shared " + field)
        _equal(gate.get(field), traffic.get(field), "formal_gate." + field)
    for scene in ("traffic", "industrial"):
        result = _path(gate, "results." + scene)
        _equal(result.get("passed"), True, "formal_gate.{}.passed".format(scene))
        checks = result.get("checks")
        _require(isinstance(checks, Mapping) and checks and all(v is True for v in checks.values()), "formal_gate.{} checks failed".format(scene))
    metrics["bf16_completed_requests"] = 3360.0
    return metrics


def _validate_provenance(
    provenance: Mapping[str, Any],
    candidate: Mapping[str, Any],
    model: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    full_identity: Mapping[str, Any],
) -> None:
    _equal(provenance.get("schema_version"), STATIC_PROVENANCE_SCHEMA, "provenance.schema_version")
    merge = _path(provenance, "merge_contract")
    _equal(merge.get("operation"), "PEFT PeftModel.merge_and_unload", "provenance.merge.operation")
    _equal(merge.get("clean_snapshot_only"), True, "provenance.merge.clean_snapshot")
    _equal(merge.get("one_joint_adapter"), True, "provenance.merge.one_joint_adapter")
    _equal(merge.get("request_level_lora_after_merge"), False, "provenance.merge.runtime_lora")
    _identity_equal(_path(provenance, "inputs.adapter_weights"), artifacts["weights"], "provenance.adapter_weights")
    _identity_equal(_path(provenance, "inputs.adapter_config"), artifacts["config"], "provenance.adapter_config")
    _identity_equal(_path(provenance, "inputs.adapter_train_metrics"), artifacts["train_metrics"], "provenance.train_metrics")
    _equal(_path(provenance, "post_evaluation_state.q5_full_passed"), True, "provenance.q5_full_passed")
    _equal(_path(provenance, "post_evaluation_state.nano_gate_authorized"), True, "provenance.nano_gate_authorized")

    _equal(candidate.get("schema_version"), STATIC_CANDIDATE_SCHEMA, "candidate.schema_version")
    _equal(candidate.get("nano_gate_authorized"), True, "candidate.nano_gate_authorized")
    _identity_equal(_path(candidate, "q5_k_m.artifact"), model, "candidate.q5_artifact")
    _identity_equal(_path(candidate, "q5_k_m.full_evidence"), full_identity, "candidate.q5_full_evidence")
    _equal(_path(candidate, "q5_k_m.full.all_gates_passed"), True, "candidate.q5_full_gates")
    _equal(_path(candidate, "q5_k_m.full.runtime_lora_count"), 0, "candidate.runtime_lora_count")
    _equal(_path(candidate, "q5_k_m.full.strict_17_to_1"), True, "candidate.17_to_1")


def _validate_static_full(
    report: Mapping[str, Any],
    model: Mapping[str, Any],
    formal_identities: Mapping[str, Mapping[str, Any]],
) -> Dict[str, float]:
    _equal(report.get("schema_version"), STATIC_FULL_SCHEMA, "static_full.schema_version")
    _identity_equal(_path(report, "runtime.static_fused_model"), model, "static_full.runtime.model")
    _equal(_path(report, "runtime.runtime_lora_count"), 0, "static_full.runtime_lora_count")
    _equal(_path(report, "runtime.lora_adapters_before"), [], "static_full.lora_adapters_before")
    _equal(_path(report, "runtime.lora_adapters_after"), [], "static_full.lora_adapters_after")
    _equal(_path(report, "runtime.request_level_lora_switching"), False, "static_full.request_switching")
    _identity_equal(_path(report, "inputs.traffic_bf16"), formal_identities["formal_traffic"], "static_full.traffic_bf16")
    _identity_equal(_path(report, "inputs.industrial_bf16"), formal_identities["formal_industrial"], "static_full.industrial_bf16")
    metrics: Dict[str, float] = {}
    for scene, count, labels, prefix in (
        ("traffic", 2400, "ABCDEF", "T"),
        ("industrial", 960, "ABC", "I"),
    ):
        section = _path(report, scene)
        classification = _path(report, scene + ".classification")
        records = _path(report, scene + ".records")
        _require(isinstance(records, list) and len(records) == count, "static_full {} records count".format(scene))
        for index, row in enumerate(records):
            field = "static_full.{}.records[{}]".format(scene, index)
            _require(isinstance(row, Mapping), field + " invalid")
            raw_prompt = row.get("raw_prompt")
            prompt = row.get("prompt")
            _require(isinstance(raw_prompt, str) and len(raw_prompt) == 16 and raw_prompt.isascii() and raw_prompt.isdigit(), field + ".raw_prompt contract")
            _equal(prompt, prefix + raw_prompt, field + ".prompt")
            _equal(row.get("prompt_n"), 17, field + ".prompt_n")
            _equal(row.get("predicted_n"), 1, field + ".predicted_n")
            _equal(row.get("request_has_lora_field"), False, field + ".request_lora")
            keys = row.get("request_payload_keys")
            _require(isinstance(keys, list) and not any("lora" in str(k).lower() for k in keys), field + ".payload_lora")
            _equal(row.get("valid"), True, field + ".valid")
            _require(row.get("prediction") in labels, field + ".prediction")
        contract = section.get("contract")
        _require(isinstance(contract, Mapping), "static_full {} contract missing".format(scene))
        for name in (
            "input_chars_all_17_prefix_plus_16_digits",
            "prompt_tokens_all_17",
            "output_tokens_all_1",
            "grammar_applied_before_sampling",
            "requests_have_no_lora_fields",
        ):
            _equal(contract.get(name), True, "static_full.{}.contract.{}".format(scene, name))
        _equal(contract.get("post_hoc_remapping"), False, "static_full.{}.contract.post_hoc".format(scene))
        calculated = _classification(records, labels)
        _equal(classification.get("count"), count, "static_full.{}.count".format(scene))
        _close(classification.get("correct"), calculated["correct"], "static_full.{}.correct".format(scene))
        for name in ("accuracy", "macro_f1", "weighted_f1", "valid_output_rate"):
            _close(classification.get(name), calculated[name], "static_full.{}.{}".format(scene, name))
        if scene == "traffic":
            _at_least(calculated["accuracy"], TRAFFIC_MIN_ACCURACY, "static_full.traffic.accuracy")
            _at_least(calculated["weighted_f1"], TRAFFIC_MIN_WEIGHTED_F1, "static_full.traffic.weighted_f1")
        else:
            for name in ("accuracy", "macro_f1", "weighted_f1"):
                _close(calculated[name], 1.0, "static_full.industrial." + name)
        metrics["static_{}_accuracy".format(scene)] = calculated["accuracy"]
        metrics["static_{}_macro_f1".format(scene)] = calculated["macro_f1"]
        metrics["static_{}_weighted_f1".format(scene)] = calculated["weighted_f1"]
        metrics["static_{}_valid_output_rate".format(scene)] = calculated["valid_output_rate"]
    gates = report.get("gates")
    _require(isinstance(gates, Mapping) and gates and all(v is True for v in gates.values()), "static_full gates failed")
    _equal(report.get("all_gates_passed"), True, "static_full.all_gates_passed")
    metrics["static_completed_requests"] = 3360.0
    return metrics


def _validate_sha_manifest(path: Path) -> Dict[str, str]:
    records: Dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split(None, 1)
        _require(len(parts) == 2, "nano SHA256SUMS line {} invalid".format(line_number))
        digest, name = parts[0], parts[1].strip()
        _require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "nano SHA256SUMS line {} digest invalid".format(line_number))
        _require(name == Path(name).name and name not in records, "nano SHA256SUMS unsafe or duplicate name: {}".format(name))
        records[name] = digest
    _equal(set(records), NANO_SHA_FILES, "nano SHA256SUMS file set")
    for name, expected in records.items():
        target = path.parent / name
        _actual_identity(target, "nano SHA256SUMS " + name)
        _equal(sha256_file(target), expected, "nano SHA256SUMS " + name)
    return records


def _command_option(tokens: Sequence[str], option: str) -> str:
    _require(tokens.count(option) == 1, "sidecar command must contain exactly one {}".format(option))
    index = tokens.index(option)
    _require(index + 1 < len(tokens), "sidecar command {} has no value".format(option))
    return tokens[index + 1]


def _validate_sidecar_command(path: Path, observation: Mapping[str, Any]) -> None:
    try:
        tokens = shlex.split(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError("nano sidecar command is invalid") from exc
    _require(bool(tokens) and Path(tokens[0]).name == "llama-server", "sidecar command is not llama-server")
    _require(not any(token == "--lora" or token.startswith("--lora-") for token in tokens), "sidecar command must not contain --lora options")
    _require("--no-mmap" not in tokens, "sidecar command must keep default mmap enabled")
    _require("--no-cuda-graph" not in tokens, "sidecar command must keep CUDA graphs enabled")
    _equal(_command_option(tokens, "--model"), _path(observation, "runtime.static_fused_model.path"), "sidecar --model")
    for option, expected in (
        ("--ctx-size", "128"),
        ("--batch-size", "16"),
        ("--ubatch-size", "16"),
        ("--parallel", "1"),
        ("--gpu-layers", "99"),
    ):
        _equal(_command_option(tokens, option), expected, "sidecar " + option)


def _memory_value(row: Mapping[str, Any], name: str, field: str) -> int:
    value = _integer(row.get(name), field + "." + name)
    _require(value >= 0, field + "." + name + " must be nonnegative")
    return value


def _memory_summary(rows: Sequence[Mapping[str, Any]], phase: str) -> Dict[str, int]:
    _require(bool(rows), phase + " memory rows empty")
    peak = 0
    minimum_available: Optional[int] = None
    longest_d = 0
    oom_values = []
    pswpin_values = []
    for index, row in enumerate(rows):
        field = "{}.memory[{}]".format(phase, index)
        _equal(row.get("phase"), phase, field + ".phase")
        vmrss = _memory_value(row, "vmrss_kib", field)
        vmhwm = _memory_value(row, "vmhwm_kib", field)
        vmswap = _memory_value(row, "vmswap_kib", field)
        available = _memory_value(row, "mem_available_kib", field) * 1024
        peak = max(peak, vmhwm * 1024, (vmrss + vmswap) * 1024)
        minimum_available = available if minimum_available is None else min(minimum_available, available)
        longest_d = max(longest_d, _memory_value(row, "consecutive_d_samples", field))
        oom_values.append(_memory_value(row, "oom_kill", field))
        pswpin_values.append(_memory_value(row, "pswpin", field))
    return {
        "peak": peak,
        "minimum_available": int(minimum_available or 0),
        "longest_d": longest_d,
        "oom_delta": oom_values[-1] - oom_values[0],
        "pswpin_delta": pswpin_values[-1] - pswpin_values[0],
        "count": len(rows),
    }


def _validate_nano(
    observation: Mapping[str, Any],
    gate: Mapping[str, Any],
    model: Mapping[str, Any],
    lora_before_path: Path,
    lora_after_path: Path,
    command_path: Path,
    startup_path: Path,
    inference_path: Path,
) -> Dict[str, float]:
    _equal(observation.get("schema_version"), NANO_OBSERVATION_SCHEMA, "nano.schema_version")
    _equal(observation.get("status"), "completed", "nano.status")
    _equal(observation.get("attempted_requests"), 500, "nano.attempted_requests")
    _equal(observation.get("completed_requests"), 500, "nano.completed_requests")
    _equal(_path(observation, "stop_reason.code"), "completed", "nano.stop_reason")
    _identity_equal(_path(observation, "runtime.static_fused_model"), model, "nano.runtime.model")
    _equal(_path(observation, "runtime.runtime_lora_count"), 0, "nano.runtime_lora_count")
    _equal(_path(observation, "runtime.reconnects"), 0, "nano.reconnects")
    contract = _path(observation, "benchmark_contract")
    for name, expected in (
        ("per_scene_requested", 250),
        ("total_requested", 500),
        ("required_prompt_tokens", 17),
        ("required_output_tokens", 1),
        ("runtime_lora_count_required", 0),
        ("request_level_lora_switching", False),
        ("requests_omit_lora_fields", True),
    ):
        _equal(contract.get(name), expected, "nano.contract." + name)
    _equal(_read_json_value(lora_before_path, "nano lora before"), [], "nano /lora-adapters before")
    _equal(_read_json_value(lora_after_path, "nano lora after"), [], "nano /lora-adapters after")
    _validate_sidecar_command(command_path, observation)

    records = observation.get("records")
    _require(isinstance(records, list) and len(records) == 500, "nano records must contain 500 rows")
    traffic_rows = []
    industrial_rows = []
    for index, row in enumerate(records):
        field = "nano.records[{}]".format(index)
        _require(isinstance(row, Mapping), field + " invalid")
        scene = "traffic" if index % 2 == 0 else "industrial"
        labels = "ABCDEF" if scene == "traffic" else "ABC"
        _equal(row.get("scene"), scene, field + ".scene")
        _equal(row.get("sequence"), index + 1, field + ".sequence")
        _equal(row.get("prompt_tokens"), 17, field + ".prompt_tokens")
        _equal(row.get("output_tokens"), 1, field + ".output_tokens")
        _equal(row.get("request_has_lora_field"), False, field + ".request_lora")
        keys = row.get("request_payload_keys")
        _require(isinstance(keys, list) and not any("lora" in str(k).lower() for k in keys), field + ".payload_lora")
        _equal(row.get("valid_output"), True, field + ".valid")
        _require(row.get("prediction") in labels, field + ".prediction")
        (traffic_rows if scene == "traffic" else industrial_rows).append(row)
    _equal(observation.get("errors"), [], "nano.errors")
    traffic = _classification(traffic_rows, "ABCDEF")
    industrial = _classification(industrial_rows, "ABC")
    _equal(int(traffic["count"]), 250, "nano.traffic_count")
    _equal(int(industrial["count"]), 250, "nano.industrial_count")
    _at_least(traffic["accuracy"], TRAFFIC_MIN_ACCURACY, "nano.traffic.accuracy")
    _at_least(traffic["weighted_f1"], TRAFFIC_MIN_WEIGHTED_F1, "nano.traffic.weighted_f1")
    for name in ("accuracy", "macro_f1", "weighted_f1"):
        _close(industrial[name], 1.0, "nano.industrial." + name)
    for scene, calculated in (("traffic", traffic), ("industrial", industrial)):
        summary = _path(observation, "summary." + scene)
        _equal(summary.get("count"), 250, "nano.summary.{}.count".format(scene))
        _close(summary.get("accuracy"), calculated["accuracy"], "nano.summary.{}.accuracy".format(scene))
        _close(summary.get("macro_f1"), calculated["macro_f1"], "nano.summary.{}.macro_f1".format(scene))
        _close(summary.get("weighted_f1"), calculated["weighted_f1"], "nano.summary.{}.weighted_f1".format(scene))
        _close(summary.get("valid_output_rate"), 1.0, "nano.summary.{}.valid".format(scene))
    latencies = [_number(row.get("latency_ms"), "nano.record.latency_ms") for row in records]
    mean_latency = sum(latencies) / len(latencies)
    _close(_path(observation, "summary.overall.latency_ms.mean"), mean_latency, "nano.mean_latency", tolerance=1e-6)
    _close(_path(observation, "summary.overall.valid_output_rate"), 1.0, "nano.valid_output_rate")
    _require(mean_latency < MAX_MEAN_LATENCY_MS, "nano mean latency must be < 200ms")

    startup_rows = _read_jsonl(startup_path, "nano startup memory")
    inference_rows = _read_jsonl(inference_path, "nano inference memory")
    startup = _memory_summary(startup_rows, "startup")
    inference = _memory_summary(inference_rows, "inference")
    strict_peak = max(startup["peak"], inference["peak"])
    _require(strict_peak <= MAX_NANO_STRICT_PEAK_BYTES, "nano strict peak exceeds 1.5GB")
    _require(inference["minimum_available"] >= MIN_NANO_MEM_AVAILABLE_BYTES, "nano minimum MemAvailable below 256MiB")
    _equal(inference["oom_delta"], 0, "nano inference OOM delta")
    _require(inference["longest_d"] < 5, "nano inference D-state consecutive samples")
    resources = observation.get("resource_samples")
    _require(isinstance(resources, list), "nano resource_samples missing")
    by_request = {row.get("request_count"): row for row in resources if isinstance(row, Mapping)}
    _require(100 in by_request and 500 in by_request, "nano resource samples must include requests 100 and 500")
    def resident(row: Mapping[str, Any]) -> int:
        return (_integer(row.get("llama_vmrss_kib"), "nano resource rss") + _integer(row.get("llama_vmswap_kib"), "nano resource swap")) * 1024
    growth = resident(by_request[500]) - resident(by_request[100])
    _require(growth <= MAX_NANO_GROWTH_BYTES, "nano post-100 growth exceeds 32MiB")

    _equal(gate.get("schema_version"), NANO_GATE_SCHEMA, "nano_gate.schema_version")
    _equal(gate.get("gate_pass"), True, "nano_gate.gate_pass")
    checks = gate.get("checks")
    _require(isinstance(checks, Mapping), "nano_gate.checks missing")
    _equal(set(checks), NANO_REQUIRED_CHECKS, "nano_gate.check set")
    _require(all(v is True for v in checks.values()), "nano_gate checks failed")
    _equal(gate.get("completed_requests"), 500, "nano_gate.completed_requests")
    _equal(gate.get("traffic_count"), 250, "nano_gate.traffic_count")
    _equal(gate.get("industrial_count"), 250, "nano_gate.industrial_count")
    _equal(_integer(gate.get("strict_peak_bytes"), "nano_gate.strict_peak_bytes"), strict_peak, "nano_gate.strict_peak_bytes")
    _equal(gate.get("threshold_bytes"), MAX_NANO_STRICT_PEAK_BYTES, "nano_gate.threshold_bytes")
    _equal(gate.get("rss_plus_swap_growth_100_to_500_bytes"), growth, "nano_gate.growth")
    _equal(gate.get("growth_threshold_bytes"), MAX_NANO_GROWTH_BYTES, "nano_gate.growth_threshold")
    _equal(gate.get("minimum_mem_available_bytes"), inference["minimum_available"], "nano_gate.minimum_mem_available")
    _equal(gate.get("oom_kill_delta"), inference["oom_delta"], "nano_gate.oom_delta")
    _equal(gate.get("inference_longest_consecutive_d_samples"), inference["longest_d"], "nano_gate.longest_d")
    _equal(gate.get("startup_sample_count"), startup["count"], "nano_gate.startup_samples")
    _equal(gate.get("inference_sample_count"), inference["count"], "nano_gate.inference_samples")
    _equal(gate.get("sample_count"), startup["count"] + inference["count"], "nano_gate.sample_count")
    _equal(gate.get("pswpin_pages_delta"), inference["pswpin_delta"], "nano_gate.pswpin_delta")
    command = gate.get("command_contract")
    _require(isinstance(command, Mapping), "nano_gate.command_contract missing")
    for name, expected in (
        ("ctx_size", 128),
        ("batch_size", 16),
        ("ubatch_size", 16),
        ("parallel", 1),
        ("gpu_layers", 99),
        ("runtime_lora_count", 0),
        ("mmap", "default_enabled"),
        ("cuda_graphs", "default_enabled"),
    ):
        _equal(command.get(name), expected, "nano_gate.command." + name)
    for name, expected in (
        ("traffic_accuracy", traffic["accuracy"]),
        ("traffic_weighted_f1", traffic["weighted_f1"]),
        ("industrial_accuracy", industrial["accuracy"]),
        ("industrial_macro_f1", industrial["macro_f1"]),
        ("industrial_weighted_f1", industrial["weighted_f1"]),
        ("valid_output_rate", 1.0),
        ("overall_mean_latency_ms", mean_latency),
    ):
        _close(gate.get(name), expected, "nano_gate." + name, tolerance=1e-6)
    return {
        "nano_completed_requests": 500.0,
        "nano_strict_peak_bytes": float(strict_peak),
        "nano_growth_100_to_500_bytes": float(growth),
        "nano_minimum_mem_available_bytes": float(inference["minimum_available"]),
        "nano_oom_kill_delta": float(inference["oom_delta"]),
        "nano_inference_longest_d_samples": float(inference["longest_d"]),
        "nano_overall_mean_latency_ms": mean_latency,
        "nano_traffic_accuracy": traffic["accuracy"],
        "nano_traffic_weighted_f1": traffic["weighted_f1"],
        "nano_industrial_accuracy": industrial["accuracy"],
        "nano_industrial_macro_f1": industrial["macro_f1"],
        "nano_industrial_weighted_f1": industrial["weighted_f1"],
        "nano_valid_output_rate": 1.0,
    }


def _gate_rows() -> list:
    return [
        {"metric": "bf16_completed_requests", "operator": "==", "value": 3360.0},
        {"metric": "bf16_traffic_accuracy", "operator": ">=", "value": TRAFFIC_MIN_ACCURACY},
        {"metric": "bf16_traffic_weighted_f1", "operator": ">=", "value": TRAFFIC_MIN_WEIGHTED_F1},
        {"metric": "bf16_industrial_accuracy", "operator": "==", "value": 1.0},
        {"metric": "bf16_industrial_macro_f1", "operator": "==", "value": 1.0},
        {"metric": "bf16_industrial_weighted_f1", "operator": "==", "value": 1.0},
        {"metric": "static_completed_requests", "operator": "==", "value": 3360.0},
        {"metric": "static_traffic_accuracy", "operator": ">=", "value": TRAFFIC_MIN_ACCURACY},
        {"metric": "static_traffic_weighted_f1", "operator": ">=", "value": TRAFFIC_MIN_WEIGHTED_F1},
        {"metric": "static_traffic_valid_output_rate", "operator": "==", "value": 1.0},
        {"metric": "static_industrial_accuracy", "operator": "==", "value": 1.0},
        {"metric": "static_industrial_macro_f1", "operator": "==", "value": 1.0},
        {"metric": "static_industrial_weighted_f1", "operator": "==", "value": 1.0},
        {"metric": "static_industrial_valid_output_rate", "operator": "==", "value": 1.0},
        {"metric": "nano_completed_requests", "operator": "==", "value": 500.0},
        {"metric": "nano_strict_peak_bytes", "operator": "<=", "value": float(MAX_NANO_STRICT_PEAK_BYTES)},
        {"metric": "nano_growth_100_to_500_bytes", "operator": "<=", "value": float(MAX_NANO_GROWTH_BYTES)},
        {"metric": "nano_minimum_mem_available_bytes", "operator": ">=", "value": float(MIN_NANO_MEM_AVAILABLE_BYTES)},
        {"metric": "nano_oom_kill_delta", "operator": "==", "value": 0.0},
        {"metric": "nano_inference_longest_d_samples", "operator": "<", "value": 5.0},
        {"metric": "nano_overall_mean_latency_ms", "operator": "<", "value": MAX_MEAN_LATENCY_MS},
        {"metric": "nano_traffic_accuracy", "operator": ">=", "value": TRAFFIC_MIN_ACCURACY},
        {"metric": "nano_traffic_weighted_f1", "operator": ">=", "value": TRAFFIC_MIN_WEIGHTED_F1},
        {"metric": "nano_industrial_accuracy", "operator": "==", "value": 1.0},
        {"metric": "nano_industrial_macro_f1", "operator": "==", "value": 1.0},
        {"metric": "nano_industrial_weighted_f1", "operator": "==", "value": 1.0},
        {"metric": "nano_valid_output_rate", "operator": "==", "value": 1.0},
    ]


def _evidence_paths(root: Path, raw: Any) -> Dict[str, Path]:
    _require(isinstance(raw, Mapping), "descriptor.evidence must be an object")
    names = (
        "formal_traffic",
        "formal_industrial",
        "formal_gate",
        "static_provenance",
        "static_candidate_summary",
        "static_full",
        "nano_observation",
        "nano_gate",
        "nano_lora_before",
        "nano_lora_after",
        "nano_sidecar_cmdline",
        "nano_startup_memory",
        "nano_inference_memory",
        "nano_sha256s",
    )
    _equal(set(raw), set(names), "descriptor.evidence fields")
    result = {}
    for name in names:
        path = _resolve(root, raw.get(name), "evidence." + name)
        _actual_identity(path, "evidence." + name)
        result[name] = path
    nano_parent = result["nano_sha256s"].parent
    for name in names[6:]:
        _equal(result[name].parent, nano_parent, "evidence.{} directory".format(name))
    return result


def _portable_sources(paths: Mapping[str, Path]) -> Dict[str, Dict[str, Any]]:
    return {
        name: _portable(_actual_identity(path, "evidence." + name))
        for name, path in paths.items()
    }


def validate_q5_static_joint_package(
    package_dir: Path, base_manifest_path: Path
) -> Dict[str, Any]:
    generic = validate_adapter_package(package_dir, base_manifest_path, require_gates=True)
    root = package_dir.resolve()
    manifest = read_json_object(root / MANIFEST_NAME)
    _equal(manifest.get("package_kind"), PACKAGE_KIND, "package.package_kind")
    _equal(manifest.get("runtime_adapters"), [], "package.runtime_adapters")
    input_contract = manifest.get("input_contract")
    _require(isinstance(input_contract, Mapping), "package.input_contract missing")
    _equal(
        input_contract.get("context_encoder"),
        CONTEXT_ENCODER,
        "package.input_contract.context_encoder",
    )
    _equal(
        input_contract.get("max_input_tokens"),
        MAX_INPUT_TOKENS,
        "package.input_contract.max_input_tokens",
    )
    lineage = manifest.get("training_lineage")
    _require(isinstance(lineage, Mapping), "package.training_lineage missing")
    _equal(lineage.get("peft_artifact_runtime_loaded"), False, "package.training_lineage.runtime_loaded")
    deployment = manifest.get("deployment")
    _require(isinstance(deployment, Mapping), "package.deployment missing")
    _equal(deployment.get("model_mode"), MODEL_MODE, "package.deployment.model_mode")
    _equal(deployment.get("runtime_adapters"), [], "package.deployment.runtime_adapters")
    _equal(deployment.get("runtime_lora_count"), 0, "package.deployment.runtime_lora_count")
    _equal(deployment.get("request_level_lora_switching"), False, "package.deployment.request_switching")
    _equal(deployment.get("scene_prefixes"), {"traffic": "T", "industrial": "I"}, "package.deployment.scene_prefixes")
    _equal(deployment.get("input_tokens"), 17, "package.deployment.input_tokens")
    _equal(deployment.get("output_tokens"), 1, "package.deployment.output_tokens")
    evidence_record = _path(manifest, "evaluation.evidence.q5_static_joint_gate_summary")
    summary_path = root / evidence_record["path"]
    summary = read_json_object(summary_path)
    _equal(summary.get("schema_version"), SUMMARY_SCHEMA, "package.summary.schema_version")
    _equal(summary.get("release_authorized_by_packager"), True, "package.summary.release_authorized")
    _equal(_path(summary, "runtime_contract.runtime_adapters"), [], "package.summary.runtime_adapters")
    _equal(_path(summary, "runtime_contract.runtime_lora_count"), 0, "package.summary.runtime_lora_count")
    model = _path(summary, "assets.static_q5_k_m")
    _equal(deployment.get("artifact_sha256"), model.get("sha256"), "package.deployment.artifact_sha256")
    _equal(deployment.get("artifact_bytes"), model.get("bytes"), "package.deployment.artifact_bytes")
    return {
        "status": "valid",
        "generic_validation": generic,
        "package_kind": PACKAGE_KIND,
        "runtime_adapters": [],
        "deployment_model_mode": MODEL_MODE,
        "static_fused_model": dict(model),
        "gate_summary_sha256": sha256_file(summary_path),
    }


def build_q5_static_joint_release(
    descriptor_path: Path, output_dir: Path
) -> Dict[str, Any]:
    descriptor_file = descriptor_path.resolve()
    root = descriptor_file.parent
    descriptor = read_json_object(descriptor_file)
    _equal(descriptor.get("schema_version"), SCHEMA, "descriptor.schema_version")
    _require("runtime_adapter" not in descriptor and "runtime_adapters" not in descriptor, "static release descriptor must not contain runtime adapters")
    output = output_dir.resolve()
    if output.exists():
        raise ManifestError("immutable Q5 static joint package output already exists: {}".format(output))

    model_path, model = _declared_identity(root, descriptor.get("static_fused_model"), "static_fused_model")
    _equal(model["bytes"], 577_990_656, "static_fused_model.bytes")
    _equal(model["sha256"], "308daa980c7ca295e18bd76e8dcf6dc1ed725ded32ada535a0c5c1910c695ce2", "static_fused_model.sha256")
    adapter_source = _resolve(root, descriptor.get("training_adapter_source"), "training_adapter_source")
    _require(adapter_source.is_dir() and not adapter_source.is_symlink(), "training_adapter_source must be a regular directory")
    artifacts = _training_artifacts(adapter_source)
    paths = _evidence_paths(root, descriptor.get("evidence"))
    evidence = {
        name: read_json_object(path)
        for name, path in paths.items()
        if name in {
            "formal_traffic",
            "formal_industrial",
            "formal_gate",
            "static_provenance",
            "static_candidate_summary",
            "static_full",
            "nano_observation",
            "nano_gate",
        }
    }
    formal_ids = {
        name: _actual_identity(paths[name], "evidence." + name)
        for name in ("formal_traffic", "formal_industrial")
    }
    metrics = _validate_formal(
        evidence["formal_traffic"], evidence["formal_industrial"], evidence["formal_gate"], artifacts
    )
    full_identity = _actual_identity(paths["static_full"], "evidence.static_full")
    _validate_provenance(
        evidence["static_provenance"], evidence["static_candidate_summary"], model, artifacts, full_identity
    )
    metrics.update(_validate_static_full(evidence["static_full"], model, formal_ids))
    sha_records = _validate_sha_manifest(paths["nano_sha256s"])
    metrics.update(
        _validate_nano(
            evidence["nano_observation"],
            evidence["nano_gate"],
            model,
            paths["nano_lora_before"],
            paths["nano_lora_after"],
            paths["nano_sidecar_cmdline"],
            paths["nano_startup_memory"],
            paths["nano_inference_memory"],
        )
    )
    _equal(
        sha_records.get("q5_static_fusion_full_gate.json"),
        full_identity["sha256"],
        "nano copy of static full evidence",
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "descriptor_sha256": sha256_file(descriptor_file),
        "assets": {"static_q5_k_m": _portable(model)},
        "training_lineage": {
            "joint_peft_artifacts": {name: _portable(identity) for name, identity in artifacts.items()},
            "merge_method": "PEFT PeftModel.merge_and_unload",
            "peft_artifact_runtime_loaded": False,
        },
        "runtime_contract": {
            "model_mode": MODEL_MODE,
            "runtime_adapters": [],
            "runtime_lora_count": 0,
            "request_level_lora_switching": False,
            "traffic_prefix": "T",
            "traffic_allowed_slots": list("ABCDEF"),
            "industrial_prefix": "I",
            "industrial_allowed_slots": list("ABC"),
            "input_tokens": 17,
            "output_tokens": 1,
        },
        "source_evidence": _portable_sources(paths),
        "nano_sha256_manifest": {
            "verified_file_count": len(sha_records),
            "all_entries_verified": True,
        },
        "metrics": metrics,
        "fixed_gates": _gate_rows(),
        "release_authorized_by_packager": True,
    }

    base_manifest = _resolve(root, descriptor.get("base_manifest"), "base_manifest")
    action_mapping = _resolve(root, descriptor.get("action_mapping"), "action_mapping")
    adapter_id = descriptor.get("adapter_id")
    version = descriptor.get("version")
    training = descriptor.get("training")
    _require(isinstance(adapter_id, str) and bool(adapter_id), "adapter_id must be non-empty")
    _require(isinstance(version, str) and bool(version), "version must be non-empty")
    _require(isinstance(training, Mapping), "training must be an object")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(tempfile.mkdtemp(prefix=output.name + ".static-stage-", dir=str(output.parent)))
    staging = staging_parent / "package"
    try:
        work = staging_parent / "inputs"
        work.mkdir()
        summary_path = work / "q5_static_joint_gate_summary.json"
        write_json_object(summary_path, summary)
        sources = {
            name: {"evidence": "q5_static_joint_gate_summary", "path": "metrics." + name}
            for name in metrics
        }
        spec = {
            "schema_version": "edge-llm-package-spec/v1",
            "adapter_id": adapter_id,
            "scene": TRAFFIC_SCENE,
            "version": version,
            "adapter_source": str(adapter_source),
            "action_mapping": str(action_mapping),
            "input_contract": {
                "event_type": "com.cloudedge.traffic.edge-event.v1",
                "data_schema": "https://cloud-edge.local/schemas/scenes/traffic-edge-event-v1.json",
                "llm_input_type": "compact_text_code",
                "context_encoder": CONTEXT_ENCODER,
                "max_input_tokens": MAX_INPUT_TOKENS,
                "direct_media_to_llm": False,
            },
            "training": dict(training),
            "evaluation": {
                "evidence": {"q5_static_joint_gate_summary": str(summary_path)},
                "metric_sources": sources,
                "gates": _gate_rows(),
            },
            "deployment": {
                "artifact": str(model_path),
                "runtime": "llama.cpp",
                "format": "gguf",
                "quantization": "Q5_K_M",
                "max_input_tokens": 17,
                "max_output_tokens": 1,
                "thinking": False,
            },
        }
        spec_path = work / "package_spec.json"
        write_json_object(spec_path, spec)
        build_adapter_package(
            project_root=root,
            base_manifest_path=base_manifest,
            spec_path=spec_path,
            output_dir=staging,
        )
        manifest_path = staging / MANIFEST_NAME
        manifest = read_json_object(manifest_path)
        manifest["package_kind"] = PACKAGE_KIND
        manifest["runtime_adapters"] = []
        manifest["training_lineage"] = {
            "peft_artifact_runtime_loaded": False,
            "role": "joint_training_and_static_merge_provenance_only",
        }
        deployment = manifest["deployment"]
        deployment.update(
            {
                "model_mode": MODEL_MODE,
                "runtime_adapters": [],
                "runtime_lora_count": 0,
                "request_level_lora_switching": False,
                "scene_prefixes": {"traffic": "T", "industrial": "I"},
                "input_tokens": 17,
                "output_tokens": 1,
            }
        )
        write_json_object(manifest_path, manifest)
        validation = validate_q5_static_joint_package(staging, base_manifest)
        os.replace(str(staging), str(output))
    finally:
        if staging_parent.exists():
            shutil.rmtree(staging_parent)
    validation["package"] = str(output)
    validation["manifest_sha256"] = sha256_file(output / MANIFEST_NAME)
    return validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed statically fused joint Q5_K_M package"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--descriptor", required=True)
    build.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--base", required=True)
    validate.add_argument("--package", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_q5_static_joint_release(Path(args.descriptor), Path(args.output))
    else:
        result = validate_q5_static_joint_package(Path(args.package), Path(args.base))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
