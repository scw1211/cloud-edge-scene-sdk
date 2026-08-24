"""Fail-closed Q4 packager and trusted Q8 candidate validator for raw16 models.

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
import hashlib
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


SCHEMA = "edge-llm-q4-static-raw16-release-input/v1"
SUMMARY_SCHEMA = "edge-llm-q4-static-raw16-release-gate-summary/v1"
PACKAGE_KIND = "joint-static-raw16-model-release"
MODEL_MODE = "joint_static_fusion"
CONTEXT_ENCODER = "scene-disjoint-decimal16@v2"
MAX_INPUT_TOKENS = 16
FORMAL_GATE_SCHEMA = "edge-llm-joint-traffic-industrial-gate/v2"
STATIC_PROVENANCE_SCHEMA = "edge-llm-joint-static-raw16-provenance/v1"
STATIC_CANDIDATE_SCHEMA = "edge-llm-joint-static-raw16-candidate-summary/v1"
STATIC_FULL_SCHEMA = "edge-llm-joint-static-q4-raw16-full-validation/v1"
NANO_OBSERVATION_SCHEMA = "edge-llm-joint-static-q4-raw16-alternating-observation/v1"
NANO_GATE_SCHEMA = "q4km-static-raw16-nano-observational-memory-gate/v1"

TRAFFIC_SCENE = "freeway_traffic_management"
TRAFFIC_MIN_ACCURACY = 0.66
TRAFFIC_MIN_WEIGHTED_F1 = 0.65
MAX_NANO_STRICT_PEAK_BYTES = 1_500_000_000
MIN_NANO_MEM_AVAILABLE_BYTES = 256 * 1024 * 1024
MAX_NANO_GROWTH_BYTES = 32 * 1024 * 1024
MAX_MEAN_LATENCY_MS = 200.0

Q8_DESCRIPTOR_SCHEMA = "edge-llm-static-raw16-release-input/v2"
Q8_PACKAGE_SUMMARY_SCHEMA = "edge-llm-static-raw16-release-gate-summary/v2"
Q8_PACKAGE_ADAPTER_ID = "joint-traffic-industrial-static-q8-raw16"
Q8_PACKAGE_VERSION = "4.0.0"
Q8_DESCRIPTOR_SHA256 = (
    "dfb2690f069e5d9c2d14adf7b8af7a96dba980874114a2c2c77f90d38c4840e4"
)
Q8_FULL_SCHEMA = "edge-llm-joint-static-q8-raw16-full-validation/v1"
Q8_NANO_STAGE_A_SCHEMA = "nano-q8-stage-a-memory-ttft/v1"
Q8_NANO_STAGE_B_SCHEMA = "nano-q8-stage-b-quality-stability/v1"
Q8_NANO_OBSERVATION_SCHEMA = (
    "edge-llm-joint-static-q8-raw16-alternating-observation/v1"
)
Q8_PAIR_SCHEMA = "edge-llm-quantized-full-precision-pair/v2"
Q8_QUANTIZATION = "Q8_0"
Q8_MODEL = {
    "bytes": 811_835_392,
    "sha256": "6dc108cfe80e8adfcd12424287b536d81f3aaa98bf537aaf8b37b0c9f09702b9",
}
Q8_FULL_REPORT_SHA256 = (
    "c25092c0b1a83e280a46e349941fc447b709e714667b22661d75d406e8a421ee"
)
Q8_PAIR_SHA256 = "0ded401f8f67c0e9be1ce599f0d6b108fc47154099d30aca1059ff534c0bd0e4"
Q8_NANO_EDGE_SERVER_SHA256 = (
    "6f531b471b6ae23d02612f77ffbac18d0e066f8f9d18152b11b40e7156765f86"
)
Q8_STAGE_A_SUMMARY_SHA256 = (
    "605e5308e82b7f11f2f6eb907dd1bf4761e160baccb9ace322354adc092bcc1d"
)
Q8_STAGE_A_RAW_SHA256 = (
    "daf279f4b290d920f07291bcde02f00ab7a3227d4bc26f68d9c31c2a43a6e580"
)
Q8_STAGE_A_SHA_MANIFEST_SHA256 = (
    "17d0e9ae0c133c89b5e6f9f816ba73cbfdaaf2e5096f5dae44305cc00600c579"
)
Q8_STAGE_B_SUMMARY_SHA256 = (
    "6e24b53d2241ef8974a23fb822121f7a07b93a29b7f5baa956ddccec822440be"
)
Q8_STAGE_B_RAW_SHA256 = (
    "a2f8011573e52a2eb3952f483fbf0bdacbf37977801c26a6fa776c7baf4a7f8c"
)
Q8_STAGE_B_SHA_MANIFEST_SHA256 = (
    "9d99d30f6ee9c5b6b4ff68089ee7b5e8d38fb366c708ed938f66287674c46daa"
)
Q8_STAGE_A_REMOTE_ROOT = Path(
    "/home/jetson02/deploy_runs/candidate_validation/20260819_q8_0_nano_stage_a_v1"
)
Q8_STAGE_B_REMOTE_ROOT = Path(
    "/home/jetson02/deploy_runs/candidate_validation/20260819_q8_0_nano_stage_b_v1"
)
Q8_FULL_LLAMA_SERVER = {
    "bytes": 17_896,
    "sha256": "319b54ddcbd7404786f17b346f67aaffb29524926091a25f32c879f3c0bb0df6",
    "version": "9859",
    "commit": "4fc4ec554",
}
Q8_DATASETS = {
    "traffic": {
        "count": 2400,
        "sha256": "604ebf54458874b744051c5d2b7bd340771138e6de1347a01ff91ea806ce55ac",
    },
    "industrial": {
        "count": 960,
        "sha256": "2d68ce94a730d95b64078ba2048b69b73cf4bb464835cede8afb624c8411c7e1",
    },
}
Q8_EXPECTED_METRICS = {
    "q8_full_completed_requests": 3360.0,
    "q8_full_traffic_accuracy": 0.67625,
    "q8_full_traffic_macro_f1": 0.6170942107122338,
    "q8_full_traffic_weighted_f1": 0.6862350502073311,
    "q8_full_traffic_valid_output_rate": 1.0,
    "q8_full_industrial_accuracy": 1.0,
    "q8_full_industrial_macro_f1": 1.0,
    "q8_full_industrial_weighted_f1": 1.0,
    "q8_full_industrial_valid_output_rate": 1.0,
    "q8_nano_stage_a_strict_peak_bytes": 1_278_586_880.0,
    "q8_nano_stage_a_minimum_mem_available_bytes": 1_405_374_464.0,
    "q8_nano_stage_a_ttft_edge_mean_ms": 73.3166688125,
    "q8_nano_stage_a_ttft_teacher_mean_ms": 336.7228226625,
    "q8_nano_stage_a_ttft_reduction": 0.782264034754823,
    "q8_nano_stage_b_completed_requests": 500.0,
    "q8_nano_stage_b_strict_peak_bytes": 1_271_853_056.0,
    "q8_nano_stage_b_growth_100_to_500_bytes": 286_720.0,
    "q8_nano_stage_b_minimum_mem_available_bytes": 1_388_498_944.0,
    "q8_nano_stage_b_mean_latency_ms": 73.421821444,
    "q8_nano_stage_b_traffic_accuracy": 0.724,
    "q8_nano_stage_b_traffic_weighted_f1": 0.7337293444698455,
    "q8_nano_stage_b_industrial_accuracy": 1.0,
    "q8_nano_stage_b_industrial_weighted_f1": 1.0,
    "q8_nano_stage_b_valid_output_rate": 1.0,
}
Q8_STAGE_A_REQUIRED_FILES = frozenset(
    {
        "RUN_STATUS.json",
        "benchmark.pid",
        "benchmark_ttft_q8_candidate.py",
        "candidate.gguf",
        "candidate_cmdline.txt",
        "candidate_health.json",
        "candidate_journal.txt",
        "candidate_lora.json",
        "candidate_props.json",
        "cloud_ollama_version.json",
        "formal_health_after.json",
        "formal_health_before.json",
        "formal_lora_after.json",
        "formal_lora_before.json",
        "formal_outbox_after.json",
        "formal_outbox_before.json",
        "formal_props_after.json",
        "formal_props_before.json",
        "formal_unit_after.txt",
        "formal_unit_before.txt",
        "inference_memory_20ms.jsonl",
        "llama_pid.txt",
        "quantization_pair_manifest.json",
        "restore_formal_q4.sh",
        "run_nano_stage_a.sh",
        "stage_a_summary.json",
        "stage_a_summary.stdout.json",
        "startup_memory_20ms.jsonl",
        "summarize_nano_stage_a.py",
        "ttft_input.json",
        "ttft_paired_raw.json",
    }
)
Q8_STAGE_B_REQUIRED_FILES = frozenset(
    {
        "RUN_STATUS.json",
        "benchmark.pid",
        "benchmark_joint_static_raw16_500.py",
        "candidate_cmdline.txt",
        "candidate_health.json",
        "candidate_journal.txt",
        "candidate_lora.json",
        "candidate_props.json",
        "formal_health_after.json",
        "formal_health_before.json",
        "formal_lora_after.json",
        "formal_lora_before.json",
        "formal_outbox_after.json",
        "formal_outbox_before.json",
        "formal_props_after.json",
        "formal_props_before.json",
        "formal_unit_after.txt",
        "formal_unit_before.txt",
        "inference_memory_20ms.jsonl",
        "llama_pid.txt",
        "q8_static_raw16_500.json",
        "restore_formal_q4_stage_b.sh",
        "run_nano_stage_b.sh",
        "stage_b_summary.json",
        "stage_b_summary.stdout.json",
        "startup_memory_20ms.jsonl",
        "summarize_nano_stage_b.py",
    }
)

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
    "nano222_joint_q4_static_raw16_observed_500_v2.json",
    "observed_gate_summary.json",
    "props_after.json",
    "props_before.json",
    "props_contract_after.json",
    "props_contract_before.json",
    "q4_static_raw16_full_gate.json",
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
    "strict_16_to_1_and_no_request_lora",
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
    _equal(_path(metrics, "runtime_contract.input_tokens"), 16, "train_metrics.input_tokens")
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
    _equal(report.get("prompt_prefix"), "", scene + ".prompt_prefix")
    _equal(report.get("required_prompt_tokens"), 16, scene + ".prompt_tokens")
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
        _equal(row.get("prompt_tokens"), 16, "{}.samples[{}].prompt_tokens".format(scene, index))
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
    _equal(_path(provenance, "post_evaluation_state.q4_full_passed"), True, "provenance.q4_full_passed")
    _equal(_path(provenance, "post_evaluation_state.nano_gate_authorized"), True, "provenance.nano_gate_authorized")

    _equal(candidate.get("schema_version"), STATIC_CANDIDATE_SCHEMA, "candidate.schema_version")
    _equal(candidate.get("nano_gate_authorized"), True, "candidate.nano_gate_authorized")
    _identity_equal(_path(candidate, "q4_k_m.artifact"), model, "candidate.q4_artifact")
    _identity_equal(_path(candidate, "q4_k_m.full_evidence"), full_identity, "candidate.q4_full_evidence")
    _equal(_path(candidate, "q4_k_m.full.all_gates_passed"), True, "candidate.q4_full_gates")
    _equal(_path(candidate, "q4_k_m.full.runtime_lora_count"), 0, "candidate.runtime_lora_count")
    _equal(_path(candidate, "q4_k_m.full.strict_16_to_1"), True, "candidate.17_to_1")


def _validate_static_full(
    report: Mapping[str, Any],
    model: Mapping[str, Any],
    formal_identities: Mapping[str, Mapping[str, Any]],
) -> Dict[str, float]:
    _equal(report.get("schema_version"), STATIC_FULL_SCHEMA, "static_full.schema_version")
    _identity_equal(_path(report, "runtime.model"), model, "static_full.runtime.model")
    _equal(_path(report, "runtime.resident_lora_count"), 0, "static_full.runtime_lora_count")
    _equal(_path(report, "runtime.lora_adapters_before"), [], "static_full.lora_adapters_before")
    _equal(_path(report, "runtime.lora_adapters_after"), [], "static_full.lora_adapters_after")
    _equal(_path(report, "runtime.request_level_lora_switching"), False, "static_full.request_switching")
    _identity_equal(_path(report, "inputs.traffic_bf16"), formal_identities["formal_traffic"], "static_full.traffic_bf16")
    _identity_equal(_path(report, "inputs.industrial_bf16"), formal_identities["formal_industrial"], "static_full.industrial_bf16")
    metrics: Dict[str, float] = {}
    for scene, count, labels, prefix in (
        ("traffic", 2400, "ABCDEF", ""),
        ("industrial", 960, "ABC", ""),
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
            _equal(prompt, raw_prompt, field + ".prompt")
            _equal(row.get("prompt_n"), 16, field + ".prompt_n")
            _equal(row.get("predicted_n"), 1, field + ".predicted_n")
            _equal(row.get("request_has_lora_field"), False, field + ".request_lora")
            keys = row.get("request_payload_keys")
            _require(isinstance(keys, list) and not any("lora" in str(k).lower() for k in keys), field + ".payload_lora")
            _equal(row.get("valid"), True, field + ".valid")
            _require(row.get("prediction") in labels, field + ".prediction")
        contract = section.get("contract")
        _require(isinstance(contract, Mapping), "static_full {} contract missing".format(scene))
        for name in (
            "input_chars_all_decimal16",
            "prompt_tokens_all_16",
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
    _equal(
        _command_option(tokens, "--model"),
        _path(observation, "runtime.static_fused_model.path"),
        "sidecar --model",
    )
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
    _identity_equal(
        _path(observation, "runtime.static_fused_model"), model, "nano.runtime.model"
    )
    _equal(_path(observation, "runtime.runtime_lora_count"), 0, "nano.runtime_lora_count")
    _equal(_path(observation, "runtime.reconnects"), 0, "nano.reconnects")
    contract = _path(observation, "benchmark_contract")
    for name, expected in (
        ("per_scene_requested", 250),
        ("total_requested", 500),
        ("required_prompt_tokens", 16),
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
        _equal(row.get("prompt_tokens"), 16, field + ".prompt_tokens")
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
    overall_pswpin_delta = _integer(
        inference_rows[-1].get("pswpin"), "nano inference final pswpin"
    ) - _integer(startup_rows[0].get("pswpin"), "nano startup initial pswpin")
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
    _equal(gate.get("pswpin_pages_delta"), overall_pswpin_delta, "nano_gate.pswpin_delta")
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


def _q8_gate_rows() -> list:
    return [
        {"metric": "q8_full_completed_requests", "operator": "==", "value": 3360.0},
        {"metric": "q8_full_traffic_accuracy", "operator": ">=", "value": TRAFFIC_MIN_ACCURACY},
        {"metric": "q8_full_traffic_weighted_f1", "operator": ">=", "value": TRAFFIC_MIN_WEIGHTED_F1},
        {"metric": "q8_full_traffic_valid_output_rate", "operator": "==", "value": 1.0},
        {"metric": "q8_full_industrial_accuracy", "operator": "==", "value": 1.0},
        {"metric": "q8_full_industrial_macro_f1", "operator": "==", "value": 1.0},
        {"metric": "q8_full_industrial_weighted_f1", "operator": "==", "value": 1.0},
        {"metric": "q8_full_industrial_valid_output_rate", "operator": "==", "value": 1.0},
        {"metric": "q8_nano_stage_a_strict_peak_bytes", "operator": "<=", "value": float(MAX_NANO_STRICT_PEAK_BYTES)},
        {"metric": "q8_nano_stage_a_minimum_mem_available_bytes", "operator": ">=", "value": float(MIN_NANO_MEM_AVAILABLE_BYTES)},
        {"metric": "q8_nano_stage_a_ttft_reduction", "operator": ">=", "value": 0.75},
        {"metric": "q8_nano_stage_b_completed_requests", "operator": "==", "value": 500.0},
        {"metric": "q8_nano_stage_b_strict_peak_bytes", "operator": "<=", "value": float(MAX_NANO_STRICT_PEAK_BYTES)},
        {"metric": "q8_nano_stage_b_growth_100_to_500_bytes", "operator": "<=", "value": float(MAX_NANO_GROWTH_BYTES)},
        {"metric": "q8_nano_stage_b_minimum_mem_available_bytes", "operator": ">=", "value": float(MIN_NANO_MEM_AVAILABLE_BYTES)},
        {"metric": "q8_nano_stage_b_mean_latency_ms", "operator": "<", "value": MAX_MEAN_LATENCY_MS},
        {"metric": "q8_nano_stage_b_traffic_accuracy", "operator": ">=", "value": TRAFFIC_MIN_ACCURACY},
        {"metric": "q8_nano_stage_b_traffic_weighted_f1", "operator": ">=", "value": TRAFFIC_MIN_WEIGHTED_F1},
        {"metric": "q8_nano_stage_b_industrial_accuracy", "operator": "==", "value": 1.0},
        {"metric": "q8_nano_stage_b_industrial_weighted_f1", "operator": "==", "value": 1.0},
        {"metric": "q8_nano_stage_b_valid_output_rate", "operator": "==", "value": 1.0},
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


def validate_q4_static_joint_package(
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
    _equal(deployment.get("scene_prefixes"), {"traffic": "", "industrial": ""}, "package.deployment.scene_prefixes")
    _equal(deployment.get("input_tokens"), 16, "package.deployment.input_tokens")
    _equal(deployment.get("output_tokens"), 1, "package.deployment.output_tokens")
    evidence_record = _path(manifest, "evaluation.evidence.q4_static_raw16_gate_summary")
    summary_path = root / evidence_record["path"]
    summary = read_json_object(summary_path)
    _equal(summary.get("schema_version"), SUMMARY_SCHEMA, "package.summary.schema_version")
    _equal(summary.get("release_authorized_by_packager"), True, "package.summary.release_authorized")
    _equal(_path(summary, "runtime_contract.runtime_adapters"), [], "package.summary.runtime_adapters")
    _equal(_path(summary, "runtime_contract.runtime_lora_count"), 0, "package.summary.runtime_lora_count")
    model = _path(summary, "assets.static_q4_k_m")
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


def build_q4_static_joint_release(
    descriptor_path: Path, output_dir: Path
) -> Dict[str, Any]:
    descriptor_file = descriptor_path.resolve()
    root = descriptor_file.parent
    descriptor = read_json_object(descriptor_file)
    _equal(descriptor.get("schema_version"), SCHEMA, "descriptor.schema_version")
    _require("runtime_adapter" not in descriptor and "runtime_adapters" not in descriptor, "static release descriptor must not contain runtime adapters")
    output = output_dir.resolve()
    if output.exists():
        raise ManifestError("immutable Q4 static joint package output already exists: {}".format(output))

    model_path, model = _declared_identity(root, descriptor.get("static_fused_model"), "static_fused_model")
    _equal(model["bytes"], 529_289_216, "static_fused_model.bytes")
    _equal(model["sha256"], "828f839873c7505005101544d8febb7408bc0443c153c4ba97ec4b3603445526", "static_fused_model.sha256")
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
        sha_records.get("q4_static_raw16_full_gate.json"),
        full_identity["sha256"],
        "nano copy of static full evidence",
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "descriptor_sha256": sha256_file(descriptor_file),
        "assets": {"static_q4_k_m": _portable(model)},
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
            "traffic_prefix": "",
            "traffic_allowed_slots": list("ABCDEF"),
            "industrial_prefix": "",
            "industrial_allowed_slots": list("ABC"),
            "input_tokens": 16,
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
        summary_path = work / "q4_static_raw16_gate_summary.json"
        write_json_object(summary_path, summary)
        sources = {
            name: {"evidence": "q4_static_raw16_gate_summary", "path": "metrics." + name}
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
                "evidence": {"q4_static_raw16_gate_summary": str(summary_path)},
                "metric_sources": sources,
                "gates": _gate_rows(),
            },
            "deployment": {
                "artifact": str(model_path),
                "runtime": "llama.cpp",
                "format": "gguf",
                "quantization": "Q4_K_M",
                "max_input_tokens": 16,
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
                "scene_prefixes": {"traffic": "", "industrial": ""},
                "input_tokens": 16,
                "output_tokens": 1,
            }
        )
        write_json_object(manifest_path, manifest)
        validation = validate_q4_static_joint_package(staging, base_manifest)
        os.replace(str(staging), str(output))
    finally:
        if staging_parent.exists():
            shutil.rmtree(staging_parent)
    validation["package"] = str(output)
    validation["manifest_sha256"] = sha256_file(output / MANIFEST_NAME)
    return validation


def _q8_identity_on_disk(record: Any, field: str) -> Tuple[Path, Dict[str, Any]]:
    _require(isinstance(record, Mapping), "{} must be an identity object".format(field))
    raw_path = record.get("path")
    _require(isinstance(raw_path, str) and bool(raw_path), "{}.path missing".format(field))
    unresolved = Path(raw_path).expanduser()
    _require(not unresolved.is_symlink(), "{} must not be a symlink".format(field))
    path = unresolved.resolve()
    actual = _actual_identity(path, field)
    _identity_equal(record, actual, field)
    return path, actual


def _q8_canonical_sha256(value: Mapping[str, Any]) -> str:
    """Match the frozen pair-v2 canonical form (compact JSON plus final LF)."""
    payload = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _q8_pair_semantic_identity(pair: Mapping[str, Any]) -> Dict[str, Any]:
    conversion = dict(_path(pair, "conversion"))
    quantization = dict(_path(pair, "quantization"))
    command = quantization.get("command")
    if isinstance(command, list) and len(command) == 4:
        normalized_command = list(command)
        normalized_command[2] = Path(str(normalized_command[2])).name
        quantization["command"] = normalized_command
    return {
        "schema_version": pair.get("schema_version"),
        "model_family": pair.get("model_family"),
        "full_precision": pair.get("full_precision"),
        "quantized": pair.get("quantized"),
        "conversion": conversion,
        "quantization": quantization,
    }


def _validate_q8_pair_document(
    pair: Mapping[str, Any],
    model: Mapping[str, Any],
    adapter_sha256: str,
    base_manifest_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    _equal(pair.get("schema_version"), Q8_PAIR_SCHEMA, "q8_pair.schema_version")
    family = pair.get("model_family")
    _require(isinstance(family, Mapping), "q8_pair.model_family missing")
    _equal(family.get("model_id"), "Qwen/Qwen3.5-0.8B", "q8_pair.model_id")
    _equal(
        family.get("revision"),
        "2fc06364715b967f1860aea9cf38778875588b17",
        "q8_pair.revision",
    )
    _equal(family.get("parameter_count"), 752_393_024, "q8_pair.parameter_count")
    lineage = family.get("lineage")
    lineage_fields = {
        "base_manifest_sha256",
        "text_snapshot_manifest_sha256",
        "adapter_artifact_sha256",
        "merged_checkpoint_sha256",
    }
    _require(
        isinstance(lineage, Mapping) and set(lineage) == lineage_fields,
        "q8_pair lineage fields invalid",
    )
    for name, value in lineage.items():
        _require(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value),
            "q8_pair lineage {} invalid".format(name),
        )
    _equal(
        family.get("lineage_sha256"),
        _q8_canonical_sha256(lineage),
        "q8_pair.lineage_sha256",
    )
    _equal(
        lineage.get("adapter_artifact_sha256"),
        adapter_sha256,
        "q8_pair.adapter_artifact_sha256",
    )
    if base_manifest_sha256 is not None:
        _equal(
            lineage.get("base_manifest_sha256"),
            base_manifest_sha256,
            "q8_pair.base_manifest_sha256",
        )

    full = pair.get("full_precision")
    quantized = pair.get("quantized")
    _require(isinstance(full, Mapping), "q8_pair.full_precision missing")
    _require(isinstance(quantized, Mapping), "q8_pair.quantized missing")
    _equal(full.get("format"), "gguf", "q8_pair.full_precision.format")
    _equal(full.get("precision"), "F16", "q8_pair.full_precision.precision")
    _equal(quantized.get("format"), "gguf", "q8_pair.quantized.format")
    _equal(quantized.get("precision"), Q8_QUANTIZATION, "q8_pair.quantized.precision")
    _identity_equal(quantized, model, "q8_pair.quantized")

    conversion = pair.get("conversion")
    _require(isinstance(conversion, Mapping), "q8_pair.conversion missing")
    for name, expected in (
        ("method", "llama.cpp/convert_hf_to_gguf.py"),
        ("source_format", "huggingface_safetensors"),
        ("output_format", "gguf"),
        ("output_precision", "F16"),
    ):
        _equal(conversion.get(name), expected, "q8_pair.conversion." + name)
    _equal(
        conversion.get("input_sha256"),
        lineage.get("merged_checkpoint_sha256"),
        "q8_pair.conversion.input_sha256",
    )
    _equal(
        conversion.get("output_sha256"),
        full.get("sha256"),
        "q8_pair.conversion.output_sha256",
    )

    quantization = pair.get("quantization")
    _require(isinstance(quantization, Mapping), "q8_pair.quantization missing")
    for name, expected in (
        ("method", "llama.cpp/llama-quantize"),
        ("type", Q8_QUANTIZATION),
        ("direct_from_full_precision", True),
        ("allow_requantize", False),
    ):
        _equal(quantization.get(name), expected, "q8_pair.quantization." + name)
    _equal(
        quantization.get("source_sha256"),
        full.get("sha256"),
        "q8_pair.quantization.source_sha256",
    )
    _equal(
        quantization.get("output_sha256"),
        model.get("sha256"),
        "q8_pair.quantization.output_sha256",
    )
    return {
        "model_family": dict(family),
        "full_precision": dict(full),
        "quantized": dict(quantized),
        "conversion": dict(conversion),
        "quantization": dict(quantization),
    }


def _q8_command_path(root: Path, token: Any, field: str) -> Path:
    _require(isinstance(token, str) and bool(token.strip()), field + " missing")
    raw = Path(token).expanduser()
    return (raw if raw.is_absolute() else root / raw).resolve()


def _validate_q8_pair_paths(
    pair_path: Path,
    pair: Mapping[str, Any],
    model_path: Path,
    model: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    root = pair_path.resolve().parent
    conversion = _path(pair, "conversion")
    quantization = _path(pair, "quantization")
    conversion_command = conversion.get("command")
    _require(
        isinstance(conversion_command, list)
        and len(conversion_command) == 6
        and all(isinstance(value, str) and value for value in conversion_command),
        "q8_pair.conversion.command invalid",
    )
    _equal(
        Path(conversion_command[0]).name,
        "convert_hf_to_gguf.py",
        "q8_pair.conversion.command converter",
    )
    _equal(conversion_command[2], "--outfile", "q8_pair.conversion.command outfile")
    _equal(conversion_command[4], "--outtype", "q8_pair.conversion.command outtype")
    _equal(conversion_command[5].lower(), "f16", "q8_pair.conversion.command precision")
    converter = _q8_command_path(root, conversion_command[0], "q8_pair.converter")
    converter_id = _actual_identity(converter, "q8_pair.converter")
    _equal(
        converter_id["sha256"],
        conversion.get("converter_sha256"),
        "q8_pair.converter_sha256",
    )
    merged_dir = _q8_command_path(root, conversion_command[1], "q8_pair.merged_hf")
    _require(
        merged_dir.is_dir() and not merged_dir.is_symlink(),
        "q8_pair merged_hf input must be a regular directory",
    )
    merged = _actual_identity(
        merged_dir / "model.safetensors", "q8_pair.merged_checkpoint"
    )
    _equal(
        merged["sha256"],
        _path(pair, "model_family.lineage.merged_checkpoint_sha256"),
        "q8_pair.merged_checkpoint_sha256",
    )
    full_path = _q8_command_path(root, conversion_command[3], "q8_pair.full_precision")
    full = _actual_identity(full_path, "q8_pair.full_precision")
    _identity_equal(pair.get("full_precision"), full, "q8_pair.full_precision")

    quantize_command = quantization.get("command")
    _require(
        isinstance(quantize_command, list)
        and len(quantize_command) == 4
        and all(isinstance(value, str) and value for value in quantize_command),
        "q8_pair.quantization.command invalid",
    )
    _equal(
        Path(quantize_command[0]).name,
        "llama-quantize",
        "q8_pair.quantization.command binary",
    )
    _equal(quantize_command[3], Q8_QUANTIZATION, "q8_pair.quantization.command type")
    _require(
        "--allow-requantize" not in quantize_command,
        "q8_pair quantization must not allow requantization",
    )
    quantizer = _actual_identity(
        _q8_command_path(root, quantize_command[0], "q8_pair.quantizer"),
        "q8_pair.quantizer",
    )
    _equal(
        quantizer["sha256"],
        quantization.get("llama_quantize_sha256"),
        "q8_pair.llama_quantize_sha256",
    )
    quantize_input = _actual_identity(
        _q8_command_path(root, quantize_command[1], "q8_pair.quantize_input"),
        "q8_pair.quantize_input",
    )
    _identity_equal(quantize_input, full, "q8_pair.quantize_input")
    quantize_output_path = _q8_command_path(
        root, quantize_command[2], "q8_pair.quantize_output"
    )
    _equal(quantize_output_path, model_path.resolve(), "q8_pair.quantize_output.path")
    quantize_output = _actual_identity(quantize_output_path, "q8_pair.quantize_output")
    _identity_equal(quantize_output, model, "q8_pair.quantize_output")
    return {
        "converter": converter_id,
        "merged_checkpoint": merged,
        "full_precision": full,
        "quantizer": quantizer,
        "quantized": quantize_output,
    }


def _validate_q8_sha_bundle(
    root: Path,
    required_files: frozenset,
    model: Mapping[str, Any],
    manifest_sha256: str,
    remote_root: Path,
) -> Dict[str, str]:
    manifest_path = root / "SHA256SUMS.txt"
    manifest = _actual_identity(manifest_path, "q8 nano SHA256SUMS")
    _equal(manifest["sha256"], manifest_sha256, "q8 nano SHA256SUMS.sha256")
    records: Dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        _require(
            len(line) > 66 and line[64:66] == "  ",
            "q8 nano SHA256SUMS line {} invalid".format(line_number),
        )
        digest, recorded_path = line[:64], line[66:]
        _require(
            len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest),
            "q8 nano SHA256SUMS line {} digest invalid".format(line_number),
        )
        recorded = Path(recorded_path)
        _require(
            recorded.is_absolute() and recorded.parent == remote_root,
            "q8 nano SHA256SUMS line {} remote path invalid".format(line_number),
        )
        name = recorded.name
        _require(
            bool(name) and name not in {".", ".."} and name not in records,
            "q8 nano SHA256SUMS unsafe or duplicate name: {}".format(name),
        )
        records[name] = digest
    _equal(set(records), set(required_files), "q8 nano SHA256SUMS file set")
    for name, digest in records.items():
        target = root / name
        if name == "candidate.gguf" and not target.exists():
            _require(
                not target.is_symlink(),
                "q8 nano remote candidate placeholder must not be a symlink",
            )
            _equal(digest, model.get("sha256"), "q8 nano remote candidate.sha256")
            continue
        identity = _actual_identity(target, "q8 nano SHA256SUMS " + name)
        _equal(identity["sha256"], digest, "q8 nano SHA256SUMS " + name)
    return records


def _validate_q8_nano_runtime(
    root: Path,
    candidate: Mapping[str, Any],
    model: Mapping[str, Any],
) -> None:
    _identity_equal(candidate, model, "q8 nano candidate")
    _equal(candidate.get("quantization"), Q8_QUANTIZATION, "q8 nano quantization")
    candidate_path = candidate.get("path")
    _require(isinstance(candidate_path, str) and bool(candidate_path), "q8 nano candidate.path")
    _equal(_read_json_value(root / "candidate_lora.json", "q8 nano lora"), [], "q8 nano /lora-adapters")
    props = read_json_object(root / "candidate_props.json")
    _equal(props.get("model_path"), candidate_path, "q8 nano props.model_path")
    _equal(props.get("model_alias"), candidate_path, "q8 nano props.model_alias")
    _equal(
        _path(props, "default_generation_settings.params.lora"),
        [],
        "q8 nano props runtime lora",
    )
    try:
        tokens = shlex.split((root / "candidate_cmdline.txt").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError("q8 nano candidate command invalid") from exc
    _require(bool(tokens) and Path(tokens[0]).name == "llama-server", "q8 nano command is not llama-server")
    _require(
        not any(token == "--lora" or token.startswith("--lora-") for token in tokens),
        "q8 nano command must not contain LoRA options",
    )
    for forbidden in ("--no-mmap", "--no-cuda-graph", "--chat-template", "--chat-template-file"):
        _require(forbidden not in tokens, "q8 nano command contains forbidden " + forbidden)
    _equal(_command_option(tokens, "--model"), candidate_path, "q8 nano command model")
    for option, expected in (
        ("--host", "127.0.0.1"),
        ("--port", "19395"),
        ("--ctx-size", "128"),
        ("--threads", "4"),
        ("--threads-batch", "4"),
        ("--batch-size", "16"),
        ("--ubatch-size", "16"),
        ("--parallel", "1"),
        ("--gpu-layers", "99"),
        ("--reasoning", "off"),
        ("--reasoning-budget", "0"),
        ("--cache-ram", "0"),
        ("--ctx-checkpoints", "0"),
        ("--poll", "100"),
        ("--poll-batch", "1"),
    ):
        _equal(_command_option(tokens, option), expected, "q8 nano command " + option)
    for flag in ("--no-cache-prompt", "--no-cache-idle-slots", "--no-webui"):
        _equal(tokens.count(flag), 1, "q8 nano command " + flag)
    _equal(props.get("total_slots"), 1, "q8 nano props.total_slots")
    run_script = (root / ("run_nano_stage_a.sh" if (root / "run_nano_stage_a.sh").is_file() else "run_nano_stage_b.sh")).read_text(encoding="utf-8")
    _require(
        "LLAMA_SERVER_SHA256=" + Q8_NANO_EDGE_SERVER_SHA256 in run_script,
        "q8 nano runner is not bound to the verified llama-server",
    )
    _require(
        str(model.get("sha256")) in run_script,
        "q8 nano runner is not bound to the Q8 model",
    )


def _q8_memory_evidence(root: Path) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    startup_rows = _read_jsonl(
        root / "startup_memory_20ms.jsonl", "q8 nano startup memory"
    )
    inference_rows = _read_jsonl(
        root / "inference_memory_20ms.jsonl", "q8 nano inference memory"
    )
    startup = _memory_summary(startup_rows, "startup")
    inference = _memory_summary(inference_rows, "inference")
    all_rows = startup_rows + inference_rows
    previous_ns: Optional[int] = None
    longest_inference_d = 0
    for phase, rows in (("startup", startup_rows), ("inference", inference_rows)):
        consecutive_d = 0
        for index, row in enumerate(rows):
            field = "q8 nano {} memory[{}]".format(phase, index)
            monotonic_ns = _integer(row.get("monotonic_ns"), field + ".monotonic_ns")
            if previous_ns is not None:
                _require(monotonic_ns > previous_ns, field + " timestamp is not increasing")
            previous_ns = monotonic_ns
            consecutive_d = consecutive_d + 1 if row.get("state") == "D" else 0
            _equal(
                row.get("consecutive_d_samples"),
                consecutive_d,
                field + ".consecutive_d_samples",
            )
            if phase == "inference":
                longest_inference_d = max(longest_inference_d, consecutive_d)
    oom_values = [_memory_value(row, "oom_kill", "q8 nano memory") for row in all_rows]
    pswpin_values = [_memory_value(row, "pswpin", "q8 nano memory") for row in all_rows]
    pswpout_values = [_memory_value(row, "pswpout", "q8 nano memory") for row in all_rows]
    swap_values = [_memory_value(row, "vmswap_kib", "q8 nano memory") for row in all_rows]
    combined = {
        "strict_peak_bytes": max(startup["peak"], inference["peak"]),
        "minimum_mem_available_bytes": min(
            startup["minimum_available"], inference["minimum_available"]
        ),
        "maximum_vmswap_bytes": max(swap_values) * 1024,
        "oom_kill_delta": max(oom_values) - min(oom_values),
        "pswpin_pages_delta": max(pswpin_values) - min(pswpin_values),
        "pswpout_pages_delta": max(pswpout_values) - min(pswpout_values),
        "inference_longest_d": longest_inference_d,
        "startup_samples": len(startup_rows),
        "inference_samples": len(inference_rows),
    }
    _at_most(combined["strict_peak_bytes"], MAX_NANO_STRICT_PEAK_BYTES, "q8 nano strict peak")
    _at_least(combined["minimum_mem_available_bytes"], MIN_NANO_MEM_AVAILABLE_BYTES, "q8 nano minimum available")
    _equal(combined["oom_kill_delta"], 0, "q8 nano oom delta")
    _require(combined["inference_longest_d"] < 5, "q8 nano inference D-state streak")
    return startup, inference, combined


def _q8_dataset_rows(
    path: Path, scene: str, labels: str, expected_count: int
) -> Tuple[Dict[str, Any], list]:
    rows = _read_jsonl(path, "q8 {} dataset".format(scene))
    _equal(len(rows), expected_count, "q8 {} dataset count".format(scene))
    contracts = []
    for index, row in enumerate(rows):
        messages = row.get("messages")
        _require(
            isinstance(messages, list)
            and len(messages) >= 2
            and isinstance(messages[0], Mapping)
            and isinstance(messages[1], Mapping),
            "q8 {} dataset row {} messages invalid".format(scene, index),
        )
        prompt = str(messages[0].get("content", ""))
        target = str(messages[1].get("content", ""))
        _require(
            len(prompt) == 16 and prompt.isascii() and prompt.isdigit(),
            "q8 {} dataset row {} prompt invalid".format(scene, index),
        )
        _require(target in labels, "q8 {} dataset row {} target invalid".format(scene, index))
        contracts.append(
            {
                "event_id": str(row.get("event_id", index + 1)),
                "raw_prompt": prompt,
                "target": target,
            }
        )
    return _actual_identity(path, "q8 {} dataset".format(scene)), contracts


def _validate_q8_full(
    report: Mapping[str, Any],
    model: Mapping[str, Any],
    verify_runtime_files: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    _equal(report.get("schema_version"), Q8_FULL_SCHEMA, "q8_full.schema_version")
    _equal(report.get("status"), "completed", "q8_full.status")
    _equal(report.get("completed_requests"), 3360, "q8_full.completed_requests")
    runtime = report.get("runtime")
    _require(isinstance(runtime, Mapping), "q8_full.runtime missing")
    _identity_equal(runtime.get("model"), model, "q8_full.runtime.model")
    _equal(
        Path(str(_path(runtime, "model.path"))).resolve(),
        Path(str(model["path"])).resolve(),
        "q8_full.runtime.model.path",
    )
    _equal(runtime.get("resident_lora_count"), 0, "q8_full.runtime_lora_count")
    _equal(runtime.get("request_level_lora_switching"), False, "q8_full.request_switching")
    _equal(runtime.get("lora_adapters_before"), [], "q8_full.lora_before")
    _equal(runtime.get("lora_adapters_after"), [], "q8_full.lora_after")
    server = runtime.get("llama_server")
    _require(isinstance(server, Mapping), "q8_full.llama_server missing")
    for name in ("bytes", "sha256", "version", "commit"):
        _equal(server.get(name), Q8_FULL_LLAMA_SERVER[name], "q8_full.llama_server." + name)
    if verify_runtime_files:
        server_path, _ = _q8_identity_on_disk(server, "q8_full.llama_server")
    else:
        server_path = Path(str(server.get("path", ""))).expanduser().resolve()
    server_command = runtime.get("server_command")
    _require(
        isinstance(server_command, list)
        and all(isinstance(token, str) and token for token in server_command),
        "q8_full.server_command invalid",
    )
    _equal(Path(server_command[0]).resolve(), server_path, "q8_full.server_command binary")
    _require(
        not any(token == "--lora" or token.startswith("--lora-") for token in server_command),
        "q8_full.server_command contains LoRA options",
    )
    _equal(_command_option(server_command, "--model"), str(model["path"]), "q8_full.server_command model")
    for option, expected in (
        ("--ctx-size", "128"),
        ("--batch-size", "16"),
        ("--ubatch-size", "16"),
        ("--parallel", "1"),
        ("--gpu-layers", "99"),
        ("--threads", "4"),
    ):
        _equal(_command_option(server_command, option), expected, "q8_full.server_command " + option)
    for props_name in ("props_before", "props_after"):
        props = runtime.get(props_name)
        _require(isinstance(props, Mapping), "q8_full.{} missing".format(props_name))
        _equal(
            _path(props, "default_generation_settings.params.lora"),
            [],
            "q8_full.{}.runtime_lora".format(props_name),
        )

    inputs = report.get("inputs")
    _require(isinstance(inputs, Mapping), "q8_full.inputs missing")
    dataset_contracts: Dict[str, list] = {}
    dataset_identities: Dict[str, Dict[str, Any]] = {}
    for scene, labels, key in (
        ("traffic", "ABCDEF", "traffic_dataset"),
        ("industrial", "ABC", "industrial_dataset"),
    ):
        path, declared = _q8_identity_on_disk(inputs.get(key), "q8_full.inputs." + key)
        _equal(
            declared["sha256"],
            Q8_DATASETS[scene]["sha256"],
            "q8_full.{}.dataset_sha256".format(scene),
        )
        actual, rows = _q8_dataset_rows(
            path, scene, labels, int(Q8_DATASETS[scene]["count"])
        )
        dataset_identities[scene] = actual
        dataset_contracts[scene] = rows

    protocol = report.get("protocol")
    _require(isinstance(protocol, Mapping), "q8_full.protocol missing")
    fixed_protocol = {
        "order": "all_traffic_then_all_industrial",
        "traffic_rows": 2400,
        "industrial_rows": 960,
        "raw_ascii_decimal_input_chars": 16,
        "runtime_prompt_tokens": 16,
        "runtime_output_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "cache_prompt": False,
        "persistent_http_connection": True,
        "no_lora": True,
        "post_hoc_remapping": False,
        "prediction_is_exact_raw_server_content": True,
    }
    for name, expected in fixed_protocol.items():
        _equal(protocol.get(name), expected, "q8_full.protocol." + name)
    _equal(protocol.get("traffic_grammar_labels"), list("ABCDEF"), "q8_full.traffic_labels")
    _equal(protocol.get("industrial_grammar_labels"), list("ABC"), "q8_full.industrial_labels")

    records = report.get("raw_records")
    _require(isinstance(records, list) and len(records) == 3360, "q8_full.raw_records count")
    by_scene = {"traffic": [], "industrial": []}
    offsets = {"traffic": 0, "industrial": 2400}
    allowed_keys = {
        "prompt",
        "n_predict",
        "temperature",
        "top_p",
        "seed",
        "cache_prompt",
        "stream",
        "grammar",
    }
    for index, row in enumerate(records):
        field = "q8_full.raw_records[{}]".format(index)
        _require(isinstance(row, Mapping), field + " invalid")
        scene = "traffic" if index < 2400 else "industrial"
        labels = "ABCDEF" if scene == "traffic" else "ABC"
        scene_index = index - offsets[scene]
        expected = dataset_contracts[scene][scene_index]
        _equal(row.get("sequence"), index + 1, field + ".sequence")
        _equal(row.get("scene_index"), scene_index, field + ".scene_index")
        _equal(row.get("scene"), scene, field + ".scene")
        for name in ("event_id", "raw_prompt", "target"):
            _equal(row.get(name), expected[name], field + "." + name)
        _equal(row.get("prompt"), expected["raw_prompt"], field + ".prompt")
        payload = row.get("request_payload")
        _require(isinstance(payload, Mapping), field + ".request_payload missing")
        _equal(set(payload), allowed_keys, field + ".request_payload keys")
        _equal(payload.get("prompt"), expected["raw_prompt"], field + ".payload.prompt")
        for name, value in (
            ("n_predict", 1),
            ("temperature", 0.0),
            ("top_p", 1.0),
            ("seed", 42),
            ("cache_prompt", False),
            ("stream", False),
        ):
            _equal(payload.get(name), value, field + ".payload." + name)
        grammar = "root ::= " + " | ".join(json.dumps(label) for label in labels) + "\n"
        _equal(payload.get("grammar"), grammar, field + ".payload.grammar")
        _require(not any("lora" in str(key).lower() for key in payload), field + ".payload_lora")
        _equal(row.get("request_has_lora_field"), False, field + ".request_lora")
        _equal(row.get("post_hoc_remapping"), False, field + ".post_hoc")
        _equal(row.get("prompt_n"), 16, field + ".prompt_n")
        _equal(row.get("predicted_n"), 1, field + ".predicted_n")
        raw_response = row.get("raw_response")
        _require(isinstance(raw_response, Mapping), field + ".raw_response missing")
        raw_text = row.get("raw_response_text")
        _require(isinstance(raw_text, str), field + ".raw_response_text missing")
        try:
            parsed_raw = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ManifestError(field + ".raw_response_text invalid") from exc
        _equal(parsed_raw, raw_response, field + ".raw_response_text")
        _equal(raw_response.get("model"), str(model["path"]), field + ".raw_response.model")
        _equal(
            _path(raw_response, "generation_settings.lora"),
            [],
            field + ".raw_response.runtime_lora",
        )
        _equal(_path(raw_response, "timings.prompt_n"), 16, field + ".raw_response.prompt_n")
        _equal(_path(raw_response, "timings.predicted_n"), 1, field + ".raw_response.predicted_n")
        prediction = row.get("prediction")
        _equal(prediction, raw_response.get("content"), field + ".raw_prediction")
        _require(prediction in labels, field + ".prediction")
        _equal(row.get("valid"), True, field + ".valid")
        _equal(row.get("correct"), prediction == expected["target"], field + ".correct")
        by_scene[scene].append(row)

    metrics: Dict[str, float] = {"q8_full_completed_requests": 3360.0}
    for scene, labels in (("traffic", "ABCDEF"), ("industrial", "ABC")):
        calculated = _classification(by_scene[scene], labels)
        declared = _path(report, scene + ".classification")
        for name in ("count", "correct", "accuracy", "macro_f1", "weighted_f1", "valid_output_rate"):
            _close(declared.get(name), calculated[name], "q8_full.{}.{}".format(scene, name), tolerance=1e-8)
        if scene == "traffic":
            _at_least(calculated["accuracy"], TRAFFIC_MIN_ACCURACY, "q8_full.traffic.accuracy")
            _at_least(calculated["weighted_f1"], TRAFFIC_MIN_WEIGHTED_F1, "q8_full.traffic.weighted_f1")
        else:
            for name in ("accuracy", "macro_f1", "weighted_f1", "valid_output_rate"):
                _close(calculated[name], 1.0, "q8_full.industrial." + name)
        for name in ("accuracy", "macro_f1", "weighted_f1", "valid_output_rate"):
            metrics["q8_full_{}_{}".format(scene, name)] = calculated[name]
    contract = report.get("contract")
    _require(isinstance(contract, Mapping) and contract, "q8_full.contract missing")
    _require(all(value is True for value in contract.values()), "q8_full.contract failed")
    return metrics, {
        "datasets": dataset_identities,
        "dataset_rows": dataset_contracts,
        "prompt_sets": {
            scene: {row["raw_prompt"] for row in rows}
            for scene, rows in dataset_contracts.items()
        },
        "full_records": by_scene,
    }


def _q8_percentile(values: Sequence[float], percentile: float) -> float:
    _require(bool(values), "q8 percentile input is empty")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _q8_latency_summary(values: Sequence[float]) -> Dict[str, float]:
    _require(bool(values), "q8 latency input is empty")
    checked = []
    for index, value in enumerate(values):
        checked.append(_number(value, "q8 latency[{}]".format(index)))
        _require(checked[-1] >= 0.0, "q8 latency must be nonnegative")
    return {
        "count": float(len(checked)),
        "mean": sum(checked) / len(checked),
        "p50": _q8_percentile(checked, 50.0),
        "p95": _q8_percentile(checked, 95.0),
        "max": max(checked),
    }


def _validate_q8_latency_declared(
    declared: Any, calculated: Mapping[str, float], field: str, digits: int = 6
) -> None:
    _require(isinstance(declared, Mapping), field + " missing")
    _close(declared.get("count"), calculated["count"], field + ".count")
    names = (
        {
            "mean": "mean_ms",
            "p50": "median_ms",
            "p95": "p95_ms",
            "max": "max_ms",
        }
        if "mean_ms" in declared
        else {name: name for name in ("mean", "p50", "p95", "max")}
    )
    for name, declared_name in names.items():
        _close(
            declared.get(declared_name),
            round(calculated[name], digits),
            field + "." + declared_name,
            tolerance=10 ** (-digits),
        )


def _validate_q8_stage_a(
    root: Path,
    model: Mapping[str, Any],
    canonical_pair: Mapping[str, Any],
    adapter_sha256: str,
    base_manifest_sha256: str,
    full_context: Mapping[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    stage_root = root.resolve()
    _require(stage_root.is_dir() and not stage_root.is_symlink(), "q8 Stage A directory invalid")
    hashes = _validate_q8_sha_bundle(
        stage_root,
        Q8_STAGE_A_REQUIRED_FILES,
        model,
        Q8_STAGE_A_SHA_MANIFEST_SHA256,
        Q8_STAGE_A_REMOTE_ROOT,
    )
    _equal(hashes.get("candidate.gguf"), model.get("sha256"), "q8 Stage A candidate bundle")
    _equal(
        read_json_object(stage_root / "RUN_STATUS.json"),
        {"candidate_exit": 0, "restore_exit": 0},
        "q8 Stage A run status",
    )
    summary_path = stage_root / "stage_a_summary.json"
    summary_id = _actual_identity(summary_path, "q8 Stage A summary")
    _equal(summary_id["sha256"], Q8_STAGE_A_SUMMARY_SHA256, "q8 Stage A summary.sha256")
    stdout_id = _actual_identity(
        stage_root / "stage_a_summary.stdout.json", "q8 Stage A stdout summary"
    )
    _equal(stdout_id["sha256"], summary_id["sha256"], "q8 Stage A stdout summary")
    summary = read_json_object(summary_path)
    _equal(summary.get("schema_version"), Q8_NANO_STAGE_A_SCHEMA, "q8 Stage A schema")
    _equal(summary.get("passed"), True, "q8 Stage A passed")
    expected_checks = {
        "inference_consecutive_d_lt_5",
        "minimum_mem_available_gte_256mib",
        "model_sha256_matches",
        "oom_kill_delta_zero",
        "paired_ttft_records_80",
        "strict_peak_bytes_lte_1_5gb",
        "ttft_client_contracts_pass",
        "ttft_identity_matches_candidate",
        "ttft_reduction_gte_75_percent",
    }
    checks = summary.get("checks")
    _require(
        isinstance(checks, Mapping)
        and set(checks) == expected_checks
        and all(value is True for value in checks.values()),
        "q8 Stage A checks invalid",
    )
    candidate = summary.get("candidate")
    _require(isinstance(candidate, Mapping), "q8 Stage A candidate missing")
    _equal(candidate.get("path"), str(Q8_STAGE_A_REMOTE_ROOT / "candidate.gguf"), "q8 Stage A candidate.path")
    _validate_q8_nano_runtime(stage_root, candidate, model)
    for name in ("formal_lora_before.json", "formal_lora_after.json"):
        _equal(_read_json_value(stage_root / name, "q8 Stage A " + name), [], "q8 Stage A " + name)

    stage_pair = read_json_object(stage_root / "quantization_pair_manifest.json")
    _validate_q8_pair_document(
        stage_pair, model, adapter_sha256, base_manifest_sha256
    )
    _equal(
        _q8_pair_semantic_identity(stage_pair),
        _q8_pair_semantic_identity(canonical_pair),
        "q8 Stage A quantization pair",
    )

    raw_path = stage_root / "ttft_paired_raw.json"
    raw_id = _actual_identity(raw_path, "q8 Stage A TTFT raw")
    _equal(raw_id["sha256"], Q8_STAGE_A_RAW_SHA256, "q8 Stage A TTFT raw.sha256")
    raw = read_json_object(raw_path)
    _equal(raw.get("schema_version"), "nano-to-cloud-paired-streaming-ttft/v1", "q8 Stage A TTFT schema")
    _equal(raw.get("passed"), True, "q8 Stage A TTFT passed")
    _close(raw.get("threshold"), 0.75, "q8 Stage A TTFT threshold")
    protocol = raw.get("protocol")
    _require(isinstance(protocol, Mapping), "q8 Stage A TTFT protocol missing")
    for name, expected in (
        ("input_contract", "decimal16"),
        ("paired_prompts", 80),
        ("traffic_prompts", 40),
        ("industrial_prompts", 40),
        ("requested_output_tokens", 1),
        ("persistent_http_connections", False),
        ("warmups_per_endpoint", 5),
    ):
        _equal(protocol.get(name), expected, "q8 Stage A TTFT protocol." + name)
    identity = raw.get("identity")
    _require(isinstance(identity, Mapping), "q8 Stage A TTFT identity missing")
    fixed_identity = {
        "edge_model_sha256": model["sha256"],
        "edge_quantization": Q8_QUANTIZATION,
        "edge_runtime": "llama.cpp b9859",
        "teacher_digest": "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7",
        "teacher_model": "qwen3.5:9b",
        "teacher_parameter_size": "9.7B",
        "teacher_quantization": "Q4_K_M",
        "teacher_runtime": "Ollama 0.31.1",
    }
    for name, expected in fixed_identity.items():
        _equal(identity.get(name), expected, "q8 Stage A TTFT identity." + name)
    _equal(
        read_json_object(stage_root / "cloud_ollama_version.json").get("version"),
        "0.31.1",
        "q8 Stage A cloud Ollama version",
    )
    input_id = _actual_identity(stage_root / "ttft_input.json", "q8 Stage A TTFT input")
    _equal(identity.get("input_sha256"), input_id["sha256"], "q8 Stage A TTFT input sha256")
    _equal(Path(str(identity.get("input_path"))).name, "ttft_input.json", "q8 Stage A TTFT input path")
    fixture = read_json_object(stage_root / "ttft_input.json")

    records = raw.get("records")
    _require(isinstance(records, list) and len(records) == 80, "q8 Stage A TTFT records count")
    edge_values = []
    teacher_values = []
    prompts = set()
    for index, row in enumerate(records):
        field = "q8 Stage A TTFT records[{}]".format(index)
        _require(isinstance(row, Mapping), field + " invalid")
        scene = "traffic" if index < 40 else "industrial"
        scene_index = index if scene == "traffic" else index - 40
        labels = "ABCDEF" if scene == "traffic" else "ABC"
        expected_prompt = full_context["dataset_rows"][scene][scene_index]["raw_prompt"]
        _equal(row.get("id"), "{}-{:03d}".format(scene, scene_index), field + ".id")
        _equal(row.get("scene"), scene, field + ".scene")
        _equal(row.get("prompt"), expected_prompt, field + ".prompt")
        _require(expected_prompt not in prompts, field + ".prompt duplicate")
        prompts.add(expected_prompt)
        fixture_records = _path(fixture, scene + ".records")
        _require(isinstance(fixture_records, list) and len(fixture_records) >= 40, "q8 Stage A prompt fixture missing")
        _equal(
            fixture_records[scene_index].get("prompt"),
            expected_prompt,
            field + ".prompt fixture",
        )
        for endpoint in ("edge", "teacher"):
            result = row.get(endpoint)
            _require(isinstance(result, Mapping), field + "." + endpoint + " missing")
            _equal(result.get("prompt_tokens"), 16, field + "." + endpoint + ".prompt_tokens")
            _equal(result.get("output_tokens"), 1, field + "." + endpoint + ".output_tokens")
            latency = _number(result.get("ttft_ms"), field + "." + endpoint + ".ttft_ms")
            _require(latency > 0.0, field + "." + endpoint + ".ttft_ms must be positive")
            (edge_values if endpoint == "edge" else teacher_values).append(latency)
        _require(_path(row, "edge.token") in labels, field + ".edge.token")
    contracts = raw.get("contracts")
    _require(
        isinstance(contracts, Mapping)
        and set(contracts)
        == {
            "edge_output_tokens_all_1",
            "edge_prompt_tokens_all_16",
            "teacher_output_tokens_all_1",
            "teacher_prompt_tokens_all_16",
        }
        and all(value is True for value in contracts.values()),
        "q8 Stage A TTFT contracts invalid",
    )
    edge_latency = _q8_latency_summary(edge_values)
    teacher_latency = _q8_latency_summary(teacher_values)
    _validate_q8_latency_declared(raw.get("edge"), edge_latency, "q8 Stage A TTFT edge")
    _validate_q8_latency_declared(raw.get("teacher"), teacher_latency, "q8 Stage A TTFT teacher")
    reduction = 1.0 - edge_latency["mean"] / teacher_latency["mean"]
    _close(raw.get("ttft_reduction_rate"), round(reduction, 9), "q8 Stage A TTFT reduction", tolerance=1e-9)
    _at_least(reduction, 0.75, "q8 Stage A TTFT reduction")
    declared_ttft = summary.get("ttft")
    _require(isinstance(declared_ttft, Mapping), "q8 Stage A summary.ttft missing")
    _close(declared_ttft.get("edge_mean_ms"), edge_latency["mean"], "q8 Stage A edge mean")
    _close(declared_ttft.get("teacher_mean_ms"), teacher_latency["mean"], "q8 Stage A teacher mean")
    _close(declared_ttft.get("reduction"), reduction, "q8 Stage A reduction")
    _close(declared_ttft.get("minimum_reduction"), 0.75, "q8 Stage A minimum reduction")
    _equal(declared_ttft.get("raw_sha256"), raw_id["sha256"], "q8 Stage A raw binding")
    _equal(Path(str(declared_ttft.get("raw_path"))).name, raw_path.name, "q8 Stage A raw path")

    _, _, memory = _q8_memory_evidence(stage_root)
    declared_memory = summary.get("memory")
    _require(isinstance(declared_memory, Mapping), "q8 Stage A summary.memory missing")
    memory_fields = {
        "startup_samples": memory["startup_samples"],
        "inference_samples": memory["inference_samples"],
        "strict_peak_bytes": memory["strict_peak_bytes"],
        "minimum_mem_available_bytes": memory["minimum_mem_available_bytes"],
        "maximum_vmswap_bytes": memory["maximum_vmswap_bytes"],
        "oom_kill_delta": memory["oom_kill_delta"],
        "inference_max_consecutive_d_samples": memory["inference_longest_d"],
        "pswpin_pages_delta": memory["pswpin_pages_delta"],
        "pswpout_pages_delta": memory["pswpout_pages_delta"],
        "limit_bytes": MAX_NANO_STRICT_PEAK_BYTES,
    }
    for name, expected in memory_fields.items():
        _equal(declared_memory.get(name), expected, "q8 Stage A memory." + name)
    return {
        "q8_nano_stage_a_strict_peak_bytes": float(memory["strict_peak_bytes"]),
        "q8_nano_stage_a_minimum_mem_available_bytes": float(memory["minimum_mem_available_bytes"]),
        "q8_nano_stage_a_ttft_edge_mean_ms": edge_latency["mean"],
        "q8_nano_stage_a_ttft_teacher_mean_ms": teacher_latency["mean"],
        "q8_nano_stage_a_ttft_reduction": reduction,
    }, {
        "sha256_manifest": _actual_identity(stage_root / "SHA256SUMS.txt", "q8 Stage A SHA manifest"),
        "summary": summary_id,
        "raw": raw_id,
        "prompt_fixture": input_id,
    }


def _validate_q8_stage_b(
    root: Path,
    model: Mapping[str, Any],
    full_context: Mapping[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    stage_root = root.resolve()
    _require(stage_root.is_dir() and not stage_root.is_symlink(), "q8 Stage B directory invalid")
    _validate_q8_sha_bundle(
        stage_root,
        Q8_STAGE_B_REQUIRED_FILES,
        model,
        Q8_STAGE_B_SHA_MANIFEST_SHA256,
        Q8_STAGE_B_REMOTE_ROOT,
    )
    _equal(
        read_json_object(stage_root / "RUN_STATUS.json"),
        {"candidate_exit": 0, "restore_exit": 0},
        "q8 Stage B run status",
    )
    summary_path = stage_root / "stage_b_summary.json"
    summary_id = _actual_identity(summary_path, "q8 Stage B summary")
    _equal(summary_id["sha256"], Q8_STAGE_B_SUMMARY_SHA256, "q8 Stage B summary.sha256")
    stdout_id = _actual_identity(
        stage_root / "stage_b_summary.stdout.json", "q8 Stage B stdout summary"
    )
    _equal(stdout_id["sha256"], summary_id["sha256"], "q8 Stage B stdout summary")
    summary = read_json_object(summary_path)
    _equal(summary.get("schema_version"), Q8_NANO_STAGE_B_SCHEMA, "q8 Stage B schema")
    _equal(summary.get("passed"), True, "q8 Stage B passed")
    expected_checks = {
        "completed_500",
        "growth_100_to_500_lte_32mib",
        "industrial_250",
        "industrial_accuracy_1",
        "industrial_macro_f1_1",
        "industrial_weighted_f1_1",
        "inference_consecutive_d_lt_5",
        "minimum_mem_available_gte_256mib",
        "model_sha256_matches",
        "oom_kill_delta_zero",
        "overall_mean_latency_lt_200ms",
        "strict_16_to_1_no_lora",
        "strict_alternation",
        "strict_peak_lte_1_5gb",
        "traffic_250",
        "traffic_accuracy_gte_0_66",
        "traffic_weighted_f1_gte_0_65",
        "valid_output_rate_1",
    }
    checks = summary.get("checks")
    _require(
        isinstance(checks, Mapping)
        and set(checks) == expected_checks
        and all(value is True for value in checks.values()),
        "q8 Stage B checks invalid",
    )
    candidate = summary.get("candidate")
    _require(isinstance(candidate, Mapping), "q8 Stage B candidate missing")
    _equal(candidate.get("path"), str(Q8_STAGE_A_REMOTE_ROOT / "candidate.gguf"), "q8 Stage B candidate.path")
    _validate_q8_nano_runtime(stage_root, candidate, model)
    for name in ("formal_lora_before.json", "formal_lora_after.json"):
        _equal(_read_json_value(stage_root / name, "q8 Stage B " + name), [], "q8 Stage B " + name)

    raw_path = stage_root / "q8_static_raw16_500.json"
    raw_id = _actual_identity(raw_path, "q8 Stage B raw")
    _equal(raw_id["sha256"], Q8_STAGE_B_RAW_SHA256, "q8 Stage B raw.sha256")
    raw = read_json_object(raw_path)
    _equal(raw.get("schema_version"), Q8_NANO_OBSERVATION_SCHEMA, "q8 Stage B raw schema")
    for name, expected in (
        ("candidate_quantization", Q8_QUANTIZATION),
        ("status", "completed"),
        ("attempted_requests", 500),
        ("completed_requests", 500),
        ("errors", []),
    ):
        _equal(raw.get(name), expected, "q8 Stage B raw." + name)
    _equal(raw.get("stop_reason"), {"code": "completed", "request_count": 500}, "q8 Stage B stop reason")
    contract = raw.get("benchmark_contract")
    _require(isinstance(contract, Mapping), "q8 Stage B contract missing")
    fixed_contract = {
        "industrial_prefix": "",
        "input_contract": CONTEXT_ENCODER,
        "minimum_mem_available_mib": 256.0,
        "order": "strict_traffic_then_industrial_alternation",
        "per_scene_requested": 250,
        "persistent_http_connection": True,
        "request_level_lora_switching": False,
        "requests_omit_lora_fields": True,
        "required_output_tokens": 1,
        "required_prompt_tokens": 16,
        "resource_sample_every_requests": 20,
        "runtime_lora_count_required": 0,
        "server_lifecycle_modified": False,
        "total_requested": 500,
        "traffic_prefix": "",
    }
    _equal(dict(contract), fixed_contract, "q8 Stage B contract")
    runtime = raw.get("runtime")
    _require(isinstance(runtime, Mapping), "q8 Stage B runtime missing")
    _identity_equal(runtime.get("static_fused_model"), model, "q8 Stage B runtime model")
    _equal(_path(runtime, "static_fused_model.path"), str(Q8_STAGE_A_REMOTE_ROOT / "candidate.gguf"), "q8 Stage B runtime model.path")
    _equal(runtime.get("runtime_lora_count"), 0, "q8 Stage B runtime lora count")
    _equal(runtime.get("reconnects"), 0, "q8 Stage B reconnects")
    datasets = raw.get("datasets")
    _require(isinstance(datasets, Mapping), "q8 Stage B datasets missing")
    _equal(datasets.get("rows_consumed_per_scene"), 250, "q8 Stage B rows consumed")
    for scene in ("traffic", "industrial"):
        _identity_equal(datasets.get(scene), full_context["datasets"][scene], "q8 Stage B dataset " + scene)

    records = raw.get("records")
    _require(isinstance(records, list) and len(records) == 500, "q8 Stage B records count")
    by_scene = {"traffic": [], "industrial": []}
    all_latencies = []
    expected_keys = [
        "cache_prompt",
        "grammar",
        "n_predict",
        "prompt",
        "seed",
        "stream",
        "temperature",
        "top_p",
    ]
    for index, row in enumerate(records):
        field = "q8 Stage B records[{}]".format(index)
        _require(isinstance(row, Mapping), field + " invalid")
        scene = "traffic" if index % 2 == 0 else "industrial"
        scene_index = index // 2
        labels = "ABCDEF" if scene == "traffic" else "ABC"
        expected = full_context["dataset_rows"][scene][scene_index]
        expected_full = full_context["full_records"][scene][scene_index]
        _equal(row.get("sequence"), index + 1, field + ".sequence")
        _equal(row.get("scene"), scene, field + ".scene")
        _equal(row.get("event_id"), expected["event_id"], field + ".event_id")
        _equal(row.get("target"), expected["target"], field + ".target")
        prediction = row.get("prediction")
        _require(prediction in labels, field + ".prediction")
        _equal(prediction, expected_full.get("prediction"), field + ".full3360_prediction")
        _equal(row.get("correct"), prediction == expected["target"], field + ".correct")
        _equal(row.get("valid_output"), True, field + ".valid_output")
        _equal(row.get("prompt_tokens"), 16, field + ".prompt_tokens")
        _equal(row.get("output_tokens"), 1, field + ".output_tokens")
        _equal(row.get("request_has_lora_field"), False, field + ".request_has_lora_field")
        _equal(row.get("request_payload_keys"), expected_keys, field + ".request_payload_keys")
        latency = _number(row.get("latency_ms"), field + ".latency_ms")
        _require(latency >= 0.0, field + ".latency_ms negative")
        for name in ("server_prompt_ms", "server_predicted_ms"):
            _require(_number(row.get(name), field + "." + name) >= 0.0, field + "." + name + " negative")
        all_latencies.append(latency)
        by_scene[scene].append(row)

    calculated_quality = {}
    for scene, labels in (("traffic", "ABCDEF"), ("industrial", "ABC")):
        calculated = _classification(by_scene[scene], labels)
        calculated_quality[scene] = calculated
        for container_name, container in (
            ("raw", _path(raw, "summary." + scene)),
            ("summary", _path(summary, "quality." + scene)),
        ):
            for name in ("count", "correct", "accuracy", "macro_f1", "weighted_f1", "valid_output_rate"):
                _close(
                    container.get(name),
                    round(calculated[name], 9),
                    "q8 Stage B {} {}.{}".format(container_name, scene, name),
                    tolerance=1e-8,
                )
        if scene == "traffic":
            _at_least(calculated["accuracy"], TRAFFIC_MIN_ACCURACY, "q8 Stage B traffic accuracy")
            _at_least(calculated["weighted_f1"], TRAFFIC_MIN_WEIGHTED_F1, "q8 Stage B traffic weighted_f1")
        else:
            for name in ("accuracy", "macro_f1", "weighted_f1", "valid_output_rate"):
                _close(calculated[name], 1.0, "q8 Stage B industrial " + name)
        scene_latency = _q8_latency_summary([row["latency_ms"] for row in by_scene[scene]])
        _validate_q8_latency_declared(
            _path(raw, "summary.{}.latency_ms".format(scene)),
            scene_latency,
            "q8 Stage B raw {} latency".format(scene),
        )
        _validate_q8_latency_declared(
            _path(summary, "quality.{}.latency_ms".format(scene)),
            scene_latency,
            "q8 Stage B summary {} latency".format(scene),
        )
    overall_latency = _q8_latency_summary(all_latencies)
    _validate_q8_latency_declared(_path(raw, "summary.overall.latency_ms"), overall_latency, "q8 Stage B raw overall latency")
    _validate_q8_latency_declared(_path(summary, "performance.overall_latency_ms"), overall_latency, "q8 Stage B summary overall latency")
    _close(_path(raw, "summary.overall.valid_output_rate"), 1.0, "q8 Stage B overall valid rate")
    _close(_path(summary, "quality.valid_output_rate"), 1.0, "q8 Stage B summary valid rate")
    _require(overall_latency["mean"] < MAX_MEAN_LATENCY_MS, "q8 Stage B mean latency failed")

    _, _, memory = _q8_memory_evidence(stage_root)
    performance = summary.get("performance")
    _require(isinstance(performance, Mapping), "q8 Stage B performance missing")
    memory_fields = {
        "startup_samples": memory["startup_samples"],
        "inference_samples": memory["inference_samples"],
        "strict_peak_bytes": memory["strict_peak_bytes"],
        "minimum_mem_available_bytes": memory["minimum_mem_available_bytes"],
        "oom_kill_delta": memory["oom_kill_delta"],
        "inference_longest_consecutive_d_samples": memory["inference_longest_d"],
    }
    for name, expected in memory_fields.items():
        _equal(performance.get(name), expected, "q8 Stage B performance." + name)

    resource = raw.get("resource_samples")
    _require(isinstance(resource, list) and len(resource) == 26, "q8 Stage B resource samples count")
    _equal([row.get("request_count") for row in resource], list(range(0, 501, 20)), "q8 Stage B resource sample order")
    for index, row in enumerate(resource):
        field = "q8 Stage B resource_samples[{}]".format(index)
        _require(isinstance(row, Mapping), field + " invalid")
        for name in (
            "llama_vmhwm_kib",
            "llama_vmrss_kib",
            "llama_vmswap_kib",
            "mem_available_kib",
            "oom_kill",
        ):
            _require(_integer(row.get(name), field + "." + name) >= 0, field + "." + name + " negative")
    at_100 = resource[5]
    at_500 = resource[-1]
    growth = (
        at_500["llama_vmrss_kib"]
        + at_500["llama_vmswap_kib"]
        - at_100["llama_vmrss_kib"]
        - at_100["llama_vmswap_kib"]
    ) * 1024
    _equal(performance.get("growth_100_to_500_bytes"), growth, "q8 Stage B growth")
    _at_most(growth, MAX_NANO_GROWTH_BYTES, "q8 Stage B growth")
    _at_most(
        max(row["llama_vmhwm_kib"] for row in resource) * 1024,
        MAX_NANO_STRICT_PEAK_BYTES,
        "q8 Stage B resource peak",
    )
    _at_least(
        min(row["mem_available_kib"] for row in resource) * 1024,
        MIN_NANO_MEM_AVAILABLE_BYTES,
        "q8 Stage B resource minimum available",
    )
    _equal(
        max(row["oom_kill"] for row in resource) - min(row["oom_kill"] for row in resource),
        0,
        "q8 Stage B resource oom delta",
    )
    _equal(summary.get("raw", {}).get("sha256"), raw_id["sha256"], "q8 Stage B raw binding")
    _equal(Path(str(summary.get("raw", {}).get("path"))).name, raw_path.name, "q8 Stage B raw path")
    return {
        "q8_nano_stage_b_completed_requests": 500.0,
        "q8_nano_stage_b_strict_peak_bytes": float(memory["strict_peak_bytes"]),
        "q8_nano_stage_b_growth_100_to_500_bytes": float(growth),
        "q8_nano_stage_b_minimum_mem_available_bytes": float(memory["minimum_mem_available_bytes"]),
        "q8_nano_stage_b_mean_latency_ms": overall_latency["mean"],
        "q8_nano_stage_b_traffic_accuracy": calculated_quality["traffic"]["accuracy"],
        "q8_nano_stage_b_traffic_weighted_f1": calculated_quality["traffic"]["weighted_f1"],
        "q8_nano_stage_b_industrial_accuracy": calculated_quality["industrial"]["accuracy"],
        "q8_nano_stage_b_industrial_weighted_f1": calculated_quality["industrial"]["weighted_f1"],
        "q8_nano_stage_b_valid_output_rate": 1.0,
    }, {
        "sha256_manifest": _actual_identity(stage_root / "SHA256SUMS.txt", "q8 Stage B SHA manifest"),
        "summary": summary_id,
        "raw": raw_id,
    }


def validate_q8_candidate_descriptor(descriptor_path: Path) -> Dict[str, Any]:
    """Validate the frozen Q8 candidate without publishing a release package.

    The source Q4 package supplies only the PEFT/config/action training lineage.
    Every reported gate below is recomputed exclusively from Q8 evidence.
    """
    descriptor_file = descriptor_path.resolve()
    descriptor_root = descriptor_file.parent
    descriptor = read_json_object(descriptor_file)
    expected_fields = {
        "schema_version",
        "static_fused_model",
        "source_q4_release_package",
        "base_manifest",
        "quantization_pair_manifest",
        "evidence",
    }
    _equal(set(descriptor), expected_fields, "q8 descriptor fields")
    _equal(descriptor.get("schema_version"), Q8_DESCRIPTOR_SCHEMA, "q8 descriptor schema")
    descriptor_sha256 = sha256_file(descriptor_file)
    _equal(descriptor_sha256, Q8_DESCRIPTOR_SHA256, "q8 descriptor.sha256")
    model_path, model = _declared_identity(
        descriptor_root, descriptor.get("static_fused_model"), "q8 static_fused_model"
    )
    _identity_equal(model, Q8_MODEL, "q8 static_fused_model")

    base_manifest = _resolve(
        descriptor_root, descriptor.get("base_manifest"), "q8 base_manifest"
    )
    base_id = _actual_identity(base_manifest, "q8 base_manifest")
    source_package = _resolve(
        descriptor_root,
        descriptor.get("source_q4_release_package"),
        "q8 source_q4_release_package",
    )
    _require(
        source_package.is_dir() and not source_package.is_symlink(),
        "q8 source Q4 package must be a regular directory",
    )
    # This proves the source package is internally sound.  Its evaluation
    # values are intentionally neither returned nor copied into Q8 metrics.
    validate_q4_static_joint_package(source_package, base_manifest)
    source_manifest_path = source_package / MANIFEST_NAME
    source_manifest_id = _actual_identity(
        source_manifest_path, "q8 source Q4 manifest"
    )
    source_manifest = read_json_object(source_manifest_path)
    _equal(
        _path(source_manifest, "deployment.quantization"),
        "Q4_K_M",
        "q8 source package quantization",
    )
    _equal(source_manifest.get("runtime_adapters"), [], "q8 source package runtime adapters")
    _equal(
        _path(source_manifest, "deployment.runtime_lora_count"),
        0,
        "q8 source package runtime lora count",
    )
    source_adapter = _actual_identity(
        source_package / "adapter_model.safetensors", "q8 source adapter"
    )
    _identity_equal(
        source_manifest.get("adapter_artifact"), source_adapter, "q8 source adapter"
    )
    source_config = _actual_identity(
        source_package / "adapter_config.json", "q8 source adapter config"
    )
    source_action = _actual_identity(
        source_package / "action_mapping.json", "q8 source action mapping"
    )
    _equal(
        source_manifest.get("action_mapping", {}).get("sha256"),
        source_action["sha256"],
        "q8 source action mapping sha256",
    )

    pair_path = _resolve(
        descriptor_root,
        descriptor.get("quantization_pair_manifest"),
        "q8 quantization_pair_manifest",
    )
    pair_id = _actual_identity(pair_path, "q8 quantization pair")
    _equal(pair_id["sha256"], Q8_PAIR_SHA256, "q8 quantization pair.sha256")
    pair = read_json_object(pair_path)
    pair_contract = _validate_q8_pair_document(
        pair, model, source_adapter["sha256"], base_id["sha256"]
    )
    pair_assets = _validate_q8_pair_paths(pair_path, pair, model_path, model)

    evidence = descriptor.get("evidence")
    _require(isinstance(evidence, Mapping), "q8 descriptor evidence missing")
    _equal(
        set(evidence),
        {"full_task", "nano_stage_a_dir", "nano_stage_b_dir"},
        "q8 descriptor evidence fields",
    )
    full_path = _resolve(
        descriptor_root, evidence.get("full_task"), "q8 evidence.full_task"
    )
    full_id = _actual_identity(full_path, "q8 full3360 evidence")
    _equal(full_id["sha256"], Q8_FULL_REPORT_SHA256, "q8 full3360 evidence.sha256")
    full_metrics, full_context = _validate_q8_full(
        read_json_object(full_path), model, verify_runtime_files=True
    )
    stage_a_root = _resolve(
        descriptor_root,
        evidence.get("nano_stage_a_dir"),
        "q8 evidence.nano_stage_a_dir",
    )
    stage_b_root = _resolve(
        descriptor_root,
        evidence.get("nano_stage_b_dir"),
        "q8 evidence.nano_stage_b_dir",
    )
    stage_a_metrics, stage_a_ids = _validate_q8_stage_a(
        stage_a_root,
        model,
        pair,
        source_adapter["sha256"],
        base_id["sha256"],
        full_context,
    )
    stage_b_metrics, stage_b_ids = _validate_q8_stage_b(
        stage_b_root, model, full_context
    )
    metrics = dict(full_metrics)
    metrics.update(stage_a_metrics)
    metrics.update(stage_b_metrics)
    _require(
        metrics and all(name.startswith("q8_") for name in metrics),
        "q8 profile metrics contain non-Q8 evidence",
    )
    _equal(set(metrics), set(Q8_EXPECTED_METRICS), "q8 profile metric fields")
    for name, expected in Q8_EXPECTED_METRICS.items():
        _close(metrics[name], expected, "q8 profile metrics." + name)
    return {
        "status": "valid",
        "schema_version": Q8_DESCRIPTOR_SCHEMA,
        "release_profile": Q8_QUANTIZATION,
        "descriptor_sha256": descriptor_sha256,
        "static_fused_model": _portable(model),
        "runtime_contract": {
            "model_mode": MODEL_MODE,
            "runtime": "llama.cpp",
            "quantization": Q8_QUANTIZATION,
            "runtime_adapters": [],
            "runtime_lora_count": 0,
            "request_level_lora_switching": False,
            "input_tokens": 16,
            "output_tokens": 1,
        },
        "training_lineage": {
            "source_release_role": "training_provenance_only",
            "source_release_evaluation_reused": False,
            "q4_metrics_reused": False,
            "source_release_manifest": _portable(source_manifest_id),
            "adapter_artifact": _portable(source_adapter),
            "adapter_config": _portable(source_config),
            "action_mapping": _portable(source_action),
            "base_manifest": _portable(base_id),
        },
        "quantization_pair": {
            "manifest": _portable(pair_id),
            "model_family": dict(pair_contract["model_family"]),
            "artifacts": {
                name: _portable(identity) for name, identity in pair_assets.items()
            },
        },
        "q8_evidence": {
            "full_task": _portable(full_id),
            "nano_stage_a": {
                name: _portable(identity) for name, identity in stage_a_ids.items()
            },
            "nano_stage_b": {
                name: _portable(identity) for name, identity in stage_b_ids.items()
            },
        },
        "metrics": metrics,
    }


def _q8_portable_identity(
    record: Any, name: str, size: int, digest: str, field: str
) -> None:
    _require(isinstance(record, Mapping), field + " identity missing")
    _equal(record.get("name"), name, field + ".name")
    _equal(record.get("bytes"), size, field + ".bytes")
    _equal(record.get("sha256"), digest, field + ".sha256")


def _validate_q8_package_summary(
    summary: Mapping[str, Any], manifest: Mapping[str, Any], package_root: Path
) -> Dict[str, Any]:
    _equal(
        summary.get("schema_version"),
        Q8_PACKAGE_SUMMARY_SCHEMA,
        "q8 package summary.schema_version",
    )
    _equal(summary.get("descriptor_sha256"), Q8_DESCRIPTOR_SHA256, "q8 package descriptor")
    _equal(
        summary.get("release_authorized_by_packager"),
        True,
        "q8 package release authorization",
    )
    model = _path(summary, "assets.static_q8_0")
    _q8_portable_identity(
        model,
        "joint_static_raw16.Q8_0.gguf",
        Q8_MODEL["bytes"],
        Q8_MODEL["sha256"],
        "q8 package model",
    )
    runtime = summary.get("runtime_contract")
    _require(isinstance(runtime, Mapping), "q8 package runtime contract missing")
    _equal(
        dict(runtime),
        {
            "model_mode": MODEL_MODE,
            "runtime": "llama.cpp",
            "quantization": Q8_QUANTIZATION,
            "runtime_adapters": [],
            "runtime_lora_count": 0,
            "request_level_lora_switching": False,
            "input_tokens": 16,
            "output_tokens": 1,
        },
        "q8 package runtime contract",
    )
    training = summary.get("training_lineage")
    _require(isinstance(training, Mapping), "q8 package training lineage missing")
    for name, expected in (
        ("source_release_role", "training_provenance_only"),
        ("source_release_evaluation_reused", False),
        ("q4_metrics_reused", False),
    ):
        _equal(training.get(name), expected, "q8 package training_lineage." + name)
    package_adapter = _actual_identity(
        package_root / "adapter_model.safetensors", "q8 package adapter"
    )
    package_config = _actual_identity(
        package_root / "adapter_config.json", "q8 package adapter config"
    )
    package_action = _actual_identity(
        package_root / "action_mapping.json", "q8 package action mapping"
    )
    _identity_equal(training.get("adapter_artifact"), package_adapter, "q8 summary adapter")
    _identity_equal(training.get("adapter_config"), package_config, "q8 summary adapter config")
    _identity_equal(training.get("action_mapping"), package_action, "q8 summary action mapping")
    _q8_portable_identity(
        training.get("base_manifest"),
        "base_manifest.json",
        2726,
        "2832e36a971974075c14694b1e2af506d0183e6a229eff2871b0283d24179af3",
        "q8 summary base manifest",
    )

    pair = summary.get("quantization_pair")
    _require(isinstance(pair, Mapping), "q8 package quantization pair missing")
    _q8_portable_identity(
        pair.get("manifest"),
        "quantization_pair_q8_v2.json",
        2337,
        Q8_PAIR_SHA256,
        "q8 package quantization pair manifest",
    )
    family = pair.get("model_family")
    _require(isinstance(family, Mapping), "q8 package model family missing")
    _equal(family.get("model_id"), "Qwen/Qwen3.5-0.8B", "q8 package model family")
    _equal(
        _path(family, "lineage.adapter_artifact_sha256"),
        package_adapter["sha256"],
        "q8 package pair adapter lineage",
    )
    pair_assets = pair.get("artifacts")
    _require(isinstance(pair_assets, Mapping), "q8 package pair artifacts missing")
    expected_pair_assets = {
        "converter": (
            "convert_hf_to_gguf.py",
            12_592,
            "c819f18fb22927b49fabc3b35d1c9e21ee638b3817eccd1bd4efbcc7116eeb4d",
        ),
        "merged_checkpoint": (
            "model.safetensors",
            1_504_827_288,
            "d62db3326680f33e442c108d71189803d1bbc76ffc9dd69db4a118ba9a166aa3",
        ),
        "full_precision": (
            "joint-static-raw16.F16.gguf",
            1_516_736_512,
            "475b46046769242447617ecae1d7cf537974ffe9202255d2814d048f2a0c3d4d",
        ),
        "quantizer": (
            "llama-quantize",
            17_912,
            "4cf43d966e52f8d1d69b66aff6d7ffa5b69b8832dbcc8fed138e3e76bdca3445",
        ),
        "quantized": (
            "joint_static_raw16.Q8_0.gguf",
            Q8_MODEL["bytes"],
            Q8_MODEL["sha256"],
        ),
    }
    _equal(set(pair_assets), set(expected_pair_assets), "q8 package pair artifact fields")
    for name, (asset_name, size, digest) in expected_pair_assets.items():
        _q8_portable_identity(
            pair_assets[name], asset_name, size, digest, "q8 package pair assets." + name
        )

    evidence = summary.get("q8_evidence")
    _require(isinstance(evidence, Mapping), "q8 package evidence missing")
    _q8_portable_identity(
        evidence.get("full_task"),
        "q8_full_traffic2400_industrial960.json",
        18_428_647,
        Q8_FULL_REPORT_SHA256,
        "q8 package full3360 evidence",
    )
    stage_a = evidence.get("nano_stage_a")
    stage_b = evidence.get("nano_stage_b")
    _require(isinstance(stage_a, Mapping), "q8 package Stage A evidence missing")
    _require(isinstance(stage_b, Mapping), "q8 package Stage B evidence missing")
    _q8_portable_identity(
        stage_a.get("sha256_manifest"),
        "SHA256SUMS.txt",
        5169,
        Q8_STAGE_A_SHA_MANIFEST_SHA256,
        "q8 package Stage A SHA manifest",
    )
    _q8_portable_identity(
        stage_a.get("summary"),
        "stage_a_summary.json",
        1433,
        Q8_STAGE_A_SUMMARY_SHA256,
        "q8 package Stage A summary",
    )
    _q8_portable_identity(
        stage_a.get("raw"),
        "ttft_paired_raw.json",
        38_285,
        Q8_STAGE_A_RAW_SHA256,
        "q8 package Stage A raw",
    )
    _q8_portable_identity(
        stage_a.get("prompt_fixture"),
        "ttft_input.json",
        2_481_336,
        "642b134414e8b1de3a1a7f297708345dc0dae8c120d74ecded17c50e20102b72",
        "q8 package Stage A prompt fixture",
    )
    _q8_portable_identity(
        stage_b.get("sha256_manifest"),
        "SHA256SUMS.txt",
        4521,
        Q8_STAGE_B_SHA_MANIFEST_SHA256,
        "q8 package Stage B SHA manifest",
    )
    _q8_portable_identity(
        stage_b.get("summary"),
        "stage_b_summary.json",
        3723,
        Q8_STAGE_B_SUMMARY_SHA256,
        "q8 package Stage B summary",
    )
    _q8_portable_identity(
        stage_b.get("raw"),
        "q8_static_raw16_500.json",
        320_548,
        Q8_STAGE_B_RAW_SHA256,
        "q8 package Stage B raw",
    )
    _equal(
        summary.get("ttft_prompt_fixture_policy"),
        {"role": "prompt_selection_only", "q4_metrics_reused": False},
        "q8 package TTFT prompt fixture policy",
    )

    metrics = summary.get("metrics")
    _require(isinstance(metrics, Mapping), "q8 package metrics missing")
    _equal(set(metrics), set(Q8_EXPECTED_METRICS), "q8 package metric fields")
    for name, expected in Q8_EXPECTED_METRICS.items():
        _close(metrics.get(name), expected, "q8 package metrics." + name)
    _equal(summary.get("fixed_gates"), _q8_gate_rows(), "q8 package fixed gates")
    deployment = manifest.get("deployment")
    _require(isinstance(deployment, Mapping), "q8 package deployment missing")
    _equal(deployment.get("artifact_sha256"), model.get("sha256"), "q8 package model SHA binding")
    _equal(deployment.get("artifact_bytes"), model.get("bytes"), "q8 package model bytes binding")
    return {
        "model": dict(model),
        "metrics": {name: float(metrics[name]) for name in metrics},
        "summary": dict(summary),
    }


def validate_q8_static_joint_package(
    package_dir: Path, base_manifest_path: Path
) -> Dict[str, Any]:
    generic = validate_adapter_package(
        package_dir, base_manifest_path, require_gates=True
    )
    root = package_dir.resolve()
    manifest = read_json_object(root / MANIFEST_NAME)
    _equal(manifest.get("package_kind"), PACKAGE_KIND, "q8 package.package_kind")
    _equal(manifest.get("release_profile"), Q8_QUANTIZATION, "q8 package.release_profile")
    _equal(
        manifest.get("candidate_descriptor_sha256"),
        Q8_DESCRIPTOR_SHA256,
        "q8 package descriptor binding",
    )
    _equal(manifest.get("runtime_adapters"), [], "q8 package.runtime_adapters")
    _equal(manifest.get("adapter_id"), Q8_PACKAGE_ADAPTER_ID, "q8 package.adapter_id")
    _equal(manifest.get("version"), Q8_PACKAGE_VERSION, "q8 package.version")
    deployment = manifest.get("deployment")
    _require(isinstance(deployment, Mapping), "q8 package deployment missing")
    fixed_deployment = {
        "runtime": "llama.cpp",
        "format": "gguf",
        "quantization": Q8_QUANTIZATION,
        "artifact_sha256": Q8_MODEL["sha256"],
        "artifact_bytes": Q8_MODEL["bytes"],
        "max_input_tokens": 16,
        "max_output_tokens": 1,
        "thinking": False,
        "model_mode": MODEL_MODE,
        "runtime_adapters": [],
        "runtime_lora_count": 0,
        "request_level_lora_switching": False,
        "scene_prefixes": {"traffic": "", "industrial": ""},
        "input_tokens": 16,
        "output_tokens": 1,
    }
    for name, expected in fixed_deployment.items():
        _equal(deployment.get(name), expected, "q8 package deployment." + name)
    training = manifest.get("training_lineage")
    _require(isinstance(training, Mapping), "q8 package training_lineage missing")
    expected_training_fields = {
        "peft_artifact_runtime_loaded": False,
        "role": "source_q4_training_provenance_only",
        "source_release_evaluation_reused": False,
        "q4_metrics_reused": False,
        "quantization_pair_manifest_sha256": Q8_PAIR_SHA256,
    }
    for name, expected in expected_training_fields.items():
        _equal(training.get(name), expected, "q8 package training_lineage." + name)
    _equal(
        manifest.get("quantization_pair"),
        {
            "manifest_sha256": Q8_PAIR_SHA256,
            "full_precision_sha256": "475b46046769242447617ecae1d7cf537974ffe9202255d2814d048f2a0c3d4d",
            "quantized_sha256": Q8_MODEL["sha256"],
        },
        "q8 package quantization pair binding",
    )
    evidence = _path(manifest, "evaluation.evidence")
    _equal(set(evidence), {"q8_static_raw16_gate_summary"}, "q8 package evidence fields")
    summary_record = evidence["q8_static_raw16_gate_summary"]
    summary_path = root / str(summary_record.get("path", ""))
    summary = read_json_object(summary_path)
    checked = _validate_q8_package_summary(summary, manifest, root)
    manifest_metrics = _path(manifest, "evaluation.metrics")
    _equal(set(manifest_metrics), set(Q8_EXPECTED_METRICS), "q8 package manifest metric fields")
    for name, expected in Q8_EXPECTED_METRICS.items():
        _close(manifest_metrics.get(name), expected, "q8 package manifest metrics." + name)
    expected_sources = {
        name: {
            "evidence": "q8_static_raw16_gate_summary",
            "path": "metrics." + name,
        }
        for name in Q8_EXPECTED_METRICS
    }
    _equal(
        _path(manifest, "evaluation.metric_sources"),
        expected_sources,
        "q8 package metric sources",
    )
    _equal(_path(manifest, "evaluation.gates"), _q8_gate_rows(), "q8 package gates")
    _equal(
        _path(summary, "training_lineage.adapter_artifact.sha256"),
        _path(manifest, "adapter_artifact.sha256"),
        "q8 package adapter training binding",
    )
    return {
        "status": "valid",
        "generic_validation": generic,
        "package_kind": PACKAGE_KIND,
        "release_profile": Q8_QUANTIZATION,
        "runtime_adapters": [],
        "static_fused_model": checked["model"],
        "gate_summary_sha256": sha256_file(summary_path),
    }


def build_q8_static_joint_release(
    descriptor_path: Path, output_dir: Path
) -> Dict[str, Any]:
    descriptor_file = descriptor_path.resolve()
    descriptor_root = descriptor_file.parent
    output = output_dir.resolve()
    if output.exists():
        raise ManifestError(
            "immutable Q8 static joint package output already exists: {}".format(
                output
            )
        )

    # No staging or package mutation is allowed until every external Q8 source
    # has passed the full candidate validator.
    profile = validate_q8_candidate_descriptor(descriptor_file)
    _equal(profile.get("status"), "valid", "q8 candidate validation status")
    descriptor = read_json_object(descriptor_file)
    model_path, model = _declared_identity(
        descriptor_root, descriptor.get("static_fused_model"), "q8 static_fused_model"
    )
    source_package = _resolve(
        descriptor_root,
        descriptor.get("source_q4_release_package"),
        "q8 source_q4_release_package",
    )
    source_manifest = read_json_object(source_package / MANIFEST_NAME)
    base_manifest = _resolve(
        descriptor_root, descriptor.get("base_manifest"), "q8 base_manifest"
    )
    summary = {
        "schema_version": Q8_PACKAGE_SUMMARY_SCHEMA,
        "descriptor_sha256": profile["descriptor_sha256"],
        "assets": {"static_q8_0": _portable(model)},
        "training_lineage": profile["training_lineage"],
        "quantization_pair": profile["quantization_pair"],
        "q8_evidence": profile["q8_evidence"],
        "ttft_prompt_fixture_policy": {
            "role": "prompt_selection_only",
            "q4_metrics_reused": False,
        },
        "runtime_contract": profile["runtime_contract"],
        "metrics": profile["metrics"],
        "fixed_gates": _q8_gate_rows(),
        "release_authorized_by_packager": True,
    }
    action_mapping = source_package / "action_mapping.json"
    training = source_manifest.get("training")
    input_contract = source_manifest.get("input_contract")
    _require(isinstance(training, Mapping), "q8 source package training missing")
    _require(
        isinstance(input_contract, Mapping),
        "q8 source package input_contract missing",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=output.name + ".q8-stage-", dir=str(output.parent))
    )
    staging = staging_parent / "package"
    try:
        work = staging_parent / "inputs"
        work.mkdir()
        summary_path = work / "q8_static_raw16_gate_summary.json"
        write_json_object(summary_path, summary)
        sources = {
            name: {
                "evidence": "q8_static_raw16_gate_summary",
                "path": "metrics." + name,
            }
            for name in profile["metrics"]
        }
        spec = {
            "schema_version": "edge-llm-package-spec/v1",
            "adapter_id": Q8_PACKAGE_ADAPTER_ID,
            "scene": TRAFFIC_SCENE,
            "version": Q8_PACKAGE_VERSION,
            "adapter_source": str(source_package),
            "action_mapping": str(action_mapping),
            "input_contract": dict(input_contract),
            "training": dict(training),
            "evaluation": {
                "evidence": {
                    "q8_static_raw16_gate_summary": str(summary_path)
                },
                "metric_sources": sources,
                "gates": _q8_gate_rows(),
            },
            "deployment": {
                "artifact": str(model_path),
                "runtime": "llama.cpp",
                "format": "gguf",
                "quantization": Q8_QUANTIZATION,
                "max_input_tokens": 16,
                "max_output_tokens": 1,
                "thinking": False,
            },
        }
        spec_path = work / "package_spec.json"
        write_json_object(spec_path, spec)
        build_adapter_package(
            project_root=descriptor_root,
            base_manifest_path=base_manifest,
            spec_path=spec_path,
            output_dir=staging,
        )
        manifest_path = staging / MANIFEST_NAME
        manifest = read_json_object(manifest_path)
        manifest["package_kind"] = PACKAGE_KIND
        manifest["release_profile"] = Q8_QUANTIZATION
        manifest["candidate_descriptor_sha256"] = Q8_DESCRIPTOR_SHA256
        manifest["runtime_adapters"] = []
        manifest["training_lineage"] = {
            "peft_artifact_runtime_loaded": False,
            "role": "source_q4_training_provenance_only",
            "source_release_evaluation_reused": False,
            "q4_metrics_reused": False,
            "source_release_manifest_sha256": profile["training_lineage"][
                "source_release_manifest"
            ]["sha256"],
            "quantization_pair_manifest_sha256": Q8_PAIR_SHA256,
        }
        manifest["quantization_pair"] = {
            "manifest_sha256": Q8_PAIR_SHA256,
            "full_precision_sha256": profile["quantization_pair"]["artifacts"][
                "full_precision"
            ]["sha256"],
            "quantized_sha256": model["sha256"],
        }
        deployment = manifest["deployment"]
        deployment.update(
            {
                "model_mode": MODEL_MODE,
                "runtime_adapters": [],
                "runtime_lora_count": 0,
                "request_level_lora_switching": False,
                "scene_prefixes": {"traffic": "", "industrial": ""},
                "input_tokens": 16,
                "output_tokens": 1,
            }
        )
        write_json_object(manifest_path, manifest)
        validation = validate_q8_static_joint_package(staging, base_manifest)
        os.replace(str(staging), str(output))
    finally:
        if staging_parent.exists():
            shutil.rmtree(staging_parent)
    final_validation = validate_q8_static_joint_package(output, base_manifest)
    final_validation["package"] = str(output)
    final_validation["manifest_sha256"] = sha256_file(output / MANIFEST_NAME)
    final_validation["staging_validation"] = validation
    return final_validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate statically fused raw16 release evidence"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--descriptor", required=True)
    build.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--base", required=True)
    validate.add_argument("--package", required=True)
    validate_q8 = subparsers.add_parser("validate-q8-candidate")
    validate_q8.add_argument("--descriptor", required=True)
    build_q8 = subparsers.add_parser("build-q8")
    build_q8.add_argument("--descriptor", required=True)
    build_q8.add_argument("--output", required=True)
    validate_q8_package = subparsers.add_parser("validate-q8-package")
    validate_q8_package.add_argument("--base", required=True)
    validate_q8_package.add_argument("--package", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_q4_static_joint_release(Path(args.descriptor), Path(args.output))
    elif args.command == "validate":
        result = validate_q4_static_joint_package(Path(args.package), Path(args.base))
    elif args.command == "validate-q8-candidate":
        result = validate_q8_candidate_descriptor(Path(args.descriptor))
    elif args.command == "build-q8":
        result = build_q8_static_joint_release(
            Path(args.descriptor), Path(args.output)
        )
    else:
        result = validate_q8_static_joint_package(
            Path(args.package), Path(args.base)
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
