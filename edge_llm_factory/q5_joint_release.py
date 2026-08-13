"""Fail-closed packager for clean Q5_K_M plus one joint process-default LoRA.

This is intentionally separate from ``q6_multiscene_release``: the latter
describes two request-selected LoRAs on Q6, while this module accepts exactly
one traffic+industrial LoRA applied process-wide at scale 1.  It validates all
raw evidence before the generic immutable adapter-package builder is invoked;
it never writes a release registry or contacts a runtime.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from edge_llm_factory.adapter_package import build_adapter_package
from edge_llm_factory.contracts import (
    ManifestError,
    read_json_object,
    sha256_file,
    write_json_object,
)


SCHEMA = "edge-llm-q5-joint-release-input/v1"
SUMMARY_SCHEMA = "edge-llm-q5-joint-release-gate-summary/v1"
FORMAL_GATE_SCHEMA = "edge-llm-joint-traffic-industrial-gate/v2"
Q5_FULL_SCHEMA = "edge-llm-joint-q5-f16-lora-full-deployment-validation/v1"
NANO_OBSERVATION_SCHEMA = "edge-llm-joint-single-adapter-alternating-observation/v1"
NANO_GATE_SCHEMA = "q5km-single-f16-nano-observational-memory-gate/v1"

TRAFFIC_SCENE = "freeway_traffic_management"
MAX_NANO_STRICT_PEAK_BYTES = 1_500_000_000
TRAFFIC_MIN_ACCURACY = 0.66
TRAFFIC_MIN_WEIGHTED_F1 = 0.65
INDUSTRIAL_REQUIRED_SCORE = 1.0
MAX_MEAN_LATENCY_MS = 200.0


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


def _equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ManifestError(
            "{} mismatch: expected {!r}, got {!r}".format(field, expected, actual)
        )


def _at_least(actual: Any, expected: float, field: str) -> float:
    value = _number(actual, field)
    if value < expected:
        raise ManifestError("{} is {} but must be >= {}".format(field, value, expected))
    return value


def _at_most(actual: Any, expected: float, field: str) -> float:
    value = _number(actual, field)
    if value > expected:
        raise ManifestError("{} is {} but must be <= {}".format(field, value, expected))
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


def _declared_identity(
    root: Path, value: Any, field: str, expected_path: Optional[Path] = None
) -> Tuple[Path, Dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ManifestError("{} must be an identity object".format(field))
    path = _resolve(root, value.get("path"), field + ".path")
    if expected_path is not None:
        _equal(path, expected_path.resolve(), field + ".path")
    actual = _actual_identity(path, field)
    _equal(value.get("bytes"), actual["bytes"], field + ".bytes")
    _equal(value.get("sha256"), actual["sha256"], field + ".sha256")
    return path, actual


def _identity_equal(record: Any, expected: Mapping[str, Any], field: str) -> None:
    if not isinstance(record, Mapping):
        raise ManifestError("{} identity is missing".format(field))
    _equal(record.get("bytes"), expected["bytes"], field + ".bytes")
    _equal(record.get("sha256"), expected["sha256"], field + ".sha256")


def _evidence_files(root: Path, value: Any) -> Dict[str, Path]:
    if not isinstance(value, Mapping):
        raise ManifestError("evidence must be an object")
    names = (
        "formal_traffic",
        "formal_industrial",
        "formal_gate",
        "q5_full",
        "nano_observation",
        "nano_gate",
    )
    result = {}
    for name in names:
        path = _resolve(root, value.get(name), "evidence." + name)
        _actual_identity(path, "evidence." + name)
        result[name] = path
    return result


def _validate_formal_report(
    scene: str,
    report: Mapping[str, Any],
    adapter_artifacts: Mapping[str, Mapping[str, Any]],
) -> Dict[str, float]:
    traffic = scene == "traffic"
    allowed = list("ABCDEF" if traffic else "ABC")
    count = 2400 if traffic else 960
    prefix = "T" if traffic else "I"
    _equal(report.get("evaluation_mode"), "joint_formal", scene + ".evaluation_mode")
    _equal(report.get("joint_scene"), scene, scene + ".joint_scene")
    _equal(report.get("prompt_format"), "raw_task", scene + ".prompt_format")
    _equal(report.get("prompt_prefix"), prefix, scene + ".prompt_prefix")
    _equal(report.get("required_prompt_tokens"), 17, scene + ".required_prompt_tokens")
    _equal(report.get("count"), count, scene + ".count")
    _equal(report.get("valid_output_rate"), 1.0, scene + ".valid_output_rate")
    _equal(report.get("test_set_used_for_training"), False, scene + ".test_set_used_for_training")
    _equal(
        report.get("expected_adapter_weights_sha256"),
        adapter_artifacts["weights"]["sha256"],
        scene + ".expected_adapter_weights_sha256",
    )
    formal_artifacts = report.get("adapter_artifacts")
    if not isinstance(formal_artifacts, Mapping):
        raise ManifestError("{} adapter_artifacts missing".format(scene))
    for name, expected in adapter_artifacts.items():
        _identity_equal(formal_artifacts.get(name), expected, scene + ".adapter_artifacts." + name)

    precision = report.get("precision")
    _require(isinstance(precision, Mapping), scene + " precision missing")
    _equal(precision.get("requested"), "bfloat16", scene + ".precision.requested")
    _equal(precision.get("effective"), "bfloat16", scene + ".precision.effective")
    _equal(precision.get("cuda_available"), True, scene + ".precision.cuda_available")
    constraint = report.get("decoding_constraint")
    _require(isinstance(constraint, Mapping), scene + " decoding constraint missing")
    _equal(constraint.get("enabled"), True, scene + ".constraint.enabled")
    _equal(constraint.get("allowed_tokens"), allowed, scene + ".constraint.allowed_tokens")
    _equal(constraint.get("applies_before_sampling"), True, scene + ".constraint.before_sampling")
    _equal(constraint.get("post_hoc_remapping"), False, scene + ".constraint.post_hoc")

    samples = report.get("samples")
    _require(isinstance(samples, list) and len(samples) == count, scene + " samples count invalid")
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ManifestError("{} sample {} invalid".format(scene, index))
        _equal(sample.get("valid"), True, "{}.samples[{}].valid".format(scene, index))
        _equal(sample.get("prompt_tokens"), 17, "{}.samples[{}].prompt_tokens".format(scene, index))
        generated = sample.get("generated_token_ids")
        _require(
            isinstance(generated, list) and len(generated) == 1,
            "{}.samples[{}] must have one output token".format(scene, index),
        )
        _require(sample.get("prediction") in allowed, "{} sample prediction outside slots".format(scene))

    accuracy = _number(report.get("decision_accuracy"), scene + ".accuracy")
    macro_f1 = _number(report.get("macro_f1"), scene + ".macro_f1")
    weighted_f1 = _number(report.get("weighted_f1"), scene + ".weighted_f1")
    if traffic:
        _at_least(accuracy, TRAFFIC_MIN_ACCURACY, scene + ".accuracy")
        _at_least(weighted_f1, TRAFFIC_MIN_WEIGHTED_F1, scene + ".weighted_f1")
    else:
        for name, value in (("accuracy", accuracy), ("macro_f1", macro_f1), ("weighted_f1", weighted_f1)):
            _equal(value, INDUSTRIAL_REQUIRED_SCORE, scene + "." + name)
    return {
        "formal_{}_accuracy".format(scene): accuracy,
        "formal_{}_macro_f1".format(scene): macro_f1,
        "formal_{}_weighted_f1".format(scene): weighted_f1,
        "formal_{}_valid_output_rate".format(scene): 1.0,
    }


def _validate_formal(
    traffic: Mapping[str, Any],
    industrial: Mapping[str, Any],
    gate: Mapping[str, Any],
    adapter_artifacts: Mapping[str, Mapping[str, Any]],
) -> Dict[str, float]:
    metrics = {}
    metrics.update(_validate_formal_report("traffic", traffic, adapter_artifacts))
    metrics.update(_validate_formal_report("industrial", industrial, adapter_artifacts))
    _equal(gate.get("schema_version"), FORMAL_GATE_SCHEMA, "formal_gate.schema_version")
    _equal(gate.get("passed"), True, "formal_gate.passed")
    _equal(gate.get("one_adapter_two_isolated_tests"), True, "formal_gate.one_adapter")
    gate_artifacts = gate.get("adapter_artifacts")
    _require(isinstance(gate_artifacts, Mapping), "formal_gate.adapter_artifacts missing")
    for name, expected in adapter_artifacts.items():
        _identity_equal(gate_artifacts.get(name), expected, "formal_gate.adapter_artifacts." + name)
    for field in (
        "base_manifest_identity",
        "snapshot_manifest_identity",
        "dataset_manifest_identity",
        "dataset_artifacts",
        "evaluator_identity",
    ):
        _equal(traffic.get(field), industrial.get(field), "formal reports " + field)
        _equal(gate.get(field), traffic.get(field), "formal_gate." + field)
    results = gate.get("results")
    _require(isinstance(results, Mapping), "formal_gate.results missing")
    for scene in ("traffic", "industrial"):
        result = results.get(scene)
        _require(isinstance(result, Mapping), "formal_gate {} result missing".format(scene))
        _equal(result.get("passed"), True, "formal_gate.{}.passed".format(scene))
        checks = result.get("checks")
        _require(
            isinstance(checks, Mapping) and checks and all(value is True for value in checks.values()),
            "formal_gate.{} checks did not all pass".format(scene),
        )
    metrics["formal_total_requests"] = 3360.0
    return metrics


def _runtime_contract(report: Mapping[str, Any], scene: str) -> None:
    contract = _path(report, scene + ".contract")
    _require(isinstance(contract, Mapping), "q5_full {} contract missing".format(scene))
    for name in (
        "input_chars_all_17_prefix_plus_16_digits",
        "prompt_tokens_all_17",
        "output_tokens_all_1",
        "grammar_applied_before_sampling",
        "requests_have_no_lora_fields",
    ):
        _equal(contract.get(name), True, "q5_full.{}.contract.{}".format(scene, name))
    _equal(contract.get("post_hoc_remapping"), False, "q5_full.{}.contract.post_hoc".format(scene))


def _validate_q5_full(
    report: Mapping[str, Any], base: Mapping[str, Any], adapter: Mapping[str, Any]
) -> Dict[str, float]:
    _equal(report.get("schema_version"), Q5_FULL_SCHEMA, "q5_full.schema_version")
    _equal(report.get("all_contract_gates_passed"), True, "q5_full.all_contract_gates_passed")
    gates = report.get("gates")
    _require(isinstance(gates, Mapping) and gates and all(value is True for value in gates.values()), "q5_full gates failed")
    _identity_equal(_path(report, "runtime.base"), base, "q5_full.runtime.base")
    _identity_equal(_path(report, "runtime.lora"), adapter, "q5_full.runtime.lora")
    _equal(_path(report, "runtime.resident_lora_count"), 1, "q5_full.runtime.resident_lora_count")
    _equal(_path(report, "runtime.resident_lora_default_scale"), 1.0, "q5_full.runtime.scale")
    _equal(_path(report, "runtime.request_level_lora_switching"), False, "q5_full.runtime.request_switching")
    _equal(_path(report, "runtime.reconnects"), 0, "q5_full.runtime.reconnects")

    metrics: Dict[str, float] = {}
    for scene, count in (("traffic", 2400), ("industrial", 960)):
        values = _path(report, scene + ".classification")
        _require(isinstance(values, Mapping), "q5_full {} classification missing".format(scene))
        _equal(values.get("count"), count, "q5_full.{}.count".format(scene))
        _equal(values.get("valid_output_rate"), 1.0, "q5_full.{}.valid_output_rate".format(scene))
        _runtime_contract(report, scene)
        records = _path(report, scene + ".records")
        _require(isinstance(records, list) and len(records) == count, "q5_full {} records count invalid".format(scene))
        for index, record in enumerate(records):
            _equal(record.get("prompt_n"), 17, "q5_full.{}.records[{}].prompt_n".format(scene, index))
            _equal(record.get("predicted_n"), 1, "q5_full.{}.records[{}].predicted_n".format(scene, index))
            _equal(record.get("request_has_lora_field"), False, "q5_full.{}.records[{}].lora".format(scene, index))
        accuracy = _number(values.get("accuracy"), "q5_full.{}.accuracy".format(scene))
        macro_f1 = _number(values.get("macro_f1"), "q5_full.{}.macro_f1".format(scene))
        weighted_f1 = _number(values.get("weighted_f1"), "q5_full.{}.weighted_f1".format(scene))
        if scene == "traffic":
            _at_least(accuracy, TRAFFIC_MIN_ACCURACY, "q5_full.traffic.accuracy")
            _at_least(weighted_f1, TRAFFIC_MIN_WEIGHTED_F1, "q5_full.traffic.weighted_f1")
        else:
            _equal(accuracy, 1.0, "q5_full.industrial.accuracy")
            _equal(macro_f1, 1.0, "q5_full.industrial.macro_f1")
            _equal(weighted_f1, 1.0, "q5_full.industrial.weighted_f1")
        metrics["q5_{}_accuracy".format(scene)] = accuracy
        metrics["q5_{}_macro_f1".format(scene)] = macro_f1
        metrics["q5_{}_weighted_f1".format(scene)] = weighted_f1
        metrics["q5_{}_valid_output_rate".format(scene)] = 1.0
    metrics["q5_completed_requests"] = 3360.0
    return metrics


def _validate_nano(
    observation: Mapping[str, Any],
    gate: Mapping[str, Any],
    base: Mapping[str, Any],
    adapter: Mapping[str, Any],
) -> Dict[str, float]:
    _equal(observation.get("schema_version"), NANO_OBSERVATION_SCHEMA, "nano_observation.schema_version")
    _equal(observation.get("status"), "completed", "nano_observation.status")
    _equal(observation.get("completed_requests"), 500, "nano_observation.completed_requests")
    _equal(_path(observation, "stop_reason.code"), "completed", "nano_observation.stop_reason")
    _identity_equal(_path(observation, "runtime.base"), base, "nano_observation.runtime.base")
    _identity_equal(_path(observation, "runtime.adapter"), adapter, "nano_observation.runtime.adapter")
    contract = _path(observation, "benchmark_contract")
    for name, expected in (
        ("per_scene_requested", 250),
        ("total_requested", 500),
        ("required_prompt_tokens", 17),
        ("required_output_tokens", 1),
        ("one_resident_adapter", True),
        ("resident_adapter_default_scale", 1.0),
        ("request_level_lora_switching", False),
        ("requests_omit_lora_fields", True),
    ):
        _equal(contract.get(name), expected, "nano_observation.contract." + name)
    records = observation.get("records")
    _require(isinstance(records, list) and len(records) == 500, "nano_observation records must contain 500 rows")
    scene_counts = {"traffic": 0, "industrial": 0}
    for index, record in enumerate(records):
        expected_scene = "traffic" if index % 2 == 0 else "industrial"
        _equal(record.get("scene"), expected_scene, "nano.records[{}].scene".format(index))
        scene_counts[expected_scene] += 1
        _equal(record.get("prompt_tokens"), 17, "nano.records[{}].prompt_tokens".format(index))
        _equal(record.get("output_tokens"), 1, "nano.records[{}].output_tokens".format(index))
        _equal(record.get("request_has_lora_field"), False, "nano.records[{}].lora".format(index))
        _equal(record.get("valid_output"), True, "nano.records[{}].valid".format(index))
        allowed = "ABCDEF" if expected_scene == "traffic" else "ABC"
        _require(
            record.get("prediction") in allowed,
            "nano.records[{}].prediction outside {} slots".format(index, expected_scene),
        )
    _equal(scene_counts["traffic"], 250, "nano_observation.traffic_count")
    _equal(scene_counts["industrial"], 250, "nano_observation.industrial_count")
    _equal(observation.get("errors"), [], "nano_observation.errors")

    _equal(gate.get("schema_version"), NANO_GATE_SCHEMA, "nano_gate.schema_version")
    _equal(gate.get("gate_pass"), True, "nano_gate.gate_pass")
    checks = gate.get("checks")
    _require(isinstance(checks, Mapping) and checks and all(value is True for value in checks.values()), "nano_gate checks failed")
    _equal(gate.get("completed_requests"), 500, "nano_gate.completed_requests")
    _equal(gate.get("traffic_count"), 250, "nano_gate.traffic_count")
    _equal(gate.get("industrial_count"), 250, "nano_gate.industrial_count")
    strict_peak = _integer(gate.get("strict_peak_bytes"), "nano_gate.strict_peak_bytes")
    if strict_peak > MAX_NANO_STRICT_PEAK_BYTES:
        raise ManifestError(
            "nano_gate.strict_peak_bytes is {} but must be <= {}".format(
                strict_peak, MAX_NANO_STRICT_PEAK_BYTES
            )
        )
    _equal(gate.get("threshold_bytes"), MAX_NANO_STRICT_PEAK_BYTES, "nano_gate.threshold_bytes")
    command = gate.get("command_contract")
    _require(isinstance(command, Mapping), "nano_gate.command_contract missing")
    _equal(command.get("resident_adapter_count"), 1, "nano_gate.resident_adapter_count")
    _equal(command.get("resident_adapter_default_scale"), 1.0, "nano_gate.resident_adapter_default_scale")

    traffic = _path(observation, "summary.traffic")
    industrial = _path(observation, "summary.industrial")
    overall = _path(observation, "summary.overall")
    traffic_accuracy = _at_least(traffic.get("accuracy"), TRAFFIC_MIN_ACCURACY, "nano.traffic.accuracy")
    traffic_wf1 = _at_least(traffic.get("weighted_f1"), TRAFFIC_MIN_WEIGHTED_F1, "nano.traffic.weighted_f1")
    industrial_accuracy = _number(industrial.get("accuracy"), "nano.industrial.accuracy")
    _equal(industrial_accuracy, 1.0, "nano.industrial.accuracy")
    industrial_macro_f1 = _number(industrial.get("macro_f1"), "nano.industrial.macro_f1")
    industrial_weighted_f1 = _number(
        industrial.get("weighted_f1"), "nano.industrial.weighted_f1"
    )
    _equal(industrial_macro_f1, 1.0, "nano.industrial.macro_f1")
    _equal(industrial_weighted_f1, 1.0, "nano.industrial.weighted_f1")
    valid_rate = _number(overall.get("valid_output_rate"), "nano.valid_output_rate")
    _equal(valid_rate, 1.0, "nano.valid_output_rate")
    mean_latency = _at_most(
        _path(overall, "latency_ms.mean"), MAX_MEAN_LATENCY_MS, "nano.overall_mean_latency_ms"
    )
    for name, expected in (
        ("traffic_accuracy", traffic_accuracy),
        ("traffic_weighted_f1", traffic_wf1),
        ("industrial_accuracy", industrial_accuracy),
        ("industrial_macro_f1", industrial_macro_f1),
        ("industrial_weighted_f1", industrial_weighted_f1),
        ("valid_output_rate", valid_rate),
        ("overall_mean_latency_ms", mean_latency),
    ):
        _equal(_number(gate.get(name), "nano_gate." + name), expected, "nano_gate." + name)
    return {
        "nano_completed_requests": 500.0,
        "nano_strict_peak_bytes": float(strict_peak),
        "nano_overall_mean_latency_ms": mean_latency,
        "nano_traffic_accuracy": traffic_accuracy,
        "nano_traffic_weighted_f1": traffic_wf1,
        "nano_industrial_accuracy": industrial_accuracy,
        "nano_industrial_macro_f1": industrial_macro_f1,
        "nano_industrial_weighted_f1": industrial_weighted_f1,
        "nano_valid_output_rate": valid_rate,
    }


def _gate_rows() -> list:
    return [
        {"metric": "formal_total_requests", "operator": "==", "value": 3360.0},
        {"metric": "formal_traffic_accuracy", "operator": ">=", "value": TRAFFIC_MIN_ACCURACY},
        {"metric": "formal_traffic_weighted_f1", "operator": ">=", "value": TRAFFIC_MIN_WEIGHTED_F1},
        {"metric": "formal_traffic_valid_output_rate", "operator": "==", "value": 1.0},
        {"metric": "formal_industrial_accuracy", "operator": "==", "value": 1.0},
        {"metric": "formal_industrial_macro_f1", "operator": "==", "value": 1.0},
        {"metric": "formal_industrial_weighted_f1", "operator": "==", "value": 1.0},
        {"metric": "formal_industrial_valid_output_rate", "operator": "==", "value": 1.0},
        {"metric": "q5_completed_requests", "operator": "==", "value": 3360.0},
        {"metric": "q5_traffic_accuracy", "operator": ">=", "value": TRAFFIC_MIN_ACCURACY},
        {"metric": "q5_traffic_weighted_f1", "operator": ">=", "value": TRAFFIC_MIN_WEIGHTED_F1},
        {"metric": "q5_traffic_valid_output_rate", "operator": "==", "value": 1.0},
        {"metric": "q5_industrial_accuracy", "operator": "==", "value": 1.0},
        {"metric": "q5_industrial_macro_f1", "operator": "==", "value": 1.0},
        {"metric": "q5_industrial_weighted_f1", "operator": "==", "value": 1.0},
        {"metric": "q5_industrial_valid_output_rate", "operator": "==", "value": 1.0},
        {"metric": "nano_completed_requests", "operator": "==", "value": 500.0},
        {"metric": "nano_strict_peak_bytes", "operator": "<=", "value": float(MAX_NANO_STRICT_PEAK_BYTES)},
        {"metric": "nano_overall_mean_latency_ms", "operator": "<", "value": MAX_MEAN_LATENCY_MS},
        {"metric": "nano_traffic_accuracy", "operator": ">=", "value": TRAFFIC_MIN_ACCURACY},
        {"metric": "nano_traffic_weighted_f1", "operator": ">=", "value": TRAFFIC_MIN_WEIGHTED_F1},
        {"metric": "nano_industrial_accuracy", "operator": "==", "value": 1.0},
        {"metric": "nano_industrial_macro_f1", "operator": "==", "value": 1.0},
        {"metric": "nano_industrial_weighted_f1", "operator": "==", "value": 1.0},
        {"metric": "nano_valid_output_rate", "operator": "==", "value": 1.0},
    ]


def build_q5_joint_release(descriptor_path: Path, output_dir: Path) -> Dict[str, Any]:
    descriptor_file = descriptor_path.resolve()
    root = descriptor_file.parent
    descriptor = read_json_object(descriptor_file)
    _equal(descriptor.get("schema_version"), SCHEMA, "descriptor.schema_version")
    output = output_dir.resolve()
    if output.exists():
        raise ManifestError("immutable Q5 joint package output already exists: {}".format(output))

    base_path, base = _declared_identity(root, descriptor.get("deployment_artifact"), "deployment_artifact")
    runtime_raw = descriptor.get("runtime_adapter")
    if not isinstance(runtime_raw, Mapping):
        raise ManifestError("runtime_adapter must be an object")
    _equal(runtime_raw.get("id"), 0, "runtime_adapter.id")
    _equal(runtime_raw.get("mode"), "process_default", "runtime_adapter.mode")
    _equal(runtime_raw.get("default_scale"), 1, "runtime_adapter.default_scale")
    adapter_path, runtime_adapter = _declared_identity(root, runtime_raw, "runtime_adapter")
    runtime_adapter.update({"id": 0, "mode": "process_default", "default_scale": 1})

    adapter_source = _resolve(root, descriptor.get("primary_adapter_source"), "primary_adapter_source")
    adapter_artifacts = {
        "weights": _actual_identity(adapter_source / "adapter_model.safetensors", "adapter weights"),
        "config": _actual_identity(adapter_source / "adapter_config.json", "adapter config"),
        "train_metrics": _actual_identity(adapter_source / "train_metrics.json", "adapter train metrics"),
    }
    evidence_paths = _evidence_files(root, descriptor.get("evidence"))
    evidence = {name: read_json_object(path) for name, path in evidence_paths.items()}
    metrics = {}
    metrics.update(
        _validate_formal(
            evidence["formal_traffic"],
            evidence["formal_industrial"],
            evidence["formal_gate"],
            adapter_artifacts,
        )
    )
    metrics.update(_validate_q5_full(evidence["q5_full"], base, runtime_adapter))
    metrics.update(
        _validate_nano(
            evidence["nano_observation"], evidence["nano_gate"], base, runtime_adapter
        )
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "descriptor_sha256": sha256_file(descriptor_file),
        "assets": {"clean_q5_k_m": base, "joint_f16_lora": runtime_adapter},
        "adapter_artifacts": adapter_artifacts,
        "bf16_formal_bindings": {
            name: evidence["formal_gate"][name]
            for name in (
                "base_manifest_identity",
                "snapshot_manifest_identity",
                "dataset_manifest_identity",
                "dataset_artifacts",
                "evaluator_identity",
            )
        },
        "runtime_contract": {
            "adapter_count": 1,
            "adapter_mode": "process_default",
            "adapter_id": 0,
            "adapter_default_scale": 1,
            "request_level_lora_switching": False,
            "traffic_prefix": "T",
            "traffic_allowed_slots": list("ABCDEF"),
            "industrial_prefix": "I",
            "industrial_allowed_slots": list("ABC"),
            "input_tokens": 17,
            "output_tokens": 1,
        },
        "source_evidence": {
            name: _actual_identity(path, "evidence." + name)
            for name, path in evidence_paths.items()
        },
        "metrics": metrics,
        "fixed_gates": _gate_rows(),
        "release_authorized_by_packager": True,
    }

    base_manifest = _resolve(root, descriptor.get("base_manifest"), "base_manifest")
    action_mapping = _resolve(root, descriptor.get("action_mapping"), "action_mapping")
    adapter_id = descriptor.get("adapter_id")
    version = descriptor.get("version")
    _require(isinstance(adapter_id, str) and bool(adapter_id), "adapter_id must be non-empty")
    _require(isinstance(version, str) and bool(version), "version must be non-empty")
    training = descriptor.get("training")
    _require(isinstance(training, Mapping), "training must be an object")

    with tempfile.TemporaryDirectory(prefix="q5-joint-package-") as temporary:
        temporary_root = Path(temporary)
        summary_path = temporary_root / "q5_joint_gate_summary.json"
        write_json_object(summary_path, summary)
        sources = {
            name: {"evidence": "q5_joint_gate_summary", "path": "metrics." + name}
            for name in metrics
        }
        spec = {
            "schema_version": "edge-llm-package-spec/v1",
            "adapter_id": adapter_id,
            # Keep the existing package-schema compatibility scene.  The
            # sanitized joint summary, not a Q6 schema, carries both scenes.
            "scene": TRAFFIC_SCENE,
            "version": version,
            "adapter_source": str(adapter_source),
            "action_mapping": str(action_mapping),
            "input_contract": {
                "event_type": "com.cloudedge.traffic.edge-event.v1",
                "data_schema": "https://cloud-edge.local/schemas/scenes/traffic-edge-event-v1.json",
                "llm_input_type": "compact_text_code",
                "context_encoder": "joint-scene-prefix-decimal17@v1",
                "max_input_tokens": 17,
                "direct_media_to_llm": False,
            },
            "training": dict(training),
            "evaluation": {
                "evidence": {"q5_joint_gate_summary": str(summary_path)},
                "metric_sources": sources,
                "gates": _gate_rows(),
            },
            "deployment": {
                "artifact": str(base_path),
                "runtime": "llama.cpp",
                "format": "gguf",
                "quantization": "Q5_K_M",
                "max_input_tokens": 17,
                "max_output_tokens": 1,
                "thinking": False,
            },
        }
        spec_path = temporary_root / "package_spec.json"
        write_json_object(spec_path, spec)
        result = build_adapter_package(
            project_root=root,
            base_manifest_path=base_manifest,
            spec_path=spec_path,
            output_dir=output,
        )
    result["runtime_adapters"] = [runtime_adapter]
    result["q5_joint_gate_summary_sha256"] = sha256_file(
        output / "evidence" / "q5_joint_gate_summary.json"
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed clean-Q5 plus one joint-LoRA package"
    )
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_q5_joint_release(Path(args.descriptor), Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
