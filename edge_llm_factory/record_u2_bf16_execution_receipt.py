"""Record the final U2 BF16 decision from already-produced evidence.

This is a recorder and verifier, not a runner.  It never imports a model,
executes an evaluation, or opens a source test/blind dataset.  Observed process
exit codes are accepted only from one explicitly supplied, SHA-256 anchored
step-receipt file.  In particular, exit code 2 is a valid gate failure only
when a complete, hash-bound gate report independently says that the gate
failed; exit code 2 by itself is never evidence of a failed metric gate.
The candidate receipt's own U2 protocol anchor is reopened and verified, then
all eight observed commands and working directories must exactly match its
``exact_execution.steps`` after resolving only the two declared SHA placeholders.
Every implementation and every non-test SHA-bound frozen input in that protocol
is then re-hashed in place.  The frozen traffic regression set is deliberately
not opened by this recorder: its protocol identity is instead cross-bound to the
traffic gate's independently recomputed frozen-test identity and immutable gate
implementation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from edge_llm_factory.contracts import HEX64, ManifestError, canonical_sha256


SCHEMA_VERSION = "edge-llm-u2-bf16-execution-receipt/v1"
STEP_RECEIPTS_SCHEMA = "edge-llm-u2-observed-step-receipts/v1"
CANDIDATE_RECEIPT_SCHEMA = "edge-llm-u2-candidate-artifact-receipt/v1"
TRAFFIC_GATE_SCHEMA = "edge-unified-traffic-bf16-paired-regression/v1"
GENERAL_CANDIDATE_SCHEMA = "edge-llm-unified-general-dev-evaluation/v1"
GENERAL_TEACHER_SCHEMA = "edge-llm-unified-general-teacher-dev-evaluation/v1"
GENERAL_GATE_SCHEMA = "edge-llm-unified-general-paired-gate/v1"
U2_PROTOCOL_SCHEMA = "edge-unified-u2-bf16-protocol/v2"
CANDIDATE_ID = "unified-v2-U2"

REQUIRED_CANDIDATE_FILES = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "train_metrics.json",
}
TRAFFIC_CHECKS = {
    "candidate_accuracy_absolute_minimum",
    "candidate_accuracy_not_below_incumbent",
    "candidate_weighted_f1_absolute_minimum",
    "candidate_weighted_f1_not_below_incumbent",
    "candidate_valid_output_rate",
}
GENERAL_CATEGORY_CHECKS = {
    "retention_at_least_0p80",
    "candidate_valid_rate_equals_1",
}
REQUIRED_STEP_OUTPUTS = {
    "train_U2_once": {
        "adapter_model",
        "adapter_config",
        "train_metrics",
    },
    "create_candidate_artifact_receipt_without_scoring": {"candidate_receipt"},
    "traffic_incumbent_bf16": {"traffic_incumbent_evaluation"},
    "traffic_candidate_bf16": {"traffic_candidate_evaluation"},
    "traffic_paired_gate": {"traffic_gate"},
    "general_teacher_fixed_800": {
        "general_teacher_samples",
        "general_teacher_summary",
    },
    "general_candidate_bf16_fixed_800": {
        "general_candidate_samples",
        "general_candidate_summary",
    },
    "general_paired_gate": {"general_gate"},
}
NON_GATE_STEPS = set(REQUIRED_STEP_OUTPUTS) - {
    "traffic_paired_gate",
    "general_paired_gate",
}
REQUIRED_IMPLEMENTATIONS = {
    "dataset_builder",
    "trainer",
    "candidate_artifact_recorder",
    "traffic_evaluator",
    "traffic_gate",
    "general_candidate_evaluator",
    "general_teacher_evaluator",
    "general_paired_gate",
    "action_constraint",
    "execution_receipt_recorder",
}
REQUIRED_SHA_BOUND_FROZEN_INPUTS = {
    "base_manifest",
    "text_snapshot_manifest",
    "u1_adapter_weights",
    "u1_adapter_config",
    "dataset_manifest",
    "training_jsonl",
    "training_validation_jsonl",
    "general_promotion_jsonl",
    "traffic_regression_jsonl",
    "production_traffic_adapter_weights",
    "production_traffic_adapter_config",
}
TRAFFIC_TEST_FROZEN_INPUT = "traffic_regression_jsonl"


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, field: str) -> str:
    text = _text(value, field).lower()
    if HEX64.fullmatch(text) is None:
        raise ManifestError(f"{field} must be a lowercase SHA-256")
    return text


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError(f"{field} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{field} must be boolean")
    return value


def _utc_timestamp(value: Any, field: str) -> str:
    text = _text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError(f"{field} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ManifestError(f"{field} must include UTC timezone information")
    if parsed.utcoffset().total_seconds() != 0:
        raise ManifestError(f"{field} must be UTC")
    return text


def _regular_file(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ManifestError(f"{label} must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ManifestError(f"{label} is not a regular file: {resolved}")
    return resolved


def _regular_directory(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ManifestError(f"{label} must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_dir():
        raise ManifestError(f"{label} is not a directory: {resolved}")
    return resolved


def _read_bytes(path: Path, label: str) -> tuple[bytes, Dict[str, Any]]:
    resolved = _regular_file(path, label)
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise ManifestError(f"cannot read {label}: {resolved}: {exc}") from exc
    return payload, {
        "path": str(resolved),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _stream_file_identity(path: Path, label: str) -> Dict[str, Any]:
    """Return a file identity without retaining the file body in memory."""

    resolved = _regular_file(path, label)
    digest = hashlib.sha256()
    total = 0
    try:
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                total += len(block)
                digest.update(block)
    except OSError as exc:
        raise ManifestError(f"cannot read {label}: {resolved}: {exc}") from exc
    return {
        "path": str(resolved),
        "bytes": total,
        "sha256": digest.hexdigest(),
    }


def _read_json(path: Path, label: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    payload, identity = _read_bytes(path, label)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{label} must be a UTF-8 JSON object") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object")
    return value, identity


def _declared_identity(value: Any, field: str) -> Dict[str, Any]:
    raw = _mapping(value, field)
    path = Path(_text(raw.get("path"), f"{field}.path")).expanduser().resolve()
    return {
        "path": str(path),
        "bytes": _integer(raw.get("bytes"), f"{field}.bytes"),
        "sha256": _sha256(raw.get("sha256"), f"{field}.sha256"),
    }


def _same_identity(actual: Mapping[str, Any], declared: Any, field: str) -> None:
    expected = _declared_identity(declared, field)
    if dict(actual) != expected:
        raise ManifestError(f"{field} path/bytes/SHA-256 does not match the file")


def _declared_path_sha(value: Any, field: str) -> Dict[str, Any]:
    """Parse a path/SHA identity whose byte count may be omitted by its schema."""

    raw = _mapping(value, field)
    result = {
        "path": str(
            Path(_text(raw.get("path"), f"{field}.path")).expanduser().resolve()
        ),
        "sha256": _sha256(raw.get("sha256"), f"{field}.sha256"),
    }
    if "bytes" in raw:
        result["bytes"] = _integer(raw.get("bytes"), f"{field}.bytes")
    return result


def _same_path_sha(actual: Mapping[str, Any], declared: Any, field: str) -> None:
    """Compare path/SHA and, when declared, the byte count."""

    expected = _declared_path_sha(declared, field)
    if actual.get("path") != expected["path"]:
        raise ManifestError(f"{field} path does not match the anchored implementation")
    if actual.get("sha256") != expected["sha256"]:
        raise ManifestError(f"{field} SHA-256 does not match the anchored implementation")
    if "bytes" in expected and actual.get("bytes") != expected["bytes"]:
        raise ManifestError(f"{field} bytes do not match the anchored implementation")


def _validate_protocol_file_bindings(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """Re-hash protocol implementations and every non-test frozen input.

    The traffic regression JSONL is the only deliberate exception.  Opening it
    here would make this recorder another source-test consumer.  Its identity is
    retained for later validation against the traffic gate report, which already
    read and re-hashed the frozen set under a SHA-bound gate implementation.
    """

    implementations = _mapping(
        protocol.get("implementations"), "u2_protocol.implementations"
    )
    if set(implementations) != REQUIRED_IMPLEMENTATIONS:
        raise ManifestError("U2 protocol implementation set mismatch")
    verified_implementations: Dict[str, Dict[str, Any]] = {}
    for name in sorted(implementations):
        field = f"u2_protocol.implementations.{name}"
        declared = _declared_path_sha(implementations[name], field)
        actual = _stream_file_identity(Path(declared["path"]), field)
        _same_path_sha(actual, implementations[name], field)
        verified_implementations[name] = {
            **actual,
            "verified_against_protocol": True,
        }

    frozen_inputs = _mapping(
        protocol.get("frozen_inputs"), "u2_protocol.frozen_inputs"
    )
    sha_bound_names = {
        name
        for name, raw in frozen_inputs.items()
        if isinstance(raw, Mapping) and "sha256" in raw
    }
    if sha_bound_names != REQUIRED_SHA_BOUND_FROZEN_INPUTS:
        raise ManifestError("U2 protocol SHA-bound frozen input set mismatch")
    verified_frozen: Dict[str, Dict[str, Any]] = {}
    for name in sorted(sha_bound_names):
        field = f"u2_protocol.frozen_inputs.{name}"
        declared = _declared_path_sha(frozen_inputs[name], field)
        if name == TRAFFIC_TEST_FROZEN_INPUT:
            verified_frozen[name] = {
                **declared,
                "verification": "deferred_to_traffic_gate_without_opening_test_content",
            }
            continue
        actual = _stream_file_identity(Path(declared["path"]), field)
        _same_path_sha(actual, frozen_inputs[name], field)
        verified_frozen[name] = {
            **actual,
            "verified_against_protocol": True,
        }
    snapshot_raw = _mapping(
        frozen_inputs.get("text_snapshot"), "u2_protocol.frozen_inputs.text_snapshot"
    )
    snapshot_directory = _regular_directory(
        Path(_text(snapshot_raw.get("path"), "u2_protocol.frozen_inputs.text_snapshot.path")),
        "u2_protocol.frozen_inputs.text_snapshot",
    )
    snapshot_manifest_path = Path(
        verified_frozen["text_snapshot_manifest"]["path"]
    )
    snapshot_manifest, snapshot_manifest_identity = _read_json(
        snapshot_manifest_path, "anchored text snapshot manifest"
    )
    _same_path_sha(
        snapshot_manifest_identity,
        frozen_inputs["text_snapshot_manifest"],
        "u2_protocol.frozen_inputs.text_snapshot_manifest",
    )
    raw_snapshot_files = snapshot_manifest.get("files")
    if not isinstance(raw_snapshot_files, list) or not raw_snapshot_files:
        raise ManifestError("anchored text snapshot manifest files must be non-empty")
    snapshot_files: Dict[str, Dict[str, Any]] = {}
    for index, raw in enumerate(raw_snapshot_files):
        field = f"text_snapshot_manifest.files[{index}]"
        row = _mapping(raw, field)
        relative_text = _text(row.get("path"), f"{field}.path")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ManifestError(f"{field}.path escapes the snapshot directory")
        normalized = relative.as_posix()
        if normalized in snapshot_files:
            raise ManifestError("anchored text snapshot manifest contains duplicate paths")
        actual = _stream_file_identity(snapshot_directory / relative, field)
        expected_bytes = _integer(row.get("bytes"), f"{field}.bytes", minimum=1)
        expected_sha = _sha256(row.get("sha256"), f"{field}.sha256")
        if actual["bytes"] != expected_bytes or actual["sha256"] != expected_sha:
            raise ManifestError(f"{field} current file differs from the anchored manifest")
        snapshot_files[normalized] = actual
    snapshot_id = _text(snapshot_manifest.get("snapshot_id"), "text_snapshot_manifest.snapshot_id")
    return {
        "implementations": verified_implementations,
        "frozen_inputs": verified_frozen,
        "text_snapshot": {
            "path": str(snapshot_directory),
            "manifest": snapshot_manifest_identity,
            "snapshot_id": snapshot_id,
            "files": snapshot_files,
            "all_manifest_files_rehashed": True,
        },
    }


def _validate_candidate_receipt(
    receipt: Mapping[str, Any], receipt_identity: Mapping[str, Any]
) -> Dict[str, Any]:
    if receipt.get("schema_version") != CANDIDATE_RECEIPT_SCHEMA:
        raise ManifestError("candidate receipt schema_version mismatch")
    if receipt.get("candidate_id") != CANDIDATE_ID:
        raise ManifestError("candidate receipt candidate_id mismatch")
    candidate = _mapping(receipt.get("candidate"), "candidate_receipt.candidate")
    candidate_dir = Path(
        _text(candidate.get("path"), "candidate_receipt.candidate.path")
    ).expanduser().resolve()
    files = _mapping(candidate.get("files"), "candidate_receipt.candidate.files")
    if set(files) != REQUIRED_CANDIDATE_FILES:
        raise ManifestError("candidate receipt must bind exactly the three U2 files")
    actual_files: Dict[str, Dict[str, Any]] = {}
    for name in sorted(REQUIRED_CANDIDATE_FILES):
        _, actual = _read_bytes(candidate_dir / name, f"candidate file {name}")
        _same_identity(actual, files.get(name), f"candidate_receipt.candidate.files.{name}")
        actual_files[name] = actual
    canonical = canonical_sha256(
        {
            name: {"bytes": row["bytes"], "sha256": row["sha256"]}
            for name, row in actual_files.items()
        }
    )
    if candidate.get("canonical_artifact_sha256") != canonical:
        raise ManifestError("candidate canonical_artifact_sha256 mismatch")
    evidence = _mapping(
        receipt.get("training_evidence"), "candidate_receipt.training_evidence"
    )
    completion = _mapping(evidence.get("completion"), "training_evidence.completion")
    if completion.get("confirmed") is not True:
        raise ManifestError("candidate receipt does not confirm completed training")
    return {
        "receipt": dict(receipt_identity),
        "candidate_path": str(candidate_dir),
        "files": actual_files,
        "canonical_artifact_sha256": canonical,
        "training_completion_confirmed": True,
    }


def _load_anchored_protocol(
    *,
    candidate_receipt: Mapping[str, Any],
    candidate_receipt_identity: Mapping[str, Any],
    candidate_model_identity: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any], list[Dict[str, Any]]]:
    anchors = _mapping(candidate_receipt.get("anchors"), "candidate_receipt.anchors")
    declared = _mapping(
        anchors.get("u2_protocol"), "candidate_receipt.anchors.u2_protocol"
    )
    protocol_path = Path(
        _text(declared.get("path"), "candidate_receipt.anchors.u2_protocol.path")
    )
    protocol, protocol_identity = _read_json(protocol_path, "anchored U2 protocol")
    _same_identity(
        protocol_identity, declared, "candidate_receipt.anchors.u2_protocol"
    )
    if declared.get("expected_sha256") != protocol_identity["sha256"]:
        raise ManifestError("candidate receipt protocol expected_sha256 mismatch")
    if declared.get("verified") is not True:
        raise ManifestError("candidate receipt does not mark the U2 protocol verified")
    if protocol.get("schema_version") != U2_PROTOCOL_SCHEMA:
        raise ManifestError("anchored U2 protocol schema_version mismatch")
    if protocol.get("candidate_id") != CANDIDATE_ID:
        raise ManifestError("anchored U2 protocol candidate_id mismatch")

    exact = _mapping(protocol.get("exact_execution"), "u2_protocol.exact_execution")
    bindings = _mapping(
        exact.get("dynamic_argument_bindings"),
        "u2_protocol.exact_execution.dynamic_argument_bindings",
    )
    expected_binding_names = {"U2_PROTOCOL_SHA256", "CANDIDATE_ADAPTER_SHA256"}
    if set(bindings) != expected_binding_names:
        raise ManifestError("U2 protocol dynamic argument binding set mismatch")
    protocol_binding = _mapping(
        bindings["U2_PROTOCOL_SHA256"],
        "u2_protocol.dynamic_argument_bindings.U2_PROTOCOL_SHA256",
    )
    if Path(
        _text(protocol_binding.get("source_file"), "U2_PROTOCOL_SHA256.source_file")
    ).expanduser().resolve() != Path(protocol_identity["path"]):
        raise ManifestError("U2_PROTOCOL_SHA256 source_file is not the anchored protocol")
    if protocol_binding.get("literal_placeholder_must_not_be_executed") is not True:
        raise ManifestError("U2_PROTOCOL_SHA256 placeholder policy is not strict")
    candidate_binding = _mapping(
        bindings["CANDIDATE_ADAPTER_SHA256"],
        "u2_protocol.dynamic_argument_bindings.CANDIDATE_ADAPTER_SHA256",
    )
    if Path(
        _text(candidate_binding.get("receipt_path"), "CANDIDATE_ADAPTER_SHA256.receipt_path")
    ).expanduser().resolve() != Path(candidate_receipt_identity["path"]):
        raise ManifestError("CANDIDATE_ADAPTER_SHA256 receipt_path mismatch")
    if candidate_binding.get("json_pointer") != (
        "/candidate/files/adapter_model.safetensors/sha256"
    ):
        raise ManifestError("CANDIDATE_ADAPTER_SHA256 json_pointer mismatch")
    if Path(
        _text(
            candidate_binding.get("must_match_file"),
            "CANDIDATE_ADAPTER_SHA256.must_match_file",
        )
    ).expanduser().resolve() != Path(candidate_model_identity["path"]):
        raise ManifestError("CANDIDATE_ADAPTER_SHA256 must_match_file mismatch")
    if candidate_binding.get("literal_placeholder_must_not_be_executed") is not True:
        raise ManifestError("CANDIDATE_ADAPTER_SHA256 placeholder policy is not strict")

    raw_steps = exact.get("steps")
    if not isinstance(raw_steps, list) or len(raw_steps) != 8:
        raise ManifestError("U2 protocol exact_execution.steps must contain exactly 8 steps")
    dynamic_values = {
        "U2_PROTOCOL_SHA256": protocol_identity["sha256"],
        "CANDIDATE_ADAPTER_SHA256": candidate_model_identity["sha256"],
    }
    resolved_steps: list[Dict[str, Any]] = []
    placeholder_counts = {name: 0 for name in dynamic_values}
    for index, raw in enumerate(raw_steps, start=1):
        step = _mapping(raw, f"u2_protocol.exact_execution.steps[{index - 1}]")
        if step.get("step") != index:
            raise ManifestError("U2 protocol step numbering must be exactly 1 through 8")
        name = _text(step.get("name"), f"u2_protocol.steps[{index}].name")
        argv = step.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) or not item for item in argv)
        ):
            raise ManifestError(f"U2 protocol step {index} argv is invalid")
        resolved_argv = []
        for token in argv:
            if token in dynamic_values:
                placeholder_counts[token] += 1
                resolved_argv.append(dynamic_values[token])
            else:
                if any(placeholder in token for placeholder in dynamic_values):
                    raise ManifestError(
                        f"U2 protocol step {index} embeds a dynamic placeholder inside an argv token"
                    )
                resolved_argv.append(token)
        resolved_steps.append(
            {
                "step": index,
                "name": name,
                "working_directory": str(
                    Path(
                        _text(
                            step.get("working_directory"),
                            f"u2_protocol.steps[{index}].working_directory",
                        )
                    ).expanduser().resolve()
                ),
                "command_argv": resolved_argv,
            }
        )
    if {row["name"] for row in resolved_steps} != set(REQUIRED_STEP_OUTPUTS):
        raise ManifestError("U2 protocol step names do not match the fixed U2 workflow")
    if any(count <= 0 for count in placeholder_counts.values()):
        raise ManifestError("every U2 protocol dynamic placeholder must be used")
    return protocol, protocol_identity, resolved_steps


def _validate_traffic_artifacts(
    artifacts: Mapping[str, Any],
    *,
    label: str,
    expected_adapter_model: Mapping[str, Any],
    expected_adapter_config: Mapping[str, Any],
    snapshot_manifest_identity: Mapping[str, Any],
    text_snapshot_binding: Mapping[str, Any],
    traffic_evaluator_identity: Mapping[str, Any],
) -> None:
    if artifacts.get("verified_unchanged_during_evaluation") is not True:
        raise ManifestError(f"{label} does not attest stable evaluation artifacts")
    _same_path_sha(
        expected_adapter_model,
        artifacts.get("adapter_model"),
        f"{label}.adapter_model",
    )
    _same_path_sha(
        expected_adapter_config,
        artifacts.get("adapter_config"),
        f"{label}.adapter_config",
    )
    _same_path_sha(
        snapshot_manifest_identity,
        artifacts.get("base_text_snapshot_manifest"),
        f"{label}.base_text_snapshot_manifest",
    )
    _same_path_sha(
        traffic_evaluator_identity,
        artifacts.get("evaluator"),
        f"{label}.evaluator",
    )
    declared_snapshot_files = _mapping(
        artifacts.get("base_text_snapshot_files"),
        f"{label}.base_text_snapshot_files",
    )
    expected_snapshot_files = _mapping(
        text_snapshot_binding.get("files"), "anchored text snapshot files"
    )
    if set(declared_snapshot_files) != set(expected_snapshot_files):
        raise ManifestError(f"{label}.base_text_snapshot_files set mismatch")
    for name, actual in expected_snapshot_files.items():
        _same_path_sha(
            _mapping(actual, f"anchored text snapshot files.{name}"),
            declared_snapshot_files[name],
            f"{label}.base_text_snapshot_files.{name}",
        )


def _validate_traffic_evaluation(
    value: Mapping[str, Any],
    label: str,
    *,
    expected_adapter_model: Mapping[str, Any],
    expected_adapter_config: Mapping[str, Any],
    snapshot_manifest_identity: Mapping[str, Any],
    text_snapshot_binding: Mapping[str, Any],
    traffic_evaluator_identity: Mapping[str, Any],
) -> None:
    if value.get("task") != "phase1_qwen_sft_generation_eval":
        raise ManifestError(f"{label} task/schema marker mismatch")
    expected = {
        "bf16": True,
        "max_seq_length": 16,
        "max_new_tokens": 1,
        "temperature": 0.0,
        "count": 2400,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise ManifestError(f"{label}.{field} must equal {wanted!r}")
    artifacts = _mapping(value.get("evaluation_artifacts"), f"{label}.evaluation_artifacts")
    _validate_traffic_artifacts(
        artifacts,
        label=f"{label}.evaluation_artifacts",
        expected_adapter_model=expected_adapter_model,
        expected_adapter_config=expected_adapter_config,
        snapshot_manifest_identity=snapshot_manifest_identity,
        text_snapshot_binding=text_snapshot_binding,
        traffic_evaluator_identity=traffic_evaluator_identity,
    )


def _validate_traffic_candidate_model(
    value: Mapping[str, Any], candidate_model_identity: Mapping[str, Any]
) -> None:
    artifacts = _mapping(
        value.get("evaluation_artifacts"),
        "traffic_candidate_evaluation.evaluation_artifacts",
    )
    _same_identity(
        candidate_model_identity,
        artifacts.get("adapter_model"),
        "traffic_candidate_evaluation.evaluation_artifacts.adapter_model",
    )


def _validate_general_summary(
    value: Mapping[str, Any], *, label: str, expected_schema: str
) -> None:
    if value.get("schema_version") != expected_schema:
        raise ManifestError(f"{label} schema_version mismatch")
    if value.get("development_only") is not True:
        raise ManifestError(f"{label} must be development_only")
    if value.get("formal_blind_test_used") is not False:
        raise ManifestError(f"{label} must declare formal_blind_test_used=false")
    dataset = _mapping(value.get("dataset"), f"{label}.dataset")
    if dataset.get("blind_test_content_loaded") is not False:
        raise ManifestError(f"{label} does not prove blind content isolation")


def _validate_sample_binding(
    summary: Mapping[str, Any], samples_identity: Mapping[str, Any], label: str
) -> None:
    artifacts = _mapping(summary.get("artifacts"), f"{label}.artifacts")
    _same_identity(samples_identity, artifacts.get("samples"), f"{label}.artifacts.samples")


def _validate_general_candidate_model_binding(
    summary: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    base_manifest_identity: Mapping[str, Any],
    snapshot_manifest_identity: Mapping[str, Any],
    text_snapshot_binding: Mapping[str, Any],
    candidate_evaluator_identity: Mapping[str, Any],
) -> str:
    adapter = _mapping(summary.get("adapter"), "general_candidate_summary.adapter")
    if Path(
        _text(adapter.get("path"), "general_candidate_summary.adapter.path")
    ).expanduser().resolve() != Path(candidate["candidate_path"]):
        raise ManifestError("general candidate adapter path differs from candidate receipt")
    files = _mapping(adapter.get("files"), "general_candidate_summary.adapter.files")
    file_bindings = {
        "adapter_weights": candidate["files"]["adapter_model.safetensors"],
        "adapter_config": candidate["files"]["adapter_config.json"],
        "train_metrics": candidate["files"]["train_metrics.json"],
    }
    if set(files) != set(file_bindings):
        raise ManifestError("general candidate adapter file set mismatch")
    for name, actual in file_bindings.items():
        _same_path_sha(
            actual,
            files[name],
            f"general_candidate_summary.adapter.files.{name}",
        )
    expected_artifact_sha = canonical_sha256(
        {
            name: {
                "bytes": actual["bytes"],
                "sha256": actual["sha256"],
            }
            for name, actual in sorted(file_bindings.items())
        }
    )
    if _sha256(
        adapter.get("artifact_sha256"),
        "general_candidate_summary.adapter.artifact_sha256",
    ) != expected_artifact_sha:
        raise ManifestError("general candidate adapter artifact SHA-256 mismatch")
    weights_binding = _mapping(
        adapter.get("weights_binding"),
        "general_candidate_summary.adapter.weights_binding",
    )
    candidate_weights_sha = candidate["files"]["adapter_model.safetensors"]["sha256"]
    for field in ("expected_sha256", "actual_sha256"):
        if _sha256(
            weights_binding.get(field),
            f"general_candidate_summary.adapter.weights_binding.{field}",
        ) != candidate_weights_sha:
            raise ManifestError("general candidate weights binding differs from receipt")
    if weights_binding.get("externally_anchored_and_recomputed") is not True:
        raise ManifestError("general candidate weights were not externally anchored")

    model = _mapping(
        summary.get("model_identity"), "general_candidate_summary.model_identity"
    )
    base_sha = _sha256(
        base_manifest_identity.get("sha256"),
        "u2_protocol.frozen_inputs.base_manifest.sha256",
    )
    snapshot_sha = _sha256(
        snapshot_manifest_identity.get("sha256"),
        "u2_protocol.frozen_inputs.text_snapshot_manifest.sha256",
    )
    if _sha256(
        model.get("base_manifest_sha256"),
        "general_candidate_summary.model_identity.base_manifest_sha256",
    ) != base_sha:
        raise ManifestError("general candidate base manifest differs from protocol")
    if _sha256(
        model.get("snapshot_manifest_sha256"),
        "general_candidate_summary.model_identity.snapshot_manifest_sha256",
    ) != snapshot_sha:
        raise ManifestError("general candidate snapshot manifest differs from protocol")
    snapshot_id = _text(
        text_snapshot_binding.get("snapshot_id"), "anchored text snapshot id"
    )
    if model.get("snapshot_id") != snapshot_id:
        raise ManifestError("general candidate snapshot_id differs from protocol")
    if model.get("adapter_artifact_sha256") != expected_artifact_sha:
        raise ManifestError("general candidate model identity does not bind its adapter")
    model_payload = {
        "base_manifest_sha256": base_sha,
        "snapshot_manifest_sha256": snapshot_sha,
        "snapshot_id": snapshot_id,
        "adapter_artifact_sha256": expected_artifact_sha,
    }
    model_sha = canonical_sha256(model_payload)
    if _sha256(
        model.get("model_sha256"),
        "general_candidate_summary.model_identity.model_sha256",
    ) != model_sha:
        raise ManifestError("general candidate model SHA-256 is inconsistent")

    snapshot_validation = _mapping(
        summary.get("snapshot_validation"),
        "general_candidate_summary.snapshot_validation",
    )
    if snapshot_validation.get("status") != "valid":
        raise ManifestError("general candidate snapshot validation is not valid")
    if snapshot_validation.get("snapshot_id") != snapshot_id:
        raise ManifestError("general candidate snapshot validation id mismatch")
    if Path(
        _text(
            snapshot_validation.get("snapshot"),
            "general_candidate_summary.snapshot_validation.snapshot",
        )
    ).expanduser().resolve() != Path(text_snapshot_binding["path"]):
        raise ManifestError("general candidate snapshot path differs from protocol")
    checked_files = snapshot_validation.get("checked_files")
    if not isinstance(checked_files, list) or set(checked_files) != set(
        _mapping(text_snapshot_binding.get("files"), "anchored snapshot files")
    ):
        raise ManifestError("general candidate checked snapshot file set mismatch")
    artifacts = _mapping(
        summary.get("artifacts"), "general_candidate_summary.artifacts"
    )
    _same_path_sha(
        candidate_evaluator_identity,
        artifacts.get("evaluator"),
        "general_candidate_summary.artifacts.evaluator",
    )
    return model_sha


def _validate_general_teacher_model_binding(
    summary: Mapping[str, Any],
    *,
    teacher_contract: Mapping[str, Any],
    teacher_evaluator_identity: Mapping[str, Any],
) -> str:
    expected_digest = _sha256(
        teacher_contract.get("expected_model_digest_sha256"),
        "u2_protocol.teacher.expected_model_digest_sha256",
    )
    expected_provider = _text(
        teacher_contract.get("provider"), "u2_protocol.teacher.provider"
    )
    expected_model = _text(
        teacher_contract.get("model"), "u2_protocol.teacher.model"
    )
    model = _mapping(
        summary.get("model_identity"), "general_teacher_summary.model_identity"
    )
    if model.get("stable_across_evaluation") is not True:
        raise ManifestError("general teacher model was not stable across evaluation")
    for field in ("expected_model_sha256", "model_sha256"):
        if _sha256(
            model.get(field), f"general_teacher_summary.model_identity.{field}"
        ) != expected_digest:
            raise ManifestError("general teacher digest differs from protocol")
    if model.get("provider") != expected_provider or model.get("model") != expected_model:
        raise ManifestError("general teacher provider/model differs from protocol")
    for phase in ("attestation_before", "attestation_after"):
        attestation = _mapping(
            model.get(phase), f"general_teacher_summary.model_identity.{phase}"
        )
        if (
            attestation.get("provider") != expected_provider
            or attestation.get("model") != expected_model
        ):
            raise ManifestError(f"general teacher {phase} provider/model mismatch")
        for field in ("expected_model_sha256", "model_sha256"):
            if _sha256(
                attestation.get(field),
                f"general_teacher_summary.model_identity.{phase}.{field}",
            ) != expected_digest:
                raise ManifestError(f"general teacher {phase} digest mismatch")
    artifacts = _mapping(
        summary.get("artifacts"), "general_teacher_summary.artifacts"
    )
    _same_path_sha(
        teacher_evaluator_identity,
        artifacts.get("evaluator"),
        "general_teacher_summary.artifacts.evaluator",
    )
    return expected_digest


def _validate_traffic_gate(
    gate: Mapping[str, Any],
    *,
    incumbent_identity: Mapping[str, Any],
    candidate_identity: Mapping[str, Any],
    candidate_model_identity: Mapping[str, Any],
    candidate_config_identity: Mapping[str, Any],
    incumbent_model_identity: Mapping[str, Any],
    incumbent_config_identity: Mapping[str, Any],
    snapshot_manifest_identity: Mapping[str, Any],
    text_snapshot_binding: Mapping[str, Any],
    traffic_evaluator_identity: Mapping[str, Any],
    traffic_gate_identity: Mapping[str, Any],
    frozen_test_protocol_identity: Mapping[str, Any],
) -> tuple[bool, Dict[str, Any]]:
    if gate.get("schema_version") != TRAFFIC_GATE_SCHEMA:
        raise ManifestError("traffic gate schema_version mismatch")
    incumbent = _mapping(gate.get("incumbent"), "traffic_gate.incumbent")
    candidate = _mapping(gate.get("candidate"), "traffic_gate.candidate")
    _same_identity(
        incumbent_identity, incumbent.get("evaluation"), "traffic_gate.incumbent.evaluation"
    )
    _same_identity(
        candidate_identity, candidate.get("evaluation"), "traffic_gate.candidate.evaluation"
    )
    _same_identity(
        candidate_model_identity,
        candidate.get("adapter_model"),
        "traffic_gate.candidate.adapter_model",
    )
    _same_path_sha(
        incumbent_model_identity,
        incumbent.get("adapter_model"),
        "traffic_gate.incumbent.adapter_model",
    )
    _same_path_sha(
        traffic_gate_identity,
        gate.get("gate_script_identity"),
        "traffic_gate.gate_script_identity",
    )
    side_bindings = {
        "incumbent": (incumbent, incumbent_model_identity, incumbent_config_identity),
        "candidate": (candidate, candidate_model_identity, candidate_config_identity),
    }
    for side_name, (side, adapter_model, adapter_config) in side_bindings.items():
        artifacts = _mapping(
            side.get("evaluation_artifacts"),
            f"traffic_gate.{side_name}.evaluation_artifacts",
        )
        _validate_traffic_artifacts(
            artifacts,
            label=f"traffic_gate.{side_name}.evaluation_artifacts",
            expected_adapter_model=adapter_model,
            expected_adapter_config=adapter_config,
            snapshot_manifest_identity=snapshot_manifest_identity,
            text_snapshot_binding=text_snapshot_binding,
            traffic_evaluator_identity=traffic_evaluator_identity,
        )
    anchors = _mapping(
        gate.get("external_sha256_anchors"), "traffic_gate.external_sha256_anchors"
    )
    if anchors.get("triple_anchor_complete") is not True:
        raise ManifestError("traffic gate is missing its triple SHA-256 anchor")
    expected_anchor_shas = {
        "frozen_test": _sha256(
            frozen_test_protocol_identity.get("sha256"),
            "u2_protocol.frozen_inputs.traffic_regression_jsonl.sha256",
        ),
        "evaluator": _sha256(
            traffic_evaluator_identity.get("sha256"),
            "u2_protocol.implementations.traffic_evaluator.sha256",
        ),
        "gate": _sha256(
            traffic_gate_identity.get("sha256"),
            "u2_protocol.implementations.traffic_gate.sha256",
        ),
    }
    for name, expected_sha in expected_anchor_shas.items():
        anchor = _mapping(
            anchors.get(name), f"traffic_gate.external_sha256_anchors.{name}"
        )
        if anchor.get("matched") is not True:
            raise ManifestError(f"traffic gate {name} SHA-256 anchor is not matched")
        if _sha256(
            anchor.get("expected_sha256"),
            f"traffic_gate.external_sha256_anchors.{name}.expected_sha256",
        ) != expected_sha:
            raise ManifestError(
                f"traffic gate {name} expected SHA-256 differs from the anchored U2 protocol"
            )
        if _sha256(
            anchor.get("actual_sha256"),
            f"traffic_gate.external_sha256_anchors.{name}.actual_sha256",
        ) != expected_sha:
            raise ManifestError(
                f"traffic gate {name} actual SHA-256 differs from the anchored U2 protocol"
            )
    frozen_test = _mapping(gate.get("frozen_test"), "traffic_gate.frozen_test")
    _same_path_sha(
        frozen_test_protocol_identity,
        frozen_test,
        "traffic_gate.frozen_test",
    )
    if _sha256(
        frozen_test.get("expected_sha256"),
        "traffic_gate.frozen_test.expected_sha256",
    ) != expected_anchor_shas["frozen_test"]:
        raise ManifestError(
            "traffic gate frozen_test expected SHA-256 differs from the anchored U2 protocol"
        )
    hard_gate = _mapping(gate.get("hard_gate"), "traffic_gate.hard_gate")
    checks = _mapping(hard_gate.get("checks"), "traffic_gate.hard_gate.checks")
    if set(checks) != TRAFFIC_CHECKS:
        raise ManifestError("traffic gate check set does not match the fixed schema")
    recomputed = all(
        _boolean(_mapping(row, f"traffic_gate.checks.{name}").get("passed"),
                 f"traffic_gate.checks.{name}.passed")
        for name, row in checks.items()
    )
    declared = _boolean(hard_gate.get("all_passed"), "traffic_gate.hard_gate.all_passed")
    if declared != recomputed:
        raise ManifestError("traffic gate all_passed is inconsistent with checks")
    return declared, {
        **_declared_path_sha(frozen_test, "traffic_gate.frozen_test"),
        "expected_sha256": expected_anchor_shas["frozen_test"],
        "verified_via_sha_bound_traffic_gate_without_opening_test_content": True,
    }


def _validate_general_gate(
    gate: Mapping[str, Any],
    *,
    candidate_samples_identity: Mapping[str, Any],
    candidate_summary_identity: Mapping[str, Any],
    teacher_samples_identity: Mapping[str, Any],
    teacher_summary_identity: Mapping[str, Any],
    candidate_model_identity: Mapping[str, Any],
    general_candidate_evaluator_identity: Mapping[str, Any],
    general_teacher_evaluator_identity: Mapping[str, Any],
    general_gate_script_identity: Mapping[str, Any],
    candidate_model_sha256: str,
    teacher_model: str,
    teacher_model_sha256: str,
) -> Dict[str, bool]:
    if gate.get("schema_version") != GENERAL_GATE_SCHEMA:
        raise ManifestError("general gate schema_version mismatch")
    if gate.get("development_only") is not True or gate.get("formal_blind_test_used") is not False:
        raise ManifestError("general gate development/blind isolation declaration invalid")
    inputs = _mapping(gate.get("inputs"), "general_gate.inputs")
    evaluation_implementations = _mapping(
        inputs.get("evaluation_implementations"),
        "general_gate.inputs.evaluation_implementations",
    )
    _same_path_sha(
        general_candidate_evaluator_identity,
        evaluation_implementations.get("candidate"),
        "general_gate.inputs.evaluation_implementations.candidate",
    )
    _same_path_sha(
        general_teacher_evaluator_identity,
        evaluation_implementations.get("teacher"),
        "general_gate.inputs.evaluation_implementations.teacher",
    )
    _same_path_sha(
        general_gate_script_identity,
        inputs.get("gate_script"),
        "general_gate.inputs.gate_script",
    )
    model_binding = _mapping(gate.get("model_binding"), "general_gate.model_binding")
    if _sha256(
        model_binding.get("candidate_model_sha256"),
        "general_gate.model_binding.candidate_model_sha256",
    ) != candidate_model_sha256:
        raise ManifestError("general gate candidate model differs from candidate summary")
    if model_binding.get("teacher_model") != teacher_model:
        raise ManifestError("general gate teacher model differs from protocol")
    if _sha256(
        model_binding.get("teacher_model_sha256"),
        "general_gate.model_binding.teacher_model_sha256",
    ) != teacher_model_sha256:
        raise ManifestError("general gate teacher digest differs from protocol")
    candidate = _mapping(inputs.get("candidate"), "general_gate.inputs.candidate")
    teacher = _mapping(inputs.get("teacher"), "general_gate.inputs.teacher")
    _same_identity(
        candidate_model_identity,
        candidate.get("adapter_model"),
        "general_gate.inputs.candidate.adapter_model",
    )
    _same_identity(
        candidate_samples_identity,
        candidate.get("samples"),
        "general_gate.inputs.candidate.samples",
    )
    _same_identity(
        candidate_summary_identity,
        candidate.get("summary"),
        "general_gate.inputs.candidate.summary",
    )
    _same_identity(
        teacher_samples_identity,
        teacher.get("samples"),
        "general_gate.inputs.teacher.samples",
    )
    _same_identity(
        teacher_summary_identity,
        teacher.get("summary"),
        "general_gate.inputs.teacher.summary",
    )
    categories = _mapping(gate.get("categories"), "general_gate.categories")
    expected_categories = {"math", "natural_language_reasoning"}
    if set(categories) != expected_categories:
        raise ManifestError("general gate must contain exactly math and natural_language_reasoning")
    result: Dict[str, bool] = {}
    for name in sorted(expected_categories):
        category = _mapping(categories[name], f"general_gate.categories.{name}")
        checks = _mapping(
            category.get("checks"), f"general_gate.categories.{name}.checks"
        )
        if set(checks) != GENERAL_CATEGORY_CHECKS:
            raise ManifestError(
                f"general gate category {name} check set does not match the fixed schema"
            )
        check_passed = {
            check_name: _boolean(
                checks[check_name],
                f"general_gate.categories.{name}.checks.{check_name}",
            )
            for check_name in sorted(GENERAL_CATEGORY_CHECKS)
        }
        passed = _boolean(
            category.get("passed"), f"general_gate.categories.{name}.passed"
        )
        if passed != all(check_passed.values()):
            raise ManifestError(
                f"general gate category {name} passed is inconsistent with checks"
            )
        result[name] = passed
    declared = _boolean(gate.get("passed"), "general_gate.passed")
    if declared != all(result.values()):
        raise ManifestError("general gate passed is inconsistent with category gates")
    return result


def _validate_step_receipts(
    value: Mapping[str, Any],
    identity: Mapping[str, Any],
    expected_sha256: str,
    protocol_steps: Sequence[Mapping[str, Any]],
) -> tuple[Dict[str, Dict[str, Any]], list[Dict[str, Any]], Dict[str, Any]]:
    expected = _sha256(expected_sha256, "expected_step_receipts_sha256")
    if identity.get("sha256") != expected:
        raise ManifestError("step receipts SHA-256 does not match the explicit anchor")
    if value.get("schema_version") != STEP_RECEIPTS_SCHEMA:
        raise ManifestError("step receipts schema_version mismatch")
    if value.get("candidate_id") != CANDIDATE_ID:
        raise ManifestError("step receipts candidate_id mismatch")
    run_id = _text(value.get("run_id"), "step_receipts.run_id")
    rows = value.get("steps")
    if not isinstance(rows, list) or len(rows) != len(protocol_steps):
        raise ManifestError("step_receipts.steps must contain the protocol's 8 steps")
    indexed: Dict[str, Dict[str, Any]] = {}
    ordered: list[Dict[str, Any]] = []
    for index, (raw, protocol_step) in enumerate(zip(rows, protocol_steps), start=1):
        step = _mapping(raw, f"step_receipts.steps[{index - 1}]")
        if step.get("step") != index:
            raise ManifestError("step receipt numbering must be exactly 1 through 8")
        name = _text(step.get("name"), f"step_receipts.steps[{index - 1}].name")
        if name in indexed:
            raise ManifestError(f"duplicate step receipt: {name}")
        if name != protocol_step.get("name"):
            raise ManifestError(f"step {index} name differs from the anchored U2 protocol")
        command = step.get("command_argv")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or not item for item in command)
        ):
            raise ManifestError(f"step {name}.command_argv must be a non-empty string array")
        if list(command) != protocol_step.get("command_argv"):
            raise ManifestError(
                f"step {index} command_argv differs from the resolved anchored U2 protocol"
            )
        working_directory = str(
            Path(
                _text(step.get("working_directory"), f"step {name}.working_directory")
            ).expanduser().resolve()
        )
        if working_directory != protocol_step.get("working_directory"):
            raise ManifestError(
                f"step {index} working_directory differs from the anchored U2 protocol"
            )
        started = _utc_timestamp(step.get("started_at"), f"step {name}.started_at")
        ended = _utc_timestamp(step.get("ended_at"), f"step {name}.ended_at")
        start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(ended.replace("Z", "+00:00"))
        if end_dt < start_dt:
            raise ManifestError(f"step {name} ended before it started")
        exit_code = _integer(step.get("observed_exit_code"), f"step {name}.observed_exit_code")
        observation = _mapping(
            step.get("exit_code_observation"), f"step {name}.exit_code_observation"
        )
        if observation.get("observed") is not True or observation.get("inferred") is not False:
            raise ManifestError(f"step {name} exit code must be explicitly observed, never inferred")
        _text(observation.get("source"), f"step {name}.exit_code_observation.source")
        outputs = _mapping(step.get("outputs"), f"step {name}.outputs")
        validated = {
            "step": index,
            "name": name,
            "command_argv": list(command),
            "working_directory": working_directory,
            "started_at": started,
            "ended_at": ended,
            "observed_exit_code": exit_code,
            "exit_code_observation": dict(observation),
            "outputs": {
                role: _declared_identity(row, f"step {name}.outputs.{role}")
                for role, row in outputs.items()
            },
        }
        indexed[name] = validated
        ordered.append(validated)
    if set(indexed) != set(REQUIRED_STEP_OUTPUTS):
        raise ManifestError("step receipt names do not match the anchored U2 workflow")
    for name, expected_outputs in REQUIRED_STEP_OUTPUTS.items():
        if set(indexed[name]["outputs"]) != expected_outputs:
            raise ManifestError(f"step {name} output roles do not match the fixed protocol")
    for name in NON_GATE_STEPS:
        if indexed[name]["observed_exit_code"] != 0:
            raise ManifestError(f"non-gate step {name} did not have observed exit code 0")
    return indexed, ordered, {
        **dict(identity),
        "expected_sha256": expected,
        "verified": True,
        "run_id": run_id,
    }


def _bind_step_output(
    steps: Mapping[str, Mapping[str, Any]],
    step_name: str,
    role: str,
    actual: Mapping[str, Any],
) -> None:
    declared = _mapping(steps[step_name]["outputs"], f"step {step_name}.outputs").get(role)
    _same_identity(actual, declared, f"step {step_name}.outputs.{role}")


def _interpret_gate_exit(step: Mapping[str, Any], passed: bool, label: str) -> str:
    exit_code = step.get("observed_exit_code")
    if passed:
        if exit_code != 0:
            raise ManifestError(f"{label} report passed but observed exit code was not 0")
        return "valid_gate_pass"
    if exit_code != 2:
        raise ManifestError(f"{label} report failed but observed exit code was not 2")
    return "valid_gate_failure_report_present"


def build_execution_receipt(
    *,
    candidate_receipt_path: Path,
    traffic_incumbent_evaluation_path: Path,
    traffic_candidate_evaluation_path: Path,
    traffic_gate_path: Path,
    general_candidate_samples_path: Path,
    general_candidate_summary_path: Path,
    general_teacher_samples_path: Path,
    general_teacher_summary_path: Path,
    general_gate_path: Path,
    step_receipts_path: Path,
    expected_step_receipts_sha256: str,
) -> Dict[str, Any]:
    candidate_receipt, candidate_receipt_identity = _read_json(
        candidate_receipt_path, "candidate artifact receipt"
    )
    candidate = _validate_candidate_receipt(
        candidate_receipt, candidate_receipt_identity
    )
    candidate_model_identity = candidate["files"]["adapter_model.safetensors"]
    protocol, protocol_identity, protocol_steps = _load_anchored_protocol(
        candidate_receipt=candidate_receipt,
        candidate_receipt_identity=candidate_receipt_identity,
        candidate_model_identity=candidate_model_identity,
    )
    protocol_file_bindings = _validate_protocol_file_bindings(protocol)
    protocol_implementations = protocol_file_bindings["implementations"]
    protocol_frozen_inputs = protocol_file_bindings["frozen_inputs"]
    text_snapshot_binding = protocol_file_bindings["text_snapshot"]
    snapshot_manifest_identity = protocol_frozen_inputs["text_snapshot_manifest"]
    base_manifest_identity = protocol_frozen_inputs["base_manifest"]
    incumbent_model_identity = protocol_frozen_inputs[
        "production_traffic_adapter_weights"
    ]
    incumbent_config_identity = protocol_frozen_inputs[
        "production_traffic_adapter_config"
    ]
    candidate_config_identity = candidate["files"]["adapter_config.json"]
    step_value, step_identity = _read_json(step_receipts_path, "observed step receipts")
    steps, ordered_steps, trusted_log = _validate_step_receipts(
        step_value,
        step_identity,
        expected_step_receipts_sha256,
        protocol_steps,
    )

    traffic_incumbent, traffic_incumbent_identity = _read_json(
        traffic_incumbent_evaluation_path, "traffic incumbent evaluation"
    )
    traffic_candidate, traffic_candidate_identity = _read_json(
        traffic_candidate_evaluation_path, "traffic candidate evaluation"
    )
    _validate_traffic_evaluation(
        traffic_incumbent,
        "traffic_incumbent_evaluation",
        expected_adapter_model=incumbent_model_identity,
        expected_adapter_config=incumbent_config_identity,
        snapshot_manifest_identity=snapshot_manifest_identity,
        text_snapshot_binding=text_snapshot_binding,
        traffic_evaluator_identity=protocol_implementations["traffic_evaluator"],
    )
    _validate_traffic_evaluation(
        traffic_candidate,
        "traffic_candidate_evaluation",
        expected_adapter_model=candidate_model_identity,
        expected_adapter_config=candidate_config_identity,
        snapshot_manifest_identity=snapshot_manifest_identity,
        text_snapshot_binding=text_snapshot_binding,
        traffic_evaluator_identity=protocol_implementations["traffic_evaluator"],
    )
    _validate_traffic_candidate_model(traffic_candidate, candidate_model_identity)
    traffic_gate, traffic_gate_identity = _read_json(traffic_gate_path, "traffic gate")
    traffic_passed, verified_traffic_test_identity = _validate_traffic_gate(
        traffic_gate,
        incumbent_identity=traffic_incumbent_identity,
        candidate_identity=traffic_candidate_identity,
        candidate_model_identity=candidate_model_identity,
        candidate_config_identity=candidate_config_identity,
        incumbent_model_identity=incumbent_model_identity,
        incumbent_config_identity=incumbent_config_identity,
        snapshot_manifest_identity=snapshot_manifest_identity,
        text_snapshot_binding=text_snapshot_binding,
        traffic_evaluator_identity=protocol_implementations["traffic_evaluator"],
        traffic_gate_identity=protocol_implementations["traffic_gate"],
        frozen_test_protocol_identity=protocol_frozen_inputs[
            TRAFFIC_TEST_FROZEN_INPUT
        ],
    )
    protocol_frozen_inputs[TRAFFIC_TEST_FROZEN_INPUT] = (
        verified_traffic_test_identity
    )

    general_candidate_summary, general_candidate_summary_identity = _read_json(
        general_candidate_summary_path, "general candidate summary"
    )
    general_teacher_summary, general_teacher_summary_identity = _read_json(
        general_teacher_summary_path, "general teacher summary"
    )
    _validate_general_summary(
        general_candidate_summary,
        label="general_candidate_summary",
        expected_schema=GENERAL_CANDIDATE_SCHEMA,
    )
    _validate_general_summary(
        general_teacher_summary,
        label="general_teacher_summary",
        expected_schema=GENERAL_TEACHER_SCHEMA,
    )
    general_candidate_model_sha = _validate_general_candidate_model_binding(
        general_candidate_summary,
        candidate=candidate,
        base_manifest_identity=base_manifest_identity,
        snapshot_manifest_identity=snapshot_manifest_identity,
        text_snapshot_binding=text_snapshot_binding,
        candidate_evaluator_identity=protocol_implementations[
            "general_candidate_evaluator"
        ],
    )
    teacher_contract = _mapping(protocol.get("teacher"), "u2_protocol.teacher")
    general_teacher_model_sha = _validate_general_teacher_model_binding(
        general_teacher_summary,
        teacher_contract=teacher_contract,
        teacher_evaluator_identity=protocol_implementations[
            "general_teacher_evaluator"
        ],
    )
    _, general_candidate_samples_identity = _read_bytes(
        general_candidate_samples_path, "general candidate samples"
    )
    _, general_teacher_samples_identity = _read_bytes(
        general_teacher_samples_path, "general teacher samples"
    )
    _validate_sample_binding(
        general_candidate_summary,
        general_candidate_samples_identity,
        "general_candidate_summary",
    )
    _validate_sample_binding(
        general_teacher_summary,
        general_teacher_samples_identity,
        "general_teacher_summary",
    )
    general_gate, general_gate_identity = _read_json(general_gate_path, "general gate")
    general_passes = _validate_general_gate(
        general_gate,
        candidate_samples_identity=general_candidate_samples_identity,
        candidate_summary_identity=general_candidate_summary_identity,
        teacher_samples_identity=general_teacher_samples_identity,
        teacher_summary_identity=general_teacher_summary_identity,
        candidate_model_identity=candidate_model_identity,
        general_candidate_evaluator_identity=protocol_implementations[
            "general_candidate_evaluator"
        ],
        general_teacher_evaluator_identity=protocol_implementations[
            "general_teacher_evaluator"
        ],
        general_gate_script_identity=protocol_implementations[
            "general_paired_gate"
        ],
        candidate_model_sha256=general_candidate_model_sha,
        teacher_model=_text(teacher_contract.get("model"), "u2_protocol.teacher.model"),
        teacher_model_sha256=general_teacher_model_sha,
    )

    actual_by_step = {
        ("train_U2_once", "adapter_model"): candidate["files"]["adapter_model.safetensors"],
        ("train_U2_once", "adapter_config"): candidate["files"]["adapter_config.json"],
        ("train_U2_once", "train_metrics"): candidate["files"]["train_metrics.json"],
        ("create_candidate_artifact_receipt_without_scoring", "candidate_receipt"): candidate_receipt_identity,
        ("traffic_incumbent_bf16", "traffic_incumbent_evaluation"): traffic_incumbent_identity,
        ("traffic_candidate_bf16", "traffic_candidate_evaluation"): traffic_candidate_identity,
        ("traffic_paired_gate", "traffic_gate"): traffic_gate_identity,
        ("general_candidate_bf16_fixed_800", "general_candidate_samples"): general_candidate_samples_identity,
        ("general_candidate_bf16_fixed_800", "general_candidate_summary"): general_candidate_summary_identity,
        ("general_teacher_fixed_800", "general_teacher_samples"): general_teacher_samples_identity,
        ("general_teacher_fixed_800", "general_teacher_summary"): general_teacher_summary_identity,
        ("general_paired_gate", "general_gate"): general_gate_identity,
    }
    for (step_name, role), actual in actual_by_step.items():
        _bind_step_output(steps, step_name, role, actual)

    traffic_exit = _interpret_gate_exit(
        steps["traffic_paired_gate"], traffic_passed, "traffic gate"
    )
    general_overall = all(general_passes.values())
    general_exit = _interpret_gate_exit(
        steps["general_paired_gate"], general_overall, "general gate"
    )
    gates = {
        "traffic": traffic_passed,
        "math": general_passes["math"],
        "natural_language_reasoning": general_passes[
            "natural_language_reasoning"
        ],
    }
    all_passed = all(gates.values())
    core = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": "u2_bf16_final_execution_decision",
        "candidate_id": CANDIDATE_ID,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "anchored_u2_protocol": {
            **protocol_identity,
            "schema_version": U2_PROTOCOL_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "dynamic_placeholders_resolved_before_comparison": True,
            "step_commands_and_working_directories_exactly_matched": True,
            "current_file_bindings": protocol_file_bindings,
        },
        "trusted_observed_step_receipts": trusted_log,
        "steps": ordered_steps,
        "evidence": {
            "candidate_artifact": candidate,
            "traffic": {
                "incumbent_evaluation": traffic_incumbent_identity,
                "candidate_evaluation": traffic_candidate_identity,
                "gate_report": traffic_gate_identity,
            },
            "general": {
                "candidate_samples": general_candidate_samples_identity,
                "candidate_summary": general_candidate_summary_identity,
                "teacher_samples": general_teacher_samples_identity,
                "teacher_summary": general_teacher_summary_identity,
                "gate_report": general_gate_identity,
            },
        },
        "gate_exit_interpretation": {
            "traffic_paired_gate": traffic_exit,
            "general_paired_gate": general_exit,
            "exit_code_2_alone_is_gate_failure_evidence": False,
        },
        "three_hard_gates": gates,
        "all_three_hard_gates_passed": all_passed,
        "decision": (
            "eligible_to_freeze_separate_merge_Q6_K_post_quantization_protocol"
            if all_passed
            else "stop_loss_no_merge_no_quantization_no_release"
        ),
        "automatic_merge_quantization_or_release_performed": False,
        "source_test_or_blind_dataset_opened_by_this_tool": False,
    }
    return {**core, "receipt_id": canonical_sha256(core)}


def write_receipt_exclusive(path: Path, receipt: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(dict(receipt), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ManifestError(f"refusing to overwrite U2 execution receipt: {output}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            output.unlink()
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-receipt", required=True)
    parser.add_argument("--traffic-incumbent-evaluation", required=True)
    parser.add_argument("--traffic-candidate-evaluation", required=True)
    parser.add_argument("--traffic-gate", required=True)
    parser.add_argument("--general-candidate-samples", required=True)
    parser.add_argument("--general-candidate-summary", required=True)
    parser.add_argument("--general-teacher-samples", required=True)
    parser.add_argument("--general-teacher-summary", required=True)
    parser.add_argument("--general-gate", required=True)
    parser.add_argument("--step-receipts", required=True)
    parser.add_argument("--expected-step-receipts-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    receipt = build_execution_receipt(
        candidate_receipt_path=Path(args.candidate_receipt),
        traffic_incumbent_evaluation_path=Path(args.traffic_incumbent_evaluation),
        traffic_candidate_evaluation_path=Path(args.traffic_candidate_evaluation),
        traffic_gate_path=Path(args.traffic_gate),
        general_candidate_samples_path=Path(args.general_candidate_samples),
        general_candidate_summary_path=Path(args.general_candidate_summary),
        general_teacher_samples_path=Path(args.general_teacher_samples),
        general_teacher_summary_path=Path(args.general_teacher_summary),
        general_gate_path=Path(args.general_gate),
        step_receipts_path=Path(args.step_receipts),
        expected_step_receipts_sha256=args.expected_step_receipts_sha256,
    )
    write_receipt_exclusive(Path(args.output), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
