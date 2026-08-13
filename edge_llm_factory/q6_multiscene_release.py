"""Build an immutable clean-Q6, two-LoRA release package from completed gates.

This module is deliberately stricter than the generic adapter packager.  It
refuses to build unless the frozen 2400/960 quality run, the mixed-adapter
isolation run, and the 250+250 Nano stability run all bind to the exact Q6
base and ordered traffic/industrial runtime LoRAs declared by the descriptor.
It never promotes a release or talks to a runtime service.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Sequence

from edge_llm_factory.adapter_package import build_adapter_package
from edge_llm_factory.contracts import (
    ManifestError,
    read_json_object,
    sha256_file,
    write_json_object,
)


SCHEMA = "edge-llm-q6-multiscene-release-input/v1"
QUALITY_SCHEMA = "q6-full-multilora-quality-gate/v1"
ISOLATION_SCHEMA = "multi-adapter-isolation-benchmark/v1"
STABILITY_SCHEMA = "multi-adapter-alternating-stability/v1"

TRAFFIC_SCENE = "freeway_traffic_management"
INDUSTRIAL_SCENE = "industrial_anomaly"

TRAFFIC_MIN_ACCURACY = 0.60
TRAFFIC_MIN_WEIGHTED_F1 = 0.60
INDUSTRIAL_MIN_ACCURACY = 1.0
MIN_MEM_AVAILABLE_MIB = 256.0
MAX_WINDOW_GROWTH_MIB = 32.0
MAX_WINDOW_PSWPIN_PAGES = 8192
BYTES_PER_KIB = 1024
BYTES_PER_MIB = 1024 * BYTES_PER_KIB
# The competition contract says 1.5 GB, so this deliberately uses decimal
# bytes rather than treating the limit as 1.5 GiB.
MAX_NANO_RESIDENT_PLUS_SWAP_BYTES = 1_500_000_000


def _resolve(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError("{} must be a non-empty path".format(field))
    raw = Path(value).expanduser()
    return (raw if raw.is_absolute() else root / raw).resolve()


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


def _path(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ManifestError("evidence is missing {}".format(dotted))
        current = current[part]
    return current


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


def _identity(path: Path, declared: Mapping[str, Any], field: str) -> Dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ManifestError("{} is missing, non-regular, or a symlink: {}".format(field, path))
    expected_bytes = _integer(declared.get("bytes"), field + ".bytes")
    expected_sha = declared.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ManifestError("{}.sha256 must be a 64-character digest".format(field))
    actual_bytes = path.stat().st_size
    actual_sha = sha256_file(path)
    _equal(actual_bytes, expected_bytes, field + ".bytes")
    _equal(actual_sha, expected_sha, field + ".sha256")
    return {"path": str(path), "bytes": actual_bytes, "sha256": actual_sha}


def _evidence_identity(
    record: Mapping[str, Any], expected: Mapping[str, Any], field: str
) -> None:
    _equal(record.get("bytes"), expected["bytes"], field + ".bytes")
    _equal(record.get("sha256"), expected["sha256"], field + ".sha256")


def _runtime_adapters(
    root: Path, value: Any
) -> list[Dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 2:
        raise ManifestError("runtime_adapters must contain exactly traffic id 0 and industrial id 1")
    expected_scenes = (TRAFFIC_SCENE, INDUSTRIAL_SCENE)
    result = []
    for adapter_id, (raw, scene) in enumerate(zip(value, expected_scenes)):
        if not isinstance(raw, dict):
            raise ManifestError("runtime_adapters[{}] must be an object".format(adapter_id))
        _equal(raw.get("id"), adapter_id, "runtime_adapters[{}].id".format(adapter_id))
        _equal(raw.get("scene"), scene, "runtime_adapters[{}].scene".format(adapter_id))
        if raw.get("default_scale") not in (0, 0.0):
            raise ManifestError("runtime adapter default_scale must be exactly zero")
        path = _resolve(root, raw.get("path"), "runtime_adapters[{}].path".format(adapter_id))
        identity = _identity(path, raw, "runtime_adapters[{}]".format(adapter_id))
        result.append({"id": adapter_id, "scene": scene, "default_scale": 0, **identity})
    return result


def _validate_quality(
    report: Mapping[str, Any], base: Mapping[str, Any], adapters: Sequence[Mapping[str, Any]]
) -> Dict[str, float]:
    _equal(report.get("schema_version"), QUALITY_SCHEMA, "quality.schema_version")
    _evidence_identity(_path(report, "runtime.base_q6"), base, "quality.runtime.base_q6")
    _evidence_identity(
        _path(report, "runtime.traffic_lora_id0"), adapters[0],
        "quality.runtime.traffic_lora_id0",
    )
    _evidence_identity(
        _path(report, "runtime.industrial_lora_id1"), adapters[1],
        "quality.runtime.industrial_lora_id1",
    )
    metrics: Dict[str, float] = {}
    for scene, count, minimum in (
        ("traffic", 2400, TRAFFIC_MIN_ACCURACY),
        ("industrial", 960, INDUSTRIAL_MIN_ACCURACY),
    ):
        prefix = scene + ".classification."
        _equal(_path(report, prefix + "count"), count, "quality." + prefix + "count")
        _equal(_path(report, prefix + "valid_output_rate"), 1.0, "quality." + prefix + "valid_output_rate")
        metrics[scene + "_accuracy"] = _at_least(
            _path(report, prefix + "accuracy"), minimum, "quality." + prefix + "accuracy"
        )
        metrics[scene + "_weighted_f1"] = _number(
            _path(report, prefix + "weighted_f1"), "quality." + prefix + "weighted_f1"
        )
        for name in (
            "input_chars_all_16_digits",
            "prompt_tokens_all_16",
            "output_tokens_all_1",
            "grammar_applied_before_sampling",
        ):
            _equal(_path(report, scene + ".contract." + name), True, "quality.{}.contract.{}".format(scene, name))
        _equal(
            _path(report, scene + ".contract.post_hoc_remapping"), False,
            "quality.{}.contract.post_hoc_remapping".format(scene),
        )
    _at_least(metrics["traffic_weighted_f1"], TRAFFIC_MIN_WEIGHTED_F1, "quality.traffic.weighted_f1")
    return metrics


def _validate_isolation(
    report: Mapping[str, Any], base: Mapping[str, Any]
) -> Dict[str, float]:
    _equal(report.get("schema_version"), ISOLATION_SCHEMA, "isolation.schema_version")
    for scene, adapter_id in (("traffic", 0), ("industrial", 1)):
        _equal(
            _path(report, scene + "_runtime.lora_adapter.id"), adapter_id,
            "isolation.{}_runtime.lora_adapter.id".format(scene),
        )
        model = Path(str(_path(report, scene + "_runtime.model"))).expanduser().resolve()
        _equal(model, Path(base["path"]), "isolation.{}_runtime.model".format(scene))
        _at_least(
            _path(report, "sequential_alternating.{}.count".format(scene)), 100,
            "isolation.sequential_alternating.{}.count".format(scene),
        )
        _equal(
            _path(report, "sequential_alternating.{}.valid_output_rate".format(scene)), 1.0,
            "isolation.sequential_alternating.{}.valid_output_rate".format(scene),
        )
        _equal(
            _path(report, "concurrent_mixed_4.{}.valid_output_rate".format(scene)), 1.0,
            "isolation.concurrent_mixed_4.{}.valid_output_rate".format(scene),
        )
        _equal(
            _path(report, "sequential_vs_concurrent_prediction_consistency.{}.consistency_rate".format(scene)),
            1.0,
            "isolation.{}.consistency_rate".format(scene),
        )
    _equal(report.get("cross_scene_adapter_contamination_detected"), False, "isolation.cross_scene_adapter_contamination_detected")
    return {
        "isolation_traffic_consistency": 1.0,
        "isolation_industrial_consistency": 1.0,
        "isolation_contamination_free": 1.0,
    }


def _nano_asset_evidence(
    report: Mapping[str, Any], base: Mapping[str, Any], adapters: Sequence[Mapping[str, Any]]
) -> None:
    evidence = report.get("asset_evidence")
    if not isinstance(evidence, dict):
        raise ManifestError(
            "Nano stability evidence lacks asset_evidence; rerun the gate with exact Q6/LoRA SHA capture"
        )
    _evidence_identity(_path(evidence, "base_q6"), base, "stability.asset_evidence.base_q6")
    rows = evidence.get("runtime_adapters")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ManifestError("stability.asset_evidence.runtime_adapters must have two records")
    for adapter_id, (row, expected) in enumerate(zip(rows, adapters)):
        if not isinstance(row, dict):
            raise ManifestError("stability runtime adapter evidence must be objects")
        _equal(row.get("id"), adapter_id, "stability.asset_evidence.runtime_adapters.id")
        _evidence_identity(row, expected, "stability.asset_evidence.runtime_adapters[{}]".format(adapter_id))


def _sample_at(samples: Any, request_count: int) -> Mapping[str, Any]:
    if not isinstance(samples, list):
        raise ManifestError("stability.resource_samples must be an array")
    matches = [row for row in samples if isinstance(row, dict) and row.get("request_count") == request_count]
    if len(matches) != 1:
        raise ManifestError("stability must contain exactly one resource sample at request {}".format(request_count))
    return matches[0]


def _positive_growth_mib(start: Mapping[str, Any], end: Mapping[str, Any], field: str) -> float:
    return max(0.0, (_number(end.get(field), field) - _number(start.get(field), field)) / 1024.0)


def _nonnegative_number(value: Any, field: str) -> float:
    result = _number(value, field)
    if result < 0:
        raise ManifestError("{} must be non-negative".format(field))
    return result


def _nonnegative_integer(value: Any, field: str) -> int:
    result = _integer(value, field)
    if result < 0:
        raise ManifestError("{} must be non-negative".format(field))
    return result


def _peak_memory_bytes(
    report: Mapping[str, Any], summary_field: str, sample_field: str
) -> int:
    """Return the most conservative available peak for one memory component.

    New evidence reports an explicit peak in MiB.  Older/raw evidence can
    provide the corresponding KiB field on every resource sample.  When both
    representations exist, take the larger value so an understated summary
    cannot weaken the hard memory gate.
    """

    candidates: list[int] = []
    resources = _path(report, "summary.resources")
    if not isinstance(resources, dict):
        raise ManifestError("stability.summary.resources must be an object")
    if summary_field in resources:
        peak_mib = _nonnegative_number(
            resources[summary_field], "stability.summary.resources." + summary_field
        )
        candidates.append(int(math.ceil(peak_mib * BYTES_PER_MIB)))

    samples = report.get("resource_samples")
    if isinstance(samples, list) and samples:
        present = [isinstance(row, dict) and sample_field in row for row in samples]
        if any(present):
            if not all(present):
                raise ManifestError(
                    "stability.resource_samples must all contain {}".format(sample_field)
                )
            raw_peak_kib = max(
                _nonnegative_integer(
                    row[sample_field],
                    "stability.resource_samples[{}].{}".format(index, sample_field),
                )
                for index, row in enumerate(samples)
            )
            candidates.append(raw_peak_kib * BYTES_PER_KIB)

    if not candidates:
        raise ManifestError(
            "stability memory evidence is missing {} and raw {}".format(
                summary_field, sample_field
            )
        )
    return max(candidates)


def _optional_peak_hwm_bytes(report: Mapping[str, Any]) -> int | None:
    resources = _path(report, "summary.resources")
    if not isinstance(resources, dict):
        raise ManifestError("stability.summary.resources must be an object")
    candidates: list[int] = []
    if "peak_llama_vmhwm_mib" in resources:
        value = _nonnegative_number(
            resources["peak_llama_vmhwm_mib"],
            "stability.summary.resources.peak_llama_vmhwm_mib",
        )
        candidates.append(int(math.ceil(value * BYTES_PER_MIB)))

    samples = report.get("resource_samples")
    if isinstance(samples, list) and samples:
        present = [isinstance(row, dict) and "llama_vmhwm_kib" in row for row in samples]
        if any(present):
            if not all(present):
                raise ManifestError(
                    "stability.resource_samples must all contain llama_vmhwm_kib when present"
                )
            peak_kib = max(
                _nonnegative_integer(
                    row["llama_vmhwm_kib"],
                    "stability.resource_samples[{}].llama_vmhwm_kib".format(index),
                )
                for index, row in enumerate(samples)
            )
            candidates.append(peak_kib * BYTES_PER_KIB)
    return max(candidates) if candidates else None


def _validate_stability(
    report: Mapping[str, Any], base: Mapping[str, Any], adapters: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    _equal(report.get("schema_version"), STABILITY_SCHEMA, "stability.schema_version")
    _equal(report.get("status"), "completed", "stability.status")
    _equal(_path(report, "stop_reason.code"), "completed", "stability.stop_reason.code")
    _equal(report.get("attempted_requests"), 500, "stability.attempted_requests")
    _equal(report.get("completed_requests"), 500, "stability.completed_requests")
    _equal(_path(report, "benchmark_contract.per_scene_requested"), 250, "stability.per_scene_requested")
    _equal(_path(report, "benchmark_contract.total_requested"), 500, "stability.total_requested")
    _equal(_path(report, "benchmark_contract.traffic_lora_id"), 0, "stability.traffic_lora_id")
    _equal(_path(report, "benchmark_contract.industrial_lora_id"), 1, "stability.industrial_lora_id")
    _nano_asset_evidence(report, base, adapters)
    for scene, minimum in (("traffic", TRAFFIC_MIN_ACCURACY), ("industrial", INDUSTRIAL_MIN_ACCURACY)):
        _equal(_path(report, "summary.{}.completed".format(scene)), 250, "stability.{}.completed".format(scene))
        _equal(_path(report, "summary.{}.valid_output_rate_on_completed".format(scene)), 1.0, "stability.{}.valid_output_rate".format(scene))
        _at_least(_path(report, "summary.{}.accuracy_on_completed".format(scene)), minimum, "stability.{}.accuracy".format(scene))
    _equal(_path(report, "summary.overall.completion_rate"), 1.0, "stability.overall.completion_rate")
    _equal(report.get("errors"), [], "stability.errors")
    minimum_mem = _at_least(
        _path(report, "summary.resources.minimum_mem_available_mib"),
        MIN_MEM_AVAILABLE_MIB,
        "stability.minimum_mem_available_mib",
    )
    peak_rss_bytes = _peak_memory_bytes(
        report, "peak_llama_rss_mib", "llama_vmrss_kib"
    )
    peak_swap_bytes = _peak_memory_bytes(
        report, "peak_llama_vmswap_mib", "llama_vmswap_kib"
    )
    peak_resident_plus_swap_bytes = peak_rss_bytes + peak_swap_bytes
    if peak_resident_plus_swap_bytes > MAX_NANO_RESIDENT_PLUS_SWAP_BYTES:
        raise ManifestError(
            "stability.peak_llama_resident_plus_swap_bytes is {} but must be <= {}".format(
                peak_resident_plus_swap_bytes,
                MAX_NANO_RESIDENT_PLUS_SWAP_BYTES,
            )
        )
    peak_hwm_bytes = _optional_peak_hwm_bytes(report)
    at_100 = _sample_at(report.get("resource_samples"), 100)
    at_500 = _sample_at(report.get("resource_samples"), 500)
    rss_growth = _positive_growth_mib(at_100, at_500, "llama_vmrss_kib")
    swap_growth = _positive_growth_mib(at_100, at_500, "llama_vmswap_kib")
    _at_most(rss_growth, MAX_WINDOW_GROWTH_MIB, "stability.rss_growth_100_to_500_mib")
    _at_most(swap_growth, MAX_WINDOW_GROWTH_MIB, "stability.swap_growth_100_to_500_mib")
    pswpin = _integer(at_500.get("pswpin_pages"), "stability.pswpin@500") - _integer(
        at_100.get("pswpin_pages"), "stability.pswpin@100"
    )
    if pswpin < 0:
        raise ManifestError("stability pswpin counter moved backwards")
    _at_most(pswpin, MAX_WINDOW_PSWPIN_PAGES, "stability.pswpin_pages_100_to_500")
    metrics = {
        "nano_completed_requests": 500.0,
        "nano_completion_rate": 1.0,
        "nano_minimum_mem_available_mib": minimum_mem,
        "nano_peak_llama_rss_bytes": peak_rss_bytes,
        "nano_peak_llama_vmswap_bytes": peak_swap_bytes,
        "nano_peak_llama_resident_plus_swap_bytes": peak_resident_plus_swap_bytes,
        "nano_rss_growth_100_to_500_mib": rss_growth,
        "nano_swap_growth_100_to_500_mib": swap_growth,
        "nano_pswpin_pages_100_to_500": float(pswpin),
    }
    if peak_hwm_bytes is not None:
        metrics["nano_peak_llama_vmhwm_bytes"] = peak_hwm_bytes
    return metrics


def _gate_rows() -> list[Dict[str, Any]]:
    return [
        {"metric": "traffic_accuracy", "operator": ">=", "value": TRAFFIC_MIN_ACCURACY},
        {"metric": "traffic_weighted_f1", "operator": ">=", "value": TRAFFIC_MIN_WEIGHTED_F1},
        {"metric": "industrial_accuracy", "operator": "==", "value": INDUSTRIAL_MIN_ACCURACY},
        {"metric": "isolation_traffic_consistency", "operator": "==", "value": 1.0},
        {"metric": "isolation_industrial_consistency", "operator": "==", "value": 1.0},
        {"metric": "isolation_contamination_free", "operator": "==", "value": 1.0},
        {"metric": "nano_completed_requests", "operator": "==", "value": 500.0},
        {"metric": "nano_completion_rate", "operator": "==", "value": 1.0},
        {"metric": "nano_minimum_mem_available_mib", "operator": ">=", "value": MIN_MEM_AVAILABLE_MIB},
        {
            "metric": "nano_peak_llama_resident_plus_swap_bytes",
            "operator": "<=",
            "value": MAX_NANO_RESIDENT_PLUS_SWAP_BYTES,
        },
        {"metric": "nano_rss_growth_100_to_500_mib", "operator": "<=", "value": MAX_WINDOW_GROWTH_MIB},
        {"metric": "nano_swap_growth_100_to_500_mib", "operator": "<=", "value": MAX_WINDOW_GROWTH_MIB},
        {"metric": "nano_pswpin_pages_100_to_500", "operator": "<=", "value": float(MAX_WINDOW_PSWPIN_PAGES)},
    ]


def build_q6_multiscene_release(descriptor_path: Path, output_dir: Path) -> Dict[str, Any]:
    descriptor_file = descriptor_path.resolve()
    root = descriptor_file.parent
    descriptor = read_json_object(descriptor_file)
    _equal(descriptor.get("schema_version"), SCHEMA, "descriptor.schema_version")
    output = output_dir.resolve()
    if output.exists():
        raise ManifestError("immutable Q6 package output already exists: {}".format(output))

    deployment_raw = descriptor.get("deployment_artifact")
    if not isinstance(deployment_raw, dict):
        raise ManifestError("deployment_artifact must be an object")
    base_path = _resolve(root, deployment_raw.get("path"), "deployment_artifact.path")
    base = _identity(base_path, deployment_raw, "deployment_artifact")
    adapters = _runtime_adapters(root, descriptor.get("runtime_adapters"))

    evidence_raw = descriptor.get("evidence")
    if not isinstance(evidence_raw, dict):
        raise ManifestError("evidence must be an object")
    evidence_paths = {
        name: _resolve(root, evidence_raw.get(name), "evidence." + name)
        for name in ("full_quality", "isolation", "nano_stability")
    }
    evidence = {name: read_json_object(path) for name, path in evidence_paths.items()}
    metrics = {}
    metrics.update(_validate_quality(evidence["full_quality"], base, adapters))
    metrics.update(_validate_isolation(evidence["isolation"], base))
    metrics.update(_validate_stability(evidence["nano_stability"], base, adapters))

    base_manifest = _resolve(root, descriptor.get("base_manifest"), "base_manifest")
    adapter_source = _resolve(root, descriptor.get("primary_adapter_source"), "primary_adapter_source")
    action_mapping = _resolve(root, descriptor.get("action_mapping"), "action_mapping")
    adapter_id = descriptor.get("adapter_id")
    version = descriptor.get("version")
    if not isinstance(adapter_id, str) or not adapter_id:
        raise ManifestError("adapter_id must be non-empty")
    if not isinstance(version, str) or not version:
        raise ManifestError("version must be non-empty")

    summary = {
        "schema_version": "edge-llm-q6-multiscene-gate-summary/v1",
        "descriptor_sha256": sha256_file(descriptor_file),
        "assets": {"base_q6": base, "runtime_adapters": adapters},
        "source_evidence": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in evidence_paths.items()
        },
        "metrics": metrics,
        "fixed_gates": _gate_rows(),
        "release_authorized_by_packager": True,
    }
    with tempfile.TemporaryDirectory(prefix="q6-multiscene-package-") as temporary:
        temporary_root = Path(temporary)
        summary_path = temporary_root / "q6_gate_summary.json"
        write_json_object(summary_path, summary)
        sources = {
            name: {"evidence": "q6_gate_summary", "path": "metrics." + name}
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
                "context_encoder": "freeway-routing-context-decimal@v2",
                "max_input_tokens": 16,
                "direct_media_to_llm": False,
            },
            "training": dict(descriptor.get("training", {})),
            "evaluation": {
                # Only the sanitized Q6 summary enters the immutable package.
                # It binds the three full raw reports by SHA/bytes/path, while
                # deliberately excluding their contextual Q8 comparison blocks.
                "evidence": {"q6_gate_summary": str(summary_path)},
                "metric_sources": sources,
                "gates": _gate_rows(),
            },
            "deployment": {
                "artifact": str(base_path),
                "runtime": "llama.cpp",
                "format": "gguf",
                "quantization": "Q6_K",
                "max_input_tokens": 16,
                "max_output_tokens": 1,
                "thinking": False,
            },
        }
        for field in ("teacher_model", "methods", "train_dataset_id", "test_dataset_id", "test_set_used_for_training"):
            if field not in spec["training"]:
                raise ManifestError("training.{} is required".format(field))
        spec_path = temporary_root / "package_spec.json"
        write_json_object(spec_path, spec)
        result = build_adapter_package(
            project_root=root,
            base_manifest_path=base_manifest,
            spec_path=spec_path,
            output_dir=output,
        )
    result["runtime_adapters"] = adapters
    result["q6_gate_summary_sha256"] = sha256_file(output / "evidence" / "q6_gate_summary.json")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a fail-closed clean-Q6 dual-LoRA package")
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_q6_multiscene_release(Path(args.descriptor), Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
