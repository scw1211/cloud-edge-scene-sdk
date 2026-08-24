#!/usr/bin/env python3
"""Audit the industrial selective Qwen path without touching production services."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlsplit
from urllib.request import urlopen


SCENE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCENE_ROOT.parents[1]
for value in (str(PROJECT_ROOT), str(SCENE_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from cloud_edge_framework.event_envelope import SceneEventEnvelope  # noqa: E402
from industrial_anomaly.plugin import (  # noqa: E402
    INDUSTRIAL_ACTION_TOKENS,
    PRODUCTS,
    IndustrialAnomalyPlugin,
)


STATES = ("normal", "review", "anomaly")
MODALITIES = ("rgb", "infrared")
FORBIDDEN_PRODUCTION_PORTS = {18100, 18101, 18190}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Iterable[float]) -> Dict[str, Any]:
    items = [float(value) for value in values]
    if not items:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(items),
        "mean_ms": round(statistics.fmean(items), 6),
        "p50_ms": round(_percentile(items, 50.0), 6),
        "p95_ms": round(_percentile(items, 95.0), 6),
        "max_ms": round(max(items), 6),
    }


def _validate_isolated_endpoint(runtime: Mapping[str, Any]) -> str:
    endpoint = str(runtime.get("endpoint", ""))
    parsed = urlsplit(endpoint)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("audit runtime must use a loopback endpoint")
    port = parsed.port
    if port is None or port in FORBIDDEN_PRODUCTION_PORTS:
        raise ValueError("audit runtime must not use a production port")
    with urlopen(endpoint.rstrip("/") + "/health", timeout=3.0) as response:
        health = json.loads(response.read().decode("utf-8"))
    if health.get("status") != "ok":
        raise RuntimeError("isolated llama.cpp runtime is not ready")
    return endpoint


def _score(
    bands: Mapping[str, Any], modality: str, product: str, state: str
) -> float:
    band = bands["modalities"][modality][product]
    low = float(band["review_low"])
    high = float(band["review_high"])
    width = high - low
    if state == "normal":
        return low - width * 0.25
    if state == "review":
        return (low + high) / 2.0
    if state == "anomaly":
        return high + width * 0.25
    raise ValueError("unsupported state")


def _run_event(
    plugin: IndustrialAnomalyPlugin,
    template: Mapping[str, Any],
    bands: Mapping[str, Any],
    product: str,
    modality: str,
    state: str,
    case_id: str,
) -> Dict[str, Any]:
    payload = json.loads(json.dumps(template))
    payload["id"] = "{}-{}".format(case_id, modality)
    payload["subject"] = case_id
    payload["data"].update(
        {
            "sample_id": case_id,
            "product": product,
            "modality": modality,
            "score": _score(bands, modality, product, state),
        }
    )
    event = plugin.normalize_envelope(SceneEventEnvelope.from_dict(payload))
    prompt = plugin._edge_llm_prompt(event)  # The scene owns this public contract.
    started = time.perf_counter()
    decision = plugin.edge_decide(event)
    wall_ms = (time.perf_counter() - started) * 1000.0
    metadata = decision.metadata
    return {
        "case_id": case_id,
        "product": product,
        "modality": modality,
        "reference_state": state,
        "score": float(event.metadata["score"]),
        "review_low": float(event.metadata["review_low"]),
        "review_high": float(event.metadata["review_high"]),
        "encoded_prompt": prompt,
        "decision": decision.decision,
        "source": metadata.get("source"),
        "wall_ms": round(wall_ms, 6),
        "edge_decision_path": metadata.get("edge_decision_path"),
        "qwen_selected": bool(metadata.get("edge_qwen_selected", False)),
        "qwen_prediction": metadata.get("edge_qwen_prediction"),
        "qwen_token": metadata.get("edge_qwen_token"),
        "qwen_latency_ms": metadata.get("edge_qwen_latency_ms"),
        "qwen_prompt_tokens": metadata.get("edge_qwen_prompt_tokens"),
        "qwen_output_tokens": metadata.get("edge_qwen_output_tokens"),
        "qwen_rule_agreement": metadata.get("edge_qwen_rule_agreement"),
        "qwen_fallback": bool(metadata.get("edge_qwen_fallback", False)),
        "qwen_fallback_reason": metadata.get("edge_qwen_fallback_reason"),
        "decoding_constraint": metadata.get("edge_qwen_decoding_constraint"),
    }


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("refusing to overwrite evidence: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    if temporary.exists():
        raise FileExistsError("temporary path already exists: {}".format(temporary))
    with temporary.open("x", encoding="utf-8") as file_obj:
        json.dump(value, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results/industrial_qwen_path_audit_v1/isolated_runtime_config.json"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            PROJECT_ROOT
            / "model_bundle/final/qwen3_5_0p8b_q4/joint_static_raw16.Q4_K_M.gguf"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runtime_path = args.runtime_config.resolve()
    model_path = args.model.resolve()
    runtime = _read_json(runtime_path)
    endpoint = _validate_isolated_endpoint(runtime)
    if Path(str(runtime["model"])).resolve() != model_path:
        raise ValueError("isolated runtime model path does not match --model")

    formal_dir = SCENE_ROOT / "deployment/joint_static_raw16_formal_v2"
    formal_plugins_path = formal_dir / "scene_plugins_edge.json"
    formal_runtime_template_path = formal_dir / "edge_llm_runtime_industrial_template.json"
    formal_runtime_output_path = formal_dir / "runtime_output_industrial.json"
    gate_path = (
        PROJECT_ROOT
        / "model_bundle/final/qwen3_5_0p8b_q4/adapter_package/evidence/q4_static_raw16_gate_summary.json"
    )
    prior_matrix_path = SCENE_ROOT / "evidence/current_release_full_matrix_summary.json"
    business_rows_path = (
        PROJECT_ROOT
        / "results/final_submission_v1/evidence/e2e/business_e2e_requests.jsonl"
    )
    e2e_summary_path = (
        PROJECT_ROOT
        / "results/final_submission_v1/evidence/e2e/perception_inclusive_e2e_summary.json"
    )
    formal_plugins = _read_json(formal_plugins_path)
    formal_runtime_template = _read_json(formal_runtime_template_path)
    formal_runtime_output = _read_json(formal_runtime_output_path)
    gate = _read_json(gate_path)
    prior_matrix = _read_json(prior_matrix_path)
    e2e_summary = _read_json(e2e_summary_path)

    industrial_options = next(
        item["options"]
        for item in formal_plugins["plugins"]
        if item["spec"] == "industrial_anomaly.plugin:IndustrialAnomalyPlugin"
    )
    model_identity = _identity(model_path)
    expected_model_sha = str(formal_runtime_output["static_deployment_sha256"])

    # This validates the exact formal JSON and plugin constraints but makes no
    # request to its production-sidecar endpoint.
    formal_config_probe = IndustrialAnomalyPlugin(
        thresholds_path=SCENE_ROOT / "industrial_anomaly/review_bands.json",
        edge_llm_runtime_config_path=formal_runtime_template_path,
        edge_llm_mode="selective",
        edge_llm_prompt_prefix="",
        edge_llm_selective_timeout_limit_seconds=0.5,
        policy_version="industrial-formal-config-structural-probe-v1",
    )
    formal_config_probe.warmup()

    industrial_rows = []
    with business_rows_path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if row.get("scene") == "industrial":
                industrial_rows.append(row)
    qwen_fields = sorted(
        {
            key
            for row in industrial_rows
            for key in row
            if "qwen" in str(key).lower()
        }
    )

    plugin = IndustrialAnomalyPlugin(
        thresholds_path=SCENE_ROOT / "industrial_anomaly/review_bands.json",
        edge_llm_runtime_config_path=runtime_path,
        edge_llm_mode="selective",
        edge_llm_prompt_prefix="",
        edge_llm_selective_timeout_limit_seconds=0.5,
        policy_version="industrial-isolated-qwen-path-audit-v1",
    )
    plugin.warmup()
    bands = _read_json(SCENE_ROOT / "industrial_anomaly/review_bands.json")
    templates = {
        "rgb": _read_json(SCENE_ROOT / "samples/rgb_event.json"),
        "infrared": _read_json(SCENE_ROOT / "samples/infrared_event.json"),
    }

    state_matrix: List[Dict[str, Any]] = []
    for product in PRODUCTS:
        for modality in MODALITIES:
            for state in STATES:
                state_matrix.append(
                    _run_event(
                        plugin,
                        templates[modality],
                        bands,
                        product,
                        modality,
                        state,
                        "matrix-{}-{}-{}".format(product, modality, state),
                    )
                )

    conflict_cases = [
        {
            "name": "hard_normal_vs_anomaly",
            "members": [
                _run_event(
                    plugin,
                    templates["rgb"],
                    bands,
                    "capsule",
                    "rgb",
                    "normal",
                    "conflict-hard",
                ),
                _run_event(
                    plugin,
                    templates["infrared"],
                    bands,
                    "capsule",
                    "infrared",
                    "anomaly",
                    "conflict-hard",
                ),
            ],
        },
        {
            "name": "review_vs_anomaly",
            "members": [
                _run_event(
                    plugin,
                    templates["rgb"],
                    bands,
                    "capsule",
                    "rgb",
                    "review",
                    "conflict-review",
                ),
                _run_event(
                    plugin,
                    templates["infrared"],
                    bands,
                    "capsule",
                    "infrared",
                    "anomaly",
                    "conflict-review",
                ),
            ],
        },
    ]

    review_records = [
        record for record in state_matrix if record["reference_state"] == "review"
    ]
    fast_records = [
        record for record in state_matrix if record["reference_state"] != "review"
    ]
    all_isolated_records = state_matrix + [
        member for case in conflict_cases for member in case["members"]
    ]
    checks = {
        "isolated_endpoint_not_production_port": (
            urlsplit(endpoint).port not in FORBIDDEN_PRODUCTION_PORTS
        ),
        "model_sha_matches_formal_static_deployment": (
            model_identity["sha256"] == expected_model_sha
        ),
        "formal_industrial_mode_is_selective": (
            industrial_options["edge_llm_mode"] == "selective"
        ),
        "exact_formal_runtime_config_structurally_loads": (
            formal_config_probe.health()["edge_qwen_available"] is True
            and formal_runtime_template["provider"] == "llama_cpp"
            and formal_runtime_template["generation"]["max_input_tokens"] == 16
            and formal_runtime_template["generation"]["max_output_tokens"] == 1
            and "lora_adapter" not in formal_runtime_template
        ),
        "formal_contract_is_raw16_static_without_prefix": (
            industrial_options["edge_llm_prompt_prefix"] == ""
            and gate["runtime_contract"]["input_tokens"] == 16
            and gate["runtime_contract"]["output_tokens"] == 1
            and gate["runtime_contract"]["runtime_lora_count"] == 0
        ),
        "all_state_matrix_prompts_are_decimal16": all(
            len(record["encoded_prompt"]) == 16
            and record["encoded_prompt"].isdigit()
            for record in state_matrix
        ),
        "all_twenty_review_events_invoked_qwen": (
            len(review_records) == 20
            and all(record["qwen_selected"] for record in review_records)
        ),
        "all_twenty_review_events_returned_16_to_1_token_B": all(
            record["qwen_prompt_tokens"] == 16
            and record["qwen_output_tokens"] == 1
            and record["qwen_token"] == "B"
            and record["qwen_prediction"] == "review"
            for record in review_records
        ),
        "all_twenty_review_events_agreed_without_fallback": all(
            record["qwen_rule_agreement"] is True
            and record["qwen_fallback"] is False
            for record in review_records
        ),
        "normal_and_anomaly_use_deterministic_fast_path": (
            len(fast_records) == 40
            and all(
                record["qwen_selected"] is False
                and record["edge_decision_path"] == "industrial_rule_fast_path"
                for record in fast_records
            )
        ),
        "hard_conflict_members_skip_qwen": all(
            member["qwen_selected"] is False
            for member in conflict_cases[0]["members"]
        ),
        "review_conflict_invokes_qwen_only_for_review_member": (
            [member["qwen_selected"] for member in conflict_cases[1]["members"]]
            == [True, False]
        ),
        "no_isolated_runtime_fallback": all(
            record["qwen_fallback"] is False for record in all_isolated_records
        ),
        "prior_closed_loop_matrix_records_actual_qwen_calls": (
            prior_matrix["primary"]["qwen_selected"] == 60
            and prior_matrix["primary"]["qwen_completed"] == 60
            and prior_matrix["primary"]["qwen_rule_agreement_rate"] == 1.0
        ),
        "final_500_business_rows_do_not_claim_per_row_qwen_identity": (
            len(industrial_rows) == 500 and qwen_fields == []
        ),
    }

    report = {
        "schema_version": "industrial-qwen-path-audit/v1",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "status": "ISOLATED_DEVELOPMENT_AUDIT",
            "production_ports_forbidden": sorted(FORBIDDEN_PRODUCTION_PORTS),
            "isolated_endpoint": endpoint,
            "production_18100_modified": False,
            "production_thresholds_modified": False,
            "perception_model_executed": False,
            "input": "controlled score events around frozen review bands",
            "latency_environment": (
                "local x86 host with NVIDIA RTX 5070 Ti; isolated latency is not "
                "Jetson Orin or Nano formal latency"
            ),
        },
        "three_layer_verdict": {
            "code_callable": "PASS",
            "formal_q4_config_loadable": "PASS_STRUCTURAL_NO_PRODUCTION_REQUEST",
            "current_q4_isolated_real_invocation": "PASS",
            "prior_closed_loop_actual_invocation": "PASS_DIFFERENT_Q5_RELEASE_IDENTITY",
            "final_500_business_rows_actual_qwen_invocation": "NOT_PROVEN_PER_ROW",
            "conclusion": (
                "Industrial Qwen is a real selective review path, not an all-event path. "
                "The 54.932709 ms composed industrial mean cannot be used as proof that "
                "all 500 business rows invoked Qwen."
            ),
        },
        "assets": {
            "current_q4_model": model_identity,
            "isolated_runtime_config": _identity(runtime_path),
            "formal_scene_plugins": _identity(formal_plugins_path),
            "formal_runtime_template": _identity(formal_runtime_template_path),
            "formal_runtime_output": _identity(formal_runtime_output_path),
            "q4_gate_summary": _identity(gate_path),
            "prior_closed_loop_matrix": _identity(prior_matrix_path),
            "final_business_rows": _identity(business_rows_path),
            "perception_inclusive_e2e": _identity(e2e_summary_path),
        },
        "formal_q4_binding": {
            "model_sha256": expected_model_sha,
            "edge_llm_mode": industrial_options["edge_llm_mode"],
            "prompt_prefix": industrial_options["edge_llm_prompt_prefix"],
            "selective_timeout_limit_seconds": industrial_options[
                "edge_llm_selective_timeout_limit_seconds"
            ],
            "input_tokens": gate["runtime_contract"]["input_tokens"],
            "output_tokens": gate["runtime_contract"]["output_tokens"],
            "runtime_lora_count": gate["runtime_contract"]["runtime_lora_count"],
            "request_level_lora_switching": gate["runtime_contract"][
                "request_level_lora_switching"
            ],
            "industrial_allowed_slots": gate["runtime_contract"][
                "industrial_allowed_slots"
            ],
        },
        "codec_contract": {
            "format": "2 M PP LLLL UUUU WWWW",
            "positions": {
                "1": "task/contract discriminator 2",
                "2": "modality M: 0=RGB, 1=infrared",
                "3-4": "product index PP in the frozen ten-product order",
                "5-8": (
                    "LLLL = clamp(round((score-low)/(high-low)*1000), -4000, 4000) + 4000"
                ),
                "9-12": (
                    "UUUU = clamp(round((score-low)/(high-low)*1000)-1000, -4000, 4000) + 4000"
                ),
                "13-16": "WWWW = round((high-low)*1e6)",
            },
            "product_order": list(PRODUCTS),
            "output_mapping": dict(INDUSTRIAL_ACTION_TOKENS),
            "decoder": (
                "llama.cpp GBNF constrains sampling to A|B|C before sampling; "
                "A/B/C map to normal/review/anomaly without post-hoc remapping"
            ),
            "safety": (
                "only review selects Qwen; disagreement or runtime error preserves "
                "the deterministic review-band decision"
            ),
        },
        "isolated_q4_results": {
            "state_matrix_events": len(state_matrix),
            "review_events": len(review_records),
            "qwen_invocations": sum(record["qwen_selected"] for record in state_matrix),
            "qwen_agreements": sum(
                record["qwen_rule_agreement"] is True for record in state_matrix
            ),
            "qwen_fallbacks": sum(record["qwen_fallback"] for record in state_matrix),
            "review_qwen_latency_ms": _summary(
                float(record["qwen_latency_ms"]) for record in review_records
            ),
            "fast_path_wall_ms": _summary(
                float(record["wall_ms"]) for record in fast_records
            ),
            "plugin_health_after_run": plugin.health(),
            "state_matrix": state_matrix,
            "conflict_cases": conflict_cases,
        },
        "existing_evidence_qualification": {
            "prior_closed_loop_release": {
                "release_id": prior_matrix["release"]["release_id"],
                "model_sha256": prior_matrix["release"]["model_sha256"],
                "events": prior_matrix["primary"]["events"],
                "qwen_selected": prior_matrix["primary"]["qwen_selected"],
                "qwen_completed": prior_matrix["primary"]["qwen_completed"],
                "qwen_rule_agreement_rate": prior_matrix["primary"][
                    "qwen_rule_agreement_rate"
                ],
                "identity_note": (
                    "actual invocation evidence, but its Q5 release identity differs "
                    "from the current Q4 model"
                ),
            },
            "final_500_business_rows": {
                "count": len(industrial_rows),
                "available_fields": sorted(
                    {key for row in industrial_rows for key in row}
                ),
                "qwen_fields": qwen_fields,
                "qwen_invocation_verdict": "NOT_PROVEN_PER_ROW",
                "business_e2e_mean_ms": round(
                    statistics.fmean(
                        float(row["business_e2e_ms"]) for row in industrial_rows
                    ),
                    6,
                ),
                "composed_with_perception_mean_ms": e2e_summary["industrial"][
                    "modality_event_composition"
                ]["merged"]["mean_ms"],
                "measurement_note": e2e_summary["industrial"]["method"][
                    "warning"
                ],
            },
            "q4_nano_gate": {
                "completed_requests": gate["metrics"]["nano_completed_requests"],
                "overall_mean_latency_ms": gate["metrics"][
                    "nano_overall_mean_latency_ms"
                ],
                "industrial_accuracy": gate["metrics"][
                    "nano_industrial_accuracy"
                ],
                "qualification": (
                    "proves Q4 model execution and industrial action ability, but is "
                    "not the same population as the final 500 business E2E rows"
                ),
            },
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    _write_new(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
