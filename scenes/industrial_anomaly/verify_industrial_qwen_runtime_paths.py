#!/usr/bin/env python3
"""Verify the isolated industrial Qwen path audit and its source bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MODEL_SHA256 = (
    "828f839873c7505005101544d8febb7408bc0443c153c4ba97ec4b3603445526"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "results/industrial_qwen_path_audit_v1/summary.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary_path = args.summary.resolve()
    summary = _read_json(summary_path)

    checks: Dict[str, bool] = {}
    checks["schema"] = summary.get("schema_version") == "industrial-qwen-path-audit/v1"
    checks["audit_all_checks"] = summary.get("all_checks_passed") is True and all(
        summary.get("checks", {}).values()
    )
    scope = summary["scope"]
    checks["isolated_scope"] = (
        scope["status"] == "ISOLATED_DEVELOPMENT_AUDIT"
        and scope["isolated_endpoint"] == "http://127.0.0.1:19190"
        and scope["production_18100_modified"] is False
        and scope["production_thresholds_modified"] is False
        and scope["perception_model_executed"] is False
    )
    verdict = summary["three_layer_verdict"]
    checks["three_layer_qualification"] = (
        verdict["code_callable"] == "PASS"
        and verdict["formal_q4_config_loadable"]
        == "PASS_STRUCTURAL_NO_PRODUCTION_REQUEST"
        and verdict["current_q4_isolated_real_invocation"] == "PASS"
        and verdict["prior_closed_loop_actual_invocation"]
        == "PASS_DIFFERENT_Q5_RELEASE_IDENTITY"
        and verdict["final_500_business_rows_actual_qwen_invocation"]
        == "NOT_PROVEN_PER_ROW"
    )
    binding = summary["formal_q4_binding"]
    checks["formal_q4_contract"] = (
        binding["model_sha256"] == EXPECTED_MODEL_SHA256
        and binding["edge_llm_mode"] == "selective"
        and binding["prompt_prefix"] == ""
        and binding["input_tokens"] == 16
        and binding["output_tokens"] == 1
        and binding["runtime_lora_count"] == 0
        and binding["request_level_lora_switching"] is False
        and binding["industrial_allowed_slots"] == ["A", "B", "C"]
    )
    results = summary["isolated_q4_results"]
    checks["matrix_population"] = (
        results["state_matrix_events"] == 60
        and results["review_events"] == 20
        and len(results["state_matrix"]) == 60
    )
    checks["real_qwen_invocations"] = (
        results["qwen_invocations"] == 20
        and results["qwen_agreements"] == 20
        and results["qwen_fallbacks"] == 0
        and results["review_qwen_latency_ms"]["count"] == 20
    )
    review_records = [
        record
        for record in results["state_matrix"]
        if record["reference_state"] == "review"
    ]
    fast_records = [
        record
        for record in results["state_matrix"]
        if record["reference_state"] != "review"
    ]
    checks["review_16_to_1"] = all(
        record["qwen_selected"] is True
        and record["qwen_prompt_tokens"] == 16
        and record["qwen_output_tokens"] == 1
        and record["qwen_token"] == "B"
        and record["qwen_rule_agreement"] is True
        and record["qwen_fallback"] is False
        and record["decoding_constraint"]["post_hoc_remapping"] is False
        for record in review_records
    )
    checks["fast_path"] = len(fast_records) == 40 and all(
        record["qwen_selected"] is False
        and record["edge_decision_path"] == "industrial_rule_fast_path"
        for record in fast_records
    )
    final_rows = summary["existing_evidence_qualification"]["final_500_business_rows"]
    checks["final_500_qualified"] = (
        final_rows["count"] == 500
        and final_rows["qwen_fields"] == []
        and final_rows["qwen_invocation_verdict"] == "NOT_PROVEN_PER_ROW"
    )
    prior = summary["existing_evidence_qualification"]["prior_closed_loop_release"]
    checks["prior_actual_qwen_qualified"] = (
        prior["qwen_selected"] == 60
        and prior["qwen_completed"] == 60
        and prior["qwen_rule_agreement_rate"] == 1.0
        and prior["model_sha256"] != EXPECTED_MODEL_SHA256
    )

    source_hashes_match = True
    for record in summary["assets"].values():
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            source_hashes_match = False
            break
        if _sha256(path) != record["sha256"]:
            source_hashes_match = False
            break
    checks["source_hashes_match"] = source_hashes_match
    checks["current_model_identity"] = (
        summary["assets"]["current_q4_model"]["sha256"]
        == EXPECTED_MODEL_SHA256
    )

    result = {
        "schema_version": "industrial-qwen-path-audit-verification/v1",
        "summary": {
            "path": str(summary_path),
            "bytes": summary_path.stat().st_size,
            "sha256": _sha256(summary_path),
        },
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    if args.output is not None:
        _write_new(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
