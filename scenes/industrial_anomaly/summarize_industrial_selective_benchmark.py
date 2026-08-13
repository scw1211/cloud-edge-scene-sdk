#!/usr/bin/env python3
"""Fail-closed summary for the 180-event industrial selective-routing run."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    source = Path(args.benchmark).resolve()
    output_json = Path(args.output_json).resolve()
    output_csv = Path(args.output_csv).resolve()
    for output in (output_json, output_csv):
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
    report = json.loads(source.read_text(encoding="utf-8"))
    primary = report["profiles"]["normal_async"]
    events: List[Dict[str, Any]] = [
        event
        for group in primary["records"]["groups"]
        for event in group["events"]
    ]
    gates = {
        "event_count_180": len(events) == 180 == int(primary["event_count"]),
        "accuracy_1": float(primary["local_accuracy"]) == 1.0,
        "compact_180_of_180": all(
            event.get("response_detail") == "compact" for event in events
        ),
        "selective_contract_180_of_180": all(
            bool(event["qwen_selected"]) == (event["reference"] == "review")
            for event in events
        ),
        "qwen_selected_60": int(primary["qwen_selected_count"]) == 60,
        "qwen_completion_1": float(primary["qwen_completion_rate"]) == 1.0,
        "qwen_agreement_1": float(primary["qwen_rule_agreement_rate"]) == 1.0,
        "authoritative_final_1": (
            float(primary["authoritative_final_rate"]) == 1.0
        ),
        "e2e_mean_under_200ms": (
            float(primary["client_input_to_provisional_ms"]["mean"]) < 200.0
        ),
    }
    summary = {
        "schema_version": "industrial-selective-routing-summary/v1",
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
        },
        "primary_metric": {
            "name": "input_to_local_executable_provisional",
            **dict(primary["client_input_to_provisional_ms"]),
            "unit": "ms",
        },
        "local_accuracy": primary["local_accuracy"],
        "authoritative_final_rate": primary["authoritative_final_rate"],
        "qwen": {
            "selected_count": primary["qwen_selected_count"],
            "selection_rate": primary["qwen_selection_rate"],
            "completion_rate": primary["qwen_completion_rate"],
            "rule_agreement_rate": primary["qwen_rule_agreement_rate"],
            "selected_only_latency_ms": primary["qwen_latency_ms"],
            "amortized_per_event_latency_ms": primary[
                "qwen_amortized_latency_ms"
            ],
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = [
        ("平均端到端时延", summary["primary_metric"]["mean"], "ms", 180),
        ("P95端到端时延", summary["primary_metric"]["p95"], "ms", 180),
        ("最大端到端时延", summary["primary_metric"]["max"], "ms", 180),
        ("本地决策准确率", summary["local_accuracy"], "ratio", 180),
        ("Qwen选择率", summary["qwen"]["selection_rate"], "ratio", 180),
        (
            "Qwen实际调用平均时延",
            summary["qwen"]["selected_only_latency_ms"]["mean"],
            "ms",
            summary["qwen"]["selected_count"],
        ),
    ]
    with output_csv.open("x", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(("指标", "数值", "单位", "样本数"))
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
