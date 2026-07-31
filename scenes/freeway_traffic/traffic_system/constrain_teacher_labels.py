"""用途：对大模型 Teacher 标签执行高速公路动作约束和安全修正。"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from traffic_system.decision_utils import read_jsonl, write_jsonl


DECISION_PRIORITY = {
    "no_action": 0,
    "congestion_warning": 1,
    "variable_speed_limit": 2,
    "ramp_metering": 3,
    "regional_coordination": 4,
    "reroute": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply auditable freeway safety constraints to Qwen teacher labels.")
    parser.add_argument("--input_jsonl", default="datasets/freeway_teacher_labels_qwen9b_joint_metis4.jsonl")
    parser.add_argument("--output_jsonl", default="datasets/freeway_teacher_labels_qwen9b_safe_joint_metis4.jsonl")
    parser.add_argument("--report_json", default="results/decision/freeway_teacher_safety_constraint_report.json")
    return parser.parse_args()


def constrain_row(row: Dict[str, Any]) -> Dict[str, Any]:
    teacher = row.get("teacher_decision", {})
    safety = row.get("rule_decision", {})
    if not isinstance(teacher, dict) or not isinstance(safety, dict):
        raise ValueError("Each row must contain teacher_decision and rule_decision objects.")
    teacher_name = str(teacher.get("decision", "no_action"))
    safety_name = str(safety.get("decision", "no_action"))
    requires_override = (
        safety_name in {"ramp_metering", "regional_coordination", "reroute"}
        and DECISION_PRIORITY.get(teacher_name, -1) < DECISION_PRIORITY.get(safety_name, -1)
    )

    result = dict(row)
    if requires_override:
        constrained = dict(safety)
        constrained["decision_source"] = "qwen9b_safety_constrained"
        constrained["reason"] = "Safety constraint raised {} to {}. {}".format(
            teacher_name,
            safety_name,
            safety.get("reason", ""),
        )[:200]
        result["teacher_decision"] = constrained
        result["safety_constraint"] = {
            "applied": True,
            "original_teacher_decision": teacher_name,
            "constrained_decision": safety_name,
            "reason": "Traffic-engineering risk floor for severe freeway states.",
        }
    else:
        constrained = dict(teacher)
        constrained["decision_source"] = "qwen9b_safety_constrained"
        result["teacher_decision"] = constrained
        result["safety_constraint"] = {
            "applied": False,
            "original_teacher_decision": teacher_name,
            "constrained_decision": teacher_name,
            "reason": "Qwen decision already satisfies the safety floor.",
        }
    result["teacher_source"] = "qwen9b_safety_constrained"
    return result


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input_jsonl))
    constrained_rows: List[Dict[str, Any]] = [constrain_row(row) for row in rows]
    write_jsonl(constrained_rows, Path(args.output_jsonl))
    decision_counts = Counter(
        str(row["teacher_decision"].get("decision", "unknown"))
        for row in constrained_rows
    )
    overrides = [row for row in constrained_rows if row["safety_constraint"]["applied"]]
    transition_counts = Counter(
        "{}->{}".format(
            row["safety_constraint"]["original_teacher_decision"],
            row["safety_constraint"]["constrained_decision"],
        )
        for row in overrides
    )
    report = {
        "task": "qwen9b_freeway_teacher_safety_constraints",
        "num_records": len(constrained_rows),
        "num_overrides": len(overrides),
        "override_rate": round(len(overrides) / max(1, len(constrained_rows)), 6),
        "decision_counts": dict(sorted(decision_counts.items())),
        "transition_counts": dict(sorted(transition_counts.items())),
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("labels:", args.output_jsonl)
    print("report:", args.report_json)


if __name__ == "__main__":
    main()
