"""用途：评估自然运行冲突率，并用多类型压力集验证全局协调能力。"""

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from traffic_system.conflict_coordinator import (
    coordinate_globally,
    detect_conflicts,
    eligibility_stats,
    pair_boundary_nodes,
)
from traffic_system.decision_utils import (
    build_decision_from_student_class,
    load_json,
    read_jsonl,
    rule_teacher_decision,
    save_json,
)
from traffic_system.edge_student import load_student_model, predict_student
from traffic_system.graph_partition import build_undirected_neighbor_map
from traffic_system.infer_joint_risk_astgcn import load_adjacency, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate natural and injected multi-edge decision conflicts.")
    parser.add_argument("--labels", default="datasets/freeway_teacher_labels_qwen9b_safe_joint_metis4.jsonl")
    parser.add_argument("--model_json", default="models/edge_student_freeway_joint_metis4.json")
    parser.add_argument("--output_json", default="results/decision/conflict_consistency_freeway_joint_metis4.json")
    parser.add_argument("--decision_source", default="student", choices=["student", "teacher", "rule"])
    parser.add_argument("--config", default="configurations/PEMS08_astgcn.conf")
    parser.add_argument("--boundary_hops", type=int, default=1)
    parser.add_argument("--max_groups", type=int, default=0, help="0 means all natural groups.")
    parser.add_argument("--stress_groups", type=int, default=12)
    return parser.parse_args()


def load_records(labels_path: Path) -> List[Dict[str, Any]]:
    records = []
    for row in read_jsonl(labels_path):
        event_path = row.get("event_path")
        if event_path:
            records.append({**row, "event": load_json(Path(str(event_path)))})
    if not records:
        raise ValueError("No records with event_path found.")
    return records


def group_by_sample(records: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        sample_id = int(record["event"].get("sample_id", -1))
        groups.setdefault(sample_id, []).append(record)
    return [sorted(rows, key=lambda row: int(row["event"].get("partition_id", 0))) for _, rows in sorted(groups.items())]


def build_decision(record: Dict[str, Any], model: Dict[str, Any], source: str) -> Dict[str, Any]:
    event = record["event"]
    if source == "teacher":
        decision = record.get("teacher_decision")
        return copy.deepcopy(decision) if isinstance(decision, dict) else rule_teacher_decision(event)
    if source == "rule":
        return rule_teacher_decision(event)
    decision_class, confidence, _ = predict_student(event, model)
    return build_decision_from_student_class(event, decision_class, confidence, "edge_student")


def coordinator_records(
    group: Sequence[Dict[str, Any]], model: Dict[str, Any], source: str
) -> List[Dict[str, Any]]:
    return [
        {"event": copy.deepcopy(record["event"]), "decision": build_decision(record, model, source)}
        for record in group
    ]


def conflict_pairs(conflicts: Sequence[Dict[str, Any]]) -> set:
    return {
        (int(conflict["left_index"]), int(conflict["right_index"]))
        for conflict in conflicts
    }


def summarize_natural(
    groups: Sequence[Sequence[Dict[str, Any]]],
    model: Dict[str, Any],
    source: str,
    neighbor_map: Dict[int, List[int]],
    boundary_hops: int,
) -> Dict[str, Any]:
    pair_totals = Counter()
    type_counts = Counter()
    initial_conflict_pairs = 0
    residual_conflict_pairs = 0
    initial_conflicts = 0
    residual_conflicts = 0
    resolved_groups = 0
    examples = []

    for group in groups:
        records = coordinator_records(group, model, source)
        stats = eligibility_stats(records, neighbor_map, boundary_hops)
        pair_totals.update(stats)
        coordinated = coordinate_globally(records, neighbor_map, boundary_hops)
        initial = coordinated["initial_conflicts"]
        residual = coordinated["residual_conflicts"]
        initial_conflict_pairs += len(conflict_pairs(initial))
        residual_conflict_pairs += len(conflict_pairs(residual))
        initial_conflicts += len(initial)
        residual_conflicts += len(residual)
        type_counts.update(str(conflict["type"]) for conflict in initial)
        if initial and not residual:
            resolved_groups += 1
        for conflict in initial:
            if len(examples) < 12:
                examples.append(
                    {
                        "sample_id": group[0]["event"].get("sample_id"),
                        "left_edge": conflict["left_edge"],
                        "right_edge": conflict["right_edge"],
                        "type": conflict["type"],
                        "nodes": conflict["nodes"],
                        "details": conflict["details"],
                        "resolved_after_global_coordination": not any(
                            item["left_index"] == conflict["left_index"]
                            and item["right_index"] == conflict["right_index"]
                            and item["type"] == conflict["type"]
                            for item in residual
                        ),
                    }
                )

    coupled = int(pair_totals["coupled_active_pairs"])
    total = int(pair_totals["total_pairs"])
    success = (initial_conflicts - residual_conflicts) / initial_conflicts if initial_conflicts else 1.0
    return {
        "num_groups": len(groups),
        "pair_denominators": dict(pair_totals),
        "initial_conflict_pair_count": initial_conflict_pairs,
        "raw_conflict_rate_among_coupled_active_pairs": round(initial_conflict_pairs / coupled, 6) if coupled else 0.0,
        "raw_conflict_rate_among_all_pairs": round(initial_conflict_pairs / total, 6) if total else 0.0,
        "initial_constraint_violation_count": initial_conflicts,
        "initial_conflict_types": dict(sorted(type_counts.items())),
        "residual_conflict_pair_count": residual_conflict_pairs,
        "residual_constraint_violation_count": residual_conflicts,
        "post_coordination_conflict_rate": round(residual_conflict_pairs / coupled, 6) if coupled else 0.0,
        "constraint_resolution_success_rate": round(success, 6),
        "groups_with_conflicts_fully_resolved": resolved_groups,
        "examples": examples,
    }


def ensure_capability(event: Dict[str, Any], key: str, node: int) -> None:
    capabilities = event.setdefault("control_capabilities", {})
    nodes = capabilities.setdefault(key, [])
    if node not in nodes:
        nodes.append(node)


def replace_actions(decision: Dict[str, Any], actions: List[Dict[str, Any]], decision_class: str) -> None:
    decision["decision"] = decision_class
    decision["global_risk_level"] = "severe"
    decision["actions"] = actions
    decision["confidence"] = 0.9
    decision["safe"] = True


def vsl_action(node: int, speed: int) -> Dict[str, Any]:
    return {
        "type": "variable_speed_limit",
        "target_nodes": [node],
        "strategy": "speed_harmonization",
        "target_speed_mph": speed,
        "duration_seconds": 300,
        "reason": "injected boundary VSL test",
    }


def ramp_action(node: int, rate: int) -> Dict[str, Any]:
    return {
        "type": "ramp_metering",
        "target_nodes": [node],
        "strategy": "regulate_ramp_inflow",
        "metering_rate_veh_per_hour": rate,
        "duration_seconds": 300,
        "reason": "injected ramp-rate test",
    }


def reroute_action(event: Dict[str, Any], node: int, ratio: float) -> Dict[str, Any]:
    return {
        "type": "reroute",
        "gateway_nodes": [node],
        "strategy": "divert_incoming_flow",
        "avoid_region": str(event.get("region_id")),
        "alternate_corridor": "shared_alternate_corridor",
        "diversion_ratio": ratio,
        "reason": "injected alternate-capacity test",
    }


def adjacent_pair(
    group: Sequence[Dict[str, Any]], neighbor_map: Dict[int, List[int]]
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], int, int]]:
    for left_index in range(len(group)):
        for right_index in range(left_index + 1, len(group)):
            left_event = group[left_index]["event"]
            right_event = group[right_index]["event"]
            left_boundary = pair_boundary_nodes(left_event, right_event, neighbor_map)
            right_boundary = pair_boundary_nodes(right_event, left_event, neighbor_map)
            if left_boundary and right_boundary:
                return group[left_index], group[right_index], min(left_boundary), min(right_boundary)
    return None


def base_stress_records(
    left_record: Dict[str, Any], right_record: Dict[str, Any]
) -> List[Dict[str, Any]]:
    records = []
    for record in (left_record, right_record):
        event = copy.deepcopy(record["event"])
        decision = build_decision_from_student_class(event, "regional_coordination", 0.9, "stress_edge")
        records.append({"event": event, "decision": decision})
    return records


def inject_stress_cases(
    group: Sequence[Dict[str, Any]], neighbor_map: Dict[int, List[int]]
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    selected = adjacent_pair(group, neighbor_map)
    if selected is None:
        return []
    left_record, right_record, left_node, right_node = selected
    cases = []

    records = base_stress_records(left_record, right_record)
    replace_actions(records[0]["decision"], [vsl_action(left_node, 30)], "regional_coordination")
    replace_actions(records[1]["decision"], [vsl_action(right_node, 65)], "regional_coordination")
    cases.append(("boundary_vsl_discontinuity", records))

    records = base_stress_records(left_record, right_record)
    for record, node in zip(records, (left_node, right_node)):
        ensure_capability(record["event"], "ramp_meter_nodes", node)
        record["decision"]["control_capabilities"] = copy.deepcopy(record["event"]["control_capabilities"])
    replace_actions(records[0]["decision"], [ramp_action(left_node, 300)], "regional_coordination")
    replace_actions(records[1]["decision"], [ramp_action(right_node, 900)], "regional_coordination")
    cases.append(("boundary_ramp_rate_discontinuity", records))

    records = base_stress_records(left_record, right_record)
    for record, node in zip(records, (left_node, right_node)):
        ensure_capability(record["event"], "reroute_gateway_nodes", node)
        record["decision"]["control_capabilities"] = copy.deepcopy(record["event"]["control_capabilities"])
    replace_actions(records[0]["decision"], [reroute_action(records[0]["event"], left_node, 0.35)], "reroute")
    replace_actions(records[1]["decision"], [reroute_action(records[1]["event"], right_node, 0.35)], "reroute")
    cases.append(("alternate_corridor_overload", records))

    records = base_stress_records(left_record, right_record)
    records[0]["decision"]["policy_version"] = "1.1.0"
    records[1]["decision"]["policy_version"] = "1.3.0"
    cases.append(("policy_version_mismatch", records))
    return cases


def evaluate_stress(
    groups: Sequence[Sequence[Dict[str, Any]]],
    neighbor_map: Dict[int, List[int]],
    boundary_hops: int,
) -> Dict[str, Any]:
    rows = []
    for group in groups:
        for injected_type, records in inject_stress_cases(group, neighbor_map):
            initial = detect_conflicts(records, neighbor_map, boundary_hops)
            coordinated = coordinate_globally(records, neighbor_map, boundary_hops)
            residual = coordinated["residual_conflicts"]
            initial_types = {str(item["type"]) for item in initial}
            residual_types = {str(item["type"]) for item in residual}
            all_safe = all(bool(record["decision"].get("safe")) for record in coordinated["records"])
            success = injected_type in initial_types and injected_type not in residual_types and all_safe
            rows.append(
                {
                    "sample_id": group[0]["event"].get("sample_id"),
                    "injected_type": injected_type,
                    "detected_types": sorted(initial_types),
                    "residual_types": sorted(residual_types),
                    "all_decisions_safe": all_safe,
                    "global_rounds": coordinated["rounds"],
                    "success": success,
                }
            )
    type_summary = {}
    for conflict_type in sorted({row["injected_type"] for row in rows}):
        selected = [row for row in rows if row["injected_type"] == conflict_type]
        type_summary[conflict_type] = {
            "cases": len(selected),
            "detected": sum(conflict_type in row["detected_types"] for row in selected),
            "resolved": sum(row["success"] for row in selected),
            "success_rate": round(sum(row["success"] for row in selected) / len(selected), 6),
        }
    success_count = sum(row["success"] for row in rows)
    return {
        "construction": "conflicts injected into real PEMS08 event pairs that share a road-graph boundary",
        "num_cases": len(rows),
        "conflict_types": type_summary,
        "success_count": success_count,
        "resolution_success_rate": round(success_count / len(rows), 6) if rows else 0.0,
        "meets_resolution_success_ge_90_percent": bool(rows and success_count / len(rows) >= 0.90),
        "cases": rows,
    }


def main() -> None:
    args = parse_args()
    records = load_records(Path(args.labels))
    groups = group_by_sample(records)
    if args.max_groups > 0:
        groups = groups[: args.max_groups]
    model = load_student_model(Path(args.model_json)) if args.decision_source == "student" else {}
    config = load_config(args.config)
    adjacency, _ = load_adjacency(config)
    neighbor_map = build_undirected_neighbor_map(adjacency, int(config["Data"]["num_of_vertices"]))

    natural = summarize_natural(groups, model, args.decision_source, neighbor_map, args.boundary_hops)
    stress = evaluate_stress(groups[: max(1, args.stress_groups)], neighbor_map, args.boundary_hops)
    result = {
        "task": "global_multi_edge_conflict_consistency_evaluation",
        "decision_source": args.decision_source,
        "conflict_coupling": "road_graph_boundary_and_control_scope",
        "boundary_hops": args.boundary_hops,
        "natural_operation": natural,
        "injected_stress_test": stress,
        "requirement_checks": {
            "natural_conflict_rate_le_5_percent": natural["raw_conflict_rate_among_coupled_active_pairs"] <= 0.05,
            "stress_resolution_success_ge_90_percent": stress["meets_resolution_success_ge_90_percent"],
        },
        "limitations": [
            "PEMS08 has no physical actuator inventory; the stress suite uses graph-boundary proxy actuators.",
            "Constraint satisfaction is validated here; traffic utility after coordination is evaluated separately in SUMO.",
        ],
    }
    save_json(result, Path(args.output_json))
    print("natural conflict rate:", natural["raw_conflict_rate_among_coupled_active_pairs"])
    print("natural residual rate:", natural["post_coordination_conflict_rate"])
    print("stress cases:", stress["num_cases"])
    print("stress resolution success:", stress["resolution_success_rate"])
    print("saved:", args.output_json)


if __name__ == "__main__":
    main()
