"""用途：构建多区域决策冲突图，并对全部关联区域执行联合参数协调。"""

import copy
import itertools
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from traffic_system.decision_utils import safe_float
from traffic_system.safety_filter import validate_and_filter_decision


ACTIVE_DECISIONS = {"variable_speed_limit", "ramp_metering", "regional_coordination", "reroute"}
DEFAULT_LIMITS = {
    "max_vsl_delta_mph": 10.0,
    "max_ramp_delta_veh_per_hour": 180.0,
    "max_combined_diversion_ratio": 0.50,
}
RISK_WEIGHT = {"low": 1.0, "medium": 2.0, "high": 3.0, "severe": 4.0, "unknown": 1.0}


def managed_nodes(event: Dict[str, Any]) -> Set[int]:
    return {
        int(node)
        for node in event.get("managed_node_ids", [])
        if not isinstance(node, bool)
    }


def action_targets(action: Dict[str, Any]) -> Set[int]:
    values = list(action.get("target_nodes", [])) + list(action.get("gateway_nodes", []))
    output = set()
    for value in values:
        try:
            if not isinstance(value, bool):
                output.add(int(value))
        except (TypeError, ValueError):
            continue
    return output


def affected_nodes(decision: Dict[str, Any]) -> Set[int]:
    nodes = set()
    for value in decision.get("affected_nodes", []):
        try:
            if not isinstance(value, bool):
                nodes.add(int(value))
        except (TypeError, ValueError):
            continue
    for action in decision.get("actions", []):
        if isinstance(action, dict):
            nodes |= action_targets(action)
    return nodes


def pair_boundary_nodes(
    event: Dict[str, Any],
    other_event: Dict[str, Any],
    neighbor_map: Dict[int, List[int]],
) -> Set[int]:
    own = managed_nodes(event)
    other = managed_nodes(other_event)
    return {
        node
        for node in own
        if any(neighbor in other for neighbor in neighbor_map.get(node, []))
    }


def decision_scope_nodes(
    event: Dict[str, Any],
    other_event: Dict[str, Any],
    decision: Dict[str, Any],
    neighbor_map: Dict[int, List[int]],
) -> Set[int]:
    nodes = affected_nodes(decision)
    if str(decision.get("decision")) in {"regional_coordination", "reroute"}:
        nodes |= pair_boundary_nodes(event, other_event, neighbor_map)
    return nodes


def nodes_within_hops(
    starts: Set[int],
    targets: Set[int],
    neighbor_map: Dict[int, List[int]],
    max_hops: int,
) -> Set[int]:
    if not starts or not targets:
        return set()
    if starts & targets:
        return starts & targets
    frontier = set(starts)
    visited = set(starts)
    for _ in range(max(0, int(max_hops))):
        next_frontier = set()
        for node in frontier:
            for neighbor in neighbor_map.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        hits = next_frontier & targets
        if hits:
            return set(starts) | hits
        frontier = next_frontier
        if not frontier:
            break
    return set()


def coupled_nodes(
    left_event: Dict[str, Any],
    right_event: Dict[str, Any],
    left_decision: Dict[str, Any],
    right_decision: Dict[str, Any],
    neighbor_map: Dict[int, List[int]],
    boundary_hops: int,
) -> Set[int]:
    left_nodes = decision_scope_nodes(left_event, right_event, left_decision, neighbor_map)
    right_nodes = decision_scope_nodes(right_event, left_event, right_decision, neighbor_map)
    return nodes_within_hops(left_nodes, right_nodes, neighbor_map, boundary_hops)


def actions_of(decision: Dict[str, Any], action_type: str) -> List[Dict[str, Any]]:
    return [
        action
        for action in decision.get("actions", [])
        if isinstance(action, dict) and action.get("type") == action_type
    ]


def numeric_values(decision: Dict[str, Any], action_type: str, field: str) -> List[float]:
    return [safe_float(action.get(field), 0.0) for action in actions_of(decision, action_type)]


def version_key(value: Any) -> Tuple[int, ...]:
    parts = [int(part) for part in re.findall(r"\d+", str(value))]
    return tuple(parts or [0])


def decision_weight(event: Dict[str, Any], decision: Dict[str, Any]) -> float:
    risk = str(decision.get("global_risk_level", "unknown"))
    confidence = max(0.05, safe_float(decision.get("confidence"), 0.5))
    summary = event.get("region_summary", {})
    score = safe_float(summary.get("region_risk_score"), 0.0) if isinstance(summary, dict) else 0.0
    return RISK_WEIGHT.get(risk, 1.0) * confidence + score


def detect_pair_conflicts(
    left_index: int,
    right_index: int,
    left: Dict[str, Any],
    right: Dict[str, Any],
    neighbor_map: Dict[int, List[int]],
    boundary_hops: int,
    limits: Dict[str, float],
) -> List[Dict[str, Any]]:
    left_decision = left["decision"]
    right_decision = right["decision"]
    if str(left_decision.get("decision")) not in ACTIVE_DECISIONS:
        return []
    if str(right_decision.get("decision")) not in ACTIVE_DECISIONS:
        return []
    coupled = coupled_nodes(
        left["event"], right["event"], left_decision, right_decision, neighbor_map, boundary_hops
    )
    if not coupled:
        return []

    common = {
        "left_index": left_index,
        "right_index": right_index,
        "left_edge": str(left["event"].get("edge_id", left_index)),
        "right_edge": str(right["event"].get("edge_id", right_index)),
        "nodes": sorted(coupled),
    }
    conflicts = []
    left_version = left_decision.get("policy_version")
    right_version = right_decision.get("policy_version")
    if left_version is not None and right_version is not None and str(left_version) != str(right_version):
        conflicts.append(
            {
                **common,
                "type": "policy_version_mismatch",
                "details": {"left": str(left_version), "right": str(right_version)},
            }
        )

    left_vsl = numeric_values(left_decision, "variable_speed_limit", "target_speed_mph")
    right_vsl = numeric_values(right_decision, "variable_speed_limit", "target_speed_mph")
    if left_vsl and right_vsl:
        delta = abs(min(left_vsl) - min(right_vsl))
        if delta > limits["max_vsl_delta_mph"]:
            conflicts.append(
                {
                    **common,
                    "type": "boundary_vsl_discontinuity",
                    "details": {"left_mph": min(left_vsl), "right_mph": min(right_vsl), "delta_mph": delta},
                }
            )

    left_ramp = numeric_values(left_decision, "ramp_metering", "metering_rate_veh_per_hour")
    right_ramp = numeric_values(right_decision, "ramp_metering", "metering_rate_veh_per_hour")
    if left_ramp and right_ramp:
        delta = abs(min(left_ramp) - min(right_ramp))
        if delta > limits["max_ramp_delta_veh_per_hour"]:
            conflicts.append(
                {
                    **common,
                    "type": "boundary_ramp_rate_discontinuity",
                    "details": {"left_rate": min(left_ramp), "right_rate": min(right_ramp), "delta": delta},
                }
            )

    left_diversion = sum(numeric_values(left_decision, "reroute", "diversion_ratio"))
    right_diversion = sum(numeric_values(right_decision, "reroute", "diversion_ratio"))
    if left_diversion > 0 and right_diversion > 0:
        total = left_diversion + right_diversion
        if total > limits["max_combined_diversion_ratio"]:
            conflicts.append(
                {
                    **common,
                    "type": "alternate_corridor_overload",
                    "details": {"left_ratio": left_diversion, "right_ratio": right_diversion, "total": total},
                }
            )
    return conflicts


def eligibility_stats(
    records: Sequence[Dict[str, Any]],
    neighbor_map: Dict[int, List[int]],
    boundary_hops: int,
) -> Dict[str, int]:
    total = active = coupled = 0
    for left, right in itertools.combinations(records, 2):
        total += 1
        left_decision = left["decision"]
        right_decision = right["decision"]
        if (
            str(left_decision.get("decision")) in ACTIVE_DECISIONS
            and str(right_decision.get("decision")) in ACTIVE_DECISIONS
        ):
            active += 1
            if coupled_nodes(
                left["event"], right["event"], left_decision, right_decision, neighbor_map, boundary_hops
            ):
                coupled += 1
    return {"total_pairs": total, "active_pairs": active, "coupled_active_pairs": coupled}


def detect_conflicts(
    records: Sequence[Dict[str, Any]],
    neighbor_map: Dict[int, List[int]],
    boundary_hops: int = 1,
    limits: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    effective_limits = {**DEFAULT_LIMITS, **(limits or {})}
    conflicts = []
    for left_index, right_index in itertools.combinations(range(len(records)), 2):
        conflicts.extend(
            detect_pair_conflicts(
                left_index,
                right_index,
                records[left_index],
                records[right_index],
                neighbor_map,
                boundary_hops,
                effective_limits,
            )
        )
    return conflicts


def set_numeric_action(decision: Dict[str, Any], action_type: str, field: str, value: float) -> None:
    for action in actions_of(decision, action_type):
        action[field] = value
        action["reason"] = "Cloud global coordinator synchronized this parameter."


def coordinate_conflict(
    records: List[Dict[str, Any]],
    conflict: Dict[str, Any],
    limits: Dict[str, float],
) -> Dict[str, Any]:
    left = records[int(conflict["left_index"])]
    right = records[int(conflict["right_index"])]
    left_decision = left["decision"]
    right_decision = right["decision"]
    kind = str(conflict["type"])
    before = copy.deepcopy(conflict.get("details", {}))

    if kind == "policy_version_mismatch":
        version = max(
            (left_decision.get("policy_version", "0"), right_decision.get("policy_version", "0")),
            key=version_key,
        )
        left_decision["policy_version"] = str(version)
        right_decision["policy_version"] = str(version)
    elif kind == "boundary_vsl_discontinuity":
        left_value = min(numeric_values(left_decision, "variable_speed_limit", "target_speed_mph"))
        right_value = min(numeric_values(right_decision, "variable_speed_limit", "target_speed_mph"))
        left_weight = decision_weight(left["event"], left_decision)
        right_weight = decision_weight(right["event"], right_decision)
        value = round((left_value * left_weight + right_value * right_weight) / (left_weight + right_weight) / 5.0) * 5
        set_numeric_action(left_decision, "variable_speed_limit", "target_speed_mph", int(value))
        set_numeric_action(right_decision, "variable_speed_limit", "target_speed_mph", int(value))
    elif kind == "boundary_ramp_rate_discontinuity":
        left_value = min(numeric_values(left_decision, "ramp_metering", "metering_rate_veh_per_hour"))
        right_value = min(numeric_values(right_decision, "ramp_metering", "metering_rate_veh_per_hour"))
        left_weight = decision_weight(left["event"], left_decision)
        right_weight = decision_weight(right["event"], right_decision)
        value = round((left_value * left_weight + right_value * right_weight) / (left_weight + right_weight) / 60.0) * 60
        set_numeric_action(left_decision, "ramp_metering", "metering_rate_veh_per_hour", int(value))
        set_numeric_action(right_decision, "ramp_metering", "metering_rate_veh_per_hour", int(value))
    elif kind == "alternate_corridor_overload":
        left_value = sum(numeric_values(left_decision, "reroute", "diversion_ratio"))
        right_value = sum(numeric_values(right_decision, "reroute", "diversion_ratio"))
        total = left_value + right_value
        scale = limits["max_combined_diversion_ratio"] / total
        set_numeric_action(left_decision, "reroute", "diversion_ratio", round(left_value * scale, 3))
        set_numeric_action(right_decision, "reroute", "diversion_ratio", round(right_value * scale, 3))

    for record in (left, right):
        decision = record["decision"]
        decision["decision_source"] = str(decision.get("decision_source", "edge")) + "_global_coordinated"
        decision["reason"] = "global conflict graph coordination"
    return {"type": kind, "left_edge": conflict["left_edge"], "right_edge": conflict["right_edge"], "before": before}


def filter_preserving_metadata(decision: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {
        key: decision[key]
        for key in ("policy_version", "decision_id", "issued_at_ms", "valid_until_ms")
        if key in decision
    }
    filtered = validate_and_filter_decision(decision)
    filtered.update(metadata)
    return filtered


def coordinate_globally(
    records: Sequence[Dict[str, Any]],
    neighbor_map: Dict[int, List[int]],
    boundary_hops: int = 1,
    limits: Optional[Dict[str, float]] = None,
    max_rounds: int = 8,
) -> Dict[str, Any]:
    effective_limits = {**DEFAULT_LIMITS, **(limits or {})}
    coordinated = copy.deepcopy(list(records))
    initial = detect_conflicts(coordinated, neighbor_map, boundary_hops, effective_limits)
    changes = []
    rounds = 0
    for round_index in range(max_rounds):
        current = detect_conflicts(coordinated, neighbor_map, boundary_hops, effective_limits)
        if not current:
            break
        rounds = round_index + 1
        for conflict in current:
            changes.append(coordinate_conflict(coordinated, conflict, effective_limits))
        for record in coordinated:
            record["decision"] = filter_preserving_metadata(record["decision"])
    residual = detect_conflicts(coordinated, neighbor_map, boundary_hops, effective_limits)
    return {
        "records": coordinated,
        "initial_conflicts": initial,
        "residual_conflicts": residual,
        "initial_conflict_count": len(initial),
        "residual_conflict_count": len(residual),
        "resolution_success_rate": round((len(initial) - len(residual)) / len(initial), 6) if initial else 1.0,
        "rounds": rounds,
        "changes": changes,
        "globally_consistent": not residual,
    }
