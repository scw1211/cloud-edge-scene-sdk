"""用途：校验决策动作能力、限制控制参数并过滤不安全或重复指令。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ALLOWED_DECISIONS = {
    "no_action",
    "congestion_warning",
    "variable_speed_limit",
    "ramp_metering",
    "regional_coordination",
    "reroute",
    "fallback_to_edge_policy",
    "emergency_fallback",
}

ALLOWED_ACTION_TYPES = {
    "traffic_advisory",
    "variable_speed_limit",
    "ramp_metering",
    "reroute",
    "regional_coordination",
    "fallback",
}

RISK_LEVELS = {"low", "medium", "high", "severe", "unknown"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and filter freeway traffic decisions.")
    parser.add_argument("--input_json", required=True, help="Input decision JSON file.")
    parser.add_argument("--output_json", default="results/decision/final_decision_check.json")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("Input JSON file not found: {}".format(path))
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError("Decision JSON must be an object.")
    return data


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)


def as_int_list(values: Any) -> List[int]:
    if not isinstance(values, list):
        return []
    result: List[int] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed not in result:
            result.append(parsed)
    return result


def as_str_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    result: List[str] = []
    for value in values:
        text = str(value)
        if text not in result:
            result.append(text)
    return result


def clamp_float(value: Any, low: float, high: float, default: float) -> Tuple[float, bool]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default, True
    clamped = max(low, min(high, number))
    return clamped, clamped != number


def clamp_int(value: Any, low: int, high: int, default: int) -> Tuple[int, bool]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default, True
    clamped = max(low, min(high, number))
    return clamped, clamped != number


def capability_nodes(capabilities: Dict[str, Any], key: str) -> List[int]:
    return as_int_list(capabilities.get(key))


def restrict_targets(
    targets: Sequence[int],
    capabilities: Dict[str, Any],
    capability_key: str,
) -> Tuple[List[int], bool]:
    allowed = capability_nodes(capabilities, capability_key)
    if not allowed:
        return list(targets), False
    allowed_set = set(allowed)
    filtered = [node for node in targets if node in allowed_set]
    return filtered, filtered != list(targets)


def sanitize_action(
    action: Any,
    capabilities: Dict[str, Any],
    affected_nodes: Sequence[int],
    affected_regions: Sequence[str],
) -> Tuple[Optional[Dict[str, Any]], bool, List[str]]:
    if not isinstance(action, dict):
        return None, True, ["Dropped non-object action."]

    action_type = str(action.get("type", ""))
    if action_type not in ALLOWED_ACTION_TYPES:
        return None, True, ["Dropped unsupported freeway action type: {}.".format(action_type)]

    sanitized = dict(action)
    changed = False
    reasons: List[str] = []
    targets = as_int_list(action.get("target_nodes", affected_nodes))

    if action_type == "traffic_advisory":
        sanitized["target_nodes"] = targets
        level = str(action.get("warning_level", "medium"))
        if level not in RISK_LEVELS:
            level = "medium"
            changed = True
            reasons.append("Normalized warning_level.")
        sanitized["warning_level"] = level
        sanitized["strategy"] = "issue_congestion_warning"

    elif action_type == "variable_speed_limit":
        targets, restricted = restrict_targets(targets, capabilities, "variable_speed_limit_nodes")
        if restricted:
            changed = True
            reasons.append("Removed nodes without variable-speed-limit capability.")
        if not targets:
            return None, True, reasons + ["Dropped VSL action without a capable target node."]
        speed, speed_changed = clamp_int(action.get("target_speed_mph", 45), 25, 65, 45)
        duration, duration_changed = clamp_int(action.get("duration_seconds", 300), 60, 1800, 300)
        changed = changed or speed_changed or duration_changed
        if speed_changed:
            reasons.append("Clamped target_speed_mph to [25, 65].")
        if duration_changed:
            reasons.append("Clamped VSL duration_seconds to [60, 1800].")
        sanitized.update(
            {
                "target_nodes": targets,
                "strategy": "speed_harmonization",
                "target_speed_mph": speed,
                "duration_seconds": duration,
            }
        )

    elif action_type == "ramp_metering":
        targets, restricted = restrict_targets(targets, capabilities, "ramp_meter_nodes")
        if restricted:
            changed = True
            reasons.append("Removed nodes without ramp-meter capability.")
        if not targets:
            return None, True, reasons + ["Dropped ramp-meter action without a capable target node."]
        rate, rate_changed = clamp_int(action.get("metering_rate_veh_per_hour", 480), 240, 900, 480)
        duration, duration_changed = clamp_int(action.get("duration_seconds", 300), 60, 1800, 300)
        changed = changed or rate_changed or duration_changed
        if rate_changed:
            reasons.append("Clamped metering rate to [240, 900] veh/h.")
        if duration_changed:
            reasons.append("Clamped ramp-meter duration_seconds to [60, 1800].")
        sanitized.update(
            {
                "target_nodes": targets,
                "strategy": "regulate_on_ramp_inflow",
                "metering_rate_veh_per_hour": rate,
                "duration_seconds": duration,
            }
        )

    elif action_type == "reroute":
        gateways = as_int_list(action.get("gateway_nodes", targets))
        gateways, restricted = restrict_targets(gateways, capabilities, "reroute_gateway_nodes")
        if restricted:
            changed = True
            reasons.append("Removed nodes without reroute-gateway capability.")
        if not gateways:
            return None, True, reasons + ["Dropped reroute action without a capable gateway."]
        ratio, ratio_changed = clamp_float(action.get("diversion_ratio", 0.2), 0.05, 0.4, 0.2)
        changed = changed or ratio_changed
        if ratio_changed:
            reasons.append("Clamped diversion_ratio to [0.05, 0.40].")
        avoid_region = str(
            action.get("avoid_region")
            or (affected_regions[0] if affected_regions else "unknown_region")
        )
        sanitized.update(
            {
                "gateway_nodes": gateways,
                "strategy": "divert_incoming_flow",
                "avoid_region": avoid_region,
                "diversion_ratio": round(ratio, 3),
                "alternate_corridor": str(action.get("alternate_corridor", "cloud_selected_corridor")),
            }
        )

    elif action_type == "regional_coordination":
        sanitized["target_regions"] = as_str_list(
            action.get("target_regions", affected_regions)
        )
        sanitized["strategy"] = "coordinate_boundary_demand"

    elif action_type == "fallback":
        sanitized["target_nodes"] = targets
        sanitized["strategy"] = "local_safe_policy"

    sanitized["reason"] = str(action.get("reason", "freeway control action"))[:160]
    return sanitized, changed, reasons


def consolidate_numeric_actions(actions: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool, List[str]]:
    result: List[Dict[str, Any]] = []
    vsl_by_node: Dict[int, Dict[str, Any]] = {}
    ramp_by_node: Dict[int, Dict[str, Any]] = {}
    changed = False

    for action in actions:
        action_type = action.get("type")
        if action_type == "variable_speed_limit":
            for node in as_int_list(action.get("target_nodes")):
                candidate = dict(action)
                candidate["target_nodes"] = [node]
                current = vsl_by_node.get(node)
                if current is None or candidate["target_speed_mph"] < current["target_speed_mph"]:
                    vsl_by_node[node] = candidate
                if current is not None:
                    changed = True
        elif action_type == "ramp_metering":
            for node in as_int_list(action.get("target_nodes")):
                candidate = dict(action)
                candidate["target_nodes"] = [node]
                current = ramp_by_node.get(node)
                if current is None or candidate["metering_rate_veh_per_hour"] < current["metering_rate_veh_per_hour"]:
                    ramp_by_node[node] = candidate
                if current is not None:
                    changed = True
        else:
            result.append(dict(action))

    grouped_vsl: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for node in sorted(vsl_by_node):
        action = vsl_by_node[node]
        key = (
            action.get("target_speed_mph"),
            action.get("duration_seconds"),
            action.get("strategy"),
            action.get("reason"),
        )
        if key not in grouped_vsl:
            grouped_vsl[key] = dict(action)
            grouped_vsl[key]["target_nodes"] = []
        grouped_vsl[key]["target_nodes"].append(node)
    grouped_ramp: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for node in sorted(ramp_by_node):
        action = ramp_by_node[node]
        key = (
            action.get("metering_rate_veh_per_hour"),
            action.get("duration_seconds"),
            action.get("strategy"),
            action.get("reason"),
        )
        if key not in grouped_ramp:
            grouped_ramp[key] = dict(action)
            grouped_ramp[key]["target_nodes"] = []
        grouped_ramp[key]["target_nodes"].append(node)
    result.extend(grouped_vsl.values())
    result.extend(grouped_ramp.values())
    reasons = ["Consolidated duplicate actuator commands conservatively."] if changed else []
    return result, changed, reasons


def infer_global_risk_level(decision: str) -> str:
    if decision in {"reroute", "regional_coordination", "ramp_metering"}:
        return "severe"
    if decision == "variable_speed_limit":
        return "high"
    if decision == "congestion_warning":
        return "medium"
    return "low"


def emergency_fallback(reason: str) -> Dict[str, Any]:
    return {
        "scene": "freeway_traffic_management",
        "decision_source": "emergency_fallback",
        "decision": "emergency_fallback",
        "global_risk_level": "unknown",
        "affected_regions": [],
        "affected_nodes": [],
        "control_capabilities": {},
        "actions": [
            {
                "type": "fallback",
                "target_nodes": [],
                "strategy": "local_safe_policy",
                "reason": reason,
            }
        ],
        "confidence": 0.0,
        "reason": reason,
        "safe": True,
        "filter_applied": True,
        "filter_reason": reason,
    }


def validate_and_filter_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(decision, dict) or "decision" not in decision:
        return emergency_fallback("Decision object or required decision field is missing.")

    filter_applied = False
    filter_reasons: List[str] = []
    raw_decision = str(decision.get("decision"))
    if raw_decision not in ALLOWED_DECISIONS:
        final_decision = "fallback_to_edge_policy"
        filter_applied = True
        filter_reasons.append("Unsupported decision replaced with edge safe policy.")
    else:
        final_decision = raw_decision

    affected_nodes = as_int_list(decision.get("affected_nodes"))
    affected_regions = as_str_list(decision.get("affected_regions"))
    capabilities = decision.get("control_capabilities", {})
    if not isinstance(capabilities, dict):
        capabilities = {}
        filter_applied = True
        filter_reasons.append("Invalid control_capabilities replaced with empty mapping.")

    confidence, confidence_changed = clamp_float(decision.get("confidence", 0.5), 0.0, 1.0, 0.5)
    filter_applied = filter_applied or confidence_changed
    if confidence_changed:
        filter_reasons.append("Clamped confidence to [0, 1].")

    raw_actions = decision.get("actions", [])
    if not isinstance(raw_actions, list):
        raw_actions = []
        filter_applied = True
        filter_reasons.append("actions must be a list.")

    actions: List[Dict[str, Any]] = []
    for action in raw_actions:
        sanitized, changed, reasons = sanitize_action(
            action,
            capabilities,
            affected_nodes,
            affected_regions,
        )
        filter_applied = filter_applied or changed
        filter_reasons.extend(reasons)
        if sanitized is not None:
            actions.append(sanitized)

    actions, consolidated, reasons = consolidate_numeric_actions(actions)
    filter_applied = filter_applied or consolidated
    filter_reasons.extend(reasons)

    if final_decision == "fallback_to_edge_policy" and not actions:
        actions = [
            {
                "type": "fallback",
                "target_nodes": affected_nodes,
                "strategy": "local_safe_policy",
                "reason": "Invalid command; keep the current safe freeway policy.",
            }
        ]

    risk_level = str(decision.get("global_risk_level") or infer_global_risk_level(final_decision))
    if risk_level not in RISK_LEVELS:
        risk_level = infer_global_risk_level(final_decision)
        filter_applied = True
        filter_reasons.append("Normalized global_risk_level.")

    filtered = {
        "scene": str(decision.get("scene", "freeway_traffic_management")),
        "decision_source": str(decision.get("decision_source", "cloud")),
        "edge_id": str(decision.get("edge_id", "unknown_edge")),
        "region_id": str(decision.get("region_id", "unknown_region")),
        "decision": final_decision,
        "global_risk_level": risk_level,
        "affected_regions": affected_regions,
        "affected_nodes": affected_nodes,
        "control_capabilities": capabilities,
        "actions": actions,
        "confidence": round(confidence, 4),
        "reason": str(decision.get("reason", "freeway traffic decision"))[:200],
        "safe": True,
        "filter_applied": filter_applied,
        "filter_reason": "; ".join(dict.fromkeys(filter_reasons)),
    }
    for key in ("policy_version", "decision_id", "issued_at_ms", "valid_until_ms"):
        if key in decision:
            filtered[key] = decision[key]
    return filtered


def main() -> None:
    args = parse_args()
    decision = load_json(Path(args.input_json))
    filtered = validate_and_filter_decision(decision)
    save_json(filtered, Path(args.output_json))
    print("decision:", filtered["decision"])
    print("safe:", filtered["safe"])
    print("output:", args.output_json)


if __name__ == "__main__":
    main()
