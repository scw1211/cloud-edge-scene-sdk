"""用途：在云端超时、断网或响应无效时执行边缘本地自治决策。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from traffic_system.safety_filter import validate_and_filter_decision
from traffic_system.decision_utils import rule_teacher_decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeway edge autonomy controller.")
    parser.add_argument(
        "--edge_event",
        default="datasets/freeway_events_joint_metis4/freeway_test_sample_0800_edge_node_0.json",
    )
    parser.add_argument("--cloud_decision", default="results/decision/cloud_decision_check.json")
    parser.add_argument("--output_json", default="results/decision/final_decision_check.json")
    parser.add_argument("--local_output_json", default="results/decision/local_decision_check.json")
    parser.add_argument(
        "--cloud_mode",
        default="normal",
        choices=["normal", "timeout", "down", "invalid"],
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("JSON file not found: {}".format(path))
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError("JSON file must contain an object: {}".format(path))
    return data


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)


def build_local_decision(edge_event: Dict[str, Any], cloud_mode: str) -> Dict[str, Any]:
    decision = rule_teacher_decision(edge_event, decision_source="local_edge_policy")
    decision["cloud_mode"] = cloud_mode
    decision["autonomy_triggered"] = True
    decision["reason"] = "Cloud unavailable; edge freeway policy executed locally. " + str(
        decision.get("reason", "")
    )
    return validate_and_filter_decision(decision)


def build_invalid_cloud_decision(edge_event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scene": "freeway_traffic_management",
        "decision_source": "cloud",
        "edge_id": str(edge_event.get("edge_id", "unknown_edge")),
        "region_id": str(edge_event.get("region_id", "unknown_region")),
        "decision": "force_all_freeway_actuators",
        "global_risk_level": "severe",
        "affected_regions": [str(edge_event.get("region_id", "unknown_region"))],
        "affected_nodes": [],
        "control_capabilities": edge_event.get("control_capabilities", {}),
        "actions": [],
        "confidence": 1.5,
        "reason": "Intentionally invalid response used by the network-fault test.",
    }


def is_cloud_decision_usable(decision: Dict[str, Any]) -> bool:
    return bool(decision.get("safe")) and decision.get("decision") not in {
        "fallback_to_edge_policy",
        "emergency_fallback",
    }


def run_autonomy_core(
    edge_event: Dict[str, Any],
    cloud_mode: str = "normal",
    cloud_decision: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        "cloud_mode": cloud_mode,
        "used_cloud_decision": False,
        "autonomy_triggered": False,
        "local_decision": None,
    }

    candidate = cloud_decision
    if cloud_mode == "invalid":
        candidate = build_invalid_cloud_decision(edge_event)
    if cloud_mode == "normal" and isinstance(candidate, dict):
        filtered = validate_and_filter_decision(candidate)
        if is_cloud_decision_usable(filtered):
            metadata["used_cloud_decision"] = True
            return filtered, metadata

    local_decision = build_local_decision(edge_event, cloud_mode)
    metadata["autonomy_triggered"] = True
    metadata["local_decision"] = local_decision
    return local_decision, metadata


def maybe_load_cloud_decision(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def main() -> None:
    args = parse_args()
    edge_event = load_json(Path(args.edge_event))
    cloud_decision = (
        maybe_load_cloud_decision(Path(args.cloud_decision))
        if args.cloud_mode == "normal"
        else None
    )
    final_decision, metadata = run_autonomy_core(
        edge_event=edge_event,
        cloud_mode=args.cloud_mode,
        cloud_decision=cloud_decision,
    )
    if metadata["local_decision"] is not None:
        save_json(metadata["local_decision"], Path(args.local_output_json))
    save_json(final_decision, Path(args.output_json))
    print("cloud_mode:", metadata["cloud_mode"])
    print("used_cloud_decision:", metadata["used_cloud_decision"])
    print("autonomy_triggered:", metadata["autonomy_triggered"])
    print("decision:", final_decision["decision"])
    print("output:", args.output_json)


if __name__ == "__main__":
    main()
