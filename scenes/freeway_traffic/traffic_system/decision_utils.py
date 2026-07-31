"""用途：定义交通决策协议、事件特征、规则决策及 JSON 读写工具。"""

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from traffic_system.safety_filter import validate_and_filter_decision


DECISION_CLASSES = [
    "no_action",
    "congestion_warning",
    "variable_speed_limit",
    "ramp_metering",
    "regional_coordination",
    "reroute",
]

ACTION_TOKEN_TO_DECISION = {
    "A": "no_action",
    "B": "congestion_warning",
    "C": "variable_speed_limit",
    "D": "ramp_metering",
    "E": "regional_coordination",
    "F": "reroute",
}

RISK_LEVELS = ["low", "medium", "high", "severe"]
RISK_TO_VALUE = {level: idx for idx, level in enumerate(RISK_LEVELS)}
UPLOAD_TO_VALUE = {
    "summary": 0,
    "feature": 1,
    "sequence": 2,
    "regional_context": 3,
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON must contain an object: {}".format(path))
    return data


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("JSONL row {} must be an object.".format(line_no))
            rows.append(row)
    return rows


def collect_event_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() == ".jsonl":
            rows = read_jsonl(input_path)
            paths = []
            for row in rows:
                event_path = row.get("event_path")
                if event_path:
                    paths.append(Path(str(event_path)))
            if paths:
                return paths
        return [input_path]
    if input_path.is_dir():
        paths = sorted(path for path in input_path.glob("*.json") if path.is_file())
        if not paths:
            raise FileNotFoundError("No event JSON files found in: {}".format(input_path))
        return paths
    raise FileNotFoundError("Input path not found: {}".format(input_path))


def event_identifier(event: Dict[str, Any]) -> str:
    edge_id = str(event.get("edge_id", "edge"))
    sample_id = str(event.get("sample_id", "sample"))
    risk_profile = str(event.get("risk_profile", event.get("upload_level", "profile")))
    return "{}_sample_{}_{}".format(edge_id, sample_id, risk_profile)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or isinstance(value, bool):
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def get_region_summary(event: Dict[str, Any]) -> Dict[str, Any]:
    summary = event.get("region_summary", {})
    return summary if isinstance(summary, dict) else {}


def get_top_nodes(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes = event.get("top_k_risk_nodes", [])
    if not isinstance(nodes, list):
        return []
    return [node for node in nodes if isinstance(node, dict)]


def get_affected_nodes(event: Dict[str, Any], levels: Sequence[str], limit: int = 10) -> List[int]:
    level_set = set(levels)
    nodes = []
    for node in get_top_nodes(event):
        if node.get("risk_level") not in level_set:
            continue
        node_id = safe_int(node.get("node_id"), default=-1)
        if node_id >= 0 and node_id not in nodes:
            nodes.append(node_id)
        if len(nodes) >= limit:
            break
    if nodes:
        return nodes

    for node in get_top_nodes(event)[:limit]:
        node_id = safe_int(node.get("node_id"), default=-1)
        if node_id >= 0 and node_id not in nodes:
            nodes.append(node_id)
    return nodes


def risk_summary(event: Dict[str, Any]) -> Dict[str, Any]:
    summary = get_region_summary(event)
    num_nodes = safe_int(summary.get("num_nodes"), safe_int(event.get("num_managed_nodes"), 1))
    if num_nodes <= 1 and isinstance(event.get("managed_node_ids"), list):
        num_nodes = len(event["managed_node_ids"])
    num_nodes = max(num_nodes, 1)
    counts = summary.get("node_risk_counts", {})
    if not isinstance(counts, dict):
        counts = {}
    severe = safe_int(summary.get("num_severe_nodes"), safe_int(counts.get("severe")))
    high = safe_int(summary.get("num_high_nodes"), safe_int(counts.get("high")))
    medium = safe_int(summary.get("num_medium_nodes"), safe_int(counts.get("medium")))
    low = safe_int(
        summary.get("num_low_nodes"),
        safe_int(counts.get("low"), max(0, num_nodes - severe - high - medium)),
    )
    max_risk = safe_float(
        summary.get("max_risk_score"),
        safe_float(summary.get("region_risk_score")),
    )
    mean_risk = safe_float(
        summary.get("mean_risk_score"),
        safe_float(summary.get("mean_node_risk_score")),
    )
    cluster = bool(summary.get("congestion_cluster_detected", severe >= 2 or severe + high >= 4))
    top_nodes = get_top_nodes(event)
    top_level = "low"
    top_score = max_risk
    if top_nodes:
        top_level = str(top_nodes[0].get("risk_level", "low"))
        top_score = safe_float(top_nodes[0].get("risk_score"), max_risk)
    return {
        "num_nodes": num_nodes,
        "num_low": max(0, low),
        "num_medium": max(0, medium),
        "num_high": max(0, high),
        "num_severe": max(0, severe),
        "max_risk": max_risk,
        "mean_risk": mean_risk,
        "cluster": cluster,
        "top_level": top_level,
        "top_score": top_score,
    }


def infer_global_risk(event: Dict[str, Any]) -> str:
    summary = risk_summary(event)
    if summary["num_severe"] > 0 or summary["top_level"] == "severe":
        return "severe"
    if summary["num_high"] > 0 or summary["top_level"] == "high":
        return "high"
    if summary["num_medium"] > 0 or summary["top_level"] == "medium":
        return "medium"
    return "low"


def control_capabilities(event: Dict[str, Any]) -> Dict[str, Any]:
    capabilities = event.get("control_capabilities", {})
    return capabilities if isinstance(capabilities, dict) else {}


def capable_nodes(event: Dict[str, Any], key: str, candidates: Sequence[int]) -> List[int]:
    allowed = control_capabilities(event).get(key, [])
    if not isinstance(allowed, list) or not allowed:
        return [int(node) for node in candidates]
    allowed_set = {safe_int(node, -1) for node in allowed}
    return [int(node) for node in candidates if int(node) in allowed_set]


def build_warning_action(nodes: Sequence[int], level: str, reason: str) -> Dict[str, Any]:
    return {
        "type": "traffic_advisory",
        "target_nodes": [int(node) for node in nodes],
        "strategy": "issue_congestion_warning",
        "warning_level": level,
        "reason": reason,
    }


def build_vsl_action(nodes: Sequence[int], speed_mph: int, reason: str) -> Dict[str, Any]:
    return {
        "type": "variable_speed_limit",
        "target_nodes": [int(node) for node in nodes],
        "strategy": "speed_harmonization",
        "target_speed_mph": int(speed_mph),
        "duration_seconds": 300,
        "reason": reason,
    }


def build_ramp_meter_action(nodes: Sequence[int], rate: int, reason: str) -> Dict[str, Any]:
    return {
        "type": "ramp_metering",
        "target_nodes": [int(node) for node in nodes],
        "strategy": "regulate_on_ramp_inflow",
        "metering_rate_veh_per_hour": int(rate),
        "duration_seconds": 300,
        "reason": reason,
    }


def build_reroute_action(
    event: Dict[str, Any],
    gateway_nodes: Sequence[int],
    diversion_ratio: float,
    reason: str,
) -> Dict[str, Any]:
    region_id = str(event.get("region_id", "unknown_region"))
    return {
        "type": "reroute",
        "gateway_nodes": [int(node) for node in gateway_nodes],
        "strategy": "divert_incoming_flow",
        "avoid_region": region_id,
        "alternate_corridor": "cloud_selected_corridor",
        "diversion_ratio": round(float(diversion_ratio), 3),
        "reason": reason,
    }


def build_regional_action(event: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "type": "regional_coordination",
        "target_regions": [str(event.get("region_id", "unknown_region"))],
        "strategy": "coordinate_boundary_demand",
        "reason": reason,
    }


def build_safe_decision(
    event: Dict[str, Any],
    decision: str,
    reason: str,
    affected_nodes: Optional[Sequence[int]] = None,
    actions: Optional[List[Dict[str, Any]]] = None,
    risk_level: Optional[str] = None,
    confidence: float = 0.75,
    decision_source: str = "rule_teacher",
) -> Dict[str, Any]:
    region_id = str(event.get("region_id", "unknown_region"))
    if affected_nodes is None:
        affected_nodes = get_affected_nodes(event, ("severe", "high", "medium"))
    raw_decision = {
        "scene": "freeway_traffic_management",
        "decision_source": decision_source,
        "edge_id": str(event.get("edge_id", "unknown_edge")),
        "region_id": region_id,
        "decision": decision,
        "global_risk_level": risk_level or infer_global_risk(event),
        "affected_regions": [region_id],
        "affected_nodes": [int(node) for node in affected_nodes],
        "control_capabilities": control_capabilities(event),
        "actions": actions or [],
        "confidence": confidence,
        "reason": reason,
    }
    return validate_and_filter_decision(raw_decision)


def rule_teacher_decision(event: Dict[str, Any], decision_source: str = "rule_teacher") -> Dict[str, Any]:
    summary = risk_summary(event)

    if not bool(event.get("upload_required", True)) and summary["num_high"] == 0 and summary["num_severe"] == 0:
        return build_safe_decision(
            event,
            "no_action",
            "no congestion risk detected",
            affected_nodes=[],
            actions=[],
            risk_level="low",
            confidence=0.72,
            decision_source=decision_source,
        )

    severe_ratio = summary["num_severe"] / max(1, summary["num_nodes"])
    if summary["num_severe"] >= 8 or severe_ratio >= 0.25 or (summary["cluster"] and summary["max_risk"] >= 0.92):
        nodes = get_affected_nodes(event, ("severe", "high"), limit=10)
        vsl_nodes = capable_nodes(event, "variable_speed_limit_nodes", nodes)
        gateways = capable_nodes(event, "reroute_gateway_nodes", nodes)
        actions = [
            build_warning_action(nodes, "severe", "Severe freeway congestion cluster detected."),
            build_vsl_action(vsl_nodes, 35, "Harmonize upstream speed before the severe cluster."),
            build_reroute_action(event, gateways, 0.30, "Divert demand around the severe cluster."),
        ]
        return build_safe_decision(
            event,
            "reroute",
            "severe congestion detected",
            affected_nodes=nodes,
            actions=actions,
            risk_level="severe",
            confidence=0.88,
            decision_source=decision_source,
        )

    if summary["num_severe"] >= 2 or summary["cluster"] or summary["num_severe"] + summary["num_high"] >= 5:
        nodes = get_affected_nodes(event, ("severe", "high"), limit=10)
        vsl_nodes = capable_nodes(event, "variable_speed_limit_nodes", nodes)
        ramp_nodes = capable_nodes(event, "ramp_meter_nodes", nodes)
        actions = [
            build_warning_action(nodes, "severe", "Regional congestion propagation risk detected."),
            build_vsl_action(vsl_nodes, 40, "Coordinate speed limits across the region."),
            build_regional_action(event, "Coordinate boundary demand with neighboring edge regions."),
        ]
        if ramp_nodes:
            actions.append(build_ramp_meter_action(ramp_nodes, 420, "Regulate on-ramp inflow near the bottleneck."))
        return build_safe_decision(
            event,
            "regional_coordination",
            "regional freeway congestion requires coordinated control",
            affected_nodes=nodes,
            actions=actions,
            risk_level="severe",
            confidence=0.84,
            decision_source=decision_source,
        )

    if summary["num_severe"] > 0:
        nodes = get_affected_nodes(event, ("severe", "high"), limit=8)
        ramp_nodes = capable_nodes(event, "ramp_meter_nodes", nodes)
        if not ramp_nodes:
            ramp_nodes = capable_nodes(
                event,
                "ramp_meter_nodes",
                control_capabilities(event).get("ramp_meter_nodes", []),
            )[:3]
        actions = [
            build_warning_action(nodes, "severe", "Severe bottleneck risk detected."),
            build_ramp_meter_action(ramp_nodes, 480, "Reduce incoming ramp demand near the bottleneck."),
        ]
        return build_safe_decision(
            event,
            "ramp_metering",
            "severe bottleneck risk requires ramp inflow regulation",
            affected_nodes=nodes,
            actions=actions,
            risk_level="severe",
            confidence=0.82,
            decision_source=decision_source,
        )

    if summary["num_high"] > 0:
        nodes = get_affected_nodes(event, ("high",), limit=8)
        vsl_nodes = capable_nodes(event, "variable_speed_limit_nodes", nodes)
        actions = [
            build_warning_action(nodes, "high", "High freeway congestion risk detected."),
            build_vsl_action(vsl_nodes, 50, "Smooth upstream speed before congestion forms."),
        ]
        return build_safe_decision(
            event,
            "variable_speed_limit",
            "high congestion risk requires speed harmonization",
            affected_nodes=nodes,
            actions=actions,
            risk_level="high",
            confidence=0.78,
            decision_source=decision_source,
        )

    if summary["num_medium"] > 0:
        nodes = get_affected_nodes(event, ("medium",), limit=8)
        actions = [build_warning_action(nodes, "medium", "Moderate traffic variation detected.")]
        return build_safe_decision(
            event,
            "congestion_warning",
            "moderate traffic variation requires early warning",
            affected_nodes=nodes,
            actions=actions,
            risk_level="medium",
            confidence=0.68,
            decision_source=decision_source,
        )

    return build_safe_decision(
        event,
        "no_action",
        "no congestion risk detected",
        affected_nodes=[],
        actions=[],
        risk_level="low",
        confidence=0.70,
        decision_source=decision_source,
    )


def compact_event_for_teacher(event: Dict[str, Any]) -> Dict[str, Any]:
    top_nodes = []
    for node in get_top_nodes(event)[:10]:
        features = node.get("features", {})
        if not isinstance(features, dict):
            features = {}
        forecast = node.get("forecast", {})
        if not isinstance(forecast, dict):
            forecast = {}
        flow_mean = safe_float(
            node.get("future_mean"),
            safe_float(features.get("future_mean"), safe_float(forecast.get("flow_mean"))),
        )
        future_max = safe_float(
            node.get("future_max"),
            safe_float(features.get("future_max"), flow_mean),
        )
        future_min = safe_float(
            node.get("future_min"),
            safe_float(features.get("future_min"), flow_mean),
        )
        compact_node = {
            "node_id": safe_int(node.get("node_id")),
            "risk_level": str(node.get("risk_level", "low")),
            "risk_score": round(safe_float(node.get("risk_score")), 4),
            "history_mean": round(
                safe_float(node.get("history_mean"), safe_float(features.get("history_mean"))), 4
            ),
            "history_last": round(
                safe_float(node.get("history_last"), safe_float(features.get("history_last"))), 4
            ),
            "future_mean": round(flow_mean, 4),
            "future_max": round(future_max, 4),
            "future_min": round(future_min, 4),
            "growth_rate": round(
                safe_float(node.get("growth_rate"), safe_float(features.get("growth_rate"))), 4
            ),
            "peak_growth_rate": round(
                safe_float(
                    node.get("peak_growth_rate"),
                    safe_float(features.get("peak_growth_rate")),
                ),
                4,
            ),
            "volatility": round(
                safe_float(features.get("volatility"), safe_float(node.get("volatility"))), 4
            ),
            "forecast": {
                "flow_mean": round(safe_float(forecast.get("flow_mean"), flow_mean), 4),
                "occupancy_mean": round(safe_float(forecast.get("occupancy_mean")), 4),
                "speed_mean": round(safe_float(forecast.get("speed_mean")), 4),
                "speed_min": round(safe_float(forecast.get("speed_min")), 4),
            },
        }
        for sequence_name in ("history_12_steps", "prediction_12_steps"):
            sequence = node.get(sequence_name)
            if isinstance(sequence, list):
                compact_node[sequence_name] = [round(safe_float(value), 4) for value in sequence]
        top_nodes.append(compact_node)
    compact = {
        "scene": event.get("scene", "freeway_traffic_management"),
        "event_id": event.get("event_id"),
        "edge_id": event.get("edge_id"),
        "region_id": event.get("region_id"),
        "sample_split": event.get("sample_split"),
        "sample_id": event.get("sample_id"),
        "upload_required": bool(event.get("upload_required", False)),
        "upload_level": event.get("upload_level"),
        "latency_ms": event.get("latency_ms"),
        "prediction_horizon_minutes": event.get("prediction_horizon_minutes", 60),
        "num_managed_nodes": len(event.get("managed_node_ids", []))
        or safe_int(event.get("num_managed_nodes"), 1),
        "region_summary": get_region_summary(event),
        "top_k_risk_nodes": top_nodes,
        "control_capabilities": control_capabilities(event),
    }
    if isinstance(event.get("neighbor_context"), list):
        compact["neighbor_context"] = event["neighbor_context"]
    return compact


def extract_feature_vector(event: Dict[str, Any]) -> Tuple[List[float], List[str]]:
    summary = risk_summary(event)
    top_nodes = get_top_nodes(event)
    num_nodes = max(summary["num_nodes"], 1)

    risk_scores = [safe_float(node.get("risk_score")) for node in top_nodes]
    growth_rates = [safe_float(node.get("growth_rate")) for node in top_nodes]
    peak_growth_rates = [safe_float(node.get("peak_growth_rate")) for node in top_nodes]
    future_means = [
        safe_float(
            node.get("future_mean"),
            safe_float(node.get("forecast", {}).get("flow_mean"))
            if isinstance(node.get("forecast"), dict)
            else 0.0,
        )
        for node in top_nodes
    ]
    top_levels = [RISK_TO_VALUE.get(str(node.get("risk_level", "low")), 0) for node in top_nodes]

    def mean_or_zero(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def seq_stats(values: Any) -> List[float]:
        if not isinstance(values, list) or not values:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        seq = np.asarray([safe_float(value) for value in values], dtype=np.float64)
        if seq.size == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        return [
            float(np.mean(seq)),
            float(np.max(seq)),
            float(np.min(seq)),
            float(np.std(seq)),
            float(seq[-1] - seq[0]) if seq.size >= 2 else 0.0,
        ]

    def pad_top_nodes(nodes: Sequence[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        padded = list(nodes[:limit])
        while len(padded) < limit:
            padded.append({})
        return padded

    def neighbor_level_stats() -> Tuple[List[str], List[float]]:
        names_out = []
        values_out = []
        contexts = event.get("neighbor_context", [])
        if not isinstance(contexts, list):
            contexts = []
        all_neighbors = []
        for context in contexts:
            if not isinstance(context, dict):
                continue
            neighbors = context.get("neighbors", [])
            if isinstance(neighbors, list):
                all_neighbors.extend(neighbor for neighbor in neighbors if isinstance(neighbor, dict))
        neighbor_scores = [safe_float(neighbor.get("risk_score")) for neighbor in all_neighbors]
        neighbor_levels = [
            RISK_TO_VALUE.get(str(neighbor.get("risk_level", "low")), 0)
            for neighbor in all_neighbors
        ]
        for level in RISK_LEVELS:
            names_out.append("neighbor_{}_ratio".format(level))
            values_out.append(
                float(sum(1 for neighbor in all_neighbors if str(neighbor.get("risk_level", "low")) == level))
                / max(1, len(all_neighbors))
            )
        names_out.extend(
            [
                "neighbor_count",
                "neighbor_mean_risk_score",
                "neighbor_max_risk_score",
                "neighbor_mean_level_id",
            ]
        )
        values_out.extend(
            [
                float(len(all_neighbors)),
                mean_or_zero(neighbor_scores),
                max(neighbor_scores) if neighbor_scores else 0.0,
                mean_or_zero(neighbor_levels),
            ]
        )
        return names_out, values_out

    upload_level = str(event.get("upload_level", "summary"))
    names = [
        "low_ratio",
        "medium_ratio",
        "high_ratio",
        "severe_ratio",
        "mean_risk_score",
        "max_risk_score",
        "cluster_detected",
        "upload_required",
        "upload_level_id",
        "top1_risk_score",
        "top1_risk_level_id",
        "top3_mean_risk_score",
        "top3_mean_growth_rate",
        "top3_mean_peak_growth_rate",
        "top3_mean_future_mean",
        "active_top_node_ratio",
        "prediction_horizon_minutes",
        "num_managed_nodes_log",
    ]
    values = [
        summary["num_low"] / num_nodes,
        summary["num_medium"] / num_nodes,
        summary["num_high"] / num_nodes,
        summary["num_severe"] / num_nodes,
        summary["mean_risk"],
        summary["max_risk"],
        1.0 if summary["cluster"] else 0.0,
        1.0 if bool(event.get("upload_required", False)) else 0.0,
        float(UPLOAD_TO_VALUE.get(upload_level, 0)),
        risk_scores[0] if risk_scores else 0.0,
        float(top_levels[0]) if top_levels else 0.0,
        mean_or_zero(risk_scores[:3]),
        mean_or_zero(growth_rates[:3]),
        mean_or_zero(peak_growth_rates[:3]),
        mean_or_zero(future_means[:3]),
        float(sum(1 for level in top_levels if level >= RISK_TO_VALUE["high"])) / max(len(top_levels), 1),
        safe_float(event.get("prediction_horizon_minutes"), 60.0),
        math.log1p(float(num_nodes)),
    ]

    for rank, node in enumerate(pad_top_nodes(top_nodes, limit=10), start=1):
        features = node.get("features", {})
        if not isinstance(features, dict):
            features = {}
        forecast = node.get("forecast", {})
        if not isinstance(forecast, dict):
            forecast = {}
        history_stats = seq_stats(node.get("history_12_steps", []))
        prediction_stats = seq_stats(node.get("prediction_12_steps", []))
        node_names = [
            "risk_score",
            "risk_level_id",
            "history_mean",
            "history_last",
            "future_mean",
            "future_max",
            "future_min",
            "growth_rate",
            "peak_growth_rate",
            "volatility",
            "hist_seq_mean",
            "hist_seq_max",
            "hist_seq_min",
            "hist_seq_std",
            "hist_seq_delta",
            "pred_seq_mean",
            "pred_seq_max",
            "pred_seq_min",
            "pred_seq_std",
            "pred_seq_delta",
        ]
        names.extend(["top{}_{}".format(rank, name) for name in node_names])
        values.extend(
            [
                safe_float(node.get("risk_score")),
                float(RISK_TO_VALUE.get(str(node.get("risk_level", "low")), 0)),
                safe_float(node.get("history_mean", features.get("history_mean"))),
                safe_float(node.get("history_last", features.get("history_last"))),
                safe_float(node.get("future_mean", features.get("future_mean", forecast.get("flow_mean")))),
                safe_float(node.get("future_max", features.get("future_max", forecast.get("flow_mean")))),
                safe_float(node.get("future_min", features.get("future_min", forecast.get("flow_mean")))),
                safe_float(node.get("growth_rate", features.get("growth_rate"))),
                safe_float(node.get("peak_growth_rate", features.get("peak_growth_rate"))),
                safe_float(node.get("volatility", features.get("volatility"))),
                *history_stats,
                *prediction_stats,
            ]
        )

    neighbor_names, neighbor_values = neighbor_level_stats()
    names.extend(neighbor_names)
    values.extend(neighbor_values)
    return values, names


def labels_from_records(records: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    if not records:
        raise ValueError("No labeled records provided.")

    features = []
    labels = []
    feature_names: List[str] = []
    event_ids = []
    for record in records:
        event = record.get("event")
        if not isinstance(event, dict):
            event_path = record.get("event_path")
            if event_path:
                event = load_json(Path(str(event_path)))
            else:
                raise ValueError("Labeled record must contain event or event_path.")
        decision = record.get("teacher_decision") or record.get("decision")
        if not isinstance(decision, dict):
            raise ValueError("Labeled record must contain teacher_decision.")
        label = str(decision.get("decision", "congestion_warning"))
        if label not in DECISION_CLASSES:
            label = rule_teacher_decision(event).get("decision", "congestion_warning")
        vector, names = extract_feature_vector(event)
        feature_names = names
        features.append(vector)
        labels.append(DECISION_CLASSES.index(label))
        event_ids.append(str(record.get("event_id", event_identifier(event))))
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        feature_names,
        event_ids,
    )


def build_decision_from_student_class(
    event: Dict[str, Any],
    decision_class: str,
    confidence: float,
    decision_source: str = "edge_student",
) -> Dict[str, Any]:
    if decision_class not in DECISION_CLASSES:
        decision_class = "congestion_warning"
    summary = risk_summary(event)
    if decision_class == "reroute":
        nodes = get_affected_nodes(event, ("severe", "high"), limit=10)
        vsl_nodes = capable_nodes(event, "variable_speed_limit_nodes", nodes)
        gateways = capable_nodes(event, "reroute_gateway_nodes", nodes)
        actions = [
            build_warning_action(nodes, "severe", "Student detected a severe freeway congestion cluster."),
            build_vsl_action(vsl_nodes, 35, "Student requests upstream speed harmonization."),
            build_reroute_action(event, gateways, 0.30, "Student recommends demand diversion."),
        ]
        risk_level = "severe"
        reason = "student severe congestion decision"
    elif decision_class == "regional_coordination":
        nodes = get_affected_nodes(event, ("severe", "high"), limit=10)
        vsl_nodes = capable_nodes(event, "variable_speed_limit_nodes", nodes)
        ramp_nodes = capable_nodes(event, "ramp_meter_nodes", nodes)
        actions = [
            build_warning_action(nodes, "severe", "Student detected regional congestion propagation risk."),
            build_vsl_action(vsl_nodes, 40, "Student requests coordinated speed harmonization."),
            build_regional_action(event, "Student requests cross-edge demand coordination."),
        ]
        if ramp_nodes:
            actions.append(build_ramp_meter_action(ramp_nodes, 420, "Student requests coordinated ramp metering."))
        risk_level = "severe" if summary["num_severe"] > 0 else "high"
        reason = "student regional coordination decision"
    elif decision_class == "ramp_metering":
        nodes = get_affected_nodes(event, ("severe", "high"), limit=8)
        ramp_nodes = capable_nodes(event, "ramp_meter_nodes", nodes)
        if not ramp_nodes:
            ramp_nodes = capable_nodes(
                event,
                "ramp_meter_nodes",
                control_capabilities(event).get("ramp_meter_nodes", []),
            )[:3]
        actions = [
            build_warning_action(nodes, "severe", "Student detected a severe bottleneck."),
            build_ramp_meter_action(ramp_nodes, 480, "Student requests ramp inflow regulation."),
        ]
        risk_level = "severe"
        reason = "student ramp-metering decision"
    elif decision_class == "variable_speed_limit":
        nodes = get_affected_nodes(event, ("high", "severe"), limit=8)
        vsl_nodes = capable_nodes(event, "variable_speed_limit_nodes", nodes)
        actions = [
            build_warning_action(nodes, "high", "Student detected high freeway congestion risk."),
            build_vsl_action(vsl_nodes, 50, "Student requests upstream speed harmonization."),
        ]
        risk_level = "high"
        reason = "student variable-speed-limit decision"
    elif decision_class == "congestion_warning":
        nodes = get_affected_nodes(event, ("medium", "high", "severe"), limit=8)
        actions = [build_warning_action(nodes, "medium", "Student recommends early congestion warning.")]
        risk_level = "medium"
        reason = "student congestion-warning decision"
    else:
        nodes = []
        actions = []
        risk_level = "low"
        reason = "student no-action decision"

    return build_safe_decision(
        event=event,
        decision=decision_class,
        reason=reason,
        affected_nodes=nodes,
        actions=actions,
        risk_level=risk_level,
        confidence=float(confidence),
        decision_source=decision_source,
    )


def build_decision_from_action_token(
    event: Dict[str, Any],
    action_token: str,
    confidence: float = 0.85,
    decision_source: str = "edge_qwen_action_token",
) -> Dict[str, Any]:
    token = str(action_token).strip().upper()
    if token not in ACTION_TOKEN_TO_DECISION:
        raise ValueError("Unsupported action token: {!r}".format(action_token))
    return build_decision_from_student_class(
        event=event,
        decision_class=ACTION_TOKEN_TO_DECISION[token],
        confidence=confidence,
        decision_source=decision_source,
    )


def build_decision_from_llm_output(
    event: Dict[str, Any],
    llm_output: Dict[str, Any],
    decision_source: str = "edge_qwen_sft",
) -> Dict[str, Any]:
    """Convert the compact SFT JSON into executable freeway control actions."""
    decision = str(llm_output.get("decision", "congestion_warning"))
    if decision not in DECISION_CLASSES:
        decision = "congestion_warning"

    top_node_ids = {
        safe_int(node.get("node_id"), -1)
        for node in get_top_nodes(event)
    }
    requested_nodes = llm_output.get("affected_nodes", [])
    if not isinstance(requested_nodes, list):
        requested_nodes = []
    nodes = []
    for value in requested_nodes:
        node_id = safe_int(value, -1)
        if node_id >= 0 and node_id in top_node_ids and node_id not in nodes:
            nodes.append(node_id)
    if not nodes and decision != "no_action":
        nodes = get_affected_nodes(event, ("severe", "high", "medium"), limit=10)

    raw_action_names = llm_output.get("actions", [])
    if not isinstance(raw_action_names, list):
        raw_action_names = []
    action_names = {
        str(name).strip().lower().replace("_", " ")
        for name in raw_action_names
    }
    actions: List[Dict[str, Any]] = []
    risk_level = str(llm_output.get("global_risk_level", infer_global_risk(event)))
    if risk_level not in RISK_LEVELS:
        risk_level = infer_global_risk(event)
    reason = str(llm_output.get("reason", "Qwen freeway decision"))[:160]

    if "congestion warning" in action_names:
        actions.append(build_warning_action(nodes, risk_level, reason))
    if "variable speed limit" in action_names:
        speed = safe_int(llm_output.get("target_speed_mph"), 40)
        vsl_nodes = capable_nodes(event, "variable_speed_limit_nodes", nodes)
        actions.append(build_vsl_action(vsl_nodes, speed, reason))
    if "ramp metering" in action_names:
        rate = safe_int(llm_output.get("metering_rate_veh_per_hour"), 420)
        ramp_nodes = capable_nodes(event, "ramp_meter_nodes", nodes)
        if not ramp_nodes:
            ramp_nodes = capable_nodes(
                event,
                "ramp_meter_nodes",
                control_capabilities(event).get("ramp_meter_nodes", []),
            )[:3]
        actions.append(build_ramp_meter_action(ramp_nodes, rate, reason))
    if "regional coordination" in action_names:
        actions.append(build_regional_action(event, reason))
    if "reroute" in action_names:
        diversion = safe_float(llm_output.get("diversion_ratio"), 0.20)
        gateways = capable_nodes(event, "reroute_gateway_nodes", nodes)
        actions.append(build_reroute_action(event, gateways, diversion, reason))

    return build_safe_decision(
        event=event,
        decision=decision,
        reason=reason,
        affected_nodes=nodes,
        actions=actions,
        risk_level=risk_level,
        confidence=safe_float(llm_output.get("confidence"), 0.75),
        decision_source=decision_source,
    )
