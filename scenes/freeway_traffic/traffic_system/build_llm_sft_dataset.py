"""用途：把 Teacher 交通决策标签转换为 Qwen Student 的 SFT 训练数据。"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from traffic_system.decision_utils import load_json, read_jsonl, safe_float, safe_int, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = """你是部署在边缘侧的高速公路交通轻量决策模型。
你只能输出一个紧凑 JSON 对象，不能输出 Markdown、解释段落或思维过程。
根据 edge_event 与 cloud_context 选择局部初步控制动作。
允许的 decision: no_action, congestion_warning, variable_speed_limit, ramp_metering, regional_coordination, reroute。
允许的 global_risk_level: low, medium, high, severe。
affected_nodes 必须来自 top_k_risk_nodes。
不得输出普通路口红绿灯或延长绿灯动作。
"""

ACTION_TOKEN_SYSTEM_PROMPT = "只答A-F。"


STANDARD_SCHEMA = {
    "decision": "no_action|congestion_warning|variable_speed_limit|ramp_metering|regional_coordination|reroute",
    "global_risk_level": "low|medium|high|severe",
    "affected_nodes": ["int node ids selected from top_k_risk_nodes"],
    "actions": ["congestion warning|variable speed limit|ramp metering|regional coordination|reroute"],
    "target_speed_mph": "integer 0 or 25..65",
    "metering_rate_veh_per_hour": "integer 0 or 240..900",
    "diversion_ratio": "number 0 or 0.05..0.40",
    "reason": "short Chinese reason, <= 30 Chinese chars",
    "confidence": "float 0..1",
}

COMPACT_DECISION_CODES = {
    "no_action": "na",
    "congestion_warning": "cw",
    "variable_speed_limit": "vsl",
    "ramp_metering": "rm",
    "regional_coordination": "rc",
    "reroute": "rr",
}

COMPACT_RISK_CODES = {
    "low": "l",
    "medium": "m",
    "high": "h",
    "severe": "s",
}

COMPACT_ACTION_CODES = {
    "congestion warning": "cw",
    "variable speed limit": "vsl",
    "ramp metering": "rm",
    "regional coordination": "rc",
    "reroute": "rr",
}

ACTION_TOKEN_CODES = {
    "no_action": "A",
    "congestion_warning": "B",
    "variable_speed_limit": "C",
    "ramp_metering": "D",
    "regional_coordination": "E",
    "reroute": "F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build phase-1 SFT data for Qwen/DeepSeek traffic decision distillation."
    )
    parser.add_argument("--teacher_labels", default="datasets/freeway_teacher_labels_qwen9b_safe_joint_metis4.jsonl")
    parser.add_argument("--output_dir", default="datasets/llm_sft_freeway_action_token")
    parser.add_argument("--teacher_model", default="qwen3.5:9b")
    parser.add_argument("--student_model", default="qwen3.5:0.8b")
    parser.add_argument(
        "--target_schema",
        default="action_token",
        choices=["standard", "compact", "action_token"],
    )
    parser.add_argument("--max_top_nodes", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance_train", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0 means use all teacher labels.")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def compact_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def compact_event_for_sft(event: Dict[str, Any], max_top_nodes: int) -> Dict[str, Any]:
    summary = event.get("region_summary", {})
    if not isinstance(summary, dict):
        summary = {}
    nodes = event.get("top_k_risk_nodes", [])
    if not isinstance(nodes, list):
        nodes = []

    top_nodes = []
    for node in nodes[:max(1, max_top_nodes)]:
        if not isinstance(node, dict):
            continue
        forecast = node.get("forecast", {})
        if not isinstance(forecast, dict):
            forecast = {}
        top_nodes.append(
            {
                "node_id": safe_int(node.get("node_id")),
                "risk_level": str(node.get("risk_level", "low")),
                "risk_score": round(safe_float(node.get("risk_score")), 4),
                "future_mean": round(safe_float(node.get("future_mean")), 3),
                "growth_rate": round(safe_float(node.get("growth_rate")), 4),
                "peak_growth_rate": round(safe_float(node.get("peak_growth_rate")), 4),
                "flow": round(safe_float(forecast.get("flow_mean", node.get("future_mean"))), 3),
                "occupancy": round(safe_float(forecast.get("occupancy_mean")), 4),
                "speed_mean": round(safe_float(forecast.get("speed_mean")), 3),
                "speed_min": round(safe_float(forecast.get("speed_min")), 3),
            }
        )

    return {
        "edge_id": event.get("edge_id"),
        "region_id": event.get("region_id"),
        "sample_id": event.get("sample_id"),
        "upload_required": bool(event.get("upload_required", False)),
        "upload_level": event.get("upload_level"),
        "latency_ms": event.get("latency_ms"),
        "summary": {
            "nodes": safe_int(summary.get("num_nodes"), len(event.get("managed_node_ids", []))),
            "low": safe_int(summary.get("num_low_nodes", summary.get("node_risk_counts", {}).get("low"))),
            "medium": safe_int(summary.get("num_medium_nodes", summary.get("node_risk_counts", {}).get("medium"))),
            "high": safe_int(summary.get("num_high_nodes", summary.get("node_risk_counts", {}).get("high"))),
            "severe": safe_int(summary.get("num_severe_nodes", summary.get("node_risk_counts", {}).get("severe"))),
            "mean_risk": round(safe_float(summary.get("mean_risk_score", summary.get("mean_node_risk_score"))), 4),
            "max_risk": round(safe_float(summary.get("max_risk_score", summary.get("region_risk_score"))), 4),
            "cluster": bool(summary.get("congestion_cluster_detected", False)),
        },
        "top_k_risk_nodes": top_nodes,
    }


def short_reason(value: Any, limit: int = 36) -> str:
    reason = str(value or "交通风险需边缘侧处理").strip()
    return reason[:limit]


def action_text(action: Any, decision: str) -> str:
    if isinstance(action, str):
        text = action.strip()
        if text:
            return text
    if not isinstance(action, dict):
        return "congestion warning"

    action_type = str(action.get("type", ""))
    return {
        "traffic_advisory": "congestion warning",
        "variable_speed_limit": "variable speed limit",
        "ramp_metering": "ramp metering",
        "regional_coordination": "regional coordination",
        "reroute": "reroute",
    }.get(action_type, "congestion warning")


def control_parameters(actions: Sequence[Any]) -> Tuple[int, int, float]:
    speed = 0
    metering_rate = 0
    diversion_ratio = 0.0
    for action in actions:
        if isinstance(action, dict):
            if action.get("type") == "variable_speed_limit":
                speed = safe_int(action.get("target_speed_mph"), 0)
            elif action.get("type") == "ramp_metering":
                metering_rate = safe_int(action.get("metering_rate_veh_per_hour"), 0)
            elif action.get("type") == "reroute":
                diversion_ratio = safe_float(action.get("diversion_ratio"), 0.0)
    return speed, metering_rate, diversion_ratio


def normalize_teacher_decision(decision: Dict[str, Any], target_schema: str) -> Any:
    decision_name = str(decision.get("decision", "congestion_warning"))
    risk_level = str(decision.get("global_risk_level", "medium"))
    affected_nodes = [
        int(node)
        for node in decision.get("affected_nodes", [])
        if not isinstance(node, bool)
    ]
    raw_actions = decision.get("actions", [])
    if not isinstance(raw_actions, list):
        raw_actions = []
    actions = []
    for raw_action in raw_actions:
        text = action_text(raw_action, decision_name)
        if text not in actions:
            actions.append(text)
    if not actions and decision_name == "no_action":
        actions = []
    elif not actions:
        actions = ["congestion warning"]

    speed, metering_rate, diversion_ratio = control_parameters(raw_actions)
    confidence = max(0.0, min(1.0, safe_float(decision.get("confidence"), 0.8)))

    if target_schema == "action_token":
        return ACTION_TOKEN_CODES.get(decision_name, "B")

    if target_schema == "compact":
        compact_actions = [
            COMPACT_ACTION_CODES.get(action, "cw")
            for action in actions
        ]
        return {
            "d": COMPACT_DECISION_CODES.get(decision_name, "cw"),
            "r": COMPACT_RISK_CODES.get(risk_level, "m"),
            "n": affected_nodes,
            "a": compact_actions,
            "s": speed,
            "m": metering_rate,
            "v": round(diversion_ratio, 3),
            "c": round(confidence, 3),
        }

    return {
        "decision": decision_name,
        "global_risk_level": risk_level,
        "affected_nodes": affected_nodes,
        "actions": actions,
        "target_speed_mph": speed,
        "metering_rate_veh_per_hour": metering_rate,
        "diversion_ratio": round(diversion_ratio, 3),
        "reason": short_reason(decision.get("reason")),
        "confidence": round(confidence, 3),
    }


def infer_network_status(row: Dict[str, Any]) -> str:
    latency = safe_float(row.get("teacher_latency_ms"), 0.0)
    if latency >= 2000:
        return "weak"
    return "normal"


def build_cloud_context(event: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    summary = event.get("region_summary", {})
    if not isinstance(summary, dict):
        summary = {}
    return {
        "network_status": str(row.get("network_status") or infer_network_status(row)),
        "coordination_goal": "prioritize safety and reduce regional congestion propagation",
        "conflict_policy": "prefer safer high-risk action when neighboring regions disagree",
        "region_congestion_cluster": bool(summary.get("congestion_cluster_detected", False)),
    }


def build_user_prompt(
    event: Dict[str, Any],
    row: Dict[str, Any],
    target_schema: str,
    max_top_nodes: int,
) -> str:
    if target_schema == "action_token":
        summary = event.get("region_summary", {})
        if not isinstance(summary, dict):
            summary = {}
        counts = summary.get("node_risk_counts", {})
        if not isinstance(counts, dict):
            counts = {}
        top_nodes = event.get("top_k_risk_nodes", [])
        if not isinstance(top_nodes, list):
            top_nodes = []
        top_nodes = [node for node in top_nodes[:3] if isinstance(node, dict)]
        risk_levels = {"low": 0, "medium": 1, "high": 2, "severe": 3}
        region_level = risk_levels.get(str(summary.get("region_risk_level", "low")), 0)
        top_level = risk_levels.get(
            str(top_nodes[0].get("risk_level", "low")) if top_nodes else "low",
            0,
        )
        top_score = max(
            [safe_float(node.get("risk_score")) for node in top_nodes] or [0.0]
        )
        speed_values = []
        occupancy_values = []
        for node in top_nodes:
            forecast = node.get("forecast", {})
            if not isinstance(forecast, dict):
                forecast = {}
            speed_values.append(safe_float(forecast.get("speed_min"), 65.0))
            occupancy_values.append(safe_float(forecast.get("occupancy_mean"), 0.0))
        capabilities = event.get("control_capabilities", {})
        if not isinstance(capabilities, dict):
            capabilities = {}
        severe = safe_int(summary.get("num_severe_nodes", counts.get("severe")))
        high = safe_int(summary.get("num_high_nodes", counts.get("high")))
        cluster = bool(
            summary.get("congestion_cluster_detected", severe >= 2 or severe + high >= 5)
        )
        fields = [
            "e{}".format(safe_int(event.get("partition_id"), 0)),
            "x{}".format(int(round(safe_float(summary.get("region_risk_score"), 0.0) * 100))),
            "r{}".format(region_level),
            "t{}".format(top_level),
            "l{}".format(safe_int(summary.get("num_low_nodes", counts.get("low")))),
            "m{}".format(safe_int(summary.get("num_medium_nodes", counts.get("medium")))),
            "h{}".format(high),
            "s{}".format(severe),
            "q{}".format(int(round(top_score * 100))),
            "v{}".format(int(round(min(speed_values) if speed_values else 65.0))),
            "o{}".format(int(round(max(occupancy_values) * 100)) if occupancy_values else 0),
            "c{}".format(1 if cluster else 0),
            "a{}".format(1 if capabilities.get("ramp_meter_nodes") else 0),
            "g{}".format(1 if capabilities.get("reroute_gateway_nodes") else 0),
        ]
        return "".join(fields)

    schema = STANDARD_SCHEMA
    if target_schema == "compact":
        schema = {
            "d": "na|cw|vsl|rm|rc|rr",
            "r": "l|m|h|s",
            "n": ["node ids"],
            "a": ["cw|vsl|rm|rc|rr"],
            "s": "speed mph",
            "m": "metering veh/h",
            "v": "diversion ratio",
            "c": "confidence 0..1",
        }
    payload = {
        "task": "根据边缘高速公路态势输出本地初步控制 JSON。",
        "output_schema": schema,
        "edge_event": compact_event_for_sft(event, max_top_nodes=max_top_nodes),
        "cloud_context": build_cloud_context(event, row),
    }
    return compact_json(payload)


def build_sft_record(
    row: Dict[str, Any],
    target_schema: str,
    teacher_model: str,
    student_model: str,
    max_top_nodes: int,
) -> Dict[str, Any]:
    event_path_value = row.get("event_path")
    if not event_path_value:
        raise ValueError("Teacher row missing event_path.")
    event_path = resolve_path(str(event_path_value))
    event = load_json(event_path)

    teacher_decision = row.get("teacher_decision")
    if not isinstance(teacher_decision, dict):
        raise ValueError("Teacher row missing teacher_decision object.")

    user_content = build_user_prompt(event, row, target_schema, max_top_nodes=max_top_nodes)
    target = normalize_teacher_decision(teacher_decision, target_schema)
    assistant_content = target if isinstance(target, str) else compact_json(target)
    system_content = ACTION_TOKEN_SYSTEM_PROMPT if target_schema == "action_token" else SYSTEM_PROMPT

    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "event_id": str(row.get("event_id") or event.get("event_id") or event_path.stem),
        "event_path": str(event_path.relative_to(PROJECT_ROOT)),
        "teacher_model": teacher_model,
        "student_model": student_model,
        "teacher_source": str(row.get("teacher_source", "unknown_teacher")),
        "target_schema": target_schema,
        "target": target,
    }


def split_records(
    records: Sequence[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    groups: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        event_id = str(record.get("event_id", ""))
        match = re.search(r"sample_(\d+)", event_id)
        if not match:
            raise ValueError("event_id does not contain a sample timestamp: {}".format(event_id))
        groups.setdefault(int(match.group(1)), []).append(record)

    ordered_samples = sorted(groups)
    if len(ordered_samples) < 3:
        raise ValueError("Need at least three timestamp groups for train/val/test.")
    train_end = max(1, int(len(ordered_samples) * train_ratio))
    val_count = max(1, int(len(ordered_samples) * val_ratio))
    val_end = min(len(ordered_samples) - 1, train_end + val_count)
    train_samples = ordered_samples[:train_end]
    val_samples = ordered_samples[train_end:val_end]
    test_samples = ordered_samples[val_end:]
    train = [record for sample in train_samples for record in groups[sample]]
    val = [record for sample in val_samples for record in groups[sample]]
    test = [record for sample in test_samples for record in groups[sample]]
    return train, val, test


def count_decisions(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        target = record.get("target", {})
        if isinstance(target, dict):
            counts[str(target.get("decision", target.get("d", "unknown")))] += 1
        elif isinstance(target, str):
            counts[target] += 1
    return dict(sorted(counts.items()))


def balance_training_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        target = str(record.get("target", "unknown"))
        grouped.setdefault(target, []).append(record)
    if not grouped:
        return []
    target_count = max(len(rows) for rows in grouped.values())
    balanced = []
    for label in sorted(grouped):
        rows = grouped[label]
        balanced.extend(rows[index % len(rows)] for index in range(target_count))
    return balanced


def main() -> None:
    args = parse_args()
    rows = read_jsonl(resolve_path(args.teacher_labels))
    if args.limit > 0:
        rows = rows[: args.limit]
    records = [
        build_sft_record(
            row,
            target_schema=args.target_schema,
            teacher_model=args.teacher_model,
            student_model=args.student_model,
            max_top_nodes=args.max_top_nodes,
        )
        for row in rows
    ]

    train, val, test = split_records(records, args.train_ratio, args.val_ratio, args.seed)
    train_unique_count = len(train)
    if args.balance_train and args.target_schema == "action_token":
        train = balance_training_records(train)
    output_dir = resolve_path(args.output_dir)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)
    write_jsonl(test, test_path)

    summary = {
        "task": "phase1_sft_distillation_dataset",
        "teacher_labels": str(resolve_path(args.teacher_labels).relative_to(PROJECT_ROOT)),
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "target_schema": args.target_schema,
        "max_top_nodes": args.max_top_nodes,
        "split_strategy": "chronological_sample_group",
        "num_total": len(records),
        "num_train_unique": train_unique_count,
        "num_train": len(train),
        "num_val": len(val),
        "num_test": len(test),
        "decision_counts_total": count_decisions(records),
        "decision_counts_train": count_decisions(train),
        "decision_counts_val": count_decisions(val),
        "decision_counts_test": count_decisions(test),
        "balance_train": bool(args.balance_train),
        "files": {
            "train": str(train_path.relative_to(PROJECT_ROOT)),
            "val": str(val_path.relative_to(PROJECT_ROOT)),
            "test": str(test_path.relative_to(PROJECT_ROOT)),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
