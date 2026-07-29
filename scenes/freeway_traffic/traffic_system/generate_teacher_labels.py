"""用途：调用 Qwen Teacher 为边缘交通事件生成结构化决策标签。"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from traffic_system.safety_filter import validate_and_filter_decision
from traffic_system.decision_utils import (
    DECISION_CLASSES,
    build_decision_from_student_class,
    collect_event_paths,
    compact_event_for_teacher,
    event_identifier,
    load_json,
    rule_teacher_decision,
    save_json,
    safe_float,
    safe_int,
    write_jsonl,
)


SYSTEM_PROMPT = """You are a cloud-side freeway traffic management teacher model.
Return only one compact JSON object. Do not output markdown, explanations, or thinking.
Allowed decision values: no_action, congestion_warning, variable_speed_limit, ramp_metering,
regional_coordination, reroute.
Allowed global_risk_level values: low, medium, high, severe.
affected_nodes must be selected from top_k_risk_nodes only.
Use no_action for low risk, congestion_warning for medium risk, variable_speed_limit for high risk,
ramp_metering for an isolated severe bottleneck, regional_coordination for a severe local cluster,
and reroute only for an extreme regional cluster. PEMS08 nodes are freeway detectors, not urban
intersections. Never output traffic-light or green-time control.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate teacher decisions for traffic edge events using Qwen/Ollama or explicit rule mode."
    )
    parser.add_argument("--events", default="datasets/freeway_events_joint_metis4_manifest.jsonl")
    parser.add_argument("--output_jsonl", default="datasets/freeway_teacher_labels_qwen9b_joint_metis4.jsonl")
    parser.add_argument("--summary_json", default="results/decision/teacher_label_summary.json")
    parser.add_argument("--teacher_mode", default="ollama", choices=["ollama", "rule"])
    parser.add_argument("--ollama_url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--limit", type=int, default=0, help="0 means all events.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_predict", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--strict_rule_audit", action="store_true", help="Also save rule decision for comparison.")
    return parser.parse_args()


def load_events(input_path: Path, limit: int) -> List[Tuple[Path, Dict[str, Any]]]:
    paths = collect_event_paths(input_path)
    rows = []
    for path in paths:
        rows.append((path, load_json(path)))
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model JSON response is not an object.")
    return parsed


def teacher_user_prompt(event: Dict[str, Any]) -> str:
    schema = {
        "decision": (
            "one of no_action, congestion_warning, variable_speed_limit, "
            "ramp_metering, regional_coordination, reroute"
        ),
        "global_risk_level": "one of low, medium, high, severe",
        "affected_nodes": ["integer node ids from top_k_risk_nodes"],
        "reason": "short Chinese reason, <= 30 characters",
        "confidence": "number 0..1",
    }
    payload = {
        "task": "Decide the cloud-side traffic action for this edge event.",
        "output_schema": schema,
        "edge_event": compact_event_for_teacher(event),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def call_ollama_teacher(args: argparse.Namespace, event: Dict[str, Any]) -> Tuple[Dict[str, Any], float, str]:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": teacher_user_prompt(event)},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": float(args.temperature),
            "num_predict": int(args.num_predict),
            "top_p": 0.8,
        },
    }
    start = time.perf_counter()
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        args.ollama_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("Ollama HTTP {}: {}".format(exc.code, error_body)) from exc
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    data = json.loads(raw_body)
    message = data.get("message", {})
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty Ollama response content.")
    raw = extract_json_object(content)
    return raw, elapsed_ms, content


def normalize_teacher_output(event: Dict[str, Any], raw: Dict[str, Any], source: str) -> Dict[str, Any]:
    decision = str(raw.get("decision", "congestion_warning"))
    if decision not in DECISION_CLASSES:
        decision = str(rule_teacher_decision(event).get("decision", "congestion_warning"))

    affected_nodes = raw.get("affected_nodes")
    if not isinstance(affected_nodes, list):
        affected_nodes = None
    reason = str(raw.get("reason", "cloud teacher decision"))[:80]
    decision_obj = build_decision_from_student_class(
        event,
        decision,
        confidence=safe_float(raw.get("confidence"), 0.86 if source == "ollama_teacher" else 0.78),
        decision_source=source,
    )
    if affected_nodes is not None:
        allowed_nodes = {
            safe_int(node.get("node_id"), -1)
            for node in event.get("top_k_risk_nodes", [])
            if isinstance(node, dict) and safe_int(node.get("node_id"), -1) >= 0
        }
        selected = []
        for node in affected_nodes:
            if isinstance(node, bool):
                continue
            try:
                node_id = int(node)
            except (TypeError, ValueError):
                continue
            if node_id in allowed_nodes and node_id not in selected:
                selected.append(node_id)
        decision_obj["affected_nodes"] = selected
        for action in decision_obj.get("actions", []):
            if action.get("type") in ("traffic_advisory", "variable_speed_limit"):
                action["target_nodes"] = selected
    decision_obj["reason"] = reason
    decision_obj["decision_source"] = source
    if str(raw.get("global_risk_level")) in {"low", "medium", "high", "severe"}:
        decision_obj["global_risk_level"] = str(raw["global_risk_level"])
    return validate_and_filter_decision(decision_obj)


def label_one_event(
    args: argparse.Namespace,
    event: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rule_decision = rule_teacher_decision(event, decision_source="rule_teacher")
    metadata: Dict[str, Any] = {
        "teacher_source": "rule_teacher",
        "teacher_latency_ms": 0.0,
        "raw_response": None,
    }

    if args.teacher_mode == "rule":
        return rule_decision, metadata

    raw, elapsed_ms, raw_text = call_ollama_teacher(args, event)
    decision = normalize_teacher_output(event, raw, source="ollama_teacher")
    metadata.update(
        {
            "teacher_source": "ollama_teacher",
            "teacher_latency_ms": round(elapsed_ms, 3),
            "raw_response": raw_text,
        }
    )
    return decision, metadata


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    source_counts: Dict[str, int] = {}
    decision_counts: Dict[str, int] = {}
    latencies = []
    for record in records:
        source = str(record.get("teacher_source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
        decision = record.get("teacher_decision", {})
        if isinstance(decision, dict):
            cls = str(decision.get("decision", "unknown"))
            decision_counts[cls] = decision_counts.get(cls, 0) + 1
        latency = safe_float(record.get("teacher_latency_ms"), 0.0)
        if latency > 0:
            latencies.append(latency)
    return {
        "num_labeled_events": len(records),
        "teacher_source_counts": source_counts,
        "decision_counts": decision_counts,
        "average_ollama_teacher_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "max_ollama_teacher_latency_ms": round(max(latencies), 3) if latencies else 0.0,
    }


def main() -> None:
    args = parse_args()
    events = load_events(Path(args.events), args.limit)
    records = []
    for index, (event_path, event) in enumerate(events, start=1):
        teacher_decision, metadata = label_one_event(args, event)
        record: Dict[str, Any] = {
            "event_id": event.get("event_id") or event_identifier(event),
            "event_path": str(event_path),
            "teacher_decision": teacher_decision,
            "teacher_source": metadata["teacher_source"],
            "teacher_latency_ms": metadata["teacher_latency_ms"],
        }
        if args.strict_rule_audit:
            record["rule_decision"] = rule_teacher_decision(event, decision_source="rule_teacher")
        records.append(record)
        print(
            "[{}/{}] {} -> {} ({}, {:.1f} ms)".format(
                index,
                len(events),
                record["event_id"],
                teacher_decision.get("decision"),
                record["teacher_source"],
                record["teacher_latency_ms"],
            )
        )

    write_jsonl(records, Path(args.output_jsonl))
    summary = summarize(records)
    save_json(summary, Path(args.summary_json))
    print("Labels saved to:", args.output_jsonl)
    print("Summary saved to:", args.summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
