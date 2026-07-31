"""用途：执行动态云边调度，并用本地持久队列保存异步云端复核任务。"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from traffic_system.autonomy import run_autonomy_core
from traffic_system.benchmark_utils import build_payload, post_json
from traffic_system.decision_utils import (
    DECISION_CLASSES,
    build_decision_from_action_token,
    build_decision_from_student_class,
    extract_feature_vector,
    load_json,
    rule_teacher_decision,
    save_json,
)
from traffic_system.defer_gate import (
    GATE_CLASSES,
    build_gate_features,
    load_defer_gate,
    predict_defer_gate,
)
from traffic_system.edge_qwen_action_infer import build_action_prompt, request_action_token
from traffic_system.edge_student import load_student_model, predict_student
from traffic_system.policy_store import PolicyStore
from traffic_system.scheduler import AdaptiveScheduler, NetworkSnapshot


class PendingReviewQueue:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, row: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())

    def rows(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        output = []
        with self.path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        output.append(value)
        return output

    def replace(self, rows: List[Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=self.path.name, dir=str(self.path.parent))
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file_obj:
                for row in rows:
                    file_obj.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def flush(self, url: str, timeout: float) -> Dict[str, int]:
        pending = self.rows()
        remaining = []
        sent = 0
        for row in pending:
            try:
                body = json.dumps(row["payload"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                post_json(url, body, timeout)
                sent += 1
            except Exception:  # noqa: BLE001
                remaining.append(row)
        self.replace(remaining)
        return {"attempted": len(pending), "sent": sent, "remaining": len(remaining)}


def load_runtime_policy(path: Path) -> Dict[str, Any]:
    try:
        current = PolicyStore(path).current()
        return current or {}
    except Exception:  # noqa: BLE001
        return {}


def policy_version(path: Path) -> str:
    current = load_runtime_policy(path)
    return str(current["policy_version"]) if current else "unmanaged"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one adaptive edge-to-cloud traffic decision.")
    parser.add_argument("--edge_event", required=True)
    parser.add_argument("--model_json", default="models/edge_student_freeway_joint_metis4.json")
    parser.add_argument("--defer_gate", default="models/edge_defer_gate.npz")
    parser.add_argument("--cloud_url", default="http://127.0.0.1:18080/api/v1/traffic/decision")
    parser.add_argument("--output_json", default="results/decision/orchestrator_check.json")
    parser.add_argument("--queue", default="runtime/pending_cloud_reviews.jsonl")
    parser.add_argument("--policy", default="deployment/policy/current_policy.json")
    parser.add_argument("--network_available", dest="network_available", action="store_true", default=True)
    parser.add_argument("--no_network_available", dest="network_available", action="store_false")
    parser.add_argument("--rtt_ms", type=float, default=15.0)
    parser.add_argument("--jitter_ms", type=float, default=3.0)
    parser.add_argument("--loss_rate", type=float, default=0.0)
    parser.add_argument("--cloud_queue_ms", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--edge_qwen_url", default=os.environ.get("EDGE_QWEN_URL", ""))
    parser.add_argument("--qwen_timeout", type=float, default=0.5)
    parser.add_argument("--qwen_confidence_threshold", type=float, default=None)
    parser.add_argument("--qwen_decision_confidence", type=float, default=0.85)
    parser.add_argument("--conflict_suspected", action="store_true")
    parser.add_argument("--flush_pending", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue = PendingReviewQueue(Path(args.queue))
    flush_result = queue.flush(args.cloud_url, args.timeout) if args.flush_pending else None
    event = load_json(Path(args.edge_event))
    model = load_student_model(Path(args.model_json))
    started = time.perf_counter()
    student_class, confidence, _ = predict_student(event, model)
    student_decision = build_decision_from_student_class(event, student_class, confidence)
    rule_decision = rule_teacher_decision(event, decision_source="local_safety_policy")
    gate = load_defer_gate(Path(args.defer_gate))
    base_vector, feature_names = extract_feature_vector(event)
    if list(feature_names) != list(gate["base_feature_names"]):
        raise ValueError("Defer gate base feature schema mismatch")
    gate_features = build_gate_features(
        np.asarray([base_vector], dtype=np.float64),
        np.asarray([DECISION_CLASSES.index(str(rule_decision["decision"]))]),
        np.asarray([DECISION_CLASSES.index(student_class)]),
        np.asarray([confidence]),
    )
    gate_choices, gate_confidences = predict_defer_gate(gate_features, gate)
    gate_choice = GATE_CLASSES[int(gate_choices[0])]
    gate_confidence = float(gate_confidences[0])
    defer_recommended = gate_choice == "defer_cloud"
    policy_bundle = load_runtime_policy(Path(args.policy))
    current_policy = str(policy_bundle.get("policy_version", "unmanaged"))
    policy_payload = policy_bundle.get("payload", {})
    scheduler_policy = policy_payload.get("scheduler", {})
    qwen_policy = policy_payload.get("edge_llm", {})
    qwen_confidence_threshold = (
        args.qwen_confidence_threshold
        if args.qwen_confidence_threshold is not None
        else float(qwen_policy.get("trigger_confidence_below", 0.75))
    )
    student_decision["policy_version"] = current_policy
    selected_edge_decision = (
        student_decision if gate_choice == "edge_student" else rule_decision
    )
    selected_edge_decision["policy_version"] = current_policy
    selected_local_class = str(selected_edge_decision["decision"])
    selected_confidence = 1.0
    qwen_review: Dict[str, Any] = {
        "enabled": bool(args.edge_qwen_url),
        "triggered": False,
        "trigger_reason": None,
        "model_disagreement": False,
        "error": None,
    }
    if args.edge_qwen_url and (defer_recommended or confidence < qwen_confidence_threshold):
        qwen_review["triggered"] = True
        qwen_review["trigger_reason"] = "edge MLP confidence below threshold"
        try:
            prompt = build_action_prompt(event, "bitpacked_decimal")
            inference = request_action_token(
                args.edge_qwen_url,
                prompt,
                args.qwen_timeout,
                prompt_format="raw_task",
            )
            qwen_decision = build_decision_from_action_token(
                event,
                inference["action_token"],
                confidence=args.qwen_decision_confidence,
            )
            qwen_decision["policy_version"] = current_policy
            qwen_review.update(
                {
                    "action_token": inference["action_token"],
                    "latency_ms": inference["latency_ms"],
                    "decision": qwen_decision["decision"],
                    "model_disagreement": qwen_decision["decision"] != selected_local_class,
                }
            )
            selected_edge_decision = qwen_decision
            selected_confidence = args.qwen_decision_confidence
        except Exception as exc:  # noqa: BLE001
            qwen_review["error"] = "{}: {}".format(type(exc).__name__, exc)
    network = NetworkSnapshot(
        available=args.network_available,
        rtt_ms=args.rtt_ms,
        jitter_ms=args.jitter_ms,
        loss_rate=args.loss_rate,
        cloud_queue_ms=args.cloud_queue_ms,
    )
    scheduler = AdaptiveScheduler(
        deadline_ms=float(scheduler_policy.get("deadline_ms", 200.0)),
        confidence_threshold=float(scheduler_policy.get("confidence_threshold", 0.70)),
        edge_compute_ms=float(scheduler_policy.get("edge_compute_ms", 52.0)),
        cloud_compute_ms=float(scheduler_policy.get("cloud_compute_ms", 12.0)),
    )
    schedule = scheduler.schedule(
        event,
        selected_confidence,
        network,
        conflict_suspected=args.conflict_suspected,
        model_disagreement=bool(qwen_review["model_disagreement"]),
        defer_recommended=defer_recommended,
        selective_defer=True,
    )
    cloud_error = None
    cloud_response = None

    if schedule.route == "cloud_sync":
        payload = build_payload(str(event.get("event_id")), event, 0.0, 0.0, compact_event=True)
        try:
            cloud_response = post_json(args.cloud_url, payload, args.timeout)
            final_decision, autonomy = run_autonomy_core(
                event, "normal", cloud_response.get("decision")
            )
        except Exception as exc:  # noqa: BLE001
            cloud_error = "{}: {}".format(type(exc).__name__, exc)
            if qwen_review["triggered"] and not qwen_review["error"]:
                final_decision = selected_edge_decision
                autonomy = {
                    "autonomy_triggered": True,
                    "used_local_qwen_decision": True,
                    "cloud_mode": "down",
                }
            else:
                final_decision, autonomy = run_autonomy_core(event, "down", None)
    elif schedule.route == "cloud_async":
        queue.append(
            {
                "event_id": event.get("event_id"),
                "queued_at_ns": time.time_ns(),
                "payload": json.loads(
                    build_payload(str(event.get("event_id")), event, 0.0, 0.0, compact_event=True).decode("utf-8")
                ),
            }
        )
        final_decision, autonomy = run_autonomy_core(
            event,
            "degraded_async",
            None,
        )
        autonomy["async_cloud_review_queued"] = True
    elif schedule.route == "local_autonomy":
        if qwen_review["triggered"] and not qwen_review["error"]:
            final_decision = selected_edge_decision
            autonomy = {
                "autonomy_triggered": True,
                "used_local_qwen_decision": True,
                "cloud_mode": "down",
            }
        else:
            final_decision, autonomy = run_autonomy_core(event, "down", None)
            final_decision["policy_version"] = current_policy
    else:
        final_decision = selected_edge_decision
        autonomy = {"autonomy_triggered": False}

    result = {
        "event_id": event.get("event_id"),
        "policy_version": current_policy,
        "scheduler_policy": {
            "deadline_ms": scheduler.deadline_ms,
            "confidence_threshold": scheduler.confidence_threshold,
            "edge_compute_ms": scheduler.edge_compute_ms,
            "cloud_compute_ms": scheduler.cloud_compute_ms,
        },
        "schedule": schedule.to_dict(),
        "student_decision": student_decision,
        "defer_gate": {
            "choice": gate_choice,
            "confidence": round(gate_confidence, 6),
            "defer_recommended": defer_recommended,
        },
        "qwen_review": qwen_review,
        "selected_edge_decision": selected_edge_decision,
        "final_decision": final_decision,
        "cloud_error": cloud_error,
        "cloud_response_received": cloud_response is not None,
        "autonomy": autonomy,
        "pending_cloud_reviews": len(queue.rows()),
        "flush_result": flush_result,
        "orchestrator_latency_ms": round((time.perf_counter() - started) * 1000.0, 6),
    }
    save_json(result, Path(args.output_json))
    print(json.dumps({
        "route": schedule.route,
        "final_decision": final_decision.get("decision"),
        "cloud_response_received": result["cloud_response_received"],
        "pending_cloud_reviews": result["pending_cloud_reviews"],
        "qwen_triggered": qwen_review["triggered"],
        "orchestrator_latency_ms": result["orchestrator_latency_ms"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
