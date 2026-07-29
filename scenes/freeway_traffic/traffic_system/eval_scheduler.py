"""用途：用真实事件和实测网络参数评估动态调度、时延、准确率及通信开销。"""

import argparse
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from traffic_system.cloud_coordinator import load_cloud_model, predict_cloud
from traffic_system.decision_utils import (
    compact_event_for_teacher,
    load_json,
    read_jsonl,
    rule_teacher_decision,
    save_json,
)
from traffic_system.edge_student import load_student_model, predict_student
from traffic_system.scheduler import AdaptiveScheduler, NetworkSnapshot


PROFILES = {
    "normal": NetworkSnapshot(True, 15.0, 3.0, 0.00, 1.0),
    "mild": NetworkSnapshot(True, 58.0, 10.0, 0.01, 1.5),
    "severe": NetworkSnapshot(True, 117.0, 30.0, 0.10, 2.0),
    "outage": NetworkSnapshot(False, 0.0, 0.0, 1.00, 0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate risk/network-aware cloud-edge scheduling.")
    parser.add_argument("--labels", default="datasets/freeway_teacher_labels_qwen9b_safe_joint_metis4.jsonl")
    parser.add_argument("--model_json", default="models/edge_student_freeway_joint_metis4.json")
    parser.add_argument(
        "--cloud_model",
        default="models/cloud_coordinator_future_calibrated.joblib",
    )
    parser.add_argument("--output_json", default="results/edge/adaptive_scheduler_eval.json")
    parser.add_argument("--deadline_ms", type=float, default=200.0)
    parser.add_argument("--edge_compute_ms", type=float, default=46.98)
    parser.add_argument("--cloud_compute_ms", type=float, default=12.0)
    parser.add_argument("--test_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def label_for(row: Dict[str, Any]) -> str:
    decision = row.get("teacher_decision", {})
    return str(decision.get("decision", "congestion_warning")) if isinstance(decision, dict) else "congestion_warning"


def payload_size(event: Dict[str, Any]) -> int:
    compact = compact_event_for_teacher(event)
    return len(json.dumps(compact, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def profile_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    routes = Counter(row["route"] for row in rows)
    return {
        "cases": len(rows),
        "route_distribution": dict(sorted(routes.items())),
        "cloud_request_rate": round(sum(row["cloud_requested"] for row in rows) / len(rows), 6),
        "synchronous_cloud_rate": round(sum(row["waits_for_cloud"] for row in rows) / len(rows), 6),
        "business_availability": round(sum(row["available"] for row in rows) / len(rows), 6),
        "immediate_decision_accuracy": round(sum(row["correct"] for row in rows) / len(rows), 6),
        "average_response_ms": round(statistics.fmean(row["response_ms"] for row in rows), 6),
        "under_200ms_rate": round(sum(row["response_ms"] <= 200.0 for row in rows) / len(rows), 6),
        "average_upload_bytes": round(statistics.fmean(row["upload_bytes"] for row in rows), 3),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    all_labels = read_jsonl(Path(args.labels))
    sample_ids = sorted(
        {int(load_json(Path(str(row["event_path"]))).get("sample_id", -1)) for row in all_labels}
    )
    test_group_count = max(1, int(round(len(sample_ids) * args.test_ratio)))
    test_samples = set(sample_ids[-test_group_count:])
    labels = [
        row
        for row in all_labels
        if int(load_json(Path(str(row["event_path"]))).get("sample_id", -1)) in test_samples
    ]
    model = load_student_model(Path(args.model_json))
    cloud_model = load_cloud_model(Path(args.cloud_model))
    scheduler = AdaptiveScheduler(
        deadline_ms=args.deadline_ms,
        edge_compute_ms=args.edge_compute_ms,
        cloud_compute_ms=args.cloud_compute_ms,
    )
    rows_by_profile: Dict[str, List[Dict[str, Any]]] = {name: [] for name in PROFILES}
    scheduler_times = []
    raw_window_bytes = 170 * 3 * 12 * 4

    for label_row in labels:
        event = load_json(Path(str(label_row["event_path"])))
        target = label_for(label_row)
        student_class, confidence, _ = predict_student(event, model)
        cloud_class, _ = predict_cloud(event, cloud_model)
        local_class = str(rule_teacher_decision(event).get("decision"))
        compact_bytes = payload_size(event)
        for profile_name, network in PROFILES.items():
            started = time.perf_counter()
            schedule = scheduler.schedule(event, confidence, network)
            scheduler_times.append((time.perf_counter() - started) * 1000.0)
            delivered = network.available and rng.random() >= network.loss_rate
            if schedule.route == "cloud_sync" and delivered:
                output_class = cloud_class
                response_ms = schedule.predicted_sync_e2e_ms
                available = True
            elif schedule.route == "local_autonomy":
                output_class = local_class
                response_ms = args.edge_compute_ms
                available = True
            else:
                output_class = student_class
                response_ms = args.edge_compute_ms
                available = True
            rows_by_profile[profile_name].append(
                {
                    "event_id": label_row.get("event_id"),
                    "route": schedule.route,
                    "cloud_requested": schedule.cloud_requested,
                    "waits_for_cloud": schedule.waits_for_cloud,
                    "delivered": delivered,
                    "available": available,
                    "correct": output_class == target,
                    "response_ms": response_ms,
                    "upload_bytes": compact_bytes if schedule.cloud_requested else 0,
                }
            )

    profile_results = {name: profile_summary(rows) for name, rows in rows_by_profile.items()}
    all_rows = [row for rows in rows_by_profile.values() for row in rows]
    adaptive_upload = sum(row["upload_bytes"] for row in all_rows)
    centralized_upload = len(all_rows) * raw_window_bytes
    always_edge_accuracy = sum(
        predict_student(load_json(Path(str(row["event_path"]))), model)[0] == label_for(row)
        for row in labels
    ) / len(labels)
    cloud_accuracy = sum(
        predict_cloud(load_json(Path(str(row["event_path"]))), cloud_model)[0] == label_for(row)
        for row in labels
    ) / len(labels)
    result = {
        "task": "adaptive_cloud_edge_scheduler_evaluation",
        "dataset_events": len(labels),
        "split": {
            "strategy": "strict_temporal_group",
            "test_ratio": args.test_ratio,
            "test_sample_ids": sorted(test_samples),
        },
        "network_profiles": {
            name: {**profile_results[name], "network": network.__dict__}
            for name, network in PROFILES.items()
        },
        "overall": {
            "cases": len(all_rows),
            "business_availability": round(sum(row["available"] for row in all_rows) / len(all_rows), 6),
            "immediate_decision_accuracy": round(sum(row["correct"] for row in all_rows) / len(all_rows), 6),
            "under_200ms_rate": round(sum(row["response_ms"] <= 200.0 for row in all_rows) / len(all_rows), 6),
            "average_response_ms": round(statistics.fmean(row["response_ms"] for row in all_rows), 6),
            "cloud_request_rate": round(sum(row["cloud_requested"] for row in all_rows) / len(all_rows), 6),
        },
        "baselines": {
            "always_edge": {"decision_accuracy": round(always_edge_accuracy, 6), "business_availability": 1.0},
            "cloud_coordinator": {
                "decision_accuracy": round(cloud_accuracy, 6),
                "model": args.cloud_model,
            },
            "always_cloud_without_edge_fallback": {
                "business_availability_by_profile": {
                    name: round(1.0 - network.loss_rate if network.available else 0.0, 6)
                    for name, network in PROFILES.items()
                }
            },
        },
        "communication": {
            "centralized_raw_window_bytes_per_case": raw_window_bytes,
            "centralized_total_bytes": centralized_upload,
            "adaptive_total_bytes": adaptive_upload,
            "upload_reduction_vs_centralized": round(1.0 - adaptive_upload / centralized_upload, 6),
        },
        "scheduler_overhead_ms": {
            "average": round(statistics.fmean(scheduler_times), 6),
            "max": round(max(scheduler_times), 6),
        },
        "measurement_note": (
            "Network profile latency/loss values are grounded in the Jetson02 HTTP fault experiments; "
            "route quality is replayed offline over real PEMS08 edge events."
        ),
    }
    save_json(result, Path(args.output_json))
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
    print("upload reduction:", result["communication"]["upload_reduction_vs_centralized"])
    print("saved:", args.output_json)


if __name__ == "__main__":
    main()
