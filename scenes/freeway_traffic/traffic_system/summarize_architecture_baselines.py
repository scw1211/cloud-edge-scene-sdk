"""用途：汇总集中式、单边缘和自适应云边协同三种架构的可比指标。"""

import argparse
from pathlib import Path
from typing import Any, Dict

from traffic_system.decision_utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize architecture baseline evidence.")
    parser.add_argument("--scheduler_result", default="results/edge/adaptive_scheduler_eval.json")
    parser.add_argument("--student_result", default="results/decision/edge_student_freeway_joint_metis4.json")
    parser.add_argument(
        "--continuous_result",
        default="results/edge/continuous_edge_cloud_final_highrisk_jetson02.json",
    )
    parser.add_argument("--output_json", default="results/edge/architecture_baseline_comparison.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scheduler = load_json(Path(args.scheduler_result))
    student = load_json(Path(args.student_result))
    continuous = load_json(Path(args.continuous_result))
    profiles = scheduler["network_profiles"]
    student_test = student["test"]
    result: Dict[str, Any] = {
        "task": "three_architecture_baseline_comparison",
        "strict_temporal_test_events": scheduler["dataset_events"],
        "architectures": {
            "centralized_cloud": {
                "path": "raw full-graph sensor window -> cloud -> decision -> edge",
                "raw_upload_bytes_per_case": scheduler["communication"]["centralized_raw_window_bytes_per_case"],
                "business_availability_by_network_profile": scheduler["baselines"]["always_cloud_without_edge_fallback"]["business_availability_by_profile"],
                "accuracy_note": "Teacher agreement is the reference label, not an independently measured accuracy.",
                "limitations": "No local decision remains when the cloud link is unavailable.",
            },
            "single_edge": {
                "path": "ASTGCN + Student local decision without cross-region coordination",
                "decision_accuracy": student_test["accuracy"],
                "weighted_f1": student_test["weighted_f1"],
                "business_availability": 1.0,
                "cloud_upload_bytes": 0,
                "limitations": "Cannot resolve decisions across edge-region boundaries.",
            },
            "adaptive_cloud_edge": {
                "path": "edge immediate decision + deadline-aware cloud sync/async coordination",
                "immediate_decision_accuracy": scheduler["overall"]["immediate_decision_accuracy"],
                "business_availability": scheduler["overall"]["business_availability"],
                "average_response_ms_replay": scheduler["overall"]["average_response_ms"],
                "under_200ms_rate_replay": scheduler["overall"]["under_200ms_rate"],
                "cloud_request_rate": scheduler["overall"]["cloud_request_rate"],
                "upload_reduction_vs_centralized": scheduler["communication"]["upload_reduction_vs_centralized"],
                "jetson_to_cloud_measured_average_ms": continuous["latency"]["total_e2e"]["average_ms"],
                "jetson_to_cloud_measured_p95_ms": continuous["latency"]["total_e2e"]["p95_ms"],
                "jetson_to_cloud_measured_max_ms": continuous["latency"]["total_e2e"]["max_ms"],
                "jetson_process_max_rss_mb": continuous["process_max_rss_mb"],
                "measured_high_risk_cases": sum(
                    bool(row.get("upload_required")) for row in continuous.get("samples", [])
                ),
                "measured_business_availability": continuous["business_availability"],
            },
        },
        "evidence_scope": {
            "measured": ["Jetson02-to-WSL continuous HTTP closed loop", "Student strict temporal split"],
            "replayed": ["four network profiles using measured Jetson fault-profile parameters"],
            "analytical": ["float32 raw-window communication baseline"],
        },
    }
    save_json(result, Path(args.output_json))
    print("saved:", args.output_json)


if __name__ == "__main__":
    main()
