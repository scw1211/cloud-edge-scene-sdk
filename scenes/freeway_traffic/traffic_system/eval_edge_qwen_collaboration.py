"""用途：评估 MLP 主判与边缘 Qwen 低置信复核的选择性协同效果。"""

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from traffic_system.decision_utils import (
    ACTION_TOKEN_TO_DECISION,
    load_json,
    read_jsonl,
    save_json,
)
from traffic_system.edge_student import load_student_model, predict_student


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECISION_TO_ACTION_TOKEN = {
    decision: token for token, decision in ACTION_TOKEN_TO_DECISION.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate confidence-gated MLP and Qwen edge collaboration."
    )
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--qwen_result_json", required=True)
    parser.add_argument(
        "--edge_model_json",
        default="models/edge_student_freeway_joint_metis4.json",
    )
    parser.add_argument("--confidence_threshold", type=float, default=0.75)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def percentile(values: List[float], ratio: float) -> float:
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * ratio))
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def latency_summary(values: List[float]) -> Dict[str, float]:
    return {
        "average_ms": round(statistics.fmean(values), 4),
        "p50_ms": round(percentile(values, 0.50), 4),
        "p95_ms": round(percentile(values, 0.95), 4),
        "max_ms": round(max(values), 4),
    }


def safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def main() -> None:
    args = parse_args()
    rows = read_jsonl(resolve_path(args.dataset_jsonl))
    qwen_result = load_json(resolve_path(args.qwen_result_json))
    qwen_samples = {
        str(sample.get("event_id")): sample
        for sample in qwen_result.get("samples", [])
        if isinstance(sample, dict)
    }
    missing = [str(row.get("event_id")) for row in rows if str(row.get("event_id")) not in qwen_samples]
    if missing:
        raise ValueError("Qwen result is missing {} dataset events.".format(len(missing)))

    edge_model = load_student_model(resolve_path(args.edge_model_json))
    samples: List[Dict[str, Any]] = []
    edge_correct = 0
    qwen_correct = 0
    cascade_correct = 0
    invoked = 0
    disagreements = 0
    invoked_edge_correct = 0
    invoked_qwen_correct = 0
    estimated_latencies: List[float] = []

    for row in rows:
        event_path = resolve_path(str(row["event_path"]))
        event = load_json(event_path)
        started = time.perf_counter()
        edge_decision, edge_confidence, _ = predict_student(event, edge_model)
        edge_latency_ms = (time.perf_counter() - started) * 1000.0
        edge_token = DECISION_TO_ACTION_TOKEN[edge_decision]
        target = str(row["target"]).strip().upper()
        qwen_sample = qwen_samples[str(row["event_id"])]
        qwen_token = str(qwen_sample.get("prediction", "")).strip().upper()
        qwen_latency_ms = float(qwen_sample.get("total_latency_ms", 0.0))
        qwen_invoked = edge_confidence < args.confidence_threshold
        selected_token = qwen_token if qwen_invoked else edge_token
        disagreement = qwen_invoked and qwen_token != edge_token

        edge_is_correct = edge_token == target
        qwen_is_correct = qwen_token == target
        selected_is_correct = selected_token == target
        edge_correct += int(edge_is_correct)
        qwen_correct += int(qwen_is_correct)
        cascade_correct += int(selected_is_correct)
        invoked += int(qwen_invoked)
        disagreements += int(disagreement)
        invoked_edge_correct += int(qwen_invoked and edge_is_correct)
        invoked_qwen_correct += int(qwen_invoked and qwen_is_correct)
        estimated_latency_ms = edge_latency_ms + (qwen_latency_ms if qwen_invoked else 0.0)
        estimated_latencies.append(estimated_latency_ms)
        samples.append(
            {
                "event_id": row["event_id"],
                "target": target,
                "edge_token": edge_token,
                "edge_confidence": round(edge_confidence, 6),
                "qwen_token": qwen_token,
                "qwen_invoked": qwen_invoked,
                "model_disagreement": disagreement,
                "selected_token": selected_token,
                "selected_correct": selected_is_correct,
                "estimated_latency_ms": round(estimated_latency_ms, 4),
            }
        )

    count = len(rows)
    result = {
        "task": "selective_edge_model_collaboration",
        "dataset_jsonl": str(resolve_path(args.dataset_jsonl)),
        "qwen_result_json": str(resolve_path(args.qwen_result_json)),
        "edge_model_json": str(resolve_path(args.edge_model_json)),
        "confidence_threshold": args.confidence_threshold,
        "samples": count,
        "accuracy": {
            "edge_mlp": safe_rate(edge_correct, count),
            "qwen_all_samples": safe_rate(qwen_correct, count),
            "selective_cascade": safe_rate(cascade_correct, count),
        },
        "selective_path": {
            "qwen_invocations": invoked,
            "qwen_invocation_rate": safe_rate(invoked, count),
            "model_disagreements": disagreements,
            "model_disagreement_rate_all_samples": safe_rate(disagreements, count),
            "model_disagreement_rate_when_invoked": safe_rate(disagreements, invoked),
            "edge_accuracy_when_qwen_invoked": safe_rate(invoked_edge_correct, invoked),
            "qwen_accuracy_when_invoked": safe_rate(invoked_qwen_correct, invoked),
        },
        "estimated_immediate_latency": latency_summary(estimated_latencies),
        "qwen_resource_measurement": {
            "runtime": qwen_result.get("runtime"),
            "peak_server_rss_mb": qwen_result.get("peak_server_rss_mb"),
            "peak_server_pss_mb": qwen_result.get("peak_server_pss_mb"),
            "system_ram_peak_delta_mb": qwen_result.get("system_ram_peak_delta_mb"),
        },
        "records": samples,
    }
    save_json(result, resolve_path(args.output_json))
    print(
        "edge={:.2%} qwen={:.2%} cascade={:.2%} invoked={:.2%} avg={:.2f}ms".format(
            result["accuracy"]["edge_mlp"],
            result["accuracy"]["qwen_all_samples"],
            result["accuracy"]["selective_cascade"],
            result["selective_path"]["qwen_invocation_rate"],
            result["estimated_immediate_latency"]["average_ms"],
        )
    )


if __name__ == "__main__":
    main()
