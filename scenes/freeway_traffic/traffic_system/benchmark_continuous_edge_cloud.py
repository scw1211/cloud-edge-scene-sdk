"""用途：连续测量 ASTGCN、Student、HTTP 云端决策组成的完整闭环性能。"""

import argparse
import concurrent.futures
import json
import os
import resource
import statistics
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traffic_system.benchmark_utils import (  # noqa: E402
    SystemMemorySampler,
    build_payload,
    post_json,
    safe_float,
    summarize,
)
from traffic_system.autonomy import run_autonomy_core  # noqa: E402
from traffic_system.conformal_risk import load_risk_calibrator  # noqa: E402
from traffic_system.defer_gate import (  # noqa: E402
    GATE_CLASSES,
    build_gate_features,
    load_defer_gate,
    predict_defer_gate,
)
from traffic_system.edge_student import load_student_model, predict_student  # noqa: E402
from traffic_system.decision_utils import (  # noqa: E402
    DECISION_CLASSES,
    build_decision_from_action_token,
    build_decision_from_student_class,
    extract_feature_vector,
    rule_teacher_decision,
    save_json,
)
from traffic_system.edge_orchestrator import PendingReviewQueue  # noqa: E402
from traffic_system.edge_qwen_action_infer import build_action_prompt, request_action_token  # noqa: E402
from traffic_system.evaluate_future_truth_policy import (  # noqa: E402
    load_evaluation_arrays,
    make_event as make_truth_event,
    one_hot_probabilities,
)
from traffic_system.generate_joint_edge_events import parse_sample_spec  # noqa: E402
from traffic_system.infer_joint_risk_astgcn import (  # noqa: E402
    build_control_capabilities,
    build_model_from_checkpoint,
    build_top_nodes,
    load_adjacency,
    load_config,
    load_inference_arrays,
    region_upload_policy,
    summarize_region,
    torch_load_trusted,
)
from traffic_system.risk_labels import RISK_CLASSES, denormalize  # noqa: E402
from traffic_system.scheduler import AdaptiveScheduler, NetworkSnapshot  # noqa: E402
from traffic_system.train_joint_risk_astgcn import clip_physical_state, select_device  # noqa: E402


NETWORK_PROFILES = {
    "normal": NetworkSnapshot(True, 15.0, 3.0, 0.0, 1.0),
    "mild": NetworkSnapshot(True, 55.0, 10.0, 0.01, 5.0),
    "severe": NetworkSnapshot(True, 160.0, 30.0, 0.10, 30.0),
    "outage": NetworkSnapshot(False, 0.0, 0.0, 1.0, 0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure a continuous Jetson-to-cloud decision loop.")
    parser.add_argument("--config", default="configurations/PEMS08_astgcn.conf")
    parser.add_argument("--data_npz", default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz")
    parser.add_argument("--risk_labels", default="datasets/risk_labels_pems08_metis4.npz")
    parser.add_argument("--risk_calibrator", default="models/region_risk_conformal.json")
    parser.add_argument(
        "--checkpoint",
        default="experiments/PEMS08/joint_risk_astgcn_metis4_flowprio2_frozen/best.pt",
    )
    parser.add_argument("--student_model", default="models/edge_student_freeway_joint_metis4.json")
    parser.add_argument("--defer_gate", default="models/edge_defer_gate.npz")
    parser.add_argument("--disable_defer_gate", action="store_true")
    parser.add_argument("--url", default="http://192.168.31.135:18080/api/v1/traffic/decision")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--samples", default="800:825")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--edge_qwen_url", default="")
    parser.add_argument("--qwen_timeout", type=float, default=0.5)
    parser.add_argument("--qwen_confidence_threshold", type=float, default=0.75)
    parser.add_argument("--qwen_decision_confidence", type=float, default=0.85)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--autonomy_on_failure", action="store_true")
    parser.add_argument("--network_profile", choices=sorted(NETWORK_PROFILES), default="normal")
    parser.add_argument("--adaptive_schedule", action="store_true")
    parser.add_argument("--network_rtt_ms", type=float, default=None)
    parser.add_argument("--network_jitter_ms", type=float, default=None)
    parser.add_argument("--network_loss_rate", type=float, default=None)
    parser.add_argument("--cloud_queue_ms", type=float, default=None)
    parser.add_argument("--scheduler_confidence_threshold", type=float, default=0.70)
    parser.add_argument("--scheduler_edge_compute_ms", type=float, default=52.0)
    parser.add_argument("--scheduler_cloud_compute_ms", type=float, default=12.0)
    parser.add_argument("--pending_queue", default="runtime/benchmark_pending_cloud_reviews.jsonl")
    parser.add_argument("--output_json", default="results/edge/continuous_edge_cloud_jetson02.json")
    return parser.parse_args()


def current_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 4)


def latency_stability(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "first_quarter_average_ms": 0.0,
            "last_quarter_average_ms": 0.0,
            "average_drift_percent": 0.0,
            "coefficient_of_variation": 0.0,
        }
    quarter = max(1, len(values) // 4)
    first_average = statistics.fmean(values[:quarter])
    last_average = statistics.fmean(values[-quarter:])
    mean_value = statistics.fmean(values)
    drift = (last_average - first_average) / first_average if first_average else 0.0
    variation = statistics.pstdev(values) / mean_value if len(values) > 1 and mean_value else 0.0
    return {
        "first_quarter_average_ms": round(first_average, 6),
        "last_quarter_average_ms": round(last_average, 6),
        "average_drift_percent": round(drift * 100.0, 6),
        "coefficient_of_variation": round(variation, 6),
    }


def build_future_references(
    args: argparse.Namespace,
    reference_arrays: Dict[str, Any],
    sample_ids: List[int],
    partitions: List[List[int]],
    adj_mx: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    if partitions != reference_arrays["label_partitions"]:
        raise ValueError("Checkpoint and future-reference partitions differ.")
    capabilities = [
        build_control_capabilities(partitions, adj_mx, partition_id)
        for partition_id in range(len(partitions))
    ]
    references: Dict[str, Dict[str, Any]] = {}
    for sample_id in sorted(set(sample_ids)):
        node_probs = one_hot_probabilities(
            reference_arrays["node_labels"][sample_id], len(RISK_CLASSES)
        )
        region_probs = one_hot_probabilities(
            reference_arrays["region_labels"][sample_id], len(RISK_CLASSES)
        )
        for partition_id in range(len(partitions)):
            event = make_truth_event(
                args.split,
                sample_id,
                partition_id,
                partitions,
                node_probs,
                region_probs,
                reference_arrays["split_target"][sample_id],
                capabilities[partition_id],
                args.top_k,
                "future_observation_fcm_reference",
            )
            references[str(event["event_id"])] = rule_teacher_decision(
                event, "future_truth_policy_reference"
            )
    return references


def network_snapshot_from_args(args: argparse.Namespace) -> NetworkSnapshot:
    base = NETWORK_PROFILES[args.network_profile]
    return NetworkSnapshot(
        available=base.available,
        rtt_ms=base.rtt_ms if args.network_rtt_ms is None else max(0.0, args.network_rtt_ms),
        jitter_ms=(
            base.jitter_ms if args.network_jitter_ms is None else max(0.0, args.network_jitter_ms)
        ),
        loss_rate=(
            base.loss_rate
            if args.network_loss_rate is None
            else min(1.0, max(0.0, args.network_loss_rate))
        ),
        cloud_queue_ms=(
            base.cloud_queue_ms if args.cloud_queue_ms is None else max(0.0, args.cloud_queue_ms)
        ),
    )


def make_event(
    args: argparse.Namespace,
    sample_id: int,
    partition_id: int,
    model: torch.nn.Module,
    split_x: np.ndarray,
    arrays: Dict[str, np.ndarray],
    partitions: List[List[int]],
    adj_mx: np.ndarray,
    adj_filename: str,
    device: torch.device,
    risk_calibrator: Dict[str, Any],
) -> Tuple[Dict[str, Any], float, float]:
    perception_started = time.perf_counter()
    x = torch.from_numpy(split_x[sample_id : sample_id + 1].astype(np.float32)).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward_started = time.perf_counter()
    with torch.no_grad():
        outputs = model(x)
    for output_name, output_tensor in outputs.items():
        if isinstance(output_tensor, torch.Tensor) and not torch.isfinite(output_tensor).all():
            raise RuntimeError(
                "ASTGCN produced NaN/Inf in {}; refuse to emit a risk decision.".format(output_name)
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - forward_started) * 1000.0

    forecast_norm = outputs["forecast"].detach().cpu().numpy()[0]
    forecast_raw = clip_physical_state(
        denormalize(forecast_norm[None, ...], arrays["mean"], arrays["std"])
    )[0]
    node_probs = F.softmax(outputs["node_logits"], dim=-1).detach().cpu().numpy()[0]
    region_probs_all = F.softmax(outputs["region_logits"], dim=-1).detach().cpu().numpy()[0]
    managed_nodes = partitions[partition_id]
    summary = summarize_region(
        managed_nodes,
        node_probs,
        region_probs_all[partition_id],
        risk_calibrator,
    )
    severe_count = int(summary["node_risk_counts"].get("severe", 0))
    high_count = int(summary["node_risk_counts"].get("high", 0))
    upload_required, upload_level = region_upload_policy(
        summary["region_risk_level"],
        summary["max_node_risk_level"],
        severe_count,
        high_count,
    )
    perception_ms = (time.perf_counter() - perception_started) * 1000.0
    edge_id = "edge_node_{}".format(partition_id)
    event = {
        "scene": "freeway_traffic_management",
        "task": "edge_freeway_congestion_risk_assessment",
        "dataset": "PEMS08",
        "model": "joint_astgcn_forecast_node_region_risk",
        "risk_source": "joint_astgcn_encoder_node_region_heads",
        "risk_calibrator": args.risk_calibrator,
        "event_id": "freeway_{}_sample_{:04d}_{}".format(args.split, sample_id, edge_id),
        "edge_id": edge_id,
        "region_id": "region_{}".format(partition_id),
        "partition_id": partition_id,
        "num_partitions": len(partitions),
        "sample_split": args.split,
        "sample_id": sample_id,
        "device": str(device),
        "model_forward_latency_ms": round(forward_ms, 6),
        "inference_latency_ms": round(perception_ms, 6),
        "time_step_minutes": 5,
        "prediction_steps": int(forecast_raw.shape[-1]),
        "prediction_horizon_minutes": int(forecast_raw.shape[-1]) * 5,
        "managed_node_ids": managed_nodes,
        "adjacency_file": adj_filename,
        "control_capabilities": build_control_capabilities(partitions, adj_mx, partition_id),
        "region_summary": summary,
        "upload_required": upload_required,
        "upload_level": upload_level,
        "top_k_risk_nodes": build_top_nodes(managed_nodes, node_probs, forecast_raw, args.top_k),
    }
    return event, forward_ms, perception_ms


def run_one(
    args: argparse.Namespace,
    run_id: int,
    sample_id: int,
    partition_id: int,
    model: torch.nn.Module,
    student_model: Dict[str, Any],
    split_x: np.ndarray,
    arrays: Dict[str, np.ndarray],
    partitions: List[List[int]],
    adj_mx: np.ndarray,
    adj_filename: str,
    device: torch.device,
    reference_labels: Dict[str, Dict[str, Any]],
    network: NetworkSnapshot,
    scheduler: AdaptiveScheduler,
    pending_queue: PendingReviewQueue,
    risk_calibrator: Dict[str, Any],
    defer_gate: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    loop_started = time.perf_counter()
    event, forward_ms, perception_ms = make_event(
        args,
        sample_id,
        partition_id,
        model,
        split_x,
        arrays,
        partitions,
        adj_mx,
        adj_filename,
        device,
        risk_calibrator,
    )
    student_started = time.perf_counter()
    student_class, student_confidence, _ = predict_student(event, student_model)
    student_ms = (time.perf_counter() - student_started) * 1000.0
    student_decision = build_decision_from_student_class(event, student_class, student_confidence)
    rule_decision = rule_teacher_decision(event, decision_source="local_safety_policy")
    gate_started = time.perf_counter()
    if defer_gate is not None:
        base_vector, feature_names = extract_feature_vector(event)
        if list(feature_names) != list(defer_gate["base_feature_names"]):
            raise ValueError("Defer gate base feature schema mismatch")
        gate_features = build_gate_features(
            np.asarray([base_vector], dtype=np.float64),
            np.asarray([DECISION_CLASSES.index(str(rule_decision["decision"]))]),
            np.asarray([DECISION_CLASSES.index(student_class)]),
            np.asarray([student_confidence]),
        )
        gate_choices, gate_confidences = predict_defer_gate(gate_features, defer_gate)
        gate_choice = GATE_CLASSES[int(gate_choices[0])]
        gate_confidence = float(gate_confidences[0])
    else:
        gate_choice = "edge_student"
        gate_confidence = float(student_confidence)
    gate_ms = (time.perf_counter() - gate_started) * 1000.0
    defer_recommended = gate_choice == "defer_cloud"
    selected_edge_decision = (
        student_decision if gate_choice == "edge_student" else rule_decision
    )
    selected_local_class = str(selected_edge_decision["decision"])
    qwen_triggered = bool(
        args.edge_qwen_url
        and (defer_recommended or student_confidence < args.qwen_confidence_threshold)
    )

    def call_qwen() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            return request_action_token(
                args.edge_qwen_url,
                build_action_prompt(event, "bitpacked_decimal"),
                args.qwen_timeout,
                prompt_format="raw_task",
            ), None
        except Exception as exc:  # noqa: BLE001
            return None, "{}: {}".format(type(exc).__name__, exc)

    body = build_payload(
        "continuous-{:04d}".format(run_id),
        event,
        perception_ms,
        student_ms,
        compact_event=True,
    )

    if args.adaptive_schedule:
        schedule = scheduler.schedule(
            event,
            1.0 if defer_gate is not None else student_confidence,
            network,
            defer_recommended=defer_recommended,
            selective_defer=defer_gate is not None,
        )
    else:
        schedule = None
    route = schedule.route if schedule is not None else "cloud_sync"

    def call_cloud() -> Tuple[Dict[str, Any], Optional[str], float]:
        network_started = time.perf_counter()
        try:
            response_value = post_json(args.url, body, args.timeout)
            return response_value, None, (time.perf_counter() - network_started) * 1000.0
        except Exception as exc:  # noqa: BLE001
            return {}, "{}: {}".format(type(exc).__name__, exc), (
                time.perf_counter() - network_started
            ) * 1000.0

    path_started = time.perf_counter()
    response: Dict[str, Any] = {}
    cloud_error: Optional[str] = None
    round_trip_ms = 0.0
    cloud_attempted = route == "cloud_sync"
    async_review_queued = False
    if route == "cloud_sync" and qwen_triggered:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            qwen_future = executor.submit(call_qwen)
            cloud_future = executor.submit(call_cloud)
            qwen_inference, qwen_error = qwen_future.result()
            response, cloud_error, round_trip_ms = cloud_future.result()
    elif route == "cloud_sync":
        qwen_inference = None
        qwen_error = None
        response, cloud_error, round_trip_ms = call_cloud()
    elif qwen_triggered:
        qwen_inference, qwen_error = call_qwen()
    else:
        qwen_inference = None
        qwen_error = None

    qwen_latency_ms = 0.0
    qwen_action_token = None
    model_disagreement = False
    if qwen_inference is not None:
        qwen_action_token = qwen_inference["action_token"]
        qwen_latency_ms = float(qwen_inference["latency_ms"])
        selected_edge_decision = build_decision_from_action_token(
            event,
            qwen_action_token,
            confidence=args.qwen_decision_confidence,
        )
        model_disagreement = selected_edge_decision["decision"] != selected_local_class

    if args.adaptive_schedule and model_disagreement and route == "edge_only":
        schedule = scheduler.schedule(
            event,
            1.0 if defer_gate is not None else student_confidence,
            network,
            model_disagreement=True,
            defer_recommended=defer_recommended,
            selective_defer=defer_gate is not None,
        )
        route = schedule.route
    if route == "cloud_async":
        pending_queue.append(
            {
                "event_id": event.get("event_id"),
                "queued_at_ns": time.time_ns(),
                "network_profile": args.network_profile,
                "payload": json.loads(body.decode("utf-8")),
            }
        )
        async_review_queued = True
    path_decision_ms = (time.perf_counter() - path_started) * 1000.0

    if cloud_error is not None and not args.autonomy_on_failure:
        raise RuntimeError(cloud_error)
    cloud_metrics = response.get("cloud_metrics", {})
    cloud_decision = response.get("decision", {})
    if route == "cloud_sync" and cloud_error is None:
        final_decision, autonomy_metadata = run_autonomy_core(
            event,
            cloud_mode="normal",
            cloud_decision=cloud_decision,
        )
    elif route == "cloud_sync":
        if qwen_triggered and qwen_error is None:
            final_decision = selected_edge_decision
            autonomy_metadata = {
                "autonomy_triggered": True,
                "used_local_qwen_decision": True,
            }
        else:
            final_decision, autonomy_metadata = run_autonomy_core(
                event,
                cloud_mode="down",
                cloud_decision=None,
            )
    elif route == "local_autonomy":
        if qwen_triggered and qwen_error is None:
            final_decision = selected_edge_decision
            autonomy_metadata = {
                "autonomy_triggered": True,
                "used_local_qwen_decision": True,
            }
        else:
            final_decision, autonomy_metadata = run_autonomy_core(
                event,
                cloud_mode="down",
                cloud_decision=None,
            )
    elif route == "cloud_async":
        final_decision, autonomy_metadata = run_autonomy_core(
            event,
            cloud_mode="degraded_async",
            cloud_decision=None,
        )
        autonomy_metadata["async_cloud_review_queued"] = async_review_queued
    else:
        final_decision = selected_edge_decision
        autonomy_metadata = {
            "autonomy_triggered": False,
            "async_cloud_review_queued": False,
        }
    reference = reference_labels.get(str(event["event_id"]))
    reference_source = "future_observation_fcm_reference"
    if not isinstance(reference, dict):
        reference = rule_teacher_decision(event, decision_source="availability_reference")
        reference_source = "predicted_event_rule_fallback"
    expected_actions = {
        str(action.get("type"))
        for action in reference.get("actions", [])
        if isinstance(action, dict)
    }
    actual_actions = {
        str(action.get("type"))
        for action in final_decision.get("actions", [])
        if isinstance(action, dict)
    }
    decision_correct = bool(
        final_decision.get("safe")
        and final_decision.get("decision") == reference.get("decision")
    )
    action_type_match = actual_actions == expected_actions
    basic_business_functional = bool(
        final_decision.get("safe")
        and final_decision.get("decision") in DECISION_CLASSES
        and isinstance(final_decision.get("actions"), list)
    )
    total_ms = (time.perf_counter() - loop_started) * 1000.0
    schedule_data = (
        schedule.to_dict()
        if schedule is not None
        else {
            "route": "cloud_sync",
            "reason": "forced synchronous benchmark path",
            "cloud_requested": True,
            "waits_for_cloud": True,
            "network": network.__dict__,
        }
    )
    return {
        "run_id": run_id,
        "sample_id": sample_id,
        "partition_id": partition_id,
        "edge_id": event["edge_id"],
        "region_id": event["region_id"],
        "upload_required": event["upload_required"],
        "upload_level": event["upload_level"],
        "model_forward_latency_ms": round(forward_ms, 6),
        "edge_perception_latency_ms": round(perception_ms, 6),
        "edge_student_latency_ms": round(student_ms, 6),
        "edge_defer_gate_latency_ms": round(gate_ms, 6),
        "defer_gate_choice": gate_choice,
        "defer_gate_confidence": round(gate_confidence, 6),
        "qwen_triggered": qwen_triggered,
        "qwen_latency_ms": round(qwen_latency_ms, 6),
        "qwen_action_token": qwen_action_token,
        "qwen_error": qwen_error,
        "model_disagreement": model_disagreement,
        "selected_edge_decision": selected_edge_decision.get("decision"),
        "schedule": schedule_data,
        "route": route,
        "cloud_attempted": cloud_attempted,
        "async_review_queued": async_review_queued,
        "round_trip_latency_ms": round(round_trip_ms, 6),
        "parallel_decision_latency_ms": round(path_decision_ms, 6),
        "cloud_decision_latency_ms": safe_float(cloud_metrics.get("cloud_decision_latency_ms")),
        "total_e2e_latency_ms": round(total_ms, 6),
        "payload_bytes": len(body),
        "student_decision": student_class,
        "student_confidence": round(student_confidence, 6),
        "cloud_success": bool(cloud_attempted and cloud_error is None),
        "cloud_response_received": bool(cloud_attempted and cloud_error is None),
        "cloud_error": cloud_error,
        "autonomy_triggered": bool(autonomy_metadata.get("autonomy_triggered")),
        "final_decision": final_decision.get("decision"),
        "reference_source": reference_source,
        "reference_decision": reference.get("decision"),
        "functional_decision": decision_correct,
        "basic_business_functional": basic_business_functional,
        "action_type_match": action_type_match,
        "safe": bool(final_decision.get("safe")),
    }


def main() -> None:
    args = parse_args()
    if args.runs <= 0 or args.warmup < 0:
        raise ValueError("runs must be positive and warmup non-negative")
    network = network_snapshot_from_args(args)
    scheduler = AdaptiveScheduler(
        confidence_threshold=args.scheduler_confidence_threshold,
        edge_compute_ms=args.scheduler_edge_compute_ms,
        cloud_compute_ms=args.scheduler_cloud_compute_ms,
    )
    pending_queue = PendingReviewQueue(Path(args.pending_queue))
    pending_queue.replace([])
    system_memory = SystemMemorySampler(0.02)
    system_memory.start()
    device = select_device(args.device)
    torch.set_num_threads(args.torch_threads)
    config = load_config(args.config)
    arrays = load_inference_arrays(Path(args.data_npz), args.split)
    reference_arrays = load_evaluation_arrays(
        Path(args.data_npz), Path(args.risk_labels), args.split
    )
    checkpoint = torch_load_trusted(Path(args.checkpoint), device)
    adj_mx, adj_filename = load_adjacency(config)
    model = build_model_from_checkpoint(config, arrays, adj_mx, checkpoint, device)
    risk_calibrator = load_risk_calibrator(Path(args.risk_calibrator))
    student_model = load_student_model(Path(args.student_model))
    defer_gate = None if args.disable_defer_gate else load_defer_gate(Path(args.defer_gate))
    split_x = arrays["split_x"]
    partitions = [[int(node) for node in part] for part in checkpoint["partitions"]]
    sample_ids = parse_sample_spec(args.samples)
    if not sample_ids or min(sample_ids) < 0 or max(sample_ids) >= len(split_x):
        raise ValueError("sample selection is empty or outside {} split".format(args.split))
    reference_labels = build_future_references(
        args,
        reference_arrays,
        sample_ids,
        partitions,
        adj_mx,
    )

    samples = []
    failures = []
    measurement_started = None
    for index in range(args.warmup + args.runs):
        measured = index >= args.warmup
        if index == args.warmup:
            pending_queue.replace([])
            measurement_started = time.perf_counter()
        sample_id = sample_ids[index % len(sample_ids)]
        partition_id = index % len(partitions)
        try:
            row = run_one(
                args,
                index,
                sample_id,
                partition_id,
                model,
                student_model,
                split_x,
                arrays,
                partitions,
                adj_mx,
                adj_filename,
                device,
                reference_labels,
                network,
                scheduler,
                pending_queue,
                risk_calibrator,
                defer_gate,
            )
            if measured:
                samples.append(row)
                print(
                    "[{}/{}] sample={} edge={} total={:.3f}ms".format(
                        len(samples), args.runs, sample_id, partition_id, row["total_e2e_latency_ms"]
                    ),
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            if measured:
                failures.append({"run_id": index, "error": "{}: {}".format(type(exc).__name__, exc)})

    system_memory.stop()
    measurement_seconds = (
        time.perf_counter() - measurement_started if measurement_started is not None else 0.0
    )

    total_values = [row["total_e2e_latency_ms"] for row in samples]
    under_200 = sum(value <= 200.0 for value in total_values)
    cloud_attempt_count = sum(row["cloud_attempted"] for row in samples)
    route_counts = {
        route: sum(row["route"] == route for row in samples)
        for route in ("cloud_sync", "cloud_async", "edge_only", "local_autonomy")
    }
    result = {
        "task": "continuous_edge_cloud_closed_loop_benchmark",
        "measurement_scope": (
            "single wall-clock interval from input-window tensor creation through ASTGCN, event construction, "
            "MLP Student, confidence-gated edge Qwen, adaptive route selection, and the selected synchronous "
            "cloud, asynchronous queue, edge-only, or local-autonomy path"
        ),
        "segmented_measurement": False,
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cloud_url": args.url,
        "network_profile": args.network_profile,
        "network_snapshot": network.__dict__,
        "adaptive_schedule": args.adaptive_schedule,
        "scheduler": {
            "confidence_threshold": args.scheduler_confidence_threshold,
            "edge_compute_ms": args.scheduler_edge_compute_ms,
            "cloud_compute_ms": args.scheduler_cloud_compute_ms,
        },
        "risk_calibrator": args.risk_calibrator,
        "defer_gate": None if defer_gate is None else args.defer_gate,
        "autonomy_on_failure": args.autonomy_on_failure,
        "reference": (
            "observed future flow/occupancy/speed -> frozen FCM risk labels -> fixed safety policy"
        ),
        "runs": args.runs,
        "warmup": args.warmup,
        "success_count": len(samples),
        "failure_count": len(failures),
        "success_rate": round(len(samples) / args.runs, 6),
        "under_200ms_rate": round(under_200 / args.runs, 6),
        "cloud_attempt_count": cloud_attempt_count,
        "cloud_success_rate": round(
            sum(row["cloud_success"] for row in samples) / cloud_attempt_count, 6
        ) if cloud_attempt_count else None,
        "cloud_request_rate": round(
            sum(row["route"] in {"cloud_sync", "cloud_async"} for row in samples)
            / len(samples),
            6,
        ) if samples else 0.0,
        "route_counts": route_counts,
        "route_rates": {
            route: round(count / len(samples), 6) if samples else 0.0
            for route, count in route_counts.items()
        },
        "pending_cloud_review_count": len(pending_queue.rows()),
        "pending_queue": str(Path(args.pending_queue)),
        "autonomy_trigger_rate": round(
            sum(row["autonomy_triggered"] for row in samples) / len(samples), 6
        ) if samples else 0.0,
        "qwen_trigger_rate": round(
            sum(row["qwen_triggered"] for row in samples) / len(samples), 6
        ) if samples else 0.0,
        "qwen_success_rate_when_triggered": round(
            sum(row["qwen_triggered"] and row["qwen_error"] is None for row in samples)
            / max(1, sum(row["qwen_triggered"] for row in samples)),
            6,
        ) if samples else 0.0,
        "edge_model_disagreement_rate": round(
            sum(row["model_disagreement"] for row in samples) / len(samples), 6
        ) if samples else 0.0,
        "defer_gate_choice_counts": {
            name: sum(row["defer_gate_choice"] == name for row in samples)
            for name in GATE_CLASSES
        },
        "decision_accuracy": round(
            sum(row["functional_decision"] for row in samples) / args.runs, 6
        ),
        "business_availability": round(
            sum(row["basic_business_functional"] for row in samples) / args.runs, 6
        ),
        "basic_business_availability": round(
            sum(row["basic_business_functional"] for row in samples) / args.runs, 6
        ),
        "business_availability_definition": (
            "a safe protocol-valid local or cloud decision is returned; all scheduled runs are in the "
            "denominator, including runtime or HTTP failures"
        ),
        "action_type_match_rate": round(
            sum(row["action_type_match"] for row in samples) / args.runs, 6
        ),
        "meets_0_2s_average": bool(total_values and statistics.fmean(total_values) <= 200.0),
        "process_max_rss_mb": current_rss_mb(),
        "system_ram_baseline_mb": system_memory.baseline_mb,
        "system_ram_peak_mb": system_memory.peak_mb,
        "system_ram_peak_delta_mb": system_memory.peak_delta_mb,
        "measurement_wall_seconds": round(measurement_seconds, 6),
        "throughput_runs_per_second": round(args.runs / measurement_seconds, 6)
        if measurement_seconds
        else 0.0,
        "latency_stability": latency_stability(total_values),
        "latency": {
            "model_forward": summarize(row["model_forward_latency_ms"] for row in samples),
            "edge_perception": summarize(row["edge_perception_latency_ms"] for row in samples),
            "edge_student": summarize(row["edge_student_latency_ms"] for row in samples),
            "edge_defer_gate": summarize(
                row["edge_defer_gate_latency_ms"] for row in samples
            ),
            "edge_qwen_triggered": summarize(
                row["qwen_latency_ms"]
                for row in samples
                if row["qwen_triggered"] and row["qwen_error"] is None
            ),
            "round_trip": summarize(
                row["round_trip_latency_ms"] for row in samples if row["cloud_attempted"]
            ),
            "parallel_edge_qwen_cloud": summarize(
                row["parallel_decision_latency_ms"] for row in samples
            ),
            "decision_path": summarize(
                row["parallel_decision_latency_ms"] for row in samples
            ),
            "cloud_decision": summarize(
                row["cloud_decision_latency_ms"]
                for row in samples
                if row["cloud_response_received"]
            ),
            "total_e2e": summarize(total_values),
        },
        "payload_average_bytes": round(statistics.fmean(row["payload_bytes"] for row in samples), 3)
        if samples
        else 0.0,
        "failures": failures,
        "samples": samples,
    }
    save_json(result, Path(args.output_json))
    print(json.dumps({key: result[key] for key in (
        "success_rate", "cloud_success_rate", "cloud_request_rate", "route_counts",
        "pending_cloud_review_count", "autonomy_trigger_rate", "qwen_trigger_rate",
        "qwen_success_rate_when_triggered", "edge_model_disagreement_rate", "decision_accuracy",
        "business_availability", "action_type_match_rate",
        "under_200ms_rate", "meets_0_2s_average", "process_max_rss_mb",
        "system_ram_baseline_mb", "system_ram_peak_mb", "system_ram_peak_delta_mb",
        "measurement_wall_seconds", "throughput_runs_per_second", "latency_stability", "latency"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
