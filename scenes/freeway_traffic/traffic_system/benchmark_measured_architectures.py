"""用途：用同一批交通窗口实测集中式、单边缘、固定云边和动态云边四种架构。"""

import argparse
import base64
import concurrent.futures
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from traffic_system.benchmark_utils import build_payload, post_json, summarize
from traffic_system.decision_utils import (
    DECISION_CLASSES,
    build_decision_from_student_class,
    rule_teacher_decision,
    save_json,
)
from traffic_system.edge_student import load_student_model, predict_student
from traffic_system.evaluate_future_truth_policy import (
    CRITICAL_DECISIONS,
    classification_report,
    make_event,
    one_hot_probabilities,
    selected_sample_ids,
    stratified_bootstrap_accuracy_ci,
)
from traffic_system.infer_joint_risk_astgcn import (
    build_control_capabilities,
    build_model_from_checkpoint,
    ensure_finite_outputs,
    load_adjacency,
    load_config,
    torch_load_trusted,
)
from traffic_system.risk_labels import RISK_CLASSES, denormalize, enable_numpy_pickle_compatibility
from traffic_system.scheduler import AdaptiveScheduler, NetworkSnapshot
from traffic_system.train_joint_risk_astgcn import clip_physical_state, select_device


ARCHITECTURES = ("centralized_cloud", "single_edge", "fixed_cloud_edge", "adaptive_cloud_edge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure four traffic architectures with one workload.")
    parser.add_argument("--config", default="configurations/PEMS08_astgcn.conf")
    parser.add_argument("--data_npz", default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz")
    parser.add_argument("--risk_labels", default="datasets/risk_labels_pems08_metis4.npz")
    parser.add_argument(
        "--checkpoint",
        default="experiments/PEMS08/joint_risk_astgcn_metis4_flowprio2_frozen/best.pt",
    )
    parser.add_argument("--student_model", default="models/edge_student_freeway_joint_metis4.json")
    parser.add_argument("--centralized_url", default="http://192.168.31.135:18081/api/v1/traffic/centralized")
    parser.add_argument("--cloud_url", default="http://192.168.31.135:18080/api/v1/traffic/decision")
    parser.add_argument("--samples", default="0:800:8")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--network_rtt_ms", type=float, default=15.0)
    parser.add_argument("--network_jitter_ms", type=float, default=3.0)
    parser.add_argument("--network_loss_rate", type=float, default=0.0)
    parser.add_argument("--cloud_queue_ms", type=float, default=1.0)
    parser.add_argument("--confidence_threshold", type=float, default=0.70)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_json",
        default="results/edge/measured_architecture_baselines.json",
    )
    parser.add_argument(
        "--report_md",
        default="results/edge/measured_architecture_baselines.md",
    )
    return parser.parse_args()


def load_arrays(data_path: Path, labels_path: Path) -> Dict[str, Any]:
    with np.load(data_path) as data:
        required = ["test_x", "test_target", "mean", "std"]
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError("Missing data arrays: {}".format(", ".join(missing)))
        arrays = {name: data[name] for name in required}
    enable_numpy_pickle_compatibility()
    with np.load(labels_path, allow_pickle=True) as labels:
        required = ["test_node_label", "test_region_label", "partitions"]
        missing = [name for name in required if name not in labels.files]
        if missing:
            raise ValueError("Missing risk labels: {}".format(", ".join(missing)))
        arrays.update(
            {
                "node_labels": labels["test_node_label"],
                "region_labels": labels["test_region_label"],
                "label_partitions": [
                    [int(node_id) for node_id in part]
                    for part in labels["partitions"].tolist()
                ],
            }
        )
    return arrays


def raw_window_payload(request_id: str, sample_id: int, partition_id: int, values: np.ndarray) -> bytes:
    little_endian = np.asarray(values, dtype="<f4")
    payload = {
        "request_id": request_id,
        "sample_split": "test",
        "sample_id": int(sample_id),
        "partition_id": int(partition_id),
        "encoding": "base64_float32_le",
        "input_shape": list(little_endian.shape),
        "input_base64": base64.b64encode(little_endian.tobytes()).decode("ascii"),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def infer_local_event(
    sample_id: int,
    partition_id: int,
    arrays: Dict[str, Any],
    model: torch.nn.Module,
    partitions: Sequence[Sequence[int]],
    capabilities: Sequence[Dict[str, Any]],
    device: torch.device,
    top_k: int,
) -> Tuple[Dict[str, Any], float]:
    tensor = torch.from_numpy(arrays["test_x"][sample_id : sample_id + 1].astype(np.float32)).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        outputs = model(tensor)
    ensure_finite_outputs(outputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    model_ms = (time.perf_counter() - started) * 1000.0
    forecast_raw = clip_physical_state(
        denormalize(outputs["forecast"].detach().cpu().numpy(), arrays["mean"], arrays["std"])
    )[0]
    node_probs = F.softmax(outputs["node_logits"], dim=-1).detach().cpu().numpy()[0]
    region_probs = F.softmax(outputs["region_logits"], dim=-1).detach().cpu().numpy()[0]
    event = make_event(
        "test",
        sample_id,
        partition_id,
        partitions,
        node_probs,
        region_probs,
        forecast_raw,
        capabilities[partition_id],
        top_k,
        "joint_astgcn_edge_prediction",
    )
    return event, model_ms


def reference_decision(
    sample_id: int,
    partition_id: int,
    arrays: Dict[str, Any],
    partitions: Sequence[Sequence[int]],
    capabilities: Sequence[Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    node_probs = one_hot_probabilities(arrays["node_labels"][sample_id], len(RISK_CLASSES))
    region_probs = one_hot_probabilities(arrays["region_labels"][sample_id], len(RISK_CLASSES))
    event = make_event(
        "test",
        sample_id,
        partition_id,
        partitions,
        node_probs,
        region_probs,
        arrays["test_target"][sample_id],
        capabilities[partition_id],
        top_k,
        "future_observation_fcm_reference",
    )
    return rule_teacher_decision(event, "future_truth_policy_reference")


def local_student_decision(event: Dict[str, Any], student_model: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    started = time.perf_counter()
    decision_class, confidence, _ = predict_student(event, student_model)
    decision = build_decision_from_student_class(
        event,
        decision_class,
        confidence,
        "qwen_distilled_edge_student",
    )
    return decision, (time.perf_counter() - started) * 1000.0


def result_row(
    architecture: str,
    sample_id: int,
    partition_id: int,
    reference: Dict[str, Any],
    decision: Optional[Dict[str, Any]],
    latency_ms: float,
    payload_bytes: int,
    cloud_requested: bool,
    cloud_waited: bool,
    model_ms: float = 0.0,
    student_ms: float = 0.0,
    route: str = "",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    decision = decision if isinstance(decision, dict) else {}
    decision_class = str(decision.get("decision", ""))
    functional = bool(
        decision.get("safe")
        and decision_class in DECISION_CLASSES
        and isinstance(decision.get("actions"), list)
    )
    return {
        "architecture": architecture,
        "sample_id": sample_id,
        "partition_id": partition_id,
        "reference_decision": str(reference.get("decision")),
        "decision": decision_class,
        "decision_match": decision_class == str(reference.get("decision")),
        "critical_reference": str(reference.get("decision")) in CRITICAL_DECISIONS,
        "critical_intervention": decision_class in CRITICAL_DECISIONS,
        "functional": functional,
        "safe": bool(decision.get("safe")),
        "latency_ms": round(latency_ms, 6),
        "model_forward_ms": round(model_ms, 6),
        "student_ms": round(student_ms, 6),
        "payload_bytes": int(payload_bytes),
        "cloud_requested": cloud_requested,
        "cloud_waited": cloud_waited,
        "route": route,
        "error": error,
    }


def run_centralized(
    args: argparse.Namespace,
    cases: Sequence[Tuple[int, int, Dict[str, Any]]],
    arrays: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    for run_id, (sample_id, partition_id, reference) in enumerate(cases):
        started = time.perf_counter()
        body = raw_window_payload(
            "architecture-centralized-{:04d}".format(run_id),
            sample_id,
            partition_id,
            arrays["test_x"][sample_id : sample_id + 1],
        )
        try:
            response = post_json(args.centralized_url, body, args.timeout)
            decision = response.get("decision")
            error = None
        except Exception as exc:  # noqa: BLE001
            decision = None
            error = "{}: {}".format(type(exc).__name__, exc)
        rows.append(
            result_row(
                "centralized_cloud",
                sample_id,
                partition_id,
                reference,
                decision,
                (time.perf_counter() - started) * 1000.0,
                len(body),
                True,
                True,
                route="cloud_sync",
                error=error,
            )
        )
    return rows


def run_edge_architecture(
    architecture: str,
    args: argparse.Namespace,
    cases: Sequence[Tuple[int, int, Dict[str, Any]]],
    arrays: Dict[str, Any],
    model: torch.nn.Module,
    student_model: Dict[str, Any],
    partitions: Sequence[Sequence[int]],
    capabilities: Sequence[Dict[str, Any]],
    device: torch.device,
    scheduler: AdaptiveScheduler,
    network: NetworkSnapshot,
) -> List[Dict[str, Any]]:
    rows = []
    async_jobs: List[Tuple[concurrent.futures.Future, int]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for run_id, (sample_id, partition_id, reference) in enumerate(cases):
            started = time.perf_counter()
            event, model_ms = infer_local_event(
                sample_id,
                partition_id,
                arrays,
                model,
                partitions,
                capabilities,
                device,
                args.top_k,
            )
            local_decision, student_ms = local_student_decision(event, student_model)
            body = build_payload(
                "architecture-{}-{:04d}".format(architecture, run_id),
                event,
                model_ms,
                student_ms,
                compact_event=True,
            )
            decision = local_decision
            payload_bytes = 0
            cloud_requested = False
            cloud_waited = False
            route = "edge_only"
            error = None

            if architecture == "fixed_cloud_edge":
                route = "cloud_sync"
                cloud_requested = cloud_waited = True
                payload_bytes = len(body)
                try:
                    response = post_json(args.cloud_url, body, args.timeout)
                    decision = response.get("decision")
                except Exception as exc:  # noqa: BLE001
                    decision = None
                    error = "{}: {}".format(type(exc).__name__, exc)
            elif architecture == "adaptive_cloud_edge":
                _, confidence, _ = predict_student(event, student_model)
                schedule = scheduler.schedule(event, confidence, network)
                route = schedule.route
                cloud_requested = schedule.cloud_requested
                cloud_waited = schedule.waits_for_cloud
                if cloud_requested:
                    payload_bytes = len(body)
                if route == "cloud_sync":
                    try:
                        response = post_json(args.cloud_url, body, args.timeout)
                        decision = response.get("decision")
                    except Exception as exc:  # noqa: BLE001
                        decision = local_decision
                        error = "{}: {}".format(type(exc).__name__, exc)
                elif route == "cloud_async":
                    async_jobs.append((executor.submit(post_json, args.cloud_url, body, args.timeout), len(rows)))
                elif route == "local_autonomy":
                    decision = local_decision

            rows.append(
                result_row(
                    architecture,
                    sample_id,
                    partition_id,
                    reference,
                    decision,
                    (time.perf_counter() - started) * 1000.0,
                    payload_bytes,
                    cloud_requested,
                    cloud_waited,
                    model_ms,
                    student_ms,
                    route,
                    error,
                )
            )
        for future, row_id in async_jobs:
            try:
                future.result()
                rows[row_id]["async_cloud_delivery_success"] = True
            except Exception as exc:  # noqa: BLE001
                rows[row_id]["async_cloud_delivery_success"] = False
                rows[row_id]["async_cloud_error"] = "{}: {}".format(type(exc).__name__, exc)
    return rows


def architecture_summary(rows: Sequence[Dict[str, Any]], bootstrap_samples: int, seed: int) -> Dict[str, Any]:
    true_labels = [row["reference_decision"] for row in rows if row["decision"] in DECISION_CLASSES]
    predictions = [row["decision"] for row in rows if row["decision"] in DECISION_CLASSES]
    classification = classification_report(true_labels, predictions, DECISION_CLASSES) if predictions else {}
    if predictions:
        classification["accuracy_95ci"] = stratified_bootstrap_accuracy_ci(
            true_labels, predictions, bootstrap_samples, seed
        )
    critical_rows = [row for row in rows if row["critical_reference"]]
    payloads = [row["payload_bytes"] for row in rows]
    return {
        "runs": len(rows),
        "success_count": sum(row["functional"] for row in rows),
        "business_availability": round(sum(row["functional"] for row in rows) / len(rows), 6),
        "safe_response_rate": round(sum(row["safe"] for row in rows) / len(rows), 6),
        "decision": classification,
        "critical_intervention_recall": round(
            sum(row["critical_intervention"] for row in critical_rows) / len(critical_rows), 6
        ) if critical_rows else None,
        "latency": summarize(row["latency_ms"] for row in rows),
        "model_forward": summarize(row["model_forward_ms"] for row in rows if row["model_forward_ms"] > 0),
        "student": summarize(row["student_ms"] for row in rows if row["student_ms"] > 0),
        "under_200ms_rate": round(sum(row["latency_ms"] <= 200.0 for row in rows) / len(rows), 6),
        "cloud_request_rate": round(sum(row["cloud_requested"] for row in rows) / len(rows), 6),
        "cloud_wait_rate": round(sum(row["cloud_waited"] for row in rows) / len(rows), 6),
        "average_payload_bytes_per_case": round(statistics.fmean(payloads), 3),
        "total_upload_bytes": int(sum(payloads)),
        "route_counts": {
            route: sum(row["route"] == route for row in rows)
            for route in ("cloud_sync", "cloud_async", "edge_only", "local_autonomy")
        },
        "failure_count": sum(bool(row["error"]) and not row["functional"] for row in rows),
        "failures": [row for row in rows if row["error"]][:20],
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    labels = {
        "centralized_cloud": "集中式云端",
        "single_edge": "单边缘",
        "fixed_cloud_edge": "固定同步云边",
        "adaptive_cloud_edge": "动态云边协同",
    }
    lines = [
        "# 四种架构统一实测基线",
        "",
        "同一设备、同一批 PEMS08 test 窗口、同一未来三变量参考策略。集中式模式实际上传原始 float32 窗口并在云端运行 ASTGCN，不使用估算时延。",
        "",
        "| 架构 | 平均时延 | P95 | ≤200ms | 决策准确率 | 高风险干预召回 | 业务可用率 | 平均上传量 | 上云率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ARCHITECTURES:
        item = result["architectures"][name]
        decision_accuracy = item.get("decision", {}).get("accuracy", 0.0)
        lines.append(
            "| {} | {:.2f} ms | {:.2f} ms | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.0f} B | {:.2%} |".format(
                labels[name],
                item["latency"]["average_ms"],
                item["latency"]["p95_ms"],
                item["under_200ms_rate"],
                decision_accuracy,
                item["critical_intervention_recall"],
                item["business_availability"],
                item["average_payload_bytes_per_case"],
                item["cloud_request_rate"],
            )
        )
    adaptive = result["architectures"]["adaptive_cloud_edge"]
    centralized = result["architectures"]["centralized_cloud"]
    lines.extend(
        [
            "",
            "## 对比结论",
            "",
            "- 动态云边平均端到端时延：{:.2f} ms，满足 0.2 s：{}。".format(
                adaptive["latency"]["average_ms"], "是" if adaptive["latency"]["average_ms"] <= 200 else "否"
            ),
            "- 相比集中式，动态云边上传字节减少：{:.2%}。".format(
                1.0 - adaptive["total_upload_bytes"] / max(1, centralized["total_upload_bytes"])
            ),
            "- 本实验为 1 块物理 Jetson 上的 4 个 METIS 逻辑区域；多物理节点另行报告。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.runs <= 0 or args.warmup < 0 or args.timeout <= 0:
        raise ValueError("runs/timeout must be positive and warmup non-negative")
    torch.set_num_threads(args.torch_threads)
    device = select_device(args.device)
    arrays = load_arrays(Path(args.data_npz), Path(args.risk_labels))
    available_ids = selected_sample_ids(args.samples, arrays["test_x"].shape[0])
    sample_ids = [available_ids[index % len(available_ids)] for index in range(args.runs)]
    config = load_config(args.config)
    checkpoint = torch_load_trusted(Path(args.checkpoint), device)
    adj_mx, _ = load_adjacency(config)
    model = build_model_from_checkpoint(
        config,
        {
            "in_channels": int(arrays["test_x"].shape[2]),
            "output_dim": int(arrays["mean"].shape[2]),
        },
        adj_mx,
        checkpoint,
        device,
    )
    partitions = [[int(node_id) for node_id in part] for part in checkpoint["partitions"]]
    if partitions != arrays["label_partitions"]:
        raise ValueError("Checkpoint and reference-label partitions differ")
    capabilities = [
        build_control_capabilities(partitions, adj_mx, partition_id)
        for partition_id in range(len(partitions))
    ]
    student_model = load_student_model(Path(args.student_model))
    scheduler = AdaptiveScheduler(
        confidence_threshold=args.confidence_threshold,
        edge_compute_ms=74.0,
        cloud_compute_ms=32.0,
    )
    network = NetworkSnapshot(
        available=True,
        rtt_ms=args.network_rtt_ms,
        jitter_ms=args.network_jitter_ms,
        loss_rate=args.network_loss_rate,
        cloud_queue_ms=args.cloud_queue_ms,
    )
    warmup_ids = available_ids[: max(1, args.warmup)]
    for index in range(args.warmup):
        infer_local_event(
            warmup_ids[index % len(warmup_ids)],
            index % len(partitions),
            arrays,
            model,
            partitions,
            capabilities,
            device,
            args.top_k,
        )
    cases = [
        (
            sample_id,
            index % len(partitions),
            reference_decision(
                sample_id,
                index % len(partitions),
                arrays,
                partitions,
                capabilities,
                args.top_k,
            ),
        )
        for index, sample_id in enumerate(sample_ids)
    ]

    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    print("running centralized_cloud", flush=True)
    all_rows["centralized_cloud"] = run_centralized(args, cases, arrays)
    for architecture in ("single_edge", "fixed_cloud_edge", "adaptive_cloud_edge"):
        print("running", architecture, flush=True)
        all_rows[architecture] = run_edge_architecture(
            architecture,
            args,
            cases,
            arrays,
            model,
            student_model,
            partitions,
            capabilities,
            device,
            scheduler,
            network,
        )
    summaries = {
        name: architecture_summary(rows, args.bootstrap_samples, args.seed + index)
        for index, (name, rows) in enumerate(all_rows.items())
    }
    result = {
        "task": "measured_four_architecture_baseline_comparison",
        "measurement_scope": (
            "one wall-clock interval from an available normalized sensor window to a final decision; "
            "centralized mode includes raw-window serialization, HTTP transfer, cloud ASTGCN and coordinator"
        ),
        "reference": "future observed flow/occupancy/speed -> frozen FCM risk -> fixed safety policy",
        "reference_status": "data-driven reference policy, not manual control ground truth",
        "hardware": {
            "client_hostname": platform.node(),
            "client_platform": platform.platform(),
            "local_device": str(device),
            "physical_edge_nodes": 1,
            "logical_edge_regions": len(partitions),
            "centralized_url": args.centralized_url,
            "cloud_url": args.cloud_url,
        },
        "workload": {
            "split": "test",
            "runs_per_architecture": args.runs,
            "warmup_runs": args.warmup,
            "sample_spec": args.samples,
            "sample_ids": sample_ids,
            "partition_assignment": "round_robin_across_metis_regions",
        },
        "network_snapshot": network.__dict__,
        "scheduler_confidence_threshold": args.confidence_threshold,
        "architecture_definitions": {
            "centralized_cloud": "raw sensor window -> HTTP -> cloud ASTGCN/risk/coordinator -> response",
            "single_edge": "edge ASTGCN/risk + Qwen-distilled MLP Student; no cloud",
            "fixed_cloud_edge": "edge ASTGCN/risk/Student -> synchronous cloud coordinator every case",
            "adaptive_cloud_edge": "edge ASTGCN/risk/Student -> deadline-aware sync/async/edge route",
        },
        "architectures": summaries,
        "records": all_rows,
    }
    save_json(result, Path(args.output_json))
    write_markdown(result, Path(args.report_md))
    print("result:", args.output_json)
    print("report:", args.report_md)
    for name in ARCHITECTURES:
        item = summaries[name]
        print(name, item["latency"]["average_ms"], item["decision"].get("accuracy"))


if __name__ == "__main__":
    main()
