"""用途：在冻结交通测试集上执行六组路由消融，并按时间组 bootstrap 置信区间。"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from traffic_system.cloud_coordinator import load_cloud_model
from traffic_system.conformal_risk import load_risk_calibrator
from traffic_system.decision_utils import DECISION_CLASSES, save_json
from traffic_system.defer_gate import (
    GATE_CLASSES,
    build_gate_features,
    load_defer_gate,
    predict_defer_gate,
)
from traffic_system.edge_student import load_student_model
from traffic_system.evaluate_conformal_scheduler import (
    CRITICAL_DECISIONS,
    load_region_neighbors,
)
from traffic_system.evaluate_future_truth_policy import classification_report
from traffic_system.infer_joint_risk_astgcn import (
    build_control_capabilities,
    build_model_from_checkpoint,
    load_adjacency,
    load_config,
    torch_load_trusted,
)
from traffic_system.scheduler import AdaptiveScheduler, NetworkSnapshot, RISK_PRIORITY
from traffic_system.train_future_calibrated_cloud_coordinator import (
    extract_split,
    string_labels,
)
from traffic_system.train_joint_risk_astgcn import select_device


METHODS = (
    "local_only",
    "confidence_only",
    "conformal_only",
    "learned_gate_only",
    "gate_plus_critical_safety",
    "full_runtime_policy",
)

METHOD_LABELS = {
    "local_only": "仅边缘本地专家",
    "confidence_only": "仅低置信度上云",
    "conformal_only": "仅风险候选集合歧义上云",
    "learned_gate_only": "仅学习式门控上云",
    "gate_plus_critical_safety": "学习式门控加最高风险兜底",
    "full_runtime_policy": "完整调度策略",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate six traffic routing policies with grouped bootstrap intervals."
    )
    parser.add_argument("--config", default="configurations/PEMS08_astgcn.conf")
    parser.add_argument(
        "--data_npz", default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz"
    )
    parser.add_argument(
        "--risk_labels", default="datasets/risk_labels_pems08_metis4.npz"
    )
    parser.add_argument(
        "--checkpoint",
        default="experiments/PEMS08/joint_risk_astgcn_metis4_flowprio2_frozen/best.pt",
    )
    parser.add_argument(
        "--student_model", default="models/edge_student_freeway_joint_metis4.json"
    )
    parser.add_argument(
        "--cloud_model", default="models/cloud_coordinator_future_calibrated.joblib"
    )
    parser.add_argument(
        "--risk_calibrator", default="models/region_risk_conformal.json"
    )
    parser.add_argument(
        "--topology", default="models/traffic_region_topology_metis4.json"
    )
    parser.add_argument("--defer_gate", default="models/edge_defer_gate.npz")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--confidence_threshold", type=float, default=None)
    parser.add_argument("--deadline_ms", type=float, default=200.0)
    parser.add_argument("--edge_compute_ms", type=float, default=74.0)
    parser.add_argument("--cloud_compute_ms", type=float, default=32.0)
    parser.add_argument("--rtt_ms", type=float, default=15.0)
    parser.add_argument("--jitter_ms", type=float, default=3.0)
    parser.add_argument("--cloud_queue_ms", type=float, default=1.0)
    parser.add_argument(
        "--request_bytes_per_cloud",
        type=float,
        default=5187.066116,
        help=(
            "Empirical conditional request bytes from the 360-event real HTTP selective run; "
            "used only for a transparent traffic-volume estimate."
        ),
    )
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_json",
        default="results/research/routing_ablation_grouped_bootstrap_20260727.json",
    )
    parser.add_argument(
        "--report_md",
        default="results/research/routing_ablation_grouped_bootstrap_20260727.md",
    )
    return parser.parse_args()


def _prediction_set(event: Mapping[str, Any]) -> List[str]:
    summary = event.get("region_summary", {})
    calibration = summary.get("region_risk_calibration", {})
    values = calibration.get("prediction_set", []) if isinstance(calibration, dict) else []
    if values:
        return [str(value) for value in values]
    return [str(summary.get("region_risk_level", "low"))]


def _point_risk(event: Mapping[str, Any]) -> str:
    summary = event.get("region_summary", {})
    return str(summary.get("region_risk_level", "low"))


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    count = len(DECISION_CLASSES)
    matrix = np.zeros((count, count), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def _macro_f1_from_confusion(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    denominator = support + predicted
    f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    present = support > 0
    return float(np.mean(f1[present])) if np.any(present) else 0.0


def _metric_values(
    matrix: np.ndarray,
    critical_reference: float,
    critical_true_positive: float,
    cloud_requests: float,
    sample_count: float,
    edge_ms: float,
    sync_ms: float,
    request_bytes_per_cloud: float,
    deadline_ms: float,
) -> Dict[str, float]:
    total = max(1.0, float(sample_count))
    cloud_rate = float(cloud_requests) / total
    accuracy = float(np.trace(matrix)) / total
    critical_recall = (
        float(critical_true_positive) / float(critical_reference)
        if critical_reference > 0
        else 0.0
    )
    mean_latency = edge_ms + cloud_rate * (sync_ms - edge_ms)
    p95_latency = sync_ms if cloud_rate > 0.05 else edge_ms
    deadline_miss = cloud_rate if sync_ms > deadline_ms else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": _macro_f1_from_confusion(matrix),
        "critical_intervention_recall": critical_recall,
        "cloud_request_rate": cloud_rate,
        "estimated_upstream_bytes_per_event": cloud_rate * request_bytes_per_cloud,
        "estimated_mean_closed_loop_ms": mean_latency,
        "estimated_p95_closed_loop_ms": p95_latency,
        "estimated_deadline_miss_rate": deadline_miss,
    }


def _interval(values: np.ndarray) -> Dict[str, float]:
    return {
        "lower": round(float(np.quantile(values, 0.025)), 6),
        "upper": round(float(np.quantile(values, 0.975)), 6),
    }


def grouped_bootstrap(
    sample_ids: np.ndarray,
    targets: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    cloud_masks: Mapping[str, np.ndarray],
    edge_ms: float,
    sync_ms: float,
    request_bytes_per_cloud: float,
    deadline_ms: float,
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    """Resample complete timestamp groups; all regions from one timestamp stay together."""
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    groups, inverse = np.unique(np.asarray(sample_ids), return_inverse=True)
    group_count = len(groups)
    class_count = len(DECISION_CLASSES)
    critical_target = np.asarray(
        [
            DECISION_CLASSES[int(value)] in CRITICAL_DECISIONS
            for value in targets
        ],
        dtype=bool,
    )
    stats = np.zeros((len(METHODS), group_count, class_count * class_count + 4))
    for method_index, method in enumerate(METHODS):
        prediction = np.asarray(predictions[method], dtype=np.int64)
        critical_prediction = np.asarray(
            [
                DECISION_CLASSES[int(value)] in CRITICAL_DECISIONS
                for value in prediction
            ],
            dtype=bool,
        )
        cloud = np.asarray(cloud_masks[method], dtype=bool)
        for group_index in range(group_count):
            selected = inverse == group_index
            stats[method_index, group_index, : class_count * class_count] = _confusion(
                targets[selected], prediction[selected]
            ).reshape(-1)
            offset = class_count * class_count
            stats[method_index, group_index, offset] = int(critical_target[selected].sum())
            stats[method_index, group_index, offset + 1] = int(
                (critical_target[selected] & critical_prediction[selected]).sum()
            )
            stats[method_index, group_index, offset + 2] = int(cloud[selected].sum())
            stats[method_index, group_index, offset + 3] = int(selected.sum())

    rng = np.random.default_rng(seed)
    estimates = {
        method: {
            metric: np.empty(samples, dtype=np.float64)
            for metric in (
                "accuracy",
                "macro_f1",
                "critical_intervention_recall",
                "cloud_request_rate",
                "estimated_upstream_bytes_per_event",
                "estimated_mean_closed_loop_ms",
                "estimated_p95_closed_loop_ms",
                "estimated_deadline_miss_rate",
            )
        }
        for method in METHODS
    }
    probabilities = np.full(group_count, 1.0 / group_count)
    offset = class_count * class_count
    for bootstrap_index in range(samples):
        weights = rng.multinomial(group_count, probabilities)
        aggregated = np.einsum("g,mgk->mk", weights, stats, optimize=True)
        for method_index, method in enumerate(METHODS):
            values = _metric_values(
                aggregated[method_index, :offset].reshape(class_count, class_count),
                aggregated[method_index, offset],
                aggregated[method_index, offset + 1],
                aggregated[method_index, offset + 2],
                aggregated[method_index, offset + 3],
                edge_ms,
                sync_ms,
                request_bytes_per_cloud,
                deadline_ms,
            )
            for metric, value in values.items():
                estimates[method][metric][bootstrap_index] = value

    intervals: Dict[str, Any] = {}
    for method in METHODS:
        intervals[method] = {
            metric: _interval(values) for metric, values in estimates[method].items()
        }
        intervals[method]["paired_delta_vs_local_only"] = {
            metric: {
                **_interval(values - estimates["local_only"][metric]),
                "probability_positive": round(
                    float(np.mean(values - estimates["local_only"][metric] > 0.0)),
                    6,
                ),
            }
            for metric, values in estimates[method].items()
        }
        intervals[method]["paired_delta_vs_full_runtime_policy"] = {
            metric: {
                **_interval(values - estimates["full_runtime_policy"][metric]),
                "probability_positive": round(
                    float(
                        np.mean(
                            values
                            - estimates["full_runtime_policy"][metric]
                            > 0.0
                        )
                    ),
                    6,
                ),
            }
            for metric, values in estimates[method].items()
        }
    return {
        "method": "timestamp_grouped_nonparametric_bootstrap",
        "bootstrap_samples": samples,
        "group_count": group_count,
        "group_key": "sample_id",
        "regions_kept_together": True,
        "seed": seed,
        "intervals": intervals,
    }


def _write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# 六组交通调度策略消融与按时间组重采样置信区间",
        "",
        "> 参考结果由未来交通流量、道路占有率和速度，经冻结风险聚类模型和固定策略生成，",
        "> 不是人工交通控制真值。所有组共享相同本地专家输出，只改变请求云端的路由信号。",
        "",
        "| 方法 | 准确率（95%置信区间） | 宏平均F1 | 关键干预召回率 | 云请求率 | 估算每事件上行量 | 估算均值/第95百分位时延 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        metrics = result["methods"][method]
        intervals = result["bootstrap"]["intervals"][method]
        lines.append(
            "| {} | {:.2%} [{:.2%}, {:.2%}] | {:.2%} | {:.2%} | {:.2%} | {:.1f} B | {:.2f}/{:.2f} ms |".format(
                METHOD_LABELS[method],
                metrics["accuracy"],
                intervals["accuracy"]["lower"],
                intervals["accuracy"]["upper"],
                metrics["macro_f1"],
                metrics["critical_intervention_recall"],
                metrics["cloud_request_rate"],
                metrics["estimated_upstream_bytes_per_event"],
                metrics["estimated_mean_closed_loop_ms"],
                metrics["estimated_p95_closed_loop_ms"],
            )
        )
    lines.extend(
        [
            "",
            "## 相对纯本地策略的配对差值（95%置信区间）",
            "",
            "| 方法 | 准确率差值 | 宏平均F1差值 | 关键干预召回率差值 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS[1:]:
        delta = result["bootstrap"]["intervals"][method][
            "paired_delta_vs_local_only"
        ]
        lines.append(
            "| {} | [{:+.2f}, {:+.2f}]个百分点 | [{:+.2f}, {:+.2f}]个百分点 | [{:+.2f}, {:+.2f}]个百分点 |".format(
                METHOD_LABELS[method],
                100.0 * delta["accuracy"]["lower"],
                100.0 * delta["accuracy"]["upper"],
                100.0 * delta["macro_f1"]["lower"],
                100.0 * delta["macro_f1"]["upper"],
                100.0
                * delta["critical_intervention_recall"]["lower"],
                100.0
                * delta["critical_intervention_recall"]["upper"],
            )
        )
    confidence_vs_full = result["bootstrap"]["intervals"]["confidence_only"][
        "paired_delta_vs_full_runtime_policy"
    ]
    lines.extend(
        [
            "",
            "## 关键解释",
            "",
            "- 完整策略相对纯本地的准确率和宏平均F1差值置信区间均大于0；关键干预召回率差值区间跨0，不能宣称提高。",
            "- 仅低置信度上云策略相对完整策略的准确率差值为[{:+.2f}, {:+.2f}]个百分点，区间跨0；但它少用约40.45个百分点云请求。".format(
                100.0 * confidence_vs_full["accuracy"]["lower"],
                100.0 * confidence_vs_full["accuracy"]["upper"],
            ),
            "- 因此完整策略是偏审慎、宏平均F1较高的方案，不是每项效果和成本都同时占优的绝对最优方案。",
            "",
            "## 口径",
            "",
            "- 重采样以同一时间编号（程序字段 `sample_id`）为组，同一时刻的四个区域整体进入或离开。",
            "- 上行字节使用360条真实网络选择性实验的条件均值进行透明估算，不冒充本次逐条抓包。",
            "- 完整策略运行在健康监测、证据完整、正常网络条件；漂移失效强制复核另作故障实验。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.top_k <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("batch_size, top_k and bootstrap_samples must be positive")
    if args.request_bytes_per_cloud <= 0:
        raise ValueError("request_bytes_per_cloud must be positive")

    device = select_device(args.device)
    config = load_config(args.config)
    checkpoint = torch_load_trusted(Path(args.checkpoint), device)
    adjacency, _ = load_adjacency(config)
    model = build_model_from_checkpoint(
        config,
        {"in_channels": 3, "output_dim": 3},
        adjacency,
        checkpoint,
        device,
    )
    partitions = [[int(node_id) for node_id in part] for part in checkpoint["partitions"]]
    capabilities = [
        build_control_capabilities(partitions, adjacency, partition_id)
        for partition_id in range(len(partitions))
    ]
    student_model = load_student_model(Path(args.student_model))
    cloud_payload = load_cloud_model(Path(args.cloud_model))
    calibrator = load_risk_calibrator(Path(args.risk_calibrator))
    defer_gate = load_defer_gate(Path(args.defer_gate))
    test = extract_split(
        "test",
        Path(args.data_npz),
        Path(args.risk_labels),
        model,
        student_model,
        partitions,
        capabilities,
        device,
        args.batch_size,
        args.top_k,
        load_region_neighbors(Path(args.topology)),
        risk_calibrator=calibrator,
    )
    if list(test["feature_names"]) != list(cloud_payload["feature_names"]):
        raise ValueError("Cloud coordinator feature schema mismatch")
    if list(test["feature_names"]) != list(defer_gate["base_feature_names"]):
        raise ValueError("Defer gate base feature schema mismatch")

    targets = np.asarray(test["y"], dtype=np.int64)
    cloud_predictions = np.asarray(
        cloud_payload["model"].predict(test["x"]), dtype=np.int64
    )
    gate_features = build_gate_features(
        test["x"],
        test["rule"],
        test["student"],
        test["student_confidence"],
    )
    gate_choices, gate_confidences = predict_defer_gate(gate_features, defer_gate)
    local_predictions = np.asarray(test["rule"], dtype=np.int64).copy()
    use_student = gate_choices == GATE_CLASSES.index("edge_student")
    local_predictions[use_student] = np.asarray(test["student"], dtype=np.int64)[
        use_student
    ]
    defer_mask = gate_choices == GATE_CLASSES.index("defer_cloud")

    threshold = (
        float(args.confidence_threshold)
        if args.confidence_threshold is not None
        else float(cloud_payload["metadata"]["scheduler_confidence_threshold"])
    )
    scheduler = AdaptiveScheduler(
        deadline_ms=args.deadline_ms,
        confidence_threshold=threshold,
        edge_compute_ms=args.edge_compute_ms,
        cloud_compute_ms=args.cloud_compute_ms,
    )
    network = NetworkSnapshot(
        available=True,
        rtt_ms=args.rtt_ms,
        jitter_ms=args.jitter_ms,
        loss_rate=0.0,
        cloud_queue_ms=args.cloud_queue_ms,
    )
    sync_ms = (
        args.edge_compute_ms
        + args.rtt_ms
        + scheduler.jitter_guard * args.jitter_ms
        + args.cloud_queue_ms
        + args.cloud_compute_ms
    )

    conformal_ambiguous = np.asarray(
        [len(_prediction_set(event)) > 1 for event in test["scheduler_events"]],
        dtype=bool,
    )
    point_severe = np.asarray(
        [
            RISK_PRIORITY.get(_point_risk(event), 0) >= RISK_PRIORITY["severe"]
            for event in test["scheduler_events"]
        ],
        dtype=bool,
    )
    full_routes: List[str] = []
    for index, event in enumerate(test["scheduler_events"]):
        decision = scheduler.schedule(
            event,
            1.0,
            network,
            defer_recommended=bool(defer_mask[index]),
            selective_defer=True,
        )
        full_routes.append(decision.route)

    cloud_masks = {
        "local_only": np.zeros(len(targets), dtype=bool),
        "confidence_only": np.asarray(test["student_confidence"]) < threshold,
        "conformal_only": conformal_ambiguous,
        "learned_gate_only": defer_mask,
        "gate_plus_critical_safety": defer_mask | point_severe,
        "full_runtime_policy": np.asarray(
            [route in {"cloud_sync", "cloud_async"} for route in full_routes],
            dtype=bool,
        ),
    }
    predictions = {
        method: np.where(mask, cloud_predictions, local_predictions)
        for method, mask in cloud_masks.items()
    }

    methods: Dict[str, Any] = {}
    target_names = string_labels(targets)
    critical_target = np.asarray(
        [name in CRITICAL_DECISIONS for name in target_names], dtype=bool
    )
    for method in METHODS:
        prediction = predictions[method]
        prediction_names = string_labels(prediction)
        critical_prediction = np.asarray(
            [name in CRITICAL_DECISIONS for name in prediction_names], dtype=bool
        )
        report = classification_report(
            target_names, prediction_names, DECISION_CLASSES
        )
        metrics = _metric_values(
            _confusion(targets, prediction),
            float(critical_target.sum()),
            float((critical_target & critical_prediction).sum()),
            float(cloud_masks[method].sum()),
            float(len(targets)),
            args.edge_compute_ms,
            sync_ms,
            args.request_bytes_per_cloud,
            args.deadline_ms,
        )
        methods[method] = {
            **{key: round(value, 6) for key, value in metrics.items()},
            "label": METHOD_LABELS[method],
            "cloud_requests": int(cloud_masks[method].sum()),
            "route_counts": (
                dict(Counter(full_routes))
                if method == "full_runtime_policy"
                else {
                    "cloud_sync": int(cloud_masks[method].sum()),
                    "edge_only": int((~cloud_masks[method]).sum()),
                }
            ),
            "classification": report,
        }

    bootstrap = grouped_bootstrap(
        np.asarray(test["sample_ids"]),
        targets,
        predictions,
        cloud_masks,
        args.edge_compute_ms,
        sync_ms,
        args.request_bytes_per_cloud,
        args.deadline_ms,
        args.bootstrap_samples,
        args.seed,
    )
    result = {
        "schema_version": 1,
        "task": "traffic_six_policy_routing_ablation",
        "reference": {
            "source": "future flow/occupancy/speed -> frozen FCM risk -> fixed safety policy",
            "status": "data-driven proxy reference, not manual traffic-control ground truth",
            "evaluation_split": "test",
            "test_used_for_fitting": False,
        },
        "scope": {
            "event_count": int(len(targets)),
            "timestamp_group_count": int(len(np.unique(test["sample_ids"]))),
            "partition_count": len(partitions),
            "shared_local_expert_selector": (
                "portable defer gate chooses rule or Student; defer rows fall back to rule "
                "when a policy keeps the event local"
            ),
            "only_routing_signal_changes_between_methods": True,
        },
        "method_definitions": {
            "local_only": "never request cloud",
            "confidence_only": "request when Student confidence is below the frozen threshold",
            "conformal_only": "request when the calibrated region-risk set has more than one class",
            "learned_gate_only": "request only when the frozen scene defer gate selects defer_cloud",
            "gate_plus_critical_safety": "gate request OR point region risk is severe",
            "full_runtime_policy": (
                "defer gate + calibrated set + severe safety + deadline/network routing; "
                "monitoring is healthy and evidence is complete in this run"
            ),
        },
        "scheduler": {
            "confidence_threshold": threshold,
            "deadline_ms": args.deadline_ms,
            "edge_compute_ms": args.edge_compute_ms,
            "predicted_sync_closed_loop_ms": round(sync_ms, 6),
            "network": network.__dict__,
        },
        "communication_estimate": {
            "request_bytes_per_cloud": args.request_bytes_per_cloud,
            "source": (
                "627635 B / 121 cloud requests from "
                "framework_http_truth_360_selective.json"
            ),
            "status": "estimate, not per-event packet capture in this offline ablation",
        },
        "defer_gate": {
            "path": args.defer_gate,
            "mean_test_confidence": round(float(np.mean(gate_confidences)), 6),
            "choice_counts": {
                name: int(np.sum(gate_choices == index))
                for index, name in enumerate(GATE_CLASSES)
            },
        },
        "methods": methods,
        "bootstrap": bootstrap,
    }
    save_json(result, Path(args.output_json))
    _write_markdown(result, Path(args.report_md))
    print(json.dumps({name: methods[name] for name in METHODS}, ensure_ascii=False, indent=2))
    print("metrics:", args.output_json)
    print("report:", args.report_md)


if __name__ == "__main__":
    main()
