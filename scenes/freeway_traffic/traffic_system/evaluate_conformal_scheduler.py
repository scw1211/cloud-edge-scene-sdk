"""用途：在独立测试时段成对比较原始置信度调度与校准风险集合调度。"""

import argparse
import json
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
from traffic_system.evaluate_future_truth_policy import classification_report
from traffic_system.infer_joint_risk_astgcn import (
    build_control_capabilities,
    build_model_from_checkpoint,
    load_adjacency,
    load_config,
    torch_load_trusted,
)
from traffic_system.scheduler import AdaptiveScheduler, NetworkSnapshot
from traffic_system.train_future_calibrated_cloud_coordinator import extract_split, string_labels
from traffic_system.train_joint_risk_astgcn import select_device


NETWORK_PROFILES = {
    "normal": NetworkSnapshot(True, 15.0, 3.0, 0.0, 1.0),
    "mild": NetworkSnapshot(True, 55.0, 10.0, 0.01, 5.0),
    "severe": NetworkSnapshot(True, 160.0, 30.0, 0.10, 30.0),
    "outage": NetworkSnapshot(False, 0.0, 0.0, 1.0, 0.0),
}

CRITICAL_DECISIONS = {
    "variable_speed_limit",
    "ramp_metering",
    "regional_coordination",
    "reroute",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw-confidence and conformal-set cloud-edge scheduling."
    )
    parser.add_argument("--config", default="configurations/PEMS08_astgcn.conf")
    parser.add_argument("--data_npz", default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz")
    parser.add_argument("--risk_labels", default="datasets/risk_labels_pems08_metis4.npz")
    parser.add_argument(
        "--checkpoint",
        default="experiments/PEMS08/joint_risk_astgcn_metis4_flowprio2_frozen/best.pt",
    )
    parser.add_argument("--student_model", default="models/edge_student_freeway_joint_metis4.json")
    parser.add_argument(
        "--cloud_model",
        default="models/cloud_coordinator_future_calibrated.joblib",
    )
    parser.add_argument("--risk_calibrator", default="models/region_risk_conformal.json")
    parser.add_argument(
        "--topology",
        default="models/traffic_region_topology_metis4.json",
    )
    parser.add_argument(
        "--deployment_method",
        default=None,
        choices=["marginal_aps", "class_conditional_aps", "simultaneous_window_aps"],
        help="Override the calibrator deployment method for an ablation run.",
    )
    parser.add_argument("--defer_gate", default="models/edge_defer_gate.npz")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--confidence_threshold", type=float, default=None)
    parser.add_argument("--deadline_ms", type=float, default=200.0)
    parser.add_argument("--edge_compute_ms", type=float, default=74.0)
    parser.add_argument("--cloud_compute_ms", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_json",
        default="results/edge/conformal_scheduler_comparison.json",
    )
    parser.add_argument(
        "--report_md",
        default="results/edge/conformal_scheduler_comparison.md",
    )
    return parser.parse_args()


def load_region_neighbors(path: Path) -> Mapping[str, Sequence[str]]:
    with path.open("r", encoding="utf-8") as file_obj:
        topology = json.load(file_obj)
    region_neighbors = topology.get("region_neighbors")
    if not isinstance(region_neighbors, dict):
        raise ValueError("traffic topology must contain region_neighbors")
    return region_neighbors


def event_without_calibration(event: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(event["region_summary"])
    summary.pop("region_risk_calibration", None)
    return {
        "upload_required": bool(event.get("upload_required", False)),
        "region_summary": summary,
    }


def immediate_prediction(
    route: str,
    cloud_delivered: bool,
    cloud_prediction: int,
    rule_prediction: int,
    student_prediction: int,
) -> Tuple[int, str]:
    if route == "cloud_sync" and cloud_delivered:
        return cloud_prediction, "cloud_coordinator"
    if route == "edge_only":
        return student_prediction, "edge_student"
    return rule_prediction, "local_safety_policy"


def metric_block(
    targets: np.ndarray,
    predictions: np.ndarray,
    routes: Sequence[str],
    sources: Sequence[str],
    sync_failures: int,
) -> Dict[str, Any]:
    target_names = string_labels(targets)
    prediction_names = string_labels(predictions)
    report = classification_report(target_names, prediction_names, DECISION_CLASSES)
    critical_target = np.asarray(
        [name in CRITICAL_DECISIONS for name in target_names], dtype=bool
    )
    critical_prediction = np.asarray(
        [name in CRITICAL_DECISIONS for name in prediction_names], dtype=bool
    )
    critical_count = int(np.sum(critical_target))
    unsafe_no_action = int(
        np.sum(critical_target & (np.asarray(prediction_names, dtype=object) == "no_action"))
    )
    route_counts = Counter(routes)
    source_counts = Counter(sources)
    cloud_requests = route_counts["cloud_sync"] + route_counts["cloud_async"]
    total = len(targets)
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1_present_classes"],
        "weighted_f1": report["weighted_f1"],
        "critical_reference_count": critical_count,
        "critical_intervention_recall": round(
            float(np.sum(critical_target & critical_prediction)) / critical_count, 6
        )
        if critical_count
        else None,
        "critical_no_action_rate": round(unsafe_no_action / critical_count, 6)
        if critical_count
        else None,
        "cloud_request_rate": round(cloud_requests / total, 6),
        "cloud_sync_rate": round(route_counts["cloud_sync"] / total, 6),
        "cloud_async_rate": round(route_counts["cloud_async"] / total, 6),
        "sync_failure_count": sync_failures,
        "route_counts": dict(route_counts),
        "decision_source_counts": dict(source_counts),
        "classification": report,
    }


def evaluate_method(
    method: str,
    events: Sequence[Dict[str, Any]],
    student_confidences: np.ndarray,
    student_predictions: np.ndarray,
    rule_predictions: np.ndarray,
    cloud_predictions: np.ndarray,
    targets: np.ndarray,
    network: NetworkSnapshot,
    scheduler: AdaptiveScheduler,
    delivery_uniforms: np.ndarray,
) -> Tuple[Dict[str, Any], np.ndarray, List[str]]:
    predictions: List[int] = []
    routes: List[str] = []
    sources: List[str] = []
    sync_failures = 0
    for index, calibrated_event in enumerate(events):
        scheduler_event = (
            calibrated_event
            if method == "conformal_set"
            else event_without_calibration(calibrated_event)
        )
        schedule = scheduler.schedule(
            scheduler_event,
            float(student_confidences[index]),
            network,
        )
        delivered = bool(
            schedule.route == "cloud_sync"
            and network.available
            and delivery_uniforms[index] >= network.loss_rate
        )
        if schedule.route == "cloud_sync" and not delivered:
            sync_failures += 1
        prediction, source = immediate_prediction(
            schedule.route,
            delivered,
            int(cloud_predictions[index]),
            int(rule_predictions[index]),
            int(student_predictions[index]),
        )
        predictions.append(prediction)
        routes.append(schedule.route)
        sources.append(source)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    return (
        metric_block(targets, prediction_array, routes, sources, sync_failures),
        prediction_array,
        routes,
    )


def evaluate_selective_defer(
    events: Sequence[Dict[str, Any]],
    gate_choices: np.ndarray,
    local_predictions: np.ndarray,
    rule_predictions: np.ndarray,
    cloud_predictions: np.ndarray,
    targets: np.ndarray,
    network: NetworkSnapshot,
    scheduler: AdaptiveScheduler,
    delivery_uniforms: np.ndarray,
) -> Tuple[Dict[str, Any], np.ndarray, List[str]]:
    predictions: List[int] = []
    routes: List[str] = []
    sources: List[str] = []
    sync_failures = 0
    defer_id = GATE_CLASSES.index("defer_cloud")
    student_id = GATE_CLASSES.index("edge_student")
    for index, event in enumerate(events):
        defer_recommended = int(gate_choices[index]) == defer_id
        schedule = scheduler.schedule(
            event,
            1.0,
            network,
            defer_recommended=defer_recommended,
            selective_defer=True,
        )
        delivered = bool(
            schedule.route == "cloud_sync"
            and network.available
            and delivery_uniforms[index] >= network.loss_rate
        )
        if schedule.route == "cloud_sync" and not delivered:
            sync_failures += 1
        if schedule.route == "cloud_sync" and delivered:
            prediction = int(cloud_predictions[index])
            source = "cloud_coordinator"
        elif schedule.route == "edge_only":
            prediction = int(local_predictions[index])
            source = (
                "defer_gate_student"
                if int(gate_choices[index]) == student_id
                else "defer_gate_rule"
            )
        else:
            prediction = int(rule_predictions[index])
            source = "local_safety_policy"
        predictions.append(prediction)
        routes.append(schedule.route)
        sources.append(source)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    return (
        metric_block(targets, prediction_array, routes, sources, sync_failures),
        prediction_array,
        routes,
    )


def paired_comparison(
    targets: np.ndarray,
    legacy_predictions: np.ndarray,
    calibrated_predictions: np.ndarray,
    legacy_routes: Sequence[str],
    calibrated_routes: Sequence[str],
) -> Dict[str, Any]:
    legacy_correct = legacy_predictions == targets
    calibrated_correct = calibrated_predictions == targets
    return {
        "route_changed_count": int(
            np.sum(np.asarray(legacy_routes, dtype=object) != np.asarray(calibrated_routes, dtype=object))
        ),
        "decision_changed_count": int(np.sum(legacy_predictions != calibrated_predictions)),
        "legacy_wrong_conformal_correct": int(np.sum(~legacy_correct & calibrated_correct)),
        "legacy_correct_conformal_wrong": int(np.sum(legacy_correct & ~calibrated_correct)),
        "both_correct": int(np.sum(legacy_correct & calibrated_correct)),
        "both_wrong": int(np.sum(~legacy_correct & ~calibrated_correct)),
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# 校准风险集合调度对比",
        "",
        "> 这里的参考答案由 PEMS08 未来 flow/occupancy/speed 经冻结 FCM 标签和固定安全策略生成，",
        "> 属于数据驱动代理真值，不是人工拥堵事件或真实控制效果标注。",
        "",
        "- 测试集不参与温度、conformal 阈值、Student、云协调器或调度阈值训练。",
        "- 两种调度共享同一批模型输出和同一组模拟丢包，仅改变不确定性表达。",
        "- `critical recall` 表示参考策略要求关键干预时，最终动作是否仍属于关键干预。",
        "",
        "| 网络 | 调度 | Accuracy | Macro-F1 | Critical recall | 云请求率 | 同步云率 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile_name, profile in result["profiles"].items():
        for method_name, label in (
            ("raw_confidence", "原始置信度"),
            ("conformal_set", "校准风险集合"),
            ("selective_defer", "选择性协同"),
        ):
            metrics = profile[method_name]
            lines.append(
                "| {} | {} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2%} |".format(
                    profile_name,
                    label,
                    metrics["accuracy"],
                    metrics["macro_f1"],
                    metrics["critical_intervention_recall"],
                    metrics["cloud_request_rate"],
                    metrics["cloud_sync_rate"],
                )
            )
    normal = result["profiles"]["normal"]
    paired = normal["paired"]
    lines.extend(
        [
            "",
            "## 正常网络成对结果",
            "",
            "- 调度路径改变：{} 个事件。".format(paired["route_changed_count"]),
            "- 原方法错误、校准集合正确：{} 个事件。".format(
                paired["legacy_wrong_conformal_correct"]
            ),
            "- 原方法正确、校准集合错误：{} 个事件。".format(
                paired["legacy_correct_conformal_wrong"]
            ),
            "",
            "结论必须结合云请求开销一起看；若准确率收益很小而请求率明显增加，",
            "该方法只保留为置信度审计与安全兜底，不应宣称提升了风险识别精度。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.top_k <= 0:
        raise ValueError("batch_size and top_k must be positive")
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
    region_neighbors = load_region_neighbors(Path(args.topology))
    student_model = load_student_model(Path(args.student_model))
    cloud_payload = load_cloud_model(Path(args.cloud_model))
    calibrator = load_risk_calibrator(Path(args.risk_calibrator))
    if args.deployment_method is not None:
        calibrator = dict(calibrator)
        calibrator["deployment_method"] = args.deployment_method
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
        region_neighbors,
        risk_calibrator=calibrator,
    )
    if list(test["feature_names"]) != list(cloud_payload["feature_names"]):
        raise ValueError("Cloud coordinator feature schema mismatch")
    cloud_predictions = np.asarray(cloud_payload["model"].predict(test["x"]), dtype=np.int64)
    if list(test["feature_names"]) != list(defer_gate["base_feature_names"]):
        raise ValueError("Defer gate base feature schema mismatch")
    gate_features = build_gate_features(
        test["x"],
        test["rule"],
        test["student"],
        test["student_confidence"],
    )
    gate_choices, gate_confidences = predict_defer_gate(gate_features, defer_gate)
    local_predictions = np.asarray(test["rule"], dtype=np.int64).copy()
    use_student = gate_choices == GATE_CLASSES.index("edge_student")
    local_predictions[use_student] = test["student"][use_student]
    threshold = args.confidence_threshold
    if threshold is None:
        threshold = float(cloud_payload["metadata"]["scheduler_confidence_threshold"])
    scheduler = AdaptiveScheduler(
        deadline_ms=args.deadline_ms,
        confidence_threshold=threshold,
        edge_compute_ms=args.edge_compute_ms,
        cloud_compute_ms=args.cloud_compute_ms,
    )

    profiles: Dict[str, Any] = {}
    for profile_index, (profile_name, network) in enumerate(NETWORK_PROFILES.items()):
        rng = np.random.default_rng(args.seed + profile_index)
        delivery_uniforms = rng.random(len(test["y"]))
        legacy_metrics, legacy_predictions, legacy_routes = evaluate_method(
            "raw_confidence",
            test["scheduler_events"],
            test["student_confidence"],
            test["student"],
            test["rule"],
            cloud_predictions,
            test["y"],
            network,
            scheduler,
            delivery_uniforms,
        )
        conformal_metrics, conformal_predictions, conformal_routes = evaluate_method(
            "conformal_set",
            test["scheduler_events"],
            test["student_confidence"],
            test["student"],
            test["rule"],
            cloud_predictions,
            test["y"],
            network,
            scheduler,
            delivery_uniforms,
        )
        selective_metrics, selective_predictions, selective_routes = evaluate_selective_defer(
            test["scheduler_events"],
            gate_choices,
            local_predictions,
            test["rule"],
            cloud_predictions,
            test["y"],
            network,
            scheduler,
            delivery_uniforms,
        )
        profiles[profile_name] = {
            "network": network.__dict__,
            "scheduler_predicted_sync_e2e_ms": round(
                args.edge_compute_ms
                + network.rtt_ms
                + scheduler.jitter_guard * network.jitter_ms
                + network.cloud_queue_ms
                + args.cloud_compute_ms,
                6,
            ),
            "raw_confidence": legacy_metrics,
            "conformal_set": conformal_metrics,
            "selective_defer": selective_metrics,
            "paired": paired_comparison(
                test["y"],
                legacy_predictions,
                conformal_predictions,
                legacy_routes,
                conformal_routes,
            ),
            "raw_vs_selective_paired": paired_comparison(
                test["y"],
                legacy_predictions,
                selective_predictions,
                legacy_routes,
                selective_routes,
            ),
        }

    result = {
        "task": "paired_conformal_scheduler_evaluation",
        "reference": {
            "source": "future flow/occupancy/speed -> frozen FCM risk -> fixed safety policy",
            "status": "data-driven proxy reference, not manual congestion or control ground truth",
            "evaluation_split": "test",
            "test_used_for_fitting": False,
        },
        "sample_count": int(len(test["y"])),
        "timestamp_count": int(len(np.unique(test["sample_ids"]))),
        "partition_count": len(partitions),
        "scheduler": {
            "confidence_threshold": threshold,
            "deadline_ms": args.deadline_ms,
            "edge_compute_ms": args.edge_compute_ms,
            "cloud_compute_ms": args.cloud_compute_ms,
        },
        "calibrator": {
            "path": args.risk_calibrator,
            "deployment_method": calibrator["deployment_method"],
            "target_coverage": calibrator["target_coverage"],
            "temperature": calibrator["temperature"],
        },
        "defer_gate": {
            "path": args.defer_gate,
            "confidence_threshold": defer_gate["metadata"]["confidence_threshold"],
            "mean_test_confidence": round(float(np.mean(gate_confidences)), 6),
            "choice_counts": {
                name: int(np.sum(gate_choices == class_id))
                for class_id, name in enumerate(GATE_CLASSES)
            },
        },
        "profiles": profiles,
    }
    save_json(result, Path(args.output_json))
    write_markdown(result, Path(args.report_md))
    print("events:", result["sample_count"])
    for profile_name, profile in profiles.items():
        print(
            profile_name,
            "raw={:.4f}/{:.4f} conformal={:.4f}/{:.4f}".format(
                profile["raw_confidence"]["accuracy"],
                profile["raw_confidence"]["cloud_request_rate"],
                profile["conformal_set"]["accuracy"],
                profile["conformal_set"]["cloud_request_rate"],
            ),
        )
        print(
            "  selective={:.4f}/{:.4f}".format(
                profile["selective_defer"]["accuracy"],
                profile["selective_defer"]["cloud_request_rate"],
            )
        )
    print("metrics:", args.output_json)


if __name__ == "__main__":
    main()
