"""用途：用验证时段未来状态校准云端快速协调器，并在独立测试时段评估。"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

from traffic_system.decision_utils import DECISION_CLASSES, extract_feature_vector, rule_teacher_decision, save_json
from traffic_system.edge_student import load_student_model, predict_student
from traffic_system.evaluate_future_truth_policy import (
    classification_report,
    load_evaluation_arrays,
    make_event,
    one_hot_probabilities,
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
from traffic_system.risk_labels import RISK_CLASSES, denormalize
from traffic_system.scheduler import AdaptiveScheduler, NetworkSnapshot
from traffic_system.train_joint_risk_astgcn import clip_physical_state, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a future-state-calibrated cloud traffic coordinator.")
    parser.add_argument("--config", default="configurations/PEMS08_astgcn.conf")
    parser.add_argument("--data_npz", default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz")
    parser.add_argument("--risk_labels", default="datasets/risk_labels_pems08_metis4.npz")
    parser.add_argument("--student_model", default="models/edge_student_freeway_joint_metis4.json")
    parser.add_argument(
        "--checkpoint",
        default="experiments/PEMS08/joint_risk_astgcn_metis4_flowprio2_frozen/best.pt",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--topology",
        default="models/traffic_region_topology_metis4.json",
    )
    parser.add_argument("--tune_ratio", type=float, default=0.20)
    parser.add_argument("--purge_radius", type=int, default=24)
    parser.add_argument("--candidate_trees", type=int, default=100)
    parser.add_argument("--final_trees", type=int, default=200)
    parser.add_argument("--max_depths", default="12,16,none")
    parser.add_argument("--min_samples_leaf", default="1,2")
    parser.add_argument("--max_features", default="sqrt")
    parser.add_argument("--scheduler_thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--scheduler_accuracy_tolerance", type=float, default=0.01)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        default="models/cloud_coordinator_future_calibrated.joblib",
    )
    parser.add_argument(
        "--metrics",
        default="results/decision/cloud_coordinator_future_calibrated.json",
    )
    parser.add_argument(
        "--report_md",
        default="results/decision/cloud_coordinator_future_calibrated.md",
    )
    return parser.parse_args()


def parse_depths(value: str) -> List[Any]:
    depths = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        depths.append(None if item == "none" else int(item))
    if not depths or any(depth is not None and depth <= 0 for depth in depths):
        raise ValueError("max_depths must contain positive integers or none")
    return depths


def parse_positive_ints(value: str, name: str) -> List[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or min(values) <= 0:
        raise ValueError("{} must contain positive integers".format(name))
    return values


def parse_thresholds(value: str) -> List[float]:
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds or any(threshold <= 0.0 or threshold >= 1.0 for threshold in thresholds):
        raise ValueError("scheduler_thresholds must be in (0, 1)")
    return sorted(set(thresholds))


def parse_max_features(value: str) -> List[Any]:
    values: List[Any] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"sqrt", "log2"}:
            values.append(item)
            continue
        number = float(item)
        if number <= 0.0 or number > 1.0:
            raise ValueError("numeric max_features values must be within (0, 1]")
        values.append(number)
    if not values:
        raise ValueError("max_features must not be empty")
    return values


def temporal_tune_indices(
    sample_ids: np.ndarray,
    tune_ratio: float,
    purge_radius: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if not 0.0 < tune_ratio < 0.5 or purge_radius < 0:
        raise ValueError("tune_ratio must be in (0, 0.5) and purge_radius non-negative")
    unique_samples = np.unique(sample_ids)
    tune_count = max(1, int(round(len(unique_samples) * tune_ratio)))
    first_tune_sample = int(unique_samples[-tune_count])
    train_last_allowed = first_tune_sample - purge_radius - 1
    train_indices = np.flatnonzero(sample_ids <= train_last_allowed)
    tune_indices = np.flatnonzero(sample_ids >= first_tune_sample)
    if not len(train_indices) or not len(tune_indices):
        raise ValueError("Temporal tuning split is empty after purge")
    return train_indices, tune_indices, {
        "unique_timestamp_groups": int(len(unique_samples)),
        "train_last_sample": int(sample_ids[train_indices].max()),
        "tune_first_sample": int(sample_ids[tune_indices].min()),
        "purged_timestamp_groups": int(
            np.sum((unique_samples > train_last_allowed) & (unique_samples < first_tune_sample))
        ),
        "purge_radius_samples": purge_radius,
        "purge_radius_minutes": purge_radius * 5,
    }


def extract_split(
    split: str,
    data_path: Path,
    labels_path: Path,
    model: torch.nn.Module,
    student_model: Dict[str, Any],
    partitions: Sequence[Sequence[int]],
    capabilities: Sequence[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
    top_k: int,
    region_neighbors: Mapping[str, Sequence[str]],
    risk_calibrator: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    arrays = load_evaluation_arrays(data_path, labels_path, split)
    features: List[List[float]] = []
    targets: List[int] = []
    rule_predictions: List[int] = []
    sample_ids: List[int] = []
    event_ids: List[str] = []
    student_predictions: List[int] = []
    student_confidences: List[float] = []
    scheduler_events: List[Dict[str, Any]] = []
    feature_names: List[str] = []
    for batch_start in range(0, arrays["split_x"].shape[0], batch_size):
        batch_end = min(batch_start + batch_size, arrays["split_x"].shape[0])
        ids = list(range(batch_start, batch_end))
        tensor = torch.from_numpy(arrays["split_x"][ids].astype(np.float32)).to(device)
        with torch.no_grad():
            outputs = model(tensor)
        ensure_finite_outputs(outputs)
        forecast_raw = clip_physical_state(
            denormalize(outputs["forecast"].detach().cpu().numpy(), arrays["mean"], arrays["std"])
        )
        node_probs = F.softmax(outputs["node_logits"], dim=-1).detach().cpu().numpy()
        region_probs = F.softmax(outputs["region_logits"], dim=-1).detach().cpu().numpy()
        for batch_index, sample_id in enumerate(ids):
            truth_node_probs = one_hot_probabilities(arrays["node_labels"][sample_id], len(RISK_CLASSES))
            truth_region_probs = one_hot_probabilities(arrays["region_labels"][sample_id], len(RISK_CLASSES))
            predicted_events = []
            truth_events = []
            for partition_id in range(len(partitions)):
                predicted_events.append(make_event(
                    split,
                    sample_id,
                    partition_id,
                    partitions,
                    node_probs[batch_index],
                    region_probs[batch_index],
                    forecast_raw[batch_index],
                    capabilities[partition_id],
                    top_k,
                    "joint_astgcn_prediction",
                    risk_calibrator=risk_calibrator,
                ))
                truth_events.append(make_event(
                    split,
                    sample_id,
                    partition_id,
                    partitions,
                    truth_node_probs,
                    truth_region_probs,
                    arrays["split_target"][sample_id],
                    capabilities[partition_id],
                    top_k,
                    "future_observation_fcm_reference",
                ))
            events_by_region = {
                str(event["region_id"]): event for event in predicted_events
            }
            for predicted_event in predicted_events:
                neighbors = []
                for neighbor_region in region_neighbors.get(
                    str(predicted_event["region_id"]), []
                ):
                    neighbor_event = events_by_region.get(str(neighbor_region))
                    if neighbor_event is None:
                        continue
                    neighbor_summary = neighbor_event["region_summary"]
                    neighbors.append(
                        {
                            "event_id": neighbor_event["event_id"],
                            "edge_id": neighbor_event["edge_id"],
                            "region_id": neighbor_event["region_id"],
                            "risk_level": neighbor_summary["region_risk_level"],
                            "risk_score": neighbor_summary["region_risk_score"],
                            "confidence": neighbor_summary["region_risk_confidence"],
                        }
                    )
                predicted_event["neighbor_context"] = [
                    {
                        "method": "road_graph_cut_edges",
                        "neighbors": neighbors,
                    }
                ]
            for predicted_event, truth_event in zip(predicted_events, truth_events):
                vector, names = extract_feature_vector(predicted_event)
                if feature_names and feature_names != list(names):
                    raise ValueError("Feature schema changed during extraction")
                feature_names = list(names)
                target_name = str(rule_teacher_decision(truth_event)["decision"])
                rule_name = str(rule_teacher_decision(predicted_event)["decision"])
                student_name, student_confidence, _ = predict_student(predicted_event, student_model)
                features.append(vector)
                targets.append(DECISION_CLASSES.index(target_name))
                rule_predictions.append(DECISION_CLASSES.index(rule_name))
                sample_ids.append(sample_id)
                event_ids.append(str(predicted_event["event_id"]))
                student_predictions.append(DECISION_CLASSES.index(student_name))
                student_confidences.append(float(student_confidence))
                summary = predicted_event["region_summary"]
                scheduler_events.append(
                    {
                        "upload_required": bool(predicted_event["upload_required"]),
                        "region_summary": {
                            "region_risk_level": summary["region_risk_level"],
                            "max_node_risk_level": summary["max_node_risk_level"],
                            "region_risk_confidence": summary["region_risk_confidence"],
                            **(
                                {
                                    "region_risk_calibration": summary[
                                        "region_risk_calibration"
                                    ]
                                }
                                if "region_risk_calibration" in summary
                                else {}
                            ),
                        },
                    }
                )
        print("{} [{}/{}]".format(split, batch_end, arrays["split_x"].shape[0]), flush=True)
    return {
        "x": np.asarray(features, dtype=np.float64),
        "y": np.asarray(targets, dtype=np.int64),
        "rule": np.asarray(rule_predictions, dtype=np.int64),
        "sample_ids": np.asarray(sample_ids, dtype=np.int64),
        "event_ids": event_ids,
        "student": np.asarray(student_predictions, dtype=np.int64),
        "student_confidence": np.asarray(student_confidences, dtype=np.float64),
        "scheduler_events": scheduler_events,
        "feature_names": feature_names,
    }


def string_labels(values: np.ndarray) -> List[str]:
    return [DECISION_CLASSES[int(value)] for value in values.tolist()]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, bootstrap_samples: int, seed: int) -> Dict[str, Any]:
    true_names = string_labels(y_true)
    predicted_names = string_labels(y_pred)
    report = classification_report(true_names, predicted_names, DECISION_CLASSES)
    report["accuracy_95ci"] = stratified_bootstrap_accuracy_ci(
        true_names,
        predicted_names,
        bootstrap_samples,
        seed,
    )
    return report


def make_model(
    trees: int,
    max_depth: Any,
    min_samples_leaf: int,
    seed: int,
    max_features: Any = "sqrt",
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=trees,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def select_scheduler_tradeoff(
    candidates: Sequence[Dict[str, Any]],
    accuracy_tolerance: float,
) -> Dict[str, Any]:
    if not candidates or accuracy_tolerance < 0.0:
        raise ValueError("Scheduler candidates are empty or tolerance is negative")
    best_accuracy = max(float(candidate["accuracy"]) for candidate in candidates)
    eligible = [
        candidate
        for candidate in candidates
        if float(candidate["accuracy"]) >= best_accuracy - accuracy_tolerance
    ]
    selected = min(
        eligible,
        key=lambda item: (
            float(item["cloud_request_rate"]),
            -float(item["accuracy"]),
            -float(item["macro_f1"]),
        ),
    )
    return {
        **selected,
        "best_candidate_accuracy": round(best_accuracy, 6),
        "accuracy_tolerance": accuracy_tolerance,
    }


def calibrate_scheduler(
    validation: Dict[str, Any],
    tune_indices: np.ndarray,
    cloud_model: ExtraTreesClassifier,
    thresholds: Sequence[float],
    accuracy_tolerance: float,
) -> Dict[str, Any]:
    cloud_predictions = cloud_model.predict(validation["x"][tune_indices])
    network = NetworkSnapshot(available=True, rtt_ms=15.0, jitter_ms=3.0, loss_rate=0.0, cloud_queue_ms=1.0)
    candidates = []
    for threshold in thresholds:
        scheduler = AdaptiveScheduler(
            confidence_threshold=threshold,
            edge_compute_ms=74.0,
            cloud_compute_ms=32.0,
        )
        final_predictions = []
        route_counts: Dict[str, int] = {}
        for local_index, row_index in enumerate(tune_indices.tolist()):
            schedule = scheduler.schedule(
                validation["scheduler_events"][row_index],
                float(validation["student_confidence"][row_index]),
                network,
            )
            route_counts[schedule.route] = route_counts.get(schedule.route, 0) + 1
            if schedule.waits_for_cloud:
                final_predictions.append(int(cloud_predictions[local_index]))
            else:
                final_predictions.append(int(validation["student"][row_index]))
        predictions = np.asarray(final_predictions, dtype=np.int64)
        targets = validation["y"][tune_indices]
        cloud_requests = sum(
            count for route, count in route_counts.items() if route in {"cloud_sync", "cloud_async"}
        )
        candidates.append(
            {
                "confidence_threshold": threshold,
                "accuracy": round(float(accuracy_score(targets, predictions)), 6),
                "macro_f1": round(float(f1_score(targets, predictions, average="macro")), 6),
                "cloud_request_rate": round(cloud_requests / len(tune_indices), 6),
                "route_counts": route_counts,
            }
        )
    return {
        "selection_rule": "lowest cloud request rate within configured accuracy tolerance of the best validation accuracy",
        "network_profile": network.__dict__,
        "candidates": candidates,
        "selected": select_scheduler_tradeoff(candidates, accuracy_tolerance),
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    test = result["test"]
    rule = result["predicted_risk_rule_test"]
    lines = [
        "# 未来状态校准云端协调器",
        "",
        "- 训练标签：仅使用 val 时段未来 flow/occupancy/speed 形成的参考策略。",
        "- 参数选择：val 内部按时间切分并隔离 {} 分钟。".format(
            result["tuning_split"]["purge_radius_minutes"]
        ),
        "- 最终评测：完整 test 时段，一次性评估，不参与选参。",
        "- 模型定位：云端毫秒级专用协调器；Qwen 仍负责异步复杂复核。",
        "- 调度阈值：仅在 val 隔离尾段选择，置信阈值为 `{}`。".format(
            result["scheduler_calibration"]["selected"]["confidence_threshold"]
        ),
        "",
        "| 模型 | Accuracy | Macro-F1 | Weighted-F1 | 95% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| 预测风险 + 固定规则 | {:.2%} | {:.2%} | {:.2%} | - |".format(
            rule["accuracy"], rule["macro_f1_present_classes"], rule["weighted_f1"]
        ),
        "| 校准云协调器 | {:.2%} | {:.2%} | {:.2%} | [{:.2%}, {:.2%}] |".format(
            test["accuracy"],
            test["macro_f1_present_classes"],
            test["weighted_f1"],
            test["accuracy_95ci"]["lower"],
            test["accuracy_95ci"]["upper"],
        ),
        "",
        "选择参数：`max_depth={}`、`min_samples_leaf={}`、`trees={}`。".format(
            result["selected_hyperparameters"]["max_depth"],
            result["selected_hyperparameters"]["min_samples_leaf"],
            result["selected_hyperparameters"]["final_trees"],
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.candidate_trees <= 0 or args.final_trees <= 0:
        raise ValueError("batch_size and tree counts must be positive")
    max_depths = parse_depths(args.max_depths)
    min_leaves = parse_positive_ints(args.min_samples_leaf, "min_samples_leaf")
    max_features_values = parse_max_features(args.max_features)
    scheduler_thresholds = parse_thresholds(args.scheduler_thresholds)
    if args.scheduler_accuracy_tolerance < 0.0:
        raise ValueError("scheduler_accuracy_tolerance must be non-negative")
    device = select_device(args.device)
    config = load_config(args.config)
    with Path(args.topology).open("r", encoding="utf-8") as file_obj:
        topology = json.load(file_obj)
    region_neighbors = topology.get("region_neighbors")
    if not isinstance(region_neighbors, dict):
        raise ValueError("traffic topology must contain region_neighbors")
    checkpoint = torch_load_trusted(Path(args.checkpoint), device)
    adj_mx, _ = load_adjacency(config)
    model = build_model_from_checkpoint(
        config,
        {"in_channels": 3, "output_dim": 3},
        adj_mx,
        checkpoint,
        device,
    )
    partitions = [[int(node_id) for node_id in part] for part in checkpoint["partitions"]]
    capabilities = [
        build_control_capabilities(partitions, adj_mx, partition_id)
        for partition_id in range(len(partitions))
    ]
    student_model = load_student_model(Path(args.student_model))
    extraction_started = time.perf_counter()
    validation = extract_split(
        "val",
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
    )
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
    )
    extraction_seconds = time.perf_counter() - extraction_started
    if validation["feature_names"] != test["feature_names"]:
        raise ValueError("Validation and test feature schemas differ")

    train_indices, tune_indices, split_info = temporal_tune_indices(
        validation["sample_ids"], args.tune_ratio, args.purge_radius
    )
    candidates = []
    for max_depth in max_depths:
        for min_leaf in min_leaves:
            for max_features in max_features_values:
                candidate = make_model(
                    args.candidate_trees,
                    max_depth,
                    min_leaf,
                    args.seed,
                    max_features,
                )
                candidate.fit(validation["x"][train_indices], validation["y"][train_indices])
                predictions = candidate.predict(validation["x"][tune_indices])
                candidates.append(
                    {
                        "max_depth": max_depth,
                        "min_samples_leaf": min_leaf,
                        "max_features": max_features,
                        "accuracy": round(float(accuracy_score(validation["y"][tune_indices], predictions)), 6),
                        "macro_f1": round(float(f1_score(validation["y"][tune_indices], predictions, average="macro")), 6),
                        "weighted_f1": round(float(f1_score(validation["y"][tune_indices], predictions, average="weighted")), 6),
                    }
                )
                print("candidate", candidates[-1], flush=True)
    selected = max(candidates, key=lambda item: (item["macro_f1"], item["accuracy"]))
    tuning_cloud_model = make_model(
        args.candidate_trees,
        selected["max_depth"],
        int(selected["min_samples_leaf"]),
        args.seed,
        selected["max_features"],
    )
    tuning_cloud_model.fit(validation["x"][train_indices], validation["y"][train_indices])
    scheduler_calibration = calibrate_scheduler(
        validation,
        tune_indices,
        tuning_cloud_model,
        scheduler_thresholds,
        args.scheduler_accuracy_tolerance,
    )
    final_model = make_model(
        args.final_trees,
        selected["max_depth"],
        int(selected["min_samples_leaf"]),
        args.seed,
        selected["max_features"],
    )
    train_started = time.perf_counter()
    final_model.fit(validation["x"], validation["y"])
    train_seconds = time.perf_counter() - train_started
    test_predictions = final_model.predict(test["x"])
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": final_model,
        "feature_names": validation["feature_names"],
        "decision_classes": DECISION_CLASSES,
        "metadata": {
            "task": "future_state_calibrated_cloud_coordinator",
            "training_split": "val",
            "evaluation_split": "test",
            "label_source": "future flow/occupancy/speed -> frozen FCM risk -> fixed safety policy",
            "context_fusion": "road-graph adjacent regions at the same timestamp",
            "selected_hyperparameters": selected,
            "scheduler_confidence_threshold": scheduler_calibration["selected"]["confidence_threshold"],
        },
    }
    joblib.dump(payload, model_path, compress=3)
    result = {
        "task": "future_state_calibrated_cloud_coordinator",
        "method": {
            "training_split": "val",
            "evaluation_split": "test",
            "test_used_for_hyperparameter_selection": False,
            "label_source": "future flow/occupancy/speed -> frozen FCM risk -> fixed safety policy",
            "reference_status": "data-driven reference policy, not manual control ground truth",
            "context_fusion": "road-graph adjacent regions at the same timestamp",
            "device_for_feature_extraction": str(device),
        },
        "sample_counts": {
            "validation_events": int(len(validation["y"])),
            "test_events": int(len(test["y"])),
            "feature_dim": int(validation["x"].shape[1]),
        },
        "tuning_split": split_info,
        "candidates": candidates,
        "selected_hyperparameters": {
            **selected,
            "candidate_trees": args.candidate_trees,
            "final_trees": args.final_trees,
        },
        "scheduler_calibration": scheduler_calibration,
        "validation_full_fit": evaluate(
            validation["y"], final_model.predict(validation["x"]), args.bootstrap_samples, args.seed
        ),
        "predicted_risk_rule_test": evaluate(
            test["y"], test["rule"], args.bootstrap_samples, args.seed + 1
        ),
        "test": evaluate(test["y"], test_predictions, args.bootstrap_samples, args.seed + 2),
        "runtime": {
            "feature_extraction_seconds": round(extraction_seconds, 6),
            "final_training_seconds": round(train_seconds, 6),
        },
        "artifact": {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
        },
    }
    save_json(result, Path(args.metrics))
    write_markdown(result, Path(args.report_md))
    print("test accuracy:", result["test"]["accuracy"])
    print("test macro F1:", result["test"]["macro_f1_present_classes"])
    print("model:", model_path)
    print("metrics:", args.metrics)


if __name__ == "__main__":
    main()
