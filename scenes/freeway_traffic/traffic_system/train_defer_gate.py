"""用途：按时间隔离训练边云选择门控器，并在独立 test 时段检验收益。"""

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier

from traffic_system.cloud_coordinator import load_cloud_model
from traffic_system.decision_utils import DECISION_CLASSES, save_json
from traffic_system.defer_gate import (
    GATE_CLASSES,
    build_gate_features,
    export_portable_defer_gate,
    predict_defer_gate,
    preferred_gate_targets,
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
from traffic_system.train_future_calibrated_cloud_coordinator import (
    extract_split,
    make_model,
    string_labels,
    temporal_tune_indices,
)
from traffic_system.train_joint_risk_astgcn import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a temporally isolated learning-to-defer gate.")
    parser.add_argument("--config", default="configurations/PEMS08_astgcn.conf")
    parser.add_argument("--data_npz", default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz")
    parser.add_argument("--risk_labels", default="datasets/risk_labels_pems08_metis4.npz")
    parser.add_argument(
        "--checkpoint",
        default="experiments/PEMS08/joint_risk_astgcn_metis4_flowprio2_frozen/best.pt",
    )
    parser.add_argument("--student_model", default="models/edge_student_freeway_joint_metis4.json")
    parser.add_argument("--cloud_model", default="models/cloud_coordinator_future_calibrated.joblib")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--tune_ratio", type=float, default=0.25)
    parser.add_argument("--purge_radius", type=int, default=24)
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--portable_trees", type=int, default=40)
    parser.add_argument("--max_depth", type=int, default=14)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    parser.add_argument("--thresholds", default="0.0,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--accuracy_tolerance", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_model", default="models/edge_defer_gate.joblib")
    parser.add_argument("--output_portable", default="models/edge_defer_gate.npz")
    parser.add_argument("--output_json", default="results/decision/edge_defer_gate_eval.json")
    parser.add_argument("--output_md", default="results/decision/edge_defer_gate_eval.md")
    return parser.parse_args()


def parse_thresholds(value: str) -> List[float]:
    thresholds = sorted(set(float(item.strip()) for item in value.split(",") if item.strip()))
    if not thresholds or thresholds[0] < 0.0 or thresholds[-1] >= 1.0:
        raise ValueError("thresholds must be in [0, 1)")
    return thresholds


def make_gate_model(args: argparse.Namespace) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )


def gated_predictions(
    choices: np.ndarray,
    rule: np.ndarray,
    student: np.ndarray,
    cloud: np.ndarray,
) -> np.ndarray:
    result = np.asarray(rule, dtype=np.int64).copy()
    result[choices == GATE_CLASSES.index("edge_student")] = student[
        choices == GATE_CLASSES.index("edge_student")
    ]
    result[choices == GATE_CLASSES.index("defer_cloud")] = cloud[
        choices == GATE_CLASSES.index("defer_cloud")
    ]
    return result


def evaluate_predictions(reference: np.ndarray, prediction: np.ndarray) -> Dict[str, Any]:
    return classification_report(
        string_labels(reference),
        string_labels(prediction),
        DECISION_CLASSES,
    )


def select_threshold(candidates: Sequence[Dict[str, Any]], tolerance: float) -> Dict[str, Any]:
    best_accuracy = max(float(row["accuracy"]) for row in candidates)
    eligible = [row for row in candidates if float(row["accuracy"]) >= best_accuracy - tolerance]
    selected = min(
        eligible,
        key=lambda row: (float(row["cloud_request_rate"]), -float(row["accuracy"])),
    )
    return {
        **selected,
        "best_accuracy": round(best_accuracy, 6),
        "accuracy_tolerance": tolerance,
    }


def choice_counts(choices: np.ndarray) -> Dict[str, int]:
    return {
        name: int(np.sum(choices == class_id))
        for class_id, name in enumerate(GATE_CLASSES)
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# 边云选择门控器评估",
        "",
        "> 决策参考仍是未来三变量生成的 FCM 代理策略，不是人工控制真值。",
        "",
        "- 门控器只在验证集前段训练；阈值在隔离后的验证集尾段选择；test 不参与训练和选参。",
        "- 本地规则正确时优先规则；仅 Student 正确时选择 Student；两者都错时学习请求云端。",
        "- 这是 learning-to-defer 思想的工程化门控器，不宣称复现论文中的端到端校准损失。",
        "",
        "| 方法 | Test Accuracy | Macro-F1 | 云请求率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("rule", "本地固定策略"),
        ("student", "边缘 Student"),
        ("cloud", "云协调器"),
        ("gated", "选择性协同"),
        ("local_oracle", "规则/Student 理论上界"),
    ):
        row = result["test"][key]
        cloud_rate = row.get("cloud_request_rate")
        lines.append(
            "| {} | {:.2%} | {:.2%} | {} |".format(
                label,
                row["accuracy"],
                row["macro_f1_present_classes"],
                "{:.2%}".format(cloud_rate) if cloud_rate is not None else "-",
            )
        )
    lines.extend(
        [
            "",
            "阈值：`{}`；test 选择计数：`{}`。".format(
                result["selected_threshold"]["confidence_threshold"],
                result["test"]["gated"]["choice_counts"],
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    if args.accuracy_tolerance < 0.0 or not 0 < args.portable_trees <= args.trees:
        raise ValueError("accuracy_tolerance or portable_trees is invalid")
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
    )
    if validation["feature_names"] != test["feature_names"]:
        raise ValueError("Validation and test feature schemas differ")
    if validation["feature_names"] != list(cloud_payload["feature_names"]):
        raise ValueError("Cloud coordinator feature schema mismatch")
    train_indices, tune_indices, split_info = temporal_tune_indices(
        validation["sample_ids"], args.tune_ratio, args.purge_radius
    )
    val_gate_x = build_gate_features(
        validation["x"],
        validation["rule"],
        validation["student"],
        validation["student_confidence"],
    )
    test_gate_x = build_gate_features(
        test["x"],
        test["rule"],
        test["student"],
        test["student_confidence"],
    )
    gate_targets = preferred_gate_targets(
        validation["y"], validation["rule"], validation["student"]
    )
    gate_model = make_gate_model(args)
    gate_model.fit(val_gate_x[train_indices], gate_targets[train_indices])

    cloud_tune_model = make_model(160, None, 2, args.seed + 1)
    cloud_tune_model.fit(validation["x"][train_indices], validation["y"][train_indices])
    tune_cloud = np.asarray(cloud_tune_model.predict(validation["x"][tune_indices]), dtype=np.int64)
    tune_probabilities = gate_model.predict_proba(val_gate_x[tune_indices])
    tune_columns = {int(value): index for index, value in enumerate(gate_model.classes_)}
    full_tune_probabilities = np.zeros((len(tune_indices), len(GATE_CLASSES)), dtype=np.float64)
    for class_id, column in tune_columns.items():
        full_tune_probabilities[:, class_id] = tune_probabilities[:, column]
    candidates = []
    for threshold in thresholds:
        choices = np.argmax(full_tune_probabilities, axis=1).astype(np.int64)
        confidence = np.max(full_tune_probabilities, axis=1)
        choices[confidence < threshold] = GATE_CLASSES.index("defer_cloud")
        predictions = gated_predictions(
            choices,
            validation["rule"][tune_indices],
            validation["student"][tune_indices],
            tune_cloud,
        )
        report = evaluate_predictions(validation["y"][tune_indices], predictions)
        candidates.append(
            {
                "confidence_threshold": threshold,
                "accuracy": report["accuracy"],
                "macro_f1": report["macro_f1_present_classes"],
                "cloud_request_rate": round(
                    float(np.mean(choices == GATE_CLASSES.index("defer_cloud"))), 6
                ),
                "choice_counts": choice_counts(choices),
            }
        )
    selected = select_threshold(candidates, args.accuracy_tolerance)
    payload = {
        "model": gate_model,
        "feature_dim": int(val_gate_x.shape[1]),
        "base_feature_names": validation["feature_names"],
        "gate_classes": GATE_CLASSES,
        "decision_classes": DECISION_CLASSES,
        "metadata": {
            "task": "temporally_isolated_edge_cloud_defer_gate",
            "training_split": "validation_early",
            "confidence_threshold": selected["confidence_threshold"],
            "target_definition": "rule if correct, else student if correct, else defer cloud",
        },
    }
    model_path = Path(args.output_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_path, compress=3)
    portable_path = Path(args.output_portable)
    portable_model = copy.copy(gate_model)
    portable_model.estimators_ = portable_model.estimators_[: args.portable_trees]
    portable_model.n_estimators = args.portable_trees
    portable_payload = dict(payload)
    portable_payload["model"] = portable_model
    export_portable_defer_gate(portable_payload, portable_path)

    test_cloud = np.asarray(cloud_payload["model"].predict(test["x"]), dtype=np.int64)
    test_choices, _ = predict_defer_gate(test_gate_x, payload)
    test_gated = gated_predictions(test_choices, test["rule"], test["student"], test_cloud)
    local_oracle = np.where(test["rule"] == test["y"], test["rule"], test["student"])
    test_result: Dict[str, Any] = {
        "rule": evaluate_predictions(test["y"], test["rule"]),
        "student": evaluate_predictions(test["y"], test["student"]),
        "cloud": evaluate_predictions(test["y"], test_cloud),
        "gated": evaluate_predictions(test["y"], test_gated),
        "local_oracle": evaluate_predictions(test["y"], local_oracle),
    }
    test_result["gated"]["cloud_request_rate"] = round(
        float(np.mean(test_choices == GATE_CLASSES.index("defer_cloud"))), 6
    )
    test_result["gated"]["choice_counts"] = choice_counts(test_choices)
    result = {
        "task": "edge_cloud_learning_to_defer_gate",
        "reference": {
            "source": "future flow/occupancy/speed -> frozen FCM risk -> fixed safety policy",
            "status": "proxy reference, not manual control ground truth",
            "test_used_for_training_or_threshold_selection": False,
        },
        "temporal_split": split_info,
        "gate_training_class_counts": choice_counts(gate_targets[train_indices]),
        "threshold_candidates": candidates,
        "selected_threshold": selected,
        "test": test_result,
        "artifact": {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "portable_path": str(portable_path),
            "portable_size_bytes": portable_path.stat().st_size,
            "portable_trees": args.portable_trees,
        },
    }
    save_json(result, Path(args.output_json))
    write_markdown(result, Path(args.output_md))
    print("selected:", selected)
    print("test gated:", test_result["gated"]["accuracy"], test_result["gated"]["cloud_request_rate"])
    print("artifact:", model_path)


if __name__ == "__main__":
    main()
