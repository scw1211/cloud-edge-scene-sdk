"""用途：用未来真实三变量状态独立评估风险预测到交通决策的完整链路。"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from traffic_system.decision_utils import (
    DECISION_CLASSES,
    build_decision_from_student_class,
    read_jsonl,
    rule_teacher_decision,
    save_json,
)
from traffic_system.edge_student import load_student_model, predict_student
from traffic_system.generate_joint_edge_events import parse_sample_spec
from traffic_system.infer_joint_risk_astgcn import (
    build_control_capabilities,
    build_model_from_checkpoint,
    build_top_nodes,
    ensure_finite_outputs,
    load_adjacency,
    load_config,
    region_upload_policy,
    summarize_region,
    torch_load_trusted,
)
from traffic_system.risk_labels import (
    RISK_CLASSES,
    confusion_matrix,
    denormalize,
    enable_numpy_pickle_compatibility,
)
from traffic_system.train_joint_risk_astgcn import clip_physical_state, select_device


CRITICAL_DECISIONS = {
    "variable_speed_limit",
    "ramp_metering",
    "regional_coordination",
    "reroute",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predicted traffic decisions against references derived from "
            "future flow/occupancy/speed and frozen FCM labels."
        )
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
        "--student_training_labels",
        default="datasets/freeway_teacher_labels_qwen9b_safe_joint_metis4.jsonl",
    )
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--samples", default="all", help="all, start:end:step, or comma-separated ids")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--student_purge_radius", type=int, default=24)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_json",
        default="results/decision/future_truth_policy_evaluation.json",
    )
    parser.add_argument(
        "--report_md",
        default="results/decision/future_truth_policy_evaluation.md",
    )
    return parser.parse_args()


def load_evaluation_arrays(data_path: Path, labels_path: Path, split: str) -> Dict[str, Any]:
    with np.load(data_path) as data:
        required = ["{}_x".format(split), "{}_target".format(split), "mean", "std"]
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError("Missing data arrays: {}".format(", ".join(missing)))
        arrays = {
            "split_x": data["{}_x".format(split)],
            "split_target": data["{}_target".format(split)],
            "mean": data["mean"],
            "std": data["std"],
        }
    enable_numpy_pickle_compatibility()
    with np.load(labels_path, allow_pickle=True) as labels:
        node_key = "{}_node_label".format(split)
        region_key = "{}_region_label".format(split)
        required = [node_key, region_key, "partitions"]
        missing = [name for name in required if name not in labels.files]
        if missing:
            raise ValueError("Missing risk-label arrays: {}".format(", ".join(missing)))
        arrays.update(
            {
                "node_labels": labels[node_key],
                "region_labels": labels[region_key],
                "label_partitions": [
                    [int(node_id) for node_id in part]
                    for part in labels["partitions"].tolist()
                ],
            }
        )
    sample_count = arrays["split_x"].shape[0]
    for name in ("split_target", "node_labels", "region_labels"):
        if arrays[name].shape[0] != sample_count:
            raise ValueError("{} has a different sample count.".format(name))
    return arrays


def selected_sample_ids(spec: str, sample_count: int) -> List[int]:
    sample_ids = list(range(sample_count)) if spec.strip().lower() == "all" else parse_sample_spec(spec)
    if not sample_ids:
        raise ValueError("Sample selection is empty.")
    if min(sample_ids) < 0 or max(sample_ids) >= sample_count:
        raise ValueError("Sample selection is outside split size {}.".format(sample_count))
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Sample selection contains duplicates.")
    return sample_ids


def one_hot_probabilities(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError("Risk label is outside the configured classes.")
    return np.eye(num_classes, dtype=np.float32)[labels]


def make_event(
    split: str,
    sample_id: int,
    partition_id: int,
    partitions: Sequence[Sequence[int]],
    node_probs: np.ndarray,
    region_probs: np.ndarray,
    traffic_state: np.ndarray,
    control_capabilities: Dict[str, Any],
    top_k: int,
    risk_source: str,
    risk_calibrator: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    managed_nodes = [int(node_id) for node_id in partitions[partition_id]]
    summary = summarize_region(
        managed_nodes,
        node_probs,
        region_probs[partition_id],
        risk_calibrator=risk_calibrator,
    )
    severe_count = int(summary["node_risk_counts"].get("severe", 0))
    high_count = int(summary["node_risk_counts"].get("high", 0))
    upload_required, upload_level = region_upload_policy(
        summary["region_risk_level"],
        summary["max_node_risk_level"],
        severe_count,
        high_count,
    )
    edge_id = "edge_node_{}".format(partition_id)
    return {
        "scene": "freeway_traffic_management",
        "task": "edge_freeway_congestion_risk_assessment",
        "dataset": "PEMS08",
        "risk_source": risk_source,
        "event_id": "freeway_{}_sample_{:04d}_{}".format(split, sample_id, edge_id),
        "edge_id": edge_id,
        "region_id": "region_{}".format(partition_id),
        "partition_id": partition_id,
        "num_partitions": len(partitions),
        "sample_split": split,
        "sample_id": sample_id,
        "time_step_minutes": 5,
        "prediction_steps": int(traffic_state.shape[-1]),
        "prediction_horizon_minutes": int(traffic_state.shape[-1]) * 5,
        "managed_node_ids": managed_nodes,
        "control_capabilities": control_capabilities,
        "region_summary": summary,
        "upload_required": upload_required,
        "upload_level": upload_level,
        "top_k_risk_nodes": build_top_nodes(managed_nodes, node_probs, traffic_state, top_k),
    }


def classification_report(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    classes: Sequence[str],
) -> Dict[str, Any]:
    class_to_id = {name: index for index, name in enumerate(classes)}
    true_ids = np.asarray([class_to_id[name] for name in true_labels], dtype=np.int64)
    pred_ids = np.asarray([class_to_id[name] for name in predicted_labels], dtype=np.int64)
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(matrix, (true_ids, pred_ids), 1)
    per_class: Dict[str, Any] = {}
    present_f1 = []
    weighted_f1 = 0.0
    for class_id, name in enumerate(classes):
        support = int(matrix[class_id].sum())
        predicted = int(matrix[:, class_id].sum())
        true_positive = int(matrix[class_id, class_id])
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            present_f1.append(f1)
            weighted_f1 += support * f1
        per_class[name] = {
            "support": support,
            "precision": round(precision, 6) if predicted else None,
            "recall": round(recall, 6) if support else None,
            "f1": round(f1, 6) if support or predicted else None,
        }
    total = len(true_ids)
    return {
        "total": total,
        "accuracy": round(float(np.mean(true_ids == pred_ids)), 6) if total else 0.0,
        "macro_f1_present_classes": round(float(np.mean(present_f1)), 6) if present_f1 else 0.0,
        "weighted_f1": round(weighted_f1 / total, 6) if total else 0.0,
        "class_names": list(classes),
        "matrix": matrix.tolist(),
        "per_class": per_class,
    }


def stratified_bootstrap_accuracy_ci(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    true_array = np.asarray(true_labels)
    predicted_array = np.asarray(predicted_labels)
    if true_array.size == 0:
        return {"method": "class_stratified_bootstrap", "samples": samples, "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(true_array == label) for label in sorted(set(true_labels))]
    estimates = np.empty(samples, dtype=np.float64)
    for bootstrap_id in range(samples):
        correct = 0
        total = 0
        for indices in strata:
            selected = rng.choice(indices, size=len(indices), replace=True)
            correct += int(np.sum(true_array[selected] == predicted_array[selected]))
            total += len(selected)
        estimates[bootstrap_id] = correct / max(1, total)
    return {
        "method": "class_stratified_bootstrap",
        "samples": samples,
        "confidence_level": 0.95,
        "lower": round(float(np.percentile(estimates, 2.5)), 6),
        "upper": round(float(np.percentile(estimates, 97.5)), 6),
        "seed": seed,
    }


def set_overlap_metrics(reference_sets: Iterable[Set[Any]], predicted_sets: Iterable[Set[Any]]) -> Dict[str, Any]:
    true_positive = false_positive = false_negative = exact = count = 0
    active_f1_values = []
    for reference, predicted in zip(reference_sets, predicted_sets):
        count += 1
        true_positive += len(reference & predicted)
        false_positive += len(predicted - reference)
        false_negative += len(reference - predicted)
        exact += int(reference == predicted)
        if reference:
            denominator = 2 * len(reference & predicted) + len(predicted - reference) + len(reference - predicted)
            active_f1_values.append(2 * len(reference & predicted) / denominator if denominator else 1.0)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": count,
        "exact_match_rate": round(exact / count, 6) if count else 0.0,
        "micro_precision": round(precision, 6),
        "micro_recall": round(recall, 6),
        "micro_f1": round(f1, 6),
        "mean_f1_reference_active": round(float(np.mean(active_f1_values)), 6) if active_f1_values else None,
    }


def decision_sets(decision: Dict[str, Any]) -> Tuple[Set[int], Set[str]]:
    nodes = {
        int(node_id)
        for node_id in decision.get("affected_nodes", [])
        if isinstance(node_id, (int, np.integer))
    }
    action_types = {
        str(action.get("type"))
        for action in decision.get("actions", [])
        if isinstance(action, dict) and action.get("type")
    }
    return nodes, action_types


def student_training_sample_ids(path: Path) -> Set[int]:
    if not path.exists():
        return set()
    sample_ids = set()
    for row in read_jsonl(path):
        event_id = str(row.get("event_id", row.get("event_path", "")))
        match = re.search(r"sample_(\d+)", event_id)
        if match:
            sample_ids.add(int(match.group(1)))
    return sample_ids


def evaluate_decisions(rows: Sequence[Dict[str, Any]], prediction_key: str, bootstrap_samples: int, seed: int) -> Dict[str, Any]:
    references = [str(row["reference_decision"]) for row in rows]
    predictions = [str(row[prediction_key]) for row in rows]
    report = classification_report(references, predictions, DECISION_CLASSES)
    report["accuracy_95ci"] = stratified_bootstrap_accuracy_ci(
        references, predictions, bootstrap_samples, seed
    )
    critical = [index for index, label in enumerate(references) if label in CRITICAL_DECISIONS]
    report["critical_intervention_recall"] = round(
        sum(predictions[index] in CRITICAL_DECISIONS for index in critical) / len(critical), 6
    ) if critical else None
    report["critical_support"] = len(critical)
    return report


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    rule = result["decision_evaluation"]["predicted_risk_rule"]
    student = result["decision_evaluation"]["edge_student_all"]
    purged = result["decision_evaluation"].get("edge_student_overlap_purged")
    node = result["risk_evaluation"]["node"]
    region = result["risk_evaluation"]["region"]
    lines = [
        "# 基于未来真实状态的交通决策评测",
        "",
        "## 评测口径",
        "",
        "- 输入：ASTGCN 根据历史窗口预测得到的节点/区域风险。",
        "- 独立参考：测试集未来 12 步真实 flow、occupancy、speed，经冻结 FCM 标签和同一安全策略映射得到。",
        "- 测试范围：{} 个时间窗口、{} 个区域事件；ASTGCN 未使用 test split 训练。".format(
            result["samples"]["timestamp_count"], result["samples"]["region_event_count"]
        ),
        "- 这里是数据驱动的参考策略评测，不冒充人工交通管控真值。",
        "",
        "## 结果",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        "| 节点风险 Accuracy / Macro-F1 | {:.2%} / {:.2%} |".format(node["accuracy"], node["macro_f1"]),
        "| 区域风险 Accuracy / Macro-F1 | {:.2%} / {:.2%} |".format(region["accuracy"], region["macro_f1"]),
        "| 预测风险 + 安全规则 决策 Accuracy | {:.2%} |".format(rule["accuracy"]),
        "| 决策 Accuracy 95% CI | [{:.2%}, {:.2%}] |".format(
            rule["accuracy_95ci"]["lower"], rule["accuracy_95ci"]["upper"]
        ),
        "| 高风险干预召回率 | {:.2%} |".format(rule["critical_intervention_recall"]),
        "| 动作类型集合 Micro-F1 | {:.2%} |".format(
            result["set_evaluation"]["predicted_risk_rule"]["action_types"]["micro_f1"]
        ),
        "| 受影响节点 Micro-F1 | {:.2%} |".format(
            result["set_evaluation"]["predicted_risk_rule"]["affected_nodes"]["micro_f1"]
        ),
        "| 当前 Qwen 蒸馏 Student Accuracy（全量，仅补充） | {:.2%} |".format(student["accuracy"]),
    ]
    if purged:
        lines.append(
            "| Student Accuracy（剔除训练窗口重叠） | {:.2%} |".format(purged["accuracy"])
        )
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "当前 Qwen 蒸馏 Student 的旧训练样本来自 test 时间轴，因此其全量结果不能作为严格泛化证据。"
            "主结果采用未从 test 决策标签学习的‘预测风险 + 固定安全规则’链路；Student 同时给出剔除重叠窗口后的补充结果。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.top_k <= 0 or args.student_purge_radius < 0:
        raise ValueError("batch_size/top_k must be positive and purge radius non-negative.")
    torch.set_num_threads(args.torch_threads)
    device = select_device(args.device)
    config = load_config(args.config)
    arrays = load_evaluation_arrays(Path(args.data_npz), Path(args.risk_labels), args.split)
    sample_ids = selected_sample_ids(args.samples, arrays["split_x"].shape[0])
    checkpoint = torch_load_trusted(Path(args.checkpoint), device)
    adj_mx, _ = load_adjacency(config)
    model_arrays = {
        "in_channels": int(arrays["split_x"].shape[2]),
        "output_dim": int(arrays["mean"].shape[2]),
    }
    model = build_model_from_checkpoint(config, model_arrays, adj_mx, checkpoint, device)
    partitions = [[int(node_id) for node_id in part] for part in checkpoint["partitions"]]
    if partitions != arrays["label_partitions"]:
        raise ValueError("Checkpoint and risk-label partitions differ.")
    capabilities = [
        build_control_capabilities(partitions, adj_mx, partition_id)
        for partition_id in range(len(partitions))
    ]
    student_model = load_student_model(Path(args.student_model))
    student_seen_ids = student_training_sample_ids(Path(args.student_training_labels))

    node_true: List[np.ndarray] = []
    node_predicted: List[np.ndarray] = []
    region_true: List[np.ndarray] = []
    region_predicted: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    forward_latencies = []
    started = time.perf_counter()

    for batch_start in range(0, len(sample_ids), args.batch_size):
        batch_ids = sample_ids[batch_start : batch_start + args.batch_size]
        x = torch.from_numpy(arrays["split_x"][batch_ids].astype(np.float32)).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_started = time.perf_counter()
        with torch.no_grad():
            outputs = model(x)
        ensure_finite_outputs(outputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_latencies.append((time.perf_counter() - batch_started) * 1000.0)

        forecast_raw = clip_physical_state(
            denormalize(outputs["forecast"].detach().cpu().numpy(), arrays["mean"], arrays["std"])
        )
        predicted_node_probs = F.softmax(outputs["node_logits"], dim=-1).detach().cpu().numpy()
        predicted_region_probs = F.softmax(outputs["region_logits"], dim=-1).detach().cpu().numpy()

        for batch_index, sample_id in enumerate(batch_ids):
            true_node_labels = arrays["node_labels"][sample_id]
            true_region_labels = arrays["region_labels"][sample_id]
            true_node_probs = one_hot_probabilities(true_node_labels, len(RISK_CLASSES))
            true_region_probs = one_hot_probabilities(true_region_labels, len(RISK_CLASSES))
            predicted_node_labels = np.argmax(predicted_node_probs[batch_index], axis=-1)
            predicted_region_labels = np.argmax(predicted_region_probs[batch_index], axis=-1)
            node_true.append(true_node_labels)
            node_predicted.append(predicted_node_labels)
            region_true.append(true_region_labels)
            region_predicted.append(predicted_region_labels)

            for partition_id in range(len(partitions)):
                predicted_event = make_event(
                    args.split,
                    sample_id,
                    partition_id,
                    partitions,
                    predicted_node_probs[batch_index],
                    predicted_region_probs[batch_index],
                    forecast_raw[batch_index],
                    capabilities[partition_id],
                    args.top_k,
                    "joint_astgcn_prediction",
                )
                reference_event = make_event(
                    args.split,
                    sample_id,
                    partition_id,
                    partitions,
                    true_node_probs,
                    true_region_probs,
                    arrays["split_target"][sample_id],
                    capabilities[partition_id],
                    args.top_k,
                    "future_observation_fcm_reference",
                )
                reference = rule_teacher_decision(reference_event, "future_truth_policy_reference")
                predicted_rule = rule_teacher_decision(predicted_event, "predicted_risk_rule")
                student_class, student_confidence, _ = predict_student(predicted_event, student_model)
                student = build_decision_from_student_class(
                    predicted_event,
                    student_class,
                    student_confidence,
                    "qwen_distilled_edge_student",
                )
                reference_nodes, reference_actions = decision_sets(reference)
                rule_nodes, rule_actions = decision_sets(predicted_rule)
                student_nodes, student_actions = decision_sets(student)
                overlaps_student_training = any(
                    abs(sample_id - seen_id) <= args.student_purge_radius
                    for seen_id in student_seen_ids
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "partition_id": partition_id,
                        "reference_decision": reference["decision"],
                        "predicted_rule_decision": predicted_rule["decision"],
                        "student_decision": student["decision"],
                        "student_confidence": round(float(student_confidence), 6),
                        "overlaps_student_training_window": overlaps_student_training,
                        "reference_nodes": reference_nodes,
                        "rule_nodes": rule_nodes,
                        "student_nodes": student_nodes,
                        "reference_actions": reference_actions,
                        "rule_actions": rule_actions,
                        "student_actions": student_actions,
                    }
                )
        completed = min(batch_start + len(batch_ids), len(sample_ids))
        print("[{}/{}] timestamp windows evaluated".format(completed, len(sample_ids)), flush=True)

    node_metrics = confusion_matrix(np.stack(node_true), np.stack(node_predicted))
    region_metrics = confusion_matrix(np.stack(region_true), np.stack(region_predicted))
    purged_rows = [row for row in rows if not row["overlaps_student_training_window"]]
    decision_evaluation = {
        "predicted_risk_rule": evaluate_decisions(
            rows, "predicted_rule_decision", args.bootstrap_samples, args.seed
        ),
        "edge_student_all": evaluate_decisions(
            rows, "student_decision", args.bootstrap_samples, args.seed + 1
        ),
    }
    if purged_rows:
        decision_evaluation["edge_student_overlap_purged"] = evaluate_decisions(
            purged_rows, "student_decision", args.bootstrap_samples, args.seed + 2
        )

    set_evaluation = {}
    for name, prefix in (("predicted_risk_rule", "rule"), ("edge_student_all", "student")):
        set_evaluation[name] = {
            "affected_nodes": set_overlap_metrics(
                (row["reference_nodes"] for row in rows),
                (row["{}_nodes".format(prefix)] for row in rows),
            ),
            "action_types": set_overlap_metrics(
                (row["reference_actions"] for row in rows),
                (row["{}_actions".format(prefix)] for row in rows),
            ),
        }

    elapsed_seconds = time.perf_counter() - started
    mistakes = [
        {
            "sample_id": row["sample_id"],
            "partition_id": row["partition_id"],
            "reference": row["reference_decision"],
            "predicted_rule": row["predicted_rule_decision"],
            "student": row["student_decision"],
        }
        for row in rows
        if row["reference_decision"] != row["predicted_rule_decision"]
    ][:50]
    result = {
        "task": "future_observation_grounded_policy_evaluation",
        "evaluation_scope": (
            "ASTGCN history-to-future prediction, node/region risk heads, and traffic decision mapping "
            "evaluated against references built from observed future flow/occupancy/speed and frozen FCM labels."
        ),
        "reference_status": (
            "data-driven reference policy, not manually annotated traffic-control ground truth"
        ),
        "strictness": {
            "astgcn_test_split_held_out_from_training": args.split == "test",
            "predicted_risk_rule_learns_no_test_decision_labels": True,
            "student_old_labels_use_test_timeline": bool(student_seen_ids and args.split == "test"),
            "student_overlap_purge_radius_samples": args.student_purge_radius,
            "student_overlap_purge_radius_minutes": args.student_purge_radius * 5,
        },
        "inputs": {
            "data_npz": args.data_npz,
            "risk_labels": args.risk_labels,
            "checkpoint": args.checkpoint,
            "student_model": args.student_model,
            "student_training_labels": args.student_training_labels,
            "split": args.split,
            "device": str(device),
        },
        "samples": {
            "timestamp_count": len(sample_ids),
            "region_event_count": len(rows),
            "sample_first": min(sample_ids),
            "sample_last": max(sample_ids),
            "student_training_timestamp_count": len(student_seen_ids),
            "student_overlap_purged_region_event_count": len(purged_rows),
        },
        "risk_evaluation": {"node": node_metrics, "region": region_metrics},
        "decision_evaluation": decision_evaluation,
        "set_evaluation": set_evaluation,
        "runtime": {
            "batch_size": args.batch_size,
            "forward_batches": len(forward_latencies),
            "mean_batch_forward_ms": round(float(np.mean(forward_latencies)), 6),
            "p95_batch_forward_ms": round(float(np.percentile(forward_latencies, 95)), 6),
            "evaluation_wall_seconds": round(elapsed_seconds, 6),
        },
        "representative_rule_mistakes": mistakes,
    }
    save_json(result, Path(args.output_json))
    write_markdown(result, Path(args.report_md))
    print("result:", args.output_json)
    print("report:", args.report_md)
    print("rule_decision_accuracy:", decision_evaluation["predicted_risk_rule"]["accuracy"])
    print("student_accuracy:", decision_evaluation["edge_student_all"]["accuracy"])


if __name__ == "__main__":
    main()
