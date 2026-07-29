"""用途：在同一严格时序测试集上比较实时 Student、边缘 Qwen 与云端 Qwen Teacher。"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from traffic_system.decision_utils import (
    ACTION_TOKEN_TO_DECISION,
    DECISION_CLASSES,
    load_json,
    read_jsonl,
    rule_teacher_decision,
    save_json,
)
from traffic_system.edge_student import load_student_model, predict_student
from traffic_system.evaluate_future_truth_policy import (
    classification_report,
    load_evaluation_arrays,
    make_event,
    one_hot_probabilities,
)
from traffic_system.risk_labels import RISK_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare edge models with the cloud Qwen teacher on one untouched test set."
    )
    parser.add_argument(
        "--strict_test_jsonl",
        default="datasets/llm_sft_freeway_action_token_v9/test.jsonl",
    )
    parser.add_argument(
        "--qwen_result_json",
        default="results/llm/llama_cpp_v9_text_only_q6_nocache_jetson02_gpu_test.json",
    )
    parser.add_argument(
        "--student_model",
        default="models/edge_student_freeway_joint_metis4.json",
    )
    parser.add_argument(
        "--student_teacher_labels",
        default="datasets/freeway_teacher_labels_qwen9b_safe_joint_metis4.jsonl",
    )
    parser.add_argument(
        "--qwen_train_jsonl",
        default="datasets/llm_sft_freeway_action_token_v9/train.jsonl",
    )
    parser.add_argument(
        "--qwen_val_jsonl",
        default="datasets/llm_sft_freeway_action_token_v9/val.jsonl",
    )
    parser.add_argument(
        "--data_npz",
        default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz",
    )
    parser.add_argument(
        "--risk_labels",
        default="datasets/risk_labels_pems08_metis4.npz",
    )
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--student_test_ratio", type=float, default=0.25)
    parser.add_argument("--retention_threshold", type=float, default=0.80)
    parser.add_argument(
        "--output_json",
        default="results/decision/edge_cloud_model_retention.json",
    )
    parser.add_argument(
        "--report_md",
        default="results/decision/edge_cloud_model_retention.md",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def action_token(value: Any, field_name: str) -> str:
    token = str(value).strip().upper()
    if token not in ACTION_TOKEN_TO_DECISION:
        raise ValueError("Unsupported {} token: {!r}".format(field_name, value))
    return token


def event_ids(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    ids = [str(row.get("event_id", "")) for row in rows]
    if any(not event_id for event_id in ids):
        raise ValueError("Every test row must contain event_id.")
    if len(ids) != len(set(ids)):
        raise ValueError("Strict test set contains duplicate event_id values.")
    return ids


def event_id_set(rows: Sequence[Mapping[str, Any]]) -> Set[str]:
    ids = {str(row.get("event_id", "")) for row in rows}
    if "" in ids:
        raise ValueError("Every dataset row must contain event_id.")
    return ids


def qwen_predictions(result: Mapping[str, Any]) -> Dict[str, str]:
    samples = result.get("samples")
    prediction_field = "prediction"
    if not isinstance(samples, list):
        samples = result.get("examples")
        prediction_field = "parsed"
    if not isinstance(samples, list):
        raise ValueError("Qwen result must contain samples or examples.")

    predictions: Dict[str, str] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        event_id = str(sample.get("event_id", ""))
        if not event_id:
            continue
        token = action_token(sample.get(prediction_field, ""), prediction_field)
        if event_id in predictions:
            raise ValueError("Duplicate Qwen prediction for {}.".format(event_id))
        predictions[event_id] = token
    return predictions


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> Dict[str, float]:
    if total <= 0:
        return {"lower": 0.0, "upper": 0.0}
    rate = correct / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    margin /= denominator
    return {
        "lower": round(max(0.0, center - margin), 6),
        "upper": round(min(1.0, center + margin), 6),
    }


def accuracy_report(references: Sequence[str], predictions: Sequence[str]) -> Dict[str, Any]:
    if len(references) != len(predictions):
        raise ValueError("Reference and prediction counts differ.")
    report = classification_report(references, predictions, DECISION_CLASSES)
    correct = sum(reference == prediction for reference, prediction in zip(references, predictions))
    report["correct"] = correct
    report["accuracy_95ci"] = {
        "method": "Wilson score interval",
        **wilson_interval(correct, len(references)),
    }
    return report


def retention_summary(
    edge_accuracy: float,
    cloud_accuracy: float,
    threshold: float,
) -> Dict[str, Any]:
    ratio = edge_accuracy / cloud_accuracy if cloud_accuracy > 0.0 else None
    return {
        "ratio": round(ratio, 6) if ratio is not None else None,
        "threshold": threshold,
        "meets_threshold": bool(ratio is not None and ratio + 1e-12 >= threshold),
    }


def sample_id_set(rows: Sequence[Mapping[str, Any]]) -> Set[int]:
    result: Set[int] = set()
    for row in rows:
        path_value = row.get("event_path")
        if not path_value:
            raise ValueError("Teacher label row is missing event_path.")
        event = load_json(resolve_path(str(path_value)))
        result.add(int(event["sample_id"]))
    return result


def student_training_sample_ids(rows: Sequence[Mapping[str, Any]], test_ratio: float) -> Set[int]:
    all_samples = sorted(sample_id_set(rows))
    if len(all_samples) < 2:
        raise ValueError("Student labels need at least two timestamp groups.")
    test_count = max(1, int(round(len(all_samples) * test_ratio)))
    return set(all_samples[:-test_count])


def ordered_sha256(values: Sequence[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def future_reference(
    row: Mapping[str, Any],
    arrays: Mapping[str, Any],
    split: str,
    top_k: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    event_path = resolve_path(str(row["event_path"]))
    event = load_json(event_path)
    sample_id = int(event["sample_id"])
    partition_id = int(event["partition_id"])
    if str(event.get("sample_split")) != split:
        raise ValueError("{} is not in the {} split.".format(event_path, split))
    if sample_id < 0 or sample_id >= arrays["split_target"].shape[0]:
        raise ValueError("sample_id {} is outside the {} split.".format(sample_id, split))

    node_probabilities = one_hot_probabilities(
        arrays["node_labels"][sample_id], len(RISK_CLASSES)
    )
    region_probabilities = one_hot_probabilities(
        arrays["region_labels"][sample_id], len(RISK_CLASSES)
    )
    reference_event = make_event(
        split,
        sample_id,
        partition_id,
        arrays["label_partitions"],
        node_probabilities,
        region_probabilities,
        arrays["split_target"][sample_id],
        event["control_capabilities"],
        top_k,
        "future_observation_fcm_reference",
    )
    reference_decision = rule_teacher_decision(
        reference_event, "future_truth_policy_reference"
    )
    return event, reference_decision


def class_distribution(values: Sequence[str]) -> Dict[str, int]:
    counts = Counter(values)
    return {name: int(counts.get(name, 0)) for name in DECISION_CLASSES}


def write_report(result: Mapping[str, Any], path: Path) -> None:
    fidelity = result["teacher_fidelity"]
    task = result["future_proxy_task"]
    count = int(result["dataset"]["events"])
    required = int(fidelity["required_correct_for_threshold"])

    lines = [
        "# 边缘模型相对云端 Qwen3.5-9B 的准确率保持",
        "",
        "## 评估口径",
        "",
        "- 三个模型使用同一批 {} 条严格时序留出事件。".format(count),
        "- Teacher 复现率：以云端 Qwen3.5-9B 的安全约束决策标签为目标。",
        "- 任务准确率：以未来真实 flow、occupancy、speed 经冻结 FCM 和固定策略得到的数据参考为目标。",
        "- 80% 保持率按 `边缘任务准确率 / 云端任务准确率` 计算。",
        "",
        "## 结果",
        "",
        "| 模型 | 9B Teacher 复现率 | 达到 80% | 未来代理任务 Accuracy | 相对 9B 任务保持率 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    display_names = {
        "cloud_qwen_9b": "云端 Qwen3.5-9B",
        "realtime_student": "实时 MLP Student",
        "edge_qwen_0_8b": "边缘 Qwen3.5-0.8B Q6",
    }
    for key in ("cloud_qwen_9b", "realtime_student", "edge_qwen_0_8b"):
        fidelity_item = fidelity["models"][key]
        task_item = task["models"][key]
        fidelity_pass = "基准" if key == "cloud_qwen_9b" else (
            "是" if fidelity_item["meets_80_percent"] else "否"
        )
        task_retention = "100.00%" if key == "cloud_qwen_9b" else "{:.2%}".format(
            task_item["retention_vs_cloud"]["ratio"]
        )
        lines.append(
            "| {} | {:.2%}（{}/{}） | {} | {:.2%}（{}/{}） | {} |".format(
                display_names[key],
                fidelity_item["metrics"]["accuracy"],
                fidelity_item["metrics"]["correct"],
                count,
                fidelity_pass,
                task_item["metrics"]["accuracy"],
                task_item["metrics"]["correct"],
                count,
                task_retention,
            )
        )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "按任务准确率保持计算，实时 Student 和边缘 Qwen 均达到云端的 80%。",
            "按逐条复现 9B Teacher 决策计算，两者目前都是 28/{}，即 77.78%；至少需要 {}/{} 才达到 80%，还差 1 条。".format(
                count, required, count
            ),
            "",
            "这组测试只有 {} 条，置信区间较宽，只能作为初步结果。云端 9B 对未来代理参考的绝对准确率也只有 {:.2%}，因此不能把 Teacher 标签当成真实道路控制真值。正式材料应扩大严格时序测试集后再下最终结论。".format(
                count, task["models"]["cloud_qwen_9b"]["metrics"]["accuracy"]
            ),
            "",
            "## 数据完整性",
            "",
            "- Student 训练时间组与本测试集重叠：{}".format(
                len(result["split_integrity"]["student_train_overlap_event_ids"])
            ),
            "- Qwen SFT 训练样本与本测试集重叠：{}".format(
                len(result["split_integrity"]["qwen_train_overlap_event_ids"])
            ),
            "- Qwen 验证样本与本测试集重叠：{}".format(
                len(result["split_integrity"]["qwen_val_overlap_event_ids"])
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.retention_threshold <= 1.0:
        raise ValueError("retention_threshold must be in (0, 1].")
    if not 0.0 < args.student_test_ratio < 1.0:
        raise ValueError("student_test_ratio must be in (0, 1).")
    if args.top_k <= 0:
        raise ValueError("top_k must be positive.")

    strict_rows = read_jsonl(resolve_path(args.strict_test_jsonl))
    strict_ids = event_ids(strict_rows)
    if not strict_rows:
        raise ValueError("Strict test set is empty.")

    qwen_result = load_json(resolve_path(args.qwen_result_json))
    qwen_by_event = qwen_predictions(qwen_result)
    missing_qwen = sorted(set(strict_ids) - set(qwen_by_event))
    extra_qwen = sorted(set(qwen_by_event) - set(strict_ids))
    if missing_qwen or extra_qwen:
        raise ValueError(
            "Qwen predictions do not match strict test IDs; missing={}, extra={}.".format(
                missing_qwen[:5], extra_qwen[:5]
            )
        )

    student_model = load_student_model(resolve_path(args.student_model))
    arrays = load_evaluation_arrays(
        resolve_path(args.data_npz), resolve_path(args.risk_labels), args.split
    )

    qwen_train_rows = read_jsonl(resolve_path(args.qwen_train_jsonl))
    qwen_val_rows = read_jsonl(resolve_path(args.qwen_val_jsonl))
    student_label_rows = read_jsonl(resolve_path(args.student_teacher_labels))
    strict_set = set(strict_ids)
    qwen_train_ids = event_id_set(qwen_train_rows)
    qwen_val_ids = event_id_set(qwen_val_rows)
    student_train_samples = student_training_sample_ids(
        student_label_rows, args.student_test_ratio
    )

    future_targets: List[str] = []
    teacher_targets: List[str] = []
    student_predictions: List[str] = []
    edge_qwen_predictions: List[str] = []
    records: List[Dict[str, Any]] = []
    student_train_overlap_event_ids: List[str] = []

    for row in strict_rows:
        event_id = str(row["event_id"])
        teacher_token = action_token(row.get("target", ""), "target")
        qwen_token = qwen_by_event[event_id]
        event, reference = future_reference(row, arrays, args.split, args.top_k)
        sample_id = int(event["sample_id"])
        if sample_id in student_train_samples:
            student_train_overlap_event_ids.append(event_id)
        student_decision, student_confidence, _ = predict_student(event, student_model)
        teacher_decision = ACTION_TOKEN_TO_DECISION[teacher_token]
        edge_qwen_decision = ACTION_TOKEN_TO_DECISION[qwen_token]
        future_decision = str(reference["decision"])

        future_targets.append(future_decision)
        teacher_targets.append(teacher_decision)
        student_predictions.append(student_decision)
        edge_qwen_predictions.append(edge_qwen_decision)
        records.append(
            {
                "event_id": event_id,
                "sample_id": sample_id,
                "partition_id": int(event["partition_id"]),
                "future_proxy_reference": future_decision,
                "cloud_qwen_9b": teacher_decision,
                "realtime_student": student_decision,
                "realtime_student_confidence": round(float(student_confidence), 6),
                "edge_qwen_0_8b": edge_qwen_decision,
                "student_matches_teacher": student_decision == teacher_decision,
                "edge_qwen_matches_teacher": edge_qwen_decision == teacher_decision,
                "cloud_matches_future_proxy": teacher_decision == future_decision,
                "student_matches_future_proxy": student_decision == future_decision,
                "edge_qwen_matches_future_proxy": edge_qwen_decision == future_decision,
            }
        )

    model_predictions = {
        "cloud_qwen_9b": teacher_targets,
        "realtime_student": student_predictions,
        "edge_qwen_0_8b": edge_qwen_predictions,
    }
    threshold_correct = int(math.ceil(args.retention_threshold * len(strict_rows)))
    fidelity_models: Dict[str, Any] = {}
    task_models: Dict[str, Any] = {}
    cloud_task_metrics = accuracy_report(future_targets, teacher_targets)

    for name, predictions in model_predictions.items():
        fidelity_metrics = accuracy_report(teacher_targets, predictions)
        task_metrics = accuracy_report(future_targets, predictions)
        fidelity_models[name] = {
            "metrics": fidelity_metrics,
            "meets_80_percent": fidelity_metrics["accuracy"] >= args.retention_threshold,
            "shortfall_correct_predictions": max(0, threshold_correct - fidelity_metrics["correct"]),
        }
        task_models[name] = {
            "metrics": task_metrics,
            "retention_vs_cloud": retention_summary(
                task_metrics["accuracy"], cloud_task_metrics["accuracy"], args.retention_threshold
            ),
        }

    result: Dict[str, Any] = {
        "task": "edge_model_cloud_qwen_accuracy_retention",
        "dataset": {
            "strict_test_jsonl": args.strict_test_jsonl,
            "events": len(strict_rows),
            "timestamp_groups": len({record["sample_id"] for record in records}),
            "ordered_event_id_sha256": ordered_sha256(strict_ids),
            "split": args.split,
            "reference_status": (
                "future flow/occupancy/speed plus frozen FCM labels and fixed safety policy; "
                "data-driven proxy, not manual traffic-control ground truth"
            ),
            "teacher_label_status": (
                "cached safety-constrained Qwen3.5-9B decisions; 100% teacher fidelity for the "
                "teacher is definitional, not an independently measured accuracy"
            ),
            "future_proxy_distribution": class_distribution(future_targets),
            "teacher_distribution": class_distribution(teacher_targets),
        },
        "split_integrity": {
            "student_train_overlap_event_ids": sorted(student_train_overlap_event_ids),
            "qwen_train_overlap_event_ids": sorted(strict_set & qwen_train_ids),
            "qwen_val_overlap_event_ids": sorted(strict_set & qwen_val_ids),
            "strict_holdout_verified": not (
                student_train_overlap_event_ids
                or strict_set & qwen_train_ids
                or strict_set & qwen_val_ids
            ),
        },
        "teacher_fidelity": {
            "definition": "exact action-class agreement with the cached cloud Qwen3.5-9B teacher",
            "threshold": args.retention_threshold,
            "required_correct_for_threshold": threshold_correct,
            "models": fidelity_models,
        },
        "future_proxy_task": {
            "definition": "decision accuracy against the independent future-observation proxy policy",
            "retention_formula": "edge_accuracy / cloud_qwen_9b_accuracy",
            "models": task_models,
        },
        "pairwise": {
            "student_qwen_agreement": round(
                sum(a == b for a, b in zip(student_predictions, edge_qwen_predictions))
                / len(strict_rows),
                6,
            )
        },
        "qwen_deployment_evidence": {
            "result_json": args.qwen_result_json,
            "runtime": qwen_result.get("runtime"),
            "model_path": qwen_result.get("model_path"),
            "valid_rate": qwen_result.get("valid_rate"),
            "ttft": qwen_result.get("ttft"),
        },
        "records": records,
    }
    output_path = resolve_path(args.output_json)
    report_path = resolve_path(args.report_md)
    save_json(result, output_path)
    write_report(result, report_path)
    print(
        json.dumps(
            {
                "events": len(strict_rows),
                "strict_holdout_verified": result["split_integrity"]["strict_holdout_verified"],
                "teacher_fidelity": {
                    name: item["metrics"]["accuracy"] for name, item in fidelity_models.items()
                },
                "future_proxy_accuracy": {
                    name: item["metrics"]["accuracy"] for name, item in task_models.items()
                },
                "future_proxy_retention": {
                    name: item["retention_vs_cloud"]["ratio"] for name, item in task_models.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("saved:", output_path)
    print("report:", report_path)


if __name__ == "__main__":
    main()
