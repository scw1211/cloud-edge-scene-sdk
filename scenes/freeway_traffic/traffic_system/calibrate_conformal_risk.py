"""用途：用时间隔离的验证集校准区域风险集合，并在独立测试集上评估。"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from traffic_system.conformal_risk import (
    SCHEMA_VERSION,
    aps_candidate_scores,
    class_conditional_quantiles,
    conformal_quantile,
    fit_temperature,
    prediction_mask,
    temperature_softmax,
    true_aps_scores,
)
from traffic_system.decision_utils import save_json
from traffic_system.infer_joint_risk_astgcn import (
    build_model_from_checkpoint,
    ensure_finite_outputs,
    load_adjacency,
    load_config,
    torch_load_trusted,
)
from traffic_system.risk_labels import RISK_CLASSES, enable_numpy_pickle_compatibility
from traffic_system.train_joint_risk_astgcn import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate conformal uncertainty for ASTGCN region risk.")
    parser.add_argument("--config", default="configurations/PEMS08_astgcn.conf")
    parser.add_argument("--data_npz", default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz")
    parser.add_argument("--risk_labels", default="datasets/risk_labels_pems08_metis4.npz")
    parser.add_argument(
        "--checkpoint",
        default="experiments/PEMS08/joint_risk_astgcn_metis4_flowprio2_frozen/best.pt",
    )
    parser.add_argument("--target_coverage", type=float, default=0.90)
    parser.add_argument("--temperature_fraction", type=float, default=0.50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--deployment_method", default="marginal_aps", choices=[
        "marginal_aps", "class_conditional_aps", "simultaneous_window_aps"
    ])
    parser.add_argument("--output_model", default="models/region_risk_conformal.json")
    parser.add_argument("--output_json", default="results/perception/region_risk_conformal_eval.json")
    parser.add_argument("--output_md", default="results/perception/region_risk_conformal_eval.md")
    return parser.parse_args()


def load_arrays(data_path: Path, labels_path: Path) -> Dict[str, np.ndarray]:
    with np.load(data_path) as data:
        required_data = ["val_x", "test_x"]
        missing_data = [name for name in required_data if name not in data.files]
        if missing_data:
            raise ValueError("Missing data arrays: {}.".format(", ".join(missing_data)))
        arrays = {name: data[name].astype(np.float32) for name in required_data}
    enable_numpy_pickle_compatibility()
    with np.load(labels_path, allow_pickle=True) as labels:
        required_labels = ["val_region_label", "test_region_label"]
        missing_labels = [name for name in required_labels if name not in labels.files]
        if missing_labels:
            raise ValueError("Missing risk labels: {}.".format(", ".join(missing_labels)))
        arrays.update({name: labels[name].astype(np.int64) for name in required_labels})
    for split in ("val", "test"):
        if arrays["{}_x".format(split)].shape[0] != arrays["{}_region_label".format(split)].shape[0]:
            raise ValueError("{} data and labels have different window counts.".format(split))
    return arrays


def infer_region_logits(
    model: torch.nn.Module,
    values: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    rows: List[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = torch.from_numpy(values[start : start + batch_size]).to(device)
        with torch.no_grad():
            outputs = model(batch)
        ensure_finite_outputs(outputs)
        rows.append(outputs["region_logits"].detach().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float64)


def log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    flat_probs = probabilities.reshape(-1, probabilities.shape[-1])
    flat_labels = labels.reshape(-1)
    return float(-np.mean(np.log(np.clip(flat_probs[np.arange(len(flat_labels)), flat_labels], 1e-12, 1.0))))


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    flat_probs = probabilities.reshape(-1, probabilities.shape[-1])
    flat_labels = labels.reshape(-1)
    target = np.eye(len(RISK_CLASSES), dtype=np.float64)[flat_labels]
    return float(np.mean(np.sum((flat_probs - target) ** 2, axis=1)))


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    flat_probs = probabilities.reshape(-1, probabilities.shape[-1])
    flat_labels = labels.reshape(-1)
    confidence = np.max(flat_probs, axis=1)
    correct = np.argmax(flat_probs, axis=1) == flat_labels
    total = len(flat_labels)
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(selected):
            error += float(np.sum(selected) / total) * abs(
                float(np.mean(confidence[selected])) - float(np.mean(correct[selected]))
            )
    return error


def class_coverage(mask: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    flat_mask = mask.reshape(-1, mask.shape[-1])
    flat_labels = labels.reshape(-1)
    result: Dict[str, float] = {}
    for class_id, name in enumerate(RISK_CLASSES):
        selected = flat_labels == class_id
        result[name] = float(np.mean(flat_mask[selected, class_id])) if np.any(selected) else float("nan")
    return result


def evaluate_sets(mask: np.ndarray, probabilities: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    flat_mask = mask.reshape(-1, mask.shape[-1])
    flat_probs = probabilities.reshape(-1, probabilities.shape[-1])
    flat_labels = labels.reshape(-1)
    rows = np.arange(len(flat_labels))
    covered = flat_mask[rows, flat_labels]
    sizes = np.sum(flat_mask, axis=1)
    point_correct = np.argmax(flat_probs, axis=1) == flat_labels
    ambiguous = sizes > 1
    errors = ~point_correct
    accepted = ~ambiguous
    window_covered = np.all(covered.reshape(labels.shape), axis=1)
    return {
        "coverage": round(float(np.mean(covered)), 6),
        "simultaneous_window_coverage": round(float(np.mean(window_covered)), 6),
        "mean_set_size": round(float(np.mean(sizes)), 6),
        "singleton_rate": round(float(np.mean(sizes == 1)), 6),
        "ambiguous_rate": round(float(np.mean(ambiguous)), 6),
        "point_accuracy": round(float(np.mean(point_correct)), 6),
        "accepted_point_accuracy": round(float(np.mean(point_correct[accepted])), 6)
        if np.any(accepted)
        else None,
        "point_error_detection_recall": round(float(np.mean(ambiguous[errors])), 6)
        if np.any(errors)
        else None,
        "point_error_detection_precision": round(float(np.mean(errors[ambiguous])), 6)
        if np.any(ambiguous)
        else None,
        "class_coverage": {
            name: round(value, 6) if math.isfinite(value) else None
            for name, value in class_coverage(flat_mask, flat_labels).items()
        },
    }


def matched_confidence_baseline(
    probabilities: np.ndarray,
    labels: np.ndarray,
    defer_count: int,
) -> Dict[str, Any]:
    flat_probs = probabilities.reshape(-1, probabilities.shape[-1])
    flat_labels = labels.reshape(-1)
    confidence = np.max(flat_probs, axis=1)
    point_correct = np.argmax(flat_probs, axis=1) == flat_labels
    order = np.argsort(confidence, kind="stable")
    deferred = np.zeros(len(flat_labels), dtype=bool)
    deferred[order[:defer_count]] = True
    accepted = ~deferred
    errors = ~point_correct
    return {
        "defer_rate": round(float(np.mean(deferred)), 6),
        "accepted_point_accuracy": round(float(np.mean(point_correct[accepted])), 6)
        if np.any(accepted)
        else None,
        "point_error_detection_recall": round(float(np.mean(deferred[errors])), 6)
        if np.any(errors)
        else None,
        "point_error_detection_precision": round(float(np.mean(errors[deferred])), 6)
        if np.any(deferred)
        else None,
        "confidence_cutoff": round(float(confidence[order[defer_count - 1]]), 6)
        if defer_count > 0
        else 0.0,
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    before = result["probability_calibration"]["before"]
    after = result["probability_calibration"]["after"]
    lines = [
        "# 区域风险置信度与 conformal 集合评估",
        "",
        "验证集按时间切成温度拟合段和 conformal 校准段，test 只用于一次最终评估。",
        "风险参照仍是冻结 FCM 代理标签，因此这里只评价代理任务置信度，不写成真实道路风险保证。",
        "",
        "## 概率校准",
        "",
        "| 指标 | 校准前 | 校准后 |",
        "| --- | ---: | ---: |",
        "| NLL | {:.4f} | {:.4f} |".format(before["nll"], after["nll"]),
        "| Brier | {:.4f} | {:.4f} |".format(before["brier"], after["brier"]),
        "| ECE | {:.4f} | {:.4f} |".format(before["ece"], after["ece"]),
        "",
        "## 测试集预测集合",
        "",
        "| 方法 | 覆盖率 | 四区域同时覆盖 | 平均集合大小 | 歧义率 | 单例准确率 | 错误检出召回 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("marginal_aps", "class_conditional_aps", "simultaneous_window_aps"):
        row = result["test"][method]
        lines.append(
            "| {} | {:.2%} | {:.2%} | {:.3f} | {:.2%} | {} | {} |".format(
                method,
                row["coverage"],
                row["simultaneous_window_coverage"],
                row["mean_set_size"],
                row["ambiguous_rate"],
                "{:.2%}".format(row["accepted_point_accuracy"])
                if row["accepted_point_accuracy"] is not None
                else "-",
                "{:.2%}".format(row["point_error_detection_recall"])
                if row["point_error_detection_recall"] is not None
                else "-",
            )
        )
    lines.extend([
        "",
        "## 口径",
        "",
        "- 这是按时间外推的经验覆盖率。交通窗口并非独立同分布，不能直接宣称严格的有限样本保证。",
        "- 歧义集合用于触发云端复核；断网时仍由本地安全策略立即执行。",
        "- 温度缩放不改变 argmax 分类，只修正置信度尺度。",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.target_coverage < 1.0:
        raise ValueError("target_coverage must be in (0, 1).")
    if not 0.1 <= args.temperature_fraction <= 0.9:
        raise ValueError("temperature_fraction must be between 0.1 and 0.9.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    torch.set_num_threads(args.torch_threads)
    device = select_device(args.device)
    arrays = load_arrays(Path(args.data_npz), Path(args.risk_labels))
    config = load_config(args.config)
    checkpoint = torch_load_trusted(Path(args.checkpoint), device)
    adj_mx, _ = load_adjacency(config)
    model_arrays = {
        "in_channels": int(arrays["val_x"].shape[2]),
        "output_dim": int(arrays["val_x"].shape[2]),
    }
    model = build_model_from_checkpoint(config, model_arrays, adj_mx, checkpoint, device)
    val_logits = infer_region_logits(model, arrays["val_x"], args.batch_size, device)
    test_logits = infer_region_logits(model, arrays["test_x"], args.batch_size, device)

    split_at = int(round(len(val_logits) * args.temperature_fraction))
    if split_at <= 0 or split_at >= len(val_logits):
        raise ValueError("validation split does not leave data for both calibration stages.")
    temperature_logits = val_logits[:split_at].reshape(-1, len(RISK_CLASSES))
    temperature_labels = arrays["val_region_label"][:split_at].reshape(-1)
    temperature = fit_temperature(temperature_logits, temperature_labels)

    conformal_logits = val_logits[split_at:]
    conformal_labels = arrays["val_region_label"][split_at:]
    conformal_probabilities = temperature_softmax(
        conformal_logits.reshape(-1, len(RISK_CLASSES)), temperature
    ).reshape(conformal_logits.shape)
    flat_conformal_probabilities = conformal_probabilities.reshape(-1, len(RISK_CLASSES))
    flat_conformal_labels = conformal_labels.reshape(-1)
    true_scores = true_aps_scores(flat_conformal_probabilities, flat_conformal_labels)
    alpha = 1.0 - args.target_coverage
    marginal_threshold = conformal_quantile(true_scores, alpha)
    class_thresholds = class_conditional_quantiles(true_scores, flat_conformal_labels, alpha)
    window_scores = true_scores.reshape(conformal_labels.shape).max(axis=1)
    simultaneous_threshold = conformal_quantile(window_scores, alpha)

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "method": "temperature_scaled_deterministic_aps",
        "deployment_method": args.deployment_method,
        "risk_classes": list(RISK_CLASSES),
        "target_coverage": round(float(args.target_coverage), 6),
        "alpha": round(float(alpha), 6),
        "temperature": round(float(temperature), 8),
        "thresholds": {
            "marginal_aps": round(float(marginal_threshold), 8),
            "class_conditional_aps": {
                name: round(float(class_thresholds[name]), 8) for name in RISK_CLASSES
            },
            "simultaneous_window_aps": round(float(simultaneous_threshold), 8),
        },
        "calibration_split": {
            "source": "validation",
            "strategy": "contiguous_time_split",
            "total_windows": int(len(val_logits)),
            "temperature_windows": [0, split_at],
            "conformal_windows": [split_at, int(len(val_logits))],
            "regions_per_window": int(val_logits.shape[1]),
        },
        "checkpoint": args.checkpoint,
        "reference": "frozen FCM proxy region labels built from future observed traffic state",
        "guarantee_note": (
            "Temporal traffic windows are not assumed exchangeable; target coverage is evaluated "
            "empirically on the later test split and is not reported as a formal i.i.d. guarantee."
        ),
    }
    save_json(artifact, Path(args.output_model))

    raw_test_probabilities = temperature_softmax(
        test_logits.reshape(-1, len(RISK_CLASSES)), 1.0
    ).reshape(test_logits.shape)
    test_probabilities = temperature_softmax(
        test_logits.reshape(-1, len(RISK_CLASSES)), temperature
    ).reshape(test_logits.shape)
    test_candidate_scores = aps_candidate_scores(
        test_probabilities.reshape(-1, len(RISK_CLASSES))
    )
    marginal_mask = prediction_mask(test_probabilities.reshape(-1, len(RISK_CLASSES)), marginal_threshold).reshape(test_logits.shape)
    class_limits = np.asarray([class_thresholds[name] for name in RISK_CLASSES], dtype=np.float64)
    class_mask = prediction_mask(test_probabilities.reshape(-1, len(RISK_CLASSES)), class_limits).reshape(test_logits.shape)
    simultaneous_mask = prediction_mask(test_probabilities.reshape(-1, len(RISK_CLASSES)), simultaneous_threshold).reshape(test_logits.shape)
    del test_candidate_scores

    set_results = {
        "marginal_aps": evaluate_sets(marginal_mask, test_probabilities, arrays["test_region_label"]),
        "class_conditional_aps": evaluate_sets(class_mask, test_probabilities, arrays["test_region_label"]),
        "simultaneous_window_aps": evaluate_sets(
            simultaneous_mask, test_probabilities, arrays["test_region_label"]
        ),
    }
    deployment_result = set_results[args.deployment_method]
    defer_count = int(round(deployment_result["ambiguous_rate"] * arrays["test_region_label"].size))
    result = {
        "task": "region_risk_conformal_calibration",
        "device": str(device),
        "model_artifact": args.output_model,
        "reference_scope": "FCM proxy risk labels, not human incident ground truth",
        "validation": {
            "total_windows": int(len(val_logits)),
            "temperature_windows": int(split_at),
            "conformal_windows": int(len(val_logits) - split_at),
        },
        "test_windows": int(len(test_logits)),
        "test_region_predictions": int(arrays["test_region_label"].size),
        "temperature": round(float(temperature), 8),
        "probability_calibration": {
            "before": {
                "nll": round(log_loss(raw_test_probabilities, arrays["test_region_label"]), 6),
                "brier": round(brier_score(raw_test_probabilities, arrays["test_region_label"]), 6),
                "ece": round(expected_calibration_error(raw_test_probabilities, arrays["test_region_label"]), 6),
            },
            "after": {
                "nll": round(log_loss(test_probabilities, arrays["test_region_label"]), 6),
                "brier": round(brier_score(test_probabilities, arrays["test_region_label"]), 6),
                "ece": round(expected_calibration_error(test_probabilities, arrays["test_region_label"]), 6),
            },
        },
        "test": set_results,
        "matched_raw_confidence_baseline": matched_confidence_baseline(
            raw_test_probabilities, arrays["test_region_label"], defer_count
        ),
        "deployment_method": args.deployment_method,
        "interpretation": (
            "Prediction-set ambiguity is an auditable defer signal. Empirical coverage is measured "
            "against the same frozen proxy labels as the risk head, not external road-event truth."
        ),
    }
    save_json(result, Path(args.output_json))
    write_markdown(result, Path(args.output_md))
    print(json.dumps({
        "temperature": result["temperature"],
        "probability_calibration": result["probability_calibration"],
        "deployment_method": args.deployment_method,
        "deployment_test": result["test"][args.deployment_method],
        "matched_raw_confidence_baseline": result["matched_raw_confidence_baseline"],
    }, ensure_ascii=False, indent=2))
    print("model:", args.output_model)
    print("evaluation:", args.output_json)


if __name__ == "__main__":
    main()
