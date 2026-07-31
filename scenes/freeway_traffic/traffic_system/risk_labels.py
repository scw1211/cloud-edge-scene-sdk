"""用途：提供风险等级、反归一化、分布统计和风险标签公共计算函数。"""

import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


RISK_CLASSES = ["low", "medium", "high", "severe"]
RISK_TO_ID = {name: idx for idx, name in enumerate(RISK_CLASSES)}
FEATURE_FLOW = 0
FEATURE_OCCUPANCY = 1
FEATURE_SPEED = 2


def enable_numpy_pickle_compatibility() -> None:
    """Read NumPy 2 object arrays on the NumPy 1.x Jetson runtime."""
    if int(np.__version__.split(".", 1)[0]) < 2:
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator + 1e-6)


def percentile_rank(values: np.ndarray, item_value: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values <= item_value))


def node_level_from_traffic_state(
    future_node: np.ndarray,
    history_node: np.ndarray,
    region_future: np.ndarray,
) -> str:
    """Build a weak-supervision congestion risk label from flow/occupancy/speed."""
    flow_future = future_node[FEATURE_FLOW, :].astype(np.float64)
    occ_future = future_node[FEATURE_OCCUPANCY, :].astype(np.float64)
    speed_future = future_node[FEATURE_SPEED, :].astype(np.float64)

    flow_hist = history_node[FEATURE_FLOW, :].astype(np.float64)
    occ_hist = history_node[FEATURE_OCCUPANCY, :].astype(np.float64)
    speed_hist = history_node[FEATURE_SPEED, :].astype(np.float64)

    region_flow_mean = np.mean(region_future[:, FEATURE_FLOW, :].astype(np.float64), axis=1)
    region_occ_mean = np.mean(region_future[:, FEATURE_OCCUPANCY, :].astype(np.float64), axis=1)
    region_speed_mean = np.mean(region_future[:, FEATURE_SPEED, :].astype(np.float64), axis=1)

    flow_mean = float(np.mean(flow_future))
    occ_mean = float(np.mean(occ_future))
    speed_mean = float(np.mean(speed_future))
    speed_min = float(np.min(speed_future))

    flow_pressure = percentile_rank(region_flow_mean, flow_mean)
    occ_pressure = percentile_rank(region_occ_mean, occ_mean)
    speed_pressure = 1.0 - percentile_rank(region_speed_mean, speed_mean)

    flow_growth = safe_ratio(flow_mean - float(np.mean(flow_hist)), float(np.mean(flow_hist)))
    occ_growth = safe_ratio(occ_mean - float(np.mean(occ_hist)), float(np.mean(occ_hist)))
    speed_drop = safe_ratio(float(np.mean(speed_hist)) - speed_mean, float(np.mean(speed_hist)))
    peak_speed_drop = safe_ratio(float(speed_hist[-1]) - speed_min, float(speed_hist[-1]))

    dynamic_pressure = max(
        0.0,
        min(1.0, flow_growth / 0.30),
        min(1.0, occ_growth / 0.35),
        min(1.0, speed_drop / 0.35),
        min(1.0, peak_speed_drop / 0.45),
    )
    risk_score = (
        0.20 * flow_pressure
        + 0.30 * occ_pressure
        + 0.35 * speed_pressure
        + 0.15 * dynamic_pressure
    )

    low_speed = speed_pressure >= 0.80 or speed_drop >= 0.12 or peak_speed_drop >= 0.20
    high_occupancy = occ_pressure >= 0.80 or occ_growth >= 0.12
    high_flow_pressure = flow_pressure >= 0.80 or flow_growth >= 0.12

    if risk_score >= 0.82 and (low_speed or high_occupancy):
        return "severe"
    if risk_score >= 0.68 and (low_speed or high_occupancy or high_flow_pressure):
        return "high"
    if risk_score >= 0.52 or low_speed or high_occupancy or high_flow_pressure:
        return "medium"
    return "low"


def denormalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return values * np.where(std == 0, 1.0, std) + mean


def build_risk_labels(
    x_norm: np.ndarray,
    future_raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    partitions: Sequence[Sequence[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create node and region risk labels for one split.

    Args:
        x_norm: normalized history, (B, N, F, T)
        future_raw: raw target values, (B, N, F, T)
        mean/std: normalization statistics for x_norm, broadcastable to x_norm
        partitions: graph partitions, each a sequence of node ids

    Returns:
        node_labels: (B, N)
        region_labels: (B, R), max severity inside each region
    """
    history_raw = x_norm * np.where(std == 0, 1.0, std) + mean
    batch_size, num_nodes = future_raw.shape[:2]
    node_labels = np.zeros((batch_size, num_nodes), dtype=np.int64)
    region_labels = np.zeros((batch_size, len(partitions)), dtype=np.int64)

    for sample_idx in range(batch_size):
        for region_idx, node_ids_raw in enumerate(partitions):
            node_ids = [int(node_id) for node_id in node_ids_raw]
            region_future = future_raw[sample_idx, node_ids, :, :]
            region_max = 0
            for node_id in node_ids:
                level = node_level_from_traffic_state(
                    future_node=future_raw[sample_idx, node_id, :, :],
                    history_node=history_raw[sample_idx, node_id, :, :],
                    region_future=region_future,
                )
                label_id = RISK_TO_ID[level]
                node_labels[sample_idx, node_id] = label_id
                region_max = max(region_max, label_id)
            region_labels[sample_idx, region_idx] = region_max
    return node_labels, region_labels


def label_distribution(labels: np.ndarray) -> Dict[str, int]:
    counts = np.bincount(labels.reshape(-1), minlength=len(RISK_CLASSES))
    return {name: int(counts[idx]) for idx, name in enumerate(RISK_CLASSES)}


def class_weights(labels: np.ndarray, power: float = 0.75) -> np.ndarray:
    flat = labels.reshape(-1)
    counts = np.bincount(flat, minlength=len(RISK_CLASSES)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = flat.shape[0] / (len(RISK_CLASSES) * counts)
    weights = np.power(weights, max(0.0, float(power)))
    return weights / weights.mean()


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    true_flat = y_true.reshape(-1).astype(np.int64)
    pred_flat = y_pred.reshape(-1).astype(np.int64)
    n = len(RISK_CLASSES)
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for true_value, pred_value in zip(true_flat.tolist(), pred_flat.tolist()):
        matrix[int(true_value)][int(pred_value)] += 1
    total = int(sum(sum(row) for row in matrix))
    correct = int(sum(matrix[i][i] for i in range(n)))
    per_class: Dict[str, Dict[str, Any]] = {}
    macro_values: List[float] = []
    weighted_f1 = 0.0
    for idx, name in enumerate(RISK_CLASSES):
        tp = matrix[idx][idx]
        support = sum(matrix[idx])
        predicted = sum(matrix[row][idx] for row in range(n))
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        if support and f1 is not None:
            macro_values.append(f1)
            weighted_f1 += f1 * support
        per_class[name] = {
            "support": int(support),
            "precision": round(precision, 6) if precision is not None else None,
            "recall": round(recall, 6) if recall is not None else None,
            "f1": round(f1, 6) if f1 is not None else None,
        }
    high_severe_mask = true_flat >= RISK_TO_ID["high"]
    if np.any(high_severe_mask):
        high_severe_recall = float(np.mean(pred_flat[high_severe_mask] >= RISK_TO_ID["high"]))
    else:
        high_severe_recall = 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_f1": round(float(np.mean(macro_values)), 6) if macro_values else 0.0,
        "weighted_f1": round(weighted_f1 / total, 6) if total else 0.0,
        "high_severe_recall": round(high_severe_recall, 6),
        "class_names": RISK_CLASSES,
        "matrix": matrix,
        "per_class": per_class,
    }
