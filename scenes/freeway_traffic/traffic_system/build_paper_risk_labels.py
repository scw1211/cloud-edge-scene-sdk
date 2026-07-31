"""用途：根据流量、占有率、速度聚类和持续拥堵条件生成风险标签。"""

import argparse
import configparser
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans

from traffic_system.graph_partition import partition_graph
from lib.utils import get_adjacency_matrix
from traffic_system.risk_labels import RISK_CLASSES, denormalize, label_distribution


FEATURE_FLOW = 0
FEATURE_OCCUPANCY = 1
FEATURE_SPEED = 2

NODE_FEATURE_NAMES = [
    "speed_mean_ratio",
    "speed_min_ratio",
    "speed_std_ratio",
    "speed_drop_ratio",
    "congestion_duration_ratio",
    "sustained_congestion_ratio",
    "occupancy_mean",
    "occupancy_max",
    "occupancy_delta",
    "flow_mean",
    "flow_log_change",
]

REGION_FEATURE_NAMES = [
    "region_speed_mean_ratio",
    "region_speed_min_ratio",
    "region_speed_std_ratio",
    "region_speed_drop_ratio",
    "region_congestion_duration_ratio",
    "region_sustained_congestion_ratio",
    "region_occupancy_mean",
    "region_occupancy_p90",
    "region_occupancy_delta",
    "region_flow_mean",
    "region_flow_log_change",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-based weak risk labels for PEMS traffic data.")
    parser.add_argument("--config", default="configurations/PEMS08_astgcn.conf")
    parser.add_argument("--data_npz", default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz")
    parser.add_argument("--output_npz", default="datasets/risk_labels_pems08_metis4.npz")
    parser.add_argument("--report_json", default="results/perception/risk_label_paper_report.json")
    parser.add_argument("--num_partitions", type=int, default=4)
    parser.add_argument(
        "--partition_method",
        default="metis",
        choices=["metis", "spectral", "graph_bfs", "graph", "contiguous"],
    )
    parser.add_argument("--overlap_hops", type=int, default=0)
    parser.add_argument("--cluster_method", default="fcm", choices=["fcm", "kmeans"])
    parser.add_argument("--num_clusters", type=int, default=4)
    parser.add_argument("--free_flow_percentile", type=float, default=85.0)
    parser.add_argument("--congestion_speed_ratio", type=float, default=0.80)
    parser.add_argument("--min_congestion_steps", type=int, default=2)
    parser.add_argument("--fcm_m", type=float, default=2.0)
    parser.add_argument("--fcm_max_iter", type=int, default=80)
    parser.add_argument("--fcm_tol", type=float, default=1e-4)
    parser.add_argument("--cluster_fit_limit", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    loaded = config.read(path)
    if not loaded:
        raise FileNotFoundError("Config file not found: {}".format(path))
    return config


def load_arrays(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError("Data file not found: {}".format(path))
    data = np.load(path)
    required = [
        "train_x",
        "train_target",
        "val_x",
        "val_target",
        "test_x",
        "test_target",
        "mean",
        "std",
    ]
    missing = [name for name in required if name not in data.files]
    if missing:
        raise ValueError("Missing keys in {}: {}".format(path, ", ".join(missing)))
    return {name: data[name] for name in data.files}


def load_adjacency(config: configparser.ConfigParser) -> Tuple[np.ndarray, str]:
    data_config = config["Data"]
    id_filename = data_config["id_filename"] if config.has_option("Data", "id_filename") else None
    adj_mx, _ = get_adjacency_matrix(
        data_config["adj_filename"],
        int(data_config["num_of_vertices"]),
        id_filename,
    )
    return adj_mx, data_config["adj_filename"]


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / (denominator + 1e-6)


def max_consecutive_ratio(mask: np.ndarray) -> np.ndarray:
    run = np.zeros(mask.shape[:2], dtype=np.float32)
    best = np.zeros(mask.shape[:2], dtype=np.float32)
    for step in range(mask.shape[-1]):
        run = (run + 1.0) * mask[:, :, step].astype(np.float32)
        best = np.maximum(best, run)
    return best / float(mask.shape[-1])


def estimate_free_flow_speed(
    train_x: np.ndarray,
    train_target: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    percentile: float,
) -> np.ndarray:
    history_raw = denormalize(train_x, mean, std)
    speed_values = np.concatenate(
        [
            history_raw[:, :, FEATURE_SPEED, :].transpose(0, 2, 1).reshape(-1, train_x.shape[1]),
            train_target[:, :, FEATURE_SPEED, :].transpose(0, 2, 1).reshape(-1, train_target.shape[1]),
        ],
        axis=0,
    )
    free_flow = np.percentile(speed_values, percentile, axis=0).astype(np.float32)
    return np.maximum(free_flow, 1.0)


def build_node_features(
    x_norm: np.ndarray,
    target_raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    free_flow_speed: np.ndarray,
    congestion_speed_ratio: float,
) -> np.ndarray:
    history_raw = denormalize(x_norm, mean, std)
    ff = free_flow_speed.reshape(1, -1)
    speed_future = target_raw[:, :, FEATURE_SPEED, :].astype(np.float32)
    speed_history = history_raw[:, :, FEATURE_SPEED, :].astype(np.float32)
    occ_future = target_raw[:, :, FEATURE_OCCUPANCY, :].astype(np.float32)
    occ_history = history_raw[:, :, FEATURE_OCCUPANCY, :].astype(np.float32)
    flow_future = target_raw[:, :, FEATURE_FLOW, :].astype(np.float32)
    flow_history = history_raw[:, :, FEATURE_FLOW, :].astype(np.float32)

    speed_mean = speed_future.mean(axis=-1)
    speed_min = speed_future.min(axis=-1)
    speed_std = speed_future.std(axis=-1)
    history_speed_mean = speed_history.mean(axis=-1)
    below_congestion = speed_future < (congestion_speed_ratio * free_flow_speed.reshape(1, -1, 1))

    occ_mean = occ_future.mean(axis=-1)
    occ_history_mean = occ_history.mean(axis=-1)
    flow_mean = flow_future.mean(axis=-1)
    flow_history_mean = flow_history.mean(axis=-1)

    features = np.stack(
        [
            safe_ratio(speed_mean, ff),
            safe_ratio(speed_min, ff),
            safe_ratio(speed_std, ff),
            np.maximum(0.0, safe_ratio(history_speed_mean - speed_mean, history_speed_mean)),
            below_congestion.mean(axis=-1),
            max_consecutive_ratio(below_congestion),
            occ_mean,
            occ_future.max(axis=-1),
            occ_mean - occ_history_mean,
            flow_mean,
            np.log1p(np.maximum(flow_mean, 0.0)) - np.log1p(np.maximum(flow_history_mean, 0.0)),
        ],
        axis=-1,
    )
    return np.nan_to_num(features.reshape(-1, len(NODE_FEATURE_NAMES))).astype(np.float32)


def build_region_features_from_node_features(
    node_features: np.ndarray,
    num_samples: int,
    num_nodes: int,
    partitions: Sequence[Sequence[int]],
) -> np.ndarray:
    node_view = node_features.reshape(num_samples, num_nodes, len(NODE_FEATURE_NAMES))
    rows = []
    idx = {name: i for i, name in enumerate(NODE_FEATURE_NAMES)}
    for sample_idx in range(num_samples):
        for node_ids_raw in partitions:
            node_ids = [int(node_id) for node_id in node_ids_raw]
            part = node_view[sample_idx, node_ids, :]
            rows.append(
                [
                    float(np.mean(part[:, idx["speed_mean_ratio"]])),
                    float(np.min(part[:, idx["speed_min_ratio"]])),
                    float(np.std(part[:, idx["speed_mean_ratio"]])),
                    float(np.mean(part[:, idx["speed_drop_ratio"]])),
                    float(np.mean(part[:, idx["congestion_duration_ratio"]])),
                    float(np.mean(part[:, idx["sustained_congestion_ratio"]])),
                    float(np.mean(part[:, idx["occupancy_mean"]])),
                    float(np.percentile(part[:, idx["occupancy_max"]], 90)),
                    float(np.mean(part[:, idx["occupancy_delta"]])),
                    float(np.mean(part[:, idx["flow_mean"]])),
                    float(np.mean(part[:, idx["flow_log_change"]])),
                ]
            )
    return np.asarray(rows, dtype=np.float32)


def standardize_fit(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((features - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / np.where(std < 1e-6, 1.0, std)).astype(np.float32)


def sample_for_cluster(features: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or features.shape[0] <= limit:
        return features
    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(features.shape[0], size=limit, replace=False)
    return features[np.sort(sample_ids)]


def squared_distances(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)


def fit_kmeans(features: np.ndarray, num_clusters: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    model = KMeans(n_clusters=num_clusters, n_init=20, random_state=seed)
    labels = model.fit_predict(features)
    return model.cluster_centers_.astype(np.float32), labels.astype(np.int64)


def fit_fcm(
    features: np.ndarray,
    num_clusters: int,
    seed: int,
    m: float,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    init_centers, _ = fit_kmeans(features, num_clusters, seed)
    centers = init_centers.astype(np.float32)
    exponent = 1.0 / max(m - 1.0, 1e-6)
    objective_history = []
    for iteration in range(1, max_iter + 1):
        dist2 = np.maximum(squared_distances(features, centers), 1e-8)
        inv = dist2 ** (-exponent)
        membership = inv / inv.sum(axis=1, keepdims=True)
        membership_m = membership ** m
        new_centers = (membership_m.T @ features) / np.maximum(membership_m.sum(axis=0)[:, None], 1e-8)
        objective = float(np.sum(membership_m * dist2))
        objective_history.append(objective)
        shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
        centers = new_centers.astype(np.float32)
        if shift < tol:
            break
    labels = nearest_cluster(features, centers)
    return centers, labels, {
        "iterations": iteration,
        "final_shift": round(shift, 8),
        "objective": round(objective_history[-1], 6),
    }


def nearest_cluster(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.argmin(squared_distances(features, centers), axis=1).astype(np.int64)


def rank_values(values: np.ndarray, lower_is_risk: bool) -> np.ndarray:
    order = np.argsort(-values if lower_is_risk else values)
    ranks = np.zeros_like(values, dtype=np.float32)
    ranks[order] = np.arange(values.shape[0], dtype=np.float32)
    return ranks


def map_clusters_to_risk(centers_raw: np.ndarray, feature_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    low_value_risk = [
        name for name in ["speed_mean_ratio", "speed_min_ratio", "region_speed_mean_ratio", "region_speed_min_ratio"]
        if name in name_to_idx
    ]
    high_value_risk = [
        name
        for name in [
            "speed_drop_ratio",
            "congestion_duration_ratio",
            "sustained_congestion_ratio",
            "occupancy_mean",
            "occupancy_max",
            "occupancy_delta",
            "flow_log_change",
            "region_speed_drop_ratio",
            "region_congestion_duration_ratio",
            "region_sustained_congestion_ratio",
            "region_occupancy_mean",
            "region_occupancy_p90",
            "region_occupancy_delta",
            "region_flow_log_change",
        ]
        if name in name_to_idx
    ]
    severity = np.zeros(centers_raw.shape[0], dtype=np.float32)
    for name in low_value_risk:
        severity += rank_values(centers_raw[:, name_to_idx[name]], lower_is_risk=True)
    for name in high_value_risk:
        severity += rank_values(centers_raw[:, name_to_idx[name]], lower_is_risk=False)
    ordered_clusters = np.argsort(severity)
    cluster_to_risk = np.zeros(centers_raw.shape[0], dtype=np.int64)
    for risk_id, cluster_id in enumerate(ordered_clusters):
        cluster_to_risk[int(cluster_id)] = int(risk_id)
    return cluster_to_risk, severity


def apply_node_calibration(
    labels: np.ndarray,
    features: np.ndarray,
    centers_raw: np.ndarray,
    cluster_to_risk: np.ndarray,
    congestion_speed_ratio: float,
    min_congestion_steps: int,
    horizon: int,
    train_occ_p75: float,
) -> np.ndarray:
    idx = {name: i for i, name in enumerate(NODE_FEATURE_NAMES)}
    calibrated = labels.copy()
    duration_threshold = min_congestion_steps / float(horizon)
    high_gate = (
        (features[:, idx["speed_min_ratio"]] <= congestion_speed_ratio)
        | (features[:, idx["congestion_duration_ratio"]] >= duration_threshold)
        | (features[:, idx["occupancy_mean"]] >= train_occ_p75)
    )
    calibrated[(calibrated >= 2) & (~high_gate)] = 1

    severe_cluster = int(np.where(cluster_to_risk == 3)[0][0])
    severe_speed = float(centers_raw[severe_cluster, idx["speed_min_ratio"]])
    severe_duration = float(centers_raw[severe_cluster, idx["congestion_duration_ratio"]])
    severe_occupancy = float(centers_raw[severe_cluster, idx["occupancy_mean"]])
    severe_gate = (
        (features[:, idx["speed_min_ratio"]] <= severe_speed)
        | (features[:, idx["congestion_duration_ratio"]] >= max(duration_threshold, 0.75 * severe_duration))
        | (features[:, idx["occupancy_mean"]] >= severe_occupancy)
    )
    calibrated[(calibrated == 3) & (~severe_gate)] = 2
    return calibrated.astype(np.int64)


def apply_region_calibration(
    labels: np.ndarray,
    features: np.ndarray,
    centers_raw: np.ndarray,
    cluster_to_risk: np.ndarray,
    congestion_speed_ratio: float,
    min_congestion_steps: int,
    horizon: int,
    train_occ_p75: float,
) -> np.ndarray:
    idx = {name: i for i, name in enumerate(REGION_FEATURE_NAMES)}
    calibrated = labels.copy()
    duration_threshold = min_congestion_steps / float(horizon)
    high_gate = (
        (features[:, idx["region_speed_min_ratio"]] <= congestion_speed_ratio)
        | (features[:, idx["region_congestion_duration_ratio"]] >= duration_threshold)
        | (features[:, idx["region_occupancy_mean"]] >= train_occ_p75)
    )
    calibrated[(calibrated >= 2) & (~high_gate)] = 1

    severe_cluster = int(np.where(cluster_to_risk == 3)[0][0])
    severe_speed = float(centers_raw[severe_cluster, idx["region_speed_min_ratio"]])
    severe_duration = float(centers_raw[severe_cluster, idx["region_congestion_duration_ratio"]])
    severe_occupancy = float(centers_raw[severe_cluster, idx["region_occupancy_mean"]])
    severe_gate = (
        (features[:, idx["region_speed_min_ratio"]] <= severe_speed)
        | (features[:, idx["region_congestion_duration_ratio"]] >= max(duration_threshold, 0.75 * severe_duration))
        | (features[:, idx["region_occupancy_mean"]] >= severe_occupancy)
    )
    calibrated[(calibrated == 3) & (~severe_gate)] = 2
    return calibrated.astype(np.int64)


def fit_cluster_model(
    train_features: np.ndarray,
    method: str,
    num_clusters: int,
    seed: int,
    fit_limit: int,
    fcm_m: float,
    fcm_max_iter: int,
    fcm_tol: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    train_std, scaler_mean, scaler_std = standardize_fit(train_features)
    fit_features = sample_for_cluster(train_std, fit_limit, seed)
    if method == "kmeans":
        centers_std, _ = fit_kmeans(fit_features, num_clusters, seed)
        extra = {"iterations": None}
    else:
        centers_std, _, extra = fit_fcm(fit_features, num_clusters, seed, fcm_m, fcm_max_iter, fcm_tol)
    return centers_std, {
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "fit_examples": int(fit_features.shape[0]),
        "extra": extra,
    }


def assign_labels(
    features: np.ndarray,
    centers_std: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    cluster_to_risk: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    features_std = standardize_apply(features, scaler_mean, scaler_std)
    clusters = nearest_cluster(features_std, centers_std)
    labels = cluster_to_risk[clusters]
    return labels.astype(np.int64), clusters


def distribution_for_splits(labels: Dict[str, np.ndarray]) -> Dict[str, Dict[str, int]]:
    return {split: label_distribution(values) for split, values in labels.items()}


def centers_to_records(
    centers_raw: np.ndarray,
    feature_names: Sequence[str],
    cluster_to_risk: np.ndarray,
    severity: np.ndarray,
) -> List[Dict[str, Any]]:
    records = []
    for cluster_id in range(centers_raw.shape[0]):
        records.append(
            {
                "cluster": int(cluster_id),
                "risk_level": RISK_CLASSES[int(cluster_to_risk[cluster_id])],
                "severity_rank_score": round(float(severity[cluster_id]), 6),
                "center": {
                    name: round(float(centers_raw[cluster_id, idx]), 6)
                    for idx, name in enumerate(feature_names)
                },
            }
        )
    risk_order = {name: idx for idx, name in enumerate(RISK_CLASSES)}
    return sorted(records, key=lambda item: risk_order[item["risk_level"]])


def main() -> None:
    args = parse_args()
    if args.num_clusters != len(RISK_CLASSES):
        raise ValueError("num_clusters must be {} for low/medium/high/severe.".format(len(RISK_CLASSES)))

    config = load_config(args.config)
    arrays = load_arrays(Path(args.data_npz))
    adj_mx, adj_filename = load_adjacency(config)
    num_nodes = int(config["Data"]["num_of_vertices"])
    horizon = int(config["Data"]["num_for_predict"])
    partitions = partition_graph(
        adj_mx,
        num_nodes,
        args.num_partitions,
        args.partition_method,
        args.overlap_hops,
    )

    free_flow_speed = estimate_free_flow_speed(
        arrays["train_x"],
        arrays["train_target"],
        arrays["mean"],
        arrays["std"],
        args.free_flow_percentile,
    )

    print("Build node features...")
    train_node_features = build_node_features(
        arrays["train_x"],
        arrays["train_target"],
        arrays["mean"],
        arrays["std"],
        free_flow_speed,
        args.congestion_speed_ratio,
    )
    val_node_features = build_node_features(
        arrays["val_x"],
        arrays["val_target"],
        arrays["mean"],
        arrays["std"],
        free_flow_speed,
        args.congestion_speed_ratio,
    )
    test_node_features = build_node_features(
        arrays["test_x"],
        arrays["test_target"],
        arrays["mean"],
        arrays["std"],
        free_flow_speed,
        args.congestion_speed_ratio,
    )

    print("Fit node {} clustering...".format(args.cluster_method.upper()))
    node_centers_std, node_cluster_info = fit_cluster_model(
        train_node_features,
        args.cluster_method,
        args.num_clusters,
        args.seed,
        args.cluster_fit_limit,
        args.fcm_m,
        args.fcm_max_iter,
        args.fcm_tol,
    )
    node_centers_raw = node_centers_std * node_cluster_info["scaler_std"] + node_cluster_info["scaler_mean"]
    node_cluster_to_risk, node_severity = map_clusters_to_risk(node_centers_raw, NODE_FEATURE_NAMES)
    train_occ_p75 = float(np.percentile(train_node_features[:, NODE_FEATURE_NAMES.index("occupancy_mean")], 75))

    train_node_label, train_node_cluster = assign_labels(
        train_node_features,
        node_centers_std,
        node_cluster_info["scaler_mean"],
        node_cluster_info["scaler_std"],
        node_cluster_to_risk,
    )
    val_node_label, val_node_cluster = assign_labels(
        val_node_features,
        node_centers_std,
        node_cluster_info["scaler_mean"],
        node_cluster_info["scaler_std"],
        node_cluster_to_risk,
    )
    test_node_label, test_node_cluster = assign_labels(
        test_node_features,
        node_centers_std,
        node_cluster_info["scaler_mean"],
        node_cluster_info["scaler_std"],
        node_cluster_to_risk,
    )
    train_node_label = apply_node_calibration(
        train_node_label,
        train_node_features,
        node_centers_raw,
        node_cluster_to_risk,
        args.congestion_speed_ratio,
        args.min_congestion_steps,
        horizon,
        train_occ_p75,
    )
    val_node_label = apply_node_calibration(
        val_node_label,
        val_node_features,
        node_centers_raw,
        node_cluster_to_risk,
        args.congestion_speed_ratio,
        args.min_congestion_steps,
        horizon,
        train_occ_p75,
    )
    test_node_label = apply_node_calibration(
        test_node_label,
        test_node_features,
        node_centers_raw,
        node_cluster_to_risk,
        args.congestion_speed_ratio,
        args.min_congestion_steps,
        horizon,
        train_occ_p75,
    )

    print("Build region features...")
    train_region_features = build_region_features_from_node_features(
        train_node_features,
        arrays["train_x"].shape[0],
        num_nodes,
        partitions,
    )
    val_region_features = build_region_features_from_node_features(
        val_node_features,
        arrays["val_x"].shape[0],
        num_nodes,
        partitions,
    )
    test_region_features = build_region_features_from_node_features(
        test_node_features,
        arrays["test_x"].shape[0],
        num_nodes,
        partitions,
    )
    del train_node_features, val_node_features, test_node_features

    print("Fit region {} clustering...".format(args.cluster_method.upper()))
    region_centers_std, region_cluster_info = fit_cluster_model(
        train_region_features,
        args.cluster_method,
        args.num_clusters,
        args.seed + 7,
        args.cluster_fit_limit,
        args.fcm_m,
        args.fcm_max_iter,
        args.fcm_tol,
    )
    region_centers_raw = region_centers_std * region_cluster_info["scaler_std"] + region_cluster_info["scaler_mean"]
    region_cluster_to_risk, region_severity = map_clusters_to_risk(region_centers_raw, REGION_FEATURE_NAMES)
    train_region_occ_p75 = float(np.percentile(train_region_features[:, REGION_FEATURE_NAMES.index("region_occupancy_mean")], 75))

    train_region_label, train_region_cluster = assign_labels(
        train_region_features,
        region_centers_std,
        region_cluster_info["scaler_mean"],
        region_cluster_info["scaler_std"],
        region_cluster_to_risk,
    )
    val_region_label, val_region_cluster = assign_labels(
        val_region_features,
        region_centers_std,
        region_cluster_info["scaler_mean"],
        region_cluster_info["scaler_std"],
        region_cluster_to_risk,
    )
    test_region_label, test_region_cluster = assign_labels(
        test_region_features,
        region_centers_std,
        region_cluster_info["scaler_mean"],
        region_cluster_info["scaler_std"],
        region_cluster_to_risk,
    )
    train_region_label = apply_region_calibration(
        train_region_label,
        train_region_features,
        region_centers_raw,
        region_cluster_to_risk,
        args.congestion_speed_ratio,
        args.min_congestion_steps,
        horizon,
        train_region_occ_p75,
    )
    val_region_label = apply_region_calibration(
        val_region_label,
        val_region_features,
        region_centers_raw,
        region_cluster_to_risk,
        args.congestion_speed_ratio,
        args.min_congestion_steps,
        horizon,
        train_region_occ_p75,
    )
    test_region_label = apply_region_calibration(
        test_region_label,
        test_region_features,
        region_centers_raw,
        region_cluster_to_risk,
        args.congestion_speed_ratio,
        args.min_congestion_steps,
        horizon,
        train_region_occ_p75,
    )

    train_node_label = train_node_label.reshape(arrays["train_x"].shape[0], num_nodes)
    val_node_label = val_node_label.reshape(arrays["val_x"].shape[0], num_nodes)
    test_node_label = test_node_label.reshape(arrays["test_x"].shape[0], num_nodes)
    train_region_label = train_region_label.reshape(arrays["train_x"].shape[0], len(partitions))
    val_region_label = val_region_label.reshape(arrays["val_x"].shape[0], len(partitions))
    test_region_label = test_region_label.reshape(arrays["test_x"].shape[0], len(partitions))
    train_node_cluster = train_node_cluster.reshape(arrays["train_x"].shape[0], num_nodes)
    val_node_cluster = val_node_cluster.reshape(arrays["val_x"].shape[0], num_nodes)
    test_node_cluster = test_node_cluster.reshape(arrays["test_x"].shape[0], num_nodes)
    train_region_cluster = train_region_cluster.reshape(arrays["train_x"].shape[0], len(partitions))
    val_region_cluster = val_region_cluster.reshape(arrays["val_x"].shape[0], len(partitions))
    test_region_cluster = test_region_cluster.reshape(arrays["test_x"].shape[0], len(partitions))

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        train_node_label=train_node_label.astype(np.int64),
        val_node_label=val_node_label.astype(np.int64),
        test_node_label=test_node_label.astype(np.int64),
        train_region_label=train_region_label.astype(np.int64),
        val_region_label=val_region_label.astype(np.int64),
        test_region_label=test_region_label.astype(np.int64),
        train_node_cluster=train_node_cluster.astype(np.int64),
        val_node_cluster=val_node_cluster.astype(np.int64),
        test_node_cluster=test_node_cluster.astype(np.int64),
        train_region_cluster=train_region_cluster.astype(np.int64),
        val_region_cluster=val_region_cluster.astype(np.int64),
        test_region_cluster=test_region_cluster.astype(np.int64),
        free_flow_speed=free_flow_speed.astype(np.float32),
        partitions=np.asarray([np.asarray(part, dtype=np.int64) for part in partitions], dtype=object),
        node_feature_names=np.asarray(NODE_FEATURE_NAMES),
        region_feature_names=np.asarray(REGION_FEATURE_NAMES),
        node_cluster_centers=node_centers_raw.astype(np.float32),
        region_cluster_centers=region_centers_raw.astype(np.float32),
        node_cluster_to_risk=node_cluster_to_risk.astype(np.int64),
        region_cluster_to_risk=region_cluster_to_risk.astype(np.int64),
        node_scaler_mean=node_cluster_info["scaler_mean"].astype(np.float32),
        node_scaler_std=node_cluster_info["scaler_std"].astype(np.float32),
        region_scaler_mean=region_cluster_info["scaler_mean"].astype(np.float32),
        region_scaler_std=region_cluster_info["scaler_std"].astype(np.float32),
        risk_classes=np.asarray(RISK_CLASSES),
    )

    report = {
        "task": "paper_based_traffic_risk_label_generation",
        "dataset": config["Data"].get("dataset_name", "unknown"),
        "data_npz": args.data_npz,
        "adjacency_file": adj_filename,
        "output_npz": str(output_npz),
        "method": {
            "cluster_method": args.cluster_method,
            "num_clusters": args.num_clusters,
            "cluster_fit_limit": args.cluster_fit_limit,
            "node_fit_examples": node_cluster_info["fit_examples"],
            "region_fit_examples": region_cluster_info["fit_examples"],
            "free_flow_percentile": args.free_flow_percentile,
            "congestion_speed_ratio": args.congestion_speed_ratio,
            "min_congestion_steps": args.min_congestion_steps,
            "high_risk_calibration": (
                "high/severe labels must satisfy speed below free-flow threshold, "
                "sustained congestion duration, or high occupancy percentile."
            ),
            "region_labeling": "region-level FCM/K-means over aggregated region traffic features; not max node label.",
            "node_cluster_extra": node_cluster_info["extra"],
            "region_cluster_extra": region_cluster_info["extra"],
        },
        "feature_names": {
            "node": NODE_FEATURE_NAMES,
            "region": REGION_FEATURE_NAMES,
        },
        "cluster_centers": {
            "node": centers_to_records(node_centers_raw, NODE_FEATURE_NAMES, node_cluster_to_risk, node_severity),
            "region": centers_to_records(region_centers_raw, REGION_FEATURE_NAMES, region_cluster_to_risk, region_severity),
        },
        "label_distribution": {
            "node": distribution_for_splits(
                {
                    "train": train_node_label,
                    "val": val_node_label,
                    "test": test_node_label,
                }
            ),
            "region": distribution_for_splits(
                {
                    "train": train_region_label,
                    "val": val_region_label,
                    "test": test_region_label,
                }
            ),
        },
        "partitions": [[int(node_id) for node_id in part] for part in partitions],
        "literature_basis": [
            "Caltrans PeMS provides speed/flow/occupancy detector data.",
            "Congestion state classification should combine speed with density/occupancy rather than flow-only.",
            "Unsupervised clustering such as FCM/K-means is commonly used to generate pseudo traffic-state labels when manual labels are unavailable.",
            "Free-flow-speed threshold and sustained duration checks reduce transient perturbation labels.",
        ],
    }
    report_json = Path(args.report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved labels:", output_npz)
    print("Saved report:", report_json)
    print("Node label distribution:", report["label_distribution"]["node"])
    print("Region label distribution:", report["label_distribution"]["region"])


if __name__ == "__main__":
    main()
