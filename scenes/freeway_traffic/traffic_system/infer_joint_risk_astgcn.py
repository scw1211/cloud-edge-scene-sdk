"""用途：运行联合 ASTGCN，输出交通状态预测、节点风险和区域风险。"""

import argparse
import configparser
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lib.utils import get_adjacency_matrix
from traffic_system.conformal_risk import calibrated_risk_set, load_risk_calibrator
from traffic_system.risk_labels import RISK_CLASSES, denormalize
from traffic_system.risk_model import JointRiskASTGCN
from traffic_system.train_joint_risk_astgcn import clip_physical_state, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run joint ASTGCN forecast/risk inference and emit an edge event.")
    parser.add_argument("--config", default="configurations/PEMS08_astgcn.conf")
    parser.add_argument("--data_npz", default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz")
    parser.add_argument(
        "--checkpoint",
        default="experiments/PEMS08/joint_risk_astgcn_metis4_flowprio2_frozen/best.pt",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample_id", type=int, default=0)
    parser.add_argument("--partition_id", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--risk_calibrator", default="models/region_risk_conformal.json")
    parser.add_argument("--output_json", default="results/perception/edge_event_check.json")
    return parser.parse_args()


def load_config(path: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    loaded = config.read(path)
    if not loaded:
        raise FileNotFoundError("Config file not found: {}".format(path))
    return config


def load_adjacency(config: configparser.ConfigParser) -> Tuple[np.ndarray, str]:
    data_config = config["Data"]
    id_filename = data_config["id_filename"] if config.has_option("Data", "id_filename") else None
    adj_mx, _ = get_adjacency_matrix(
        data_config["adj_filename"],
        int(data_config["num_of_vertices"]),
        id_filename,
    )
    return adj_mx, data_config["adj_filename"]


def torch_load_trusted(path: Path, device: torch.device) -> Dict[str, Any]:
    if int(np.__version__.split(".", 1)[0]) < 2:
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_inference_arrays(path: Path, split: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("Data file not found: {}".format(path))
    split_key = "{}_x".format(split)
    with np.load(path) as data:
        required = [split_key, "mean", "std"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError("Missing inference arrays: {}".format(", ".join(missing)))
        split_x = data[split_key]
        mean = data["mean"]
        std = data["std"]
    return {
        "split_x": split_x,
        "mean": mean,
        "std": std,
        "in_channels": int(split_x.shape[2]),
        "output_dim": int(mean.shape[2]),
    }


def build_model_from_checkpoint(
    config: configparser.ConfigParser,
    arrays: Dict[str, np.ndarray],
    adj_mx: np.ndarray,
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> JointRiskASTGCN:
    data_config = config["Data"]
    training_config = config["Training"]
    model_config = checkpoint.get("config", {})
    model = JointRiskASTGCN(
        device=device,
        nb_block=int(training_config["nb_block"]),
        in_channels=int(arrays["in_channels"]),
        k_order=int(training_config["K"]),
        nb_chev_filter=int(training_config["nb_chev_filter"]),
        nb_time_filter=int(training_config["nb_time_filter"]),
        time_strides=int(training_config["num_of_hours"]),
        adj_mx=adj_mx,
        num_for_predict=int(data_config["num_for_predict"]),
        len_input=int(data_config["len_input"]),
        num_of_vertices=int(data_config["num_of_vertices"]),
        partitions=checkpoint["partitions"],
        output_dim=int(arrays["output_dim"]),
        risk_hidden_dim=int(model_config.get("risk_hidden_dim", 96)),
        dropout=float(model_config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def risk_score_from_probs(probs: np.ndarray) -> float:
    severity = np.arange(len(RISK_CLASSES), dtype=np.float32)
    return float(np.dot(probs, severity) / max(1, len(RISK_CLASSES) - 1))


def ensure_finite_outputs(outputs: Dict[str, torch.Tensor]) -> None:
    for output_name, output_tensor in outputs.items():
        if isinstance(output_tensor, torch.Tensor) and not torch.isfinite(output_tensor).all():
            raise RuntimeError(
                "ASTGCN produced NaN/Inf in {}; refusing to emit a risk event.".format(output_name)
            )


def region_upload_policy(region_level: str, max_node_level: str, severe_count: int, high_count: int) -> Tuple[bool, str]:
    if region_level == "severe" or max_node_level == "severe" or severe_count > 0:
        return True, "regional_context"
    if region_level == "high" or max_node_level == "high" or high_count >= 2:
        return True, "sequence"
    if region_level == "medium" or high_count >= 1:
        return True, "feature"
    return False, "summary"


def build_top_nodes(
    managed_node_ids: Sequence[int],
    node_probs: np.ndarray,
    forecast_raw: np.ndarray,
    top_k: int,
) -> List[Dict[str, Any]]:
    rows = []
    for node_id in managed_node_ids:
        probs = node_probs[int(node_id)]
        risk_id = int(np.argmax(probs))
        risk_level = RISK_CLASSES[risk_id]
        risk_score = risk_score_from_probs(probs)
        node_forecast = forecast_raw[int(node_id)]
        rows.append(
            {
                "node_id": int(node_id),
                "risk_level": risk_level,
                "risk_score": round(risk_score, 6),
                "risk_confidence": round(float(np.max(probs)), 6),
                "risk_probabilities": {
                    name: round(float(probs[idx]), 6)
                    for idx, name in enumerate(RISK_CLASSES)
                },
                "forecast": {
                    "flow_mean": round(float(np.mean(node_forecast[0, :])), 6),
                    "occupancy_mean": round(float(np.mean(node_forecast[1, :])), 6),
                    "speed_mean": round(float(np.mean(node_forecast[2, :])), 6),
                    "speed_min": round(float(np.min(node_forecast[2, :])), 6),
                },
            }
        )
    rows.sort(key=lambda item: (RISK_CLASSES.index(item["risk_level"]), item["risk_score"]), reverse=True)
    return rows[:top_k]


def summarize_region(
    managed_node_ids: Sequence[int],
    node_probs: np.ndarray,
    region_probs: np.ndarray,
    risk_calibrator: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    node_labels = np.argmax(node_probs[list(managed_node_ids)], axis=1)
    counts = {
        name: int(np.sum(node_labels == idx))
        for idx, name in enumerate(RISK_CLASSES)
    }
    region_id = int(np.argmax(region_probs))
    summary = {
        "region_risk_level": RISK_CLASSES[region_id],
        "region_risk_score": round(risk_score_from_probs(region_probs), 6),
        "region_risk_confidence": round(float(np.max(region_probs)), 6),
        "region_risk_probabilities": {
            name: round(float(region_probs[idx]), 6)
            for idx, name in enumerate(RISK_CLASSES)
        },
        "node_risk_counts": counts,
        "mean_node_risk_score": round(float(np.mean([risk_score_from_probs(p) for p in node_probs[list(managed_node_ids)]])), 6),
        "max_node_risk_level": RISK_CLASSES[int(np.max(node_labels))],
    }
    if risk_calibrator is not None:
        summary["region_risk_calibration"] = calibrated_risk_set(
            region_probs,
            risk_calibrator,
        )
    return summary


def build_control_capabilities(
    partitions: Sequence[Sequence[int]],
    adj_mx: np.ndarray,
    partition_id: int,
) -> Dict[str, Any]:
    """Build a reproducible graph-proxy actuator map for the PEMS08 prototype."""
    managed = [int(node) for node in partitions[partition_id]]
    own = set(managed)
    boundary = []
    for node in managed:
        neighbors = np.where((adj_mx[node] != 0) | (adj_mx[:, node] != 0))[0].astype(int).tolist()
        if any(neighbor not in own for neighbor in neighbors):
            boundary.append(node)
    if not boundary:
        raise ValueError("Partition {} has no cross-region boundary node.".format(partition_id))

    degrees = {
        node: int(np.count_nonzero((adj_mx[node] != 0) | (adj_mx[:, node] != 0)))
        for node in boundary
    }
    ranked = sorted(boundary, key=lambda node: (-degrees[node], node))
    ramp_count = max(2, min(6, len(ranked) // 3 or 1))
    gateway_count = max(3, min(10, len(ranked)))
    return {
        "mapping_type": "road_graph_proxy_actuator_map",
        "mapping_note": (
            "PEMS08 has detector nodes but no actuator inventory; boundary nodes are "
            "used as reproducible ramp/gateway proxies for prototype validation."
        ),
        "variable_speed_limit_nodes": managed,
        "ramp_meter_nodes": ranked[:ramp_count],
        "reroute_gateway_nodes": ranked[:gateway_count],
    }
def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    config = load_config(args.config)
    arrays = load_inference_arrays(Path(args.data_npz), args.split)
    checkpoint = torch_load_trusted(Path(args.checkpoint), device)
    adj_mx, adj_filename = load_adjacency(config)
    model = build_model_from_checkpoint(config, arrays, adj_mx, checkpoint, device)
    risk_calibrator = load_risk_calibrator(Path(args.risk_calibrator))

    split_x = arrays["split_x"]
    if args.sample_id < 0 or args.sample_id >= split_x.shape[0]:
        raise ValueError("sample_id out of range for {} split: {}".format(args.split, args.sample_id))
    partitions = [[int(node_id) for node_id in part] for part in checkpoint["partitions"]]
    if args.partition_id < 0 or args.partition_id >= len(partitions):
        raise ValueError("partition_id out of range: {}".format(args.partition_id))

    x = torch.from_numpy(split_x[args.sample_id : args.sample_id + 1].astype(np.float32)).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(x)
    ensure_finite_outputs(outputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0

    forecast_norm = outputs["forecast"].detach().cpu().numpy()[0]
    forecast_raw = clip_physical_state(denormalize(forecast_norm[None, ...], arrays["mean"], arrays["std"]))[0]
    node_probs = F.softmax(outputs["node_logits"], dim=-1).detach().cpu().numpy()[0]
    region_probs_all = F.softmax(outputs["region_logits"], dim=-1).detach().cpu().numpy()[0]
    region_probs = region_probs_all[args.partition_id]
    managed_node_ids = partitions[args.partition_id]

    top_nodes = build_top_nodes(managed_node_ids, node_probs, forecast_raw, args.top_k)
    region_summary = summarize_region(
        managed_node_ids,
        node_probs,
        region_probs,
        risk_calibrator,
    )
    high_count = region_summary["node_risk_counts"].get("high", 0)
    severe_count = region_summary["node_risk_counts"].get("severe", 0)
    upload_required, upload_level = region_upload_policy(
        region_summary["region_risk_level"],
        region_summary["max_node_risk_level"],
        severe_count,
        high_count,
    )

    event = {
        "scene": "freeway_traffic_management",
        "task": "edge_traffic_risk_assessment",
        "dataset": config["Data"].get("dataset_name", "PEMS08"),
        "model": "joint_astgcn_forecast_node_region_risk",
        "risk_source": "joint_astgcn_encoder_node_region_heads",
        "risk_calibrator": args.risk_calibrator,
        "checkpoint": args.checkpoint,
        "edge_id": "edge_node_{}".format(args.partition_id),
        "region_id": "region_{}".format(args.partition_id),
        "partition_id": int(args.partition_id),
        "num_partitions": len(partitions),
        "sample_split": args.split,
        "sample_id": int(args.sample_id),
        "time_step_minutes": 5,
        "prediction_steps": int(forecast_raw.shape[-1]),
        "prediction_horizon_minutes": int(forecast_raw.shape[-1]) * 5,
        "device": str(device),
        "inference_latency_ms": round(float(latency_ms), 6),
        "input_shape": list(x.shape),
        "managed_node_ids": managed_node_ids,
        "control_capabilities": build_control_capabilities(
            partitions,
            adj_mx,
            args.partition_id,
        ),
        "adjacency_file": adj_filename,
        "region_summary": region_summary,
        "upload_required": upload_required,
        "upload_level": upload_level,
        "top_k_risk_nodes": top_nodes,
    }
    save_json(event, Path(args.output_json))
    print("Saved edge event:", args.output_json)
    print(
        "partition={}, region_level={}, upload={}, latency_ms={:.3f}".format(
            args.partition_id,
            region_summary["region_risk_level"],
            upload_level,
            latency_ms,
        )
    )


if __name__ == "__main__":
    main()
