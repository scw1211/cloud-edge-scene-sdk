"""用途：训练交通状态预测、节点风险和区域风险联合 ASTGCN 模型。"""

import argparse
import configparser
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from traffic_system.graph_partition import partition_graph
from lib.utils import get_adjacency_matrix
from traffic_system.risk_labels import (
    RISK_CLASSES,
    class_weights,
    confusion_matrix,
    denormalize,
    label_distribution,
)
from traffic_system.risk_model import JointRiskASTGCN, aggregate_node_probs_to_regions


FEATURE_NAMES = ["flow", "occupancy", "speed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train joint ASTGCN forecast and risk heads.")
    parser.add_argument("--config", default="configurations/PEMS08_astgcn.conf")
    parser.add_argument("--data_npz", default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz")
    parser.add_argument("--risk_label_npz", default="datasets/risk_labels_pems08_metis4.npz")
    parser.add_argument("--pretrained_forecast", default="experiments/PEMS08/astgcn_multitask_flowprio2/best.params")
    parser.add_argument("--output_dir", default="experiments/PEMS08/joint_risk_astgcn_metis4")
    parser.add_argument("--metrics_json", default="results/perception/joint_risk_astgcn_metis4.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num_partitions", type=int, default=4)
    parser.add_argument(
        "--partition_method",
        default="metis",
        choices=["metis", "spectral", "graph_bfs", "graph", "contiguous"],
    )
    parser.add_argument("--overlap_hops", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--risk_hidden_dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--forecast_weight", type=float, default=1.0)
    parser.add_argument("--node_risk_weight", type=float, default=0.8)
    parser.add_argument("--region_risk_weight", type=float, default=0.6)
    parser.add_argument("--consistency_weight", type=float, default=0.15)
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--class_weight_power", type=float, default=0.75)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=2)
    parser.add_argument("--train_limit_samples", type=int, default=0)
    parser.add_argument("--val_limit_samples", type=int, default=0)
    parser.add_argument("--test_limit_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda:0")
    return torch.device("cpu")


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
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError("Missing keys in {}: {}".format(path, ", ".join(missing)))
    return {key: data[key] for key in data.files}


def load_risk_label_arrays(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            "Risk label file not found: {}. Run `python -m traffic_system.build_paper_risk_labels` first.".format(path)
        )
    data = np.load(path, allow_pickle=True)
    required = [
        "train_node_label",
        "val_node_label",
        "test_node_label",
        "train_region_label",
        "val_region_label",
        "test_region_label",
    ]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError("Risk label file missing keys: {}".format(", ".join(missing)))
    return {key: data[key] for key in data.files}


def limit_samples(x: np.ndarray, y: np.ndarray, limit: int) -> Tuple[np.ndarray, np.ndarray]:
    if limit and limit > 0:
        return x[:limit], y[:limit]
    return x, y


def normalize_target(target: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (target - mean) / np.where(std == 0, 1.0, std)


def clip_physical_state(values: np.ndarray) -> np.ndarray:
    clipped = values.copy()
    clipped[:, :, 0, :] = np.maximum(clipped[:, :, 0, :], 0.0)
    clipped[:, :, 1, :] = np.clip(clipped[:, :, 1, :], 0.0, 1.0)
    clipped[:, :, 2, :] = np.maximum(clipped[:, :, 2, :], 0.0)
    return clipped


def masked_mape_value(y_true: np.ndarray, y_pred: np.ndarray, null_val: float = 0.0) -> float:
    mask = np.not_equal(y_true, null_val)
    if not np.any(mask):
        return 0.0
    denom = np.where(np.abs(y_true) < 1e-5, np.nan, y_true)
    mape = np.abs((y_pred - y_true) / denom)
    mape = np.where(mask, mape, np.nan)
    return float(np.nanmean(mape))


def make_loader(
    x: np.ndarray,
    y_raw: np.ndarray,
    node_labels: np.ndarray,
    region_labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    y_norm = normalize_target(y_raw, mean, std).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y_norm),
        torch.from_numpy(node_labels.astype(np.int64)),
        torch.from_numpy(region_labels.astype(np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_adjacency(config: configparser.ConfigParser) -> Tuple[np.ndarray, str]:
    data_config = config["Data"]
    id_filename = data_config["id_filename"] if config.has_option("Data", "id_filename") else None
    adj_mx, _ = get_adjacency_matrix(
        data_config["adj_filename"],
        int(data_config["num_of_vertices"]),
        id_filename,
    )
    return adj_mx, data_config["adj_filename"]


def build_model(
    config: configparser.ConfigParser,
    arrays: Dict[str, np.ndarray],
    adj_mx: np.ndarray,
    partitions,
    args: argparse.Namespace,
    device: torch.device,
) -> JointRiskASTGCN:
    data_config = config["Data"]
    training_config = config["Training"]
    return JointRiskASTGCN(
        device=device,
        nb_block=int(training_config["nb_block"]),
        in_channels=int(arrays["train_x"].shape[2]),
        k_order=int(training_config["K"]),
        nb_chev_filter=int(training_config["nb_chev_filter"]),
        nb_time_filter=int(training_config["nb_time_filter"]),
        time_strides=int(training_config["num_of_hours"]),
        adj_mx=adj_mx,
        num_for_predict=int(data_config["num_for_predict"]),
        len_input=int(data_config["len_input"]),
        num_of_vertices=int(data_config["num_of_vertices"]),
        partitions=partitions,
        output_dim=int(arrays["train_target"].shape[2]),
        risk_hidden_dim=int(args.risk_hidden_dim),
        dropout=float(args.dropout),
    ).to(device)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    num_classes = logits.shape[-1]
    logits_flat = logits.reshape(-1, num_classes)
    targets_flat = targets.reshape(-1)
    ce = F.cross_entropy(logits_flat, targets_flat, weight=weights, reduction="none")
    plain_ce = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    pt = torch.exp(-plain_ce)
    return (((1.0 - pt) ** gamma) * ce).mean()


def consistency_loss(model: JointRiskASTGCN, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    target = aggregate_node_probs_to_regions(
        outputs["node_logits"],
        model.region_mask,
        model.region_size,
    ).detach()
    return F.kl_div(F.log_softmax(outputs["region_logits"], dim=-1), target, reduction="batchmean")


def set_backbone_trainable(model: JointRiskASTGCN, trainable: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = trainable


def run_epoch(
    model: JointRiskASTGCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    node_weights: torch.Tensor,
    region_weights: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "forecast": 0.0, "node": 0.0, "region": 0.0, "consistency": 0.0}
    num_batches = 0
    for x, y, node_labels, region_labels in loader:
        x = x.to(device)
        y = y.to(device)
        node_labels = node_labels.to(device)
        region_labels = region_labels.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        forecast = F.smooth_l1_loss(outputs["forecast"], y)
        node = focal_loss(outputs["node_logits"], node_labels, node_weights, args.focal_gamma)
        region = focal_loss(outputs["region_logits"], region_labels, region_weights, args.focal_gamma)
        consistency = consistency_loss(model, outputs)
        loss = (
            args.forecast_weight * forecast
            + args.node_risk_weight * node
            + args.region_risk_weight * region
            + args.consistency_weight * consistency
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        totals["loss"] += float(loss.item())
        totals["forecast"] += float(forecast.item())
        totals["node"] += float(node.item())
        totals["region"] += float(region.item())
        totals["consistency"] += float(consistency.item())
        num_batches += 1
    return {key: round(value / max(1, num_batches), 6) for key, value in totals.items()}


def evaluate(
    model: JointRiskASTGCN,
    loader: DataLoader,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    forecast_pred = []
    forecast_true = []
    node_pred = []
    node_true = []
    region_pred = []
    region_true = []
    start = time.perf_counter()
    with torch.no_grad():
        for x, y, node_labels, region_labels in loader:
            outputs = model(x.to(device))
            forecast_pred.append(outputs["forecast"].detach().cpu().numpy())
            forecast_true.append(y.numpy())
            node_pred.append(outputs["node_logits"].argmax(dim=-1).detach().cpu().numpy())
            node_true.append(node_labels.numpy())
            region_pred.append(outputs["region_logits"].argmax(dim=-1).detach().cpu().numpy())
            region_true.append(region_labels.numpy())
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    pred_norm = np.concatenate(forecast_pred, axis=0)
    true_norm = np.concatenate(forecast_true, axis=0)
    pred_raw = clip_physical_state(denormalize(pred_norm, mean, std))
    true_raw = denormalize(true_norm, mean, std)
    feature_metrics: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(FEATURE_NAMES):
        diff = pred_raw[:, :, idx, :] - true_raw[:, :, idx, :]
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mape = masked_mape_value(true_raw[:, :, idx, :], pred_raw[:, :, idx, :], 0.0)
        feature_metrics[name] = {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "mape": round(mape, 6),
        }
    node_true_arr = np.concatenate(node_true, axis=0)
    node_pred_arr = np.concatenate(node_pred, axis=0)
    region_true_arr = np.concatenate(region_true, axis=0)
    region_pred_arr = np.concatenate(region_pred, axis=0)
    return {
        "latency_ms_per_sample": round(elapsed_ms / max(1, pred_norm.shape[0]), 6),
        "forecast": feature_metrics,
        "node_risk": confusion_matrix(node_true_arr, node_pred_arr),
        "region_risk": confusion_matrix(region_true_arr, region_pred_arr),
    }


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def torch_load_trusted(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    config = load_config(args.config)
    arrays = load_arrays(Path(args.data_npz))
    risk_labels = load_risk_label_arrays(Path(args.risk_label_npz))
    adj_mx, adj_filename = load_adjacency(config)
    partitions = partition_graph(
        adj_mx,
        int(config["Data"]["num_of_vertices"]),
        args.num_partitions,
        args.partition_method,
        args.overlap_hops,
    )

    train_x, train_y = limit_samples(arrays["train_x"], arrays["train_target"], args.train_limit_samples)
    val_x, val_y = limit_samples(arrays["val_x"], arrays["val_target"], args.val_limit_samples)
    test_x, test_y = limit_samples(arrays["test_x"], arrays["test_target"], args.test_limit_samples)
    mean = arrays["mean"]
    std = arrays["std"]

    train_node = risk_labels["train_node_label"][: train_x.shape[0]]
    train_region = risk_labels["train_region_label"][: train_x.shape[0]]
    val_node = risk_labels["val_node_label"][: val_x.shape[0]]
    val_region = risk_labels["val_region_label"][: val_x.shape[0]]
    test_node = risk_labels["test_node_label"][: test_x.shape[0]]
    test_region = risk_labels["test_region_label"][: test_x.shape[0]]
    expected_nodes = int(config["Data"]["num_of_vertices"])
    expected_regions = len(partitions)
    for name, labels in [("train_node", train_node), ("val_node", val_node), ("test_node", test_node)]:
        if labels.shape[1] != expected_nodes:
            raise ValueError("{} label node dimension mismatch: {} != {}".format(name, labels.shape[1], expected_nodes))
    for name, labels in [("train_region", train_region), ("val_region", val_region), ("test_region", test_region)]:
        if labels.shape[1] != expected_regions:
            raise ValueError(
                "{} label region dimension mismatch: {} != {}. Rebuild risk labels for this partition setting.".format(
                    name,
                    labels.shape[1],
                    expected_regions,
                )
            )

    train_loader = make_loader(train_x, train_y, train_node, train_region, mean, std, args.batch_size, True)
    val_loader = make_loader(val_x, val_y, val_node, val_region, mean, std, args.batch_size, False)
    test_loader = make_loader(test_x, test_y, test_node, test_region, mean, std, args.batch_size, False)

    model = build_model(config, arrays, adj_mx, partitions, args, device)
    if args.pretrained_forecast:
        weight_path = Path(args.pretrained_forecast)
        if not weight_path.exists():
            raise FileNotFoundError("Pretrained forecast weight not found: {}".format(weight_path))
        model.backbone.load_state_dict(torch.load(weight_path, map_location=device))

    node_weights = torch.tensor(
        class_weights(train_node, args.class_weight_power),
        dtype=torch.float32,
        device=device,
    )
    region_weights = torch.tensor(
        class_weights(train_region, args.class_weight_power),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_val = None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        set_backbone_trainable(model, epoch > args.freeze_backbone_epochs)
        if epoch == args.freeze_backbone_epochs + 1:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        train_losses = run_epoch(model, train_loader, optimizer, device, node_weights, region_weights, args)
        val_metrics = evaluate(model, val_loader, mean, std, device)
        val_score = val_metrics["node_risk"]["weighted_f1"] + val_metrics["region_risk"]["weighted_f1"]
        record = {"epoch": epoch, "train": train_losses, "val_score": round(val_score, 6)}
        history.append(record)
        print(
            "epoch={:03d} loss={:.6f} val_node_f1={:.6f} val_region_f1={:.6f}".format(
                epoch,
                train_losses["loss"],
                val_metrics["node_risk"]["weighted_f1"],
                val_metrics["region_risk"]["weighted_f1"],
            )
        )
        if best_val is None or val_score > best_val:
            best_val = val_score
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "num_partitions": args.num_partitions,
                        "partition_method": args.partition_method,
                        "overlap_hops": args.overlap_hops,
                        "risk_hidden_dim": args.risk_hidden_dim,
                        "dropout": args.dropout,
                    },
                    "partitions": [[int(node_id) for node_id in part] for part in partitions],
                    "risk_classes": RISK_CLASSES,
                    "mean": mean,
                    "std": std,
                },
                output_dir / "best.pt",
            )

    checkpoint = torch_load_trusted(output_dir / "best.pt", device)
    model.load_state_dict(checkpoint["state_dict"])
    final_metrics = {
        "task": "joint_astgcn_forecast_node_region_risk",
        "dataset": config["Data"].get("dataset_name", "unknown"),
        "adjacency_file": adj_filename,
        "device": str(device),
        "partitions": [[int(node_id) for node_id in part] for part in partitions],
        "risk_classes": RISK_CLASSES,
        "label_distribution": {
            "train_node": label_distribution(train_node),
            "train_region": label_distribution(train_region),
            "val_node": label_distribution(val_node),
            "val_region": label_distribution(val_region),
            "test_node": label_distribution(test_node),
            "test_region": label_distribution(test_region),
        },
        "train_config": vars(args),
        "history": history,
        "val": evaluate(model, val_loader, mean, std, device),
        "test": evaluate(model, test_loader, mean, std, device),
        "checkpoint": str(output_dir / "best.pt"),
    }
    save_json(final_metrics, Path(args.metrics_json))
    print("Saved checkpoint:", output_dir / "best.pt")
    print("Saved metrics:", args.metrics_json)


if __name__ == "__main__":
    main()
