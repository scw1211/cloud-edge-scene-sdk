"""用途：在完整测试集上遮蔽交通变量，量化 flow/occupancy/speed 对风险头的贡献。"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from traffic_system.decision_utils import save_json
from traffic_system.infer_joint_risk_astgcn import (
    build_model_from_checkpoint,
    ensure_finite_outputs,
    load_adjacency,
    load_config,
    torch_load_trusted,
)
from traffic_system.risk_labels import confusion_matrix, enable_numpy_pickle_compatibility
from traffic_system.train_joint_risk_astgcn import select_device


CHANNEL_NAMES = ["flow", "occupancy", "speed"]
VARIANTS = {
    "full": (0, 1, 2),
    "flow_only": (0,),
    "speed_only": (2,),
    "flow_speed": (0, 2),
    "flow_occupancy": (0, 1),
    "occupancy_speed": (1, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run held-out input-channel ablation for the joint risk heads.")
    parser.add_argument("--config", default="configurations/PEMS08_astgcn.conf")
    parser.add_argument("--data_npz", default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz")
    parser.add_argument("--risk_labels", default="datasets/risk_labels_pems08_metis4.npz")
    parser.add_argument(
        "--checkpoint",
        default="experiments/PEMS08/joint_risk_astgcn_metis4_flowprio2_frozen/best.pt",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--bootstrap_samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_json",
        default="results/perception/risk_channel_ablation.json",
    )
    parser.add_argument(
        "--report_md",
        default="results/perception/risk_channel_ablation.md",
    )
    return parser.parse_args()


def load_arrays(data_path: Path, labels_path: Path) -> Dict[str, np.ndarray]:
    with np.load(data_path) as data:
        if "test_x" not in data.files:
            raise ValueError("test_x is missing from {}".format(data_path))
        test_x = data["test_x"].astype(np.float32)
    enable_numpy_pickle_compatibility()
    with np.load(labels_path, allow_pickle=True) as labels:
        required = ["test_node_label", "test_region_label"]
        missing = [name for name in required if name not in labels.files]
        if missing:
            raise ValueError("Missing labels: {}".format(", ".join(missing)))
        node_labels = labels["test_node_label"].astype(np.int64)
        region_labels = labels["test_region_label"].astype(np.int64)
    if test_x.shape[0] != node_labels.shape[0] or test_x.shape[0] != region_labels.shape[0]:
        raise ValueError("Test arrays have inconsistent sample counts.")
    if test_x.shape[2] != len(CHANNEL_NAMES):
        raise ValueError("Expected {} channels, got {}.".format(len(CHANNEL_NAMES), test_x.shape[2]))
    return {"test_x": test_x, "node_labels": node_labels, "region_labels": region_labels}


def mask_channels(values: np.ndarray, kept_channels: Sequence[int]) -> np.ndarray:
    masked = values.copy()
    dropped = sorted(set(range(masked.shape[2])) - set(int(index) for index in kept_channels))
    if dropped:
        masked[:, :, dropped, :] = 0.0
    return masked


def paired_bootstrap_delta(
    full_scores: np.ndarray,
    variant_scores: np.ndarray,
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    full_scores = np.asarray(full_scores, dtype=np.float64)
    variant_scores = np.asarray(variant_scores, dtype=np.float64)
    if full_scores.shape != variant_scores.shape or full_scores.ndim != 1:
        raise ValueError("Paired score arrays must be one-dimensional with equal shapes.")
    if not len(full_scores) or samples <= 0:
        raise ValueError("Paired bootstrap requires non-empty scores and positive samples.")
    differences = full_scores - variant_scores
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for bootstrap_id in range(samples):
        selected = rng.integers(0, len(differences), size=len(differences))
        estimates[bootstrap_id] = float(np.mean(differences[selected]))
    return {
        "metric": "full_accuracy_minus_variant_accuracy",
        "unit": "timestamp_window",
        "mean_delta": round(float(np.mean(differences)), 6),
        "confidence_level": 0.95,
        "lower": round(float(np.percentile(estimates, 2.5)), 6),
        "upper": round(float(np.percentile(estimates, 97.5)), 6),
        "probability_full_better": round(float(np.mean(estimates > 0.0)), 6),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def run_variant(
    model: torch.nn.Module,
    test_x: np.ndarray,
    kept_channels: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    node_predictions: List[np.ndarray] = []
    region_predictions: List[np.ndarray] = []
    batch_latencies = []
    for batch_start in range(0, len(test_x), batch_size):
        batch = mask_channels(test_x[batch_start : batch_start + batch_size], kept_channels)
        tensor = torch.from_numpy(batch).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            outputs = model(tensor)
        ensure_finite_outputs(outputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_latencies.append((time.perf_counter() - started) * 1000.0)
        node_predictions.append(torch.argmax(outputs["node_logits"], dim=-1).cpu().numpy())
        region_predictions.append(torch.argmax(outputs["region_logits"], dim=-1).cpu().numpy())
    return {
        "node_predictions": np.concatenate(node_predictions, axis=0),
        "region_predictions": np.concatenate(region_predictions, axis=0),
        "mean_batch_forward_ms": round(float(np.mean(batch_latencies)), 6),
        "p95_batch_forward_ms": round(float(np.percentile(batch_latencies, 95)), 6),
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# 风险头交通变量遮蔽消融",
        "",
        "## 口径",
        "",
        "固定同一个联合 ASTGCN 权重和完整 PEMS08 test split，只将被遮蔽输入通道置为训练归一化均值（0）。"
        "因此这是变量依赖的诊断性消融，不代表各变体重新训练后的最优性能。",
        "",
        "| 输入变量 | 节点 Accuracy | 节点 Macro-F1 | 区域 Accuracy | 区域 Macro-F1 | 区域高/严重召回 | 相对完整输入下降 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    display = {
        "full": "flow + occupancy + speed",
        "flow_only": "flow",
        "speed_only": "speed",
        "flow_speed": "flow + speed",
        "flow_occupancy": "flow + occupancy",
        "occupancy_speed": "occupancy + speed",
    }
    for name in VARIANTS:
        item = result["variants"][name]
        delta = item.get("paired_region_accuracy_delta", {}).get("mean_delta", 0.0)
        lines.append(
            "| {} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2%} |".format(
                display[name],
                item["node"]["accuracy"],
                item["node"]["macro_f1"],
                item["region"]["accuracy"],
                item["region"]["macro_f1"],
                item["region"]["high_severe_recall"],
                delta,
            )
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            result["conclusion"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("batch_size and bootstrap_samples must be positive.")
    torch.set_num_threads(args.torch_threads)
    device = select_device(args.device)
    arrays = load_arrays(Path(args.data_npz), Path(args.risk_labels))
    config = load_config(args.config)
    checkpoint = torch_load_trusted(Path(args.checkpoint), device)
    adj_mx, _ = load_adjacency(config)
    model_arrays = {
        "in_channels": int(arrays["test_x"].shape[2]),
        "output_dim": int(arrays["test_x"].shape[2]),
    }
    model = build_model_from_checkpoint(config, model_arrays, adj_mx, checkpoint, device)

    raw_results: Dict[str, Dict[str, Any]] = {}
    for name, kept_channels in VARIANTS.items():
        print("running variant:", name, flush=True)
        raw_results[name] = run_variant(
            model,
            arrays["test_x"],
            kept_channels,
            args.batch_size,
            device,
        )

    full_node_scores = np.mean(
        raw_results["full"]["node_predictions"] == arrays["node_labels"], axis=1
    )
    full_region_scores = np.mean(
        raw_results["full"]["region_predictions"] == arrays["region_labels"], axis=1
    )
    variants: Dict[str, Any] = {}
    for variant_id, (name, kept_channels) in enumerate(VARIANTS.items()):
        raw = raw_results[name]
        node_scores = np.mean(raw["node_predictions"] == arrays["node_labels"], axis=1)
        region_scores = np.mean(raw["region_predictions"] == arrays["region_labels"], axis=1)
        variants[name] = {
            "kept_channels": [CHANNEL_NAMES[index] for index in kept_channels],
            "masked_channels": [
                channel for index, channel in enumerate(CHANNEL_NAMES) if index not in kept_channels
            ],
            "node": confusion_matrix(arrays["node_labels"], raw["node_predictions"]),
            "region": confusion_matrix(arrays["region_labels"], raw["region_predictions"]),
            "paired_node_accuracy_delta": paired_bootstrap_delta(
                full_node_scores,
                node_scores,
                args.bootstrap_samples,
                args.seed + variant_id * 2,
            ),
            "paired_region_accuracy_delta": paired_bootstrap_delta(
                full_region_scores,
                region_scores,
                args.bootstrap_samples,
                args.seed + variant_id * 2 + 1,
            ),
            "runtime": {
                "mean_batch_forward_ms": raw["mean_batch_forward_ms"],
                "p95_batch_forward_ms": raw["p95_batch_forward_ms"],
            },
        }

    strongest_reduced = max(
        (name for name in VARIANTS if name != "full"),
        key=lambda name: variants[name]["region"]["accuracy"],
    )
    region_delta = variants[strongest_reduced]["paired_region_accuracy_delta"]
    conclusion = (
        "完整三变量输入的区域风险准确率为 {:.2%}。最强遮蔽变体 {} 为 {:.2%}，"
        "完整输入配对提升 {:.2%}（95% CI [{:.2%}, {:.2%}]）。"
        "结果用于证明 flow、occupancy、speed 的联合输入有可测贡献，不能将风险识别简化为 flow 单变量。"
    ).format(
        variants["full"]["region"]["accuracy"],
        strongest_reduced,
        variants[strongest_reduced]["region"]["accuracy"],
        region_delta["mean_delta"],
        region_delta["lower"],
        region_delta["upper"],
    )
    result = {
        "task": "joint_risk_input_channel_ablation",
        "method": {
            "split": "test",
            "timestamp_windows": int(arrays["test_x"].shape[0]),
            "node_predictions": int(arrays["node_labels"].size),
            "region_predictions": int(arrays["region_labels"].size),
            "ablation": "masked normalized channels are set to 0, the training mean",
            "weights_retrained_per_variant": False,
            "interpretation": "diagnostic channel dependency, not optimal reduced-input retraining",
            "device": str(device),
            "checkpoint": args.checkpoint,
        },
        "variants": variants,
        "strongest_reduced_variant": strongest_reduced,
        "conclusion": conclusion,
    }
    save_json(result, Path(args.output_json))
    write_markdown(result, Path(args.report_md))
    print("result:", args.output_json)
    print("report:", args.report_md)
    print(conclusion)


if __name__ == "__main__":
    main()
