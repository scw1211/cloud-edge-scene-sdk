"""用途：批量运行联合 ASTGCN，并为四个 METIS 区域生成边缘事件数据。"""

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from traffic_system.benchmark_utils import SystemMemorySampler
from traffic_system.conformal_risk import load_risk_calibrator
from traffic_system.decision_utils import write_jsonl
from traffic_system.infer_joint_risk_astgcn import (
    build_control_capabilities,
    build_model_from_checkpoint,
    build_top_nodes,
    ensure_finite_outputs,
    load_adjacency,
    load_config,
    load_inference_arrays,
    region_upload_policy,
    summarize_region,
    torch_load_trusted,
)
from traffic_system.risk_labels import denormalize
from traffic_system.train_joint_risk_astgcn import clip_physical_state, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate joint ASTGCN freeway edge events in one model session.")
    parser.add_argument("--config", default="configurations/PEMS08_astgcn.conf")
    parser.add_argument("--data_npz", default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz")
    parser.add_argument(
        "--checkpoint",
        default="experiments/PEMS08/joint_risk_astgcn_metis4_flowprio2_frozen/best.pt",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--samples", default="0:360:8", help="start:end:step or comma-separated ids")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--risk_calibrator", default="models/region_risk_conformal.json")
    parser.add_argument("--output_dir", default="datasets/freeway_events_joint_metis4")
    parser.add_argument("--manifest", default="datasets/freeway_events_joint_metis4_manifest.jsonl")
    parser.add_argument("--benchmark_json", default="")
    parser.add_argument("--runtime_note", default="")
    return parser.parse_args()


def parse_sample_spec(spec: str) -> List[int]:
    if ":" not in spec:
        return [int(part.strip()) for part in spec.split(",") if part.strip()]
    parts = [int(part.strip()) for part in spec.split(":")]
    if len(parts) not in (2, 3):
        raise ValueError("samples must use start:end or start:end:step")
    start, end = parts[:2]
    step = parts[2] if len(parts) == 3 else 1
    if step <= 0:
        raise ValueError("sample step must be positive")
    return list(range(start, end, step))


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def current_rss_mb() -> float:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return round(float(line.split()[1]) / 1024.0, 4)
    return 0.0


def cuda_memory_mb(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(device) / 1048576.0, 4),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / 1048576.0, 4),
    }


def record_memory(stages: Dict[str, Any], name: str, device: torch.device) -> None:
    stages[name] = {
        "rss_mb": current_rss_mb(),
        "cuda": cuda_memory_mb(device),
    }


def main() -> None:
    args = parse_args()
    system_sampler = SystemMemorySampler(0.02)
    system_sampler.start()
    device = select_device(args.device)
    memory_stages: Dict[str, Any] = {}
    record_memory(memory_stages, "runtime_initialized", device)
    config = load_config(args.config)
    arrays = load_inference_arrays(Path(args.data_npz), args.split)
    record_memory(memory_stages, "split_arrays_loaded", device)
    checkpoint = torch_load_trusted(Path(args.checkpoint), device)
    record_memory(memory_stages, "checkpoint_loaded", device)
    adj_mx, adj_filename = load_adjacency(config)
    model = build_model_from_checkpoint(config, arrays, adj_mx, checkpoint, device)
    risk_calibrator = load_risk_calibrator(Path(args.risk_calibrator))
    record_memory(memory_stages, "model_built", device)
    partitions = [[int(node) for node in part] for part in checkpoint["partitions"]]
    split_x = arrays["split_x"]
    sample_ids = parse_sample_spec(args.samples)
    invalid = [sample for sample in sample_ids if sample < 0 or sample >= split_x.shape[0]]
    if invalid:
        raise ValueError("sample ids out of range: {}".format(invalid[:10]))

    output_dir = Path(args.output_dir)
    rows: List[Dict[str, Any]] = []
    latencies: List[float] = []
    warmup_x = torch.from_numpy(split_x[sample_ids[0] : sample_ids[0] + 1].astype(np.float32)).to(device)
    with torch.no_grad():
        model(warmup_x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    record_memory(memory_stages, "warmup_completed", device)
    for sample_index, sample_id in enumerate(sample_ids, start=1):
        x = torch.from_numpy(split_x[sample_id : sample_id + 1].astype(np.float32)).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            outputs = model(x)
        ensure_finite_outputs(outputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)
        if sample_index == 1:
            record_memory(memory_stages, "first_inference_completed", device)

        forecast_norm = outputs["forecast"].detach().cpu().numpy()[0]
        forecast_raw = clip_physical_state(
            denormalize(forecast_norm[None, ...], arrays["mean"], arrays["std"])
        )[0]
        node_probs = F.softmax(outputs["node_logits"], dim=-1).detach().cpu().numpy()[0]
        region_probs_all = F.softmax(outputs["region_logits"], dim=-1).detach().cpu().numpy()[0]

        for partition_id, managed_node_ids in enumerate(partitions):
            region_probs = region_probs_all[partition_id]
            top_nodes = build_top_nodes(managed_node_ids, node_probs, forecast_raw, args.top_k)
            summary = summarize_region(
                managed_node_ids,
                node_probs,
                region_probs,
                risk_calibrator,
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
            region_id = "region_{}".format(partition_id)
            event_id = "freeway_{}_sample_{:04d}_{}".format(args.split, sample_id, edge_id)
            event = {
                "scene": "freeway_traffic_management",
                "task": "edge_freeway_congestion_risk_assessment",
                "dataset": config["Data"].get("dataset_name", "PEMS08"),
                "model": "joint_astgcn_forecast_node_region_risk",
                "risk_source": "joint_astgcn_encoder_node_region_heads",
                "risk_calibrator": args.risk_calibrator,
                "checkpoint": args.checkpoint,
                "event_id": event_id,
                "edge_id": edge_id,
                "region_id": region_id,
                "partition_id": partition_id,
                "num_partitions": len(partitions),
                "sample_split": args.split,
                "sample_id": sample_id,
                "device": str(device),
                "inference_latency_ms": round(float(latency_ms), 6),
                "time_step_minutes": 5,
                "prediction_steps": int(forecast_raw.shape[-1]),
                "prediction_horizon_minutes": int(forecast_raw.shape[-1]) * 5,
                "input_shape": list(x.shape),
                "managed_node_ids": managed_node_ids,
                "adjacency_file": adj_filename,
                "control_capabilities": build_control_capabilities(
                    partitions,
                    adj_mx,
                    partition_id,
                ),
                "region_summary": summary,
                "upload_required": upload_required,
                "upload_level": upload_level,
                "top_k_risk_nodes": top_nodes,
            }
            event_path = output_dir / "{}.json".format(event_id)
            save_json(event, event_path)
            rows.append(
                {
                    "event_id": event_id,
                    "event_path": str(event_path),
                    "sample_split": args.split,
                    "sample_id": sample_id,
                    "edge_id": edge_id,
                    "region_id": region_id,
                    "region_risk_level": summary["region_risk_level"],
                    "upload_required": upload_required,
                    "upload_level": upload_level,
                }
            )
        print("[{}/{}] sample={} latency_ms={:.3f}".format(sample_index, len(sample_ids), sample_id, latency_ms))

    write_jsonl(rows, Path(args.manifest))
    system_sampler.stop()
    benchmark = {
        "task": "joint_astgcn_jetson_inference_benchmark",
        "device": str(device),
        "checkpoint": args.checkpoint,
        "split": args.split,
        "sample_ids": sample_ids,
        "warmup_runs": 1,
        "measured_runs": len(latencies),
        "average_inference_latency_ms": round(float(np.mean(latencies)), 6),
        "p50_inference_latency_ms": round(float(np.percentile(latencies, 50)), 6),
        "p95_inference_latency_ms": round(float(np.percentile(latencies, 95)), 6),
        "max_inference_latency_ms": round(float(np.max(latencies)), 6),
        "process_max_rss_mb": round(
            float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
            4,
        ),
        "rss_scope": "full Python process including PyTorch runtime, model, and loaded dataset arrays",
        "system_ram_baseline_mb": system_sampler.baseline_mb,
        "system_ram_peak_mb": system_sampler.peak_mb,
        "system_ram_peak_delta_mb": round(
            system_sampler.peak_mb - system_sampler.baseline_mb,
            4,
        ),
        "runtime_note": args.runtime_note,
        "memory_stages": memory_stages,
    }
    if args.benchmark_json:
        save_json(benchmark, Path(args.benchmark_json))
    print("events:", len(rows))
    print("average_inference_latency_ms:", round(float(np.mean(latencies)), 4))
    print("process_max_rss_mb:", benchmark["process_max_rss_mb"])
    print("manifest:", args.manifest)


if __name__ == "__main__":
    main()
