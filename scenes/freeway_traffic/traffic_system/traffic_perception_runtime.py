"""用途：常驻加载联合 ASTGCN，并把一次时空预测转换为各 METIS 区域事件。"""

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from traffic_system.conformal_risk import load_risk_calibrator
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


@dataclass(frozen=True)
class TrafficPerceptionResult:
    sample_id: int
    model_forward_ms: float
    perception_ms: float
    events: List[Dict[str, Any]]


class JointTrafficPerceptionRuntime:
    """One resident model session shared by repeated traffic acceptance samples."""

    def __init__(
        self,
        config_path: Path,
        data_path: Path,
        checkpoint_path: Path,
        risk_calibrator_path: Path,
        split: str,
        device_name: str,
        top_k: int,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.config_path = Path(config_path).resolve()
        self.data_path = Path(data_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.risk_calibrator_path = Path(risk_calibrator_path).resolve()
        self.split = split
        self.top_k = int(top_k)
        self.device = select_device(device_name)

        started = time.perf_counter()
        self.config = load_config(str(self.config_path))
        self.arrays = load_inference_arrays(self.data_path, split)
        self.checkpoint = torch_load_trusted(self.checkpoint_path, self.device)
        self.adjacency, self.adjacency_filename = load_adjacency(self.config)
        self.model = build_model_from_checkpoint(
            self.config,
            self.arrays,
            self.adjacency,
            self.checkpoint,
            self.device,
        )
        self.model.eval()
        self.risk_calibrator = load_risk_calibrator(self.risk_calibrator_path)
        self.partitions = [
            [int(node) for node in partition]
            for partition in self.checkpoint["partitions"]
        ]
        self.split_x = self.arrays["split_x"]
        self.load_latency_ms = round((time.perf_counter() - started) * 1000.0, 6)

    @property
    def sample_count(self) -> int:
        return int(self.split_x.shape[0])

    @property
    def partition_count(self) -> int:
        return len(self.partitions)

    def validate_sample_ids(self, sample_ids: Sequence[int]) -> List[int]:
        normalized = [int(value) for value in sample_ids]
        if not normalized:
            raise ValueError("sample selection must not be empty")
        invalid = [
            sample_id
            for sample_id in normalized
            if sample_id < 0 or sample_id >= self.sample_count
        ]
        if invalid:
            raise ValueError(
                "sample ids outside {} split: {}".format(
                    self.split, invalid[:10]
                )
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("sample ids must be unique for an acceptance run")
        return normalized

    def warmup(self, sample_id: int) -> float:
        sample_id = self.validate_sample_ids([sample_id])[0]
        value = torch.from_numpy(
            self.split_x[sample_id : sample_id + 1].astype(np.float32)
        ).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(value)
        ensure_finite_outputs(outputs)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return round((time.perf_counter() - started) * 1000.0, 6)

    def infer_sample(self, sample_id: int) -> TrafficPerceptionResult:
        sample_id = self.validate_sample_ids([sample_id])[0]
        perception_started = time.perf_counter()
        value = torch.from_numpy(
            self.split_x[sample_id : sample_id + 1].astype(np.float32)
        ).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        forward_started = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(value)
        ensure_finite_outputs(outputs)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - forward_started) * 1000.0

        forecast_norm = outputs["forecast"].detach().cpu().numpy()[0]
        forecast_raw = clip_physical_state(
            denormalize(
                forecast_norm[None, ...],
                self.arrays["mean"],
                self.arrays["std"],
            )
        )[0]
        node_probabilities = (
            F.softmax(outputs["node_logits"], dim=-1).detach().cpu().numpy()[0]
        )
        region_probabilities = (
            F.softmax(outputs["region_logits"], dim=-1).detach().cpu().numpy()[0]
        )

        event_parts: List[Dict[str, Any]] = []
        for partition_id, managed_node_ids in enumerate(self.partitions):
            summary = summarize_region(
                managed_node_ids,
                node_probabilities,
                region_probabilities[partition_id],
                self.risk_calibrator,
            )
            severe_count = int(summary["node_risk_counts"].get("severe", 0))
            high_count = int(summary["node_risk_counts"].get("high", 0))
            upload_required, upload_level = region_upload_policy(
                summary["region_risk_level"],
                summary["max_node_risk_level"],
                severe_count,
                high_count,
            )
            event_parts.append(
                {
                    "partition_id": partition_id,
                    "managed_node_ids": managed_node_ids,
                    "summary": summary,
                    "upload_required": upload_required,
                    "upload_level": upload_level,
                    "top_nodes": build_top_nodes(
                        managed_node_ids,
                        node_probabilities,
                        forecast_raw,
                        self.top_k,
                    ),
                }
            )

        perception_ms = (time.perf_counter() - perception_started) * 1000.0
        events = [
            self._build_event(
                sample_id,
                value,
                forecast_raw,
                forward_ms,
                perception_ms,
                part,
            )
            for part in event_parts
        ]
        return TrafficPerceptionResult(
            sample_id=sample_id,
            model_forward_ms=round(forward_ms, 6),
            perception_ms=round(perception_ms, 6),
            events=events,
        )

    def _build_event(
        self,
        sample_id: int,
        value: torch.Tensor,
        forecast_raw: np.ndarray,
        forward_ms: float,
        perception_ms: float,
        part: Dict[str, Any],
    ) -> Dict[str, Any]:
        partition_id = int(part["partition_id"])
        edge_id = "edge_node_{}".format(partition_id)
        region_id = "region_{}".format(partition_id)
        return {
            "scene": "freeway_traffic_management",
            "task": "edge_freeway_congestion_risk_assessment",
            "dataset": self.config["Data"].get("dataset_name", "PEMS08"),
            "model": "joint_astgcn_forecast_node_region_risk",
            "risk_source": "joint_astgcn_encoder_node_region_heads",
            "risk_calibrator": str(self.risk_calibrator_path),
            "checkpoint": str(self.checkpoint_path),
            "event_id": "freeway_{}_sample_{:04d}_{}".format(
                self.split, sample_id, edge_id
            ),
            "edge_id": edge_id,
            "region_id": region_id,
            "partition_id": partition_id,
            "num_partitions": self.partition_count,
            "sample_split": self.split,
            "sample_id": sample_id,
            "device": str(self.device),
            "model_forward_latency_ms": round(float(forward_ms), 6),
            "inference_latency_ms": round(float(perception_ms), 6),
            "time_step_minutes": 5,
            "prediction_steps": int(forecast_raw.shape[-1]),
            "prediction_horizon_minutes": int(forecast_raw.shape[-1]) * 5,
            "input_shape": list(value.shape),
            "managed_node_ids": list(part["managed_node_ids"]),
            "adjacency_file": self.adjacency_filename,
            "control_capabilities": build_control_capabilities(
                self.partitions,
                self.adjacency,
                partition_id,
            ),
            "region_summary": dict(part["summary"]),
            "upload_required": bool(part["upload_required"]),
            "upload_level": str(part["upload_level"]),
            "top_k_risk_nodes": list(part["top_nodes"]),
        }
