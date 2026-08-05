"""直接根据交通观测窗口生成风险事件，不加载 ASTGCN 或 PyTorch。"""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from traffic_system.risk_labels import RISK_CLASSES


FEATURE_FLOW = 0
FEATURE_OCCUPANCY = 1
FEATURE_SPEED = 2
_RISK_PRIORITIES = np.linspace(0.0, 1.0, len(RISK_CLASSES), dtype=np.float64)


@dataclass(frozen=True)
class TrafficPerceptionResult:
    sample_id: int
    model_forward_ms: float
    perception_ms: float
    events: List[Dict[str, Any]]


def _bounded(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip((values - lower) / max(upper - lower, 1e-6), 0.0, 1.0)


def _risk_probabilities(
    scores: np.ndarray,
    centers: np.ndarray,
    width: float,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    logits = -np.square((values - centers.reshape(1, -1)) / width)
    logits -= np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def current_window_risk(
    raw_sample: np.ndarray,
    config: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    """Compute current node risk from [nodes, features, history] observations."""
    sample = np.asarray(raw_sample, dtype=np.float64)
    if sample.ndim != 3 or sample.shape[1] < 3 or sample.shape[2] < 2:
        raise ValueError(
            "current-state input must have shape [nodes, >=3 features, >=2 steps]"
        )
    if not np.isfinite(sample).all():
        raise ValueError("current-state input contains NaN or Inf")

    speed = sample[:, FEATURE_SPEED, :]
    occupancy = sample[:, FEATURE_OCCUPANCY, :]
    flow = sample[:, FEATURE_FLOW, :]
    reference_speed = float(config["reference_speed"])
    congestion_ratio = float(config["congestion_speed_ratio"])
    if reference_speed <= 0.0 or not 0.0 < congestion_ratio < 1.0:
        raise ValueError("invalid current-state speed thresholds")

    speed_mean_ratio = np.mean(speed, axis=1) / reference_speed
    speed_min_ratio = np.min(speed, axis=1) / reference_speed
    congested_duration = np.mean(
        speed < (reference_speed * congestion_ratio), axis=1
    )
    occupancy_mean = np.mean(occupancy, axis=1)

    split_at = max(1, speed.shape[1] // 2)
    older_speed = np.mean(speed[:, :split_at], axis=1)
    recent_speed = np.mean(speed[:, split_at:], axis=1)
    recent_drop = np.maximum(
        0.0,
        (older_speed - recent_speed) / np.maximum(older_speed, 1e-6),
    )

    weights = dict(config["risk_weights"])
    scores = (
        float(weights["mean_speed_pressure"])
        * _bounded(0.95 - speed_mean_ratio, 0.0, 0.45)
        + float(weights["minimum_speed_pressure"])
        * _bounded(0.85 - speed_min_ratio, 0.0, 0.50)
        + float(weights["congestion_duration"]) * congested_duration
        + float(weights["occupancy_pressure"])
        * _bounded(occupancy_mean, 0.06, 0.22)
        + float(weights["recent_speed_drop"])
        * _bounded(recent_drop, 0.0, 0.30)
    )
    scores = np.clip(scores, 0.0, 1.0)
    centers = np.asarray(config["risk_score_centers"], dtype=np.float64)
    if centers.shape != (len(RISK_CLASSES),):
        raise ValueError("risk_score_centers must contain four values")
    probabilities = _risk_probabilities(
        scores,
        centers,
        float(config["risk_score_width"]),
    )
    return {
        "scores": scores,
        "probabilities": probabilities,
        "flow_mean": np.mean(flow, axis=1),
        "occupancy_mean": occupancy_mean,
        "speed_mean": np.mean(speed, axis=1),
        "speed_min": np.min(speed, axis=1),
        "speed_history": speed,
    }


def _risk_score(probabilities: np.ndarray) -> float:
    return float(
        np.dot(np.asarray(probabilities, dtype=np.float64), _RISK_PRIORITIES)
    )


def _mean_risk_score(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    return float(np.mean(np.dot(values, _RISK_PRIORITIES)))


class CurrentStateTrafficPerceptionRuntime:
    """Pure NumPy runtime used to benchmark the no-forecast data path."""

    def __init__(
        self,
        data_path: Path,
        rule_config_path: Path,
        topology_path: Path,
        split: str,
        top_k: int,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        started = time.perf_counter()
        self.data_path = Path(data_path).resolve()
        self.rule_config_path = Path(rule_config_path).resolve()
        self.topology_path = Path(topology_path).resolve()
        self.split = split
        self.top_k = int(top_k)
        self.device = "cpu:numpy"

        self.rule_config = json.loads(
            self.rule_config_path.read_text(encoding="utf-8")
        )
        self.topology = json.loads(self.topology_path.read_text(encoding="utf-8"))
        self.partitions = [
            [int(node) for node in partition]
            for partition in self.rule_config["partitions"]
        ]
        flattened = sorted(node for partition in self.partitions for node in partition)
        if flattened != list(range(len(flattened))):
            raise ValueError("current-state partitions must cover every node exactly once")

        split_key = "{}_x".format(split)
        with np.load(self.data_path) as data:
            required = {split_key, "mean", "std"}
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError("missing current-state arrays: {}".format(missing))
            self.split_x = data[split_key]
            self.mean = data["mean"].reshape(1, -1, 1).astype(np.float32)
            self.std = data["std"].reshape(1, -1, 1).astype(np.float32)
        if self.split_x.shape[1] != len(flattened):
            raise ValueError("partition node count does not match traffic input")
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
            raise ValueError("sample ids outside {} split: {}".format(self.split, invalid[:10]))
        return normalized

    def warmup(self, sample_id: int) -> float:
        self.validate_sample_ids([sample_id])
        return 0.0

    def _control_capabilities(self, partition_id: int) -> Dict[str, Any]:
        region_id = "region_{}".format(partition_id)
        managed = list(self.partitions[partition_id])
        boundary = [
            int(node)
            for node in self.topology.get("region_boundary_nodes", {}).get(
                region_id, []
            )
        ]
        if not boundary:
            boundary = managed[: min(6, len(managed))]
        return {
            "mapping_type": "road_graph_proxy_actuator_map",
            "mapping_note": "Current-state path reuses the frozen METIS boundary actuator map.",
            "variable_speed_limit_nodes": managed,
            "ramp_meter_nodes": boundary[: min(6, len(boundary))],
            "reroute_gateway_nodes": boundary[: min(10, len(boundary))],
        }

    def _top_nodes(
        self,
        managed_nodes: Sequence[int],
        state: Mapping[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        probabilities = state["probabilities"]
        ranked_nodes = []
        for position, raw_node_id in enumerate(managed_nodes):
            node_id = int(raw_node_id)
            node_probs = probabilities[node_id]
            label_id = int(np.argmax(node_probs))
            # The legacy implementation sorts on the six-decimal value exposed
            # in the event, not on the unrounded score.  Keep that exact key and
            # the original managed-node position so equal keys remain stable.
            risk_score = round(_risk_score(node_probs), 6)
            ranked_nodes.append((label_id, risk_score, position, node_id))
        ranked_nodes.sort(key=lambda item: (-item[0], -item[1], item[2]))

        rows = []
        for label_id, risk_score, _, node_id in ranked_nodes[: self.top_k]:
            node_probs = probabilities[node_id]
            speed_history = state["speed_history"][node_id]
            rows.append(
                {
                    "node_id": node_id,
                    "risk_level": RISK_CLASSES[label_id],
                    "risk_score": risk_score,
                    "risk_confidence": round(float(np.max(node_probs)), 6),
                    "risk_probabilities": {
                        name: round(float(node_probs[index]), 6)
                        for index, name in enumerate(RISK_CLASSES)
                    },
                    "history_mean": round(float(np.mean(speed_history)), 6),
                    "history_last": round(float(speed_history[-1]), 6),
                    "volatility": round(float(np.std(speed_history)), 6),
                    "history_12_steps": [
                        round(float(value), 6) for value in speed_history
                    ],
                    "current_observation": {
                        "flow_mean": round(float(state["flow_mean"][node_id]), 6),
                        "occupancy_mean": round(
                            float(state["occupancy_mean"][node_id]), 6
                        ),
                        "speed_mean": round(float(state["speed_mean"][node_id]), 6),
                        "speed_min": round(float(state["speed_min"][node_id]), 6),
                    },
                }
            )
        return rows

    def infer_sample(self, sample_id: int) -> TrafficPerceptionResult:
        sample_id = self.validate_sample_ids([sample_id])[0]
        started = time.perf_counter()
        normalized = self.split_x[sample_id].astype(np.float32, copy=False)
        raw_sample = normalized * self.std + self.mean
        state = current_window_risk(raw_sample, self.rule_config)
        events = []
        for partition_id, managed_nodes in enumerate(self.partitions):
            managed = np.asarray(managed_nodes, dtype=np.int64)
            node_probs = state["probabilities"][managed]
            node_labels = np.argmax(node_probs, axis=1)
            counts = {
                name: int(np.sum(node_labels == index))
                for index, name in enumerate(RISK_CLASSES)
            }
            mean_probs = np.mean(node_probs, axis=0)
            worst_probs = node_probs[int(np.argmax(state["scores"][managed]))]
            region_probs = 0.75 * mean_probs + 0.25 * worst_probs
            region_probs = region_probs / np.sum(region_probs)
            region_label_id = int(np.argmax(region_probs))
            max_label_id = int(np.max(node_labels))
            high_count = counts["high"]
            severe_count = counts["severe"]
            if severe_count:
                upload_required, upload_level = True, "regional_context"
            elif high_count >= 1:
                upload_required, upload_level = True, "sequence"
            elif counts["medium"] >= 1:
                upload_required, upload_level = True, "feature"
            else:
                upload_required, upload_level = False, "summary"
            summary = {
                "region_risk_level": RISK_CLASSES[region_label_id],
                "region_risk_score": round(_risk_score(region_probs), 6),
                "region_risk_confidence": round(float(np.max(region_probs)), 6),
                "region_risk_probabilities": {
                    name: round(float(region_probs[index]), 6)
                    for index, name in enumerate(RISK_CLASSES)
                },
                "node_risk_counts": counts,
                "mean_node_risk_score": round(_mean_risk_score(node_probs), 6),
                "max_node_risk_level": RISK_CLASSES[max_label_id],
            }
            events.append(
                {
                    "scene": "freeway_traffic_management",
                    "task": "edge_freeway_current_state_risk_assessment",
                    "dataset": "PEMS08",
                    "model": "current_window_risk_rules_v1",
                    "model_version": "current-state-v1",
                    "output_type": "current_state_risk",
                    "risk_source": "current_observed_12_step_window",
                    "checkpoint": "none",
                    "event_id": "freeway_{}_sample_{:04d}_edge_node_{}".format(
                        self.split, sample_id, partition_id
                    ),
                    "edge_id": "edge_node_{}".format(partition_id),
                    "region_id": "region_{}".format(partition_id),
                    "partition_id": partition_id,
                    "num_partitions": self.partition_count,
                    "sample_split": self.split,
                    "sample_id": sample_id,
                    "device": self.device,
                    "model_forward_latency_ms": 0.0,
                    "inference_latency_ms": 0.0,
                    "time_step_minutes": 5,
                    "observation_window_minutes": int(raw_sample.shape[-1]) * 5,
                    "prediction_steps": 0,
                    "prediction_horizon_minutes": 0,
                    "input_shape": list(normalized.shape),
                    "managed_node_ids": list(managed_nodes),
                    "control_capabilities": self._control_capabilities(partition_id),
                    "region_summary": summary,
                    "upload_required": upload_required,
                    "upload_level": upload_level,
                    "top_k_risk_nodes": self._top_nodes(managed_nodes, state),
                    "perception_mode": "current_state",
                }
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        for event in events:
            event["inference_latency_ms"] = round(elapsed_ms, 6)
        return TrafficPerceptionResult(
            sample_id=sample_id,
            model_forward_ms=0.0,
            perception_ms=round(elapsed_ms, 6),
            events=events,
        )
