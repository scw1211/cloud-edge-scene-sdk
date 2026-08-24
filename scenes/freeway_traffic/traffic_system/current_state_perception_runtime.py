"""直接根据交通观测窗口生成风险事件，不加载 ASTGCN 或 PyTorch。"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from traffic_system.road_sets import (
    build_overlap_partial_assessments,
    build_road_set_catalog,
)
from traffic_system.overlap_observations import (  # noqa: E402
    build_canonical_overlap_observation,
)
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
        self.source_dataset_sha256 = _sha256_file(self.data_path)
        self.model_artifact_sha256 = _sha256_file(self.rule_config_path)

        self.rule_config = json.loads(
            self.rule_config_path.read_text(encoding="utf-8")
        )
        self.topology = json.loads(self.topology_path.read_text(encoding="utf-8"))
        self.road_set_catalog = build_road_set_catalog(self.topology)
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
        state_node_indices: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        probabilities = state["probabilities"]
        ranked_nodes = []
        if state_node_indices is None:
            state_node_indices = managed_nodes
        if len(state_node_indices) != len(managed_nodes):
            raise ValueError("state_node_indices must match managed_nodes")
        for position, (raw_node_id, raw_state_index) in enumerate(
            zip(managed_nodes, state_node_indices)
        ):
            node_id = int(raw_node_id)
            state_index = int(raw_state_index)
            node_probs = probabilities[state_index]
            label_id = int(np.argmax(node_probs))
            # The legacy implementation sorts on the six-decimal value exposed
            # in the event, not on the unrounded score.  Keep that exact key and
            # the original managed-node position so equal keys remain stable.
            risk_score = round(_risk_score(node_probs), 6)
            ranked_nodes.append(
                (label_id, risk_score, position, node_id, state_index)
            )
        ranked_nodes.sort(key=lambda item: (-item[0], -item[1], item[2]))

        rows = []
        for label_id, risk_score, _, node_id, state_index in ranked_nodes[
            : self.top_k
        ]:
            node_probs = probabilities[state_index]
            speed_history = state["speed_history"][state_index]
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
                        "flow_mean": round(
                            float(state["flow_mean"][state_index]), 6
                        ),
                        "occupancy_mean": round(
                            float(state["occupancy_mean"][state_index]), 6
                        ),
                        "speed_mean": round(
                            float(state["speed_mean"][state_index]), 6
                        ),
                        "speed_min": round(
                            float(state["speed_min"][state_index]), 6
                        ),
                    },
                }
            )
        return rows

    def _partition_event(
        self,
        sample_id: int,
        partition_id: int,
        managed_nodes: Sequence[int],
        state: Mapping[str, np.ndarray],
        state_node_indices: Sequence[int],
        input_shape: Sequence[int],
        observation_steps: int,
        overlap_observations: Sequence[Mapping[str, Any]] = (),
    ) -> Dict[str, Any]:
        """Materialize one already-assigned METIS region as one native event."""
        indices = np.asarray(state_node_indices, dtype=np.int64)
        node_probs = state["probabilities"][indices]
        node_labels = np.argmax(node_probs, axis=1)
        counts = {
            name: int(np.sum(node_labels == index))
            for index, name in enumerate(RISK_CLASSES)
        }
        mean_probs = np.mean(node_probs, axis=0)
        local_scores = state["scores"][indices]
        worst_probs = node_probs[int(np.argmax(local_scores))]
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
            # This compact physical-state summary covers every node in the
            # already-assigned METIS partition, not only the reported top-k.
            "current_observation": {
                "node_count": int(len(indices)),
                "flow_mean": round(
                    float(np.mean(state["flow_mean"][indices])), 6
                ),
                "occupancy_mean": round(
                    float(np.mean(state["occupancy_mean"][indices])), 6
                ),
                "speed_mean": round(
                    float(np.mean(state["speed_mean"][indices])), 6
                ),
                "speed_min": round(
                    float(np.min(state["speed_min"][indices])), 6
                ),
            },
        }
        top_nodes = self._top_nodes(
            managed_nodes,
            state,
            state_node_indices=state_node_indices,
        )
        region_id = "region_{}".format(partition_id)
        aggregation_member = "edge_node_{}".format(partition_id)
        incident_overlap_observations = [
            dict(observation)
            for observation in overlap_observations
            if region_id in observation.get("required_regions", [])
        ]
        road_set_assessments = build_overlap_partial_assessments(
            self.road_set_catalog,
            region_id,
            aggregation_member,
            incident_overlap_observations,
            self.model_artifact_sha256,
            self.rule_config,
            "edge_local_raw",
        )
        return {
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
            "edge_id": aggregation_member,
            "region_id": region_id,
            "partition_id": partition_id,
            "num_partitions": self.partition_count,
            "sample_split": self.split,
            "sample_id": sample_id,
            "device": self.device,
            "model_forward_latency_ms": 0.0,
            "inference_latency_ms": 0.0,
            "time_step_minutes": 5,
            "observation_window_minutes": int(observation_steps) * 5,
            "prediction_steps": 0,
            "prediction_horizon_minutes": 0,
            "input_shape": [int(value) for value in input_shape],
            "managed_node_ids": [int(value) for value in managed_nodes],
            "control_capabilities": self._control_capabilities(partition_id),
            "region_summary": summary,
            "upload_required": upload_required,
            "upload_level": upload_level,
            "top_k_risk_nodes": top_nodes,
            "road_set_overlap_observations": incident_overlap_observations,
            "road_set_assessments": road_set_assessments,
            "perception_mode": "current_state",
        }

    def infer_sample(self, sample_id: int) -> TrafficPerceptionResult:
        sample_id = self.validate_sample_ids([sample_id])[0]
        started = time.perf_counter()
        normalized = self.split_x[sample_id].astype(np.float32, copy=False)
        raw_sample = normalized * self.std + self.mean
        state = current_window_risk(raw_sample, self.rule_config)
        overlap_observations = [
            build_canonical_overlap_observation(
                road_set,
                raw_sample[
                    np.asarray(road_set["member_nodes"], dtype=np.int64)
                ],
                road_set["member_nodes"],
                self.rule_config,
                dataset="PEMS08",
                split=self.split,
                sample_id=sample_id,
                source_dataset_sha256=self.source_dataset_sha256,
                model_artifact_sha256=self.model_artifact_sha256,
            )
            for road_set in self.road_set_catalog["road_sets"]
        ]
        events = []
        for partition_id, managed_nodes in enumerate(self.partitions):
            events.append(
                self._partition_event(
                    sample_id=sample_id,
                    partition_id=partition_id,
                    managed_nodes=managed_nodes,
                    state=state,
                    state_node_indices=managed_nodes,
                    input_shape=normalized.shape,
                    observation_steps=raw_sample.shape[-1],
                    overlap_observations=overlap_observations,
                )
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PartitionCurrentStateTrafficPerceptionRuntime(
    CurrentStateTrafficPerceptionRuntime
):
    """Resident runtime for one primary shard plus read-only overlap sidecars."""

    def __init__(
        self,
        manifest_path: Path,
        partition_id: int,
        rule_config_path: Path,
        topology_path: Path,
        split: str,
        top_k: int,
        verify_sha256: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        started = time.perf_counter()
        self.manifest_path = Path(manifest_path).resolve()
        self.rule_config_path = Path(rule_config_path).resolve()
        self.topology_path = Path(topology_path).resolve()
        self.split = split
        self.top_k = int(top_k)
        self.device = "cpu:numpy"
        self.partition_id = int(partition_id)

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", 0)) != 2:
            raise ValueError("unsupported partition manifest schema")
        if str(manifest.get("format", "")) != (
            "pems08_preassigned_metis_edge_partitions"
        ):
            raise ValueError("partition manifest format is not recognized")
        self.rule_config = json.loads(
            self.rule_config_path.read_text(encoding="utf-8")
        )
        self.topology = json.loads(self.topology_path.read_text(encoding="utf-8"))
        self.road_set_catalog = build_road_set_catalog(self.topology)
        self.partitions = [
            [int(node) for node in partition]
            for partition in self.rule_config["partitions"]
        ]
        if self.partition_id < 0 or self.partition_id >= len(self.partitions):
            raise ValueError("partition_id is outside configured partitions")
        flattened = sorted(
            node for partition in self.partitions for node in partition
        )
        if flattened != list(range(len(flattened))):
            raise ValueError(
                "current-state partitions must cover every node exactly once"
            )

        actual_rule_sha256 = _sha256_file(self.rule_config_path)
        actual_topology_sha256 = _sha256_file(self.topology_path)
        contract = manifest.get("partition_contract", {})
        if not isinstance(contract, dict):
            raise ValueError("partition contract must be an object")
        if int(contract.get("partition_count", 0)) != len(self.partitions):
            raise ValueError("partition manifest count does not match rule config")
        if contract.get("runtime_repartition_allowed") is not False:
            raise ValueError("partition manifest must prohibit runtime repartition")
        if actual_rule_sha256 != str(contract.get("rule_config_sha256", "")):
            raise ValueError("rule config SHA-256 does not match manifest")
        if actual_topology_sha256 != str(contract.get("topology_sha256", "")):
            raise ValueError("topology SHA-256 does not match manifest")

        from traffic_system.overlap_observations import (
            OVERLAP_PREPROCESS_VERSION,
            OVERLAP_SCHEMA_VERSION,
        )

        overlap_contract = manifest.get("overlap_contract")
        if not isinstance(overlap_contract, dict):
            raise ValueError("overlap contract must be an object")
        if int(overlap_contract.get("schema_version", 0)) != int(
            OVERLAP_SCHEMA_VERSION
        ):
            raise ValueError("overlap contract schema does not match runtime")
        if str(overlap_contract.get("kind", "")) != (
            "canonical_shared_overlap_subscription"
        ):
            raise ValueError("overlap contract kind is not recognized")
        if str(overlap_contract.get("dataset", "")) != "PEMS08":
            raise ValueError("overlap contract dataset is not recognized")
        if str(overlap_contract.get("preprocess_version", "")) != (
            OVERLAP_PREPROCESS_VERSION
        ):
            raise ValueError("overlap preprocess identity does not match runtime")
        if str(overlap_contract.get("topology_sha256", "")) != (
            actual_topology_sha256
        ):
            raise ValueError("overlap topology identity does not match runtime")
        if str(overlap_contract.get("model_artifact_sha256", "")) != (
            actual_rule_sha256
        ):
            raise ValueError("overlap model identity does not match runtime")
        source = manifest.get("source")
        if not isinstance(source, dict):
            raise ValueError("partition source identity must be an object")
        source_dataset_sha256 = str(source.get("sha256", ""))
        if (
            len(source_dataset_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source_dataset_sha256.lower()
            )
        ):
            raise ValueError("partition source SHA-256 is invalid")
        if str(overlap_contract.get("source_dataset_sha256", "")) != (
            source_dataset_sha256
        ):
            raise ValueError("overlap source identity does not match manifest")
        try:
            source_dataset_bytes = int(source["bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("partition source byte identity is invalid") from error
        if source_dataset_bytes <= 0:
            raise ValueError("partition source byte identity is invalid")
        if overlap_contract.get("primary_sensor_ownership_changed") is not False:
            raise ValueError("overlap contract must preserve primary ownership")
        if overlap_contract.get("sidecar_nodes_are_managed_nodes") is not False:
            raise ValueError("overlap sidecar nodes must not become managed nodes")
        if int(overlap_contract.get("owners_per_subscription", 0)) != 2:
            raise ValueError("each overlap subscription must have two owners")

        records = manifest.get("partitions", [])
        if not isinstance(records, list):
            raise ValueError("partition manifest records must be a list")
        if len(records) != len(self.partitions):
            raise ValueError("partition manifest must contain every primary shard")
        records_by_id: Dict[int, Mapping[str, Any]] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("partition manifest record must be an object")
            try:
                record_partition_id = int(record["partition_id"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("partition manifest owner mapping is invalid") from error
            if (
                record_partition_id < 0
                or record_partition_id >= len(self.partitions)
                or record_partition_id in records_by_id
            ):
                raise ValueError("partition manifest owner mapping is invalid")
            expected_nodes = self.partitions[record_partition_id]
            try:
                record_nodes = [
                    int(node) for node in record["global_node_ids"]
                ]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("partition manifest node mapping is invalid") from error
            if (
                str(record.get("region_id", ""))
                != "region_{}".format(record_partition_id)
                or str(record.get("edge_id", ""))
                != "edge_node_{}".format(record_partition_id)
                or int(record.get("node_count", -1)) != len(expected_nodes)
                or record_nodes != expected_nodes
            ):
                raise ValueError("partition manifest node assignment has drifted")
            records_by_id[record_partition_id] = record

        record = records_by_id[self.partition_id]
        filename = str(record.get("file", "")).strip()
        record_sha256 = str(record.get("sha256", "")).strip().lower()
        if not filename or len(record_sha256) != 64:
            raise ValueError("partition shard file identity is incomplete")
        self.data_path = self._manifest_child_path(filename)
        if not self.data_path.is_file():
            raise ValueError("partition shard file is missing")
        if self.data_path.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError("partition shard size does not match manifest")
        if verify_sha256 and _sha256_file(self.data_path) != record_sha256:
            raise ValueError("partition shard SHA-256 does not match manifest")

        split_key = "{}_x".format(split)
        with np.load(self.data_path, allow_pickle=False) as data:
            required = {
                split_key,
                "mean",
                "std",
                "global_node_ids",
                "partition_id",
                "num_partitions",
                "feature_names",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError("missing partition arrays: {}".format(missing))
            stored_partition_id = int(np.asarray(data["partition_id"]).item())
            stored_partition_count = int(np.asarray(data["num_partitions"]).item())
            if stored_partition_id != self.partition_id:
                raise ValueError("partition shard identity does not match request")
            if stored_partition_count != len(self.partitions):
                raise ValueError("partition shard count does not match rule config")
            self.managed_node_ids = [
                int(node) for node in np.asarray(data["global_node_ids"]).tolist()
            ]
            if self.managed_node_ids != self.partitions[self.partition_id]:
                raise ValueError("partition shard node assignment has drifted")
            self.split_x = np.asarray(data[split_key])
            self.mean = np.asarray(data["mean"]).reshape(1, -1, 1).astype(
                np.float32
            )
            self.std = np.asarray(data["std"]).reshape(1, -1, 1).astype(
                np.float32
            )
            self.feature_names = np.asarray(data["feature_names"])
        if self.split_x.shape[1] != len(self.managed_node_ids):
            raise ValueError("partition shard node count does not match assignment")
        if self.split_x.ndim != 4:
            raise ValueError("partition shard observations must be rank four")

        catalog_by_id = {
            str(road_set["road_set_id"]): road_set
            for road_set in self.road_set_catalog["road_sets"]
        }
        overlap_records = manifest.get("road_sets")
        if not isinstance(overlap_records, list):
            raise ValueError("overlap road-set records must be a list")
        if int(overlap_contract.get("subscription_count", -1)) != len(
            overlap_records
        ):
            raise ValueError("overlap subscription count does not match records")
        if len(overlap_records) != len(catalog_by_id):
            raise ValueError("overlap records do not cover the topology catalog")

        region_to_partition = {
            "region_{}".format(partition_id): partition_id
            for partition_id in range(len(self.partitions))
        }
        overlap_records_by_id: Dict[str, Mapping[str, Any]] = {}
        for overlap_record in overlap_records:
            if not isinstance(overlap_record, Mapping):
                raise ValueError("overlap road-set record must be an object")
            road_set_id = str(overlap_record.get("road_set_id", "")).strip()
            if not road_set_id or road_set_id in overlap_records_by_id:
                raise ValueError("overlap road-set identity is invalid")
            road_set = catalog_by_id.get(road_set_id)
            if road_set is None:
                raise ValueError("overlap road set is absent from topology")
            expected_regions = [
                str(region) for region in road_set["required_regions"]
            ]
            if any(region not in region_to_partition for region in expected_regions):
                raise ValueError("overlap owner region is absent from partitions")
            expected_partition_ids = sorted(
                region_to_partition[region] for region in expected_regions
            )
            expected_partitions = [
                "edge_node_{}".format(partition_id)
                for partition_id in expected_partition_ids
            ]
            try:
                record_regions = [
                    str(region) for region in overlap_record["required_regions"]
                ]
                record_partition_ids = [
                    int(value)
                    for value in overlap_record["required_partition_ids"]
                ]
                record_partitions = [
                    str(value)
                    for value in overlap_record["required_partitions"]
                ]
                record_nodes = [
                    int(node) for node in overlap_record["global_node_ids"]
                ]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("overlap owner or node mapping is incomplete") from error
            if (
                record_regions != expected_regions
                or record_partition_ids != expected_partition_ids
                or record_partitions != expected_partitions
                or len(set(record_partition_ids)) != 2
                or record_nodes != list(road_set["member_nodes"])
                or record_nodes != sorted(set(record_nodes))
                or int(overlap_record.get("node_count", -1)) != len(record_nodes)
            ):
                raise ValueError("overlap owner or topology mapping has drifted")

            sidecar_filename = str(overlap_record.get("file", "")).strip()
            sidecar_sha256 = str(
                overlap_record.get("sha256", "")
            ).strip().lower()
            try:
                sidecar_bytes = int(overlap_record["bytes"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("overlap sidecar byte identity is invalid") from error
            if (
                not sidecar_filename
                or sidecar_bytes <= 0
                or len(sidecar_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in sidecar_sha256
                )
            ):
                raise ValueError("overlap sidecar file identity is incomplete")
            sidecar_path = self._manifest_child_path(sidecar_filename)
            if not sidecar_path.is_file():
                raise ValueError("overlap sidecar file is missing")
            if sidecar_path.stat().st_size != sidecar_bytes:
                raise ValueError("overlap sidecar size does not match manifest")
            # This integrity check is deliberately unconditional: --skip-sha
            # may relax a primary shard replay, never a mirrored overlap input.
            if _sha256_file(sidecar_path) != sidecar_sha256:
                raise ValueError("overlap sidecar SHA-256 does not match manifest")
            overlap_records_by_id[road_set_id] = {
                **dict(overlap_record),
                "resolved_path": sidecar_path,
            }

        if set(overlap_records_by_id) != set(catalog_by_id):
            raise ValueError("overlap records do not match topology road sets")

        self.source_dataset_sha256 = source_dataset_sha256
        self.model_artifact_sha256 = actual_rule_sha256
        self.overlap_subscriptions: List[Dict[str, Any]] = []
        for road_set_id in sorted(overlap_records_by_id):
            record_with_path = overlap_records_by_id[road_set_id]
            required_partition_ids = [
                int(value)
                for value in record_with_path["required_partition_ids"]
            ]
            if self.partition_id not in required_partition_ids:
                continue
            sidecar_path = Path(record_with_path["resolved_path"])
            with np.load(sidecar_path, allow_pickle=False) as sidecar:
                required_arrays = {
                    split_key,
                    "schema_version",
                    "kind",
                    "dataset",
                    "road_set_id",
                    "required_regions",
                    "required_partition_ids",
                    "required_partitions",
                    "global_node_ids",
                    "preprocess_version",
                    "source_dataset_sha256",
                    "source_dataset_bytes",
                    "model_artifact_sha256",
                    "topology_sha256",
                    "mean",
                    "std",
                    "feature_names",
                }
                missing = sorted(required_arrays - set(sidecar.files))
                if missing:
                    raise ValueError(
                        "missing overlap sidecar arrays: {}".format(missing)
                    )
                sidecar_nodes = [
                    int(node)
                    for node in np.asarray(sidecar["global_node_ids"]).tolist()
                ]
                sidecar_regions = [
                    str(value)
                    for value in np.asarray(sidecar["required_regions"]).tolist()
                ]
                sidecar_partition_ids = [
                    int(value)
                    for value in np.asarray(
                        sidecar["required_partition_ids"]
                    ).tolist()
                ]
                sidecar_partitions = [
                    str(value)
                    for value in np.asarray(
                        sidecar["required_partitions"]
                    ).tolist()
                ]
                if (
                    int(np.asarray(sidecar["schema_version"]).item())
                    != OVERLAP_SCHEMA_VERSION
                    or str(np.asarray(sidecar["kind"]).item())
                    != "canonical_shared_overlap_subscription"
                    or str(np.asarray(sidecar["dataset"]).item()) != "PEMS08"
                    or str(np.asarray(sidecar["road_set_id"]).item())
                    != road_set_id
                    or sidecar_regions
                    != list(record_with_path["required_regions"])
                    or sidecar_partition_ids != required_partition_ids
                    or sidecar_partitions
                    != list(record_with_path["required_partitions"])
                    or sidecar_nodes
                    != list(record_with_path["global_node_ids"])
                    or str(np.asarray(sidecar["preprocess_version"]).item())
                    != OVERLAP_PREPROCESS_VERSION
                    or str(
                        np.asarray(sidecar["source_dataset_sha256"]).item()
                    )
                    != source_dataset_sha256
                    or int(np.asarray(sidecar["source_dataset_bytes"]).item())
                    != source_dataset_bytes
                    or str(
                        np.asarray(sidecar["model_artifact_sha256"]).item()
                    )
                    != actual_rule_sha256
                    or str(np.asarray(sidecar["topology_sha256"]).item())
                    != actual_topology_sha256
                ):
                    raise ValueError("overlap sidecar identity has drifted")
                sidecar_mean = np.asarray(sidecar["mean"]).reshape(
                    1, -1, 1
                ).astype(np.float32)
                sidecar_std = np.asarray(sidecar["std"]).reshape(
                    1, -1, 1
                ).astype(np.float32)
                sidecar_feature_names = np.asarray(sidecar["feature_names"])
                if (
                    not np.array_equal(sidecar_mean, self.mean)
                    or not np.array_equal(sidecar_std, self.std)
                    or not np.array_equal(
                        sidecar_feature_names, self.feature_names
                    )
                ):
                    raise ValueError("overlap preprocessing arrays have drifted")
                sidecar_split_x = np.asarray(sidecar[split_key])
            if (
                sidecar_split_x.ndim != 4
                or sidecar_split_x.shape[0] != self.split_x.shape[0]
                or sidecar_split_x.shape[1] != len(sidecar_nodes)
                or sidecar_split_x.shape[2:] != self.split_x.shape[2:]
            ):
                raise ValueError("overlap sidecar observation shape has drifted")
            self.overlap_subscriptions.append(
                {
                    "road_set": catalog_by_id[road_set_id],
                    "global_node_ids": sidecar_nodes,
                    "split_x": sidecar_split_x,
                    "mean": sidecar_mean,
                    "std": sidecar_std,
                    "file": str(sidecar_path),
                    "sha256": str(record_with_path["sha256"]),
                }
            )

        expected_incident_ids = sorted(
            road_set_id
            for road_set_id, road_set in catalog_by_id.items()
            if "region_{}".format(self.partition_id)
            in road_set["required_regions"]
        )
        loaded_incident_ids = [
            str(subscription["road_set"]["road_set_id"])
            for subscription in self.overlap_subscriptions
        ]
        if loaded_incident_ids != expected_incident_ids:
            raise ValueError("overlap owner subscriptions are incomplete")
        self.load_latency_ms = round((time.perf_counter() - started) * 1000.0, 6)

    def _manifest_child_path(self, filename: str) -> Path:
        parent = self.manifest_path.parent.resolve()
        candidate = (parent / filename).resolve()
        try:
            candidate.relative_to(parent)
        except ValueError as error:
            raise ValueError("manifest file must remain inside shard directory") from error
        return candidate

    @property
    def overlap_road_set_ids(self) -> List[str]:
        return [
            str(subscription["road_set"]["road_set_id"])
            for subscription in self.overlap_subscriptions
        ]

    def canonical_overlap_observations(
        self, sample_id: int
    ) -> List[Dict[str, Any]]:
        """Return owner-independent observations from the shared sidecars."""
        sample_id = self.validate_sample_ids([sample_id])[0]
        observations = []
        for subscription in self.overlap_subscriptions:
            normalized = subscription["split_x"][sample_id].astype(
                np.float32, copy=False
            )
            raw_fragment = (
                normalized * subscription["std"] + subscription["mean"]
            )
            observations.append(
                build_canonical_overlap_observation(
                    subscription["road_set"],
                    raw_fragment,
                    subscription["global_node_ids"],
                    self.rule_config,
                    dataset="PEMS08",
                    split=self.split,
                    sample_id=sample_id,
                    source_dataset_sha256=self.source_dataset_sha256,
                    model_artifact_sha256=self.model_artifact_sha256,
                )
            )
        return observations

    def infer_sample(self, sample_id: int) -> TrafficPerceptionResult:
        sample_id = self.validate_sample_ids([sample_id])[0]
        started = time.perf_counter()
        normalized = self.split_x[sample_id].astype(np.float32, copy=False)
        raw_sample = normalized * self.std + self.mean
        state = current_window_risk(raw_sample, self.rule_config)
        overlap_observations = self.canonical_overlap_observations(sample_id)
        event = self._partition_event(
            sample_id=sample_id,
            partition_id=self.partition_id,
            managed_nodes=self.managed_node_ids,
            state=state,
            state_node_indices=list(range(len(self.managed_node_ids))),
            input_shape=normalized.shape,
            observation_steps=raw_sample.shape[-1],
            overlap_observations=overlap_observations,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        event["inference_latency_ms"] = round(elapsed_ms, 6)
        event["partition_data_preassigned"] = True
        return TrafficPerceptionResult(
            sample_id=sample_id,
            model_forward_ms=0.0,
            perception_ms=round(elapsed_ms, 6),
            events=[event],
        )
