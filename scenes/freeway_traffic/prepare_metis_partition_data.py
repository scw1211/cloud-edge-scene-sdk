#!/usr/bin/env python3
"""离线把 PEMS08 按冻结的 METIS 映射切成四份边缘输入。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from traffic_system.overlap_observations import (
    OVERLAP_PREPROCESS_VERSION,
    OVERLAP_SCHEMA_VERSION,
)
from traffic_system.road_sets import build_road_set_catalog


SCENE_ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_partition_data(
    source_path: Path,
    rule_config_path: Path,
    topology_path: Path,
    output_directory: Path,
) -> Dict[str, Any]:
    source_path = Path(source_path).resolve()
    rule_config_path = Path(rule_config_path).resolve()
    topology_path = Path(topology_path).resolve()
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    rule_config = json.loads(rule_config_path.read_text(encoding="utf-8"))
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    road_set_catalog = build_road_set_catalog(topology)
    partitions = [
        [int(node) for node in partition]
        for partition in rule_config["partitions"]
    ]
    flattened = sorted(node for partition in partitions for node in partition)
    if flattened != list(range(len(flattened))):
        raise ValueError("METIS partitions must cover every node exactly once")
    region_to_partition = {
        "region_{}".format(partition_id): partition_id
        for partition_id in range(len(partitions))
    }
    partition_sets = [set(nodes) for nodes in partitions]
    source_sha256 = _sha256(source_path)
    source_bytes = source_path.stat().st_size
    rule_config_sha256 = _sha256(rule_config_path)
    topology_sha256 = _sha256(topology_path)

    records = []
    overlap_records = []
    with np.load(source_path, allow_pickle=False) as source:
        required = {
            "train_x",
            "val_x",
            "test_x",
            "mean",
            "std",
            "feature_names",
        }
        missing = sorted(required - set(source.files))
        if missing:
            raise ValueError("source PEMS08 arrays are missing: {}".format(missing))
        for split in ("train", "val", "test"):
            if source["{}_x".format(split)].shape[1] != len(flattened):
                raise ValueError("{} node count does not match METIS map".format(split))

        for partition_id, managed_nodes in enumerate(partitions):
            node_ids = np.asarray(managed_nodes, dtype=np.int64)
            filename = "pems08_metis4_edge_node_{}.npz".format(partition_id)
            final_path = output_directory / filename
            temporary = output_directory / (
                ".{}.tmp.{}.npz".format(filename, os.getpid())
            )
            payload = {
                "schema_version": np.asarray(1, dtype=np.int64),
                "partition_id": np.asarray(partition_id, dtype=np.int64),
                "num_partitions": np.asarray(len(partitions), dtype=np.int64),
                "global_node_ids": node_ids,
                "mean": np.asarray(source["mean"]),
                "std": np.asarray(source["std"]),
                "feature_names": np.asarray(source["feature_names"]),
            }
            for split in ("train", "val", "test"):
                payload["{}_x".format(split)] = np.asarray(
                    source["{}_x".format(split)][:, node_ids, :, :]
                )
                timestamp_key = "{}_timestamp".format(split)
                if timestamp_key in source.files:
                    payload[timestamp_key] = np.asarray(source[timestamp_key])
            try:
                np.savez_compressed(temporary, **payload)
                os.replace(str(temporary), str(final_path))
            finally:
                if temporary.exists():
                    temporary.unlink()
            records.append(
                {
                    "partition_id": partition_id,
                    "edge_id": "edge_node_{}".format(partition_id),
                    "region_id": "region_{}".format(partition_id),
                    "node_count": len(managed_nodes),
                    "global_node_ids": managed_nodes,
                    "file": filename,
                    "bytes": final_path.stat().st_size,
                    "sha256": _sha256(final_path),
                }
            )

        for road_set in road_set_catalog["road_sets"]:
            road_set_id = str(road_set["road_set_id"])
            required_regions = [
                str(region) for region in road_set["required_regions"]
            ]
            if len(required_regions) != 2 or any(
                region not in region_to_partition
                for region in required_regions
            ):
                raise ValueError(
                    "road set {} has an invalid two-owner region mapping".format(
                        road_set_id
                    )
                )
            required_partition_ids = sorted(
                region_to_partition[region] for region in required_regions
            )
            required_partitions = [
                "edge_node_{}".format(partition_id)
                for partition_id in required_partition_ids
            ]
            if len(set(required_partition_ids)) != 2:
                raise ValueError(
                    "road set {} must map to two distinct partitions".format(
                        road_set_id
                    )
                )
            for region in required_regions:
                partition_id = region_to_partition[region]
                declared_nodes = {
                    int(node)
                    for node in road_set["member_nodes_by_region"][region]
                }
                if not declared_nodes or not declared_nodes.issubset(
                    partition_sets[partition_id]
                ):
                    raise ValueError(
                        "road set {} topology nodes do not belong to {}".format(
                            road_set_id, region
                        )
                    )

            global_node_ids = [
                int(node) for node in road_set["member_nodes"]
            ]
            if global_node_ids != sorted(set(global_node_ids)):
                raise ValueError(
                    "road set {} nodes must be a sorted unique union".format(
                        road_set_id
                    )
                )
            node_ids = np.asarray(global_node_ids, dtype=np.int64)
            filename = "pems08_metis4_overlap_{}.npz".format(road_set_id)
            final_path = output_directory / filename
            temporary = output_directory / (
                ".{}.tmp.{}.npz".format(filename, os.getpid())
            )
            payload = {
                "schema_version": np.asarray(
                    OVERLAP_SCHEMA_VERSION, dtype=np.int64
                ),
                "kind": np.asarray(
                    "canonical_shared_overlap_subscription"
                ),
                "dataset": np.asarray("PEMS08"),
                "road_set_id": np.asarray(road_set_id),
                "required_regions": np.asarray(required_regions),
                "required_partition_ids": np.asarray(
                    required_partition_ids, dtype=np.int64
                ),
                "required_partitions": np.asarray(required_partitions),
                "global_node_ids": node_ids,
                "preprocess_version": np.asarray(
                    OVERLAP_PREPROCESS_VERSION
                ),
                "source_dataset_sha256": np.asarray(source_sha256),
                "source_dataset_bytes": np.asarray(
                    source_bytes, dtype=np.int64
                ),
                "model_artifact_sha256": np.asarray(
                    rule_config_sha256
                ),
                "topology_sha256": np.asarray(topology_sha256),
                "mean": np.asarray(source["mean"]),
                "std": np.asarray(source["std"]),
                "feature_names": np.asarray(source["feature_names"]),
            }
            for split in ("train", "val", "test"):
                payload["{}_x".format(split)] = np.asarray(
                    source["{}_x".format(split)][:, node_ids, :, :]
                )
                timestamp_key = "{}_timestamp".format(split)
                if timestamp_key in source.files:
                    payload[timestamp_key] = np.asarray(source[timestamp_key])
            try:
                np.savez_compressed(temporary, **payload)
                os.replace(str(temporary), str(final_path))
            finally:
                if temporary.exists():
                    temporary.unlink()
            overlap_records.append(
                {
                    "road_set_id": road_set_id,
                    "required_regions": required_regions,
                    "required_partition_ids": required_partition_ids,
                    "required_partitions": required_partitions,
                    "global_node_ids": global_node_ids,
                    "node_count": len(global_node_ids),
                    "file": filename,
                    "bytes": final_path.stat().st_size,
                    "sha256": _sha256(final_path),
                }
            )

    manifest = {
        "schema_version": 2,
        "format": "pems08_preassigned_metis_edge_partitions",
        "source": {
            "file": str(source_path),
            "bytes": source_bytes,
            "sha256": source_sha256,
        },
        "partition_contract": {
            "method": "frozen_metis",
            "partition_count": len(partitions),
            "rule_config_sha256": rule_config_sha256,
            "topology_sha256": topology_sha256,
            "runtime_repartition_allowed": False,
        },
        "overlap_contract": {
            "schema_version": OVERLAP_SCHEMA_VERSION,
            "kind": "canonical_shared_overlap_subscription",
            "dataset": "PEMS08",
            "subscription_count": len(overlap_records),
            "owners_per_subscription": 2,
            "preprocess_version": OVERLAP_PREPROCESS_VERSION,
            "source_dataset_sha256": source_sha256,
            "model_artifact_sha256": rule_config_sha256,
            "topology_sha256": topology_sha256,
            "primary_sensor_ownership_changed": False,
            "sidecar_nodes_are_managed_nodes": False,
        },
        "partitions": records,
        "road_sets": overlap_records,
    }
    manifest_path = output_directory / "manifest.json"
    temporary_manifest = output_directory / (
        ".manifest.json.tmp.{}".format(os.getpid())
    )
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary_manifest), str(manifest_path))
    return {**manifest, "manifest_path": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="提前把 PEMS08 切成四个固定 METIS 边缘分区"
    )
    parser.add_argument(
        "--source",
        default=str(
            SCENE_ROOT
            / "assets"
            / "downloads"
            / "PEMS08_r1_d0_w0_astcgn_multitask.npz"
        ),
    )
    parser.add_argument(
        "--rule-config",
        default=str(
            SCENE_ROOT / "assets" / "models" / "current_state_perception_v1.json"
        ),
    )
    parser.add_argument(
        "--topology",
        default=str(
            SCENE_ROOT / "assets" / "models" / "traffic_region_topology_metis4.json"
        ),
    )
    parser.add_argument(
        "--output-directory",
        default=str(SCENE_ROOT / "runtime" / "pems08_metis4_partitions"),
    )
    args = parser.parse_args()
    result = prepare_partition_data(
        Path(args.source),
        Path(args.rule_config),
        Path(args.topology),
        Path(args.output_directory),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
