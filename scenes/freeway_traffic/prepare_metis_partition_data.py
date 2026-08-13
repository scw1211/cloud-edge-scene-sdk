#!/usr/bin/env python3
"""离线把 PEMS08 按冻结的 METIS 映射切成四份边缘输入。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


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
    partitions = [
        [int(node) for node in partition]
        for partition in rule_config["partitions"]
    ]
    flattened = sorted(node for partition in partitions for node in partition)
    if flattened != list(range(len(flattened))):
        raise ValueError("METIS partitions must cover every node exactly once")

    records = []
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

    manifest = {
        "schema_version": 1,
        "format": "pems08_preassigned_metis_edge_partitions",
        "source": {
            "file": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": _sha256(source_path),
        },
        "partition_contract": {
            "method": "frozen_metis",
            "partition_count": len(partitions),
            "rule_config_sha256": _sha256(rule_config_path),
            "topology_sha256": _sha256(topology_path),
            "runtime_repartition_allowed": False,
        },
        "partitions": records,
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
