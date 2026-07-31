"""用途：从 PEMS08 路网边和区域事件生成可复现的区域邻接与边界拓扑。"""

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_regions(events_dir: Path) -> Dict[str, Set[int]]:
    regions: Dict[str, Set[int]] = {}
    for path in sorted(events_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as file_obj:
            event = json.load(file_obj)
        region_id = str(event.get("region_id", "")).strip()
        nodes = event.get("managed_node_ids", [])
        if not region_id or not isinstance(nodes, list) or not nodes:
            continue
        node_set = {int(node) for node in nodes}
        previous = regions.get(region_id)
        if previous is not None and previous != node_set:
            raise ValueError("managed nodes changed within region {}".format(region_id))
        regions[region_id] = node_set
    if len(regions) < 2:
        raise ValueError("at least two event regions are required")
    all_nodes = [node for nodes in regions.values() for node in nodes]
    if len(all_nodes) != len(set(all_nodes)):
        raise ValueError("core traffic regions must not overlap")
    return regions


def build_topology(events_dir: Path, adjacency_path: Path) -> Dict[str, Any]:
    regions = _load_regions(events_dir)
    node_region = {
        node: region_id for region_id, nodes in regions.items() for node in nodes
    }
    pair_edges: Dict[Tuple[str, str], List[List[int]]] = {}
    with adjacency_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if not reader.fieldnames or not {"from", "to"}.issubset(reader.fieldnames):
            raise ValueError("adjacency CSV must contain from and to columns")
        for row in reader:
            left = int(row["from"])
            right = int(row["to"])
            left_region = node_region.get(left)
            right_region = node_region.get(right)
            if left_region is None or right_region is None or left_region == right_region:
                continue
            pair = tuple(sorted((left_region, right_region)))
            oriented = [left, right] if pair[0] == left_region else [right, left]
            if oriented not in pair_edges.setdefault(pair, []):
                pair_edges[pair].append(oriented)

    neighbors: Dict[str, Set[str]] = {region_id: set() for region_id in regions}
    boundary_nodes: Dict[str, Set[int]] = {region_id: set() for region_id in regions}
    pairs = []
    for pair, edges in sorted(pair_edges.items()):
        left_region, right_region = pair
        neighbors[left_region].add(right_region)
        neighbors[right_region].add(left_region)
        left_nodes = sorted({edge[0] for edge in edges})
        right_nodes = sorted({edge[1] for edge in edges})
        boundary_nodes[left_region].update(left_nodes)
        boundary_nodes[right_region].update(right_nodes)
        pairs.append(
            {
                "left_region": left_region,
                "right_region": right_region,
                "cut_edge_count": len(edges),
                "left_boundary_nodes": left_nodes,
                "right_boundary_nodes": right_nodes,
            }
        )
    return {
        "schema_version": 1,
        "method": "road_graph_cut_edges",
        "adjacency_sha256": _sha256_file(adjacency_path),
        "region_count": len(regions),
        "region_neighbors": {
            region_id: sorted(values) for region_id, values in sorted(neighbors.items())
        },
        "region_boundary_nodes": {
            region_id: sorted(values)
            for region_id, values in sorted(boundary_nodes.items())
        },
        "region_pairs": pairs,
    }


def save_atomic(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output_path.name + ".", suffix=".tmp", dir=str(output_path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build traffic region topology.")
    parser.add_argument("--events_dir", default="datasets/freeway_events_joint_metis4")
    parser.add_argument("--adjacency", default="../data/PEMS08/PEMS08.csv")
    parser.add_argument("--output", default="models/traffic_region_topology_metis4.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_topology(Path(args.events_dir), Path(args.adjacency))
    save_atomic(payload, Path(args.output))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
