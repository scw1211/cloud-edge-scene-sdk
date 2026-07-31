"""用途：根据道路邻接图执行 METIS 等图分区并计算区域边界关系。"""

from collections import deque
from typing import Deque, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


def contiguous_partitions(num_nodes: int, num_partitions: int) -> List[List[int]]:
    """Split node ids by index. Kept only as a baseline/debug option."""
    validate_partition_args(num_nodes, num_partitions)
    partitions = []
    base_size = num_nodes // num_partitions
    remainder = num_nodes % num_partitions
    for partition_id in range(num_partitions):
        start = partition_id * base_size + min(partition_id, remainder)
        size = base_size + (1 if partition_id < remainder else 0)
        partitions.append(list(range(start, start + size)))
    return partitions


def validate_partition_args(num_nodes: int, num_partitions: int) -> None:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive.")
    if num_partitions > num_nodes:
        raise ValueError("num_partitions cannot exceed num_nodes.")


def build_undirected_neighbor_map(adj_mx: np.ndarray, num_nodes: int) -> Dict[int, List[int]]:
    """Convert a directed/weighted adjacency matrix to an undirected neighbor map."""
    if adj_mx is None:
        raise ValueError("adj_mx is required for graph partitioning.")
    adj_np = np.asarray(adj_mx)
    if adj_np.ndim != 2 or adj_np.shape[0] < num_nodes or adj_np.shape[1] < num_nodes:
        raise ValueError("adj_mx must be a square matrix covering all nodes.")

    adj_bool = (adj_np[:num_nodes, :num_nodes] != 0) | (adj_np[:num_nodes, :num_nodes].T != 0)
    neighbor_map: Dict[int, List[int]] = {}
    for node_id in range(num_nodes):
        neighbors = np.where(adj_bool[node_id])[0].astype(int).tolist()
        neighbor_map[node_id] = sorted(node for node in neighbors if node != node_id)
    return neighbor_map


def build_undirected_adjacency(adj_mx: np.ndarray, num_nodes: int) -> np.ndarray:
    """Return a binary undirected adjacency matrix."""
    if adj_mx is None:
        raise ValueError("adj_mx is required for graph partitioning.")
    adj_np = np.asarray(adj_mx)
    if adj_np.ndim != 2 or adj_np.shape[0] < num_nodes or adj_np.shape[1] < num_nodes:
        raise ValueError("adj_mx must be a square matrix covering all nodes.")
    adj_bool = (adj_np[:num_nodes, :num_nodes] != 0) | (adj_np[:num_nodes, :num_nodes].T != 0)
    np.fill_diagonal(adj_bool, False)
    return adj_bool.astype(np.float64)


def bfs_distances(neighbor_map: Dict[int, List[int]], sources: Iterable[int]) -> List[float]:
    """Shortest unweighted graph distance from any source."""
    num_nodes = len(neighbor_map)
    dist = [float("inf")] * num_nodes
    queue: Deque[int] = deque()
    for source in sources:
        source = int(source)
        if 0 <= source < num_nodes and dist[source] == float("inf"):
            dist[source] = 0.0
            queue.append(source)

    while queue:
        node = queue.popleft()
        for neighbor in neighbor_map[node]:
            if dist[neighbor] != float("inf"):
                continue
            dist[neighbor] = dist[node] + 1.0
            queue.append(neighbor)
    return dist


def choose_farthest_seeds(neighbor_map: Dict[int, List[int]], num_partitions: int) -> List[int]:
    """Choose deterministic, well-spread graph seeds for balanced BFS partitioning."""
    degrees = {node: len(neighbors) for node, neighbors in neighbor_map.items()}
    first_seed = max(degrees, key=lambda node: (degrees[node], -node))
    seeds = [int(first_seed)]

    while len(seeds) < num_partitions:
        dist = bfs_distances(neighbor_map, seeds)
        candidates = [node for node in neighbor_map if node not in seeds]
        next_seed = max(
            candidates,
            key=lambda node: (
                dist[node] if dist[node] != float("inf") else 10**9,
                degrees[node],
                -node,
            ),
        )
        seeds.append(int(next_seed))
    return seeds


def target_partition_sizes(num_nodes: int, num_partitions: int) -> List[int]:
    base_size = num_nodes // num_partitions
    remainder = num_nodes % num_partitions
    return [base_size + (1 if pid < remainder else 0) for pid in range(num_partitions)]


def assign_node(
    node: int,
    partition_id: int,
    partitions: List[List[int]],
    assigned: List[int],
    unassigned: Set[int],
    queues: List[deque],
) -> None:
    partitions[partition_id].append(int(node))
    assigned[node] = partition_id
    unassigned.remove(node)
    queues[partition_id].append(int(node))


def grow_balanced_bfs_partitions(
    neighbor_map: Dict[int, List[int]],
    seeds: Sequence[int],
) -> List[List[int]]:
    """Grow balanced graph partitions from seed nodes using BFS frontiers."""
    num_nodes = len(neighbor_map)
    num_partitions = len(seeds)
    targets = target_partition_sizes(num_nodes, num_partitions)
    degrees = {node: len(neighbors) for node, neighbors in neighbor_map.items()}
    seed_distances = [bfs_distances(neighbor_map, [seed]) for seed in seeds]

    partitions: List[List[int]] = [[] for _ in range(num_partitions)]
    assigned = [-1] * num_nodes
    unassigned: Set[int] = set(range(num_nodes))
    queues: List[deque] = [deque() for _ in range(num_partitions)]

    for pid, seed in enumerate(seeds):
        assign_node(int(seed), pid, partitions, assigned, unassigned, queues)

    while unassigned:
        progress = False
        order = sorted(
            range(num_partitions),
            key=lambda pid: (len(partitions[pid]) / float(targets[pid]), pid),
        )

        for pid in order:
            if len(partitions[pid]) >= targets[pid]:
                continue
            while queues[pid] and len(partitions[pid]) < targets[pid]:
                node = queues[pid].popleft()
                candidates = [
                    neighbor for neighbor in neighbor_map[node] if neighbor in unassigned
                ]
                candidates.sort(key=lambda item: (-degrees[item], item))
                for neighbor in candidates:
                    if len(partitions[pid]) >= targets[pid]:
                        break
                    assign_node(neighbor, pid, partitions, assigned, unassigned, queues)
                    progress = True

        if progress:
            continue

        # If a frontier is exhausted, attach the nearest remaining node to the
        # most underfilled partition. This keeps the algorithm robust on sparse graphs.
        candidate_moves = []
        for pid in range(num_partitions):
            if len(partitions[pid]) >= targets[pid]:
                continue
            partition_set = set(partitions[pid])
            for node in unassigned:
                touches = sum(1 for neighbor in neighbor_map[node] if neighbor in partition_set)
                dist = seed_distances[pid][node]
                if dist == float("inf"):
                    dist = 10**9
                candidate_moves.append(
                    (
                        0 if touches > 0 else 1,
                        dist,
                        -touches,
                        -degrees[node],
                        pid,
                        node,
                    )
                )
        if not candidate_moves:
            break
        _, _, _, _, best_pid, best_node = min(candidate_moves)
        assign_node(best_node, best_pid, partitions, assigned, unassigned, queues)

    return [sorted(nodes) for nodes in partitions]


def spectral_split_nodes(adj_undirected: np.ndarray, nodes: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Split a node set by the Fiedler vector of its induced graph Laplacian."""
    node_list = sorted(int(node) for node in nodes)
    n = len(node_list)
    if n < 2:
        return node_list, []
    if n == 2:
        return [node_list[0]], [node_list[1]]

    sub_adj = adj_undirected[np.ix_(node_list, node_list)]
    degrees = sub_adj.sum(axis=1)

    # If the induced subgraph is disconnected, split by connected components first.
    if np.any(degrees == 0):
        return split_by_components_or_ids(sub_adj, node_list)

    laplacian = np.diag(degrees) - sub_adj
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    except np.linalg.LinAlgError:
        return split_by_ids(node_list)

    if eigenvectors.shape[1] < 2:
        return split_by_ids(node_list)
    fiedler = eigenvectors[:, 1]
    order = sorted(range(n), key=lambda idx: (float(fiedler[idx]), node_list[idx]))
    midpoint = n // 2
    left = sorted(node_list[idx] for idx in order[:midpoint])
    right = sorted(node_list[idx] for idx in order[midpoint:])
    if not left or not right:
        return split_by_ids(node_list)
    return left, right


def split_by_ids(nodes: Sequence[int]) -> Tuple[List[int], List[int]]:
    node_list = sorted(int(node) for node in nodes)
    midpoint = max(1, len(node_list) // 2)
    return node_list[:midpoint], node_list[midpoint:]


def split_by_components_or_ids(sub_adj: np.ndarray, node_list: Sequence[int]) -> Tuple[List[int], List[int]]:
    components = []
    visited = set()
    for local_id in range(len(node_list)):
        if local_id in visited:
            continue
        stack = [local_id]
        visited.add(local_id)
        component = []
        while stack:
            current = stack.pop()
            component.append(int(node_list[current]))
            for neighbor in np.where(sub_adj[current] != 0)[0].astype(int).tolist():
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))

    if len(components) < 2:
        return split_by_ids(node_list)
    components.sort(key=lambda part: (-len(part), part[0]))
    left: List[int] = []
    right: List[int] = []
    for component in components:
        target = left if len(left) <= len(right) else right
        target.extend(component)
    if not left or not right:
        return split_by_ids(node_list)
    return sorted(left), sorted(right)


def spectral_partitions(
    adj_mx: np.ndarray,
    num_nodes: int,
    num_partitions: int,
    overlap_hops: int = 0,
) -> List[List[int]]:
    """Recursive spectral bisection partitioning over the road-network graph."""
    validate_partition_args(num_nodes, num_partitions)
    adj_undirected = build_undirected_adjacency(adj_mx, num_nodes)
    partitions: List[List[int]] = [list(range(num_nodes))]

    while len(partitions) < num_partitions:
        split_index = max(range(len(partitions)), key=lambda idx: len(partitions[idx]))
        current = partitions.pop(split_index)
        left, right = spectral_split_nodes(adj_undirected, current)
        if not left or not right:
            left, right = split_by_ids(current)
        partitions.append(left)
        partitions.append(right)

    partitions = sorted((sorted(partition) for partition in partitions), key=lambda part: (part[0], len(part)))
    neighbor_map = build_undirected_neighbor_map(adj_mx, num_nodes)
    return add_overlap_hops(partitions, neighbor_map, overlap_hops)


def metis_partitions(
    adj_mx: np.ndarray,
    num_nodes: int,
    num_partitions: int,
    overlap_hops: int = 0,
) -> List[List[int]]:
    """METIS multilevel k-way graph partitioning via pymetis."""
    validate_partition_args(num_nodes, num_partitions)
    try:
        import pymetis
    except ImportError as exc:
        raise RuntimeError(
            "partition_method=metis requires pymetis. Install it with: "
            "conda run -n traffic python -m pip install pymetis"
        ) from exc

    neighbor_map = build_undirected_neighbor_map(adj_mx, num_nodes)
    adjacency = [neighbor_map[node] for node in range(num_nodes)]
    _, membership = pymetis.part_graph(num_partitions, adjacency=adjacency)
    partitions: List[List[int]] = [[] for _ in range(num_partitions)]
    for node_id, partition_id in enumerate(membership):
        partitions[int(partition_id)].append(int(node_id))
    partitions = [sorted(nodes) for nodes in partitions]
    return add_overlap_hops(partitions, neighbor_map, overlap_hops)


def add_overlap_hops(
    partitions: Sequence[Sequence[int]],
    neighbor_map: Dict[int, List[int]],
    overlap_hops: int,
) -> List[List[int]]:
    """Expand each core partition with k-hop boundary nodes from adjacent partitions."""
    if overlap_hops <= 0:
        return [sorted(int(node) for node in nodes) for nodes in partitions]

    expanded = []
    for nodes in partitions:
        visited = set(int(node) for node in nodes)
        frontier = set(visited)
        for _ in range(overlap_hops):
            next_frontier = set()
            for node in frontier:
                for neighbor in neighbor_map[node]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        expanded.append(sorted(visited))
    return expanded


def graph_partitions(
    adj_mx: np.ndarray,
    num_nodes: int,
    num_partitions: int,
    overlap_hops: int = 0,
) -> List[List[int]]:
    """Partition a traffic graph into balanced topology-aware regions."""
    validate_partition_args(num_nodes, num_partitions)
    neighbor_map = build_undirected_neighbor_map(adj_mx, num_nodes)
    seeds = choose_farthest_seeds(neighbor_map, num_partitions)
    partitions = grow_balanced_bfs_partitions(neighbor_map, seeds)
    return add_overlap_hops(partitions, neighbor_map, overlap_hops)


def partition_graph(
    adj_mx: np.ndarray,
    num_nodes: int,
    num_partitions: int,
    method: str,
    overlap_hops: int = 0,
) -> List[List[int]]:
    """Dispatch supported partitioning methods."""
    if method == "contiguous":
        return contiguous_partitions(num_nodes, num_partitions)
    if method in ("graph", "graph_bfs"):
        return graph_partitions(adj_mx, num_nodes, num_partitions, overlap_hops)
    if method == "spectral":
        return spectral_partitions(adj_mx, num_nodes, num_partitions, overlap_hops)
    if method == "metis":
        return metis_partitions(adj_mx, num_nodes, num_partitions, overlap_hops)
    raise ValueError("Unsupported partition method: {}".format(method))


def summarize_partitions(adj_mx: np.ndarray, partitions: Sequence[Sequence[int]]) -> Dict[str, object]:
    """Return simple topology quality stats for partition sanity checks."""
    num_nodes = int(np.asarray(adj_mx).shape[0])
    neighbor_map = build_undirected_neighbor_map(adj_mx, num_nodes)
    node_to_parts: Dict[int, Set[int]] = {}
    for pid, nodes in enumerate(partitions):
        for node in nodes:
            node_to_parts.setdefault(int(node), set()).add(pid)

    cut_edges = 0
    internal_edges = 0
    seen_edges = set()
    for node, neighbors in neighbor_map.items():
        for neighbor in neighbors:
            edge = tuple(sorted((node, neighbor)))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            node_parts = node_to_parts.get(node, set())
            neighbor_parts = node_to_parts.get(neighbor, set())
            if node_parts & neighbor_parts:
                internal_edges += 1
            else:
                cut_edges += 1

    sizes = [len(nodes) for nodes in partitions]
    duplicate_nodes = sum(1 for parts in node_to_parts.values() if len(parts) > 1)
    component_counts = [
        count_partition_components(neighbor_map, [int(node) for node in nodes])
        for nodes in partitions
    ]
    boundary_nodes = count_boundary_nodes(neighbor_map, node_to_parts)
    return {
        "num_partitions": len(partitions),
        "partition_sizes": sizes,
        "min_size": min(sizes) if sizes else 0,
        "max_size": max(sizes) if sizes else 0,
        "balance_ratio": round((max(sizes) / max(1, min(sizes))) if sizes else 0.0, 4),
        "duplicate_nodes": duplicate_nodes,
        "component_counts": component_counts,
        "connected_partition_count": sum(1 for count in component_counts if count == 1),
        "boundary_node_count": boundary_nodes,
        "internal_edges": internal_edges,
        "cut_edges": cut_edges,
        "cut_edge_ratio": round(cut_edges / max(1, internal_edges + cut_edges), 4),
    }


def count_partition_components(
    neighbor_map: Dict[int, List[int]],
    nodes: Sequence[int],
) -> int:
    node_set = set(int(node) for node in nodes)
    if not node_set:
        return 0
    visited = set()
    components = 0
    for node in sorted(node_set):
        if node in visited:
            continue
        components += 1
        stack = [node]
        visited.add(node)
        while stack:
            current = stack.pop()
            for neighbor in neighbor_map[current]:
                if neighbor not in node_set or neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
    return components


def count_boundary_nodes(
    neighbor_map: Dict[int, List[int]],
    node_to_parts: Dict[int, Set[int]],
) -> int:
    boundary_nodes = set()
    for node, node_parts in node_to_parts.items():
        for neighbor in neighbor_map[node]:
            neighbor_parts = node_to_parts.get(neighbor, set())
            if not node_parts & neighbor_parts:
                boundary_nodes.add(node)
                boundary_nodes.add(neighbor)
    return len(boundary_nodes)
