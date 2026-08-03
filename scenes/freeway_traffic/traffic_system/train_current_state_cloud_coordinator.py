"""训练与当前态感知合同一致的云端 ExtraTrees 协调器。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

from traffic_system.build_current_state_qwen_dataset import build_future_truth_event
from traffic_system.current_state_perception_runtime import (
    CurrentStateTrafficPerceptionRuntime,
)
from traffic_system.decision_utils import (
    DECISION_CLASSES,
    extract_feature_vector,
    rule_teacher_decision,
)
from traffic_system.risk_labels import enable_numpy_pickle_compatibility


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_npz",
        default="assets/downloads/PEMS08_r1_d0_w0_astcgn_multitask.npz",
    )
    parser.add_argument("--risk_labels", required=True)
    parser.add_argument(
        "--current_state_config",
        default="assets/models/current_state_perception_v1.json",
    )
    parser.add_argument(
        "--topology", default="assets/models/traffic_region_topology_metis4.json"
    )
    parser.add_argument(
        "--model",
        default="assets/models/cloud_coordinator_current_state_future_v1.joblib",
    )
    parser.add_argument(
        "--metrics",
        default="results/decision/cloud_coordinator_current_state_future_v1.json",
    )
    parser.add_argument(
        "--feature_cache",
        default="datasets/current_state_cloud_future_v1_features.npz",
    )
    parser.add_argument("--candidate_trees", type=int, default=100)
    parser.add_argument("--final_trees", type=int, default=50)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--force_rebuild_cache", action="store_true")
    return parser.parse_args()


def _fuse_neighbors(
    events: Sequence[Mapping[str, Any]],
    region_neighbors: Mapping[str, Sequence[str]],
) -> List[Dict[str, Any]]:
    by_region = {str(event["region_id"]): dict(event) for event in events}
    fused = []
    for event in events:
        payload = dict(event)
        neighbors = []
        for neighbor_region in region_neighbors.get(str(event["region_id"]), []):
            neighbor = by_region.get(str(neighbor_region))
            if neighbor is None:
                continue
            summary = neighbor["region_summary"]
            neighbors.append(
                {
                    "event_id": neighbor["event_id"],
                    "edge_id": neighbor["edge_id"],
                    "region_id": neighbor["region_id"],
                    "risk_level": summary["region_risk_level"],
                    "risk_score": summary["region_risk_score"],
                    "confidence": summary["region_risk_confidence"],
                }
            )
        payload["neighbor_context"] = [
            {"method": "road_graph_cut_edges", "neighbors": neighbors}
        ]
        fused.append(payload)
    return fused


def _extract_split(
    split: str,
    data_path: Path,
    labels_path: Path,
    current_config: Path,
    topology_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    runtime = CurrentStateTrafficPerceptionRuntime(
        data_path=data_path,
        rule_config_path=current_config,
        topology_path=topology_path,
        split=split,
        top_k=10,
    )
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    region_neighbors = topology.get("region_neighbors")
    if not isinstance(region_neighbors, dict):
        raise ValueError("traffic topology has no region_neighbors")
    enable_numpy_pickle_compatibility()
    with np.load(labels_path, allow_pickle=True) as labels:
        node_labels = labels["{}_node_label".format(split)]
        region_labels = labels["{}_region_label".format(split)]
        label_partitions = [
            [int(value) for value in part] for part in labels["partitions"].tolist()
        ]
    if label_partitions != runtime.partitions:
        raise ValueError("risk-label partitions differ from current-state partitions")

    features: List[List[float]] = []
    targets: List[int] = []
    sample_ids: List[int] = []
    feature_names: List[str] = []
    for sample_id in range(runtime.sample_count):
        current_events = _fuse_neighbors(
            runtime.infer_sample(sample_id).events, region_neighbors
        )
        for event in current_events:
            partition_id = int(event["partition_id"])
            vector, names = extract_feature_vector(event)
            if feature_names and feature_names != list(names):
                raise ValueError("current-state cloud feature schema changed")
            feature_names = list(names)
            truth_event = build_future_truth_event(
                event,
                node_labels[sample_id],
                int(region_labels[sample_id, partition_id]),
            )
            target = str(rule_teacher_decision(truth_event)["decision"])
            features.append(vector)
            targets.append(DECISION_CLASSES.index(target))
            sample_ids.append(sample_id)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.int64),
        np.asarray(sample_ids, dtype=np.int64),
        feature_names,
    )


def _build_or_load_features(
    cache_path: Path,
    data_path: Path,
    labels_path: Path,
    current_config: Path,
    topology: Path,
    force: bool,
) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, str]]:
    input_hashes = {
        "data": file_sha256(data_path),
        "labels": file_sha256(labels_path),
        "current_config": file_sha256(current_config),
        "topology": file_sha256(topology),
    }
    if cache_path.is_file() and not force:
        with np.load(cache_path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata_json"].item()))
            if metadata.get("input_hashes") == input_hashes:
                arrays = {
                    "x_{}".format(split): cached["x_{}".format(split)]
                    for split in ("train", "val", "test")
                }
                arrays.update(
                    {
                        "y_{}".format(split): cached["y_{}".format(split)]
                        for split in ("train", "val", "test")
                    }
                )
                arrays.update(
                    {
                        "sample_{}".format(split): cached[
                            "sample_{}".format(split)
                        ]
                        for split in ("train", "val", "test")
                    }
                )
                return arrays, list(metadata["feature_names"]), input_hashes

    arrays: Dict[str, np.ndarray] = {}
    feature_names: List[str] = []
    for split in ("train", "val", "test"):
        x, y, sample_ids, names = _extract_split(
            split, data_path, labels_path, current_config, topology
        )
        if feature_names and feature_names != names:
            raise ValueError("current-state cloud feature schema differs across splits")
        feature_names = names
        arrays["x_{}".format(split)] = x
        arrays["y_{}".format(split)] = y
        arrays["sample_{}".format(split)] = sample_ids
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        **arrays,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "schema_version": 1,
                    "input_hashes": input_hashes,
                    "feature_names": feature_names,
                },
                sort_keys=True,
            )
        ),
    )
    return arrays, feature_names, input_hashes


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        "count": int(y_true.size),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 6),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, average="weighted")), 6
        ),
    }


def _grouped_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_ids: np.ndarray,
    samples: int,
    seed: int,
) -> List[float]:
    groups = np.unique(sample_ids)
    indices = {group: np.flatnonzero(sample_ids == group) for group in groups}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(max(1, samples)):
        selected = rng.choice(groups, size=groups.size, replace=True)
        rows = np.concatenate([indices[group] for group in selected])
        values.append(float(np.mean(y_true[rows] == y_pred[rows])))
    low, high = np.percentile(values, [2.5, 97.5])
    return [round(float(low), 6), round(float(high), 6)]


def _make_model(
    trees: int, max_depth: Any, min_samples_leaf: int, seed: int
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=trees,
        max_features="sqrt",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data_npz)
    labels_path = resolve_path(args.risk_labels)
    current_config = resolve_path(args.current_state_config)
    topology = resolve_path(args.topology)
    model_path = resolve_path(args.model)
    metrics_path = resolve_path(args.metrics)
    cache_path = resolve_path(args.feature_cache)
    if args.candidate_trees <= 0 or args.final_trees <= 0:
        raise ValueError("tree counts must be positive")

    started = time.perf_counter()
    arrays, feature_names, input_hashes = _build_or_load_features(
        cache_path,
        data_path,
        labels_path,
        current_config,
        topology,
        args.force_rebuild_cache,
    )
    extraction_seconds = time.perf_counter() - started
    candidates = []
    for max_depth in (12, 16, None):
        for min_samples_leaf in (1, 2):
            model = _make_model(
                args.candidate_trees,
                max_depth,
                min_samples_leaf,
                args.seed,
            )
            model.fit(arrays["x_train"], arrays["y_train"])
            prediction = model.predict(arrays["x_val"])
            candidate = {
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                **_metrics(arrays["y_val"], prediction),
            }
            candidates.append(candidate)
            print("candidate", candidate, flush=True)
    selected = max(
        candidates,
        key=lambda item: (item["macro_f1"], item["accuracy"]),
    )
    x_train = np.concatenate([arrays["x_train"], arrays["x_val"]], axis=0)
    y_train = np.concatenate([arrays["y_train"], arrays["y_val"]], axis=0)
    final_model = _make_model(
        args.final_trees,
        selected["max_depth"],
        int(selected["min_samples_leaf"]),
        args.seed,
    )
    train_started = time.perf_counter()
    final_model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - train_started
    test_prediction = final_model.predict(arrays["x_test"])
    test_metrics = _metrics(arrays["y_test"], test_prediction)
    test_metrics["grouped_bootstrap_accuracy_95ci"] = _grouped_accuracy_ci(
        arrays["y_test"],
        test_prediction,
        arrays["sample_test"],
        args.bootstrap_samples,
        args.seed + 1,
    )

    payload = {
        "model": final_model,
        "feature_names": feature_names,
        "decision_classes": DECISION_CLASSES,
        "metadata": {
            "task": "current_state_future_observed_cloud_coordinator_v1",
            "input_contract": "current_state_risk_with_same_sample_topology",
            "training_splits": ["train", "val"],
            "evaluation_split": "test",
            "test_used_for_training_or_selection": False,
            "label_source": "future observed FCM labels -> fixed traffic policy",
            "selected_hyperparameters": selected,
            "candidate_trees": int(args.candidate_trees),
            "final_trees": int(args.final_trees),
            "input_hashes": input_hashes,
        },
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_path, compress=3)
    result = {
        "task": "current_state_future_observed_cloud_coordinator_v1",
        "split_contract": "train for candidates; val for selection; train+val final fit; test once",
        "sample_counts": {
            split: int(arrays["y_{}".format(split)].size)
            for split in ("train", "val", "test")
        },
        "feature_count": len(feature_names),
        "candidates": candidates,
        "selected": selected,
        "candidate_trees": int(args.candidate_trees),
        "final_trees": int(args.final_trees),
        "test": test_metrics,
        "runtime": {
            "feature_extraction_seconds": round(extraction_seconds, 6),
            "final_training_seconds": round(training_seconds, 6),
        },
        "artifact": {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": file_sha256(model_path),
        },
        "inputs": input_hashes,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
