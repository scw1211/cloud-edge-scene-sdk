"""训练与当前态感知合同一致的轻量 Student，并在原始 val/test 切分上评估。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier

from traffic_system.build_current_state_qwen_dataset import build_future_truth_event
from traffic_system.current_state_perception_runtime import (
    CurrentStateTrafficPerceptionRuntime,
)
from traffic_system.decision_utils import (
    DECISION_CLASSES,
    extract_feature_vector,
    rule_teacher_decision,
)
from traffic_system.edge_student import load_student_model, predict_student
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


def parse_hidden_dims(value: str) -> Tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("hidden_dims must contain positive comma-separated integers")
    return dims


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
        "--model_json",
        default="assets/models/edge_student_freeway_current_state_future_v1.json",
    )
    parser.add_argument(
        "--metrics_json",
        default="results/decision/edge_student_current_state_future_v1.json",
    )
    parser.add_argument(
        "--feature_cache",
        default="datasets/current_state_student_future_v1_features.npz",
    )
    parser.add_argument("--train_per_class", type=int, default=4000)
    parser.add_argument("--hidden_dims", default="64,32")
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--force_rebuild_cache", action="store_true")
    return parser.parse_args()


def _split_arrays(
    split: str,
    data_path: Path,
    labels_path: Path,
    current_config: Path,
    topology: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    runtime = CurrentStateTrafficPerceptionRuntime(
        data_path=data_path,
        rule_config_path=current_config,
        topology_path=topology,
        split=split,
        top_k=10,
    )
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
        perception = runtime.infer_sample(sample_id)
        for current_event in perception.events:
            partition_id = int(current_event["partition_id"])
            vector, names = extract_feature_vector(current_event)
            if feature_names and names != feature_names:
                raise ValueError("current-state feature schema changed within one split")
            feature_names = list(names)
            truth_event = build_future_truth_event(
                current_event,
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


def build_or_load_features(
    cache_path: Path,
    data_path: Path,
    labels_path: Path,
    current_config: Path,
    topology: Path,
    force: bool,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
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
                return arrays, list(metadata["feature_names"])

    arrays: Dict[str, np.ndarray] = {}
    feature_names: List[str] = []
    for split in ("train", "val", "test"):
        x, y, sample_ids, names = _split_arrays(
            split, data_path, labels_path, current_config, topology
        )
        if feature_names and names != feature_names:
            raise ValueError("feature schema differs across train/val/test")
        feature_names = names
        arrays["x_{}".format(split)] = x
        arrays["y_{}".format(split)] = y
        arrays["sample_{}".format(split)] = sample_ids
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "input_hashes": input_hashes,
        "feature_names": feature_names,
        "target": "future_observed_fcm_policy_action",
        "split_contract": "original PEMS08 train/val/test",
    }
    np.savez_compressed(
        cache_path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return arrays, feature_names


def balanced_indices(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    if per_class <= 0:
        return np.arange(y.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(len(DECISION_CLASSES)):
        candidates = np.flatnonzero(y == class_id)
        if candidates.size == 0:
            raise ValueError("training data has no {} targets".format(DECISION_CLASSES[class_id]))
        selected.append(
            rng.choice(candidates, size=per_class, replace=candidates.size < per_class)
        )
    indices = np.concatenate(selected).astype(np.int64)
    rng.shuffle(indices)
    return indices


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def manual_predict(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    layers: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    current = (x.astype(np.float64) - mean) / np.where(std < 1e-8, 1.0, std)
    for index, layer in enumerate(layers):
        current = current @ np.asarray(layer["weights"], dtype=np.float64)
        current += np.asarray(layer["bias"], dtype=np.float64)
        if index < len(layers) - 1:
            current = np.maximum(current, 0.0)
    probabilities = softmax(current)
    return np.argmax(probabilities, axis=1), probabilities


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray
) -> Dict[str, Any]:
    per_class: Dict[str, Any] = {}
    f1_values = []
    weighted_f1 = 0.0
    for class_id, name in enumerate(DECISION_CLASSES):
        true_mask = y_true == class_id
        pred_mask = y_pred == class_id
        support = int(np.sum(true_mask))
        correct = int(np.sum(true_mask & pred_mask))
        predicted = int(np.sum(pred_mask))
        precision = correct / predicted if predicted else 0.0
        recall = correct / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            f1_values.append(f1)
            weighted_f1 += support * f1
        per_class[name] = {
            "support": support,
            "predicted": predicted,
            "accuracy": round(correct / support, 6) if support else None,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    return {
        "count": int(y_true.size),
        "accuracy": round(float(np.mean(y_true == y_pred)), 6),
        "macro_f1": round(float(np.mean(f1_values)), 6),
        "weighted_f1": round(weighted_f1 / max(1, int(y_true.size)), 6),
        "mean_confidence": round(float(np.mean(np.max(probabilities, axis=1))), 6),
        "target_counts": {
            DECISION_CLASSES[index]: int(count)
            for index, count in enumerate(np.bincount(y_true, minlength=len(DECISION_CLASSES)))
        },
        "per_class": per_class,
    }


def grouped_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_ids: np.ndarray,
    seed: int,
    iterations: int = 2000,
) -> List[float]:
    unique = np.unique(sample_ids)
    rng = np.random.default_rng(seed)
    values = []
    group_indices = {group: np.flatnonzero(sample_ids == group) for group in unique}
    for _ in range(iterations):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        values.append(float(np.mean(y_true[indices] == y_pred[indices])))
    low, high = np.percentile(values, [2.5, 97.5])
    return [round(float(low), 6), round(float(high), 6)]


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data_npz)
    labels_path = resolve_path(args.risk_labels)
    current_config = resolve_path(args.current_state_config)
    topology = resolve_path(args.topology)
    model_path = resolve_path(args.model_json)
    metrics_path = resolve_path(args.metrics_json)
    cache_path = resolve_path(args.feature_cache)
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    arrays, feature_names = build_or_load_features(
        cache_path,
        data_path,
        labels_path,
        current_config,
        topology,
        args.force_rebuild_cache,
    )
    train_indices = balanced_indices(
        arrays["y_train"], args.train_per_class, args.seed
    )
    x_train_raw = arrays["x_train"][train_indices].astype(np.float64)
    y_train = arrays["y_train"][train_indices]
    mean = np.mean(x_train_raw, axis=0)
    std = np.std(x_train_raw, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x_train = (x_train_raw - mean) / std

    classifier = MLPClassifier(
        hidden_layer_sizes=hidden_dims,
        activation="relu",
        solver="adam",
        alpha=args.alpha,
        batch_size=args.batch_size,
        learning_rate_init=args.learning_rate,
        max_iter=args.max_iter,
        shuffle=True,
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-5,
    )
    started = time.perf_counter()
    classifier.fit(x_train, y_train)
    training_seconds = time.perf_counter() - started
    if list(classifier.classes_.astype(int)) != list(range(len(DECISION_CLASSES))):
        raise ValueError("trained classifier output classes are incomplete or reordered")

    layers = []
    for index, (weights, bias) in enumerate(zip(classifier.coefs_, classifier.intercepts_)):
        layers.append(
            {
                "type": "dense_softmax" if index == len(classifier.coefs_) - 1 else "dense_relu",
                "weights": np.asarray(weights, dtype=np.float64).round(10).tolist(),
                "bias": np.asarray(bias, dtype=np.float64).round(10).tolist(),
            }
        )

    metrics: Dict[str, Any] = {
        "task": "current_state_to_future_observed_policy_student_v1",
        "split_contract": "original PEMS08 train/val/test; no test data used in training",
        "train_selection": {
            "strategy": "balanced_by_action_class",
            "per_class": args.train_per_class,
            "selected_rows": int(train_indices.size),
            "source_rows": int(arrays["y_train"].size),
        },
        "training": {
            "implementation": "sklearn.neural_network.MLPClassifier",
            "hidden_dims": list(hidden_dims),
            "iterations": int(classifier.n_iter_),
            "training_seconds": round(training_seconds, 6),
            "loss": round(float(classifier.loss_), 8),
            "best_validation_score": round(float(classifier.best_validation_score_), 8),
            "seed": args.seed,
        },
        "inputs": {
            "data_sha256": file_sha256(data_path),
            "risk_labels_sha256": file_sha256(labels_path),
            "current_state_config_sha256": file_sha256(current_config),
            "topology_sha256": file_sha256(topology),
            "feature_cache": str(cache_path),
        },
    }
    predictions: Dict[str, np.ndarray] = {}
    probabilities: Dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        pred, probs = manual_predict(
            arrays["x_{}".format(split)], mean, std, layers
        )
        predictions[split] = pred
        probabilities[split] = probs
        metrics[split] = classification_metrics(
            arrays["y_{}".format(split)], pred, probs
        )
        if split in {"val", "test"}:
            metrics[split]["grouped_bootstrap_accuracy_95ci"] = grouped_accuracy_ci(
                arrays["y_{}".format(split)],
                pred,
                arrays["sample_{}".format(split)],
                args.seed + (1 if split == "val" else 2),
            )

    sklearn_probs = classifier.predict_proba(
        (arrays["x_test"].astype(np.float64) - mean) / std
    )
    metrics["export_equivalence"] = {
        "decision_match_rate": round(
            float(np.mean(np.argmax(sklearn_probs, axis=1) == predictions["test"])), 8
        ),
        "probability_max_abs_diff": float(
            np.max(np.abs(sklearn_probs - probabilities["test"]))
        ),
    }

    model = {
        "task": "edge_traffic_student_current_state_future_observed_policy_v1",
        "model_type": "numpy_mlp",
        "decision_classes": DECISION_CLASSES,
        "feature_names": feature_names,
        "normalization": {
            "mean": mean.round(10).tolist(),
            "std": std.round(10).tolist(),
        },
        "hidden_dims": list(hidden_dims),
        "layers": layers,
        "metrics": {
            "val_accuracy": metrics["val"]["accuracy"],
            "test_accuracy": metrics["test"]["accuracy"],
            "test_macro_f1": metrics["test"]["macro_f1"],
        },
        "provenance": metrics["inputs"],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 用真实事件覆盖“导出后可被生产 NumPy runtime 读取和执行”这一门禁。
    exported = load_student_model(model_path)
    runtime = CurrentStateTrafficPerceptionRuntime(
        data_path=data_path,
        rule_config_path=current_config,
        topology_path=topology,
        split="test",
        top_k=10,
    )
    event = runtime.infer_sample(0).events[0]
    for _ in range(20):
        predict_student(event, exported)
    latency_started = time.perf_counter()
    for _ in range(1000):
        predict_student(event, exported)
    metrics["runtime_gate"] = {
        "production_loader_and_predict_passed": True,
        "mean_latency_ms_1000_runs": round(
            (time.perf_counter() - latency_started) * 1000.0 / 1000.0, 6
        ),
        "model_size_bytes": model_path.stat().st_size,
        "feature_count": len(feature_names),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
