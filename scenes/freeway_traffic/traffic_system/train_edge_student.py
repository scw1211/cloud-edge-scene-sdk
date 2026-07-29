"""用途：使用 Teacher 决策标签训练边缘实时 MLP Student。"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from traffic_system.decision_utils import DECISION_CLASSES, labels_from_records, read_jsonl, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny edge-side student model from teacher traffic decisions."
    )
    parser.add_argument("--labels", default="datasets/freeway_teacher_labels_qwen9b_safe_joint_metis4.jsonl")
    parser.add_argument("--model_json", default="models/edge_student_freeway_joint_metis4.json")
    parser.add_argument("--metrics_json", default="results/decision/edge_student_freeway_joint_metis4.json")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--learning_rate", type=float, default=0.08)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--model_type", default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--hidden_dim", type=int, default=24)
    parser.add_argument(
        "--hidden_dims",
        default="",
        help="Comma-separated hidden dimensions for MLP, e.g. 128,64. Overrides hidden_dim.",
    )
    parser.add_argument("--test_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_strategy",
        default="temporal_group",
        choices=["temporal_group", "stratified_random"],
        help="temporal_group keeps all regions from one timestamp in the same split.",
    )
    return parser.parse_args()


def parse_hidden_dims(hidden_dims: str, hidden_dim: int) -> List[int]:
    if hidden_dims.strip():
        dims = [int(part.strip()) for part in hidden_dims.split(",") if part.strip()]
    else:
        dims = [int(hidden_dim)]
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("All hidden dimensions must be positive.")
    return dims


def split_indices(labels: np.ndarray, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    n = labels.shape[0]
    if n < 2:
        return np.arange(n), np.arange(n)

    rng = random.Random(seed)
    train_indices: List[int] = []
    test_indices: List[int] = []
    for cls in sorted(set(labels.tolist())):
        cls_indices = [int(idx) for idx, value in enumerate(labels.tolist()) if value == cls]
        rng.shuffle(cls_indices)
        if len(cls_indices) == 1:
            train_indices.extend(cls_indices)
            continue
        test_count = max(1, int(round(len(cls_indices) * test_ratio)))
        test_indices.extend(cls_indices[:test_count])
        train_indices.extend(cls_indices[test_count:])

    if not test_indices:
        all_indices = list(range(n))
        rng.shuffle(all_indices)
        test_count = max(1, int(round(n * test_ratio)))
        test_indices = all_indices[:test_count]
        train_indices = all_indices[test_count:] or all_indices[:]

    return np.asarray(sorted(train_indices), dtype=np.int64), np.asarray(sorted(test_indices), dtype=np.int64)


def split_indices_temporal_group(
    event_ids: Sequence[str],
    test_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    groups: Dict[int, List[int]] = {}
    for index, event_id in enumerate(event_ids):
        match = re.search(r"sample_(\d+)", str(event_id))
        if not match:
            raise ValueError("event_id does not contain a sample timestamp: {}".format(event_id))
        groups.setdefault(int(match.group(1)), []).append(index)
    ordered_samples = sorted(groups)
    if len(ordered_samples) < 2:
        raise ValueError("Need at least two sample groups for temporal evaluation.")
    test_group_count = max(1, int(round(len(ordered_samples) * test_ratio)))
    test_samples = set(ordered_samples[-test_group_count:])
    train_indices = [index for sample in ordered_samples if sample not in test_samples for index in groups[sample]]
    test_indices = [index for sample in ordered_samples if sample in test_samples for index in groups[sample]]
    return np.asarray(train_indices, dtype=np.int64), np.asarray(test_indices, dtype=np.int64)


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    y[np.arange(labels.shape[0]), labels] = 1.0
    return y


def class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = labels.shape[0] / (num_classes * counts)
    weights = weights / weights.mean()
    return weights


def train_softmax_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    rng = np.random.default_rng(42)
    weights = rng.normal(0.0, 0.02, size=(x_train.shape[1], num_classes))
    bias = np.zeros((num_classes,), dtype=np.float64)
    y_oh = one_hot(y_train, num_classes)
    cls_weights = class_weights(y_train, num_classes)
    sample_weights = cls_weights[y_train]
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        logits = x_train @ weights + bias
        probs = softmax(logits)
        weighted_error = (probs - y_oh) * sample_weights[:, None] / x_train.shape[0]
        grad_w = x_train.T @ weighted_error + weight_decay * weights
        grad_b = weighted_error.sum(axis=0)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            eps = 1e-12
            loss = -np.sum(y_oh * np.log(np.maximum(probs, eps)) * sample_weights[:, None]) / x_train.shape[0]
            loss += 0.5 * weight_decay * float(np.sum(weights * weights))
            pred = probs.argmax(axis=1)
            acc = float(np.mean(pred == y_train))
            history.append({"epoch": float(epoch), "loss": round(float(loss), 6), "train_accuracy": round(acc, 4)})
    return weights, bias, history


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def mlp_forward(
    x: np.ndarray,
    params: List[Dict[str, np.ndarray]],
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    activations = [x]
    pre_activations: List[np.ndarray] = []
    current = x
    for layer in params[:-1]:
        z = current @ layer["weights"] + layer["bias"]
        pre_activations.append(z)
        current = relu(z)
        activations.append(current)
    output_layer = params[-1]
    logits = current @ output_layer["weights"] + output_layer["bias"]
    return activations, pre_activations, logits


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    hidden_dims: Sequence[int],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, float]]]:
    if not hidden_dims:
        raise ValueError("hidden_dims must not be empty.")
    rng = np.random.default_rng(seed)
    dims = [int(x_train.shape[1])] + [int(dim) for dim in hidden_dims] + [int(num_classes)]
    params: List[Dict[str, np.ndarray]] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        params.append(
            {
                "weights": rng.normal(0.0, np.sqrt(2.0 / max(1, in_dim)), size=(in_dim, out_dim)),
                "bias": np.zeros((out_dim,), dtype=np.float64),
            }
        )

    y_oh = one_hot(y_train, num_classes)
    cls_weights = class_weights(y_train, num_classes)
    sample_weights = cls_weights[y_train]
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        activations, pre_activations, logits = mlp_forward(x_train, params)
        probs = softmax(logits)

        weighted_error = (probs - y_oh) * sample_weights[:, None] / x_train.shape[0]
        delta = weighted_error
        grad_weights: List[np.ndarray] = []
        grad_biases: List[np.ndarray] = []

        for layer_idx in reversed(range(len(params))):
            grad_w = activations[layer_idx].T @ delta + weight_decay * params[layer_idx]["weights"]
            grad_b = delta.sum(axis=0)
            grad_weights.insert(0, grad_w)
            grad_biases.insert(0, grad_b)
            if layer_idx > 0:
                delta = (delta @ params[layer_idx]["weights"].T) * (pre_activations[layer_idx - 1] > 0.0)

        for layer_idx, layer in enumerate(params):
            layer["weights"] -= learning_rate * grad_weights[layer_idx]
            layer["bias"] -= learning_rate * grad_biases[layer_idx]

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            eps = 1e-12
            loss = -np.sum(y_oh * np.log(np.maximum(probs, eps)) * sample_weights[:, None]) / x_train.shape[0]
            loss += 0.5 * weight_decay * float(
                sum(np.sum(layer["weights"] * layer["weights"]) for layer in params)
            )
            pred = probs.argmax(axis=1)
            acc = float(np.mean(pred == y_train))
            history.append({"epoch": float(epoch), "loss": round(float(loss), 6), "train_accuracy": round(acc, 4)})

    return params, history


def predict(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    probs = softmax(x @ weights + bias)
    return probs.argmax(axis=1), probs


def predict_mlp(x: np.ndarray, params: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    _, _, logits = mlp_forward(x, params)
    probs = softmax(logits)
    return probs.argmax(axis=1), probs


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    ids: Sequence[str],
) -> Dict[str, Any]:
    accuracy = float(np.mean(y_true == y_pred)) if y_true.size else 0.0
    per_class = {}
    f1_values = []
    weighted_f1_sum = 0.0
    for class_id, class_name in enumerate(DECISION_CLASSES):
        mask = y_true == class_id
        total = int(mask.sum())
        correct = int(((y_true == y_pred) & mask).sum())
        predicted = int((y_pred == class_id).sum())
        precision = correct / predicted if predicted else 0.0
        recall = correct / total if total else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if total:
            f1_values.append(f1)
            weighted_f1_sum += total * f1
        per_class[class_name] = {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else None,
            "precision": round(precision, 4) if total or predicted else None,
            "recall": round(recall, 4) if total else None,
            "f1": round(f1, 4) if total or predicted else None,
        }
    mistakes = []
    for idx, true_value, pred_value in zip(range(len(y_true)), y_true.tolist(), y_pred.tolist()):
        if true_value == pred_value:
            continue
        mistakes.append(
            {
                "event_id": ids[idx] if idx < len(ids) else str(idx),
                "true": DECISION_CLASSES[true_value],
                "pred": DECISION_CLASSES[pred_value],
                "confidence": round(float(np.max(probs[idx])), 4),
            }
        )
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1_present_classes": round(float(np.mean(f1_values)), 4) if f1_values else 0.0,
        "weighted_f1": round(weighted_f1_sum / max(1, int(y_true.size)), 4),
        "per_class": per_class,
        "num_mistakes": len(mistakes),
        "mistakes": mistakes[:20],
    }


def base_model_payload(
    path: Path,
    feature_names: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "task": "edge_traffic_student_decision",
        "decision_classes": DECISION_CLASSES,
        "feature_names": list(feature_names),
        "normalization": {
            "mean": mean.round(10).tolist(),
            "std": std.round(10).tolist(),
        },
        "metrics": metrics,
    }


def save_linear_model(
    path: Path,
    feature_names: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    metrics: Dict[str, Any],
) -> None:
    model = base_model_payload(path, feature_names, mean, std, metrics)
    model.update(
        {
            "model_type": "numpy_softmax_regression",
            "weights": weights.round(10).tolist(),
            "bias": bias.round(10).tolist(),
        }
    )
    save_json(model, path)


def save_mlp_model(
    path: Path,
    feature_names: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    params: List[Dict[str, np.ndarray]],
    metrics: Dict[str, Any],
) -> None:
    model = base_model_payload(path, feature_names, mean, std, metrics)
    layers = []
    for idx, layer in enumerate(params):
        layers.append(
            {
                "type": "dense_softmax" if idx == len(params) - 1 else "dense_relu",
                "weights": layer["weights"].round(10).tolist(),
                "bias": layer["bias"].round(10).tolist(),
            }
        )
    model.update(
        {
            "model_type": "numpy_mlp",
            "hidden_dims": [len(layer["bias"]) for layer in params[:-1]],
            "layers": layers,
        }
    )
    save_json(model, path)


def main() -> None:
    args = parse_args()
    records = read_jsonl(Path(args.labels))
    x, y, feature_names, event_ids = labels_from_records(records)
    if len(set(y.tolist())) < 2:
        raise ValueError("Need at least two decision classes to train a useful student.")

    if args.split_strategy == "temporal_group":
        train_idx, test_idx = split_indices_temporal_group(event_ids, args.test_ratio)
    else:
        train_idx, test_idx = split_indices(y, args.test_ratio, args.seed)
    x_train_raw = x[train_idx]
    x_test_raw = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    train_ids = [event_ids[int(idx)] for idx in train_idx]
    test_ids = [event_ids[int(idx)] for idx in test_idx]
    x_train, x_test, mean, std = standardize_train_test(x_train_raw, x_test_raw)
    hidden_dims = parse_hidden_dims(args.hidden_dims, args.hidden_dim)

    if args.model_type == "linear":
        weights, bias, history = train_softmax_regression(
            x_train=x_train,
            y_train=y_train,
            num_classes=len(DECISION_CLASSES),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        train_pred, train_probs = predict(x_train, weights, bias)
        test_pred, test_probs = predict(x_test, weights, bias)
        model_params = {"weights": weights, "bias": bias}
    else:
        params, history = train_mlp(
            x_train=x_train,
            y_train=y_train,
            num_classes=len(DECISION_CLASSES),
            hidden_dims=hidden_dims,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        train_pred, train_probs = predict_mlp(x_train, params)
        test_pred, test_probs = predict_mlp(x_test, params)
        model_params = params

    metrics = {
        "num_records": int(x.shape[0]),
        "num_train": int(train_idx.shape[0]),
        "num_test": int(test_idx.shape[0]),
        "feature_dim": int(x.shape[1]),
        "model_type": args.model_type,
        "hidden_dims": hidden_dims if args.model_type == "mlp" else [],
        "hidden_dim": hidden_dims[0] if args.model_type == "mlp" else 0,
        "split_strategy": args.split_strategy,
        "train": classification_metrics(y_train, train_pred, train_probs, train_ids),
        "test": classification_metrics(y_test, test_pred, test_probs, test_ids),
        "history": history,
    }
    if args.model_type == "linear":
        save_linear_model(
            Path(args.model_json),
            feature_names,
            mean,
            std,
            model_params["weights"],
            model_params["bias"],
            metrics,
        )
    else:
        save_mlp_model(Path(args.model_json), feature_names, mean, std, model_params, metrics)
    save_json(metrics, Path(args.metrics_json))

    print("Student model saved to:", args.model_json)
    print("Metrics saved to:", args.metrics_json)
    print("train_accuracy:", metrics["train"]["accuracy"])
    print("test_accuracy:", metrics["test"]["accuracy"])
    print("num_records:", metrics["num_records"])


if __name__ == "__main__":
    main()
