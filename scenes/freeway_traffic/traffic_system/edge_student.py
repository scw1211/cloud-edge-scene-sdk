"""用途：加载轻量 MLP Student，并对单个边缘事件进行毫秒级初步决策。"""

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from traffic_system.decision_utils import (
    DECISION_CLASSES,
    build_decision_from_student_class,
    extract_feature_vector,
    load_json,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the edge-side lightweight student decision model.")
    parser.add_argument("--edge_event", required=True)
    parser.add_argument("--model_json", default="models/edge_student_freeway_joint_metis4.json")
    parser.add_argument("--output_json", default="results/decision/edge_student_check.json")
    parser.add_argument("--runs", type=int, default=1000)
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def load_student_model(path: Path) -> Dict[str, Any]:
    model = load_json(path)
    required = ["decision_classes", "normalization"]
    missing = [key for key in required if key not in model]
    if missing:
        raise ValueError("Student model missing fields: {}".format(", ".join(missing)))
    model["_compiled_mean"] = np.asarray(model["normalization"]["mean"], dtype=np.float64)
    model["_compiled_std"] = np.asarray(model["normalization"]["std"], dtype=np.float64)
    model_type = str(model.get("model_type", "numpy_softmax_regression"))
    if model_type == "numpy_mlp":
        if "layers" not in model:
            raise ValueError("MLP student model missing field: layers")
        model["_compiled_layers"] = [
            {
                "type": str(layer.get("type", "")),
                "weights": np.asarray(layer["weights"], dtype=np.float64),
                "bias": np.asarray(layer["bias"], dtype=np.float64),
            }
            for layer in model["layers"]
        ]
    else:
        for key in ("weights", "bias"):
            if key not in model:
                raise ValueError("Linear student model missing field: {}".format(key))
        model["_compiled_weights"] = np.asarray(model["weights"], dtype=np.float64)
        model["_compiled_bias"] = np.asarray(model["bias"], dtype=np.float64)
    return model


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def dense_vector(values: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Evaluate one small dense layer without starting a BLAS thread pool."""
    return np.sum(weights * values[:, None], axis=0) + bias


def predict_student(event: Dict[str, Any], model: Dict[str, Any]) -> Tuple[str, float, np.ndarray]:
    vector, _ = extract_feature_vector(event)
    x = np.asarray(vector, dtype=np.float64)
    mean = model.get("_compiled_mean")
    std = model.get("_compiled_std")
    if mean is None or std is None:
        mean = np.asarray(model["normalization"]["mean"], dtype=np.float64)
        std = np.asarray(model["normalization"]["std"], dtype=np.float64)
    x_norm = (x - mean) / np.where(std < 1e-8, 1.0, std)
    if str(model.get("model_type", "numpy_softmax_regression")) == "numpy_mlp":
        layers = model.get("_compiled_layers") or model["layers"]
        current = x_norm
        for layer_idx, layer in enumerate(layers):
            weights = layer["weights"] if isinstance(layer["weights"], np.ndarray) else np.asarray(layer["weights"], dtype=np.float64)
            bias = layer["bias"] if isinstance(layer["bias"], np.ndarray) else np.asarray(layer["bias"], dtype=np.float64)
            current = dense_vector(current, weights, bias)
            if layer_idx < len(layers) - 1:
                current = relu(current)
        probs = softmax(current)
    else:
        weights = model.get("_compiled_weights")
        bias = model.get("_compiled_bias")
        if weights is None or bias is None:
            weights = np.asarray(model["weights"], dtype=np.float64)
            bias = np.asarray(model["bias"], dtype=np.float64)
        probs = softmax(dense_vector(x_norm, weights, bias))
    class_id = int(np.argmax(probs))
    classes = list(model.get("decision_classes", DECISION_CLASSES))
    decision = classes[class_id] if class_id < len(classes) else "monitor"
    confidence = float(probs[class_id])
    return decision, confidence, probs


def measure_student_latency(event: Dict[str, Any], model: Dict[str, Any], runs: int) -> float:
    if runs <= 0:
        raise ValueError("runs must be positive.")
    for _ in range(min(20, runs)):
        predict_student(event, model)
    start = time.perf_counter()
    for _ in range(runs):
        predict_student(event, model)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms / float(runs)


def current_max_rss_mb() -> float:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(float(rss_kb) / 1024.0, 4)


def main() -> None:
    args = parse_args()
    event = load_json(Path(args.edge_event))
    model = load_student_model(Path(args.model_json))
    decision_class, confidence, probs = predict_student(event, model)
    latency_ms = measure_student_latency(event, model, args.runs)
    decision = build_decision_from_student_class(
        event,
        decision_class,
        confidence=confidence,
        decision_source="edge_student",
    )
    decision["student_metrics"] = {
        "latency_ms": round(latency_ms, 6),
        "max_rss_mb": current_max_rss_mb(),
        "class_probabilities": {
            cls: round(float(prob), 6)
            for cls, prob in zip(model.get("decision_classes", DECISION_CLASSES), probs.tolist())
        },
    }
    save_json(decision, Path(args.output_json))
    print("decision:", decision["decision"])
    print("confidence:", decision["confidence"])
    print("latency_ms:", decision["student_metrics"]["latency_ms"])
    print("max_rss_mb:", decision["student_metrics"]["max_rss_mb"])
    print("JSON saved to:", args.output_json)


if __name__ == "__main__":
    main()
