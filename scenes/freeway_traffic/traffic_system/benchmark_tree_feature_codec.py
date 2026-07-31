"""用途：比较交通树路径语义编码与常见数值传输基线的通信量和决策一致性。"""

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence
import zlib

import joblib
import numpy as np

from traffic_system.decision_utils import extract_feature_vector
from traffic_system.tree_feature_codec import TreeRoutingFeatureCodec


def _summary(values: Sequence[int]) -> Dict[str, float]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0}

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            return float(ordered[low])
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "mean": round(sum(ordered) / len(ordered), 6),
        "p50": round(percentile(0.50), 6),
        "p95": round(percentile(0.95), 6),
        "max": max(ordered),
    }


def _atomic_save_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file_obj:
            json.dump(value, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _quality(model: Any, original: np.ndarray, decoded: np.ndarray) -> Dict[str, Any]:
    original_predictions = np.asarray(model.predict(original))
    decoded_predictions = np.asarray(model.predict(decoded))
    prediction_matches = original_predictions == decoded_predictions

    original_probabilities = np.asarray(model.predict_proba(original), dtype=np.float64)
    decoded_probabilities = np.asarray(model.predict_proba(decoded), dtype=np.float64)

    route_matches = 0
    route_total = 0
    samples_with_all_routes_preserved = np.ones(original.shape[0], dtype=bool)
    for estimator in model.estimators_:
        matches = np.asarray(estimator.apply(original)) == np.asarray(
            estimator.apply(decoded)
        )
        route_matches += int(matches.sum())
        route_total += int(matches.size)
        samples_with_all_routes_preserved &= matches

    return {
        "prediction_match_count": int(prediction_matches.sum()),
        "prediction_match_rate": round(float(prediction_matches.mean()), 9),
        "samples_with_all_tree_routes_preserved": int(
            samples_with_all_routes_preserved.sum()
        ),
        "all_tree_routes_sample_rate": round(
            float(samples_with_all_routes_preserved.mean()), 9
        ),
        "tree_sample_route_matches": route_matches,
        "tree_sample_route_total": route_total,
        "tree_sample_route_match_rate": round(route_matches / route_total, 9),
        "max_probability_absolute_error": float(
            np.max(np.abs(original_probabilities - decoded_probabilities))
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark decision-path-preserving tree feature encoding."
    )
    parser.add_argument(
        "--events-dir", default="datasets/freeway_events_joint_metis4"
    )
    parser.add_argument(
        "--cloud-model", default="models/cloud_coordinator_topology_fused.joblib"
    )
    parser.add_argument(
        "--codec", default="models/traffic_tree_feature_codec_topology_v1.npz"
    )
    parser.add_argument(
        "--output", default="results/research/tree_codec_ablation_20260727.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_paths = sorted(Path(args.events_dir).glob("*.json"))
    if not event_paths:
        raise ValueError("no traffic event JSON files found")

    model_payload = joblib.load(Path(args.cloud_model))
    model = model_payload["model"]
    expected_names = [str(value) for value in model_payload["feature_names"]]
    codec = TreeRoutingFeatureCodec(Path(args.codec))

    rows: List[List[float]] = []
    json_bytes: List[int] = []
    full_float32_bytes: List[int] = []
    full_float32_zlib_bytes: List[int] = []
    active_float32_bytes: List[int] = []
    active_float32_zlib_bytes: List[int] = []
    active_float16_bytes: List[int] = []
    active_float16_zlib_bytes: List[int] = []
    routing_codec_bytes: List[int] = []
    routing_codec_base64_bytes: List[int] = []
    routing_decoded: List[List[float]] = []

    for path in event_paths:
        with path.open("r", encoding="utf-8") as file_obj:
            event = json.load(file_obj)
        values, names = extract_feature_vector(event)
        if list(names) != expected_names:
            raise ValueError("feature schema mismatch in {}".format(path))
        vector = np.asarray(values, dtype="<f4")
        active = vector[codec.active_indices]

        rows.append(vector.tolist())
        json_bytes.append(
            len(
                json.dumps(
                    vector.tolist(), ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            )
        )
        full_payload = vector.tobytes()
        full_float32_bytes.append(len(full_payload))
        full_float32_zlib_bytes.append(len(zlib.compress(full_payload, level=6)))
        active_payload = np.asarray(active, dtype="<f4").tobytes()
        active_float32_bytes.append(len(active_payload))
        active_float32_zlib_bytes.append(len(zlib.compress(active_payload, level=6)))
        half_payload = np.asarray(active, dtype="<f2").tobytes()
        active_float16_bytes.append(len(half_payload))
        active_float16_zlib_bytes.append(len(zlib.compress(half_payload, level=6)))

        evidence = codec.encode(path.stem, vector.tolist())
        routing_codec_bytes.append(evidence.size_bytes)
        routing_codec_base64_bytes.append(int(evidence.codec["base64_size_bytes"]))
        routing_decoded.append(
            codec.decode(evidence, codec.metadata["source_model_sha256"])
        )

    original = np.asarray(rows, dtype="<f4")
    active_float32_decoded = np.zeros_like(original)
    active_float32_decoded[:, codec.active_indices] = original[:, codec.active_indices]
    active_float16_decoded = np.zeros_like(original)
    active_float16_decoded[:, codec.active_indices] = (
        original[:, codec.active_indices].astype("<f2").astype("<f4")
    )
    routing_decoded_array = np.asarray(routing_decoded, dtype="<f4")

    methods = {
        "json_values_only": {
            "description": "十进制 JSON 数组；不含字段名、schema 和协议封装",
            "payload_bytes": _summary(json_bytes),
            "quality": _quality(model, original, original.copy()),
        },
        "full_float32_raw": {
            "description": "全部特征的原始 float32 数组",
            "payload_bytes": _summary(full_float32_bytes),
            "quality": _quality(model, original, original.copy()),
        },
        "full_float32_zlib": {
            "description": "全部 float32 特征逐事件 zlib 压缩",
            "payload_bytes": _summary(full_float32_zlib_bytes),
            "quality": _quality(model, original, original.copy()),
        },
        "active_float32_raw": {
            "description": "仅传树模型实际使用的特征；双方共享活动特征索引",
            "payload_bytes": _summary(active_float32_bytes),
            "quality": _quality(model, original, active_float32_decoded),
        },
        "active_float32_zlib": {
            "description": "活动 float32 特征逐事件 zlib 压缩",
            "payload_bytes": _summary(active_float32_zlib_bytes),
            "quality": _quality(model, original, active_float32_decoded),
        },
        "active_float16_raw": {
            "description": "活动特征直接转 float16；可能跨越树分裂阈值",
            "payload_bytes": _summary(active_float16_bytes),
            "quality": _quality(model, original, active_float16_decoded),
        },
        "active_float16_zlib": {
            "description": "活动 float16 特征逐事件 zlib 压缩",
            "payload_bytes": _summary(active_float16_zlib_bytes),
            "quality": _quality(model, original, active_float16_decoded),
        },
        "tree_routing_uint16": {
            "description": "按冻结树模型全部分裂阈值编码活动特征所在区间",
            "payload_bytes": _summary(routing_codec_bytes),
            "base64_bytes": _summary(routing_codec_base64_bytes),
            "quality": _quality(model, original, routing_decoded_array),
        },
    }

    reference_mean = methods["full_float32_raw"]["payload_bytes"]["mean"]
    for metrics in methods.values():
        mean_bytes = float(metrics["payload_bytes"]["mean"])
        metrics["reduction_vs_full_float32"] = round(
            1.0 - mean_bytes / reference_mean, 9
        )

    result = {
        "schema_version": 1,
        "task": "tree_routing_feature_codec_ablation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "events_directory": str(Path(args.events_dir)),
            "sample_count": len(event_paths),
            "feature_count": int(original.shape[1]),
            "active_feature_count": int(codec.active_indices.size),
            "tree_count": len(model.estimators_),
            "tree_sample_route_comparisons": len(event_paths)
            * len(model.estimators_),
            "source_model_sha256": codec.metadata["source_model_sha256"],
            "codec_artifact_id": codec.metadata["artifact_id"],
        },
        "measurement_boundary": (
            "这里只比较每事件特征数据面载荷；共享 codec artifact、字段名、"
            "schema、base64 和 HTTP/JSON 信封开销单独说明，不混入核心载荷。"
        ),
        "methods": methods,
    }
    _atomic_save_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
