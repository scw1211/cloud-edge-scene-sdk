#!/usr/bin/env python3
"""Build and evaluate PatchCore banks from the deployed ONNX feature extractor.

The ONNX model produces a ``[1, 384, 20, 20]`` patch feature map.  This tool
turns normal training images into the exact ``.pcbank`` format consumed by the
existing TensorRT/CUDA C++ runtimes and evaluates the resulting image/pixel
anomaly scores on MulSen-AD without using test data during bank construction.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import uuid

import numpy as np


MAGIC = b"PCBNK01\0"
VERSION = 1
HEADER = struct.Struct("<8sIIIff")
IMAGE_SIZE = 160
FEATURE_CHANNELS = 384
FEATURE_GRID = 20
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def preprocess_image(path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Match the production PIL/ImageNet preprocessing path exactly."""
    from PIL import Image

    with Image.open(str(path)) as image:
        image = image.convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BICUBIC
        )
        pixels = np.asarray(image, dtype=np.float32)
    return ((pixels / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(
        np.float32, copy=False
    )


class OnnxFeatureExtractor:
    """Strict ONNX Runtime wrapper for the frozen ViT patch extractor."""

    def __init__(self, model_path: Path):
        import onnxruntime as ort

        self.model_path = model_path.resolve()
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "image":
            raise ValueError("ONNX input must be exactly 'image'")
        if len(outputs) != 1 or outputs[0].name != "patch_features":
            raise ValueError("ONNX output must be exactly 'patch_features'")

    def infer(self, image_path: Path) -> Tuple[np.ndarray, float, float]:
        started = time.perf_counter()
        tensor = preprocess_image(image_path)
        preprocessed = time.perf_counter()
        features = self.session.run(None, {"image": tensor})[0]
        finished = time.perf_counter()
        expected = (1, FEATURE_CHANNELS, FEATURE_GRID, FEATURE_GRID)
        if tuple(features.shape) != expected or features.dtype != np.float32:
            raise ValueError(
                "unexpected ONNX output: {} {} (expected {} float32)".format(
                    features.shape, features.dtype, expected
                )
            )
        return (
            features[0],
            (preprocessed - started) * 1000.0,
            (finished - preprocessed) * 1000.0,
        )


def _modality_dir(product_root: Path, modality: str) -> Path:
    canonical = {"rgb": "RGB", "infrared": "Infrared"}[modality]
    path = product_root.resolve() / canonical
    if not path.is_dir():
        raise FileNotFoundError("MulSen-AD modality directory is missing: {}".format(path))
    return path


def _pngs(path: Path) -> List[Path]:
    values = sorted(item for item in path.rglob("*.png") if item.is_file())
    if not values:
        raise ValueError("no PNG images found under {}".format(path))
    return values


def extract_patch_rows(
    extractor: OnnxFeatureExtractor, images: Sequence[Path]
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    rows: List[np.ndarray] = []
    timings: List[Dict[str, Any]] = []
    for index, image in enumerate(images):
        feature_map, preprocessing_ms, inference_ms = extractor.infer(image)
        rows.append(feature_map.reshape(FEATURE_CHANNELS, -1).T)
        timings.append(
            {
                "index": index,
                "path": str(image.resolve()),
                "preprocessing_ms": preprocessing_ms,
                "inference_ms": inference_ms,
            }
        )
    return np.concatenate(rows, axis=0).astype(np.float32), timings


def _sparse_projection(features: np.ndarray, epsilon: float, seed: int) -> np.ndarray:
    from sklearn.random_projection import (
        SparseRandomProjection,
        johnson_lindenstrauss_min_dim,
    )

    target = int(johnson_lindenstrauss_min_dim(features.shape[0], eps=epsilon))
    if target >= features.shape[1]:
        return np.asarray(features, dtype=np.float32)

    projection = SparseRandomProjection(
        n_components=target, eps=epsilon, random_state=seed
    )
    return np.asarray(projection.fit_transform(features), dtype=np.float32)


def select_kcenter(
    features: np.ndarray, count: int, epsilon: float = 0.9, seed: int = 42
) -> np.ndarray:
    """Deterministic PatchCore greedy k-center selection after JL projection."""
    if features.ndim != 2 or not 0 < count <= features.shape[0]:
        raise ValueError("invalid feature matrix or coreset count")
    if count == features.shape[0]:
        return np.arange(count, dtype=np.int64)
    projected = _sparse_projection(features, epsilon, seed)
    rng = np.random.RandomState(seed)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(rng.randint(projected.shape[0]))
    min_squared = np.full(projected.shape[0], np.inf, dtype=np.float32)
    for offset in range(count):
        if offset:
            selected[offset] = int(np.argmax(min_squared))
        center = projected[selected[offset]]
        squared = np.sum((projected - center) ** 2, axis=1, dtype=np.float32)
        np.minimum(min_squared, squared, out=min_squared)
        min_squared[selected[: offset + 1]] = -1.0
    if len(set(int(value) for value in selected)) != count:
        raise RuntimeError("k-center returned duplicate rows")
    return selected


def write_pcbank(
    output: Path, normalized_patches: np.ndarray, mean: float, stddev: float
) -> None:
    if normalized_patches.ndim != 2 or normalized_patches.shape[1] != FEATURE_CHANNELS:
        raise ValueError("memory bank must be [rows, 384]")
    if not np.isfinite(normalized_patches).all() or not math.isfinite(stddev) or stddev <= 0:
        raise ValueError("memory bank values must be finite and stddev must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    with temporary.open("wb") as handle:
        handle.write(
            HEADER.pack(
                MAGIC,
                VERSION,
                normalized_patches.shape[0],
                normalized_patches.shape[1],
                float(mean),
                float(stddev),
            )
        )
        handle.write(
            np.ascontiguousarray(normalized_patches, dtype=np.float16).tobytes()
        )
    temporary.replace(output)


def read_pcbank(path: Path) -> Tuple[np.ndarray, float, float]:
    payload = path.read_bytes()
    if len(payload) < HEADER.size:
        raise ValueError("memory bank header is missing")
    magic, version, rows, cols, mean, stddev = HEADER.unpack_from(payload)
    if magic != MAGIC or version != VERSION or cols != FEATURE_CHANNELS:
        raise ValueError("unsupported memory bank identity")
    expected = HEADER.size + rows * cols * np.dtype(np.float16).itemsize
    if len(payload) != expected or rows <= 0 or stddev <= 0:
        raise ValueError("invalid memory bank size or statistics")
    bank = np.frombuffer(payload, dtype=np.float16, offset=HEADER.size).reshape(rows, cols)
    return bank.astype(np.float32), float(mean), float(stddev)


def patchcore_distances(
    feature_map: np.ndarray, bank: np.ndarray, mean: float, stddev: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    patches = feature_map.reshape(FEATURE_CHANNELS, -1).T.astype(np.float32)
    patches = (patches - mean) / stddev
    patch_norms = np.sum(patches * patches, axis=1, keepdims=True)
    bank_norms = np.sum(bank * bank, axis=1)[None, :]
    squared = patch_norms + bank_norms - 2.0 * np.matmul(patches, bank.T)
    np.maximum(squared, 0.0, out=squared)
    indices = np.argmin(squared, axis=1)
    distances = np.sqrt(squared[np.arange(squared.shape[0]), indices])
    return patches, distances.astype(np.float32), indices.astype(np.int64)


def patchcore_image_score(
    patches: np.ndarray, distances: np.ndarray, indices: np.ndarray, bank: np.ndarray
) -> float:
    patch_index = int(np.argmax(distances))
    s_star = float(distances[patch_index]) / 1000.0
    reference = bank[int(indices[patch_index])]
    reference_distances = np.linalg.norm(bank - reference[None, :], axis=1)
    neighbours = np.argsort(reference_distances)[: min(3, bank.shape[0])]
    if neighbours.size <= 1:
        return 0.0
    neighbour_distances = np.linalg.norm(
        bank[neighbours[1:]] - patches[patch_index][None, :], axis=1
    ) / 1000.0
    dimension = math.sqrt(float(bank.shape[1]))
    denominator = float(np.exp(neighbour_distances / dimension).sum())
    weight = 1.0 - math.exp(s_star / dimension) / denominator
    return float(weight * s_star)


def _upsample_map(distances: np.ndarray, blur_radius: float) -> np.ndarray:
    from PIL import Image

    patch_map = distances.reshape(FEATURE_GRID, FEATURE_GRID).astype(np.float32)
    image = Image.fromarray(patch_map, mode="F").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    result = np.asarray(image, dtype=np.float32)
    if blur_radius > 0:
        radius = int(math.ceil(3.0 * blur_radius))
        offsets = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(offsets * offsets) / (2.0 * blur_radius * blur_radius))
        kernel /= kernel.sum()
        horizontal = np.pad(result, ((0, 0), (radius, radius)), mode="reflect")
        result = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="valid"), 1, horizontal
        )
        vertical = np.pad(result, ((radius, radius), (0, 0)), mode="reflect")
        result = np.apply_along_axis(
            lambda column: np.convolve(column, kernel, mode="valid"), 0, vertical
        )
    return result.astype(np.float32, copy=False)


def _ground_truth(image_path: Path, modality_root: Path) -> Tuple[int, np.ndarray]:
    from PIL import Image

    defect = image_path.parent.name
    if defect == "good":
        return 0, np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    label = 1
    csv_path = modality_root / "GT" / defect / "data.csv"
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        row = rows[int(image_path.stem)]
        label = int(
            int(row.get("RGB", 0))
            or int(row.get("infrared", row.get("Infrared", 0)))
            or int(row.get("pointcloud", row.get("Pointcloud", 0)))
        )
    mask_path = modality_root / "GT" / defect / image_path.name
    if not mask_path.is_file():
        return label, np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask = Image.open(mask_path).convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
    )
    return label, (np.asarray(mask) > 127).astype(np.uint8)


def _latency(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
    )

    y_true = np.asarray(labels, dtype=np.uint8)
    y_score = np.asarray(scores, dtype=np.float64)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    threshold = (
        float(thresholds[min(best, len(thresholds) - 1)])
        if len(thresholds)
        else float("inf")
    )
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "max_f1": float(f1[best]),
        "oracle_threshold": threshold,
        "positive": int(y_true.sum()),
        "negative": int((1 - y_true).sum()),
    }


def _threshold_metrics(
    labels: Sequence[int], scores: Sequence[float], review_low: float, review_high: float
) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    y_true = np.asarray(labels, dtype=np.uint8)
    y_score = np.asarray(scores, dtype=np.float64)

    def one(threshold: float) -> Dict[str, Any]:
        predicted = (y_score >= threshold).astype(np.uint8)
        matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
        return {
            "threshold": threshold,
            "accuracy": float(accuracy_score(y_true, predicted)),
            "f1": float(f1_score(y_true, predicted, zero_division=0)),
            "confusion_matrix_tn_fp_fn_tp": [
                int(matrix[0, 0]),
                int(matrix[0, 1]),
                int(matrix[1, 0]),
                int(matrix[1, 1]),
            ],
        }

    states = np.where(y_score < review_low, "normal", np.where(y_score < review_high, "review", "anomaly"))
    return {
        "review_low": review_low,
        "review_high": review_high,
        "review_or_anomaly_is_positive": one(review_low),
        "anomaly_only_is_positive": one(review_high),
        "state_counts": {
            state: int(np.count_nonzero(states == state))
            for state in ("normal", "review", "anomaly")
        },
    }


def build_bank(args: argparse.Namespace) -> Dict[str, Any]:
    product_root = Path(args.product_root)
    modality_root = _modality_dir(product_root, args.modality)
    images = _pngs(modality_root / "train")
    extractor = OnnxFeatureExtractor(Path(args.onnx))
    patches, timings = extract_patch_rows(extractor, images)
    mean = float(patches.mean(dtype=np.float64))
    stddev = float(patches.std(dtype=np.float64))
    normalized = (patches - mean) / stddev
    count = max(1, int(round(normalized.shape[0] * args.coreset_fraction)))
    indices = select_kcenter(normalized, count, args.projection_epsilon, args.seed)
    output = Path(args.output)
    write_pcbank(output, normalized[indices], mean, stddev)
    manifest = {
        "schema_version": "industrial-patchcore-bank/v1",
        "product": product_root.name,
        "modality": args.modality,
        "model": _identity(Path(args.onnx)),
        "dataset": {
            "product_root": str(product_root.resolve()),
            "train_images": len(images),
            "train_file_sha256": {
                str(path.relative_to(product_root)): _sha256(path) for path in images
            },
            "test_data_used": False,
        },
        "preprocessing": {
            "image_size": IMAGE_SIZE,
            "resize": "PIL bicubic",
            "color": "RGB (infrared grayscale replicated to RGB)",
            "mean": MEAN.tolist(),
            "std": STD.tolist(),
        },
        "feature_contract": {
            "input": "image:float32[1,3,160,160]",
            "output": "patch_features:float32[1,384,20,20]",
            "raw_patch_rows": int(patches.shape[0]),
            "channels": int(patches.shape[1]),
            "scalar_mean": mean,
            "scalar_stddev": stddev,
        },
        "coreset": {
            "algorithm": "greedy-kcenter-after-sparse-random-projection",
            "fraction": args.coreset_fraction,
            "rows": int(indices.size),
            "projection_epsilon": args.projection_epsilon,
            "seed": args.seed,
        },
        "bank": _identity(output),
        "timing_ms": {
            "preprocessing": _latency([row["preprocessing_ms"] for row in timings]),
            "onnx_inference": _latency([row["inference_ms"] for row in timings]),
        },
    }
    _atomic_json(Path(args.manifest), manifest)
    return manifest


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    product_root = Path(args.product_root)
    modality_root = _modality_dir(product_root, args.modality)
    images = _pngs(modality_root / "test")
    bank, mean, stddev = read_pcbank(Path(args.bank))
    extractor = OnnxFeatureExtractor(Path(args.onnx))
    blur_radius = 0.0 if args.modality == "rgb" else 4.0
    records: List[Dict[str, Any]] = []
    pixel_labels: List[np.ndarray] = []
    pixel_scores: List[np.ndarray] = []
    for image in images:
        started = time.perf_counter()
        feature_map, preprocessing_ms, inference_ms = extractor.infer(image)
        post_started = time.perf_counter()
        patches, distances, indices = patchcore_distances(
            feature_map, bank, mean, stddev
        )
        score = patchcore_image_score(patches, distances, indices, bank)
        score_map = _upsample_map(distances, blur_radius)
        label, mask = _ground_truth(image, modality_root)
        finished = time.perf_counter()
        record = {
            "path": str(image.resolve()),
            "defect": image.parent.name,
            "label": label,
            "image_score": score,
            "preprocessing_ms": preprocessing_ms,
            "onnx_inference_ms": inference_ms,
            "patchcore_ms": (finished - post_started) * 1000.0,
            "end_to_end_ms": (finished - started) * 1000.0,
        }
        records.append(record)
        pixel_labels.append(mask.reshape(-1))
        pixel_scores.append(score_map.reshape(-1))
    image_metrics = _metrics(
        [row["label"] for row in records], [row["image_score"] for row in records]
    )
    pixel_metrics = _metrics(
        np.concatenate(pixel_labels).tolist(), np.concatenate(pixel_scores).tolist()
    )
    result = {
        "schema_version": "industrial-patchcore-onnx-evaluation/v1",
        "product": product_root.name,
        "modality": args.modality,
        "model": _identity(Path(args.onnx)),
        "bank": _identity(Path(args.bank)),
        "contract": {
            "test_images": len(images),
            "test_used_for_training_or_coreset": False,
            "post_hoc_remapping": False,
            "score": "PatchCore nearest-neighbour plus paper reweighting",
        },
        "image_metrics": image_metrics,
        "pixel_metrics": pixel_metrics,
        "latency_ms": {
            "preprocessing": _latency([row["preprocessing_ms"] for row in records]),
            "onnx_inference": _latency([row["onnx_inference_ms"] for row in records]),
            "patchcore": _latency([row["patchcore_ms"] for row in records]),
            "end_to_end": _latency([row["end_to_end_ms"] for row in records]),
        },
        "records": records,
    }
    if args.review_bands:
        bands_path = Path(args.review_bands)
        bands = json.loads(bands_path.read_text(encoding="utf-8"))
        values = bands["modalities"][args.modality][product_root.name]
        result["frozen_review_band"] = {
            "source": _identity(bands_path),
            **_threshold_metrics(
                [row["label"] for row in records],
                [row["image_score"] for row in records],
                float(values["review_low"]),
                float(values["review_high"]),
            ),
        }
    _atomic_json(Path(args.output), result)
    return result


def infer_event(args: argparse.Namespace) -> Dict[str, Any]:
    """Run one real image and emit a plugin-compatible CloudEvent payload."""
    image_path = Path(args.image).resolve()
    bank_path = Path(args.bank).resolve()
    model_path = Path(args.onnx).resolve()
    bank, mean, stddev = read_pcbank(bank_path)
    extractor = OnnxFeatureExtractor(model_path)
    started = time.perf_counter()
    feature_map, preprocessing_ms, inference_ms = extractor.infer(image_path)
    post_started = time.perf_counter()
    patches, distances, indices = patchcore_distances(
        feature_map, bank, mean, stddev
    )
    score = patchcore_image_score(patches, distances, indices, bank)
    score_map = _upsample_map(
        distances, 0.0 if args.modality == "rgb" else 4.0
    )
    heatmap_path = Path(args.heatmap).resolve()
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_heatmap = heatmap_path.with_name(heatmap_path.name + ".part")
    score_map.astype(np.float32, copy=False).tofile(str(temporary_heatmap))
    temporary_heatmap.replace(heatmap_path)
    finished = time.perf_counter()
    sample_id = args.sample_id or image_path.stem
    event_id = args.event_id or "industrial-perception-{}".format(uuid.uuid4().hex)
    result = {
        "specversion": "1.0",
        "id": event_id,
        "source": "urn:edge:industrial-{}:onnx-patchcore".format(args.modality),
        "type": "com.example.industrial.anomaly-map.v1",
        "scene": "industrial_anomaly",
        "edgeid": args.edge_id,
        "subject": "{}/{}/{}".format(args.product, args.modality, sample_id),
        "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "datacontenttype": "application/json",
        "dataschema": "https://cloud-edge.local/schemas/examples/industrial-anomaly-map-v1.json",
        "data": {
            "sample_id": sample_id,
            "product": args.product,
            "modality": args.modality,
            "score": score,
            "raw_uri": image_path.as_uri(),
            "heatmap_uri": args.heatmap_uri or heatmap_path.as_uri(),
            "heatmap_size_bytes": heatmap_path.stat().st_size,
            "heatmap_sha256": _sha256(heatmap_path),
            "preprocessing_latency_ms": preprocessing_ms,
            "inference_ms": (finished - started) * 1000.0,
            "model": {
                "name": "vit-small-patch8-160-onnx-patchcore",
                "version": _sha256(model_path)[:12],
                "backbone_sha256": _sha256(model_path),
                "memory_bank_sha256": _sha256(bank_path),
                "onnx_inference_ms": inference_ms,
                "patchcore_ms": (finished - post_started) * 1000.0,
            },
        },
    }
    _atomic_json(Path(args.output), result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    builder = subparsers.add_parser("build-bank")
    builder.add_argument("--onnx", required=True)
    builder.add_argument("--product-root", required=True)
    builder.add_argument("--modality", choices=("rgb", "infrared"), required=True)
    builder.add_argument("--coreset-fraction", type=float, required=True)
    builder.add_argument("--projection-epsilon", type=float, default=0.9)
    builder.add_argument("--seed", type=int, default=42)
    builder.add_argument("--output", required=True)
    builder.add_argument("--manifest", required=True)
    builder.set_defaults(handler=build_bank)
    evaluator = subparsers.add_parser("evaluate")
    evaluator.add_argument("--onnx", required=True)
    evaluator.add_argument("--bank", required=True)
    evaluator.add_argument("--product-root", required=True)
    evaluator.add_argument("--modality", choices=("rgb", "infrared"), required=True)
    evaluator.add_argument("--review-bands")
    evaluator.add_argument("--output", required=True)
    evaluator.set_defaults(handler=evaluate)
    inference = subparsers.add_parser("infer-event")
    inference.add_argument("--onnx", required=True)
    inference.add_argument("--bank", required=True)
    inference.add_argument("--image", required=True)
    inference.add_argument("--product", default="capsule")
    inference.add_argument("--modality", choices=("rgb", "infrared"), required=True)
    inference.add_argument("--sample-id")
    inference.add_argument("--event-id")
    inference.add_argument("--edge-id", default="industrial-perception-edge")
    inference.add_argument("--heatmap", required=True)
    inference.add_argument(
        "--heatmap-uri",
        help=(
            "Published URI visible to the edge service; defaults to the local "
            "--heatmap file URI"
        ),
    )
    inference.add_argument("--output", required=True)
    inference.set_defaults(handler=infer_event)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if getattr(args, "coreset_fraction", 0.5) <= 0 or getattr(
        args, "coreset_fraction", 0.5
    ) > 1:
        raise SystemExit("--coreset-fraction must be in (0, 1]")
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
