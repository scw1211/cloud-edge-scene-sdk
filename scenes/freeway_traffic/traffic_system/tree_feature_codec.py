"""用途：将交通云协调器特征编码为保持树模型分支结果的紧凑数据面载荷。"""

import argparse
import base64
import hashlib
import json
import math
import os
import tempfile
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np

from cloud_edge_framework.contracts import Evidence


CODEC_NAME = "tree_routing_uint16"
CODEC_VERSION = 1


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_schema_sha256(feature_names: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_json([str(name) for name in feature_names])).hexdigest()


def _artifact_id(
    feature_schema: str,
    active_indices: np.ndarray,
    offsets: np.ndarray,
    thresholds: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(feature_schema.encode("ascii"))
    digest.update(np.asarray(active_indices, dtype="<u2").tobytes())
    digest.update(np.asarray(offsets, dtype="<u4").tobytes())
    digest.update(np.asarray(thresholds, dtype="<f8").tobytes())
    return digest.hexdigest()


def export_codec_artifact(model_path: Path, output_path: Path) -> Dict[str, Any]:
    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "model" not in payload or "feature_names" not in payload:
        raise ValueError("cloud model payload must contain model and feature_names")
    model = payload["model"]
    feature_names = [str(name) for name in payload["feature_names"]]
    if not hasattr(model, "estimators_"):
        raise ValueError("tree-routing codec requires an ensemble exposing estimators_")

    threshold_sets: List[set] = [set() for _ in feature_names]
    for estimator in model.estimators_:
        for feature_index, threshold in zip(estimator.tree_.feature, estimator.tree_.threshold):
            index = int(feature_index)
            if index >= 0:
                threshold_sets[index].add(float(threshold))

    active_indices = np.asarray(
        [index for index, values in enumerate(threshold_sets) if values], dtype="<u2"
    )
    offsets = [0]
    flattened: List[float] = []
    for index in active_indices:
        values = sorted(threshold_sets[int(index)])
        if len(values) >= 65535:
            raise ValueError("feature {} has too many routing intervals".format(int(index)))
        flattened.extend(values)
        offsets.append(len(flattened))
    offsets_array = np.asarray(offsets, dtype="<u4")
    thresholds = np.asarray(flattened, dtype="<f8")
    schema_hash = feature_schema_sha256(feature_names)
    artifact_id = _artifact_id(schema_hash, active_indices, offsets_array, thresholds)
    metadata = {
        "codec": CODEC_NAME,
        "codec_version": CODEC_VERSION,
        "artifact_id": artifact_id,
        "feature_schema_sha256": schema_hash,
        "feature_count": len(feature_names),
        "active_feature_count": int(active_indices.size),
        "threshold_count": int(thresholds.size),
        "source_model_sha256": _sha256_file(model_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output_path.name + ".", suffix=".tmp", dir=str(output_path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as file_obj:
            np.savez_compressed(
                file_obj,
                metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"))),
                feature_names=np.asarray(feature_names),
                active_indices=active_indices,
                offsets=offsets_array,
                thresholds=thresholds,
            )
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return {**metadata, "artifact_path": str(output_path), "artifact_bytes": output_path.stat().st_size}


class TreeRoutingFeatureCodec:
    """Encode only model-active features as tree routing interval identifiers."""

    def __init__(self, artifact_path: Path) -> None:
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.is_file():
            raise FileNotFoundError("traffic feature codec not found: {}".format(self.artifact_path))
        with np.load(self.artifact_path, allow_pickle=False) as artifact:
            self.metadata = json.loads(str(artifact["metadata_json"].item()))
            self.feature_names = [str(value) for value in artifact["feature_names"].tolist()]
            self.active_indices = np.asarray(artifact["active_indices"], dtype=np.uint16)
            self.offsets = np.asarray(artifact["offsets"], dtype=np.uint32)
            self.thresholds = np.asarray(artifact["thresholds"], dtype=np.float64)
        self._validate_artifact()

    def _validate_artifact(self) -> None:
        if self.metadata.get("codec") != CODEC_NAME:
            raise ValueError("unsupported traffic feature codec")
        if int(self.metadata.get("codec_version", 0)) != CODEC_VERSION:
            raise ValueError("unsupported traffic feature codec version")
        if len(self.feature_names) != int(self.metadata.get("feature_count", -1)):
            raise ValueError("feature codec count mismatch")
        if self.offsets.size != self.active_indices.size + 1:
            raise ValueError("feature codec offsets are invalid")
        if int(self.offsets[-1]) != int(self.thresholds.size):
            raise ValueError("feature codec threshold array is invalid")
        schema_hash = feature_schema_sha256(self.feature_names)
        if schema_hash != self.metadata.get("feature_schema_sha256"):
            raise ValueError("feature codec schema checksum mismatch")
        artifact_id = _artifact_id(
            schema_hash, self.active_indices, self.offsets, self.thresholds
        )
        if artifact_id != self.metadata.get("artifact_id"):
            raise ValueError("feature codec artifact checksum mismatch")

    def encode(self, event_id: str, values: Sequence[float]) -> Evidence:
        if len(values) != len(self.feature_names):
            raise ValueError("feature vector does not match codec schema")
        # sklearn tree estimators convert inference inputs to float32 before
        # comparing them with their float64 thresholds.  Encode that effective
        # value so the cloud-side reconstruction follows the same branches.
        vector = np.asarray(values, dtype="<f4").astype(np.float64)
        if not np.all(np.isfinite(vector)):
            raise ValueError("feature vector contains non-finite values")
        codes = np.empty(self.active_indices.size, dtype="<u2")
        for position, feature_index in enumerate(self.active_indices):
            start = int(self.offsets[position])
            end = int(self.offsets[position + 1])
            thresholds = self.thresholds[start:end]
            codes[position] = np.searchsorted(
                thresholds, vector[int(feature_index)], side="left"
            )
        raw_payload = codes.tobytes()
        compressed = zlib.compress(raw_payload, level=6)
        if len(compressed) < len(raw_payload):
            payload = compressed
            encoding = CODEC_NAME + "+zlib+base64"
        else:
            payload = raw_payload
            encoding = CODEC_NAME + "+base64"
        inline = base64.b64encode(payload).decode("ascii")
        source_size = int(vector.size * np.dtype("<f4").itemsize)
        return Evidence(
            evidence_id="{}_task_features".format(event_id),
            level="feature",
            modality="traffic_task_features",
            encoding=encoding,
            inline=inline,
            shape=[int(vector.size)],
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            content_type="application/vnd.cloud-edge.tree-routing-features",
            codec={
                "name": CODEC_NAME,
                "version": CODEC_VERSION,
                "artifact_id": self.metadata["artifact_id"],
                "feature_schema_sha256": self.metadata["feature_schema_sha256"],
                "source_dtype": "float32",
                "source_size_bytes": source_size,
                "active_feature_count": int(self.active_indices.size),
                "base64_size_bytes": len(inline),
                "decision_preserving_for_model": self.metadata["source_model_sha256"],
            },
        )

    def decode(
        self,
        evidence: Evidence,
        expected_model_sha256: Optional[str] = None,
    ) -> List[float]:
        codec = evidence.codec
        if codec.get("artifact_id") != self.metadata["artifact_id"]:
            raise ValueError("feature evidence uses a different codec artifact")
        if codec.get("feature_schema_sha256") != self.metadata["feature_schema_sha256"]:
            raise ValueError("feature evidence schema mismatch")
        if expected_model_sha256 is not None and expected_model_sha256 != self.metadata.get(
            "source_model_sha256"
        ):
            raise ValueError("feature codec was not exported from the active cloud model")
        if not isinstance(evidence.inline, str):
            raise ValueError("feature evidence payload must be base64 text")
        try:
            payload = base64.b64decode(evidence.inline.encode("ascii"), validate=True)
        except Exception as exc:
            raise ValueError("feature evidence is not valid base64") from exc
        if len(payload) != evidence.size_bytes:
            raise ValueError("feature evidence size mismatch")
        if hashlib.sha256(payload).hexdigest() != evidence.sha256:
            raise ValueError("feature evidence checksum mismatch")
        if "+zlib+" in evidence.encoding:
            try:
                raw_payload = zlib.decompress(payload)
            except zlib.error as exc:
                raise ValueError("feature evidence decompression failed") from exc
        else:
            raw_payload = payload
        codes = np.frombuffer(raw_payload, dtype="<u2")
        if codes.size != self.active_indices.size:
            raise ValueError("feature evidence code count mismatch")

        vector = np.zeros(len(self.feature_names), dtype=np.float64)
        for position, feature_index in enumerate(self.active_indices):
            start = int(self.offsets[position])
            end = int(self.offsets[position + 1])
            thresholds = self.thresholds[start:end]
            interval = int(codes[position])
            if interval > thresholds.size:
                raise ValueError("feature evidence interval is invalid")
            if interval == thresholds.size:
                lower = float(thresholds[-1])
                candidate = np.float32(lower)
                if float(candidate) <= lower:
                    candidate = np.nextafter(candidate, np.float32(math.inf))
            else:
                upper = float(thresholds[interval])
                candidate = np.float32(upper)
                if float(candidate) > upper:
                    candidate = np.nextafter(candidate, np.float32(-math.inf))
                if interval > 0 and float(candidate) <= float(thresholds[interval - 1]):
                    raise ValueError("feature codec interval has no float32 representative")
            value = float(candidate)
            vector[int(feature_index)] = value
        return vector.tolist()

    def describe(self) -> Dict[str, Any]:
        return {
            **self.metadata,
            "artifact_path": str(self.artifact_path),
            "artifact_bytes": self.artifact_path.stat().st_size,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a decision-preserving traffic feature codec."
    )
    parser.add_argument(
        "--cloud_model", default="models/cloud_coordinator_topology_fused.joblib"
    )
    parser.add_argument(
        "--output", default="models/traffic_tree_feature_codec_topology_v1.npz"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_codec_artifact(Path(args.cloud_model), Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
