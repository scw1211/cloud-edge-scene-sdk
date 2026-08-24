"""Canonical shared-observation objects for two-owner traffic road sets.

Primary sensor ownership remains mutually exclusive.  This module builds one
content-addressed, read-only observation from the complete cut-edge endpoint
union and mirrors that exact object to both subscribers.  Subscriber identity
is deliberately absent from every content/decision digest.
"""

import base64
import binascii
import hashlib
import json
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from traffic_system.risk_labels import RISK_CLASSES


OVERLAP_SCHEMA_VERSION = 2
OVERLAP_PREPROCESS_VERSION = "pems08-road-set-f32le-c-v1"
ROAD_SET_MODEL_ID = "current_window_overlap_risk"
ROAD_SET_MODEL_VERSION = "current-window-overlap-risk-v2"
ROAD_SET_POLICY_ID = "road_set_deterministic_policy"
ROAD_SET_POLICY_VERSION = "road-set-deterministic-policy-v1"
ROAD_SET_POLICY_DEFINITION = {
    "risk_source": "canonical_cut_edge_endpoint_union",
    "low": {"action_type": "no_action"},
    "medium": {
        "action_type": "traffic_advisory",
        "strategy": "shared_boundary_watch",
    },
    "high": {
        "action_type": "variable_speed_limit",
        "strategy": "shared_boundary_speed_harmonization",
        "target_speed_mph": 50,
        "duration_seconds": 300,
    },
    "severe": {
        "action_type": "variable_speed_limit",
        "strategy": "shared_boundary_speed_harmonization",
        "target_speed_mph": 35,
        "duration_seconds": 300,
    },
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


ROAD_SET_POLICY_ARTIFACT_SHA256 = canonical_sha256(
    ROAD_SET_POLICY_DEFINITION
)


def _risk_score(probabilities: np.ndarray) -> float:
    priorities = np.linspace(
        0.0, 1.0, len(RISK_CLASSES), dtype=np.float64
    )
    return float(np.dot(np.asarray(probabilities, dtype=np.float64), priorities))


def _canonical_action(risk_level: str) -> Dict[str, Any]:
    definition = dict(ROAD_SET_POLICY_DEFINITION[risk_level])
    action_type = str(definition.pop("action_type"))
    return {
        "action_type": action_type,
        "parameters": definition,
    }


def road_set_decision_from_feature_matrix(
    feature_matrix: np.ndarray,
) -> Dict[str, Any]:
    """Recompute the deterministic output from the portable feature matrix."""

    matrix = np.ascontiguousarray(feature_matrix, dtype=np.dtype("<f4"))
    expected_columns = len(RISK_CLASSES) + 5
    if (
        matrix.ndim != 2
        or matrix.shape[0] <= 0
        or matrix.shape[1] != expected_columns
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("canonical overlap feature matrix shape is invalid")
    probabilities = np.asarray(
        matrix[:, : len(RISK_CLASSES)], dtype=np.float64
    )
    if (
        np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(
            np.sum(probabilities, axis=1), 1.0, rtol=1e-5, atol=1e-5
        )
    ):
        raise ValueError("canonical overlap probabilities are invalid")
    scores = np.asarray(matrix[:, len(RISK_CLASSES)], dtype=np.float64)
    labels = np.argmax(probabilities, axis=1)
    risk_counts = {
        name: int(np.sum(labels == index))
        for index, name in enumerate(RISK_CLASSES)
    }
    risk_level = RISK_CLASSES[int(np.max(labels))]
    representative = probabilities[int(np.argmax(scores))]
    action = _canonical_action(risk_level)
    output = {
        "risk_level": risk_level,
        "risk_score": round(_risk_score(representative), 6),
        "confidence": round(float(np.max(representative)), 6),
        "node_risk_counts": risk_counts,
        "action": action,
    }
    return {
        "output": output,
        "output_digest": canonical_sha256(output),
        "action": action,
        "action_digest": canonical_sha256(action),
    }


def _hex_sha256(value: Any, field: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("{} must be a SHA-256 hex digest".format(field))
    return normalized


def _decode_base64(value: Any, field: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("{} must be non-empty base64".format(field))
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("{} is invalid base64".format(field)) from error


def validate_canonical_overlap_observation(
    observation: Mapping[str, Any],
    *,
    require_raw: bool = False,
    expected_model_artifact_sha256: str = "",
) -> Dict[str, Any]:
    """Validate a full edge observation or a feature-only cloud bundle.

    The feature callback intentionally omits the raw float window.  Its cache
    capability authenticates the source-content claim, while this validator
    still verifies the feature, output, action, policy and identity digests.
    A full edge/sidecar observation additionally recomputes the source digest
    from the canonical header and raw C-order bytes.
    """

    value = dict(observation)
    if (
        int(value.get("schema_version", 0)) != OVERLAP_SCHEMA_VERSION
        or value.get("kind") != "canonical_shared_overlap_subscription"
        or value.get("subscription_semantics")
        != "read_only_mirrored_boundary_observation"
        or value.get("primary_sensor_ownership_changed") is not False
    ):
        raise ValueError("canonical overlap observation contract is invalid")
    road_set_id = str(value.get("road_set_id", "")).strip()
    if not road_set_id:
        raise ValueError("canonical overlap observation requires road_set_id")
    nodes = [int(node) for node in value.get("global_node_ids", [])]
    shape = [int(dimension) for dimension in value.get("shape", [])]
    if (
        nodes != sorted(set(nodes))
        or len(shape) != 3
        or shape[0] != len(nodes)
        or any(dimension <= 0 for dimension in shape)
        or value.get("dtype") != "float32_le"
        or value.get("memory_order") != "C"
        or value.get("preprocess_version") != OVERLAP_PREPROCESS_VERSION
    ):
        raise ValueError("canonical overlap array identity is invalid")
    source_dataset_sha256 = _hex_sha256(
        value.get("source_dataset_sha256"), "source_dataset_sha256"
    )
    raw_fragment_sha256 = _hex_sha256(
        value.get("raw_fragment_sha256"), "raw_fragment_sha256"
    )
    source_content_sha256 = _hex_sha256(
        value.get("source_content_sha256"), "source_content_sha256"
    )
    feature_content_sha256 = _hex_sha256(
        value.get("feature_content_sha256"), "feature_content_sha256"
    )
    model_artifact_sha256 = _hex_sha256(
        value.get("model_artifact_sha256"), "model_artifact_sha256"
    )
    if expected_model_artifact_sha256 and model_artifact_sha256 != _hex_sha256(
        expected_model_artifact_sha256,
        "expected_model_artifact_sha256",
    ):
        raise ValueError("canonical overlap model artifact is not approved")
    _hex_sha256(value.get("policy_artifact_sha256"), "policy_artifact_sha256")
    _hex_sha256(value.get("output_digest"), "output_digest")
    _hex_sha256(value.get("action_digest"), "action_digest")
    if (
        value.get("model_id") != ROAD_SET_MODEL_ID
        or value.get("model_version") != ROAD_SET_MODEL_VERSION
        or value.get("policy_id") != ROAD_SET_POLICY_ID
        or value.get("policy_version") != ROAD_SET_POLICY_VERSION
        or value.get("policy_artifact_sha256")
        != ROAD_SET_POLICY_ARTIFACT_SHA256
        or value.get("deterministic_decoding") is not True
    ):
        raise ValueError("canonical overlap model or policy identity is invalid")

    feature_bytes = _decode_base64(
        value.get("feature_bytes_base64"), "feature_bytes_base64"
    )
    feature_shape = [
        int(dimension) for dimension in value.get("feature_shape", [])
    ]
    if (
        value.get("feature_encoding")
        != "road_set_node_risk_features_f32le_c_v2"
        or len(feature_shape) != 2
        or feature_shape[0] != len(nodes)
        or feature_shape[1] != len(RISK_CLASSES) + 5
        or len(feature_bytes) != int(value.get("feature_bytes", -1))
        or len(feature_bytes) != 4 * feature_shape[0] * feature_shape[1]
        or hashlib.sha256(feature_bytes).hexdigest()
        != feature_content_sha256
    ):
        raise ValueError("canonical overlap feature payload is invalid")

    output = value.get("output")
    action = value.get("action")
    if not isinstance(output, Mapping) or not isinstance(action, Mapping):
        raise ValueError("canonical overlap decision payload is invalid")
    if output.get("action") != action:
        raise ValueError("canonical overlap output/action content disagrees")
    if canonical_sha256(output) != value.get("output_digest"):
        raise ValueError("canonical overlap output digest is invalid")
    if canonical_sha256(action) != value.get("action_digest"):
        raise ValueError("canonical overlap action digest is invalid")
    if str(output.get("risk_level", "")) not in RISK_CLASSES:
        raise ValueError("canonical overlap risk level is invalid")
    decoded_feature_matrix = np.frombuffer(
        feature_bytes, dtype=np.dtype("<f4")
    ).reshape(feature_shape)
    recomputed = road_set_decision_from_feature_matrix(
        decoded_feature_matrix
    )
    for field in ("output", "output_digest", "action", "action_digest"):
        if value.get(field) != recomputed[field]:
            raise ValueError(
                "canonical overlap {} was not recomputed from features".format(
                    field
                )
            )
    value["cloud_feature_recomputation_verified"] = True
    value["raw_fragment_digest_recomputed"] = False

    has_raw = "raw_fragment_base64" in value
    if require_raw and not has_raw:
        raise ValueError("canonical overlap raw fragment is required")
    if has_raw:
        raw_bytes = _decode_base64(
            value.get("raw_fragment_base64"), "raw_fragment_base64"
        )
        if (
            len(raw_bytes) != int(value.get("raw_fragment_bytes", -1))
            or len(raw_bytes) != 4 * int(np.prod(shape))
            or hashlib.sha256(raw_bytes).hexdigest()
            != raw_fragment_sha256
        ):
            raise ValueError("canonical overlap raw fragment is invalid")
        digest_header = {
            "schema_version": OVERLAP_SCHEMA_VERSION,
            "dataset": str(value.get("dataset", "")),
            "split": str(value.get("split", "")),
            "sample_id": int(value.get("sample_id", -1)),
            "window_id": str(value.get("window_id", "")),
            "road_set_id": road_set_id,
            "global_node_ids": nodes,
            "dtype": "float32_le",
            "shape": shape,
            "memory_order": "C",
            "preprocess_version": OVERLAP_PREPROCESS_VERSION,
            "source_dataset_sha256": source_dataset_sha256,
        }
        if hashlib.sha256(
            canonical_json_bytes(digest_header) + b"\x00" + raw_bytes
        ).hexdigest() != source_content_sha256:
            raise ValueError("canonical overlap source-content digest is invalid")
        value["raw_fragment_digest_recomputed"] = True
    expected_observation_id = "road_set_observation_{}".format(
        source_content_sha256[:32]
    )
    if value.get("observation_id") != expected_observation_id:
        raise ValueError("canonical overlap observation identity is invalid")
    if value.get("resource_id") != "traffic_road_set:{}".format(road_set_id):
        raise ValueError("canonical overlap resource identity is invalid")
    return value


def build_canonical_overlap_observation(
    road_set: Mapping[str, Any],
    raw_fragment: np.ndarray,
    global_node_ids: Sequence[int],
    rule_config: Mapping[str, Any],
    *,
    dataset: str,
    split: str,
    sample_id: int,
    source_dataset_sha256: str,
    model_artifact_sha256: str,
) -> Dict[str, Any]:
    """Build one deterministic object from an already-selected raw fragment."""

    road_set_id = str(road_set.get("road_set_id", "")).strip()
    if not road_set_id:
        raise ValueError("canonical overlap observation requires road_set_id")
    nodes = [int(value) for value in global_node_ids]
    expected_nodes = [int(value) for value in road_set.get("member_nodes", [])]
    if nodes != sorted(nodes) or nodes != expected_nodes:
        raise ValueError(
            "canonical overlap nodes must equal the sorted cut-edge endpoint union"
        )
    fragment = np.ascontiguousarray(raw_fragment, dtype=np.dtype("<f4"))
    if fragment.ndim != 3 or fragment.shape[0] != len(nodes):
        raise ValueError("canonical overlap fragment shape does not match nodes")
    if not np.isfinite(fragment).all():
        raise ValueError("canonical overlap fragment contains NaN or Inf")
    if not source_dataset_sha256 or len(source_dataset_sha256) != 64:
        raise ValueError("canonical overlap source dataset SHA-256 is invalid")
    if not model_artifact_sha256 or len(model_artifact_sha256) != 64:
        raise ValueError("canonical overlap model artifact SHA-256 is invalid")

    raw_bytes = fragment.tobytes(order="C")
    window_id = "{}:{}:{}".format(dataset, split, int(sample_id))
    digest_header = {
        "schema_version": OVERLAP_SCHEMA_VERSION,
        "dataset": str(dataset),
        "split": str(split),
        "sample_id": int(sample_id),
        "window_id": window_id,
        "road_set_id": road_set_id,
        "global_node_ids": nodes,
        "dtype": "float32_le",
        "shape": [int(value) for value in fragment.shape],
        "memory_order": "C",
        "preprocess_version": OVERLAP_PREPROCESS_VERSION,
        "source_dataset_sha256": source_dataset_sha256,
    }
    raw_fragment_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    source_content_sha256 = hashlib.sha256(
        canonical_json_bytes(digest_header) + b"\x00" + raw_bytes
    ).hexdigest()

    # Import lazily to avoid a module cycle: the perception runtime calls this
    # builder only after its own risk function has been defined.
    from traffic_system.current_state_perception_runtime import current_window_risk

    state = current_window_risk(fragment, rule_config)
    feature_matrix = np.concatenate(
        [
            np.asarray(state["probabilities"], dtype=np.float32),
            np.stack(
                [
                    np.asarray(state["scores"], dtype=np.float32),
                    np.asarray(state["flow_mean"], dtype=np.float32),
                    np.asarray(state["occupancy_mean"], dtype=np.float32),
                    np.asarray(state["speed_mean"], dtype=np.float32),
                    np.asarray(state["speed_min"], dtype=np.float32),
                ],
                axis=1,
            ),
        ],
        axis=1,
    ).astype(np.dtype("<f4"), copy=False)
    feature_bytes = np.ascontiguousarray(feature_matrix).tobytes(order="C")
    feature_content_sha256 = hashlib.sha256(feature_bytes).hexdigest()
    recomputed = road_set_decision_from_feature_matrix(feature_matrix)
    observation_id = "road_set_observation_{}".format(
        source_content_sha256[:32]
    )
    observation = {
        "schema_version": OVERLAP_SCHEMA_VERSION,
        "kind": "canonical_shared_overlap_subscription",
        "subscription_semantics": "read_only_mirrored_boundary_observation",
        "primary_sensor_ownership_changed": False,
        "road_set_id": road_set_id,
        "resource_id": str(road_set["resource_id"]),
        "required_regions": list(road_set["required_regions"]),
        "global_node_ids": nodes,
        "window_id": window_id,
        "observation_id": observation_id,
        "dataset": str(dataset),
        "split": str(split),
        "sample_id": int(sample_id),
        "dtype": "float32_le",
        "shape": [int(value) for value in fragment.shape],
        "memory_order": "C",
        "preprocess_version": OVERLAP_PREPROCESS_VERSION,
        "source_dataset_sha256": source_dataset_sha256,
        "raw_fragment_sha256": raw_fragment_sha256,
        "source_content_sha256": source_content_sha256,
        "raw_fragment_bytes": len(raw_bytes),
        "raw_fragment_base64": base64.b64encode(raw_bytes).decode("ascii"),
        "feature_encoding": "road_set_node_risk_features_f32le_c_v2",
        "feature_shape": [int(value) for value in feature_matrix.shape],
        "feature_content_sha256": feature_content_sha256,
        "feature_bytes": len(feature_bytes),
        "feature_bytes_base64": base64.b64encode(feature_bytes).decode("ascii"),
        "model_id": ROAD_SET_MODEL_ID,
        "model_version": ROAD_SET_MODEL_VERSION,
        "model_artifact_sha256": model_artifact_sha256,
        "policy_id": ROAD_SET_POLICY_ID,
        "policy_version": ROAD_SET_POLICY_VERSION,
        "policy_artifact_sha256": ROAD_SET_POLICY_ARTIFACT_SHA256,
        "deterministic_decoding": True,
        "output": recomputed["output"],
        "output_digest": recomputed["output_digest"],
        "action": recomputed["action"],
        "action_digest": recomputed["action_digest"],
    }
    return validate_canonical_overlap_observation(
        observation,
        require_raw=True,
        expected_model_artifact_sha256=model_artifact_sha256,
    )


def recompute_canonical_overlap_observation_from_raw(
    observation: Mapping[str, Any],
    rule_config: Mapping[str, Any],
    expected_model_artifact_sha256: str,
) -> Dict[str, Any]:
    """Rebuild the complete signed-identity object from pulled raw bytes."""

    declared = dict(observation)
    raw_bytes = _decode_base64(
        declared.get("raw_fragment_base64"), "raw_fragment_base64"
    )
    shape = [int(value) for value in declared.get("shape", [])]
    if len(shape) != 3 or len(raw_bytes) != 4 * int(np.prod(shape)):
        raise ValueError("pulled canonical overlap raw shape is invalid")
    raw_fragment = np.frombuffer(
        raw_bytes, dtype=np.dtype("<f4")
    ).reshape(shape)
    road_set = {
        "road_set_id": str(declared.get("road_set_id", "")),
        "resource_id": str(declared.get("resource_id", "")),
        "required_regions": list(declared.get("required_regions", [])),
        "member_nodes": [int(value) for value in declared.get("global_node_ids", [])],
    }
    rebuilt = build_canonical_overlap_observation(
        road_set,
        raw_fragment,
        road_set["member_nodes"],
        rule_config,
        dataset=str(declared.get("dataset", "")),
        split=str(declared.get("split", "")),
        sample_id=int(declared.get("sample_id", -1)),
        source_dataset_sha256=str(
            declared.get("source_dataset_sha256", "")
        ),
        model_artifact_sha256=expected_model_artifact_sha256,
    )
    compared_fields = (
        "schema_version",
        "kind",
        "subscription_semantics",
        "primary_sensor_ownership_changed",
        "road_set_id",
        "resource_id",
        "required_regions",
        "global_node_ids",
        "window_id",
        "observation_id",
        "dataset",
        "split",
        "sample_id",
        "dtype",
        "shape",
        "memory_order",
        "preprocess_version",
        "source_dataset_sha256",
        "raw_fragment_sha256",
        "source_content_sha256",
        "raw_fragment_bytes",
        "raw_fragment_base64",
        "feature_encoding",
        "feature_shape",
        "feature_content_sha256",
        "feature_bytes",
        "feature_bytes_base64",
        "model_id",
        "model_version",
        "model_artifact_sha256",
        "policy_id",
        "policy_version",
        "policy_artifact_sha256",
        "deterministic_decoding",
        "output",
        "output_digest",
        "action",
        "action_digest",
    )
    mismatched = [
        field
        for field in compared_fields
        if declared.get(field) != rebuilt.get(field)
    ]
    if mismatched:
        raise ValueError(
            "pulled raw overlap object failed deterministic recomputation: {}".format(
                ", ".join(mismatched)
            )
        )
    rebuilt["raw_fragment_digest_recomputed"] = True
    rebuilt["cloud_feature_recomputation_verified"] = True
    rebuilt["deterministic_policy_recomputed"] = True
    rebuilt["recomputation_source"] = "canonical_raw_fragment"
    return rebuilt
