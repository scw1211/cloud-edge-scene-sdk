"""Versioned neighbor-aware traffic coordination policy.

This module deliberately sits *after* the local 16-token Qwen decision and is
independent from the framework route scheduler.  It consumes a compact neighbor
summary and a deterministic policy package; it never changes route selection,
never calls a cloud model per event, and never claims to update model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


RISK_LEVELS = ("low", "medium", "high", "severe")
RISK_CODE = {name: index for index, name in enumerate(RISK_LEVELS)}
TREND_LEVELS = ("falling", "stable", "rising")
TREND_CODE = {name: index for index, name in enumerate(TREND_LEVELS)}
ACTIONS = ("A", "B", "C", "D", "E", "F")
ACTION_CODE = {name: index for index, name in enumerate(ACTIONS)}
ACTION_TO_DECISION = {
    "A": "no_action",
    "B": "congestion_warning",
    "C": "variable_speed_limit",
    "D": "ramp_metering",
    "E": "regional_coordination",
    "F": "reroute",
}
DECISION_TO_ACTION = {value: key for key, value in ACTION_TO_DECISION.items()}

PACKAGE_PAYLOAD_FILES = (
    "policy.json",
    "feature_schema.json",
    "objective_config.json",
    "validation_results.json",
)
PACKAGE_FILES = PACKAGE_PAYLOAD_FILES + ("manifest.json", "sha256.txt")


def canonical_json(value: Any, *, indent: Optional[int] = None) -> bytes:
    if indent is None:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
        )
    return (rendered + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError("{} must be numeric".format(name))
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError("{} must be an integer".format(name))
    result = int(value)
    if result != value and not isinstance(value, str):
        raise ValueError("{} must be an integer".format(name))
    return result


@dataclass(frozen=True)
class NeighborContext:
    """Five-field aggregate.  Full neighbor sequences are intentionally absent."""

    neighbor_risk_max: str
    neighbor_risk_trend: str
    neighbor_remaining_capacity: float
    neighbor_planned_action: str
    snapshot_age_ms: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NeighborContext":
        if not isinstance(value, Mapping):
            raise ValueError("neighbor_context must be an object")
        forbidden = {
            "raw_sequence",
            "raw_series",
            "node_sequence",
            "neighbor_events",
            "sensor_readings",
        }
        leaked = sorted(forbidden.intersection(value))
        if leaked:
            raise ValueError(
                "neighbor_context must not contain raw neighbor data: {}".format(
                    ",".join(leaked)
                )
            )
        required = {
            "neighbor_risk_max",
            "neighbor_risk_trend",
            "neighbor_remaining_capacity",
            "neighbor_planned_action",
            "snapshot_age_ms",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(
                "neighbor_context missing fields: {}".format(",".join(missing))
            )
        risk = str(value["neighbor_risk_max"]).strip().lower()
        trend = str(value["neighbor_risk_trend"]).strip().lower()
        action = str(value["neighbor_planned_action"]).strip().upper()
        if risk not in RISK_CODE:
            raise ValueError("neighbor_risk_max is invalid")
        if trend not in TREND_CODE:
            raise ValueError("neighbor_risk_trend is invalid")
        if action not in ACTION_CODE:
            raise ValueError("neighbor_planned_action must be A-F")
        capacity = _finite_float(
            value["neighbor_remaining_capacity"],
            "neighbor_remaining_capacity",
        )
        if not 0.0 <= capacity <= 1.0:
            raise ValueError("neighbor_remaining_capacity must be in [0, 1]")
        age = _strict_int(value["snapshot_age_ms"], "snapshot_age_ms")
        if age < 0:
            raise ValueError("snapshot_age_ms must be non-negative")
        return cls(risk, trend, capacity, action, age)

    def compact_code(self) -> str:
        """Return a 10-digit, fixed-width, lossy transport summary.

        Layout: risk(1), trend(1), remaining-capacity percent(3), action(1),
        snapshot-age in 10 ms buckets(4).  The formal Q8 raw-decimal contract
        has one tokenizer token per ASCII digit, so this is also ten transport
        token equivalents.  It is *not* appended to the current Q8 prompt.
        """

        capacity_percent = max(
            0, min(100, int(round(self.neighbor_remaining_capacity * 100.0)))
        )
        age_bucket = max(0, min(9999, int(round(self.snapshot_age_ms / 10.0))))
        return "{}{}{:03d}{}{:04d}".format(
            RISK_CODE[self.neighbor_risk_max],
            TREND_CODE[self.neighbor_risk_trend],
            capacity_percent,
            ACTION_CODE[self.neighbor_planned_action],
            age_bucket,
        )

    def to_dict(self) -> Dict[str, Any]:
        code = self.compact_code()
        return {
            "neighbor_risk_max": self.neighbor_risk_max,
            "neighbor_risk_trend": self.neighbor_risk_trend,
            "neighbor_remaining_capacity": round(
                self.neighbor_remaining_capacity, 6
            ),
            "neighbor_planned_action": self.neighbor_planned_action,
            "snapshot_age_ms": self.snapshot_age_ms,
            "compact_code": code,
            "compact_encoding": "traffic-neighbor-decimal10/v1",
            "added_token_equivalents": len(code),
            "encoded_bytes": len(code.encode("ascii")),
            "q8_model_input_token_delta": 0,
        }


def summarize_neighbor_snapshots(
    snapshots: Sequence[Mapping[str, Any]],
) -> NeighborContext:
    """Reduce already-aggregated adjacent-region snapshots to five fields.

    Each input row is a current regional aggregate, not a time series.  The
    most conservative risk/trend, minimum remaining capacity, highest-priority
    planned action and oldest age are retained.  This function intentionally
    has no API for raw node samples.
    """

    if not isinstance(snapshots, Sequence) or isinstance(
        snapshots, (str, bytes)
    ):
        raise ValueError("neighbor snapshots must be a sequence")
    if not snapshots:
        raise ValueError("at least one neighbor snapshot is required")
    normalized = [NeighborContext.from_mapping(value) for value in snapshots]
    risk = max(normalized, key=lambda value: RISK_CODE[value.neighbor_risk_max])
    trend = max(
        normalized,
        key=lambda value: TREND_CODE[value.neighbor_risk_trend],
    )
    # Higher letters represent more globally coupled traffic interventions in
    # the frozen A-F decision contract, so keep the most coordination-sensitive
    # neighbor action.
    action = max(
        normalized,
        key=lambda value: ACTION_CODE[value.neighbor_planned_action],
    )
    return NeighborContext(
        neighbor_risk_max=risk.neighbor_risk_max,
        neighbor_risk_trend=trend.neighbor_risk_trend,
        neighbor_remaining_capacity=min(
            value.neighbor_remaining_capacity for value in normalized
        ),
        neighbor_planned_action=action.neighbor_planned_action,
        snapshot_age_ms=max(value.snapshot_age_ms for value in normalized),
    )


@dataclass(frozen=True)
class CoordinationRecommendation:
    local_action: str
    selected_action: str
    changed: bool
    reason_code: str
    policy_version: str
    neighbor_context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "local_action": self.local_action,
            "selected_action": self.selected_action,
            "changed": self.changed,
            "reason_code": self.reason_code,
            "policy_version": self.policy_version,
            "neighbor_context": dict(self.neighbor_context),
        }


class CoordinationPolicy:
    """Deterministic edge policy selected by frozen cloud replay."""

    def __init__(self, policy: Mapping[str, Any]) -> None:
        if not isinstance(policy, Mapping):
            raise ValueError("coordination policy must be an object")
        if policy.get("schema_version") != "coordination-policy/v1":
            raise ValueError("unsupported coordination policy schema")
        version = str(policy.get("policy_version", "")).strip()
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", version):
            raise ValueError("coordination policy version must be numeric dotted form")
        parameters = policy.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("coordination policy parameters are required")
        threshold = _finite_float(
            parameters.get("diversion_capacity_threshold"),
            "diversion_capacity_threshold",
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("diversion_capacity_threshold must be in [0, 1]")
        max_age = _strict_int(
            parameters.get("max_snapshot_age_ms"), "max_snapshot_age_ms"
        )
        if max_age <= 0:
            raise ValueError("max_snapshot_age_ms must be positive")
        conflict_matrix = parameters.get("action_conflict_matrix")
        if not isinstance(conflict_matrix, Mapping):
            raise ValueError("action_conflict_matrix must be an object")
        normalized_matrix: Dict[str, str] = {}
        for key, replacement in conflict_matrix.items():
            parts = str(key).split("/")
            if len(parts) != 2 or any(part not in ACTIONS for part in parts):
                raise ValueError("invalid action conflict matrix key")
            replacement = str(replacement).upper()
            if replacement not in ACTIONS:
                raise ValueError("invalid action conflict replacement")
            normalized_matrix[str(key)] = replacement
        for field in (
            "capacity_fallback_action",
            "rising_risk_fallback_action",
            "high_risk_neighbor_action",
        ):
            if str(parameters.get(field, "")).upper() not in ACTIONS:
                raise ValueError("{} must be A-F".format(field))
        for field in (
            "neighbor_load_penalty",
            "minimum_neighbor_load_score",
            "rising_trend_penalty",
            "trend_activation_threshold",
        ):
            number = _finite_float(parameters.get(field), field)
            if number < 0.0:
                raise ValueError("{} must be non-negative".format(field))
        biases = parameters.get("regional_coordination_bias")
        if not isinstance(biases, Mapping) or not biases:
            raise ValueError("regional_coordination_bias must be a non-empty object")
        normalized_biases: Dict[str, float] = {}
        for action, raw_bias in biases.items():
            normalized_action = str(action).upper()
            if normalized_action not in ACTIONS:
                raise ValueError("regional_coordination_bias action must be A-F")
            normalized_biases[normalized_action] = _finite_float(
                raw_bias, "regional_coordination_bias.{}".format(action)
            )
        self.payload = dict(policy)
        self.version = version
        self.parameters = dict(parameters)
        self.capacity_threshold = threshold
        self.max_snapshot_age_ms = max_age
        self.conflict_matrix = normalized_matrix
        self.coordination_bias = normalized_biases

    def _biased_fallback(self, configured: str) -> str:
        configured = str(configured).upper()
        candidates = sorted(set(self.coordination_bias) | {configured})
        return max(
            candidates,
            key=lambda action: (
                self.coordination_bias.get(action, 0.0),
                action == configured,
                -ACTION_CODE[action],
            ),
        )

    def recommend(
        self,
        local_action: str,
        neighbor_context: Mapping[str, Any],
    ) -> CoordinationRecommendation:
        action = str(local_action).strip().upper()
        if action not in ACTIONS:
            raise ValueError("local_action must be A-F")
        context = NeighborContext.from_mapping(neighbor_context)
        selected = action
        reason = "LOCAL_ACTION_RETAINED"
        if context.snapshot_age_ms > self.max_snapshot_age_ms:
            reason = "STALE_NEIGHBOR_SNAPSHOT_RETAIN_LOCAL"
        else:
            matrix_key = "{}/{}".format(
                action, context.neighbor_planned_action
            )
            if matrix_key in self.conflict_matrix:
                selected = self.conflict_matrix[matrix_key]
                reason = "ACTION_CONFLICT_MATRIX"
            elif (
                action == "F"
                and context.neighbor_remaining_capacity
                < self.capacity_threshold
                and float(self.parameters["neighbor_load_penalty"])
                * (1.0 - context.neighbor_remaining_capacity)
                >= float(self.parameters["minimum_neighbor_load_score"])
            ):
                selected = self._biased_fallback(
                    str(self.parameters["capacity_fallback_action"])
                )
                reason = "NEIGHBOR_CAPACITY_GUARD"
            elif (
                action == "F"
                and context.neighbor_risk_trend == "rising"
                and float(self.parameters["rising_trend_penalty"])
                >= float(self.parameters["trend_activation_threshold"])
                and RISK_CODE[context.neighbor_risk_max]
                >= RISK_CODE[
                    str(
                        self.parameters.get(
                            "rising_risk_min_level", "high"
                        )
                    ).lower()
                ]
            ):
                selected = self._biased_fallback(
                    str(self.parameters["rising_risk_fallback_action"])
                )
                reason = "NEIGHBOR_RISK_TREND_GUARD"
            elif (
                action in {"A", "B"}
                and context.neighbor_risk_trend == "rising"
                and RISK_CODE[context.neighbor_risk_max]
                >= RISK_CODE["high"]
            ):
                selected = self._biased_fallback(
                    str(self.parameters["high_risk_neighbor_action"])
                )
                reason = "NEIGHBOR_PROPAGATION_GUARD"
        return CoordinationRecommendation(
            local_action=action,
            selected_action=selected,
            changed=selected != action,
            reason_code=reason,
            policy_version=self.version,
            neighbor_context=context.to_dict(),
        )


def _load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def verify_policy_package(package_dir: Path) -> Dict[str, Any]:
    root = Path(package_dir).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("coordination policy package must be a real directory")
    observed = sorted(path.name for path in root.iterdir())
    if observed != sorted(PACKAGE_FILES):
        raise ValueError("coordination policy package file set is not closed")
    for name in PACKAGE_FILES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("coordination policy package contains a non-file")
    checksum_rows: Dict[str, str] = {}
    for raw_line in (root / "sha256.txt").read_text(encoding="utf-8").splitlines():
        if not raw_line:
            continue
        parts = raw_line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError("invalid sha256.txt row")
        digest, name = parts
        if name in checksum_rows or name not in PACKAGE_FILES[:-1]:
            raise ValueError("unexpected or duplicate sha256.txt path")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("invalid sha256.txt digest")
        checksum_rows[name] = digest
    if sorted(checksum_rows) != sorted(PACKAGE_FILES[:-1]):
        raise ValueError("sha256.txt does not cover the package payload")
    for name, expected in checksum_rows.items():
        if sha256_file(root / name) != expected:
            raise ValueError("coordination policy checksum mismatch: {}".format(name))
    manifest = _load_json(root / "manifest.json")
    if manifest.get("schema_version") != "coordination-policy-manifest/v1":
        raise ValueError("unsupported coordination policy manifest")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("coordination policy manifest files are required")
    manifest_files = {str(row.get("path")): row for row in files if isinstance(row, dict)}
    if sorted(manifest_files) != sorted(PACKAGE_PAYLOAD_FILES):
        raise ValueError("coordination policy manifest payload set differs")
    for name, row in manifest_files.items():
        path = root / name
        if int(row.get("size_bytes", -1)) != path.stat().st_size:
            raise ValueError("coordination policy manifest size mismatch")
        if str(row.get("sha256")) != sha256_file(path):
            raise ValueError("coordination policy manifest hash mismatch")
    total_bytes = sum((root / name).stat().st_size for name in PACKAGE_FILES)
    if int(manifest.get("package_size_bytes", -1)) != total_bytes:
        raise ValueError("coordination policy package size mismatch")
    policy = _load_json(root / "policy.json")
    validation = _load_json(root / "validation_results.json")
    feature_schema = _load_json(root / "feature_schema.json")
    objective = _load_json(root / "objective_config.json")
    if str(manifest.get("policy_version")) != str(policy.get("policy_version")):
        raise ValueError("coordination policy version binding mismatch")
    if feature_schema.get("schema_version") != "neighbor-context/v1":
        raise ValueError("unsupported neighbor feature schema")
    if objective.get("schema_version") != "coordination-objective/v1":
        raise ValueError("unsupported coordination objective")
    CoordinationPolicy(policy)
    activation = validation.get("activation")
    if not isinstance(activation, dict):
        raise ValueError("coordination validation activation record is required")
    criteria = activation.get("criteria")
    if not isinstance(criteria, dict) or not criteria:
        raise ValueError("coordination activation criteria are required")
    if any(type(value) is not bool for value in criteria.values()):
        raise ValueError("coordination activation criteria must be booleans")
    expected_activation = all(criteria.values())
    if type(activation.get("production_activation_allowed")) is not bool:
        raise ValueError("production_activation_allowed must be boolean")
    if activation["production_activation_allowed"] != expected_activation:
        raise ValueError("production activation result differs from its criteria")
    return {
        "root": str(root),
        "policy": policy,
        "validation": validation,
        "feature_schema": feature_schema,
        "objective_config": objective,
        "manifest": manifest,
        "package_size_bytes": total_bytes,
        "package_sha256": sha256_file(root / "sha256.txt"),
        "production_activation_allowed": bool(
            validation.get("activation", {}).get(
                "production_activation_allowed", False
            )
        ),
    }


class CoordinationPolicyCache:
    """Atomic edge cache retaining the last fully verified active package."""

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = Path(cache_root)
        self.packages_root = self.cache_root / "packages"
        self.current_path = self.cache_root / "current.json"

    def stage(self, package_dir: Path) -> Dict[str, Any]:
        verified = verify_policy_package(package_dir)
        version = str(verified["policy"]["policy_version"])
        self.packages_root.mkdir(parents=True, exist_ok=True)
        destination = self.packages_root / version
        if destination.exists():
            existing = verify_policy_package(destination)
            if existing["package_sha256"] != verified["package_sha256"]:
                raise ValueError("equal coordination policy version has different bytes")
            return {
                "staged": True,
                "cache_reused": True,
                "policy_version": version,
                "package_sha256": verified["package_sha256"],
            }
        temporary = Path(
            tempfile.mkdtemp(prefix=".coordination-policy-", dir=str(self.packages_root))
        )
        try:
            for name in PACKAGE_FILES:
                shutil.copyfile(str(Path(package_dir) / name), str(temporary / name))
            verify_policy_package(temporary)
            os.replace(str(temporary), str(destination))
        finally:
            if temporary.exists():
                shutil.rmtree(str(temporary))
        return {
            "staged": True,
            "cache_reused": False,
            "policy_version": version,
            "package_sha256": verified["package_sha256"],
        }

    def activate(self, version: str) -> Dict[str, Any]:
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", str(version)):
            raise ValueError("coordination policy version must be numeric dotted form")
        package_dir = self.packages_root / str(version)
        verified = verify_policy_package(package_dir)
        if not verified["production_activation_allowed"]:
            return {
                "activated": False,
                "reason": "PRODUCTION_GATE_NOT_PASSED",
                "policy_version": str(version),
            }
        pointer = {
            "schema_version": "coordination-policy-pointer/v1",
            "policy_version": str(version),
            "package_sha256": verified["package_sha256"],
        }
        self.cache_root.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=".current.", suffix=".json", dir=str(self.cache_root)
        )
        try:
            with os.fdopen(descriptor, "wb") as file_obj:
                file_obj.write(canonical_json(pointer, indent=2))
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(name, str(self.current_path))
        finally:
            if os.path.exists(name):
                os.unlink(name)
        return {
            "activated": True,
            "reason": "VERIFIED_POLICY_POINTER_UPDATED",
            "policy_version": str(version),
        }

    def current(self) -> Optional[Dict[str, Any]]:
        if not self.current_path.is_file():
            return None
        pointer = _load_json(self.current_path)
        if pointer.get("schema_version") != "coordination-policy-pointer/v1":
            raise ValueError("invalid coordination policy pointer")
        version = str(pointer.get("policy_version", ""))
        verified = verify_policy_package(self.packages_root / version)
        if verified["package_sha256"] != pointer.get("package_sha256"):
            raise ValueError("active coordination policy pointer hash mismatch")
        if not verified["production_activation_allowed"]:
            raise ValueError("active coordination policy no longer passes its frozen gate")
        return verified


def write_policy_package(
    output_dir: Path,
    policy: Mapping[str, Any],
    feature_schema: Mapping[str, Any],
    objective_config: Mapping[str, Any],
    validation_results: Mapping[str, Any],
    manifest_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write a deterministic, closed-world policy package.

    This helper is intentionally separate from cloud publication.  Publication
    can copy the verified package; it cannot alter any file without breaking the
    manifest and checksum envelope.
    """

    output = Path(output_dir)
    CoordinationPolicy(policy)
    if output.exists():
        raise FileExistsError("coordination policy output already exists")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".coordination-policy-v1-", dir=str(parent)))
    try:
        values = {
            "policy.json": dict(policy),
            "feature_schema.json": dict(feature_schema),
            "objective_config.json": dict(objective_config),
            "validation_results.json": dict(validation_results),
        }
        identities: List[Dict[str, Any]] = []
        for name in PACKAGE_PAYLOAD_FILES:
            body = canonical_json(values[name], indent=2)
            (staging / name).write_bytes(body)
            identities.append(
                {
                    "path": name,
                    "size_bytes": len(body),
                    "sha256": sha256_bytes(body),
                }
            )
        manifest: Dict[str, Any] = {
            "schema_version": "coordination-policy-manifest/v1",
            "policy_version": str(policy["policy_version"]),
            "update_kind": "decision_logic_parameters_only",
            "edge_model_parameters_changed": False,
            "collaboration_scheduler_changed": False,
            "files": identities,
            "package_size_bytes": 0,
        }
        if manifest_metadata:
            manifest["provenance"] = dict(manifest_metadata)
        for _ in range(16):
            manifest_body = canonical_json(manifest, indent=2)
            checksums = identities + [
                {
                    "path": "manifest.json",
                    "size_bytes": len(manifest_body),
                    "sha256": sha256_bytes(manifest_body),
                }
            ]
            checksum_body = "".join(
                "{}  {}\n".format(row["sha256"], row["path"])
                for row in checksums
            ).encode("utf-8")
            total = (
                sum(int(row["size_bytes"]) for row in identities)
                + len(manifest_body)
                + len(checksum_body)
            )
            if total == manifest["package_size_bytes"]:
                break
            manifest["package_size_bytes"] = total
        else:
            raise RuntimeError("coordination package size did not converge")
        manifest_body = canonical_json(manifest, indent=2)
        checksums = identities + [
            {
                "path": "manifest.json",
                "size_bytes": len(manifest_body),
                "sha256": sha256_bytes(manifest_body),
            }
        ]
        checksum_body = "".join(
            "{}  {}\n".format(row["sha256"], row["path"])
            for row in checksums
        ).encode("utf-8")
        (staging / "manifest.json").write_bytes(manifest_body)
        (staging / "sha256.txt").write_bytes(checksum_body)
        for name in PACKAGE_FILES:
            with (staging / name).open("rb") as file_obj:
                os.fsync(file_obj.fileno())
        os.replace(str(staging), str(output))
    finally:
        if staging.exists():
            shutil.rmtree(str(staging))
    return verify_policy_package(output)
