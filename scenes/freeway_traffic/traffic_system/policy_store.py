"""用途：创建、校验并原子更新带版本和校验和的边缘交通策略包。"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Tuple


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def checksum_payload(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version_key(value: Any) -> Tuple[int, ...]:
    parts = [int(part) for part in re.findall(r"\d+", str(value))]
    if not parts:
        raise ValueError("Policy version must contain at least one number.")
    return tuple(parts)


def validate_artifact_manifest(artifacts: Any) -> Tuple[bool, str]:
    if artifacts is None:
        return True, "valid"
    if not isinstance(artifacts, list):
        return False, "artifacts_not_array"
    seen_paths = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            return False, "artifact_{}_not_object".format(index)
        required = {"role", "path", "size_bytes", "sha256"}
        missing = sorted(required - set(artifact))
        if missing:
            return False, "artifact_{}_missing:{}".format(index, ",".join(missing))
        relative = PurePosixPath(str(artifact["path"]))
        if relative.is_absolute() or ".." in relative.parts or str(relative) in {"", "."}:
            return False, "artifact_{}_unsafe_path".format(index)
        if str(relative) in seen_paths:
            return False, "artifact_{}_duplicate_path".format(index)
        seen_paths.add(str(relative))
        try:
            size_bytes = int(artifact["size_bytes"])
        except (TypeError, ValueError):
            return False, "artifact_{}_invalid_size".format(index)
        if size_bytes < 0:
            return False, "artifact_{}_invalid_size".format(index)
        if not re.fullmatch(r"[0-9a-f]{64}", str(artifact["sha256"])):
            return False, "artifact_{}_invalid_sha256".format(index)
        if str(artifact.get("target", "edge")) not in {"edge", "cloud", "all"}:
            return False, "artifact_{}_invalid_target".format(index)
    return True, "valid"


def build_bundle(version: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    version_key(version)
    envelope = {
        "schema_version": 1,
        "policy_version": str(version),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    envelope["checksum_sha256"] = checksum_payload(envelope)
    return envelope


def verify_bundle(bundle: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(bundle, dict):
        return False, "bundle_not_object"
    required = {"schema_version", "policy_version", "created_at_utc", "payload", "checksum_sha256"}
    missing = sorted(required - set(bundle))
    if missing:
        return False, "missing_fields:" + ",".join(missing)
    try:
        schema_version = int(bundle.get("schema_version", 0))
    except (TypeError, ValueError):
        return False, "unsupported_schema_version"
    if schema_version != 1:
        return False, "unsupported_schema_version"
    try:
        version_key(bundle["policy_version"])
    except ValueError:
        return False, "invalid_policy_version"
    if not isinstance(bundle.get("payload"), dict):
        return False, "payload_not_object"
    artifacts_valid, artifacts_reason = validate_artifact_manifest(
        bundle["payload"].get("artifacts")
    )
    if not artifacts_valid:
        return False, artifacts_reason
    signed = {key: value for key, value in bundle.items() if key != "checksum_sha256"}
    expected = checksum_payload(signed)
    if str(bundle.get("checksum_sha256")) != expected:
        return False, "checksum_mismatch"
    return True, "valid"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError("JSON must contain an object: {}".format(path))
    return value


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file_obj:
            json.dump(value, file_obj, ensure_ascii=False, indent=2)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


class PolicyStore:
    def __init__(self, current_path: Path) -> None:
        self.current_path = current_path

    def current(self) -> Optional[Dict[str, Any]]:
        if not self.current_path.exists():
            return None
        bundle = load_json(self.current_path)
        valid, reason = verify_bundle(bundle)
        if not valid:
            raise ValueError("Current policy is invalid: {}".format(reason))
        return bundle

    def apply(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        valid, reason = verify_bundle(candidate)
        current = self.current()
        current_version = str(current.get("policy_version")) if current else None
        if not valid:
            return {"applied": False, "reason": reason, "current_version": current_version}
        candidate_version = str(candidate["policy_version"])
        if current and version_key(candidate_version) <= version_key(current_version):
            return {
                "applied": False,
                "reason": "stale_or_equal_version",
                "candidate_version": candidate_version,
                "current_version": current_version,
            }
        if current:
            atomic_write_json(self.current_path.with_suffix(self.current_path.suffix + ".bak"), current)
        atomic_write_json(self.current_path, candidate)
        return {
            "applied": True,
            "reason": "atomic_update_complete",
            "candidate_version": candidate_version,
            "previous_version": current_version,
            "current_version": candidate_version,
        }

    def rollback(self) -> Dict[str, Any]:
        backup_path = self.current_path.with_suffix(self.current_path.suffix + ".bak")
        current = self.current()
        current_version = str(current["policy_version"]) if current else None
        if not backup_path.exists():
            return {
                "rolled_back": False,
                "reason": "backup_not_found",
                "current_version": current_version,
            }
        backup = load_json(backup_path)
        valid, reason = verify_bundle(backup)
        if not valid:
            return {
                "rolled_back": False,
                "reason": "invalid_backup:" + reason,
                "current_version": current_version,
            }
        atomic_write_json(self.current_path, backup)
        return {
            "rolled_back": True,
            "reason": "rollback_complete",
            "previous_version": current_version,
            "current_version": str(backup["policy_version"]),
        }


def default_payload(project_root: Path) -> Dict[str, Any]:
    astgcn_path = (
        project_root
        / "experiments/PEMS08/joint_risk_astgcn_metis4_flowprio2_frozen/best.pt"
    )
    astgcn_config_path = project_root / "configurations/PEMS08_astgcn.conf"
    student_path = project_root / "models/edge_student_freeway_joint_metis4.json"
    risk_calibrator_path = project_root / "models/region_risk_conformal.json"
    defer_gate_path = project_root / "models/edge_defer_gate.npz"
    qwen_path = project_root / "models/gguf/qwen35_0_8b_freeway_action_token_v9_text_only_q6_k.gguf"
    cloud_path = project_root / "models/cloud_coordinator_future_calibrated.joblib"
    runtime_config_path = project_root / "deployment/edge/edge_runtime_config.json"
    artifacts = []
    for role, target, path in (
        ("edge_perception_model", "edge", astgcn_path),
        ("edge_perception_config", "edge", astgcn_config_path),
        ("realtime_student", "edge", student_path),
        ("region_risk_calibrator", "edge", risk_calibrator_path),
        ("selective_defer_gate", "edge", defer_gate_path),
        ("cloud_coordinator", "cloud", cloud_path),
        ("edge_llm_student", "edge", qwen_path),
        ("edge_runtime_config", "edge", runtime_config_path),
    ):
        if path.exists():
            artifacts.append(
                {
                    "role": role,
                    "target": target,
                    "path": str(path.relative_to(project_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": artifact_sha256(path),
                }
            )
    return {
        "scene": "freeway_traffic_management",
        "scheduler": {
            "deadline_ms": 200.0,
            "confidence_threshold": 0.70,
            "edge_compute_ms": 52.0,
            "cloud_compute_ms": 12.0,
            "routing_revision": "conformal_selective_defer_v1",
            "calibration_source": "temporally_isolated_validation_future_state_reference",
            "risk_scope": "region_head_with_node_distribution_in_gate",
            "severe_always_reviewed": True,
        },
        "selective_defer": {
            "enabled": True,
            "gate": "models/edge_defer_gate.npz",
            "risk_calibrator": "models/region_risk_conformal.json",
            "conformal_method": "marginal_aps",
            "local_experts": ["fixed_safety_policy", "distilled_edge_student"],
        },
        "edge_llm": {
            "modality": "text_only",
            "runtime": "llama.cpp_gpu",
            "input_encoding": "bitpacked_decimal",
            "context_tokens": 16,
            "output_tokens": 1,
            "trigger_confidence_below": 0.75,
            "prompt_cache": False,
        },
        "conflict_limits": {
            "max_vsl_delta_mph": 10.0,
            "max_ramp_delta_veh_per_hour": 180.0,
            "max_combined_diversion_ratio": 0.5,
        },
        "safety_limits": {
            "target_speed_mph": [25, 65],
            "metering_rate_veh_per_hour": [240, 900],
            "diversion_ratio": [0.05, 0.40],
        },
        "artifacts": artifacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create, verify, or apply a traffic policy bundle.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--version", default="1.0.0")
    create.add_argument("--output", default="deployment/policy/traffic_policy_candidate.json")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--bundle", required=True)
    apply_parser.add_argument("--current", default="deployment/policy/current_policy.json")
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--current", default="deployment/policy/current_policy.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if args.command == "create":
        bundle = build_bundle(args.version, default_payload(project_root))
        atomic_write_json(project_root / args.output, bundle)
        print("created:", args.output)
        print("checksum:", bundle["checksum_sha256"])
    elif args.command == "verify":
        valid, reason = verify_bundle(load_json(Path(args.bundle)))
        print(json.dumps({"valid": valid, "reason": reason}, ensure_ascii=False))
        if not valid:
            raise SystemExit(1)
    elif args.command == "apply":
        result = PolicyStore(Path(args.current)).apply(load_json(Path(args.bundle)))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["applied"]:
            raise SystemExit(1)
    else:
        result = PolicyStore(Path(args.current)).rollback()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["rolled_back"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
