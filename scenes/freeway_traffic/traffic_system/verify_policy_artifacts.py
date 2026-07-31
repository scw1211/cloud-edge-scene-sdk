"""用途：在部署节点逐项核验策略清单中的制品大小、SHA-256 和目标范围。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from traffic_system.decision_utils import save_json
from traffic_system.policy_store import artifact_sha256, verify_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify deployed policy artifacts.")
    parser.add_argument("--policy", default="deployment/policy/current_policy.json")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--target", default="edge", choices=["edge", "cloud", "all"])
    parser.add_argument("--output_json", default="results/edge/policy_artifact_verify.json")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Policy bundle must be a JSON object")
    return value


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    bundle = load_json(Path(args.policy))
    bundle_valid, bundle_reason = verify_bundle(bundle)
    records: List[Dict[str, Any]] = []
    for artifact in bundle.get("payload", {}).get("artifacts", []):
        if args.target != "all" and artifact.get("target") != args.target:
            continue
        path = root / str(artifact["path"])
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        size_matches = exists and actual_size == int(artifact["size_bytes"])
        actual_sha256 = artifact_sha256(path) if size_matches else None
        sha256_matches = actual_sha256 == str(artifact["sha256"])
        records.append(
            {
                "role": artifact["role"],
                "target": artifact["target"],
                "path": artifact["path"],
                "exists": exists,
                "size_matches": size_matches,
                "sha256_matches": sha256_matches,
                "actual_size_bytes": actual_size,
                "actual_sha256": actual_sha256,
            }
        )
    all_valid = bool(
        bundle_valid
        and records
        and all(row["exists"] and row["size_matches"] and row["sha256_matches"] for row in records)
    )
    result = {
        "task": "deployed_policy_artifact_verification",
        "policy_version": bundle.get("policy_version"),
        "target": args.target,
        "bundle_valid": bundle_valid,
        "bundle_reason": bundle_reason,
        "artifact_count": len(records),
        "all_artifacts_valid": all_valid,
        "artifacts": records,
    }
    save_json(result, Path(args.output_json))
    print(json.dumps({key: result[key] for key in (
        "policy_version", "bundle_valid", "artifact_count", "all_artifacts_valid"
    )}, ensure_ascii=False, indent=2))
    if not all_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
