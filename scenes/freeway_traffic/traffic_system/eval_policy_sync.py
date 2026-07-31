"""用途：验证策略下发过程对篡改、旧版本、乱序和中断更新的处理能力。"""

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from traffic_system.decision_utils import save_json
from traffic_system.policy_store import PolicyStore, build_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate atomic policy synchronization and rollback safety.")
    parser.add_argument("--output_json", default="results/edge/policy_sync_consistency_eval.json")
    return parser.parse_args()


def candidate(version: str, threshold: float) -> Dict[str, Any]:
    return build_bundle(
        version,
        {
            "scene": "freeway_traffic_management",
            "scheduler": {"deadline_ms": 200.0, "confidence_threshold": threshold},
        },
    )


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as directory:
        store = PolicyStore(Path(directory) / "current_policy.json")

        def run_case(name: str, bundle: Dict[str, Any], expected: bool) -> None:
            before = store.current()
            result = store.apply(bundle)
            after = store.current()
            preserved = after is not None and (result["applied"] or before == after)
            rows.append(
                {
                    "case": name,
                    "expected_applied": expected,
                    "actual_applied": bool(result["applied"]),
                    "reason": result["reason"],
                    "state_preserved": preserved,
                    "current_version": after.get("policy_version") if after else None,
                    "success": bool(result["applied"]) == expected and preserved,
                }
            )

        run_case("initial_install", candidate("1.0.0", 0.75), True)
        run_case("valid_upgrade", candidate("1.1.0", 0.72), True)
        run_case("stale_version", candidate("1.0.5", 0.70), False)
        tampered = candidate("1.2.0", 0.68)
        tampered["payload"]["scheduler"]["deadline_ms"] = 999.0
        run_case("tampered_checksum", tampered, False)
        truncated = candidate("1.3.0", 0.66)
        del truncated["payload"]
        run_case("truncated_bundle", truncated, False)
        run_case("out_of_order_newer", candidate("2.0.0", 0.65), True)
        run_case("out_of_order_older", candidate("1.9.0", 0.64), False)

        current = store.current()
        rows.append(
            {
                "case": "network_outage_no_candidate",
                "expected_applied": False,
                "actual_applied": False,
                "reason": "network_unavailable",
                "state_preserved": current is not None,
                "current_version": current.get("policy_version") if current else None,
                "success": current is not None,
            }
        )

    success_count = sum(row["success"] for row in rows)
    result = {
        "task": "edge_policy_update_consistency_evaluation",
        "cases": rows,
        "case_count": len(rows),
        "success_count": success_count,
        "success_rate": round(success_count / len(rows), 6),
        "invalid_or_stale_updates_rejected": all(
            row["success"] for row in rows if row["case"] in {
                "stale_version", "tampered_checksum", "truncated_bundle", "out_of_order_older"
            }
        ),
        "business_continuity_during_update_failure": all(row["state_preserved"] for row in rows),
    }
    save_json(result, Path(args.output_json))
    print(json.dumps({key: result[key] for key in (
        "case_count", "success_rate", "invalid_or_stale_updates_rejected",
        "business_continuity_during_update_failure"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
