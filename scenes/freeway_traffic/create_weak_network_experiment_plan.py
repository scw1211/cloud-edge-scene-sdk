#!/usr/bin/env python3
"""在弱网实测开始前生成并冻结样本、故障和资产计划。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCENE_ROOT = Path(__file__).resolve().parent
SDK_ROOT = SCENE_ROOT.parents[1]
for _import_root in (SDK_ROOT, SCENE_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from benchmark_weak_network_retention import (  # noqa: E402
    PROFILE_ORDER,
    PROFILES,
    _git_identity,
    _hardware_id,
    _load_experiment_plan,
    _model_ids,
    _sha256,
    _utc_now,
    _write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成交通弱网预注册实验计划")
    parser.add_argument("--project-root", default=str(SDK_ROOT))
    parser.add_argument(
        "--manifest",
        default=str(
            SCENE_ROOT / "runtime" / "pems08_metis4_partitions" / "manifest.json"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--hardware-id", default="")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--samples-per-profile", type=int, default=25)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--business-deadline-ms", type=float, default=200.0)
    parser.add_argument("--aggregation-timeout-ms", type=int, default=150)
    parser.add_argument("--dispatch-lead-ms", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    scene_root = project_root / "scenes" / "freeway_traffic"
    manifest_path = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("plan output already exists: {}".format(output))
    if not manifest_path.is_file():
        raise FileNotFoundError("partition manifest not found: {}".format(manifest_path))
    if args.sample_start < 0 or args.samples_per_profile <= 0:
        raise ValueError("sample-start must be non-negative and count must be positive")
    if min(
        args.top_k,
        args.business_deadline_ms,
        args.aggregation_timeout_ms,
    ) <= 0:
        raise ValueError("top-k/deadline/aggregation timeout must be positive")
    if args.dispatch_lead_ms < 0 or args.seed < 0:
        raise ValueError("dispatch lead and seed must be non-negative")

    commit, dirty = _git_identity(project_root)
    if dirty:
        raise RuntimeError(
            "formal experiment plan requires a clean worktree: {}".format(dirty)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sorted(int(row["partition_id"]) for row in manifest["partitions"]) != list(
        range(4)
    ):
        raise ValueError("weak-network plan requires four METIS partitions")

    next_sample = int(args.sample_start)
    sample_ids = {}
    for profile in PROFILE_ORDER:
        sample_ids[profile] = list(
            range(next_sample, next_sample + int(args.samples_per_profile))
        )
        next_sample += int(args.samples_per_profile)
    plan = {
        "schema_version": "1.0",
        "plan_id": str(args.plan_id),
        "created_at": _utc_now(),
        "scene": "freeway_traffic",
        "git_commit": commit,
        "dataset_id": "pems08_metis4:{}".format(_sha256(manifest_path)),
        "model_ids": _model_ids(scene_root),
        "hardware_id": _hardware_id(args.hardware_id),
        "seed": int(args.seed),
        "profile_order": list(PROFILE_ORDER) + ["recovery"],
        "fault_profiles": {name: dict(PROFILES[name]) for name in PROFILE_ORDER},
        "sample_ids_by_profile": sample_ids,
        "business_deadline_ms": float(args.business_deadline_ms),
        "aggregation_timeout_ms": int(args.aggregation_timeout_ms),
        "dispatch_lead_ms": float(args.dispatch_lead_ms),
        "split": str(args.split),
        "top_k": int(args.top_k),
    }
    _write_json(output, plan)
    validated, digest = _load_experiment_plan(output)
    print(
        json.dumps(
            {
                "status": "plan_locked",
                "path": str(output),
                "sha256": digest,
                "plan_id": validated["plan_id"],
                "sample_count": sum(
                    len(values) for values in validated["sample_ids_by_profile"].values()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
