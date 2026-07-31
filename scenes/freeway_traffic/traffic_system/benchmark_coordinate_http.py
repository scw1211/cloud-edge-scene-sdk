"""用途：通过真实 HTTP 提交同一时刻的多区域交通事件并记录冲突协调闭环。"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

from cloud_edge_framework.evidence import EvidencePlanner
from cloud_edge_framework.registry import build_default_registry
from cloud_edge_framework.transport import HttpCloudClient
from traffic_system.scene_event import traffic_envelope_from_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark real multi-edge HTTP coordination.")
    parser.add_argument(
        "--event_glob",
        default="datasets/freeway_events_joint_metis4/freeway_test_sample_0048_*.json",
    )
    parser.add_argument("--cloud_base_url", default="http://127.0.0.1:18100")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--output",
        default="results/framework/traffic_http_coordinate_conflict.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("event file must contain an object: {}".format(path))
    return value


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = sorted(root.glob(args.event_glob))
    if len(paths) < 2:
        raise ValueError("event_glob must match at least two edge events")

    registry = build_default_registry(root)
    planner = EvidencePlanner()
    compact_events = []
    try:
        plugin = registry.get("traffic")
        for path in paths:
            event = plugin.normalize(traffic_envelope_from_output(load_json(path)))
            plan = planner.plan(event)
            selected_ids = set(plan.selected_evidence_ids)
            selected = replace(
                event,
                evidence=[
                    evidence
                    for evidence in event.evidence
                    if evidence.evidence_id in selected_ids
                ],
            )
            compact_events.append(
                plugin.prepare_cloud_event(selected, plan.required_level)
            )

        client = HttpCloudClient(args.cloud_base_url, args.timeout)
        started = time.perf_counter()
        coordination = client.coordinate(compact_events)
        wall_ms = (time.perf_counter() - started) * 1000.0
    finally:
        registry.close()

    output: Dict[str, Any] = {
        "task": "real_http_multi_edge_conflict_coordination",
        "input_files": [str(path.relative_to(root)) for path in paths],
        "event_count": len(compact_events),
        "client_wall_ms": round(wall_ms, 6),
        **coordination,
    }
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event_count": output["event_count"],
                "client_wall_ms": output["client_wall_ms"],
                "transport": output.get("transport"),
                "initial_conflict_count": output["initial_conflict_count"],
                "residual_conflict_count": output["residual_conflict_count"],
                "resolution_success_rate": output["resolution_success_rate"],
                "changes": output["changes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
