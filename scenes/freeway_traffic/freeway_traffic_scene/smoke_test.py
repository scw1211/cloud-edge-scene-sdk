"""用途：无服务器验证交通插件、双边缘汇聚、冲突消解和结果回填。"""

import json
from pathlib import Path

from cloud_edge_framework.registry import SceneRegistry
from cloud_edge_framework.runtime import CloudRuntime, EdgeRuntime
from cloud_edge_framework.scheduling import NetworkSnapshot
from freeway_traffic_scene.plugin import FreewayTrafficPlugin


class _MemoryAggregationCloud:
    def __init__(self, cloud):
        self.cloud = cloud
        self.groups = {}

    def aggregate(self, event):
        spec = event.metadata["aggregation"]
        group = self.groups.setdefault(spec["key"], {})
        group[spec["member"]] = event
        aggregation = {
            "group_id": "memory:" + spec["key"],
            "state": "waiting",
            "received_members": sorted(group),
            "missing_members": sorted(
                set(spec["expected_members"]) - set(group)
            ),
            "completion_reason": "",
            "finality": "pending",
            "evidence_complete": False,
            "global_confirmation": False,
            "result_revision": 0,
        }
        coordination = None
        if set(spec["expected_members"]).issubset(group):
            coordination = self.cloud.coordinate(
                [group[name] for name in sorted(group)]
            )
            aggregation["state"] = "completed"
            aggregation["completion_reason"] = "all_expected_members"
            aggregation["finality"] = "final"
            aggregation["evidence_complete"] = True
            aggregation["global_confirmation"] = bool(
                coordination.get("globally_consistent", False)
            )
            aggregation["result_revision"] = 1
        return {
            "aggregation": aggregation,
            "coordination": coordination,
            "transport": {
                "request_bytes": 512,
                "response_bytes": 512,
                "http_round_trip_ms": 1.0,
            },
        }

    def submit_feedback(self, *args):
        del args
        return True


def _load(name):
    path = Path(__file__).resolve().parents[1] / "samples" / name
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    registry = SceneRegistry([FreewayTrafficPlugin()])
    cloud = _MemoryAggregationCloud(CloudRuntime(registry))
    edge_a = EdgeRuntime(registry, cloud=cloud)
    edge_b = EdgeRuntime(registry, cloud=cloud)
    network = NetworkSnapshot(
        available=True,
        rtt_ms=5,
        jitter_ms=1,
        loss_rate=0,
        cloud_queue_ms=1,
        cloud_compute_ms=2,
    )
    first = edge_a.process(_load("edge_a_event.json"), network)
    second = edge_b.process(_load("edge_b_event.json"), network)
    first_delivery = edge_a.flush_pending()
    second_delivery = edge_b.flush_pending()
    assert first["final_decision"]["status"] == "provisional"
    assert first["final_decision"]["metadata"]["aggregation"]["state"] == "waiting"
    assert second["final_decision"]["status"] == "final"
    assert second["final_decision"]["metadata"]["aggregation"]["state"] == "completed"
    assert first_delivery["completed"] == 1
    assert second_delivery["attempted"] == 0
    coordination = first_delivery["coordination"]["coordination"]
    assert coordination["initial_conflict_count"] >= 1
    assert coordination["residual_conflict_count"] == 0
    print(
        json.dumps(
            {
                "status": "traffic_smoke_test_passed",
                "first_edge_initial_state": "provisional",
                "second_edge_initial_state": "final",
                "explicit_cloud_confirmation": True,
                "first_edge_result_backfill_completed": first_delivery["completed"],
                "initial_conflicts": coordination["initial_conflict_count"],
                "residual_conflicts": coordination["residual_conflict_count"],
                "model_weights_required": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
