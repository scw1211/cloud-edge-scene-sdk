"""用途：无服务器验证交通插件、双边缘汇聚、冲突消解和结果回填。"""

import json
from pathlib import Path

from cloud_edge_framework.contracts import stable_id
from cloud_edge_framework.registry import SceneRegistry
from cloud_edge_framework.runtime import CloudRuntime, EdgeRuntime
from cloud_edge_framework.scheduling import NetworkSnapshot
from freeway_traffic_scene.plugin import FreewayTrafficPlugin


class _MemoryAggregationCloud:
    def __init__(self, cloud):
        self.cloud = cloud
        self.groups = {}
        self.results = {}
        self.summary_uploads = 0

    @staticmethod
    def _group_id(event, spec):
        return stable_id("aggregation", event.scene, spec["key"])

    def _snapshot(self, event, spec):
        group = self.groups[spec["key"]]
        coordination = self.results.get(spec["key"])
        completed = coordination is not None
        return {
            "group_id": self._group_id(event, spec),
            "state": "completed" if completed else "waiting",
            "received_members": sorted(group),
            "missing_members": sorted(
                set(spec["expected_members"]) - set(group)
            ),
            "completion_reason": (
                "all_expected_members" if completed else ""
            ),
            "finality": "final" if completed else "pending",
            "evidence_complete": completed,
            "global_confirmation": bool(
                completed and coordination.get("globally_consistent", False)
            ),
            "result_revision": int(completed),
        }

    def _coordinate_if_ready(self, spec):
        group = self.groups[spec["key"]]
        if (
            spec["key"] not in self.results
            and set(spec["expected_members"]).issubset(group)
        ):
            self.results[spec["key"]] = self.cloud.coordinate(
                [group[name] for name in sorted(group)]
            )

    def aggregate(self, event):
        spec = event.metadata["aggregation"]
        group = self.groups.setdefault(spec["key"], {})
        if spec["member"] not in group:
            self.summary_uploads += 1
            group[spec["member"]] = event
        # Capture the durable-accept response before the in-memory worker is
        # allowed to coordinate. This mirrors the real endpoint: ingress does
        # not hold the request for peers or model execution.
        aggregation = self._snapshot(event, spec)
        response = {
            "aggregation": aggregation,
            "coordination": self.results.get(spec["key"]),
            "transport": {
                "request_bytes": 512,
                "response_bytes": 512,
                "http_round_trip_ms": 1.0,
            },
        }
        self._coordinate_if_ready(spec)
        return response

    def aggregate_batch(self, events, wait_seconds=0.0):
        assert float(wait_seconds) == 0.0
        responses = [(event, self.aggregate(event)) for event in events]
        return {
            "items": [
                {
                    "event_id": event.event_id,
                    "group_id": response["aggregation"]["group_id"],
                    "aggregation": response["aggregation"],
                    "coordination": response["coordination"],
                }
                for event, response in responses
            ],
            "transport": {
                "request_bytes": 512 * len(responses),
                "response_bytes": 512 * len(responses),
                "http_round_trip_ms": 1.0,
            },
        }

    def aggregation_results_batch(self, events, event_group_ids):
        items = []
        for event in events:
            spec = event.metadata["aggregation"]
            aggregation = self._snapshot(event, spec)
            assert aggregation["group_id"] == event_group_ids[event.event_id]
            items.append(
                {
                    "event_id": event.event_id,
                    "group_id": aggregation["group_id"],
                    "aggregation": aggregation,
                    "coordination": self.results.get(spec["key"]),
                }
            )
        return {
            "items": items,
            "transport": {
                "request_bytes": 128 * len(items),
                "response_bytes": 512 * len(items),
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
    first_submission = edge_a.flush_pending(waiting_poll_seconds=0.001)
    second_submission = edge_b.flush_pending(waiting_poll_seconds=0.001)
    first_delivery = edge_a.flush_pending(waiting_poll_seconds=0.001)
    second_delivery = edge_b.flush_pending(waiting_poll_seconds=0.001)
    assert first["final_decision"]["status"] == "provisional"
    assert second["final_decision"]["status"] == "provisional"
    assert first_submission["aggregation_waiting"] == 1
    assert second_submission["aggregation_waiting"] == 1
    assert first_delivery["completed"] == 1
    assert second_delivery["completed"] == 1
    assert cloud.summary_uploads == 2
    coordination = first_delivery["coordination"]["coordination"]
    assert coordination["initial_conflict_count"] >= 1
    assert coordination["residual_conflict_count"] == 0
    print(
        json.dumps(
            {
                "status": "traffic_smoke_test_passed",
                "first_edge_initial_state": "provisional",
                "second_edge_initial_state": "provisional",
                "explicit_cloud_confirmation": True,
                "first_edge_result_backfill_completed": first_delivery["completed"],
                "second_edge_result_backfill_completed": second_delivery["completed"],
                "unique_summary_uploads": cloud.summary_uploads,
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
