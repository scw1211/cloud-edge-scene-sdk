"""Boundary road-set coordination contract tests for the four traffic edges.

These fixtures deliberately stop at the stable topology contract.  The cloud
supplemental-fetch tests reuse them so that selective fetching cannot silently
turn mutually exclusive sensor ownership into duplicated perception.
"""

import json
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRAFFIC_ROOT = REPOSITORY_ROOT / "scenes" / "freeway_traffic"
for import_root in (REPOSITORY_ROOT, TRAFFIC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from cloud_edge_framework.event_envelope import SceneEventEnvelope  # noqa: E402
from cloud_edge_framework.cloud_service import CloudApiService  # noqa: E402
from cloud_edge_framework.registry import SceneRegistry  # noqa: E402
from cloud_edge_framework.runtime import CloudRuntime  # noqa: E402
from cloud_edge_framework.selective_evidence_pull import (  # noqa: E402
    SelectiveEvidencePullPlanner,
)
from freeway_traffic_full.plugin_impl import TrafficPlugin  # noqa: E402
from traffic_system.current_state_perception_runtime import (  # noqa: E402
    CurrentStateTrafficPerceptionRuntime,
)
from traffic_system.road_sets import (  # noqa: E402
    build_partial_assessments,
    build_road_set_catalog,
    fuse_road_set_contexts,
)
from traffic_system.scene_event import traffic_envelope_from_output  # noqa: E402


PERCEPTION_PATH = (
    TRAFFIC_ROOT / "assets" / "models" / "current_state_perception_v1.json"
)
TOPOLOGY_PATH = (
    TRAFFIC_ROOT / "assets" / "models" / "traffic_region_topology_metis4.json"
)
EXPECTED_PARTITION_SIZES = [43, 42, 42, 43]
DATA_PATH = (
    TRAFFIC_ROOT
    / "assets"
    / "downloads"
    / "PEMS08_r1_d0_w0_astcgn_multitask.npz"
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sensor_owners():
    perception = _load_json(PERCEPTION_PATH)
    return {
        "region_{}".format(index): {
            "member": "edge_node_{}".format(index),
            "sensor_ids": {int(node) for node in nodes},
        }
        for index, nodes in enumerate(perception["partitions"])
    }


def _boundary_road_set_fixtures():
    """Return one two-sided partial-assessment fixture per adjacent pair."""

    topology = _load_json(TOPOLOGY_PATH)
    owners = _sensor_owners()
    fixtures = []
    for pair in topology["region_pairs"]:
        left_region = str(pair["left_region"])
        right_region = str(pair["right_region"])
        road_set_id = "rs_{}__{}".format(left_region, right_region)
        fixtures.append(
            {
                "road_set_id": road_set_id,
                "resource_id": "traffic_road_set:{}".format(road_set_id),
                "cut_edge_count": int(pair["cut_edge_count"]),
                "assessments": [
                    {
                        "road_set_id": road_set_id,
                        "member": owners[left_region]["member"],
                        "region_id": left_region,
                        "assessment_scope": "partial",
                        "owned_sensor_ids": {
                            int(node) for node in pair["left_boundary_nodes"]
                        },
                    },
                    {
                        "road_set_id": road_set_id,
                        "member": owners[right_region]["member"],
                        "region_id": right_region,
                        "assessment_scope": "partial",
                        "owned_sensor_ids": {
                            int(node) for node in pair["right_boundary_nodes"]
                        },
                    },
                ],
            }
        )
    return fixtures


def _catalog():
    return build_road_set_catalog(_load_json(TOPOLOGY_PATH))


@lru_cache(maxsize=4)
def _canonical_native_events(sample_id=0):
    runtime = CurrentStateTrafficPerceptionRuntime(
        DATA_PATH,
        PERCEPTION_PATH,
        TOPOLOGY_PATH,
        split="test",
        top_k=10,
    )
    return tuple(runtime.infer_sample(sample_id).events)


def _owner_partial_records(
    *,
    top_nodes_by_region=None,
    summaries_by_region=None,
    actions_by_region=None,
):
    owners = _sensor_owners()
    catalog = _catalog()
    top_nodes_by_region = top_nodes_by_region or {}
    summaries_by_region = summaries_by_region or {}
    actions_by_region = actions_by_region or {}
    records = []
    for region_id, owner in sorted(owners.items()):
        summary = {
            "region_risk_level": "low",
            "region_risk_score": 0.1,
            "region_risk_confidence": 0.9,
            **summaries_by_region.get(region_id, {}),
        }
        records.append(
            {
                "region_id": region_id,
                "aggregation_member": owner["member"],
                "road_set_assessments": build_partial_assessments(
                    catalog,
                    region_id,
                    owner["member"],
                    sorted(owner["sensor_ids"]),
                    top_nodes_by_region.get(region_id, []),
                    summary,
                    actions_by_region.get(region_id, []),
                ),
            }
        )
    return records


def _raw_traffic_event(partition_id, *, sample_id=9100, summary=None, top_nodes=None):
    owners = _sensor_owners()
    region_id = "region_{}".format(partition_id)
    member = owners[region_id]["member"]
    region_summary = {
        "region_risk_level": "low",
        "region_risk_score": 0.1,
        "region_risk_confidence": 0.9,
        "region_risk_probabilities": {"low": 0.9, "medium": 0.1},
        "region_risk_calibration": {
            "method": "test_calibration",
            "calibrated_confidence": 0.9,
            "prediction_set": ["low"],
        },
        "node_risk_counts": {
            "low": len(owners[region_id]["sensor_ids"]),
            "medium": 0,
            "high": 0,
            "severe": 0,
        },
    }
    if summary:
        region_summary.update(summary)
    data = {
        "sample_id": sample_id,
        "sample_split": "test",
        "dataset": "PEMS08",
        "region_id": region_id,
        "partition_id": partition_id,
        "num_partitions": 4,
        "aggregation_member": member,
        "aggregation_expected_members": [
            "edge_node_{}".format(index) for index in range(4)
        ],
        "aggregation_minimum_members": 4,
        "managed_node_ids": sorted(owners[region_id]["sensor_ids"]),
        "prediction_horizon_minutes": 60,
        "region_summary": region_summary,
        "top_k_risk_nodes": list(top_nodes or []),
        "control_capabilities": {
            "variable_speed_limit_nodes": sorted(
                owners[region_id]["sensor_ids"]
            ),
            "ramp_meter_nodes": [],
        },
        "operational_safety_risk": {
            "level": "low",
            "score": 0.1,
            "source": "test_action_authority",
        },
        "upload_required": False,
        "deadline_ms": 500,
        "preprocessing_latency_ms": 2,
        "inference_latency_ms": 20,
        "model": "test_current_state",
    }
    return {
        "specversion": "1.0",
        "id": "road-set-{}-{}".format(sample_id, partition_id),
        "source": "urn:edge:test:partition:{}".format(partition_id),
        "type": "com.cloudedge.traffic.edge-event.v1",
        "scene": "freeway_traffic_management",
        "edgeid": member,
        "subject": str(sample_id),
        "time": "2026-08-01T00:00:00Z",
        "datacontenttype": "application/json",
        "dataschema": (
            "https://cloud-edge.local/schemas/scenes/traffic-edge-event-v1.json"
        ),
        "data": data,
    }


class BoundaryRoadSetFixtureTests(unittest.TestCase):
    def test_sensor_ownership_is_an_exact_mutually_exclusive_170_node_cover(self):
        owners = _sensor_owners()
        ordered_regions = ["region_{}".format(index) for index in range(4)]

        self.assertEqual(
            EXPECTED_PARTITION_SIZES,
            [len(owners[region]["sensor_ids"]) for region in ordered_regions],
        )
        all_sensor_ids = set()
        for index, region in enumerate(ordered_regions):
            sensor_ids = owners[region]["sensor_ids"]
            self.assertEqual("edge_node_{}".format(index), owners[region]["member"])
            self.assertFalse(all_sensor_ids & sensor_ids)
            all_sensor_ids.update(sensor_ids)
        self.assertEqual(set(range(170)), all_sensor_ids)

    def test_each_boundary_road_set_has_two_owner_local_partial_assessments(self):
        owners = _sensor_owners()
        fixtures = _boundary_road_set_fixtures()

        self.assertEqual(5, len(fixtures))
        self.assertEqual(26, sum(item["cut_edge_count"] for item in fixtures))
        self.assertEqual(
            len(fixtures), len({item["road_set_id"] for item in fixtures})
        )
        for fixture in fixtures:
            assessments = fixture["assessments"]
            self.assertEqual(2, len(assessments))
            self.assertEqual(
                {fixture["road_set_id"]},
                {item["road_set_id"] for item in assessments},
            )
            self.assertEqual(
                "traffic_road_set:{}".format(fixture["road_set_id"]),
                fixture["resource_id"],
            )
            self.assertEqual(
                {"partial"},
                {item["assessment_scope"] for item in assessments},
            )
            self.assertEqual(2, len({item["member"] for item in assessments}))
            left, right = assessments
            self.assertTrue(left["owned_sensor_ids"])
            self.assertTrue(right["owned_sensor_ids"])
            self.assertFalse(
                left["owned_sensor_ids"] & right["owned_sensor_ids"]
            )
            for assessment in assessments:
                owner = owners[assessment["region_id"]]
                self.assertEqual(owner["member"], assessment["member"])
                self.assertLessEqual(
                    assessment["owned_sensor_ids"], owner["sensor_ids"]
                )

    def test_topology_neighbors_are_exactly_the_road_set_memberships(self):
        topology = _load_json(TOPOLOGY_PATH)
        memberships = {
            region: set() for region in topology["region_neighbors"]
        }
        for fixture in _boundary_road_set_fixtures():
            left, right = fixture["assessments"]
            memberships[left["region_id"]].add(right["region_id"])
            memberships[right["region_id"]].add(left["region_id"])

        self.assertEqual(
            {
                region: set(neighbors)
                for region, neighbors in topology["region_neighbors"].items()
            },
            memberships,
        )


class RoadSetDecisionContractTests(unittest.TestCase):
    def test_catalog_and_owner_partials_match_the_frozen_topology(self):
        owners = _sensor_owners()
        catalog = _catalog()
        fixtures = {
            item["road_set_id"]: item
            for item in _boundary_road_set_fixtures()
        }

        self.assertEqual(set(fixtures), {
            item["road_set_id"] for item in catalog["road_sets"]
        })
        records = _owner_partial_records()
        partials = [
            assessment
            for record in records
            for assessment in record["road_set_assessments"]
        ]
        self.assertEqual(10, len(partials))
        for road_set_id, fixture in fixtures.items():
            matching = [
                item for item in partials if item["road_set_id"] == road_set_id
            ]
            self.assertEqual(2, len(matching))
            self.assertEqual({True}, {item["partial"] for item in matching})
            self.assertEqual(
                {item["member"] for item in fixture["assessments"]},
                {item["aggregation_member"] for item in matching},
            )
            for partial in matching:
                self.assertEqual([], partial["required_members"])
                self.assertEqual(
                    fixture["resource_id"], partial["resource_id"]
                )
                self.assertLessEqual(
                    set(partial["member_nodes"]),
                    owners[partial["member_region"]]["sensor_ids"],
                )

    def test_consistent_partials_preserve_five_road_set_decisions_without_pull(self):
        contexts = fuse_road_set_contexts(_owner_partial_records())

        self.assertEqual(5, len(contexts))
        self.assertEqual(
            5, len({context["road_set_id"] for context in contexts})
        )
        for context in contexts:
            self.assertTrue(context["complete"])
            self.assertEqual(2, len(context["partial_assessments"]))
            self.assertEqual(2, len(context["observed_members"]))
            self.assertFalse(context["road_set_conflict_suspected"])
            self.assertEqual([], context["required_members"])
            self.assertNotIn("required_evidence_level", context)

    def test_different_owner_local_risk_does_not_trigger_a_shared_pull(self):
        records = _owner_partial_records(
            top_nodes_by_region={
                "region_0": [
                    {
                        "node_id": 5,
                        "risk_level": "severe",
                        "risk_score": 0.95,
                        "risk_confidence": 0.96,
                    }
                ],
                "region_1": [
                    {
                        "node_id": 18,
                        "risk_level": "low",
                        "risk_score": 0.1,
                        "risk_confidence": 0.95,
                    }
                ],
            }
        )
        contexts = fuse_road_set_contexts(records)
        requested = [
            context for context in contexts if context["required_members"]
        ]

        # The two values describe mutually exclusive primary sensor domains.
        # They are not two evaluations of the same shared observation.
        self.assertEqual([], requested)

    def test_region_proxy_does_not_manufacture_a_road_set_conflict(self):
        topology = _load_json(TOPOLOGY_PATH)
        owners = _sensor_owners()
        proxy_nodes = {
            region_id: min(
                owner["sensor_ids"]
                - set(topology["region_boundary_nodes"][region_id])
            )
            for region_id, owner in owners.items()
            if region_id in {"region_0", "region_1"}
        }
        records = _owner_partial_records(
            top_nodes_by_region={
                "region_0": [
                    {
                        "node_id": proxy_nodes["region_0"],
                        "risk_level": "severe",
                        "risk_score": 0.95,
                        "risk_confidence": 0.96,
                    }
                ],
                "region_1": [
                    {
                        "node_id": proxy_nodes["region_1"],
                        "risk_level": "low",
                        "risk_score": 0.1,
                        "risk_confidence": 0.95,
                    }
                ],
            }
        )

        contexts = fuse_road_set_contexts(records)

        self.assertTrue(
            all(
                partial["assessment_source"] == "owner_region_summary_proxy"
                for context in contexts
                for partial in context["partial_assessments"]
            )
        )
        self.assertFalse(
            any(context["road_set_conflict_suspected"] for context in contexts)
        )
        self.assertTrue(
            all(context["required_members"] == [] for context in contexts)
        )

    def test_legacy_region_action_views_do_not_trigger_a_shared_pull(self):
        resource_id = "traffic_road_set:rs_region_1__region_2"
        records = _owner_partial_records(
            actions_by_region={
                "region_1": [
                    {
                        "action_type": "variable_speed_limit",
                        "parameters": {"target_speed_mph": 55},
                        "resource_ids": [resource_id],
                    }
                ],
                "region_2": [
                    {
                        "action_type": "variable_speed_limit",
                        "parameters": {"target_speed_mph": 35},
                        "resource_ids": [resource_id],
                    }
                ],
            }
        )
        contexts = fuse_road_set_contexts(records)
        requested = [
            context for context in contexts if context["required_members"]
        ]

        # Only the versioned deterministic overlap policy may author a shared
        # road-set action.  Legacy region actions cannot manufacture one.
        self.assertEqual([], requested)


class TrafficPluginRoadSetIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.plugin = TrafficPlugin(topology_path=TOPOLOGY_PATH)
        self.normalized = [
            self.plugin.normalize(traffic_envelope_from_output(event))
            for event in _canonical_native_events(0)
        ]
        self.prepared = [
            self.plugin.prepare_cloud_event(event, "summary")
            for event in self.normalized
        ]

    def test_normalize_prepare_fuse_keeps_owner_partials_and_road_set_identity(self):
        incident_counts = [2, 3, 2, 3]
        for index, (normalized, prepared) in enumerate(
            zip(self.normalized, self.prepared)
        ):
            assessments = normalized.scene_payload["road_set_assessments"]
            self.assertEqual(incident_counts[index], len(assessments))
            self.assertEqual(
                incident_counts[index],
                len(
                    {
                        resource
                        for resource in normalized.scope.shared_resources
                        if resource.startswith("traffic_road_set:")
                    }
                ),
            )
            self.assertEqual(
                assessments, prepared.scene_payload["road_set_assessments"]
            )
            self.assertTrue(
                all(item["partial"] is True for item in assessments)
            )
            self.assertTrue(
                all(
                    item["aggregation_member"] == "edge_node_{}".format(index)
                    for item in assessments
                )
            )

        fused = list(self.plugin.fuse_cloud_context(self.prepared))
        contexts_by_id = {}
        for event in fused:
            self.assertFalse(event.metadata["road_set_conflict_suspected"])
            self.assertEqual([], event.metadata["required_members"])
            for context in event.scene_payload["road_set_contexts"]:
                contexts_by_id.setdefault(context["road_set_id"], []).append(
                    context
                )
        self.assertEqual(5, len(contexts_by_id))
        for copies in contexts_by_id.values():
            self.assertEqual(2, len(copies))
            self.assertEqual(copies[0], copies[1])
            self.assertEqual(2, len(copies[0]["partial_assessments"]))
            self.assertEqual([], copies[0]["required_members"])

    def test_cloud_decide_and_coordinate_keep_four_members_and_road_set_decisions(self):
        fused = list(self.plugin.fuse_cloud_context(self.prepared))
        decisions = list(self.plugin.cloud_decide_batch(fused))

        self.assertEqual(4, len(decisions))
        self.assertEqual(
            [2, 3, 2, 3],
            [item.metadata["road_set_decision_count"] for item in decisions],
        )
        self.assertTrue(
            all(item.metadata["decision_granularity"] == "road_set" for item in decisions)
        )
        self.assertTrue(
            all(item.metadata["global_action_collapsed"] is False for item in decisions)
        )
        self.assertEqual(
            5,
            len(
                {
                    road_set["road_set_id"]
                    for decision in decisions
                    for road_set in decision.metadata["road_set_decisions"]
                }
            ),
        )

        result = CloudRuntime(SceneRegistry([self.plugin])).coordinate(self.prepared)
        self.assertEqual(4, result["event_count"])
        self.assertEqual(4, len(result["decisions"]))
        self.assertEqual(
            {event.event_id for event in self.prepared},
            {
                event_id
                for decision in result["decisions"]
                for event_id in decision["event_ids"]
            },
        )
        self.assertTrue(
            all(
                decision["metadata"]["decision_granularity"] == "road_set"
                and decision["metadata"]["global_action_collapsed"] is False
                for decision in result["decisions"]
            )
        )

    def test_region_actions_are_not_promoted_and_canonical_controls_are_shared(self):
        high_event = self.normalized[0]
        self.assertTrue(
            any(
                action.action_type
                in {
                    "variable_speed_limit",
                    "ramp_metering",
                    "regional_coordination",
                    "reroute",
                }
                for action in high_event.candidate_actions
            )
        )
        self.assertTrue(
            all(
                not value.startswith("traffic_road_set:")
                for action in high_event.candidate_actions
                for value in action.target_ids + action.resource_ids
            )
        )

        local = self.plugin.edge_decide(high_event)
        canonical_actions = [
            action
            for action in local.actions
            if action.parameters.get("canonical_road_set_action")
        ]
        self.assertTrue(canonical_actions)
        self.assertTrue(
            all(
                len(action.resource_ids) == 1
                and action.resource_ids[0].startswith("traffic_road_set:")
                for action in canonical_actions
            )
        )
        executable = [
            action
            for action in canonical_actions
            if action.action_type
            in {"variable_speed_limit", "ramp_metering", "reroute"}
        ]
        self.assertTrue(executable)
        self.assertTrue(
            all(
                action.parameters.get("requires_cloud_confirmation") is True
                for action in executable
            )
        )

        lease = SimpleNamespace(
            group_id="road-set-authorization-test",
            group_key="PEMS08:test:9101",
            completion_reason="all_expected_members",
            expected_members=[
                "edge_node_{}".format(index) for index in range(4)
            ],
            received_members=[
                "edge_node_{}".format(index) for index in range(4)
            ],
            missing_members=[],
            result_revision=1,
        )
        base_coordination = {
            "decisions": [local.to_dict()],
            "globally_consistent": True,
        }
        failed_pull = CloudApiService._mark_aggregation_finality(
            {
                **base_coordination,
                "evidence_pull": {
                    "triggered": True,
                    "rerun_applied": False,
                    "initial_evidence_sufficient": False,
                },
            },
            lease,
        )
        failed_authorization = failed_pull["decisions"][0]["metadata"][
            "action_authorization"
        ]
        self.assertFalse(failed_pull["global_confirmation"])
        self.assertIn(
            "variable_speed_limit",
            failed_authorization["deferred_action_types"],
        )
        self.assertFalse(failed_authorization["all_actions_authorized"])

        successful_pull = CloudApiService._mark_aggregation_finality(
            {
                **base_coordination,
                "evidence_pull": {
                    "triggered": True,
                    "rerun_applied": True,
                    "initial_evidence_sufficient": False,
                },
            },
            lease,
        )
        successful_authorization = successful_pull["decisions"][0]["metadata"][
            "action_authorization"
        ]
        self.assertTrue(successful_pull["global_confirmation"])
        self.assertNotIn(
            "variable_speed_limit",
            successful_authorization["deferred_action_types"],
        )
        self.assertTrue(successful_authorization["all_actions_authorized"])

    def test_only_a_canonical_digest_mismatch_is_visible_to_pull_planner(self):
        divergent = deepcopy(self.prepared)
        target_index = next(
            index
            for index, event in enumerate(divergent)
            if event.scope.region_id == "region_1"
        )
        payload = dict(divergent[target_index].scene_payload)
        assessments = deepcopy(payload["road_set_assessments"])
        target = next(
            item
            for item in assessments
            if item["road_set_id"] == "rs_region_0__region_1"
        )
        target["source_content_sha256"] = "f" * 64
        payload["road_set_assessments"] = assessments
        divergent[target_index] = replace(
            divergent[target_index], scene_payload=payload
        )

        result = CloudRuntime(SceneRegistry([self.plugin])).coordinate(divergent)
        road_set_rows = [
            item
            for decision in result["decisions"]
            for item in decision["metadata"]["road_set_decisions"]
            if item["road_set_id"] == "rs_region_0__region_1"
        ]
        self.assertEqual(2, len(road_set_rows))
        self.assertTrue(
            all(item["road_set_conflict_suspected"] for item in road_set_rows)
        )

        plan = SelectiveEvidencePullPlanner().plan(divergent, result)
        self.assertTrue(plan.triggered)
        self.assertEqual(["rs_region_0__region_1"], plan.road_set_ids)
        self.assertEqual(
            ["edge_node_0", "edge_node_1"], plan.requested_members
        )


if __name__ == "__main__":
    unittest.main()
