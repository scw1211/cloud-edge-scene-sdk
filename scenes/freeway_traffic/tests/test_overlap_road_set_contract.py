"""Canonical shared-overlap road-set contract tests."""

import base64
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRAFFIC_ROOT = REPOSITORY_ROOT / "scenes" / "freeway_traffic"
for import_root in (REPOSITORY_ROOT, TRAFFIC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from cloud_edge_framework.registry import SceneRegistry  # noqa: E402
from cloud_edge_framework.runtime import CloudRuntime  # noqa: E402
from freeway_traffic_full.plugin_impl import TrafficPlugin  # noqa: E402
from traffic_system.current_state_perception_runtime import (  # noqa: E402
    CurrentStateTrafficPerceptionRuntime,
)
from traffic_system.overlap_observations import canonical_sha256  # noqa: E402
from traffic_system.road_sets import fuse_road_set_contexts  # noqa: E402
from traffic_system.scene_event import traffic_envelope_from_output  # noqa: E402


MODEL_ROOT = TRAFFIC_ROOT / "assets" / "models"
DATA_PATH = (
    TRAFFIC_ROOT
    / "assets"
    / "downloads"
    / "PEMS08_r1_d0_w0_astcgn_multitask.npz"
)
RULE_PATH = MODEL_ROOT / "current_state_perception_v1.json"
TOPOLOGY_PATH = MODEL_ROOT / "traffic_region_topology_metis4.json"


def _runtime():
    return CurrentStateTrafficPerceptionRuntime(
        DATA_PATH,
        RULE_PATH,
        TOPOLOGY_PATH,
        split="test",
        top_k=10,
    )


def _plugin():
    return TrafficPlugin(
        cloud_model_path=MODEL_ROOT / "cloud_coordinator_topology_fused.joblib",
        current_state_cloud_model_path=(
            MODEL_ROOT / "cloud_coordinator_current_state_future_v1.joblib"
        ),
        edge_student_path=MODEL_ROOT / "edge_student_freeway_joint_metis4.json",
        current_state_edge_student_path=(
            MODEL_ROOT / "edge_student_freeway_current_state_future_v1.json"
        ),
        feature_codec_path=(
            MODEL_ROOT / "traffic_tree_feature_codec_topology_v1.npz"
        ),
        current_state_feature_codec_path=(
            MODEL_ROOT / "traffic_tree_feature_codec_current_state_v1.npz"
        ),
        topology_path=TOPOLOGY_PATH,
        edge_llm_mode="disabled",
    )


def _records(events):
    return [
        {
            "aggregation_member": event["edge_id"],
            "road_set_assessments": event["road_set_assessments"],
        }
        for event in events
    ]


class CanonicalOverlapObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.perception = _runtime()
        cls.native_events = cls.perception.infer_sample(0).events

    def test_primary_ownership_stays_exclusive_while_overlap_is_identical(self):
        owners = [set(event["managed_node_ids"]) for event in self.native_events]
        self.assertEqual([43, 42, 42, 43], [len(nodes) for nodes in owners])
        self.assertEqual(set(range(170)), set().union(*owners))
        for left_index in range(4):
            for right_index in range(left_index + 1, 4):
                self.assertFalse(owners[left_index] & owners[right_index])

        copies = {}
        for event in self.native_events:
            for observation in event["road_set_overlap_observations"]:
                copies.setdefault(observation["road_set_id"], []).append(observation)
        self.assertEqual(5, len(copies))
        for road_set_id, replicas in copies.items():
            self.assertEqual(2, len(replicas), road_set_id)
            self.assertEqual(replicas[0], replicas[1])
            observation = replicas[0]
            self.assertEqual("canonical_shared_overlap_subscription", observation["kind"])
            self.assertFalse(observation["primary_sensor_ownership_changed"])
            raw_bytes = base64.b64decode(observation["raw_fragment_base64"])
            feature_bytes = base64.b64decode(observation["feature_bytes_base64"])
            self.assertEqual(
                hashlib.sha256(raw_bytes).hexdigest(),
                observation["raw_fragment_sha256"],
            )
            self.assertEqual(
                hashlib.sha256(feature_bytes).hexdigest(),
                observation["feature_content_sha256"],
            )

    def test_same_overlap_bytes_and_versions_fuse_without_conflict(self):
        contexts = fuse_road_set_contexts(_records(self.native_events))
        self.assertEqual(5, len(contexts))
        for context in contexts:
            self.assertTrue(context["complete"])
            self.assertEqual(2, len(context["partial_assessments"]))
            self.assertFalse(context["road_set_conflict_suspected"])
            self.assertEqual([], context["required_members"])
            left, right = context["partial_assessments"]
            for field in (
                "observation_id",
                "window_id",
                "source_content_sha256",
                "feature_content_sha256",
                "preprocess_version",
                "model_artifact_sha256",
                "model_version",
                "policy_artifact_sha256",
                "policy_version",
                "output_digest",
                "action_digest",
            ):
                self.assertEqual(left[field], right[field], field)

    def test_only_canonical_content_version_or_output_mismatch_requests_exact_pair(self):
        cases = (
            ("source_content_sha256", "f" * 64, "overlap_content_digest_mismatch"),
            ("preprocess_version", "tampered-preprocess", "overlap_preprocess_version_mismatch"),
            ("model_version", "tampered-model", "overlap_model_version_mismatch"),
            ("policy_version", "tampered-policy", "overlap_policy_version_mismatch"),
            ("output_digest", "e" * 64, "overlap_output_digest_mismatch"),
            ("action_digest", "d" * 64, "overlap_action_digest_mismatch"),
        )
        for field, value, expected_kind in cases:
            with self.subTest(field=field):
                records = deepcopy(_records(self.native_events))
                target = next(
                    assessment
                    for record in records
                    if record["aggregation_member"] == "edge_node_1"
                    for assessment in record["road_set_assessments"]
                    if assessment["road_set_id"] == "rs_region_0__region_1"
                )
                target[field] = value
                contexts = fuse_road_set_contexts(records)
                conflicts = [
                    context
                    for context in contexts
                    if context["road_set_conflict_suspected"]
                ]
                self.assertEqual(1, len(conflicts))
                self.assertEqual("rs_region_0__region_1", conflicts[0]["road_set_id"])
                self.assertEqual(expected_kind, conflicts[0]["conflict_kind"])
                self.assertEqual(
                    ["edge_node_0", "edge_node_1"],
                    conflicts[0]["required_members"],
                )
                self.assertEqual("raw", conflicts[0]["required_evidence_level"])

    def test_normal_cloud_decisions_do_not_create_generic_corridor_conflicts(self):
        plugin = _plugin()
        semantic = [
            plugin.normalize(traffic_envelope_from_output(event))
            for event in self.native_events
        ]
        prepared = [plugin.prepare_cloud_event(event, "feature") for event in semantic]
        result = CloudRuntime(SceneRegistry([plugin])).coordinate(prepared)
        self.assertEqual(0, result["initial_conflict_count"])
        self.assertEqual(0, result["residual_conflict_count"])
        self.assertTrue(result["globally_consistent"])
        for decision in result["decisions"]:
            for action in decision["actions"]:
                road_sets = [
                    resource
                    for resource in action["resource_ids"]
                    if resource.startswith("traffic_road_set:")
                ]
                if road_sets:
                    self.assertTrue(action["parameters"].get("canonical_road_set_action"))

    def test_self_consistent_forged_compact_action_triggers_raw_repair(self):
        plugin = _plugin()
        semantic = [
            plugin.normalize(traffic_envelope_from_output(event))
            for event in self.native_events
        ]
        prepared = [
            plugin.prepare_cloud_event(
                replace(
                    event,
                    evidence=[item for item in event.evidence if item.level == "summary"],
                ),
                "summary",
            )
            for event in semantic
        ]
        forged = []
        for event in prepared:
            if event.scope.region_id not in {"region_0", "region_1"}:
                forged.append(event)
                continue
            payload = dict(event.scene_payload)
            assessments = deepcopy(payload["road_set_assessments"])
            target = next(
                item
                for item in assessments
                if item["road_set_id"] == "rs_region_0__region_1"
            )
            action = deepcopy(target["action"])
            action["parameters"] = {
                **action.get("parameters", {}),
                "target_speed_mph": 15,
            }
            output = deepcopy(target["output"])
            output["action"] = action
            target["action"] = action
            target["action_digest"] = canonical_sha256(action)
            target["output"] = output
            target["output_digest"] = canonical_sha256(output)
            payload["road_set_assessments"] = assessments
            forged.append(replace(event, scene_payload=payload))

        fused = list(plugin.fuse_cloud_context(forged))
        contexts = {
            context["road_set_id"]: context
            for event in fused
            for context in event.scene_payload["road_set_contexts"]
        }
        conflict = contexts["rs_region_0__region_1"]
        self.assertTrue(conflict["road_set_conflict_suspected"])
        self.assertEqual(
            "overlap_compact_declaration_mismatch",
            conflict["conflict_kind"],
        )
        self.assertEqual("raw", conflict["required_evidence_level"])
        decisions = plugin.cloud_decide_batch(fused)
        canonical = [
            action
            for decision in decisions
            for action in decision.actions
            if action.parameters.get("canonical_road_set_action")
            and "rs_region_0__region_1"
            in action.parameters.get("road_set_ids", [])
        ]
        self.assertEqual(2, len(canonical))
        self.assertTrue(
            all(action.parameters.get("target_speed_mph") != 15 for action in canonical)
        )

    def test_raw_recompute_rejects_feature_tamper_and_cross_window_replay(self):
        plugin = _plugin()
        tampered = deepcopy(self.native_events[0])
        observation = tampered["road_set_overlap_observations"][0]
        feature = bytearray(base64.b64decode(observation["feature_bytes_base64"]))
        feature[0] ^= 1
        observation["feature_bytes_base64"] = base64.b64encode(feature).decode("ascii")
        observation["feature_content_sha256"] = hashlib.sha256(feature).hexdigest()
        with self.assertRaisesRegex(ValueError, "deterministic recomputation"):
            plugin.normalize(traffic_envelope_from_output(tampered))

        sample_one = deepcopy(self.perception.infer_sample(1).events[0])
        sample_zero_observation = deepcopy(
            next(
                item
                for item in self.native_events[0]["road_set_overlap_observations"]
                if item["road_set_id"] == "rs_region_0__region_1"
            )
        )
        observations = sample_one["road_set_overlap_observations"]
        observations[
            next(
                index
                for index, item in enumerate(observations)
                if item["road_set_id"] == "rs_region_0__region_1"
            )
        ] = sample_zero_observation
        with self.assertRaisesRegex(ValueError, "another event window"):
            plugin.normalize(traffic_envelope_from_output(sample_one))

    def test_compact_owner_spoof_missing_and_duplicate_fail_closed(self):
        plugin = _plugin()
        semantic = [
            plugin.normalize(traffic_envelope_from_output(event))
            for event in self.native_events
        ]
        prepared = [plugin.prepare_cloud_event(event, "summary") for event in semantic]
        mutations = {}

        owner_payload = dict(prepared[0].scene_payload)
        owner_rows = deepcopy(owner_payload["road_set_assessments"])
        owner_rows[0]["member_region"] = "region_1"
        owner_payload["road_set_assessments"] = owner_rows
        mutations["owner binding"] = replace(prepared[0], scene_payload=owner_payload)

        missing_payload = dict(prepared[0].scene_payload)
        missing_payload["road_set_assessments"] = deepcopy(
            missing_payload["road_set_assessments"][:-1]
        )
        mutations["incomplete"] = replace(prepared[0], scene_payload=missing_payload)

        duplicate_payload = dict(prepared[0].scene_payload)
        duplicate_rows = deepcopy(duplicate_payload["road_set_assessments"])
        duplicate_rows.append(deepcopy(duplicate_rows[0]))
        duplicate_payload["road_set_assessments"] = duplicate_rows
        mutations["coverage"] = replace(prepared[0], scene_payload=duplicate_payload)

        for expected, mutated in mutations.items():
            with self.subTest(expected=expected):
                events = [mutated, *prepared[1:]]
                with self.assertRaisesRegex(ValueError, expected):
                    plugin.fuse_cloud_context(events)


if __name__ == "__main__":
    unittest.main()
