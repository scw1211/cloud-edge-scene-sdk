"""Traffic SUMMARY_FIRST / FEATURE_ON_DEMAND / RAW_ON_DEMAND contracts."""

from dataclasses import replace
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRAFFIC_ROOT = REPOSITORY_ROOT / "scenes" / "freeway_traffic"
for import_root in (REPOSITORY_ROOT, TRAFFIC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from cloud_edge_framework.evidence import EvidencePlanner  # noqa: E402
from cloud_edge_framework.registry import SceneRegistry  # noqa: E402
from cloud_edge_framework.runtime import CloudRuntime  # noqa: E402
from cloud_edge_framework.selective_evidence_pull import (  # noqa: E402
    BoundedEvidenceCache,
    EVIDENCE_CACHE_FETCH_ENDPOINT,
    HttpEvidencePullClient,
    PullPlan,
    PullTarget,
    SelectiveEvidencePullPlanner,
)
from freeway_traffic_full.plugin_impl import TrafficPlugin  # noqa: E402
from traffic_system.current_state_perception_runtime import (  # noqa: E402
    CurrentStateTrafficPerceptionRuntime,
)
from traffic_system.scene_event import traffic_envelope_from_output  # noqa: E402
from traffic_system.summary_first_evidence import (  # noqa: E402
    FEATURE_ON_DEMAND,
    SUMMARY_FIRST,
    SummaryFirstEvidenceCoordinator,
    SummaryFirstTrafficPlugin,
)


MODEL_ROOT = TRAFFIC_ROOT / "assets" / "models"
DATA_PATH = (
    TRAFFIC_ROOT
    / "assets"
    / "downloads"
    / "PEMS08_r1_d0_w0_astcgn_multitask.npz"
)
RULE_PATH = MODEL_ROOT / "current_state_perception_v1.json"
TOPOLOGY_PATH = MODEL_ROOT / "traffic_region_topology_metis4.json"
MEMBERS = ["edge_node_{}".format(index) for index in range(4)]
BASE_URLS = {
    member: "http://127.0.0.1:{}".format(19601 + index)
    for index, member in enumerate(MEMBERS)
}


def _plugin(plugin_type):
    return plugin_type(
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


def _caches():
    return {
        member: BoundedEvidenceCache(
            BASE_URLS[member],
            ttl_seconds=300.0,
            max_entries=128,
            max_bytes=128 * 1024 * 1024,
            signing_key=hashlib.sha256(
                "summary-first-test:{}".format(member).encode()
            ).digest(),
        )
        for member in MEMBERS
    }


class _Dispatcher:
    def __init__(self, caches, fail_member="", fail_level=""):
        self.caches = dict(caches)
        self.fail_member = fail_member
        self.fail_level = fail_level
        self.calls = []

    def __call__(self, url, payload, _timeout, _limit):
        origin = url[: -len(EVIDENCE_CACHE_FETCH_ENDPOINT)]
        member = str(payload["member"])
        if origin != BASE_URLS[member]:
            raise ValueError("callback did not reach the owning edge cache")
        self.calls.append((member, str(payload["requested_level"])))
        if member == self.fail_member and (
            not self.fail_level or payload["requested_level"] == self.fail_level
        ):
            raise TimeoutError("controlled feature callback failure")
        return self.caches[member].fetch(payload)


def _client(dispatcher):
    return HttpEvidencePullClient(
        list(BASE_URLS.values()), timeout_seconds=0.5, opener=dispatcher
    )


def _decision_core(result):
    return [
        {
            "decision": decision["decision"],
            "actions": [
                {
                    "action_type": action["action_type"],
                    "target_ids": action["target_ids"],
                    "resource_ids": action["resource_ids"],
                    "parameters": action["parameters"],
                }
                for action in decision["actions"]
            ],
        }
        for decision in result["decisions"]
    ]


class _RecordingRoadSetPlanner(SelectiveEvidencePullPlanner):
    def __init__(self):
        super().__init__()
        self.calls = []

    def plan(self, events, initial_coordination):
        self.calls.append(list(events))
        return PullPlan(
            triggered=False,
            trigger_reasons=[],
            road_set_ids=[],
            requested_members=[],
            requested_level=None,
            road_set_members={},
            road_set_levels={},
            initial_evidence_sufficient=False,
            targets=[],
            no_pull_reason="recording_no_conflict",
        )


class _ControlledRawPlanner(SelectiveEvidencePullPlanner):
    def __init__(self, residual=False, no_targets=False):
        super().__init__()
        self.residual = residual
        self.no_targets = no_targets
        self.calls = 0

    def plan(self, events, initial_coordination):
        self.calls += 1
        requested = ["edge_node_0", "edge_node_1"]
        if self.calls > 1:
            return PullPlan(
                triggered=self.residual,
                trigger_reasons=["controlled_residual"] if self.residual else [],
                road_set_ids=["rs_region_0__region_1"] if self.residual else [],
                requested_members=requested if self.residual else [],
                requested_level="raw" if self.residual else None,
                road_set_members={
                    "rs_region_0__region_1": requested
                } if self.residual else {},
                road_set_levels={
                    "rs_region_0__region_1": "raw"
                } if self.residual else {},
                initial_evidence_sufficient=self.residual,
                targets=[],
                no_pull_reason=(
                    "controlled_residual" if self.residual else "no_conflict"
                ),
            )
        event_by_member = {
            str(event.metadata["aggregation"]["member"]): event
            for event in events
        }
        targets = [] if self.no_targets else [
            PullTarget(
                road_set_id="rs_region_0__region_1",
                road_set_ids=["rs_region_0__region_1"],
                member=member,
                event_id=event_by_member[member].event_id,
                requested_level="raw",
                locator=dict(event_by_member[member].metadata["evidence_pull_locator"]),
            )
            for member in requested
        ]
        return PullPlan(
            triggered=True,
            trigger_reasons=["controlled_post_feature_conflict"],
            road_set_ids=["rs_region_0__region_1"],
            requested_members=requested,
            requested_level="raw",
            road_set_members={"rs_region_0__region_1": requested},
            road_set_levels={"rs_region_0__region_1": "raw"},
            initial_evidence_sufficient=False,
            targets=targets,
            no_pull_reason="raw_unavailable" if self.no_targets else "",
        )


class SummaryFirstEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.perception = CurrentStateTrafficPerceptionRuntime(
            DATA_PATH,
            RULE_PATH,
            TOPOLOGY_PATH,
            split="test",
            top_k=10,
        )

    def _summary_events(self, sample_id=0):
        plugin = _plugin(SummaryFirstTrafficPlugin)
        caches = _caches()
        events = []
        directives = {}
        for native in self.perception.infer_sample(sample_id).events:
            event = plugin.normalize(traffic_envelope_from_output(dict(native)))
            local = plugin.edge_decide(event)
            member = str(event.metadata["aggregation"]["member"])
            locator = caches[member].store(event)
            cloud, directive = plugin.prepare_summary_first_upload(
                event, local, locator
            )
            events.append(cloud)
            directives[member] = directive
        return plugin, caches, events, directives

    def _old_events(self, sample_id=0):
        plugin = _plugin(TrafficPlugin)
        caches = _caches()
        events = []
        for native in self.perception.infer_sample(sample_id).events:
            event = plugin.normalize(traffic_envelope_from_output(dict(native)))
            local = plugin.edge_decide(event)
            member = str(event.metadata["aggregation"]["member"])
            locator = caches[member].store(event)
            plan = EvidencePlanner().plan(
                event,
                scene_policy=plugin.evidence_advice(event, local, False),
            )
            selected_ids = set(plan.selected_evidence_ids)
            selected = replace(
                event,
                evidence=[
                    item
                    for item in event.evidence
                    if item.evidence_id in selected_ids
                ],
            )
            cloud = plugin.prepare_cloud_event(selected, plan.required_level)
            events.append(
                replace(
                    cloud,
                    metadata={**cloud.metadata, "evidence_pull_locator": locator},
                )
            )
        return plugin, events

    @staticmethod
    def _inject_content_declaration_mismatch(events):
        result = []
        for event in events:
            if event.scope.region_id != "region_1":
                result.append(event)
                continue
            payload = dict(event.scene_payload)
            assessments = deepcopy(payload["road_set_assessments"])
            target = next(
                row
                for row in assessments
                if row["road_set_id"] == "rs_region_0__region_1"
            )
            target["source_content_sha256"] = "f" * 64
            payload["road_set_assessments"] = assessments
            result.append(replace(event, scene_payload=payload))
        return result

    def test_every_first_upload_is_summary_and_feature_bytes_are_absent(self):
        _, _, events, directives = self._summary_events()
        self.assertEqual(len(events), 4)
        self.assertTrue(
            all({item.level for item in event.evidence} == {"summary"} for event in events)
        )
        self.assertTrue(
            all(
                "canonical_feature_observation" not in assessment
                for event in events
                for assessment in event.scene_payload["road_set_assessments"]
            )
        )
        self.assertEqual(
            sum(value.feature_pull_requested for value in directives.values()), 3
        )
        self.assertEqual(
            directives["edge_node_2"].oracle_required_level, "summary"
        )
        self.assertFalse(directives["edge_node_2"].feature_pull_requested)

    def test_old_raw_oracle_is_never_a_first_raw_upload(self):
        plugin = _plugin(SummaryFirstTrafficPlugin)
        caches = _caches()
        native = dict(self.perception.infer_sample(0).events[0])
        native["upload_required"] = True
        native["upload_level"] = "raw"
        event = plugin.normalize(traffic_envelope_from_output(native))
        local = plugin.edge_decide(event)
        member = str(event.metadata["aggregation"]["member"])
        cloud, directive = plugin.prepare_summary_first_upload(
            event, local, caches[member].store(event)
        )
        self.assertEqual(directive.oracle_required_level, "raw")
        self.assertTrue(directive.feature_pull_requested)
        self.assertEqual({item.level for item in cloud.evidence}, {"summary"})
        self.assertFalse(directive.raw_pull_requested)

    def test_feature_rerun_matches_the_frozen_mixed_level_cloud_result(self):
        old_plugin, old_events = self._old_events()
        new_plugin, caches, new_events, _ = self._summary_events()
        old_result = CloudRuntime(SceneRegistry([old_plugin])).coordinate(old_events)
        dispatcher = _Dispatcher(caches)
        result = SummaryFirstEvidenceCoordinator(
            CloudRuntime(SceneRegistry([new_plugin])), _client(dispatcher)
        ).coordinate(new_events)
        self.assertTrue(result.authoritative)
        self.assertEqual(result.final_state, FEATURE_ON_DEMAND)
        self.assertEqual(len(result.feature_plan.requested_members), 3)
        self.assertEqual(
            sorted(dispatcher.calls),
            [
                ("edge_node_0", "feature"),
                ("edge_node_1", "feature"),
                ("edge_node_3", "feature"),
            ],
        )
        self.assertEqual(_decision_core(result.final_result), _decision_core(old_result))
        self.assertFalse(result.raw_plan.triggered)

    def test_raw_conflict_planner_is_not_called_until_feature_stage_finishes(self):
        plugin, caches, events, _ = self._summary_events()
        planner = _RecordingRoadSetPlanner()
        dispatcher = _Dispatcher(caches)
        result = SummaryFirstEvidenceCoordinator(
            CloudRuntime(SceneRegistry([plugin])),
            _client(dispatcher),
            road_set_planner=planner,
        ).coordinate(events)
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(result.final_state, FEATURE_ON_DEMAND)
        requested = set(result.feature_plan.requested_members)
        for event in planner.calls[0]:
            member = str(event.metadata["aggregation"]["member"])
            levels = {item.level for item in event.evidence}
            if member in requested:
                self.assertIn("feature", levels)
            else:
                self.assertEqual(levels, {"summary"})

    def test_incomplete_feature_callback_keeps_summary_non_authoritative(self):
        plugin, caches, events, _ = self._summary_events()
        dispatcher = _Dispatcher(caches, fail_member="edge_node_1")
        result = SummaryFirstEvidenceCoordinator(
            CloudRuntime(SceneRegistry([plugin])), _client(dispatcher)
        ).coordinate(events)
        self.assertFalse(result.authoritative)
        self.assertEqual(result.final_state, SUMMARY_FIRST)
        self.assertEqual(result.final_result, result.initial_result)
        self.assertTrue(result.feature_errors)
        self.assertFalse(result.raw_plan.triggered)

    def test_missing_or_mismatched_feature_locator_is_fail_closed(self):
        for mode in ("missing", "mismatch"):
            with self.subTest(mode=mode):
                plugin, caches, events, _ = self._summary_events()
                changed = []
                for event in events:
                    metadata = dict(event.metadata)
                    if metadata["aggregation"]["member"] == "edge_node_0":
                        if mode == "missing":
                            metadata.pop("evidence_pull_locator")
                        else:
                            locator = dict(metadata["evidence_pull_locator"])
                            locator["member"] = "edge_node_9"
                            metadata["evidence_pull_locator"] = locator
                    changed.append(replace(event, metadata=metadata))
                result = SummaryFirstEvidenceCoordinator(
                    CloudRuntime(SceneRegistry([plugin])),
                    _client(_Dispatcher(caches)),
                ).coordinate(changed)
                self.assertFalse(result.authoritative)
                self.assertEqual(result.final_state, SUMMARY_FIRST)
                self.assertTrue(result.feature_plan.triggered)
                self.assertEqual(result.feature_plan.targets, [])
                self.assertTrue(result.feature_errors)

    def test_required_raw_with_zero_targets_is_fail_closed(self):
        plugin, caches, events, _ = self._summary_events()
        planner = _ControlledRawPlanner(no_targets=True)
        result = SummaryFirstEvidenceCoordinator(
            CloudRuntime(SceneRegistry([plugin])),
            _client(_Dispatcher(caches)),
            road_set_planner=planner,
        ).coordinate(events)
        self.assertFalse(result.authoritative)
        self.assertEqual(result.final_state, FEATURE_ON_DEMAND)
        self.assertTrue(result.raw_plan.triggered)
        self.assertEqual(result.raw_plan.targets, [])
        self.assertIn("raw_unavailable", result.raw_errors)

    def test_raw_rerun_with_residual_conflict_is_fail_closed(self):
        plugin, caches, events, _ = self._summary_events()
        planner = _ControlledRawPlanner(residual=True)
        dispatcher = _Dispatcher(caches)
        result = SummaryFirstEvidenceCoordinator(
            CloudRuntime(SceneRegistry([plugin])),
            _client(dispatcher),
            road_set_planner=planner,
        ).coordinate(events)
        self.assertFalse(result.authoritative)
        self.assertEqual(planner.calls, 2)
        self.assertEqual(
            [call for call in dispatcher.calls if call[1] == "raw"],
            [("edge_node_0", "raw"), ("edge_node_1", "raw")],
        )
        self.assertIn("raw_rerun_did_not_clear_conflict", result.raw_errors)

    def test_real_declaration_mismatch_uses_feature_then_exact_two_raw_and_repairs(self):
        plugin, caches, events, _ = self._summary_events()
        dispatcher = _Dispatcher(caches)
        result = SummaryFirstEvidenceCoordinator(
            CloudRuntime(SceneRegistry([plugin])), _client(dispatcher)
        ).coordinate(self._inject_content_declaration_mismatch(events))
        self.assertTrue(result.authoritative)
        self.assertEqual(result.final_state, "RAW_ON_DEMAND")
        feature_positions = [
            index for index, call in enumerate(dispatcher.calls) if call[1] == "feature"
        ]
        raw_positions = [
            index for index, call in enumerate(dispatcher.calls) if call[1] == "raw"
        ]
        self.assertTrue(feature_positions)
        self.assertTrue(raw_positions)
        self.assertLess(max(feature_positions), min(raw_positions))
        self.assertEqual(
            sorted(call for call in dispatcher.calls if call[1] == "raw"),
            [("edge_node_0", "raw"), ("edge_node_1", "raw")],
        )
        self.assertEqual(result.final_result["initial_conflict_count"], 0)
        self.assertEqual(result.final_result["residual_conflict_count"], 0)

    def test_real_raw_one_side_timeout_keeps_feature_result_non_authoritative(self):
        plugin, caches, events, _ = self._summary_events()
        dispatcher = _Dispatcher(
            caches, fail_member="edge_node_1", fail_level="raw"
        )
        result = SummaryFirstEvidenceCoordinator(
            CloudRuntime(SceneRegistry([plugin])), _client(dispatcher)
        ).coordinate(self._inject_content_declaration_mismatch(events))
        self.assertFalse(result.authoritative)
        self.assertEqual(result.final_state, FEATURE_ON_DEMAND)
        self.assertEqual(result.final_result, result.feature_result)
        self.assertTrue(result.raw_errors)
        self.assertEqual(
            sorted(call for call in dispatcher.calls if call[1] == "raw"),
            [("edge_node_0", "raw"), ("edge_node_1", "raw")],
        )


if __name__ == "__main__":
    unittest.main()
