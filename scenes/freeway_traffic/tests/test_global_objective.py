"""Traffic joint candidate utility and framework hook regressions."""

from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRAFFIC_ROOT = REPOSITORY_ROOT / "scenes" / "freeway_traffic"
for import_root in (REPOSITORY_ROOT, TRAFFIC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from cloud_edge_framework.conflicts import ConflictCoordinator  # noqa: E402
from cloud_edge_framework.contracts import (  # noqa: E402
    Action,
    Evidence,
    EventScope,
    Prediction,
    Risk,
    SemanticEvent,
    Timing,
    Uncertainty,
    build_decision,
)
from cloud_edge_framework.registry import SceneRegistry  # noqa: E402
from freeway_traffic_full.plugin_impl import TrafficPlugin  # noqa: E402
from traffic_system.global_objective import TrafficGlobalObjective  # noqa: E402


OBJECTIVE_PATH = (
    TRAFFIC_ROOT / "assets" / "models" / "traffic_global_utility_v1.json"
)


def _action(speed: float, resource: str = "corridor:shared", node: int = 1):
    return Action(
        action_type="variable_speed_limit",
        target_ids=["traffic_node:{}".format(node)],
        resource_ids=[resource],
        parameters={
            "target_speed_mph": speed,
            "requires_cloud_confirmation": True,
        },
        reason="test traffic control",
        priority=70,
    )


def _event(
    index: int,
    risk_score: float = 0.95,
    expected_members=None,
    allowed_node: int = 1,
):
    expected = expected_members or [
        "edge_node_{}".format(value) for value in range(4)
    ]
    risk_level = "severe" if risk_score >= 0.90 else "low"
    return SemanticEvent(
        event_id="traffic-global-{}".format(index),
        scene="traffic",
        task="traffic_control",
        edge_id="edge_node_{}".format(index),
        occurred_at_ms=1000,
        scope=EventScope(
            entity_id="sample:1",
            subsystem="freeway",
            state_variable="traffic_risk",
            region_id="region_{}".format(index),
            shared_resources=["corridor:shared"],
            correlation_keys=["sample:1"],
            window_start_ms=1000,
            window_end_ms=1100,
        ),
        prediction=Prediction(
            label=risk_level,
            confidence=0.95,
            probabilities={risk_level: 0.95},
        ),
        risk=Risk(level=risk_level, score=risk_score),
        uncertainty=Uncertainty(
            confidence=0.95,
            calibrated=True,
            prediction_set=[risk_level],
            method="test",
        ),
        timing=Timing(deadline_ms=200.0),
        evidence=[
            Evidence(
                evidence_id="evidence-{}".format(index),
                level="summary",
                modality="traffic",
                encoding="json",
                inline={"risk_score": risk_score},
            )
        ],
        candidate_actions=[],
        model={"name": "test"},
        scene_payload={
            "region_summary": {
                "current_observation": {
                    "node_count": 40,
                    "flow_mean": 420.0 if risk_score >= 0.90 else 100.0,
                    "occupancy_mean": 0.20 if risk_score >= 0.90 else 0.06,
                    "speed_mean": 25.0 if risk_score >= 0.90 else 65.0,
                    "speed_min": 18.0 if risk_score >= 0.90 else 60.0,
                }
            },
            "control_capabilities": {
                "variable_speed_limit_nodes": [allowed_node]
            }
        },
        metadata={
            "aggregation": {
                "authority": "cloud_aggregation_lease",
                "member": "edge_node_{}".format(index),
                "expected_members": list(expected),
                "received_members": list(expected),
                "missing_members": [],
                "completion_reason": "all_expected_members",
                "evidence_complete": True,
                "finality": "final",
            }
        },
    )


def _decision(event: SemanticEvent, actions):
    return build_decision(
        event=event,
        decision="variable_speed_limit" if actions else "no_action",
        actions=actions,
        confidence=0.9,
        reason="test cloud proposal",
        source="test_cloud",
        policy_version="test-1",
        route="cloud_sync",
    )


class TrafficGlobalObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.objective = TrafficGlobalObjective.from_path(OBJECTIVE_PATH)

    def test_active_selects_positive_high_risk_bundle_exactly(self):
        events = [_event(index) for index in range(4)]
        decisions = [_decision(event, [_action(45, "corridor:{}".format(index))])
                     for index, event in enumerate(events)]
        selected, metadata = self.objective.optimize(
            events,
            decisions,
            TrafficPlugin().action_conflict,
            "active",
        )
        self.assertTrue(metadata["applied"])
        self.assertTrue(metadata["candidate_set_optimality_verified"])
        self.assertEqual(metadata["solver"], "exact_enumeration")
        self.assertEqual(len(metadata["objective_definition_sha256"]), 64)
        self.assertEqual(
            metadata["baseline_utility"], metadata["baseline"]["utility"]
        )
        self.assertEqual(
            metadata["selected_utility"], metadata["selected"]["utility"]
        )
        self.assertEqual([len(item.actions) for item in selected], [1, 1, 1, 1])
        self.assertGreaterEqual(metadata["utility_delta"], 0.0)
        self.assertEqual(metadata["selected"]["traffic_observation_count"], 4)
        self.assertEqual(
            metadata["expected_members"],
            ["edge_node_0", "edge_node_1", "edge_node_2", "edge_node_3"],
        )
        self.assertGreater(
            metadata["selected"]["delay_proxy_reduction"], 0.0
        )
        self.assertFalse(metadata.get("global_optimality_claimed", False))

    def test_low_risk_or_unsupported_action_is_not_selected(self):
        low = _event(0, risk_score=0.10)
        unsupported = _event(1, allowed_node=2)
        quiet = [_event(2, risk_score=0.10), _event(3, risk_score=0.10)]
        decisions = [
            _decision(low, [_action(45, "corridor:low")]),
            _decision(unsupported, [_action(45, "corridor:unsupported", node=1)]),
            _decision(quiet[0], []),
            _decision(quiet[1], []),
        ]
        selected, metadata = self.objective.optimize(
            [low, unsupported] + quiet,
            decisions,
            TrafficPlugin().action_conflict,
            "active",
        )
        self.assertEqual([len(item.actions) for item in selected], [0, 0, 0, 0])
        self.assertTrue(metadata["selected"]["feasible"])

    def test_conflicting_joint_plan_is_rejected(self):
        events = [_event(index) for index in range(4)]
        decisions = [
            _decision(events[0], [_action(30)]),
            _decision(events[1], [_action(65)]),
            _decision(events[2], []),
            _decision(events[3], []),
        ]
        selected, metadata = self.objective.optimize(
            events,
            decisions,
            TrafficPlugin().action_conflict,
            "active",
        )
        self.assertLessEqual(sum(len(item.actions) for item in selected), 1)
        self.assertEqual(metadata["selected"]["conflict_count"], 0)

    def test_shadow_records_choice_without_changing_actions(self):
        events = [_event(index) for index in range(4)]
        decisions = [_decision(event, [_action(45, "corridor:{}".format(index))])
                     for index, event in enumerate(events)]
        selected, metadata = self.objective.optimize(
            events,
            decisions,
            TrafficPlugin().action_conflict,
            "shadow",
        )
        self.assertFalse(metadata["applied"])
        self.assertEqual(
            [[action.to_dict() for action in item.actions] for item in selected],
            [[action.to_dict() for action in item.actions] for item in decisions],
        )
        self.assertTrue(
            all("global_optimization" in item.metadata for item in selected)
        )

    def test_client_reported_subset_cannot_shrink_active_member_contract(self):
        subset = ["edge_node_0", "edge_node_1"]
        events = [_event(index, expected_members=subset) for index in range(2)]
        decisions = [_decision(event, [_action(45)]) for event in events]
        selected, metadata = self.objective.optimize(
            events,
            decisions,
            TrafficPlugin().action_conflict,
            "active",
        )
        self.assertFalse(metadata["authoritative_input"])
        self.assertFalse(metadata["applied"])
        self.assertEqual(
            metadata["aggregation_completeness_reason"],
            "expected_member_contract_mismatch",
        )
        self.assertEqual(
            [[action.to_dict() for action in item.actions] for item in selected],
            [[action.to_dict() for action in item.actions] for item in decisions],
        )

    def test_client_metadata_without_cloud_lease_authority_cannot_apply(self):
        events = [_event(index) for index in range(4)]
        for event in events:
            event.metadata["aggregation"].pop("authority")
        decisions = [_decision(event, [_action(45)]) for event in events]
        _selected, metadata = self.objective.optimize(
            events,
            decisions,
            TrafficPlugin().action_conflict,
            "active",
        )
        self.assertFalse(metadata["applied"])
        self.assertEqual(
            metadata["aggregation_completeness_reason"],
            "untrusted_aggregation_context",
        )

    def test_partial_aggregation_cannot_apply_active_plan(self):
        events = [_event(0), _event(1)]
        decisions = [_decision(event, [_action(45)]) for event in events]
        selected, metadata = self.objective.optimize(
            events,
            decisions,
            TrafficPlugin().action_conflict,
            "active",
        )
        self.assertFalse(metadata["authoritative_input"])
        self.assertFalse(metadata["applied"])
        self.assertEqual(metadata["application_blocked_reason"], "incomplete_aggregation")
        self.assertEqual(
            [[action.to_dict() for action in item.actions] for item in selected],
            [[action.to_dict() for action in item.actions] for item in decisions],
        )

    def test_missing_expected_members_cannot_apply_active_plan(self):
        events = [_event(index) for index in range(4)]
        for event in events:
            event.metadata["aggregation"].pop("expected_members")
        decisions = [_decision(event, [_action(45)]) for event in events]
        selected, metadata = self.objective.optimize(
            events,
            decisions,
            TrafficPlugin().action_conflict,
            "active",
        )
        self.assertFalse(metadata["authoritative_input"])
        self.assertFalse(metadata["applied"])
        self.assertEqual(metadata["application_blocked_reason"], "incomplete_aggregation")
        self.assertEqual(
            [[action.to_dict() for action in item.actions] for item in selected],
            [[action.to_dict() for action in item.actions] for item in decisions],
        )

    def test_framework_coordinator_invokes_scene_optimizer(self):
        events = [_event(index) for index in range(4)]
        decisions = [_decision(event, [_action(45, "corridor:{}".format(index))])
                     for index, event in enumerate(events)]
        plugin = TrafficPlugin(
            global_objective_path=OBJECTIVE_PATH,
            global_optimizer_mode="shadow",
        )
        result = ConflictCoordinator(SceneRegistry([plugin])).coordinate(
            events,
            decisions,
        )
        payload = result.to_dict()
        self.assertEqual(len(payload["global_optimizations"]), 1)
        self.assertEqual(
            payload["global_optimizations"][0]["objective_id"],
            "traffic_candidate_global_utility_v1",
        )
        self.assertFalse(payload["global_optimizations"][0]["applied"])


if __name__ == "__main__":
    unittest.main()
