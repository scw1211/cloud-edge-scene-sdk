"""Regression and ownership tests for the three-stage routing pipeline."""

from cloud_edge_framework.contracts import SemanticEvent
from cloud_edge_framework.routing_pipeline import (
    apply_route_selection,
    execution_failover,
    feasible_action_filter,
    route_selector,
)
from cloud_edge_framework.scheduling import CollaborationScheduler, NetworkSnapshot


def _event(risk="low", confidence=0.95, deadline_ms=200.0):
    return SemanticEvent.from_dict(
        {
            "schema_version": "1.0",
            "event_id": "routing-regression",
            "scene": "routing-fixture",
            "task": "control",
            "edge_id": "edge-a",
            "occurred_at_ms": 1000,
            "scope": {
                "entity_id": "entity-a",
                "subsystem": "fixture",
                "state_variable": "risk",
                "region_id": "region-a",
                "window_start_ms": 900,
                "window_end_ms": 1000,
            },
            "prediction": {
                "label": risk,
                "confidence": confidence,
                "probabilities": {risk: confidence},
            },
            "risk": {"level": risk, "score": 0.9 if risk == "high" else 0.1},
            "uncertainty": {
                "confidence": confidence,
                "calibrated": True,
                "prediction_set": [risk],
                "method": "fixture",
            },
            "timing": {
                "deadline_ms": deadline_ms,
                "preprocessing_ms": 1.0,
                "edge_inference_ms": 2.0,
            },
            "evidence": [
                {
                    "evidence_id": "routing-summary",
                    "level": "summary",
                    "modality": "fixture",
                    "encoding": "json",
                    "inline": {"risk": risk},
                    "size_bytes": 16,
                    "content_type": "application/json",
                }
            ],
            "candidate_actions": [],
        }
    )


def _new_rule_path(schedule, network, summary=False, provisional_first=False):
    effective_cloud = network.available and network.loss_rate < 0.95
    prohibitions = {}
    if summary and effective_cloud:
        prohibitions["edge_only"] = "SUMMARY_DELIVERY_REQUIRED"
    if provisional_first:
        prohibitions["cloud_sync"] = "CALLER_REQUIRES_PROVISIONAL_FIRST"
    predicted = schedule.predicted_closed_loop_ms
    if network.loss_rate >= 0.20:
        predicted = max(predicted, schedule.deadline_ms + 1.0)
    feasibility = feasible_action_filter(
        cloud_available=effective_cloud,
        predicted_sync_ms=predicted,
        sync_deadline_ms=schedule.deadline_ms,
        prohibited_actions=prohibitions,
    )
    selection = route_selector(
        rule_decision=schedule,
        feasibility=feasibility,
    )
    failover = execution_failover(
        selection.selected_action,
        cloud_available=effective_cloud,
    )
    return apply_route_selection(schedule, selection, failover)


def _legacy_effective_route(schedule, network, summary=False, provisional_first=False):
    route = schedule.route
    if provisional_first and route == "cloud_sync":
        route = "cloud_async"
    if (
        summary
        and network.available
        and network.loss_rate < 0.95
        and route == "edge_only"
    ):
        route = "cloud_async"
    return route


def test_frozen_rule_path_outputs_are_equivalent_before_and_after_slimming():
    scheduler = CollaborationScheduler()
    normal = NetworkSnapshot()
    weak = NetworkSnapshot(loss_rate=0.30)
    outage = NetworkSnapshot(available=False, loss_rate=1.0)
    cases = [
        (_event(), normal, False, False),
        (_event(risk="high"), normal, False, False),
        (_event(risk="high"), weak, False, False),
        (_event(), outage, False, False),
        (_event(), normal, True, False),
        (_event(risk="high"), normal, False, True),
        (_event(risk="high"), normal, True, True),
    ]
    for event, network, summary, provisional_first in cases:
        schedule = scheduler.schedule(event, network)
        observed = _new_rule_path(
            schedule,
            network,
            summary=summary,
            provisional_first=provisional_first,
        )
        expected = _legacy_effective_route(
            schedule,
            network,
            summary=summary,
            provisional_first=provisional_first,
        )
        assert observed.route == expected
        assert observed.waits_for_cloud is (expected == "cloud_sync")
        assert observed.cloud_requested is (
            expected in {"cloud_async", "cloud_sync"}
        )


def test_feasibility_has_only_availability_deadline_and_business_bans():
    result = feasible_action_filter(
        cloud_available=True,
        predicted_sync_ms=10.0,
        sync_deadline_ms=20.0,
        prohibited_actions={"edge_only": "BUSINESS_REQUIRES_CLOUD"},
    )
    assert result.feasible_actions == ("cloud_async", "cloud_sync")
    assert result.rejected_actions == {
        "edge_only": "BUSINESS_REQUIRES_CLOUD"
    }


def test_route_selector_keeps_collaboration_scheduler_as_sole_authority():
    schedule = CollaborationScheduler().schedule(_event(), NetworkSnapshot())
    selection = route_selector(
        rule_decision=schedule,
        feasibility=feasible_action_filter(
            cloud_available=True,
            predicted_sync_ms=10.0,
            sync_deadline_ms=20.0,
        ),
    )
    assert selection.selected_action == schedule.route
    assert selection.selector_authority == "CollaborationScheduler"


def test_execution_failover_is_the_only_local_autonomy_trigger():
    edge = execution_failover(
        "edge_only", cloud_available=True, cloud_execution_failed=True
    )
    cloud = execution_failover(
        "cloud_sync", cloud_available=True, cloud_execution_failed=True
    )
    outage = execution_failover("edge_only", cloud_available=False)
    assert edge.execution_route == "edge_only"
    assert not edge.triggered
    assert cloud.execution_route == "local_autonomy"
    assert cloud.triggered
    assert outage.execution_route == "local_autonomy"
    assert outage.triggered
