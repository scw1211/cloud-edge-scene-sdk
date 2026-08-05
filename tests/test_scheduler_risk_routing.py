"""Safety boundaries for ordinary versus synchronous cloud routing."""

from __future__ import annotations

import unittest

from cloud_edge_framework.contracts import SemanticEvent
from cloud_edge_framework.scheduling import CollaborationScheduler, NetworkSnapshot


def _event(
    *,
    risk_level: str = "low",
    confidence: float = 0.95,
    prediction_set=None,
) -> SemanticEvent:
    return SemanticEvent.from_dict(
        {
            "schema_version": "1.0",
            "event_id": "scheduler-event",
            "scene": "scheduler-fixture",
            "task": "control",
            "edge_id": "edge-a",
            "occurred_at_ms": 1000,
            "scope": {
                "entity_id": "entity-a",
                "subsystem": "fixture",
                "state_variable": "state",
                "region_id": "region-a",
                "window_start_ms": 900,
                "window_end_ms": 1000,
            },
            "prediction": {
                "label": risk_level,
                "confidence": confidence,
                "probabilities": {risk_level: confidence},
            },
            "risk": {
                "level": risk_level,
                "score": 0.9 if risk_level in {"high", "severe"} else 0.1,
            },
            "uncertainty": {
                "confidence": confidence,
                "calibrated": True,
                "prediction_set": list(prediction_set or [risk_level]),
                "method": "fixture",
            },
            "timing": {
                "deadline_ms": 500.0,
                "preprocessing_ms": 1.0,
                "edge_inference_ms": 2.0,
            },
            "evidence": [
                {
                    "evidence_id": "scheduler-summary",
                    "level": "summary",
                    "modality": "fixture",
                    "encoding": "json",
                    "inline": {"risk": risk_level},
                    "size_bytes": 16,
                    "content_type": "application/json",
                }
            ],
            "candidate_actions": [],
        }
    )


class SchedulerRiskRoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = CollaborationScheduler()
        self.network = NetworkSnapshot()

    def test_ordinary_confident_low_risk_stays_local(self) -> None:
        result = self.scheduler.schedule(_event(), self.network)
        self.assertEqual(result.route, "edge_only")
        self.assertFalse(result.waits_for_cloud)

    def test_each_advanced_review_trigger_waits_when_feasible(self) -> None:
        cases = {
            "high_risk": (_event(risk_level="high"), {}),
            "uncertainty": (_event(confidence=0.5), {}),
            "model_disagreement": (_event(), {"model_disagreement": True}),
            "local_model_uncertainty": (_event(), {"decision_uncertain": True}),
            "cross_region_conflict": (_event(), {"conflict_suspected": True}),
            "policy_forced": (_event(), {"cloud_review_requested": True}),
        }
        for name, (event, controls) in cases.items():
            with self.subTest(name=name):
                result = self.scheduler.schedule(event, self.network, **controls)
                self.assertEqual(result.route, "cloud_sync")
                self.assertTrue(result.waits_for_cloud)

    def test_scene_authoritative_uncertainty_can_resolve_generic_confidence(self) -> None:
        result = self.scheduler.schedule(
            _event(confidence=0.5),
            self.network,
            decision_uncertain=False,
        )
        self.assertEqual(result.route, "edge_only")
        self.assertFalse(result.uncertain)

    def test_resolved_uncertainty_never_overrides_safety_boundaries(self) -> None:
        cases = {
            "high_risk": (_event(risk_level="high"), {}),
            "cross_region_conflict": (_event(), {"conflict_suspected": True}),
            "policy_forced": (_event(), {"cloud_review_requested": True}),
        }
        for name, (event, controls) in cases.items():
            with self.subTest(name=name):
                result = self.scheduler.schedule(
                    event,
                    self.network,
                    decision_uncertain=False,
                    **controls,
                )
                self.assertEqual(result.route, "cloud_sync")
                self.assertTrue(result.waits_for_cloud)

    def test_generic_possible_high_remains_synchronous(self) -> None:
        result = self.scheduler.schedule(
            _event(prediction_set=["high"]),
            self.network,
        )
        self.assertEqual(result.route, "cloud_sync")
        self.assertTrue(result.critical)

    def test_prediction_set_ambiguity_survives_scene_resolution(self) -> None:
        result = self.scheduler.schedule(
            _event(prediction_set=["low", "medium"]),
            self.network,
            decision_uncertain=False,
        )
        self.assertEqual(result.route, "cloud_sync")
        self.assertTrue(result.uncertain)

    def test_explicit_low_operational_risk_can_separate_descriptive_risk(self) -> None:
        result = self.scheduler.schedule(
            _event(prediction_set=["high"]),
            self.network,
            routing_risk_level="low",
        )
        self.assertEqual(result.route, "edge_only")
        self.assertFalse(result.critical)


if __name__ == "__main__":
    unittest.main()
