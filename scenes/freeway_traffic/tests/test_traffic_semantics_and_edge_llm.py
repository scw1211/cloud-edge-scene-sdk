"""Traffic semantic separation and selective Edge-Qwen routing regressions."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRAFFIC_ROOT = REPOSITORY_ROOT / "scenes" / "freeway_traffic"
for import_root in (REPOSITORY_ROOT, TRAFFIC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from cloud_edge_framework.contracts import build_decision  # noqa: E402
from cloud_edge_framework.event_envelope import SceneEventEnvelope  # noqa: E402
from cloud_edge_framework.registry import SceneRegistry  # noqa: E402
from cloud_edge_framework.runtime import CloudRuntime  # noqa: E402
from freeway_traffic_full.edge_llm import TrafficEdgeLLMController  # noqa: E402
from freeway_traffic_full.plugin_impl import TrafficPlugin  # noqa: E402


def _raw_event(**overrides):
    data = {
        "sample_id": 100,
        "sample_split": "test",
        "dataset": "PEMS08",
        "region_id": "region_0",
        "prediction_horizon_minutes": 60,
        "managed_node_ids": [1, 2, 3, 4],
        "region_summary": {
            "region_risk_level": "severe",
            "region_risk_score": 0.94,
            "region_risk_confidence": 0.95,
            "region_risk_probabilities": {
                "low": 0.01,
                "medium": 0.01,
                "high": 0.03,
                "severe": 0.95,
            },
            "region_risk_calibration": {
                "method": "test_calibration",
                "calibrated_confidence": 0.95,
                "prediction_set": ["severe"],
            },
            "node_risk_counts": {
                "low": 4,
                "medium": 0,
                "high": 0,
                "severe": 0,
            },
        },
        "top_k_risk_nodes": [
            {
                "node_id": 4,
                "risk_level": "severe",
                "risk_score": 0.98,
                "risk_confidence": 0.97,
            }
        ],
        "control_capabilities": {
            "variable_speed_limit_nodes": [1, 2, 3, 4],
            "ramp_meter_nodes": [4],
        },
        "operational_safety_risk": {
            "level": "low",
            "score": 0.2,
            "source": "test_action_authority",
        },
        "upload_required": False,
        "escalation_expected_gain": {
            "edge_qwen": -0.1,
            "cloud": 0.1,
            "source": "held_out_role_ablation",
        },
        "deadline_ms": 500,
        "preprocessing_latency_ms": 2,
        "inference_latency_ms": 20,
        "model": "joint_astgcn",
    }
    data.update(overrides)
    return {
        "specversion": "1.0",
        "id": "traffic-semantic-test",
        "source": "urn:edge:test:astgcn",
        "type": "com.cloudedge.traffic.edge-event.v1",
        "scene": "freeway_traffic_management",
        "edgeid": "edge_node_0",
        "subject": "100",
        "time": "2026-08-01T00:00:00Z",
        "datacontenttype": "application/json",
        "dataschema": (
            "https://cloud-edge.local/schemas/scenes/traffic-edge-event-v1.json"
        ),
        "data": data,
    }


def _normalize(**overrides):
    plugin = TrafficPlugin()
    envelope = SceneEventEnvelope.from_dict(_raw_event(**overrides))
    return plugin, plugin.normalize(envelope)


def _student(event, confidence=0.95, decision="no_action", metadata=None):
    result = build_decision(
        event=event,
        decision=decision,
        actions=[],
        confidence=confidence,
        reason="test student",
        source="test_student",
        policy_version="test",
    )
    return replace(result, metadata={**result.metadata, **(metadata or {})})


def _controller():
    controller = TrafficEdgeLLMController(
        Path("unused-release.json"),
        Path("unused-runtime.json"),
        mode="selective",
        student_confidence_threshold=0.75,
        min_expected_gain=0.05,
    )
    controller.active = SimpleNamespace(
        model=SimpleNamespace(
            validation={"metrics": {"average_ttft_ms": 10.0}}
        )
    )
    return controller


class TrafficConstructorCompatibilityTests(unittest.TestCase):
    def test_edge_llm_old_positional_arguments_keep_their_meaning(self):
        controller = TrafficEdgeLLMController(
            None,
            None,
            "disabled",
            "medium",
            0.75,
            23.0,
            7,
            9.0,
        )
        self.assertEqual(controller.deadline_margin_ms, 23.0)
        self.assertEqual(controller.deadline_probe_interval, 7)
        self.assertEqual(controller.runtime_failure_cooldown_seconds, 9.0)
        self.assertEqual(controller.min_expected_gain, 0.05)

    def test_plugin_old_positional_arguments_keep_policy_version(self):
        plugin = TrafficPlugin(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "disabled",
            "medium",
            0.75,
            23.0,
            7,
            9.0,
            "legacy-policy",
        )
        self.assertEqual(plugin.policy_version, "legacy-policy")
        self.assertEqual(plugin._edge_llm.deadline_margin_ms, 23.0)
        self.assertEqual(plugin._edge_llm.min_expected_gain, 0.05)
        self.assertEqual(plugin.cloud_llm_min_expected_gain, 0.05)


class TrafficSemanticTests(unittest.TestCase):
    def test_normalize_emits_five_independent_semantics(self):
        _, event = _normalize()

        self.assertEqual(event.metadata["traffic_semantics_version"], "2.0")
        self.assertEqual(event.metadata["regional_state"]["level"], "severe")
        self.assertEqual(
            event.metadata["operational_safety_risk"]["level"], "low"
        )
        self.assertEqual(event.metadata["model_uncertainty"]["prediction_set_size"], 1)
        self.assertEqual(event.metadata["escalation_expected_gain"]["edge_qwen"], -0.1)
        self.assertTrue(event.metadata["evidence_completeness"]["complete"])

        # The public field is retained only as a compatibility projection. New
        # action/model routing must use the explicit scene semantics above.
        self.assertEqual(event.risk.level, "severe")
        self.assertIn("max(regional_state", event.metadata["legacy_risk_semantics"])

    def test_summary_is_default_and_feature_is_explicit(self):
        plugin, ordinary = _normalize()
        _, escalated = _normalize(upload_level="feature", upload_required=True)

        self.assertEqual(ordinary.metadata["minimum_evidence_level"], "summary")
        self.assertEqual(
            ordinary.metadata["evidence_completeness"]["minimum_required_level"],
            "summary",
        )
        self.assertEqual(escalated.metadata["minimum_evidence_level"], "feature")
        self.assertEqual(
            plugin.evidence_advice(ordinary, _student(ordinary))["required_level"],
            "summary",
        )
        self.assertEqual(
            plugin.evidence_advice(escalated, _student(escalated))["required_level"],
            "feature",
        )

    def test_evidence_policy_upgrades_for_uncertainty_conflict_or_raw_hint(self):
        plugin, event = _normalize()
        uncertain = replace(
            event,
            metadata={
                **event.metadata,
                "model_uncertainty": {
                    **event.metadata["model_uncertainty"],
                    "requires_review": True,
                },
            },
        )
        _, raw = _normalize(upload_level="raw", upload_required=True)

        self.assertEqual(
            plugin.evidence_advice(uncertain, _student(uncertain))["required_level"],
            "feature",
        )
        self.assertEqual(
            plugin.evidence_advice(
                event,
                _student(
                    event,
                    metadata={
                        "model_uncertainty": {
                            "requires_review": True,
                            "student_rule_disagreement": True,
                        }
                    },
                ),
            )["required_level"],
            "feature",
        )
        self.assertEqual(
            plugin.evidence_advice(
                event, _student(event), conflict_suspected=True
            )["required_level"],
            "feature",
        )
        self.assertEqual(
            plugin.evidence_advice(raw, _student(raw))["required_level"], "raw"
        )

    def test_evidence_policy_uses_real_edge_decision_signals(self):
        model_root = TRAFFIC_ROOT / "assets" / "models"
        plugin = TrafficPlugin(
            edge_student_path=model_root / "edge_student_freeway_joint_metis4.json",
            feature_codec_path=(
                model_root / "traffic_tree_feature_codec_topology_v1.npz"
            ),
        )
        event = plugin.normalize(SceneEventEnvelope.from_dict(_raw_event()))
        local = plugin.edge_decide(event)

        self.assertIsNotNone(
            local.metadata["model_uncertainty"]["student_confidence"]
        )
        self.assertTrue(
            local.metadata["model_uncertainty"]["student_rule_disagreement"]
        )
        self.assertFalse(event.metadata["model_uncertainty"]["requires_review"])
        self.assertEqual(
            plugin.evidence_advice(event, local)["required_level"], "feature"
        )

    def test_member_completeness_is_separate_from_evidence_level(self):
        _, event = _normalize(
            num_partitions=2,
            aggregation_member="edge_node_0",
            aggregation_expected_members=["edge_node_0", "edge_node_1"],
            aggregation_minimum_members=1,
        )
        completeness = event.metadata["evidence_completeness"]
        self.assertTrue(completeness["minimum_level_available"])
        self.assertFalse(completeness["aggregation_complete"])
        self.assertFalse(completeness["complete"])

    def test_cloud_llm_policy_ignores_risk_and_requires_uncertainty_plus_gain(self):
        plugin, ordinary = _normalize()
        ordinary_policy = ordinary.metadata["cloud_llm_review_policy"]
        self.assertEqual(ordinary.risk.level, "severe")
        self.assertFalse(ordinary_policy["eligible"])
        self.assertFalse(ordinary_policy["legacy_risk_trigger_used"])

        raw = _raw_event()
        raw["data"]["region_summary"]["region_risk_calibration"][
            "prediction_set"
        ] = ["high", "severe"]
        uncertain_with_gain = plugin.normalize(SceneEventEnvelope.from_dict(raw))
        policy = uncertain_with_gain.metadata["cloud_llm_review_policy"]
        self.assertTrue(policy["eligible"])
        self.assertEqual(policy["reason"], "traffic_uncertainty_with_cloud_gain")

    def test_cloud_llm_policy_rejects_unestimated_gain_but_allows_explicit_request(self):
        plugin = TrafficPlugin()
        raw = _raw_event()
        raw["data"].pop("escalation_expected_gain")
        raw["data"]["region_summary"]["region_risk_calibration"][
            "prediction_set"
        ] = ["high", "severe"]
        unestimated = plugin.normalize(SceneEventEnvelope.from_dict(raw))
        self.assertFalse(unestimated.metadata["cloud_llm_review_policy"]["eligible"])
        self.assertEqual(
            unestimated.metadata["cloud_llm_review_policy"]["reason"],
            "traffic_cloud_gain_not_estimated",
        )

        explicit = plugin.normalize(
            SceneEventEnvelope.from_dict(
                _raw_event(cloud_llm_review_requested=True)
            )
        )
        self.assertTrue(explicit.metadata["cloud_llm_review_policy"]["eligible"])
        self.assertEqual(
            explicit.metadata["cloud_llm_review_policy"]["reason"],
            "traffic_explicit_cloud_llm_review",
        )

    def test_cloud_submission_uses_student_uncertainty_not_only_perception(self):
        plugin, event = _normalize()
        local = _student(
            event,
            confidence=0.55,
            metadata={
                "model_uncertainty": {
                    "requires_review": True,
                    "student_confidence": 0.55,
                    "student_rule_disagreement": True,
                },
                "escalation_expected_gain": event.metadata[
                    "escalation_expected_gain"
                ],
            },
        )
        metadata = plugin.cloud_submission_metadata(event, local)

        self.assertTrue(metadata["model_uncertainty"]["requires_review"])
        self.assertTrue(metadata["cloud_llm_review_policy"]["eligible"])
        self.assertEqual(
            metadata["cloud_llm_review_policy"]["reason"],
            "traffic_uncertainty_with_cloud_gain",
        )

    def test_cloud_llm_does_not_promote_broad_async_review_to_sync(self):
        plugin, event = _normalize()
        local = _student(
            event,
            confidence=0.55,
            metadata={
                "model_uncertainty": {
                    "requires_review": True,
                    "requires_synchronous_review": False,
                    "student_confidence": 0.55,
                    "student_rule_disagreement": True,
                },
                "escalation_expected_gain": event.metadata[
                    "escalation_expected_gain"
                ],
            },
        )
        metadata = plugin.cloud_submission_metadata(event, local)

        policy = metadata["cloud_llm_review_policy"]
        self.assertTrue(policy["model_uncertainty_requires_review"])
        self.assertFalse(
            policy["model_uncertainty_requires_synchronous_review"]
        )
        self.assertFalse(policy["eligible"])
        self.assertEqual(policy["reason"], "traffic_no_model_uncertainty")

        explicit_plugin, explicit_event = _normalize(
            cloud_llm_review_requested=True
        )
        explicit = explicit_plugin.cloud_submission_metadata(
            explicit_event,
            local,
        )
        self.assertTrue(explicit["cloud_llm_review_policy"]["eligible"])
        self.assertEqual(
            explicit["cloud_llm_review_policy"]["reason"],
            "traffic_explicit_cloud_llm_review",
        )

    def test_defer_preparation_preserves_student_confidence_and_disagreement(self):
        plugin = TrafficPlugin(edge_student_path=Path("student-not-loaded.json"))
        event = plugin.normalize(SceneEventEnvelope.from_dict(_raw_event()))
        prepared = plugin._apply_defer_gate(event, _student(event, 0.91, "reroute"))

        self.assertEqual(prepared.metadata["traffic_student_candidate_confidence"], 0.91)
        self.assertTrue(prepared.metadata["traffic_student_rule_disagreement"])
        self.assertTrue(
            prepared.metadata["model_uncertainty"]["student_rule_disagreement"]
        )

    def test_safety_floor_uses_action_safety_not_congestion_level(self):
        plugin, event = _normalize()
        decision = _student(event, 0.95, "no_action")
        unchanged = plugin._ensure_operational_safety(event, decision)
        self.assertEqual(unchanged.decision, "no_action")
        self.assertFalse(unchanged.metadata.get("operational_safety_override", False))

        high_safety = replace(
            event,
            metadata={
                **event.metadata,
                "operational_safety_risk": {
                    "level": "high",
                    "score": 0.8,
                    "source": "test",
                },
            },
        )
        overridden = plugin._ensure_operational_safety(high_safety, decision)
        self.assertTrue(overridden.metadata["operational_safety_override"])

        offline = replace(
            event,
            metadata={
                **event.metadata,
                "edge_runtime_network_available": False,
            },
        )
        cloud_only = _student(
            offline, 0.95, "regional_coordination"
        )
        offline_fallback = plugin._ensure_operational_safety(offline, cloud_only)
        self.assertTrue(offline_fallback.metadata["operational_safety_override"])
        self.assertEqual(
            offline_fallback.metadata["operational_safety_risk"]["level"], "high"
        )


class TrafficCloudBatchDecisionTests(unittest.TestCase):
    def setUp(self):
        model_root = TRAFFIC_ROOT / "assets" / "models"
        self.plugin = TrafficPlugin(
            cloud_model_path=(
                model_root / "cloud_coordinator_topology_fused.joblib"
            ),
            feature_codec_path=(
                model_root / "traffic_tree_feature_codec_topology_v1.npz"
            ),
            topology_path=(
                model_root / "traffic_region_topology_metis4.json"
            ),
        )
        self.plugin.warmup()
        self.events = []
        for partition_id in range(4):
            raw = _raw_event(
                sample_id=200,
                region_id="region_{}".format(partition_id),
                partition_id=partition_id,
                num_partitions=4,
                managed_node_ids=[partition_id * 4 + value for value in (1, 2, 3, 4)],
            )
            raw["id"] = "traffic-cloud-batch-{}".format(partition_id)
            raw["edgeid"] = "edge_node_{}".format(partition_id)
            event = self.plugin.normalize(SceneEventEnvelope.from_dict(raw))
            self.events.append(self.plugin.prepare_cloud_event(event, "feature"))

    @staticmethod
    def _without_batch_observability(decision):
        value = decision.to_dict()
        value["metadata"].pop("cloud_inference_batch_size", None)
        return value

    def test_batch_decisions_match_single_event_decisions(self):
        fused = list(self.plugin.fuse_cloud_context(self.events))
        individual = [self.plugin.cloud_decide(event) for event in fused]
        batched = list(self.plugin.cloud_decide_batch(fused))

        self.assertEqual(len(batched), 4)
        self.assertEqual(
            [self._without_batch_observability(item) for item in individual],
            [self._without_batch_observability(item) for item in batched],
        )
        self.assertTrue(
            all(item.metadata["cloud_inference_batch_size"] == 4 for item in batched)
        )

    def test_cloud_runtime_runs_one_probability_batch_for_four_regions(self):
        model = self.plugin._cloud_model["model"]
        original_predict = model.predict
        original_predict_proba = model.predict_proba
        probability_shapes = []

        def forbidden_predict(_matrix):
            raise AssertionError("batch cloud inference must not traverse trees twice")

        def counted_predict_proba(matrix):
            probability_shapes.append(tuple(matrix.shape))
            return original_predict_proba(matrix)

        model.predict = forbidden_predict
        model.predict_proba = counted_predict_proba
        registry = SceneRegistry([self.plugin])
        try:
            result = CloudRuntime(registry).coordinate(self.events)
        finally:
            model.predict = original_predict
            model.predict_proba = original_predict_proba

        self.assertEqual(probability_shapes, [(4, 226)])
        self.assertEqual(result["event_count"], 4)
        self.assertEqual(len(result["decisions"]), 4)

    def test_summary_only_events_use_lightweight_cloud_policy(self):
        summaries = [
            self.plugin.prepare_cloud_event(
                replace(
                    event,
                    evidence=[item for item in event.evidence if item.level == "summary"],
                ),
                "summary",
            )
            for event in self.events
        ]
        result = CloudRuntime(SceneRegistry([self.plugin])).coordinate(summaries)

        self.assertEqual(result["event_count"], 4)
        self.assertEqual(len(result["decisions"]), 4)
        self.assertTrue(
            all(
                item["metadata"]["cloud_inference_path"]
                == "semantic_summary_policy"
                for item in result["decisions"]
            )
        )
        self.assertTrue(
            all(
                item["metadata"]["cloud_summary_batch_size"] == 4
                for item in result["decisions"]
            )
        )

    def test_mixed_evidence_batches_only_feature_events_through_model(self):
        mixed = list(self.events)
        for index in (2, 3):
            mixed[index] = self.plugin.prepare_cloud_event(
                replace(
                    mixed[index],
                    evidence=[
                        item
                        for item in mixed[index].evidence
                        if item.level == "summary"
                    ],
                ),
                "summary",
            )
        model = self.plugin._cloud_model["model"]
        original_predict_proba = model.predict_proba
        probability_shapes = []

        def counted_predict_proba(matrix):
            probability_shapes.append(tuple(matrix.shape))
            return original_predict_proba(matrix)

        model.predict_proba = counted_predict_proba
        try:
            decisions = list(
                self.plugin.cloud_decide_batch(
                    self.plugin.fuse_cloud_context(mixed)
                )
            )
        finally:
            model.predict_proba = original_predict_proba

        self.assertEqual(probability_shapes, [(2, 226)])
        self.assertEqual(len(decisions), 4)
        self.assertEqual(
            sum(
                decision.metadata.get("cloud_inference_path")
                == "semantic_summary_policy"
                for decision in decisions
            ),
            2,
        )


class EdgeQwenSelectionTests(unittest.TestCase):
    def setUp(self):
        _, event = _normalize()
        self.event = replace(
            event,
            metadata={**event.metadata, "edge_runtime_network_available": True},
        )
        self.controller = _controller()

    def test_high_congestion_alone_does_not_select_qwen(self):
        selected, reason = self.controller._selection(
            self.event, _student(self.event, 0.95)
        )
        self.assertFalse(selected)
        self.assertEqual(reason, "no_model_escalation_signal")

    def test_student_uncertainty_without_gain_does_not_select_qwen(self):
        selected, reason = self.controller._selection(
            self.event, _student(self.event, 0.60)
        )
        self.assertFalse(selected)
        self.assertEqual(reason, "edge_qwen_expected_gain_below_threshold")

    def test_positive_gain_and_student_uncertainty_select_qwen(self):
        event = replace(
            self.event,
            metadata={
                **self.event.metadata,
                "escalation_expected_gain": {
                    "edge_qwen": 0.08,
                    "cloud": 0.0,
                    "source": "held_out_gain_model",
                },
            },
        )
        selected, reason = self.controller._selection(event, _student(event, 0.60))
        self.assertTrue(selected)
        self.assertIn("student_low_confidence", reason)
        self.assertIn("expected_gain=0.080000", reason)

    def test_prediction_set_ambiguity_selects_qwen(self):
        uncertainty = {
            **self.event.metadata["model_uncertainty"],
            "prediction_set": ["high", "severe"],
            "prediction_set_size": 2,
        }
        event = replace(
            self.event,
            metadata={
                **self.event.metadata,
                "model_uncertainty": uncertainty,
                "escalation_expected_gain": {
                    "edge_qwen": 0.08,
                    "cloud": 0.0,
                    "source": "held_out_gain_model",
                },
            },
        )
        selected, reason = self.controller._selection(event, _student(event, 0.95))
        self.assertTrue(selected)
        self.assertIn("prediction_set_ambiguous", reason)

    def test_student_rule_disagreement_selects_qwen(self):
        event = replace(
            self.event,
            metadata={
                **self.event.metadata,
                "escalation_expected_gain": {
                    "edge_qwen": 0.08,
                    "cloud": 0.0,
                    "source": "held_out_gain_model",
                },
            },
        )
        selected, reason = self.controller._selection(
            event,
            _student(
                event,
                0.95,
                metadata={"traffic_student_rule_disagreement": True},
            ),
        )
        self.assertTrue(selected)
        self.assertIn("student_rule_disagreement", reason)

    def test_positive_expected_gain_alone_does_not_select_qwen(self):
        event = replace(
            self.event,
            metadata={
                **self.event.metadata,
                "escalation_expected_gain": {
                    "edge_qwen": 0.08,
                    "cloud": 0.0,
                    "source": "held_out_gain_model",
                },
            },
        )
        selected, reason = self.controller._selection(event, _student(event, 0.95))
        self.assertFalse(selected)
        self.assertEqual(reason, "no_model_escalation_signal")


if __name__ == "__main__":
    unittest.main()
