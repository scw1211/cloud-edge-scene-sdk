from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRAFFIC_ROOT = REPOSITORY_ROOT / "scenes" / "freeway_traffic"
for import_root in (REPOSITORY_ROOT, TRAFFIC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from cloud_edge_framework.contracts import Action, build_decision
from cloud_edge_framework.event_envelope import SceneEventEnvelope
from edge_llm_factory.adapter_package import validate_adapter_package
from edge_llm_factory.contracts import read_json_object
from edge_llm_factory.runtime import ActionDecoder
from freeway_traffic_full.edge_llm import TrafficEdgeLLMController
from freeway_traffic_full.plugin_impl import TrafficPlugin
from traffic_system.build_current_state_qwen_dataset import (
    outcome_reasons,
    routing_reasons,
)
from traffic_system.ultracompact_codec import (
    decode_routing_context_v2_prompt,
    encode_routing_context_v2_prompt,
)


class CurrentStateQwenContractTests(unittest.TestCase):
    @staticmethod
    def _current_state_event():
        raw = {
            "specversion": "1.0",
            "id": "current-state-qwen-advisory-test",
            "source": "urn:edge:test",
            "type": "com.cloudedge.traffic.edge-event.v1",
            "scene": "freeway_traffic_management",
            "edgeid": "edge_node_0",
            "subject": "125",
            "time": "2026-08-03T00:00:00Z",
            "datacontenttype": "application/json",
            "dataschema": "https://cloud-edge.local/schemas/scenes/traffic-edge-event-v1.json",
            "data": {
                "sample_id": 125,
                "sample_split": "test",
                "dataset": "PEMS08",
                "region_id": "region_0",
                "prediction_horizon_minutes": 0,
                "managed_node_ids": [1],
                "region_summary": {
                    "region_risk_level": "low",
                    "region_risk_score": 0.2,
                    "region_risk_confidence": 0.87,
                    "region_risk_probabilities": {
                        "low": 0.87,
                        "medium": 0.1,
                        "high": 0.02,
                        "severe": 0.01,
                    },
                    "node_risk_counts": {
                        "low": 1,
                        "medium": 0,
                        "high": 0,
                        "severe": 0,
                    },
                },
                "top_k_risk_nodes": [],
                "control_capabilities": {},
                "upload_required": False,
                "deadline_ms": 500,
                "preprocessing_latency_ms": 1,
                "inference_latency_ms": 2,
                "perception_mode": "current_state",
                "output_type": "current_state_risk",
            },
        }
        event = TrafficPlugin().normalize(SceneEventEnvelope.from_dict(raw))
        return replace(
            event,
            metadata={
                **event.metadata,
                "edge_runtime_network_available": True,
                "edge_runtime_network_status": "normal",
            },
        )

    @staticmethod
    def _advisory():
        return Action(
            action_type="traffic_advisory",
            target_ids=["traffic_node:1"],
            resource_ids=["traffic_node:1"],
            parameters={"strategy": "issue_congestion_warning"},
            reason="Student recommends an early congestion warning.",
            priority=30,
        )

    @staticmethod
    def _decoder_controller():
        base = read_json_object(
            TRAFFIC_ROOT / "assets" / "edge_llm" / "base_manifest.json"
        )
        mapping = read_json_object(
            TRAFFIC_ROOT
            / "assets"
            / "edge_llm"
            / "adapter_package_current_state_v2"
            / "action_mapping.json"
        )
        decoder = ActionDecoder(base, mapping)

        class DecoderBackedModel:
            validation = {"metrics": {"decision_accuracy": 0.6613}}

            @staticmethod
            def decide(_prompt, event, network_available):
                return {
                    "inference": {
                        "slot": "B",
                        "token": "B",
                        "latency_ms": 1.0,
                        "prompt_tokens": 16,
                        "output_tokens": 1,
                    },
                    "decision": decoder.decode("B", event, network_available),
                }

        controller = TrafficEdgeLLMController(
            Path("release.json"), Path("runtime.json"), mode="primary"
        )
        controller.context_encoder = "freeway-routing-context-decimal@v2"
        controller.active = SimpleNamespace(
            release_id="current-state-test",
            revision=1,
            model=DecoderBackedModel(),
        )
        return controller

    def test_routing_reasons_do_not_read_future_correctness(self) -> None:
        reasons = routing_reasons(
            student_decision="no_action",
            student_confidence=0.95,
            rule_decision="no_action",
            prediction_set_size=1,
            confidence_threshold=0.75,
        )
        outcomes = outcome_reasons(
            target="reroute",
            student_decision="no_action",
            rule_decision="no_action",
            reference="reroute",
        )

        self.assertEqual(reasons, [])
        self.assertIn("student_wrong", outcomes)
        self.assertIn("rule_wrong", outcomes)
        self.assertIn("future_action_requires_coordination", outcomes)

    def test_routing_context_v2_round_trip(self) -> None:
        legacy = "r1t0l10m5h2s1q75v42o18c1a1g0"
        context = {
            "student_decision": "variable_speed_limit",
            "rule_decision": "regional_coordination",
            "student_confidence": 0.62,
            "prediction_set_size": 2,
            "network_status": "weak",
        }
        encoded = encode_routing_context_v2_prompt(legacy, context)
        decoded = decode_routing_context_v2_prompt(encoded)

        self.assertEqual(len(encoded), 16)
        self.assertTrue(encoded.isdigit())
        self.assertEqual(decoded["traffic_code"], "102100754218")
        self.assertEqual(decoded["student_decision"], "variable_speed_limit")
        self.assertEqual(decoded["rule_decision"], "regional_coordination")
        self.assertEqual(decoded["student_confidence_bucket"], 2)
        self.assertEqual(decoded["prediction_set_size"], 2)
        self.assertEqual(decoded["network_status"], "weak")

    def test_routing_context_v2_changes_for_each_routing_signal(self) -> None:
        legacy = "r0t1l10m15h5s0q25v55o12c0a1g1"
        base = {
            "student_decision": "no_action",
            "rule_decision": "no_action",
            "student_confidence": 0.9,
            "prediction_set_size": 1,
            "network_status": "normal",
        }
        variants = []
        for key, value in (
            ("student_decision", "reroute"),
            ("rule_decision", "congestion_warning"),
            ("student_confidence", 0.1),
            ("prediction_set_size", 2),
            ("network_status", "offline"),
        ):
            changed = dict(base)
            changed[key] = value
            variants.append(encode_routing_context_v2_prompt(legacy, changed))

        encoded_base = encode_routing_context_v2_prompt(legacy, base)
        self.assertTrue(all(value != encoded_base for value in variants))
        self.assertEqual(len(set(variants)), len(variants))

    def test_current_state_payload_selects_current_state_student(self) -> None:
        plugin = TrafficPlugin(
            edge_student_path=Path("forecast.json"),
            current_state_edge_student_path=Path("current.json"),
        )
        plugin._edge_student = {"name": "forecast"}
        plugin._current_state_edge_student = {"name": "current"}

        model, path, contract = plugin._edge_student_for_payload(
            {"perception_mode": "current_state"}
        )
        self.assertEqual(model["name"], "current")
        self.assertEqual(path.name, "current.json")
        self.assertEqual(contract, "current_state_future_v1")

        model, path, contract = plugin._edge_student_for_payload(
            {"perception_mode": "astgcn"}
        )
        self.assertEqual(model["name"], "forecast")
        self.assertEqual(path.name, "forecast.json")
        self.assertEqual(contract, "forecast_joint_v1")

    def test_current_state_separates_async_and_synchronous_uncertainty(self) -> None:
        event = self._current_state_event()
        plugin = TrafficPlugin(
            current_state_edge_student_path=Path("current.json"),
            current_state_sync_confidence_threshold=0.50,
        )

        cases = (
            ("congestion_warning", 0.90, True, False),
            ("congestion_warning", 0.60, True, False),
            ("congestion_warning", 0.49, True, True),
            ("no_action", 0.49, True, False),
        )
        for decision_name, confidence, broad_review, synchronous in cases:
            with self.subTest(decision=decision_name, confidence=confidence):
                student = build_decision(
                    event=event,
                    decision=decision_name,
                    actions=[self._advisory()]
                    if decision_name == "congestion_warning"
                    else [],
                    confidence=confidence,
                    reason="test Student decision",
                    source="test_student",
                    policy_version="test",
                )
                selected = plugin._apply_defer_gate(event, student)
                uncertainty = selected.metadata["model_uncertainty"]
                self.assertEqual(
                    uncertainty["requires_review"], broad_review
                )
                self.assertEqual(
                    uncertainty["requires_synchronous_review"], synchronous
                )

        ambiguous_event = replace(
            event,
            metadata={
                **event.metadata,
                "model_uncertainty": {
                    **event.metadata["model_uncertainty"],
                    "prediction_set": ["low", "medium"],
                },
            },
        )
        student = build_decision(
            event=ambiguous_event,
            decision="no_action",
            actions=[],
            confidence=0.90,
            reason="test ambiguous prediction set",
            source="test_student",
            policy_version="test",
        )
        uncertainty = plugin._apply_defer_gate(
            ambiguous_event, student
        ).metadata["model_uncertainty"]
        self.assertTrue(uncertainty["requires_synchronous_review"])
        self.assertIn(
            "prediction_set_ambiguous",
            uncertainty["synchronous_review_reasons"],
        )

    def test_agreeing_qwen_resolves_only_student_synchronous_uncertainty(self) -> None:
        event = self._current_state_event()
        plugin = TrafficPlugin(current_state_edge_student_path=Path("current.json"))
        student = build_decision(
            event=event,
            decision="congestion_warning",
            actions=[self._advisory()],
            confidence=0.49,
            reason="test Student warning",
            source="test_student",
            policy_version="test",
        )
        selected = plugin._apply_defer_gate(event, student)
        metadata = {
            **selected.metadata,
            "edge_decision_path": "edge_qwen",
            "edge_llm_model_disagreement": False,
            "edge_llm_requires_cloud": False,
        }
        corroborated = plugin._resolve_current_state_synchronous_uncertainty(
            event,
            replace(selected, metadata=metadata),
        )
        uncertainty = corroborated.metadata["model_uncertainty"]
        self.assertTrue(uncertainty["requires_review"])
        self.assertFalse(uncertainty["requires_synchronous_review"])
        self.assertEqual(
            uncertainty["synchronous_review_resolution"],
            "edge_qwen_corroborated_student",
        )

        disagreement = plugin._resolve_current_state_synchronous_uncertainty(
            event,
            replace(
                selected,
                metadata={
                    **metadata,
                    "edge_llm_model_disagreement": True,
                },
            ),
        )
        self.assertTrue(
            disagreement.metadata["model_uncertainty"][
                "requires_synchronous_review"
            ]
        )

        missing_flags = plugin._resolve_current_state_synchronous_uncertainty(
            event,
            replace(
                selected,
                metadata={
                    **selected.metadata,
                    "edge_decision_path": "edge_qwen",
                },
            ),
        )
        self.assertTrue(
            missing_flags.metadata["model_uncertainty"][
                "requires_synchronous_review"
            ]
        )

        requires_cloud = plugin._resolve_current_state_synchronous_uncertainty(
            event,
            replace(
                selected,
                metadata={
                    **metadata,
                    "edge_llm_requires_cloud": True,
                },
            ),
        )
        self.assertTrue(
            requires_cloud.metadata["model_uncertainty"][
                "requires_synchronous_review"
            ]
        )

        unknown_reason = plugin._resolve_current_state_synchronous_uncertainty(
            event,
            replace(
                selected,
                metadata={
                    **metadata,
                    "model_uncertainty": {
                        **selected.metadata["model_uncertainty"],
                        "synchronous_review_reasons": ["unknown_signal"],
                    },
                },
            ),
        )
        self.assertTrue(
            unknown_reason.metadata["model_uncertainty"][
                "requires_synchronous_review"
            ]
        )

        ambiguous_uncertainty = {
            **selected.metadata["model_uncertainty"],
            "synchronous_review_reasons": [
                "student_no_majority_and_rule_disagreement",
                "prediction_set_ambiguous",
            ],
        }
        ambiguous = plugin._resolve_current_state_synchronous_uncertainty(
            event,
            replace(
                selected,
                metadata={
                    **metadata,
                    "model_uncertainty": ambiguous_uncertainty,
                },
            ),
        )
        self.assertTrue(
            ambiguous.metadata["model_uncertainty"][
                "requires_synchronous_review"
            ]
        )
        self.assertEqual(
            ambiguous.metadata["model_uncertainty"][
                "synchronous_review_reasons"
            ],
            ["prediction_set_ambiguous"],
        )

    def test_gain_profile_routes_only_validation_accepted_stratum(self) -> None:
        raw = {
            "specversion": "1.0",
            "id": "gain-profile-test",
            "source": "urn:edge:test",
            "type": "com.cloudedge.traffic.edge-event.v1",
            "scene": "freeway_traffic_management",
            "edgeid": "edge_node_0",
            "subject": "0",
            "time": "2026-08-03T00:00:00Z",
            "datacontenttype": "application/json",
            "dataschema": "https://cloud-edge.local/schemas/scenes/traffic-edge-event-v1.json",
            "data": {
                "sample_id": 0,
                "sample_split": "test",
                "dataset": "PEMS08",
                "region_id": "region_0",
                "prediction_horizon_minutes": 0,
                "managed_node_ids": [0],
                "region_summary": {
                    "region_risk_level": "low",
                    "region_risk_score": 0.1,
                    "region_risk_confidence": 0.9,
                    "region_risk_probabilities": {
                        "low": 0.9,
                        "medium": 0.1,
                        "high": 0.0,
                        "severe": 0.0,
                    },
                    "node_risk_counts": {
                        "low": 1,
                        "medium": 0,
                        "high": 0,
                        "severe": 0,
                    },
                },
                "top_k_risk_nodes": [],
                "control_capabilities": {},
                "upload_required": False,
                "deadline_ms": 500,
                "preprocessing_latency_ms": 1,
                "inference_latency_ms": 2,
                "perception_mode": "current_state",
                "output_type": "current_state_risk",
            },
        }
        plugin = TrafficPlugin()
        event = plugin.normalize(SceneEventEnvelope.from_dict(raw))
        event = replace(
            event,
            metadata={
                **event.metadata,
                "edge_runtime_network_available": True,
                "edge_runtime_network_status": "normal",
                "escalation_expected_gain": {},
            },
        )
        student = build_decision(
            event=event,
            decision="no_action",
            actions=[],
            confidence=0.95,
            reason="test",
            source="test_student",
            policy_version="test",
        )
        student = replace(
            student,
            metadata={
                **student.metadata,
                "traffic_student_candidate_decision": "no_action",
                "traffic_student_candidate_confidence": 0.95,
                "traffic_rule_candidate_decision": "congestion_warning",
                "traffic_student_rule_disagreement": True,
                "model_uncertainty": {
                    "student_available": True,
                    "student_confidence": 0.95,
                    "student_rule_disagreement": True,
                    "prediction_set": ["low"],
                },
            },
        )
        controller = TrafficEdgeLLMController(
            Path("release.json"),
            Path("runtime.json"),
            mode="selective",
            min_expected_gain=0.05,
        )
        controller.active = type(
            "Active",
            (),
            {"model": type("Model", (), {"validation": {"metrics": {"average_ttft_ms": 10.0}}})()},
        )()
        controller.gain_profile = {
            "accepted_strata": {
                "normal|no_action|congestion_warning|3|0": {
                    "validation_gain": 0.2
                }
            }
        }

        selected, reason = controller._selection(event, student)
        self.assertTrue(selected)
        self.assertIn("expected_gain=0.200000", reason)
        gain, source, qualified = controller._expected_gain(event, student)
        self.assertEqual(gain, 0.2)
        self.assertEqual(source, "validated_current_state_gain_profile")
        self.assertTrue(qualified)

        controller.gain_profile = {"accepted_strata": {}}
        selected, reason = controller._selection(event, student)
        self.assertFalse(selected)
        self.assertEqual(reason, "edge_qwen_expected_gain_below_threshold")

    def test_future_warning_can_use_only_student_authorized_advisory(self) -> None:
        event = self._current_state_event()
        student = build_decision(
            event=event,
            decision="congestion_warning",
            actions=[self._advisory()],
            confidence=0.371,
            reason="test Student warning",
            source="test_student",
            policy_version="test",
        )

        decision = self._decoder_controller().decide(event, student, "test")

        self.assertEqual(decision.decision, "congestion_warning")
        self.assertEqual(
            [action.action_type for action in decision.actions],
            ["traffic_advisory"],
        )
        self.assertEqual(decision.metadata["edge_decision_path"], "edge_qwen")
        self.assertFalse(decision.metadata["edge_llm_safety_fallback"])
        self.assertFalse(decision.metadata["edge_llm_model_disagreement"])
        self.assertTrue(
            decision.metadata["edge_llm_student_advisory_whitelisted"]
        )

    def test_same_decision_with_different_action_semantics_is_disagreement(self) -> None:
        event = self._current_state_event()
        student = build_decision(
            event=event,
            decision="congestion_warning",
            actions=[self._advisory()],
            confidence=0.371,
            reason="test Student warning",
            source="test_student",
            policy_version="test",
        )
        controller = self._decoder_controller()
        base_decide = controller.active.model.decide

        class ChangedActionModel:
            validation = {"metrics": {"decision_accuracy": 0.6613}}

            @staticmethod
            def decide(prompt, decoder_event, network_available):
                result = base_decide(prompt, decoder_event, network_available)
                decoded = dict(result["decision"])
                actions = [dict(value) for value in decoded["actions"]]
                actions[0] = {
                    **actions[0],
                    "parameters": {
                        **actions[0].get("parameters", {}),
                        "warning_level": "high",
                    },
                }
                return {
                    **result,
                    "decision": {**decoded, "actions": actions},
                }

        controller.active.model = ChangedActionModel()
        decision = controller.decide(event, student, "test")

        self.assertEqual(decision.decision, "congestion_warning")
        self.assertTrue(decision.metadata["edge_llm_model_disagreement"])

        reason_controller = self._decoder_controller()
        reason_base_decide = reason_controller.active.model.decide

        class ChangedReasonModel:
            validation = {"metrics": {"decision_accuracy": 0.6613}}

            @staticmethod
            def decide(prompt, decoder_event, network_available):
                result = reason_base_decide(
                    prompt, decoder_event, network_available
                )
                decoded = dict(result["decision"])
                actions = [dict(value) for value in decoded["actions"]]
                actions[0] = {
                    **actions[0],
                    "reason": "different operational justification",
                }
                return {
                    **result,
                    "decision": {**decoded, "actions": actions},
                }

        reason_controller.active.model = ChangedReasonModel()
        reason_decision = reason_controller.decide(event, student, "test")
        self.assertTrue(
            reason_decision.metadata["edge_llm_model_disagreement"]
        )

    def test_future_warning_without_student_advisory_still_falls_back(self) -> None:
        event = self._current_state_event()
        student = build_decision(
            event=event,
            decision="no_action",
            actions=[],
            confidence=0.7,
            reason="test Student abstains",
            source="test_student",
            policy_version="test",
        )

        decision = self._decoder_controller().decide(event, student, "test")

        self.assertEqual(decision.decision, "no_action")
        self.assertEqual(
            decision.metadata["edge_llm_fallback_reason"],
            "authorized_candidate_action_missing",
        )
        self.assertEqual(
            decision.metadata["edge_decision_path"], "student_safety_fallback"
        )

    def test_decode_whitelist_never_injects_student_control_actions(self) -> None:
        event = self._current_state_event()
        controls = [
            Action(
                action_type=name,
                target_ids=["traffic_node:1"],
                resource_ids=["traffic_node:1"],
                parameters={},
                reason="must remain outside the decoder whitelist",
                priority=90,
            )
            for name in ("variable_speed_limit", "ramp_metering", "reroute")
        ]
        student = build_decision(
            event=event,
            decision="reroute",
            actions=[self._advisory(), *controls],
            confidence=0.5,
            reason="test mixed Student actions",
            source="test_student",
            policy_version="test",
        )
        controller = self._decoder_controller()

        decoder_event, injected = controller._decoder_event(event, student)

        self.assertTrue(injected)
        self.assertEqual(
            {value["action_type"] for value in decoder_event["candidate_actions"]},
            {"traffic_advisory"},
        )
        astgcn = replace(
            event,
            scene_payload={
                **event.scene_payload,
                "perception_mode": "astgcn",
                "output_type": "forecast_and_risk",
            },
        )
        legacy_event, legacy_injected = controller._decoder_event(astgcn, student)
        self.assertFalse(legacy_injected)
        self.assertEqual(legacy_event["candidate_actions"], [])

    def test_current_state_action_map_changes_only_proactive_advisory(self) -> None:
        current = read_json_object(
            TRAFFIC_ROOT
            / "assets"
            / "edge_llm"
            / "adapter_package_current_state_v2"
            / "action_mapping.json"
        )
        legacy = read_json_object(
            TRAFFIC_ROOT
            / "assets"
            / "edge_llm"
            / "adapter_package"
            / "action_mapping.json"
        )
        current_slots = {row["slot"]: row for row in current["entries"]}
        legacy_slots = {row["slot"]: row for row in legacy["entries"]}

        self.assertEqual(current_slots["B"]["min_risk_level"], "low")
        self.assertEqual(legacy_slots["B"]["min_risk_level"], "medium")
        for slot in ("C", "D", "E", "F"):
            self.assertEqual(current_slots[slot], legacy_slots[slot])

    def test_current_state_adapter_package_remains_release_valid(self) -> None:
        result = validate_adapter_package(
            TRAFFIC_ROOT / "assets" / "edge_llm" / "adapter_package_current_state_v2",
            TRAFFIC_ROOT / "assets" / "edge_llm" / "base_manifest.json",
            require_gates=True,
        )
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["version"], "2.0.2")


if __name__ == "__main__":
    unittest.main()
