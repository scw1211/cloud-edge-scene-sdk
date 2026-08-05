"""不经过 ASTGCN 的当前态势感知路径回归测试。"""

from pathlib import Path
import sys
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRAFFIC_ROOT = REPOSITORY_ROOT / "scenes" / "freeway_traffic"
for import_root in (REPOSITORY_ROOT, TRAFFIC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from freeway_traffic_full.plugin_impl import TrafficPlugin  # noqa: E402
from traffic_system.current_state_perception_runtime import (  # noqa: E402
    CurrentStateTrafficPerceptionRuntime,
    _mean_risk_score,
    _risk_score,
    current_window_risk,
)
from traffic_system.risk_labels import RISK_CLASSES  # noqa: E402
from traffic_system.scene_event import traffic_envelope_from_output  # noqa: E402


RULE_CONFIG = {
    "reference_speed": 68.0,
    "congestion_speed_ratio": 0.8,
    "risk_score_centers": [0.1, 0.37, 0.63, 0.9],
    "risk_score_width": 0.18,
    "risk_weights": {
        "mean_speed_pressure": 0.3,
        "minimum_speed_pressure": 0.15,
        "congestion_duration": 0.25,
        "occupancy_pressure": 0.2,
        "recent_speed_drop": 0.1,
    },
}


def _synthetic_state(node_count=9):
    raw_probabilities = np.arange(
        1,
        node_count * len(RISK_CLASSES) + 1,
        dtype=np.float64,
    ).reshape(node_count, len(RISK_CLASSES))
    probabilities = raw_probabilities / np.sum(
        raw_probabilities, axis=1, keepdims=True
    )
    # Exercise stable ties in a deliberately non-sorted managed-node order.
    probabilities[2] = [0.05, 0.15, 0.70, 0.10]
    probabilities[5] = probabilities[2]
    # These exact scores differ by less than half a micro-unit but both round
    # to the same six-decimal public score.  The legacy sort treats them as a
    # tie and therefore keeps their managed-node order.
    probabilities[1] = [0.10, 0.50, 0.20, 0.20]
    probabilities[7] = [0.10, 0.50, 0.2000003, 0.1999997]
    speed_history = np.arange(node_count * 12, dtype=np.float64).reshape(
        node_count, 12
    )
    return {
        "probabilities": probabilities,
        "flow_mean": np.linspace(100.0, 200.0, node_count),
        "occupancy_mean": np.linspace(0.05, 0.25, node_count),
        "speed_mean": np.mean(speed_history, axis=1),
        "speed_min": np.min(speed_history, axis=1),
        "speed_history": speed_history,
    }


def _legacy_top_nodes(managed_nodes, state, top_k):
    rows = []
    for node_id in managed_nodes:
        node_id = int(node_id)
        node_probs = state["probabilities"][node_id]
        label_id = int(np.argmax(node_probs))
        speed_history = state["speed_history"][node_id]
        rows.append(
            {
                "node_id": node_id,
                "risk_level": RISK_CLASSES[label_id],
                "risk_score": round(_risk_score(node_probs), 6),
                "risk_confidence": round(float(np.max(node_probs)), 6),
                "risk_probabilities": {
                    name: round(float(node_probs[index]), 6)
                    for index, name in enumerate(RISK_CLASSES)
                },
                "history_mean": round(float(np.mean(speed_history)), 6),
                "history_last": round(float(speed_history[-1]), 6),
                "volatility": round(float(np.std(speed_history)), 6),
                "history_12_steps": [
                    round(float(value), 6) for value in speed_history
                ],
                "current_observation": {
                    "flow_mean": round(float(state["flow_mean"][node_id]), 6),
                    "occupancy_mean": round(
                        float(state["occupancy_mean"][node_id]), 6
                    ),
                    "speed_mean": round(float(state["speed_mean"][node_id]), 6),
                    "speed_min": round(float(state["speed_min"][node_id]), 6),
                },
            }
        )
    rows.sort(
        key=lambda item: (
            RISK_CLASSES.index(item["risk_level"]),
            item["risk_score"],
        ),
        reverse=True,
    )
    return rows[:top_k]


class CurrentStateRiskTests(unittest.TestCase):
    def test_top_nodes_matches_legacy_reference_for_ties_and_boundaries(self):
        runtime = object.__new__(CurrentStateTrafficPerceptionRuntime)
        managed_nodes = [5, 2, 8, 1, 7, 0, 4, 3, 6]
        state = _synthetic_state()

        for top_k in (1, 3, len(managed_nodes), len(managed_nodes) + 5):
            with self.subTest(top_k=top_k):
                runtime.top_k = top_k
                self.assertEqual(
                    runtime._top_nodes(managed_nodes, state),
                    _legacy_top_nodes(managed_nodes, state, top_k),
                )

        runtime.top_k = len(managed_nodes)
        node_ids = [
            row["node_id"] for row in runtime._top_nodes(managed_nodes, state)
        ]
        self.assertLess(node_ids.index(5), node_ids.index(2))
        self.assertLess(node_ids.index(1), node_ids.index(7))

    def test_vectorized_mean_risk_score_matches_scalar_reference(self):
        rng = np.random.default_rng(20260805)
        candidates = [_synthetic_state(node_count=170)["probabilities"]]
        for node_count in (1, 4, 43, 170):
            raw = rng.random((node_count, len(RISK_CLASSES)))
            candidates.append(raw / np.sum(raw, axis=1, keepdims=True))

        for probabilities in candidates:
            with self.subTest(node_count=len(probabilities)):
                scalar_reference = float(
                    np.mean([_risk_score(value) for value in probabilities])
                )
                self.assertEqual(
                    round(_mean_risk_score(probabilities), 6),
                    round(scalar_reference, 6),
                )

    def test_congested_window_scores_above_free_flow_window(self):
        free = np.zeros((2, 3, 12), dtype=np.float32)
        free[:, 0, :] = 180.0
        free[:, 1, :] = 0.04
        free[:, 2, :] = 67.0
        congested = free.copy()
        congested[:, 0, :] = 420.0
        congested[:, 1, :] = 0.30
        congested[:, 2, :] = 22.0

        free_result = current_window_risk(free, RULE_CONFIG)
        congested_result = current_window_risk(congested, RULE_CONFIG)

        self.assertTrue(np.all(free_result["scores"] < 0.25))
        self.assertTrue(np.all(congested_result["scores"] > 0.75))
        self.assertTrue(
            np.all(
                np.argmax(congested_result["probabilities"], axis=1)
                > np.argmax(free_result["probabilities"], axis=1)
            )
        )

    def test_current_window_rejects_nonfinite_input(self):
        sample = np.zeros((2, 3, 12), dtype=np.float32)
        for invalid in (np.nan, np.inf):
            with self.subTest(invalid=invalid):
                candidate = sample.copy()
                candidate[0, 0, 0] = invalid
                with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                    current_window_risk(candidate, RULE_CONFIG)

    def test_real_asset_emits_four_valid_current_state_events(self):
        runtime = CurrentStateTrafficPerceptionRuntime(
            data_path=(
                TRAFFIC_ROOT
                / "assets"
                / "downloads"
                / "PEMS08_r1_d0_w0_astcgn_multitask.npz"
            ),
            rule_config_path=(
                TRAFFIC_ROOT
                / "assets"
                / "models"
                / "current_state_perception_v1.json"
            ),
            topology_path=(
                TRAFFIC_ROOT
                / "assets"
                / "models"
                / "traffic_region_topology_metis4.json"
            ),
            split="test",
            top_k=10,
        )

        result = runtime.infer_sample(0)

        self.assertEqual(result.model_forward_ms, 0.0)
        self.assertEqual(len(result.events), 4)
        self.assertEqual(
            sorted(
                node
                for event in result.events
                for node in event["managed_node_ids"]
            ),
            list(range(170)),
        )
        plugin = TrafficPlugin()
        for native_event in result.events:
            self.assertEqual(native_event["prediction_horizon_minutes"], 0)
            self.assertEqual(native_event["prediction_steps"], 0)
            self.assertEqual(native_event["output_type"], "current_state_risk")
            envelope = traffic_envelope_from_output(native_event)
            self.assertTrue(envelope.source.endswith(":current-state"))
            semantic = plugin.normalize(envelope)
            self.assertEqual(semantic.model["output_type"], "current_state_risk")
            self.assertEqual(semantic.model["version"], "current-state-v1")

    def test_full_plugin_keeps_student_and_cloud_model_for_latency_test(self):
        model_root = TRAFFIC_ROOT / "assets" / "models"
        runtime = CurrentStateTrafficPerceptionRuntime(
            data_path=(
                TRAFFIC_ROOT
                / "assets"
                / "downloads"
                / "PEMS08_r1_d0_w0_astcgn_multitask.npz"
            ),
            rule_config_path=model_root / "current_state_perception_v1.json",
            topology_path=model_root / "traffic_region_topology_metis4.json",
            split="test",
            top_k=10,
        )
        plugin = TrafficPlugin(
            cloud_model_path=model_root / "cloud_coordinator_topology_fused.joblib",
            current_state_cloud_model_path=(
                model_root / "cloud_coordinator_current_state_future_v1.joblib"
            ),
            edge_student_path=model_root / "edge_student_freeway_joint_metis4.json",
            current_state_edge_student_path=(
                model_root / "edge_student_freeway_current_state_future_v1.json"
            ),
            defer_gate_path=model_root / "edge_defer_gate.npz",
            feature_codec_path=(
                model_root / "traffic_tree_feature_codec_topology_v1.npz"
            ),
            current_state_feature_codec_path=(
                model_root / "traffic_tree_feature_codec_current_state_v1.npz"
            ),
            topology_path=model_root / "traffic_region_topology_metis4.json",
            edge_llm_mode="disabled",
        )
        events = [
            plugin.normalize(traffic_envelope_from_output(native_event))
            for native_event in runtime.infer_sample(0).events
        ]

        local_decisions = [plugin.edge_decide(event) for event in events]
        cloud_events = plugin.fuse_cloud_context(
            [plugin.prepare_cloud_event(event, "feature") for event in events]
        )
        cloud_decisions = list(plugin.cloud_decide_batch(cloud_events))

        self.assertEqual(len(local_decisions), 4)
        self.assertEqual(len(cloud_decisions), 4)
        self.assertTrue(
            all(
                decision.metadata.get("edge_student_contract")
                == "current_state_future_v1"
                and decision.metadata.get(
                    "traffic_defer_gate_skipped_for_current_state"
                )
                is True
                and decision.metadata.get("edge_decision_path") == "student"
                for decision in local_decisions
            )
        )
        self.assertTrue(
            all(
                decision.metadata.get("source")
                == "cloud_extratrees_task_feature_coordinator"
                and decision.metadata.get("cloud_inference_batch_size") == 4
                and decision.metadata.get("cloud_model_contract")
                == "current_state_future_v1"
                for decision in cloud_decisions
            )
        )
        self.assertIsNone(plugin._edge_student)
        self.assertIsNotNone(plugin._current_state_edge_student)
        self.assertIsNone(plugin._cloud_model)
        self.assertIsNotNone(plugin._current_state_cloud_model)
        self.assertTrue(
            all(
                event.scene_payload.get("perception_mode") == "current_state"
                and event.scene_payload.get("output_type") == "current_state_risk"
                for event in cloud_events
            )
        )


if __name__ == "__main__":
    unittest.main()
