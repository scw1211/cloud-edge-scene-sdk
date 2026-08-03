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
    current_window_risk,
)
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


class CurrentStateRiskTests(unittest.TestCase):
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
