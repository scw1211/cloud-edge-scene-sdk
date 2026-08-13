from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from cloud_edge_framework.runtime import EdgeRuntime
from scenes.industrial_anomaly import benchmark_industrial_closed_loop as benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Decision:
    def to_dict(self):
        return {
            "decision_id": "decision-1",
            "event_id": "event-1",
            "scene": "industrial_anomaly",
            "decision": "review",
            "status": "provisional",
            "route": "cloud_async",
            "actions": [],
            "metadata": {
                "edge_decision_path": "industrial_edge_qwen",
                "edge_qwen_selection_reason": "review_requires_corroboration",
                "edge_qwen_selected": True,
                "edge_qwen_mode": "selective",
                "edge_qwen_prediction": "review",
                "edge_qwen_token": "B",
                "edge_qwen_latency_ms": 91.25,
                "edge_qwen_prompt_tokens": 16,
                "edge_qwen_output_tokens": 1,
                "edge_qwen_rule_agreement": True,
                "edge_qwen_fallback": False,
                "edge_qwen_fallback_reason": "",
                "edge_qwen_runtime_error": "must-not-leak",
                "product": "capsule",
            },
        }


class IndustrialBenchmarkTests(unittest.TestCase):
    def test_selective_qwen_statistics_exclude_unselected_events(self):
        events = [
            {
                "local_decision": "normal",
                "reference": "normal",
                "qwen_selected": False,
                "qwen_rule_agreement": None,
                "qwen_latency_ms": None,
                "pipeline_stage_ms": {},
                "client_wall_ms": 10.0,
                "edge_service_wall_ms": 9.0,
                "edge_preliminary_decision_ms": 8.0,
                "selected_request_bytes": 100,
                "client_request_bytes": 200,
                "client_response_bytes": 300,
            },
            {
                "local_decision": "review",
                "reference": "review",
                "qwen_selected": True,
                "qwen_rule_agreement": True,
                "qwen_latency_ms": 90.0,
                "pipeline_stage_ms": {},
                "client_wall_ms": 100.0,
                "edge_service_wall_ms": 99.0,
                "edge_preliminary_decision_ms": 98.0,
                "selected_request_bytes": 100,
                "client_request_bytes": 200,
                "client_response_bytes": 300,
            },
            {
                "local_decision": "anomaly",
                "reference": "anomaly",
                "qwen_selected": False,
                "qwen_rule_agreement": None,
                "qwen_latency_ms": None,
                "pipeline_stage_ms": {},
                "client_wall_ms": 11.0,
                "edge_service_wall_ms": 10.0,
                "edge_preliminary_decision_ms": 9.0,
                "selected_request_bytes": 100,
                "client_request_bytes": 200,
                "client_response_bytes": 300,
            },
        ]
        summary = benchmark._profile_summary(
            [{"client_group_wall_ms": 100.0, "events": events}], [], []
        )
        self.assertEqual(summary["qwen_selection_rate"], 0.333333)
        self.assertEqual(summary["qwen_rule_agreement_rate"], 1.0)
        self.assertEqual(summary["qwen_latency_ms"]["count"], 1)
        self.assertEqual(summary["qwen_latency_ms"]["mean"], 90.0)
        self.assertEqual(summary["qwen_amortized_latency_ms"]["mean"], 30.0)
        self.assertTrue(summary["qwen_routing_contract_satisfied"])

    def test_compact_post_event_uses_frozen_bands_and_retains_qwen_fields(self):
        bands = {
            "modalities": {
                "rgb": {
                    "capsule": {"review_low": 0.4, "review_high": 0.6}
                }
            }
        }
        event = {
            "id": "industrial-primary-0001-rgb",
            "traceid": "trace-industrial-primary-0001-rgb",
            "data": {
                "modality": "rgb",
                "product": "capsule",
                "score": 0.5,
            },
        }
        response = {
            "response_detail": "compact",
            "final_decision": {
                "decision": "review",
                "status": "provisional",
                "route": "cloud_async",
                "metadata": {
                    "edge_qwen_selection_reason": "review_requires_corroboration",
                    "edge_qwen_selected": True,
                    "edge_qwen_prediction": "review",
                    "edge_qwen_token": "B",
                    "edge_qwen_latency_ms": 91.25,
                    "edge_qwen_prompt_tokens": 16,
                    "edge_qwen_output_tokens": 1,
                    "edge_qwen_rule_agreement": True,
                },
            },
            "schedule": {"route": "cloud_async"},
            "data_plane": {
                "selected_request_bytes": 3840,
                "request_reduction_ratio": 0.96,
            },
            "framework_runtime_ms": 100.0,
            "edge_service_wall_ms": 102.0,
            "closed_loop_accounting": {
                "edge_preliminary_decision_ms": 92.0,
                "accounted_closed_loop_ms": 100.0,
                "pipeline_stage_ms": {"edge_decision": 91.5},
            },
        }
        with patch.object(
            benchmark,
            "_request_json",
            return_value=(response, 256, 512, 104.0),
        ) as request_json:
            record = benchmark._post_event(
                "http://127.0.0.1:19501",
                event,
                bands,
                async_mode=True,
                force_sync=False,
                timeout=5.0,
            )

        self.assertEqual(
            request_json.call_args.kwargs["headers"]["X-Response-Detail"],
            "compact",
        )
        self.assertEqual(
            request_json.call_args.kwargs["headers"]["Prefer"],
            "respond-async",
        )
        self.assertEqual(record["reference"], "review")
        self.assertEqual(record["local_decision"], "review")
        self.assertEqual(record["response_detail"], "compact")
        self.assertTrue(record["qwen_selected"])
        self.assertEqual(
            record["qwen_selection_reason"], "review_requires_corroboration"
        )
        self.assertEqual(record["qwen_prediction"], "review")
        self.assertEqual(record["qwen_token"], "B")
        self.assertEqual(record["qwen_latency_ms"], 91.25)
        self.assertEqual(record["qwen_prompt_tokens"], 16)
        self.assertEqual(record["qwen_output_tokens"], 1)
        self.assertTrue(record["qwen_rule_agreement"])

    def test_compact_decision_retains_only_bounded_industrial_qwen_metadata(self):
        compact = EdgeRuntime._compact_decision(_Decision())

        self.assertEqual(compact["decision"], "review")
        self.assertEqual(compact["status"], "provisional")
        self.assertEqual(compact["route"], "cloud_async")
        metadata = compact["metadata"]
        self.assertEqual(
            metadata["edge_qwen_selection_reason"],
            "review_requires_corroboration",
        )
        self.assertTrue(metadata["edge_qwen_selected"])
        self.assertEqual(metadata["edge_qwen_mode"], "selective")
        self.assertEqual(metadata["edge_qwen_prediction"], "review")
        self.assertEqual(metadata["edge_qwen_token"], "B")
        self.assertEqual(metadata["edge_qwen_latency_ms"], 91.25)
        self.assertEqual(metadata["edge_qwen_prompt_tokens"], 16)
        self.assertEqual(metadata["edge_qwen_output_tokens"], 1)
        self.assertTrue(metadata["edge_qwen_rule_agreement"])
        self.assertFalse(metadata["edge_qwen_fallback"])
        self.assertNotIn("edge_qwen_runtime_error", metadata)
        self.assertNotIn("product", metadata)

    def test_primary_only_cli_runs_one_frozen_180_event_profile(self):
        groups = [
            {
                "client_group_wall_ms": 10.0,
                "events": [
                    {"event_id": "primary-{:03d}-rgb".format(index)},
                    {"event_id": "primary-{:03d}-infrared".format(index)},
                ],
            }
            for index in range(90)
        ]
        primary = {
            "event_count": 180,
            "local_accuracy": 1.0,
            "client_input_to_provisional_ms": {
                "count": 180,
                "mean": 72.0,
                "p50": 70.0,
                "p95": 150.0,
                "max": 180.0,
            },
            "qwen_selection_rate": 0.333333,
            "qwen_selected_count": 60,
            "qwen_completion_rate": 1.0,
            "qwen_rule_agreement_rate": 1.0,
            "qwen_routing_contract_satisfied": True,
            "qwen_latency_ms": {
                "count": 60,
                "mean": 90.0,
                "p50": 90.0,
                "p95": 100.0,
                "max": 110.0,
            },
            "qwen_amortized_latency_ms": {
                "count": 180,
                "mean": 30.0,
                "p50": 0.0,
                "p95": 100.0,
                "max": 110.0,
            },
            "records": {"groups": groups, "reviews": []},
        }
        thresholds = (
            PROJECT_ROOT
            / "scenes/industrial_anomaly/industrial_anomaly/review_bands.json"
        )
        rgb_template = (
            PROJECT_ROOT / "scenes/industrial_anomaly/samples/rgb_event.json"
        )
        infrared_template = (
            PROJECT_ROOT / "scenes/industrial_anomaly/samples/infrared_event.json"
        )
        with tempfile.TemporaryDirectory(prefix="industrial-primary-only-") as directory:
            output = Path(directory) / "primary.json"
            argv = [
                "benchmark_industrial_closed_loop.py",
                "--primary-only",
                "--thresholds",
                str(thresholds),
                "--rgb-template",
                str(rgb_template),
                "--infrared-template",
                str(infrared_template),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv), \
                    patch.object(
                        benchmark, "_set_profile", return_value={}
                    ) as set_profile, \
                    patch.object(
                        benchmark, "_proxy_status", return_value={}
                    ) as proxy_status, \
                    patch.object(
                        benchmark, "_run_profile", return_value=primary
                    ) as run_profile, \
                    redirect_stdout(io.StringIO()):
                result = benchmark.main()

            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        set_profile.assert_called_once_with("http://127.0.0.1:19510", "normal")
        proxy_status.assert_called_once_with("http://127.0.0.1:19510")
        run_profile.assert_called_once()
        self.assertEqual(run_profile.call_args.args[0], "normal-async")
        self.assertEqual(run_profile.call_args.args[1], 90)
        self.assertEqual(
            run_profile.call_args.args[2],
            (("normal", "normal"), ("review", "review"), ("anomaly", "anomaly")),
        )
        self.assertEqual(set(report["profiles"]), {"normal_async"})
        self.assertTrue(report["scope_notes"]["primary_only"])
        self.assertEqual(report["scope_notes"]["request_event_count"], 180)
        self.assertNotIn("service_evidence", report)


if __name__ == "__main__":
    unittest.main()
