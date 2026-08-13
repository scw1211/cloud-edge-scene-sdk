from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDUSTRIAL_ROOT = PROJECT_ROOT / "scenes" / "industrial_anomaly"
TRAFFIC_ROOT = PROJECT_ROOT / "scenes" / "freeway_traffic"
for value in (str(INDUSTRIAL_ROOT), str(TRAFFIC_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from cloud_edge_framework.aggregation import AggregationSpec, MultiEdgeEventAggregator
from cloud_edge_framework.cloud_service import CloudApiService
from cloud_edge_framework.event_envelope import SceneEventEnvelope
from cloud_edge_framework.handoff import DurableOutboxHandoff
from cloud_edge_framework.registry import SceneRegistry
from cloud_edge_framework.reliability import SQLiteOutbox
from cloud_edge_framework.review_tracking import ReviewLifecycleStore
from cloud_edge_framework.runtime import CloudRuntime, EdgeRuntime
from cloud_edge_framework.scheduling import NetworkSnapshot
from industrial_anomaly.plugin import IndustrialAnomalyPlugin
from freeway_traffic_scene.plugin import FreewayTrafficPlugin


SAMPLE_ROOT = INDUSTRIAL_ROOT / "samples"


def _sample(name):
    return json.loads((SAMPLE_ROOT / name).read_text(encoding="utf-8"))


def _with_score(payload, event_id, score, modality="rgb"):
    value = copy.deepcopy(payload)
    value["id"] = event_id
    value["edgeid"] = "industrial-{}-edge".format(modality)
    value["data"]["modality"] = modality
    value["data"]["score"] = score
    return value


class _NoCloudCalls:
    def aggregate(self, event):
        del event
        raise AssertionError("/decide must persist before the background cloud path")

    def coordinate(self, events):
        del events
        raise AssertionError("/decide must not call cloud coordination inline")


class _ManagerSnapshot:
    def __init__(self, registry):
        self.registry = registry


class _Manager:
    def __init__(self, registry):
        self.snapshot = _ManagerSnapshot(registry)

    @contextmanager
    def lease(self):
        yield self.snapshot


class _IndustrialActionClient:
    def __init__(self, state="review", error=None):
        self.state = state
        self.error = error
        self.prompts = []

    def predict(self, prompt, valid_tokens):
        self.prompts.append((prompt, dict(valid_tokens)))
        if self.error is not None:
            raise self.error
        return {
            "slot": self.state,
            "token": valid_tokens[self.state],
            "latency_ms": 12.5,
            "prompt_tokens": len(prompt),
            "output_tokens": 1,
            "decoding_constraint": {"allowed_tokens": ["A", "B", "C"]},
        }


class IndustrialAnomalyPluginTests(unittest.TestCase):
    def setUp(self):
        self.plugin = IndustrialAnomalyPlugin()

    def tearDown(self):
        self.plugin.close()

    def _normalize(self, payload):
        return self.plugin.normalize_envelope(SceneEventEnvelope.from_dict(payload))

    def test_existing_cpp_payload_is_accepted_without_explicit_product(self):
        payload = _sample("rgb_event.json")
        payload["data"].pop("product")
        payload["data"].update(
            {
                "asset_id": "192.168.31.100",
                "region_id": "1",
                "threshold": "0.5",
                "proposed_limit_percent": "50",
                "shared_resource": [],
            }
        )

        event = self._normalize(payload)

        self.assertEqual(event.metadata["product"], "capsule")
        self.assertEqual(event.metadata["modality"], "rgb")
        self.assertEqual(event.metadata["aggregation"]["member"], "rgb")
        self.assertEqual(
            event.metadata["aggregation"]["expected_members"],
            ["rgb", "infrared"],
        )

    def test_strict_schema_rejects_missing_score(self):
        payload = _sample("rgb_event.json")
        payload["data"].pop("score")
        with self.assertRaisesRegex(ValueError, "score"):
            self._normalize(payload)

    def test_review_bands_return_normal_review_and_anomaly(self):
        base = _sample("rgb_event.json")
        cases = (
            ("normal", 0.0060),
            ("review", 0.00745),
            ("anomaly", 0.0082),
        )
        for expected, score in cases:
            with self.subTest(expected=expected):
                event = self._normalize(
                    _with_score(base, "band-{}".format(expected), score)
                )
                decision = self.plugin.edge_decide(event)
                self.assertEqual(decision.decision, expected)
                self.assertFalse(decision.metadata["edge_qwen_selected"])
                if expected == "normal":
                    self.assertEqual(decision.actions, [])
                else:
                    self.assertEqual(len(decision.actions), 1)

    def test_industrial_edge_llm_corroborates_rule_with_decimal16_prompt(self):
        plugin = IndustrialAnomalyPlugin(
            edge_llm_runtime_config_path=Path("unused-runtime.json"),
            edge_llm_mode="corroborate",
        )
        client = _IndustrialActionClient("review")
        plugin._edge_llm_client = client
        try:
            event = plugin.normalize_envelope(
                SceneEventEnvelope.from_dict(_sample("rgb_event.json"))
            )
            decision = plugin.edge_decide(event)
            self.assertEqual(decision.decision, "review")
            self.assertEqual(decision.metadata["edge_decision_path"], "industrial_edge_qwen")
            self.assertTrue(decision.metadata["edge_qwen_rule_agreement"])
            self.assertEqual(len(client.prompts[0][0]), 16)
            self.assertTrue(client.prompts[0][0].isdigit())
            self.assertTrue(client.prompts[0][0].startswith("2"))
            self.assertEqual(
                client.prompts[0][1],
                {"normal": "A", "review": "B", "anomaly": "C"},
            )
        finally:
            plugin.close()

    def test_industrial_joint_runtime_uses_i_prefix_without_request_lora(self):
        config = {
            "authentication": {"api_key_env": ""},
            "endpoint": "http://127.0.0.1:18590",
            "generation": {
                "keep_alive": "30m",
                "max_input_tokens": 17,
                "max_output_tokens": 1,
                "seed": 42,
                "temperature": 0.0,
                "thinking": False,
                "top_p": 1.0,
            },
            "model": "traffic-industrial-joint-test",
            "provider": "llama_cpp",
            "schema_version": "edge-llm-runtime/v1",
            "timeout_seconds": 0.18,
        }
        with tempfile.TemporaryDirectory(prefix="industrial-joint-runtime-") as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            plugin = IndustrialAnomalyPlugin(
                edge_llm_runtime_config_path=path,
                edge_llm_mode="selective",
                edge_llm_prompt_prefix="I",
            )
            try:
                plugin.warmup()
                self.assertIsNone(
                    plugin._edge_llm_client.describe()["lora_adapter"]
                )
                client = _IndustrialActionClient("review")
                plugin._edge_llm_client = client
                event = plugin.normalize_envelope(
                    SceneEventEnvelope.from_dict(_sample("rgb_event.json"))
                )
                decision = plugin.edge_decide(event)
                prompt = client.prompts[0][0]
                self.assertEqual(len(prompt), 17)
                self.assertEqual(prompt[0], "I")
                self.assertTrue(prompt[1:].isdigit())
                self.assertEqual(decision.metadata["edge_qwen_prompt_tokens"], 17)
            finally:
                plugin.close()

    def test_industrial_joint_runtime_rejects_request_level_lora(self):
        config = {
            "authentication": {"api_key_env": ""},
            "endpoint": "http://127.0.0.1:18590",
            "generation": {
                "keep_alive": "30m",
                "max_input_tokens": 17,
                "max_output_tokens": 1,
                "seed": 42,
                "temperature": 0.0,
                "thinking": False,
                "top_p": 1.0,
            },
            "lora_adapter": {"id": 0, "scale": 1.0},
            "model": "traffic-industrial-joint-test",
            "provider": "llama_cpp",
            "schema_version": "edge-llm-runtime/v1",
            "timeout_seconds": 0.18,
        }
        with tempfile.TemporaryDirectory(prefix="industrial-joint-runtime-") as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            plugin = IndustrialAnomalyPlugin(
                edge_llm_runtime_config_path=path,
                edge_llm_mode="selective",
                edge_llm_prompt_prefix="I",
            )
            try:
                with self.assertRaisesRegex(ValueError, "omit request-level LoRA"):
                    plugin.warmup()
            finally:
                plugin.close()

    def test_industrial_selective_edge_llm_only_runs_for_review(self):
        plugin = IndustrialAnomalyPlugin(
            edge_llm_runtime_config_path=Path("unused-runtime.json"),
            edge_llm_mode="selective",
        )
        client = _IndustrialActionClient("review")
        plugin._edge_llm_client = client
        try:
            base = _sample("rgb_event.json")
            decisions = []
            for state, score in (
                ("normal", 0.0060),
                ("review", 0.00745),
                ("anomaly", 0.0082),
            ):
                event = plugin.normalize_envelope(
                    SceneEventEnvelope.from_dict(
                        _with_score(base, "selective-{}".format(state), score)
                    )
                )
                decisions.append(plugin.edge_decide(event))

            self.assertEqual([item.decision for item in decisions], [
                "normal",
                "review",
                "anomaly",
            ])
            self.assertEqual(len(client.prompts), 1)
            self.assertEqual(
                decisions[0].metadata["edge_decision_path"],
                "industrial_rule_fast_path",
            )
            self.assertFalse(decisions[0].metadata["edge_qwen_selected"])
            self.assertTrue(decisions[1].metadata["edge_qwen_selected"])
            self.assertEqual(
                decisions[2].metadata["edge_qwen_selection_reason"],
                "deterministic_normal_or_anomaly",
            )
            self.assertEqual(plugin.health()["edge_qwen_invocations"], 1)
        finally:
            plugin.close()

    def test_industrial_selective_fast_path_does_not_require_loaded_runtime(self):
        plugin = IndustrialAnomalyPlugin(
            edge_llm_runtime_config_path=Path("unused-runtime.json"),
            edge_llm_mode="selective",
        )
        try:
            event = plugin.normalize_envelope(
                SceneEventEnvelope.from_dict(
                    _with_score(_sample("rgb_event.json"), "selective-fast", 0.0060)
                )
            )
            decision = plugin.edge_decide(event)
            self.assertEqual(decision.decision, "normal")
            self.assertEqual(
                decision.metadata["edge_decision_path"],
                "industrial_rule_fast_path",
            )
            self.assertNotIn("edge_qwen_fallback", decision.metadata)
        finally:
            plugin.close()

    def test_industrial_selective_routing_matches_half_open_review_band(self):
        plugin = IndustrialAnomalyPlugin(
            edge_llm_runtime_config_path=Path("unused-runtime.json"),
            edge_llm_mode="selective",
        )
        client = _IndustrialActionClient("review")
        plugin._edge_llm_client = client
        try:
            base = _sample("rgb_event.json")
            low, high = plugin._band("rgb", "capsule")
            observed = []
            for index, score in enumerate((low - 1e-9, low, high - 1e-9, high)):
                event = plugin.normalize_envelope(
                    SceneEventEnvelope.from_dict(
                        _with_score(base, "selective-boundary-{}".format(index), score)
                    )
                )
                observed.append(plugin.edge_decide(event))
            self.assertEqual(
                [item.decision for item in observed],
                ["normal", "review", "review", "anomaly"],
            )
            self.assertEqual(len(client.prompts), 2)
        finally:
            plugin.close()

    def test_industrial_selective_warmup_rejects_timeout_over_budget(self):
        config = {
            "authentication": {"api_key_env": ""},
            "endpoint": "http://127.0.0.1:18590",
            "generation": {
                "keep_alive": "30m",
                "max_input_tokens": 16,
                "max_output_tokens": 1,
                "seed": 42,
                "temperature": 0.0,
                "thinking": False,
                "top_p": 1.0,
            },
            "lora_adapter": {"id": 1, "scale": 1.0},
            "model": "industrial-test",
            "provider": "llama_cpp",
            "schema_version": "edge-llm-runtime/v1",
            "timeout_seconds": 0.181,
        }
        with tempfile.TemporaryDirectory(prefix="industrial-runtime-") as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            plugin = IndustrialAnomalyPlugin(
                edge_llm_runtime_config_path=path,
                edge_llm_mode="selective",
            )
            try:
                with self.assertRaisesRegex(ValueError, "must be <= 0.18"):
                    plugin.warmup()
                config["timeout_seconds"] = 0.18
                path.write_text(json.dumps(config), encoding="utf-8")
                plugin.warmup()
                self.assertIsNotNone(plugin._edge_llm_client)
            finally:
                plugin.close()

    def test_joint_static_selective_timeout_limit_is_explicit_and_bounded(self):
        config = {
            "authentication": {"api_key_env": ""},
            "endpoint": "http://127.0.0.1:18590",
            "generation": {
                "keep_alive": "30m",
                "max_input_tokens": 17,
                "max_output_tokens": 1,
                "seed": 42,
                "temperature": 0.0,
                "thinking": False,
                "top_p": 1.0,
            },
            "model": "joint-static-industrial-test",
            "provider": "llama_cpp",
            "schema_version": "edge-llm-runtime/v1",
            "timeout_seconds": 0.25,
        }
        with tempfile.TemporaryDirectory(prefix="industrial-joint-runtime-") as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            plugin = IndustrialAnomalyPlugin(
                edge_llm_runtime_config_path=path,
                edge_llm_mode="selective",
                edge_llm_prompt_prefix="I",
                edge_llm_selective_timeout_limit_seconds=0.25,
            )
            try:
                plugin.warmup()
                self.assertIsNotNone(plugin._edge_llm_client)
                self.assertEqual(
                    plugin.health()["edge_qwen_selective_timeout_limit_seconds"],
                    0.25,
                )
            finally:
                plugin.close()
        with self.assertRaisesRegex(ValueError, "must be in"):
            IndustrialAnomalyPlugin(
                edge_llm_runtime_config_path=path,
                edge_llm_mode="selective",
                edge_llm_prompt_prefix="I",
                edge_llm_selective_timeout_limit_seconds=0.251,
            )

    def test_release_disabled_runtime_keeps_deterministic_industrial_path(self):
        disabled = {
            "schema_version": "edge-llm-runtime-disabled/v1",
            "enabled": False,
            "reason": "release_missing_runtime_adapter",
            "required_adapter_id": 1,
            "release_binding": {
                "release_id": "traffic-only-legacy",
                "revision": 7,
                "binding_fingerprint": "a" * 64,
                "deployment_sha256": "b" * 64,
                "runtime_output": "industrial",
                "adapter_id": None,
                "adapter_sha256": None,
            },
        }
        with tempfile.TemporaryDirectory(prefix="industrial-disabled-") as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(json.dumps(disabled), encoding="utf-8")
            plugin = IndustrialAnomalyPlugin(
                edge_llm_runtime_config_path=path,
                edge_llm_mode="selective",
            )
            try:
                plugin.warmup()
                health = plugin.health()
                self.assertTrue(health["edge_qwen_release_disabled"])
                self.assertFalse(health["edge_qwen_available"])
                self.assertEqual(health["decision_engine"], "deterministic_review_bands")
                event = plugin.normalize_envelope(
                    SceneEventEnvelope.from_dict(_sample("rgb_event.json"))
                )
                decision = plugin.edge_decide(event)
                self.assertEqual(decision.decision, "review")
                self.assertEqual(
                    decision.metadata["edge_decision_path"],
                    "industrial_rule_release_fallback",
                )
                self.assertEqual(plugin._edge_llm_invocations, 0)
            finally:
                plugin.close()

    def test_industrial_edge_llm_disagreement_falls_back_to_rule(self):
        plugin = IndustrialAnomalyPlugin(
            edge_llm_runtime_config_path=Path("unused-runtime.json"),
            edge_llm_mode="corroborate",
        )
        plugin._edge_llm_client = _IndustrialActionClient("normal")
        try:
            event = plugin.normalize_envelope(
                SceneEventEnvelope.from_dict(_sample("rgb_event.json"))
            )
            decision = plugin.edge_decide(event)
            self.assertEqual(decision.decision, "review")
            self.assertEqual(decision.metadata["edge_decision_path"], "rule_safety_fallback")
            self.assertEqual(
                decision.metadata["edge_qwen_fallback_reason"],
                "model_rule_disagreement",
            )
        finally:
            plugin.close()

    def test_industrial_edge_llm_runtime_error_falls_back_to_rule(self):
        plugin = IndustrialAnomalyPlugin(
            edge_llm_runtime_config_path=Path("unused-runtime.json"),
            edge_llm_mode="corroborate",
        )
        plugin._edge_llm_client = _IndustrialActionClient(
            error=TimeoutError("test timeout")
        )
        try:
            event = plugin.normalize_envelope(
                SceneEventEnvelope.from_dict(_sample("rgb_event.json"))
            )
            decision = plugin.edge_decide(event)
            self.assertEqual(decision.decision, "review")
            self.assertEqual(decision.metadata["edge_decision_path"], "rule_runtime_fallback")
            self.assertIn("TimeoutError", decision.metadata["edge_qwen_runtime_error"])
        finally:
            plugin.close()

    def test_cross_modal_policy_matrix(self):
        rgb = _sample("rgb_event.json")
        infrared = _sample("infrared_event.json")
        cases = (
            (0.0060, 0.0070, "normal"),
            (0.0082, 0.0082, "anomaly"),
            (0.0060, 0.0082, "review"),
            (0.00745, 0.0082, "review"),
        )
        for index, (rgb_score, infrared_score, expected) in enumerate(cases):
            with self.subTest(expected=expected):
                events = [
                    self._normalize(
                        _with_score(rgb, "matrix-{}-rgb".format(index), rgb_score)
                    ),
                    self._normalize(
                        _with_score(
                            infrared,
                            "matrix-{}-infrared".format(index),
                            infrared_score,
                            modality="infra",
                        )
                    ),
                ]
                fused = self.plugin.fuse_cloud_context(events)
                decisions = self.plugin.cloud_decide_batch(fused)
                self.assertEqual([item.decision for item in decisions], [expected] * 2)
                self.assertTrue(
                    all(item.metadata["cross_modal_complete"] for item in decisions)
                )

    def test_incomplete_member_is_review_not_global_final(self):
        payload = _sample("rgb_event.json")
        plugin = IndustrialAnomalyPlugin(aggregation_timeout_ms=1)
        registry = SceneRegistry([plugin])
        aggregator = MultiEdgeEventAggregator()
        try:
            event = plugin.normalize_envelope(SceneEventEnvelope.from_dict(payload))
            aggregator.submit(event, AggregationSpec.from_dict(plugin.aggregation_spec(event)))
            time.sleep(0.005)
            leases = aggregator.claim_due(1)
            self.assertEqual(len(leases), 1)
            trusted = CloudApiService._events_with_trusted_aggregation_context(
                leases[0], registry
            )
            coordination = CloudRuntime(registry).coordinate(trusted)
            marked = CloudApiService._mark_aggregation_finality(
                coordination, leases[0]
            )
            self.assertEqual(marked["aggregation_finality"], "partial_final")
            self.assertFalse(marked["global_confirmation"])
            self.assertEqual(marked["decisions"][0]["decision"], "review")
            self.assertEqual(marked["decisions"][0]["status"], "provisional")
        finally:
            aggregator.close()
            registry.close()

    def test_complete_pair_becomes_authoritative_final(self):
        registry = SceneRegistry([self.plugin])
        aggregator = MultiEdgeEventAggregator()
        try:
            events = [
                self._normalize(_sample("rgb_event.json")),
                self._normalize(_sample("infrared_event.json")),
            ]
            for event in events:
                aggregator.submit(
                    event,
                    AggregationSpec.from_dict(self.plugin.aggregation_spec(event)),
                )
            leases = aggregator.claim_due(1)
            self.assertEqual(len(leases), 1)
            trusted = CloudApiService._events_with_trusted_aggregation_context(
                leases[0], registry
            )
            coordination = CloudRuntime(registry).coordinate(trusted)
            marked = CloudApiService._mark_aggregation_finality(
                coordination, leases[0]
            )
            self.assertEqual(marked["aggregation_finality"], "final")
            self.assertTrue(marked["global_confirmation"])
            self.assertEqual(len(marked["decisions"]), 2)
            self.assertTrue(
                all(item["status"] == "final" for item in marked["decisions"])
            )
        finally:
            aggregator.close()
            registry.close()

    def test_edge_decide_persists_summary_without_inline_cloud_wait(self):
        with tempfile.TemporaryDirectory(prefix="industrial-outbox-") as directory:
            root = Path(directory)
            registry = SceneRegistry([self.plugin])
            outbox = SQLiteOutbox(root / "outbox.sqlite3")
            tracker = ReviewLifecycleStore(root / "reviews.sqlite3")
            handoff = DurableOutboxHandoff(outbox, root / "handoff.jsonl")
            try:
                runtime = EdgeRuntime(
                    registry=registry,
                    cloud=_NoCloudCalls(),
                    review_store=outbox,
                    review_tracker=tracker,
                    durable_handoff=handoff,
                )
                result = runtime.process(
                    _sample("rgb_event.json"),
                    network=NetworkSnapshot(available=True, rtt_ms=5.0),
                    response_detail="compact",
                    return_provisional_immediately=True,
                )
                self.assertEqual(result["scene"], "industrial_anomaly")
                self.assertEqual(
                    result["final_decision"]["decision"], "review"
                )
                self.assertEqual(
                    result["final_decision"]["status"], "provisional"
                )
                self.assertTrue(result["summary_delivery"]["required"])
                deadline = time.monotonic() + 1.0
                while outbox.count() == 0 and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertEqual(outbox.count(), 1)
            finally:
                handoff.close()
                tracker.close()
                outbox.close()
                registry.close()

    def test_registry_routes_industrial_and_traffic_by_envelope(self):
        registry = SceneRegistry([self.plugin, FreewayTrafficPlugin()])
        try:
            industrial = SceneEventEnvelope.from_dict(_sample("rgb_event.json"))
            traffic = SceneEventEnvelope.from_dict(
                json.loads(
                    (
                        PROJECT_ROOT
                        / "scenes/freeway_traffic/samples/edge_a_event.json"
                    ).read_text(encoding="utf-8")
                )
            )
            self.assertIs(registry.for_envelope(industrial), self.plugin)
            self.assertEqual(registry.for_envelope(traffic).scene, "traffic")
        finally:
            registry.close()


if __name__ == "__main__":
    unittest.main()
