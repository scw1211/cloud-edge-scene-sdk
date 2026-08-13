"""Regression tests for pre-sampling single-token action constraints."""

import json
from pathlib import Path
import sys
from unittest.mock import patch
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAFFIC_ROOT = REPOSITORY_ROOT / "scenes" / "freeway_traffic"
if str(TRAFFIC_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAFFIC_ROOT))

from edge_llm_factory.action_constraints import (
    action_token_constraint_evidence,
    build_action_token_gbnf,
)
from edge_llm_factory.contracts import ManifestError, read_json_object
from edge_llm_factory.evaluate_action_tokens import (
    _hf_action_token_ids,
    _hf_constraint_kwargs,
)
from edge_llm_factory.providers import LlamaCppProvider
from edge_llm_factory.runtime import (
    ActionDecoder,
    ConfiguredActionClient,
    LlamaCppActionClient,
)
from traffic_system.evaluate_llama_cpp_action_tokens import request_token


TOKENS = {"normal": "A", "warn": "B", "defer": "F"}


def _runtime_config() -> dict:
    return {
        "schema_version": "edge-llm-runtime/v1",
        "provider": "llama_cpp",
        "endpoint": "http://127.0.0.1:18190",
        "model": "test.gguf",
        "timeout_seconds": 1.0,
        "generation": {
            "max_input_tokens": 16,
            "max_output_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "thinking": False,
            "keep_alive": "1m",
        },
        "authentication": {"api_key_env": ""},
    }


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _Tokenizer:
    ids = {"A": 32, "B": 33, "F": 37}

    def __call__(self, token, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("action token lookup must not add special tokens")
        return {"input_ids": [self.ids[token]]}


class ActionTokenConstraintTests(unittest.TestCase):
    def test_traffic_reserved_slots_are_decoder_only_not_model_choices(self):
        traffic = REPOSITORY_ROOT / "scenes" / "freeway_traffic"
        decoder = ActionDecoder(
            read_json_object(traffic / "assets" / "edge_llm" / "base_manifest.json"),
            read_json_object(
                traffic
                / "assets"
                / "edge_llm"
                / "adapter_package_current_state_v2"
                / "action_mapping.json"
            ),
        )

        self.assertEqual(tuple(decoder.valid_tokens.values()), tuple("ABCDEF"))
        self.assertEqual(decoder.reserved_tokens, {"G": "G", "H": "H"})
        self.assertIn("G", decoder.entries)
        self.assertIn("H", decoder.entries)

    def test_gbnf_is_an_exact_choice_and_evidence_denies_post_hoc_mapping(self):
        grammar = build_action_token_gbnf(TOKENS)
        self.assertEqual(grammar, 'root ::= "A" | "B" | "F"\n')
        evidence = action_token_constraint_evidence(
            TOKENS,
            reserved_tokens=("G", "H"),
        )
        self.assertEqual(evidence["allowed_tokens"], ["A", "B", "F"])
        self.assertEqual(evidence["reserved_tokens_excluded_from_sampling"], ["G", "H"])
        self.assertTrue(evidence["applies_before_sampling"])
        self.assertFalse(evidence["post_hoc_remapping"])
        self.assertEqual(len(evidence["grammar_sha256"]), 64)

    def test_configured_production_client_sends_gbnf_before_sampling(self):
        provider = LlamaCppProvider(_runtime_config())
        client = ConfiguredActionClient(provider)
        response = {
            "content": "B",
            "timings": {"prompt_n": 16, "predicted_n": 1},
        }
        with patch.object(provider, "_post", return_value=response) as post:
            result = client.predict("0123456789012345", TOKENS)

        payload = post.call_args.args[1]
        self.assertEqual(payload["grammar"], build_action_token_gbnf(TOKENS))
        self.assertEqual(payload["n_predict"], 1)
        self.assertEqual(result["token"], "B")
        self.assertTrue(result["decoding_constraint"]["enabled"])
        self.assertFalse(result["decoding_constraint"]["post_hoc_remapping"])

    def test_configured_client_rejects_malformed_text_without_extracting_token(self):
        provider = LlamaCppProvider(_runtime_config())
        client = ConfiguredActionClient(provider)
        with patch.object(provider, "_post", return_value={"content": "answer=B"}):
            with self.assertRaisesRegex(ManifestError, "不符合单 token 协议"):
                client.predict("0123456789012345", TOKENS)

    def test_legacy_endpoint_client_also_sends_the_same_constraint(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _Response(
                {"content": "F", "timings": {"prompt_n": 16, "predicted_n": 1}}
            )

        with patch("edge_llm_factory.runtime.urllib.request.urlopen", fake_urlopen):
            result = LlamaCppActionClient("http://127.0.0.1:18190", 2.0).predict(
                "0123456789012345", TOKENS
            )

        self.assertEqual(captured["payload"]["grammar"], build_action_token_gbnf(TOKENS))
        self.assertEqual(captured["timeout"], 2.0)
        self.assertEqual(result["token"], "F")

    def test_hf_whitelist_uses_declared_single_token_ids(self):
        ids = _hf_action_token_ids(
            _Tokenizer(),
            ("A", "B", "F"),
            {"A": 32, "B": 33, "F": 37},
        )
        callback = _hf_constraint_kwargs(ids)["prefix_allowed_tokens_fn"]
        self.assertEqual(ids, {"A": 32, "B": 33, "F": 37})
        self.assertEqual(callback(0, object()), [32, 33, 37])

    def test_llama_cpp_evaluator_flag_sends_grammar_and_does_not_case_remap(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return _Response(
                {"content": "b", "timings": {"prompt_n": 16, "predicted_n": 1}}
            )

        with patch(
            "traffic_system.evaluate_llama_cpp_action_tokens.urllib.request.urlopen",
            fake_urlopen,
        ):
            constrained = request_token(
                "http://127.0.0.1:18190",
                "0123456789012345",
                2.0,
                constrain_action_tokens=True,
            )

        self.assertIn("grammar", captured["payload"])
        self.assertIsNone(constrained["parsed"])
        self.assertEqual(constrained["raw_output"], "b")
        self.assertFalse(
            constrained["decoding_constraint"]["post_hoc_remapping"]
        )


if __name__ == "__main__":
    unittest.main()
