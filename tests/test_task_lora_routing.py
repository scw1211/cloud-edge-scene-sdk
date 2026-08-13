"""Per-scene llama.cpp LoRA routing must happen before inference."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from edge_llm_factory.contracts import ManifestError
from edge_llm_factory.providers import LlamaCppProvider, validate_runtime_config
from edge_llm_factory.runtime import ConfiguredActionClient
from edge_llm_factory.serve_release import (
    ActiveReleaseLlamaServer,
    main as serve_release_main,
)


def _config(adapter=None):
    config = {
        "schema_version": "edge-llm-runtime/v1",
        "provider": "llama_cpp",
        "endpoint": "http://127.0.0.1:18190",
        "model": "qwen3.5-0.8b-q8.gguf",
        "timeout_seconds": 1.0,
        "generation": {
            "max_input_tokens": 16,
            "max_output_tokens": 1,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "thinking": False,
            "keep_alive": "1m",
        },
        "authentication": {"api_key_env": ""},
    }
    if adapter is not None:
        config["lora_adapter"] = adapter
    return config


class TaskLoraRoutingTests(unittest.TestCase):
    def test_serve_release_cli_forwards_no_mmap(self):
        with (
            patch("edge_llm_factory.serve_release.ActiveReleaseLlamaServer") as server,
            patch("edge_llm_factory.serve_release.signal.signal"),
        ):
            serve_release_main(
                [
                    "--registry",
                    "/runtime/registry.json",
                    "--runtime-config",
                    "/runtime/runtime.json",
                    "--binary",
                    "/opt/llama-server",
                    "--no-mmap",
                ]
            )

        self.assertTrue(server.call_args.kwargs["no_mmap"])
        server.return_value.run.assert_called_once_with()
        server.return_value.stop.assert_called_once_with()

    def test_server_preloads_adapters_in_stable_id_order_and_disables_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "llama-server"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            binary.chmod(0o700)
            adapters = [root / "traffic.gguf", root / "math.gguf", root / "logic.gguf"]
            for adapter in adapters:
                adapter.write_bytes(b"ggla")
            supervisor = ActiveReleaseLlamaServer(
                registry_path=root / "registry.json",
                runtime_config_path=root / "runtime.json",
                binary=binary,
                host="127.0.0.1",
                port=18190,
                context_tokens=1024,
                threads=4,
                parallel=1,
                gpu_layers=99,
                poll_seconds=2,
                startup_timeout_seconds=10,
                lora_adapters=adapters,
            )

            command = supervisor.command(root / "base.gguf")
            self.assertEqual(command[command.index("--batch-size") + 1], "16")
            self.assertEqual(command[command.index("--ubatch-size") + 1], "16")
            self.assertNotIn("--no-mmap", command)
            self.assertNotIn("--lora", command)
            self.assertEqual(command.count("--lora-scaled"), 1)
            lora_value = command[command.index("--lora-scaled") + 1]
            self.assertEqual(
                lora_value,
                ",".join("{}:0".format(path.resolve()) for path in adapters),
            )
            self.assertNotIn("--lora-init-without-apply", command)

    def test_server_forwards_explicit_batch_and_ubatch_sizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "llama-server"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            binary.chmod(0o700)
            supervisor = ActiveReleaseLlamaServer(
                registry_path=root / "registry.json",
                runtime_config_path=root / "runtime.json",
                binary=binary,
                host="127.0.0.1",
                port=18190,
                context_tokens=32,
                threads=4,
                parallel=1,
                gpu_layers=99,
                poll_seconds=2,
                startup_timeout_seconds=10,
                batch_size=8,
                ubatch_size=4,
                no_mmap=True,
            )

            command = supervisor.command(root / "base.gguf")
            self.assertEqual(command[command.index("--batch-size") + 1], "8")
            self.assertEqual(command[command.index("--ubatch-size") + 1], "4")
            self.assertEqual(command.count("--no-mmap"), 1)

    def test_scene_runtime_selects_exactly_one_preloaded_adapter(self):
        provider = LlamaCppProvider(_config({"id": 2, "scale": 1.0}))
        with patch.object(
            provider,
            "_post",
            return_value={"content": "FINAL: 42", "timings": {}},
        ) as post:
            result = provider.generate("21 + 21")

        self.assertEqual(result.text, "FINAL: 42")
        self.assertEqual(post.call_args.args[1]["lora"], [{"id": 2, "scale": 1.0}])
        self.assertEqual(provider.describe()["lora_adapter"], {"id": 2, "scale": 1.0})

    def test_traffic_constraint_and_adapter_are_applied_together(self):
        provider = LlamaCppProvider(_config({"id": 0, "scale": 1.0}))
        client = ConfiguredActionClient(provider)
        with patch.object(
            provider,
            "_post",
            return_value={"content": "C", "timings": {}},
        ) as post:
            result = client.predict("0123456789012345", {"normal": "A", "warn": "C"})

        payload = post.call_args.args[1]
        self.assertEqual(payload["lora"], [{"id": 0, "scale": 1.0}])
        self.assertIn('"A"', payload["grammar"])
        self.assertIn('"C"', payload["grammar"])
        self.assertEqual(result["token"], "C")

    def test_omitting_adapter_preserves_single_model_runtime(self):
        provider = LlamaCppProvider(_config())
        with patch.object(
            provider,
            "_post",
            return_value={"content": "A", "timings": {}},
        ) as post:
            provider.generate("prompt")

        self.assertNotIn("lora", post.call_args.args[1])
        self.assertIsNone(provider.describe()["lora_adapter"])

    def test_non_llama_provider_cannot_claim_request_lora_support(self):
        config = _config({"id": 1, "scale": 1.0})
        config.update(
            {
                "provider": "ollama",
                "endpoint": "http://127.0.0.1:11434",
                "model": "qwen3.5:9b",
            }
        )
        with self.assertRaisesRegex(ManifestError, "只有 llama_cpp"):
            validate_runtime_config(config)

    def test_adapter_id_scale_and_fields_fail_closed(self):
        invalid = (
            {"id": -1, "scale": 1.0},
            {"id": 0, "scale": 0.0},
            {"id": 0, "scale": 1.0, "name": "math"},
        )
        for adapter in invalid:
            with self.subTest(adapter=adapter):
                with self.assertRaises(ManifestError):
                    validate_runtime_config(_config(adapter))


if __name__ == "__main__":
    unittest.main()
