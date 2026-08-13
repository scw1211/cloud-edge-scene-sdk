import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

from edge_llm_factory.serve_release import (
    ActiveReleaseLlamaServer,
    validate_startup_probe_config,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EdgeLLMStartupGateTests(unittest.TestCase):
    def _supervisor(self, root, artifact, adapters, release_id="release-multi"):
        binary = root / "llama-server"
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        binary.chmod(0o700)
        probe_config = root / "startup-probes.json"
        probe_config.write_text(
            json.dumps(
                {
                    "schema_version": "edge-llm-startup-probes/v1",
                    "require_for_runtime_adapters": True,
                    "releases": {
                        release_id: {
                            "deployment_sha256": _sha(artifact),
                            "adapters": [
                                {
                                    "id": index,
                                    "adapter_sha256": _sha(adapter),
                                    "prompt": "probe-{}".format(index),
                                    "allowed_tokens": ["A", "B", "C"],
                                    "expected_token": token,
                                    "timeout_seconds": 1.0,
                                }
                                for index, (adapter, token) in enumerate(
                                    zip(adapters, ("A", "B"))
                                )
                            ],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return ActiveReleaseLlamaServer(
            registry_path=root / "registry.json",
            runtime_config_path=root / "runtime.json",
            binary=binary,
            host="127.0.0.1",
            port=18190,
            context_tokens=128,
            threads=1,
            parallel=1,
            gpu_layers=0,
            poll_seconds=1,
            startup_timeout_seconds=2,
            startup_probe_config_path=probe_config,
        )

    def test_health_followup_verifies_layout_and_warms_each_adapter(self):
        with tempfile.TemporaryDirectory(prefix="startup-gate-") as directory:
            root = Path(directory)
            artifact = root / "base.gguf"
            artifact.write_bytes(b"base")
            adapters = [root / "traffic.gguf", root / "industrial.gguf"]
            adapters[0].write_bytes(b"traffic")
            adapters[1].write_bytes(b"industrial")
            supervisor = self._supervisor(root, artifact, adapters)
            layout = [
                {"id": index, "path": str(path.resolve()), "scale": 0.0}
                for index, path in enumerate(adapters)
            ]
            record = {
                "deployment_artifact": {
                    "path": str(artifact.resolve()),
                    "sha256": _sha(artifact),
                },
                "runtime_adapters": [
                    {"id": index, "path": str(path), "sha256": _sha(path)}
                    for index, path in enumerate(adapters)
                ],
            }
            with patch.object(
                supervisor,
                "_request_json",
                side_effect=[
                    {"model_path": str(artifact.resolve())},
                    layout,
                    {"content": "A", "timings": {"predicted_n": 1}},
                    {"content": "B", "timings": {"predicted_n": 1}},
                    layout,
                ],
            ) as request_json:
                evidence = supervisor._verify_startup_gate(
                    "release-multi", record, artifact, adapters
                )

            self.assertEqual(evidence["status"], "passed")
            self.assertEqual([row["adapter_id"] for row in evidence["warmups"]], [0, 1])
            self.assertEqual(evidence["post_warmup_runtime_adapters"], layout)
            first_payload = request_json.call_args_list[2].kwargs["payload"]
            second_payload = request_json.call_args_list[3].kwargs["payload"]
            self.assertEqual(first_payload["lora"], [{"id": 0, "scale": 1.0}])
            self.assertEqual(second_payload["lora"], [{"id": 1, "scale": 1.0}])
            self.assertEqual(first_payload["n_predict"], 1)
            self.assertFalse(first_payload["stream"])
            self.assertIn('"A"', first_payload["grammar"])
            self.assertEqual(
                request_json.call_args_list[-1],
                call("/lora-adapters", timeout_seconds=1.0),
            )

    def test_wrong_model_or_nonzero_default_scale_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="startup-gate-bad-") as directory:
            root = Path(directory)
            artifact = root / "base.gguf"
            adapter = root / "adapter.gguf"
            artifact.write_bytes(b"base")
            adapter.write_bytes(b"adapter")
            supervisor = self._supervisor(root, artifact, [adapter])
            record = {
                "deployment_artifact": {"sha256": _sha(artifact)},
                "runtime_adapters": [
                    {"id": 0, "path": str(adapter), "sha256": _sha(adapter)}
                ],
            }
            with patch.object(
                supervisor,
                "_request_json",
                return_value={"model_path": "/wrong/model.gguf"},
            ):
                with self.assertRaisesRegex(RuntimeError, "model_path mismatch"):
                    supervisor._verify_startup_gate(
                        "release-multi", record, artifact, [adapter]
                    )

            with self.assertRaisesRegex(RuntimeError, "default scale"):
                supervisor._verify_lora_layout(
                    [{"id": 0, "path": str(adapter.resolve()), "scale": 1.0}],
                    [adapter],
                )

    def test_process_default_adapter_gate_omits_request_lora(self):
        with tempfile.TemporaryDirectory(prefix="startup-gate-joint-") as directory:
            root = Path(directory)
            artifact = root / "base-q4km.gguf"
            adapter = root / "joint-f16-lora.gguf"
            artifact.write_bytes(b"base")
            adapter.write_bytes(b"joint")
            supervisor = self._supervisor(root, artifact, [adapter])
            supervisor._runtime_adapter_default_scales = [1]
            supervisor.startup_probe_config = validate_startup_probe_config(
                {
                    "schema_version": "edge-llm-startup-probes/v1",
                    "require_for_runtime_adapters": True,
                    "releases": {
                        "release-multi": {
                            "deployment_sha256": _sha(artifact),
                            "adapters": [
                                {
                                    "id": 0,
                                    "adapter_sha256": _sha(adapter),
                                    "prompt": "T0371009919313402",
                                    "allowed_tokens": list("ABCDEF"),
                                    "expected_token": "F",
                                    "expected_prompt_tokens": 17,
                                    "timeout_seconds": 1.0,
                                },
                                {
                                    "id": 0,
                                    "adapter_sha256": _sha(adapter),
                                    "prompt": "I2100399929990806",
                                    "allowed_tokens": list("ABC"),
                                    "expected_token": "A",
                                    "expected_prompt_tokens": 17,
                                    "timeout_seconds": 1.0,
                                },
                            ],
                        }
                    },
                }
            )
            supervisor.startup_probe_config["source"] = {
                "path": str(root / "startup-probes.json"),
                "sha256": "0" * 64,
            }
            layout = [
                {"id": 0, "path": str(adapter.resolve()), "scale": 1.0}
            ]
            record = {
                "deployment_artifact": {"sha256": _sha(artifact)},
                "runtime_adapters": [
                    {
                        "id": 0,
                        "path": str(adapter),
                        "sha256": _sha(adapter),
                        "default_scale": 1,
                    }
                ],
            }
            with patch.object(
                supervisor,
                "_request_json",
                side_effect=[
                    {"model_path": str(artifact.resolve())},
                    layout,
                    {
                        "content": "F",
                        "timings": {"prompt_n": 17, "predicted_n": 1},
                    },
                    {
                        "content": "A",
                        "timings": {"prompt_n": 17, "predicted_n": 1},
                    },
                    layout,
                ],
            ) as request_json:
                evidence = supervisor._verify_startup_gate(
                    "release-multi", record, artifact, [adapter]
                )

            payloads = [
                request_json.call_args_list[index].kwargs["payload"]
                for index in (2, 3)
            ]
            self.assertTrue(all("lora" not in payload for payload in payloads))
            self.assertEqual(
                [row["adapter_activation"] for row in evidence["warmups"]],
                ["process_default", "process_default"],
            )
            self.assertEqual(
                [row["expected_token"] for row in evidence["warmups"]],
                ["F", "A"],
            )
            self.assertEqual(
                [row["prompt_tokens"] for row in evidence["warmups"]],
                [17, 17],
            )
            self.assertEqual(evidence["runtime_adapters"], layout)

            for bad_timings in (
                {"prompt_n": 16, "predicted_n": 1},
                {"predicted_n": 1},
            ):
                with self.subTest(bad_timings=bad_timings), patch.object(
                    supervisor,
                    "_request_json",
                    side_effect=[
                        {"model_path": str(artifact.resolve())},
                        layout,
                        {"content": "F", "timings": bad_timings},
                    ],
                ):
                    with self.assertRaisesRegex(RuntimeError, "prompt tokens"):
                        supervisor._verify_startup_gate(
                            "release-multi", record, artifact, [adapter]
                        )

    def test_static_fusion_gate_runs_two_scene_probes_without_lora(self):
        with tempfile.TemporaryDirectory(prefix="startup-gate-static-") as directory:
            root = Path(directory)
            artifact = root / "joint-static-q5km.gguf"
            artifact.write_bytes(b"static-joint")
            supervisor = self._supervisor(root, artifact, [])
            supervisor.runtime_outputs = [
                {
                    "adapter_mode": "static",
                    "static_deployment_sha256": _sha(artifact),
                }
            ]
            supervisor.startup_probe_config = validate_startup_probe_config(
                {
                    "schema_version": "edge-llm-startup-probes/v1",
                    "require_for_runtime_adapters": True,
                    "require_for_static_fusion": True,
                    "releases": {
                        "release-multi": {
                            "deployment_sha256": _sha(artifact),
                            "adapters": [],
                            "static": [
                                {
                                    "prompt": "T0371009919313402",
                                    "allowed_tokens": list("ABCDEF"),
                                    "expected_token": "F",
                                    "expected_prompt_tokens": 17,
                                    "timeout_seconds": 1.0,
                                },
                                {
                                    "prompt": "I2100399929990806",
                                    "allowed_tokens": list("ABC"),
                                    "expected_token": "A",
                                    "expected_prompt_tokens": 17,
                                    "timeout_seconds": 1.0,
                                },
                            ],
                        }
                    },
                }
            )
            supervisor.startup_probe_config["source"] = {
                "path": str(root / "startup-probes.json"),
                "sha256": "0" * 64,
            }
            record = {
                "deployment_artifact": {"sha256": _sha(artifact)},
                "runtime_adapters": [],
            }
            with patch.object(
                supervisor,
                "_request_json",
                side_effect=[
                    {"model_path": str(artifact.resolve())},
                    [],
                    {
                        "content": "F",
                        "timings": {"prompt_n": 17, "predicted_n": 1},
                    },
                    {
                        "content": "A",
                        "timings": {"prompt_n": 17, "predicted_n": 1},
                    },
                    [],
                ],
            ) as request_json:
                evidence = supervisor._verify_startup_gate(
                    "release-multi", record, artifact, []
                )

            payloads = [
                request_json.call_args_list[index].kwargs["payload"]
                for index in (2, 3)
            ]
            self.assertTrue(all("lora" not in payload for payload in payloads))
            self.assertEqual(evidence["runtime_adapters"], [])
            self.assertEqual(evidence["post_warmup_runtime_adapters"], [])
            self.assertEqual(
                [row["adapter_activation"] for row in evidence["warmups"]],
                ["static_fusion", "static_fusion"],
            )
            self.assertEqual(
                [row["adapter_id"] for row in evidence["warmups"]],
                [None, None],
            )
            self.assertEqual(
                [row["expected_token"] for row in evidence["warmups"]],
                ["F", "A"],
            )


if __name__ == "__main__":
    unittest.main()
