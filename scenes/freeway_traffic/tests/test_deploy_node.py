"""Focused tests for immutable service-template selection and child startup."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import call, patch


SCENE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SCENE_ROOT.parents[1]
if str(SCENE_ROOT) not in sys.path:
    sys.path.insert(0, str(SCENE_ROOT))

import deploy_node  # noqa: E402


class DeployNodeConfigTests(unittest.TestCase):
    @staticmethod
    def _write_template(path, role="edge", port=19501, host="0.0.0.0"):
        value = {
            "schema_version": 1,
            "role": role,
            "plugin_config": (
                "scenes/industrial_anomaly/deployment/"
                "scene_plugins_edge_traffic_industrial.json"
            ),
            "listen": {"host": host, "port": port},
            "cloud": {"base_url": "http://old-cloud:18100"},
            "release_watch": {
                "enabled": True,
                "registry": "scenes/freeway_traffic/runtime/edge_llm_release_store.json",
            },
        }
        path.write_text(json.dumps(value), encoding="utf-8")
        return value

    def test_explicit_template_is_copied_and_cloud_url_is_overridden(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "candidate-edge.json"
            original = self._write_template(template)
            with patch.object(deploy_node, "RUNTIME_ROOT", root / "runtime"):
                generated = deploy_node._generated_service_config(
                    "edge",
                    "http://192.0.2.10:18100/",
                    False,
                    service_config=template,
                )

            result = json.loads(generated.read_text(encoding="utf-8"))
            self.assertEqual(result["cloud"]["base_url"], "http://192.0.2.10:18100")
            self.assertEqual(result["listen"]["port"], 19501)
            self.assertEqual(
                result["plugin_config"],
                original["plugin_config"],
            )
            self.assertEqual(
                json.loads(template.read_text(encoding="utf-8")),
                original,
            )

    def test_omitting_explicit_template_preserves_full_deployment_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime_root = Path(temporary) / "runtime"
            with patch.object(deploy_node, "RUNTIME_ROOT", runtime_root):
                generated = deploy_node._generated_service_config(
                    "edge",
                    "http://192.0.2.20:18100",
                    False,
                )

            result = json.loads(generated.read_text(encoding="utf-8"))
            self.assertEqual(result["role"], "edge")
            self.assertEqual(
                result["plugin_config"],
                "scenes/freeway_traffic/deployment/full/scene_plugins_edge.json",
            )
            self.assertEqual(result["cloud"]["base_url"], "http://192.0.2.20:18100")

    def test_explicit_template_must_match_requested_role(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "cloud.json"
            self._write_template(template, role="cloud", port=19500)
            with patch.object(deploy_node, "RUNTIME_ROOT", root / "runtime"):
                with self.assertRaisesRegex(ValueError, "角色不匹配"):
                    deploy_node._generated_service_config(
                        "edge",
                        "http://192.0.2.10:18100",
                        False,
                        service_config=template,
                    )

    def test_readiness_url_uses_candidate_port_and_stays_local(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wildcard = root / "wildcard.json"
            ipv6 = root / "ipv6.json"
            remote = root / "remote.json"
            self._write_template(wildcard, port=19501)
            self._write_template(ipv6, port=19502, host="::")
            self._write_template(remote, port=19503, host="example.com")

            self.assertEqual(
                deploy_node._service_readiness_url(wildcard),
                "http://127.0.0.1:19501/ready",
            )
            self.assertEqual(
                deploy_node._service_readiness_url(ipv6),
                "http://[::1]:19502/ready",
            )
            with self.assertRaisesRegex(ValueError, "远端地址"):
                deploy_node._service_readiness_url(remote)

    def test_service_release_registry_resolves_from_project_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "edge.json"
            self._write_template(config)
            self.assertEqual(
                deploy_node._service_release_registry(config),
                (
                    REPOSITORY_ROOT
                    / "scenes/freeway_traffic/runtime/edge_llm_release_store.json"
                ).resolve(),
            )

    def test_multi_runtime_outputs_must_match_enabled_scene_plugins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traffic_runtime = root / "runtime" / "traffic.json"
            industrial_runtime = root / "runtime" / "industrial.json"
            plugin_config = root / "plugins.json"
            service_config = root / "service.json"
            plugin_config.write_text(
                json.dumps(
                    {
                        "plugins": [
                            {
                                "enabled": True,
                                "options": {
                                    "edge_llm_mode": "selective",
                                    "edge_llm_runtime_config_path": str(
                                        traffic_runtime
                                    ),
                                },
                            },
                            {
                                "enabled": True,
                                "options": {
                                    "edge_llm_mode": "selective",
                                    "edge_llm_runtime_config_path": str(
                                        industrial_runtime
                                    ),
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            service_config.write_text(
                json.dumps({"plugin_config": str(plugin_config)}),
                encoding="utf-8",
            )
            matched = deploy_node._verify_runtime_outputs_match_service(
                service_config,
                [
                    {"output": traffic_runtime},
                    {"output": industrial_runtime},
                ],
            )
            self.assertEqual(matched["status"], "matched")
            with self.assertRaisesRegex(ValueError, "不一致"):
                deploy_node._verify_runtime_outputs_match_service(
                    service_config,
                    [{"output": traffic_runtime}],
                )

    def test_child_pythonpath_adds_both_scenes_without_mutating_parent(self):
        inherited = os.pathsep.join(
            [
                str(SCENE_ROOT),
                "relative-support",
                "",
            ]
        )
        with patch.dict(os.environ, {"PYTHONPATH": inherited}, clear=False):
            environment = deploy_node._child_environment()
            self.assertEqual(os.environ["PYTHONPATH"], inherited)

        entries = environment["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(entries[0], str(REPOSITORY_ROOT.resolve()))
        self.assertEqual(entries[1], str(SCENE_ROOT.resolve()))
        self.assertIn(
            str((REPOSITORY_ROOT / "scenes" / "industrial_anomaly").resolve()),
            entries,
        )
        self.assertEqual(entries.count(str(SCENE_ROOT.resolve())), 1)
        self.assertNotIn("", entries)
        self.assertIn(str((Path.cwd() / "relative-support").resolve()), entries)

    def test_active_release_replaces_legacy_catalogued_edge_qwen(self):
        catalog = {
            "downloaded_assets": {
                "edge_qwen_gguf": {"file": "retired-q6.gguf"},
                "pems08_inference_array": {
                    "file": "pems08.npz",
                    "startup_required": False,
                },
                "required_runtime_asset": {"file": "required.bin"},
            }
        }
        release = {
            "active_release_id": "clean-q8-plus-two-loras",
            "releases": {
                "clean-q8-plus-two-loras": {
                    "deployment_artifact": {
                        "path": "/models/base.Q8_0.gguf",
                        "bytes": 811835392,
                        "sha256": "6" * 64,
                    }
                }
            },
        }
        with patch.object(
            deploy_node,
            "_verify_asset",
            return_value={"path": "/assets/pems08.npz", "status": "verified"},
        ) as verify_asset:
            results = deploy_node._verify_edge_downloaded_assets(catalog, release)

        verify_asset.assert_called_once_with({"file": "required.bin"})
        self.assertEqual(results[0]["status"], "verified_by_active_release")
        self.assertEqual(results[0]["release_id"], "clean-q8-plus-two-loras")
        self.assertEqual(results[0]["path"], "/models/base.Q8_0.gguf")
        self.assertEqual(results[1]["catalog_asset"], "pems08_inference_array")
        self.assertEqual(
            results[1]["status"], "not_required_for_service_startup"
        )
        self.assertEqual(results[2]["catalog_asset"], "required_runtime_asset")

    def test_cli_forwards_explicit_service_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            template = Path(temporary) / "candidate-edge.json"
            self._write_template(template)
            with patch.object(deploy_node, "run_node") as run_node:
                deploy_node.main(
                    [
                        "run",
                        "--role",
                        "edge",
                        "--cloud-url",
                        "http://192.0.2.10:18100",
                        "--llama-binary",
                        "/opt/llama-server",
                        "--llama-registry",
                        "/runtime/static-candidate-release-store.json",
                        "--service-config",
                        str(template),
                        "--llama-no-mmap",
                    ]
                )

            args = run_node.call_args.args[0]
            self.assertEqual(args.service_config, str(template))
            self.assertEqual(
                args.llama_registry,
                "/runtime/static-candidate-release-store.json",
            )
            self.assertFalse(args.with_cloud_qwen9b)
            self.assertTrue(args.llama_no_mmap)

    def test_cli_forwards_repeatable_release_runtime_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            template = Path(temporary) / "candidate-edge.json"
            self._write_template(template)
            with patch.object(deploy_node, "run_node") as run_node:
                deploy_node.main(
                    [
                        "run",
                        "--role",
                        "edge",
                        "--cloud-url",
                        "http://192.0.2.10:18100",
                        "--llama-binary",
                        "/opt/llama-server",
                        "--service-config",
                        str(template),
                        "--llama-runtime-output",
                        "/config/traffic-runtime-output.json",
                        "--llama-runtime-output",
                        "/config/industrial-runtime-output.json",
                    ]
                )

        args = run_node.call_args.args[0]
        self.assertEqual(
            args.llama_runtime_output,
            [
                "/config/traffic-runtime-output.json",
                "/config/industrial-runtime-output.json",
            ],
        )

    def test_run_node_uses_candidate_port_and_child_environment(self):
        class Process:
            returncode = 1

            def __init__(self, pid, failed=False):
                self.pid = pid
                self.failed = failed

            def poll(self):
                return 1 if self.failed else None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "candidate-edge.json"
            self._write_template(template, port=19501)
            template_value = json.loads(template.read_text(encoding="utf-8"))
            template_value["release_watch"]["registry"] = str(
                root / "candidate-release-store.json"
            )
            template.write_text(json.dumps(template_value), encoding="utf-8")
            args = SimpleNamespace(
                role="edge",
                cloud_url="http://192.0.2.10:18100",
                with_cloud_qwen9b=False,
                service_config=str(template),
                llama_binary="/opt/llama-server",
                llama_registry=str(root / "candidate-release-store.json"),
                device="cuda",
                llama_port=18190,
                context_tokens=128,
                threads=4,
                llama_batch_size=8,
                llama_ubatch_size=8,
                parallel=1,
                gpu_layers=99,
                llama_no_mmap=True,
                disable_cuda_graphs=True,
                llama_lora_adapter=[],
                llama_startup_probes=None,
                startup_timeout_seconds=5.0,
            )
            model_process = Process(1001)
            service_process = Process(1002, failed=True)
            with (
                patch.object(deploy_node, "RUNTIME_ROOT", root / "runtime"),
                patch.object(
                    deploy_node, "check_installation"
                ) as check_installation,
                patch.object(deploy_node.signal, "signal"),
                patch.object(
                    deploy_node,
                    "_wait_startup_gate",
                    return_value={"status": "passed"},
                ) as wait_startup_gate,
                patch.object(deploy_node, "_wait_health") as wait_health,
                patch.object(deploy_node, "_stop_processes"),
                patch("builtins.print"),
                patch.object(
                    deploy_node.subprocess,
                    "Popen",
                    side_effect=[model_process, service_process],
                ) as popen,
            ):
                with patch.dict(
                    os.environ,
                    {"GGML_CUDA_DISABLE_GRAPHS": "ambient-parent-value"},
                    clear=False,
                ):
                    with self.assertRaisesRegex(RuntimeError, "子进程异常退出"):
                        deploy_node.run_node(args)
                    self.assertEqual(
                        os.environ["GGML_CUDA_DISABLE_GRAPHS"],
                        "ambient-parent-value",
                    )

            self.assertEqual(
                wait_health.call_args_list,
                [
                    call(
                        "http://127.0.0.1:19501/ready",
                        5.0,
                        service_process,
                    ),
                ],
            )
            wait_startup_gate.assert_called_once_with(
                root / "runtime/generated/llama_startup_gate_18190.json",
                5.0,
                model_process,
            )
            release_command = popen.call_args_list[0].args[0]
            check_installation.assert_called_once_with(
                "edge",
                "/opt/llama-server",
                "cuda",
                llama_registry=(root / "candidate-release-store.json").resolve(),
            )
            self.assertEqual(
                release_command[release_command.index("--registry") + 1],
                str((root / "candidate-release-store.json").resolve()),
            )
            self.assertIn("--startup-gate-receipt", release_command)
            self.assertIn("--runtime-config", release_command)
            self.assertNotIn("--runtime-output", release_command)
            self.assertEqual(
                release_command[release_command.index("--batch-size") + 1],
                "8",
            )
            self.assertEqual(
                release_command[release_command.index("--ubatch-size") + 1],
                "8",
            )
            self.assertEqual(release_command.count("--no-mmap"), 1)
            self.assertEqual(
                popen.call_args_list[0].kwargs["env"][
                    "GGML_CUDA_DISABLE_GRAPHS"
                ],
                "1",
            )
            self.assertNotIn(
                "GGML_CUDA_DISABLE_GRAPHS",
                popen.call_args_list[1].kwargs["env"],
            )
            for invocation in popen.call_args_list:
                child_path = invocation.kwargs["env"]["PYTHONPATH"].split(
                    os.pathsep
                )
                self.assertIn(
                    str(
                        (
                            REPOSITORY_ROOT
                            / "scenes"
                            / "industrial_anomaly"
                        ).resolve()
                    ),
                    child_path,
                )

    def test_run_cli_defaults_keep_existing_llama_batch_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            template = Path(temporary) / "candidate-edge.json"
            self._write_template(template)
            with patch.object(deploy_node, "run_node") as run_node:
                deploy_node.main(
                    [
                        "run",
                        "--role",
                        "edge",
                        "--cloud-url",
                        "http://192.0.2.10:18100",
                        "--llama-binary",
                        "/opt/llama-server",
                        "--service-config",
                        str(template),
                    ]
                )

        args = run_node.call_args.args[0]
        self.assertEqual(args.llama_batch_size, 16)
        self.assertEqual(args.llama_ubatch_size, 16)
        self.assertFalse(args.llama_no_mmap)
        self.assertFalse(args.disable_cuda_graphs)

    def test_llama_batch_cli_values_must_be_positive(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            deploy_node._positive_int("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            deploy_node._positive_int("-1")
        self.assertEqual(deploy_node._positive_int("8"), 8)


if __name__ == "__main__":
    unittest.main()
