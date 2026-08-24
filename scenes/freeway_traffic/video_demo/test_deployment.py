import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scenes.freeway_traffic.video_demo.deployment import (
    DeploymentError,
    GuidedDeploymentController,
)


def _profile() -> dict:
    return {
        "schema_version": "guided-deployment-profiles/v1",
        "profiles": [
            {
                "id": "lab",
                "label": "比赛设备",
                "enabled": True,
                "allow_deploy": True,
                "cloud_url": "http://192.0.2.10:18100",
                "edge_url": "http://192.0.2.11:18101",
                "cloud": {
                    "label": "云端3090",
                    "host": "192.0.2.10",
                    "user": "tester",
                    "ssh_port": 22,
                    "identity_file": "/keys/id_ed25519",
                    "sdk_root": "/opt/cloud-edge",
                },
                "edge": {
                    "label": "Orin Nano",
                    "host": "192.0.2.11",
                    "user": "tester",
                    "ssh_port": 22,
                    "identity_file": "/keys/id_ed25519",
                    "sdk_root": "/opt/cloud-edge",
                    "llama_binary": "/opt/cloud-edge/runtime/bin/llama-server",
                },
            }
        ],
    }


def _profile_v2() -> dict:
    """Two clouds and three edges with distinct public and SSH addresses."""

    nodes = [
        {
            "id": "cloud-a",
            "role": "cloud",
            "label": "云端 A",
            "service_url": "http://192.0.2.10:18100",
            "host": "198.51.100.10",
            "user": "cloud_a_operator",
            "ssh_port": 22,
            "identity_file": "/run/secrets/cloud-a-ed25519",
            "sdk_root": "/opt/cloud-edge/cloud-a",
        },
        {
            "id": "cloud-b",
            "role": "cloud",
            "label": "云端 B",
            "service_url": "http://192.0.2.11:18100",
            "ssh": {
                "host": "198.51.100.11",
                "user": "cloud_b_operator",
                "ssh_port": 2222,
                "identity_file": "/run/secrets/cloud-b-ed25519",
            },
            "sdk_root": "/opt/cloud-edge/cloud-b",
        },
    ]
    for index, name in enumerate(("a", "b", "c"), start=1):
        nodes.append(
            {
                "id": f"edge-{name}",
                "role": "edge",
                "label": f"边缘 {name.upper()}",
                "service_url": f"http://192.0.2.{20 + index}:18101",
                "host": f"198.51.100.{20 + index}",
                "user": f"edge_{name}_operator",
                "ssh_port": 22,
                "identity_file": f"/run/secrets/edge-{name}-ed25519",
                "sdk_root": f"/opt/cloud-edge/edge-{name}",
                "llama_binary": f"/opt/cloud-edge/edge-{name}/runtime/llama-server",
            }
        )
    return {
        "schema_version": "guided-deployment-profiles/v2",
        "profiles": [
            {
                "id": "multi-lab",
                "label": "多节点比赛设备",
                "enabled": True,
                "allow_deploy": True,
                "nodes": nodes,
                "default_topology": {
                    "primary_cloud_id": "cloud-a",
                    "cloud_node_ids": ["cloud-a", "cloud-b"],
                    "edge_bindings": [
                        {"edge_id": "edge-a", "cloud_id": "cloud-a"},
                        {"edge_id": "edge-b", "cloud_id": "cloud-b"},
                        {"edge_id": "edge-c", "cloud_id": "cloud-a"},
                    ],
                },
            }
        ],
    }


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


class FakeRunner:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        source = kwargs.get("input", "")
        if "platform.system" in source:
            output = {
                "ok": True,
                "system": "Linux",
                "architecture": "aarch64",
                "python": "3.10.12",
                "paths": {"/opt/cloud-edge": True},
                "memory_bytes": 8_000_000_000,
                "disk_free_bytes": 40_000_000_000,
                "cuda_visible": True,
                "hostname": "node",
            }
        elif "install_traffic_systemd.sh" in source:
            output = {"returncode": 0, "stdout": "active", "stderr": ""}
        elif "run_partitioned_current_state_edges.py" in source:
            output = {
                "returncode": 0,
                "sample": {"status": "passed", "samples": [{"final": True}]},
                "stdout": "passed",
                "stderr": "",
            }
        elif "systemctl','disable" in source:
            output = {"returncode": 0, "stdout": "stopped", "stderr": ""}
        else:
            raise AssertionError(source)
        return subprocess.CompletedProcess(command, 0, json.dumps(output), "")


class TopologyRunner:
    """Exercise the real SSH bridge while recording node-level effects."""

    HOST_TO_NODE = {
        "198.51.100.10": "cloud-a",
        "198.51.100.11": "cloud-b",
        "198.51.100.21": "edge-a",
        "198.51.100.22": "edge-b",
        "198.51.100.23": "edge-c",
    }

    def __init__(self, fail_install_node=None) -> None:
        self.fail_install_node = fail_install_node
        self.calls = []
        self.operations = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        source = kwargs.get("input", "")
        target = command[-3]
        host = target.rsplit("@", 1)[-1]
        node_id = self.HOST_TO_NODE[host]
        if "platform.system" in source:
            self.operations.append(("preflight", node_id))
            output = {
                "ok": True,
                "system": "Linux",
                "architecture": "aarch64" if node_id.startswith("edge") else "x86_64",
                "python": "3.10.12",
                "paths": {},
                "memory_bytes": 8_000_000_000,
                "disk_free_bytes": 40_000_000_000,
                "cuda_visible": True,
                "hostname": node_id,
            }
        elif "install_traffic_systemd.sh" in source:
            role = "edge" if 'role="edge"' in source else "cloud"
            self.operations.append(("install", node_id, role, source))
            output = {
                "returncode": 19 if node_id == self.fail_install_node else 0,
                "stdout": "active" if node_id != self.fail_install_node else "",
                "stderr": "controlled install failure"
                if node_id == self.fail_install_node
                else "",
            }
        elif "systemctl','disable" in source:
            role = "edge" if "cloud-edge-traffic-edge" in source else "cloud"
            self.operations.append(("stop", node_id, role, source))
            output = {"returncode": 0, "stdout": "stopped", "stderr": ""}
        else:
            raise AssertionError(source)
        return subprocess.CompletedProcess(command, 0, json.dumps(output), "")


class GuidedDeploymentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.profile_path = root / "profiles.json"
        self.profile_path.write_text(json.dumps(_profile()), encoding="utf-8")
        self.runner = FakeRunner()
        self.controller = GuidedDeploymentController(
            profiles_path=self.profile_path,
            state_root=root / "state",
            runner=self.runner,
        )

    def v2_controller(self, *, runner=None, payload=None):
        root = Path(self.temp.name)
        profile_path = root / "profiles-v2.json"
        profile_path.write_text(
            json.dumps(payload or _profile_v2()), encoding="utf-8"
        )
        return GuidedDeploymentController(
            profiles_path=profile_path,
            state_root=root / "state-v2",
            runner=runner or TopologyRunner(),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_no_profile_is_explanation_only(self) -> None:
        controller = GuidedDeploymentController(
            profiles_path=None,
            state_root=Path(self.temp.name) / "empty",
        )
        self.assertFalse(controller.catalog()["live_deployment_available"])
        with self.assertRaisesRegex(DeploymentError, "尚未配置"):
            controller.create_session(
                {"scenario": "traffic", "template": "recommended", "profile_id": ""}
            )

    def test_profile_catalog_does_not_expose_ssh_credentials_or_paths(self) -> None:
        catalog = self.controller.catalog()
        encoded = json.dumps(catalog)
        self.assertTrue(catalog["live_deployment_available"])
        forbidden_keys = {
            "host",
            "user",
            "ssh_port",
            "identity_file",
            "sdk_root",
            "llama_binary",
            "ssh",
            "path",
        }
        self.assertTrue(forbidden_keys.isdisjoint(set(_all_keys(catalog))))
        self.assertNotIn("/keys/id_ed25519", encoded)
        self.assertNotIn("/opt/cloud-edge", encoded)
        self.assertIn("http://192.0.2.10:18100", encoded)

    def test_v1_profile_is_normalized_without_breaking_the_original_contract(self) -> None:
        profile = self.controller.catalog()["profiles"][0]
        self.assertEqual(
            profile["profile_schema_version"], "guided-deployment-profiles/v1"
        )
        self.assertEqual(
            profile["default_topology"],
            {
                "primary_cloud_id": "cloud",
                "cloud_node_ids": ["cloud"],
                "edge_bindings": [{"edge_id": "edge", "cloud_id": "cloud"}],
            },
        )
        session = self.controller.create_session(
            {"scenario": "traffic", "template": "recommended", "profile_id": "lab"}
        )
        self.assertEqual(session["topology"], profile["default_topology"])
        self.assertEqual(len(session["topology_sha256"]), 64)

    def test_v2_catalog_exposes_ids_and_health_origins_but_no_ssh_secrets(self) -> None:
        controller = self.v2_controller()
        catalog = controller.catalog()
        encoded = json.dumps(catalog)
        profile = catalog["profiles"][0]
        self.assertEqual(
            profile["profile_schema_version"], "guided-deployment-profiles/v2"
        )
        self.assertEqual(len(profile["nodes"]), 5)
        self.assertTrue(all(node["credential_ready"] for node in profile["nodes"]))
        self.assertIn("http://192.0.2.10:18100", encoded)
        for secret in (
            "198.51.100.",
            "cloud_a_operator",
            "edge_a_operator",
            "/run/secrets/",
            "/opt/cloud-edge/",
        ):
            self.assertNotIn(secret, encoded)
        forbidden_keys = {
            "host",
            "user",
            "ssh_port",
            "identity_file",
            "sdk_root",
            "llama_binary",
            "ssh",
            "path",
        }
        self.assertTrue(forbidden_keys.isdisjoint(set(_all_keys(catalog))))

    def test_v2_freezes_two_cloud_three_edge_bindings_by_authorized_id(self) -> None:
        runner = TopologyRunner()
        controller = self.v2_controller(runner=runner)
        request = {
            "scenario": "traffic",
            "template": "recommended",
            "profile_id": "multi-lab",
            "topology": {
                "primary_cloud_id": "cloud-b",
                "cloud_node_ids": ["cloud-b", "cloud-a"],
                "edge_bindings": [
                    {"edge_id": "edge-c", "cloud_id": "cloud-b"},
                    {"edge_id": "edge-a", "cloud_id": "cloud-b"},
                    {"edge_id": "edge-b", "cloud_id": "cloud-a"},
                ],
            },
        }
        session = controller.create_session(request)
        expected = {
            "primary_cloud_id": "cloud-b",
            "cloud_node_ids": ["cloud-a", "cloud-b"],
            "edge_bindings": [
                {"edge_id": "edge-a", "cloud_id": "cloud-b"},
                {"edge_id": "edge-b", "cloud_id": "cloud-a"},
                {"edge_id": "edge-c", "cloud_id": "cloud-b"},
            ],
        }
        self.assertEqual(session["topology"], expected)
        self.assertEqual(len(session["topology_sha256"]), 64)
        request["topology"]["edge_bindings"][0]["cloud_id"] = "cloud-a"
        self.assertEqual(controller.get_session(session["session_id"])["topology"], expected)
        self.assertEqual(runner.calls, [])

        tampered = controller.get_session(session["session_id"])
        tampered["topology"]["edge_bindings"][0]["cloud_id"] = "cloud-a"
        controller._save(tampered)
        with self.assertRaisesRegex(DeploymentError, "拓扑"):
            controller.run_action(session["session_id"], "preflight", {})
        self.assertEqual(runner.calls, [])

    def test_v2_rejects_invalid_or_forged_topologies_without_remote_calls(self) -> None:
        runner = TopologyRunner()
        controller = self.v2_controller(runner=runner)
        base = {
            "scenario": "traffic",
            "template": "recommended",
            "profile_id": "multi-lab",
        }
        invalid_topologies = {
            "zero_cloud": {
                "primary_cloud_id": "cloud-a",
                "cloud_node_ids": [],
                "edge_bindings": [{"edge_id": "edge-a", "cloud_id": "cloud-a"}],
            },
            "zero_edge": {
                "primary_cloud_id": "cloud-a",
                "cloud_node_ids": ["cloud-a"],
                "edge_bindings": [],
            },
            "primary_missing": {
                "primary_cloud_id": "cloud-missing",
                "cloud_node_ids": ["cloud-a"],
                "edge_bindings": [{"edge_id": "edge-a", "cloud_id": "cloud-a"}],
            },
            "edge_bound_to_unselected_cloud": {
                "primary_cloud_id": "cloud-a",
                "cloud_node_ids": ["cloud-a"],
                "edge_bindings": [{"edge_id": "edge-a", "cloud_id": "cloud-b"}],
            },
            "forged_edge_id": {
                "primary_cloud_id": "cloud-a",
                "cloud_node_ids": ["cloud-a"],
                "edge_bindings": [{"edge_id": "edge-attacker", "cloud_id": "cloud-a"}],
            },
            "forged_cloud_id": {
                "primary_cloud_id": "cloud-attacker",
                "cloud_node_ids": ["cloud-attacker"],
                "edge_bindings": [{"edge_id": "edge-a", "cloud_id": "cloud-attacker"}],
            },
            "browser_supplied_node_record": {
                "primary_cloud_id": "cloud-a",
                "cloud_node_ids": ["cloud-a"],
                "edge_bindings": [{"edge_id": "edge-a", "cloud_id": "cloud-a"}],
                "nodes": [
                    {
                        "id": "edge-attacker",
                        "host": "attacker.invalid",
                        "identity_file": "/tmp/key",
                    }
                ],
            },
        }
        for name, topology in invalid_topologies.items():
            with self.subTest(name=name), self.assertRaises(DeploymentError):
                controller.create_session({**base, "topology": topology})
        state_root = Path(self.temp.name) / "state-v2"
        self.assertFalse(state_root.exists())
        self.assertEqual(runner.calls, [])

    def test_v2_profile_rejects_duplicate_node_ids_without_remote_calls(self) -> None:
        payload = _profile_v2()
        payload["profiles"][0]["nodes"][1]["id"] = "cloud-a"
        runner = TopologyRunner()
        with self.assertRaisesRegex(ValueError, "duplicate node id"):
            self.v2_controller(runner=runner, payload=payload)
        self.assertEqual(runner.calls, [])

    def test_runtime_status_requires_real_health_responses(self) -> None:
        empty = GuidedDeploymentController(
            profiles_path=None,
            state_root=Path(self.temp.name) / "empty-runtime",
        )
        self.assertFalse(empty.runtime_status()["online"])
        with mock.patch.object(
            GuidedDeploymentController,
            "_request_json",
            side_effect=[{"status": "ok"}, OSError("offline")],
        ):
            status = self.controller.runtime_status()
        self.assertTrue(status["configured"])
        self.assertFalse(status["online"])
        self.assertTrue(status["nodes"]["cloud"]["online"])
        self.assertFalse(status["nodes"]["edge"]["online"])

    def test_read_only_connection_probe_summarizes_runtime_identity(self) -> None:
        controller = GuidedDeploymentController(
            profiles_path=None,
            state_root=Path(self.temp.name) / "connection-probe",
            connection_defaults={
                "edge_url": "http://192.0.2.11:18101",
                "cloud_url": "http://127.0.0.1:18100",
                "cloud_advertised_url": "http://192.0.2.10:18100",
            },
        )
        health = {
            "status": "ok",
            "ready": True,
            "role": "edge",
            "framework_version": "0.4.1",
            "runtime": {
                "generation": 1,
                "scenes": ["industrial_anomaly", "traffic"],
                "plugins": [
                    {
                        "scene": "industrial_anomaly",
                        "health": {"product_count": 10},
                    },
                    {
                        "scene": "traffic",
                        "health": {
                            "edge_llm": {
                                "active": {
                                    "release_id": "joint-q4",
                                    "deployment": {
                                        "quantization": "Q4_K_M",
                                        "input_tokens": 16,
                                        "output_tokens": 1,
                                        "thinking": False,
                                    },
                                }
                            }
                        },
                    },
                ],
            },
        }
        with mock.patch.object(controller, "_request_json", return_value=health):
            result = controller.runtime_status()
        self.assertTrue(result["online"])
        self.assertFalse(result["partial_online"])
        self.assertEqual(result["cloud_advertised_url"], "http://192.0.2.10:18100")
        edge = result["nodes"]["edge"]["health"]
        self.assertEqual(edge["model"]["quantization"], "Q4_K_M")
        self.assertEqual(edge["industrial_product_count"], 10)

    def test_connection_probe_rejects_credentials_paths_and_non_http(self) -> None:
        for endpoint in (
            "ftp://127.0.0.1:18100",
            "http://user:secret@127.0.0.1:18100",
            "http://127.0.0.1:18100/health",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(DeploymentError):
                self.controller.probe_endpoints(
                    edge_url=endpoint,
                    cloud_url="http://127.0.0.1:18100",
                )

    def test_runtime_snapshot_is_bounded_and_read_only(self) -> None:
        def response(url: str, _timeout: float = 0) -> dict:
            if url.endswith("/api/v1/framework/outbox"):
                return {"active": 2, "states": {"pending": 2, "completed": 8}}
            if url.endswith("/api/v1/collaboration/reviews"):
                return {
                    "summary": {
                        "total": 11,
                        "states": {"queued": 1, "completed": 10},
                        "authoritative_completed": 9,
                    },
                    "recent": [
                        {
                            "event_id": f"event-{index}",
                            "scene": "traffic",
                            "requested_route": "cloud_async",
                            "state": "completed",
                            "preliminary_latency_ms": 12.5,
                            "planned_request_bytes": 640,
                            "decision_changed": False,
                            "completion_stage": "lightweight_final",
                            "attempts": 1,
                        }
                        for index in range(12)
                    ],
                }
            if url.endswith("/api/v1/framework/edge-llm/release"):
                return {"active_release_id": "q4", "applied_revision": 4, "running": True}
            if "18101" in url:
                return {
                    "counters": {
                        "edge_requests_total": 20,
                        "local_autonomy_total": 1,
                        "async_cloud_delivery_successes_total": 18,
                        "async_cloud_delivery_failures_total": 2,
                    }
                }
            if url.endswith("/api/v1/collaboration/aggregations"):
                return {"states": {"waiting": 1, "completed": 7}, "event_count": 21}
            return {
                "counters": {
                    "cloud_requests_total": 19,
                    "coordination_events_total": 16,
                    "coordination_conflicts_initial_total": 3,
                    "coordination_conflicts_residual_total": 0,
                    "coordination_conflict_resolution_successes_total": 3,
                },
                "distributions": {"cloud_service_runtime_ms": {"p50": 2.0, "p95": 8.0}},
            }

        with mock.patch.object(self.controller, "_request_json", side_effect=response):
            snapshot = self.controller.runtime_snapshot(
                edge_url="http://192.0.2.11:18101",
                cloud_url="http://192.0.2.10:18100",
            )
        self.assertTrue(snapshot["read_only"])
        self.assertEqual(snapshot["edge"]["outbox"]["active"], 2)
        self.assertEqual(snapshot["cloud"]["conflicts_residual_total"], 0)
        self.assertEqual(len(snapshot["recent_events"]), 8)

    def test_state_machine_rejects_skipping_and_requires_confirmation(self) -> None:
        session = self.controller.create_session(
            {"scenario": "traffic", "template": "recommended", "profile_id": "lab"}
        )
        with self.assertRaisesRegex(DeploymentError, "环境"):
            self.controller.run_action(session["session_id"], "generate_plan", {})
        session = self.controller.run_action(session["session_id"], "preflight", {})
        session = self.controller.run_action(session["session_id"], "generate_plan", {})
        with self.assertRaisesRegex(DeploymentError, "确认"):
            self.controller.run_action(session["session_id"], "deploy", {})

    def test_v2_deploys_all_clouds_before_edges_and_preserves_each_binding(self) -> None:
        runner = TopologyRunner()
        controller = self.v2_controller(runner=runner)
        session = controller.create_session(
            {
                "scenario": "traffic",
                "template": "recommended",
                "profile_id": "multi-lab",
                "topology": {
                    "primary_cloud_id": "cloud-b",
                    "cloud_node_ids": ["cloud-a", "cloud-b"],
                    "edge_bindings": [
                        {"edge_id": "edge-a", "cloud_id": "cloud-b"},
                        {"edge_id": "edge-b", "cloud_id": "cloud-a"},
                        {"edge_id": "edge-c", "cloud_id": "cloud-b"},
                    ],
                },
            }
        )
        session = controller.run_action(session["session_id"], "preflight", {})
        session = controller.run_action(session["session_id"], "generate_plan", {})
        session = controller.run_action(
            session["session_id"],
            "deploy",
            {"confirmation": "START_TRAFFIC_DEPLOYMENT"},
        )

        install_operations = [
            operation for operation in runner.operations if operation[0] == "install"
        ]
        self.assertEqual(
            [operation[1] for operation in install_operations],
            ["cloud-b", "cloud-a", "edge-a", "edge-b", "edge-c"],
        )
        self.assertEqual(
            [operation[2] for operation in install_operations],
            ["cloud", "cloud", "edge", "edge", "edge"],
        )
        self.assertEqual(
            session["deployment"]["install_order"],
            ["cloud-b", "cloud-a", "edge-a", "edge-b", "edge-c"],
        )
        deploy_result = next(
            stage["result"] for stage in session["stages"] if stage["id"] == "deploy"
        )
        self.assertEqual(deploy_result["primary_cloud_id"], "cloud-b")
        self.assertEqual(deploy_result["nodes"]["edge-a"]["cloud_id"], "cloud-b")
        self.assertEqual(deploy_result["nodes"]["edge-b"]["cloud_id"], "cloud-a")
        self.assertEqual(deploy_result["nodes"]["edge-c"]["cloud_id"], "cloud-b")
        install_source = {operation[1]: operation[3] for operation in install_operations}
        self.assertIn("http://192.0.2.11:18100", install_source["edge-a"])
        self.assertIn("http://192.0.2.10:18100", install_source["edge-b"])
        self.assertIn("http://192.0.2.11:18100", install_source["edge-c"])

    def test_v2_partial_failure_compensates_only_started_nodes_in_reverse_order(self) -> None:
        runner = TopologyRunner(fail_install_node="edge-b")
        controller = self.v2_controller(runner=runner)
        session = controller.create_session(
            {
                "scenario": "traffic",
                "template": "recommended",
                "profile_id": "multi-lab",
            }
        )
        session = controller.run_action(session["session_id"], "preflight", {})
        session = controller.run_action(session["session_id"], "generate_plan", {})
        with self.assertRaisesRegex(DeploymentError, "安装"):
            controller.run_action(
                session["session_id"],
                "deploy",
                {"confirmation": "START_TRAFFIC_DEPLOYMENT"},
            )

        failed = controller.get_session(session["session_id"])
        installs = [
            operation[1]
            for operation in runner.operations
            if operation[0] == "install"
        ]
        stops = [
            operation[1] for operation in runner.operations if operation[0] == "stop"
        ]
        self.assertEqual(installs, ["cloud-a", "cloud-b", "edge-a", "edge-b"])
        self.assertEqual(stops, ["edge-a", "cloud-b", "cloud-a"])
        self.assertNotIn("edge-b", stops)
        self.assertNotIn("edge-c", stops)
        self.assertEqual(
            failed["deployment"]["installed_node_ids"],
            ["cloud-a", "cloud-b", "edge-a"],
        )
        self.assertEqual(failed["deployment"]["active_node_ids"], [])
        compensation = failed["deployment"]["compensation"]
        self.assertEqual(
            compensation["attempted_node_ids"],
            ["edge-a", "cloud-b", "cloud-a"],
        )
        self.assertEqual(
            compensation["completed_node_ids"],
            ["edge-a", "cloud-b", "cloud-a"],
        )
        self.assertEqual(compensation["failures"], {})
        self.assertEqual(failed["status"], "failed")

    def test_v2_verification_failure_compensates_every_active_node_in_reverse(self) -> None:
        runner = TopologyRunner()
        controller = self.v2_controller(runner=runner)
        session = controller.create_session(
            {
                "scenario": "traffic",
                "template": "recommended",
                "profile_id": "multi-lab",
            }
        )
        session = controller.run_action(session["session_id"], "preflight", {})
        session = controller.run_action(session["session_id"], "generate_plan", {})
        session = controller.run_action(
            session["session_id"],
            "deploy",
            {"confirmation": "START_TRAFFIC_DEPLOYMENT"},
        )
        with mock.patch.object(
            controller, "_request_json", side_effect=OSError("controlled offline")
        ), self.assertRaisesRegex(DeploymentError, "健康检查"):
            controller.run_action(session["session_id"], "verify", {})

        failed = controller.get_session(session["session_id"])
        stops = [
            operation[1] for operation in runner.operations if operation[0] == "stop"
        ]
        self.assertEqual(
            stops, ["edge-c", "edge-b", "edge-a", "cloud-b", "cloud-a"]
        )
        self.assertEqual(failed["deployment"]["active_node_ids"], [])
        self.assertEqual(
            failed["deployment"]["compensation"]["attempted_node_ids"], stops
        )
        verify_stage = next(
            stage for stage in failed["stages"] if stage["id"] == "verify"
        )
        self.assertEqual(verify_stage["status"], "failed")
        self.assertEqual(verify_stage["error"]["code"], "HEALTH_CHECK_FAILED")

    def test_v2_rollback_refuses_to_touch_nodes_not_installed_by_the_session(self) -> None:
        runner = TopologyRunner()
        controller = self.v2_controller(runner=runner)
        session = controller.create_session(
            {
                "scenario": "traffic",
                "template": "recommended",
                "profile_id": "multi-lab",
            }
        )
        with self.assertRaisesRegex(DeploymentError, "没有可回滚"):
            controller.run_action(
                session["session_id"],
                "rollback",
                {"confirmation": "ROLLBACK_TRAFFIC_DEPLOYMENT"},
            )
        self.assertEqual(runner.calls, [])

        forged = controller.get_session(session["session_id"])
        forged["deployment"]["active_node_ids"] = ["edge-a"]
        forged["deployment"]["installed_node_ids"] = []
        controller._save(forged)
        with self.assertRaisesRegex(DeploymentError, "范围"):
            controller.run_action(
                session["session_id"],
                "rollback",
                {"confirmation": "ROLLBACK_TRAFFIC_DEPLOYMENT"},
            )
        self.assertEqual(runner.calls, [])

    def test_topology_probe_accepts_at_most_32_bounded_valid_endpoints(self) -> None:
        nodes = [
            {
                "id": f"node-{index}",
                "role": "cloud" if index == 0 else "edge",
                "label": f"节点 {index}",
                "endpoint": f"http://192.0.2.{index + 1}:{18100 + index}",
            }
            for index in range(32)
        ]

        def probe(role, endpoint, timeout):
            return {
                "online": True,
                "role": role,
                "endpoint": endpoint,
                "latency_ms": 1.0,
                "timeout": timeout,
            }

        with mock.patch.object(
            self.controller, "_probe_node", side_effect=probe
        ) as probe_node:
            result = self.controller.probe_topology(
                nodes=nodes, timeout_seconds=99
            )
        self.assertEqual(result["node_count"], 32)
        self.assertEqual(result["online_count"], 32)
        self.assertTrue(result["online"])
        self.assertTrue(result["read_only"])
        self.assertEqual(probe_node.call_count, 32)
        self.assertTrue(all(call.args[2] == 5.0 for call in probe_node.call_args_list))

    def test_topology_probe_rejects_oversize_or_invalid_inputs_before_requests(self) -> None:
        valid = {
            "id": "cloud-a",
            "role": "cloud",
            "label": "云端 A",
            "endpoint": "http://192.0.2.10:18100",
        }
        oversized = [
            {
                "id": f"node-{index}",
                "role": "edge",
                "label": f"节点 {index}",
                "endpoint": f"http://192.0.2.{index + 1}:{18100 + index}",
            }
            for index in range(33)
        ]
        invalid_cases = [
            [],
            oversized,
            [{**valid, "endpoint": "https://192.0.2.10:18100"}],
            [{**valid, "endpoint": "http://user:secret@192.0.2.10:18100"}],
            [{**valid, "endpoint": "http://192.0.2.10:18100/health"}],
            [{**valid, "endpoint": "http://192.0.2.10:18100?admin=1"}],
            [{**valid, "unexpected": "field"}],
            [valid, {**valid, "id": "cloud-b"}],
            [valid, {**valid, "endpoint": "http://192.0.2.11:18100"}],
        ]
        with mock.patch.object(self.controller, "_probe_node") as probe_node:
            for index, nodes in enumerate(invalid_cases):
                with self.subTest(case=index), self.assertRaises(DeploymentError):
                    self.controller.probe_topology(nodes=nodes)
        probe_node.assert_not_called()

    def test_complete_real_path_generates_required_deliverables(self) -> None:
        session = self.controller.create_session(
            {"scenario": "traffic", "template": "recommended", "profile_id": "lab"}
        )
        session = self.controller.run_action(session["session_id"], "preflight", {})
        session = self.controller.run_action(session["session_id"], "generate_plan", {})
        session = self.controller.run_action(
            session["session_id"],
            "deploy",
            {"confirmation": "START_TRAFFIC_DEPLOYMENT"},
        )
        with mock.patch.object(
            GuidedDeploymentController,
            "_request_json",
            side_effect=[{"status": "ready", "model": "cloud"}, {"status": "ready", "model": "edge"}],
        ):
            session = self.controller.run_action(session["session_id"], "verify", {})
        self.assertEqual(session["status"], "completed")
        expected = {
            "deployment_manifest.json",
            "active_model_manifest.json",
            "health_check_report.json",
            "system_topology.json",
            "deployment_report.html",
            "rollback.sh",
        }
        self.assertTrue(expected.issubset(session["artifacts"]))
        for name in expected:
            self.assertTrue(Path(session["artifacts"][name]).is_file())
        commands = [call[0] for call in self.runner.calls]
        self.assertTrue(all(command[0] == "ssh" for command in commands))
        self.assertTrue(all("ControlPath=none" in command for command in commands))

    def test_unsupported_scene_and_template_fail_closed(self) -> None:
        for request in (
            {"scenario": "industrial", "template": "recommended", "profile_id": "lab"},
            {"scenario": "traffic", "template": "offline", "profile_id": "lab"},
        ):
            with self.subTest(request=request), self.assertRaises(DeploymentError):
                self.controller.create_session(request)


if __name__ == "__main__":
    unittest.main()
