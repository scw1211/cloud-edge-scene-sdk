"""四分区独立边缘服务配置与进程隔离测试。"""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRAFFIC_ROOT = REPOSITORY_ROOT / "scenes" / "freeway_traffic"
for import_root in (REPOSITORY_ROOT, TRAFFIC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from run_partitioned_current_state_edges import (  # noqa: E402
    _business_completion_ms,
    _four_edge_provisional_barrier_ms,
    _launch_isolated_edge_services,
    _stop_edge_services,
)


class _FakeProcess:
    next_pid = 41000

    def __init__(self, command, **kwargs):
        del kwargs
        self.command = list(command)
        self.pid = _FakeProcess.next_pid
        _FakeProcess.next_pid += 1
        self.return_code = None

    def poll(self):
        return self.return_code

    def terminate(self):
        self.return_code = 0

    def wait(self, timeout=None):
        del timeout
        return self.return_code

    def kill(self):
        self.return_code = -9


class PartitionedEdgeServiceTests(unittest.TestCase):
    def test_business_completion_is_controlled_by_policy_route(self):
        for policy_route in ("edge_only", "local_autonomy", "cloud_async"):
            event = {
                "policy_route": policy_route,
                "local_actions_authorized": True,
                "input_to_provisional_ms": 31.5,
            }
            self.assertEqual(31.5, _business_completion_ms(event, 180.0))

            event["local_actions_authorized"] = False
            self.assertEqual(180.0, _business_completion_ms(event, 180.0))

        sync_event = {
            "policy_route": "cloud_sync",
            "local_actions_authorized": True,
            "input_to_provisional_ms": 31.5,
        }
        self.assertEqual(180.0, _business_completion_ms(sync_event, 180.0))

        with self.assertRaisesRegex(RuntimeError, "unknown policy route"):
            _business_completion_ms(
                {
                    "policy_route": "mystery",
                    "local_actions_authorized": True,
                    "input_to_provisional_ms": 31.5,
                },
                180.0,
            )

    def test_provisional_barrier_rejects_direct_final_response(self):
        provisional = [
            {
                "partition_id": partition_id,
                "status": "provisional",
                "input_to_provisional_ms": 10.0 + partition_id,
            }
            for partition_id in range(4)
        ]
        self.assertEqual(13.0, _four_edge_provisional_barrier_ms(provisional))

        mixed = list(provisional)
        mixed[2] = {
            "partition_id": 2,
            "status": "final",
            "input_to_provisional_ms": None,
        }
        with self.assertRaisesRegex(RuntimeError, "cloud final response"):
            _four_edge_provisional_barrier_ms(mixed)

    def test_four_services_have_distinct_ports_and_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_root = root / "scenes" / "freeway_traffic"
            template_path = scene_root / "deployment" / "full" / "edge_service.json"
            template_path.parent.mkdir(parents=True)
            template_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "role": "edge",
                        "plugin_config": "plugins.json",
                        "listen": {"host": "0.0.0.0", "port": 18101},
                        "storage": {},
                        "cloud": {"base_url": "http://127.0.0.1:18100"},
                        "release_watch": {
                            "enabled": True,
                            "registry": "release.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            _FakeProcess.next_pid = 41000
            with mock.patch(
                "run_partitioned_current_state_edges._port_is_free",
                return_value=True,
            ), mock.patch(
                "run_partitioned_current_state_edges._wait_service_ready"
            ), mock.patch(
                "run_partitioned_current_state_edges.subprocess.Popen",
                side_effect=_FakeProcess,
            ):
                services = _launch_isolated_edge_services(
                    project_root=root,
                    scene_root=scene_root,
                    experiment_id="test-four-edges",
                    template_path=template_path,
                    cloud_url="http://192.0.2.10:18100",
                    port_base=19101,
                    startup_timeout_seconds=1.0,
                )

            self.assertEqual(4, len(services))
            self.assertEqual(4, len({item["process"].pid for item in services}))
            self.assertEqual(4, len({item["endpoint"] for item in services}))
            self.assertEqual(4, len({item["state_root"] for item in services}))
            for partition_id, service in enumerate(services):
                config = json.loads(
                    Path(service["config_path"]).read_text(encoding="utf-8")
                )
                self.assertEqual(19101 + partition_id, config["listen"]["port"])
                self.assertEqual("127.0.0.1", config["listen"]["host"])
                self.assertEqual(
                    "http://192.0.2.10:18100", config["cloud"]["base_url"]
                )
                self.assertFalse(config["release_watch"]["enabled"])
                paths = list(config["storage"].values())
                self.assertTrue(paths)
                self.assertTrue(
                    all(
                        "edge_node_{}".format(partition_id) in path
                        for path in paths
                    )
                )
            _stop_edge_services(services)
            self.assertTrue(
                all(service["process"].poll() == 0 for service in services)
            )


if __name__ == "__main__":
    unittest.main()
