import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
FORMAL = ROOT / "scenes" / "industrial_anomaly" / "deployment" / "joint_static_formal_v1"
MODEL_SHA = "308daa980c7ca295e18bd76e8dcf6dc1ed725ded32ada535a0c5c1910c695ce2"
REGISTRY = "runtime/traffic_industrial_joint_static_q5km_formal_v1/edge_llm_release_store.json"


def load(name):
    return json.loads((FORMAL / name).read_text(encoding="utf-8"))


class JointStaticFormalConfigTest(unittest.TestCase):
    def test_formal_ports_cloud_and_registry_are_consistent(self):
        service = load("edge_service.json")
        plugins = load("scene_plugins_edge.json")
        unit = (FORMAL / "cloud-edge-edge-18101-joint-static-q5-v1.service").read_text(
            encoding="utf-8"
        )

        self.assertEqual(service["listen"], {"host": "0.0.0.0", "port": 18101,
                                              "max_body_bytes": 8388608,
                                              "access_log": False})
        self.assertEqual(service["cloud"]["base_url"], "http://192.168.31.135:18100")
        self.assertEqual(service["release_watch"]["registry"], REGISTRY)
        self.assertIn(REGISTRY, unit)
        self.assertIn("--llama-port 18190", unit)
        self.assertIn("--llama-registry ", unit)
        traffic = next(plugin for plugin in plugins["plugins"] if "freeway" in plugin["spec"])
        industrial = next(
            plugin for plugin in plugins["plugins"] if "industrial" in plugin["spec"]
        )
        self.assertEqual(traffic["options"]["edge_llm_release_registry_path"], REGISTRY)
        self.assertEqual(
            industrial["options"]["edge_llm_selective_timeout_limit_seconds"], 0.5
        )

    def test_static_runtime_contract_has_no_request_adapter(self):
        outputs = [load("runtime_output_traffic.json"), load("runtime_output_industrial.json")]
        templates = [
            load("edge_llm_runtime_traffic_template.json"),
            load("edge_llm_runtime_industrial_template.json"),
        ]
        unit = (FORMAL / "cloud-edge-edge-18101-joint-static-q5-v1.service").read_text(
            encoding="utf-8"
        )

        for output in outputs:
            self.assertEqual(output["adapter_mode"], "static")
            self.assertEqual(output["static_deployment_sha256"], MODEL_SHA)
        for template in templates:
            self.assertEqual(template["generation"]["max_input_tokens"], 17)
            self.assertEqual(template["generation"]["max_output_tokens"], 1)
            self.assertNotIn("lora", template)
        self.assertEqual(templates[1]["timeout_seconds"], 0.5)
        for forbidden in ("--lora ", "--lora-scaled", "--llama-no-mmap",
                          "--disable-cuda-graphs"):
            self.assertNotIn(forbidden, unit)

    def test_startup_probes_bind_both_scenes_without_adapters(self):
        probes = load("startup_probes.json")
        release = probes["releases"]["traffic-industrial-joint-static-v2-q5km"]
        self.assertEqual(release["deployment_sha256"], MODEL_SHA)
        self.assertEqual(release["adapters"], [])
        self.assertEqual([item["prompt"][0] for item in release["static"]], ["T", "I"])
        self.assertEqual([item["expected_prompt_tokens"] for item in release["static"]], [17, 17])


if __name__ == "__main__":
    unittest.main()
