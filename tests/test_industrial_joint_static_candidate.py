import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "scenes"
    / "industrial_anomaly"
    / "deployment"
    / "joint_static_candidate_v3"
)


def read_json(name: str):
    return json.loads((CANDIDATE / name).read_text(encoding="utf-8"))


class IndustrialJointStaticCandidateTests(unittest.TestCase):
    def test_candidate_manifest_covers_and_verifies_every_file(self) -> None:
        rows = []
        for line in (CANDIDATE / "SHA256SUMS.txt").read_text(
            encoding="utf-8"
        ).splitlines():
            digest, name = line.split("  ", 1)
            rows.append((digest, name))
        expected = {
            path.name
            for path in CANDIDATE.iterdir()
            if path.is_file() and path.name != "SHA256SUMS.txt"
        }
        self.assertEqual({name for _, name in rows}, expected)
        for digest, name in rows:
            self.assertEqual(
                hashlib.sha256((CANDIDATE / name).read_bytes()).hexdigest(),
                digest,
                name,
            )

    def test_identity_binds_passed_sidecar_and_static_q5_package(self) -> None:
        identity = read_json("candidate_identity.json")
        self.assertEqual(
            identity["release_id"],
            "traffic-industrial-joint-static-v2-q5km",
        )
        self.assertEqual(
            identity["status"],
            "package_validated_sidecar_passed_formal_not_promoted",
        )
        self.assertEqual(identity["deployment"]["runtime_adapters"], [])
        self.assertEqual(identity["runtime_contract"]["runtime_lora_count"], 0)
        self.assertFalse(identity["runtime_contract"]["request_level_lora"])
        self.assertEqual(identity["runtime_contract"]["input_tokens"], 17)
        self.assertEqual(identity["runtime_contract"]["output_tokens"], 1)
        self.assertEqual(
            identity["adapter_package"]["manifest_sha256"],
            "504eda480843e3e0540d8495bbab880b66a61651cbc7eb9a177e5f43e7cfc24c",
        )
        sidecar = identity["sidecar_validation"]
        self.assertEqual(sidecar["gate_exit"], 0)
        self.assertEqual(sidecar["restore_exit"], 0)
        self.assertTrue(sidecar["traffic_smoke_passed"])
        self.assertTrue(sidecar["industrial_rgb_smoke_passed"])
        self.assertTrue(sidecar["industrial_infrared_smoke_passed"])
        self.assertLess(sidecar["industrial_selected_mean_latency_ms"], 200.0)
        self.assertFalse(sidecar["authoritative_final_claimed"])

    def test_runtime_outputs_and_plugins_are_one_consistent_v3_generation(self) -> None:
        identity = read_json("candidate_identity.json")
        deployment_sha = identity["deployment"]["sha256"]
        model_path = identity["deployment"]["path"]
        service = read_json("edge_service.json")
        self.assertEqual(service["listen"], {
            "host": "127.0.0.1",
            "port": 19391,
            "max_body_bytes": 8388608,
            "access_log": False,
        })
        self.assertIn("candidate_v3", service["plugin_config"])
        self.assertIn("candidate_v3", service["release_watch"]["registry"])

        for scene in ("traffic", "industrial"):
            runtime = read_json(f"edge_llm_runtime_{scene}_template.json")
            output = read_json(f"runtime_output_{scene}.json")
            self.assertEqual(runtime["model"], model_path)
            self.assertEqual(runtime["generation"]["max_input_tokens"], 17)
            self.assertEqual(runtime["generation"]["max_output_tokens"], 1)
            self.assertNotIn("lora_adapter", runtime)
            self.assertEqual(output["adapter_mode"], "static")
            self.assertEqual(output["static_deployment_sha256"], deployment_sha)
            self.assertIn("candidate-v3", output["name"])
            self.assertIn("candidate_v3", output["output"])

        plugins = read_json("scene_plugins_edge.json")["plugins"]
        self.assertEqual(len(plugins), 2)
        by_spec = {row["spec"]: row["options"] for row in plugins}
        traffic = by_spec["freeway_traffic_full.plugin_impl:TrafficPlugin"]
        industrial = by_spec["industrial_anomaly.plugin:IndustrialAnomalyPlugin"]
        self.assertEqual(traffic["edge_llm_prompt_prefix"], "T")
        self.assertEqual(industrial["edge_llm_prompt_prefix"], "I")
        self.assertEqual(
            industrial["edge_llm_selective_timeout_limit_seconds"], 0.25
        )
        self.assertIn("candidate_v3", traffic["edge_llm_runtime_config_path"])
        self.assertIn("candidate_v3", industrial["edge_llm_runtime_config_path"])

    def test_startup_probes_cover_both_scene_prefixes_without_adapters(self) -> None:
        identity = read_json("candidate_identity.json")
        release = read_json("startup_probes.json")["releases"][
            identity["release_id"]
        ]
        self.assertEqual(release["deployment_sha256"], identity["deployment"]["sha256"])
        self.assertEqual(release["adapters"], [])
        probes = release["static"]
        self.assertEqual([probe["prompt"][0] for probe in probes], ["T", "I"])
        self.assertEqual([probe["expected_prompt_tokens"] for probe in probes], [17, 17])
        self.assertEqual([probe["expected_token"] for probe in probes], ["F", "A"])


if __name__ == "__main__":
    unittest.main()
