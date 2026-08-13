import json
from pathlib import Path
import tempfile
import unittest

from scenes.industrial_anomaly import benchmark_multi_adapter_stability as benchmark


class _StubClient:
    def __init__(self, lora_id, predictions):
        self.lora_id = lora_id
        self.predictions = dict(predictions)

    def describe(self):
        return {
            "endpoint": "http://127.0.0.1:18190",
            "model": "/models/base.gguf",
            "timeout_seconds": 0.5,
            "lora_adapter": {"id": self.lora_id, "scale": 1.0},
        }

    def predict(self, prompt, mapping):
        return {
            "token": self.predictions[prompt],
            "latency_ms": 10.0 + self.lora_id,
            "prompt_tokens": 16,
            "output_tokens": 1,
        }


def _snapshot(request_count, mem_available_kib=1024 * 1024, rss_kib=600 * 1024):
    return {
        "captured_at": "2026-08-12T00:00:00Z",
        "request_count": request_count,
        "mem_available_kib": mem_available_kib,
        "swap_total_kib": 2 * 1024 * 1024,
        "swap_free_kib": 2 * 1024 * 1024,
        "swap_used_kib": 0,
        "pgpgin_counter": 1000 + request_count,
        "pswpin_pages": 0,
        "llama_pid": 1234,
        "llama_state": "S (sleeping)",
        "llama_vmrss_kib": rss_kib,
        "llama_vmswap_kib": 0,
        "llama_vmhwm_kib": rss_kib,
    }


class _StubSampler:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.calls = []

    def sample(self, request_count):
        self.calls.append(request_count)
        return _snapshot(
            request_count,
            mem_available_kib=self.values.get(request_count, 1024 * 1024),
            rss_kib=600 * 1024 + request_count,
        )


class MultiAdapterStabilityBenchmarkTests(unittest.TestCase):
    def test_timeout_override_is_in_memory_and_reported(self):
        runtime = {
            "schema_version": "edge-llm-runtime/v1",
            "provider": "llama_cpp",
            "endpoint": "http://127.0.0.1:18190",
            "model": "/models/base.gguf",
            "timeout_seconds": 0.18,
            "generation": {
                "max_input_tokens": 16,
                "max_output_tokens": 1,
                "temperature": 0,
                "top_p": 1,
                "seed": 42,
                "thinking": False,
                "keep_alive": "30m",
            },
            "authentication": {"api_key_env": ""},
            "lora_adapter": {"id": 1, "scale": 1.0},
        }
        with tempfile.TemporaryDirectory(prefix="adapter-stability-runtime-") as directory:
            path = Path(directory) / "runtime.json"
            original = json.dumps(runtime, sort_keys=True)
            path.write_text(original, encoding="utf-8")

            client, evidence = benchmark._load_client(path, 2.5)

            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(client.describe()["timeout_seconds"], 2.5)
            self.assertEqual(evidence["original_timeout_seconds"], 0.18)
            self.assertEqual(evidence["effective_timeout_seconds"], 2.5)
            self.assertTrue(evidence["benchmark_only_timeout_override"])
            self.assertFalse(evidence["runtime_file_modified"])

    def test_strict_alternation_and_complete_summary(self):
        traffic_rows = [("T" * 15 + str(index), token) for index, token in enumerate("AB")]
        industrial_rows = [("I" * 15 + str(index), token) for index, token in enumerate("BC")]
        traffic = _StubClient(0, dict(traffic_rows))
        industrial = _StubClient(1, dict(industrial_rows))
        sampler = _StubSampler()

        report = benchmark.run_benchmark(
            traffic_rows,
            industrial_rows,
            traffic,
            industrial,
            sampler,
            sample_every=2,
            min_mem_available_mib=256,
            runtime_evidence={"timeout_override_note": "none"},
        )

        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["completed_requests"], 4)
        self.assertEqual(
            [row["scene"] for row in report["records"]],
            ["traffic", "industrial", "traffic", "industrial"],
        )
        self.assertEqual([row["lora_id"] for row in report["records"]], [0, 1, 0, 1])
        self.assertEqual(report["summary"]["traffic"]["accuracy_on_completed"], 1.0)
        self.assertEqual(report["summary"]["industrial"]["valid_output_rate_on_completed"], 1.0)
        self.assertEqual(report["summary"]["overall"]["latency_ms"]["count"], 4)
        self.assertEqual(sampler.calls, [0, 2, 4])
        self.assertEqual(
            report["summary"]["resources"]["growth"]["llama_rss_mib"],
            round(4 / 1024, 6),
        )

    def test_low_memory_stops_before_next_sampling_block(self):
        traffic_rows = [("T" * 15 + str(index), "A") for index in range(3)]
        industrial_rows = [("I" * 15 + str(index), "B") for index in range(3)]
        traffic = _StubClient(0, {prompt: target for prompt, target in traffic_rows})
        industrial = _StubClient(1, {prompt: target for prompt, target in industrial_rows})
        sampler = _StubSampler({2: 200 * 1024})

        report = benchmark.run_benchmark(
            traffic_rows,
            industrial_rows,
            traffic,
            industrial,
            sampler,
            sample_every=2,
            min_mem_available_mib=256,
            runtime_evidence={},
        )

        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["completed_requests"], 2)
        self.assertEqual(
            report["stop_reason"]["code"], "mem_available_below_threshold"
        )
        self.assertEqual(sampler.calls, [0, 2])
        self.assertEqual(report["summary"]["overall"]["completion_rate"], 0.333333)

    def test_proc_sampler_and_atomic_json_use_only_explicit_paths(self):
        with tempfile.TemporaryDirectory(prefix="adapter-stability-proc-") as directory:
            root = Path(directory)
            (root / "4321").mkdir()
            (root / "meminfo").write_text(
                "MemAvailable: 524288 kB\nSwapTotal: 1048576 kB\nSwapFree: 786432 kB\n",
                encoding="utf-8",
            )
            (root / "vmstat").write_text("pgpgin 12345\npswpin 67\n", encoding="utf-8")
            (root / "4321/status").write_text(
                "Name:\tllama-server\nState:\tS (sleeping)\nVmHWM:\t700000 kB\n"
                "VmRSS:\t650000 kB\nVmSwap:\t12000 kB\n",
                encoding="utf-8",
            )
            sample = benchmark.ProcResourceSampler(4321, root).sample(20)
            output = root / "report.json"
            benchmark._atomic_write_json(output, {"sample": sample})

            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(persisted["sample"]["request_count"], 20)
            self.assertEqual(persisted["sample"]["swap_used_kib"], 262144)
            self.assertEqual(persisted["sample"]["llama_vmrss_kib"], 650000)
            self.assertEqual(persisted["sample"]["llama_vmswap_kib"], 12000)
            self.assertEqual(persisted["sample"]["pgpgin_counter"], 12345)
            self.assertEqual(persisted["sample"]["pswpin_pages"], 67)
            with self.assertRaises(FileExistsError):
                benchmark._atomic_write_json(output, {})


if __name__ == "__main__":
    unittest.main()
