"""在 Jetson 本机独立采集当前交通 Edge-Qwen 的单次推理内存证据。

本脚本只生成比赛第 3 项证据，不运行云端 9B，也不生成能力或 TTFT 结果。
正式执行时会从本机内核信息确认当前主机是 aarch64 Jetson；服务器上的
内存数据即使字段齐全，也不能通过统一门禁。
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCENE_ROOT = Path(__file__).resolve().parents[1]
if str(SCENE_ROOT) not in sys.path:
    sys.path.insert(0, str(SCENE_ROOT))

from traffic_system.benchmark_utils import (
    ProcessMemorySampler,
    stop_server,
    wait_until_ready,
)
from traffic_system.eval_general_capability_retention import task_prompt
from traffic_system.measure_current_edge_llm_targets import (
    REPOSITORY_ROOT,
    MeasurementContractError,
    _assert_loopback,
    _git_state,
    _openai_stream_chat,
    _sha256,
    _validate_dataset,
    _write_json,
    validate_edge_model_asset,
)


MEMORY_SEMANTICS = "peak process-tree VmRSS during measured inference window"
MEASUREMENT_SCOPE = "jetson_edge_single_inference"


def detect_jetson_hardware(
    *,
    model_path: Path = Path("/proc/device-tree/model"),
    machine: Optional[str] = None,
    hostname: Optional[str] = None,
) -> Dict[str, str]:
    """Return kernel-derived Jetson identity or reject a non-Jetson host."""

    observed_machine = machine or platform.machine()
    observed_hostname = hostname or platform.node()
    try:
        device_model = model_path.read_bytes().rstrip(b"\x00").decode(
            "utf-8", errors="replace"
        ).strip()
    except OSError as exc:
        raise MeasurementContractError(
            "formal memory evidence must run on a Jetson with /proc/device-tree/model"
        ) from exc
    normalized_model = device_model.lower()
    is_nvidia_jetson_family = "nvidia" in normalized_model and any(
        family in normalized_model for family in ("jetson", "orin", "xavier")
    )
    if observed_machine != "aarch64" or not is_nvidia_jetson_family:
        raise MeasurementContractError(
            "formal memory evidence requires an aarch64 NVIDIA Jetson host"
        )
    return {
        "role": "edge_device",
        "platform": "nvidia_jetson",
        "machine": observed_machine,
        "device_model": device_model,
        "hostname": observed_hostname,
    }


def build_memory_evidence(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the strict target-3 document from per-request Jetson samples."""

    hardware = raw.get("hardware_id")
    if not isinstance(hardware, Mapping):
        raise MeasurementContractError(
            "memory hardware_id must be a Jetson identity object"
        )
    if (
        hardware.get("role") != "edge_device"
        or hardware.get("platform") != "nvidia_jetson"
        or hardware.get("machine") != "aarch64"
        or "nvidia"
        not in str(hardware.get("device_model", "")).lower()
        or not any(
            family in str(hardware.get("device_model", "")).lower()
            for family in ("jetson", "orin", "xavier")
        )
    ):
        raise MeasurementContractError(
            "memory evidence is not attested by a Jetson edge host"
        )
    samples = raw.get("memory_samples")
    if not isinstance(samples, list) or not samples:
        raise MeasurementContractError("memory_samples must be a non-empty list")
    rss_values: List[float] = []
    pss_values: List[float] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise MeasurementContractError(
                "memory_samples[{}] must be an object".format(index)
            )
        rss = float(sample.get("peak_process_tree_rss_mb", 0.0))
        pss = float(sample.get("peak_process_tree_pss_mb", 0.0))
        if rss <= 0.0 or pss < 0.0:
            raise MeasurementContractError(
                "memory samples require positive RSS and nonnegative PSS"
            )
        rss_values.append(rss)
        pss_values.append(pss)
    return {
        "provenance": {
            "git_commit": raw["git_commit"],
            "dataset_id": raw["dataset_id"],
            "hardware_id": dict(hardware),
            "model_ids": [raw["edge_model_id"]],
            "metric_semantics": "single_inference_peak_memory_mb",
            "run_id": raw["run_id"],
            "generated_at": raw["generated_at"],
        },
        "measurement_scope": MEASUREMENT_SCOPE,
        "measurement_host": dict(hardware),
        "measurement_window": {
            "measured_requests_per_window": 1,
            "window_count": len(samples),
            "sampler_interval_ms": float(raw["sampler_interval_ms"]),
            "warmup_requests": int(raw["warmup_requests"]),
        },
        "peak_memory_mb": max(rss_values),
        "peak_pss_mb": max(pss_values),
        "memory_semantics": MEMORY_SEMANTICS,
        "sample_count": len(samples),
    }


def _manifest_fragment(evidence_path: Path, evidence: Mapping[str, Any]) -> Dict[str, Any]:
    expected = dict(evidence["provenance"])
    expected.pop("run_id", None)
    expected.pop("generated_at", None)
    return {
        "targets": {
            "single_inference_memory": {
                "evidence": {
                    "path": evidence_path.name,
                    "sha256": _sha256(evidence_path),
                },
                "expected": expected,
                "criteria": {"maximum_peak_memory_mb": 1536},
            }
        }
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 Jetson 上独立实测 Edge-Qwen 内存")
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--edge-model", required=True)
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--dataset-metadata", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--edge-host", default="127.0.0.1")
    parser.add_argument("--edge-port", type=int, default=18292)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measurement-count", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--startup-timeout", type=int, default=120)
    parser.add_argument("--sample-interval-ms", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    _assert_loopback("http://{}:{}".format(args.edge_host, args.edge_port), "edge-host")
    if args.measurement_count <= 0:
        raise MeasurementContractError("measurement-count must be positive")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise MeasurementContractError("output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, dataset_sha = _validate_dataset(
        Path(args.dataset_jsonl).resolve(), Path(args.dataset_metadata).resolve()
    )
    edge_model = Path(args.edge_model).resolve()
    model_hashes = validate_edge_model_asset(edge_model)
    hardware_id = detect_jetson_hardware()
    git_commit, dirty = _git_state()
    if dirty:
        raise MeasurementContractError("formal evidence requires a clean Git worktree")

    edge_binary = Path(args.llama_server).resolve()
    edge_base_url = "http://{}:{}".format(args.edge_host, args.edge_port)
    command = [
        str(edge_binary),
        "-m",
        str(edge_model),
        "--host",
        args.edge_host,
        "--port",
        str(args.edge_port),
        "--ctx-size",
        str(args.ctx_size),
        "--threads",
        str(args.threads),
        "--threads-batch",
        str(args.threads),
        "--parallel",
        "1",
        "--gpu-layers",
        str(args.gpu_layers),
        "--cache-ram",
        "0",
        "--ctx-checkpoints",
        "0",
        "--no-cache-prompt",
        "--no-cache-idle-slots",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
    ]
    samples: List[Dict[str, Any]] = []
    log_tail = ""
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=str(REPOSITORY_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_until_ready(edge_base_url, process, args.startup_timeout)
            for index in range(args.warmup):
                row = rows[index % len(rows)]
                system_prompt, user_prompt, max_tokens = task_prompt(row)
                _openai_stream_chat(
                    edge_base_url,
                    "current-edge-qwen",
                    system_prompt,
                    user_prompt,
                    max_tokens,
                    args.timeout,
                )
            for index in range(args.measurement_count):
                row = rows[index % len(rows)]
                system_prompt, user_prompt, max_tokens = task_prompt(row)
                sampler = ProcessMemorySampler(
                    process.pid, args.sample_interval_ms / 1000.0
                )
                sampler.start()
                try:
                    response = _openai_stream_chat(
                        edge_base_url,
                        "current-edge-qwen",
                        system_prompt,
                        user_prompt,
                        max_tokens,
                        args.timeout,
                    )
                finally:
                    sampler.stop()
                samples.append(
                    {
                        "sample_id": row["sample_id"],
                        "category": row["category"],
                        "ttft_ms": response["ttft_ms"],
                        "wall_time_ms": response["wall_time_ms"],
                        "peak_process_tree_rss_mb": sampler.peak_mb,
                        "peak_process_tree_pss_mb": sampler.peak_pss_mb,
                    }
                )
        finally:
            stop_server(process)
            log.seek(0)
            log_tail = "\n".join(log.read().splitlines()[-40:])

    generated_at = datetime.now(timezone.utc).astimezone().isoformat()
    raw = {
        "task": "current_traffic_edge_llm_target_3_jetson",
        "git_commit": git_commit,
        "worktree_dirty": dirty,
        "run_id": args.run_id,
        "generated_at": generated_at,
        "dataset_id": "general_capability_eval_v1@sha256:" + dataset_sha,
        "dataset_sha256": dataset_sha,
        "hardware_id": hardware_id,
        "edge_model_id": "gguf:{}@sha256:{}".format(
            edge_model.name, model_hashes["edge_asset_sha256"]
        ),
        "model_hashes": model_hashes,
        "llama_server_sha256": _sha256(edge_binary),
        "llama_server_command": command,
        "sampler_interval_ms": args.sample_interval_ms,
        "warmup_requests": args.warmup,
        "memory_samples": samples,
        "edge_server_log_tail": log_tail,
    }
    raw_path = output_dir / "target_3_jetson_raw.json"
    _write_json(raw_path, raw)
    evidence = build_memory_evidence(raw)
    evidence["raw_measurement"] = {"path": raw_path.name, "sha256": _sha256(raw_path)}
    evidence_path = output_dir / "single_inference_memory.json"
    _write_json(evidence_path, evidence)
    fragment = _manifest_fragment(evidence_path, evidence)
    _write_json(output_dir / "target_3_manifest_fragment.json", fragment)
    print(json.dumps(fragment, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
