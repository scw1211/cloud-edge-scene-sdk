#!/usr/bin/env python3
"""Safely exercise two request-selected LoRAs on one local llama-server.

The measured order is deterministic: traffic, industrial, traffic, industrial,
until ``--per-scene`` requests have been completed for each scene.  The script
does not start, stop, or reconfigure llama-server.  It only reads ``/proc`` and
uses the two supplied runtime configuration files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCENE_ROOT = Path(__file__).resolve().parent
SDK_ROOT = SCENE_ROOT.parents[1]
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from edge_llm_factory.runtime import ConfiguredActionClient


KIB_PER_MIB = 1024.0
TRAFFIC_TOKENS = "ABCDEF"
INDUSTRIAL_TOKENS = "ABC"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_identity(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("asset must be a regular non-symlink file: {}".format(resolved))
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Iterable[float]) -> Dict[str, Any]:
    items = [float(value) for value in values]
    if not items:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(items),
        "mean": round(statistics.fmean(items), 6),
        "p50": round(_percentile(items, 50), 6),
        "p95": round(_percentile(items, 95), 6),
        "max": round(max(items), 6),
    }


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 6) if denominator else None


def _read_json_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return dict(value)


def _rows(path: Path, allowed: str, limit: int) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line)
            messages = value.get("messages") if isinstance(value, dict) else None
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError("{}:{} has no user/assistant messages".format(path, line_number))
            if not isinstance(messages[0], dict) or not isinstance(messages[1], dict):
                raise ValueError("{}:{} messages must be JSON objects".format(path, line_number))
            prompt = str(messages[0].get("content", ""))
            target = str(messages[1].get("content", ""))
            if len(prompt) != 16 or target not in allowed:
                raise ValueError(
                    "{}:{} violates the 16-input/one-action-token contract".format(
                        path, line_number
                    )
                )
            result.append((prompt, target))
            if len(result) == limit:
                break
    if len(result) != limit:
        raise ValueError("{} has fewer than {} valid rows".format(path, limit))
    return result


def _parse_kib_file(path: Path) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw_line:
            continue
        name, raw_value = raw_line.split(":", 1)
        parts = raw_value.strip().split()
        if not parts:
            continue
        try:
            values[name] = int(parts[0])
        except ValueError:
            continue
    return values


def _parse_counter_file(path: Path) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) != 2:
            continue
        try:
            values[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return values


def _required(values: Mapping[str, int], names: Sequence[str], source: Path) -> None:
    missing = [name for name in names if name not in values]
    if missing:
        raise RuntimeError("{} lacks fields: {}".format(source, ", ".join(missing)))


class ProcResourceSampler:
    """Read-only sampler for host memory and one explicitly identified process."""

    def __init__(self, llama_pid: int, proc_root: Path = Path("/proc")) -> None:
        if isinstance(llama_pid, bool) or int(llama_pid) <= 1:
            raise ValueError("llama_pid must be an explicit process id greater than 1")
        self.llama_pid = int(llama_pid)
        self.proc_root = Path(proc_root)

    def sample(self, request_count: int) -> Dict[str, Any]:
        meminfo_path = self.proc_root / "meminfo"
        vmstat_path = self.proc_root / "vmstat"
        status_path = self.proc_root / str(self.llama_pid) / "status"
        meminfo = _parse_kib_file(meminfo_path)
        vmstat = _parse_counter_file(vmstat_path)
        status = _parse_kib_file(status_path)
        _required(meminfo, ("MemAvailable", "SwapTotal", "SwapFree"), meminfo_path)
        _required(vmstat, ("pgpgin", "pswpin"), vmstat_path)
        _required(status, ("VmRSS", "VmSwap"), status_path)
        state = ""
        process_name = ""
        for raw_line in status_path.read_text(encoding="utf-8").splitlines():
            if raw_line.startswith("State:"):
                state = raw_line.split(":", 1)[1].strip()
            elif raw_line.startswith("Name:"):
                process_name = raw_line.split(":", 1)[1].strip()
        if not state:
            raise RuntimeError("{} lacks State".format(status_path))
        if "llama" not in process_name.lower():
            raise RuntimeError(
                "explicit pid {} is {!r}, not a llama process".format(
                    self.llama_pid, process_name
                )
            )
        return {
            "captured_at": _utc_now(),
            "request_count": int(request_count),
            "mem_available_kib": meminfo["MemAvailable"],
            "swap_total_kib": meminfo["SwapTotal"],
            "swap_free_kib": meminfo["SwapFree"],
            "swap_used_kib": meminfo["SwapTotal"] - meminfo["SwapFree"],
            "pgpgin_counter": vmstat["pgpgin"],
            "pswpin_pages": vmstat["pswpin"],
            "llama_pid": self.llama_pid,
            "llama_name": process_name,
            "llama_state": state,
            "llama_vmrss_kib": status["VmRSS"],
            "llama_vmswap_kib": status["VmSwap"],
            "llama_vmhwm_kib": status.get("VmHWM"),
        }


def _load_client(
    path: Path,
    timeout_override_seconds: Optional[float],
) -> Tuple[ConfiguredActionClient, Dict[str, Any]]:
    raw = _read_json_object(path)
    original_timeout = raw.get("timeout_seconds", 5.0)
    if timeout_override_seconds is not None:
        if not math.isfinite(timeout_override_seconds) or timeout_override_seconds <= 0:
            raise ValueError("benchmark timeout override must be a positive finite number")
        # Deliberately modify only the in-memory copy.  Runtime files remain immutable.
        raw["timeout_seconds"] = float(timeout_override_seconds)
    client = ConfiguredActionClient.from_config(raw)
    return client, {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "original_timeout_seconds": float(original_timeout),
        "effective_timeout_seconds": float(client.describe()["timeout_seconds"]),
        "benchmark_only_timeout_override": timeout_override_seconds is not None,
        "runtime_file_modified": False,
        "resolved_runtime": client.describe(),
    }


def _stop_for_snapshot(
    snapshot: Mapping[str, Any], min_mem_available_kib: int
) -> Optional[Dict[str, Any]]:
    if int(snapshot["mem_available_kib"]) < int(min_mem_available_kib):
        return {
            "code": "mem_available_below_threshold",
            "detail": "MemAvailable {} KiB is below the {} KiB safety floor".format(
                snapshot["mem_available_kib"], min_mem_available_kib
            ),
            "request_count": int(snapshot["request_count"]),
        }
    state = str(snapshot.get("llama_state", ""))
    if state.startswith("Z") or state.startswith("X"):
        return {
            "code": "llama_process_not_runnable",
            "detail": "llama process state is {!r}".format(state),
            "request_count": int(snapshot["request_count"]),
        }
    return None


def _scene_summary(
    records: Sequence[Mapping[str, Any]], scene: str, requested: int
) -> Dict[str, Any]:
    selected = [record for record in records if record["scene"] == scene]
    correct = sum(bool(record["correct"]) for record in selected)
    valid = sum(bool(record["valid_output"]) for record in selected)
    return {
        "requested": int(requested),
        "completed": len(selected),
        "completion_rate": _rate(len(selected), requested),
        "correct": correct,
        "accuracy_on_completed": _rate(correct, len(selected)),
        "valid_outputs": valid,
        "valid_output_rate_on_completed": _rate(valid, len(selected)),
        "latency_ms": _summary(record["latency_ms"] for record in selected),
        "prompt_token_contract_rate": _rate(
            sum(record.get("prompt_tokens") == 16 for record in selected), len(selected)
        ),
        "output_token_contract_rate": _rate(
            sum(record.get("output_tokens") == 1 for record in selected), len(selected)
        ),
    }


def _resource_summary(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {
            "sample_count": 0,
            "baseline": None,
            "final": None,
            "growth": None,
        }
    first = samples[0]
    last = samples[-1]

    def delta(name: str) -> int:
        return int(last[name]) - int(first[name])

    return {
        "sample_count": len(samples),
        "baseline": dict(first),
        "final": dict(last),
        "minimum_mem_available_mib": round(
            min(int(row["mem_available_kib"]) for row in samples) / KIB_PER_MIB, 6
        ),
        "peak_system_swap_used_mib": round(
            max(int(row["swap_used_kib"]) for row in samples) / KIB_PER_MIB, 6
        ),
        "peak_llama_rss_mib": round(
            max(int(row["llama_vmrss_kib"]) for row in samples) / KIB_PER_MIB, 6
        ),
        "peak_llama_vmswap_mib": round(
            max(int(row["llama_vmswap_kib"]) for row in samples) / KIB_PER_MIB, 6
        ),
        "growth": {
            "system_swap_used_mib": round(delta("swap_used_kib") / KIB_PER_MIB, 6),
            "llama_rss_mib": round(delta("llama_vmrss_kib") / KIB_PER_MIB, 6),
            "llama_vmswap_mib": round(delta("llama_vmswap_kib") / KIB_PER_MIB, 6),
            "pgpgin_counter_delta": delta("pgpgin_counter"),
            "pswpin_pages_delta": delta("pswpin_pages"),
        },
    }


def run_benchmark(
    traffic_rows: Sequence[Tuple[str, str]],
    industrial_rows: Sequence[Tuple[str, str]],
    traffic_client: ConfiguredActionClient,
    industrial_client: ConfiguredActionClient,
    sampler: ProcResourceSampler,
    sample_every: int,
    min_mem_available_mib: float,
    runtime_evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    if len(traffic_rows) != len(industrial_rows) or not traffic_rows:
        raise ValueError("traffic and industrial rows must have the same non-zero length")
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")
    if not math.isfinite(min_mem_available_mib) or min_mem_available_mib <= 0:
        raise ValueError("min_mem_available_mib must be a positive finite number")
    traffic_description = traffic_client.describe()
    industrial_description = industrial_client.describe()
    traffic_adapter = traffic_description.get("lora_adapter")
    industrial_adapter = industrial_description.get("lora_adapter")
    if not isinstance(traffic_adapter, dict) or not isinstance(industrial_adapter, dict):
        raise ValueError("both benchmark runtimes must select an explicit LoRA adapter")
    if traffic_adapter.get("id") == industrial_adapter.get("id"):
        raise ValueError("traffic and industrial runtimes must select distinct LoRA ids")
    for shared_field in ("endpoint", "model"):
        traffic_value = traffic_description.get(shared_field)
        industrial_value = industrial_description.get(shared_field)
        if traffic_value != industrial_value:
            raise ValueError(
                "both runtimes must use one shared llama-server; {} differs".format(
                    shared_field
                )
            )
    per_scene = len(traffic_rows)
    min_mem_available_kib = int(math.ceil(min_mem_available_mib * KIB_PER_MIB))
    definitions: List[Tuple[str, ConfiguredActionClient, str, str, Dict[str, str], str]] = []
    for index in range(per_scene):
        definitions.append(
            (
                "traffic",
                traffic_client,
                traffic_rows[index][0],
                traffic_rows[index][1],
                {token: token for token in TRAFFIC_TOKENS},
                TRAFFIC_TOKENS,
            )
        )
        definitions.append(
            (
                "industrial",
                industrial_client,
                industrial_rows[index][0],
                industrial_rows[index][1],
                {"normal": "A", "review": "B", "anomaly": "C"},
                INDUSTRIAL_TOKENS,
            )
        )

    records: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    attempted_requests = 0
    stop_reason: Optional[Dict[str, Any]] = None

    try:
        initial = sampler.sample(0)
        samples.append(initial)
        stop_reason = _stop_for_snapshot(initial, min_mem_available_kib)
    except Exception as exc:  # no model request is safe without a baseline
        stop_reason = {
            "code": "resource_sampling_failed",
            "detail": "{}: {}".format(type(exc).__name__, exc),
            "request_count": 0,
        }

    if stop_reason is None:
        for sequence, definition in enumerate(definitions, start=1):
            scene, client, prompt, target, mapping, allowed = definition
            attempted_requests = sequence
            try:
                result = client.predict(prompt, mapping)
                prediction = str(result.get("token", ""))
                records.append(
                    {
                        "sequence": sequence,
                        "scene": scene,
                        "prompt": prompt,
                        "target": target,
                        "prediction": prediction,
                        "correct": prediction == target,
                        "valid_output": prediction in allowed,
                        "latency_ms": float(result["latency_ms"]),
                        "prompt_tokens": result.get("prompt_tokens"),
                        "output_tokens": result.get("output_tokens"),
                        "lora_id": client.describe()["lora_adapter"]["id"],
                    }
                )
            except Exception as exc:
                error = {
                    "sequence": sequence,
                    "scene": scene,
                    "exception_type": type(exc).__name__,
                    "detail": str(exc),
                }
                errors.append(error)
                stop_reason = {
                    "code": "model_request_failed",
                    "detail": "{} request {} failed: {}".format(scene, sequence, exc),
                    "request_count": attempted_requests,
                }
                break

            if attempted_requests % sample_every == 0 or attempted_requests == len(definitions):
                try:
                    snapshot = sampler.sample(attempted_requests)
                    samples.append(snapshot)
                    stop_reason = _stop_for_snapshot(snapshot, min_mem_available_kib)
                except Exception as exc:
                    stop_reason = {
                        "code": "resource_sampling_failed",
                        "detail": "{}: {}".format(type(exc).__name__, exc),
                        "request_count": attempted_requests,
                    }
                if stop_reason is not None:
                    break

    if samples and samples[-1]["request_count"] != attempted_requests:
        try:
            final_sample = sampler.sample(attempted_requests)
            samples.append(final_sample)
            threshold_reason = _stop_for_snapshot(final_sample, min_mem_available_kib)
            if stop_reason is None and threshold_reason is not None:
                stop_reason = threshold_reason
        except Exception as exc:
            if stop_reason is None:
                stop_reason = {
                    "code": "resource_sampling_failed",
                    "detail": "{}: {}".format(type(exc).__name__, exc),
                    "request_count": attempted_requests,
                }

    completed = len(records) == len(definitions) and stop_reason is None
    if completed:
        stop_reason = {
            "code": "completed",
            "detail": "all deterministic alternating requests completed",
            "request_count": len(records),
        }
    report = {
        "schema_version": "multi-adapter-alternating-stability/v1",
        "created_at": _utc_now(),
        "status": "completed" if completed else "stopped",
        "benchmark_contract": {
            "order": "strict_traffic_then_industrial_alternation",
            "per_scene_requested": per_scene,
            "total_requested": per_scene * 2,
            "resource_sample_every_requests": int(sample_every),
            "minimum_mem_available_mib": float(min_mem_available_mib),
            "local_proc_read_only": True,
            "server_lifecycle_modified": False,
            "shared_endpoint": traffic_description.get("endpoint"),
            "shared_model": traffic_description.get("model"),
            "traffic_lora_id": traffic_adapter["id"],
            "industrial_lora_id": industrial_adapter["id"],
        },
        "runtime_evidence": dict(runtime_evidence),
        "attempted_requests": attempted_requests,
        "completed_requests": len(records),
        "stop_reason": stop_reason,
        "summary": {
            "overall": {
                "requested": len(definitions),
                "completed": len(records),
                "completion_rate": _rate(len(records), len(definitions)),
                "accuracy_on_completed": _rate(
                    sum(bool(record["correct"]) for record in records), len(records)
                ),
                "valid_output_rate_on_completed": _rate(
                    sum(bool(record["valid_output"]) for record in records), len(records)
                ),
                "latency_ms": _summary(record["latency_ms"] for record in records),
            },
            "traffic": _scene_summary(records, "traffic", per_scene),
            "industrial": _scene_summary(records, "industrial", per_scene),
            "resources": _resource_summary(samples),
        },
        "resource_samples": samples,
        "errors": errors,
        "records": records,
    }
    return report


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file_obj:
            json.dump(value, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary_name, str(path))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a guarded 250+250 alternating traffic/industrial LoRA test"
    )
    parser.add_argument("--traffic-runtime", required=True)
    parser.add_argument("--industrial-runtime", required=True)
    parser.add_argument("--traffic-jsonl", required=True)
    parser.add_argument("--industrial-jsonl", required=True)
    parser.add_argument("--llama-pid", type=int, required=True)
    parser.add_argument(
        "--base-gguf",
        required=True,
        help="exact base GGUF loaded by the measured llama-server",
    )
    parser.add_argument(
        "--runtime-adapter",
        action="append",
        required=True,
        help="ordered runtime LoRA path; pass traffic id 0 then industrial id 1",
    )
    parser.add_argument("--per-scene", type=int, default=250)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--min-mem-available-mib", type=float, default=256.0)
    parser.add_argument(
        "--benchmark-timeout-override-seconds",
        type=float,
        default=None,
        help=(
            "override both request timeouts only in this process; the report records "
            "the original/effective values and runtime files are never rewritten"
        ),
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.per_scene <= 0:
        raise ValueError("per-scene must be positive")
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    traffic_runtime_path = Path(args.traffic_runtime).resolve()
    industrial_runtime_path = Path(args.industrial_runtime).resolve()
    traffic_client, traffic_evidence = _load_client(
        traffic_runtime_path, args.benchmark_timeout_override_seconds
    )
    industrial_client, industrial_evidence = _load_client(
        industrial_runtime_path, args.benchmark_timeout_override_seconds
    )
    if len(args.runtime_adapter) != 2:
        raise ValueError("--runtime-adapter must be passed exactly twice in id order")
    base_asset = _asset_identity(Path(args.base_gguf))
    runtime_model = Path(traffic_client.describe()["model"]).expanduser().resolve()
    if runtime_model != Path(base_asset["path"]):
        raise ValueError(
            "--base-gguf does not match the model selected by both runtime configs"
        )
    adapter_assets = [
        {"id": adapter_id, **_asset_identity(Path(path))}
        for adapter_id, path in enumerate(args.runtime_adapter)
    ]
    report = run_benchmark(
        _rows(Path(args.traffic_jsonl), TRAFFIC_TOKENS, args.per_scene),
        _rows(Path(args.industrial_jsonl), INDUSTRIAL_TOKENS, args.per_scene),
        traffic_client,
        industrial_client,
        ProcResourceSampler(args.llama_pid),
        args.sample_every,
        args.min_mem_available_mib,
        {
            "traffic": traffic_evidence,
            "industrial": industrial_evidence,
            "timeout_override_note": (
                "A benchmark-only in-memory timeout override was applied; committed "
                "and deployed runtime configuration files were not modified."
                if args.benchmark_timeout_override_seconds is not None
                else "No benchmark timeout override was applied."
            ),
        },
    )
    report["dataset_evidence"] = {
        "traffic": {
            "path": str(Path(args.traffic_jsonl).resolve()),
            "sha256": _sha256(Path(args.traffic_jsonl)),
            "rows_consumed": args.per_scene,
        },
        "industrial": {
            "path": str(Path(args.industrial_jsonl).resolve()),
            "sha256": _sha256(Path(args.industrial_jsonl)),
            "rows_consumed": args.per_scene,
        },
    }
    report["asset_evidence"] = {
        "base_q6": base_asset,
        "runtime_adapters": adapter_assets,
    }
    _atomic_write_json(output_path, report)
    printable = dict(report)
    printable.pop("records", None)
    printable.pop("resource_samples", None)
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
