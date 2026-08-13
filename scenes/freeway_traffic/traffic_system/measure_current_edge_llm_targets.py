"""Measure competition targets 1-2 for the *current* traffic Edge-Qwen asset.

This command deliberately does not discover or reuse historical result files.  It
starts llama-server from the GGUF pinned by ``asset_catalog.json``, evaluates that
process and the pinned Ollama 9B teacher on the same frozen prompt set, and emits
the capability-retention and TTFT evidence accepted by
``scripts/evaluate_competition_targets.py``.

Jetson memory is deliberately not measured here.  This command is normally run
on the GPU server so its RSS cannot be used as formal edge-memory evidence.  Use
``measure_current_edge_llm_memory.py`` on the Jetson for target 3.

The command is measurement-only: it neither imports an Ollama model nor changes a
deployed service.  The caller must provide the installed Ollama manifest/blob paths
so the teacher alias can be tied to the catalogued bytes rather than trusted by
name alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCENE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SCENE_ROOT.parents[1]
if str(SCENE_ROOT) not in sys.path:
    sys.path.insert(0, str(SCENE_ROOT))

from traffic_system.benchmark_utils import (  # noqa: E402
    stop_server,
    wait_until_ready,
)
from traffic_system.eval_general_capability_retention import (  # noqa: E402
    evaluate_response,
    model_summary,
    task_prompt,
    unload_model,
)


ASSET_CATALOG = SCENE_ROOT / "asset_catalog.json"
MODEL_CATALOG = REPOSITORY_ROOT / "model_bundle" / "catalog.json"
METRIC_SEMANTICS = {
    "capability_retention": "macro_capability_retention",
    "ttft_reduction": "same_protocol_ttft_reduction",
}
REQUIRED_CAPABILITY_CATEGORIES = frozenset(
    {"math", "code", "natural_language_reasoning"}
)


class MeasurementContractError(ValueError):
    """Raised when current-version provenance cannot be proven."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise MeasurementContractError("{} must contain a JSON object".format(path))
    return value


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise MeasurementContractError(
                    "{}:{} must contain a JSON object".format(path, line_number)
                )
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise MeasurementContractError("refusing to overwrite {}".format(path))
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_state() -> Tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPOSITORY_ROOT),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=str(REPOSITORY_ROOT),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _assert_loopback(url: str, field: str) -> None:
    hostname = urllib.parse.urlparse(url).hostname
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise MeasurementContractError(
            "{} must be loopback so both models run on the declared host".format(field)
        )


def _validate_dataset(dataset: Path, metadata: Path) -> Tuple[List[Dict[str, Any]], str]:
    rows = _read_jsonl(dataset)
    metadata_value = _read_json(metadata)
    expected_total = int(metadata_value.get("total_samples", 0))
    expected_counts = metadata_value.get("category_counts", {})
    counts: Dict[str, int] = {}
    sample_ids = []
    for row in rows:
        category = str(row.get("category", ""))
        sample_id = str(row.get("sample_id", ""))
        if not category or not sample_id:
            raise MeasurementContractError("dataset rows require category and sample_id")
        counts[category] = counts.get(category, 0) + 1
        sample_ids.append(sample_id)
    if set(counts) != REQUIRED_CAPABILITY_CATEGORIES:
        raise MeasurementContractError(
            "dataset categories must be exactly math, code, and "
            "natural_language_reasoning"
        )
    if any(counts[category] <= 0 for category in REQUIRED_CAPABILITY_CATEGORIES):
        raise MeasurementContractError("every required capability category needs samples")
    if (
        not isinstance(expected_counts, Mapping)
        or set(expected_counts) != REQUIRED_CAPABILITY_CATEGORIES
    ):
        raise MeasurementContractError(
            "metadata category_counts must contain exactly the three required categories"
        )
    normalized_expected_counts = {
        str(category): int(count) for category, count in expected_counts.items()
    }
    if len(rows) != expected_total or counts != normalized_expected_counts:
        raise MeasurementContractError("dataset does not match its frozen metadata")
    if len(sample_ids) != len(set(sample_ids)):
        raise MeasurementContractError("dataset sample_id values must be unique")
    return rows, _sha256(dataset)


def _catalog_contracts() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    asset = _read_json(ASSET_CATALOG)["downloaded_assets"]["edge_qwen_gguf"]
    teacher = _read_json(MODEL_CATALOG)["cloud_teacher"]
    return dict(asset), dict(teacher)


def validate_model_assets(
    edge_model: Path,
    teacher_manifest: Path,
    teacher_blob: Path,
) -> Dict[str, str]:
    edge_contract, teacher_contract = _catalog_contracts()
    expected_edge_name = Path(str(edge_contract["file"])).name
    if edge_model.name != expected_edge_name:
        raise MeasurementContractError(
            "edge model must be current catalog asset {}, got {}".format(
                expected_edge_name, edge_model.name
            )
        )
    observed = {
        "edge_asset_sha256": _sha256(edge_model),
        "teacher_manifest_sha256": _sha256(teacher_manifest),
        "teacher_blob_sha256": _sha256(teacher_blob),
    }
    expected = {
        "edge_asset_sha256": str(edge_contract["sha256"]),
        "teacher_manifest_sha256": str(teacher_contract["ollama_manifest_sha256"]),
        "teacher_blob_sha256": str(teacher_contract["model_blob_sha256"]),
    }
    if observed != expected:
        differences = [
            "{} expected {}, observed {}".format(key, expected[key], observed[key])
            for key in sorted(expected)
            if observed[key] != expected[key]
        ]
        raise MeasurementContractError("model identity mismatch: " + "; ".join(differences))
    return observed


def validate_edge_model_asset(edge_model: Path) -> Dict[str, str]:
    """Validate only the current edge GGUF, for the Jetson memory runner."""

    edge_contract, _ = _catalog_contracts()
    expected_edge_name = Path(str(edge_contract["file"])).name
    if edge_model.name != expected_edge_name:
        raise MeasurementContractError(
            "edge model must be current catalog asset {}, got {}".format(
                expected_edge_name, edge_model.name
            )
        )
    observed = _sha256(edge_model)
    expected = str(edge_contract["sha256"])
    if observed != expected:
        raise MeasurementContractError(
            "edge model identity mismatch: expected {}, observed {}".format(
                expected, observed
            )
        )
    return {"edge_asset_sha256": observed}


def _openai_stream_chat(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "max_tokens": max_tokens,
        "think": False,
        # Ollama's OpenAI-compatible endpoint ignores ``think`` for Qwen3.5,
        # while ``reasoning_effort=none`` is the supported non-thinking
        # control.  Keep both fields because llama-server accepts ``think``
        # and Ollama accepts ``reasoning_effort``; this gives the teacher and
        # edge model the same answer-only protocol instead of accidentally
        # spending the output budget on hidden reasoning tokens.
        "reasoning_effort": "none",
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_content_at: Optional[float] = None
    parts: List[str] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            encoded = line[5:].strip()
            if not encoded or encoded == "[DONE]":
                continue
            chunk = json.loads(encoded)
            choices = chunk.get("choices", [])
            if not choices:
                continue
            content = choices[0].get("delta", {}).get("content")
            if content:
                if first_content_at is None:
                    first_content_at = time.perf_counter()
                parts.append(str(content))
    finished = time.perf_counter()
    if first_content_at is None:
        raise MeasurementContractError("{} returned no content token".format(model))
    return {
        "text": "".join(parts).strip(),
        "ttft_ms": round((first_content_at - started) * 1000.0, 6),
        "wall_time_ms": round((finished - started) * 1000.0, 6),
    }


def _evaluate_model(
    rows: Sequence[Dict[str, Any]],
    base_url: str,
    model: str,
    timeout: int,
    code_timeout: float,
) -> List[Dict[str, Any]]:
    samples = []
    for row in rows:
        system_prompt, user_prompt, max_tokens = task_prompt(row)
        response = _openai_stream_chat(
            base_url, model, system_prompt, user_prompt, max_tokens, timeout
        )
        evaluated = evaluate_response(response["text"], row, code_timeout)
        samples.append(
            {
                "sample_id": row["sample_id"],
                "benchmark": row["benchmark"],
                "category": row["category"],
                "correct": bool(evaluated["correct"]),
                "prediction": evaluated.get("prediction"),
                "reference": evaluated.get("reference"),
                "execution_error": evaluated.get("execution_error"),
                "raw_output": response["text"],
                "ttft_ms": response["ttft_ms"],
                "wall_time_ms": response["wall_time_ms"],
            }
        )
    return samples


def _ollama_runtime_digest(host: str, model: str) -> str:
    with urllib.request.urlopen(host.rstrip("/") + "/api/ps", timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    expected_name = model[:-7] if model.endswith(":latest") else model
    for item in value.get("models", []):
        observed_name = str(item.get("name", ""))
        if observed_name.endswith(":latest"):
            observed_name = observed_name[:-7]
        if observed_name == expected_name:
            return str(item.get("digest", ""))
    return ""


def _mean(values: Iterable[float]) -> float:
    return round(statistics.fmean(float(value) for value in values), 6)


def _validate_raw_samples(
    raw: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dataset_ids = [str(value) for value in raw.get("dataset_sample_ids", [])]
    teacher = list(raw.get("teacher_samples", []))
    edge = list(raw.get("edge_samples", []))
    for label, samples in (("teacher", teacher), ("edge", edge)):
        ids = [str(sample.get("sample_id", "")) for sample in samples]
        if ids != dataset_ids:
            raise MeasurementContractError(
                "{} samples are incomplete, duplicated, or reordered".format(label)
            )
        if any(float(sample.get("ttft_ms", 0.0)) <= 0.0 for sample in samples):
            raise MeasurementContractError("{} TTFT samples must be positive".format(label))
    if not teacher:
        raise MeasurementContractError("measurement requires at least one sample")
    teacher_categories = [str(sample.get("category", "")) for sample in teacher]
    edge_categories = [str(sample.get("category", "")) for sample in edge]
    if teacher_categories != edge_categories:
        raise MeasurementContractError("teacher and edge category order must match")
    if set(teacher_categories) != REQUIRED_CAPABILITY_CATEGORIES:
        raise MeasurementContractError(
            "measured samples must contain math, code, and "
            "natural_language_reasoning"
        )
    return teacher, edge


def build_gate_evidence(raw: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert a server run into strict capability and TTFT documents only."""

    teacher, edge = _validate_raw_samples(raw)
    teacher_summary = model_summary(teacher)
    edge_summary = model_summary(edge)
    common = {
        "git_commit": raw["git_commit"],
        "dataset_id": raw["dataset_id"],
        "hardware_id": raw["hardware_id"],
        "run_id": raw["run_id"],
        "generated_at": raw["generated_at"],
    }
    teacher_id = raw["teacher_model_id"]
    edge_id = raw["edge_model_id"]
    capability = {
        "provenance": {
            **common,
            "model_ids": {"teacher": teacher_id, "edge": edge_id},
            "metric_semantics": METRIC_SEMANTICS["capability_retention"],
        },
        "teacher_macro_score": teacher_summary["overall_macro_score"],
        "edge_macro_score": edge_summary["overall_macro_score"],
        "category_scores": {
            category: {
                "teacher": teacher_summary["categories"][category]["score"],
                "edge": edge_summary["categories"][category]["score"],
                "sample_count": teacher_summary["categories"][category]["total"],
            }
            for category in sorted(teacher_summary["categories"])
        },
        "category_sample_counts": {
            category: teacher_summary["categories"][category]["total"]
            for category in sorted(teacher_summary["categories"])
        },
        "sample_count": len(teacher),
    }
    ttft = {
        "provenance": {
            **common,
            "model_ids": {"baseline": teacher_id, "edge": edge_id},
            "metric_semantics": METRIC_SEMANTICS["ttft_reduction"],
        },
        "baseline_ttft_ms": [float(sample["ttft_ms"]) for sample in teacher],
        "edge_ttft_ms": [float(sample["ttft_ms"]) for sample in edge],
        "paired_sample_ids": list(raw["dataset_sample_ids"]),
        "baseline_mean_ms": _mean(sample["ttft_ms"] for sample in teacher),
        "edge_mean_ms": _mean(sample["ttft_ms"] for sample in edge),
    }
    return {
        "capability_retention": capability,
        "ttft_reduction": ttft,
    }


def _manifest_fragment(
    evidence_paths: Mapping[str, Path], evidence: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    targets: Dict[str, Any] = {}
    criteria = {
        "capability_retention": {"minimum_retention_rate": 0.8},
        "ttft_reduction": {"minimum_reduction_rate": 0.75},
    }
    for target in evidence:
        targets[target] = {
            "evidence": {
                "path": evidence_paths[target].name,
                "sha256": _sha256(evidence_paths[target]),
            },
            "expected": dict(evidence[target]["provenance"]),
            "criteria": criteria[target],
        }
        for key in ("run_id", "generated_at"):
            targets[target]["expected"].pop(key, None)
    return {"targets": targets}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    edge_contract, teacher_contract = _catalog_contracts()
    parser = argparse.ArgumentParser(description="Measure current traffic Edge-Qwen targets 1-2")
    parser.add_argument("--llama-server", required=True)
    parser.add_argument(
        "--edge-model",
        default=str(SCENE_ROOT / str(edge_contract["file"])),
    )
    parser.add_argument("--teacher-model", default=str(teacher_contract["model"]))
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--teacher-blob", required=True)
    parser.add_argument("--teacher-host", default="http://127.0.0.1:11434")
    parser.add_argument("--edge-host", default="127.0.0.1")
    parser.add_argument("--edge-port", type=int, default=18291)
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--dataset-metadata", required=True)
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--startup-timeout", type=int, default=120)
    parser.add_argument("--code-timeout", type=float, default=3.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    _assert_loopback(args.teacher_host, "teacher-host")
    _assert_loopback("http://{}:{}".format(args.edge_host, args.edge_port), "edge-host")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise MeasurementContractError("output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Path(args.dataset_jsonl).resolve()
    metadata = Path(args.dataset_metadata).resolve()
    rows, dataset_sha = _validate_dataset(dataset, metadata)
    model_hashes = validate_model_assets(
        Path(args.edge_model).resolve(),
        Path(args.teacher_manifest).resolve(),
        Path(args.teacher_blob).resolve(),
    )
    git_commit, dirty = _git_state()
    if dirty:
        raise MeasurementContractError("formal evidence requires a clean Git worktree")

    edge_binary = Path(args.llama_server).resolve()
    edge_base_url = "http://{}:{}".format(args.edge_host, args.edge_port)
    command = [
        str(edge_binary),
        "-m",
        str(Path(args.edge_model).resolve()),
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
    teacher_samples: List[Dict[str, Any]] = []
    try:
        teacher_samples = _evaluate_model(
            rows, args.teacher_host, args.teacher_model, args.timeout, args.code_timeout
        )
        runtime_teacher_digest = _ollama_runtime_digest(args.teacher_host, args.teacher_model)
        if runtime_teacher_digest != model_hashes["teacher_manifest_sha256"]:
            raise MeasurementContractError(
                "Ollama runtime digest does not match pinned teacher manifest"
            )
    finally:
        unload_model(args.teacher_host, args.teacher_model)

    edge_samples: List[Dict[str, Any]] = []
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
            edge_samples = _evaluate_model(
                rows,
                edge_base_url,
                "current-edge-qwen",
                args.timeout,
                args.code_timeout,
            )
        finally:
            stop_server(process)
            log.seek(0)
            log_tail = "\n".join(log.read().splitlines()[-40:])

    generated_at = datetime.now(timezone.utc).astimezone().isoformat()
    dataset_id = "general_capability_eval_v1@sha256:" + dataset_sha
    teacher_id = "ollama:{}@sha256:{}".format(
        args.teacher_model, model_hashes["teacher_manifest_sha256"]
    )
    edge_id = "gguf:{}@sha256:{}".format(
        Path(args.edge_model).name, model_hashes["edge_asset_sha256"]
    )
    raw = {
        "task": "current_traffic_edge_llm_targets_1_2",
        "git_commit": git_commit,
        "worktree_dirty": dirty,
        "run_id": args.run_id,
        "generated_at": generated_at,
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_sha,
        "dataset_sample_ids": [row["sample_id"] for row in rows],
        "hardware_id": args.hardware_id,
        "teacher_model_id": teacher_id,
        "edge_model_id": edge_id,
        "model_hashes": model_hashes,
        "llama_server_sha256": _sha256(edge_binary),
        "llama_server_command": command,
        "teacher_samples": teacher_samples,
        "edge_samples": edge_samples,
        "edge_server_log_tail": log_tail,
    }
    raw_path = output_dir / "targets_1_2_raw.json"
    _write_json(raw_path, raw)
    evidence = build_gate_evidence(raw)
    evidence_paths = {
        target: output_dir / "{}.json".format(target) for target in METRIC_SEMANTICS
    }
    for target, path in evidence_paths.items():
        value = dict(evidence[target])
        value["raw_measurement"] = {"path": raw_path.name, "sha256": _sha256(raw_path)}
        _write_json(path, value)
    fragment = _manifest_fragment(evidence_paths, evidence)
    _write_json(output_dir / "targets_1_2_manifest_fragment.json", fragment)
    print(json.dumps(fragment, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
