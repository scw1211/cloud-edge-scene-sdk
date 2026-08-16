#!/usr/bin/env python3
"""Exercise the industrial ExtraTrees -> selective Qwen cloud chain over HTTP."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from urllib.request import urlopen


SCENE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCENE_ROOT.parents[1]
for value in (str(PROJECT_ROOT), str(SCENE_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from cloud_edge_framework.event_envelope import SceneEventEnvelope  # noqa: E402
from cloud_edge_framework.transport import HttpCloudClient  # noqa: E402
from industrial_anomaly.plugin import IndustrialAnomalyPlugin  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def _health(base_url: str, timeout_seconds: float) -> Dict[str, Any]:
    with urlopen(base_url.rstrip("/") + "/health", timeout=timeout_seconds) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("cloud health must be a JSON object")
    return value


def _events(
    plugin: IndustrialAnomalyPlugin,
    templates: Mapping[str, Mapping[str, Any]],
    sample_id: str,
    scores: Mapping[str, float],
) -> List[Any]:
    events = []
    for modality in ("rgb", "infrared"):
        payload = json.loads(json.dumps(templates[modality]))
        payload["id"] = "{}-{}".format(sample_id, modality)
        payload["subject"] = "capsule-{}".format(sample_id)
        payload["data"]["sample_id"] = sample_id
        payload["data"]["modality"] = modality
        payload["data"]["score"] = float(scores[modality])
        event = plugin.normalize_envelope(SceneEventEnvelope.from_dict(payload))
        local = plugin.edge_decide(event)
        event = replace(
            event,
            metadata={
                **event.metadata,
                **plugin.cloud_submission_metadata(event, local),
            },
            evidence=[item for item in event.evidence if item.level == "summary"],
        )
        events.append(plugin.prepare_cloud_event(event, "summary"))
    return events


def _run_case(
    client: HttpCloudClient,
    plugin: IndustrialAnomalyPlugin,
    templates: Mapping[str, Mapping[str, Any]],
    name: str,
    scores: Mapping[str, float],
    timeout_seconds: float,
) -> Dict[str, Any]:
    sample_id = "industrial-cloud-{}-{}".format(name, time.time_ns())
    events = _events(plugin, templates, sample_id, scores)
    started = time.perf_counter()
    submission = client.aggregate_batch(events, timeout_seconds=timeout_seconds)
    group_ids = {
        str(item["event_id"]): str(item["group_id"])
        for item in submission["items"]
    }
    deadline = time.monotonic() + timeout_seconds
    result = submission
    while time.monotonic() < deadline:
        if result.get("all_terminal") is True:
            break
        time.sleep(0.02)
        result = client.aggregation_results_batch(
            events,
            group_ids,
            timeout_seconds=max(0.01, deadline - time.monotonic()),
        )
        if all(
            item.get("aggregation", {}).get("state") == "completed"
            for item in result.get("items", [])
        ):
            break
    wall_ms = (time.perf_counter() - started) * 1000.0
    groups = result.get("groups", [])
    if len(groups) != 1 or not isinstance(groups[0].get("coordination"), dict):
        raise RuntimeError("industrial cloud aggregation did not become final")
    group = groups[0]
    coordination = group["coordination"]
    decisions = coordination.get("decisions", [])
    if len(decisions) != 2:
        raise RuntimeError("industrial cloud aggregation must return two decisions")
    metadata = [dict(item.get("metadata", {})) for item in decisions]
    reviews = [item.get("cloud_llm_review") for item in metadata]
    return {
        "name": name,
        "sample_id": sample_id,
        "scores": {key: float(value) for key, value in scores.items()},
        "http_wall_ms": round(wall_ms, 6),
        "cloud_runtime_ms": float(coordination["cloud_runtime_ms"]),
        "decisions": [str(item["decision"]) for item in decisions],
        "source": str(metadata[0].get("source", "")),
        "extratrees_confidence": float(metadata[0]["cloud_model_confidence"]),
        "extratrees_probabilities": dict(metadata[0]["cloud_model_probabilities"]),
        "qwen_review_policy": dict(metadata[0]["cloud_llm_review_policy"]),
        "qwen_review_present": [review is not None for review in reviews],
        "qwen_review": reviews[0],
        "same_qwen_review_reused_for_both_modalities": (
            reviews[0] is not None and reviews[0] == reviews[1]
        ),
        "global_confirmation": bool(coordination["global_confirmation"]),
        "globally_consistent": bool(coordination["globally_consistent"]),
        "resolution_success_rate": float(coordination["resolution_success_rate"]),
        "received_members": list(group["aggregation"]["received_members"]),
    }


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("refusing to overwrite existing evidence: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    if temporary.exists():
        raise FileExistsError("temporary evidence already exists: {}".format(temporary))
    with temporary.open("x", encoding="utf-8") as file_obj:
        json.dump(value, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud-url", default="http://127.0.0.1:19102")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New evidence path; an existing file is never selected implicitly.",
    )
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")

    thresholds = SCENE_ROOT / "industrial_anomaly/review_bands.json"
    model = SCENE_ROOT / "assets/models/industrial_cloud_extratrees_capsule_v1.joblib"
    templates = {
        "rgb": _read_json(SCENE_ROOT / "samples/rgb_event.json"),
        "infrared": _read_json(SCENE_ROOT / "samples/infrared_event.json"),
    }
    health = _health(args.cloud_url, args.timeout_seconds)
    plugins = {
        item["scene"]: item
        for item in health.get("runtime", {}).get("plugins", [])
        if isinstance(item, dict) and item.get("scene")
    }
    industrial_health = plugins.get("industrial_anomaly", {}).get("health", {})
    cloud_llm = health.get("runtime", {}).get("cloud_llm", {})
    if health.get("ready") is not True:
        raise RuntimeError("cloud sidecar is not ready")
    if industrial_health.get("cloud_decision_engine") != (
        "industrial_extratrees_with_selective_qwen9b_review"
    ):
        raise RuntimeError("cloud sidecar is not using the industrial ExtraTrees chain")
    if cloud_llm.get("enabled") is not True:
        raise RuntimeError("cloud sidecar Qwen reviewer is not enabled")

    plugin = IndustrialAnomalyPlugin(thresholds_path=thresholds)
    client = HttpCloudClient(args.cloud_url, timeout_seconds=args.timeout_seconds)
    fast = _run_case(
        client,
        plugin,
        templates,
        "high-confidence",
        {"rgb": 0.006, "infrared": 0.007},
        args.timeout_seconds,
    )
    review = _run_case(
        client,
        plugin,
        templates,
        "low-confidence",
        {"rgb": 0.00745, "infrared": 0.00802},
        args.timeout_seconds,
    )
    gates = {
        "service_ready_with_two_scenes": health.get("runtime", {}).get("scenes")
        == ["industrial_anomaly", "traffic"],
        "extratrees_model_identity_matches": (
            industrial_health.get("cloud_model", {}).get("model_id")
            == "industrial-capsule-cross-modal-extratrees-v1"
        ),
        "qwen9b_reviewer_enabled": cloud_llm.get("runtime", {}).get("model")
        == "qwen3.5:9b",
        "fast_path_is_extratrees": fast["source"]
        == "industrial_cloud_extratrees_coordinator",
        "fast_path_final_normal": fast["decisions"] == ["normal", "normal"],
        "fast_path_skips_qwen": fast["qwen_review_present"] == [False, False],
        "fast_path_cloud_runtime_under_200ms": fast["cloud_runtime_ms"] < 200.0,
        "review_path_starts_from_extratrees": review["source"]
        == "industrial_cloud_extratrees_coordinator",
        "review_path_qwen_is_cost_selected": review["qwen_review_policy"].get(
            "eligible"
        )
        is True,
        "review_path_reuses_one_group_review": review[
            "same_qwen_review_reused_for_both_modalities"
        ],
        "qwen_is_advisory_and_preserves_extratrees": review["decisions"]
        == ["normal", "normal"],
        "both_paths_globally_consistent": fast["globally_consistent"]
        and review["globally_consistent"],
    }
    report = {
        "schema_version": "industrial-cloud-chain-http-evidence/v1",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "cloud_url": args.cloud_url,
        "architecture": {
            "aggregation": "paired RGB/infrared compact summaries",
            "primary": "industrial ExtraTrees",
            "selective_reviewer": "Qwen3.5 9B via Ollama",
            "review_safety": (
                "Qwen is advisory; accept and challenge both preserve ExtraTrees"
            ),
        },
        "assets": {
            "extratrees_model": _identity(model),
            "review_bands": _identity(thresholds),
        },
        "service": {
            "status": health.get("status"),
            "ready": health.get("ready"),
            "scenes": health.get("runtime", {}).get("scenes"),
            "framework_version": health.get("framework_version"),
            "industrial_policy_version": industrial_health.get("policy_version"),
            "cloud_llm": cloud_llm,
        },
        "cases": [fast, review],
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    _write_atomic(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
