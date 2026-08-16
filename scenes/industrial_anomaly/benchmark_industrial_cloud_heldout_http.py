#!/usr/bin/env python3
"""Run the frozen 18-pair industrial heldout set through the cloud HTTP API."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
import sys
from typing import Any, Dict, Mapping, Sequence, Tuple


SCENE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCENE_ROOT.parents[1]
for value in (str(PROJECT_ROOT), str(SCENE_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmark_industrial_cloud_chain import (  # noqa: E402
    _health,
    _identity,
    _read_json,
    _run_case,
    _write_atomic,
)
from cloud_edge_framework.transport import HttpCloudClient  # noqa: E402
from industrial_anomaly.plugin import IndustrialAnomalyPlugin  # noqa: E402


EXPECTED_INPUTS = {
    "rgb": {
        "bytes": 6415,
        "sha256": "7a485739a5e2630237a69497efc16a0b90619aed298bc695e750b6385a236517",
    },
    "infrared": {
        "bytes": 6695,
        "sha256": "48f20ebe0b16a92acbe077f7f02ab918a86a787790abd620d327cf9626f8fc2e",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_scores(path: Path) -> Dict[Tuple[str, str], float]:
    result = {}
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            image = PurePosixPath(str(row["path"]))
            key = (image.parent.name, image.stem)
            if key in result:
                raise ValueError("duplicate industrial score identity: {}".format(key))
            result[key] = float(row["image_score"])
    if not result:
        raise ValueError("industrial score CSV is empty")
    return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency(values: Sequence[float]) -> Dict[str, Any]:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6),
        "p50": round(_percentile(values, 50.0), 6),
        "p95": round(_percentile(values, 95.0), 6),
        "max": round(max(values), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud-url", default="http://127.0.0.1:18100")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--rgb-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--infrared-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout-fraction", type=float, default=0.31)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New evidence path; committed evidence is never overwritten implicitly.",
    )
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")

    paths = {
        "rgb": args.rgb_predictions.resolve(),
        "infrared": args.infrared_predictions.resolve(),
    }
    for modality, path in paths.items():
        expected = EXPECTED_INPUTS[modality]
        if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
            raise ValueError("{} score evidence identity mismatch".format(modality))
    scores = {modality: _read_scores(path) for modality, path in paths.items()}
    if set(scores["rgb"]) != set(scores["infrared"]):
        raise ValueError("RGB and infrared score identities differ")
    keys = sorted(scores["rgb"])
    labels = [0 if key[0] == "good" else 1 for key in keys]

    import numpy as np
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(keys), dtype=np.int64)
    _, heldout_indices = train_test_split(
        indices,
        test_size=float(args.heldout_fraction),
        random_state=int(args.seed),
        stratify=np.asarray(labels, dtype=np.int64),
    )
    ordered_heldout = sorted(int(value) for value in heldout_indices)
    identity = hashlib.sha256(
        "\n".join("{}/{}".format(*keys[index]) for index in ordered_heldout).encode()
    ).hexdigest()
    if identity != "cda56e8a5553bdad05ea707aa7211e4fcf23540ff11e7228ec841274ced6a5e3":
        raise ValueError("heldout split identity mismatch")

    health = _health(args.cloud_url, args.timeout_seconds)
    plugin_health = {
        item["scene"]: item["health"]
        for item in health.get("runtime", {}).get("plugins", [])
    }
    industrial_health = plugin_health.get("industrial_anomaly", {})
    if health.get("ready") is not True or industrial_health.get(
        "cloud_decision_engine"
    ) != "industrial_extratrees_with_selective_qwen9b_review":
        raise RuntimeError("formal cloud is not running the industrial ExtraTrees chain")

    thresholds = SCENE_ROOT / "industrial_anomaly/review_bands.json"
    model = SCENE_ROOT / "assets/models/industrial_cloud_extratrees_capsule_v1.joblib"
    templates = {
        "rgb": _read_json(SCENE_ROOT / "samples/rgb_event.json"),
        "infrared": _read_json(SCENE_ROOT / "samples/infrared_event.json"),
    }
    plugin = IndustrialAnomalyPlugin(thresholds_path=thresholds)
    client = HttpCloudClient(args.cloud_url, timeout_seconds=args.timeout_seconds)
    records = []
    for position, index in enumerate(ordered_heldout):
        key = keys[index]
        record = _run_case(
            client,
            plugin,
            templates,
            "heldout-{:02d}".format(position),
            {
                "rgb": scores["rgb"][key],
                "infrared": scores["infrared"][key],
            },
            args.timeout_seconds,
        )
        record["identity"] = "{}/{}".format(*key)
        record["expected"] = "normal" if labels[index] == 0 else "anomaly"
        record["extratrees_decision"] = max(
            record["extratrees_probabilities"],
            key=record["extratrees_probabilities"].get,
        )
        record["extratrees_correct"] = (
            record["extratrees_decision"] == record["expected"]
        )
        records.append(record)

    selected = [record for record in records if any(record["qwen_review_present"])]
    http_latency = _latency([record["http_wall_ms"] for record in records])
    cloud_latency = _latency([record["cloud_runtime_ms"] for record in records])
    accuracy = sum(record["extratrees_correct"] for record in records) / len(records)
    gates = {
        "heldout_identity_matches_training_evidence": True,
        "completed_18_pairs": len(records) == 18,
        "all_pairs_cross_modal_final": all(
            record["global_confirmation"]
            and record["received_members"] == ["infrared", "rgb"]
            for record in records
        ),
        "extratrees_accuracy_is_1": accuracy == 1.0,
        "qwen_selection_rate_at_most_0_20": len(selected) / len(records) <= 0.20,
        "population_http_mean_under_200ms": http_latency["mean"] < 200.0,
        "population_cloud_runtime_mean_under_200ms": cloud_latency["mean"] < 200.0,
        "all_results_globally_consistent": all(
            record["globally_consistent"] for record in records
        ),
        "selected_reviews_are_group_deduplicated": all(
            record["same_qwen_review_reused_for_both_modalities"]
            for record in selected
        ),
    }
    report = {
        "schema_version": "industrial-cloud-heldout-http-evidence/v1",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "cloud_url": args.cloud_url,
        "inputs": {key: _identity(value) for key, value in paths.items()},
        "model": _identity(model),
        "split": {
            "seed": int(args.seed),
            "heldout_fraction": float(args.heldout_fraction),
            "heldout_pairs": len(records),
            "heldout_identity_sha256": identity,
        },
        "summary": {
            "extratrees_accuracy": round(accuracy, 9),
            "qwen_selected_pairs": len(selected),
            "qwen_selection_rate": round(len(selected) / len(records), 9),
            "http_final_latency_ms": http_latency,
            "cloud_runtime_ms": cloud_latency,
        },
        "records": records,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    _write_atomic(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
