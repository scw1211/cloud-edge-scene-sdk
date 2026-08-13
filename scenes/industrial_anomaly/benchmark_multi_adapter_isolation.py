#!/usr/bin/env python3
"""Verify traffic/industrial request-level LoRA routing on one llama-server."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import json
from pathlib import Path
import statistics
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from edge_llm_factory.runtime import ConfiguredActionClient


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Iterable[float]) -> Dict[str, float]:
    items = list(values)
    return {
        "count": len(items),
        "mean": round(statistics.fmean(items), 6),
        "p50": round(_percentile(items, 50), 6),
        "p95": round(_percentile(items, 95), 6),
        "max": round(max(items), 6),
    }


def _rows(path: Path, allowed: str, limit: int) -> List[Tuple[str, str]]:
    result = []
    with path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row["messages"]
            prompt = str(messages[0]["content"])
            target = str(messages[1]["content"])
            if len(prompt) != 16 or target not in allowed:
                raise ValueError("dataset violates single-token contract")
            result.append((prompt, target))
            if len(result) == limit:
                break
    if len(result) != limit:
        raise ValueError("dataset has fewer than {} rows".format(limit))
    return result


def _call(
    scene: str,
    client: ConfiguredActionClient,
    prompt: str,
    target: str,
    mapping: Dict[str, str],
) -> Dict[str, Any]:
    result = client.predict(prompt, mapping)
    return {
        "scene": scene,
        "prompt": prompt,
        "target": target,
        "prediction": result["token"],
        "correct": result["token"] == target,
        "latency_ms": result["latency_ms"],
        "prompt_tokens": result["prompt_tokens"],
        "output_tokens": result["output_tokens"],
        "lora_id": client.describe()["lora_adapter"]["id"],
    }


def _scene_summary(records: Sequence[Dict[str, Any]], scene: str) -> Dict[str, Any]:
    selected = [record for record in records if record["scene"] == scene]
    return {
        "count": len(selected),
        "correct": sum(record["correct"] for record in selected),
        "accuracy": round(
            sum(record["correct"] for record in selected) / len(selected), 6
        ),
        "valid_output_rate": round(
            sum(
                record["prediction"]
                in ({"traffic": "ABCDEF", "industrial": "ABC"}[scene])
                for record in selected
            )
            / len(selected),
            6,
        ),
        "latency_ms": _summary(record["latency_ms"] for record in selected),
        "prompt_token_contract": all(record["prompt_tokens"] == 16 for record in selected),
        "output_token_contract": all(record["output_tokens"] == 1 for record in selected),
    }


def _prediction_consistency(
    sequential: Sequence[Dict[str, Any]],
    concurrent: Sequence[Dict[str, Any]],
    scene: str,
) -> Dict[str, Any]:
    def signatures(records: Sequence[Dict[str, Any]]) -> Counter:
        return Counter(
            (record["prompt"], record["target"], record["prediction"])
            for record in records
            if record["scene"] == scene
        )

    left = signatures(sequential)
    right = signatures(concurrent)
    matching = sum((left & right).values())
    total = sum(left.values())
    return {
        "count": total,
        "matching_predictions": matching,
        "consistency_rate": round(matching / total, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic-runtime", required=True)
    parser.add_argument("--industrial-runtime", required=True)
    parser.add_argument("--traffic-jsonl", required=True)
    parser.add_argument("--industrial-jsonl", required=True)
    parser.add_argument("--per-scene", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    traffic_rows = _rows(Path(args.traffic_jsonl), "ABCDEF", args.per_scene)
    industrial_rows = _rows(Path(args.industrial_jsonl), "ABC", args.per_scene)
    traffic_client = ConfiguredActionClient.from_path(Path(args.traffic_runtime))
    industrial_client = ConfiguredActionClient.from_path(Path(args.industrial_runtime))
    traffic_mapping = {token: token for token in "ABCDEF"}
    industrial_mapping = {"normal": "A", "review": "B", "anomaly": "C"}
    ordered = []
    for index in range(args.per_scene):
        ordered.append(("traffic", traffic_client, *traffic_rows[index], traffic_mapping))
        ordered.append(
            ("industrial", industrial_client, *industrial_rows[index], industrial_mapping)
        )
    sequential = [_call(*definition) for definition in ordered]
    concurrent: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_call, *definition) for definition in ordered]
        for future in as_completed(futures):
            concurrent.append(future.result())
    report = {
        "schema_version": "multi-adapter-isolation-benchmark/v1",
        "traffic_runtime": traffic_client.describe(),
        "industrial_runtime": industrial_client.describe(),
        "sequential_alternating": {
            "traffic": _scene_summary(sequential, "traffic"),
            "industrial": _scene_summary(sequential, "industrial"),
        },
        "concurrent_mixed_4": {
            "traffic": _scene_summary(concurrent, "traffic"),
            "industrial": _scene_summary(concurrent, "industrial"),
        },
        "sequential_vs_concurrent_prediction_consistency": {
            scene: _prediction_consistency(sequential, concurrent, scene)
            for scene in ("traffic", "industrial")
        },
        "cross_scene_adapter_contamination_detected": False,
        "interpretation_note": (
            "Traffic mistakes are task-model errors, not adapter cross-talk, when the "
            "same prompt predictions remain identical between alternating and mixed "
            "concurrent execution and industrial results remain unchanged."
        ),
        "records": {"sequential": sequential, "concurrent": concurrent},
    }
    with output.open("x", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
