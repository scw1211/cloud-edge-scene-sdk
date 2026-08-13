#!/usr/bin/env python3
"""Evaluate the cloud 9B model on the industrial cross-modal review policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import urllib.request


LABELS = ("A", "B", "C")
SCORE_SYSTEM_PROMPT = (
    "你是工业双模态质量复核器。分别根据score与review_low/review_high判定："
    "score<review_low为normal；review_low<=score<review_high为review；"
    "score>=review_high为anomaly。若RGB和红外都normal输出A；都anomaly输出C；"
    "其他组合输出B。必须只输出一个大写字母A、B或C。"
)
STATE_SYSTEM_PROMPT = (
    "你是工业双模态质量复核器。输入已经给出RGB和红外的边缘状态。"
    "若两路都normal输出A；若两路都anomaly输出C；其他组合输出B。"
    "必须只输出一个大写字母A、B或C。"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Iterable[float]) -> Dict[str, float]:
    items = list(float(value) for value in values)
    return {
        "count": len(items),
        "mean": round(statistics.fmean(items), 6),
        "p50": round(_percentile(items, 50.0), 6),
        "p95": round(_percentile(items, 95.0), 6),
        "max": round(max(items), 6),
    }


def _get_json(base_url: str, path: str) -> Any:
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=30) as response:
        return json.load(response)


def _post_json(base_url: str, path: str, payload: Mapping[str, Any]) -> Any:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _score(state: str, low: float, high: float, index: int) -> float:
    width = high - low
    fraction = 0.1 + (index % 9) * 0.1
    if state == "normal":
        return max(0.0, low - width * fraction)
    if state == "review":
        return low + width * fraction
    if state == "anomaly":
        return high + width * fraction
    raise ValueError(state)


def _cases(
    thresholds: Mapping[str, Any], count: int, prompt_mode: str
) -> List[Dict[str, Any]]:
    modalities = thresholds["modalities"]
    products = sorted(modalities["rgb"])
    review_pairs = (
        ("normal", "review"),
        ("normal", "anomaly"),
        ("review", "normal"),
        ("review", "review"),
        ("review", "anomaly"),
        ("anomaly", "normal"),
        ("anomaly", "review"),
    )
    result = []
    for index in range(count):
        product = products[index % len(products)]
        label = LABELS[index % len(LABELS)]
        if label == "A":
            states = ("normal", "normal")
        elif label == "C":
            states = ("anomaly", "anomaly")
        else:
            states = review_pairs[(index // len(LABELS)) % len(review_pairs)]
        values = {}
        for modality, state in zip(("rgb", "infrared"), states):
            band = modalities[modality][product]
            low = float(band["review_low"])
            high = float(band["review_high"])
            values[modality] = {
                "state": state,
                "score": _score(state, low, high, index),
                "review_low": low,
                "review_high": high,
            }
        if prompt_mode == "scores":
            prompt = (
                "product={product}; RGB score={rs:.9g} review_low={rl:.9g} "
                "review_high={rh:.9g}; infrared score={iscore:.9g} "
                "review_low={il:.9g} review_high={ih:.9g}"
            ).format(
                product=product,
                rs=values["rgb"]["score"],
                rl=values["rgb"]["review_low"],
                rh=values["rgb"]["review_high"],
                iscore=values["infrared"]["score"],
                il=values["infrared"]["review_low"],
                ih=values["infrared"]["review_high"],
            )
        elif prompt_mode == "states":
            prompt = "product={}; RGB={}; infrared={}".format(
                product, states[0], states[1]
            )
        else:
            raise ValueError("unsupported prompt_mode")
        result.append(
            {
                "case_id": "industrial-cloud9b-{:04d}".format(index),
                "product": product,
                "states": {"rgb": states[0], "infrared": states[1]},
                "prompt": prompt,
                "target": label,
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--prompt-mode", choices=("scores", "states"), default="states")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    thresholds_path = Path(args.thresholds).resolve()
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    cases = _cases(thresholds, args.count, args.prompt_mode)
    system_prompt = (
        STATE_SYSTEM_PROMPT if args.prompt_mode == "states" else SCORE_SYSTEM_PROMPT
    )
    tags_before = _get_json(args.base_url, "/api/tags")
    model_rows = [row for row in tags_before["models"] if row["name"] == args.model]
    if len(model_rows) != 1:
        raise RuntimeError("expected exactly one model identity")
    version_before = _get_json(args.base_url, "/api/version")

    records = []
    for case in cases:
        started = time.perf_counter()
        response = _post_json(
            args.base_url,
            "/api/generate",
            {
                "model": args.model,
                "system": system_prompt,
                "prompt": case["prompt"],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": 42,
                    "num_ctx": 512,
                    "num_predict": 1,
                },
            },
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        raw_output = str(response.get("response", ""))
        valid = raw_output in LABELS
        records.append(
            {
                **case,
                "raw_output": raw_output,
                "valid": valid,
                "correct": valid and raw_output == case["target"],
                "wall_ms": round(wall_ms, 6),
                "server_total_ms": round(float(response["total_duration"]) / 1e6, 6),
                "load_ms": round(float(response["load_duration"]) / 1e6, 6),
                "prompt_eval_count": int(response["prompt_eval_count"]),
                "eval_count": int(response["eval_count"]),
                "done": bool(response.get("done")),
                "model": response.get("model"),
            }
        )
    tags_after = _get_json(args.base_url, "/api/tags")
    version_after = _get_json(args.base_url, "/api/version")
    correct = sum(row["correct"] for row in records)
    valid = sum(row["valid"] for row in records)
    by_label = {
        label: {
            "count": sum(row["target"] == label for row in records),
            "correct": sum(row["target"] == label and row["correct"] for row in records),
        }
        for label in LABELS
    }
    for value in by_label.values():
        value["recall"] = round(value["correct"] / value["count"], 6)
    report = {
        "schema_version": "industrial-cloud9b-policy-evaluation/v1",
        "created_at_epoch_ms": int(time.time() * 1000),
        "claim_boundary": (
            "This evaluates the 9B model's execution of the industrial RGB/infrared "
            "cross-modal review policy on controlled held-out cases. "
            "It is not an image anomaly perception evaluation."
        ),
        "model_identity": model_rows[0],
        "runtime_version_before": version_before,
        "runtime_version_after": version_after,
        "model_identity_stable": tags_before == tags_after,
        "thresholds": {
            "path": str(thresholds_path),
            "bytes": thresholds_path.stat().st_size,
            "sha256": _sha256(thresholds_path),
        },
        "protocol": {
            "prompt_mode": args.prompt_mode,
            "input_semantics": (
                "states already produced by edge perception and threshold calibration"
                if args.prompt_mode == "states"
                else "raw scores and calibrated thresholds"
            ),
            "system_prompt": system_prompt,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "num_ctx": 512,
            "num_predict": 1,
            "thinking": False,
            "strict_outputs": list(LABELS),
        },
        "metrics": {
            "count": len(records),
            "correct": correct,
            "accuracy": round(correct / len(records), 6),
            "valid": valid,
            "valid_rate": round(valid / len(records), 6),
            "by_label": by_label,
            "wall_ms": _summary(row["wall_ms"] for row in records),
            "server_total_ms": _summary(row["server_total_ms"] for row in records),
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    print(json.dumps({"output": str(output), **report["metrics"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
