"""Build a frozen industrial review-band distillation dataset.

The learned task is deliberately narrow: compare a quantized anomaly score with
the product/modality review band and emit one action token.  It is evidence for
policy distillation, not evidence for RGB/infrared perception accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = "industrial-rule-distillation/v1"
TOKEN_BY_STATE = {"normal": "A", "review": "B", "anomaly": "C"}
MODALITY_CODE = {"rgb": "0", "infrared": "1"}
TRAIN_PER_CELL = 64
VALIDATION_PER_CELL = 8
TEST_PER_CELL = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _quantize(value: Any) -> int:
    scaled = int(round(float(value) * 1_000_000.0))
    if scaled < 0 or scaled > 9999:
        raise ValueError("industrial score/band is outside the four-digit contract")
    return scaled


def encode_prompt(
    modality: str,
    product_index: int,
    score: int,
    review_low: int,
    review_high: int,
) -> str:
    if modality not in MODALITY_CODE:
        raise ValueError("unsupported modality")
    if not 0 <= product_index <= 99:
        raise ValueError("product index is outside the two-digit contract")
    if not 0 <= score <= 9999 or not 0 <= review_low < review_high <= 9999:
        raise ValueError("score/review band is outside the four-digit contract")
    prompt = "1{}{:02d}{:04d}{:04d}{:04d}".format(
        MODALITY_CODE[modality],
        product_index,
        score,
        review_low,
        review_high,
    )
    if len(prompt) != 16 or not prompt.isdigit():
        raise AssertionError("industrial prompt must be exactly 16 decimal tokens")
    return prompt


def state_for_score(score: int, review_low: int, review_high: int) -> str:
    if score < review_low:
        return "normal"
    if score < review_high:
        return "review"
    return "anomaly"


def _domain(state: str, low: int, high: int) -> range:
    if state == "normal":
        return range(0, low)
    if state == "review":
        return range(low, high)
    if state == "anomaly":
        return range(high, 10000)
    raise ValueError("unknown industrial state")


def _anchors(state: str, low: int, high: int) -> Dict[str, Sequence[int]]:
    if state == "normal":
        return {"test": (low - 1,), "validation": (low - 2,), "train": (low - 3,)}
    if state == "review":
        return {
            "test": (low, high - 1),
            "validation": (low + 1, high - 2),
            "train": (low + 2, high - 3),
        }
    return {"test": (high,), "validation": (high + 1,), "train": (high + 2,)}


def _ranked_values(
    seed: int,
    modality: str,
    product: str,
    state: str,
    values: Iterable[int],
) -> List[int]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            "{}|{}|{}|{}|{}".format(
                seed, modality, product, state, value
            ).encode("ascii")
        ).digest(),
    )


def _partition_values(
    seed: int,
    modality: str,
    product: str,
    state: str,
    low: int,
    high: int,
) -> Dict[str, List[int]]:
    counts = {
        "test": TEST_PER_CELL,
        "validation": VALIDATION_PER_CELL,
        "train": TRAIN_PER_CELL,
    }
    domain = list(_domain(state, low, high))
    if len(domain) < sum(counts.values()):
        raise ValueError(
            "review band is too narrow for disjoint splits: {}/{} {}".format(
                modality, product, state
            )
        )
    selected = {name: [] for name in counts}
    used = set()
    for split, values in _anchors(state, low, high).items():
        for value in values:
            if value in domain and value not in used and len(selected[split]) < counts[split]:
                selected[split].append(value)
                used.add(value)
    ranked = _ranked_values(seed, modality, product, state, domain)
    for split in ("test", "validation", "train"):
        for value in ranked:
            if len(selected[split]) >= counts[split]:
                break
            if value not in used:
                selected[split].append(value)
                used.add(value)
        if len(selected[split]) != counts[split]:
            raise AssertionError("unable to fill industrial split")
    return selected


def _row(
    split: str,
    modality: str,
    product: str,
    product_index: int,
    score: int,
    low: int,
    high: int,
) -> Dict[str, Any]:
    state = state_for_score(score, low, high)
    prompt = encode_prompt(modality, product_index, score, low, high)
    return {
        "event_id": "industrial:{}:{}:{}:{}".format(
            split, modality, product, score
        ),
        "category": "industrial_rule_distillation",
        "prompt_format": "raw_task",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": TOKEN_BY_STATE[state]},
        ],
        "metadata": {
            "modality": modality,
            "product": product,
            "product_index": product_index,
            "score_q6": score,
            "review_low_q6": low,
            "review_high_q6": high,
            "state": state,
            "target_token": TOKEN_BY_STATE[state],
            "input_contract": "industrial-review-band-decimal16@v1",
        },
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(_canonical_json(row).decode("utf-8") + "\n")
    return {
        "path": str(path),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build(thresholds_path: Path, output_dir: Path, seed: int) -> Dict[str, Any]:
    thresholds_path = thresholds_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    bands = json.loads(thresholds_path.read_text(encoding="utf-8"))
    modalities = bands.get("modalities")
    if not isinstance(modalities, dict) or set(modalities) != set(MODALITY_CODE):
        raise ValueError("thresholds must contain exactly rgb and infrared")
    products = sorted(modalities["rgb"])
    if set(products) != set(modalities["infrared"]):
        raise ValueError("modalities must use identical product sets")

    rows: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for modality in sorted(modalities):
        for product_index, product in enumerate(products):
            band = modalities[modality][product]
            low = _quantize(band["review_low"])
            high = _quantize(band["review_high"])
            if high - low < TRAIN_PER_CELL + VALIDATION_PER_CELL + TEST_PER_CELL:
                raise ValueError("quantized review band is too narrow")
            for state in TOKEN_BY_STATE:
                split_values = _partition_values(
                    seed, modality, product, state, low, high
                )
                for split, values in split_values.items():
                    rows[split].extend(
                        _row(
                            split,
                            modality,
                            product,
                            product_index,
                            score,
                            low,
                            high,
                        )
                        for score in values
                    )

    for split in rows:
        rows[split].sort(key=lambda row: str(row["event_id"]))
    prompt_sets = {
        split: {row["messages"][0]["content"] for row in split_rows}
        for split, split_rows in rows.items()
    }
    if any(
        prompt_sets[left] & prompt_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise AssertionError("industrial split prompts overlap")

    artifacts = {
        split: _write_jsonl(output_dir / "{}.jsonl".format(split), split_rows)
        for split, split_rows in rows.items()
    }
    distribution: Dict[str, Dict[str, int]] = {}
    for split, split_rows in rows.items():
        counts = {token: 0 for token in TOKEN_BY_STATE.values()}
        for row in split_rows:
            counts[row["messages"][-1]["content"]] += 1
        distribution[split] = counts
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": "industrial_review_band_policy_distillation",
        "claim_boundary": (
            "Measures reproduction of the deterministic review-band policy; "
            "does not measure RGB/infrared anomaly perception accuracy."
        ),
        "seed": seed,
        "input_contract": {
            "name": "industrial-review-band-decimal16@v1",
            "layout": "version1+modality1+product2+score4+review_low4+review_high4",
            "prompt_tokens": 16,
            "scale": "round(value*1e6)",
        },
        "output_contract": {
            "A": "normal",
            "B": "review",
            "C": "anomaly",
            "output_tokens": 1,
        },
        "source": {
            "thresholds_path": str(thresholds_path),
            "thresholds_sha256": _sha256(thresholds_path),
            "builder_path": str(Path(__file__).resolve()),
            "builder_sha256": _sha256(Path(__file__).resolve()),
        },
        "coverage": {
            "modalities": sorted(modalities),
            "products": products,
            "product_modality_cells": len(products) * len(modalities),
            "scores_per_state_cell": {
                "train": TRAIN_PER_CELL,
                "validation": VALIDATION_PER_CELL,
                "test": TEST_PER_CELL,
            },
        },
        "artifacts": artifacts,
        "target_distribution": distribution,
        "split_prompt_overlap": {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        },
        "test_used_for_training": False,
        "real_perception_labels_present": False,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        file_obj.write("\n")
    return {**manifest, "manifest_sha256": _sha256(manifest_path)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a frozen industrial rule-distillation dataset."
    )
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args(argv)
    result = build(Path(args.thresholds), Path(args.output_dir), args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
