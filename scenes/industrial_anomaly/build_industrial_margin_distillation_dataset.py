"""Build v2 industrial action data from normalized review-band margins.

The deterministic codec performs feature extraction only.  The LoRA learns the
mapping from modality/product/margins to the normal/review/anomaly action slot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from scenes.industrial_anomaly.industrial_anomaly.action_codec import (
    encode_margin_prompt,
    margin_position,
    prompt_from_values,
)


SCHEMA_VERSION = "industrial-margin-distillation/v2"
TOKEN_BY_STATE = {"normal": "A", "review": "B", "anomaly": "C"}
MODALITY_CODE = {"rgb": "0", "infrared": "1"}
TRAIN_PER_CELL = 64
VALIDATION_PER_CELL = 8
TEST_PER_CELL = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state(position_milli: int) -> str:
    if position_milli < 0:
        return "normal"
    if position_milli < 1000:
        return "review"
    return "anomaly"


def _positions(state: str) -> range:
    if state == "normal":
        return range(-4000, 0)
    if state == "review":
        return range(0, 1000)
    if state == "anomaly":
        return range(1000, 5001)
    raise ValueError("unknown industrial state")


def _ranked(seed: int, modality: str, product: str, state: str) -> List[int]:
    return sorted(
        _positions(state),
        key=lambda value: hashlib.sha256(
            "{}|{}|{}|{}|{}".format(
                seed, modality, product, state, value
            ).encode("ascii")
        ).digest(),
    )


def _partition(seed: int, modality: str, product: str, state: str) -> Dict[str, List[int]]:
    counts = {
        "test": TEST_PER_CELL,
        "validation": VALIDATION_PER_CELL,
        "train": TRAIN_PER_CELL,
    }
    boundaries = {
        "normal": {"test": (-1,), "validation": (-2,), "train": (-3,)},
        "review": {
            "test": (0, 999),
            "validation": (1, 998),
            "train": (2, 997),
        },
        "anomaly": {"test": (1000,), "validation": (1001,), "train": (1002,)},
    }
    selected = {name: list(boundaries[state][name]) for name in counts}
    used = {value for values in selected.values() for value in values}
    for split in ("test", "validation", "train"):
        for value in _ranked(seed, modality, product, state):
            if len(selected[split]) >= counts[split]:
                break
            if value not in used:
                selected[split].append(value)
                used.add(value)
    return selected


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    with path.open("x", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return {
        "path": str(path),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build(thresholds_path: Path, output_dir: Path, seed: int) -> Dict[str, Any]:
    thresholds_path = thresholds_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(thresholds_path.read_text(encoding="utf-8"))
    modalities = source.get("modalities")
    if not isinstance(modalities, dict) or set(modalities) != set(MODALITY_CODE):
        raise ValueError("thresholds must contain rgb and infrared")
    products = sorted(modalities["rgb"])
    if set(products) != set(modalities["infrared"]):
        raise ValueError("industrial products differ by modality")

    rows: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for modality in sorted(modalities):
        for product_index, product in enumerate(products):
            band = modalities[modality][product]
            width_q6 = int(
                round(
                    (float(band["review_high"]) - float(band["review_low"]))
                    * 1_000_000.0
                )
            )
            if not 1 <= width_q6 <= 9999:
                raise ValueError("industrial band width is outside decimal16 contract")
            for state in TOKEN_BY_STATE:
                for split, positions in _partition(
                    seed, modality, product, state
                ).items():
                    for position in positions:
                        prompt = encode_margin_prompt(
                            modality, product_index, position, width_q6
                        )
                        rows[split].append(
                            {
                                "event_id": "industrial-v2:{}:{}:{}:{}".format(
                                    split, modality, product, position
                                ),
                                "category": "industrial_margin_distillation",
                                "prompt_format": "raw_task",
                                "messages": [
                                    {"role": "user", "content": prompt},
                                    {
                                        "role": "assistant",
                                        "content": TOKEN_BY_STATE[state],
                                    },
                                ],
                                "metadata": {
                                    "modality": modality,
                                    "product": product,
                                    "product_index": product_index,
                                    "relative_position_milli": position,
                                    "band_width_q6": width_q6,
                                    "state": state,
                                    "target_token": TOKEN_BY_STATE[state],
                                    "input_contract": "industrial-margin-decimal16@v2",
                                },
                            }
                        )
    for split in rows:
        rows[split].sort(key=lambda row: str(row["event_id"]))
    prompts = {
        split: {row["messages"][0]["content"] for row in split_rows}
        for split, split_rows in rows.items()
    }
    overlap = {
        "train_validation": len(prompts["train"] & prompts["validation"]),
        "train_test": len(prompts["train"] & prompts["test"]),
        "validation_test": len(prompts["validation"] & prompts["test"]),
    }
    if any(overlap.values()):
        raise AssertionError("industrial v2 split prompts overlap")
    artifacts = {
        split: _write_jsonl(output_dir / (split + ".jsonl"), split_rows)
        for split, split_rows in rows.items()
    }
    distribution = {}
    for split, split_rows in rows.items():
        distribution[split] = {
            token: sum(row["messages"][-1]["content"] == token for row in split_rows)
            for token in TOKEN_BY_STATE.values()
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": "industrial_margin_policy_distillation",
        "claim_boundary": (
            "The codec deterministically extracts normalized distances to the "
            "calibrated review band. The LoRA maps those structured features to "
            "actions. This does not measure image anomaly perception accuracy."
        ),
        "seed": seed,
        "input_contract": {
            "name": "industrial-margin-decimal16@v2",
            "layout": "version1+modality1+product2+lower_margin4+upper_margin4+band_width4",
            "prompt_tokens": 16,
            "margin_center": 4000,
            "margin_scale": 1000,
        },
        "output_contract": {"A": "normal", "B": "review", "C": "anomaly"},
        "source": {
            "thresholds_path": str(thresholds_path),
            "thresholds_sha256": sha256_file(thresholds_path),
            "builder_path": str(Path(__file__).resolve()),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
        },
        "coverage": {
            "modalities": sorted(modalities),
            "products": products,
            "product_modality_cells": 20,
        },
        "artifacts": artifacts,
        "target_distribution": distribution,
        "split_prompt_overlap": overlap,
        "test_used_for_training": False,
        "previous_v1_test_used_for_training": False,
        "real_perception_labels_present": False,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        file_obj.write("\n")
    return {**manifest, "manifest_sha256": sha256_file(manifest_path)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build industrial margin KD v2.")
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build(Path(args.thresholds), Path(args.output_dir), args.seed),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
