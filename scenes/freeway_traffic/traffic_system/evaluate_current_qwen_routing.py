"""用独立验证集校准 Edge-Qwen 让权，并在测试集比较五种决策路径。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from traffic_system.decision_utils import ACTION_TOKEN_TO_DECISION, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val_jsonl", default="datasets/edge_qwen_current_state_future_v2/val.jsonl"
    )
    parser.add_argument("--val_qwen_json", required=True)
    parser.add_argument(
        "--test_jsonl", default="datasets/edge_qwen_current_state_future_v2/test.jsonl"
    )
    parser.add_argument("--test_qwen_json", required=True)
    parser.add_argument(
        "--output_json", default="results/llm/current_state_qwen_routing_eval.json"
    )
    parser.add_argument("--minimum_stratum_support", type=int, default=12)
    parser.add_argument("--minimum_validation_gain", type=float, default=0.03)
    parser.add_argument("--maximum_validation_route_rate", type=float, default=0.20)
    parser.add_argument("--cloud_uncertainty_threshold", type=float, default=0.60)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--profile_json",
        default="assets/models/edge_qwen_gain_router_current_state_v1.json",
    )
    return parser.parse_args()


def load_qwen(path: Path) -> Dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples", [])
    if not isinstance(examples, list):
        raise ValueError("Qwen evaluation JSON has no examples list")
    output = {}
    for example in examples:
        event_id = str(example.get("event_id", ""))
        if not event_id or event_id in output:
            raise ValueError("Qwen evaluation event_id is missing or duplicated")
        output[event_id] = example
    return output


def joined_rows(data_path: Path, qwen_path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(data_path)
    qwen = load_qwen(qwen_path)
    output = []
    for row in rows:
        event_id = str(row["event_id"])
        result = qwen.get(event_id)
        if result is None:
            raise ValueError("Qwen result missing event {}".format(event_id))
        token = result.get("parsed")
        output.append(
            {
                **row,
                "qwen_token": token,
                "qwen_decision": ACTION_TOKEN_TO_DECISION.get(str(token)),
                "qwen_valid": str(token) in ACTION_TOKEN_TO_DECISION,
                "qwen_probabilities": dict(result.get("class_probabilities", {})),
                "qwen_latency_ms": float(result.get("latency_ms", 0.0)),
            }
        )
    extra = set(qwen) - {str(row["event_id"]) for row in rows}
    if extra:
        raise ValueError("Qwen evaluation has {} unexpected events".format(len(extra)))
    return output


def observable_stratum(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    confidence_bucket = min(3, max(0, int(float(row["student_confidence"]) * 4.0)))
    return (
        str(row["network_status"]),
        str(row["student_decision"]),
        str(row["rule_decision"]),
        confidence_bucket,
        int(row["prediction_set_size"]) > 1,
    )


def calibrate_router(
    rows: Sequence[Mapping[str, Any]],
    minimum_support: int,
    minimum_gain: float,
    maximum_route_rate: float,
) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[observable_stratum(row)].append(row)
    candidates = []
    for key, values in grouped.items():
        if len(values) < minimum_support:
            continue
        student_correct = sum(
            row["student_decision"] == row["target_decision"] for row in values
        )
        qwen_correct = sum(row["qwen_decision"] == row["target_decision"] for row in values)
        gain = (qwen_correct - student_correct) / len(values)
        corrections = sum(
            row["student_decision"] != row["target_decision"]
            and row["qwen_decision"] == row["target_decision"]
            for row in values
        )
        regressions = sum(
            row["student_decision"] == row["target_decision"]
            and row["qwen_decision"] != row["target_decision"]
            for row in values
        )
        if gain >= minimum_gain and corrections > regressions:
            candidates.append((key, {
                "support": len(values),
                "validation_gain": gain,
                "corrections": corrections,
                "regressions": regressions,
            }))
    maximum_rows = max(0, int(len(rows) * maximum_route_rate))
    accepted: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    selected_rows = 0
    for key, value in sorted(
        candidates,
        key=lambda item: (
            float(item[1]["validation_gain"]),
            int(item[1]["support"]),
        ),
        reverse=True,
    ):
        support = int(value["support"])
        if selected_rows + support > maximum_rows:
            continue
        accepted[key] = value
        selected_rows += support
    return accepted


def stratum_key(values: Sequence[Any]) -> str:
    return "|".join(
        [
            str(values[0]),
            str(values[1]),
            str(values[2]),
            str(int(values[3])),
            "1" if bool(values[4]) else "0",
        ]
    )


def accuracy(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([row[key] == row["target_decision"] for row in rows]))


def grouped_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: Optional[str],
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    groups: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["sample_id"])].append(row)
    group_ids = sorted(groups)
    rng = random.Random(seed)

    def score(selected: Iterable[int]) -> float:
        values = [row for group_id in selected for row in groups[group_id]]
        left_value = accuracy(values, left)
        return left_value if right is None else left_value - accuracy(values, right)

    estimate = score(group_ids)
    boot = [
        score([rng.choice(group_ids) for _ in group_ids])
        for _ in range(max(1, samples))
    ]
    low, high = np.percentile(np.asarray(boot), [2.5, 97.5]).tolist()
    return {
        "estimate": round(estimate, 6),
        "ci95": [round(float(low), 6), round(float(high), 6)],
        "groups": len(group_ids),
        "bootstrap_samples": max(1, samples),
    }


def main() -> None:
    args = parse_args()
    val_rows = joined_rows(resolve_path(args.val_jsonl), resolve_path(args.val_qwen_json))
    test_rows = joined_rows(resolve_path(args.test_jsonl), resolve_path(args.test_qwen_json))
    router = calibrate_router(
        val_rows,
        args.minimum_stratum_support,
        args.minimum_validation_gain,
        args.maximum_validation_route_rate,
    )
    for row in test_rows:
        routed = observable_stratum(row) in router
        row["selective_qwen_routed"] = routed
        row["selective_decision"] = (
            row["qwen_decision"]
            if routed and row["qwen_valid"]
            else row["student_decision"]
        )
        probabilities = row["qwen_probabilities"]
        max_probability = max((float(value) for value in probabilities.values()), default=0.0)
        row["cloud_review_eligible"] = bool(
            row["hard_sample"]
            and not routed
            and (
                max_probability < args.cloud_uncertainty_threshold
                or row["qwen_decision"] != row["rule_decision"]
            )
        )

    strategies = {
        "rule": "rule_decision",
        "student": "student_decision",
        "edge_qwen": "qwen_decision",
        "student_plus_gain_routed_qwen": "selective_decision",
    }
    metrics = {
        name: grouped_bootstrap(
            test_rows, key, None, args.bootstrap_samples, args.seed + index
        )
        for index, (name, key) in enumerate(strategies.items())
    }
    deltas = {
        "edge_qwen_minus_rule": grouped_bootstrap(
            test_rows,
            "qwen_decision",
            "rule_decision",
            args.bootstrap_samples,
            args.seed + 101,
        ),
        "selective_minus_rule": grouped_bootstrap(
            test_rows,
            "selective_decision",
            "rule_decision",
            args.bootstrap_samples,
            args.seed + 102,
        ),
        "selective_minus_student": grouped_bootstrap(
            test_rows,
            "selective_decision",
            "student_decision",
            args.bootstrap_samples,
            args.seed + 104,
        ),
        "edge_qwen_minus_student": grouped_bootstrap(
            test_rows,
            "qwen_decision",
            "student_decision",
            args.bootstrap_samples,
            args.seed + 103,
        ),
    }
    route_rows = [row for row in test_rows if row["selective_qwen_routed"]]
    corrections = sum(
        row["student_decision"] != row["target_decision"]
        and row["selective_decision"] == row["target_decision"]
        for row in route_rows
    )
    regressions = sum(
        row["student_decision"] == row["target_decision"]
        and row["selective_decision"] != row["target_decision"]
        for row in route_rows
    )
    summary = {
        "task": "current_state_future_grounded_edge_qwen_routing_eval_v2",
        "test_rows": len(test_rows),
        "sample_groups": len({int(row["sample_id"]) for row in test_rows}),
        "validation_rows": len(val_rows),
        "router_contract": {
            "features": [
                "network_status",
                "student_decision",
                "rule_decision",
                "student_confidence_bucket",
                "prediction_set_ambiguity",
            ],
            "minimum_stratum_support": args.minimum_stratum_support,
            "minimum_validation_gain": args.minimum_validation_gain,
            "maximum_validation_route_rate": args.maximum_validation_route_rate,
            "accepted_strata": len(router),
            "test_route_count": len(route_rows),
            "test_route_rate": round(len(route_rows) / max(1, len(test_rows)), 6),
            "corrections": corrections,
            "regressions": regressions,
            "net_corrections": corrections - regressions,
        },
        "metrics": metrics,
        "paired_deltas": deltas,
        "edge_qwen_latency_ms": {
            "mean": round(float(np.mean([row["qwen_latency_ms"] for row in test_rows])), 3),
            "p95": round(float(np.percentile([row["qwen_latency_ms"] for row in test_rows], 95)), 3),
            "environment": "evaluation_host_only_not_jetson",
        },
        "cloud_review": {
            "quality_measured": False,
            "reason": "no cloud 9B outputs were supplied; only eligible rows are counted",
            "eligible_count": sum(row["cloud_review_eligible"] for row in test_rows),
            "eligible_rate": round(
                sum(row["cloud_review_eligible"] for row in test_rows)
                / max(1, len(test_rows)),
                6,
            ),
        },
        "accepted_validation_strata": [
            {"key": list(key), **value} for key, value in sorted(router.items())
        ],
    }
    output_path = resolve_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    profile = {
        "schema_version": 1,
        "task": "edge_qwen_gain_router_current_state_v1",
        "baseline": "current_state_student",
        "features": summary["router_contract"]["features"],
        "minimum_validation_gain": args.minimum_validation_gain,
        "maximum_validation_route_rate": args.maximum_validation_route_rate,
        "validation_rows": len(val_rows),
        "accepted_strata": {
            stratum_key(key): {
                "validation_gain": round(float(value["validation_gain"]), 8),
                "support": int(value["support"]),
                "corrections": int(value["corrections"]),
                "regressions": int(value["regressions"]),
            }
            for key, value in sorted(router.items())
        },
    }
    profile_path = resolve_path(args.profile_json)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
