"""构建当前观测+Student/规则/网络上下文到未来真实动作的 Edge-Qwen 数据。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from traffic_system.build_llm_sft_dataset import ACTION_TOKEN_CODES, build_user_prompt
from traffic_system.current_state_perception_runtime import (
    CurrentStateTrafficPerceptionRuntime,
)
from traffic_system.decision_utils import (
    DECISION_CLASSES,
    rule_teacher_decision,
)
from traffic_system.edge_student import load_student_model, predict_student
from traffic_system.risk_labels import RISK_CLASSES, enable_numpy_pickle_compatibility
from traffic_system.ultracompact_codec import encode_routing_context_v2_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLOUD_REQUIRED_DECISIONS = {"regional_coordination", "reroute"}
NETWORK_STATES = ("normal", "weak", "offline")


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_npz",
        default="assets/downloads/PEMS08_r1_d0_w0_astcgn_multitask.npz",
    )
    parser.add_argument("--risk_labels", required=True)
    parser.add_argument(
        "--student_model",
        default="assets/models/edge_student_freeway_current_state_future_v1.json",
    )
    parser.add_argument(
        "--current_state_config",
        default="assets/models/current_state_perception_v1.json",
    )
    parser.add_argument(
        "--topology", default="assets/models/traffic_region_topology_metis4.json"
    )
    parser.add_argument(
        "--output_dir", default="datasets/edge_qwen_current_state_future_v2"
    )
    parser.add_argument("--train_per_class", type=int, default=600)
    parser.add_argument("--val_sample_groups", type=int, default=100)
    parser.add_argument("--test_sample_groups", type=int, default=200)
    parser.add_argument("--student_confidence_threshold", type=float, default=0.75)
    parser.add_argument("--ambiguity_margin", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def _one_hot(label_id: int) -> Dict[str, float]:
    return {
        name: 1.0 if index == int(label_id) else 0.0
        for index, name in enumerate(RISK_CLASSES)
    }


def build_future_truth_event(
    current_event: Mapping[str, Any],
    node_labels: np.ndarray,
    region_label: int,
) -> Dict[str, Any]:
    managed_nodes = [int(value) for value in current_event["managed_node_ids"]]
    labels = np.asarray(node_labels, dtype=np.int64)
    selected = labels[managed_nodes]
    counts = {
        name: int(np.sum(selected == index))
        for index, name in enumerate(RISK_CLASSES)
    }
    max_label = int(np.max(selected)) if selected.size else 0
    top_nodes = sorted(
        (
            {
                "node_id": node_id,
                "risk_level": RISK_CLASSES[int(labels[node_id])],
                "risk_score": round(float(labels[node_id]) / 3.0, 6),
                "risk_confidence": 1.0,
            }
            for node_id in managed_nodes
        ),
        key=lambda item: (RISK_CLASSES.index(item["risk_level"]), item["node_id"]),
        reverse=True,
    )[:10]
    cluster = counts["severe"] >= 2 or counts["severe"] + counts["high"] >= 4
    summary = {
        "num_nodes": len(managed_nodes),
        "num_low_nodes": counts["low"],
        "num_medium_nodes": counts["medium"],
        "num_high_nodes": counts["high"],
        "num_severe_nodes": counts["severe"],
        "node_risk_counts": counts,
        "region_risk_level": RISK_CLASSES[int(region_label)],
        "region_risk_score": round(float(region_label) / 3.0, 6),
        "region_risk_confidence": 1.0,
        "region_risk_probabilities": _one_hot(int(region_label)),
        "max_node_risk_level": RISK_CLASSES[max_label],
        "max_risk_score": round(float(max_label) / 3.0, 6),
        "mean_node_risk_score": round(float(np.mean(selected)) / 3.0, 6),
        "mean_risk_score": round(float(np.mean(selected)) / 3.0, 6),
        "congestion_cluster_detected": cluster,
    }
    return {
        **dict(current_event),
        "task": "future_observation_grounded_edge_decision",
        "risk_source": "future_observed_fcm_labels",
        "upload_required": bool(max_label > 0),
        "upload_level": "regional_context" if max_label >= 2 else "summary",
        "region_summary": summary,
        "top_k_risk_nodes": top_nodes,
    }


def ambiguity_size(event: Mapping[str, Any], margin: float) -> int:
    probabilities = event.get("region_summary", {}).get(
        "region_risk_probabilities", {}
    )
    values = sorted(
        (float(value) for value in probabilities.values()), reverse=True
    )
    return 2 if len(values) >= 2 and values[0] - values[1] < margin else 1


def offline_target(reference: str, current_rule: str, truth_event: Mapping[str, Any]) -> str:
    if reference not in CLOUD_REQUIRED_DECISIONS:
        return reference
    if current_rule not in CLOUD_REQUIRED_DECISIONS:
        return current_rule
    counts = truth_event["region_summary"]["node_risk_counts"]
    capabilities = truth_event.get("control_capabilities", {})
    if counts.get("severe", 0) and capabilities.get("ramp_meter_nodes"):
        return "ramp_metering"
    if counts.get("high", 0) or counts.get("severe", 0):
        return "variable_speed_limit"
    return "congestion_warning"


def routing_reasons(
    student_decision: str,
    student_confidence: float,
    rule_decision: str,
    prediction_set_size: int,
    confidence_threshold: float,
) -> List[str]:
    reasons = []
    if student_decision != rule_decision:
        reasons.append("student_rule_disagreement")
    if student_confidence < confidence_threshold:
        reasons.append("student_low_confidence")
    if prediction_set_size > 1:
        reasons.append("current_state_ambiguous")
    return reasons


def outcome_reasons(
    target: str,
    student_decision: str,
    rule_decision: str,
    reference: str,
) -> List[str]:
    reasons = []
    if student_decision != target:
        reasons.append("student_wrong")
    if rule_decision != target:
        reasons.append("rule_wrong")
    if reference in CLOUD_REQUIRED_DECISIONS:
        reasons.append("future_action_requires_coordination")
    return reasons


def build_row(
    event: Mapping[str, Any],
    split: str,
    student_decision: str,
    student_confidence: float,
    rule_decision: str,
    reference_decision: str,
    network_status: str,
    prediction_set_size: int,
    confidence_threshold: float,
    truth_event: Mapping[str, Any],
) -> Dict[str, Any]:
    target_decision = (
        offline_target(reference_decision, rule_decision, truth_event)
        if network_status == "offline"
        else reference_decision
    )
    target = ACTION_TOKEN_CODES[target_decision]
    context = {
        "student_decision": student_decision,
        "rule_decision": rule_decision,
        "student_confidence": student_confidence,
        "prediction_set_size": prediction_set_size,
        "network_status": network_status,
    }
    legacy_prompt = build_user_prompt(dict(event), {}, "action_token", max_top_nodes=3)
    prompt = encode_routing_context_v2_prompt(legacy_prompt, context)
    if len(prompt) != 16:
        raise AssertionError("routing-context-v2 prompt must be 16 characters")
    reasons = routing_reasons(
        student_decision,
        student_confidence,
        rule_decision,
        prediction_set_size,
        confidence_threshold,
    )
    outcomes = outcome_reasons(
        target_decision,
        student_decision,
        rule_decision,
        reference_decision,
    )
    source_event_id = str(event["event_id"])
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "prompt_format": "raw_task",
        "input_encoding": "routing_context_v2",
        "feature_code": prompt,
        "target": target,
        "target_decision": target_decision,
        "future_reference_decision": reference_decision,
        "target_source": "future_observed_fcm_policy",
        "source_event_id": source_event_id,
        "event_id": "{}:{}".format(source_event_id, network_status),
        "sample_id": int(event["sample_id"]),
        "partition_id": int(event["partition_id"]),
        "split": split,
        "network_status": network_status,
        "student_decision": student_decision,
        "student_confidence": round(float(student_confidence), 8),
        "rule_decision": rule_decision,
        "prediction_set_size": prediction_set_size,
        "hard_reasons": reasons,
        "hard_sample": bool(reasons),
        "outcome_reasons": outcomes,
    }


def rows_for_split(
    split: str,
    data_path: Path,
    labels_path: Path,
    student: Mapping[str, Any],
    current_config: Path,
    topology: Path,
    confidence_threshold: float,
    ambiguity_margin: float,
) -> List[Dict[str, Any]]:
    runtime = CurrentStateTrafficPerceptionRuntime(
        data_path=data_path,
        rule_config_path=current_config,
        topology_path=topology,
        split=split,
        top_k=10,
    )
    enable_numpy_pickle_compatibility()
    with np.load(labels_path, allow_pickle=True) as labels:
        node_labels = labels["{}_node_label".format(split)]
        region_labels = labels["{}_region_label".format(split)]
        label_partitions = [
            [int(value) for value in part] for part in labels["partitions"].tolist()
        ]
    if label_partitions != runtime.partitions:
        raise ValueError("risk-label partitions differ from current-state partitions")
    rows = []
    for sample_id in range(runtime.sample_count):
        perception = runtime.infer_sample(sample_id)
        for current_event in perception.events:
            partition_id = int(current_event["partition_id"])
            student_decision, student_confidence, _ = predict_student(
                current_event, student
            )
            rule_decision = str(rule_teacher_decision(current_event)["decision"])
            truth_event = build_future_truth_event(
                current_event,
                node_labels[sample_id],
                int(region_labels[sample_id, partition_id]),
            )
            reference = str(rule_teacher_decision(truth_event)["decision"])
            prediction_set_size = ambiguity_size(current_event, ambiguity_margin)
            for network_status in NETWORK_STATES:
                rows.append(
                    build_row(
                        current_event,
                        split,
                        student_decision,
                        student_confidence,
                        rule_decision,
                        reference,
                        network_status,
                        prediction_set_size,
                        confidence_threshold,
                        truth_event,
                    )
                )
    return rows


def _interleave_networks(rows: Sequence[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pools[str(row["network_status"])].append(row)
    for values in pools.values():
        rng.shuffle(values)
        # ``pop()`` 从末尾取样，因此把困难样本排在末尾，确保优先入选。
        values.sort(key=lambda row: bool(row["hard_sample"]))
    output = []
    while any(pools.values()):
        for network in NETWORK_STATES:
            if pools[network]:
                output.append(pools[network].pop())
    return output


def select_balanced_training(
    rows: Sequence[Dict[str, Any]], per_class: int, seed: int
) -> List[Dict[str, Any]]:
    if per_class <= 0:
        raise ValueError("train_per_class must be positive")
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["target"])].append(row)
    rng = random.Random(seed)
    selected = []
    for target in sorted(ACTION_TOKEN_CODES.values()):
        pool = _interleave_networks(grouped.get(target, []), rng)
        if not pool:
            raise ValueError("training split has no target {}".format(target))
        chosen = [dict(pool[index % len(pool)]) for index in range(per_class)]
        for repeat_index, row in enumerate(chosen):
            row["training_repeat"] = repeat_index // len(pool)
        selected.extend(chosen)
    rng.shuffle(selected)
    return selected


def evenly_spaced_groups(rows: Sequence[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    sample_ids = sorted({int(row["sample_id"]) for row in rows})
    if count <= 0 or count >= len(sample_ids):
        chosen = set(sample_ids)
    else:
        positions = np.linspace(0, len(sample_ids) - 1, num=count, dtype=np.int64)
        chosen = {sample_ids[int(position)] for position in positions.tolist()}
    return [row for row in rows if int(row["sample_id"]) in chosen]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(
                json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def row_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "rows": len(rows),
        "sample_groups": len({int(row["sample_id"]) for row in rows}),
        "unique_source_events": len({str(row["source_event_id"]) for row in rows}),
        "target_counts": dict(sorted(Counter(str(row["target"]) for row in rows).items())),
        "network_counts": dict(
            sorted(Counter(str(row["network_status"]) for row in rows).items())
        ),
        "hard_rows": sum(bool(row["hard_sample"]) for row in rows),
        "student_wrong_rows": sum(
            "student_wrong" in row["outcome_reasons"] for row in rows
        ),
        "rule_wrong_rows": sum(
            "rule_wrong" in row["outcome_reasons"] for row in rows
        ),
        "student_rule_disagreement_rows": sum(
            "student_rule_disagreement" in row["hard_reasons"] for row in rows
        ),
    }


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data_npz)
    labels_path = resolve_path(args.risk_labels)
    student_path = resolve_path(args.student_model)
    current_config = resolve_path(args.current_state_config)
    topology = resolve_path(args.topology)
    output_dir = resolve_path(args.output_dir)
    student = load_student_model(student_path)

    source_rows = {
        split: rows_for_split(
            split,
            data_path,
            labels_path,
            student,
            current_config,
            topology,
            args.student_confidence_threshold,
            args.ambiguity_margin,
        )
        for split in ("train", "val", "test")
    }
    datasets = {
        "train": select_balanced_training(
            source_rows["train"], args.train_per_class, args.seed
        ),
        "val": evenly_spaced_groups(source_rows["val"], args.val_sample_groups),
        "test": evenly_spaced_groups(source_rows["test"], args.test_sample_groups),
    }
    hard_test = [row for row in datasets["test"] if row["hard_sample"]]
    easy_test = [row for row in datasets["test"] if not row["hard_sample"]]
    online_test = [
        row for row in datasets["test"] if row["network_status"] == "normal"
    ]
    for name, rows in datasets.items():
        write_jsonl(output_dir / (name + ".jsonl"), rows)
    write_jsonl(output_dir / "hard_test.jsonl", hard_test)
    write_jsonl(output_dir / "easy_test.jsonl", easy_test)
    write_jsonl(output_dir / "online_test.jsonl", online_test)

    summary = {
        "task": "edge_qwen_current_state_future_grounded_dataset_v2",
        "input_contract": {
            "context_encoder": "freeway-routing-context-decimal@v2",
            "prompt_format": "raw_task",
            "prompt_characters": 16,
            "output": "single token A-F",
            "inputs": [
                "current_observed_12_step_traffic_state",
                "student_decision_and_confidence",
                "rule_decision",
                "prediction_set_ambiguity",
                "network_status",
            ],
        },
        "target_contract": {
            "normal_or_weak": "future observed FCM risk -> deterministic policy action",
            "offline": "future action with cloud-required E/F replaced by safe local action",
            "test_split_used_for_training": False,
        },
        "inputs": {
            "data_npz": str(data_path),
            "data_sha256": file_sha256(data_path),
            "risk_labels": str(labels_path),
            "risk_labels_sha256": file_sha256(labels_path),
            "student_model": str(student_path),
            "student_model_sha256": file_sha256(student_path),
            "current_state_config": str(current_config),
            "current_state_config_sha256": file_sha256(current_config),
        },
        "selection": {
            "train_per_class": args.train_per_class,
            "val_sample_groups": args.val_sample_groups,
            "test_sample_groups": args.test_sample_groups,
            "student_confidence_threshold": args.student_confidence_threshold,
            "ambiguity_margin": args.ambiguity_margin,
            "seed": args.seed,
        },
        "source": {
            split: row_summary(rows) for split, rows in source_rows.items()
        },
        "datasets": {
            **{name: row_summary(rows) for name, rows in datasets.items()},
            "hard_test": row_summary(hard_test),
            "easy_test": row_summary(easy_test),
            "online_test": row_summary(online_test),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
