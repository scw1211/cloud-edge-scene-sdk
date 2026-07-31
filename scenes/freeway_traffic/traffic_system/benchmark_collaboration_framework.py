"""用途：在同一批交通事件上比较边缘、完整云端与自适应云边协同路径。"""

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

from sklearn.metrics import accuracy_score, f1_score

from cloud_edge_framework.evidence import EvidencePlanner
from cloud_edge_framework.registry import build_default_registry
from cloud_edge_framework.runtime import CloudRuntime
from cloud_edge_framework.scheduling import CollaborationScheduler, NetworkSnapshot
from traffic_system.decision_utils import save_json
from traffic_system.decision_utils import rule_teacher_decision
from traffic_system.evaluate_future_truth_policy import (
    load_evaluation_arrays,
    make_event,
    one_hot_probabilities,
)
from traffic_system.risk_labels import RISK_CLASSES
from traffic_system.scene_event import traffic_envelope_from_output


NETWORK_PROFILES = {
    "normal": NetworkSnapshot(True, 15.0, 3.0, 0.0, 1.0, 12.0, 100.0, 100.0, 2048),
    "mild": NetworkSnapshot(True, 45.0, 12.0, 0.01, 3.0, 15.0, 20.0, 50.0, 2048),
    "severe": NetworkSnapshot(True, 160.0, 40.0, 0.15, 8.0, 20.0, 2.0, 10.0, 2048),
    "outage": NetworkSnapshot(False, 0.0, 0.0, 1.0, 0.0, 0.0, 0.001, 0.001, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the collaboration framework.")
    parser.add_argument(
        "--labels",
        default="datasets/freeway_teacher_labels_qwen9b_safe_joint_metis4_expanded.jsonl",
    )
    parser.add_argument(
        "--data_npz",
        default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz",
    )
    parser.add_argument(
        "--risk_labels",
        default="datasets/risk_labels_pems08_metis4.npz",
    )
    parser.add_argument(
        "--output",
        default="results/framework/collaboration_framework_benchmark.json",
    )
    parser.add_argument(
        "--report",
        default="results/framework/collaboration_framework_benchmark.md",
    )
    return parser.parse_args()


def request_bytes(event: Any, include_scene_payload: bool) -> int:
    return len(
        json.dumps(
            {"event": event.to_dict(include_scene_payload=include_scene_payload)},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def quality(reference: List[str], predictions: List[str]) -> Dict[str, float]:
    return {
        "teacher_agreement": round(float(accuracy_score(reference, predictions)), 6),
        "macro_f1": round(float(f1_score(reference, predictions, average="macro")), 6),
    }


def mean(values: List[float]) -> float:
    return round(sum(values) / max(1, len(values)), 6)


def write_report(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# 组件化云边协同框架对比",
        "",
        "- 样本：{} 条事件，同时保留未来状态代理和 Qwen 策略两种参考。".format(
            result["sample_count"]
        ),
        "- 未来状态代理来自测试窗口真实 flow/occupancy/speed，不等同于现场控制收益。",
        "- 网络时延为同一调度公式下的预算仿真；真实 HTTP 回路另行实测。",
        "",
        "| 推理路径 | 未来状态代理准确率 | Macro-F1 | Qwen 策略一致率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("edge_student", "cloud_individual", "cloud_topology_fused"):
        item = result["quality"][name]
        lines.append(
            "| {} | {:.2%} | {:.2%} | {:.2%} |".format(
                name,
                item["future_state_proxy"]["teacher_agreement"],
                item["future_state_proxy"]["macro_f1"],
                item["qwen_policy"]["teacher_agreement"],
            )
        )
    lines.extend(
        [
            "",
            "完整事件平均请求：{:.1f} B；紧凑请求：{:.1f} B；减少 {:.2%}。".format(
                result["data_plane"]["full_request_bytes_mean"],
                result["data_plane"]["compact_request_bytes_mean"],
                result["data_plane"]["request_reduction_ratio"],
            ),
            "关联事件对冲突率：{:.2%}；消解后 {:.2%}；消解成功率 {:.2%}。".format(
                result["coordination"]["initial_conflicting_pair_rate"],
                result["coordination"]["residual_conflicting_pair_rate"],
                result["coordination"]["resolution_success_rate"],
            ),
            "",
            "| 网络 | 未来状态代理准确率 | 即时响应均值/ms | 同步闭环均值/ms | 路由 |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for name, item in result["adaptive"].items():
        lines.append(
            "| {} | {:.2%} | {:.3f} | {} | `{}` |".format(
                name,
                item["quality"]["future_state_proxy"]["teacher_agreement"],
                item["immediate_response_ms_mean"],
                "{:.3f}".format(item["sync_closed_loop_ms_mean"])
                if item["sync_count"]
                else "-",
                json.dumps(item["route_counts"], ensure_ascii=False, separators=(",", ":")),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    registry = build_default_registry(root)
    cloud = CloudRuntime(registry)
    cloud.warmup()
    plugin = registry.get("traffic")
    planner = EvidencePlanner()
    scheduler = CollaborationScheduler()

    rows = []
    with Path(args.labels).open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                rows.append(json.loads(line))
    source_rows = []
    partitions_by_id: Dict[int, List[int]] = {}
    for row in rows:
        event_path = Path(row["event_path"])
        if not event_path.is_absolute():
            event_path = root / event_path
        with event_path.open("r", encoding="utf-8") as file_obj:
            raw = json.load(file_obj)
        partition_id = int(raw["partition_id"])
        nodes = [int(node) for node in raw["managed_node_ids"]]
        if partition_id in partitions_by_id and partitions_by_id[partition_id] != nodes:
            raise ValueError("managed nodes changed for partition {}".format(partition_id))
        partitions_by_id[partition_id] = nodes
        source_rows.append((row, raw))
    partitions = [partitions_by_id[index] for index in sorted(partitions_by_id)]
    arrays = load_evaluation_arrays(
        Path(args.data_npz), Path(args.risk_labels), "test"
    )

    entries = []
    groups: Dict[Any, List[Any]] = defaultdict(list)
    for row, raw in source_rows:
        sample_id = int(raw["sample_id"])
        partition_id = int(raw["partition_id"])
        truth_event = make_event(
            "test",
            sample_id,
            partition_id,
            partitions,
            one_hot_probabilities(
                arrays["node_labels"][sample_id], len(RISK_CLASSES)
            ),
            one_hot_probabilities(
                arrays["region_labels"][sample_id], len(RISK_CLASSES)
            ),
            arrays["split_target"][sample_id],
            raw["control_capabilities"],
            10,
            "future_observation_fcm_reference",
        )
        event = plugin.normalize(traffic_envelope_from_output(raw))
        local_started = time.perf_counter()
        local = plugin.edge_decide(event)
        student_ms = (time.perf_counter() - local_started) * 1000.0
        event = replace(
            event,
            timing=replace(
                event.timing,
                edge_inference_ms=event.timing.edge_inference_ms + student_ms,
            ),
        )
        plan = planner.plan(event)
        selected = replace(
            event,
            evidence=[
                item for item in event.evidence if item.level in {"summary", "feature"}
            ],
        )
        compact = plugin.prepare_cloud_event(selected, "feature")
        entry = {
            "qwen_reference": str(row["teacher_decision"]["decision"]),
            "future_reference": str(rule_teacher_decision(truth_event)["decision"]),
            "event": event,
            "local": local,
            "compact": compact,
            "evidence_level": plan.required_level,
            "full_bytes": request_bytes(selected, True),
            "compact_bytes": request_bytes(compact, True),
            "student_ms": student_ms,
        }
        entries.append(entry)
        groups[raw.get("sample_id")].append(entry)

    individual = {}
    for entry in entries:
        individual[entry["event"].event_id] = cloud.decide(entry["compact"])
    fused = {}
    initial_conflicts = 0
    residual_conflicts = 0
    initial_conflicting_pairs = set()
    residual_conflicting_pairs = set()
    initial_conflict_examples = []
    residual_conflict_examples = []
    correlated_pair_count = 0
    for sample_id, group in groups.items():
        result = cloud.coordinate([entry["compact"] for entry in group])
        correlated_pair_count += len(group) * (len(group) - 1) // 2
        initial_conflicts += int(result["initial_conflict_count"])
        residual_conflicts += int(result["residual_conflict_count"])
        initial_conflicting_pairs.update(
            tuple(sorted((item["left_event_id"], item["right_event_id"])))
            for item in result["initial_conflicts"]
        )
        residual_conflicting_pairs.update(
            tuple(sorted((item["left_event_id"], item["right_event_id"])))
            for item in result["residual_conflicts"]
        )
        for conflict in result["initial_conflicts"]:
            if len(initial_conflict_examples) >= 5:
                break
            initial_conflict_examples.append(
                {
                    "sample_id": sample_id,
                    "conflict": conflict,
                    "resolution_changes": [
                        change
                        for change in result["changes"]
                        if change["conflict_id"] == conflict["conflict_id"]
                    ],
                }
            )
        for conflict in result["residual_conflicts"]:
            if len(residual_conflict_examples) >= 5:
                break
            residual_conflict_examples.append(
                {"sample_id": sample_id, "conflict": conflict}
            )
        for decision in result["decisions"]:
            fused[decision["event_ids"][0]] = decision

    qwen_references = [entry["qwen_reference"] for entry in entries]
    future_references = [entry["future_reference"] for entry in entries]
    local_predictions = [entry["local"].decision for entry in entries]
    individual_predictions = [individual[entry["event"].event_id].decision for entry in entries]
    fused_predictions = [fused[entry["event"].event_id]["decision"] for entry in entries]
    full_bytes = [entry["full_bytes"] for entry in entries]
    compact_bytes = [entry["compact_bytes"] for entry in entries]

    def quality_bundle(predictions: List[str]) -> Dict[str, Any]:
        return {
            "future_state_proxy": quality(future_references, predictions),
            "qwen_policy": quality(qwen_references, predictions),
        }

    adaptive = {}
    for profile_name, network in NETWORK_PROFILES.items():
        predictions = []
        route_counts: Counter = Counter()
        immediate_ms = []
        sync_ms = []
        review_ms = []
        deadline_met = 0
        for entry in entries:
            schedule = scheduler.schedule(
                entry["event"],
                network,
                upload_bytes=entry["compact_bytes"],
                evidence_level=entry["evidence_level"],
            )
            route_counts[schedule.route] += 1
            if schedule.route == "cloud_sync":
                predictions.append(fused[entry["event"].event_id]["decision"])
                immediate_ms.append(schedule.predicted_closed_loop_ms)
                sync_ms.append(schedule.predicted_closed_loop_ms)
                review_ms.append(schedule.predicted_closed_loop_ms)
                deadline_met += int(
                    schedule.predicted_closed_loop_ms <= entry["event"].timing.deadline_ms
                )
            else:
                predictions.append(entry["local"].decision)
                immediate_ms.append(
                    entry["event"].timing.preprocessing_ms
                    + entry["event"].timing.edge_inference_ms
                )
                if schedule.cloud_requested and network.available:
                    review_ms.append(schedule.predicted_closed_loop_ms)
        adaptive[profile_name] = {
            "quality": quality_bundle(predictions),
            "route_counts": dict(sorted(route_counts.items())),
            "immediate_response_ms_mean": mean(immediate_ms),
            "sync_closed_loop_ms_mean": mean(sync_ms),
            "async_review_completion_ms_mean": mean(review_ms),
            "sync_count": len(sync_ms),
            "deadline_success_rate_for_sync": round(
                deadline_met / max(1, len(sync_ms)), 6
            ),
        }

    output = {
        "task": "componentized_cloud_edge_framework_benchmark",
        "sample_count": len(entries),
        "references": {
            "future_state_proxy": {
                "name": "future flow/occupancy/speed -> frozen FCM -> fixed safety policy",
                "limitation": "data-driven proxy rather than field actuator ground truth",
            },
            "qwen_policy": {
                "name": "qwen9b_safety_constrained_teacher",
                "limitation": "policy fidelity metric rather than physical traffic-control ground truth",
            },
        },
        "quality": {
            "edge_student": quality_bundle(local_predictions),
            "cloud_individual": quality_bundle(individual_predictions),
            "cloud_topology_fused": quality_bundle(fused_predictions),
            "fusion_changed_decisions": sum(
                left != right
                for left, right in zip(individual_predictions, fused_predictions)
            ),
        },
        "data_plane": {
            "full_request_bytes_mean": mean([float(value) for value in full_bytes]),
            "compact_request_bytes_mean": mean([float(value) for value in compact_bytes]),
            "request_reduction_ratio": round(
                1.0 - sum(compact_bytes) / max(1, sum(full_bytes)), 6
            ),
            "edge_student_runtime_ms_mean": mean(
                [float(entry["student_ms"]) for entry in entries]
            ),
        },
        "coordination": {
            "timestamp_groups": len(groups),
            "correlated_pair_count": correlated_pair_count,
            "initial_conflicts": initial_conflicts,
            "residual_conflicts": residual_conflicts,
            "initial_conflicting_pair_count": len(initial_conflicting_pairs),
            "residual_conflicting_pair_count": len(residual_conflicting_pairs),
            "initial_conflict_examples": initial_conflict_examples,
            "residual_conflict_examples": residual_conflict_examples,
            "initial_conflicting_pair_rate": round(
                len(initial_conflicting_pairs) / max(1, correlated_pair_count), 6
            ),
            "residual_conflicting_pair_rate": round(
                len(residual_conflicting_pairs) / max(1, correlated_pair_count), 6
            ),
            "resolution_success_rate": round(
                (initial_conflicts - residual_conflicts) / initial_conflicts, 6
            )
            if initial_conflicts
            else 1.0,
            "requirements_met": {
                "conflicting_pair_rate_le_5_percent": len(initial_conflicting_pairs)
                / max(1, correlated_pair_count)
                <= 0.05,
                "resolution_success_rate_ge_90_percent": (
                    (initial_conflicts - residual_conflicts) / initial_conflicts
                    if initial_conflicts
                    else 1.0
                )
                >= 0.90,
            },
        },
        "adaptive": adaptive,
        "latency_note": "network profiles use the same closed-loop budget model; they are not HTTP measurements",
    }
    save_json(output, Path(args.output))
    write_report(output, Path(args.report))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
