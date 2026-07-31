"""Evaluate four Edge-Qwen roles against Rule and Student on the strict holdout."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from cloud_edge_framework.plugins.traffic import TrafficPlugin
from edge_llm_factory.runtime import ActionDecoder
from traffic_system.decision_utils import (
    ACTION_TOKEN_TO_DECISION,
    DECISION_CLASSES,
    extract_feature_vector,
    load_json,
    read_jsonl,
    rule_teacher_decision,
    save_json,
)
from traffic_system.defer_gate import (
    GATE_CLASSES,
    build_gate_features,
    load_defer_gate,
    predict_defer_gate,
)
from traffic_system.evaluate_conformal_scheduler import CRITICAL_DECISIONS
from traffic_system.evaluate_future_truth_policy import classification_report
from traffic_system.scene_event import traffic_envelope_from_output


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLE_ORDER = (
    "local_rule_student_expert",
    "edge_qwen_raw_primary",
    "edge_qwen_safety_decoded",
    "edge_qwen_safety_selective_cloud",
)
REFERENCE_ORDER = ("rule_only", "student_only")
DISPLAY_NAMES = {
    "rule_only": "仅规则策略",
    "student_only": "仅交通专用小模型",
    "local_rule_student_expert": "规则/交通专用小模型本地专家",
    "edge_qwen_raw_primary": "边缘0.8B大模型原始主判",
    "edge_qwen_safety_decoded": "边缘0.8B大模型加安全解码",
    "edge_qwen_safety_selective_cloud": "边缘0.8B大模型加安全解码和选择性云复核",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict_test_jsonl",
        default="datasets/llm_sft_freeway_action_token_v9/test.jsonl",
    )
    parser.add_argument(
        "--qwen_result_json",
        default="results/llm/llama_cpp_v9_text_only_q6_nocache_jetson02_gpu_test.json",
    )
    parser.add_argument(
        "--retention_json",
        default="results/decision/edge_cloud_model_retention.json",
    )
    parser.add_argument(
        "--selective_latency_json",
        default=(
            "results/decision/"
            "edge_qwen_selective_collaboration_v9_text_only_q6_gpu_jetson02_test.json"
        ),
    )
    parser.add_argument("--defer_gate", default="models/edge_defer_gate.npz")
    parser.add_argument(
        "--base_manifest",
        default="deployment/edge_llm/base/qwen35_0_8b_text/base_manifest.json",
    )
    parser.add_argument(
        "--action_mapping",
        default="deployment/edge_llm/scenes/freeway_traffic_v9/action_mapping.json",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_json",
        default="results/research/edge_qwen_role_ablation_20260727.json",
    )
    parser.add_argument(
        "--report_md",
        default="results/research/edge_qwen_role_ablation_20260727.md",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _confusion(references: Sequence[str], predictions: Sequence[str]) -> np.ndarray:
    class_to_index = {name: index for index, name in enumerate(DECISION_CLASSES)}
    matrix = np.zeros((len(DECISION_CLASSES), len(DECISION_CLASSES)), dtype=np.int64)
    for reference, prediction in zip(references, predictions):
        matrix[class_to_index[reference], class_to_index[prediction]] += 1
    return matrix


def _metrics(references: Sequence[str], predictions: Sequence[str]) -> Dict[str, float]:
    matrix = _confusion(references, predictions).astype(np.float64)
    total = max(1.0, float(matrix.sum()))
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    denominator = support + predicted
    f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    present = support > 0
    critical_reference = np.asarray(
        [value in CRITICAL_DECISIONS for value in references], dtype=bool
    )
    critical_prediction = np.asarray(
        [value in CRITICAL_DECISIONS for value in predictions], dtype=bool
    )
    return {
        "accuracy": float(np.trace(matrix) / total),
        "macro_f1_present_classes": (
            float(np.mean(f1[present])) if np.any(present) else 0.0
        ),
        "critical_intervention_recall": (
            float(np.mean(critical_prediction[critical_reference]))
            if np.any(critical_reference)
            else 0.0
        ),
    }


def _interval(values: np.ndarray) -> Dict[str, float]:
    return {
        "lower": round(float(np.quantile(values, 0.025)), 6),
        "upper": round(float(np.quantile(values, 0.975)), 6),
    }


def _grouped_bootstrap(
    sample_ids: Sequence[int],
    future_references: Sequence[str],
    teacher_references: Sequence[str],
    predictions: Mapping[str, Sequence[str]],
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    group_values = np.asarray(sample_ids, dtype=np.int64)
    unique_groups = np.unique(group_values)
    indices_by_group = {
        int(group): np.flatnonzero(group_values == group) for group in unique_groups
    }
    future = np.asarray(future_references, dtype=object)
    teacher = np.asarray(teacher_references, dtype=object)
    prediction_arrays = {
        name: np.asarray(values, dtype=object) for name, values in predictions.items()
    }
    metric_names = (
        "accuracy",
        "macro_f1_present_classes",
        "critical_intervention_recall",
    )
    estimates = {
        name: {
            "future_proxy": {
                metric: np.zeros(samples, dtype=np.float64)
                for metric in metric_names
            },
            "teacher_fidelity": {
                metric: np.zeros(samples, dtype=np.float64)
                for metric in metric_names
            },
        }
        for name in prediction_arrays
    }
    rng = np.random.default_rng(seed)
    for bootstrap_index in range(samples):
        selected_groups = rng.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        selected_indices = np.concatenate(
            [indices_by_group[int(group)] for group in selected_groups]
        )
        for name, values in prediction_arrays.items():
            future_metrics = _metrics(
                future[selected_indices].tolist(),
                values[selected_indices].tolist(),
            )
            teacher_metrics = _metrics(
                teacher[selected_indices].tolist(),
                values[selected_indices].tolist(),
            )
            for metric in metric_names:
                estimates[name]["future_proxy"][metric][bootstrap_index] = (
                    future_metrics[metric]
                )
                estimates[name]["teacher_fidelity"][metric][bootstrap_index] = (
                    teacher_metrics[metric]
                )

    intervals: Dict[str, Any] = {}
    comparison_names = ("rule_only", "student_only", "local_rule_student_expert")
    for name in prediction_arrays:
        intervals[name] = {}
        for reference_kind in ("future_proxy", "teacher_fidelity"):
            intervals[name][reference_kind] = {
                metric: _interval(
                    estimates[name][reference_kind][metric]
                )
                for metric in metric_names
            }
            intervals[name][reference_kind]["paired_delta"] = {}
            for baseline in comparison_names:
                intervals[name][reference_kind]["paired_delta"][baseline] = {}
                for metric in metric_names:
                    delta = (
                        estimates[name][reference_kind][metric]
                        - estimates[baseline][reference_kind][metric]
                    )
                    intervals[name][reference_kind]["paired_delta"][baseline][
                        metric
                    ] = {
                        **_interval(delta),
                        "probability_positive": round(
                            float(np.mean(delta > 0.0)), 6
                        ),
                    }
    return {
        "method": "sample_id_grouped_nonparametric_bootstrap",
        "bootstrap_samples": samples,
        "group_key": "sample_id",
        "group_count": int(len(unique_groups)),
        "regions_kept_together": True,
        "seed": seed,
        "intervals": intervals,
    }


def _latency_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(array)), 6),
        "p50_ms": round(float(np.quantile(array, 0.50)), 6),
        "p95_ms": round(float(np.quantile(array, 0.95)), 6),
        "max_ms": round(float(np.max(array)), 6),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# 边缘0.8B大模型四组角色消融",
        "",
        "> 严格时序留出集仅 36 个事件、9 个时间组。未来代理参考不是人工交通控制真值，",
        "> 云复核组使用缓存的安全约束Qwen3.5-9B参考结果，不是本次实时网络调用。",
        "",
        "| 方案 | 未来代理准确率（95%置信区间） | 宏平均F1 | 云端9B参考复现率 | 边缘大模型调用率 | 安全回退率 | 云复核率 | 决策层均值/第95百分位时延 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in (*REFERENCE_ORDER, *ROLE_ORDER):
        item = report["methods"][name]
        interval = report["bootstrap"]["intervals"][name]["future_proxy"]["accuracy"]
        lines.append(
            "| {} | {:.2%} [{:.2%}, {:.2%}] | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2f}/{:.2f} ms |".format(
                DISPLAY_NAMES[name],
                item["future_proxy"]["accuracy"],
                interval["lower"],
                interval["upper"],
                item["future_proxy"]["macro_f1_present_classes"],
                item["teacher_fidelity"]["accuracy"],
                item["qwen_invocation_rate"],
                item["safety_fallback_rate"],
                item["cloud_review_rate"],
                item["decision_layer_latency"]["mean_ms"],
                item["decision_layer_latency"]["p95_ms"],
            )
        )
    conclusion = report["conclusion"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- {}".format(conclusion["direct_answer"]),
            "- {}".format(conclusion["statistical_boundary"]),
            "- {}".format(conclusion["recommended_role"]),
            "",
            "## 四组定义",
            "",
            "1. 本地专家：冻结的学习式让权门控在规则与交通专用小模型中选择；若门控要求上云但本组强制本地，则回退规则。",
            "2. 边缘0.8B大模型原始主判：直接采用模型输出的合法单动作标记，不做动作授权校验。",
            "3. 边缘0.8B大模型加安全解码：检查风险范围、候选动作和云依赖；失败时回到本地专家。",
            "4. 安全解码加选择性云复核：遇到需云动作、安全回退、模型分歧或门控让权时，采用缓存的云端9B复核结果。",
            "",
            "决策层时延来自同一Jetson02设备六位量化图形处理器留出运行的逐事件记录；安全解码开销未单独测量，",
            "云复核按异步最终结果统计，未叠加到边缘即时响应时延。实验室仍需重新实测完整三进程闭环。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    strict_rows = read_jsonl(resolve_path(args.strict_test_jsonl))
    retention = load_json(resolve_path(args.retention_json))
    qwen_result = load_json(resolve_path(args.qwen_result_json))
    selective_latency = load_json(resolve_path(args.selective_latency_json))
    gate = load_defer_gate(resolve_path(args.defer_gate))
    decoder = ActionDecoder(
        load_json(resolve_path(args.base_manifest)),
        load_json(resolve_path(args.action_mapping)),
    )
    plugin = TrafficPlugin()

    retention_by_event = {
        str(record["event_id"]): record for record in retention["records"]
    }
    qwen_by_event = {
        str(record["event_id"]): record for record in qwen_result["samples"]
    }
    latency_by_event = {
        str(record["event_id"]): record
        for record in selective_latency["records"]
    }
    strict_ids = [str(row["event_id"]) for row in strict_rows]
    for name, values in (
        ("retention", retention_by_event),
        ("Qwen", qwen_by_event),
        ("latency", latency_by_event),
    ):
        if set(values) != set(strict_ids):
            raise ValueError("{} evidence does not match strict holdout IDs".format(name))

    methods: Dict[str, List[str]] = {
        name: [] for name in (*REFERENCE_ORDER, *ROLE_ORDER)
    }
    future_references: List[str] = []
    teacher_references: List[str] = []
    sample_ids: List[int] = []
    qwen_latencies: List[float] = []
    student_latencies: List[float] = []
    safety_fallbacks: List[bool] = []
    cloud_reviews: List[bool] = []
    records: List[Dict[str, Any]] = []

    for row in strict_rows:
        event_id = str(row["event_id"])
        record = retention_by_event[event_id]
        native_event = load_json(resolve_path(str(row["event_path"])))
        rule = rule_teacher_decision(
            native_event, decision_source="role_ablation_rule"
        )
        rule_decision = str(rule["decision"])
        student_decision = str(record["realtime_student"])
        student_confidence = float(record["realtime_student_confidence"])
        vector, feature_names = extract_feature_vector(native_event)
        if list(feature_names) != list(gate["base_feature_names"]):
            raise ValueError("defer gate feature schema mismatch")
        gate_features = build_gate_features(
            np.asarray([vector], dtype=np.float64),
            np.asarray(
                [DECISION_CLASSES.index(rule_decision)], dtype=np.int64
            ),
            np.asarray(
                [DECISION_CLASSES.index(student_decision)], dtype=np.int64
            ),
            np.asarray([student_confidence], dtype=np.float64),
        )
        gate_choice_ids, gate_confidences = predict_defer_gate(gate_features, gate)
        gate_choice = GATE_CLASSES[int(gate_choice_ids[0])]
        local_decision = (
            student_decision if gate_choice == "edge_student" else rule_decision
        )

        qwen_sample = qwen_by_event[event_id]
        qwen_token = str(qwen_sample["prediction"]).strip().upper()
        qwen_raw = ACTION_TOKEN_TO_DECISION[qwen_token]
        semantic_event = plugin.normalize(
            traffic_envelope_from_output(native_event)
        )
        decoded = decoder.decode(
            qwen_token,
            semantic_event.to_dict(include_scene_payload=False),
            network_available=True,
        )
        safety_fallback = bool(decoded["safety_fallback"]) or str(
            decoded["decision"]
        ) in {"abstain", "request_cloud"}
        qwen_safe = (
            local_decision if safety_fallback else str(decoded["decision"])
        )
        cloud_review = bool(
            decoded["requires_cloud"]
            or safety_fallback
            or qwen_safe != local_decision
            or gate_choice == "defer_cloud"
        )
        cloud_final = (
            str(record["cloud_qwen_9b"]) if cloud_review else qwen_safe
        )

        qwen_latency = float(qwen_sample["total_latency_ms"])
        combined_latency = float(
            latency_by_event[event_id]["estimated_latency_ms"]
        )
        student_latency = max(
            0.0,
            combined_latency
            - (
                qwen_latency
                if bool(latency_by_event[event_id]["qwen_invoked"])
                else 0.0
            ),
        )

        future = str(record["future_proxy_reference"])
        teacher = str(record["cloud_qwen_9b"])
        future_references.append(future)
        teacher_references.append(teacher)
        sample_ids.append(int(record["sample_id"]))
        qwen_latencies.append(qwen_latency)
        student_latencies.append(student_latency)
        safety_fallbacks.append(safety_fallback)
        cloud_reviews.append(cloud_review)

        predictions = {
            "rule_only": rule_decision,
            "student_only": student_decision,
            "local_rule_student_expert": local_decision,
            "edge_qwen_raw_primary": qwen_raw,
            "edge_qwen_safety_decoded": qwen_safe,
            "edge_qwen_safety_selective_cloud": cloud_final,
        }
        for name, prediction in predictions.items():
            methods[name].append(prediction)
        records.append(
            {
                "event_id": event_id,
                "sample_id": int(record["sample_id"]),
                "partition_id": int(record["partition_id"]),
                "future_proxy_reference": future,
                "cloud_qwen_9b": teacher,
                "rule_only": rule_decision,
                "student_only": student_decision,
                "student_confidence": student_confidence,
                "gate_choice": gate_choice,
                "gate_confidence": round(float(gate_confidences[0]), 6),
                "local_rule_student_expert": local_decision,
                "edge_qwen_token": qwen_token,
                "edge_qwen_raw_primary": qwen_raw,
                "decoder": decoded,
                "edge_qwen_safety_decoded": qwen_safe,
                "cloud_review_requested": cloud_review,
                "edge_qwen_safety_selective_cloud": cloud_final,
                "student_latency_ms": round(student_latency, 6),
                "qwen_latency_ms": round(qwen_latency, 6),
            }
        )

    count = len(strict_rows)
    fallback_rate = float(np.mean(safety_fallbacks))
    cloud_rate = float(np.mean(cloud_reviews))
    result_methods: Dict[str, Any] = {}
    for name in (*REFERENCE_ORDER, *ROLE_ORDER):
        qwen_rate = 0.0 if name in {
            "rule_only",
            "student_only",
            "local_rule_student_expert",
        } else 1.0
        method_fallback_rate = (
            fallback_rate
            if name in {
                "edge_qwen_safety_decoded",
                "edge_qwen_safety_selective_cloud",
            }
            else 0.0
        )
        method_cloud_rate = (
            cloud_rate
            if name == "edge_qwen_safety_selective_cloud"
            else 0.0
        )
        latency = (
            student_latencies
            if qwen_rate == 0.0
            else (
                qwen_latencies
                if name == "edge_qwen_raw_primary"
                else [
                    student + qwen
                    for student, qwen in zip(
                        student_latencies, qwen_latencies
                    )
                ]
            )
        )
        result_methods[name] = {
            "label": DISPLAY_NAMES[name],
            "future_proxy": {
                **{
                    key: round(value, 6)
                    for key, value in _metrics(
                        future_references, methods[name]
                    ).items()
                },
                "classification": classification_report(
                    future_references, methods[name], DECISION_CLASSES
                ),
            },
            "teacher_fidelity": {
                **{
                    key: round(value, 6)
                    for key, value in _metrics(
                        teacher_references, methods[name]
                    ).items()
                },
                "classification": classification_report(
                    teacher_references, methods[name], DECISION_CLASSES
                ),
            },
            "qwen_invocation_rate": qwen_rate,
            "safety_fallback_rate": round(method_fallback_rate, 6),
            "cloud_review_rate": round(method_cloud_rate, 6),
            "decision_layer_latency": _latency_summary(latency),
            "prediction_distribution": dict(Counter(methods[name])),
        }

    bootstrap = _grouped_bootstrap(
        sample_ids,
        future_references,
        teacher_references,
        methods,
        args.bootstrap_samples,
        args.seed,
    )
    raw_delta_student = bootstrap["intervals"]["edge_qwen_raw_primary"][
        "future_proxy"
    ]["paired_delta"]["student_only"]["accuracy"]
    raw_delta_rule = bootstrap["intervals"]["edge_qwen_raw_primary"][
        "future_proxy"
    ]["paired_delta"]["rule_only"]["accuracy"]
    raw_vs_student_pp = 100.0 * (
        result_methods["edge_qwen_raw_primary"]["future_proxy"]["accuracy"]
        - result_methods["student_only"]["future_proxy"]["accuracy"]
    )
    raw_vs_rule_pp = 100.0 * (
        result_methods["edge_qwen_raw_primary"]["future_proxy"]["accuracy"]
        - result_methods["rule_only"]["future_proxy"]["accuracy"]
    )
    full_vs_local = bootstrap["intervals"][
        "edge_qwen_safety_selective_cloud"
    ]["future_proxy"]["paired_delta"]["local_rule_student_expert"]["accuracy"]

    result: Dict[str, Any] = {
        "schema_version": 1,
        "task": "edge_qwen_four_role_ablation",
        "dataset": {
            "strict_test_jsonl": args.strict_test_jsonl,
            "events": count,
            "timestamp_groups": len(set(sample_ids)),
            "strict_holdout_verified": retention["split_integrity"][
                "strict_holdout_verified"
            ],
            "future_reference": retention["dataset"]["reference_status"],
            "cloud_reference": retention["dataset"]["teacher_label_status"],
        },
        "role_definitions": {
            "local_rule_student_expert": (
                "可移植的学习式让权门控选择规则或交通专用小模型；"
                "本组不使用云端，所以要求上云时回退到规则"
            ),
            "edge_qwen_raw_primary": (
                "所有事件直接使用缓存的边缘0.8B大模型合法单动作输出，"
                "不检查动作授权"
            ),
            "edge_qwen_safety_decoded": (
                "安全解码器检查风险范围、候选动作授权和云端依赖；"
                "失败时回退到规则/交通专用小模型本地专家"
            ),
            "edge_qwen_safety_selective_cloud": (
                "动作需要云端、安全解码失败、边缘大模型与本地专家分歧，"
                "或门控要求让权时，使用缓存的云端9B最终结果"
            ),
        },
        "methods": result_methods,
        "bootstrap": bootstrap,
        "protocol_safety": {
            "qwen_valid_output_rate": float(qwen_result["valid_rate"]),
            "decoder_fallback_count": int(sum(safety_fallbacks)),
            "decoder_fallback_rate": round(fallback_rate, 6),
            "decoder_fallback_reasons": dict(
                Counter(
                    str(record["decoder"]["fallback_reason"])
                    for record in records
                    if record["decoder"]["safety_fallback"]
                )
            ),
        },
        "cloud_review_simulation": {
            "review_count": int(sum(cloud_reviews)),
            "review_rate": round(cloud_rate, 6),
            "trigger_counts_nonexclusive": {
                "decoder_requires_cloud": sum(
                    bool(record["decoder"]["requires_cloud"])
                    for record in records
                ),
                "decoder_safety_fallback": sum(
                    bool(record["decoder"]["safety_fallback"])
                    for record in records
                ),
                "qwen_local_disagreement": sum(
                    record["edge_qwen_safety_decoded"]
                    != record["local_rule_student_expert"]
                    for record in records
                ),
                "defer_gate": sum(
                    record["gate_choice"] == "defer_cloud"
                    for record in records
                ),
            },
            "status": (
                "counterfactual replay with cached Qwen3.5-9B teacher outputs; "
                "not a new live cloud latency measurement"
            ),
        },
        "latency_evidence": {
            "source": args.qwen_result_json,
            "runtime": qwen_result["runtime"],
            "qwen_ttft": qwen_result["ttft"],
            "peak_server_pss_mb": qwen_result["peak_server_pss_mb"],
            "scope": (
                "decision-layer cached Jetson02 timings; safety decoder overhead "
                "and asynchronous cloud eventual latency are excluded"
            ),
        },
        "conclusion": {
            "direct_answer": (
                "边缘0.8B大模型原始主判相对交通专用小模型的未来代理准确率提高"
                "{:+.2f} 个百分点，相对规则 {:+.2f} 个百分点；"
                "但它与交通专用小模型的云端9B参考复现率同为{:.2%}。".format(
                    raw_vs_student_pp,
                    raw_vs_rule_pp,
                    result_methods["edge_qwen_raw_primary"][
                        "teacher_fidelity"
                    ]["accuracy"],
                )
            ),
            "statistical_boundary": (
                "按同一时间编号分组重复重采样，边缘大模型与交通专用小模型的准确率差值"
                "95%置信区间为[{:+.2f}, {:+.2f}]个百分点，与规则的差值为"
                "[{:+.2f}, {:+.2f}]；完整角色相对本地专家为"
                "[{:+.2f}, {:+.2f}]。36 条数据不足以宣称稳定显著优越。".format(
                    100.0 * raw_delta_student["lower"],
                    100.0 * raw_delta_student["upper"],
                    100.0 * raw_delta_rule["lower"],
                    100.0 * raw_delta_rule["upper"],
                    100.0 * full_vs_local["lower"],
                    100.0 * full_vs_local["upper"],
                )
            ),
            "recommended_role": (
                "现有证据支持把边缘0.8B大模型定位为“受约束的局部语义决策/复核者”，"
                "不支持它替换规则、交通专用小模型或云端；安全解码和让权机制必须保留。"
            ),
        },
        "records": records,
    }
    output = resolve_path(args.output_json)
    markdown = resolve_path(args.report_md)
    save_json(result, output)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "events": count,
                "groups": len(set(sample_ids)),
                "methods": {
                    name: {
                        "future_accuracy": result_methods[name][
                            "future_proxy"
                        ]["accuracy"],
                        "teacher_fidelity": result_methods[name][
                            "teacher_fidelity"
                        ]["accuracy"],
                        "qwen_rate": result_methods[name][
                            "qwen_invocation_rate"
                        ],
                        "fallback_rate": result_methods[name][
                            "safety_fallback_rate"
                        ],
                        "cloud_rate": result_methods[name][
                            "cloud_review_rate"
                        ],
                    }
                    for name in (*REFERENCE_ORDER, *ROLE_ORDER)
                },
                "conclusion": result["conclusion"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("metrics:", output)
    print("report:", markdown)


if __name__ == "__main__":
    main()
