"""用途：汇总 Jetson 闭环在正常、弱网和断网条件下的真实故障注入结果。"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence

from traffic_system.benchmark_utils import summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize adaptive network resilience results.")
    parser.add_argument(
        "--normal",
        default="results/edge/continuous_v1_6_normal_1000runs.json",
    )
    parser.add_argument(
        "--mild",
        default="results/edge/continuous_v1_6_fault_mild_adaptive_200runs.json",
    )
    parser.add_argument(
        "--severe",
        default="results/edge/continuous_v1_6_fault_severe_adaptive_200runs.json",
    )
    parser.add_argument(
        "--outage",
        default="results/edge/continuous_v1_6_fault_outage_adaptive_200runs.json",
    )
    parser.add_argument(
        "--severe_fixed",
        default="results/edge/continuous_v1_6_fault_severe_fixed_sync_100runs.json",
    )
    parser.add_argument(
        "--outage_fixed",
        default="results/edge/continuous_v1_6_fault_outage_fixed_sync_20runs.json",
    )
    parser.add_argument(
        "--mild_proxy_log",
        default="results/edge/network_fault_proxy_v1_6_mild.jsonl",
    )
    parser.add_argument(
        "--severe_proxy_log",
        default="results/edge/network_fault_proxy_v1_6_severe_fixed_final.jsonl",
    )
    parser.add_argument(
        "--outage_proxy_log",
        default="results/edge/network_fault_proxy_v1_6_outage_fixed_final.jsonl",
    )
    parser.add_argument(
        "--output_json",
        default="results/edge/network_resilience_v1_6_summary.json",
    )
    parser.add_argument(
        "--report_md",
        default="results/edge/network_resilience_v1_6_summary.md",
    )
    parser.add_argument("--policy", default="deployment/policy/current_policy.json")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: {}".format(path))
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def profile_summary(result: Dict[str, Any], evidence: Path) -> Dict[str, Any]:
    latency = result["latency"]["total_e2e"]
    return {
        "runs": int(result["runs"]),
        "success_rate": float(result["success_rate"]),
        "average_e2e_ms": float(latency["average_ms"]),
        "p95_e2e_ms": float(latency["p95_ms"]),
        "p99_e2e_ms": float(latency["p99_ms"]),
        "under_200ms_rate": float(result["under_200ms_rate"]),
        "meets_200ms_average": bool(result["meets_0_2s_average"]),
        "decision_accuracy": float(result["decision_accuracy"]),
        "business_availability": float(result["business_availability"]),
        "cloud_request_rate": float(result["cloud_request_rate"]),
        "cloud_success_rate_when_attempted": result["cloud_success_rate"],
        "autonomy_trigger_rate": float(result["autonomy_trigger_rate"]),
        "route_counts": result["route_counts"],
        "pending_cloud_review_count": int(result["pending_cloud_review_count"]),
        "process_max_rss_mb": float(result["process_max_rss_mb"]),
        "evidence": str(evidence),
    }


def proxy_summary(path: Path) -> Dict[str, Any]:
    rows = read_jsonl(path)
    dropped = [row for row in rows if row.get("dropped")]
    delivered = [row for row in rows if not row.get("dropped")]
    delays = [float(row.get("delay_ms", 0.0)) for row in delivered]
    return {
        "requests": len(rows),
        "delivered_requests": len(delivered),
        "dropped_requests": len(dropped),
        "observed_loss_rate": round(len(dropped) / len(rows), 6) if rows else 0.0,
        "delivered_average_injected_delay_ms": round(statistics.fmean(delays), 6)
        if delays
        else 0.0,
        "evidence": str(path),
    }


def signature(rows: Sequence[Dict[str, Any]]) -> List[List[int]]:
    return [[int(row["sample_id"]), int(row["partition_id"])] for row in rows]


def paired_comparison(
    fixed: Dict[str, Any], adaptive: Dict[str, Any], name: str
) -> Dict[str, Any]:
    fixed_rows = fixed.get("samples", [])
    adaptive_rows = adaptive.get("samples", [])[: len(fixed_rows)]
    if not fixed_rows or signature(fixed_rows) != signature(adaptive_rows):
        raise ValueError("{} fixed/adaptive samples are not paired.".format(name))
    fixed_latency = [float(row["total_e2e_latency_ms"]) for row in fixed_rows]
    adaptive_latency = [float(row["total_e2e_latency_ms"]) for row in adaptive_rows]
    fixed_summary = summarize(fixed_latency)
    adaptive_summary = summarize(adaptive_latency)
    fixed_average = fixed_summary["average_ms"]
    adaptive_average = adaptive_summary["average_ms"]
    fixed_p95 = fixed_summary["p95_ms"]
    adaptive_p95 = adaptive_summary["p95_ms"]
    count = len(fixed_rows)
    return {
        "paired_cases": count,
        "same_ordered_samples": True,
        "fixed_sync": {
            "average_e2e_ms": fixed_average,
            "p95_e2e_ms": fixed_p95,
            "under_200ms_rate": round(sum(value <= 200.0 for value in fixed_latency) / count, 6),
            "decision_accuracy": round(
                sum(bool(row["functional_decision"]) for row in fixed_rows) / count, 6
            ),
            "business_availability": round(
                sum(bool(row["basic_business_functional"]) for row in fixed_rows) / count, 6
            ),
        },
        "adaptive": {
            "average_e2e_ms": adaptive_average,
            "p95_e2e_ms": adaptive_p95,
            "under_200ms_rate": round(sum(value <= 200.0 for value in adaptive_latency) / count, 6),
            "decision_accuracy": round(
                sum(bool(row["functional_decision"]) for row in adaptive_rows) / count, 6
            ),
            "business_availability": round(
                sum(bool(row["basic_business_functional"]) for row in adaptive_rows) / count, 6
            ),
        },
        "improvement": {
            "average_latency_reduction_percent": round(
                (fixed_average - adaptive_average) / fixed_average * 100.0, 6
            ),
            "p95_latency_reduction_percent": round(
                (fixed_p95 - adaptive_p95) / fixed_p95 * 100.0, 6
            ),
        },
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    labels = {"normal": "正常网络", "mild": "轻度抖动", "severe": "严重弱网", "outage": "完全断网"}
    lines = [
        "# Jetson02 网络韧性实测",
        "",
        "参考决策由未来真实 flow、occupancy、speed 经冻结 FCM 风险标签和固定安全策略生成。",
        "",
        "| 网络状态 | 次数 | 平均闭环 | P95 | <=200ms | 决策准确率 | 业务可用率 | 路由 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for profile in ("normal", "mild", "severe", "outage"):
        item = result["adaptive_profiles"][profile]
        lines.append(
            "| {} | {} | {:.2f} ms | {:.2f} ms | {:.2%} | {:.2%} | {:.2%} | {} |".format(
                labels[profile],
                item["runs"],
                item["average_e2e_ms"],
                item["p95_e2e_ms"],
                item["under_200ms_rate"],
                item["decision_accuracy"],
                item["business_availability"],
                json.dumps(item["route_counts"], ensure_ascii=False, separators=(",", ":")),
            )
        )
    lines.extend(["", "## 固定同步对照", ""])
    for profile in ("severe", "outage"):
        item = result["paired_fixed_vs_adaptive"][profile]
        lines.append(
            "- {}：{} 个同序样本，平均时延 {:.2f} -> {:.2f} ms，P95 {:.2f} -> {:.2f} ms，平均时延降低 {:.2f}%。".format(
                labels[profile],
                item["paired_cases"],
                item["fixed_sync"]["average_e2e_ms"],
                item["adaptive"]["average_e2e_ms"],
                item["fixed_sync"]["p95_e2e_ms"],
                item["adaptive"]["p95_e2e_ms"],
                item["improvement"]["average_latency_reduction_percent"],
            )
        )
    lines.extend(
        [
            "",
            "业务可用率只表示返回了安全且协议有效的本地或云端决策；它不等同于决策准确率。风险参考是数据驱动参考，不是人工交通控制真值。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = {
        "normal": Path(args.normal),
        "mild": Path(args.mild),
        "severe": Path(args.severe),
        "outage": Path(args.outage),
    }
    adaptive = {name: load_json(path) for name, path in paths.items()}
    severe_fixed = load_json(Path(args.severe_fixed))
    outage_fixed = load_json(Path(args.outage_fixed))
    profiles = {name: profile_summary(adaptive[name], paths[name]) for name in paths}
    comparisons = {
        "severe": paired_comparison(severe_fixed, adaptive["severe"], "severe"),
        "outage": paired_comparison(outage_fixed, adaptive["outage"], "outage"),
    }
    output = {
        "task": "measured_edge_cloud_network_resilience",
        "device": "Jetson02",
        "policy_version": str(load_json(Path(args.policy))["policy_version"]),
        "reference": "future observed flow/occupancy/speed -> frozen FCM risk -> fixed safety policy",
        "adaptive_profiles": profiles,
        "paired_fixed_vs_adaptive": comparisons,
        "fault_proxy_observations": {
            "mild": proxy_summary(Path(args.mild_proxy_log)),
            "severe_fixed": proxy_summary(Path(args.severe_proxy_log)),
            "outage_fixed": proxy_summary(Path(args.outage_proxy_log)),
            "severe_adaptive_http_requests": int(adaptive["severe"]["cloud_attempt_count"]),
            "outage_adaptive_http_requests": int(adaptive["outage"]["cloud_attempt_count"]),
        },
        "acceptance": {
            "normal_long_run_zero_failures": int(adaptive["normal"]["failure_count"]) == 0,
            "all_adaptive_average_under_200ms": all(
                item["meets_200ms_average"] for item in profiles.values()
            ),
            "all_adaptive_business_availability_at_least_90pct": all(
                item["business_availability"] >= 0.90 for item in profiles.values()
            ),
            "severe_has_no_synchronous_cloud_wait": profiles["severe"]["route_counts"]["cloud_sync"] == 0,
            "severe_async_reviews_are_durable": (
                profiles["severe"]["route_counts"]["cloud_async"]
                == profiles["severe"]["pending_cloud_review_count"]
            ),
            "outage_is_fully_local": profiles["outage"]["route_counts"]["local_autonomy"]
            == profiles["outage"]["runs"],
            "fixed_sync_outage_misses_200ms": not bool(outage_fixed["meets_0_2s_average"]),
            "paired_samples_verified": all(
                item["same_ordered_samples"] for item in comparisons.values()
            ),
        },
        "metric_note": (
            "Business availability is protocol validity and safety, not decision accuracy. "
            "The future-state reference is data-derived rather than human control ground truth."
        ),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output, Path(args.report_md))
    print(json.dumps({"acceptance": output["acceptance"], "paired": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
