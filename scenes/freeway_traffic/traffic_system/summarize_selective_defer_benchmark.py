"""用途：汇总 1.8 选择性协同在 Jetson 四档网络及同序基线上的实测结果。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from traffic_system.decision_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize selective-defer Jetson benchmarks.")
    parser.add_argument("--normal", default="results/edge/continuous_v1_8_1_normal_200runs.json")
    parser.add_argument(
        "--normal_baseline",
        default="results/edge/continuous_v1_8_1_normal_no_gate_baseline_200runs.json",
    )
    parser.add_argument("--mild", default="results/edge/continuous_v1_8_1_mild_200runs.json")
    parser.add_argument("--severe", default="results/edge/continuous_v1_8_1_severe_200runs.json")
    parser.add_argument("--outage", default="results/edge/continuous_v1_8_1_outage_100runs.json")
    parser.add_argument("--mild_proxy_log", default="results/edge/network_fault_proxy_v1_8_1_mild.jsonl")
    parser.add_argument("--policy", default="deployment/policy/current_policy.json")
    parser.add_argument("--output_json", default="results/edge/selective_defer_v1_8_1_summary.json")
    parser.add_argument("--output_md", default="results/edge/selective_defer_v1_8_1_summary.md")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: {}".format(path))
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def event_signature(result: Dict[str, Any]) -> List[Tuple[int, int]]:
    return [
        (int(row["sample_id"]), int(row["partition_id"]))
        for row in result.get("samples", [])
    ]


def summarize_profile(result: Dict[str, Any], path: Path) -> Dict[str, Any]:
    latency = result["latency"]["total_e2e"]
    gate_latency = result["latency"]["edge_defer_gate"]
    return {
        "runs": int(result["runs"]),
        "success_rate": float(result["success_rate"]),
        "average_e2e_ms": float(latency["average_ms"]),
        "p95_e2e_ms": float(latency["p95_ms"]),
        "p99_e2e_ms": float(latency["p99_ms"]),
        "under_200ms_rate": float(result["under_200ms_rate"]),
        "decision_accuracy": float(result["decision_accuracy"]),
        "business_availability": float(result["business_availability"]),
        "cloud_request_rate": float(result["cloud_request_rate"]),
        "cloud_success_rate": result["cloud_success_rate"],
        "route_counts": result["route_counts"],
        "pending_cloud_review_count": int(result["pending_cloud_review_count"]),
        "gate_average_ms": float(gate_latency["average_ms"]),
        "gate_p95_ms": float(gate_latency["p95_ms"]),
        "process_max_rss_mb": float(result["process_max_rss_mb"]),
        "evidence": str(path),
    }


def normal_comparison(
    baseline: Dict[str, Any], selective: Dict[str, Any]
) -> Dict[str, Any]:
    if event_signature(baseline) != event_signature(selective):
        raise ValueError("Normal baseline and selective runs are not event-paired")
    baseline_latency = float(baseline["latency"]["total_e2e"]["average_ms"])
    selective_latency = float(selective["latency"]["total_e2e"]["average_ms"])
    baseline_cloud = float(baseline["cloud_request_rate"])
    selective_cloud = float(selective["cloud_request_rate"])
    return {
        "paired_events": len(event_signature(selective)),
        "same_ordered_events": True,
        "baseline": {
            "accuracy": float(baseline["decision_accuracy"]),
            "cloud_request_rate": baseline_cloud,
            "average_e2e_ms": baseline_latency,
        },
        "selective_defer": {
            "accuracy": float(selective["decision_accuracy"]),
            "cloud_request_rate": selective_cloud,
            "average_e2e_ms": selective_latency,
        },
        "difference": {
            "accuracy_percentage_points": round(
                (float(selective["decision_accuracy"]) - float(baseline["decision_accuracy"]))
                * 100.0,
                6,
            ),
            "cloud_request_percentage_points": round((selective_cloud - baseline_cloud) * 100.0, 6),
            "cloud_request_relative_reduction": round(
                (baseline_cloud - selective_cloud) / baseline_cloud, 6
            )
            if baseline_cloud
            else None,
            "average_e2e_delta_ms": round(selective_latency - baseline_latency, 6),
        },
    }


def proxy_observation(path: Path) -> Dict[str, Any]:
    rows = [row for row in read_jsonl(path) if row.get("path") != "/health"]
    dropped = [row for row in rows if row.get("dropped")]
    delivered = [row for row in rows if not row.get("dropped")]
    return {
        "requests": len(rows),
        "delivered": len(delivered),
        "dropped": len(dropped),
        "observed_loss_rate": round(len(dropped) / len(rows), 6) if rows else 0.0,
        "configured_profile": "40 ms mean delay, 10 ms jitter, 1% loss",
        "evidence": str(path),
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    labels = {
        "normal": "正常网络",
        "mild": "轻度故障注入",
        "severe": "严重弱网调度",
        "outage": "完全断网",
    }
    lines = [
        "# Jetson02 选择性协同 1.8.1 实测",
        "",
        "> 决策参考由未来 flow/occupancy/speed 经冻结 FCM 标签和固定策略生成，不是人工控制真值。",
        "> 四档网络样本数不完全相同，跨档准确率不作配对比较；正常网络 A/B 使用同序 200 事件。",
        "",
        "| 网络 | 次数 | 平均 / P95 | <=200 ms | 准确率 | 可用率 | 上云率 | 路由 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in ("normal", "mild", "severe", "outage"):
        row = result["profiles"][name]
        lines.append(
            "| {} | {} | {:.2f} / {:.2f} ms | {:.2%} | {:.2%} | {:.2%} | {:.2%} | `{}` |".format(
                labels[name],
                row["runs"],
                row["average_e2e_ms"],
                row["p95_e2e_ms"],
                row["under_200ms_rate"],
                row["decision_accuracy"],
                row["business_availability"],
                row["cloud_request_rate"],
                json.dumps(row["route_counts"], ensure_ascii=False, separators=(",", ":")),
            )
        )
    paired = result["normal_paired_comparison"]
    difference = paired["difference"]
    lines.extend(
        [
            "",
            "## 正常网络同序 A/B",
            "",
            "- 准确率：{:.2%} -> {:.2%}（{:+.2f} 个百分点）。".format(
                paired["baseline"]["accuracy"],
                paired["selective_defer"]["accuracy"],
                difference["accuracy_percentage_points"],
            ),
            "- 上云率：{:.2%} -> {:.2%}（相对减少 {:.2%}）。".format(
                paired["baseline"]["cloud_request_rate"],
                paired["selective_defer"]["cloud_request_rate"],
                difference["cloud_request_relative_reduction"],
            ),
            "- 平均闭环：{:.2f} -> {:.2f} ms（{:+.2f} ms）。".format(
                paired["baseline"]["average_e2e_ms"],
                paired["selective_defer"]["average_e2e_ms"],
                difference["average_e2e_delta_ms"],
            ),
            "",
            "严重弱网没有同步云等待；异步请求落入持久队列。断网时全部由本地自治完成。",
            "业务可用率只表示返回安全且协议有效的决策，不等于决策准确率。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = {
        "normal": Path(args.normal),
        "mild": Path(args.mild),
        "severe": Path(args.severe),
        "outage": Path(args.outage),
    }
    profiles_raw = {name: load_json(path) for name, path in paths.items()}
    baseline = load_json(Path(args.normal_baseline))
    profiles = {
        name: summarize_profile(profiles_raw[name], paths[name]) for name in paths
    }
    result = {
        "task": "jetson_selective_defer_network_benchmark",
        "device": "Jetson02",
        "policy_version": str(load_json(Path(args.policy))["policy_version"]),
        "reference": {
            "source": "future flow/occupancy/speed -> frozen FCM risk -> fixed safety policy",
            "status": "data-driven proxy, not manual traffic-control ground truth",
        },
        "profiles": profiles,
        "normal_paired_comparison": normal_comparison(baseline, profiles_raw["normal"]),
        "mild_fault_proxy": proxy_observation(Path(args.mild_proxy_log)),
        "acceptance": {
            "all_runs_successful": all(row["success_rate"] == 1.0 for row in profiles.values()),
            "all_average_under_200ms": all(row["average_e2e_ms"] <= 200.0 for row in profiles.values()),
            "all_business_availability_at_least_90pct": all(
                row["business_availability"] >= 0.9 for row in profiles.values()
            ),
            "severe_has_zero_sync_wait": profiles["severe"]["route_counts"]["cloud_sync"] == 0,
            "severe_async_queue_is_durable": (
                profiles["severe"]["route_counts"]["cloud_async"]
                == profiles["severe"]["pending_cloud_review_count"]
            ),
            "outage_is_fully_local": (
                profiles["outage"]["route_counts"]["local_autonomy"]
                == profiles["outage"]["runs"]
            ),
            "normal_cloud_request_rate_reduced": (
                profiles["normal"]["cloud_request_rate"]
                < float(baseline["cloud_request_rate"])
            ),
        },
    }
    save_json(result, Path(args.output_json))
    write_markdown(result, Path(args.output_md))
    print(json.dumps(result["acceptance"], ensure_ascii=False, indent=2))
    print("summary:", args.output_json)


if __name__ == "__main__":
    main()
