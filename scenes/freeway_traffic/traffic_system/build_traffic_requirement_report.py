"""用途：汇总交通场景复测与 Jetson 实测证据，生成竞赛要求矩阵。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str) -> Dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    with path.open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def evidence(relative_path: str) -> str:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return relative_path


def percentage(value: float) -> str:
    return "{:.2f}%".format(float(value) * 100.0)


def status_label(value: str) -> str:
    return {
        "pass": "达标",
        "qualified_pass": "条件达标",
        "partial": "部分达标",
        "fail": "未达标",
        "not_tested": "未完成",
    }[value]


def build_report() -> Dict[str, Any]:
    fresh_root = "results/competition/traffic_full_retest_20260726"
    future = load_json(f"{fresh_root}/future_truth_policy.json")
    conformal = load_json(f"{fresh_root}/region_risk_conformal_eval.json")
    scheduler = load_json(f"{fresh_root}/conformal_scheduler.json")
    retention = load_json(f"{fresh_root}/edge_cloud_model_retention.json")
    conflict = load_json(f"{fresh_root}/conflict_consistency.json")
    policy = load_json(f"{fresh_root}/policy_sync_consistency.json")
    collaboration = load_json(f"{fresh_root}/collaboration_framework.json")
    online = load_json(f"{fresh_root}/framework_http_360.json")
    outage = load_json(f"{fresh_root}/framework_outage_360.json")
    recovery = load_json(f"{fresh_root}/recovery_status.json")
    monitoring = load_json(f"{fresh_root}/monitoring_hot_path.json")
    regression = load_json(f"{fresh_root}/regression_status.json")
    perception = load_json(
        "results/perception/joint_risk_astgcn_metis4_flowprio2_frozen.json"
    )
    architectures = load_json("results/edge/measured_architecture_baselines.json")
    network = load_json("results/edge/network_resilience_v1_6_summary.json")
    multi_edge = load_json("results/edge/multi_edge_cloud_load_v1_6_1_2_4_8.json")
    memory = load_json("results/edge/integrated_edge_memory_summary.json")
    qwen = load_json("results/llm/qwen_v9_text_only_competition_summary.json")
    general_kd = load_json("results/edge_llm/qwen35_0_8b_general_kd_summary.json")
    jetson_loop = load_json(
        "results/framework/jetson_to_wsl_traffic_framework_50runs_20260726.json"
    )
    sumo = load_json("results/simulation/sumo_freeway_closed_loop_adaptive_10seeds.json")
    cloud_llm = load_json(
        "results/framework/cloud_qwen9b_structured_recheck_20260726.json"
    )

    risk_node = future["risk_evaluation"]["node"]
    risk_region = future["risk_evaluation"]["region"]
    decision = future["decision_evaluation"]["predicted_risk_rule"]
    action_types = future["set_evaluation"]["predicted_risk_rule"]["action_types"]
    affected_nodes = future["set_evaluation"]["predicted_risk_rule"]["affected_nodes"]
    online_steady = online["steady_state"]
    outage_steady = outage["steady_state"]
    general_q4 = general_kd["capability"]["models"]["general_kd_q4"]["retention"]
    general_f16 = general_kd["capability"]["models"]["general_kd_f16"]["retention"]
    ttft_reduction = qwen["same_model_full_input_baseline"]["ttft_reduction"]
    architecture_items = architectures["architectures"]
    adaptive_arch = architecture_items["adaptive_cloud_edge"]
    centralized_arch = architecture_items["centralized_cloud"]
    upload_reduction = 1.0 - (
        adaptive_arch["average_payload_bytes_per_case"]
        / centralized_arch["average_payload_bytes_per_case"]
    )
    conformal_deployment = conformal["test"][conformal["deployment_method"]]
    selective_normal = scheduler["profiles"]["normal"]["selective_defer"]
    sumo_cloud = sumo["architectures"]["cloud_edge_collaboration"]["aggregate"]

    hard_requirements: List[Dict[str, Any]] = [
        {
            "id": "H1",
            "requirement": "边侧轻量大模型在数学、代码、自然语言推理保持满血模型 80%–90% 能力",
            "threshold": "三类均至少 80%",
            "result": (
                "最佳可部署候选 Q4 宏保持率 63.33%；代码 33.33%、数学 "
                "61.90%、自然语言推理 90.48%。F16 宏保持率 69.17%，代码仍仅 41.67%。"
            ),
            "status": "fail",
            "evidence": [
                evidence("results/edge_llm/qwen35_0_8b_general_kd_summary.json")
            ],
            "qualification": "这是通用能力硬缺口；交通专用任务保持率不能替代该指标。",
        },
        {
            "id": "H2",
            "requirement": "TTFT 减少 75%",
            "threshold": "相对明确基线至少减少 75%",
            "result": "同一交通 Qwen 模型长输入 1309.999 ms → 13-token 输入 82.620 ms，减少 {}。".format(
                percentage(ttft_reduction)
            ),
            "status": "qualified_pass",
            "evidence": [
                evidence("results/llm/qwen_v9_text_only_competition_summary.json"),
                evidence(
                    "results/llm/llama_cpp_v9_text_only_q6_full_input_gpu_ttft_baseline.json"
                ),
            ],
            "qualification": "达标来自交通协议和提示压缩；基线是同一 0.8B 模型，不是 9B 云模型。",
        },
        {
            "id": "H3",
            "requirement": "边侧单次推理内存占用不超过 1.5 GB",
            "threshold": "≤1536 MB",
            "result": "0.8B Qwen 系统 RAM 增量 1164.87 MB；与 ASTGCN 共驻保守 RSS 和 1067.07 MB。",
            "status": "pass",
            "evidence": [
                evidence("results/edge/integrated_edge_memory_summary.json"),
                evidence(
                    "results/llm/llama_cpp_v9_text_only_q6_nocache_jetson02_gpu_test.json"
                ),
            ],
            "qualification": "Jetson02 实测。",
        },
        {
            "id": "H4",
            "requirement": "云边协同网络波动期间基本业务功能保持率至少 90%",
            "threshold": "≥90%",
            "result": (
                "本轮真实断云 HTTP 340/340 返回安全本地结果，保持率 100%；"
                "Jetson 四网络档历史实测也均为 100%。"
            ),
            "status": "pass",
            "evidence": [
                evidence(f"{fresh_root}/framework_outage_360.json"),
                evidence("results/edge/network_resilience_v1_6_summary.json"),
            ],
            "qualification": "保持率表示协议有效和业务可执行，不表示断网仍有完整全局多边缘能力。",
        },
        {
            "id": "H5",
            "requirement": "至少部署 2 类差异明显场景",
            "threshold": "≥2 类场景",
            "result": "交通场景已完整接入并复测；工业插件虽已编写，但本轮按要求未部署、未计分。",
            "status": "not_tested",
            "evidence": [evidence(f"{fresh_root}/framework_http_360.json")],
            "qualification": "单靠交通场景不能满足该硬要求。",
        },
        {
            "id": "H6",
            "requirement": "至少两类场景平均端到端时延小于 0.2 s",
            "threshold": "两场景平均 E2E <200 ms",
            "result": (
                "交通本轮本机真实 HTTP 均值 81.40 ms、P95 96.48 ms；"
                "Jetson→WSL 含 Edge-Qwen 的 accounted 均值 185.21 ms、P95 191.68 ms。"
            ),
            "status": "partial",
            "evidence": [
                evidence(f"{fresh_root}/framework_http_360.json"),
                evidence(
                    "results/framework/jetson_to_wsl_traffic_framework_50runs_20260726.json"
                ),
            ],
            "qualification": "交通达标；第二场景尚无部署实测，所以总要求只能判部分达标。",
        },
        {
            "id": "H7",
            "requirement": "关联多边缘节点决策冲突比例不超过 5%",
            "threshold": "≤5%",
            "result": "自然运行耦合活跃对冲突率 4.48%，协调后残余冲突率 0%。",
            "status": "pass",
            "evidence": [evidence(f"{fresh_root}/conflict_consistency.json")],
            "qualification": "分母为共享道路边界且动作活跃的关联边缘对。",
        },
        {
            "id": "H8",
            "requirement": "冲突解决成功率至少 90%",
            "threshold": "≥90%",
            "result": "自然冲突 6/6 消解；48 个注入压力冲突 48/48 消解，均为 100%。",
            "status": "pass",
            "evidence": [evidence(f"{fresh_root}/conflict_consistency.json")],
            "qualification": "交通效用另由 SUMO 闭环评估，不能只看约束满足。",
        },
    ]

    metrics = [
        {
            "category": "感知",
            "metric": "flow / occupancy / speed MAPE",
            "result": "{:.2%} / {:.2%} / {:.2%}".format(
                perception["test"]["forecast"]["flow"]["mape"],
                perception["test"]["forecast"]["occupancy"]["mape"],
                perception["test"]["forecast"]["speed"]["mape"],
            ),
            "sample": "PEMS08 test",
            "freshness": "固定检查点既有实测",
            "evidence": "results/perception/joint_risk_astgcn_metis4_flowprio2_frozen.json",
        },
        {
            "category": "感知",
            "metric": "节点风险 Accuracy / Macro-F1 / 高严重召回",
            "result": "{:.2%} / {:.2%} / {:.2%}".format(
                risk_node["accuracy"],
                risk_node["macro_f1"],
                risk_node["high_severe_recall"],
            ),
            "sample": "{} 节点预测".format(risk_node["total"]),
            "freshness": "本轮复测",
            "evidence": f"{fresh_root}/future_truth_policy.json",
        },
        {
            "category": "感知",
            "metric": "区域风险 Accuracy / Macro-F1 / 高严重召回",
            "result": "{:.2%} / {:.2%} / {:.2%}".format(
                risk_region["accuracy"],
                risk_region["macro_f1"],
                risk_region["high_severe_recall"],
            ),
            "sample": "{} 区域事件".format(risk_region["total"]),
            "freshness": "本轮复测",
            "evidence": f"{fresh_root}/future_truth_policy.json",
        },
        {
            "category": "决策",
            "metric": "固定安全策略 Accuracy / Macro-F1 / 关键干预召回",
            "result": "{:.2%} / {:.2%} / {:.2%}".format(
                decision["accuracy"],
                decision["macro_f1_present_classes"],
                decision["critical_intervention_recall"],
            ),
            "sample": "{} 区域事件".format(decision["total"]),
            "freshness": "本轮复测",
            "evidence": f"{fresh_root}/future_truth_policy.json",
        },
        {
            "category": "决策",
            "metric": "动作类型 / 受影响节点 Micro-F1",
            "result": "{:.2%} / {:.2%}".format(
                action_types["micro_f1"], affected_nodes["micro_f1"]
            ),
            "sample": "{} 区域事件".format(decision["total"]),
            "freshness": "本轮复测",
            "evidence": f"{fresh_root}/future_truth_policy.json",
        },
        {
            "category": "校准",
            "metric": "ECE（温度缩放前→后）",
            "result": "{:.2%} → {:.2%}".format(
                conformal["probability_calibration"]["before"]["ece"],
                conformal["probability_calibration"]["after"]["ece"],
            ),
            "sample": "{} 区域预测".format(conformal["test_region_predictions"]),
            "freshness": "本轮复测",
            "evidence": f"{fresh_root}/region_risk_conformal_eval.json",
        },
        {
            "category": "校准",
            "metric": "风险集合覆盖率 / 平均集合大小 / 接受样本准确率",
            "result": "{:.2%} / {:.3f} / {:.2%}".format(
                conformal_deployment["coverage"],
                conformal_deployment["mean_set_size"],
                conformal_deployment["accepted_point_accuracy"],
            ),
            "sample": "{} 区域预测".format(conformal["test_region_predictions"]),
            "freshness": "本轮复测",
            "evidence": f"{fresh_root}/region_risk_conformal_eval.json",
        },
        {
            "category": "调度",
            "metric": "选择性协同 Accuracy / 云请求率（正常网络）",
            "result": "{:.2%} / {:.2%}".format(
                selective_normal["accuracy"], selective_normal["cloud_request_rate"]
            ),
            "sample": "{} 区域事件".format(scheduler["sample_count"]),
            "freshness": "本轮复测",
            "evidence": f"{fresh_root}/conformal_scheduler.json",
        },
        {
            "category": "在线闭环",
            "metric": "真实 HTTP 成功率 / client E2E 均值 / P95",
            "result": "{:.2%} / {:.2f} ms / {:.2f} ms".format(
                online_steady["success_rate"],
                online_steady["client_wall_ms"]["mean"],
                online_steady["client_wall_ms"]["p95"],
            ),
            "sample": "{} 稳态多样事件".format(online_steady["attempted"]),
            "freshness": "本轮复测（WSL）",
            "evidence": f"{fresh_root}/framework_http_360.json",
        },
        {
            "category": "弱网",
            "metric": "真实断云保持率 / E2E 均值 / P95",
            "result": "{:.2%} / {:.2f} ms / {:.2f} ms".format(
                outage_steady["success_rate"],
                outage_steady["client_wall_ms"]["mean"],
                outage_steady["client_wall_ms"]["p95"],
            ),
            "sample": "{} 稳态多样事件".format(outage_steady["attempted"]),
            "freshness": "本轮复测（真实停云）",
            "evidence": f"{fresh_root}/framework_outage_360.json",
        },
        {
            "category": "恢复",
            "metric": "断网队列补传",
            "result": "{} 条完成，pending=0，完成率 100%".format(
                recovery["outbox"]["completed"]
            ),
            "sample": "{} 次投递".format(recovery["outbox"]["delivery_attempts"]),
            "freshness": "本轮复测",
            "evidence": f"{fresh_root}/recovery_status.json",
        },
        {
            "category": "一致性",
            "metric": "自然冲突率 / 残余冲突率 / 压力消解率",
            "result": "{:.2%} / {:.2%} / {:.2%}".format(
                conflict["natural_operation"][
                    "raw_conflict_rate_among_coupled_active_pairs"
                ],
                conflict["natural_operation"]["post_coordination_conflict_rate"],
                conflict["injected_stress_test"]["resolution_success_rate"],
            ),
            "sample": "134 个耦合活跃对 + 48 个压力用例",
            "freshness": "本轮复测",
            "evidence": f"{fresh_root}/conflict_consistency.json",
        },
        {
            "category": "更新",
            "metric": "策略原子更新、校验、乱序和断网用例",
            "result": "{}/{} 成功，成功率 {:.2%}".format(
                policy["success_count"], policy["case_count"], policy["success_rate"]
            ),
            "sample": "{} 类更新故障".format(policy["case_count"]),
            "freshness": "本轮复测",
            "evidence": f"{fresh_root}/policy_sync_consistency.json",
        },
        {
            "category": "工程",
            "metric": "完整回归测试",
            "result": "{}/{} 通过".format(
                regression["tests_run"] - regression["failures"] - regression["errors"],
                regression["tests_run"],
            ),
            "sample": "{:.3f} s".format(regression["duration_seconds"]),
            "freshness": "本轮复测",
            "evidence": f"{fresh_root}/regression_status.json",
        },
        {
            "category": "通信",
            "metric": "集中式原始窗口→动态云边平均上传量",
            "result": "{:.2f} KB → {:.2f} KB，减少 {:.2%}".format(
                centralized_arch["average_payload_bytes_per_case"] / 1024.0,
                adaptive_arch["average_payload_bytes_per_case"] / 1024.0,
                upload_reduction,
            ),
            "sample": "每架构 100 次",
            "freshness": "Jetson02 既有实测",
            "evidence": "results/edge/measured_architecture_baselines.json",
        },
        {
            "category": "通信",
            "metric": "本轮正式框架实际 HTTP 请求 / 响应",
            "result": "{:.2f} KB / {:.2f} KB".format(
                online_steady["transport"]["request_bytes"]["mean"] / 1024.0,
                online_steady["transport"]["response_bytes"]["mean"] / 1024.0,
            ),
            "sample": "{} 稳态事件".format(online_steady["attempted"]),
            "freshness": "本轮复测",
            "evidence": f"{fresh_root}/framework_http_360.json",
        },
        {
            "category": "扩展性",
            "metric": "8 并发吞吐 / P95 / 成功率",
            "result": "{:.2f} req/s / {:.2f} ms / {:.2%}".format(
                multi_edge["levels"][-1]["throughput_requests_per_second"],
                multi_edge["levels"][-1]["request_latency"]["p95_ms"],
                multi_edge["levels"][-1]["success_rate"],
            ),
            "sample": "1/2/4/8 并发，每档 200 请求",
            "freshness": "Jetson02 既有实测",
            "evidence": "results/edge/multi_edge_cloud_load_v1_6_1_2_4_8.json",
        },
        {
            "category": "监测",
            "metric": "有界增量监测均值 / P95 / 加速",
            "result": "{:.3f} ms / {:.3f} ms / {:.2f}×".format(
                monitoring["bounded_incremental_evaluation"]["full_window_ms"]["mean"],
                monitoring["bounded_incremental_evaluation"]["full_window_ms"]["p95"],
                monitoring["full_window_mean_speedup"],
            ),
            "sample": "{} 次".format(monitoring["iterations"]),
            "freshness": "本轮复测（WSL）",
            "evidence": f"{fresh_root}/monitoring_hot_path.json",
        },
        {
            "category": "模型协同",
            "metric": "交通专用 0.8B Qwen TTFT / 合法输出率 / 单模准确率 / 级联准确率",
            "result": "{:.2f} ms / {:.2%} / {:.2%} / {:.2%}".format(
                qwen["jetson_gpu"]["average_ttft_ms"],
                qwen["strict_future_test"]["valid_output_rate"],
                qwen["strict_future_test"]["qwen_accuracy"],
                qwen["strict_future_test"]["selective_cascade_accuracy"],
            ),
            "sample": "{} 个严格留出事件".format(qwen["strict_future_test"]["samples"]),
            "freshness": "Jetson02 既有实测",
            "evidence": "results/llm/qwen_v9_text_only_competition_summary.json",
        },
        {
            "category": "模型协同",
            "metric": "Edge-Qwen / 实时 Student 对云端未来代理的任务保持率",
            "result": "{:.2%} / {:.2%}".format(
                retention["future_proxy_task"]["models"]["edge_qwen_0_8b"][
                    "retention_vs_cloud"
                ]["ratio"],
                retention["future_proxy_task"]["models"]["realtime_student"][
                    "retention_vs_cloud"
                ]["ratio"],
            ),
            "sample": "36 个严格留出事件",
            "freshness": "本轮复算",
            "evidence": f"{fresh_root}/edge_cloud_model_retention.json",
        },
        {
            "category": "应用效用",
            "metric": "SUMO 云边协同相对无控制",
            "result": (
                "系统时间 -{:.2%}，平均行程 -{:.2%}，P95 行程 -{:.2%}，等待 -{:.2%}"
            ).format(
                sumo_cloud["system_time_reduction"],
                sumo_cloud["in_network_travel_time_reduction"],
                sumo_cloud["p95_travel_time_reduction"],
                sumo_cloud["waiting_time_reduction"],
            ),
            "sample": "10 个随机种子",
            "freshness": "既有 SUMO 实测",
            "evidence": "results/simulation/sumo_freeway_closed_loop_adaptive_10seeds.json",
        },
        {
            "category": "云端增强",
            "metric": "真实 Qwen3.5-9B 严格 JSON / 热态时延",
            "result": "3/3 合法；两次热态 {:.2f}–{:.2f} ms（另有一次冷启动 {:.2f} ms）".format(
                min(item["latency_ms"] for item in cloud_llm["reviews"][1:]),
                max(item["latency_ms"] for item in cloud_llm["reviews"][1:]),
                cloud_llm["reviews"][0]["latency_ms"],
            ),
            "sample": "3 次结构化复核",
            "freshness": "既有实测",
            "evidence": "results/framework/cloud_qwen9b_structured_recheck_20260726.json",
        },
    ]

    scorecard = [
        {
            "item": "实时性改进",
            "max_score": 15,
            "estimated_score": 12,
            "basis": (
                "弱网相对固定同步均值降 69.19%/82.42%；但正常局域网动态方案 "
                "68.78 ms 慢于集中式 24.17 ms，不能宣称所有网络下都更快。"
            ),
        },
        {
            "item": "感知与决策效果",
            "max_score": 15,
            "estimated_score": 13,
            "basis": "全测试集风险、决策、校准和 10-seed SUMO 齐全；参考仍是未来状态代理而非现场控制真值。",
        },
        {
            "item": "资源与通信效率",
            "max_score": 10,
            "estimated_score": 9,
            "basis": "上传量较集中式降 86.38%，内存达标；正式框架完整 HTTP 包仍约 4.84 KB。",
        },
        {
            "item": "方案完整性",
            "max_score": 15,
            "estimated_score": 11,
            "basis": "云边、断网、补传、更新、监测闭环完整；第二场景尚未部署验证。",
        },
        {
            "item": "可扩展性与适应性",
            "max_score": 10,
            "estimated_score": 8,
            "basis": "插件化和 1/2/4/8 并发有证据；仍是单台 Jetson 模拟四逻辑区域。",
        },
        {
            "item": "稳定性表现",
            "max_score": 10,
            "estimated_score": 10,
            "basis": "正常/断云均 340/340 成功，补传队列归零，策略更新失败保持旧状态。",
        },
        {
            "item": "决策一致性",
            "max_score": 10,
            "estimated_score": 10,
            "basis": "自然冲突 4.48%，残余 0%，压力用例消解 100%。",
        },
        {
            "item": "创新性",
            "max_score": 10,
            "estimated_score": 8,
            "basis": "多级证据、可校准风险集合、边缘自治和异步复核有亮点；学习式路由器仍是 shadow。",
        },
        {
            "item": "应用价值",
            "max_score": 5,
            "estimated_score": 4,
            "basis": "交通闭环和 SUMO 效用明确；尚缺真实道路/第二行业部署。",
        },
    ]

    return {
        "schema_version": 1,
        "task": "competition_full_requirement_audit_traffic_scene",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "scene": "freeway_traffic_management",
            "fresh_retest_directory": fresh_root,
            "fresh_retest_note": (
                "算法、框架正常/断云 HTTP、恢复、监测、冲突和更新在当前 WSL 复测；"
                "只能在 Jetson 上重现的 TTFT、内存、四架构、弱网和并发采用已保存原始实测。"
            ),
            "reference": (
                "PEMS08 观测未来 flow/occupancy/speed 经冻结 FCM 与固定安全策略形成代理真值；"
                "它不是人工标注事故真值或真实道路执行收益。"
            ),
        },
        "summary": {
            "hard_requirement_counts": {
                status: sum(row["status"] == status for row in hard_requirements)
                for status in (
                    "pass",
                    "qualified_pass",
                    "partial",
                    "fail",
                    "not_tested",
                )
            },
            "estimated_score": sum(row["estimated_score"] for row in scorecard),
            "estimated_score_max": sum(row["max_score"] for row in scorecard),
            "score_range": [82, 87],
            "score_note": (
                "非官方估分。按现有证据中位估计 85/100；通用能力和第二场景两个硬缺口使 90+ "
                "目前不可防守。"
            ),
        },
        "hard_requirements": hard_requirements,
        "traffic_metrics": metrics,
        "scorecard": scorecard,
        "architecture_boundary": {
            "student_policy": (
                "框架不要求每个场景必须提供同一种 Student。框架只要求场景插件实现本地决策能力；"
                "交通 MLP Student、交通 0.8B 动作模型及其特征编码均属于交通插件。其他场景可使用"
                "自身轻量模型、校准器或规则安全层。"
            ),
            "cloud_llm_policy": (
                "9B 云模型只做高风险、分歧或解释性复核，不进入 200 ms 同步硬路径。"
            ),
        },
        "important_limitations": [
            "本轮没有把工业插件算作已部署的第二场景。",
            "通用数学/代码/自然语言能力仍未达到三类均 80% 的硬指标。",
            "当前正式 HTTP 摘要包平均约 4.84 KB，不应宣传成只有数百字节；其中树模型编码特征本体为 144 B，但协议元数据仍较大。",
            "360 条组件化框架样本上，拓扑融合为消除冲突改变了 35 个决策，但未来代理点准确率未提升；创新点应表述为一致性和安全约束，不是无条件精度提升。",
            "学习式效用路由器仍处于 shadow，不能作为已上线收益计分。",
        ],
        "supporting_values": {
            "general_q4_macro_retention": general_q4["macro_retention_ratio"],
            "general_f16_macro_retention": general_f16["macro_retention_ratio"],
            "ttft_reduction": ttft_reduction,
            "memory_system_ram_mb": qwen["jetson_gpu"]["system_ram_footprint_mb"],
            "co_resident_rss_sum_mb": memory["co_resident_application_upper_bounds"][
                "rss_sum_mb"
            ],
            "jetson_accounted_e2e_mean_ms": jetson_loop["steady_state"][
                "accounted_closed_loop_ms"
            ]["mean"],
            "jetson_accounted_e2e_p95_ms": jetson_loop["steady_state"][
                "accounted_closed_loop_ms"
            ]["p95"],
            "upload_reduction_vs_centralized": upload_reduction,
            "framework_component_quality": collaboration["quality"],
            "weak_network": network["adaptive_profiles"],
        },
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# 交通场景竞赛要求全量复测矩阵",
        "",
        "结论：当前证据中位估分约 **{}/{}**（合理区间 {}–{}）。交通场景本身的实时性、"
        "弱网保持、一致性、资源和闭环均较强，但“通用三类能力 80%”及“第二场景部署”"
        "仍是两个硬缺口，因此现在不能稳妥宣称 90+。".format(
            report["summary"]["estimated_score"],
            report["summary"]["estimated_score_max"],
            report["summary"]["score_range"][0],
            report["summary"]["score_range"][1],
        ),
        "",
        "## 一、题目硬指标",
        "",
        "| 编号 | 题目要求 | 门槛 | 实测结果 | 判定 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["hard_requirements"]:
        lines.append(
            "| {id} | {requirement} | {threshold} | {result} | **{label}** |".format(
                label=status_label(row["status"]), **row
            )
        )
    lines.extend(
        [
            "",
            "说明：`条件达标` 表示数值达到，但基线定义比题目可能采用的严格口径窄；"
            "`部分达标` 表示交通通过、跨场景总要求尚未通过。",
            "",
            "## 二、交通场景量化结果",
            "",
            "| 类别 | 指标 | 结果 | 样本/负载 | 证据新鲜度 |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for row in report["traffic_metrics"]:
        lines.append(
            "| {category} | {metric} | {result} | {sample} | {freshness} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 三、按评审细则估分",
            "",
            "| 评分项 | 满分 | 当前估分 | 依据 |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in report["scorecard"]:
        lines.append(
            "| {item} | {max_score} | {estimated_score} | {basis} |".format(**row)
        )
    lines.extend(
        [
            "| **合计** | **{}** | **{}** | 非官方证据审计估分 |".format(
                report["summary"]["estimated_score_max"],
                report["summary"]["estimated_score"],
            ),
            "",
            "## 四、必须诚实写进报告的边界",
            "",
        ]
    )
    for limitation in report["important_limitations"]:
        lines.append("- " + limitation)
    lines.extend(
        [
            "",
            "## 五、Student 的框架边界",
            "",
            report["architecture_boundary"]["student_policy"],
            "",
            report["architecture_boundary"]["cloud_llm_policy"],
            "",
            "## 六、原始证据索引",
            "",
        ]
    )
    evidence_paths = sorted(
        {
            path_value
            for row in report["hard_requirements"]
            for path_value in row["evidence"]
        }
        | {row["evidence"] for row in report["traffic_metrics"]}
    )
    for path_value in evidence_paths:
        lines.append("- `{}`".format(path_value))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the traffic competition requirement report.")
    parser.add_argument(
        "--output-json",
        default="results/competition/traffic_full_retest_20260726/requirement_matrix.json",
    )
    parser.add_argument(
        "--output-md",
        default="docs/traffic_requirement_full_retest_20260726.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()
    output_json = PROJECT_ROOT / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_md = PROJECT_ROOT / args.output_md
    write_markdown(report, output_md)
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "hard_requirement_counts": report["summary"][
                    "hard_requirement_counts"
                ],
                "estimated_score": report["summary"]["estimated_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
