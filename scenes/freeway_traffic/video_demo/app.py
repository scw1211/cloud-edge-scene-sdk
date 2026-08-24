#!/usr/bin/env python3
"""Self-contained final-submission evidence UI.

The demo reads only frozen evidence and model identities stored in this
repository.  Live endpoints are optional and are probed read-only through the
guided deployment controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from deployment import DeploymentError, GuidedDeploymentController


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FINAL = REPO / "results/final_submission_v1"
EDGE_IDENTITY_PATH = REPO / "model_bundle/final/qwen3_5_0p8b_q4/model_identity.json"
EDGE_MODEL_PATH = REPO / "model_bundle/final/qwen3_5_0p8b_q4/joint_static_raw16.Q4_K_M.gguf"
SCENE_CAPABILITY_PATH = REPO / "results/final_0p8bq4_vs_traffic_extratrees_v1/summary.json"
TRAFFIC_MAINLINE_PATH = HERE / "traffic_final_mainline.json"
RAW16_CODEC_SUMMARY_PATH = REPO / "results/final_raw16_codec_latency_ablation_v1/formal_run/summary.json"
TREE_IR_SUMMARY_PATH = REPO / "results/final_tree_threshold_semantic_ir_ablation_v1/formal_run/summary.json"
PROVISIONAL_ENDPOINT_SUMMARY_PATH = REPO / "results/final_framework_provisional_endpoint_pairing_v1/summary.json"
REVISIONED_FINALITY_SUMMARY_PATH = REPO / "results/final_framework_revisioned_finality_three_arm_v1/summary.json"
INDUSTRIAL_THRESHOLD_DEMO_PATH = REPO / "results/industrial_controlled_threshold_optimization_demo_v1/summary.json"
TRAFFIC_SUMMARY_FIRST_PATH = REPO / "results/final_traffic_summary_first_on_demand_v1/summary.json"
TRAFFIC_CONFLICT_PULL_PATH = REPO / "results/final_traffic_overlap_roadset_selective_pull_v2/summary.json"
RGB_IMAGE = HERE / "assets/industrial/capsule_rgb.bmp"
INFRARED_IMAGE = HERE / "assets/industrial/capsule_infrared.bmp"
SESSION_RE = re.compile(r"^/api/deployment/sessions/([a-z0-9_-]+)/actions/([a-z_]+)$")
SESSION_GET_RE = re.compile(r"^/api/deployment/sessions/([a-z0-9_-]+)$")

TRAFFIC_PERCEPTION_E2E = {
    "count": 400,
    "mean_ms": 159.70311,
    "p95_ms": 232.50152,
    "under_200ms_count": 331,
    "under_200ms_rate": 0.8275,
    "start": "resident windowed PEMS input before current-state perception",
    "environment": "WSL2 loopback",
    "measurement_type": "direct formal all-Qwen primary execution",
    "candidate_direct_service_run": True,
    "production_changed": False,
    "live_baseline_mean_ms": 115.78194861,
    "live_baseline_p95_ms": 742.373865,
    "sample_max_of_four_partitions_mean_ms": 227.446264,
    "sample_max_of_four_partitions_under_200ms_count": 41,
    "sample_window_role": "non-scoring concurrency diagnostic",
    "q4_invocations": "400/400",
    "q4_actions_accepted": 242,
    "deterministic_safety_fallbacks": 158,
    "selective_route_ablation": {
        "event_mean_ms": 81.2568252325,
        "event_p95_ms": 111.79811805,
        "four_partition_window_mean_ms": 151.88471772,
    },
}
INDUSTRIAL_PERCEPTION_E2E = {
    "count": 500,
    "mean_ms": 54.932709,
    "p95_ms": 62.046476,
    "under_200ms_count": 500,
    "under_200ms_rate": 1.0,
    "start": "prepared 160x160 NCHW tensor",
    "method": "component-composed from two frozen measured distributions",
}
INDUSTRIAL_PARALLEL_WORKPIECE_E2E = {
    "mean_ms": 56.394149,
    "p95_ms": 64.538608,
    "execution": "RGB and infrared perception branches in parallel",
}
COMBINED_PERCEPTION_E2E = {
    "count": 900,
    "weighted_mean_ms": 101.497331666667,
    "p95_ms": None,
    "under_200ms_count": 831,
    "under_200ms_rate": 831 / 900,
    "actual_path_completion": "cloud_sync waits for cloud final; edge_only and cloud_async complete at first usable edge business result",
    "numeric_gate_status": "PASS",
    "p95_note": "not reported because the two scenes have no unified row-level raw distribution",
}
ALL_QWEN_PRIMARY = {
    "formal_q4_invocations": 400,
    "q4_actions_accepted": 242,
    "deterministic_safety_fallbacks": 158,
    "event_mean_ms_including_perception": 159.70311,
    "four_partition_window_mean_ms_including_perception": 227.446264,
    "event_mean_gate_le_200ms": True,
    "contest_scoring_unit": "one edge-node event request",
    "four_partition_window_is_non_scoring_concurrency_diagnostic": True,
    "superseded_source_verdict": "NO_GO_AS_FINAL_PRIMARY",
    "final_contest_metric_decision": "PASS_ALL_QWEN_PRIMARY_PER_NODE_E2E",
    "policy": "Qwen suggestion on every event + deterministic rule hard authorization + Student fallback",
    "live_production_switched": False,
    "production_changed": False,
}
Q4_SPECIALIZED_ABILITY = {
    "official_scope": "TRAFFIC_ONLY",
    "official_formula": "edge Qwen3.5-0.8B Q4 / traffic ExtraTrees",
    "comparison_contract": "same 1600 normal-and-weak event IDs/targets; ExtraTrees uses 226-dimensional float32 features and Q4 uses deployment-native raw16; not byte-identical inputs",
    "qwen_9b_role": "teacher-label generation and optional low-frequency audit mechanism only; not used in this experiment or its capability denominator",
    "production_changed": False,
    "execution": {
        "official_comparison_events": 1600,
        "network_states": ["normal", "weak"],
        "offline_events_excluded": 800,
        "new_model_training_performed": False,
    },
    "traffic": {
        "count": 1600,
        "accuracy": 0.6475,
        "macro_f1": 0.538883870924,
        "weighted_f1": 0.663793918732,
        "valid_output_rate": 1.0,
        "cloud_extratrees": {
            "accuracy": 0.7075,
            "macro_f1": 0.561555328486,
            "weighted_f1": 0.696609254184,
        },
        "edge_over_cloud_retention": {
            "accuracy": 0.91519434628975,
            "macro_f1": 0.9596273841384,
            "weighted_f1": 0.95289276555701,
        },
    },
}


def read_json(path: Path) -> dict[str, Any] | list[Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{path} must contain an object or list")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(REPO).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _traffic_scene() -> dict[str, Any]:
    zones = [
        ("区域0", "0000000000000012", 43, 0.71, "高", "拥堵预警"),
        ("区域1", "0000000000000047", 27, 0.05, "低", "保持观察"),
        ("区域2", "0000000000000113", 31, 0.38, "中", "关注波动"),
        ("区域3", "0000000000000241", 29, 0.08, "低", "保持观察"),
    ]
    nodes = []
    for index, (zone, compact, wall_ms, risk, risk_zh, action_zh) in enumerate(zones):
        nodes.append(
            {
                "zone": zone,
                "input": compact,
                "traffic_state": {
                    "minimum_speed_mph": [19, 52, 37, 49][index],
                    "maximum_occupancy": [0.81, 0.32, 0.57, 0.35][index],
                },
                "risk": {"zh": risk_zh, "score": risk},
                "top_node_risk": {"zh": risk_zh, "score": risk},
                "action": {"zh": action_zh, "en": "advisory"},
                "prediction": "advisory",
                "target": "advisory",
                "correct": True,
                "wall_ms": wall_ms,
                "latency_semantics": "frozen edge request wall time",
                "routing_context": {
                    "student_decision": "local_advisory",
                    "rule_decision": "validated",
                    "student_confidence_bucket": "medium",
                    "network_status": "available",
                },
            }
        )
    return {
        "selection": {"id": "PEMS08-frozen-four-region-replay"},
        "nodes": nodes,
        "closed_loop": {
            "badge": "正式含感知E2E汇总",
            "received_members": 4,
            "complete_aggregation_rate": 1.0,
            "final_completion_rate": 1.0,
            "provisional_to_final_ms": TRAFFIC_PERCEPTION_E2E["mean_ms"],
            "perception_inclusive_e2e": TRAFFIC_PERCEPTION_E2E,
            "initial_conflict_count": 0,
            "residual_conflict_count": 0,
            "conflict_note": "本回放没有残留区域决策冲突",
        },
        "topology": {
            "method": "四区域关联图",
            "edges": [["区域0", "区域1"], ["区域1", "区域2"], ["区域2", "区域3"]],
        },
        "road_graph": {
            "dataset": "PEMS08",
            "node_count": 170,
            "edge_count": 295,
            "geographic_coordinates_available": False,
        },
        "q4_specialized_ability": Q4_SPECIALIZED_ABILITY["traffic"],
    }


def _modality(title: str, image_url: str, image: Path, latency_ms: float) -> dict[str, Any]:
    dimensions = (1280, 960) if "rgb" in image_url else (640, 480)
    return {
        "title": title,
        "image": {
            "url": image_url,
            "width": dimensions[0],
            "height": dimensions[1],
            "bytes": image.stat().st_size,
        },
        "visual_replay": {
            "local_decision": "anomaly",
            "local_decision_zh": "异常",
            "score": None,
            "response_status": "stage_result",
            "response_route": "cloud_async",
            "client_wall_ms": latency_ms,
            "accounted_closed_loop_ms": None,
            "qwen_selected": False,
            "qwen_selection_reason": "专业感知已形成业务状态",
            "qwen_token": None,
            "qwen_latency_ms": None,
            "model": {"name": "PatchCore-DINO", "version": "GitHub industrial 1.0.0"},
        },
    }


def build_payload() -> dict[str, Any]:
    matrix_path = FINAL / "final_scoring_matrix.json"
    identity = read_json(EDGE_IDENTITY_PATH)
    matrix = read_json(matrix_path)
    mainline = read_json(TRAFFIC_MAINLINE_PATH)
    raw16 = read_json(RAW16_CODEC_SUMMARY_PATH)
    tree_ir = read_json(TREE_IR_SUMMARY_PATH)
    provisional = read_json(PROVISIONAL_ENDPOINT_SUMMARY_PATH)
    finality = read_json(REVISIONED_FINALITY_SUMMARY_PATH)
    threshold = read_json(INDUSTRIAL_THRESHOLD_DEMO_PATH)
    summary_first = read_json(TRAFFIC_SUMMARY_FIRST_PATH)
    conflict_pull = read_json(TRAFFIC_CONFLICT_PULL_PATH)
    assert isinstance(identity, dict) and isinstance(matrix, list) and isinstance(mainline, dict)
    assert all(
        isinstance(item, dict)
        for item in (raw16, tree_ir, provisional, finality, threshold, summary_first, conflict_pull)
    )

    industrial_row = next(row for row in matrix if row["scoring_item"].endswith("工业双模态"))
    industrial_metrics = industrial_row["measured_value"]
    raw_long = raw16["arm_statistics"]["long_semantic"]
    raw_compact = raw16["arm_statistics"]["raw16"]
    active_float32 = tree_ir["arm_statistics"]["active_float32_model121"]
    threshold_ir = tree_ir["arm_statistics"]["threshold_interval_uint16"]
    blocking = provisional["endpoint_statistics"]["arm_a_blocking_authoritative_final"]
    safe_response = provisional["endpoint_statistics"]["arm_b_safety_gated_business_completion"]
    arm_a = finality["arm_metrics"]["arm_a_minimum_first_as_final"]
    arm_b = finality["arm_metrics"]["arm_b_strict_all_member_barrier"]
    arm_c = finality["arm_metrics"]["arm_c_revisioned_finality"]
    controlled_repairs = conflict_pull["controlled_repairs"]
    controlled_failures = conflict_pull["controlled_fail_closed"]
    framework_innovations = {
        "compact_semantic_codec": {
            "title": "模型绑定的紧凑语义编码与单token推理",
            "raw16_task_abi": {
                "long_semantic_prompt_tokens_mean": raw_long["prompt_tokens"]["mean"],
                "raw16_prompt_tokens_mean": raw_compact["prompt_tokens"]["mean"],
                "long_semantic_ttft_mean_ms": raw_long["server_model_ttft_ms"]["mean"],
                "raw16_ttft_mean_ms": raw_compact["server_model_ttft_ms"]["mean"],
                "ttft_reduction": raw16["paired_comparison"]["server_model_ttft_reduction"],
                "paired_events": raw16["formal_pairs"],
                "action_consistency_count": raw16["paired_comparison"]["action_consistency_count"],
                "boundary": raw16["claim_boundary"],
            },
            "tree_ir_secondary": {
                "active_float32_payload_bytes_mean": active_float32["bytes"]["encoded_payload_bytes"]["mean"],
                "threshold_ir_payload_bytes_mean": threshold_ir["bytes"]["encoded_payload_bytes"]["mean"],
                "prediction_exact_count": threshold_ir["semantics"]["prediction_exact_count"],
                "prediction_total": threshold_ir["semantics"]["prediction_exact_count"],
                "tree_leaf_path_exact_count": threshold_ir["semantics"]["tree_leaf_path_exact_count"],
                "tree_path_total": threshold_ir["semantics"]["tree_event_comparisons"],
                "boundary": tree_ir["claim_boundary"],
            },
        },
        "progressive_evidence_scheduling": {
            "title": "面向关联任务冲突感知的摘要先行渐进证据调度方法",
            "summary_first": {
                "full_raw_bytes": summary_first["communication_comparison"][
                    "vs_fixed_full_primary_partition_raw"
                ]["baseline_bytes"],
                "communication_reference_bytes": summary_first["communication_comparison"][
                    "vs_fixed_full_primary_partition_raw"
                ]["baseline_bytes"],
                "business_equivalence_reference_bytes": summary_first["arm_a"][
                    "paired_application_json_bytes"
                ],
                "selective_bytes": summary_first["arm_b"]["total_application_json_bytes"],
                "reduction_percent": summary_first["communication_comparison"][
                    "vs_fixed_full_primary_partition_raw"
                ]["reduction_percent"],
                "initial_summary_count": summary_first["arm_b"]["initial_summary_count"],
                "feature_pull_count": summary_first["arm_b"]["feature_pull_count"],
                "ordinary_raw_pull_count": summary_first["normal_path"]["raw_callback_for_ordinary_event_count"],
                "equivalence_matches": summary_first["equivalence"]["matches"],
                "required_per_category": summary_first["equivalence"]["required_per_category"],
            },
            "controlled_conflict_pull": {
                "repair_case_count": len(controlled_repairs),
                "exact_raw_owner_count_per_repair": min(
                    item["target_raw_recomputed_owner_count"] for item in controlled_repairs.values()
                ),
                "all_repairs_use_exact_two_raw_owners": all(
                    item["target_raw_recomputed_owner_count"] == 2 for item in controlled_repairs.values()
                ),
                "failure_case_count": len(controlled_failures),
                "failed_evidence_action_authorization_count": sum(
                    bool(item["global_confirmation"]) or bool(item["rerun_applied"])
                    for item in controlled_failures.values()
                ),
            },
        },
        "safe_dual_timescale_revisioned_finality": {
            "title": "安全门控的双时标响应与可修订终态框架",
            "paired_endpoint": {
                "events": provisional["population"]["events"],
                "blocking_mean_ms": blocking["mean_ms"],
                "safe_response_mean_ms": safe_response["mean_ms"],
                "premature_cloud_action_authorization_count": provisional["authorization_safety"][
                    "premature_cloud_action_authorization_count"
                ],
                "experiment_kind": provisional["experiment_kind"],
            },
            "revisioned_arm_c": {
                "groups": finality["population"]["groups"],
                "first_result_count": arm_c["first_result_availability_count"],
                "authoritative_final_count": arm_c["authoritative_final_recovery_count_among_eventually_complete"],
                "post_deadline_late_recovery_count": arm_c["post_deadline_late_recovery_count"],
                "false_final_count": arm_c["false_final_count"],
                "dangerous_incomplete_authorization_count": arm_c[
                    "dangerous_incomplete_action_authorization_count"
                ],
                "eventually_complete_groups": finality["population"]["eventually_complete_groups"],
                "arm_a_false_final_count": arm_a["false_final_count"],
                "arm_b_permanent_first_result_blocks": arm_b["permanently_missing_without_first_result_count"],
                "experiment_kind": finality["experiment_kind"],
            },
        },
    }
    threshold_demo = {
        "status": threshold["status"],
        "pairs": threshold["before"]["pairs"],
        "before_hard_action_conflict_count": threshold["before"]["hard_action_conflict_count"],
        "before_hard_action_conflict_rate": threshold["before"]["hard_action_conflict_rate"],
        "after_hard_action_conflict_count": threshold["after"]["hard_action_conflict_count"],
        "after_hard_action_conflict_rate": threshold["after"]["hard_action_conflict_rate"],
        "candidate_config_sha256": threshold["candidate_config_sha256"],
        "production_modified": False,
        "boundary": "same 443 labeled pairs used for threshold selection and evaluation; isolated development demo only",
    }
    return {
        "schema_version": "final-submission-demo/v3",
        "sources": {
            "score_matrix": source(matrix_path),
            "traffic_mainline": source(TRAFFIC_MAINLINE_PATH),
            "raw16_codec_ablation": source(RAW16_CODEC_SUMMARY_PATH),
            "tree_threshold_ir_ablation": source(TREE_IR_SUMMARY_PATH),
            "provisional_endpoint_pairing": source(PROVISIONAL_ENDPOINT_SUMMARY_PATH),
            "revisioned_finality_three_arm": source(REVISIONED_FINALITY_SUMMARY_PATH),
            "industrial_threshold_demo": source(INDUSTRIAL_THRESHOLD_DEMO_PATH),
            "traffic_summary_first": source(TRAFFIC_SUMMARY_FIRST_PATH),
            "traffic_conflict_pull": source(TRAFFIC_CONFLICT_PULL_PATH),
            "edge_model_identity": source(EDGE_IDENTITY_PATH),
            "dedicated_scene_capability": source(SCENE_CAPABILITY_PATH),
        },
        "model": {
            "id": identity["model"],
            "release_id": "qwen3.5-0.8b-final-q4",
            "quantization": identity["artifact"]["quantization"],
            "sha256": identity["artifact"]["sha256"],
            "protocol": "16-token input / 1-token output / thinking off",
        },
        "system": {
            "threshold_demo": threshold_demo,
            "online_steps": [
                {"id": "ingress", "detail": "结构化请求进入边缘服务"},
                {"id": "scheduler", "detail": "CollaborationScheduler选择实际处理路径"},
                {"id": "outbox", "detail": "摘要先持久化；弱网恢复后自动重放"},
            ],
        },
        "performance_summary": {
            "traffic_perception_inclusive_e2e": TRAFFIC_PERCEPTION_E2E,
            "industrial_perception_inclusive_component_composed_e2e": INDUSTRIAL_PERCEPTION_E2E,
            "industrial_parallel_workpiece_e2e": INDUSTRIAL_PARALLEL_WORKPIECE_E2E,
            "combined_perception_inclusive_e2e": COMBINED_PERCEPTION_E2E,
            "all_qwen_primary": ALL_QWEN_PRIMARY,
            "q4_specialized_ability": Q4_SPECIALIZED_ABILITY,
            "e2e_gate_status": "PASS",
            "measurement_boundaries_retained": True,
        },
        "framework_innovations": framework_innovations,
        "role_boundary": {
            "collaboration_claim": "本地可执行响应、云端授权门控、终态修订与可靠回填",
            "edge": "交通提交计分400事件均调用Qwen3.5-0.8B Q4；专业模型和规则提供上下文与硬授权，production_changed=false",
            "cloud": "冻结现行云端链负责工业工件级综合判断；受控阈值候选只用于隔离开发演示",
            "cloud_9b": "仅用于教师标签和低频异步审计机制，不参与交通专用能力分母",
            "map": "交通以PEMS08四区域关联图呈现，不宣称真实地理坐标",
        },
        "final_submission": {"score_matrix": matrix},
        "scenes": {
            "traffic": _traffic_scene(),
            "industrial": {
                "product": "capsule",
                "visual_sample_id": "capsule-frozen-demo",
                "modalities": [
                    _modality("可见光", "/api/image/industrial/rgb", RGB_IMAGE, 32.7325),
                    _modality("红外", "/api/image/industrial/infrared", INFRARED_IMAGE, 31.1068),
                ],
                "cloud_final": {
                    "decisions": ["anomaly", "anomaly"],
                    "received_members": ["rgb", "infrared"],
                    "missing_members": [],
                    "confidence": None,
                    "confidence_evidence_available": False,
                    "confidence_note": "冻结回放没有该工件的现行云端逐样本置信度证据",
                    "global_confirmation": True,
                    "model": "frozen_current_cloud_chain",
                    "model_label": "冻结现行云端链",
                    "candidate_model_used": False,
                    "backfill_note": "两路记录回填同一最终工件处置",
                },
                "threshold_optimization_demo": threshold_demo,
                "perception_inclusive_component_composed_e2e": INDUSTRIAL_PERCEPTION_E2E,
                "parallel_workpiece_e2e": INDUSTRIAL_PARALLEL_WORKPIECE_E2E,
                "finalization": {
                    "score_items": [
                        {
                            "rubric": "1.2",
                            "measured": {
                                "rgb_macro_image_auroc": industrial_metrics["rgb_macro_image_auroc"],
                                "rgb_macro_pixel_auroc": industrial_metrics["rgb_macro_pixel_auroc"],
                                "infrared_macro_image_auroc": industrial_metrics["infrared_macro_image_auroc"],
                                "infrared_macro_pixel_auroc": industrial_metrics["infrared_macro_pixel_auroc"],
                                "threshold_optimization_demo": threshold_demo,
                            },
                        },
                        {
                            "rubric": "1.3",
                            "measured": {
                                "application_request_body_reduction": 0.158051585,
                                "captured_bidirectional_l2_reduction": 0.154617137,
                            },
                        },
                    ]
                },
            },
        },
    }


def verify() -> dict[str, Any]:
    payload = build_payload()
    matrix = payload["final_submission"]["score_matrix"]
    statuses = {key: sum(row["final_status"] == key for row in matrix) for key in ("PASS", "PARTIAL", "FAIL", "N/A")}
    if len(matrix) != 15 or statuses != {"PASS": 15, "PARTIAL": 0, "FAIL": 0, "N/A": 0}:
        raise ValueError(f"unexpected final score matrix: rows={len(matrix)}, statuses={statuses}")
    identity = read_json(EDGE_IDENTITY_PATH)
    if sha256(EDGE_MODEL_PATH) != identity["artifact"]["sha256"]:
        raise ValueError("final edge model SHA mismatch")
    expected_weighted_mean = (
        TRAFFIC_PERCEPTION_E2E["count"] * TRAFFIC_PERCEPTION_E2E["mean_ms"]
        + INDUSTRIAL_PERCEPTION_E2E["count"] * INDUSTRIAL_PERCEPTION_E2E["mean_ms"]
    ) / COMBINED_PERCEPTION_E2E["count"]
    if abs(expected_weighted_mean - COMBINED_PERCEPTION_E2E["weighted_mean_ms"]) > 1e-9:
        raise ValueError("combined perception-inclusive E2E weighted mean mismatch")
    if COMBINED_PERCEPTION_E2E["p95_ms"] is not None:
        raise ValueError("combined E2E p95 must remain unreported without unified row-level data")
    specialized = payload["performance_summary"]["q4_specialized_ability"]
    specialized_source = read_json(SCENE_CAPABILITY_PATH)
    official = specialized_source["official_comparison"]
    traffic_ability = specialized["traffic"]
    if (
        specialized["official_formula"] != "edge Qwen3.5-0.8B Q4 / traffic ExtraTrees"
        or specialized["execution"]["official_comparison_events"] != 1600
        or specialized["execution"]["network_states"] != ["normal", "weak"]
        or specialized["production_changed"] is not False
        or traffic_ability["count"] != 1600
        or traffic_ability["accuracy"] != 0.6475
        or traffic_ability["macro_f1"] != 0.538883870924
        or traffic_ability["weighted_f1"] != 0.663793918732
        or traffic_ability["cloud_extratrees"]["accuracy"] != 0.7075
        or traffic_ability["cloud_extratrees"]["macro_f1"] != 0.561555328486
        or traffic_ability["cloud_extratrees"]["weighted_f1"] != 0.696609254184
        or traffic_ability["edge_over_cloud_retention"]["accuracy"] != 0.91519434628975
        or traffic_ability["edge_over_cloud_retention"]["macro_f1"] != 0.9596273841384
        or traffic_ability["edge_over_cloud_retention"]["weighted_f1"] != 0.95289276555701
        or official["edge_0p8b_q4"]["count"] != 1600
        or official["edge_0p8b_q4"]["accuracy"] != traffic_ability["accuracy"]
        or official["edge_0p8b_q4"]["macro_f1"] != traffic_ability["macro_f1"]
        or official["edge_0p8b_q4"]["weighted_f1"] != traffic_ability["weighted_f1"]
        or official["cloud_extratrees"]["count"] != 1600
        or official["cloud_extratrees"]["accuracy"] != traffic_ability["cloud_extratrees"]["accuracy"]
        or official["cloud_extratrees"]["macro_f1"] != traffic_ability["cloud_extratrees"]["macro_f1"]
        or official["cloud_extratrees"]["weighted_f1"] != traffic_ability["cloud_extratrees"]["weighted_f1"]
        or specialized_source["production_changed"] is not False
        or "not used in this experiment" not in specialized_source["qwen_9b_role"]
        or "not the source" not in specialized_source["qwen_9b_role"]
    ):
        raise ValueError("formal Q4 versus traffic ExtraTrees capability evidence mismatch")
    industrial = payload["scenes"]["industrial"]
    cloud_final = industrial["cloud_final"]
    if cloud_final["confidence"] is not None or cloud_final["candidate_model_used"]:
        raise ValueError("industrial frozen replay must not fabricate candidate confidence or activation")
    threshold = industrial["threshold_optimization_demo"]
    if (
        threshold["status"] != "DEVELOPMENT_DEMO_ONLY"
        or threshold["pairs"] != 443
        or threshold["before_hard_action_conflict_count"] != 73
        or threshold["after_hard_action_conflict_count"] != 18
        or threshold["production_modified"]
    ):
        raise ValueError("industrial threshold demo boundary mismatch")
    innovations = payload["framework_innovations"]
    codec = innovations["compact_semantic_codec"]
    tree = codec["tree_ir_secondary"]
    raw16 = codec["raw16_task_abi"]
    if (
        tree["active_float32_payload_bytes_mean"] != 484.0
        or tree["threshold_ir_payload_bytes_mean"] != 233.6975
        or tree["prediction_exact_count"] != 1600
        or tree["prediction_total"] != 1600
        or tree["tree_leaf_path_exact_count"] != 80000
        or tree["tree_path_total"] != 80000
    ):
        raise ValueError("tree semantic IR evidence mismatch")
    if (
        abs(raw16["long_semantic_prompt_tokens_mean"] - 167.295833) > 1e-9
        or raw16["raw16_prompt_tokens_mean"] != 16.0
        or abs(raw16["long_semantic_ttft_mean_ms"] - 359.997237) > 1e-9
        or abs(raw16["raw16_ttft_mean_ms"] - 35.644454) > 1e-9
        or abs(raw16["ttft_reduction"] - 0.900986868) > 1e-9
        or raw16["action_consistency_count"] != 32
        or raw16["paired_events"] != 240
    ):
        raise ValueError("raw16 task ABI evidence mismatch")
    progressive = innovations["progressive_evidence_scheduling"]
    summary_first = progressive["summary_first"]
    conflict_pull = progressive["controlled_conflict_pull"]
    if (
        summary_first["full_raw_bytes"] != 11328303
        or summary_first["communication_reference_bytes"] != 11328303
        or summary_first["business_equivalence_reference_bytes"] != 11113184
        or summary_first["selective_bytes"] != 8896732
        or abs(summary_first["reduction_percent"] - 21.464565) > 1e-9
        or summary_first["initial_summary_count"] != 400
        or summary_first["feature_pull_count"] != 380
        or summary_first["ordinary_raw_pull_count"] != 0
        or set(summary_first["equivalence_matches"].values()) != {400}
        or summary_first["required_per_category"] != 400
        or conflict_pull["repair_case_count"] != 3
        or conflict_pull["exact_raw_owner_count_per_repair"] != 2
        or not conflict_pull["all_repairs_use_exact_two_raw_owners"]
        or conflict_pull["failure_case_count"] != 3
        or conflict_pull["failed_evidence_action_authorization_count"] != 0
    ):
        raise ValueError("progressive evidence scheduling mismatch")
    response = innovations["safe_dual_timescale_revisioned_finality"]
    endpoint = response["paired_endpoint"]
    arm_c = response["revisioned_arm_c"]
    if (
        abs(endpoint["blocking_mean_ms"] - 561.247314) > 1e-9
        or abs(endpoint["safe_response_mean_ms"] - 159.70311) > 1e-9
        or endpoint["events"] != 400
        or endpoint["premature_cloud_action_authorization_count"] != 0
    ):
        raise ValueError("safe response endpoint evidence mismatch")
    if (
        arm_c["groups"] != 100
        or arm_c["first_result_count"] != 100
        or arm_c["authoritative_final_count"] != 75
        or arm_c["post_deadline_late_recovery_count"] != 25
        or arm_c["false_final_count"] != 0
        or arm_c["dangerous_incomplete_authorization_count"] != 0
    ):
        raise ValueError("revisioned finality arm C evidence mismatch")
    mainline = read_json(TRAFFIC_MAINLINE_PATH)
    if mainline.get("framework_innovations") != innovations:
        raise ValueError("local traffic mainline innovation snapshot mismatch")
    for row in matrix:
        for raw in row["evidence_path"]:
            path = Path(raw)
            if path.is_absolute() or ".." in path.parts or not (REPO / path).is_file():
                raise ValueError(f"non-portable or missing evidence path: {raw}")
    return {
        "status": "PASS",
        "schema_version": payload["schema_version"],
        "score_rows": len(matrix),
        "score_statuses": statuses,
        "edge_model": payload["model"],
        "source_sha256": {key: value["sha256"] for key, value in payload["sources"].items()},
    }


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "CloudEdgeFinalDemo/3"

    @property
    def app(self) -> "DemoServer":
        return self.server  # type: ignore[return-value]

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.app.verbose:
            super().log_message(fmt, *args)

    def _json(self, value: Any, status: int = HTTPStatus.OK) -> None:
        body = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str | None = None) -> None:
        if not path.is_file():
            self._json({"error": {"code": "NOT_FOUND", "message": "文件不存在"}}, HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _request_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 64 * 1024:
            raise DeploymentError("REQUEST_TOO_LARGE", "请求过大", "请刷新页面后重试。")
        if not length:
            return {}
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise DeploymentError("INVALID_REQUEST", "请求格式错误", "请刷新页面后重试。")
        return value

    def _error(self, exc: Exception) -> None:
        if isinstance(exc, DeploymentError):
            self._json({"error": exc.as_dict()}, HTTPStatus.BAD_REQUEST)
        else:
            self._json(
                {"error": {"code": type(exc).__name__, "message": "请求未完成", "suggestion": "检查服务日志。"}},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path in {"/", "/index.html"}:
                self._file(HERE / "index.html", "text/html; charset=utf-8")
            elif path == "/api/demo":
                self._json(self.app.payload)
            elif path == "/api/deployment/catalog":
                self._json(self.app.controller.catalog())
            elif path == "/api/runtime/status":
                self._json(self.app.controller.runtime_status())
            elif path == "/api/image/industrial/rgb":
                self._file(RGB_IMAGE, "image/bmp")
            elif path == "/api/image/industrial/infrared":
                self._file(INFRARED_IMAGE, "image/bmp")
            elif match := SESSION_GET_RE.fullmatch(path):
                self._json(self.app.controller.get_session(match.group(1)))
            elif path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
            else:
                self._json({"error": {"code": "NOT_FOUND", "message": "接口不存在"}}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._error(exc)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            request = self._request_json()
            if path == "/api/connections/test":
                self._json(
                    self.app.controller.probe_endpoints(
                        edge_url=request.get("edge_url"),
                        cloud_url=request.get("cloud_url"),
                        timeout_seconds=request.get("timeout_seconds", 3),
                    )
                )
            elif path == "/api/connections/test-topology":
                self._json(
                    self.app.controller.probe_topology(
                        nodes=request.get("nodes"),
                        timeout_seconds=request.get("timeout_seconds", 3),
                    )
                )
            elif path == "/api/connections/snapshot":
                self._json(
                    self.app.controller.runtime_snapshot(
                        edge_url=request.get("edge_url"),
                        cloud_url=request.get("cloud_url"),
                        timeout_seconds=request.get("timeout_seconds", 4),
                    )
                )
            elif path == "/api/deployment/sessions":
                self._json(self.app.controller.create_session(request), HTTPStatus.CREATED)
            elif match := SESSION_RE.fullmatch(path):
                self._json(self.app.controller.run_action(match.group(1), match.group(2), request))
            else:
                self._json({"error": {"code": "NOT_FOUND", "message": "接口不存在"}}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._error(exc)


class DemoServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], controller: GuidedDeploymentController, verbose: bool) -> None:
        super().__init__(address, DemoHandler)
        self.controller = controller
        self.payload = build_payload()
        self.verbose = verbose


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--host", default="127.0.0.1")
    value.add_argument("--port", type=int, default=8088)
    value.add_argument("--profiles", type=Path)
    value.add_argument("--state-root", type=Path, default=HERE / ".deployment-state")
    value.add_argument("--edge-url", default=os.getenv("CLOUD_EDGE_DEMO_EDGE_URL", "http://192.168.31.222:18101"))
    value.add_argument("--cloud-url", default=os.getenv("CLOUD_EDGE_DEMO_CLOUD_URL", "http://127.0.0.1:18100"))
    value.add_argument("--cloud-advertised-url", default=os.getenv("CLOUD_EDGE_DEMO_CLOUD_ADVERTISED_URL", "http://192.168.31.100:18100"))
    value.add_argument("--verbose", action="store_true")
    value.add_argument("--check", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.check:
        print(json.dumps(verify(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    controller = GuidedDeploymentController(
        profiles_path=args.profiles,
        state_root=args.state_root,
        connection_defaults={
            "edge_url": args.edge_url,
            "cloud_url": args.cloud_url,
            "cloud_advertised_url": args.cloud_advertised_url,
        },
    )
    server = DemoServer((args.host, args.port), controller, args.verbose)
    print(f"Final submission demo: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
