"""Run a reproducible provisional-to-final reliability and safety fault matrix."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List

from cloud_edge_framework.conflicts import ConflictCoordinator
from cloud_edge_framework.contracts import SemanticEvent, build_decision
from cloud_edge_framework.event_envelope import SceneEventEnvelope
from cloud_edge_framework.plugins.base import ScenePlugin
from cloud_edge_framework.registry import SceneRegistry
from cloud_edge_framework.reliability import (
    IdempotencyConflictError,
    SQLiteIdempotencyStore,
    SQLiteOutbox,
)
from cloud_edge_framework.review_tracking import ReviewLifecycleStore
from cloud_edge_framework.runtime import CloudRuntime, EdgeRuntime
from cloud_edge_framework.scheduling import NetworkSnapshot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = (
    ROOT / "results" / "research" / "provisional_final_fault_matrix_20260727.json"
)
DEFAULT_MARKDOWN = (
    ROOT / "results" / "research" / "provisional_final_fault_matrix_20260727.md"
)


class MatrixPlugin(ScenePlugin):
    scene = "fault_matrix"
    aliases = ()
    event_types = ("org.example.fault-matrix.v1",)
    data_schema_id = "https://example.org/schemas/fault-matrix-v1.json"
    policy_version = "matrix-1.9.0"

    def payload_schema(self) -> Dict[str, Any]:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": self.data_schema_id,
            "type": "object",
            "required": [
                "entity",
                "resource",
                "window_start_ms",
                "window_end_ms",
                "action_value",
                "cloud_override",
            ],
            "properties": {
                "entity": {"type": "string", "minLength": 1},
                "resource": {"type": "string", "minLength": 1},
                "window_start_ms": {"type": "integer", "minimum": 0},
                "window_end_ms": {"type": "integer", "minimum": 0},
                "action_value": {"type": "number"},
                "cloud_override": {"type": "boolean"},
            },
            "additionalProperties": False,
        }

    def normalize(self, envelope: SceneEventEnvelope) -> SemanticEvent:
        self.validate_envelope(envelope)
        data = dict(envelope.data)
        return SemanticEvent.from_dict(
            {
                "schema_version": "1.0",
                "event_id": envelope.event_id,
                "scene": self.scene,
                "task": "fault_matrix_control",
                "edge_id": envelope.edge_id,
                "occurred_at_ms": envelope.occurred_at_ms,
                "scope": {
                    "entity_id": data["entity"],
                    "subsystem": "fault_matrix",
                    "state_variable": "control_state",
                    "region_id": "matrix_region",
                    "shared_resources": [data["resource"]],
                    "correlation_keys": ["matrix:" + data["entity"]],
                    "window_start_ms": data["window_start_ms"],
                    "window_end_ms": data["window_end_ms"],
                },
                "prediction": {
                    "label": "high",
                    "confidence": 0.92,
                    "probabilities": {"high": 0.92},
                    "values": {"action_value": data["action_value"]},
                },
                "risk": {"level": "high", "score": 0.90},
                "uncertainty": {
                    "confidence": 0.91,
                    "calibrated": True,
                    "prediction_set": ["high"],
                    "method": "fault_matrix_fixture",
                },
                "timing": {
                    "deadline_ms": 200.0,
                    "preprocessing_ms": 1.0,
                    "edge_inference_ms": 2.0,
                },
                "evidence": [
                    {
                        "evidence_id": envelope.event_id + "_summary",
                        "level": "summary",
                        "modality": "fixture",
                        "encoding": "json",
                        "inline": {"value": data["action_value"]},
                        "size_bytes": 16,
                        "content_type": "application/json",
                    },
                    {
                        "evidence_id": envelope.event_id + "_feature",
                        "level": "feature",
                        "modality": "fixture",
                        "encoding": "json",
                        "inline": [data["action_value"]],
                        "size_bytes": 8,
                        "content_type": "application/json",
                    },
                ],
                "candidate_actions": [
                    {
                        "action_type": "commit_control",
                        "target_ids": [data["entity"]],
                        "resource_ids": [data["resource"]],
                        "parameters": {
                            "min_risk_level": "high",
                            "value": data["action_value"],
                            "requires_cloud_confirmation": True,
                        },
                        "reason": "fixture irreversible action",
                        "priority": 90,
                    }
                ],
                "model": {"name": "matrix_fixture", "version": "1"},
                "scene_payload": data,
                "metadata": {"transport_include_scene_payload": True},
            }
        )

    def edge_decide(self, event: SemanticEvent):
        return self.decision_from_candidates(event, "matrix_edge", 0.92)

    def cloud_decide(self, event: SemanticEvent):
        if bool(event.scene_payload.get("cloud_override", False)):
            return build_decision(
                event=event,
                decision="monitor",
                actions=[],
                confidence=0.98,
                reason="cloud evidence rejects the provisional actuator action",
                source="matrix_cloud",
                policy_version=self.policy_version,
            )
        return self.decision_from_candidates(event, "matrix_cloud", 0.98)


class FailBeforeCommitCloud:
    def __init__(self) -> None:
        self.commit_count = 0

    def decide(self, event: SemanticEvent):
        del event
        raise ConnectionError("injected failure before cloud commit")


def _envelope(
    event_id: str,
    occurred_at_ms: int,
    resource: str,
    action_value: float = 1.0,
    cloud_override: bool = False,
    edge_id: str = "edge-matrix-1",
) -> Dict[str, Any]:
    event_time = datetime.fromtimestamp(
        occurred_at_ms / 1000.0, tz=timezone.utc
    ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return {
        "specversion": "1.0",
        "id": event_id,
        "source": "urn:edge:{}:fault-matrix".format(edge_id),
        "type": MatrixPlugin.event_types[0],
        "scene": MatrixPlugin.scene,
        "edgeid": edge_id,
        "time": event_time,
        "datacontenttype": "application/json",
        "dataschema": MatrixPlugin.data_schema_id,
        "data": {
            "entity": "entity-" + event_id,
            "resource": resource,
            "window_start_ms": occurred_at_ms - 100,
            "window_end_ms": occurred_at_ms,
            "action_value": action_value,
            "cloud_override": cloud_override,
        },
    }


def _normal_network() -> NetworkSnapshot:
    return NetworkSnapshot(
        available=True,
        rtt_ms=8.0,
        jitter_ms=1.0,
        cloud_queue_ms=1.0,
        cloud_compute_ms=5.0,
        uplink_mbps=100.0,
        downlink_mbps=100.0,
    )


def _offline_network() -> NetworkSnapshot:
    return replace(_normal_network(), available=False)


def _scenario(
    name: str,
    injected_fault: str,
    invariant: str,
    observed: Dict[str, Any],
    passed: bool,
) -> Dict[str, Any]:
    return {
        "scenario": name,
        "injected_fault": injected_fault,
        "invariant": invariant,
        "observed": observed,
        "passed": bool(passed),
    }


def run_matrix() -> Dict[str, Any]:
    scenarios: List[Dict[str, Any]] = []
    plugin = MatrixPlugin()
    registry = SceneRegistry([plugin])

    with tempfile.TemporaryDirectory(prefix="framework_fault_matrix_") as directory:
        temp = Path(directory)

        # 1. Cloud is already unavailable before an event is sent.
        tracker = ReviewLifecycleStore(temp / "pre_send_reviews.sqlite3")
        outbox = SQLiteOutbox(temp / "pre_send_outbox.sqlite3")
        runtime = EdgeRuntime(
            registry=registry,
            cloud=CloudRuntime(registry),
            review_store=outbox,
            review_tracker=tracker,
        )
        result = runtime.process(
            _envelope("pre-send", 1_000, "resource-pre-send"),
            network=_offline_network(),
        )
        authorization = result["local_decision"]["metadata"]["action_authorization"]
        passed = (
            result["local_decision"]["status"] == "provisional"
            and result["final_decision"]["route"] == "local_autonomy"
            and result["final_decision"]["status"] == "provisional"
            and result["review"]["state"] == "queued"
            and outbox.count() == 1
            and authorization["immediate_action_types"] == []
            and authorization["deferred_action_types"] == ["commit_control"]
        )
        scenarios.append(
            _scenario(
                "pre_send_cloud_outage",
                "network unavailable before transmission",
                "produce a provisional local result, persist exactly one review, and do not authorize the irreversible action",
                {
                    "local_status": result["local_decision"]["status"],
                    "immediate_route": result["final_decision"]["route"],
                    "global_status": result["final_decision"]["status"],
                    "review_state": result["review"]["state"],
                    "outbox_active": outbox.count(),
                    "action_authorization": authorization,
                },
                passed,
            )
        )
        tracker.close()

        # 2. The synchronous cloud path fails before any cloud decision commits.
        tracker = ReviewLifecycleStore(temp / "before_commit_reviews.sqlite3")
        outbox = SQLiteOutbox(temp / "before_commit_outbox.sqlite3")
        failing_cloud = FailBeforeCommitCloud()
        runtime = EdgeRuntime(
            registry=registry,
            cloud=failing_cloud,
            review_store=outbox,
            review_tracker=tracker,
        )
        result = runtime.process(
            _envelope("before-commit", 2_000, "resource-before-commit"),
            network=_normal_network(),
        )
        passed = (
            result["schedule"]["route"] == "cloud_sync"
            and result["final_decision"]["route"] == "local_autonomy"
            and result["final_decision"]["status"] == "provisional"
            and result["review"]["state"] == "queued"
            and failing_cloud.commit_count == 0
            and outbox.count() == 1
        )
        scenarios.append(
            _scenario(
                "failure_before_cloud_commit",
                "connection failure inside synchronous request before commit",
                "fall back locally and retain the event for replay without claiming a cloud commit",
                {
                    "scheduled_route": result["schedule"]["route"],
                    "fallback_route": result["final_decision"]["route"],
                    "global_status": result["final_decision"]["status"],
                    "review_state": result["review"]["state"],
                    "cloud_commit_count": failing_cloud.commit_count,
                    "outbox_active": outbox.count(),
                },
                passed,
            )
        )
        tracker.close()

        # 3. Cloud commits, but the response is lost. A retry must replay the cache.
        idempotency = SQLiteIdempotencyStore(
            temp / "cloud_idempotency.sqlite3", ttl_seconds=60.0, max_entries=100
        )
        commit_counter = {"count": 0}

        def commit_operation() -> Dict[str, Any]:
            commit_counter["count"] += 1
            return {"decision": "committed", "commit_sequence": commit_counter["count"]}

        request = {"event_id": "after-commit", "payload": {"value": 1}}
        first, first_replayed = idempotency.execute(
            "after-commit-key", request, commit_operation
        )
        # The first response is intentionally discarded to model a lost reply.
        second, second_replayed = idempotency.execute(
            "after-commit-key", request, commit_operation
        )
        passed = (
            not first_replayed
            and second_replayed
            and first == second
            and commit_counter["count"] == 1
        )
        scenarios.append(
            _scenario(
                "response_lost_after_cloud_commit",
                "discard the first response after durable idempotency commit",
                "retry returns the original response and executes the operation once",
                {
                    "first_replayed": first_replayed,
                    "retry_replayed": second_replayed,
                    "responses_equal": first == second,
                    "cloud_commit_count": commit_counter["count"],
                },
                passed,
            )
        )

        # 4. Restart the edge process while a review is queued, then replay it.
        restart_outbox_path = temp / "restart_outbox.sqlite3"
        restart_review_path = temp / "restart_reviews.sqlite3"
        tracker_one = ReviewLifecycleStore(restart_review_path)
        runtime_one = EdgeRuntime(
            registry=registry,
            cloud=CloudRuntime(registry),
            review_store=SQLiteOutbox(restart_outbox_path),
            review_tracker=tracker_one,
        )
        queued = runtime_one.process(
            _envelope("restart-recovery", 3_000, "resource-restart"),
            network=_offline_network(),
        )
        tracker_one.close()
        tracker_two = ReviewLifecycleStore(restart_review_path)
        restart_outbox = SQLiteOutbox(restart_outbox_path)
        runtime_two = EdgeRuntime(
            registry=registry,
            cloud=CloudRuntime(registry),
            review_store=restart_outbox,
            review_tracker=tracker_two,
        )
        flushed = runtime_two.flush_pending()
        record = tracker_two.get("restart-recovery")
        passed = (
            queued["review"]["state"] == "queued"
            and flushed["completed"] == 1
            and flushed["remaining"] == 0
            and record["state"] == "completed"
            and record["attempts"] == 1
        )
        scenarios.append(
            _scenario(
                "edge_restart_queue_recovery",
                "close the first edge runtime with one durable queued review",
                "a new runtime resumes the same outbox and completes the review exactly once",
                {
                    "state_before_restart": queued["review"]["state"],
                    "replay_completed": flushed["completed"],
                    "outbox_remaining": flushed["remaining"],
                    "review_state_after_restart": record["state"],
                    "review_attempts": record["attempts"],
                },
                passed,
            )
        )
        tracker_two.close()

        # 5. Related edges use numerically different policy versions.
        first_event = plugin.normalize(
            SceneEventEnvelope.from_dict(
                _envelope(
                    "policy-left",
                    4_000,
                    "resource-policy",
                    action_value=5.0,
                    edge_id="edge-left",
                )
            )
        )
        second_event = plugin.normalize(
            SceneEventEnvelope.from_dict(
                _envelope(
                    "policy-right",
                    4_100,
                    "resource-policy",
                    action_value=5.0,
                    edge_id="edge-right",
                )
            )
        )
        second_event = replace(
            second_event,
            scope=replace(
                second_event.scope,
                entity_id=first_event.scope.entity_id,
                correlation_keys=list(first_event.scope.correlation_keys),
            ),
        )
        left = replace(plugin.edge_decide(first_event), policy_version="matrix-1.9.0")
        right = replace(plugin.edge_decide(second_event), policy_version="matrix-1.10.0")
        coordination = ConflictCoordinator(registry).coordinate(
            [first_event, second_event], [left, right]
        )
        versions = [decision.policy_version for decision in coordination.decisions]
        passed = (
            len(coordination.initial_conflicts) == 1
            and not coordination.residual_conflicts
            and versions == ["matrix-1.10.0", "matrix-1.10.0"]
        )
        scenarios.append(
            _scenario(
                "policy_version_mismatch",
                "correlated edge decisions use versions 1.9.0 and 1.10.0",
                "cloud performs numeric version reconciliation and leaves no residual conflict",
                {
                    "initial_conflicts": len(coordination.initial_conflicts),
                    "residual_conflicts": len(coordination.residual_conflicts),
                    "final_versions": versions,
                    "resolution_success_rate": coordination.to_dict()[
                        "resolution_success_rate"
                    ],
                },
                passed,
            )
        )

        # 6. Events arrive out of timestamp order and one event is duplicated.
        replay_tracker = ReviewLifecycleStore(temp / "replay_reviews.sqlite3")
        replay_outbox = SQLiteOutbox(temp / "replay_outbox.sqlite3")
        replay_runtime = EdgeRuntime(
            registry=registry,
            cloud=CloudRuntime(registry),
            review_store=replay_outbox,
            review_tracker=replay_tracker,
        )
        ordered_inputs = [
            _envelope("replay-3", 6_000, "resource-replay-3"),
            _envelope("replay-1", 5_000, "resource-replay-1"),
            _envelope("replay-2", 5_500, "resource-replay-2"),
        ]
        for item in ordered_inputs:
            replay_runtime.process(item, network=_offline_network())
        # Direct runtime calls bypass the service idempotency cache.  An exact
        # source event retry must still be accepted even though its measured
        # runtime fields differ.  A real business-payload change under the same
        # event id must be rejected.
        replay_runtime.process(ordered_inputs[1], network=_offline_network())
        exact_duplicate_accepted = replay_outbox.count() == 3
        changed_duplicate = json.loads(json.dumps(ordered_inputs[1]))
        changed_duplicate["data"]["action_value"] = 0.12345
        changed_duplicate_rejected = False
        try:
            replay_runtime.process(changed_duplicate, network=_offline_network())
        except IdempotencyConflictError:
            changed_duplicate_rejected = True
        active_before = replay_outbox.count()
        flushed = replay_runtime.flush_pending()
        second_flush = replay_runtime.flush_pending()
        snapshot = replay_tracker.snapshot()
        passed = (
            active_before == 3
            and exact_duplicate_accepted
            and changed_duplicate_rejected
            and flushed["completed"] == 3
            and flushed["remaining"] == 0
            and second_flush["attempted"] == 0
            and snapshot["total"] == 3
            and snapshot["completed"] == 3
        )
        scenarios.append(
            _scenario(
                "duplicate_and_out_of_order_replay",
                "queue timestamps in order 6000, 5000, 5500 ms and submit the 5000 ms event twice",
                "accept an exact source retry, reject changed business data under the same event_id, accept out-of-order occurrence time, and complete each unique event once",
                {
                    "unique_outbox_events_before_replay": active_before,
                    "exact_duplicate_accepted": exact_duplicate_accepted,
                    "changed_duplicate_rejected": changed_duplicate_rejected,
                    "first_replay_completed": flushed["completed"],
                    "second_replay_attempted": second_flush["attempted"],
                    "review_total": snapshot["total"],
                    "review_completed": snapshot["completed"],
                },
                passed,
            )
        )
        replay_tracker.close()

        # 7. Cloud overturns a provisional irreversible action.
        overturn_tracker = ReviewLifecycleStore(temp / "overturn_reviews.sqlite3")
        overturn_runtime = EdgeRuntime(
            registry=registry,
            cloud=CloudRuntime(registry),
            review_store=SQLiteOutbox(temp / "overturn_outbox.sqlite3"),
            review_tracker=overturn_tracker,
        )
        overturned = overturn_runtime.process(
            _envelope(
                "cloud-overturn",
                7_000,
                "resource-overturn",
                cloud_override=True,
            ),
            network=_normal_network(),
        )
        local_auth = overturned["local_decision"]["metadata"]["action_authorization"]
        final_auth = overturned["final_decision"]["metadata"]["action_authorization"]
        violation_count = int(
            "commit_control" in local_auth["immediate_action_types"]
        )
        passed = (
            overturned["local_decision"]["status"] == "provisional"
            and overturned["local_decision"]["decision"] == "commit_control"
            and overturned["final_decision"]["status"] == "final"
            and overturned["final_decision"]["decision"] == "monitor"
            and overturned["review"]["decision_changed"] is True
            and violation_count == 0
            and local_auth["deferred_action_types"] == ["commit_control"]
            and final_auth["cloud_confirmed"] is True
        )
        scenarios.append(
            _scenario(
                "cloud_overturns_provisional_action",
                "cloud evidence rejects a proposed action that requires confirmation",
                "track the correction and never authorize the irreversible action before the final cloud decision",
                {
                    "local_status": overturned["local_decision"]["status"],
                    "local_decision": overturned["local_decision"]["decision"],
                    "final_status": overturned["final_decision"]["status"],
                    "final_decision": overturned["final_decision"]["decision"],
                    "decision_changed": overturned["review"]["decision_changed"],
                    "local_action_authorization": local_auth,
                    "final_action_authorization": final_auth,
                    "irreversible_action_violation_count": violation_count,
                },
                passed,
            )
        )
        overturn_tracker.close()

    passed_count = sum(bool(item["passed"]) for item in scenarios)
    return {
        "schema_version": 1,
        "benchmark": "provisional_to_final_fault_matrix",
        "scope": (
            "component-level deterministic fault injection over EdgeRuntime, "
            "SQLiteOutbox, ReviewLifecycleStore, SQLiteIdempotencyStore, and "
            "ConflictCoordinator; this is not a physical network-partition test"
        ),
        "scenario_count": len(scenarios),
        "passed_count": passed_count,
        "failed_count": len(scenarios) - passed_count,
        "all_passed": passed_count == len(scenarios),
        "safety_invariants": {
            "irreversible_action_violation_count": sum(
                int(
                    item["observed"].get(
                        "irreversible_action_violation_count", 0
                    )
                )
                for item in scenarios
            ),
            "duplicate_cloud_commit_count": max(
                0,
                int(
                    next(
                        item["observed"]["cloud_commit_count"]
                        for item in scenarios
                        if item["scenario"]
                        == "response_lost_after_cloud_commit"
                    )
                )
                - 1,
            ),
        },
        "scenarios": scenarios,
    }


def _markdown(report: Dict[str, Any]) -> str:
    descriptions = {
        "pre_send_cloud_outage": (
            "发送前云端断开",
            "事件发送前网络已经不可用",
            "必须给出边缘临时判断、持久保存一次复核任务，且不得提前执行不可逆动作",
        ),
        "failure_before_cloud_commit": (
            "同步复核提交前失败",
            "同步请求在云端提交结果前连接失败",
            "转入断网自治并保存待补传任务，不能声称云端已经提交",
        ),
        "response_lost_after_cloud_commit": (
            "云端提交后响应丢失",
            "云端已持久提交，但第一次响应在返回途中丢失",
            "重试必须返回原结果，云端操作只能执行一次",
        ),
        "edge_restart_queue_recovery": (
            "边缘重启后恢复队列",
            "仍有一条待复核任务时关闭边缘进程",
            "新进程必须从同一持久队列恢复，并且只完成一次复核",
        ),
        "policy_version_mismatch": (
            "决策策略版本不一致",
            "相关边缘分别使用1.9.0和1.10.0版本",
            "云端按数值版本协调到较新版本，且不能留下未解决冲突",
        ),
        "duplicate_and_out_of_order_replay": (
            "重复和乱序补传",
            "事件按乱序时间进入队列，其中一条又被重复提交",
            "必须按事件编号去重、接受乱序时间，并让每条唯一事件只完成一次",
        ),
        "cloud_overturns_provisional_action": (
            "云端推翻边缘临时动作",
            "云端证据否决一个需要确认的不可逆动作",
            "必须记录修正，且云端最终结果产生前不得授权不可逆动作",
        ),
    }
    observations = {
        "pre_send_cloud_outage": (
            "边缘状态：临时；全局状态：待复核；待补传1条；"
            "不可逆动作已延后"
        ),
        "failure_before_cloud_commit": (
            "原计划同步复核；失败后转断网自治；全局状态：待复核；"
            "云提交0次；待补传1条"
        ),
        "response_lost_after_cloud_commit": (
            "首次不是缓存重放；重试命中幂等缓存；两次响应一致；云提交1次"
        ),
        "edge_restart_queue_recovery": (
            "重启前待复核；重启后完成1条；队列剩余0条；实际尝试1次"
        ),
        "policy_version_mismatch": (
            "初始冲突1个；剩余冲突0个；两边最终均使用1.10.0版本"
        ),
        "duplicate_and_out_of_order_replay": (
            "补传前唯一事件3条；首次完成3条；再次补传0条；重复事件未新增"
        ),
        "cloud_overturns_provisional_action": (
            "边缘建议执行控制；云端最终改为继续监测；修正已记录；"
            "不可逆动作提前执行0次"
        ),
    }
    lines = [
        "# 边缘临时判断到云端最终结果的完整故障矩阵",
        "",
        "- 范围：对框架运行时、持久待传队列、复核生命周期、幂等缓存和冲突协调器进行组件级确定性故障注入；这不是物理网络分区实验。",
        "- 结果：{}/{} 通过；不可逆动作违规 {} 次；重复云提交 {} 次。".format(
            report["passed_count"],
            report["scenario_count"],
            report["safety_invariants"][
                "irreversible_action_violation_count"
            ],
            report["safety_invariants"]["duplicate_cloud_commit_count"],
        ),
        "",
        "| 场景 | 注入故障 | 必须保持的不变量 | 关键观测 | 结果 |",
        "|---|---|---|---|---|",
    ]
    for item in report["scenarios"]:
        title, fault, invariant = descriptions[item["scenario"]]
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                title,
                fault,
                invariant,
                observations[item["scenario"]],
                "通过" if item["passed"] else "失败",
            )
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "该矩阵验证的是框架状态机、持久队列、幂等缓存和冲突协调器。它不替代实验室应做的进程强制终止、真实断网、磁盘故障和服务器重启实验。",
            "",
            "框架现在明确区分“边缘临时判断”和“全局最终结果”。断网后需要补传复核的事件，全局状态为“待复核”，不再错误标成“最终”。动作若标记为“需要云端确认”，在确认前只会进入延后动作列表，不会进入立即动作列表。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = run_matrix()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
