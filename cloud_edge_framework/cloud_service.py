"""用途：运行只承担复核、协调、反馈和幂等去重的独立云端服务。"""

import argparse
from dataclasses import replace
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, List, Mapping, Tuple

from cloud_edge_framework.aggregation import AggregationSpec, MultiEdgeEventAggregator
from cloud_edge_framework.artifacts import EvidenceArtifactStore
from cloud_edge_framework.contracts import (
    SCHEMA_VERSION,
    DecisionEnvelope,
    SemanticEvent,
    stable_id,
)
from cloud_edge_framework.feedback import DecisionFeedbackStore
from cloud_edge_framework.http_api import ApiNotFoundError, create_http_server
from cloud_edge_framework.metrics import FrameworkMetrics
from cloud_edge_framework.plugin_manager import PluginRuntimeManager
from cloud_edge_framework.reliability import SQLiteIdempotencyStore
from cloud_edge_framework.service_config import FrameworkServiceConfig, load_service_config
from cloud_edge_framework.selective_evidence_pull import (
    HttpEvidencePullClient,
    SelectiveEvidencePullPlanner,
    fetch_pull_plan,
    merge_pulled_evidence,
)
from cloud_edge_framework.version import FRAMEWORK_VERSION


CLOUD_DECISION_ENDPOINT = "/api/v1/collaboration/cloud-decision"
COORDINATE_ENDPOINT = "/api/v1/collaboration/coordinate"
FEEDBACK_ENDPOINT = "/api/v1/collaboration/feedback"
PLUGINS_ENDPOINT = "/api/v1/collaboration/plugins"
RELOAD_ENDPOINT = "/api/v1/collaboration/plugins/reload"
SCHEMA_ENDPOINT = "/api/v1/collaboration/schema"
METRICS_ENDPOINT = "/api/v1/framework/metrics"
EVIDENCE_ENDPOINT_PREFIX = "/api/v1/evidence/"
AGGREGATE_ENDPOINT = "/api/v1/collaboration/aggregate"
AGGREGATE_BATCH_ENDPOINT = AGGREGATE_ENDPOINT + "/batch"
AGGREGATE_RESULTS_BATCH_ENDPOINT = AGGREGATE_ENDPOINT + "/results/batch"
AGGREGATE_FLUSH_ENDPOINT = AGGREGATE_ENDPOINT + "/flush"
AGGREGATIONS_ENDPOINT = "/api/v1/collaboration/aggregations"
AGGREGATIONS_ENDPOINT_PREFIX = AGGREGATIONS_ENDPOINT + "/"


class CloudApiService:
    role = "cloud"

    def __init__(self, project_root: Path, config: FrameworkServiceConfig) -> None:
        if config.role != self.role:
            raise ValueError("CloudApiService requires a cloud config")
        self.project_root = project_root.resolve()
        self.config = config
        artifact_root = config.storage.artifacts or (
            self.project_root / "runtime" / "framework_cloud_artifacts"
        )
        self.artifact_store = EvidenceArtifactStore(artifact_root)
        aggregation_path = config.storage.aggregations or (
            self.project_root / "runtime" / "framework_cloud_aggregations.sqlite3"
        )
        self.aggregator = MultiEdgeEventAggregator(aggregation_path)
        self.cloud_reviewer = None
        if config.cloud_llm is not None and config.cloud_llm.enabled:
            if config.cloud_llm.runtime_config is None:
                raise ValueError("enabled cloud_llm requires runtime_config")
            from edge_llm_factory.providers import load_provider
            from cloud_edge_framework.cloud_llm import CloudLLMReviewer

            self.cloud_reviewer = CloudLLMReviewer(
                load_provider(config.cloud_llm.runtime_config),
                min_risk_level=config.cloud_llm.min_risk_level,
            )
        feedback_store = DecisionFeedbackStore(config.storage.feedback)
        feedback_path = config.storage.feedback
        audit_path = (
            feedback_path.with_name(
                feedback_path.stem + "_cloud_llm_audits.jsonl"
            )
            if feedback_path is not None
            else self.project_root
            / "runtime"
            / "framework_cloud_llm_audits.jsonl"
        )
        self.cloud_audit_store = DecisionFeedbackStore(audit_path)
        self._cloud_audit_completed_ids = {
            str(record.get("feedback_id", ""))
            for record in self.cloud_audit_store.records()
        }
        self._cloud_audit_queue: queue.Queue = queue.Queue(maxsize=256)
        self._cloud_audit_pending = set()
        self._cloud_audit_lock = threading.RLock()
        self._cloud_audit_stop = threading.Event()
        self._cloud_audit_accepting = True
        self._cloud_audit_state: Dict[str, Any] = {
            "running": True,
            "queued": 0,
            "completed": 0,
            "failed": 0,
            "dropped": 0,
            "shutdown_incomplete": False,
            "errors": [],
            "business_result_blocking": False,
        }
        self._cloud_audit_worker = threading.Thread(
            target=self._cloud_audit_loop,
            name="cloud-llm-async-audit",
            daemon=True,
        )
        self.manager = PluginRuntimeManager(
            project_root=self.project_root,
            config_path=config.plugin_config,
            feedback_store=feedback_store,
            role="cloud",
            cloud_reviewer=self.cloud_reviewer,
        )
        idempotency_path = config.storage.idempotency or (
            self.project_root / "runtime" / "framework_cloud_idempotency.sqlite3"
        )
        self.idempotency = SQLiteIdempotencyStore(
            idempotency_path,
            ttl_seconds=config.idempotency.ttl_seconds,
            max_entries=config.idempotency.max_entries,
        )
        self.metrics = FrameworkMetrics(self.role)
        self.evidence_pull_client = None
        self.evidence_pull_planner = None
        evidence_pull = config.evidence_pull
        if evidence_pull is not None and evidence_pull.enabled:
            self.evidence_pull_client = HttpEvidencePullClient(
                evidence_pull.allowed_edge_base_urls,
                timeout_seconds=evidence_pull.fetch_timeout_seconds,
                max_response_bytes=evidence_pull.max_response_bytes,
            )
            self.evidence_pull_planner = SelectiveEvidencePullPlanner(
                evidence_pull.max_members_per_group
            )
        self._aggregation_stop = threading.Event()
        self._aggregation_wakeup = threading.Event()
        self._aggregation_worker_state: Dict[str, Any] = {
            "running": True,
            "cycles": 0,
            "completed": 0,
            "errors": [],
        }
        self._aggregation_worker = threading.Thread(
            target=self._aggregation_flush_loop,
            name="cloud-aggregation-timeout-flusher",
            daemon=True,
        )
        self._cloud_audit_worker.start()
        self._aggregation_worker.start()

    def _cloud_audit_loop(self) -> None:
        """Run optional LLM audits after the business result is available."""

        try:
            while True:
                try:
                    task = self._cloud_audit_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if task is None:
                    self._cloud_audit_queue.task_done()
                    break
                audit_id = str(task["audit_id"])
                try:
                    review = self.cloud_reviewer.review(
                        task["event"], task["decision"]
                    ).to_dict()
                    self.cloud_audit_store.append_record(
                        {
                            "schema_version": 1,
                            "feedback_id": audit_id,
                            "record_type": "cloud_llm_async_audit",
                            "created_at_ms": int(time.time() * 1000),
                            "group_id": task["group_id"],
                            "group_key": task["group_key"],
                            "event_id": task["event"].event_id,
                            "scene": task["event"].scene,
                            "business_decision_id": task["decision"].decision_id,
                            "business_decision": task["decision"].decision,
                            "business_result_changed": False,
                            "review": review,
                            "audit_latency_ms": review.get("latency_ms"),
                        }
                    )
                    with self._cloud_audit_lock:
                        self._cloud_audit_completed_ids.add(audit_id)
                        self._cloud_audit_state["completed"] += 1
                except Exception as exc:  # noqa: BLE001
                    with self._cloud_audit_lock:
                        self._cloud_audit_state["failed"] += 1
                        self._cloud_audit_state["errors"] = (
                            self._cloud_audit_state["errors"]
                            + ["{}: {}".format(type(exc).__name__, exc)]
                        )[-10:]
                finally:
                    with self._cloud_audit_lock:
                        self._cloud_audit_pending.discard(audit_id)
                    self._cloud_audit_queue.task_done()
        finally:
            with self._cloud_audit_lock:
                self._cloud_audit_state["running"] = False
                self._cloud_audit_state["shutdown_incomplete"] = False

    def _enqueue_cloud_audits(
        self,
        group_ids: List[str],
        event_groups: List[List[SemanticEvent]],
        coordination_groups: List[Dict[str, Any]],
    ) -> None:
        """Queue at most one audit for each fused group without delaying it."""

        if getattr(self, "cloud_reviewer", None) is None:
            return
        for group_id, events, coordination in zip(
            group_ids, event_groups, coordination_groups
        ):
            raw_decisions = coordination.get("decisions", [])
            if not isinstance(raw_decisions, list):
                continue
            for event, raw_decision in zip(events, raw_decisions):
                decision = DecisionEnvelope.from_dict(raw_decision)
                if decision.metadata.get("cloud_llm_audit_requested") is not True:
                    continue
                group_key = str(
                    decision.metadata.get("cloud_llm_review_group_key", group_id)
                )
                audit_id = stable_id(
                    "cloud_llm_async_audit", group_key, decision.decision_id
                )
                with self._cloud_audit_lock:
                    if not self._cloud_audit_accepting:
                        self._cloud_audit_state["dropped"] += 1
                        break
                    if (
                        audit_id in self._cloud_audit_pending
                        or audit_id in self._cloud_audit_completed_ids
                    ):
                        break
                    self._cloud_audit_pending.add(audit_id)
                    try:
                        self._cloud_audit_queue.put_nowait(
                            {
                                "audit_id": audit_id,
                                "group_id": group_id,
                                "group_key": group_key,
                                "event": event,
                                "decision": decision,
                            }
                        )
                    except queue.Full:
                        self._cloud_audit_pending.discard(audit_id)
                        self._cloud_audit_state["failed"] += 1
                        self._cloud_audit_state["dropped"] += 1
                        self._cloud_audit_state["errors"] = (
                            self._cloud_audit_state["errors"]
                            + ["async audit queue is full"]
                        )[-10:]
                    else:
                        self._cloud_audit_state["queued"] += 1
                break

    def _try_enqueue_cloud_audits(
        self,
        group_ids: List[str],
        event_groups: List[List[SemanticEvent]],
        coordination_groups: List[Dict[str, Any]],
    ) -> None:
        """Keep audit bookkeeping outside the business-result failure path."""

        try:
            self._enqueue_cloud_audits(
                group_ids,
                event_groups,
                coordination_groups,
            )
        except Exception as audit_exc:  # noqa: BLE001
            audit_lock = getattr(self, "_cloud_audit_lock", None)
            audit_state = getattr(self, "_cloud_audit_state", None)
            if audit_lock is not None and audit_state is not None:
                with audit_lock:
                    audit_state["failed"] += 1
                    audit_state["errors"] = (
                        audit_state["errors"]
                        + ["{}: {}".format(type(audit_exc).__name__, audit_exc)]
                    )[-10:]

    def _aggregation_flush_loop(self) -> None:
        while not self._aggregation_stop.is_set():
            self._aggregation_wakeup.wait(0.05)
            self._aggregation_wakeup.clear()
            if self._aggregation_stop.is_set():
                break
            try:
                self._aggregation_worker_state["cycles"] += 1
                while not self._aggregation_stop.is_set():
                    result = self.flush_aggregations(64)
                    self._aggregation_worker_state["completed"] += int(
                        result["completed"]
                    )
                    if result["errors"]:
                        self._aggregation_worker_state["errors"] = result[
                            "errors"
                        ][-10:]
                    if int(result["attempted"]) < 64:
                        break
            except Exception as exc:  # noqa: BLE001
                self._aggregation_worker_state["errors"] = [
                    "{}: {}".format(type(exc).__name__, exc)
                ]

    def _notify_aggregation_worker(self) -> None:
        wakeup = getattr(self, "_aggregation_wakeup", None)
        if wakeup is not None:
            wakeup.set()

    def health(self) -> Dict[str, Any]:
        evidence_pull_client = getattr(self, "evidence_pull_client", None)
        return {
            "status": "ok",
            "ready": True,
            "role": self.role,
            "framework_version": FRAMEWORK_VERSION,
            "schema_version": SCHEMA_VERSION,
            "runtime": self.manager.health(),
            "idempotency": self.idempotency.snapshot(),
            "artifacts": self.artifact_store.snapshot(),
            "aggregations": self.aggregator.snapshot(),
            "aggregation_worker": {
                **self._aggregation_worker_state,
                "running": self._aggregation_worker.is_alive(),
            },
            "cloud_llm_async_audit": {
                **self._cloud_audit_state,
                "running": self._cloud_audit_worker.is_alive(),
                "pending": self._cloud_audit_queue.unfinished_tasks,
                "records": self.cloud_audit_store.count(),
            },
            "selective_evidence_pull": {
                "enabled": evidence_pull_client is not None,
                "normal_path": "summary_only_no_pull",
                "owners_per_road_set": 2,
                "unique_member_cap_per_group": 4,
                "allowed_edge_base_urls": sorted(
                    evidence_pull_client.allowed_edge_base_urls
                )
                if evidence_pull_client is not None
                else [],
            },
        }

    def protocol(self) -> Dict[str, Any]:
        evidence_pull_client = getattr(self, "evidence_pull_client", None)
        return {
            "schema_version": SCHEMA_VERSION,
            "role": self.role,
            "accepted_input": "normalized SemanticEvent",
            "endpoints": {
                "cloud_decision": CLOUD_DECISION_ENDPOINT,
                "coordinate": COORDINATE_ENDPOINT,
                "feedback": FEEDBACK_ENDPOINT,
                "plugins": PLUGINS_ENDPOINT,
                "reload_plugins": RELOAD_ENDPOINT,
                "metrics": METRICS_ENDPOINT,
                "evidence": EVIDENCE_ENDPOINT_PREFIX + "{sha256}",
                "aggregate": AGGREGATE_ENDPOINT,
                "aggregate_batch": AGGREGATE_BATCH_ENDPOINT,
                "aggregate_results_batch": AGGREGATE_RESULTS_BATCH_ENDPOINT,
                "flush_aggregations": AGGREGATE_FLUSH_ENDPOINT,
                "aggregations": AGGREGATIONS_ENDPOINT,
                "selective_evidence_pull": (
                    "edge capability callback after a shared road-set trigger"
                ),
            },
            "selective_evidence_pull": {
                "enabled": evidence_pull_client is not None,
                "owners_per_road_set": 2,
                "unique_member_cap_per_group": 4,
                "allowed_edge_base_urls": sorted(
                    evidence_pull_client.allowed_edge_base_urls
                )
                if evidence_pull_client is not None
                else [],
            },
        }

    @staticmethod
    def _idempotency_key(
        headers: Mapping[str, str],
        prefix: str,
        *parts: str,
    ) -> str:
        supplied = str(headers.get("idempotency-key", "")).strip()
        return supplied or stable_id(prefix, *parts)

    def cloud_decision(
        self,
        payload: Dict[str, Any],
        headers: Mapping[str, str],
    ) -> Dict[str, Any]:
        cloud_accepted_at_ms = int(time.time() * 1000)
        event = payload.get("event")
        if not isinstance(event, dict):
            raise ValueError("request.event must be an object")
        event_id = str(event.get("event_id", "")).strip()
        if not event_id:
            raise ValueError("request.event.event_id must not be empty")
        request_key = self._idempotency_key(headers, "cloud_request", event_id)
        started = time.perf_counter()
        with self.manager.lease() as snapshot:
            def decide_once() -> Dict[str, Any]:
                response = dict(snapshot.require_cloud().decide_payload(event))
                response["cloud_accepted_at_ms"] = cloud_accepted_at_ms
                return response

            result, replayed = self.idempotency.execute(
                request_key,
                payload,
                decide_once,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result["idempotency_key"] = request_key
        result["idempotency_replay"] = replayed
        # Legacy cache rows may not contain the ingress timestamp.  New rows
        # persist it inside the idempotent response so retries retain the first
        # accepted time rather than reporting the retry time.
        result.setdefault("cloud_accepted_at_ms", cloud_accepted_at_ms)
        result["trace_id"] = str(headers.get("x-trace-id", "")) or str(
            event.get("metadata", {}).get("trace_id", "")
        )
        if getattr(self, "cloud_reviewer", None) is not None:
            self._try_enqueue_cloud_audits(
                [stable_id("direct_cloud_decision", event_id)],
                [[SemanticEvent.from_dict(event)]],
                [{"decisions": [result["decision"]]}],
            )
        self.metrics.record_cloud_request("cloud_decision", elapsed_ms, replayed)
        return result

    def coordinate(
        self,
        payload: Dict[str, Any],
        headers: Mapping[str, str],
    ) -> Dict[str, Any]:
        cloud_accepted_at_ms = int(time.time() * 1000)
        events = payload.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError("request.events must be a non-empty list")
        if not all(isinstance(item, dict) for item in events):
            raise ValueError("request.events must contain only objects")
        event_ids = [str(item.get("event_id", "")).strip() for item in events]
        if any(not value for value in event_ids):
            raise ValueError("every coordinated event must provide event_id")
        request_key = self._idempotency_key(
            headers, "coordinate_request", *sorted(event_ids)
        )
        started = time.perf_counter()
        with self.manager.lease() as snapshot:
            def coordinate_once() -> Dict[str, Any]:
                response = dict(snapshot.require_cloud().coordinate_payloads(events))
                response["cloud_accepted_at_ms"] = cloud_accepted_at_ms
                return response

            result, replayed = self.idempotency.execute(
                request_key,
                payload,
                coordinate_once,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result["idempotency_key"] = request_key
        result["idempotency_replay"] = replayed
        result.setdefault("cloud_accepted_at_ms", cloud_accepted_at_ms)
        result["trace_id"] = str(headers.get("x-trace-id", ""))
        if getattr(self, "cloud_reviewer", None) is not None:
            self._try_enqueue_cloud_audits(
                [stable_id("direct_coordinate", *sorted(event_ids))],
                [[SemanticEvent.from_dict(event) for event in events]],
                [result],
            )
        self.metrics.record_cloud_request("coordinate", elapsed_ms, replayed)
        self.metrics.record_coordination_result(result, replayed)
        return result

    def _complete_aggregation_lease(self, lease: Any) -> Dict[str, Any]:
        completed, errors = self._complete_aggregation_leases_batch([lease])
        if errors:
            raise RuntimeError(errors[0]["error"])
        return completed[0]

    @staticmethod
    def _coordinate_runtime_groups(
        runtime: Any,
        groups: List[List[SemanticEvent]],
    ) -> List[Dict[str, Any]]:
        if hasattr(runtime, "coordinate_groups"):
            return list(runtime.coordinate_groups(groups))
        return [runtime.coordinate(events) for events in groups]

    def _coordinate_with_selective_evidence_pull(
        self,
        runtime: Any,
        trusted_groups: List[List[SemanticEvent]],
    ) -> Tuple[List[Dict[str, Any]], List[List[SemanticEvent]]]:
        """Run summary coordination, then rerun after all missing targets arrive.

        Pulling is deliberately absent from the normal path.  A malformed,
        expired, missing, or timed-out two-member callback keeps the complete
        first-pass coordination result and never grants additional authority.
        """

        coordinated = self._coordinate_runtime_groups(runtime, trusted_groups)
        evidence_pull_client = getattr(self, "evidence_pull_client", None)
        evidence_pull_planner = getattr(self, "evidence_pull_planner", None)
        if evidence_pull_client is None or evidence_pull_planner is None:
            return coordinated, trusted_groups

        effective_groups = list(trusted_groups)
        final_results: List[Dict[str, Any]] = []
        for group_index, (events, initial) in enumerate(
            zip(trusted_groups, coordinated)
        ):
            try:
                plan = evidence_pull_planner.plan(events, initial)
            except Exception as exc:  # noqa: BLE001
                # Selective enrichment is not allowed to discard a completed
                # first pass.  Treat an unexpected planner failure as a
                # required-but-incomplete pull so finality remains conservative.
                selected_result = dict(initial)
                selected_result["evidence_pull"] = {
                    "schema_version": 1,
                    "triggered": True,
                    "trigger_reasons": ["pull_planner_failure"],
                    "road_set_ids": [],
                    "road_set_members": {},
                    "road_set_levels": {},
                    "requested_members": [],
                    "fetch_target_members": [],
                    "requested_level": None,
                    "initial_evidence_sufficient": False,
                    "budgets": {
                        "owners_per_road_set": 2,
                        "unique_members_per_group": 4,
                        "road_sets_per_group": 5,
                    },
                    "attempted": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "rerun_applied": False,
                    "fallback": "initial_summary_coordination",
                    "no_pull_reason": "pull_planner_failed",
                    "errors": ["{}: {}".format(type(exc).__name__, exc)],
                }
                self.metrics.increment("evidence_pull_groups_total")
                self.metrics.increment("evidence_pull_triggered_groups_total")
                self.metrics.increment("evidence_pull_fail_closed_total")
                final_results.append(selected_result)
                continue
            diagnostics: Dict[str, Any] = {
                "schema_version": 1,
                "triggered": plan.triggered,
                "trigger_reasons": list(plan.trigger_reasons),
                "road_set_ids": list(plan.road_set_ids),
                "road_set_members": dict(plan.road_set_members),
                "road_set_levels": dict(plan.road_set_levels),
                "requested_members": list(plan.requested_members),
                "fetch_target_members": [
                    target.member for target in plan.targets
                ],
                "requested_level": plan.requested_level,
                "initial_evidence_sufficient": (
                    plan.initial_evidence_sufficient
                ),
                "budgets": {
                    "owners_per_road_set": 2,
                    "unique_members_per_group": 4,
                    "road_sets_per_group": 5,
                },
                "attempted": 0,
                "succeeded": 0,
                "failed": 0,
                "rerun_applied": False,
                "fallback": None,
                "no_pull_reason": plan.no_pull_reason,
                "errors": [],
            }
            self.metrics.increment("evidence_pull_groups_total")
            selected_result = dict(initial)
            if not plan.triggered:
                self.metrics.increment("evidence_pull_no_trigger_total")
                selected_result["evidence_pull"] = diagnostics
                final_results.append(selected_result)
                continue

            self.metrics.increment("evidence_pull_triggered_groups_total")
            if not plan.targets:
                if plan.initial_evidence_sufficient:
                    self.metrics.increment(
                        "evidence_pull_already_present_total"
                    )
                    selected_result["evidence_pull"] = diagnostics
                    final_results.append(selected_result)
                    continue
                diagnostics["fallback"] = "initial_summary_coordination"
                self.metrics.increment("evidence_pull_fail_closed_total")
                selected_result["evidence_pull"] = diagnostics
                final_results.append(selected_result)
                continue

            diagnostics["attempted"] = len(plan.targets)
            self.metrics.increment(
                "evidence_pull_fetch_attempts_total", amount=len(plan.targets)
            )
            pull_started = time.perf_counter()
            pulled, errors = fetch_pull_plan(evidence_pull_client, plan)
            diagnostics["succeeded"] = len(pulled)
            diagnostics["failed"] = len(plan.targets) - len(pulled)
            diagnostics["errors"] = list(errors)
            self.metrics.increment(
                "evidence_pull_fetch_successes_total", amount=len(pulled)
            )
            self.metrics.increment(
                "evidence_pull_fetch_failures_total",
                amount=len(plan.targets) - len(pulled),
            )
            self.metrics.observe(
                "evidence_pull_pair_latency_ms",
                (time.perf_counter() - pull_started) * 1000.0,
            )
            self.metrics.observe(
                "evidence_pull_response_bytes",
                sum(value.response_bytes for value in pulled.values()),
            )
            if errors or len(pulled) != len(plan.targets):
                diagnostics["fallback"] = "initial_summary_coordination"
                diagnostics["no_pull_reason"] = "fetch_targets_incomplete"
                self.metrics.increment("evidence_pull_fail_closed_total")
                selected_result["evidence_pull"] = diagnostics
                final_results.append(selected_result)
                continue

            try:
                enriched = merge_pulled_evidence(events, pulled)
                rerun = self._coordinate_runtime_groups(runtime, [enriched])[0]
            except Exception as exc:  # noqa: BLE001
                diagnostics["errors"] = [
                    "rerun:{}: {}".format(type(exc).__name__, exc)
                ]
                diagnostics["fallback"] = "initial_summary_coordination"
                diagnostics["no_pull_reason"] = "evidence_rerun_failed"
                self.metrics.increment("evidence_pull_fail_closed_total")
            else:
                effective_groups[group_index] = enriched
                selected_result = dict(rerun)
                diagnostics["rerun_applied"] = True
                diagnostics["no_pull_reason"] = ""
                diagnostics["initial_summary_result"] = {
                    "initial_conflict_count": int(
                        initial.get("initial_conflict_count", 0)
                    ),
                    "residual_conflict_count": int(
                        initial.get("residual_conflict_count", 0)
                    ),
                    "globally_consistent": bool(
                        initial.get("globally_consistent", False)
                    ),
                }
                self.metrics.increment("evidence_pull_reruns_total")
            selected_result["evidence_pull"] = diagnostics
            final_results.append(selected_result)
        return final_results, effective_groups

    def _complete_aggregation_leases_batch(
        self,
        leases: List[Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        """Run one model batch and commit each sample as an isolated result.

        If one malformed group makes a scene plugin fail, recursively split
        the batch.  Healthy samples still complete, while the bad group alone
        returns to the durable retry queue.
        """
        if not leases:
            return [], []
        started = time.perf_counter()
        try:
            with self.manager.lease() as snapshot:
                runtime = snapshot.require_cloud()
                trusted_groups = [
                    self._events_with_trusted_aggregation_context(
                        lease, snapshot.registry
                    )
                    for lease in leases
                ]
                coordinated, audit_groups = (
                    self._coordinate_with_selective_evidence_pull(
                        runtime, trusted_groups
                    )
                )
            if len(coordinated) != len(leases):
                raise ValueError(
                    "cloud group batch changed aggregation result count"
                )
            marked = [
                self._mark_aggregation_finality(result, lease)
                for lease, result in zip(leases, coordinated)
            ]
            # Inference and validation finish before any group is committed.
            # SQLite then preserves the whole ready batch or no result.
            if hasattr(self.aggregator, "complete_many"):
                self.aggregator.complete_many(
                    [
                        (lease.group_id, result)
                        for lease, result in zip(leases, marked)
                    ]
                )
            else:
                for lease, result in zip(leases, marked):
                    self.aggregator.complete(lease.group_id, result)
            # The business result is durable before any optional 9B work is
            # queued.  Audit failures therefore cannot delay or replace it.
            self._try_enqueue_cloud_audits(
                [lease.group_id for lease in leases],
                audit_groups,
                marked,
            )
        except Exception as exc:
            if len(leases) > 1:
                middle = len(leases) // 2
                left_completed, left_errors = (
                    self._complete_aggregation_leases_batch(leases[:middle])
                )
                right_completed, right_errors = (
                    self._complete_aggregation_leases_batch(leases[middle:])
                )
                return (
                    left_completed + right_completed,
                    left_errors + right_errors,
                )
            lease = leases[0]
            error = "{}: {}".format(type(exc).__name__, exc)
            self.aggregator.release(lease.group_id, error)
            return [], [{"group_id": lease.group_id, "error": error}]

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.record_cloud_request("aggregate_batch_worker", elapsed_ms, False)
        completed = []
        for lease, result in zip(leases, marked):
            self.metrics.record_coordination_result(result, False)
            completed.append(self.aggregator.get(lease.group_id))
        self.metrics.increment(
            "aggregation_worker_groups_total", amount=len(completed)
        )
        return completed, []

    @staticmethod
    def _events_with_trusted_aggregation_context(
        lease: Any,
        registry: Any,
    ) -> List[SemanticEvent]:
        """Replace client member claims with the durable cloud lease contract."""
        expected_members = [str(value) for value in lease.expected_members]
        received_members = [str(value) for value in lease.received_members]
        missing_members = [str(value) for value in lease.missing_members]
        if (
            not expected_members
            or len(set(expected_members)) != len(expected_members)
            or len(set(received_members)) != len(received_members)
            or bool(set(received_members) - set(expected_members))
            or set(missing_members) != set(expected_members) - set(received_members)
        ):
            raise ValueError("aggregation lease member contract is inconsistent")
        complete = bool(
            lease.completion_reason == "all_expected_members"
            and not missing_members
            and set(received_members) == set(expected_members)
        )
        events: List[SemanticEvent] = []
        derived_members: List[str] = []
        for event in lease.events:
            if event.scene != str(lease.scene):
                raise ValueError(
                    "leased event {} disagrees with aggregation scene".format(
                        event.event_id
                    )
                )
            plugin = registry.get(event.scene)
            raw_spec = plugin.aggregation_spec(event)
            if raw_spec is None:
                raise ValueError(
                    "leased event {} has no aggregation spec".format(event.event_id)
                )
            spec = AggregationSpec.from_dict(raw_spec)
            if (
                spec.key != str(lease.group_key)
                or spec.expected_members != sorted(expected_members)
                or spec.member not in received_members
                or spec.member in derived_members
            ):
                raise ValueError(
                    "leased event {} disagrees with durable aggregation state".format(
                        event.event_id
                    )
                )
            derived_members.append(spec.member)
            metadata = dict(event.metadata)
            previous = metadata.get("aggregation")
            aggregation = dict(previous) if isinstance(previous, dict) else {}
            aggregation.update(
                {
                    "authority": "cloud_aggregation_lease",
                    "group_id": str(lease.group_id),
                    "key": str(lease.group_key),
                    "member": spec.member,
                    "expected_members": list(expected_members),
                    "received_members": list(received_members),
                    "missing_members": list(missing_members),
                    "completion_reason": str(lease.completion_reason),
                    "evidence_complete": complete,
                    "finality": "final" if complete else "partial_final",
                    "result_revision": int(lease.result_revision),
                }
            )
            metadata["aggregation"] = aggregation
            events.append(replace(event, metadata=metadata))
        if set(derived_members) != set(received_members):
            raise ValueError(
                "durable aggregation members do not match leased event payloads"
            )
        return events

    @staticmethod
    def _mark_aggregation_finality(
        coordination: Dict[str, Any],
        lease: Any,
    ) -> Dict[str, Any]:
        """Mark a timeout result as useful but non-authoritative.

        ``DecisionEnvelope.status`` deliberately keeps the version-1.0 enum:
        an incomplete cloud result is a cloud-derived provisional decision,
        while ``metadata.aggregation.finality`` carries the additive
        ``partial_final`` refinement.
        """
        if not isinstance(coordination, dict):
            raise ValueError("aggregation coordination result must be an object")
        expected_members = list(lease.expected_members)
        received_members = list(lease.received_members)
        missing_members = list(lease.missing_members)
        evidence_complete = (
            lease.completion_reason == "all_expected_members"
            and not missing_members
            and set(expected_members).issubset(set(received_members))
        )
        finality = "final" if evidence_complete else "partial_final"
        observed_members_consistent = bool(
            coordination.get("globally_consistent", False)
        )
        evidence_pull = coordination.get("evidence_pull", {})
        evidence_pull = (
            dict(evidence_pull) if isinstance(evidence_pull, dict) else {}
        )
        selective_pull_required = bool(evidence_pull.get("triggered", False))
        selective_pull_complete = bool(
            not selective_pull_required
            or evidence_pull.get("rerun_applied", False)
            or evidence_pull.get("initial_evidence_sufficient", False)
        )
        global_confirmation = bool(
            evidence_complete
            and observed_members_consistent
            and selective_pull_complete
        )
        aggregation_metadata = {
            "group_id": lease.group_id,
            "group_key": lease.group_key,
            "completion_reason": lease.completion_reason,
            "expected_members": expected_members,
            "received_members": received_members,
            "missing_members": missing_members,
            "finality": finality,
            "evidence_complete": evidence_complete,
            "completeness_basis": "expected_aggregation_members",
            "cloud_confirmed": global_confirmation,
            "global_confirmation": global_confirmation,
            "selective_evidence_pull_required": selective_pull_required,
            "selective_evidence_pull_complete": selective_pull_complete,
            "result_revision": int(lease.result_revision),
        }
        decisions = []
        for index, raw_decision in enumerate(coordination.get("decisions", [])):
            if not isinstance(raw_decision, dict):
                raise ValueError(
                    "aggregation coordination decision {} must be an object".format(
                        index
                    )
                )
            decision = DecisionEnvelope.from_dict(raw_decision)
            metadata = dict(decision.metadata)
            previous_aggregation = metadata.get("aggregation")
            merged_aggregation = (
                dict(previous_aggregation)
                if isinstance(previous_aggregation, dict)
                else {}
            )
            merged_aggregation.update(aggregation_metadata)
            metadata["aggregation"] = merged_aggregation
            immediate_actions = []
            deferred_actions = []
            for action in decision.actions:
                if (
                    action.parameters.get("requires_cloud_confirmation") is True
                    and not global_confirmation
                ):
                    deferred_actions.append(action.action_type)
                else:
                    immediate_actions.append(action.action_type)
            metadata.update(
                {
                    # The cloud did inspect the members that were present, but
                    # only a complete, consistent group is globally confirmed.
                    "cloud_reviewed": True,
                    "cloud_verified": global_confirmation,
                    "action_authorization": {
                        "cloud_confirmed": global_confirmation,
                        "immediate_action_types": immediate_actions,
                        "deferred_action_types": deferred_actions,
                        "all_actions_authorized": not deferred_actions,
                    },
                }
            )
            decisions.append(
                replace(
                    decision,
                    route="cloud_sync" if evidence_complete else "cloud_async",
                    status="final" if evidence_complete else "provisional",
                    metadata=metadata,
                ).to_dict()
            )
        result = dict(coordination)
        result.update(
            {
                "decisions": decisions,
                "aggregation_finality": finality,
                "evidence_complete": evidence_complete,
                "global_confirmation": global_confirmation,
                "result_revision": int(lease.result_revision),
                "observed_members_consistent": observed_members_consistent,
                # Consistency among received members is not proof of global
                # consistency while one or more expected members are absent.
                "globally_consistent": global_confirmation,
            }
        )
        return result

    def aggregate(
        self, payload: Dict[str, Any], headers: Mapping[str, str]
    ) -> Dict[str, Any]:
        request_received_at_ms = int(time.time() * 1000)
        del headers
        raw_event = payload.get("event")
        if not isinstance(raw_event, dict):
            raise ValueError("request.event must be an object")
        event = SemanticEvent.from_dict(raw_event)
        with self.manager.lease() as snapshot:
            plugin = snapshot.registry.get(event.scene)
            raw_spec = plugin.aggregation_spec(event)
        if raw_spec is None:
            raise ValueError("scene event does not request multi-edge aggregation")
        spec = AggregationSpec.from_dict(raw_spec)
        submission = self.aggregator.submit(event, spec)
        cloud_accepted_at_ms = int(
            submission.get("submitted_event_received_at_ms", request_received_at_ms)
        )
        # Ingress owns only durable acceptance.  The worker is notified after
        # the transaction commits and coordinates every currently-ready group
        # without holding this HTTP request open.
        self._notify_aggregation_worker()
        return {
            "aggregation": submission,
            "coordination": submission.get("result"),
            "late_submission": False,
            "cloud_accepted_at_ms": cloud_accepted_at_ms,
        }

    def aggregate_batch(
        self,
        payload: Dict[str, Any],
        headers: Mapping[str, str],
    ) -> Dict[str, Any]:
        """Durably accept summaries and return without waiting for peers.

        ``wait_ms`` from the short-lived 0.13.1 protocol is accepted only for
        rolling-upgrade compatibility and deliberately ignored.  Join
        completeness belongs to durable cloud state; model batching belongs
        to the background ready queue.
        """
        del headers
        raw_events = payload.get("events")
        if not isinstance(raw_events, list) or not raw_events:
            raise ValueError("request.events must be a non-empty list")
        if not all(isinstance(item, dict) for item in raw_events):
            raise ValueError("request.events must contain only objects")
        if len(raw_events) > 10000:
            raise ValueError("request.events exceeds the aggregation batch limit")
        try:
            requested_wait_ms = int(payload.get("wait_ms", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("request.wait_ms must be an integer") from exc
        if requested_wait_ms < 0 or requested_wait_ms > 2000:
            raise ValueError("request.wait_ms must be between 0 and 2000")

        events = [SemanticEvent.from_dict(item) for item in raw_events]
        event_ids = [event.event_id for event in events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("request.events contains duplicate event_id values")

        entries = []
        with self.manager.lease() as snapshot:
            for event in events:
                plugin = snapshot.registry.get(event.scene)
                raw_spec = plugin.aggregation_spec(event)
                if raw_spec is None:
                    raise ValueError(
                        "scene event {} does not request multi-edge aggregation".format(
                            event.event_id
                        )
                    )
                entries.append((event, AggregationSpec.from_dict(raw_spec)))

        started = time.perf_counter()
        accepted_at: Dict[str, int] = {}
        event_group_ids: Dict[str, str] = {}
        group_ids: List[str] = []
        for event, spec in entries:
            submission = self.aggregator.submit(event, spec)
            group_id = str(submission["group_id"])
            if group_id not in group_ids:
                group_ids.append(group_id)
            event_group_ids[event.event_id] = group_id
            accepted_at[event.event_id] = int(
                submission.get(
                    "submitted_event_received_at_ms", int(time.time() * 1000)
                )
            )

        self._notify_aggregation_worker()

        groups = []
        group_terminal: Dict[str, bool] = {}
        for group_id in group_ids:
            aggregation = self.aggregator.get(group_id)
            coordination = aggregation.pop("result", None)
            groups.append(
                {
                    "group_id": group_id,
                    "aggregation": aggregation,
                    "coordination": coordination,
                }
            )
            group_terminal[group_id] = aggregation.get("state") == "completed"

        items = []
        for event, _ in entries:
            group_id = event_group_ids[event.event_id]
            items.append(
                {
                    "event_id": event.event_id,
                    "group_id": group_id,
                    "late_submission": False,
                    "cloud_accepted_at_ms": accepted_at[event.event_id],
                }
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.record_cloud_request("aggregate_batch", elapsed_ms, False)
        self.metrics.increment(
            "aggregation_batch_events_total", amount=len(events)
        )
        return {
            "items": items,
            "groups": groups,
            "event_count": len(items),
            "group_count": len(group_ids),
            "wait_ms": 0,
            "requested_wait_ms_ignored": requested_wait_ms,
            "processing_mode": "durable_accept_then_background_ready_batch",
            "batch_runtime_ms": round(elapsed_ms, 6),
            "all_terminal": all(group_terminal.values()),
        }

    def aggregation_results_batch(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Read several durable group results without resubmitting summaries."""
        raw_items = payload.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("request.items must be a non-empty list")
        if len(raw_items) > 10000:
            raise ValueError("request.items exceeds the result batch limit")
        entries: List[Tuple[str, str]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                raise ValueError("request.items must contain only objects")
            event_id = str(item.get("event_id", "")).strip()
            group_id = str(item.get("group_id", "")).strip()
            if not event_id or not group_id:
                raise ValueError(
                    "every result item requires event_id and group_id"
                )
            entries.append((event_id, group_id))
        if len({event_id for event_id, _ in entries}) != len(entries):
            raise ValueError("request.items contains duplicate event_id values")

        started = time.perf_counter()
        snapshots: Dict[str, Dict[str, Any]] = {}
        accepted_at: Dict[str, int] = {}
        for event_id, group_id in entries:
            snapshot = self.aggregator.get(
                group_id, submitted_event_id=event_id
            )
            received_at_ms = snapshot.get("submitted_event_received_at_ms")
            if received_at_ms is None:
                raise ValueError(
                    "event {} is not a member of aggregation {}".format(
                        event_id, group_id
                    )
                )
            snapshots[group_id] = snapshot
            accepted_at[event_id] = int(received_at_ms)

        groups = []
        for group_id in dict.fromkeys(group_id for _, group_id in entries):
            aggregation = dict(snapshots[group_id])
            aggregation.pop("submitted_event_received_at_ms", None)
            coordination = aggregation.pop("result", None)
            groups.append(
                {
                    "group_id": group_id,
                    "aggregation": aggregation,
                    "coordination": coordination,
                }
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.record_cloud_request(
            "aggregate_results_batch", elapsed_ms, False
        )
        return {
            "items": [
                {
                    "event_id": event_id,
                    "group_id": group_id,
                    "cloud_accepted_at_ms": accepted_at[event_id],
                }
                for event_id, group_id in entries
            ],
            "groups": groups,
            "event_count": len(entries),
            "group_count": len(groups),
            "processing_mode": "result_lookup_without_summary_resubmission",
        }

    def flush_aggregations(self, limit: int = 64) -> Dict[str, Any]:
        leases = self.aggregator.claim_due(limit)
        completed, errors = self._complete_aggregation_leases_batch(leases)
        return {
            "attempted": len(leases),
            "completed": len(completed),
            "groups": completed,
            "errors": errors,
            "summary": self.aggregator.snapshot(),
        }

    def add_feedback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = payload.get("record")
        if not isinstance(record, dict):
            raise ValueError("request.record must be an object")
        accepted = self.manager.feedback_store.append_record(record)
        return {
            "accepted": accepted,
            "count": self.manager.feedback_store.count(),
        }

    def handle_get(self, path: str, headers: Mapping[str, str]) -> Dict[str, Any]:
        del headers
        if path == "/health":
            return self.health()
        if path == "/ready":
            return {"status": "ready", "ready": True, "role": self.role}
        if path == METRICS_ENDPOINT:
            return self.metrics.snapshot()
        if path == SCHEMA_ENDPOINT:
            return self.protocol()
        if path == PLUGINS_ENDPOINT:
            with self.manager.lease() as snapshot:
                return snapshot.describe()
        if path == FEEDBACK_ENDPOINT:
            return {
                "count": self.manager.feedback_store.count(),
                "recent": self.manager.feedback_store.recent(20),
            }
        if path == AGGREGATIONS_ENDPOINT:
            return self.aggregator.snapshot()
        if path.startswith(AGGREGATIONS_ENDPOINT_PREFIX):
            return self.aggregator.get(path[len(AGGREGATIONS_ENDPOINT_PREFIX):])
        if path.startswith(EVIDENCE_ENDPOINT_PREFIX):
            return self.artifact_store.describe(path[len(EVIDENCE_ENDPOINT_PREFIX):])
        raise ApiNotFoundError(path)

    def handle_post(
        self,
        path: str,
        payload: Dict[str, Any],
        headers: Mapping[str, str],
    ) -> Dict[str, Any]:
        if path == CLOUD_DECISION_ENDPOINT:
            return self.cloud_decision(payload, headers)
        if path == COORDINATE_ENDPOINT:
            return self.coordinate(payload, headers)
        if path == AGGREGATE_ENDPOINT:
            return self.aggregate(payload, headers)
        if path == AGGREGATE_BATCH_ENDPOINT:
            return self.aggregate_batch(payload, headers)
        if path == AGGREGATE_RESULTS_BATCH_ENDPOINT:
            return self.aggregation_results_batch(payload)
        if path == AGGREGATE_FLUSH_ENDPOINT:
            return self.flush_aggregations(int(payload.get("limit", 64)))
        if path == FEEDBACK_ENDPOINT:
            return self.add_feedback(payload)
        if path == RELOAD_ENDPOINT:
            return self.manager.reload()
        raise ApiNotFoundError(path)

    def handle_put(
        self,
        path: str,
        body: bytes,
        headers: Mapping[str, str],
    ) -> Dict[str, Any]:
        if not path.startswith(EVIDENCE_ENDPOINT_PREFIX):
            raise ApiNotFoundError(path)
        digest = path[len(EVIDENCE_ENDPOINT_PREFIX):]
        result = self.artifact_store.put(
            body,
            digest,
            content_type=str(headers.get("content-type", "application/octet-stream")),
            evidence_id=str(headers.get("x-evidence-id", "")),
        )
        result["received_bytes"] = len(body)
        self.metrics.increment("evidence_uploads_total")
        self.metrics.observe("evidence_upload_bytes", len(body))
        return result

    def record_failure(self, method: str, path: str) -> None:
        self.metrics.record_failure("{} {}".format(method, path))

    def close(self) -> None:
        with self._cloud_audit_lock:
            self._cloud_audit_accepting = False
        self._aggregation_stop.set()
        self._aggregation_wakeup.set()
        self._aggregation_worker.join(timeout=1.0)
        self._aggregation_worker_state["running"] = False
        self._cloud_audit_stop.set()
        dropped_ids = []
        with self._cloud_audit_lock:
            while True:
                try:
                    task = self._cloud_audit_queue.get_nowait()
                except queue.Empty:
                    break
                self._cloud_audit_queue.task_done()
                if task is not None:
                    audit_id = str(task["audit_id"])
                    dropped_ids.append(audit_id)
                    self._cloud_audit_pending.discard(audit_id)
            if dropped_ids:
                self._cloud_audit_state["dropped"] += len(dropped_ids)
                self._cloud_audit_state["errors"] = (
                    self._cloud_audit_state["errors"]
                    + [
                        "shutdown dropped {} queued async audit(s)".format(
                            len(dropped_ids)
                        )
                    ]
                )[-10:]
            if self._cloud_audit_worker.is_alive():
                self._cloud_audit_queue.put_nowait(None)
        self._cloud_audit_worker.join(timeout=1.0)
        with self._cloud_audit_lock:
            worker_alive = self._cloud_audit_worker.is_alive()
            self._cloud_audit_state["running"] = worker_alive
            self._cloud_audit_state["shutdown_incomplete"] = worker_alive
        self.manager.close()
        self.aggregator.close()
        self.idempotency.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cloud-only coordination service.")
    parser.add_argument(
        "--config",
        default="deployment/framework/cloud_service.json",
    )
    parser.add_argument(
        "--project_root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_service_config(config_path, project_root, expected_role="cloud")
    service = CloudApiService(project_root, config)
    server = create_http_server(
        service,
        config.listen.host,
        config.listen.port,
        config.listen.max_body_bytes,
        config.listen.access_log,
    )
    print(
        "Cloud service listening on http://{}:{}".format(
            config.listen.host, config.listen.port
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
