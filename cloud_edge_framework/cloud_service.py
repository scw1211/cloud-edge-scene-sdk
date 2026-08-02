"""用途：运行只承担复核、协调、反馈和幂等去重的独立云端服务。"""

import argparse
from dataclasses import replace
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Mapping

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
        self._aggregation_stop = threading.Event()
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
        self._aggregation_worker.start()

    def _aggregation_flush_loop(self) -> None:
        while not self._aggregation_stop.wait(0.05):
            try:
                result = self.flush_aggregations(64)
                self._aggregation_worker_state["cycles"] += 1
                self._aggregation_worker_state["completed"] += int(
                    result["completed"]
                )
                if result["errors"]:
                    self._aggregation_worker_state["errors"] = result[
                        "errors"
                    ][-10:]
            except Exception as exc:  # noqa: BLE001
                self._aggregation_worker_state["errors"] = [
                    "{}: {}".format(type(exc).__name__, exc)
                ]

    def health(self) -> Dict[str, Any]:
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
        }

    def protocol(self) -> Dict[str, Any]:
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
                "flush_aggregations": AGGREGATE_FLUSH_ENDPOINT,
                "aggregations": AGGREGATIONS_ENDPOINT,
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
        self.metrics.record_cloud_request("coordinate", elapsed_ms, replayed)
        self.metrics.record_coordination_result(result, replayed)
        return result

    def _complete_aggregation_lease(self, lease: Any) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            with self.manager.lease() as snapshot:
                coordination = snapshot.require_cloud().coordinate(lease.events)
            coordination = self._mark_aggregation_finality(coordination, lease)
            self.aggregator.complete(lease.group_id, coordination)
        except Exception as exc:
            self.aggregator.release(
                lease.group_id, "{}: {}".format(type(exc).__name__, exc)
            )
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.record_cloud_request("aggregate", elapsed_ms, False)
        self.metrics.record_coordination_result(coordination, False)
        return self.aggregator.get(lease.group_id)

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
        global_confirmation = bool(
            evidence_complete and observed_members_consistent
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
        lease = self.aggregator.claim(str(submission["group_id"]))
        if lease is None:
            coordination = submission.get("result")
            return {
                "aggregation": submission,
                "coordination": coordination,
                "late_submission": False,
                "cloud_accepted_at_ms": cloud_accepted_at_ms,
            }
        completed = self._complete_aggregation_lease(lease)
        return {
            "aggregation": completed,
            "coordination": completed.get("result"),
            "late_submission": False,
            "cloud_accepted_at_ms": cloud_accepted_at_ms,
        }

    def aggregate_batch(
        self,
        payload: Dict[str, Any],
        headers: Mapping[str, str],
    ) -> Dict[str, Any]:
        """Persist a group of edge summaries before attempting coordination.

        A short optional wait lets independently connected edges converge on
        the same completed group without making every member poll by itself.
        The endpoint remains durable and idempotent because member identity is
        still enforced by ``MultiEdgeEventAggregator.submit``.
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
            wait_ms = int(payload.get("wait_ms", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("request.wait_ms must be an integer") from exc
        if wait_ms < 0 or wait_ms > 2000:
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

        def complete_ready_groups() -> None:
            for group_id in group_ids:
                lease = self.aggregator.claim(group_id)
                if lease is not None:
                    self._complete_aggregation_lease(lease)

        complete_ready_groups()
        wait_deadline = time.monotonic() + (wait_ms / 1000.0)
        snapshots = {
            group_id: self.aggregator.get(group_id) for group_id in group_ids
        }
        while wait_ms and any(
            item.get("state") != "completed" for item in snapshots.values()
        ):
            remaining = wait_deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.005, remaining))
            complete_ready_groups()
            snapshots = {
                group_id: self.aggregator.get(group_id) for group_id in group_ids
            }

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
            "wait_ms": wait_ms,
            "batch_runtime_ms": round(elapsed_ms, 6),
            "all_terminal": all(group_terminal.values()),
        }

    def flush_aggregations(self, limit: int = 64) -> Dict[str, Any]:
        leases = self.aggregator.claim_due(limit)
        completed = []
        errors = []
        for lease in leases:
            try:
                completed.append(self._complete_aggregation_lease(lease))
            except Exception as exc:
                errors.append(
                    {
                        "group_id": lease.group_id,
                        "error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
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
        self._aggregation_stop.set()
        self._aggregation_worker.join(timeout=1.0)
        self._aggregation_worker_state["running"] = False
        self.manager.close()
        self.aggregator.close()


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
