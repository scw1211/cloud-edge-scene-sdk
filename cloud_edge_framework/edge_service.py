"""用途：运行负责场景接入、本地决策、调度、自治和可靠重放的独立边缘服务。"""

import argparse
from pathlib import Path
import time
from typing import Any, Dict, Mapping

from cloud_edge_framework.contracts import SCHEMA_VERSION, stable_id
from cloud_edge_framework.feedback import DecisionFeedbackStore
from cloud_edge_framework.http_api import ApiNotFoundError, create_http_server
from cloud_edge_framework.metrics import FrameworkMetrics
from cloud_edge_framework.monitoring import CalibrationDriftMonitor, MonitoringPolicy
from cloud_edge_framework.networking import CloudNetworkMonitor
from cloud_edge_framework.performance import PerformanceProfileStore
from cloud_edge_framework.plugin_manager import PluginRuntimeManager
from cloud_edge_framework.release_watcher import EdgeLLMReleaseWatcher
from cloud_edge_framework.reliability import SQLiteIdempotencyStore, SQLiteOutbox
from cloud_edge_framework.review_tracking import ReviewLifecycleStore
from cloud_edge_framework.reliable_transport import ReliableHttpCloudClient
from cloud_edge_framework.replay import OutboxReplayWorker
from cloud_edge_framework.scheduling import CollaborationScheduler
from cloud_edge_framework.service_config import FrameworkServiceConfig, load_service_config
from cloud_edge_framework.version import FRAMEWORK_VERSION
from cloud_edge_framework.utility_routing import LearnedUtilityRouter


DECIDE_ENDPOINT = "/api/v1/collaboration/decide"
FLUSH_ENDPOINT = "/api/v1/collaboration/flush-pending"
PLUGINS_ENDPOINT = "/api/v1/collaboration/plugins"
RELOAD_ENDPOINT = "/api/v1/collaboration/plugins/reload"
SCHEMA_ENDPOINT = "/api/v1/collaboration/schema"
METRICS_ENDPOINT = "/api/v1/framework/metrics"
OUTBOX_ENDPOINT = "/api/v1/framework/outbox"
EDGE_LLM_RELEASE_ENDPOINT = "/api/v1/framework/edge-llm/release"
EDGE_LLM_RELOAD_ENDPOINT = "/api/v1/framework/edge-llm/reload"
REVIEWS_ENDPOINT = "/api/v1/collaboration/reviews"
REVIEWS_ENDPOINT_PREFIX = REVIEWS_ENDPOINT + "/"
MONITORING_ENDPOINT = "/api/v1/collaboration/monitoring"
MONITORING_ENDPOINT_PREFIX = MONITORING_ENDPOINT + "/"
MONITORING_OUTCOME_ENDPOINT = MONITORING_ENDPOINT + "/outcome"
MONITORING_REFERENCE_ENDPOINT = MONITORING_ENDPOINT + "/reference"
ROUTING_DATASET_ENDPOINT = "/api/v1/collaboration/routing-dataset"


class EdgeApiService:
    role = "edge"

    def __init__(
        self,
        project_root: Path,
        config: FrameworkServiceConfig,
        network_monitor: Any = None,
    ) -> None:
        if config.role != self.role or config.cloud is None:
            raise ValueError("EdgeApiService requires an edge config with cloud settings")
        self.project_root = project_root.resolve()
        self.config = config
        outbox_path = config.storage.outbox or (
            self.project_root / "runtime" / "framework_edge_outbox.sqlite3"
        )
        idempotency_path = config.storage.idempotency or (
            self.project_root / "runtime" / "framework_edge_idempotency.sqlite3"
        )
        self.outbox = SQLiteOutbox(outbox_path)
        self.idempotency = SQLiteIdempotencyStore(
            idempotency_path,
            ttl_seconds=config.idempotency.ttl_seconds,
            max_entries=config.idempotency.max_entries,
        )
        self.performance_store = PerformanceProfileStore(
            config.storage.performance_profiles,
            synchronous_persistence=False,
        )
        self.feedback_store = DecisionFeedbackStore(config.storage.feedback)
        review_path = config.storage.reviews or (
            self.project_root / "runtime" / "framework_edge_reviews.sqlite3"
        )
        self.review_tracker = ReviewLifecycleStore(review_path)
        monitoring_path = config.storage.monitoring or (
            self.project_root / "runtime" / "framework_edge_monitoring.sqlite3"
        )
        monitoring_config = config.monitoring
        monitoring_enabled = (
            monitoring_config is None or monitoring_config.enabled
        )
        monitoring_policy = (
            MonitoringPolicy()
            if monitoring_config is None
            else MonitoringPolicy(
                window_size=monitoring_config.window_size,
                bins=monitoring_config.bins,
                min_labeled_samples=monitoring_config.min_labeled_samples,
                min_drift_samples=monitoring_config.min_drift_samples,
                bootstrap_reference_size=monitoring_config.bootstrap_reference_size,
                max_ece=monitoring_config.max_ece,
                target_coverage=monitoring_config.target_coverage,
                coverage_tolerance=monitoring_config.coverage_tolerance,
                max_psi=monitoring_config.max_psi,
                evaluation_interval_events=(
                    monitoring_config.evaluation_interval_events
                ),
                evaluation_max_staleness_ms=(
                    monitoring_config.evaluation_max_staleness_ms
                ),
            )
        )
        self.calibration_monitor = (
            CalibrationDriftMonitor(monitoring_path, monitoring_policy)
            if monitoring_enabled
            else None
        )
        self.cloud_client = ReliableHttpCloudClient(
            config.cloud.base_url,
            timeout_seconds=config.cloud.timeout_seconds,
            max_attempts=config.cloud.max_attempts,
            retry_backoff_seconds=config.cloud.retry_backoff_seconds,
        )
        self.scheduler = CollaborationScheduler(
            confidence_threshold=config.scheduler.confidence_threshold,
            jitter_guard=config.scheduler.jitter_guard,
        )
        self.utility_router = None
        if config.utility_router is not None and config.utility_router.enabled:
            if config.utility_router.artifact is None:
                raise ValueError("enabled utility router requires an artifact")
            self.utility_router = LearnedUtilityRouter.load(
                config.utility_router.artifact, mode=config.utility_router.mode
            )
        self.metrics = FrameworkMetrics(self.role)
        self.manager = PluginRuntimeManager(
            project_root=self.project_root,
            config_path=config.plugin_config,
            review_store=self.outbox,
            performance_store=self.performance_store,
            feedback_store=self.feedback_store,
            review_tracker=self.review_tracker,
            role="edge",
            remote_cloud=self.cloud_client,
            scheduler=self.scheduler,
            calibration_monitor=self.calibration_monitor,
            utility_router=self.utility_router,
        )
        self.release_watcher = None
        if config.release_watch is not None and config.release_watch.enabled:
            if config.release_watch.registry is None:
                raise ValueError("enabled release watcher requires registry")
            self.release_watcher = EdgeLLMReleaseWatcher(
                manager=self.manager,
                registry_path=config.release_watch.registry,
                interval_seconds=config.release_watch.interval_seconds,
            )
            self.release_watcher.start()

        self.network_monitor = network_monitor or CloudNetworkMonitor(
            config.cloud.base_url,
            timeout_seconds=config.cloud.timeout_seconds,
            config=config.network_probe,
        )
        self.network_monitor.start()
        self.replay_worker = OutboxReplayWorker(
            manager=self.manager,
            outbox=self.outbox,
            network_monitor=self.network_monitor,
            config=config.replay,
            metrics=self.metrics,
        )
        self.replay_worker.start()

    def health(self) -> Dict[str, Any]:
        network = self.network_monitor.health()
        return {
            "status": "ok",
            "ready": True,
            "role": self.role,
            "framework_version": FRAMEWORK_VERSION,
            "schema_version": SCHEMA_VERSION,
            "cloud_available": bool(network["snapshot"]["available"]),
            "runtime": self.manager.health(),
            "network": network,
            "outbox": self.outbox.snapshot(),
            "replay": self.replay_worker.health(),
            "idempotency": self.idempotency.snapshot(),
            "reviews": self.review_tracker.snapshot(),
            "monitoring": (
                self.calibration_monitor.snapshot()
                if self.calibration_monitor is not None
                else {"status": "disabled"}
            ),
            "edge_llm_release": self.release_watcher.health()
            if self.release_watcher is not None
            else {"status": "disabled"},
        }

    def protocol(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "role": self.role,
            "accepted_input": "SceneEventEnvelope",
            "network_source": "edge-owned active HTTP probe",
            "endpoints": {
                "decide": DECIDE_ENDPOINT,
                "flush_pending": FLUSH_ENDPOINT,
                "plugins": PLUGINS_ENDPOINT,
                "reload_plugins": RELOAD_ENDPOINT,
                "edge_llm_release": EDGE_LLM_RELEASE_ENDPOINT,
                "reload_edge_llm": EDGE_LLM_RELOAD_ENDPOINT,
                "metrics": METRICS_ENDPOINT,
                "outbox": OUTBOX_ENDPOINT,
                "reviews": REVIEWS_ENDPOINT,
                "review": REVIEWS_ENDPOINT_PREFIX + "{review_or_event_id}",
                "monitoring": MONITORING_ENDPOINT,
                "monitoring_scene": MONITORING_ENDPOINT_PREFIX + "{scene}",
                "monitoring_outcome": MONITORING_OUTCOME_ENDPOINT,
                "monitoring_reference": MONITORING_REFERENCE_ENDPOINT,
                "routing_dataset": ROUTING_DATASET_ENDPOINT,
            },
        }

    def decide(
        self,
        payload: Dict[str, Any],
        headers: Mapping[str, str],
    ) -> Dict[str, Any]:
        event = payload.get("event")
        if not isinstance(event, dict):
            raise ValueError("request.event must be an object")
        event_id = str(event.get("id", "")).strip()
        if not event_id:
            raise ValueError("request.event.id must not be empty")
        supplied_trace = str(headers.get("x-trace-id", "")).strip()
        if supplied_trace and not event.get("traceid"):
            event = dict(event)
            event["traceid"] = supplied_trace
        request_key = str(headers.get("idempotency-key", "")).strip() or stable_id(
            "edge_request", event_id
        )
        network = self.network_monitor.snapshot()
        started = time.perf_counter()

        def operation() -> Dict[str, Any]:
            with self.manager.lease() as snapshot:
                return snapshot.require_edge().process(
                    event,
                    network,
                    conflict_suspected=bool(
                        payload.get("conflict_suspected", False)
                    ),
                    model_disagreement=bool(payload.get("model_disagreement", False)),
                )

        result, replayed = self.idempotency.execute(request_key, payload, operation)
        # Waking the worker unconditionally is cheaper than asking the Outbox
        # how much work exists: `count()` is a second SQLite round trip on the
        # request path, while a spurious wake only costs the background thread
        # one empty claim.  The worker owns all HTTP delivery, so provisional
        # latency is never tied to cloud acknowledgement latency.
        self.replay_worker.notify()
        result["idempotency_key"] = request_key
        result["idempotency_replay"] = replayed
        result["edge_service_wall_ms"] = round(
            (time.perf_counter() - started) * 1000.0, 6
        )
        self.metrics.record_edge_result(result)
        if replayed:
            self.metrics.increment("idempotency_replays_total", operation="edge_decide")
        return result

    def handle_get(self, path: str, headers: Mapping[str, str]) -> Dict[str, Any]:
        del headers
        if path == "/health":
            return self.health()
        if path == "/ready":
            return {
                "status": "ready",
                "ready": True,
                "role": self.role,
                "cloud_required_for_readiness": False,
            }
        if path == METRICS_ENDPOINT:
            result = self.metrics.snapshot()
            result["review_lifecycle"] = self.review_tracker.snapshot()
            result["calibration_drift_monitor"] = (
                self.calibration_monitor.snapshot()
                if self.calibration_monitor is not None
                else {"status": "disabled"}
            )
            return result
        if path == OUTBOX_ENDPOINT:
            return self.outbox.snapshot()
        if path == ROUTING_DATASET_ENDPOINT:
            return self.review_tracker.routing_dataset()
        if path == EDGE_LLM_RELEASE_ENDPOINT:
            return (
                self.release_watcher.health()
                if self.release_watcher is not None
                else {"status": "disabled"}
            )
        if path == REVIEWS_ENDPOINT:
            return {
                "summary": self.review_tracker.snapshot(),
                "recent": self.review_tracker.recent(20),
            }
        if path.startswith(REVIEWS_ENDPOINT_PREFIX):
            return self.review_tracker.get(path[len(REVIEWS_ENDPOINT_PREFIX):])
        if path == MONITORING_ENDPOINT:
            return (
                self.calibration_monitor.snapshot()
                if self.calibration_monitor is not None
                else {"status": "disabled"}
            )
        if path.startswith(MONITORING_ENDPOINT_PREFIX):
            if self.calibration_monitor is None:
                return {"status": "disabled"}
            return self.calibration_monitor.scene_snapshot(
                path[len(MONITORING_ENDPOINT_PREFIX):]
            )
        if path == SCHEMA_ENDPOINT:
            return self.protocol()
        if path == PLUGINS_ENDPOINT:
            with self.manager.lease() as snapshot:
                return snapshot.describe()
        raise ApiNotFoundError(path)

    def handle_post(
        self,
        path: str,
        payload: Dict[str, Any],
        headers: Mapping[str, str],
    ) -> Dict[str, Any]:
        if path == DECIDE_ENDPOINT:
            return self.decide(payload, headers)
        if path == FLUSH_ENDPOINT:
            return self.replay_worker.run_once()
        if path == RELOAD_ENDPOINT:
            return self.manager.reload()
        if path == MONITORING_OUTCOME_ENDPOINT:
            if self.calibration_monitor is None:
                raise ValueError("calibration and drift monitoring is disabled")
            return self.calibration_monitor.record_outcome(
                str(payload.get("event_id", "")),
                str(payload.get("true_label", "")),
            )
        if path == MONITORING_REFERENCE_ENDPOINT:
            if self.calibration_monitor is None:
                raise ValueError("calibration and drift monitoring is disabled")
            samples = payload.get("samples")
            if not isinstance(samples, list):
                raise ValueError("monitoring reference samples must be a list")
            return self.calibration_monitor.set_reference(
                str(payload.get("scene", "")),
                str(payload.get("signal", "")),
                samples,
                source=str(payload.get("source", "validation")),
            )
        if path == EDGE_LLM_RELOAD_ENDPOINT:
            if self.release_watcher is None:
                raise ValueError("Edge LLM release watcher is disabled")
            return self.release_watcher.run_once(force=True)
        raise ApiNotFoundError(path)

    def record_failure(self, method: str, path: str) -> None:
        self.metrics.record_failure("{} {}".format(method, path))

    def close(self) -> None:
        if self.release_watcher is not None:
            self.release_watcher.stop()
        self.replay_worker.stop()
        self.network_monitor.stop()
        self.cloud_client.flush_feedback()
        self.manager.close()
        if self.calibration_monitor is not None:
            self.calibration_monitor.close()
        self.review_tracker.close()
        # Released last: the replay worker and runtime above may still touch the
        # durable Outbox while they wind down.
        self.outbox.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the edge-only collaboration service.")
    parser.add_argument(
        "--config",
        default="deployment/framework/edge_service.json",
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
    config = load_service_config(config_path, project_root, expected_role="edge")
    service = EdgeApiService(project_root, config)
    server = create_http_server(
        service,
        config.listen.host,
        config.listen.port,
        config.listen.max_body_bytes,
        config.listen.access_log,
    )
    print(
        "Edge service listening on http://{}:{}".format(
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
