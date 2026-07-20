"""用途：运行负责场景接入、本地决策、调度、自治和可靠重放的独立边缘服务。"""

import argparse
from pathlib import Path
import time
from typing import Any, Dict, Mapping

from cloud_edge_framework.contracts import SCHEMA_VERSION, stable_id
from cloud_edge_framework.feedback import DecisionFeedbackStore
from cloud_edge_framework.http_api import ApiNotFoundError, create_http_server
from cloud_edge_framework.metrics import FrameworkMetrics
from cloud_edge_framework.networking import CloudNetworkMonitor
from cloud_edge_framework.performance import PerformanceProfileStore
from cloud_edge_framework.plugin_manager import PluginRuntimeManager
from cloud_edge_framework.release_watcher import EdgeLLMReleaseWatcher
from cloud_edge_framework.reliability import SQLiteIdempotencyStore, SQLiteOutbox
from cloud_edge_framework.reliable_transport import ReliableHttpCloudClient
from cloud_edge_framework.replay import OutboxReplayWorker
from cloud_edge_framework.scheduling import CollaborationScheduler
from cloud_edge_framework.service_config import FrameworkServiceConfig, load_service_config
from cloud_edge_framework.version import FRAMEWORK_VERSION


DECIDE_ENDPOINT = "/api/v1/collaboration/decide"
FLUSH_ENDPOINT = "/api/v1/collaboration/flush-pending"
PLUGINS_ENDPOINT = "/api/v1/collaboration/plugins"
RELOAD_ENDPOINT = "/api/v1/collaboration/plugins/reload"
SCHEMA_ENDPOINT = "/api/v1/collaboration/schema"
METRICS_ENDPOINT = "/api/v1/framework/metrics"
OUTBOX_ENDPOINT = "/api/v1/framework/outbox"
EDGE_LLM_RELEASE_ENDPOINT = "/api/v1/framework/edge-llm/release"
EDGE_LLM_RELOAD_ENDPOINT = "/api/v1/framework/edge-llm/reload"


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
        self.metrics = FrameworkMetrics(self.role)
        self.manager = PluginRuntimeManager(
            project_root=self.project_root,
            config_path=config.plugin_config,
            review_store=self.outbox,
            performance_store=self.performance_store,
            feedback_store=self.feedback_store,
            role="edge",
            remote_cloud=self.cloud_client,
            scheduler=self.scheduler,
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
            return self.metrics.snapshot()
        if path == OUTBOX_ENDPOINT:
            return self.outbox.snapshot()
        if path == EDGE_LLM_RELEASE_ENDPOINT:
            return (
                self.release_watcher.health()
                if self.release_watcher is not None
                else {"status": "disabled"}
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
