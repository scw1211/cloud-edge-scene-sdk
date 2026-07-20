"""用途：运行只承担复核、协调、反馈和幂等去重的独立云端服务。"""

import argparse
from pathlib import Path
import time
from typing import Any, Dict, Mapping

from cloud_edge_framework.contracts import SCHEMA_VERSION, stable_id
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


class CloudApiService:
    role = "cloud"

    def __init__(self, project_root: Path, config: FrameworkServiceConfig) -> None:
        if config.role != self.role:
            raise ValueError("CloudApiService requires a cloud config")
        self.project_root = project_root.resolve()
        self.config = config
        feedback_store = DecisionFeedbackStore(config.storage.feedback)
        self.manager = PluginRuntimeManager(
            project_root=self.project_root,
            config_path=config.plugin_config,
            feedback_store=feedback_store,
            role="cloud",
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

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "ready": True,
            "role": self.role,
            "framework_version": FRAMEWORK_VERSION,
            "schema_version": SCHEMA_VERSION,
            "runtime": self.manager.health(),
            "idempotency": self.idempotency.snapshot(),
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
        event = payload.get("event")
        if not isinstance(event, dict):
            raise ValueError("request.event must be an object")
        event_id = str(event.get("event_id", "")).strip()
        if not event_id:
            raise ValueError("request.event.event_id must not be empty")
        request_key = self._idempotency_key(headers, "cloud_request", event_id)
        started = time.perf_counter()
        with self.manager.lease() as snapshot:
            result, replayed = self.idempotency.execute(
                request_key,
                payload,
                lambda: snapshot.require_cloud().decide_payload(event),
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result["idempotency_key"] = request_key
        result["idempotency_replay"] = replayed
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
            result, replayed = self.idempotency.execute(
                request_key,
                payload,
                lambda: snapshot.require_cloud().coordinate_payloads(events),
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result["idempotency_key"] = request_key
        result["idempotency_replay"] = replayed
        result["trace_id"] = str(headers.get("x-trace-id", ""))
        self.metrics.record_cloud_request("coordinate", elapsed_ms, replayed)
        return result

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
        if path == FEEDBACK_ENDPOINT:
            return self.add_feedback(payload)
        if path == RELOAD_ENDPOINT:
            return self.manager.reload()
        raise ApiNotFoundError(path)

    def record_failure(self, method: str, path: str) -> None:
        self.metrics.record_failure("{} {}".format(method, path))

    def close(self) -> None:
        self.manager.close()


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
