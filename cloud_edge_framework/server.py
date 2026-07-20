"""用途：提供只依赖动态场景插件的云边调度、复核、协调和热重载 HTTP 服务。"""

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from cloud_edge_framework.contracts import ContractError, EVIDENCE_LEVELS, RISK_LEVELS, SCHEMA_VERSION
from cloud_edge_framework.feedback import DecisionFeedbackStore
from cloud_edge_framework.plugin_manager import PluginRuntimeManager
from cloud_edge_framework.performance import PerformanceProfileStore
from cloud_edge_framework.registry import PluginLoadError
from cloud_edge_framework.review_queue import PendingReviewStore
from cloud_edge_framework.scheduling import NetworkSnapshot


DECIDE_ENDPOINT = "/api/v1/collaboration/decide"
CLOUD_ENDPOINT = "/api/v1/collaboration/cloud-decision"
COORDINATE_ENDPOINT = "/api/v1/collaboration/coordinate"
FLUSH_ENDPOINT = "/api/v1/collaboration/flush-pending"
SCHEMA_ENDPOINT = "/api/v1/collaboration/schema"
PLUGINS_ENDPOINT = "/api/v1/collaboration/plugins"
RELOAD_ENDPOINT = "/api/v1/collaboration/plugins/reload"
FEEDBACK_ENDPOINT = "/api/v1/collaboration/feedback"


def protocol_description() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "external_ingress": {
            "format": "CloudEvents 1.0 structured JSON with framework routing extensions",
            "required_fields": [
                "specversion",
                "id",
                "source",
                "type",
                "scene",
                "edgeid",
                "time",
                "datacontenttype",
                "dataschema",
            ],
            "payload_rule": "exactly one of data or data_base64",
            "schema_file": "schemas/scene_event_envelope.schema.json",
        },
        "internal_cloud_transport": {
            "format": "normalized SemanticEvent; produced only by a scene plugin",
            "schema_file": "schemas/semantic_event.schema.json",
        },
        "risk_levels": list(RISK_LEVELS),
        "evidence_levels": list(EVIDENCE_LEVELS),
        "schema_files": [
            "schemas/scene_event_envelope.schema.json",
            "schemas/semantic_event.schema.json",
            "schemas/decision_envelope.schema.json",
        ],
        "endpoints": {
            "decide": DECIDE_ENDPOINT,
            "cloud_decision": CLOUD_ENDPOINT,
            "coordinate": COORDINATE_ENDPOINT,
            "flush_pending": FLUSH_ENDPOINT,
            "plugins": PLUGINS_ENDPOINT,
            "reload_plugins": RELOAD_ENDPOINT,
            "feedback": FEEDBACK_ENDPOINT,
        },
    }


class CollaborationService:
    def __init__(
        self,
        project_root: Path,
        review_queue_path: Optional[Path] = None,
        plugin_config_path: Optional[Path] = None,
        performance_profile_path: Optional[Path] = None,
        feedback_path: Optional[Path] = None,
    ) -> None:
        self.manager = PluginRuntimeManager(
            project_root=project_root,
            config_path=plugin_config_path,
            review_store=PendingReviewStore(review_queue_path),
            performance_store=PerformanceProfileStore(
                performance_profile_path,
                synchronous_persistence=False,
            ),
            feedback_store=DecisionFeedbackStore(feedback_path),
        )

    def health(self) -> Dict[str, Any]:
        runtime = self.manager.health()
        return {
            "status": "ok",
            "service": "componentized_cloud_edge_runtime",
            "schema_version": SCHEMA_VERSION,
            **runtime,
        }

    def plugins(self) -> Dict[str, Any]:
        with self.manager.lease() as snapshot:
            return snapshot.describe()

    def reload_plugins(self) -> Dict[str, Any]:
        return self.manager.reload()

    def feedback(self) -> Dict[str, Any]:
        return {
            "count": self.manager.feedback_store.count(),
            "recent": self.manager.feedback_store.recent(20),
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

    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        event = payload.get("event")
        if not isinstance(event, dict):
            raise ValueError("request.event must be an object")
        network = NetworkSnapshot.from_dict(payload.get("network", {}))
        with self.manager.lease() as snapshot:
            return snapshot.edge.process(
                event,
                network,
                conflict_suspected=bool(payload.get("conflict_suspected", False)),
                model_disagreement=bool(payload.get("model_disagreement", False)),
            )

    def cloud_decision(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        event = payload.get("event")
        if not isinstance(event, dict):
            raise ValueError("request.event must be an object")
        with self.manager.lease() as snapshot:
            return snapshot.cloud.decide_payload(event)

    def coordinate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        events = payload.get("events")
        if not isinstance(events, list) or not all(isinstance(item, dict) for item in events):
            raise ValueError("request.events must be a list of event objects")
        with self.manager.lease() as snapshot:
            return snapshot.cloud.coordinate_payloads(events)

    def flush_pending(self) -> Dict[str, Any]:
        with self.manager.lease() as snapshot:
            return snapshot.edge.flush_pending()

    def close(self) -> None:
        self.manager.close()


def build_handler(service: CollaborationService, max_body_bytes: int, access_log: bool):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CloudEdgeComponents/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            if access_log:
                super().log_message(fmt, *args)

        def send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_json(self) -> Dict[str, Any]:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if content_length <= 0 or content_length > max_body_bytes:
                raise ValueError("request body size must be within 1 and {} bytes".format(max_body_bytes))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            return payload

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/health":
                self.send_json(HTTPStatus.OK, service.health())
            elif path == SCHEMA_ENDPOINT:
                self.send_json(HTTPStatus.OK, protocol_description())
            elif path == PLUGINS_ENDPOINT:
                self.send_json(HTTPStatus.OK, service.plugins())
            elif path == FEEDBACK_ENDPOINT:
                self.send_json(HTTPStatus.OK, service.feedback())
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            try:
                if path == RELOAD_ENDPOINT:
                    self.read_json()
                    result = service.reload_plugins()
                elif path == FLUSH_ENDPOINT:
                    self.read_json()
                    result = service.flush_pending()
                else:
                    payload = self.read_json()
                    if path == DECIDE_ENDPOINT:
                        result = service.decide(payload)
                    elif path == CLOUD_ENDPOINT:
                        result = service.cloud_decision(payload)
                    elif path == COORDINATE_ENDPOINT:
                        result = service.coordinate(payload)
                    elif path == FEEDBACK_ENDPOINT:
                        result = service.add_feedback(payload)
                    else:
                        self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                        return
                self.send_json(HTTPStatus.OK, result)
            except (ContractError, PluginLoadError, KeyError, ValueError, json.JSONDecodeError) as exc:
                self.send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid_request", "detail": str(exc)},
                )
            except Exception as exc:  # noqa: BLE001
                self.send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "error": "framework_failure",
                        "detail": "{}: {}".format(type(exc).__name__, exc),
                    },
                )

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the componentized cloud-edge service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--plugin_config", default="deployment/framework/scene_plugins.json")
    parser.add_argument("--max_body_bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--review_queue", default="runtime/framework_pending_reviews.jsonl")
    parser.add_argument(
        "--performance_profiles",
        default="runtime/framework_performance_profiles.json",
    )
    parser.add_argument(
        "--feedback",
        default="runtime/framework_decision_feedback.jsonl",
    )
    parser.add_argument("--access_log", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.port <= 0 or args.max_body_bytes <= 0:
        raise ValueError("port and max_body_bytes must be positive")
    project_root = Path(args.project_root).resolve()
    plugin_config = Path(args.plugin_config)
    if not plugin_config.is_absolute():
        plugin_config = project_root / plugin_config
    review_queue_path = Path(args.review_queue) if args.review_queue else None
    if review_queue_path is not None and not review_queue_path.is_absolute():
        review_queue_path = project_root / review_queue_path
    performance_profile_path = (
        Path(args.performance_profiles) if args.performance_profiles else None
    )
    if performance_profile_path is not None and not performance_profile_path.is_absolute():
        performance_profile_path = project_root / performance_profile_path
    feedback_path = Path(args.feedback) if args.feedback else None
    if feedback_path is not None and not feedback_path.is_absolute():
        feedback_path = project_root / feedback_path
    service = CollaborationService(
        project_root,
        review_queue_path,
        plugin_config,
        performance_profile_path,
        feedback_path,
    )
    server = ThreadingHTTPServer(
        (args.host, args.port),
        build_handler(service, args.max_body_bytes, args.access_log),
    )
    server.daemon_threads = True
    server.request_queue_size = 128
    print("Cloud-edge component runtime listening on http://{}:{}".format(args.host, args.port))
    print("Scenes: {}".format(", ".join(service.manager.snapshot().registry.scenes())))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
