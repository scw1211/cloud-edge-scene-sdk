"""用途：为严格分角色的边缘与云端服务提供统一 JSON HTTP 外壳。"""
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any, Dict, Mapping
from urllib.parse import urlsplit

from cloud_edge_framework.contracts import ContractError
from cloud_edge_framework.registry import PluginLoadError
from cloud_edge_framework.reliability import IdempotencyConflictError


class ApiNotFoundError(LookupError):
    pass

@dataclass(frozen=True)
class RawResponse:
    """在统一 JSON HTTP 外壳中返回非 JSON 的二进制响应。"""

    body: bytes
    content_type: str = "application/octet-stream"

def _headers(handler: BaseHTTPRequestHandler) -> Dict[str, str]:
    return {str(name).lower(): str(value) for name, value in handler.headers.items()}


def build_role_handler(service: Any, max_body_bytes: int, access_log: bool):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CloudEdgeFramework/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            if access_log:
                super().log_message(fmt, *args)

        def send_json(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(
                dict(payload), ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            trace_id = payload.get("trace_id")
            if trace_id:
                self.send_header("X-Trace-ID", str(trace_id))
            self.end_headers()
            self.wfile.write(body)

        def read_json(self) -> Dict[str, Any]:
            body = self.read_body()
            try:
                payload = json.loads(body.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ValueError("request body must use UTF-8") from exc
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            return payload

        def read_body(self) -> bytes:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if content_length <= 0 or content_length > max_body_bytes:
                raise ValueError(
                    "request body size must be within 1 and {} bytes".format(
                        max_body_bytes
                    )
                )
            return self.rfile.read(content_length)

        def _handle_error(self, exc: Exception) -> None:
            if isinstance(exc, ApiNotFoundError):
                status = HTTPStatus.NOT_FOUND
                code = "not_found"
            elif isinstance(exc, IdempotencyConflictError):
                status = HTTPStatus.CONFLICT
                code = "idempotency_conflict"
            elif isinstance(
                exc,
                (ContractError, PluginLoadError, KeyError, ValueError, json.JSONDecodeError),
            ):
                status = HTTPStatus.BAD_REQUEST
                code = "invalid_request"
            else:
                status = HTTPStatus.INTERNAL_SERVER_ERROR
                code = "framework_failure"
            self.send_json(
                status,
                {
                    "error": code,
                    "detail": "{}: {}".format(type(exc).__name__, exc),
                    "role": service.role,
                },
            )

        def do_GET(self) -> None:
            try:
                result = service.handle_get(urlsplit(self.path).path, _headers(self))
                self.send_json(HTTPStatus.OK, result)
            except Exception as exc:  # noqa: BLE001
                service.record_failure("GET", urlsplit(self.path).path)
                self._handle_error(exc)

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            try:
                payload = self.read_json()
                result = service.handle_post(path, payload, _headers(self))
                self.send_json(HTTPStatus.OK, result)
            except Exception as exc:  # noqa: BLE001
                service.record_failure("POST", path)
                self._handle_error(exc)

        def do_PUT(self) -> None:
            path = urlsplit(self.path).path
            try:
                if not hasattr(service, "handle_put"):
                    raise ApiNotFoundError(path)
                result = service.handle_put(path, self.read_body(), _headers(self))
                self.send_json(HTTPStatus.OK, result)
            except Exception as exc:  # noqa: BLE001
                service.record_failure("PUT", path)
                self._handle_error(exc)

    return Handler


def create_http_server(service: Any, host: str, port: int, max_body_bytes: int, access_log: bool):
    server = ThreadingHTTPServer(
        (host, int(port)),
        build_role_handler(service, int(max_body_bytes), bool(access_log)),
    )
    server.daemon_threads = True
    server.request_queue_size = 128
    return server
