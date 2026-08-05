"""用途：为严格分角色的边缘与云端服务提供统一 JSON HTTP 外壳。"""

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any, Dict, Mapping
from urllib.parse import urlsplit

from cloud_edge_framework.contracts import ContractError
from cloud_edge_framework.registry import PluginLoadError
from cloud_edge_framework.reliability import IdempotencyConflictError


_KEEP_ALIVE_IDLE_TIMEOUT_SECONDS = 30.0


class ApiNotFoundError(LookupError):
    pass


def _headers(handler: BaseHTTPRequestHandler) -> Dict[str, str]:
    return {str(name).lower(): str(value) for name, value in handler.headers.items()}


def build_role_handler(service: Any, max_body_bytes: int, access_log: bool):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CloudEdgeFramework/0.1"
        # Every JSON response carries an exact Content-Length, so one accepted
        # TCP connection can safely serve the next request.  The default on
        # BaseHTTPRequestHandler is HTTP/1.0, which forces clients to reconnect
        # and hides connection setup inside every /decide latency sample.
        protocol_version = "HTTP/1.1"
        keep_alive_idle_timeout_seconds = _KEEP_ALIVE_IDLE_TIMEOUT_SECONDS

        def handle(self) -> None:
            """Bound only the idle wait for each HTTP request line.

            ``BaseHTTPRequestHandler.handle`` otherwise waits forever for both
            the first request and later requests on an HTTP/1.1 connection,
            pinning one server thread per idle client. The timeout is removed as
            soon as a request line is complete, so request-body reads and
            long-running service handlers retain their existing semantics.
            """
            self.close_connection = True
            normal_timeout = self.connection.gettimeout()
            self._normal_request_timeout = normal_timeout
            idle_timeout = float(self.keep_alive_idle_timeout_seconds)
            if normal_timeout is not None:
                idle_timeout = min(idle_timeout, float(normal_timeout))
            while True:
                self._waiting_for_request_line = True
                self.connection.settimeout(idle_timeout)
                self.handle_one_request()
                if self.close_connection:
                    break

        def parse_request(self) -> bool:
            if getattr(self, "_waiting_for_request_line", False):
                # handle_one_request has received the full request line. Restore
                # the accepted socket's original timeout before parsing headers,
                # reading a body, or invoking potentially long-running work.
                self.connection.settimeout(self._normal_request_timeout)
                self._waiting_for_request_line = False
            return super().parse_request()

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
            if self.close_connection:
                self.send_header("Connection", "close")
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
            # A malformed request may leave unread bytes in rfile.  Closing an
            # exceptional connection prevents those bytes from being parsed as
            # the next HTTP/1.1 request while normal responses stay persistent.
            self.close_connection = True
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
