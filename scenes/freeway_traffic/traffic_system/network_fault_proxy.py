"""用途：在真实 HTTP 链路中注入延迟、抖动、丢包和断网故障。"""

import argparse
import ipaddress
import json
import random
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit


PROFILES = {
    "normal": {"delay_ms": 0.0, "jitter_ms": 0.0, "loss_rate": 0.0},
    "mild": {"delay_ms": 40.0, "jitter_ms": 10.0, "loss_rate": 0.01},
    "severe": {"delay_ms": 80.0, "jitter_ms": 30.0, "loss_rate": 0.10},
    "outage": {"delay_ms": 0.0, "jitter_ms": 0.0, "loss_rate": 1.0},
}

CONTROL_STATUS_PATH = "/__fault__/status"
CONTROL_PROFILE_PATH = "/__fault__/profile"
FORWARDED_REQUEST_HEADERS = (
    "Accept",
    "Content-Type",
    "Idempotency-Key",
    "X-Trace-ID",
    "X-Request-ID",
    "Prefer",
    "X-Response-Detail",
)


def _request_class(path: str) -> str:
    value = urlsplit(path).path
    if value == "/health":
        return "health_probe"
    if value.endswith("/aggregate/batch"):
        return "summary_data_plane"
    if value.endswith("/aggregate/results/batch"):
        return "result_data_plane"
    if value.endswith("/cloud-decision"):
        return "legacy_cloud_decision"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject real HTTP delay, jitter, and loss.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", default="http://127.0.0.1:18080")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--drop_hold_seconds", type=float, default=1.0)
    parser.add_argument("--backend_timeout", type=float, default=5.0)
    parser.add_argument("--log_jsonl", default="")
    parser.add_argument(
        "--control-token",
        default="",
        help="optional token for the loopback-only runtime control endpoint",
    )
    return parser.parse_args()


def append_jsonl(path: Optional[Path], row: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


class FaultState:
    def __init__(self, profile: str, seed: int, log_path: Optional[Path]) -> None:
        self.seed = int(seed)
        self.profile_name = ""
        self.profile: Dict[str, float] = {}
        self.rng = random.Random(self.seed)
        self.lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.request_count = 0
        self.profile_request_count = 0
        self.dropped_count = 0
        self.profile_dropped_count = 0
        self.profile_delayed_count = 0
        self.profile_delay_ms_total = 0.0
        self.request_class_counts: Dict[str, int] = {}
        self.profile_request_class_counts: Dict[str, int] = {}
        self.switch_count = 0
        self.log_path = log_path
        self.set_profile(profile, initial=True)

    def _snapshot_unlocked(self) -> Dict[str, Any]:
        return {
            "profile": self.profile_name,
            "parameters": dict(self.profile),
            "seed": self.seed,
            "request_count": self.request_count,
            "profile_request_count": self.profile_request_count,
            "dropped_count": self.dropped_count,
            "profile_dropped_count": self.profile_dropped_count,
            "profile_delayed_count": self.profile_delayed_count,
            "profile_delay_ms_total": round(self.profile_delay_ms_total, 6),
            "request_class_counts": dict(self.request_class_counts),
            "profile_request_class_counts": dict(
                self.profile_request_class_counts
            ),
            "switch_count": self.switch_count,
        }

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return self._snapshot_unlocked()

    def append_log(self, row: Dict[str, Any]) -> None:
        with self.log_lock:
            append_jsonl(self.log_path, row)

    def set_profile(self, profile: str, initial: bool = False) -> Dict[str, Any]:
        name = str(profile).strip().lower()
        if name not in PROFILES:
            raise ValueError("unknown fault profile: {}".format(name))
        with self.lock:
            self.profile_name = name
            self.profile = dict(PROFILES[name])
            # Resetting the generator makes every preregistered profile repeat
            # the same seeded sequence when the run is repeated.
            self.rng = random.Random(self.seed)
            self.profile_request_count = 0
            self.profile_dropped_count = 0
            self.profile_delayed_count = 0
            self.profile_delay_ms_total = 0.0
            self.profile_request_class_counts = {}
            if not initial:
                self.switch_count += 1
            return self._snapshot_unlocked()

    def sample(self, path: str = "") -> Dict[str, Any]:
        with self.lock:
            self.request_count += 1
            self.profile_request_count += 1
            request_id = self.request_count
            profile_request_id = self.profile_request_count
            profile_name = self.profile_name
            request_class = _request_class(path)
            self.request_class_counts[request_class] = (
                self.request_class_counts.get(request_class, 0) + 1
            )
            self.profile_request_class_counts[request_class] = (
                self.profile_request_class_counts.get(request_class, 0) + 1
            )
            dropped = self.rng.random() < float(self.profile["loss_rate"])
            delay_ms = max(
                0.0,
                self.rng.gauss(
                    float(self.profile["delay_ms"]),
                    float(self.profile["jitter_ms"]),
                ),
            )
            if dropped:
                self.dropped_count += 1
                self.profile_dropped_count += 1
            if delay_ms > 0.0:
                self.profile_delayed_count += 1
                self.profile_delay_ms_total += delay_ms
        return {
            "proxy_request_id": request_id,
            "profile_request_id": profile_request_id,
            "profile": profile_name,
            "request_class": request_class,
            "dropped": dropped,
            "delay_ms": delay_ms,
        }


def _loopback_client(address: str) -> bool:
    try:
        value = ipaddress.ip_address(str(address).split("%", 1)[0])
    except ValueError:
        return False
    if value.is_loopback:
        return True
    return bool(
        value.version == 6
        and value.ipv4_mapped is not None
        and value.ipv4_mapped.is_loopback
    )


def _forward_headers(headers: Any) -> Dict[str, str]:
    return {
        name: str(headers.get(name))
        for name in FORWARDED_REQUEST_HEADERS
        if headers.get(name) is not None
    }


def build_handler(
    state: FaultState,
    backend: str,
    drop_hold_seconds: float,
    backend_timeout: float,
    control_token: str = "",
):
    class FaultProxyHandler(BaseHTTPRequestHandler):
        server_version = "TrafficFaultProxy/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def send_body(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, value: Dict[str, Any]) -> None:
            self.send_body(
                status,
                json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                ),
                "application/json",
            )

        def control_allowed(self) -> bool:
            if not _loopback_client(str(self.client_address[0])):
                return False
            if not control_token:
                return True
            supplied = str(self.headers.get("X-Fault-Control-Token", ""))
            authorization = str(self.headers.get("Authorization", ""))
            return supplied == control_token or authorization == "Bearer " + control_token

        def handle_control(self) -> bool:
            path = urlsplit(self.path).path
            if path not in {CONTROL_STATUS_PATH, CONTROL_PROFILE_PATH}:
                return False
            if not self.control_allowed():
                self.send_json(
                    HTTPStatus.FORBIDDEN,
                    {"error": "fault control is restricted to loopback clients"},
                )
                return True
            if path == CONTROL_STATUS_PATH:
                if self.command != "GET":
                    self.send_json(
                        HTTPStatus.METHOD_NOT_ALLOWED, {"error": "GET required"}
                    )
                else:
                    self.send_json(HTTPStatus.OK, state.snapshot())
                return True
            if self.command != "POST":
                self.send_json(
                    HTTPStatus.METHOD_NOT_ALLOWED, {"error": "POST required"}
                )
                return True
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length <= 0 or length > 4096:
                    raise ValueError("control body must contain 1..4096 bytes")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("control body must be an object")
                result = state.set_profile(str(payload.get("profile", "")))
            except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            else:
                state.append_log({"control": "set_profile", **result})
                self.send_json(HTTPStatus.OK, result)
            return True

        def proxy(self) -> None:
            if self.handle_control():
                return
            sampled = state.sample(self.path)
            started = time.perf_counter()
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            request_body = self.rfile.read(content_length) if content_length > 0 else None
            if sampled["dropped"]:
                time.sleep(drop_hold_seconds)
                self.close_connection = True
                state.append_log(
                    {
                        **sampled,
                        "method": self.command,
                        "path": self.path,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4),
                    },
                )
                return

            time.sleep(sampled["delay_ms"] / 1000.0)
            headers = _forward_headers(self.headers)
            request = urllib.request.Request(
                backend.rstrip("/") + self.path,
                data=request_body,
                headers=headers,
                method=self.command,
            )
            try:
                with urllib.request.urlopen(request, timeout=backend_timeout) as response:
                    response_body = response.read()
                    status = response.status
                    content_type = response.headers.get("Content-Type", "application/json")
                    backend_error = ""
            except urllib.error.HTTPError as exc:
                response_body = exc.read()
                status = exc.code
                content_type = exc.headers.get("Content-Type", "application/json")
                backend_error = ""
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                status = HTTPStatus.BAD_GATEWAY
                content_type = "application/json"
                backend_error = "{}: {}".format(type(exc).__name__, exc)
                response_body = json.dumps(
                    {"error": "fault proxy backend unavailable"},
                    separators=(",", ":"),
                ).encode("utf-8")
            client_write_error = ""
            try:
                self.send_body(status, response_body, content_type)
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                # A deliberately dropped or timed-out client may close before
                # the proxy can relay the backend response.  Keep that fact in
                # the evidence log instead of emitting an unstructured server
                # traceback from the handler thread.
                client_write_error = "{}: {}".format(type(exc).__name__, exc)
            state.append_log(
                {
                    **sampled,
                    "method": self.command,
                    "path": self.path,
                    "status": status,
                    "backend_error": backend_error or None,
                    "client_write_error": client_write_error or None,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4),
                },
            )

        def do_GET(self) -> None:
            self.proxy()

        def do_POST(self) -> None:
            self.proxy()

    return FaultProxyHandler


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_jsonl) if args.log_jsonl else None
    state = FaultState(args.profile, args.seed, log_path)
    handler = build_handler(
        state,
        args.backend,
        args.drop_hold_seconds,
        args.backend_timeout,
        args.control_token,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        "Fault proxy {} listening on {}:{} -> {}".format(
            args.profile, args.host, args.port, args.backend
        ),
        flush=True,
    )
    print(json.dumps(PROFILES[args.profile], ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
