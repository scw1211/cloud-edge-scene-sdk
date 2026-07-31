"""用途：在真实 HTTP 链路中注入延迟、抖动、丢包和断网故障。"""

import argparse
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


PROFILES = {
    "mild": {"delay_ms": 40.0, "jitter_ms": 10.0, "loss_rate": 0.01},
    "severe": {"delay_ms": 80.0, "jitter_ms": 30.0, "loss_rate": 0.10},
    "outage": {"delay_ms": 0.0, "jitter_ms": 0.0, "loss_rate": 1.0},
}


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
    return parser.parse_args()


def append_jsonl(path: Optional[Path], row: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


class FaultState:
    def __init__(self, profile: str, seed: int, log_path: Optional[Path]) -> None:
        self.profile_name = profile
        self.profile = PROFILES[profile]
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.request_count = 0
        self.log_path = log_path

    def sample(self) -> Dict[str, Any]:
        with self.lock:
            self.request_count += 1
            request_id = self.request_count
            dropped = self.rng.random() < float(self.profile["loss_rate"])
            delay_ms = max(
                0.0,
                self.rng.gauss(
                    float(self.profile["delay_ms"]),
                    float(self.profile["jitter_ms"]),
                ),
            )
        return {"proxy_request_id": request_id, "dropped": dropped, "delay_ms": delay_ms}


def build_handler(
    state: FaultState,
    backend: str,
    drop_hold_seconds: float,
    backend_timeout: float,
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

        def proxy(self) -> None:
            sampled = state.sample()
            started = time.perf_counter()
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            request_body = self.rfile.read(content_length) if content_length > 0 else None
            if sampled["dropped"]:
                time.sleep(drop_hold_seconds)
                self.close_connection = True
                append_jsonl(
                    state.log_path,
                    {
                        **sampled,
                        "profile": state.profile_name,
                        "method": self.command,
                        "path": self.path,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4),
                    },
                )
                return

            time.sleep(sampled["delay_ms"] / 1000.0)
            headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
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
            except urllib.error.HTTPError as exc:
                response_body = exc.read()
                status = exc.code
                content_type = exc.headers.get("Content-Type", "application/json")
            self.send_body(status, response_body, content_type)
            append_jsonl(
                state.log_path,
                {
                    **sampled,
                    "profile": state.profile_name,
                    "method": self.command,
                    "path": self.path,
                    "status": status,
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
