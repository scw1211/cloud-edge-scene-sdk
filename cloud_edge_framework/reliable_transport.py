"""用途：为边缘到云端 HTTP 调用增加有限重试、追踪头和幂等请求键。"""

import json
import time
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cloud_edge_framework.contracts import stable_id
from cloud_edge_framework.transport import CloudTransportError, HttpCloudClient


class ReliableHttpCloudClient(HttpCloudClient):
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 0.5,
        max_attempts: int = 2,
        retry_backoff_seconds: float = 0.025,
    ) -> None:
        super().__init__(base_url, timeout_seconds)
        self.max_attempts = int(max_attempts)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")

    @staticmethod
    def _request_identity(path: str, payload: Dict[str, Any]) -> Tuple[str, str]:
        trace_id = ""
        if path.endswith("/cloud-decision"):
            event = payload.get("event", {})
            if isinstance(event, dict):
                event_id = str(event.get("event_id", ""))
                metadata = event.get("metadata", {})
                if isinstance(metadata, dict):
                    trace_id = str(metadata.get("trace_id", ""))
                if event_id:
                    return stable_id("cloud_request", event_id), trace_id
        if path.endswith("/coordinate"):
            events = payload.get("events", [])
            event_ids = sorted(
                str(event.get("event_id", ""))
                for event in events
                if isinstance(event, dict) and event.get("event_id")
            )
            for event in events:
                if not isinstance(event, dict):
                    continue
                metadata = event.get("metadata", {})
                if isinstance(metadata, dict) and metadata.get("trace_id"):
                    trace_id = str(metadata["trace_id"])
                    break
            if event_ids:
                return stable_id("coordinate_request", *event_ids), trace_id
        if path.endswith("/aggregate/batch"):
            events = payload.get("events", [])
            event_ids = sorted(
                str(event.get("event_id", ""))
                for event in events
                if isinstance(event, dict) and event.get("event_id")
            )
            for event in events:
                if not isinstance(event, dict):
                    continue
                metadata = event.get("metadata", {})
                if isinstance(metadata, dict) and metadata.get("trace_id"):
                    trace_id = str(metadata["trace_id"])
                    break
            if event_ids:
                return stable_id("aggregate_batch_request", *event_ids), trace_id
        if path.endswith("/aggregate/results/batch"):
            items = payload.get("items", [])
            identities = sorted(
                "{}:{}".format(
                    str(item.get("event_id", "")),
                    str(item.get("group_id", "")),
                )
                for item in items
                if isinstance(item, dict)
                and item.get("event_id")
                and item.get("group_id")
            )
            if identities:
                return stable_id(
                    "aggregate_results_batch_request", *identities
                ), trace_id
        if path.endswith("/feedback"):
            record = payload.get("record", {})
            if isinstance(record, dict) and record.get("feedback_id"):
                return stable_id("feedback_request", record["feedback_id"]), trace_id
        material = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return stable_id("http_request", path, material), trace_id

    def _put_bytes(
        self,
        path: str,
        data: bytes,
        headers: Dict[str, str],
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        deadline = self._timeout_deadline(timeout_seconds)
        last_error: Exception = CloudTransportError("artifact upload did not start")
        for attempt in range(1, self.max_attempts + 1):
            try:
                remaining = self._remaining_timeout(deadline)
                result = (
                    super()._put_bytes(path, data, headers)
                    if remaining is None
                    else super()._put_bytes(
                        path,
                        data,
                        headers,
                        timeout_seconds=min(self.timeout_seconds, remaining),
                    )
                )
                if deadline is not None and time.monotonic() > deadline:
                    raise CloudTransportError(
                        "cloud request exceeded its timeout budget"
                    )
                result["attempts"] = attempt
                return result
            except CloudTransportError as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= delay:
                            raise CloudTransportError(
                                "cloud request exceeded its timeout budget"
                            ) from last_error
                    time.sleep(delay)
        raise last_error

    def _post(
        self,
        path: str,
        payload: Dict[str, Any],
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request_id, trace_id = self._request_identity(path, payload)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Request-ID": request_id,
            "Idempotency-Key": request_id,
        }
        if trace_id:
            headers["X-Trace-ID"] = trace_id
        started = time.perf_counter()
        deadline = self._timeout_deadline(timeout_seconds)
        response_body = b""
        last_error: Exception = CloudTransportError("cloud request did not start")
        attempts = 0
        for attempt in range(1, self.max_attempts + 1):
            attempts = attempt
            remaining = self._remaining_timeout(deadline)
            attempt_timeout = (
                self.timeout_seconds
                if remaining is None
                else min(self.timeout_seconds, remaining)
            )
            request = Request(
                self.base_url + path,
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urlopen(request, timeout=attempt_timeout) as response:
                    response_body = response.read()
                if deadline is not None and time.monotonic() > deadline:
                    raise CloudTransportError(
                        "cloud request exceeded its timeout budget"
                    )
                last_error = CloudTransportError("")
                break
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = CloudTransportError(
                    "cloud returned HTTP {}: {}".format(exc.code, detail)
                )
                if exc.code < 500:
                    raise last_error from exc
            except (TimeoutError, URLError, OSError) as exc:
                last_error = CloudTransportError("cloud request failed: {}".format(exc))
            if attempt < self.max_attempts:
                delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= delay:
                        raise CloudTransportError(
                            "cloud request exceeded its timeout budget"
                        ) from last_error
                time.sleep(delay)
        else:
            raise last_error
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        try:
            value = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CloudTransportError("cloud returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise CloudTransportError("cloud response must be an object")
        value["_transport_metrics"] = {
            "request_id": request_id,
            "trace_id": trace_id,
            "attempts": attempts,
            "request_bytes": len(body),
            "response_bytes": len(response_body),
            "http_round_trip_ms": round(elapsed_ms, 6),
        }
        self._remaining_timeout(deadline)
        return value
