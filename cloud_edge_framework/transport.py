"""用途：从真实边缘进程通过 HTTP 调用云端决策与多事件协调接口。"""

import json
import queue
import threading
import time
from dataclasses import replace
from typing import Any, Dict, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cloud_edge_framework.contracts import DecisionEnvelope, SemanticEvent
from cloud_edge_framework.feedback import DecisionFeedbackStore


class CloudTransportError(RuntimeError):
    """Raised when the remote cloud service cannot return a valid decision."""


class HttpCloudClient:
    def __init__(self, base_url: str, timeout_seconds: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        if not self.base_url:
            raise ValueError("base_url must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._feedback_queue: queue.Queue = queue.Queue()
        self._feedback_lock = threading.Lock()
        self._feedback_ids = set()
        self._feedback_errors = []
        self._feedback_worker_thread = None

    def _ensure_feedback_worker(self) -> None:
        with self._feedback_lock:
            if (
                self._feedback_worker_thread is not None
                and self._feedback_worker_thread.is_alive()
            ):
                return
            self._feedback_worker_thread = threading.Thread(
                target=self._feedback_worker,
                name="cloud-feedback-sync",
                daemon=True,
            )
            self._feedback_worker_thread.start()

    def _feedback_worker(self) -> None:
        while True:
            record = self._feedback_queue.get()
            feedback_id = str(record["feedback_id"])
            try:
                self._post("/api/v1/collaboration/feedback", {"record": record})
            except Exception as exc:  # noqa: BLE001
                with self._feedback_lock:
                    self._feedback_errors.append(
                        "{}: {}".format(type(exc).__name__, exc)
                    )
                    self._feedback_ids.discard(feedback_id)
            finally:
                self._feedback_queue.task_done()

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise CloudTransportError("cloud returned HTTP {}: {}".format(exc.code, detail)) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise CloudTransportError("cloud request failed: {}".format(exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        try:
            value = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CloudTransportError("cloud returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise CloudTransportError("cloud response must be an object")
        value["_transport_metrics"] = {
            "request_bytes": len(body),
            "response_bytes": len(response_body),
            "http_round_trip_ms": round(elapsed_ms, 6),
        }
        return value

    def decide(self, event: SemanticEvent) -> DecisionEnvelope:
        include_scene_payload = bool(
            event.metadata.get("transport_include_scene_payload", False)
        )
        response = self._post(
            "/api/v1/collaboration/cloud-decision",
            {"event": event.to_dict(include_scene_payload=include_scene_payload)},
        )
        raw_decision = response.get("decision")
        if not isinstance(raw_decision, dict):
            raise CloudTransportError("cloud response is missing decision")
        decision = DecisionEnvelope.from_dict(raw_decision)
        metadata = dict(decision.metadata)
        metadata["transport"] = response["_transport_metrics"]
        metadata["cloud_runtime_ms"] = response.get("cloud_runtime_ms")
        return replace(decision, metadata=metadata)

    def coordinate(self, events: Sequence[SemanticEvent]) -> Dict[str, Any]:
        response = self._post(
            "/api/v1/collaboration/coordinate",
            {
                "events": [
                    event.to_dict(
                        include_scene_payload=bool(
                            event.metadata.get("transport_include_scene_payload", False)
                        )
                    )
                    for event in events
                ]
            },
        )
        transport = response.pop("_transport_metrics")
        response["transport"] = transport
        return response

    def submit_feedback(
        self,
        event: SemanticEvent,
        local: DecisionEnvelope,
        cloud: DecisionEnvelope,
        evidence_level: str,
        network_class: str,
        request_bytes: int,
    ) -> bool:
        record = DecisionFeedbackStore.build_record(
            event,
            local,
            cloud,
            evidence_level,
            network_class,
            request_bytes,
        )
        feedback_id = str(record["feedback_id"])
        with self._feedback_lock:
            if feedback_id in self._feedback_ids:
                return False
            self._feedback_ids.add(feedback_id)
        self._ensure_feedback_worker()
        self._feedback_queue.put(record)
        return True

    def flush_feedback(self, timeout_seconds: float = 2.0) -> Dict[str, Any]:
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while self._feedback_queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)
        with self._feedback_lock:
            return {
                "complete": self._feedback_queue.unfinished_tasks == 0,
                "queued_ids": len(self._feedback_ids),
                "errors": list(self._feedback_errors),
            }
