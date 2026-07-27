"""用途：从真实边缘进程通过 HTTP 调用云端决策与多事件协调接口。"""

import hashlib
import json
import queue
import threading
import time
from dataclasses import replace
from typing import Any, Dict, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cloud_edge_framework.artifacts import optional_artifact_path
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
        self._artifact_lock = threading.RLock()
        self._uploaded_artifacts = set()

    def _put_bytes(
        self,
        path: str,
        data: bytes,
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Accept": "application/json", **headers},
            method="PUT",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise CloudTransportError(
                "artifact upload returned HTTP {}: {}".format(exc.code, detail)
            ) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise CloudTransportError("artifact upload failed: {}".format(exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        try:
            value = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CloudTransportError("artifact upload returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise CloudTransportError("artifact upload response must be an object")
        return {
            "response": value,
            "request_bytes": len(data),
            "response_bytes": len(response_body),
            "http_round_trip_ms": round(elapsed_ms, 6),
        }

    def _materialize_artifacts(
        self, event: SemanticEvent
    ) -> Tuple[SemanticEvent, Dict[str, Any]]:
        evidence_items = []
        uploaded_bytes = 0
        response_bytes = 0
        upload_ms = 0.0
        artifact_count = 0
        uploaded_count = 0
        for evidence in event.evidence:
            path = optional_artifact_path(evidence.uri)
            if path is None:
                evidence_items.append(evidence)
                continue
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if evidence.sha256 is not None and evidence.sha256 != digest:
                raise CloudTransportError(
                    "local evidence sha256 mismatch for {}".format(evidence.evidence_id)
                )
            if evidence.size_bytes not in {0, len(data)}:
                raise CloudTransportError(
                    "local evidence size mismatch for {}".format(evidence.evidence_id)
                )
            artifact_count += 1
            with self._artifact_lock:
                already_uploaded = digest in self._uploaded_artifacts
            if not already_uploaded:
                uploaded = self._put_bytes(
                    "/api/v1/evidence/" + digest,
                    data,
                    {
                        "Content-Type": evidence.content_type,
                        "X-Evidence-ID": evidence.evidence_id,
                        "Idempotency-Key": "evidence_" + digest,
                    },
                )
                uploaded_bytes += int(uploaded["request_bytes"])
                response_bytes += int(uploaded["response_bytes"])
                upload_ms += float(uploaded["http_round_trip_ms"])
                uploaded_count += 1
                with self._artifact_lock:
                    self._uploaded_artifacts.add(digest)
            evidence_items.append(
                replace(
                    evidence,
                    uri="evidence://" + digest,
                    size_bytes=len(data),
                    sha256=digest,
                )
            )
        return replace(event, evidence=evidence_items), {
            "artifact_count": artifact_count,
            "artifact_uploaded_count": uploaded_count,
            "artifact_request_bytes": uploaded_bytes,
            "artifact_response_bytes": response_bytes,
            "artifact_upload_ms": round(upload_ms, 6),
        }

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
        cloud_event, artifact_metrics = self._materialize_artifacts(event)
        include_scene_payload = bool(
            cloud_event.metadata.get("transport_include_scene_payload", False)
        )
        response = self._post(
            "/api/v1/collaboration/cloud-decision",
            {
                "event": cloud_event.to_dict(
                    include_scene_payload=include_scene_payload
                )
            },
        )
        raw_decision = response.get("decision")
        if not isinstance(raw_decision, dict):
            raise CloudTransportError("cloud response is missing decision")
        decision = DecisionEnvelope.from_dict(raw_decision)
        transport = dict(response["_transport_metrics"])
        transport["json_request_bytes"] = int(transport["request_bytes"])
        transport.update(artifact_metrics)
        transport["request_bytes"] = (
            int(transport["json_request_bytes"])
            + int(artifact_metrics["artifact_request_bytes"])
        )
        transport["response_bytes"] = (
            int(transport["response_bytes"])
            + int(artifact_metrics["artifact_response_bytes"])
        )
        transport["http_round_trip_ms"] = round(
            float(transport["http_round_trip_ms"])
            + float(artifact_metrics["artifact_upload_ms"]),
            6,
        )
        metadata = dict(decision.metadata)
        metadata["transport"] = transport
        metadata["cloud_runtime_ms"] = response.get("cloud_runtime_ms")
        return replace(decision, metadata=metadata)

    def aggregate(self, event: SemanticEvent) -> Dict[str, Any]:
        cloud_event, artifact_metrics = self._materialize_artifacts(event)
        response = self._post(
            "/api/v1/collaboration/aggregate",
            {
                "event": cloud_event.to_dict(
                    include_scene_payload=bool(
                        cloud_event.metadata.get(
                            "transport_include_scene_payload", False
                        )
                    )
                )
            },
        )
        transport = dict(response.pop("_transport_metrics"))
        transport["json_request_bytes"] = int(transport["request_bytes"])
        transport.update(artifact_metrics)
        transport["request_bytes"] = (
            int(transport["json_request_bytes"])
            + int(artifact_metrics["artifact_request_bytes"])
        )
        transport["response_bytes"] = (
            int(transport["response_bytes"])
            + int(artifact_metrics["artifact_response_bytes"])
        )
        transport["http_round_trip_ms"] = round(
            float(transport["http_round_trip_ms"])
            + float(artifact_metrics["artifact_upload_ms"]),
            6,
        )
        response["transport"] = transport
        return response

    def coordinate(self, events: Sequence[SemanticEvent]) -> Dict[str, Any]:
        cloud_events = []
        artifact_metrics = {
            "artifact_count": 0,
            "artifact_uploaded_count": 0,
            "artifact_request_bytes": 0,
            "artifact_response_bytes": 0,
            "artifact_upload_ms": 0.0,
        }
        for event in events:
            cloud_event, item_metrics = self._materialize_artifacts(event)
            cloud_events.append(cloud_event)
            for name in artifact_metrics:
                artifact_metrics[name] += item_metrics[name]
        response = self._post(
            "/api/v1/collaboration/coordinate",
            {
                "events": [
                    event.to_dict(
                        include_scene_payload=bool(
                            event.metadata.get("transport_include_scene_payload", False)
                        )
                    )
                    for event in cloud_events
                ]
            },
        )
        transport = dict(response.pop("_transport_metrics"))
        transport["json_request_bytes"] = int(transport["request_bytes"])
        transport.update(artifact_metrics)
        transport["request_bytes"] = (
            int(transport["json_request_bytes"])
            + int(artifact_metrics["artifact_request_bytes"])
        )
        transport["response_bytes"] = (
            int(transport["response_bytes"])
            + int(artifact_metrics["artifact_response_bytes"])
        )
        transport["http_round_trip_ms"] = round(
            float(transport["http_round_trip_ms"])
            + float(artifact_metrics["artifact_upload_ms"]),
            6,
        )
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
