"""用途：持久记录边缘初判与云端复核差异，形成可追溯的纠错训练数据。"""

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cloud_edge_framework.contracts import DecisionEnvelope, SemanticEvent, stable_id


class DecisionFeedbackStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._memory: List[Dict[str, Any]] = []
        self._known_ids = set()
        self._pending_ids = set()
        self._errors: List[str] = []
        self._queue: queue.Queue = queue.Queue()
        self._worker_thread = None
        if self.path is not None and self.path.is_file():
            for record in self._read_file():
                self._known_ids.add(str(record["feedback_id"]))

    def _read_file(self) -> List[Dict[str, Any]]:
        if self.path is None or not self.path.is_file():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "invalid decision feedback at line {}".format(line_number)
                    ) from exc
                if not isinstance(record, dict) or not record.get("feedback_id"):
                    raise ValueError(
                        "decision feedback line {} is invalid".format(line_number)
                    )
                records.append(record)
        return records

    @staticmethod
    def build_record(
        event: SemanticEvent,
        local: DecisionEnvelope,
        cloud: DecisionEnvelope,
        evidence_level: str,
        network_class: str,
        request_bytes: int,
    ) -> Dict[str, Any]:
        feedback_id = stable_id(
            "feedback",
            event.event_id,
            local.decision_id,
            cloud.decision_id,
            cloud.policy_version,
        )
        return {
            "schema_version": 1,
            "feedback_id": feedback_id,
            "created_at_ms": int(time.time() * 1000),
            "event_id": event.event_id,
            "scene": event.scene,
            "edge_id": event.edge_id,
            "evidence_level": evidence_level,
            "network_class": network_class,
            "request_bytes": max(0, int(request_bytes)),
            "decision_changed": local.decision != cloud.decision,
            "policy_version": cloud.policy_version,
            "event": event.to_dict(
                include_scene_payload=bool(
                    event.metadata.get("transport_include_scene_payload", False)
                )
            ),
            "edge_decision": local.to_dict(),
            "cloud_target": cloud.to_dict(),
        }

    def append_record(self, record: Dict[str, Any]) -> bool:
        if not isinstance(record, dict) or int(record.get("schema_version", 0)) != 1:
            raise ValueError("invalid decision feedback record")
        feedback_id = str(record.get("feedback_id", "")).strip()
        if not feedback_id:
            raise ValueError("decision feedback record requires feedback_id")
        with self._lock:
            if feedback_id in self._known_ids:
                return False
            if self.path is None:
                self._memory.append(record)
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as file_obj:
                    file_obj.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    file_obj.flush()
                    os.fsync(file_obj.fileno())
            self._known_ids.add(feedback_id)
            return True

    def append(
        self,
        event: SemanticEvent,
        local: DecisionEnvelope,
        cloud: DecisionEnvelope,
        evidence_level: str,
        network_class: str,
        request_bytes: int,
    ) -> bool:
        return self.append_record(
            self.build_record(
                event,
                local,
                cloud,
                evidence_level,
                network_class,
                request_bytes,
            )
        )

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._worker_thread = threading.Thread(
                target=self._worker,
                name="local-feedback-writer",
                daemon=True,
            )
            self._worker_thread.start()

    def _worker(self) -> None:
        while True:
            record = self._queue.get()
            feedback_id = str(record["feedback_id"])
            try:
                self.append_record(record)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._errors.append(
                        "{}: {}".format(type(exc).__name__, exc)
                    )
            finally:
                with self._lock:
                    self._pending_ids.discard(feedback_id)
                self._queue.task_done()

    def enqueue(
        self,
        event: SemanticEvent,
        local: DecisionEnvelope,
        cloud: DecisionEnvelope,
        evidence_level: str,
        network_class: str,
        request_bytes: int,
    ) -> bool:
        record = self.build_record(
            event,
            local,
            cloud,
            evidence_level,
            network_class,
            request_bytes,
        )
        feedback_id = str(record["feedback_id"])
        with self._lock:
            if feedback_id in self._known_ids or feedback_id in self._pending_ids:
                return False
            self._pending_ids.add(feedback_id)
        self._ensure_worker()
        self._queue.put(record)
        return True

    def flush(self, timeout_seconds: float = 2.0) -> Dict[str, Any]:
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)
        with self._lock:
            return {
                "complete": self._queue.unfinished_tasks == 0 and not self._errors,
                "pending": self._queue.unfinished_tasks,
                "count": len(self._known_ids),
                "errors": list(self._errors),
            }

    def records(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._read_file() if self.path is not None else list(self._memory)

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        if limit <= 0:
            raise ValueError("feedback limit must be positive")
        return self.records()[-limit:]

    def count(self) -> int:
        with self._lock:
            return len(self._known_ids)
