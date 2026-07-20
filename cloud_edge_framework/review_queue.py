"""用途：持久保存弱网和云端失败期间尚未完成的异步复核事件。"""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

from cloud_edge_framework.contracts import SemanticEvent


class PendingReviewStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path
        self._memory: List[SemanticEvent] = []
        self._lock = threading.RLock()

    @staticmethod
    def _serialized(event: SemanticEvent) -> dict:
        include_scene_payload = bool(
            event.metadata.get("transport_include_scene_payload", False)
        )
        return event.to_dict(include_scene_payload=include_scene_payload)

    def append(self, event: SemanticEvent) -> bool:
        with self._lock:
            if any(item.event_id == event.event_id for item in self.events()):
                return False
            if self.path is None:
                self._memory.append(event)
                return True
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file_obj:
                file_obj.write(
                    json.dumps(self._serialized(event), ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                file_obj.flush()
                os.fsync(file_obj.fileno())
            return True

    def events(self) -> List[SemanticEvent]:
        with self._lock:
            if self.path is None:
                return list(self._memory)
            if not self.path.exists():
                return []
            events: List[SemanticEvent] = []
            with self.path.open("r", encoding="utf-8") as file_obj:
                for line_number, line in enumerate(file_obj, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                        events.append(SemanticEvent.from_dict(value))
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise ValueError(
                            "invalid pending review at line {}: {}".format(line_number, exc)
                        ) from exc
            return events

    def clear(self) -> None:
        with self._lock:
            if self.path is None:
                self._memory = []
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=self.path.name + ".", dir=str(self.path.parent)
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as file_obj:
                    file_obj.flush()
                    os.fsync(file_obj.fileno())
                os.replace(temporary_name, self.path)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)

    def count(self) -> int:
        return len(self.events())
