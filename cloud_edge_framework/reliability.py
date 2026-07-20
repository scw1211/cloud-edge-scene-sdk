"""用途：提供可确认的 SQLite 弱网 Outbox 和云端幂等响应缓存。"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

from cloud_edge_framework.contracts import SemanticEvent


class IdempotencyConflictError(ValueError):
    """Raised when one idempotency key is reused for a different request."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _request_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OutboxLease:
    event: SemanticEvent
    attempts: int


class SQLiteOutbox:
    """Durable at-least-once queue with leases and stable event-id deduplication."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS outbox_events (
                    event_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('pending','inflight','completed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    available_at_ms INTEGER NOT NULL,
                    lease_until_ms INTEGER,
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    completed_at_ms INTEGER,
                    last_error TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_outbox_ready "
                "ON outbox_events(state, available_at_ms, created_at_ms)"
            )

    @staticmethod
    def _serialized(event: SemanticEvent) -> str:
        include_scene_payload = bool(
            event.metadata.get("transport_include_scene_payload", False)
        )
        return _canonical_json(
            event.to_dict(include_scene_payload=include_scene_payload)
        )

    def append(self, event: SemanticEvent) -> bool:
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO outbox_events(
                    event_id, payload_json, state, attempts, available_at_ms,
                    created_at_ms, updated_at_ms
                ) VALUES (?, ?, 'pending', 0, ?, ?, ?)
                """,
                (event.event_id, self._serialized(event), now_ms, now_ms, now_ms),
            )
            return cursor.rowcount == 1

    def claim(self, limit: int, lease_seconds: float) -> List[OutboxLease]:
        if limit <= 0 or lease_seconds <= 0:
            raise ValueError("outbox claim limit and lease_seconds must be positive")
        now_ms = int(time.time() * 1000)
        lease_until_ms = now_ms + int(lease_seconds * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE outbox_events
                SET state='pending', lease_until_ms=NULL, updated_at_ms=?
                WHERE state='inflight' AND lease_until_ms IS NOT NULL AND lease_until_ms <= ?
                """,
                (now_ms, now_ms),
            )
            rows = connection.execute(
                """
                SELECT event_id, payload_json, attempts
                FROM outbox_events
                WHERE state='pending' AND available_at_ms <= ?
                ORDER BY created_at_ms, event_id
                LIMIT ?
                """,
                (now_ms, int(limit)),
            ).fetchall()
            if rows:
                connection.executemany(
                    """
                    UPDATE outbox_events
                    SET state='inflight', attempts=attempts+1,
                        lease_until_ms=?, updated_at_ms=?
                    WHERE event_id=? AND state='pending'
                    """,
                    [(lease_until_ms, now_ms, str(row["event_id"])) for row in rows],
                )
            connection.commit()
        return [
            OutboxLease(
                event=SemanticEvent.from_dict(json.loads(str(row["payload_json"]))),
                attempts=int(row["attempts"]) + 1,
            )
            for row in rows
        ]

    def acknowledge(self, event_ids: Sequence[str]) -> None:
        ids = [str(value) for value in event_ids]
        if not ids:
            return
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.executemany(
                """
                UPDATE outbox_events
                SET state='completed', lease_until_ms=NULL, completed_at_ms=?,
                    updated_at_ms=?, last_error=''
                WHERE event_id=?
                """,
                [(now_ms, now_ms, event_id) for event_id in ids],
            )

    def release(
        self,
        event_ids: Sequence[str],
        error: str,
        max_backoff_seconds: float,
    ) -> None:
        ids = [str(value) for value in event_ids]
        if not ids:
            return
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT event_id, attempts FROM outbox_events WHERE event_id IN ({})".format(
                    ",".join("?" for _ in ids)
                ),
                ids,
            ).fetchall()
            updates = []
            for row in rows:
                attempts = max(1, int(row["attempts"]))
                backoff_seconds = min(float(max_backoff_seconds), 2.0 ** min(attempts - 1, 16))
                updates.append(
                    (
                        now_ms + int(backoff_seconds * 1000),
                        now_ms,
                        str(error)[:2000],
                        str(row["event_id"]),
                    )
                )
            connection.executemany(
                """
                UPDATE outbox_events
                SET state='pending', available_at_ms=?, lease_until_ms=NULL,
                    updated_at_ms=?, last_error=?
                WHERE event_id=?
                """,
                updates,
            )

    def events(self) -> List[SemanticEvent]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM outbox_events
                WHERE state IN ('pending','inflight')
                ORDER BY created_at_ms, event_id
                """
            ).fetchall()
        return [SemanticEvent.from_dict(json.loads(str(row["payload_json"]))) for row in rows]

    def clear(self) -> None:
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE outbox_events
                SET state='completed', lease_until_ms=NULL, completed_at_ms=?, updated_at_ms=?
                WHERE state IN ('pending','inflight')
                """,
                (now_ms, now_ms),
            )

    def count(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM outbox_events "
                "WHERE state IN ('pending','inflight')"
            ).fetchone()
        return int(row["count"])

    def snapshot(self) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            counts = {
                str(row["state"]): int(row["count"])
                for row in connection.execute(
                    "SELECT state, COUNT(*) AS count FROM outbox_events GROUP BY state"
                ).fetchall()
            }
            oldest = connection.execute(
                "SELECT MIN(created_at_ms) AS value FROM outbox_events "
                "WHERE state IN ('pending','inflight')"
            ).fetchone()["value"]
            attempts = connection.execute(
                "SELECT COALESCE(SUM(attempts), 0) AS value FROM outbox_events"
            ).fetchone()["value"]
        return {
            "path": str(self.path),
            "states": {
                "pending": counts.get("pending", 0),
                "inflight": counts.get("inflight", 0),
                "completed": counts.get("completed", 0),
            },
            "active": counts.get("pending", 0) + counts.get("inflight", 0),
            "oldest_active_age_ms": max(0, now_ms - int(oldest)) if oldest else 0,
            "delivery_attempts": int(attempts),
        }


class SQLiteIdempotencyStore:
    """Caches completed cloud responses and rejects key reuse with a new payload."""

    def __init__(self, path: Path, ttl_seconds: float, max_entries: int) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        if self.ttl_seconds <= 0 or self.max_entries <= 0:
            raise ValueError("idempotency ttl_seconds and max_entries must be positive")
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency_responses (
                    request_key TEXT PRIMARY KEY,
                    request_sha256 TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    expires_at_ms INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_idempotency_expiry "
                "ON idempotency_responses(expires_at_ms)"
            )

    def execute(
        self,
        request_key: str,
        request_payload: Any,
        operation: Callable[[], Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], bool]:
        key = str(request_key).strip()
        if not key or len(key) > 256:
            raise ValueError("idempotency key must contain 1 to 256 characters")
        request_hash = _request_sha256(request_payload)
        now_ms = int(time.time() * 1000)
        expires_at_ms = now_ms + int(self.ttl_seconds * 1000)
        with self._lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM idempotency_responses WHERE expires_at_ms <= ?",
                (now_ms,),
            )
            row = connection.execute(
                "SELECT request_sha256, response_json FROM idempotency_responses "
                "WHERE request_key=?",
                (key,),
            ).fetchone()
            if row is not None:
                if str(row["request_sha256"]) != request_hash:
                    raise IdempotencyConflictError(
                        "idempotency key was already used for a different request"
                    )
                response = json.loads(str(row["response_json"]))
                return dict(response), True

            response = operation()
            if not isinstance(response, dict):
                raise ValueError("idempotent operation must return an object")
            connection.execute(
                """
                INSERT INTO idempotency_responses(
                    request_key, request_sha256, response_json, created_at_ms, expires_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (key, request_hash, _canonical_json(response), now_ms, expires_at_ms),
            )
            overflow = connection.execute(
                "SELECT MAX(0, COUNT(*) - ?) AS value FROM idempotency_responses",
                (self.max_entries,),
            ).fetchone()["value"]
            if int(overflow) > 0:
                connection.execute(
                    """
                    DELETE FROM idempotency_responses WHERE request_key IN (
                        SELECT request_key FROM idempotency_responses
                        ORDER BY created_at_ms LIMIT ?
                    )
                    """,
                    (int(overflow),),
                )
            return dict(response), False

    def snapshot(self) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM idempotency_responses WHERE expires_at_ms > ?",
                (now_ms,),
            ).fetchone()
        return {
            "path": str(self.path),
            "active_entries": int(row["count"]),
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
        }
