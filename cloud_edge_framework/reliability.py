"""用途：提供可确认的 SQLite 弱网 Outbox 和云端幂等响应缓存。"""

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from cloud_edge_framework.contracts import SemanticEvent


_SOURCE_ENVELOPE_SHA256_KEY = "_source_envelope_sha256"
_SOURCE_BUSINESS_CONTEXT_KEY = "_source_business_control_context"
_SOURCE_BUSINESS_CONTEXT_FIELDS = (
    "conflict_suspected",
    "model_disagreement",
)
_SQLITE_BUSY_TIMEOUT_MS = 10_000


def _configure_sqlite_connection(
    connection: sqlite3.Connection,
    enable_wal: bool,
) -> None:
    """Apply connection pragmas, retrying WAL negotiation across processes."""
    connection.execute(
        "PRAGMA busy_timeout={}".format(_SQLITE_BUSY_TIMEOUT_MS)
    )
    if not enable_wal:
        return
    deadline = time.monotonic() + (_SQLITE_BUSY_TIMEOUT_MS / 1000.0)
    while True:
        try:
            current = connection.execute("PRAGMA journal_mode").fetchone()
            if current is not None and str(current[0]).lower() == "wal":
                break
            # Preserve SQLite's prior fallback behavior on filesystems that do
            # not support WAL: a successful PRAGMA may select another mode.
            connection.execute("PRAGMA journal_mode=WAL").fetchone()
            break
        except sqlite3.OperationalError as error:
            message = str(error).lower()
            if "locked" not in message and "busy" not in message:
                raise
            if time.monotonic() >= deadline:
                raise
        time.sleep(0.01)
    connection.execute("PRAGMA synchronous=FULL")


class IdempotencyConflictError(ValueError):
    """Raised when one idempotency key is reused for a different request."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _request_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def source_submission_identity(metadata: Dict[str, Any]) -> str:
    """Return the stable identity shared by Outbox and review lifecycle rows.

    Runtime measurements deliberately stay outside this identity, while
    request-level business controls that can change evidence or routing are
    included.  Missing control metadata is interpreted as the historical
    default (both flags false), so pending rows written by older versions keep
    their exact-retry behavior after an upgrade.
    """
    if not isinstance(metadata, dict):
        return ""
    source_hash = str(metadata.get(_SOURCE_ENVELOPE_SHA256_KEY, "")).strip().lower()
    if not (
        len(source_hash) == 64
        and all(character in "0123456789abcdef" for character in source_hash)
    ):
        return ""
    raw_context = metadata.get(_SOURCE_BUSINESS_CONTEXT_KEY, {})
    if raw_context is None:
        raw_context = {}
    if not isinstance(raw_context, dict):
        return ""
    context: Dict[str, bool] = {}
    for field_name in _SOURCE_BUSINESS_CONTEXT_FIELDS:
        value = raw_context.get(field_name, False)
        if not isinstance(value, bool):
            return ""
        context[field_name] = value
    return "source_envelope:{}|business_context:{}".format(
        source_hash,
        _request_sha256(context),
    )


@dataclass(frozen=True)
class OutboxLease:
    event: SemanticEvent
    attempts: int
    reconciliation: bool = False
    aggregation_submitted: bool = False


class SQLiteOutbox:
    """Durable at-least-once queue with leases and stable event-id deduplication."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        try:
            connection.row_factory = sqlite3.Row
            _configure_sqlite_connection(connection, enable_wal=True)
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            # Serialize schema inspection and ALTER TABLE across processes.
            # The column list is deliberately read only after this lock is held.
            connection.execute("BEGIN IMMEDIATE")
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
                    aggregation_wait_started_at_ms INTEGER,
                    last_error TEXT NOT NULL DEFAULT ''
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(outbox_events)"
                ).fetchall()
            }
            if "aggregation_wait_started_at_ms" not in columns:
                connection.execute(
                    "ALTER TABLE outbox_events "
                    "ADD COLUMN aggregation_wait_started_at_ms INTEGER"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_outbox_ready "
                "ON outbox_events(state, available_at_ms, created_at_ms)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS outbox_reconciliation (
                    event_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL CHECK(state IN ('pending','inflight','completed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    available_at_ms INTEGER NOT NULL,
                    lease_until_ms INTEGER,
                    started_at_ms INTEGER NOT NULL,
                    deadline_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    completed_at_ms INTEGER,
                    last_error TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(event_id) REFERENCES outbox_events(event_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_outbox_reconciliation_ready "
                "ON outbox_reconciliation(state, available_at_ms, deadline_at_ms)"
            )

    @staticmethod
    def _serialized(event: SemanticEvent) -> str:
        include_scene_payload = bool(
            event.metadata.get("transport_include_scene_payload", False)
        )
        return _canonical_json(
            event.to_dict(include_scene_payload=include_scene_payload)
        )

    @staticmethod
    def _identity(serialized: str) -> str:
        """Return an immutable-input identity, not a runtime-measurement hash."""
        try:
            value = json.loads(serialized)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            metadata = value.get("metadata")
            if isinstance(metadata, dict):
                source_identity = source_submission_identity(metadata)
                if source_identity:
                    return source_identity
        return "serialized_payload:" + hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def append(self, event: SemanticEvent) -> bool:
        now_ms = int(time.time() * 1000)
        serialized = self._serialized(event)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO outbox_events(
                    event_id, payload_json, state, attempts, available_at_ms,
                    created_at_ms, updated_at_ms
                ) VALUES (?, ?, 'pending', 0, ?, ?, ?)
                """,
                (event.event_id, serialized, now_ms, now_ms, now_ms),
            )
            if cursor.rowcount == 1:
                return True
            existing = connection.execute(
                "SELECT payload_json FROM outbox_events WHERE event_id=?",
                (event.event_id,),
            ).fetchone()
            if existing is None or self._identity(
                str(existing["payload_json"])
            ) != self._identity(serialized):
                raise IdempotencyConflictError(
                    "outbox event_id was already used for a different cloud submission"
                )
            return False

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
                SELECT event_id, payload_json, attempts,
                       aggregation_wait_started_at_ms
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
                reconciliation=False,
                aggregation_submitted=(
                    row["aggregation_wait_started_at_ms"] is not None
                ),
            )
            for row in rows
        ]

    def claim_reconciliation(
        self,
        limit: int,
        lease_seconds: float,
    ) -> List[OutboxLease]:
        """Claim due, still-bounded late-result reconciliation work."""
        if limit <= 0 or lease_seconds <= 0:
            raise ValueError(
                "reconciliation claim limit and lease_seconds must be positive"
            )
        now_ms = int(time.time() * 1000)
        lease_until_ms = now_ms + int(lease_seconds * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE outbox_reconciliation
                SET state='pending', lease_until_ms=NULL, updated_at_ms=?
                WHERE state='inflight' AND lease_until_ms IS NOT NULL
                  AND lease_until_ms <= ? AND deadline_at_ms > ?
                """,
                (now_ms, now_ms, now_ms),
            )
            rows = connection.execute(
                """
                SELECT reconciliation.event_id, events.payload_json,
                       reconciliation.attempts
                FROM outbox_reconciliation AS reconciliation
                JOIN outbox_events AS events
                  ON events.event_id=reconciliation.event_id
                WHERE reconciliation.state='pending'
                  AND reconciliation.available_at_ms <= ?
                  AND reconciliation.deadline_at_ms > ?
                ORDER BY reconciliation.available_at_ms,
                         reconciliation.started_at_ms,
                         reconciliation.event_id
                LIMIT ?
                """,
                (now_ms, now_ms, int(limit)),
            ).fetchall()
            if rows:
                connection.executemany(
                    """
                    UPDATE outbox_reconciliation
                    SET state='inflight', attempts=attempts+1,
                        lease_until_ms=?, updated_at_ms=?
                    WHERE event_id=? AND state='pending'
                    """,
                    [
                        (lease_until_ms, now_ms, str(row["event_id"]))
                        for row in rows
                    ],
                )
            connection.commit()
        return [
            OutboxLease(
                event=SemanticEvent.from_dict(json.loads(str(row["payload_json"]))),
                attempts=int(row["attempts"]) + 1,
                reconciliation=True,
                aggregation_submitted=True,
            )
            for row in rows
        ]

    def sweep_expired_reconciliation(self) -> List[str]:
        """Terminalize bounded pollers and return their ids for observability."""
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT event_id FROM outbox_reconciliation "
                "WHERE state IN ('pending','inflight') AND deadline_at_ms <= ?",
                (now_ms,),
            ).fetchall()
            expired_ids = [str(row["event_id"]) for row in rows]
            if expired_ids:
                connection.executemany(
                    """
                    UPDATE outbox_reconciliation
                    SET state='completed', lease_until_ms=NULL,
                        completed_at_ms=?, updated_at_ms=?
                    WHERE event_id=? AND state IN ('pending','inflight')
                    """,
                    [(now_ms, now_ms, event_id) for event_id in expired_ids],
                )
        return expired_ids

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

    def mark_aggregation_submitted(
        self, event_ids: Sequence[str]
    ) -> None:
        """Remember durable cloud acceptance without completing the Outbox."""
        ids = [str(value) for value in event_ids]
        if not ids:
            return
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.executemany(
                """
                UPDATE outbox_events
                SET aggregation_wait_started_at_ms=COALESCE(
                        aggregation_wait_started_at_ms, ?
                    ),
                    updated_at_ms=?
                WHERE event_id=?
                """,
                [(now_ms, now_ms, event_id) for event_id in ids],
            )

    def acknowledge_reconciliation(self, event_ids: Sequence[str]) -> None:
        ids = [str(value) for value in event_ids]
        if not ids:
            return
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.executemany(
                """
                UPDATE outbox_reconciliation
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

    def release_reconciliation(
        self,
        event_ids: Sequence[str],
        error: str,
        max_backoff_seconds: float,
    ) -> List[str]:
        """Release failed reconciliation attempts without crossing their deadline."""
        ids = [str(value) for value in event_ids]
        if not ids:
            return []
        now_ms = int(time.time() * 1000)
        expired_ids: List[str] = []
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT event_id, attempts, deadline_at_ms "
                "FROM outbox_reconciliation WHERE event_id IN ({})".format(
                    ",".join("?" for _ in ids)
                ),
                ids,
            ).fetchall()
            pending_updates = []
            completed_updates = []
            for row in rows:
                event_id = str(row["event_id"])
                deadline_at_ms = int(row["deadline_at_ms"])
                if deadline_at_ms <= now_ms:
                    expired_ids.append(event_id)
                    completed_updates.append((now_ms, now_ms, event_id))
                    continue
                attempts = max(1, int(row["attempts"]))
                backoff_seconds = min(
                    float(max_backoff_seconds),
                    2.0 ** min(attempts - 1, 16),
                )
                available_at_ms = min(
                    deadline_at_ms,
                    now_ms + max(1, int(backoff_seconds * 1000)),
                )
                pending_updates.append(
                    (
                        available_at_ms,
                        now_ms,
                        str(error)[:2000],
                        event_id,
                    )
                )
            if pending_updates:
                connection.executemany(
                    """
                    UPDATE outbox_reconciliation
                    SET state='pending', available_at_ms=?, lease_until_ms=NULL,
                        updated_at_ms=?, last_error=?
                    WHERE event_id=?
                    """,
                    pending_updates,
                )
            if completed_updates:
                connection.executemany(
                    """
                    UPDATE outbox_reconciliation
                    SET state='completed', lease_until_ms=NULL,
                        completed_at_ms=?, updated_at_ms=?
                    WHERE event_id=?
                    """,
                    completed_updates,
                )
        return expired_ids

    def defer_reconciliation(
        self,
        event_ids: Sequence[str],
        poll_seconds: float,
    ) -> List[str]:
        """Schedule the next low-frequency poll, or close an expired poller."""
        ids = [str(value) for value in event_ids]
        if not ids:
            return []
        delay_seconds = float(poll_seconds)
        if delay_seconds <= 0:
            raise ValueError("reconciliation poll_seconds must be positive")
        now_ms = int(time.time() * 1000)
        expired_ids: List[str] = []
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT event_id, deadline_at_ms FROM outbox_reconciliation "
                "WHERE event_id IN ({})".format(
                    ",".join("?" for _ in ids)
                ),
                ids,
            ).fetchall()
            pending_updates = []
            completed_updates = []
            for row in rows:
                event_id = str(row["event_id"])
                deadline_at_ms = int(row["deadline_at_ms"])
                if deadline_at_ms <= now_ms:
                    expired_ids.append(event_id)
                    completed_updates.append((now_ms, now_ms, event_id))
                else:
                    pending_updates.append(
                        (
                            min(
                                deadline_at_ms,
                                now_ms + max(1, int(delay_seconds * 1000)),
                            ),
                            now_ms,
                            event_id,
                        )
                    )
            if pending_updates:
                connection.executemany(
                    """
                    UPDATE outbox_reconciliation
                    SET state='pending', available_at_ms=?, lease_until_ms=NULL,
                        updated_at_ms=?, last_error=''
                    WHERE event_id=?
                    """,
                    pending_updates,
                )
            if completed_updates:
                connection.executemany(
                    """
                    UPDATE outbox_reconciliation
                    SET state='completed', lease_until_ms=NULL,
                        completed_at_ms=?, updated_at_ms=?
                    WHERE event_id=?
                    """,
                    completed_updates,
                )
        return expired_ids

    def defer_waiting(
        self,
        event_ids: Sequence[str],
        poll_seconds: float,
    ) -> None:
        """Release successfully delivered events for a short result poll.

        Waiting for aggregation peers is not a transport failure and therefore
        must not inherit the exponential failure backoff.
        """
        ids = [str(value) for value in event_ids]
        if not ids:
            return
        delay_seconds = float(poll_seconds)
        if delay_seconds <= 0:
            raise ValueError("waiting poll_seconds must be positive")
        now_ms = int(time.time() * 1000)
        available_at_ms = now_ms + max(1, int(delay_seconds * 1000))
        with self._lock, self._connect() as connection:
            connection.executemany(
                """
                UPDATE outbox_events
                SET state='pending', available_at_ms=?, lease_until_ms=NULL,
                    updated_at_ms=?, last_error=''
                WHERE event_id=?
                """,
                [(available_at_ms, now_ms, event_id) for event_id in ids],
            )

    def defer_aggregation_wait(
        self,
        event_ids: Sequence[str],
        poll_seconds: float,
        max_wait_seconds: float,
        reconciliation_poll_seconds: float = 5.0,
        reconciliation_max_wait_seconds: float = 60.0,
    ) -> List[str]:
        """Poll an accepted aggregation for a bounded amount of time.

        The cloud may need time to receive the remaining members, but a
        permanently missing peer must not keep an edge Outbox row alive
        forever.  The timeout begins on the first successful aggregation
        response, not when the event entered the Outbox; an event accumulated
        during an outage therefore receives a full aggregation wait window
        after connectivity returns.  Rows that keep waiting beyond
        ``max_wait_seconds`` become terminal and are returned to the caller so
        its review lifecycle can be closed with an explicit
        partial/local-timeout stage.
        """
        ids = [str(value) for value in event_ids]
        if not ids:
            return []
        delay_seconds = float(poll_seconds)
        wait_seconds = float(max_wait_seconds)
        reconcile_delay_seconds = float(reconciliation_poll_seconds)
        reconcile_wait_seconds = float(reconciliation_max_wait_seconds)
        if (
            delay_seconds <= 0
            or wait_seconds <= 0
            or reconcile_delay_seconds <= 0
            or reconcile_wait_seconds <= 0
        ):
            raise ValueError(
                "aggregation and reconciliation poll/max wait seconds must be positive"
            )
        now_ms = int(time.time() * 1000)
        deadline_age_ms = max(1, int(wait_seconds * 1000))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT event_id, aggregation_wait_started_at_ms "
                "FROM outbox_events "
                "WHERE event_id IN ({})".format(
                    ",".join("?" for _ in ids)
                ),
                ids,
            ).fetchall()
            started_now_ids = [
                str(row["event_id"])
                for row in rows
                if row["aggregation_wait_started_at_ms"] is None
            ]
            expired_ids = [
                str(row["event_id"])
                for row in rows
                if row["aggregation_wait_started_at_ms"] is not None
                and now_ms - int(row["aggregation_wait_started_at_ms"])
                >= deadline_age_ms
            ]
            expired_set = set(expired_ids)
            pending_ids = [
                str(row["event_id"])
                for row in rows
                if str(row["event_id"]) not in expired_set
            ]
            if pending_ids:
                available_at_ms = now_ms + max(1, int(delay_seconds * 1000))
                connection.executemany(
                    """
                    UPDATE outbox_events
                    SET state='pending', available_at_ms=?, lease_until_ms=NULL,
                        updated_at_ms=?, last_error=''
                    WHERE event_id=?
                    """,
                    [
                        (available_at_ms, now_ms, event_id)
                        for event_id in pending_ids
                    ],
                )
            if started_now_ids:
                connection.executemany(
                    "UPDATE outbox_events "
                    "SET aggregation_wait_started_at_ms=? WHERE event_id=?",
                    [(now_ms, event_id) for event_id in started_now_ids],
                )
            if expired_ids:
                connection.executemany(
                    """
                    UPDATE outbox_events
                    SET state='completed', lease_until_ms=NULL,
                        completed_at_ms=?, updated_at_ms=?, last_error=''
                    WHERE event_id=?
                    """,
                    [(now_ms, now_ms, event_id) for event_id in expired_ids],
                )
                reconciliation_available_at_ms = now_ms + max(
                    1, int(reconcile_delay_seconds * 1000)
                )
                reconciliation_deadline_at_ms = now_ms + max(
                    1, int(reconcile_wait_seconds * 1000)
                )
                connection.executemany(
                    """
                    INSERT INTO outbox_reconciliation(
                        event_id, state, attempts, available_at_ms,
                        started_at_ms, deadline_at_ms, updated_at_ms
                    ) VALUES (?, 'pending', 0, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO UPDATE SET
                        state='pending', attempts=0,
                        available_at_ms=excluded.available_at_ms,
                        lease_until_ms=NULL,
                        started_at_ms=excluded.started_at_ms,
                        deadline_at_ms=excluded.deadline_at_ms,
                        updated_at_ms=excluded.updated_at_ms,
                        completed_at_ms=NULL, last_error=''
                    """,
                    [
                        (
                            event_id,
                            reconciliation_available_at_ms,
                            now_ms,
                            reconciliation_deadline_at_ms,
                            now_ms,
                        )
                        for event_id in expired_ids
                    ],
                )
        return expired_ids

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
            connection.execute(
                """
                UPDATE outbox_reconciliation
                SET state='completed', lease_until_ms=NULL, completed_at_ms=?,
                    updated_at_ms=?
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

    def reconciliation_count(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM outbox_reconciliation "
                "WHERE state IN ('pending','inflight')"
            ).fetchone()
        return int(row["count"])

    def work_count(self) -> int:
        """Return ordinary delivery plus bounded reconciliation work."""
        with self._lock, self._connect() as connection:
            ordinary = connection.execute(
                "SELECT COUNT(*) AS count FROM outbox_events "
                "WHERE state IN ('pending','inflight')"
            ).fetchone()["count"]
            reconciliation = connection.execute(
                "SELECT COUNT(*) AS count FROM outbox_reconciliation "
                "WHERE state IN ('pending','inflight')"
            ).fetchone()["count"]
        return int(ordinary) + int(reconciliation)

    def next_available_delay(self) -> Optional[float]:
        """Seconds until the next queue item is claimable, or ``None`` if idle.

        Inflight lease expiry is included so a crashed sender cannot make the
        worker sleep past recovery.  Reconciliation deadlines are also wakeup
        points, allowing bounded work to become terminal without another event.
        """
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT MIN(ready_at_ms) AS ready_at_ms FROM (
                    SELECT available_at_ms AS ready_at_ms
                    FROM outbox_events WHERE state='pending'
                    UNION ALL
                    SELECT lease_until_ms AS ready_at_ms
                    FROM outbox_events
                    WHERE state='inflight' AND lease_until_ms IS NOT NULL
                    UNION ALL
                    SELECT MIN(available_at_ms, deadline_at_ms) AS ready_at_ms
                    FROM outbox_reconciliation WHERE state='pending'
                    UNION ALL
                    SELECT MIN(lease_until_ms, deadline_at_ms) AS ready_at_ms
                    FROM outbox_reconciliation
                    WHERE state='inflight' AND lease_until_ms IS NOT NULL
                )
                """
            ).fetchone()
        ready_at_ms = row["ready_at_ms"]
        if ready_at_ms is None:
            return None
        return max(0.0, (int(ready_at_ms) - now_ms) / 1000.0)

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
            reconciliation_counts = {
                str(row["state"]): int(row["count"])
                for row in connection.execute(
                    "SELECT state, COUNT(*) AS count "
                    "FROM outbox_reconciliation GROUP BY state"
                ).fetchall()
            }
            reconciliation_attempts = connection.execute(
                "SELECT COALESCE(SUM(attempts), 0) AS value "
                "FROM outbox_reconciliation"
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
            "reconciliation": {
                "states": {
                    "pending": reconciliation_counts.get("pending", 0),
                    "inflight": reconciliation_counts.get("inflight", 0),
                    "completed": reconciliation_counts.get("completed", 0),
                },
                "active": reconciliation_counts.get("pending", 0)
                + reconciliation_counts.get("inflight", 0),
                "attempts": int(reconciliation_attempts),
            },
            "next_available_delay_seconds": self.next_available_delay(),
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

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA busy_timeout=10000")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

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
