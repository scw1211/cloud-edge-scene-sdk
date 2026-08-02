"""用途：在云端按场景关联键持久汇聚来自多个边缘节点的语义事件。"""

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from cloud_edge_framework.contracts import SemanticEvent, stable_id
from cloud_edge_framework.reliability import (
    _configure_sqlite_connection,
    IdempotencyConflictError,
)


@dataclass(frozen=True)
class AggregationSpec:
    key: str
    member: str
    expected_members: List[str]
    minimum_members: int
    timeout_ms: int

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "AggregationSpec":
        if not isinstance(value, dict):
            raise ValueError("aggregation spec must be an object")
        key = str(value.get("key", "")).strip()
        member = str(value.get("member", "")).strip()
        if not key or not member:
            raise ValueError("aggregation key and member must not be empty")
        raw_expected = value.get("expected_members", [])
        if not isinstance(raw_expected, list) or not raw_expected:
            raise ValueError("aggregation expected_members must be a non-empty list")
        expected: List[str] = []
        for raw_member in raw_expected:
            name = str(raw_member).strip()
            if not name:
                raise ValueError("aggregation expected member must not be empty")
            if name not in expected:
                expected.append(name)
        expected = sorted(expected)
        if member not in expected:
            raise ValueError("aggregation member must be listed in expected_members")
        minimum = int(value.get("minimum_members", len(expected)))
        if minimum <= 0 or minimum > len(expected):
            raise ValueError("aggregation minimum_members is outside expected member count")
        timeout_ms = int(value.get("timeout_ms", 200))
        if timeout_ms <= 0 or timeout_ms > 600000:
            raise ValueError("aggregation timeout_ms must be within 1 and 600000")
        return cls(
            key=key,
            member=member,
            expected_members=expected,
            minimum_members=minimum,
            timeout_ms=timeout_ms,
        )


@dataclass(frozen=True)
class AggregationLease:
    group_id: str
    scene: str
    group_key: str
    completion_reason: str
    expected_members: List[str]
    received_members: List[str]
    missing_members: List[str]
    result_revision: int
    events: List[SemanticEvent]


class MultiEdgeEventAggregator:
    """Durable event-time join with duplicate, missing-member and timeout handling."""

    def __init__(
        self,
        path: Optional[Path] = None,
        retry_base_seconds: float = 0.25,
        retry_max_seconds: float = 30.0,
    ) -> None:
        self.path = Path(path).resolve() if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.retry_base_seconds = float(retry_base_seconds)
        self.retry_max_seconds = float(retry_max_seconds)
        if (
            self.retry_base_seconds <= 0
            or self.retry_max_seconds <= 0
            or self.retry_base_seconds > self.retry_max_seconds
        ):
            raise ValueError(
                "aggregation retry seconds must be positive and base must not exceed max"
            )
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path) if self.path is not None else ":memory:",
            timeout=10.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        _configure_sqlite_connection(
            self._connection,
            enable_wal=self.path is not None,
        )
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._initialize_locked()
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _initialize_locked(self) -> None:
        """Create/migrate the schema while holding SQLite's write lock."""
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS aggregation_groups (
                group_id TEXT PRIMARY KEY,
                scene TEXT NOT NULL,
                group_key TEXT NOT NULL,
                expected_members_json TEXT NOT NULL,
                expected_member_count INTEGER NOT NULL,
                minimum_members INTEGER NOT NULL,
                timeout_ms INTEGER NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('waiting','inflight','completed')),
                first_received_at_ms INTEGER NOT NULL,
                deadline_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                completion_reason TEXT NOT NULL DEFAULT '',
                result_revision INTEGER NOT NULL DEFAULT 0,
                claimed_member_count INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at_ms INTEGER NOT NULL DEFAULT 0,
                result_json TEXT,
                last_error TEXT NOT NULL DEFAULT '',
                UNIQUE(scene, group_key)
            )
            """
        )
        # Re-read only after BEGIN IMMEDIATE succeeds. Another process may have
        # completed the same migration while this connection waited.
        columns = {
            str(row["name"])
            for row in self._connection.execute(
                "PRAGMA table_info(aggregation_groups)"
            ).fetchall()
        }
        if "result_revision" not in columns:
            self._connection.execute(
                "ALTER TABLE aggregation_groups "
                "ADD COLUMN result_revision INTEGER NOT NULL DEFAULT 0"
            )
        if "claimed_member_count" not in columns:
            self._connection.execute(
                "ALTER TABLE aggregation_groups "
                "ADD COLUMN claimed_member_count INTEGER NOT NULL DEFAULT 0"
            )
        if "attempts" not in columns:
            self._connection.execute(
                "ALTER TABLE aggregation_groups "
                "ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0"
            )
        if "next_attempt_at_ms" not in columns:
            self._connection.execute(
                "ALTER TABLE aggregation_groups "
                "ADD COLUMN next_attempt_at_ms INTEGER NOT NULL DEFAULT 0"
            )
        if "expected_member_count" not in columns:
            self._connection.execute(
                "ALTER TABLE aggregation_groups "
                "ADD COLUMN expected_member_count INTEGER NOT NULL DEFAULT 0"
            )
        # Avoid depending on SQLite's optional JSON1 extension during a rolling
        # upgrade.  Existing groups are few and this migration runs once under
        # BEGIN IMMEDIATE, so decoding the stored contract in Python is both
        # portable and deterministic.
        legacy_groups = self._connection.execute(
            "SELECT group_id, expected_members_json FROM aggregation_groups "
            "WHERE expected_member_count <= 0"
        ).fetchall()
        if legacy_groups:
            self._connection.executemany(
                "UPDATE aggregation_groups SET expected_member_count=? "
                "WHERE group_id=?",
                [
                    (
                        len(json.loads(str(row["expected_members_json"]))),
                        str(row["group_id"]),
                    )
                    for row in legacy_groups
                ],
            )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS aggregation_events (
                group_id TEXT NOT NULL,
                event_id TEXT NOT NULL UNIQUE,
                member TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                received_at_ms INTEGER NOT NULL,
                PRIMARY KEY(group_id, member),
                FOREIGN KEY(group_id) REFERENCES aggregation_groups(group_id)
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_aggregation_state "
            "ON aggregation_groups(state, deadline_at_ms)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_aggregation_retry_ready "
            "ON aggregation_groups(state, next_attempt_at_ms, deadline_at_ms)"
        )

    @staticmethod
    def _serialize_event(event: SemanticEvent) -> str:
        include_scene_payload = bool(
            event.metadata.get("transport_include_scene_payload", False)
        )
        return json.dumps(
            event.to_dict(include_scene_payload=include_scene_payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def submit(self, event: SemanticEvent, spec: AggregationSpec) -> Dict[str, Any]:
        group_id = stable_id("aggregation", event.scene, spec.key)
        now_ms = int(time.time() * 1000)
        expected_json = json.dumps(spec.expected_members, separators=(",", ":"))
        serialized_event = self._serialize_event(event)
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM aggregation_groups WHERE group_id=?", (group_id,)
            ).fetchone()
            if row is None:
                self._connection.execute(
                    """
                    INSERT INTO aggregation_groups(
                        group_id, scene, group_key, expected_members_json,
                        expected_member_count, minimum_members, timeout_ms, state,
                        first_received_at_ms, deadline_at_ms, updated_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'waiting', ?, ?, ?)
                    """,
                    (
                        group_id,
                        event.scene,
                        spec.key,
                        expected_json,
                        len(spec.expected_members),
                        spec.minimum_members,
                        spec.timeout_ms,
                        now_ms,
                        now_ms + spec.timeout_ms,
                        now_ms,
                    ),
                )
            else:
                if (
                    str(row["scene"]) != event.scene
                    or str(row["expected_members_json"]) != expected_json
                    or int(row["minimum_members"]) != spec.minimum_members
                    or int(row["timeout_ms"]) != spec.timeout_ms
                ):
                    raise ValueError("aggregation group was reused with a different policy")
            existing = self._connection.execute(
                "SELECT event_id, payload_json FROM aggregation_events "
                "WHERE group_id=? AND member=?",
                (group_id, spec.member),
            ).fetchone()
            if existing is not None:
                if str(existing["event_id"]) != event.event_id:
                    raise ValueError(
                        "aggregation member {} already submitted another event".format(
                            spec.member
                        )
                    )
                if str(existing["payload_json"]) != serialized_event:
                    raise IdempotencyConflictError(
                        "aggregation event_id/member was already used for a "
                        "different semantic event"
                    )
                # An exact retry is a no-op in every group state.  In
                # particular, do not reopen a partial result or rewrite the
                # original receive timestamp.
                return self.get(group_id, submitted_event_id=event.event_id)
            reused_event = self._connection.execute(
                "SELECT group_id, member, payload_json FROM aggregation_events "
                "WHERE event_id=?",
                (event.event_id,),
            ).fetchone()
            if (
                reused_event is not None
                and str(reused_event["group_id"]) != group_id
            ):
                raise IdempotencyConflictError(
                    "aggregation event_id was already used for another group"
                )
            if reused_event is not None:
                if (
                    str(reused_event["member"]) != spec.member
                    or str(reused_event["payload_json"]) != serialized_event
                ):
                    raise IdempotencyConflictError(
                        "aggregation event_id was already used for a different "
                        "cloud submission"
                    )
                return self.get(group_id, submitted_event_id=event.event_id)
            if (
                row is not None
                and str(row["state"]) == "completed"
                and str(row["completion_reason"]) == "all_expected_members"
            ):
                return self.get(group_id, submitted_event_id=event.event_id)
            inserted = self._connection.execute(
                """
                INSERT OR IGNORE INTO aggregation_events(
                    group_id, event_id, member, payload_json, received_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    group_id,
                    event.event_id,
                    spec.member,
                    serialized_event,
                    now_ms,
                ),
            )
            if inserted.rowcount != 1:
                # This is only reachable when another aggregator process won a
                # cross-process UNIQUE race after the checks above.  Re-read
                # the winning row and apply the same exact-retry contract.
                raced = self._connection.execute(
                    "SELECT group_id, member, payload_json "
                    "FROM aggregation_events WHERE event_id=?",
                    (event.event_id,),
                ).fetchone()
                if (
                    raced is None
                    or str(raced["group_id"]) != group_id
                    or str(raced["member"]) != spec.member
                    or str(raced["payload_json"]) != serialized_event
                ):
                    raise IdempotencyConflictError(
                        "aggregation event_id was concurrently used for a "
                        "different cloud submission"
                    )
                return self.get(group_id, submitted_event_id=event.event_id)
            if (
                row is not None
                and str(row["state"]) == "completed"
                and str(row["completion_reason"]) != "all_expected_members"
                and inserted.rowcount == 1
            ):
                # A timeout result is explicitly partial.  A previously missing
                # member may reopen it so the cloud can issue a later, complete
                # correction.  A retry from an already stored member leaves the
                # partial result untouched.
                self._connection.execute(
                    """
                    UPDATE aggregation_groups
                    SET state='waiting', completion_reason='', updated_at_ms=?,
                        claimed_member_count=0, attempts=0,
                        next_attempt_at_ms=0, last_error=''
                    WHERE group_id=? AND state='completed'
                    """,
                    (now_ms, group_id),
                )
            else:
                self._connection.execute(
                    "UPDATE aggregation_groups "
                    "SET updated_at_ms=?, attempts=0, next_attempt_at_ms=0, "
                    "last_error='' WHERE group_id=?",
                    (now_ms, group_id),
                )
        return self.get(group_id, submitted_event_id=event.event_id)

    @staticmethod
    def _members(connection: sqlite3.Connection, group_id: str) -> List[str]:
        return [
            str(row["member"])
            for row in connection.execute(
                "SELECT member FROM aggregation_events WHERE group_id=? ORDER BY member",
                (group_id,),
            ).fetchall()
        ]

    def _claim(self, group_id: str, now_ms: int) -> Optional[AggregationLease]:
        row = self._connection.execute(
            "SELECT * FROM aggregation_groups WHERE group_id=?", (group_id,)
        ).fetchone()
        if row is None or str(row["state"]) != "waiting":
            return None
        if now_ms < int(row["next_attempt_at_ms"]):
            return None
        members = self._members(self._connection, group_id)
        expected = json.loads(str(row["expected_members_json"]))
        complete = set(expected).issubset(set(members))
        timed_out = now_ms >= int(row["deadline_at_ms"])
        minimum_met = len(members) >= int(row["minimum_members"])
        if not complete and not (timed_out and minimum_met):
            return None
        reason = "all_expected_members" if complete else "timeout_with_partial_members"
        updated = self._connection.execute(
            """
            UPDATE aggregation_groups
            SET state='inflight', completion_reason=?, claimed_member_count=?,
                updated_at_ms=?
            WHERE group_id=? AND state='waiting' AND next_attempt_at_ms <= ?
            """,
            (reason, len(members), now_ms, group_id, now_ms),
        )
        if updated.rowcount != 1:
            return None
        event_rows = self._connection.execute(
            """
            SELECT payload_json FROM aggregation_events
            WHERE group_id=? ORDER BY received_at_ms, event_id
            """,
            (group_id,),
        ).fetchall()
        return AggregationLease(
            group_id=group_id,
            scene=str(row["scene"]),
            group_key=str(row["group_key"]),
            completion_reason=reason,
            expected_members=list(expected),
            received_members=list(members),
            missing_members=sorted(set(expected) - set(members)),
            result_revision=int(row["result_revision"]) + 1,
            events=[
                SemanticEvent.from_dict(json.loads(str(item["payload_json"])))
                for item in event_rows
            ],
        )

    def claim(self, group_id: str) -> Optional[AggregationLease]:
        with self._lock, self._connection:
            return self._claim(str(group_id), int(time.time() * 1000))

    def claim_due(self, limit: int = 64) -> List[AggregationLease]:
        if limit <= 0:
            raise ValueError("aggregation claim limit must be positive")
        now_ms = int(time.time() * 1000)
        leases: List[AggregationLease] = []
        with self._lock, self._connection:
            rows = self._connection.execute(
                """
                SELECT group_id FROM aggregation_groups
                WHERE state='waiting' AND next_attempt_at_ms <= ? AND (
                    deadline_at_ms <= ? OR attempts > 0 OR
                    (SELECT COUNT(*) FROM aggregation_events AS events
                     WHERE events.group_id=aggregation_groups.group_id)
                        >= expected_member_count
                )
                ORDER BY
                    CASE WHEN deadline_at_ms <= ? THEN 0 ELSE 1 END,
                    next_attempt_at_ms, deadline_at_ms
                LIMIT ?
                """,
                (now_ms, now_ms, now_ms, int(limit)),
            ).fetchall()
            for row in rows:
                lease = self._claim(str(row["group_id"]), now_ms)
                if lease is not None:
                    leases.append(lease)
        return leases

    def complete(self, group_id: str, result: Dict[str, Any]) -> None:
        self.complete_many([(str(group_id), result)])

    def complete_many(
        self,
        results: Sequence[Any],
    ) -> None:
        """Atomically commit several independently coordinated sample groups."""
        items = [(str(group_id), result) for group_id, result in results]
        if not items:
            return
        if len({group_id for group_id, _ in items}) != len(items):
            raise ValueError("aggregation result batch contains duplicate groups")
        if any(not isinstance(result, dict) for _, result in items):
            raise ValueError("aggregation result must be an object")
        now_ms = int(time.time() * 1000)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                updates = []
                for group_id, result in items:
                    row = self._connection.execute(
                        "SELECT state, completion_reason, claimed_member_count "
                        "FROM aggregation_groups WHERE group_id=?",
                        (group_id,),
                    ).fetchone()
                    if row is None or str(row["state"]) != "inflight":
                        raise ValueError(
                            "aggregation group is not inflight: {}".format(
                                group_id
                            )
                        )
                    current_member_count = len(
                        self._members(self._connection, group_id)
                    )
                    members_arrived_during_claim = (
                        current_member_count
                        > int(row["claimed_member_count"])
                    )
                    updates.append(
                        (
                            (
                                "waiting"
                                if members_arrived_during_claim
                                else "completed"
                            ),
                            (
                                ""
                                if members_arrived_during_claim
                                else str(row["completion_reason"])
                            ),
                            json.dumps(
                                result,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            now_ms,
                            group_id,
                        )
                    )
                self._connection.executemany(
                    """
                    UPDATE aggregation_groups
                    SET state=?, result_revision=result_revision+1,
                        completion_reason=?, claimed_member_count=0,
                        attempts=0, next_attempt_at_ms=0,
                        result_json=?, updated_at_ms=?, last_error=''
                    WHERE group_id=? AND state='inflight'
                    """,
                    updates,
                )
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def release(self, group_id: str, error: str) -> None:
        now_ms = int(time.time() * 1000)
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT attempts, claimed_member_count FROM aggregation_groups "
                "WHERE group_id=? AND state='inflight'",
                (str(group_id),),
            ).fetchone()
            if row is None:
                return
            current_member_count = len(
                self._members(self._connection, str(group_id))
            )
            members_arrived_during_claim = (
                current_member_count > int(row["claimed_member_count"])
            )
            if members_arrived_during_claim:
                # The failed lease did not contain the newly arrived evidence;
                # let the richer revision run immediately once before applying
                # backoff to any subsequent failure.
                attempts = 0
                next_attempt_at_ms = 0
            else:
                attempts = max(0, int(row["attempts"])) + 1
                delay_seconds = min(
                    self.retry_max_seconds,
                    self.retry_base_seconds * (2.0 ** min(attempts - 1, 30)),
                )
                next_attempt_at_ms = now_ms + max(
                    1, int(delay_seconds * 1000.0)
                )
            self._connection.execute(
                """
                UPDATE aggregation_groups
                SET state='waiting', claimed_member_count=0,
                    attempts=?, next_attempt_at_ms=?,
                    last_error=?, updated_at_ms=?
                WHERE group_id=? AND state='inflight'
                """,
                (
                    attempts,
                    next_attempt_at_ms,
                    str(error)[:2000],
                    now_ms,
                    str(group_id),
                ),
            )

    def get(
        self,
        group_id: str,
        submitted_event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        key = str(group_id).strip()
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM aggregation_groups WHERE group_id=?", (key,)
            ).fetchone()
            if row is None:
                raise KeyError("aggregation group not found: {}".format(key))
            members = self._members(self._connection, key)
            submitted_received_at_ms = None
            if submitted_event_id is not None:
                submitted_row = self._connection.execute(
                    "SELECT received_at_ms FROM aggregation_events "
                    "WHERE group_id=? AND event_id=?",
                    (key, str(submitted_event_id)),
                ).fetchone()
                if submitted_row is not None:
                    submitted_received_at_ms = int(
                        submitted_row["received_at_ms"]
                    )
        expected = json.loads(str(row["expected_members_json"]))
        result_json = row["result_json"]
        result = json.loads(str(result_json)) if result_json else None
        missing_members = sorted(set(expected) - set(members))
        state = str(row["state"])
        completion_reason = str(row["completion_reason"])
        evidence_complete = (
            state == "completed"
            and completion_reason == "all_expected_members"
            and not missing_members
        )
        if state == "completed" and result_json:
            finality = "final" if evidence_complete else "partial_final"
        else:
            finality = "pending"
        global_confirmation = bool(
            evidence_complete
            and isinstance(result, dict)
            and result.get("global_confirmation", False)
        )
        response = {
            "group_id": key,
            "scene": str(row["scene"]),
            "group_key": str(row["group_key"]),
            "state": state,
            "expected_members": expected,
            "received_members": members,
            "missing_members": missing_members,
            "evidence_complete": evidence_complete,
            "finality": finality,
            "cloud_confirmed": global_confirmation,
            "global_confirmation": global_confirmation,
            "result_revision": int(row["result_revision"]),
            "minimum_members": int(row["minimum_members"]),
            "timeout_ms": int(row["timeout_ms"]),
            "first_received_at_ms": int(row["first_received_at_ms"]),
            "deadline_at_ms": int(row["deadline_at_ms"]),
            "completion_reason": completion_reason,
            "result": result,
            "attempts": int(row["attempts"]),
            "next_attempt_at_ms": int(row["next_attempt_at_ms"]),
            "last_error": str(row["last_error"]),
        }
        if submitted_received_at_ms is not None:
            response["submitted_event_received_at_ms"] = submitted_received_at_ms
        return response

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counts = {
                str(row["state"]): int(row["count"])
                for row in self._connection.execute(
                    "SELECT state, COUNT(*) AS count FROM aggregation_groups GROUP BY state"
                ).fetchall()
            }
            event_count = int(
                self._connection.execute(
                    "SELECT COUNT(*) AS count FROM aggregation_events"
                ).fetchone()["count"]
            )
            retry_waiting = int(
                self._connection.execute(
                    "SELECT COUNT(*) AS count FROM aggregation_groups "
                    "WHERE state='waiting' AND attempts > 0"
                ).fetchone()["count"]
            )
        return {
            "path": str(self.path) if self.path is not None else ":memory:",
            "states": {
                "waiting": counts.get("waiting", 0),
                "inflight": counts.get("inflight", 0),
                "completed": counts.get("completed", 0),
            },
            "event_count": event_count,
            "retry_waiting": retry_waiting,
        }

    def close(self) -> None:
        with self._lock:
            self._connection.close()
