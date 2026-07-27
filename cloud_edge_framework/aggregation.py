"""用途：在云端按场景关联键持久汇聚来自多个边缘节点的语义事件。"""

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from cloud_edge_framework.contracts import SemanticEvent, stable_id


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
    events: List[SemanticEvent]


class MultiEdgeEventAggregator:
    """Durable event-time join with duplicate, missing-member and timeout handling."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path).resolve() if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path) if self.path is not None else ":memory:",
            timeout=10.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=10000")
        if self.path is not None:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS aggregation_groups (
                    group_id TEXT PRIMARY KEY,
                    scene TEXT NOT NULL,
                    group_key TEXT NOT NULL,
                    expected_members_json TEXT NOT NULL,
                    minimum_members INTEGER NOT NULL,
                    timeout_ms INTEGER NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('waiting','inflight','completed')),
                    first_received_at_ms INTEGER NOT NULL,
                    deadline_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    completion_reason TEXT NOT NULL DEFAULT '',
                    result_json TEXT,
                    last_error TEXT NOT NULL DEFAULT '',
                    UNIQUE(scene, group_key)
                )
                """
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
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM aggregation_groups WHERE group_id=?", (group_id,)
            ).fetchone()
            if row is None:
                self._connection.execute(
                    """
                    INSERT INTO aggregation_groups(
                        group_id, scene, group_key, expected_members_json,
                        minimum_members, timeout_ms, state, first_received_at_ms,
                        deadline_at_ms, updated_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, 'waiting', ?, ?, ?)
                    """,
                    (
                        group_id,
                        event.scene,
                        spec.key,
                        expected_json,
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
                if str(row["state"]) == "completed":
                    return self.get(group_id)

            existing = self._connection.execute(
                "SELECT event_id FROM aggregation_events WHERE group_id=? AND member=?",
                (group_id, spec.member),
            ).fetchone()
            if existing is not None and str(existing["event_id"]) != event.event_id:
                raise ValueError(
                    "aggregation member {} already submitted another event".format(
                        spec.member
                    )
                )
            reused_event = self._connection.execute(
                "SELECT group_id, member FROM aggregation_events WHERE event_id=?",
                (event.event_id,),
            ).fetchone()
            if (
                reused_event is not None
                and str(reused_event["group_id"]) != group_id
            ):
                raise ValueError(
                    "aggregation event {} already belongs to another group".format(
                        event.event_id
                    )
                )
            self._connection.execute(
                """
                INSERT OR IGNORE INTO aggregation_events(
                    group_id, event_id, member, payload_json, received_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    group_id,
                    event.event_id,
                    spec.member,
                    self._serialize_event(event),
                    now_ms,
                ),
            )
            self._connection.execute(
                "UPDATE aggregation_groups SET updated_at_ms=? WHERE group_id=?",
                (now_ms, group_id),
            )
        return self.get(group_id)

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
            SET state='inflight', completion_reason=?, updated_at_ms=?, last_error=''
            WHERE group_id=? AND state='waiting'
            """,
            (reason, now_ms, group_id),
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
                WHERE state='waiting' AND deadline_at_ms <= ?
                ORDER BY deadline_at_ms LIMIT ?
                """,
                (now_ms, int(limit)),
            ).fetchall()
            for row in rows:
                lease = self._claim(str(row["group_id"]), now_ms)
                if lease is not None:
                    leases.append(lease)
        return leases

    def complete(self, group_id: str, result: Dict[str, Any]) -> None:
        if not isinstance(result, dict):
            raise ValueError("aggregation result must be an object")
        now_ms = int(time.time() * 1000)
        with self._lock, self._connection:
            updated = self._connection.execute(
                """
                UPDATE aggregation_groups
                SET state='completed', result_json=?, updated_at_ms=?, last_error=''
                WHERE group_id=? AND state='inflight'
                """,
                (
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    now_ms,
                    str(group_id),
                ),
            )
            if updated.rowcount != 1:
                raise ValueError("aggregation group is not inflight: {}".format(group_id))

    def release(self, group_id: str, error: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE aggregation_groups
                SET state='waiting', last_error=?, updated_at_ms=?
                WHERE group_id=? AND state='inflight'
                """,
                (str(error)[:2000], int(time.time() * 1000), str(group_id)),
            )

    def get(self, group_id: str) -> Dict[str, Any]:
        key = str(group_id).strip()
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM aggregation_groups WHERE group_id=?", (key,)
            ).fetchone()
            if row is None:
                raise KeyError("aggregation group not found: {}".format(key))
            members = self._members(self._connection, key)
        expected = json.loads(str(row["expected_members_json"]))
        result_json = row["result_json"]
        return {
            "group_id": key,
            "scene": str(row["scene"]),
            "group_key": str(row["group_key"]),
            "state": str(row["state"]),
            "expected_members": expected,
            "received_members": members,
            "missing_members": sorted(set(expected) - set(members)),
            "minimum_members": int(row["minimum_members"]),
            "timeout_ms": int(row["timeout_ms"]),
            "first_received_at_ms": int(row["first_received_at_ms"]),
            "deadline_at_ms": int(row["deadline_at_ms"]),
            "completion_reason": str(row["completion_reason"]),
            "result": json.loads(str(result_json)) if result_json else None,
            "last_error": str(row["last_error"]),
        }

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
        return {
            "path": str(self.path) if self.path is not None else ":memory:",
            "states": {
                "waiting": counts.get("waiting", 0),
                "inflight": counts.get("inflight", 0),
                "completed": counts.get("completed", 0),
            },
            "event_count": event_count,
        }

    def close(self) -> None:
        with self._lock:
            self._connection.close()
