"""用途：持久跟踪边缘临时决策到云端最终复核的完整生命周期。"""

import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from cloud_edge_framework.contracts import DecisionEnvelope, SemanticEvent, stable_id


REVIEW_STATES = {"queued", "inflight", "completed"}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decision_signature(decision: DecisionEnvelope) -> str:
    return _canonical(
        {
            "decision": decision.decision,
            "risk_level": decision.risk_level,
            "actions": [action.to_dict() for action in decision.actions],
        }
    )


def _summary(values: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            return ordered[low]
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 6),
        "p50": round(percentile(0.50), 6),
        "p95": round(percentile(0.95), 6),
        "max": round(max(ordered), 6),
    }


class ReviewLifecycleStore:
    """SQLite-backed review state machine; an in-memory database is used in unit runtimes."""

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
                CREATE TABLE IF NOT EXISTS review_lifecycle (
                    review_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL UNIQUE,
                    trace_id TEXT NOT NULL,
                    scene TEXT NOT NULL,
                    requested_route TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('queued','inflight','completed')),
                    requested_at_ms INTEGER NOT NULL,
                    queued_at_ms INTEGER NOT NULL,
                    started_at_ms INTEGER,
                    completed_at_ms INTEGER,
                    preliminary_latency_ms REAL NOT NULL,
                    eventual_completion_ms REAL,
                    evidence_level TEXT NOT NULL,
                    planned_request_bytes INTEGER NOT NULL,
                    routing_features_json TEXT NOT NULL DEFAULT '{}',
                    local_decision_json TEXT NOT NULL,
                    final_decision_json TEXT,
                    decision_changed INTEGER,
                    completion_mode TEXT NOT NULL DEFAULT '',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT ''
                )
                """
            )
            columns = {
                str(row["name"])
                for row in self._connection.execute(
                    "PRAGMA table_info(review_lifecycle)"
                ).fetchall()
            }
            if "routing_features_json" not in columns:
                self._connection.execute(
                    "ALTER TABLE review_lifecycle "
                    "ADD COLUMN routing_features_json TEXT NOT NULL DEFAULT '{}'"
                )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_review_state "
                "ON review_lifecycle(state, queued_at_ms)"
            )

    def queue(
        self,
        event: SemanticEvent,
        local: DecisionEnvelope,
        requested_route: str,
        evidence_level: str,
        requested_at_ms: int,
        preliminary_latency_ms: float,
        planned_request_bytes: int,
        routing_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        review_id = stable_id("review", event.event_id, local.decision_id)
        now_ms = int(time.time() * 1000)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO review_lifecycle(
                    review_id, event_id, trace_id, scene, requested_route, state,
                    requested_at_ms, queued_at_ms, preliminary_latency_ms,
                    evidence_level, planned_request_bytes, routing_features_json,
                    local_decision_json
                ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    event.event_id,
                    str(event.metadata.get("trace_id", "")),
                    event.scene,
                    str(requested_route),
                    int(requested_at_ms),
                    now_ms,
                    max(0.0, float(preliminary_latency_ms)),
                    str(evidence_level),
                    max(0, int(planned_request_bytes)),
                    _canonical(dict(routing_features or {})),
                    _canonical(local.to_dict()),
                ),
            )
        return review_id

    def start(self, event_ids: Sequence[str], mode: str) -> None:
        ids = [str(value) for value in event_ids]
        if not ids:
            return
        now_ms = int(time.time() * 1000)
        with self._lock, self._connection:
            self._connection.executemany(
                """
                UPDATE review_lifecycle
                SET state='inflight', started_at_ms=?, completion_mode=?,
                    attempts=attempts+1, last_error=''
                WHERE event_id=? AND state!='completed'
                """,
                [(now_ms, str(mode), event_id) for event_id in ids],
            )

    def retry(self, event_ids: Sequence[str], error: str) -> None:
        ids = [str(value) for value in event_ids]
        if not ids:
            return
        with self._lock, self._connection:
            self._connection.executemany(
                """
                UPDATE review_lifecycle
                SET state='queued', last_error=?
                WHERE event_id=? AND state!='completed'
                """,
                [(str(error)[:2000], event_id) for event_id in ids],
            )

    def complete(
        self,
        event_id: str,
        cloud: DecisionEnvelope,
        completion_mode: str,
        completed_at_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        finished_ms = int(completed_at_ms or int(time.time() * 1000))
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM review_lifecycle WHERE event_id=?", (str(event_id),)
            ).fetchone()
            if row is None:
                raise KeyError("review lifecycle not found for event: {}".format(event_id))
            local = DecisionEnvelope.from_dict(json.loads(str(row["local_decision_json"])))
            changed = _decision_signature(local) != _decision_signature(cloud)
            eventual_ms = max(0.0, float(finished_ms - int(row["requested_at_ms"])))
            self._connection.execute(
                """
                UPDATE review_lifecycle
                SET state='completed', completed_at_ms=?, eventual_completion_ms=?,
                    final_decision_json=?, decision_changed=?, completion_mode=?,
                    last_error=''
                WHERE event_id=?
                """,
                (
                    finished_ms,
                    eventual_ms,
                    _canonical(cloud.to_dict()),
                    1 if changed else 0,
                    str(completion_mode),
                    str(event_id),
                ),
            )
        return self.get(str(event_id))

    @staticmethod
    def _record(row: sqlite3.Row) -> Dict[str, Any]:
        result = {name: row[name] for name in row.keys()}
        result["decision_changed"] = (
            None if row["decision_changed"] is None else bool(row["decision_changed"])
        )
        result["routing_features"] = json.loads(
            str(row["routing_features_json"])
        )
        result.pop("routing_features_json", None)
        result["local_decision"] = json.loads(str(row["local_decision_json"]))
        result.pop("local_decision_json", None)
        final_json = row["final_decision_json"]
        result["final_decision"] = json.loads(str(final_json)) if final_json else None
        result.pop("final_decision_json", None)
        return result

    def get(self, review_or_event_id: str) -> Dict[str, Any]:
        key = str(review_or_event_id).strip()
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM review_lifecycle WHERE review_id=? OR event_id=?",
                (key, key),
            ).fetchone()
        if row is None:
            raise KeyError("review lifecycle not found: {}".format(key))
        return self._record(row)

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        if limit <= 0:
            raise ValueError("review limit must be positive")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM review_lifecycle ORDER BY queued_at_ms DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._record(row) for row in rows]

    def routing_dataset(self, limit: int = 10000) -> Dict[str, Any]:
        if limit <= 0:
            raise ValueError("routing dataset limit must be positive")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM review_lifecycle "
                "WHERE state='completed' AND routing_features_json!='{}' "
                "ORDER BY completed_at_ms DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        samples = []
        for row in rows:
            record = self._record(row)
            samples.append(
                {
                    "event_id": record["event_id"],
                    "scene": record["scene"],
                    "features": record["routing_features"],
                    "observed_route": record["requested_route"],
                    "outcome": {
                        "cloud_changed_decision": bool(record["decision_changed"]),
                        "eventual_completion_ms": record["eventual_completion_ms"],
                        "completion_mode": record["completion_mode"],
                    },
                }
            )
        return {
            "schema_version": 1,
            "sample_count": len(samples),
            "selection": "cloud-reviewed events only; use controlled probes to reduce selection bias",
            "target": "cloud_changed_decision is a defer proxy, not task ground truth",
            "samples": samples,
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM review_lifecycle ORDER BY queued_at_ms"
            ).fetchall()
        records = [self._record(row) for row in rows]
        states = {name: 0 for name in sorted(REVIEW_STATES)}
        for record in records:
            states[str(record["state"])] += 1
        completed = [item for item in records if item["state"] == "completed"]
        corrections = sum(bool(item["decision_changed"]) for item in completed)
        sync = [
            float(item["eventual_completion_ms"])
            for item in completed
            if item["completion_mode"] == "sync"
        ]
        asynchronous = [
            float(item["eventual_completion_ms"])
            for item in completed
            if item["completion_mode"] == "replay"
        ]
        preliminary = [float(item["preliminary_latency_ms"]) for item in records]
        return {
            "path": str(self.path) if self.path is not None else ":memory:",
            "states": states,
            "total": len(records),
            "completed": len(completed),
            "corrections": corrections,
            "cloud_correction_rate": round(corrections / len(completed), 6)
            if completed
            else 0.0,
            "latency_ms": {
                "edge_preliminary": _summary(preliminary),
                "synchronous_cloud_closed_loop": _summary(sync),
                "asynchronous_cloud_eventual": _summary(asynchronous),
            },
        }

    def close(self) -> None:
        with self._lock:
            self._connection.close()
