"""用途：持久跟踪边缘临时决策到云端最终复核的完整生命周期。"""

import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from cloud_edge_framework.contracts import DecisionEnvelope, SemanticEvent, stable_id
from cloud_edge_framework.reliability import (
    _configure_sqlite_connection,
    IdempotencyConflictError,
    source_submission_identity,
)


REVIEW_STATES = {"queued", "inflight", "completed"}
COMPLETION_STAGES = {
    "lightweight_final",
    "large_model_review",
    "large_model_correction",
    "partial_final",
    "local_only_timeout",
}
NON_AUTHORITATIVE_STAGES = {"partial_final", "local_only_timeout"}


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


def _legacy_completion_stage(
    final_decision_json: Any,
    decision_changed: Any,
    completion_mode: Any,
) -> str:
    """Infer the explicit v2 stage for a completed row written by v1."""
    mode = str(completion_mode or "")
    if mode == "partial_timeout":
        return "partial_final"
    if mode == "aggregation_timeout":
        return "local_only_timeout"
    try:
        final_decision = json.loads(str(final_decision_json))
    except (TypeError, ValueError, json.JSONDecodeError):
        final_decision = None
    metadata = (
        final_decision.get("metadata", {})
        if isinstance(final_decision, dict)
        else {}
    )
    review = metadata.get("cloud_llm_review") if isinstance(metadata, dict) else None
    if isinstance(review, dict) and bool(review):
        return (
            "large_model_correction"
            if bool(decision_changed)
            else "large_model_review"
        )
    return "lightweight_final"


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
        _configure_sqlite_connection(
            self._connection,
            enable_wal=self.path is not None,
            synchronous="NORMAL",
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
                cloud_received_at_ms INTEGER,
                completed_at_ms INTEGER,
                preliminary_latency_ms REAL NOT NULL,
                cloud_receipt_latency_ms REAL,
                eventual_completion_ms REAL,
                evidence_level TEXT NOT NULL,
                planned_request_bytes INTEGER NOT NULL,
                source_identity TEXT NOT NULL DEFAULT '',
                routing_features_json TEXT NOT NULL DEFAULT '{}',
                local_decision_json TEXT NOT NULL,
                final_decision_json TEXT,
                decision_changed INTEGER,
                completion_mode TEXT NOT NULL DEFAULT '',
                completion_stage TEXT NOT NULL DEFAULT '',
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT ''
            )
            """
        )
        # Re-read only after BEGIN IMMEDIATE succeeds. Another process may have
        # completed the same migration while this connection waited for the
        # database write lock.
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
        if "source_identity" not in columns:
            self._connection.execute(
                "ALTER TABLE review_lifecycle "
                "ADD COLUMN source_identity TEXT NOT NULL DEFAULT ''"
            )
        if "cloud_received_at_ms" not in columns:
            self._connection.execute(
                "ALTER TABLE review_lifecycle ADD COLUMN cloud_received_at_ms INTEGER"
            )
        if "cloud_receipt_latency_ms" not in columns:
            self._connection.execute(
                "ALTER TABLE review_lifecycle ADD COLUMN cloud_receipt_latency_ms REAL"
            )
        if "completion_stage" not in columns:
            self._connection.execute(
                "ALTER TABLE review_lifecycle "
                "ADD COLUMN completion_stage TEXT NOT NULL DEFAULT ''"
            )
        legacy_rows = self._connection.execute(
            "SELECT event_id, final_decision_json, decision_changed, completion_mode "
            "FROM review_lifecycle "
            "WHERE state='completed' AND completion_stage=''"
        ).fetchall()
        if legacy_rows:
            self._connection.executemany(
                "UPDATE review_lifecycle SET completion_stage=? "
                "WHERE event_id=? AND completion_stage=''",
                [
                    (
                        _legacy_completion_stage(
                            row["final_decision_json"],
                            row["decision_changed"],
                            row["completion_mode"],
                        ),
                        str(row["event_id"]),
                    )
                    for row in legacy_rows
                ],
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
        source_identity = source_submission_identity(event.metadata)
        now_ms = int(time.time() * 1000)
        # These structures can be several kilobytes. Serialize before entering
        # the shared SQLite critical section so concurrent partitions do not
        # block on each other's pure JSON work.
        routing_features_json = _canonical(dict(routing_features or {}))
        local_decision_json = _canonical(local.to_dict())
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT review_id, source_identity FROM review_lifecycle "
                "WHERE event_id=?",
                (event.event_id,),
            ).fetchone()
            if existing is not None:
                existing_identity = str(existing["source_identity"])
                if (
                    existing_identity
                    and source_identity
                    and existing_identity != source_identity
                ):
                    raise IdempotencyConflictError(
                        "review event_id was already used with different "
                        "business control context"
                    )
                if not existing_identity and source_identity:
                    self._connection.execute(
                        "UPDATE review_lifecycle SET source_identity=? "
                        "WHERE event_id=? AND source_identity=''",
                        (source_identity, event.event_id),
                    )
                return str(existing["review_id"])
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO review_lifecycle(
                    review_id, event_id, trace_id, scene, requested_route, state,
                    requested_at_ms, queued_at_ms, preliminary_latency_ms,
                    evidence_level, planned_request_bytes, source_identity,
                    routing_features_json, local_decision_json
                ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)
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
                    source_identity,
                    routing_features_json,
                    local_decision_json,
                ),
            )
            if cursor.rowcount == 1:
                return review_id
            # Another store/process may have inserted the same event between
            # the pre-check and INSERT OR IGNORE.  Apply the same conflict
            # contract to the winning row instead of returning a nonexistent
            # locally-derived review_id.
            existing = self._connection.execute(
                "SELECT review_id, source_identity FROM review_lifecycle "
                "WHERE event_id=?",
                (event.event_id,),
            ).fetchone()
            if existing is None:
                raise RuntimeError("review lifecycle insert was not persisted")
            existing_identity = str(existing["source_identity"])
            if (
                existing_identity
                and source_identity
                and existing_identity != source_identity
            ):
                raise IdempotencyConflictError(
                    "review event_id was concurrently used with different "
                    "business control context"
                )
            if not existing_identity and source_identity:
                self._connection.execute(
                    "UPDATE review_lifecycle SET source_identity=? "
                    "WHERE event_id=? AND source_identity=''",
                    (source_identity, event.event_id),
                )
            return str(existing["review_id"])

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

    def waiting(self, event_ids: Sequence[str], reason: str = "") -> None:
        """Return received aggregation items to queued without recording failure."""
        ids = [str(value) for value in event_ids]
        if not ids:
            return
        del reason
        with self._lock, self._connection:
            self._connection.executemany(
                """
                UPDATE review_lifecycle
                SET state='queued', last_error=''
                WHERE event_id=? AND state!='completed'
                """,
                [(event_id,) for event_id in ids],
            )

    def received(
        self,
        event_ids: Sequence[str],
        received_at_ms: Optional[int] = None,
    ) -> None:
        """Record the first time the cloud accepted each durable submission."""
        ids = [str(value) for value in event_ids]
        if not ids:
            return
        observed_ms = int(
            received_at_ms
            if received_at_ms is not None
            else int(time.time() * 1000)
        )
        with self._lock, self._connection:
            self._connection.executemany(
                """
                UPDATE review_lifecycle
                SET cloud_received_at_ms=COALESCE(cloud_received_at_ms, ?),
                    cloud_receipt_latency_ms=COALESCE(
                        cloud_receipt_latency_ms,
                        MAX(0.0, CAST(? - requested_at_ms AS REAL))
                    )
                WHERE event_id=?
                """,
                [(observed_ms, observed_ms, event_id) for event_id in ids],
            )

    @staticmethod
    def _completion_stage(
        cloud: DecisionEnvelope,
        requested_stage: Optional[str],
        decision_changed: bool,
    ) -> str:
        if requested_stage is not None:
            stage = str(requested_stage).strip()
            if stage not in COMPLETION_STAGES:
                raise ValueError(
                    "completion_stage must be one of {}".format(
                        ", ".join(sorted(COMPLETION_STAGES))
                    )
                )
            return stage
        review = cloud.metadata.get("cloud_llm_review")
        if isinstance(review, dict) and bool(review):
            return (
                "large_model_correction"
                if decision_changed
                else "large_model_review"
            )
        return "lightweight_final"

    def complete(
        self,
        event_id: str,
        cloud: DecisionEnvelope,
        completion_mode: str,
        completed_at_ms: Optional[int] = None,
        completion_stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        finished_ms = int(
            completed_at_ms
            if completed_at_ms is not None
            else int(time.time() * 1000)
        )
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM review_lifecycle WHERE event_id=?", (str(event_id),)
            ).fetchone()
            if row is None:
                raise KeyError("review lifecycle not found for event: {}".format(event_id))
            local = DecisionEnvelope.from_dict(json.loads(str(row["local_decision_json"])))
            changed = _decision_signature(local) != _decision_signature(cloud)
            stage = self._completion_stage(cloud, completion_stage, changed)
            eventual_ms = max(0.0, float(finished_ms - int(row["requested_at_ms"])))
            self._connection.execute(
                """
                UPDATE review_lifecycle
                SET state='completed', completed_at_ms=?, eventual_completion_ms=?,
                    final_decision_json=?, decision_changed=?, completion_mode=?,
                    completion_stage=?, last_error=''
                WHERE event_id=?
                """,
                (
                    finished_ms,
                    eventual_ms,
                    _canonical(cloud.to_dict()),
                    1 if changed else 0,
                    str(completion_mode),
                    stage,
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
                "AND completion_stage NOT IN ('partial_final','local_only_timeout') "
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
                        "cloud_receipt_latency_ms": record[
                            "cloud_receipt_latency_ms"
                        ],
                        "eventual_completion_ms": record["eventual_completion_ms"],
                        "completion_mode": record["completion_mode"],
                        "completion_stage": record["completion_stage"],
                    },
                }
            )
        return {
            "schema_version": 2,
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
        authoritative = [
            item
            for item in completed
            if str(item["completion_stage"] or "lightweight_final")
            not in NON_AUTHORITATIVE_STAGES
        ]
        non_authoritative = [
            item
            for item in completed
            if str(item["completion_stage"] or "lightweight_final")
            in NON_AUTHORITATIVE_STAGES
        ]
        corrections = sum(bool(item["decision_changed"]) for item in authoritative)
        sync = [
            float(item["eventual_completion_ms"])
            for item in completed
            if item["completion_mode"] == "sync"
        ]
        asynchronous = [
            float(item["eventual_completion_ms"])
            for item in completed
            if item["completion_mode"] in {"replay", "reconciliation"}
        ]
        preliminary = [float(item["preliminary_latency_ms"]) for item in records]
        receipt = [
            float(item["cloud_receipt_latency_ms"])
            for item in records
            if item["cloud_receipt_latency_ms"] is not None
        ]
        lightweight = [
            float(item["eventual_completion_ms"])
            for item in completed
            if item["completion_stage"] in {"", "lightweight_final"}
        ]
        large_model_reviews = [
            float(item["eventual_completion_ms"])
            for item in completed
            if item["completion_stage"] == "large_model_review"
        ]
        large_model_corrections = [
            float(item["eventual_completion_ms"])
            for item in completed
            if item["completion_stage"] == "large_model_correction"
        ]
        partial_finals = [
            float(item["eventual_completion_ms"])
            for item in completed
            if item["completion_stage"] == "partial_final"
        ]
        local_only_timeouts = [
            float(item["eventual_completion_ms"])
            for item in completed
            if item["completion_stage"] == "local_only_timeout"
        ]
        stage_counts = {name: 0 for name in sorted(COMPLETION_STAGES)}
        for item in completed:
            stage = str(item["completion_stage"] or "lightweight_final")
            if stage in stage_counts:
                stage_counts[stage] += 1
        edge_provisional = _summary(preliminary)
        cloud_receipt = _summary(receipt)
        lightweight_final = _summary(lightweight)
        large_model_review = _summary(large_model_reviews)
        large_model_correction = _summary(large_model_corrections)
        return {
            "path": str(self.path) if self.path is not None else ":memory:",
            "states": states,
            "total": len(records),
            # `completed` remains as the v1 alias for existing dashboards.
            "completed": len(completed),
            "terminal_completed": len(completed),
            "authoritative_completed": len(authoritative),
            "non_authoritative_completed": len(non_authoritative),
            "completion_count_semantics": {
                "version": 2,
                "completed_alias": "terminal_completed",
                "terminal_completed": "all records whose lifecycle state is completed",
                "authoritative_completed": (
                    "terminal records excluding partial_final and local_only_timeout"
                ),
                "non_authoritative_completed": (
                    "terminal records at partial_final or local_only_timeout"
                ),
                "cloud_correction_rate_denominator": "authoritative_completed",
            },
            "latency_measurement_semantics": {
                "version": 2,
                "edge_provisional": (
                    "edge-local measured work; independent of cloud clock"
                ),
                "cloud_receipt": (
                    "cloud first durable-acceptance wall time minus edge request "
                    "wall time; valid only when edge and cloud clocks are synchronized"
                ),
                "cloud_receipt_clock_requirement": (
                    "record NTP/chrony offset and exclude runs whose absolute "
                    "cross-host offset exceeds the experiment tolerance"
                ),
                "closed_loop": (
                    "edge-observed completion from the original request time"
                ),
            },
            "completion_stages": stage_counts,
            "corrections": corrections,
            "cloud_correction_rate": round(corrections / len(authoritative), 6)
            if authoritative
            else 0.0,
            "latency_ms": {
                "edge_provisional": edge_provisional,
                "cloud_receipt": cloud_receipt,
                "lightweight_final": lightweight_final,
                "large_model_review": large_model_review,
                "large_model_correction": large_model_correction,
                "partial_final": _summary(partial_finals),
                "local_only_timeout": _summary(local_only_timeouts),
                # v1 aliases retained for existing reports and dashboards.
                "edge_preliminary": edge_provisional,
                "synchronous_cloud_closed_loop": _summary(sync),
                "asynchronous_cloud_eventual": _summary(asynchronous),
            },
        }

    def close(self) -> None:
        with self._lock:
            self._connection.close()
