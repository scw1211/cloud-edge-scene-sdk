"""用途：校验场景 JSON 文件，并通过持久化 Outbox 可靠提交到边缘框架。"""

import argparse
import ctypes
import hashlib
import json
import os
import select
import shutil
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError

from cloud_edge_framework.contracts import ContractError
from cloud_edge_framework.event_envelope import SceneEventEnvelope


class BridgeValidationError(ValueError):
    """Raised when an input file cannot enter the durable delivery queue."""


class DeliveryError(RuntimeError):
    """HTTP delivery failure annotated with retryability and measured wall time."""

    def __init__(self, message: str, retryable: bool, elapsed_ms: float) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)
        self.elapsed_ms = float(elapsed_ms)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _elapsed_ms(started_ns: int) -> float:
    return round((time.perf_counter_ns() - started_ns) / 1_000_000.0, 6)


def _safe_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in str(value)
    ).strip("._")
    return (cleaned or "event")[:160]


def _load_json_object(path: Path, description: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeValidationError(
            "cannot read {} {}: {}".format(description, path, exc)
        ) from exc
    if not isinstance(value, dict):
        raise BridgeValidationError("{} must contain a JSON object".format(description))
    return value


def _first_schema_error(
    validator: Draft202012Validator,
    payload: Any,
    prefix: str,
) -> Optional[str]:
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if not errors:
        return None
    error = errors[0]
    location = ".".join(str(part) for part in error.absolute_path) or prefix
    return "{} {}: {}".format(prefix, location, error.message)


class LocalEventValidator:
    """Caches the envelope and scene payload validators for low-latency ingestion."""

    def __init__(
        self,
        envelope_schema_path: Path,
        schema_directories: Sequence[Path],
        verify_local_evidence: bool = False,
    ) -> None:
        envelope_schema = _load_json_object(envelope_schema_path, "envelope schema")
        try:
            Draft202012Validator.check_schema(envelope_schema)
        except SchemaError as exc:
            raise BridgeValidationError(
                "invalid envelope schema: {}".format(exc.message)
            ) from exc
        self.envelope_validator = Draft202012Validator(
            envelope_schema,
            format_checker=FormatChecker(),
        )
        self.payload_validators: Dict[str, Draft202012Validator] = {}
        self.payload_schema_paths: Dict[str, str] = {}
        self.verify_local_evidence = bool(verify_local_evidence)
        for directory in schema_directories:
            self._load_schema_directory(Path(directory))

    def _load_schema_directory(self, directory: Path) -> None:
        path = directory.resolve()
        if not path.is_dir():
            raise BridgeValidationError("schema directory does not exist: {}".format(path))
        for schema_path in sorted(path.rglob("*.json")):
            schema = _load_json_object(schema_path, "payload schema")
            schema_id = schema.get("$id")
            if not isinstance(schema_id, str) or not schema_id.strip():
                continue
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as exc:
                raise BridgeValidationError(
                    "invalid payload schema {}: {}".format(schema_path, exc.message)
                ) from exc
            normalized_id = schema_id.strip()
            previous = self.payload_schema_paths.get(normalized_id)
            if previous is not None:
                raise BridgeValidationError(
                    "duplicate payload schema id {} in {} and {}".format(
                        normalized_id,
                        previous,
                        schema_path,
                    )
                )
            self.payload_validators[normalized_id] = Draft202012Validator(
                schema,
                format_checker=FormatChecker(),
            )
            self.payload_schema_paths[normalized_id] = str(schema_path.resolve())

    @staticmethod
    def _file_uris(value: Any) -> Iterable[str]:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key) == "uri" and isinstance(child, str):
                    yield child
                yield from LocalEventValidator._file_uris(child)
        elif isinstance(value, list):
            for child in value:
                yield from LocalEventValidator._file_uris(child)

    def _validate_local_evidence(self, payload: Any) -> None:
        if not self.verify_local_evidence:
            return
        for uri in self._file_uris(payload):
            parsed = urlsplit(uri)
            if parsed.scheme != "file":
                continue
            evidence_path = Path(unquote(parsed.path))
            if not evidence_path.is_file():
                raise BridgeValidationError(
                    "local evidence file does not exist: {}".format(evidence_path)
                )

    def validate(self, value: Dict[str, Any]) -> Tuple[SceneEventEnvelope, Dict[str, float]]:
        envelope_started = time.perf_counter_ns()
        envelope_error = _first_schema_error(
            self.envelope_validator,
            value,
            "event envelope",
        )
        if envelope_error:
            raise BridgeValidationError(envelope_error)
        try:
            envelope = SceneEventEnvelope.from_dict(value)
        except ContractError as exc:
            raise BridgeValidationError(str(exc)) from exc
        envelope_ms = _elapsed_ms(envelope_started)

        payload_started = time.perf_counter_ns()
        payload_validator = self.payload_validators.get(envelope.dataschema)
        if payload_validator is None:
            available = ", ".join(sorted(self.payload_validators)) or "none"
            raise BridgeValidationError(
                "no local payload schema for {}; available schema ids: {}".format(
                    envelope.dataschema,
                    available,
                )
            )
        payload_error = _first_schema_error(
            payload_validator,
            envelope.payload_for_validation(),
            "scene payload",
        )
        if payload_error:
            raise BridgeValidationError(payload_error)
        self._validate_local_evidence(envelope.payload_for_validation())
        payload_ms = _elapsed_ms(payload_started)
        return envelope, {
            "envelope_validation_ms": envelope_ms,
            "payload_validation_ms": payload_ms,
            "validation_ms": round(envelope_ms + payload_ms, 6),
        }

    def describe(self) -> Dict[str, Any]:
        return {
            "payload_schema_count": len(self.payload_validators),
            "payload_schemas": dict(sorted(self.payload_schema_paths.items())),
            "verify_local_evidence": self.verify_local_evidence,
        }


@dataclass(frozen=True)
class OutboxItem:
    event_id: str
    envelope: Dict[str, Any]
    source_file: str
    attempts: int
    created_at_ms: int
    ingestion_timings: Dict[str, float]


class DurableEnvelopeOutbox:
    """SQLite at-least-once queue for validated external event envelopes."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = self._open_connection()
        self._initialize()

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path),
            timeout=10.0,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _connect(self) -> sqlite3.Connection:
        return self._connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS bridge_events (
                    event_id TEXT PRIMARY KEY,
                    body_sha256 TEXT NOT NULL,
                    envelope_json TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(
                        state IN ('pending','inflight','completed','failed')
                    ),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    available_at_ms INTEGER NOT NULL,
                    lease_until_ms INTEGER,
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    delivered_at_ms INTEGER,
                    last_error TEXT NOT NULL DEFAULT '',
                    ingestion_timings_json TEXT NOT NULL,
                    response_json TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_bridge_ready "
                "ON bridge_events(state, available_at_ms, created_at_ms)"
            )

    def enqueue(
        self,
        envelope: Dict[str, Any],
        source_file: str,
        ingestion_timings: Dict[str, float],
    ) -> str:
        event_id = str(envelope["id"])
        serialized = _canonical_json(envelope)
        body_sha256 = _sha256_text(serialized)
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT body_sha256, state FROM bridge_events WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if row is not None:
                if str(row["body_sha256"]) != body_sha256:
                    raise BridgeValidationError(
                        "event id {} was already used for different content".format(event_id)
                    )
                return "duplicate_{}".format(str(row["state"]))
            connection.execute(
                """
                INSERT INTO bridge_events(
                    event_id, body_sha256, envelope_json, source_file, state,
                    attempts, available_at_ms, created_at_ms, updated_at_ms,
                    ingestion_timings_json
                ) VALUES (?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    body_sha256,
                    serialized,
                    source_file,
                    now_ms,
                    now_ms,
                    now_ms,
                    _canonical_json(ingestion_timings),
                ),
            )
        return "enqueued"

    def claim(self, limit: int, lease_seconds: float) -> List[OutboxItem]:
        if limit <= 0 or lease_seconds <= 0:
            raise ValueError("claim limit and lease_seconds must be positive")
        now_ms = int(time.time() * 1000)
        lease_until_ms = now_ms + int(float(lease_seconds) * 1000)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE bridge_events
                SET state='pending', lease_until_ms=NULL, updated_at_ms=?
                WHERE state='inflight' AND lease_until_ms <= ?
                """,
                (now_ms, now_ms),
            )
            rows = connection.execute(
                """
                SELECT event_id, envelope_json, source_file, attempts,
                       created_at_ms, ingestion_timings_json
                FROM bridge_events
                WHERE state='pending' AND available_at_ms <= ?
                ORDER BY created_at_ms, event_id
                LIMIT ?
                """,
                (now_ms, int(limit)),
            ).fetchall()
            if rows:
                connection.executemany(
                    """
                    UPDATE bridge_events
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
            OutboxItem(
                event_id=str(row["event_id"]),
                envelope=dict(json.loads(str(row["envelope_json"]))),
                source_file=str(row["source_file"]),
                attempts=int(row["attempts"]) + 1,
                created_at_ms=int(row["created_at_ms"]),
                ingestion_timings={
                    str(key): float(value)
                    for key, value in dict(
                        json.loads(str(row["ingestion_timings_json"]))
                    ).items()
                },
            )
            for row in rows
        ]

    def acknowledge(self, event_id: str, response: Dict[str, Any]) -> None:
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE bridge_events
                SET state='completed', lease_until_ms=NULL, delivered_at_ms=?,
                    updated_at_ms=?, last_error='', response_json=?
                WHERE event_id=?
                """,
                (now_ms, now_ms, _canonical_json(response), str(event_id)),
            )

    def fail(self, event_id: str, error: str) -> None:
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE bridge_events
                SET state='failed', lease_until_ms=NULL, updated_at_ms=?, last_error=?
                WHERE event_id=?
                """,
                (now_ms, str(error)[:4000], str(event_id)),
            )

    def release(self, event_id: str, attempts: int, error: str, max_backoff: float) -> None:
        now_ms = int(time.time() * 1000)
        backoff_seconds = min(float(max_backoff), 2.0 ** min(max(0, attempts - 1), 16))
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE bridge_events
                SET state='pending', lease_until_ms=NULL, available_at_ms=?,
                    updated_at_ms=?, last_error=?
                WHERE event_id=?
                """,
                (
                    now_ms + int(backoff_seconds * 1000),
                    now_ms,
                    str(error)[:4000],
                    str(event_id),
                ),
            )

    def snapshot(self) -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            states = {
                str(row["state"]): int(row["count"])
                for row in connection.execute(
                    "SELECT state, COUNT(*) AS count FROM bridge_events GROUP BY state"
                ).fetchall()
            }
            attempts = connection.execute(
                "SELECT COALESCE(SUM(attempts), 0) AS value FROM bridge_events"
            ).fetchone()["value"]
        return {
            "path": str(self.path),
            "states": {
                name: states.get(name, 0)
                for name in ("pending", "inflight", "completed", "failed")
            },
            "delivery_attempts": int(attempts),
        }

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class EdgeEventSender:
    def __init__(
        self,
        edge_base_url: str,
        timeout_seconds: float,
        conflict_suspected: bool = False,
        model_disagreement: bool = False,
    ) -> None:
        self.url = (
            edge_base_url.rstrip("/") + "/api/v1/collaboration/decide"
        )
        self.timeout_seconds = float(timeout_seconds)
        self.conflict_suspected = bool(conflict_suspected)
        self.model_disagreement = bool(model_disagreement)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def send(self, envelope: Dict[str, Any]) -> Tuple[Dict[str, Any], float, int, int]:
        payload = {
            "event": envelope,
            "conflict_suspected": self.conflict_suspected,
            "model_disagreement": self.model_disagreement,
        }
        body = _canonical_json(payload).encode("utf-8")
        event_id = str(envelope["id"])
        request_id = "bridge-{}".format(_sha256_text(event_id)[:32])
        request = Request(
            self.url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Idempotency-Key": request_id,
                "X-Request-ID": request_id,
            },
            method="POST",
        )
        started_ns = time.perf_counter_ns()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read()
        except HTTPError as exc:
            elapsed_ms = _elapsed_ms(started_ns)
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise DeliveryError(
                "edge returned HTTP {}: {}".format(exc.code, detail),
                retryable=exc.code >= 500 or exc.code in {408, 429},
                elapsed_ms=elapsed_ms,
            ) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise DeliveryError(
                "edge request failed: {}".format(exc),
                retryable=True,
                elapsed_ms=_elapsed_ms(started_ns),
            ) from exc
        elapsed_ms = _elapsed_ms(started_ns)
        try:
            result = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DeliveryError(
                "edge returned invalid JSON",
                retryable=True,
                elapsed_ms=elapsed_ms,
            ) from exc
        if not isinstance(result, dict):
            raise DeliveryError(
                "edge response must be an object",
                retryable=True,
                elapsed_ms=elapsed_ms,
            )
        return result, elapsed_ms, len(body), len(response_body)


class FileEventBridge:
    def __init__(
        self,
        input_directory: Path,
        state_directory: Path,
        validator: LocalEventValidator,
        sender: EdgeEventSender,
        lease_seconds: float = 30.0,
        max_backoff_seconds: float = 60.0,
        batch_size: int = 32,
    ) -> None:
        self.input_directory = Path(input_directory).resolve()
        self.state_directory = Path(state_directory).resolve()
        self.accepted_directory = self.state_directory / "accepted"
        self.rejected_directory = self.state_directory / "rejected"
        self.receipt_directory = self.state_directory / "receipts"
        for directory in (
            self.input_directory,
            self.state_directory,
            self.accepted_directory,
            self.rejected_directory,
            self.receipt_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        if self.state_directory == self.input_directory:
            raise ValueError("state_directory must be different from input_directory")
        self.validator = validator
        self.sender = sender
        self.lease_seconds = float(lease_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self.batch_size = int(batch_size)
        if self.lease_seconds <= 0 or self.max_backoff_seconds <= 0:
            raise ValueError("lease and backoff values must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.outbox = DurableEnvelopeOutbox(self.state_directory / "outbox.sqlite3")
        self._recent_ingestion_timings: Dict[str, Dict[str, float]] = {}
        self.stats = {
            "files_seen": 0,
            "events_enqueued": 0,
            "duplicates": 0,
            "files_rejected": 0,
            "delivery_completed": 0,
            "delivery_retried": 0,
            "delivery_failed": 0,
        }

    @staticmethod
    def _eligible(path: Path) -> bool:
        return (
            path.is_file()
            and path.suffix.lower() == ".json"
            and not path.name.startswith(".")
            and not path.name.endswith(".error.json")
        )

    @staticmethod
    def _unique_destination(directory: Path, preferred_name: str) -> Path:
        candidate = directory / preferred_name
        if not candidate.exists():
            return candidate
        stem = candidate.stem
        suffix = candidate.suffix
        for index in range(1, 100000):
            alternate = directory / "{}-{}{}".format(stem, index, suffix)
            if not alternate.exists():
                return alternate
        raise RuntimeError("cannot allocate archive filename in {}".format(directory))

    def _archive(self, path: Path, directory: Path, prefix: str = "") -> Path:
        preferred_name = "{}{}".format(prefix, path.name)
        destination = self._unique_destination(directory, preferred_name)
        return Path(shutil.move(str(path), str(destination)))

    def _write_json(self, path: Path, value: Dict[str, Any]) -> None:
        temporary = path.with_name(".{}.tmp".format(path.name))
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(path))

    def _reject_file(
        self,
        path: Path,
        error: Exception,
        started_ns: int,
    ) -> Dict[str, Any]:
        archived = self._archive(path, self.rejected_directory)
        record = {
            "status": "rejected",
            "source_file": str(archived),
            "error_type": type(error).__name__,
            "error": str(error),
            "rejected_at_ms": int(time.time() * 1000),
            "bridge_processing_ms": _elapsed_ms(started_ns),
        }
        error_path = archived.with_name(archived.name + ".error.json")
        self._write_json(error_path, record)
        self.stats["files_rejected"] += 1
        return record

    def ingest_file(self, path: Path) -> Dict[str, Any]:
        source = Path(path).resolve()
        if not self._eligible(source):
            return {"status": "ignored", "source_file": str(source)}
        self.stats["files_seen"] += 1
        started_ns = time.perf_counter_ns()
        read_started_ns = time.perf_counter_ns()
        try:
            value = _load_json_object(source, "event file")
            read_ms = _elapsed_ms(read_started_ns)
            envelope, validation_timings = self.validator.validate(value)
            prefix = "{}__".format(_safe_name(envelope.event_id))
            archive_destination = self._unique_destination(
                self.accepted_directory,
                "{}{}".format(prefix, source.name),
            )
            enqueue_started_ns = time.perf_counter_ns()
            ingestion_timings = {
                "read_parse_ms": read_ms,
                **validation_timings,
            }
            queue_status = self.outbox.enqueue(
                envelope.to_dict(),
                str(archive_destination),
                ingestion_timings,
            )
            ingestion_timings["durable_enqueue_ms"] = _elapsed_ms(
                enqueue_started_ns
            )
            archived = Path(shutil.move(str(source), str(archive_destination)))
            ingestion_timings["ingestion_total_ms"] = _elapsed_ms(started_ns)
            if queue_status == "enqueued":
                self._recent_ingestion_timings[envelope.event_id] = dict(
                    ingestion_timings
                )
                self.stats["events_enqueued"] += 1
            else:
                self.stats["duplicates"] += 1
            return {
                "status": queue_status,
                "event_id": envelope.event_id,
                "source_file": str(archived),
                "timings": ingestion_timings,
            }
        except (BridgeValidationError, ContractError, ValueError) as exc:
            return self._reject_file(source, exc, started_ns)

    def scan(self) -> List[Dict[str, Any]]:
        return [
            self.ingest_file(path)
            for path in sorted(self.input_directory.iterdir())
            if self._eligible(path)
        ]

    def _receipt_path(self, event_id: str) -> Path:
        return self.receipt_directory / "{}.json".format(_safe_name(event_id))

    def deliver_ready(self) -> List[Dict[str, Any]]:
        outcomes = []
        for item in self.outbox.claim(self.batch_size, self.lease_seconds):
            ingestion_timings = self._recent_ingestion_timings.get(
                item.event_id,
                item.ingestion_timings,
            )
            try:
                response, http_ms, request_bytes, response_bytes = self.sender.send(
                    item.envelope
                )
            except DeliveryError as exc:
                outcome = {
                    "event_id": item.event_id,
                    "status": "retry_pending" if exc.retryable else "failed",
                    "attempt": item.attempts,
                    "error": str(exc),
                    "http_round_trip_ms": exc.elapsed_ms,
                    "timings": {
                        **ingestion_timings,
                        "http_round_trip_ms": exc.elapsed_ms,
                    },
                }
                if exc.retryable:
                    self.outbox.release(
                        item.event_id,
                        item.attempts,
                        str(exc),
                        self.max_backoff_seconds,
                    )
                    self.stats["delivery_retried"] += 1
                else:
                    self.outbox.fail(item.event_id, str(exc))
                    self.stats["delivery_failed"] += 1
                    self._recent_ingestion_timings.pop(item.event_id, None)
                    self._write_json(self._receipt_path(item.event_id), outcome)
                outcomes.append(outcome)
                continue

            self.outbox.acknowledge(item.event_id, response)
            completed_at_ms = int(time.time() * 1000)
            receipt = {
                "event_id": item.event_id,
                "status": "delivered",
                "attempt": item.attempts,
                "source_file": item.source_file,
                "completed_at_ms": completed_at_ms,
                "file_queue_to_ack_ms": max(0, completed_at_ms - item.created_at_ms),
                "timings": {
                    **ingestion_timings,
                    "http_round_trip_ms": http_ms,
                },
                "request_bytes": request_bytes,
                "response_bytes": response_bytes,
                "edge_response": response,
            }
            self._write_json(self._receipt_path(item.event_id), receipt)
            self._recent_ingestion_timings.pop(item.event_id, None)
            self.stats["delivery_completed"] += 1
            outcomes.append(receipt)
        return outcomes

    def snapshot(self) -> Dict[str, Any]:
        return {
            "input_directory": str(self.input_directory),
            "state_directory": str(self.state_directory),
            "validator": self.validator.describe(),
            "session": dict(self.stats),
            "outbox": self.outbox.snapshot(),
        }

    def close(self) -> None:
        self.outbox.close()


class InotifyDirectoryWatcher:
    """Minimal Linux inotify wrapper for close-write and atomic-rename events."""

    IN_CLOSE_WRITE = 0x00000008
    IN_MOVED_TO = 0x00000080
    IN_Q_OVERFLOW = 0x00004000
    WATCH_MASK = IN_CLOSE_WRITE | IN_MOVED_TO | IN_Q_OVERFLOW
    EVENT_HEADER = struct.Struct("iIII")

    def __init__(self, directory: Path) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        init = libc.inotify_init1
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        add_watch = libc.inotify_add_watch
        add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add_watch.restype = ctypes.c_int
        self.fd = int(init(os.O_NONBLOCK | os.O_CLOEXEC))
        if self.fd < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        encoded_path = os.fsencode(str(Path(directory).resolve()))
        self.watch_descriptor = int(add_watch(self.fd, encoded_path, self.WATCH_MASK))
        if self.watch_descriptor < 0:
            error_number = ctypes.get_errno()
            os.close(self.fd)
            raise OSError(error_number, os.strerror(error_number))

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def wait(self, timeout_seconds: float) -> Tuple[List[str], bool]:
        readable, _, _ = select.select([self.fd], [], [], max(0.0, timeout_seconds))
        if not readable:
            return [], False
        try:
            block = os.read(self.fd, 64 * 1024)
        except BlockingIOError:
            return [], False
        names: List[str] = []
        overflow = False
        offset = 0
        while offset + self.EVENT_HEADER.size <= len(block):
            _, mask, _, name_length = self.EVENT_HEADER.unpack_from(block, offset)
            offset += self.EVENT_HEADER.size
            name_bytes = block[offset : offset + name_length]
            offset += name_length
            if mask & self.IN_Q_OVERFLOW:
                overflow = True
            if mask & (self.IN_CLOSE_WRITE | self.IN_MOVED_TO):
                name = os.fsdecode(name_bytes.split(b"\0", 1)[0])
                if name:
                    names.append(name)
        return names, overflow

    def __enter__(self) -> "InotifyDirectoryWatcher":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _write_snapshot(path: Optional[Path], value: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp".format(path.name))
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _run_watch(
    bridge: FileEventBridge,
    retry_interval_seconds: float,
    rescan_interval_seconds: float,
    metrics_path: Optional[Path],
) -> None:
    bridge.scan()
    bridge.deliver_ready()
    _write_snapshot(metrics_path, bridge.snapshot())
    last_retry = time.monotonic()
    last_rescan = last_retry
    with InotifyDirectoryWatcher(bridge.input_directory) as watcher:
        while True:
            names, overflow = watcher.wait(min(1.0, retry_interval_seconds))
            for name in sorted(set(names)):
                bridge.ingest_file(bridge.input_directory / name)
            now = time.monotonic()
            if overflow or now - last_rescan >= rescan_interval_seconds:
                bridge.scan()
                last_rescan = now
            if names or now - last_retry >= retry_interval_seconds:
                bridge.deliver_ready()
                last_retry = now
                snapshot = bridge.snapshot()
                _write_snapshot(metrics_path, snapshot)
                print(json.dumps(snapshot, ensure_ascii=False), flush=True)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate local scene JSON files and reliably submit them to an edge service."
    )
    parser.add_argument("mode", choices=("once", "watch"))
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument(
        "--envelope-schema",
        default=str(project_root / "schemas" / "scene_event_envelope.schema.json"),
    )
    parser.add_argument(
        "--schema-dir",
        action="append",
        default=[],
        help="Directory containing plugin payload schemas; may be repeated.",
    )
    parser.add_argument("--edge-base-url", default="http://127.0.0.1:18101")
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--lease-seconds", type=float, default=30.0)
    parser.add_argument("--max-backoff-seconds", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retry-interval-seconds", type=float, default=1.0)
    parser.add_argument("--rescan-interval-seconds", type=float, default=30.0)
    parser.add_argument("--metrics-output", default="")
    parser.add_argument("--verify-local-evidence", action="store_true")
    parser.add_argument("--conflict-suspected", action="store_true")
    parser.add_argument("--model-disagreement", action="store_true")
    args = parser.parse_args(argv)
    if not args.schema_dir:
        args.schema_dir = [str(project_root / "schemas" / "scenes")]
    if args.retry_interval_seconds <= 0 or args.rescan_interval_seconds <= 0:
        parser.error("retry and rescan intervals must be positive")
    return args


def run(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = _parse_args(argv)
    validator = LocalEventValidator(
        Path(args.envelope_schema),
        [Path(value) for value in args.schema_dir],
        verify_local_evidence=args.verify_local_evidence,
    )
    bridge = FileEventBridge(
        input_directory=Path(args.input_dir),
        state_directory=Path(args.state_dir),
        validator=validator,
        sender=EdgeEventSender(
            args.edge_base_url,
            args.timeout_seconds,
            conflict_suspected=args.conflict_suspected,
            model_disagreement=args.model_disagreement,
        ),
        lease_seconds=args.lease_seconds,
        max_backoff_seconds=args.max_backoff_seconds,
        batch_size=args.batch_size,
    )
    metrics_path = Path(args.metrics_output).resolve() if args.metrics_output else None
    if args.mode == "watch":
        try:
            _run_watch(
                bridge,
                args.retry_interval_seconds,
                args.rescan_interval_seconds,
                metrics_path,
            )
        except KeyboardInterrupt:
            pass
    else:
        bridge.scan()
        bridge.deliver_ready()
    snapshot = bridge.snapshot()
    _write_snapshot(metrics_path, snapshot)
    bridge.close()
    return snapshot


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
