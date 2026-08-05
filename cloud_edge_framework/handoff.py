"""Durable, low-latency handoff from request threads to an Outbox.

The handoff journal is intentionally much smaller than the Outbox database.  A
request only waits for one append plus ``fsync`` (shared by a batch of request
threads); a separate worker performs the potentially slow Outbox append.  Each
journal record is self-checking so an interrupted final write can be discarded
without mistaking damaged durable data for a valid submission.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional

from cloud_edge_framework.contracts import SemanticEvent
from cloud_edge_framework.reliability import (
    IdempotencyConflictError,
    source_submission_identity,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _checksum(body: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def _record(body: Dict[str, Any]) -> bytes:
    envelope = {"body": body, "checksum": _checksum(body)}
    return (_canonical_json(envelope) + "\n").encode("utf-8")


def _event_payload(event: SemanticEvent) -> Dict[str, Any]:
    include_scene_payload = bool(
        event.metadata.get("transport_include_scene_payload", False)
    )
    return event.to_dict(include_scene_payload=include_scene_payload)


def _event_identity(payload: Dict[str, Any]) -> str:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        source_identity = source_submission_identity(metadata)
        if source_identity:
            return source_identity
    serialized = _canonical_json(payload)
    return "serialized_payload:" + hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


@dataclass
class _PendingEntry:
    event: SemanticEvent
    identity: str
    retries: int = 0
    next_attempt_at: float = 0.0


class _SubmitTicket:
    def __init__(self, event: SemanticEvent, payload: Dict[str, Any], identity: str) -> None:
        self.event = event
        self.payload = payload
        self.identity = identity
        self.done = threading.Event()


class DurableOutboxHandoff:
    """Crash-safe asynchronous adapter in front of an Outbox.

    ``submit`` returning ``True`` means the put record has survived ``fsync``.
    It does *not* mean that ``outbox.append`` has completed.  A duplicate of a
    pending or already-Outbox-durable event returns ``False`` when its immutable
    submission identity matches and raises :class:`IdempotencyConflictError`
    otherwise.
    """

    def __init__(
        self,
        outbox: Any,
        journal_path: Optional[Path] = None,
        persisted_callback: Optional[Callable[[SemanticEvent, bool], None]] = None,
        max_batch_size: int = 64,
        retry_backoff_seconds: float = 0.01,
        max_retry_backoff_seconds: float = 1.0,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if retry_backoff_seconds <= 0:
            raise ValueError("retry_backoff_seconds must be positive")
        if max_retry_backoff_seconds < retry_backoff_seconds:
            raise ValueError(
                "max_retry_backoff_seconds must be >= retry_backoff_seconds"
            )
        if not hasattr(outbox, "append") or not hasattr(
            outbox, "submission_identity"
        ):
            raise TypeError(
                "outbox must provide append(event) and submission_identity(event_id)"
            )

        self.outbox = outbox
        self.persisted_callback = persisted_callback
        self.max_batch_size = int(max_batch_size)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.max_retry_backoff_seconds = float(max_retry_backoff_seconds)
        self.path = self._resolve_path(outbox, journal_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._condition = threading.Condition(threading.RLock())
        self._journal_lock = threading.Lock()
        self._write_queue: Deque[_SubmitTicket] = deque()
        self._pending: Dict[str, _PendingEntry] = {}
        # Includes both journal-writer reservations and durable pending puts.
        self._identities: Dict[str, str] = {}
        self._retry_count = 0
        self._last_error: Optional[str] = None
        self._recovered = 0
        self._closing = False
        self._stop_requested = False
        self._journal_owner_descriptor: Optional[int] = None
        self._workers_exited = 0
        self._workers_stopped = threading.Event()

        self._ensure_journal_file()
        self._acquire_journal_ownership()
        try:
            recovered = self._recover()
        except BaseException:
            self._release_journal_ownership()
            raise
        with self._condition:
            self._pending.update(recovered)
            self._identities.update(
                {event_id: entry.identity for event_id, entry in recovered.items()}
            )
            self._recovered = len(recovered)

        self._writer = threading.Thread(
            target=self._writer_worker,
            name="durable-outbox-journal",
            daemon=True,
        )
        self._delivery = threading.Thread(
            target=self._delivery_worker,
            name="durable-outbox-delivery",
            daemon=True,
        )
        self._writer.start()
        self._delivery.start()

    @staticmethod
    def _resolve_path(outbox: Any, journal_path: Optional[Path]) -> Path:
        if journal_path is not None:
            return Path(journal_path).resolve()
        outbox_path = getattr(outbox, "path", None)
        if outbox_path is None:
            raise ValueError("journal_path is required when outbox has no path")
        resolved = Path(outbox_path).resolve()
        return resolved.with_name(resolved.name + ".handoff.jsonl")

    def _ensure_journal_file(self) -> None:
        created = not self.path.exists()
        descriptor = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if created:
            # Persist the new directory entry as well as the empty file.
            try:
                directory = os.open(str(self.path.parent), os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except OSError:
                # Some filesystems do not allow fsync on a directory.  Every
                # put still receives a file fsync; this is only an extra guard
                # for first creation on filesystems that support it.
                pass

    def _acquire_journal_ownership(self) -> None:
        """Hold one process-wide owner lock for the journal's full lifetime."""
        descriptor = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(descriptor)
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise RuntimeError(
                    "durable Outbox handoff journal is already owned by another instance"
                ) from exc
            raise
        self._journal_owner_descriptor = descriptor

    def _release_journal_ownership(self) -> None:
        descriptor = self._journal_owner_descriptor
        if descriptor is None:
            return
        self._journal_owner_descriptor = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _recover(self) -> Dict[str, _PendingEntry]:
        with self._journal_lock:
            raw = self.path.read_bytes()
            complete_length = raw.rfind(b"\n") + 1
            complete = raw[:complete_length]
            if complete_length != len(raw):
                self._truncate_locked(complete_length)

            pending: Dict[str, _PendingEntry] = {}
            for line_number, raw_line in enumerate(complete.splitlines(), start=1):
                if not raw_line:
                    raise ValueError(
                        "handoff journal contains an empty record at line {}".format(
                            line_number
                        )
                    )
                body = self._validated_body(raw_line, line_number)
                operation = body.get("op")
                event_id = str(body.get("event_id", ""))
                identity = str(body.get("identity", ""))
                if not event_id or not identity:
                    raise ValueError(
                        "handoff journal record {} has no identity".format(line_number)
                    )
                if operation == "put":
                    payload = body.get("event")
                    if not isinstance(payload, dict):
                        raise ValueError(
                            "handoff journal put {} has no event".format(line_number)
                        )
                    event = SemanticEvent.from_dict(payload)
                    if event.event_id != event_id or _event_identity(payload) != identity:
                        raise ValueError(
                            "handoff journal put {} identity is inconsistent".format(
                                line_number
                            )
                        )
                    existing = pending.get(event_id)
                    if existing is not None and existing.identity != identity:
                        raise IdempotencyConflictError(
                            "handoff journal contains conflicting pending submissions"
                        )
                    if existing is None:
                        pending[event_id] = _PendingEntry(event, identity)
                elif operation == "ack":
                    existing = pending.get(event_id)
                    if existing is not None:
                        if existing.identity != identity:
                            raise ValueError(
                                "handoff journal ack {} has the wrong identity".format(
                                    line_number
                                )
                            )
                        del pending[event_id]
                else:
                    raise ValueError(
                        "handoff journal record {} has unknown operation".format(
                            line_number
                        )
                    )

            if not pending and complete:
                self._truncate_locked(0)
            return pending

    @staticmethod
    def _validated_body(raw_line: bytes, line_number: int) -> Dict[str, Any]:
        try:
            decoded = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "handoff journal record {} is not valid JSON".format(line_number)
            ) from exc
        if not isinstance(decoded, dict):
            raise ValueError(
                "handoff journal record {} must be an object".format(line_number)
            )
        body = decoded.get("body")
        checksum = decoded.get("checksum")
        if not isinstance(body, dict) or not isinstance(checksum, str):
            raise ValueError(
                "handoff journal record {} has an invalid envelope".format(line_number)
            )
        if not hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest() == checksum:
            raise ValueError(
                "handoff journal checksum mismatch at line {}".format(line_number)
            )
        return body

    def submit(self, event: SemanticEvent, timeout_seconds: float = 10.0) -> bool:
        """Durably enqueue ``event`` without waiting for the Outbox append."""
        if not isinstance(event, SemanticEvent):
            raise TypeError("event must be a SemanticEvent")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        payload = _event_payload(event)
        identity = _event_identity(payload)
        # Rebuild from the serialized form so later caller mutation cannot alter
        # what recovery or the background Outbox observes.
        durable_event = SemanticEvent.from_dict(payload)
        ticket = _SubmitTicket(durable_event, payload, identity)

        with self._condition:
            if self._closing:
                raise RuntimeError("durable Outbox handoff is closed")
            existing_identity = self._identities.get(event.event_id)
            if existing_identity is not None:
                if existing_identity == identity:
                    return False
                raise IdempotencyConflictError(
                    "handoff event_id was already used for a different cloud submission"
                )
            # Reserve before the independent Outbox lookup so concurrent request
            # threads cannot both pass a negative historical lookup.
            self._identities[event.event_id] = identity

        try:
            persisted_identity = self.outbox.submission_identity(event.event_id)
        except BaseException:
            with self._condition:
                if (
                    self._identities.get(event.event_id) == identity
                    and event.event_id not in self._pending
                ):
                    self._identities.pop(event.event_id, None)
            raise

        if persisted_identity is not None:
            with self._condition:
                if (
                    self._identities.get(event.event_id) == identity
                    and event.event_id not in self._pending
                ):
                    self._identities.pop(event.event_id, None)
            if persisted_identity == identity:
                return False
            raise IdempotencyConflictError(
                "handoff event_id was already used for a different cloud submission"
            )

        with self._condition:
            if self._closing:
                if (
                    self._identities.get(event.event_id) == identity
                    and event.event_id not in self._pending
                ):
                    self._identities.pop(event.event_id, None)
                raise RuntimeError("durable Outbox handoff is closed")
            self._write_queue.append(ticket)
            self._condition.notify_all()

        if not ticket.done.wait(timeout=float(timeout_seconds)):
            raise TimeoutError("timed out waiting for durable handoff journal fsync")
        return True

    def _writer_loop(self) -> None:
        while True:
            with self._condition:
                while not self._write_queue and not self._stop_requested:
                    if self._closing:
                        return
                    self._condition.wait()
                if self._stop_requested:
                    return
                batch: List[_SubmitTicket] = []
                while self._write_queue and len(batch) < self.max_batch_size:
                    batch.append(self._write_queue.popleft())

            attempt = 0
            while True:
                try:
                    records = [
                        _record(
                            {
                                "event": ticket.payload,
                                "event_id": ticket.event.event_id,
                                "identity": ticket.identity,
                                "op": "put",
                                "version": 1,
                            }
                        )
                        for ticket in batch
                    ]
                    self._append_records(records)
                    break
                except Exception as exc:  # noqa: BLE001
                    attempt += 1
                    self._record_failure(exc)
                    delay = self._retry_delay(attempt)
                    with self._condition:
                        if self._stop_requested:
                            return
                        self._condition.wait(timeout=delay)

            with self._condition:
                for ticket in batch:
                    self._pending[ticket.event.event_id] = _PendingEntry(
                        ticket.event,
                        ticket.identity,
                    )
                    ticket.done.set()
                self._last_error = None
                self._condition.notify_all()

    def _writer_worker(self) -> None:
        try:
            self._writer_loop()
        finally:
            self._worker_exited()

    def _delivery_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stop_requested:
                    if self._closing and not self._write_queue:
                        return
                    self._condition.wait()
                if self._stop_requested:
                    return

                now = time.monotonic()
                event_id, entry = min(
                    self._pending.items(),
                    key=lambda item: item[1].next_attempt_at,
                )
                delay = entry.next_attempt_at - now
                if delay > 0:
                    self._condition.wait(timeout=delay)
                    continue

            try:
                inserted = bool(self.outbox.append(entry.event))
                self._append_records(
                    [
                        _record(
                            {
                                "event_id": event_id,
                                "identity": entry.identity,
                                "op": "ack",
                                "version": 1,
                            }
                        )
                    ]
                )
            except Exception as exc:  # noqa: BLE001
                with self._condition:
                    current = self._pending.get(event_id)
                    if current is entry:
                        entry.retries += 1
                        entry.next_attempt_at = (
                            time.monotonic() + self._retry_delay(entry.retries)
                        )
                    self._retry_count += 1
                    self._last_error = "{}: {}".format(type(exc).__name__, exc)
                    self._condition.notify_all()
                continue

            with self._condition:
                current = self._pending.get(event_id)
                if current is entry:
                    del self._pending[event_id]
                    self._identities.pop(event_id, None)
                self._last_error = None
                self._condition.notify_all()

            self._maybe_compact()
            if self.persisted_callback is not None:
                try:
                    self.persisted_callback(entry.event, inserted)
                except Exception as exc:  # noqa: BLE001
                    # The Outbox and ACK are already durable.  A notification
                    # callback cannot roll them back or make this item pending
                    # again; expose its failure for operators instead.
                    with self._condition:
                        self._last_error = "{}: {}".format(
                            type(exc).__name__, exc
                        )

    def _delivery_worker(self) -> None:
        try:
            self._delivery_loop()
        finally:
            self._worker_exited()

    def _worker_exited(self) -> None:
        release_owner = False
        workers_stopped = False
        with self._condition:
            self._workers_exited += 1
            if self._workers_exited >= 2:
                workers_stopped = True
                release_owner = (
                    self._closing and self._journal_owner_descriptor is not None
                )
        if release_owner:
            # Both loops have finished every possible journal access.  This path
            # also releases ownership when close's drain deadline expires just
            # before a worker observes the stop notification.
            self._maybe_compact()
            self._release_journal_ownership()
        if workers_stopped:
            # A waiter that observes completion may immediately construct the
            # next owner, so publish it only after ownership is released.
            self._workers_stopped.set()

    def _retry_delay(self, failures: int) -> float:
        exponent = max(0, min(int(failures) - 1, 30))
        return min(
            self.max_retry_backoff_seconds,
            self.retry_backoff_seconds * (2 ** exponent),
        )

    def _record_failure(self, exc: Exception) -> None:
        with self._condition:
            self._retry_count += 1
            self._last_error = "{}: {}".format(type(exc).__name__, exc)

    def _append_records(self, records: Iterable[bytes]) -> None:
        data = b"".join(records)
        if not data:
            return
        with self._journal_lock:
            descriptor = os.open(
                str(self.path),
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            start = os.lseek(descriptor, 0, os.SEEK_END)
            try:
                view = memoryview(data)
                written = 0
                while written < len(view):
                    count = os.write(descriptor, view[written:])
                    if count <= 0:
                        raise OSError("journal write made no progress")
                    written += count
                os.fsync(descriptor)
            except BaseException:
                # Never leave a failed write followed by a valid record: that
                # would turn a recoverable incomplete tail into mid-file damage.
                try:
                    os.ftruncate(descriptor, start)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                raise
            else:
                os.close(descriptor)

    def _truncate_locked(self, length: int) -> None:
        descriptor = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.ftruncate(descriptor, max(0, int(length)))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _maybe_compact(self) -> None:
        with self._condition:
            if self._pending or self._identities or self._write_queue:
                return
        with self._journal_lock:
            # A submit can reserve a new identity while this method waits for
            # the journal lock, and its writer may even append first.  Recheck
            # reservations as well as durable pending entries so that such a
            # freshly fsynced put can never be truncated by stale emptiness.
            with self._condition:
                if self._pending or self._identities or self._write_queue:
                    return
            self._truncate_locked(0)

    def snapshot(self) -> Dict[str, Any]:
        with self._condition:
            pending = len(self._identities)
            return {
                "pending": pending,
                "retries": self._retry_count,
                "last_error": self._last_error,
                "recovered": self._recovered,
                "path": str(self.path),
                "durable_pending_count": len(self._pending),
                "retry_count": self._retry_count,
            }

    def close(self, timeout_seconds: float = 10.0) -> None:
        """Best-effort drain; uncompleted durable puts remain in the journal."""
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative")
        deadline = time.monotonic() + float(timeout_seconds)
        with self._condition:
            if self._journal_owner_descriptor is None:
                return
            if not self._closing:
                self._closing = True
            self._condition.notify_all()

        for thread in (self._writer, self._delivery):
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)

        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
        for thread in (self._writer, self._delivery):
            if thread.is_alive():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))

        # Stop notifications wake retry waits immediately, but a zero remaining
        # drain budget can otherwise return before the scheduled worker gets one
        # CPU slice to release the lifetime journal lock.  This is a shutdown
        # grace period, not additional delivery time; a worker blocked in an
        # external Outbox remains the owner until it actually exits.
        self._workers_stopped.wait(timeout=0.05)

        if not self._writer.is_alive() and not self._delivery.is_alive():
            self._maybe_compact()
            self._release_journal_ownership()
