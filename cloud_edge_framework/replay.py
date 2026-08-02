"""用途：网络恢复后在后台自动领取、协调并确认边缘 Outbox 事件。"""

import threading
import time
from typing import Any, Dict, Optional

from cloud_edge_framework.metrics import FrameworkMetrics
from cloud_edge_framework.plugin_manager import PluginRuntimeManager
from cloud_edge_framework.reliability import SQLiteOutbox
from cloud_edge_framework.service_config import ReplayConfig


class OutboxReplayWorker:
    def __init__(
        self,
        manager: PluginRuntimeManager,
        outbox: SQLiteOutbox,
        network_monitor: Any,
        config: ReplayConfig,
        metrics: FrameworkMetrics,
    ) -> None:
        self.manager = manager
        self.outbox = outbox
        self.network_monitor = network_monitor
        self.config = config
        self.metrics = metrics
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._last_delivery_result: Optional[Dict[str, Any]] = None
        self._last_result: Dict[str, Any] = {
            "status": "not_started",
            "attempted": 0,
            "completed": 0,
        }

    def _work_count(self) -> int:
        if hasattr(self.outbox, "work_count"):
            return int(self.outbox.work_count())
        return int(self.outbox.count())

    def _next_wait_seconds(self, result: Dict[str, Any]) -> float:
        """Sleep until durable work is actually due, while remaining wakeable."""
        if result.get("status") in {"cloud_unavailable", "worker_error"}:
            return float(self.config.interval_seconds)
        if hasattr(self.outbox, "next_available_delay"):
            delay = self.outbox.next_available_delay()
            if delay is not None:
                # Avoid a zero-delay busy loop at millisecond timestamp
                # boundaries.  notify() still interrupts longer waits when a
                # newly appended ordinary event is ready sooner.
                return max(0.001, float(delay))
        return float(self.config.interval_seconds)

    def run_once(self) -> Dict[str, Any]:
        network = self.network_monitor.snapshot()
        if not network.available:
            result = {
                "status": "cloud_unavailable",
                "attempted": 0,
                "completed": 0,
                "remaining": self._work_count(),
            }
        elif self._work_count() == 0:
            result = {
                "status": "idle",
                "attempted": 0,
                "completed": 0,
                "remaining": 0,
            }
        else:
            with self.manager.lease() as snapshot:
                result = snapshot.require_edge().flush_pending(
                    batch_size=self.config.batch_size,
                    lease_seconds=self.config.lease_seconds,
                    max_backoff_seconds=self.config.max_backoff_seconds,
                    waiting_poll_seconds=self.config.waiting_poll_seconds,
                    partial_poll_seconds=self.config.partial_poll_seconds,
                    aggregation_max_wait_seconds=(
                        self.config.aggregation_max_wait_seconds
                    ),
                    reconciliation_poll_seconds=(
                        self.config.reconciliation_poll_seconds
                    ),
                    reconciliation_max_wait_seconds=(
                        self.config.reconciliation_max_wait_seconds
                    ),
                    aggregation_batch_wait_seconds=(
                        self.config.aggregation_batch_wait_seconds
                    ),
                )
            result = dict(result)
            error_count = len(result.get("errors", []))
            waiting_count = int(
                result.get("aggregation_waiting", result.get("waiting", 0))
            )
            expired_count = int(result.get("aggregation_expired", 0))
            reconciliation_expired_count = int(
                result.get("reconciliation_expired", 0)
            )
            if error_count:
                result["status"] = "failed"
            elif waiting_count:
                result["status"] = "waiting"
            elif expired_count:
                result["status"] = "aggregation_timeout"
            elif reconciliation_expired_count:
                result["status"] = "reconciliation_expired"
            elif int(result.get("attempted", 0)) == 0 and self._work_count() > 0:
                result["status"] = "scheduled"
            else:
                result["status"] = "completed"
            self.metrics.record_replay(
                int(result.get("attempted", 0)),
                int(result.get("completed", 0)),
                waiting=waiting_count,
                errors=error_count,
            )
            if expired_count:
                self.metrics.increment(
                    "outbox_aggregation_timeout_total", expired_count
                )
                self.metrics.increment(
                    "outbox_partial_final_total",
                    int(result.get("partial_expired", 0)),
                )
                self.metrics.increment(
                    "outbox_local_only_timeout_total",
                    int(result.get("local_timeout_expired", 0)),
                )
            if reconciliation_expired_count:
                self.metrics.increment(
                    "outbox_reconciliation_expired_total",
                    reconciliation_expired_count,
                )
            for delivery in result.get("deliveries", []):
                if not isinstance(delivery, dict):
                    continue
                self.metrics.record_async_delivery(
                    float(delivery.get("http_round_trip_ms", 0.0)),
                    int(delivery.get("request_bytes", 0)),
                    int(delivery.get("response_bytes", 0)),
                    success=bool(delivery.get("success", False)),
                )
        result["observed_at_ms"] = int(time.time() * 1000)
        with self._lock:
            self._last_result = result
            if int(result.get("attempted", 0)) > 0:
                self._last_delivery_result = dict(result)
        return result

    def _run(self) -> None:
        wait_seconds = self.config.interval_seconds
        while not self._stop_event.is_set():
            notified = self._wake_event.wait(wait_seconds)
            self._wake_event.clear()
            if self._stop_event.is_set():
                break
            if (
                notified
                and self._work_count() > 0
                and self.config.batch_coalesce_seconds > 0
            ):
                # A fixed, very short window lets near-simultaneous summaries
                # from one sample share a physical request. It never waits for
                # the next sample and remains interruptible during shutdown.
                if self._stop_event.wait(self.config.batch_coalesce_seconds):
                    break
                # Consume notifications for events already visible to the
                # imminent claim. A later append still wakes the next cycle.
                self._wake_event.clear()
            try:
                result = self.run_once()
                wait_seconds = self._next_wait_seconds(result)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._last_result = {
                        "status": "worker_error",
                        "attempted": 0,
                        "completed": 0,
                        "error": "{}: {}".format(type(exc).__name__, exc),
                        "observed_at_ms": int(time.time() * 1000),
                    }
                    self._last_delivery_result = dict(self._last_result)
                self.metrics.record_failure("outbox_replay")
                wait_seconds = self.config.interval_seconds

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._wake_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="edge-outbox-replay",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        self._stop_event.set()
        self._wake_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(max(0.0, timeout_seconds))

    def notify(self) -> None:
        """Wake the background sender after a durable Outbox append."""
        self._wake_event.set()

    def health(self) -> Dict[str, Any]:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            last_result = dict(self._last_result)
            last_delivery_result = (
                dict(self._last_delivery_result)
                if self._last_delivery_result is not None
                else None
            )
        return {
            "running": running,
            "interval_seconds": self.config.interval_seconds,
            "batch_size": self.config.batch_size,
            "batch_coalesce_seconds": self.config.batch_coalesce_seconds,
            "waiting_poll_seconds": self.config.waiting_poll_seconds,
            "partial_poll_seconds": self.config.partial_poll_seconds,
            "aggregation_max_wait_seconds": (
                self.config.aggregation_max_wait_seconds
            ),
            "reconciliation_poll_seconds": (
                self.config.reconciliation_poll_seconds
            ),
            "reconciliation_max_wait_seconds": (
                self.config.reconciliation_max_wait_seconds
            ),
            "aggregation_batch_wait_seconds": (
                self.config.aggregation_batch_wait_seconds
            ),
            "next_available_delay_seconds": (
                self.outbox.next_available_delay()
                if hasattr(self.outbox, "next_available_delay")
                else None
            ),
            "last_result": last_result,
            "last_delivery_result": last_delivery_result,
        }
