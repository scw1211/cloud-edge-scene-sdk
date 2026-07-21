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
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._last_delivery_result: Optional[Dict[str, Any]] = None
        self._last_result: Dict[str, Any] = {
            "status": "not_started",
            "attempted": 0,
            "completed": 0,
        }

    def run_once(self) -> Dict[str, Any]:
        network = self.network_monitor.snapshot()
        if not network.available:
            result = {
                "status": "cloud_unavailable",
                "attempted": 0,
                "completed": 0,
                "remaining": self.outbox.count(),
            }
        elif self.outbox.count() == 0:
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
                )
            result = dict(result)
            result["status"] = "completed" if result.get("completed") else "failed"
            self.metrics.record_replay(
                int(result.get("attempted", 0)),
                int(result.get("completed", 0)),
            )
        result["observed_at_ms"] = int(time.time() * 1000)
        with self._lock:
            self._last_result = result
            if int(result.get("attempted", 0)) > 0:
                self._last_delivery_result = dict(result)
        return result

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
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
            self._stop_event.wait(self.config.interval_seconds)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="edge-outbox-replay",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(max(0.0, timeout_seconds))

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
            "last_result": last_result,
            "last_delivery_result": last_delivery_result,
        }
