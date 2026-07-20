"""用途：监听 Edge LLM active release，并通过插件快照重载安全应用新版本。"""

import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from edge_llm_factory.release_store import ReleaseStore


class EdgeLLMReleaseWatcher:
    def __init__(
        self,
        manager: Any,
        registry_path: Path,
        interval_seconds: float = 2.0,
        store: Optional[Any] = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("release watcher interval_seconds must be positive")
        self.manager = manager
        self.registry_path = Path(registry_path).resolve()
        self.interval_seconds = float(interval_seconds)
        self.store = store or ReleaseStore(self.registry_path)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None
        self._reload_count = 0
        self._failed_revision: Optional[int] = None
        self._failure_count = 0
        self._retry_not_before = 0.0
        initial = self.store.status(verify_active=True)
        self._active_release_id = initial.get("active_release_id")
        self._applied_revision = int(initial.get("revision", 0))
        self._observed_revision = self._applied_revision

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="edge-llm-release-watcher",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.run_once()

    def run_once(self, force: bool = False) -> Dict[str, Any]:
        with self._lock:
            try:
                quick = self.store.status(verify_active=False)
                revision = int(quick.get("revision", 0))
                release_id = quick.get("active_release_id")
                self._observed_revision = revision
                if not force and revision == self._applied_revision:
                    return {
                        "status": "unchanged",
                        "active_release_id": release_id,
                        "revision": revision,
                    }
                now = time.monotonic()
                if (
                    not force
                    and revision == self._failed_revision
                    and now < self._retry_not_before
                ):
                    return {
                        "status": "retry_deferred",
                        "active_release_id": self._active_release_id,
                        "applied_revision": self._applied_revision,
                        "observed_revision": revision,
                        "retry_after_seconds": round(self._retry_not_before - now, 6),
                    }
                verified = self.store.status(verify_active=True)
                if int(verified.get("revision", -1)) != revision:
                    raise RuntimeError("release store changed while verifying active artifacts")
                if verified.get("active_release_id") != release_id:
                    raise RuntimeError("active release changed while verifying artifacts")
                if release_id is None:
                    raise RuntimeError("release store has no active release")
                reload_result = self.manager.reload()
                self._applied_revision = revision
                self._active_release_id = release_id
                self._reload_count += 1
                self._last_error = None
                self._failed_revision = None
                self._failure_count = 0
                self._retry_not_before = 0.0
                return {
                    "status": "reloaded",
                    "active_release_id": release_id,
                    "revision": revision,
                    "runtime": reload_result,
                }
            except Exception as exc:  # noqa: BLE001
                self._last_error = "{}: {}".format(type(exc).__name__, exc)
                if self._failed_revision == self._observed_revision:
                    self._failure_count += 1
                else:
                    self._failed_revision = self._observed_revision
                    self._failure_count = 1
                retry_seconds = min(
                    60.0, self.interval_seconds * (2 ** min(self._failure_count, 5))
                )
                self._retry_not_before = time.monotonic() + retry_seconds
                return {
                    "status": "reload_failed",
                    "active_release_id": self._active_release_id,
                    "applied_revision": self._applied_revision,
                    "observed_revision": self._observed_revision,
                    "error": self._last_error,
                    "retry_after_seconds": round(retry_seconds, 6),
                }

    def health(self) -> Dict[str, Any]:
        with self._lock:
            retry_after = max(0.0, self._retry_not_before - time.monotonic())
            return {
                "status": "ok" if self._last_error is None else "degraded",
                "registry": str(self.registry_path),
                "active_release_id": self._active_release_id,
                "applied_revision": self._applied_revision,
                "observed_revision": self._observed_revision,
                "reload_count": self._reload_count,
                "last_error": self._last_error,
                "failed_revision": self._failed_revision,
                "failure_count": self._failure_count,
                "retry_after_seconds": round(retry_after, 6),
                "running": self._thread is not None and self._thread.is_alive(),
            }

    def stop(self, timeout_seconds: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout_seconds)
        self._thread = None
