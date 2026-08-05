"""用途：原子构建和热重载场景插件运行快照，失败时保持上一版本继续服务。"""

import hashlib
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from cloud_edge_framework.registry import SceneRegistry, build_default_registry
from cloud_edge_framework.feedback import DecisionFeedbackStore
from cloud_edge_framework.performance import PerformanceProfileStore
from cloud_edge_framework.review_queue import PendingReviewStore
from cloud_edge_framework.review_tracking import ReviewLifecycleStore
from cloud_edge_framework.monitoring import CalibrationDriftMonitor
from cloud_edge_framework.runtime import CloudRuntime, EdgeRuntime
from cloud_edge_framework.scheduling import CollaborationScheduler


RUNTIME_ROLES = {"edge", "cloud", "combined"}


@dataclass(frozen=True)
class RuntimeSnapshot:
    role: str
    generation: int
    loaded_at_utc: str
    config_sha256: str
    registry: SceneRegistry
    cloud: Optional[CloudRuntime]
    edge: Optional[EdgeRuntime]

    def require_cloud(self) -> CloudRuntime:
        if self.cloud is None:
            raise RuntimeError("cloud runtime is not available in {} role".format(self.role))
        return self.cloud

    def require_edge(self) -> EdgeRuntime:
        if self.edge is None:
            raise RuntimeError("edge runtime is not available in {} role".format(self.role))
        return self.edge

    def describe(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "generation": self.generation,
            "loaded_at_utc": self.loaded_at_utc,
            "config_sha256": self.config_sha256,
            "scenes": self.registry.scenes(),
            "plugins": self.registry.descriptors(),
        }


class PluginRuntimeManager:
    def __init__(
        self,
        project_root: Path,
        config_path: Optional[Path] = None,
        review_store: Optional[PendingReviewStore] = None,
        performance_store: Optional[PerformanceProfileStore] = None,
        feedback_store: Optional[DecisionFeedbackStore] = None,
        review_tracker: Optional[ReviewLifecycleStore] = None,
        role: str = "combined",
        remote_cloud: Optional[Any] = None,
        scheduler: Optional[CollaborationScheduler] = None,
        cloud_reviewer: Optional[Any] = None,
        calibration_monitor: Optional[CalibrationDriftMonitor] = None,
        utility_router: Optional[Any] = None,
        durable_handoff: Optional[Any] = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.config_path = (
            config_path.resolve()
            if config_path is not None
            else self.project_root / "deployment" / "framework" / "scene_plugins.json"
        )
        self.review_store = review_store or PendingReviewStore()
        self.performance_store = performance_store or PerformanceProfileStore()
        self.feedback_store = feedback_store or DecisionFeedbackStore()
        self._owns_review_tracker = review_tracker is None
        self.review_tracker = review_tracker or ReviewLifecycleStore()
        self.role = str(role)
        if self.role not in RUNTIME_ROLES:
            raise ValueError("runtime role must be one of {}".format(sorted(RUNTIME_ROLES)))
        if self.role == "edge" and remote_cloud is None:
            raise ValueError("edge runtime manager requires remote_cloud")
        self.remote_cloud = remote_cloud
        self.scheduler = scheduler
        self.cloud_reviewer = cloud_reviewer
        self.calibration_monitor = calibration_monitor
        self.utility_router = utility_router
        self.durable_handoff = durable_handoff
        self._lock = threading.RLock()
        self._reload_lock = threading.Lock()
        self._active: Optional[RuntimeSnapshot] = None
        self._retired: Dict[int, RuntimeSnapshot] = {}
        self._inflight: Dict[int, int] = {}
        self.reload()

    def _config_sha256(self) -> str:
        if not self.config_path.is_file():
            return "builtin_default"
        return hashlib.sha256(self.config_path.read_bytes()).hexdigest()

    def _build_snapshot(self, generation: int) -> RuntimeSnapshot:
        registry = build_default_registry(self.project_root, self.config_path)
        cloud: Optional[CloudRuntime] = None
        edge: Optional[EdgeRuntime] = None
        try:
            if self.role in {"cloud", "combined"}:
                cloud = CloudRuntime(registry, reviewer=self.cloud_reviewer)
                cloud.warmup()
            else:
                registry.warmup()
            if self.role in {"edge", "combined"}:
                edge_cloud = cloud if self.role == "combined" else self.remote_cloud
                edge = EdgeRuntime(
                    registry=registry,
                    cloud=edge_cloud,
                    scheduler=self.scheduler,
                    review_store=self.review_store,
                    performance_store=self.performance_store,
                    feedback_store=self.feedback_store,
                    review_tracker=self.review_tracker,
                    calibration_monitor=self.calibration_monitor,
                    utility_router=self.utility_router,
                    durable_handoff=self.durable_handoff,
                )
        except Exception:
            registry.close()
            raise
        return RuntimeSnapshot(
            role=self.role,
            generation=generation,
            loaded_at_utc=datetime.now(timezone.utc).isoformat(),
            config_sha256=self._config_sha256(),
            registry=registry,
            cloud=cloud,
            edge=edge,
        )

    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            if self._active is None:
                raise RuntimeError("plugin runtime is not initialized")
            return self._active

    @contextmanager
    def lease(self) -> Iterator[RuntimeSnapshot]:
        with self._lock:
            if self._active is None:
                raise RuntimeError("plugin runtime is not initialized")
            snapshot = self._active
            self._inflight[snapshot.generation] = self._inflight.get(snapshot.generation, 0) + 1
        try:
            yield snapshot
        finally:
            close_snapshot = None
            with self._lock:
                remaining = self._inflight.get(snapshot.generation, 1) - 1
                if remaining <= 0:
                    self._inflight.pop(snapshot.generation, None)
                    close_snapshot = self._retired.pop(snapshot.generation, None)
                else:
                    self._inflight[snapshot.generation] = remaining
            if close_snapshot is not None:
                close_snapshot.registry.close()

    def reload(self) -> Dict[str, Any]:
        with self._reload_lock:
            with self._lock:
                generation = 1 if self._active is None else self._active.generation + 1
            candidate = self._build_snapshot(generation)
            close_previous = None
            with self._lock:
                previous = self._active
                self._active = candidate
                if previous is not None:
                    if self._inflight.get(previous.generation, 0) > 0:
                        self._retired[previous.generation] = previous
                    else:
                        close_previous = previous
                retired_count = len(self._retired)
            if close_previous is not None:
                close_previous.registry.close()
            result = candidate.describe()
            result.update(
                {
                    "reloaded": True,
                    "previous_generation": previous.generation if previous else None,
                    "retired_snapshot_count": retired_count,
                }
            )
            return result

    def health(self) -> Dict[str, Any]:
        # Hold a lease, not the manager lock, while collecting store/plugin
        # diagnostics. Some snapshots execute SQLite queries and parse history;
        # keeping the manager lock across them stalls every /decide lease.
        with self.lease() as snapshot:
            with self._lock:
                retired_count = len(self._retired)
                inflight = sum(self._inflight.values())
            result = {
                **snapshot.describe(),
                "pending_reviews": self.review_store.count(),
                "performance_profiles": len(
                    self.performance_store.snapshot()["profiles"]
                ),
                "decision_feedback_records": self.feedback_store.count(),
                "review_lifecycle": self.review_tracker.snapshot(),
                "calibration_drift_monitor": (
                    self.calibration_monitor.snapshot()
                    if self.calibration_monitor is not None
                    else {"status": "disabled"}
                ),
                "utility_router": (
                    self.utility_router.describe()
                    if self.utility_router is not None
                    else {"enabled": False}
                ),
                "cloud_llm": (
                    self.cloud_reviewer.describe()
                    if self.cloud_reviewer is not None
                    else {"enabled": False}
                ),
                "retired_snapshot_count": retired_count,
                "inflight_requests": inflight,
            }
            if self.durable_handoff is not None:
                result["durable_handoff"] = self.durable_handoff.snapshot()
            return result

    def close(self) -> None:
        self.feedback_store.flush()
        self.performance_store.flush()
        with self._lock:
            snapshots = ([self._active] if self._active is not None else []) + list(
                self._retired.values()
            )
            self._active = None
            self._retired = {}
            self._inflight = {}
        errors = []
        for snapshot in reversed(snapshots):
            try:
                snapshot.registry.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
        if self._owns_review_tracker:
            self.review_tracker.close()
        if errors:
            raise RuntimeError("failed to close plugin runtime: {}".format("; ".join(errors)))
