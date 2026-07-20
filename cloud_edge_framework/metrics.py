"""用途：以统一 JSON 口径记录边缘、云端和弱网重放的运行指标。"""

from collections import Counter, defaultdict, deque
import math
import threading
import time
from typing import Any, Deque, Dict, Iterable, Mapping


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile / 100.0
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


class FrameworkMetrics:
    def __init__(self, role: str, sample_limit: int = 10000) -> None:
        self.role = str(role)
        self.started_at_ms = int(time.time() * 1000)
        self.sample_limit = int(sample_limit)
        if self.sample_limit <= 0:
            raise ValueError("metrics sample_limit must be positive")
        self._lock = threading.RLock()
        self._counters: Counter = Counter()
        self._labels: Dict[str, Counter] = defaultdict(Counter)
        self._samples: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.sample_limit)
        )

    def increment(self, name: str, amount: int = 1, **labels: str) -> None:
        with self._lock:
            self._counters[str(name)] += int(amount)
            for label_name, label_value in labels.items():
                self._labels["{}:{}".format(name, label_name)][str(label_value)] += int(
                    amount
                )

    def observe(self, name: str, value: float) -> None:
        numeric = float(value)
        if not math.isfinite(numeric):
            return
        with self._lock:
            self._samples[str(name)].append(numeric)

    def record_edge_result(self, result: Mapping[str, Any]) -> None:
        schedule = result.get("schedule", {})
        final = result.get("final_decision", {})
        data_plane = result.get("data_plane", {})
        accounting = result.get("closed_loop_accounting", {})
        metadata = final.get("metadata", {}) if isinstance(final, dict) else {}
        transport = metadata.get("transport", {}) if isinstance(metadata, dict) else {}
        route = str(final.get("route", schedule.get("route", "unknown")))
        self.increment("edge_requests_total", route=route)
        if route == "local_autonomy":
            self.increment("local_autonomy_total")
        self.observe("framework_runtime_ms", result.get("framework_runtime_ms", 0.0))
        self.observe(
            "accounted_closed_loop_ms",
            accounting.get("accounted_closed_loop_ms", 0.0),
        )
        self.observe("selected_request_bytes", data_plane.get("selected_request_bytes", 0))
        if isinstance(transport, dict) and transport:
            self.observe("http_round_trip_ms", transport.get("http_round_trip_ms", 0.0))
            self.observe("http_request_bytes", transport.get("request_bytes", 0))
            self.observe("http_response_bytes", transport.get("response_bytes", 0))
            self.observe("http_attempts", transport.get("attempts", 1))

    def record_cloud_request(
        self,
        operation: str,
        elapsed_ms: float,
        replayed: bool,
    ) -> None:
        self.increment("cloud_requests_total", operation=operation)
        if replayed:
            self.increment("idempotency_replays_total", operation=operation)
        self.observe("cloud_service_runtime_ms", elapsed_ms)

    def record_failure(self, operation: str) -> None:
        self.increment("request_failures_total", operation=operation)

    def record_replay(self, attempted: int, completed: int) -> None:
        self.increment("outbox_replay_runs_total")
        self.increment("outbox_replay_attempted_total", attempted)
        self.increment("outbox_replay_completed_total", completed)
        if attempted and not completed:
            self.increment("outbox_replay_failures_total")

    @staticmethod
    def _sample_summary(values: Deque[float]) -> Dict[str, float]:
        items = list(values)
        if not items:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "count": len(items),
            "mean": round(sum(items) / len(items), 6),
            "p50": round(_percentile(items, 50.0), 6),
            "p95": round(_percentile(items, 95.0), 6),
            "max": round(max(items), 6),
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            labels = {name: dict(values) for name, values in self._labels.items()}
            samples = {
                name: self._sample_summary(values)
                for name, values in self._samples.items()
            }
        return {
            "role": self.role,
            "started_at_ms": self.started_at_ms,
            "uptime_ms": max(0, int(time.time() * 1000) - self.started_at_ms),
            "counters": counters,
            "labels": labels,
            "distributions": samples,
        }
