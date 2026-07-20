"""用途：在边缘侧主动探测云端健康状态并生成调度所需网络快照。"""

from collections import deque
from dataclasses import asdict, dataclass
import math
import threading
import time
from typing import Deque, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cloud_edge_framework.scheduling import NetworkSnapshot
from cloud_edge_framework.service_config import NetworkProbeConfig


@dataclass(frozen=True)
class ProbeSample:
    observed_at_ms: int
    success: bool
    rtt_ms: float
    error: str = ""


class CloudNetworkMonitor:
    """Measures application-level reachability instead of trusting request metadata."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        config: NetworkProbeConfig,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.config = config
        if not self.base_url:
            raise ValueError("network monitor base_url must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("network monitor timeout_seconds must be positive")
        if config.failure_threshold > config.window_size:
            raise ValueError("failure_threshold must not exceed window_size")
        self._samples: Deque[ProbeSample] = deque(maxlen=config.window_size)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._consecutive_failures = 0

    def probe_once(self) -> ProbeSample:
        request = Request(
            self.base_url + "/health",
            headers={"Accept": "application/json"},
            method="GET",
        )
        started = time.perf_counter()
        error = ""
        success = False
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response.read(4096)
                success = 200 <= int(response.status) < 300
                if not success:
                    error = "HTTP {}".format(response.status)
        except HTTPError as exc:
            error = "HTTP {}".format(exc.code)
        except (TimeoutError, URLError, OSError) as exc:
            error = "{}: {}".format(type(exc).__name__, exc)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        sample = ProbeSample(
            observed_at_ms=int(time.time() * 1000),
            success=success,
            rtt_ms=elapsed_ms if success else 0.0,
            error=error,
        )
        with self._lock:
            self._samples.append(sample)
            if success:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
        return sample

    def _run(self) -> None:
        while not self._stop_event.wait(self.config.interval_seconds):
            self.probe_once()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
        self.probe_once()
        with self._lock:
            self._thread = threading.Thread(
                target=self._run,
                name="cloud-network-probe",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(max(0.0, timeout_seconds))

    def snapshot(self) -> NetworkSnapshot:
        with self._lock:
            samples = list(self._samples)
            consecutive_failures = self._consecutive_failures
        successes = [sample.rtt_ms for sample in samples if sample.success]
        loss_rate = (
            sum(not sample.success for sample in samples) / len(samples)
            if samples
            else 1.0
        )
        rtt_ms = sum(successes) / len(successes) if successes else 0.0
        if len(successes) >= 2:
            variance = sum((value - rtt_ms) ** 2 for value in successes) / len(successes)
            jitter_ms = math.sqrt(variance)
        else:
            jitter_ms = 0.0
        available = bool(
            samples
            and samples[-1].success
            and consecutive_failures < self.config.failure_threshold
        )
        return NetworkSnapshot(
            available=available,
            rtt_ms=rtt_ms,
            jitter_ms=jitter_ms,
            loss_rate=loss_rate,
            cloud_queue_ms=self.config.cloud_queue_ms,
            cloud_compute_ms=self.config.cloud_compute_ms,
            uplink_mbps=self.config.uplink_mbps,
            downlink_mbps=self.config.downlink_mbps,
            expected_response_bytes=self.config.expected_response_bytes,
        )

    def health(self) -> Dict[str, object]:
        with self._lock:
            samples = list(self._samples)
            failures = self._consecutive_failures
            running = self._thread is not None and self._thread.is_alive()
        return {
            "running": running,
            "base_url": self.base_url,
            "sample_count": len(samples),
            "consecutive_failures": failures,
            "latest": asdict(samples[-1]) if samples else None,
            "snapshot": asdict(self.snapshot()),
            "measurement": "application_http_health_probe",
        }


class StaticNetworkMonitor:
    """Injectable deterministic monitor used by framework tests."""

    def __init__(self, network: NetworkSnapshot) -> None:
        self._network = network

    def set_snapshot(self, network: NetworkSnapshot) -> None:
        self._network = network

    def start(self) -> None:
        return

    def stop(self, timeout_seconds: float = 2.0) -> None:
        del timeout_seconds

    def snapshot(self) -> NetworkSnapshot:
        return self._network

    def health(self) -> Dict[str, object]:
        return {
            "running": True,
            "snapshot": asdict(self._network),
            "measurement": "injected_test_snapshot",
        }
