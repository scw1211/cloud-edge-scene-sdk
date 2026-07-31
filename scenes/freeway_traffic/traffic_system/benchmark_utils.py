"""用途：提供 HTTP 闭环、延迟统计、进程内存采样和服务管理公共工具。"""

import json
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * ratio))
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def summarize(values: Iterable[float]) -> Dict[str, float]:
    data = [float(value) for value in values]
    if not data:
        return {
            "count": 0,
            "average_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(data),
        "average_ms": round(statistics.fmean(data), 6),
        "p50_ms": round(percentile(data, 0.50), 6),
        "p95_ms": round(percentile(data, 0.95), 6),
        "p99_ms": round(percentile(data, 0.99), 6),
        "min_ms": round(min(data), 6),
        "max_ms": round(max(data), 6),
    }


def build_payload(
    request_id: str,
    event: Dict[str, Any],
    perception_latency_ms: float,
    student_latency_ms: float,
    compact_event: bool = False,
) -> bytes:
    if compact_event:
        from traffic_system.decision_utils import compact_event_for_teacher

        event = compact_event_for_teacher(event)
    payload = {
        "request_id": request_id,
        "client_sent_at_ns": time.time_ns(),
        "edge_perception_latency_ms": perception_latency_ms,
        "edge_student_latency_ms": student_latency_ms,
        "edge_compute_latency_ms": perception_latency_ms + student_latency_ms,
        "edge_event": event,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def post_json(url: str, body: bytes, timeout: float) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Cloud response must be a JSON object.")
    return data


def read_rss_kb(pid: int) -> int:
    try:
        status_path = Path("/proc") / str(pid) / "status"
        for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                return int(parts[1]) if len(parts) >= 2 else 0
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        pass
    return 0


def read_pss_kb(pid: int) -> int:
    """Read proportional set size so shared mmap pages are not double-counted."""
    try:
        rollup_path = Path("/proc") / str(pid) / "smaps_rollup"
        for line in rollup_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("Pss:"):
                parts = line.split()
                return int(parts[1]) if len(parts) >= 2 else 0
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        pass
    return 0


def child_pids(pid: int) -> List[int]:
    try:
        children_path = Path("/proc") / str(pid) / "task" / str(pid) / "children"
        value = children_path.read_text(encoding="utf-8", errors="ignore").strip()
        return [int(child) for child in value.split()] if value else []
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return []


def process_tree_rss_mb(pid: int) -> float:
    total_kb = sum(read_rss_kb(item) for item in [pid] + child_pids(pid))
    return round(total_kb / 1024.0, 4)


def process_tree_pss_mb(pid: int) -> float:
    total_kb = sum(read_pss_kb(item) for item in [pid] + child_pids(pid))
    return round(total_kb / 1024.0, 4)


class ProcessMemorySampler:
    def __init__(self, pid: int, interval: float) -> None:
        self.pid = pid
        self.interval = max(0.01, interval)
        self.samples_mb: List[float] = []
        self.pss_samples_mb: List[float] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        next_pss_sample = 0.0
        while not self._stop_event.is_set():
            self.samples_mb.append(process_tree_rss_mb(self.pid))
            now = time.monotonic()
            if now >= next_pss_sample:
                self.pss_samples_mb.append(process_tree_pss_mb(self.pid))
                next_pss_sample = now + 1.0
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        self.samples_mb.append(process_tree_rss_mb(self.pid))
        self.pss_samples_mb.append(process_tree_pss_mb(self.pid))
        self._thread.start()

    def stop(self) -> None:
        self.samples_mb.append(process_tree_rss_mb(self.pid))
        self.pss_samples_mb.append(process_tree_pss_mb(self.pid))
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return round(max(self.samples_mb), 4) if self.samples_mb else 0.0

    @property
    def peak_pss_mb(self) -> float:
        return round(max(self.pss_samples_mb), 4) if self.pss_samples_mb else 0.0


def system_used_memory_mb() -> float:
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    return round((total - available) / 1024.0, 4)


class SystemMemorySampler:
    def __init__(self, interval: float) -> None:
        self.interval = max(0.01, interval)
        self.baseline_mb = system_used_memory_mb()
        self.samples_mb: List[float] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.samples_mb.append(system_used_memory_mb())
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        self.samples_mb.append(system_used_memory_mb())
        self._thread.start()

    def stop(self) -> None:
        self.samples_mb.append(system_used_memory_mb())
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return round(max(self.samples_mb), 4) if self.samples_mb else 0.0

    @property
    def peak_delta_mb(self) -> float:
        return round(max(0.0, self.peak_mb - self.baseline_mb), 4)


def wait_until_ready(base_url: str, proc: subprocess.Popen, timeout: int) -> float:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited before it became ready.")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=1) as response:
                if response.status == 200:
                    return round((time.perf_counter() - started) * 1000.0, 4)
        except urllib.error.HTTPError as exc:
            if exc.code == 200:
                return round((time.perf_counter() - started) * 1000.0, 4)
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.1)
    raise TimeoutError("Timed out waiting for llama-server: {}".format(base_url))


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def ollama_pids() -> List[int]:
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmdline = (proc / "cmdline").read_text(encoding="utf-8", errors="ignore")
            comm = (proc / "comm").read_text(encoding="utf-8", errors="ignore")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        executable = Path(cmdline.split("\0", 1)[0]).name
        if comm.strip() in {"ollama", "llama-server"} or executable in {"ollama", "llama-server"}:
            pids.append(int(proc.name))
    return sorted(set(pids))


def ollama_rss_mb() -> float:
    return round(sum(read_rss_kb(pid) for pid in ollama_pids()) / 1024.0, 4)
