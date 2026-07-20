"""用途：按场景、证据层级和网络等级保存云边闭环实测性能画像。"""

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from cloud_edge_framework.scheduling import NetworkSnapshot


def network_class(network: NetworkSnapshot) -> str:
    if not network.available or network.loss_rate >= 0.95:
        return "outage"
    if network.rtt_ms <= 20.0 and network.loss_rate < 0.02:
        return "good"
    if network.rtt_ms <= 80.0 and network.loss_rate < 0.10:
        return "normal"
    return "degraded"


@dataclass(frozen=True)
class PerformanceEstimate:
    sample_count: int
    success_rate: float
    cloud_path_ms: float
    request_bytes: float
    response_bytes: float
    network_class: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceProfileStore:
    def __init__(
        self,
        path: Optional[Path] = None,
        alpha: float = 0.25,
        minimum_samples: int = 3,
        synchronous_persistence: bool = True,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("performance alpha must be within (0, 1]")
        if minimum_samples <= 0:
            raise ValueError("minimum_samples must be positive")
        self.path = Path(path) if path is not None else None
        self.alpha = float(alpha)
        self.minimum_samples = int(minimum_samples)
        self.synchronous_persistence = bool(synchronous_persistence)
        self._lock = threading.RLock()
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        self._load()

    @staticmethod
    def _key(scene: str, evidence_level: str, network_name: str) -> str:
        return "{}|{}|{}".format(scene, evidence_level, network_name)

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
            raise ValueError("invalid performance profile store")
        profiles = payload.get("profiles", {})
        if not isinstance(profiles, dict):
            raise ValueError("performance profiles must be an object")
        self._profiles = {str(key): dict(value) for key, value in profiles.items()}

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file_obj:
                json.dump(
                    {"schema_version": 1, "profiles": self._profiles},
                    file_obj,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def estimate(
        self,
        scene: str,
        evidence_level: str,
        network: NetworkSnapshot,
    ) -> Optional[PerformanceEstimate]:
        name = network_class(network)
        key = self._key(scene, evidence_level, name)
        with self._lock:
            profile = self._profiles.get(key)
            if profile is None or int(profile.get("sample_count", 0)) < self.minimum_samples:
                return None
            return PerformanceEstimate(
                sample_count=int(profile["sample_count"]),
                success_rate=float(profile["success_rate"]),
                cloud_path_ms=float(profile["cloud_path_ms"]),
                request_bytes=float(profile["request_bytes"]),
                response_bytes=float(profile["response_bytes"]),
                network_class=name,
            )

    def record(
        self,
        scene: str,
        evidence_level: str,
        network: NetworkSnapshot,
        success: bool,
        cloud_path_ms: float,
        request_bytes: int,
        response_bytes: int,
    ) -> None:
        name = network_class(network)
        key = self._key(scene, evidence_level, name)
        values = {
            "success_rate": 1.0 if success else 0.0,
            "cloud_path_ms": max(0.0, float(cloud_path_ms)),
            "request_bytes": max(0.0, float(request_bytes)),
            "response_bytes": max(0.0, float(response_bytes)),
        }
        with self._lock:
            previous = self._profiles.get(key)
            if previous is None:
                profile = {"sample_count": 1, **values}
            else:
                profile = {"sample_count": int(previous.get("sample_count", 0)) + 1}
                for field_name, current in values.items():
                    profile[field_name] = (
                        self.alpha * current
                        + (1.0 - self.alpha) * float(previous.get(field_name, current))
                    )
            self._profiles[key] = profile
            if self.synchronous_persistence:
                self._persist()
            else:
                self._dirty = True

    def flush(self) -> None:
        with self._lock:
            if self._dirty:
                self._persist()
                self._dirty = False

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1,
                "minimum_samples": self.minimum_samples,
                "profiles": {key: dict(value) for key, value in self._profiles.items()},
            }
