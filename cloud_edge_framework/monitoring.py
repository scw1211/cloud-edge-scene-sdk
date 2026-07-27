"""用途：为所有场景统一提供延迟标签校准评估、风险集合覆盖率和输入漂移监测。"""

import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from cloud_edge_framework.contracts import SemanticEvent


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _finite_probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("monitoring signal {} must be numeric".format(name))
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("monitoring signal {} must be within [0, 1]".format(name))
    return result


def _histogram(values: Sequence[float], bins: int) -> List[float]:
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int(float(value) * bins))
        counts[index] += 1
    total = max(1, len(values))
    return [count / total for count in counts]


def _population_stability_index(
    reference: Sequence[float], current: Sequence[float]
) -> float:
    epsilon = 1e-6
    total = 0.0
    for expected, observed in zip(reference, current):
        expected_safe = max(epsilon, float(expected))
        observed_safe = max(epsilon, float(observed))
        total += (observed_safe - expected_safe) * math.log(
            observed_safe / expected_safe
        )
    return float(total)


@dataclass(frozen=True)
class MonitoringPolicy:
    window_size: int = 500
    bins: int = 10
    min_labeled_samples: int = 50
    min_drift_samples: int = 50
    bootstrap_reference_size: int = 200
    max_ece: float = 0.10
    target_coverage: float = 0.90
    coverage_tolerance: float = 0.05
    max_psi: float = 0.20
    evaluation_interval_events: int = 25
    evaluation_max_staleness_ms: int = 1000

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("monitoring window_size must be at least 2")
        if self.bins < 2 or self.bins > 100:
            raise ValueError("monitoring bins must be within [2, 100]")
        for name in (
            "min_labeled_samples",
            "min_drift_samples",
            "bootstrap_reference_size",
        ):
            if int(getattr(self, name)) < 2:
                raise ValueError("monitoring {} must be at least 2".format(name))
        for name in ("max_ece", "target_coverage", "coverage_tolerance"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError("monitoring {} must be within [0, 1]".format(name))
        if self.max_psi < 0.0:
            raise ValueError("monitoring max_psi must not be negative")
        if self.evaluation_interval_events < 1:
            raise ValueError(
                "monitoring evaluation_interval_events must be at least 1"
            )
        if self.evaluation_max_staleness_ms < 1:
            raise ValueError(
                "monitoring evaluation_max_staleness_ms must be at least 1"
            )


class CalibrationDriftMonitor:
    """Persists monitoring observations and never treats cloud output as ground truth."""

    def __init__(
        self,
        path: Optional[Path] = None,
        policy: Optional[MonitoringPolicy] = None,
    ) -> None:
        self.path = Path(path).resolve() if path is not None else None
        self.policy = policy or MonitoringPolicy()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path) if self.path is not None else ":memory:",
            timeout=10.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=10000")
        if self.path is not None:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
        self._runtime_snapshot_cache: Dict[str, Dict[str, Any]] = {}
        self._events_since_evaluation: Dict[str, int] = {}
        self._last_evaluation_monotonic_ms: Dict[str, float] = {}
        self._scene_observation_totals: Dict[str, int] = {}
        self._reference_signal_cache: Dict[str, set] = {}
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS monitoring_observation (
                    event_id TEXT PRIMARY KEY,
                    scene TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    prediction_set_json TEXT NOT NULL,
                    signals_json TEXT NOT NULL,
                    observed_at_ms INTEGER NOT NULL,
                    true_label TEXT,
                    correct INTEGER,
                    covered INTEGER,
                    labeled_at_ms INTEGER
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_monitor_scene_time "
                "ON monitoring_observation(scene, observed_at_ms DESC)"
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS monitoring_reference (
                    scene TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    histogram_json TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    PRIMARY KEY(scene, signal)
                )
                """
            )

    @staticmethod
    def standard_signals(event: SemanticEvent) -> Dict[str, float]:
        return {
            "prediction_confidence": float(event.prediction.confidence),
            "risk_score": float(event.risk.score),
            "uncertainty_confidence": float(event.uncertainty.confidence),
        }

    def observe(
        self,
        event: SemanticEvent,
        scene_signals: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        signals = self.standard_signals(event)
        for raw_name, raw_value in dict(scene_signals or {}).items():
            name = str(raw_name).strip()
            if not name or len(name) > 100:
                raise ValueError("monitoring signal name is invalid")
            signals[name] = _finite_probability(raw_value, name)
        signals = {
            name: _finite_probability(value, name) for name, value in signals.items()
        }
        prediction_set = list(event.uncertainty.prediction_set)
        now_ms = int(time.time() * 1000)
        inserted = False
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO monitoring_observation(
                    event_id, scene, predicted_label, confidence,
                    prediction_set_json, signals_json, observed_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.scene,
                    event.risk.level,
                    float(event.uncertainty.confidence),
                    _canonical(prediction_set),
                    _canonical(signals),
                    now_ms,
                ),
            )
            inserted = cursor.rowcount > 0
        if inserted:
            self._register_observation(event.scene)
            self._bootstrap_references(event.scene, signals)
        return self._runtime_scene_snapshot(event.scene)

    def _register_observation(self, scene: str) -> None:
        scene = str(scene)
        with self._lock:
            previous = self._scene_observation_totals.get(scene)
            if previous is None:
                row = self._connection.execute(
                    "SELECT COUNT(*) AS count FROM monitoring_observation WHERE scene=?",
                    (scene,),
                ).fetchone()
                self._scene_observation_totals[scene] = int(row["count"])
            else:
                self._scene_observation_totals[scene] = previous + 1
            self._events_since_evaluation[scene] = (
                self._events_since_evaluation.get(scene, 0) + 1
            )

    def _scene_observation_total(self, scene: str) -> int:
        scene = str(scene)
        with self._lock:
            cached = self._scene_observation_totals.get(scene)
            if cached is not None:
                return cached
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM monitoring_observation WHERE scene=?",
                (scene,),
            ).fetchone()
            total = int(row["count"])
            self._scene_observation_totals[scene] = total
            return total

    def _runtime_scene_snapshot(self, scene: str) -> Dict[str, Any]:
        """Return a bounded-staleness status for the per-event scheduling path.

        Public monitoring endpoints still call ``scene_snapshot`` and therefore always
        produce a fresh full-window evaluation. The hot path reuses the last evaluation
        for at most ``evaluation_interval_events - 1`` newly inserted events.
        """
        scene = str(scene).strip()
        with self._lock:
            cached = self._runtime_snapshot_cache.get(scene)
            event_lag = self._events_since_evaluation.get(scene, 0)
            last_evaluated_ms = self._last_evaluation_monotonic_ms.get(scene)
        now_monotonic_ms = time.monotonic() * 1000.0
        elapsed_ms = (
            now_monotonic_ms - last_evaluated_ms
            if last_evaluated_ms is not None
            else None
        )
        refresh = (
            cached is None
            or event_lag >= self.policy.evaluation_interval_events
            or elapsed_ms is None
            or elapsed_ms >= self.policy.evaluation_max_staleness_ms
        )
        if cached is not None and not refresh:
            current_window_count = min(
                self._scene_observation_total(scene), self.policy.window_size
            )
            for signal in cached.get("drift", {}).get("signals", {}).values():
                if (
                    not bool(signal.get("ready", False))
                    and current_window_count >= self.policy.min_drift_samples
                ):
                    refresh = True
                    break
        if refresh:
            return self.scene_snapshot(scene)
        snapshot = copy.deepcopy(cached)
        snapshot["observed_count"] = min(
            self._scene_observation_total(scene), self.policy.window_size
        )
        snapshot["evaluation"] = {
            "fresh": False,
            "event_lag": event_lag,
            "max_event_lag": self.policy.evaluation_interval_events - 1,
            "elapsed_since_full_evaluation_ms": round(elapsed_ms, 6),
            "max_staleness_ms": self.policy.evaluation_max_staleness_ms,
        }
        return snapshot

    def _recent_signal_values(
        self, scene: str, signal: str, limit: int
    ) -> List[float]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT signals_json FROM monitoring_observation "
                "WHERE scene=? ORDER BY observed_at_ms DESC LIMIT ?",
                (str(scene), int(limit)),
            ).fetchall()
        values: List[float] = []
        for row in reversed(rows):
            payload = json.loads(str(row["signals_json"]))
            if signal in payload:
                values.append(_finite_probability(payload[signal], signal))
        return values

    def _has_reference(self, scene: str, signal: str) -> bool:
        with self._lock:
            scene = str(scene)
            signals = self._reference_signal_cache.get(scene)
            if signals is None:
                rows = self._connection.execute(
                    "SELECT signal FROM monitoring_reference WHERE scene=?",
                    (scene,),
                ).fetchall()
                signals = {str(row["signal"]) for row in rows}
                self._reference_signal_cache[scene] = signals
            return str(signal) in signals

    def _bootstrap_references(
        self, scene: str, signals: Mapping[str, float]
    ) -> None:
        required = self.policy.bootstrap_reference_size
        if self._scene_observation_total(scene) < required:
            return
        for signal in sorted(signals):
            if self._has_reference(scene, signal):
                continue
            values = self._recent_signal_values(scene, signal, required)
            if len(values) < required:
                continue
            self.set_reference(scene, signal, values, source="online_bootstrap")

    def set_reference(
        self,
        scene: str,
        signal: str,
        samples: Sequence[Any],
        source: str = "validation",
    ) -> Dict[str, Any]:
        scene = str(scene).strip()
        signal = str(signal).strip()
        source = str(source).strip()
        if not scene or not signal or not source:
            raise ValueError("reference scene, signal and source must not be empty")
        values = [_finite_probability(value, signal) for value in samples]
        if len(values) < 2:
            raise ValueError("monitoring reference needs at least two samples")
        histogram = _histogram(values, self.policy.bins)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO monitoring_reference(
                    scene, signal, histogram_json, sample_count, source, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(scene, signal) DO UPDATE SET
                    histogram_json=excluded.histogram_json,
                    sample_count=excluded.sample_count,
                    source=excluded.source,
                    created_at_ms=excluded.created_at_ms
                """,
                (
                    scene,
                    signal,
                    _canonical(histogram),
                    len(values),
                    source,
                    int(time.time() * 1000),
                ),
            )
            self._reference_signal_cache.setdefault(scene, set()).add(signal)
        return self.scene_snapshot(scene)

    def record_outcome(self, event_id: str, true_label: str) -> Dict[str, Any]:
        event_id = str(event_id).strip()
        true_label = str(true_label).strip()
        if not event_id or not true_label:
            raise ValueError("event_id and true_label must not be empty")
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT scene, predicted_label, prediction_set_json "
                "FROM monitoring_observation WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise KeyError("monitoring event not found: {}".format(event_id))
            prediction_set = json.loads(str(row["prediction_set_json"]))
            self._connection.execute(
                """
                UPDATE monitoring_observation
                SET true_label=?, correct=?, covered=?, labeled_at_ms=?
                WHERE event_id=?
                """,
                (
                    true_label,
                    1 if str(row["predicted_label"]) == true_label else 0,
                    1 if true_label in prediction_set else 0,
                    int(time.time() * 1000),
                    event_id,
                ),
            )
            scene = str(row["scene"])
        return {"event_id": event_id, "true_label": true_label, "monitoring": self.scene_snapshot(scene)}

    def _scene_rows(self, scene: str) -> List[sqlite3.Row]:
        with self._lock:
            return self._connection.execute(
                "SELECT * FROM monitoring_observation WHERE scene=? "
                "ORDER BY observed_at_ms DESC LIMIT ?",
                (str(scene), int(self.policy.window_size)),
            ).fetchall()

    def _references(self, scene: str) -> List[sqlite3.Row]:
        with self._lock:
            return self._connection.execute(
                "SELECT * FROM monitoring_reference WHERE scene=? ORDER BY signal",
                (str(scene),),
            ).fetchall()

    def scene_snapshot(self, scene: str) -> Dict[str, Any]:
        scene = str(scene).strip()
        rows = self._scene_rows(scene)
        labeled = [row for row in rows if row["correct"] is not None]
        calibration_reasons: List[str] = []
        ece: Optional[float] = None
        accuracy: Optional[float] = None
        coverage: Optional[float] = None
        average_set_size: Optional[float] = None
        prediction_set_samples = 0
        if labeled:
            accuracy = sum(int(row["correct"]) for row in labeled) / len(labeled)
            ece_total = 0.0
            for bin_id in range(self.policy.bins):
                lower = bin_id / self.policy.bins
                upper = (bin_id + 1) / self.policy.bins
                selected = [
                    row
                    for row in labeled
                    if lower <= float(row["confidence"])
                    and (
                        float(row["confidence"]) < upper
                        or (bin_id == self.policy.bins - 1 and float(row["confidence"]) <= upper)
                    )
                ]
                if not selected:
                    continue
                bin_confidence = sum(float(row["confidence"]) for row in selected) / len(selected)
                bin_accuracy = sum(int(row["correct"]) for row in selected) / len(selected)
                ece_total += len(selected) / len(labeled) * abs(bin_accuracy - bin_confidence)
            ece = ece_total
            set_rows = [
                row
                for row in labeled
                if json.loads(str(row["prediction_set_json"]))
            ]
            prediction_set_samples = len(set_rows)
            if set_rows:
                coverage = sum(int(row["covered"]) for row in set_rows) / len(set_rows)
                average_set_size = sum(
                    len(json.loads(str(row["prediction_set_json"]))) for row in set_rows
                ) / len(set_rows)
        calibration_ready = len(labeled) >= self.policy.min_labeled_samples
        if calibration_ready and ece is not None and ece > self.policy.max_ece:
            calibration_reasons.append("ece_exceeded")
        coverage_ready = prediction_set_samples >= self.policy.min_labeled_samples
        minimum_coverage = self.policy.target_coverage - self.policy.coverage_tolerance
        if coverage_ready and coverage is not None and coverage < minimum_coverage:
            calibration_reasons.append("prediction_set_undercoverage")

        drift_signals: Dict[str, Any] = {}
        drift_reasons: List[str] = []
        references = self._references(scene)
        for reference in references:
            signal = str(reference["signal"])
            values: List[float] = []
            for row in reversed(rows):
                payload = json.loads(str(row["signals_json"]))
                if signal in payload:
                    values.append(_finite_probability(payload[signal], signal))
            ready = len(values) >= self.policy.min_drift_samples
            psi = None
            if ready:
                expected = json.loads(str(reference["histogram_json"]))
                observed = _histogram(values, self.policy.bins)
                psi = _population_stability_index(expected, observed)
                if psi > self.policy.max_psi:
                    drift_reasons.append("psi_exceeded:{}".format(signal))
            drift_signals[signal] = {
                "sample_count": len(values),
                "reference_sample_count": int(reference["sample_count"]),
                "reference_source": str(reference["source"]),
                "psi": round(psi, 6) if psi is not None else None,
                "ready": ready,
            }

        reasons = calibration_reasons + drift_reasons
        enough_drift_data = bool(drift_signals) and all(
            bool(value["ready"]) for value in drift_signals.values()
        )
        if reasons:
            status = "degraded"
        elif calibration_ready and enough_drift_data:
            status = "healthy"
        else:
            status = "collecting"
        snapshot = {
            "scene": scene,
            "status": status,
            "force_cloud_review": status == "degraded",
            "reasons": reasons,
            "observed_count": len(rows),
            "policy": asdict(self.policy),
            "calibration": {
                "status": "degraded"
                if calibration_reasons
                else ("ready" if calibration_ready else "collecting"),
                "labeled_count": len(labeled),
                "accuracy": round(accuracy, 6) if accuracy is not None else None,
                "ece": round(ece, 6) if ece is not None else None,
                "max_ece": self.policy.max_ece,
                "prediction_set_samples": prediction_set_samples,
                "coverage": round(coverage, 6) if coverage is not None else None,
                "target_coverage": self.policy.target_coverage,
                "average_prediction_set_size": round(average_set_size, 6)
                if average_set_size is not None
                else None,
            },
            "drift": {
                "status": "degraded"
                if drift_reasons
                else ("ready" if enough_drift_data else "collecting"),
                "max_psi": self.policy.max_psi,
                "signals": drift_signals,
            },
            "evaluation": {
                "fresh": True,
                "event_lag": 0,
                "max_event_lag": self.policy.evaluation_interval_events - 1,
                "elapsed_since_full_evaluation_ms": 0.0,
                "max_staleness_ms": self.policy.evaluation_max_staleness_ms,
            },
        }
        with self._lock:
            self._runtime_snapshot_cache[scene] = copy.deepcopy(snapshot)
            self._events_since_evaluation[scene] = 0
            self._last_evaluation_monotonic_ms[scene] = time.monotonic() * 1000.0
            self._scene_observation_total(scene)
        return snapshot

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT DISTINCT scene FROM monitoring_observation ORDER BY scene"
            ).fetchall()
        scenes = {str(row["scene"]): self.scene_snapshot(str(row["scene"])) for row in rows}
        return {
            "path": str(self.path) if self.path is not None else ":memory:",
            "scene_count": len(scenes),
            "scenes": scenes,
        }

    def close(self) -> None:
        with self._lock:
            self._connection.close()
