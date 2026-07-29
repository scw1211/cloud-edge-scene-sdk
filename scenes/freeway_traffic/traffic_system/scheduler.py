"""用途：根据风险、置信度、网络质量和截止时间动态选择边缘或云端路径。"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from traffic_system.decision_utils import safe_float


RISK_PRIORITY = {"low": 0, "medium": 1, "high": 2, "severe": 3}


@dataclass(frozen=True)
class NetworkSnapshot:
    available: bool = True
    rtt_ms: float = 15.0
    jitter_ms: float = 3.0
    loss_rate: float = 0.0
    cloud_queue_ms: float = 1.0

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "NetworkSnapshot":
        return cls(
            available=bool(value.get("available", True)),
            rtt_ms=max(0.0, safe_float(value.get("rtt_ms"), 15.0)),
            jitter_ms=max(0.0, safe_float(value.get("jitter_ms"), 3.0)),
            loss_rate=min(1.0, max(0.0, safe_float(value.get("loss_rate"), 0.0))),
            cloud_queue_ms=max(0.0, safe_float(value.get("cloud_queue_ms"), 1.0)),
        )


@dataclass(frozen=True)
class ScheduleDecision:
    route: str
    reason: str
    predicted_sync_e2e_ms: float
    immediate_deadline_ms: float
    cloud_requested: bool
    waits_for_cloud: bool
    risk_prediction_set: List[str]
    risk_set_size: int
    uncertainty_source: str
    defer_recommended: bool
    selective_defer: bool
    network: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdaptiveScheduler:
    def __init__(
        self,
        deadline_ms: float = 200.0,
        confidence_threshold: float = 0.75,
        edge_compute_ms: float = 47.0,
        cloud_compute_ms: float = 12.0,
        jitter_guard: float = 1.645,
    ) -> None:
        self.deadline_ms = float(deadline_ms)
        self.confidence_threshold = float(confidence_threshold)
        self.edge_compute_ms = float(edge_compute_ms)
        self.cloud_compute_ms = float(cloud_compute_ms)
        self.jitter_guard = float(jitter_guard)

    def schedule(
        self,
        event: Dict[str, Any],
        student_confidence: float,
        network: NetworkSnapshot,
        conflict_suspected: bool = False,
        model_disagreement: bool = False,
        defer_recommended: bool = False,
        selective_defer: bool = False,
    ) -> ScheduleDecision:
        summary = event.get("region_summary", {})
        if not isinstance(summary, dict):
            summary = {}
        region_level = str(summary.get("region_risk_level", "low"))
        max_node_level = str(summary.get("max_node_risk_level", region_level))
        region_priority = RISK_PRIORITY.get(region_level, 0)
        max_node_priority = RISK_PRIORITY.get(max_node_level, 0)
        risk_priority = (
            region_priority
            if selective_defer
            else max(region_priority, max_node_priority)
        )
        risk_confidence = safe_float(summary.get("region_risk_confidence"), 1.0)
        calibration = summary.get("region_risk_calibration")
        has_calibrated_set = isinstance(calibration, dict)
        prediction_set = (
            [str(name) for name in calibration.get("prediction_set", [])]
            if has_calibrated_set
            else [region_level]
        )
        if not prediction_set:
            raise ValueError("calibrated region risk set must not be empty.")
        invalid_classes = [name for name in prediction_set if name not in RISK_PRIORITY]
        if invalid_classes:
            raise ValueError("calibrated region risk set has unknown classes: {}.".format(invalid_classes))
        risk_set_size = len(prediction_set)
        calibrated_ambiguous = has_calibrated_set and risk_set_size > 1
        calibrated_critical = has_calibrated_set and any(
            RISK_PRIORITY[name] >= RISK_PRIORITY["high"] for name in prediction_set
        )
        calibrated_severe = has_calibrated_set and "severe" in prediction_set
        effective_confidence = min(float(student_confidence), risk_confidence)
        predicted = (
            self.edge_compute_ms
            + network.rtt_ms
            + self.jitter_guard * network.jitter_ms
            + network.cloud_queue_ms
            + self.cloud_compute_ms
        )
        sync_feasible = (
            network.available
            and network.loss_rate < 0.20
            and predicted <= self.deadline_ms
        )
        point_critical = risk_priority >= RISK_PRIORITY["high"]
        point_severe = risk_priority >= RISK_PRIORITY["severe"]
        point_high = risk_priority == RISK_PRIORITY["high"]
        critical = (
            point_severe or calibrated_severe
            if selective_defer
            else point_critical or calibrated_critical
        )
        if has_calibrated_set:
            uncertain = (
                float(student_confidence) < self.confidence_threshold
                or calibrated_ambiguous
                or defer_recommended
            )
            uncertainty_source = "conformal_set_and_defer_gate" if selective_defer else "conformal_set_and_student"
        else:
            uncertain = effective_confidence < self.confidence_threshold or defer_recommended
            uncertainty_source = "raw_softmax_and_defer_gate" if selective_defer else "raw_softmax_and_student"
        upload_hint = bool(event.get("upload_required", False))

        if not network.available or network.loss_rate >= 0.95:
            route = "local_autonomy"
            reason = "cloud unavailable; keep business on the local edge policy"
        elif conflict_suspected and sync_feasible:
            route = "cloud_sync"
            reason = "cross-region conflict requires synchronous global coordination"
        elif conflict_suspected:
            route = "cloud_async"
            reason = "conflict exists but cloud round trip may miss the real-time deadline"
        elif calibrated_critical and not point_critical and sync_feasible:
            route = "cloud_sync"
            reason = "calibrated risk set includes a possible high-risk state"
        elif calibrated_critical and not point_critical:
            route = "cloud_async"
            reason = "possible high-risk state uses local action and asynchronous cloud review"
        elif critical and sync_feasible:
            route = "cloud_sync"
            reason = "high-risk event can complete cloud coordination before the deadline"
        elif critical:
            route = "cloud_async"
            reason = "high-risk event uses immediate edge action and asynchronous cloud review"
        elif model_disagreement:
            route = "cloud_async"
            reason = "edge models disagree; execute locally and request asynchronous cloud review"
        elif uncertain and sync_feasible:
            route = "cloud_sync"
            reason = "selective edge result requires cloud review"
        elif selective_defer and point_high and not uncertain:
            route = "edge_only"
            reason = "calibrated high-risk state is covered by a confident local expert"
        elif upload_hint and not sync_feasible:
            route = "cloud_async"
            reason = "upload requested but synchronous cloud response is not deadline-feasible"
        elif not sync_feasible:
            route = "local_autonomy"
            reason = "cloud path cannot meet the deadline; execute the validated local safety policy"
        else:
            route = "edge_only"
            reason = "non-critical high-confidence event is handled locally"

        return ScheduleDecision(
            route=route,
            reason=reason,
            predicted_sync_e2e_ms=round(predicted, 4),
            immediate_deadline_ms=self.deadline_ms,
            cloud_requested=route in {"cloud_sync", "cloud_async"},
            waits_for_cloud=route == "cloud_sync",
            risk_prediction_set=prediction_set,
            risk_set_size=risk_set_size,
            uncertainty_source=uncertainty_source,
            defer_recommended=bool(defer_recommended),
            selective_defer=bool(selective_defer),
            network=asdict(network),
        )
