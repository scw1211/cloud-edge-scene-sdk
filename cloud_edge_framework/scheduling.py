"""用途：依据风险、载荷、网络状态、历史实测和闭环预算选择计算路径。"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from cloud_edge_framework.contracts import SemanticEvent


RISK_PRIORITY = {"low": 0, "medium": 1, "high": 2, "severe": 3}


@dataclass(frozen=True)
class NetworkSnapshot:
    available: bool = True
    rtt_ms: float = 15.0
    jitter_ms: float = 3.0
    loss_rate: float = 0.0
    cloud_queue_ms: float = 1.0
    cloud_compute_ms: float = 12.0
    uplink_mbps: float = 100.0
    downlink_mbps: float = 100.0
    expected_response_bytes: int = 2048

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "NetworkSnapshot":
        if not isinstance(value, dict):
            raise ValueError("network must be an object")

        def number(name: str, default: float, low: float, high: float = float("inf")) -> float:
            try:
                result = float(value.get(name, default))
            except (TypeError, ValueError) as exc:
                raise ValueError("network.{} must be numeric".format(name)) from exc
            if result < low or result > high:
                raise ValueError("network.{} is outside [{}, {}]".format(name, low, high))
            return result

        return cls(
            available=bool(value.get("available", True)),
            rtt_ms=number("rtt_ms", 15.0, 0.0),
            jitter_ms=number("jitter_ms", 3.0, 0.0),
            loss_rate=number("loss_rate", 0.0, 0.0, 1.0),
            cloud_queue_ms=number("cloud_queue_ms", 1.0, 0.0),
            cloud_compute_ms=number("cloud_compute_ms", 12.0, 0.0),
            uplink_mbps=number("uplink_mbps", 100.0, 0.001),
            downlink_mbps=number("downlink_mbps", 100.0, 0.001),
            expected_response_bytes=int(
                number("expected_response_bytes", 2048.0, 0.0)
            ),
        )


@dataclass(frozen=True)
class ScheduleDecision:
    route: str
    reason: str
    predicted_closed_loop_ms: float
    deadline_ms: float
    cloud_requested: bool
    waits_for_cloud: bool
    uncertain: bool
    critical: bool
    explicit_cloud_review_requested: bool
    sla_probe_requested: bool
    evidence_level: str
    upload_bytes: int
    estimated_transfer_ms: float
    analytic_cloud_path_ms: float
    profile_source: str
    network: Dict[str, Any]
    selective_defer: bool
    defer_recommended: bool
    routing_risk_level: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CollaborationScheduler:
    def __init__(self, confidence_threshold: float = 0.75, jitter_guard: float = 1.645) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.jitter_guard = float(jitter_guard)

    def schedule(
        self,
        event: SemanticEvent,
        network: NetworkSnapshot,
        conflict_suspected: bool = False,
        model_disagreement: bool = False,
        cloud_review_requested: bool = False,
        sla_probe_requested: bool = False,
        upload_bytes: int = 0,
        evidence_level: str = "summary",
        measured_cloud_path_ms: Optional[float] = None,
        selective_defer: bool = False,
        defer_recommended: bool = False,
        routing_risk_level: Optional[str] = None,
        decision_uncertain: bool = False,
    ) -> ScheduleDecision:
        upload_bytes = max(0, int(upload_bytes))
        transfer_ms = (
            upload_bytes * 8.0 / (network.uplink_mbps * 1_000_000.0) * 1000.0
            + network.expected_response_bytes
            * 8.0
            / (network.downlink_mbps * 1_000_000.0)
            * 1000.0
        )
        analytic_cloud_path_ms = (
            network.rtt_ms
            + self.jitter_guard * network.jitter_ms
            + transfer_ms
            + network.cloud_queue_ms
            + network.cloud_compute_ms
        )
        if measured_cloud_path_ms is not None and measured_cloud_path_ms >= 0.0:
            cloud_path_ms = 0.7 * float(measured_cloud_path_ms) + 0.3 * analytic_cloud_path_ms
            profile_source = "measured_ewma_blended"
        else:
            cloud_path_ms = analytic_cloud_path_ms
            profile_source = "network_snapshot"
        predicted = (
            event.timing.preprocessing_ms
            + event.timing.edge_inference_ms
            + cloud_path_ms
        )
        prediction_set = event.uncertainty.prediction_set or [event.risk.level]
        if selective_defer:
            # The scene gate already consumes model confidence. Reapplying the
            # generic confidence threshold here would undo learning-to-defer and
            # force nearly every traffic event back to the cloud. A conformal
            # set, when present, remains an independent ambiguity signal.
            uncertain = len(prediction_set) > 1 or defer_recommended
        else:
            uncertain = (
                event.uncertainty.confidence < self.confidence_threshold
                or event.prediction.confidence < self.confidence_threshold
                or len(prediction_set) > 1
                or defer_recommended
            )
        # Scene-local models can expose uncertainty after normalization (for
        # example Student confidence or Student/rule disagreement).  It is a
        # first-class scheduling signal and must not be lost merely because the
        # upstream perception confidence was high.
        uncertain = bool(uncertain or decision_uncertain)
        possible_high = any(
            RISK_PRIORITY.get(level, 0) >= RISK_PRIORITY["high"]
            for level in prediction_set
        )
        possible_severe = "severe" in prediction_set
        has_explicit_routing_risk = routing_risk_level is not None
        route_risk_level = str(routing_risk_level or event.risk.level)
        if route_risk_level not in RISK_PRIORITY:
            raise ValueError("routing_risk_level is invalid")
        route_risk_priority = RISK_PRIORITY[route_risk_level]
        point_critical = route_risk_priority >= RISK_PRIORITY["high"]
        if selective_defer:
            critical = (
                route_risk_priority >= RISK_PRIORITY["severe"] or possible_severe
            )
        elif has_explicit_routing_risk:
            # ``routing_risk_level`` may deliberately separate operational
            # action safety from a scene's descriptive state (for example,
            # severe congestion with only an advisory action). A possible high
            # label is an escalation signal when it is ambiguous/uncertain, but
            # must not override a confident low operational-risk assessment by
            # itself. Generic plugins that do not provide an override still use
            # event.risk here, so their high/severe semantics are unchanged.
            critical = point_critical or (possible_high and uncertain)
        else:
            # Without an explicit scene-level operational-risk override, the
            # conformal set is part of the generic safety boundary even when a
            # point confidence happens to be high.
            critical = point_critical or possible_high
        sync_feasible = (
            network.available
            and network.loss_rate < 0.20
            and predicted <= event.timing.deadline_ms
        )

        if not network.available or network.loss_rate >= 0.95:
            route = "local_autonomy"
            reason = "cloud is unavailable; execute the local scene safety policy"
        elif sla_probe_requested and network.loss_rate < 0.20:
            route = "cloud_sync"
            reason = "bounded SLA probe measures the complete Edge-Qwen and cloud path"
        elif conflict_suspected and sync_feasible:
            route = "cloud_sync"
            reason = "correlated edge decisions require synchronous cloud coordination"
        elif conflict_suspected:
            route = "cloud_async"
            reason = "conflict review is required but the cloud loop cannot meet the deadline"
        elif cloud_review_requested and sync_feasible:
            route = "cloud_sync"
            reason = "the scene policy explicitly requests synchronous cloud verification"
        elif cloud_review_requested:
            route = "cloud_async"
            reason = "the scene policy requests cloud verification outside the synchronous budget"
        elif (
            selective_defer
            and possible_high
            and not point_critical
            and sync_feasible
        ):
            route = "cloud_sync"
            reason = "the calibrated risk set includes a possible high-risk state"
        elif selective_defer and possible_high and not point_critical:
            route = "cloud_async"
            reason = "possible high-risk state uses local action and asynchronous cloud review"
        elif critical and sync_feasible:
            route = "cloud_sync"
            reason = "critical event can finish cloud verification within the deadline"
        elif critical:
            route = "cloud_async"
            reason = "critical event executes locally and queues cloud verification"
        elif (uncertain or model_disagreement) and sync_feasible:
            route = "cloud_sync"
            reason = "uncertain local result requires cloud verification"
        elif uncertain or model_disagreement:
            route = "cloud_async"
            reason = "uncertain result uses a provisional local action and asynchronous review"
        elif selective_defer and route_risk_level == "high":
            route = "edge_only"
            reason = "calibrated high-risk state is covered by a confident local expert"
        else:
            route = "edge_only"
            reason = "stable non-critical event is handled by the edge model"

        return ScheduleDecision(
            route=route,
            reason=reason,
            predicted_closed_loop_ms=round(predicted, 6),
            deadline_ms=event.timing.deadline_ms,
            cloud_requested=route in {"cloud_sync", "cloud_async"},
            waits_for_cloud=route == "cloud_sync",
            uncertain=uncertain,
            critical=critical,
            explicit_cloud_review_requested=bool(cloud_review_requested),
            sla_probe_requested=bool(sla_probe_requested),
            evidence_level=evidence_level,
            upload_bytes=upload_bytes,
            estimated_transfer_ms=round(transfer_ms, 6),
            analytic_cloud_path_ms=round(analytic_cloud_path_ms, 6),
            profile_source=profile_source,
            network=asdict(network),
            selective_defer=bool(selective_defer),
            defer_recommended=bool(defer_recommended),
            routing_risk_level=route_risk_level,
        )
