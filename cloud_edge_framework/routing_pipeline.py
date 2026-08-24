"""Deterministic routing projection for edge/cloud execution.

The deployed routing path has exactly three decision stages:

1. :func:`feasible_action_filter` applies transport/deadline/business hard
   constraints.
2. :func:`route_selector` projects the existing
   :class:`CollaborationScheduler` decision onto that feasible set.
3. :func:`execution_failover` changes an execution state to
   ``local_autonomy`` only when cloud execution is unavailable or fails.

No secondary component may replace this scheduler decision.
"""

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple

from cloud_edge_framework.scheduling import ScheduleDecision


PIPELINE_VERSION = "simplified-routing/v1"
ACTIONS = ("edge_only", "cloud_async", "cloud_sync")


@dataclass(frozen=True)
class FeasibleActionResult:
    feasible_actions: Tuple[str, ...]
    rejected_actions: Dict[str, str]
    cloud_available: bool
    predicted_sync_ms: float
    sync_deadline_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteSelection:
    selected_action: str
    preferred_action: str
    rule_action: str
    selector_authority: str
    reason_code: str
    reason: str
    feasibility: FeasibleActionResult

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["pipeline_version"] = PIPELINE_VERSION
        return value


@dataclass(frozen=True)
class ExecutionFailoverResult:
    selected_action: str
    execution_route: str
    triggered: bool
    reason_code: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def feasible_action_filter(
    *,
    cloud_available: bool,
    predicted_sync_ms: float,
    sync_deadline_ms: float,
    prohibited_actions: Optional[Mapping[str, str]] = None,
) -> FeasibleActionResult:
    """Apply the complete hard-constraint set exactly once.

    ``prohibited_actions`` is reserved for documented business invariants,
    such as a caller forbidding a blocking response or a scene contract
    requiring summary delivery.  It must not be populated from model risk,
    confidence, entropy, or communication-cost thresholds.
    """

    predicted = max(0.0, float(predicted_sync_ms))
    deadline = max(0.0, float(sync_deadline_ms))
    rejected: Dict[str, str] = {}

    if not bool(cloud_available):
        rejected["cloud_async"] = "CLOUD_UNAVAILABLE"
        rejected["cloud_sync"] = "CLOUD_UNAVAILABLE"
    elif predicted > deadline:
        rejected["cloud_sync"] = "SYNCHRONOUS_DEADLINE_UNSATISFIABLE"

    for action, reason in dict(prohibited_actions or {}).items():
        if action not in ACTIONS:
            raise ValueError("business prohibition contains an unknown action")
        reason_code = str(reason).strip()
        if not reason_code:
            raise ValueError("business prohibition reason must not be empty")
        rejected[action] = reason_code

    feasible = tuple(action for action in ACTIONS if action not in rejected)
    if not feasible:
        raise ValueError("hard constraints removed every routing action")
    return FeasibleActionResult(
        feasible_actions=feasible,
        rejected_actions=rejected,
        cloud_available=bool(cloud_available),
        predicted_sync_ms=predicted,
        sync_deadline_ms=deadline,
    )


def _project_to_feasible(
    action: str, feasibility: FeasibleActionResult
) -> Tuple[str, Optional[str]]:
    """Project a preference through the already-computed feasibility mask."""

    preferred = str(action)
    feasible = feasibility.feasible_actions
    if preferred == "local_autonomy":
        # local_autonomy is an execution state, never a selectable action.
        if "edge_only" not in feasible:
            raise ValueError("local autonomy requires a feasible edge action")
        return "edge_only", "RULE_LOCAL_AUTONOMY_CANONICALIZED"
    if preferred in feasible:
        return preferred, None
    if preferred == "cloud_sync" and "cloud_async" in feasible:
        return "cloud_async", "PREFERRED_ACTION_INFEASIBLE"
    if "edge_only" in feasible:
        return "edge_only", "PREFERRED_ACTION_INFEASIBLE"
    return feasible[0], "PREFERRED_ACTION_INFEASIBLE"


def route_selector(
    *,
    rule_decision: ScheduleDecision,
    feasibility: FeasibleActionResult,
) -> RouteSelection:
    """Project the CollaborationScheduler route through hard constraints."""

    raw_rule_action = str(rule_decision.route)
    rule_action, rule_projection = _project_to_feasible(
        raw_rule_action, feasibility
    )
    reason_code = rule_projection or "RULE_ROUTER_SELECTED"

    return RouteSelection(
        selected_action=rule_action,
        preferred_action=raw_rule_action,
        rule_action=raw_rule_action,
        selector_authority="CollaborationScheduler",
        reason_code=reason_code,
        reason=str(rule_decision.reason),
        feasibility=feasibility,
    )


def execution_failover(
    selected_action: str,
    *,
    cloud_available: bool,
    cloud_execution_failed: bool = False,
    failure_reason_code: Optional[str] = None,
) -> ExecutionFailoverResult:
    """Map a selected action to its runtime route without re-running policy."""

    selected = str(selected_action)
    if selected not in ACTIONS:
        raise ValueError("execution failover received an unknown selected action")
    cloud_action = selected in {"cloud_async", "cloud_sync"}
    if not cloud_available:
        return ExecutionFailoverResult(
            selected_action=selected,
            execution_route="local_autonomy",
            triggered=True,
            reason_code="CLOUD_UNAVAILABLE",
        )
    if cloud_action and cloud_execution_failed:
        return ExecutionFailoverResult(
            selected_action=selected,
            execution_route="local_autonomy",
            triggered=True,
            reason_code=str(failure_reason_code or "CLOUD_EXECUTION_FAILED"),
        )
    return ExecutionFailoverResult(
        selected_action=selected,
        execution_route=selected,
        triggered=False,
        reason_code=None,
    )


def apply_route_selection(
    rule_decision: ScheduleDecision,
    selection: RouteSelection,
    failover: ExecutionFailoverResult,
) -> ScheduleDecision:
    """Materialize the single routing decision in the legacy response schema."""

    reason = selection.reason
    if failover.triggered:
        reason = "{}; execution failover: {}".format(
            reason, failover.reason_code
        )
    return replace(
        rule_decision,
        route=failover.execution_route,
        reason=reason,
        cloud_requested=failover.execution_route in {"cloud_async", "cloud_sync"},
        waits_for_cloud=failover.execution_route == "cloud_sync",
    )
