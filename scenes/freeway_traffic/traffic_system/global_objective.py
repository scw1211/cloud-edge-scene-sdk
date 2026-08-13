"""Finite-candidate joint traffic utility used by the cloud coordinator.

The score is an auditable surrogate over the actions already proposed by the
scene models.  It is not a traffic simulator and must not be reported as an
optimum of the physical road network.  With four METIS regions the current
candidate set is small enough to enumerate exactly.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from cloud_edge_framework.contracts import Action, DecisionEnvelope, SemanticEvent


ActionConflict = Callable[[Action, Action], Tuple[bool, str]]


_DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": 1,
    "objective_id": "traffic_candidate_global_utility_v1",
    "expected_members": [
        "edge_node_0",
        "edge_node_1",
        "edge_node_2",
        "edge_node_3",
    ],
    "weights": {
        "delay_reduction": 0.45,
        "queue_reduction": 0.30,
        "throughput_gain": 0.15,
        "risk_relief": 0.10,
        "switching_cost": 0.20,
        "conflict_penalty": 2.0,
        "capability_penalty": 10.0,
        "safety_penalty": 10.0,
    },
    "action_effect": {
        "traffic_advisory": 0.0,
        "variable_speed_limit": 0.25,
        "ramp_metering": 0.30,
        "regional_coordination": 0.32,
        "reroute": 0.45,
    },
    "traffic_normalization": {
        "reference_speed": 68.0,
        "flow_capacity": 600.0,
        "occupancy_capacity": 0.22,
    },
    "unknown_switching_cost": 0.10,
    "maximum_combined_diversion_ratio": 0.50,
    "parameter_bounds": {
        "target_speed_mph": [25.0, 65.0],
        "metering_rate_veh_per_hour": [120.0, 900.0],
        "diversion_ratio": [0.0, 0.50],
    },
    "max_exact_candidates": 256,
    "beam_width": 256,
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError("{} must be numeric".format(field))
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(field))
    return result


def _bounded(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = 0.0
    if not math.isfinite(result):
        result = 0.0
    return max(low, min(high, result))


def _action_signature(action: Action) -> Dict[str, Any]:
    return {
        "action_type": action.action_type,
        "target_ids": sorted(action.target_ids),
        "resource_ids": sorted(action.resource_ids),
        "parameters": dict(action.parameters),
    }


def _bundle_signature(actions: Sequence[Action]) -> str:
    return _sha256_json([_action_signature(action) for action in actions])


def _node_ids(actions: Sequence[Action]) -> List[int]:
    result: List[int] = []
    for action in actions:
        for target in action.target_ids:
            prefix = "traffic_node:"
            if not str(target).startswith(prefix):
                continue
            try:
                node = int(str(target)[len(prefix) :])
            except ValueError:
                continue
            if node not in result:
                result.append(node)
    return result


def _decision_name(actions: Sequence[Action], risk_level: str) -> str:
    if not actions:
        return "no_action" if risk_level == "low" else "monitor"
    selected = max(actions, key=lambda action: action.priority)
    if selected.action_type == "traffic_advisory":
        return "congestion_warning"
    return selected.action_type


class TrafficGlobalObjective:
    """Score and select a joint action bundle for one aggregation group."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        merged = json.loads(json.dumps(_DEFAULT_CONFIG))
        for key, value in dict(config).items():
            if key in {
                "weights",
                "action_effect",
                "parameter_bounds",
                "traffic_normalization",
            }:
                if not isinstance(value, dict):
                    raise ValueError("global objective {} must be an object".format(key))
                merged[key].update(value)
            else:
                merged[key] = value
        if int(merged.get("schema_version", 0)) != 1:
            raise ValueError("traffic global objective schema_version must be 1")
        objective_id = str(merged.get("objective_id", "")).strip()
        if not objective_id:
            raise ValueError("traffic global objective_id must not be empty")
        expected_members = merged.get("expected_members")
        if not isinstance(expected_members, list) or not expected_members:
            raise ValueError("traffic global objective expected_members must be non-empty")
        normalized_members = [str(value).strip() for value in expected_members]
        if (
            any(not value for value in normalized_members)
            or len(set(normalized_members)) != len(normalized_members)
        ):
            raise ValueError(
                "traffic global objective expected_members must be unique non-empty strings"
            )
        merged["expected_members"] = normalized_members
        for name, value in merged["weights"].items():
            if _finite_number(value, "weights.{}".format(name)) < 0.0:
                raise ValueError("traffic global objective weights must be non-negative")
        for name, value in merged["action_effect"].items():
            number = _finite_number(value, "action_effect.{}".format(name))
            if not 0.0 <= number <= 1.0:
                raise ValueError("traffic action effects must be in [0, 1]")
        for name, value in merged["traffic_normalization"].items():
            if _finite_number(
                value, "traffic_normalization.{}".format(name)
            ) <= 0.0:
                raise ValueError(
                    "traffic normalization values must be positive"
                )
        for name, value in merged["parameter_bounds"].items():
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("parameter_bounds.{} must contain [low, high]".format(name))
            low = _finite_number(value[0], "parameter_bounds.{}[0]".format(name))
            high = _finite_number(value[1], "parameter_bounds.{}[1]".format(name))
            if high < low:
                raise ValueError("parameter_bounds.{} is reversed".format(name))
            merged["parameter_bounds"][name] = [low, high]
        for name in ("max_exact_candidates", "beam_width"):
            if int(merged[name]) <= 0:
                raise ValueError("{} must be positive".format(name))
            merged[name] = int(merged[name])
        merged["unknown_switching_cost"] = _bounded(
            merged.get("unknown_switching_cost", 0.10)
        )
        merged["maximum_combined_diversion_ratio"] = _bounded(
            merged.get("maximum_combined_diversion_ratio", 0.50)
        )
        self.config = merged
        self.objective_id = objective_id
        self.objective_sha256 = _sha256_json(merged)
        self.definition_file_sha256: Optional[str] = None

    @classmethod
    def from_path(cls, path: Path) -> "TrafficGlobalObjective":
        normalized_path = Path(path)
        with normalized_path.open("r", encoding="utf-8") as file_obj:
            value = json.load(file_obj)
        if not isinstance(value, dict):
            raise ValueError("traffic global objective file must contain an object")
        objective = cls(value)
        objective.definition_file_sha256 = hashlib.sha256(
            normalized_path.read_bytes()
        ).hexdigest()
        return objective

    def _variants(self, decision: DecisionEnvelope) -> List[Tuple[str, List[Action]]]:
        advisory = [
            action
            for action in decision.actions
            if action.action_type == "traffic_advisory"
        ]
        raw = [
            ("no_action", []),
            ("advisory_only", advisory),
            ("model_bundle", list(decision.actions)),
        ]
        variants: List[Tuple[str, List[Action]]] = []
        seen = set()
        for name, actions in raw:
            signature = _bundle_signature(actions)
            if signature in seen:
                continue
            seen.add(signature)
            variants.append((name, list(actions)))
        return variants

    def _action_effect(self, action: Action) -> float:
        maximum = _bounded(
            self.config["action_effect"].get(action.action_type, 0.0)
        )
        if action.action_type == "variable_speed_limit":
            speed = _bounded(
                action.parameters.get("target_speed_mph", 65.0), 25.0, 65.0
            )
            return maximum * _bounded((65.0 - speed) / 40.0)
        if action.action_type == "ramp_metering":
            rate = _bounded(
                action.parameters.get("metering_rate_veh_per_hour", 900.0),
                120.0,
                900.0,
            )
            return maximum * _bounded((900.0 - rate) / 780.0)
        if action.action_type == "reroute":
            ratio = _bounded(action.parameters.get("diversion_ratio", 0.0))
            return maximum * _bounded(ratio / 0.40)
        return maximum

    def _bundle_effect(self, actions: Sequence[Action]) -> float:
        residual = 1.0
        for action in actions:
            residual *= 1.0 - self._action_effect(action)
        return _bounded(1.0 - residual)

    def _traffic_proxies(self, event: SemanticEvent) -> Dict[str, Any]:
        """Build dimensionless traffic proxies from a regional observation."""
        summary = event.scene_payload.get("region_summary")
        observation = (
            summary.get("current_observation")
            if isinstance(summary, dict)
            else None
        )
        if not isinstance(observation, dict):
            return {
                "observed": False,
                "delay": 0.0,
                "queue": 0.0,
                "throughput": 0.0,
            }
        normalization = self.config["traffic_normalization"]
        speed_ratio = _bounded(
            _finite_number(
                observation.get("speed_mean", 0.0),
                "current_observation.speed_mean",
            )
            / normalization["reference_speed"],
            0.0,
            1.5,
        )
        flow_ratio = _bounded(
            _finite_number(
                observation.get("flow_mean", 0.0),
                "current_observation.flow_mean",
            )
            / normalization["flow_capacity"]
        )
        occupancy_ratio = _bounded(
            _finite_number(
                observation.get("occupancy_mean", 0.0),
                "current_observation.occupancy_mean",
            )
            / normalization["occupancy_capacity"]
        )
        speed_deficit = _bounded(1.0 - speed_ratio)
        return {
            "observed": True,
            # Auditable proxies only: these are not a traffic simulation or a
            # claim of optimality over the physical road network.
            "delay": flow_ratio * speed_deficit,
            "queue": occupancy_ratio * (0.5 + 0.5 * speed_deficit),
            "throughput": flow_ratio * min(1.0, speed_ratio),
        }

    def _aggregation_complete(
        self, events: Sequence[SemanticEvent]
    ) -> Tuple[bool, List[str], List[str], str]:
        configured = list(self.config["expected_members"])
        configured_set = set(configured)
        if not events:
            return False, [], configured, "no_events"
        observed = set()
        for event in events:
            raw = event.metadata.get("aggregation")
            if not isinstance(raw, dict):
                return False, sorted(observed), configured, "missing_aggregation_contract"
            if raw.get("authority") != "cloud_aggregation_lease":
                return False, sorted(observed), configured, "untrusted_aggregation_context"
            values = raw.get("expected_members", [])
            if not isinstance(values, list) or not values:
                return False, sorted(observed), configured, "missing_expected_members"
            declared = [str(value).strip() for value in values]
            if len(declared) != len(values) or len(declared) != len(set(declared)):
                return False, sorted(observed), configured, "invalid_expected_members"
            if set(declared) != configured_set:
                return (
                    False,
                    sorted(observed),
                    configured,
                    "expected_member_contract_mismatch",
                )
            received_values = raw.get("received_members")
            if not isinstance(received_values, list):
                return False, sorted(observed), configured, "missing_received_members"
            received = [str(value).strip() for value in received_values]
            missing_values = raw.get("missing_members")
            if not isinstance(missing_values, list):
                return False, sorted(observed), configured, "missing_member_audit"
            missing = [str(value).strip() for value in missing_values]
            if (
                any(not value for value in received + missing)
                or len(set(received)) != len(received)
                or len(set(missing)) != len(missing)
                or set(received) != configured_set
                or missing
                or raw.get("completion_reason") != "all_expected_members"
                or raw.get("evidence_complete") is not True
                or raw.get("finality") != "final"
            ):
                return False, sorted(observed), configured, "incomplete_aggregation_lease"
            member = raw.get("member")
            if member is None or not str(member):
                return False, sorted(observed), configured, "missing_aggregation_member"
            normalized_member = str(member)
            if normalized_member not in configured_set:
                return False, sorted(observed), configured, "unexpected_aggregation_member"
            if normalized_member in observed:
                return False, sorted(observed), configured, "duplicate_aggregation_member"
            observed.add(normalized_member)
        complete = observed == configured_set
        return (
            complete,
            sorted(observed),
            configured,
            "complete" if complete else "missing_expected_members",
        )

    def _switching_cost(
        self,
        event: SemanticEvent,
        actions: Sequence[Action],
    ) -> Tuple[float, bool]:
        current = event.scene_payload.get("current_control_state")
        if not isinstance(current, dict):
            control_count = sum(
                action.action_type != "traffic_advisory" for action in actions
            )
            return (
                self.config["unknown_switching_cost"] * min(1.0, control_count),
                False,
            )
        raw_actions = current.get("actions", [])
        if not isinstance(raw_actions, list):
            raw_actions = []
        current_types = {
            str(value.get("action_type", value.get("type", "")))
            for value in raw_actions
            if isinstance(value, dict)
        }
        selected_types = {
            action.action_type
            for action in actions
            if action.action_type != "traffic_advisory"
        }
        union = current_types | selected_types
        if not union:
            return 0.0, True
        return len(current_types ^ selected_types) / len(union), True

    def _capability_violations(
        self,
        event: SemanticEvent,
        actions: Sequence[Action],
    ) -> Tuple[int, int]:
        capabilities = event.scene_payload.get("control_capabilities")
        if not isinstance(capabilities, dict):
            return 0, 0
        keys = {
            "variable_speed_limit": "variable_speed_limit_nodes",
            "ramp_metering": "ramp_meter_nodes",
            "reroute": "reroute_gateway_nodes",
        }
        violations = 0
        observed = 0
        for action in actions:
            key = keys.get(action.action_type)
            if key is None or key not in capabilities:
                continue
            allowed_raw = capabilities.get(key)
            if not isinstance(allowed_raw, list):
                continue
            observed += 1
            allowed = set()
            for value in allowed_raw:
                try:
                    allowed.add(int(value))
                except (TypeError, ValueError):
                    continue
            targets = _node_ids([action])
            if targets and not set(targets).issubset(allowed):
                violations += 1
        return violations, observed

    def _safety_violations(
        self,
        actions_by_event: Sequence[Sequence[Action]],
        complete: bool,
    ) -> Tuple[int, List[str]]:
        violations: List[str] = []
        bounds = self.config["parameter_bounds"]
        combined_diversion = 0.0
        parameter_names = {
            "variable_speed_limit": "target_speed_mph",
            "ramp_metering": "metering_rate_veh_per_hour",
            "reroute": "diversion_ratio",
        }
        for actions in actions_by_event:
            for action in actions:
                if (
                    action.parameters.get("requires_cloud_confirmation") is True
                    and not complete
                ):
                    violations.append("incomplete_cloud_confirmation")
                field = parameter_names.get(action.action_type)
                if field is not None and field in action.parameters:
                    try:
                        value = _finite_number(
                            action.parameters[field],
                            "{}.{}".format(action.action_type, field),
                        )
                    except (TypeError, ValueError):
                        violations.append("invalid_{}".format(field))
                        continue
                    low, high = bounds[field]
                    if not low <= value <= high:
                        violations.append("out_of_bounds_{}".format(field))
                if action.action_type == "reroute":
                    combined_diversion += _bounded(
                        action.parameters.get("diversion_ratio", 0.0)
                    )
        if combined_diversion > self.config["maximum_combined_diversion_ratio"] + 1e-9:
            violations.append("combined_diversion_limit")
        return len(violations), violations

    @staticmethod
    def _conflicts(
        actions_by_event: Sequence[Sequence[Action]],
        action_conflict: ActionConflict,
    ) -> Tuple[int, List[str]]:
        count = 0
        kinds: List[str] = []
        for left_index, right_index in itertools.combinations(
            range(len(actions_by_event)), 2
        ):
            for left in actions_by_event[left_index]:
                left_resources = set(left.resource_ids or left.target_ids)
                for right in actions_by_event[right_index]:
                    right_resources = set(right.resource_ids or right.target_ids)
                    if not left_resources.intersection(right_resources):
                        continue
                    incompatible, kind = action_conflict(left, right)
                    if incompatible:
                        count += 1
                        kinds.append(str(kind or "action_conflict"))
        return count, kinds

    def _score(
        self,
        events: Sequence[SemanticEvent],
        actions_by_event: Sequence[Sequence[Action]],
        action_conflict: ActionConflict,
        complete: bool,
    ) -> Dict[str, Any]:
        event_count = max(1, len(events))
        risk_relief = 0.0
        switching_cost = 0.0
        switching_observed = 0
        capability_violations = 0
        capability_checks = 0
        transfer_penalty = 0.0
        delay_reduction = 0.0
        queue_reduction = 0.0
        throughput_gain = 0.0
        traffic_observation_count = 0
        baseline_delay_proxy = 0.0
        baseline_queue_proxy = 0.0
        baseline_throughput_proxy = 0.0
        risk_weights = [
            _bounded(event.risk.score) * _bounded(event.uncertainty.confidence)
            for event in events
        ]
        for index, (event, actions) in enumerate(zip(events, actions_by_event)):
            effect = self._bundle_effect(actions)
            risk_relief += risk_weights[index] * effect
            traffic = self._traffic_proxies(event)
            if traffic["observed"]:
                traffic_observation_count += 1
                baseline_delay_proxy += traffic["delay"]
                baseline_queue_proxy += traffic["queue"]
                baseline_throughput_proxy += traffic["throughput"]
                delay_reduction += traffic["delay"] * effect
                queue_reduction += traffic["queue"] * effect
                throughput_gain += max(
                    0.0, risk_weights[index] - traffic["throughput"]
                ) * effect
            switch, observed = self._switching_cost(event, actions)
            switching_cost += switch
            switching_observed += int(observed)
            violations, checks = self._capability_violations(event, actions)
            capability_violations += violations
            capability_checks += checks
            for action in actions:
                if action.action_type != "reroute":
                    continue
                ratio = _bounded(action.parameters.get("diversion_ratio", 0.0))
                neighbor_risk = statistics_mean(
                    risk_weights[other]
                    for other in range(len(events))
                    if other != index
                )
                transfer_penalty += ratio * neighbor_risk
        risk_relief = max(0.0, risk_relief - transfer_penalty) / event_count
        delay_reduction /= event_count
        queue_reduction /= event_count
        throughput_gain /= event_count
        baseline_delay_proxy /= event_count
        baseline_queue_proxy /= event_count
        baseline_throughput_proxy /= event_count
        switching_cost /= event_count
        conflict_count, conflict_kinds = self._conflicts(
            actions_by_event, action_conflict
        )
        safety_count, safety_reasons = self._safety_violations(
            actions_by_event, complete
        )
        weights = self.config["weights"]
        utility = (
            weights["delay_reduction"] * delay_reduction
            + weights["queue_reduction"] * queue_reduction
            + weights["throughput_gain"] * throughput_gain
            + weights["risk_relief"] * risk_relief
            - weights["switching_cost"] * switching_cost
            - weights["conflict_penalty"] * conflict_count
            - weights["capability_penalty"] * capability_violations
            - weights["safety_penalty"] * safety_count
        )
        feasible = not (
            conflict_count or capability_violations or safety_count
        )
        return {
            "utility": round(float(utility), 9),
            "feasible": feasible,
            "risk_relief": round(float(risk_relief), 9),
            "delay_proxy_reduction": round(float(delay_reduction), 9),
            "queue_proxy_reduction": round(float(queue_reduction), 9),
            "throughput_proxy_gain": round(float(throughput_gain), 9),
            "baseline_delay_proxy": round(float(baseline_delay_proxy), 9),
            "baseline_queue_proxy": round(float(baseline_queue_proxy), 9),
            "baseline_throughput_proxy": round(
                float(baseline_throughput_proxy), 9
            ),
            "traffic_observation_count": traffic_observation_count,
            "traffic_observation_missing_count": (
                len(events) - traffic_observation_count
            ),
            "reroute_transfer_penalty": round(float(transfer_penalty / event_count), 9),
            "switching_cost": round(float(switching_cost), 9),
            "switching_state_observed_count": switching_observed,
            "switching_state_missing_count": len(events) - switching_observed,
            "conflict_count": conflict_count,
            "conflict_kinds": sorted(conflict_kinds),
            "capability_violation_count": capability_violations,
            "capability_check_count": capability_checks,
            "safety_violation_count": safety_count,
            "safety_violations": sorted(safety_reasons),
        }

    def optimize(
        self,
        events: Sequence[SemanticEvent],
        decisions: Sequence[DecisionEnvelope],
        action_conflict: ActionConflict,
        mode: str,
    ) -> Tuple[List[DecisionEnvelope], Dict[str, Any]]:
        normalized_events = list(events)
        normalized_decisions = list(decisions)
        if len(normalized_events) != len(normalized_decisions) or not normalized_events:
            raise ValueError("traffic global optimization requires aligned non-empty inputs")
        if mode not in {"shadow", "active"}:
            raise ValueError("traffic global optimizer mode must be shadow or active")
        variants = [self._variants(decision) for decision in normalized_decisions]
        candidate_space_size = math.prod(len(value) for value in variants)
        (
            complete,
            observed_members,
            expected_members,
            completeness_reason,
        ) = self._aggregation_complete(normalized_events)

        def evaluate(selection: Sequence[int]) -> Dict[str, Any]:
            actions = [variants[index][choice][1] for index, choice in enumerate(selection)]
            score = self._score(
                normalized_events,
                actions,
                action_conflict,
                complete,
            )
            score["selection"] = list(selection)
            score["variants"] = [
                variants[index][choice][0] for index, choice in enumerate(selection)
            ]
            score["action_count"] = sum(len(value) for value in actions)
            score["plan_sha256"] = _sha256_json(
                [[_action_signature(action) for action in value] for value in actions]
            )
            score["actions"] = actions
            return score

        baseline_selection = [
            next(
                index
                for index, (name, _actions) in enumerate(event_variants)
                if name == "model_bundle"
            )
            if any(name == "model_bundle" for name, _actions in event_variants)
            else len(event_variants) - 1
            for event_variants in variants
        ]
        baseline = evaluate(baseline_selection)
        evaluated: List[Dict[str, Any]] = []
        exact = candidate_space_size <= self.config["max_exact_candidates"]
        if exact:
            for selection in itertools.product(
                *(range(len(value)) for value in variants)
            ):
                evaluated.append(evaluate(selection))
            solver = "exact_enumeration"
        else:
            partial: List[Tuple[int, ...]] = [tuple()]
            for event_index, event_variants in enumerate(variants):
                expanded = [
                    selection + (choice,)
                    for selection in partial
                    for choice in range(len(event_variants))
                ]
                padded = [
                    selection
                    + tuple(0 for _ in range(len(variants) - event_index - 1))
                    for selection in expanded
                ]
                ranked = sorted(
                    (evaluate(selection) for selection in padded),
                    key=self._rank_key,
                    reverse=True,
                )[: self.config["beam_width"]]
                partial = [
                    tuple(item["selection"][: event_index + 1]) for item in ranked
                ]
            evaluated = [evaluate(selection) for selection in partial]
            solver = "deterministic_beam"
        selected = max(evaluated, key=self._rank_key)
        apply_selected = bool(mode == "active" and complete)
        output: List[DecisionEnvelope] = []
        for index, decision in enumerate(normalized_decisions):
            selected_actions = list(selected["actions"][index])
            metadata = dict(decision.metadata)
            metadata["global_optimization"] = {
                "objective_id": self.objective_id,
                "objective_sha256": self.objective_sha256,
                "mode": mode,
                "applied": apply_selected,
                "authoritative_input": complete,
                "selected_variant": selected["variants"][index],
                "selected_plan_sha256": selected["plan_sha256"],
            }
            if apply_selected:
                output.append(
                    replace(
                        decision,
                        decision=_decision_name(
                            selected_actions, decision.risk_level
                        ),
                        actions=selected_actions,
                        reason=(
                            "traffic joint candidate utility selected {}"
                        ).format(selected["variants"][index]),
                        metadata=metadata,
                    )
                )
            else:
                output.append(replace(decision, metadata=metadata))
        baseline_public = {
            key: value
            for key, value in baseline.items()
            if key != "actions"
        }
        selected_public = {
            key: value
            for key, value in selected.items()
            if key != "actions"
        }
        metadata = {
            "schema_version": 1,
            "scene": "traffic",
            "objective_id": self.objective_id,
            "objective_sha256": self.objective_sha256,
            "objective_definition_sha256": self.definition_file_sha256,
            "mode": mode,
            "applied": apply_selected,
            "application_blocked_reason": (
                "incomplete_aggregation" if mode == "active" and not complete else None
            ),
            "authoritative_input": complete,
            "observed_members": observed_members,
            "expected_members": expected_members,
            "observed_member_count": len(observed_members),
            "expected_member_count": len(expected_members),
            "aggregation_completeness_reason": completeness_reason,
            "candidate_space_size": candidate_space_size,
            "evaluated_candidate_count": len(evaluated),
            "solver": solver,
            "solver_exact": exact,
            "candidate_set_optimality_verified": exact,
            "candidate_set_complete": exact,
            "exact_search": exact,
            "constraints_satisfied": bool(selected["feasible"]),
            "baseline_utility": float(baseline["utility"]),
            "selected_utility": float(selected["utility"]),
            "baseline": baseline_public,
            "selected": selected_public,
            "utility_delta": round(
                float(selected["utility"] - baseline["utility"]), 9
            ),
            "claim_scope": (
                "finite_candidate_delay_queue_throughput_surrogate_only; "
                "not physical-road-network optimality"
            ),
        }
        return output, metadata

    @staticmethod
    def _rank_key(value: Mapping[str, Any]) -> Tuple[Any, ...]:
        return (
            bool(value["feasible"]),
            float(value["utility"]),
            -float(value["switching_cost"]),
            -int(value["action_count"]),
            str(value["plan_sha256"]),
        )


def statistics_mean(values: Sequence[float]) -> float:
    normalized = list(values)
    return sum(normalized) / len(normalized) if normalized else 0.0
