"""用途：按实体、状态变量、时间窗和共享执行资源检测并消解跨边缘冲突。"""

import itertools
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Sequence, Set, Tuple

from cloud_edge_framework.contracts import Action, DecisionEnvelope, SemanticEvent, stable_id
from cloud_edge_framework.registry import SceneRegistry


@dataclass(frozen=True)
class ConflictRecord:
    conflict_id: str
    kind: str
    scene: str
    left_index: int
    right_index: int
    left_event_id: str
    right_event_id: str
    left_action_index: int
    right_action_index: int
    shared_resources: List[str]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoordinationResult:
    decisions: List[DecisionEnvelope]
    initial_conflicts: List[ConflictRecord]
    residual_conflicts: List[ConflictRecord]
    changes: List[Dict[str, Any]]
    rounds: int

    def to_dict(self) -> Dict[str, Any]:
        initial_count = len(self.initial_conflicts)
        residual_count = len(self.residual_conflicts)
        return {
            "decisions": [decision.to_dict() for decision in self.decisions],
            "initial_conflicts": [conflict.to_dict() for conflict in self.initial_conflicts],
            "residual_conflicts": [conflict.to_dict() for conflict in self.residual_conflicts],
            "initial_conflict_count": initial_count,
            "residual_conflict_count": residual_count,
            "resolution_success_rate": round(
                (initial_count - residual_count) / initial_count, 6
            )
            if initial_count
            else 1.0,
            "globally_consistent": residual_count == 0,
            "rounds": self.rounds,
            "changes": list(self.changes),
        }


def _time_related(left: SemanticEvent, right: SemanticEvent, tolerance_ms: int) -> bool:
    return not (
        left.scope.window_end_ms + tolerance_ms < right.scope.window_start_ms
        or right.scope.window_end_ms + tolerance_ms < left.scope.window_start_ms
    )


def scopes_correlated(
    left: SemanticEvent,
    right: SemanticEvent,
    tolerance_ms: int = 5000,
) -> bool:
    if left.scene != right.scene or not _time_related(left, right, tolerance_ms):
        return False
    if left.scope.state_variable != right.scope.state_variable:
        return False
    same_entity = left.scope.entity_id == right.scope.entity_id
    shared_scope_resource = bool(
        set(left.scope.shared_resources) & set(right.scope.shared_resources)
    )
    shared_key = bool(set(left.scope.correlation_keys) & set(right.scope.correlation_keys))
    return same_entity or shared_scope_resource or shared_key


def correlation_groups(events: Sequence[SemanticEvent], tolerance_ms: int = 5000) -> List[List[int]]:
    parents = list(range(len(events)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left, right in itertools.combinations(range(len(events)), 2):
        if scopes_correlated(events[left], events[right], tolerance_ms):
            union(left, right)
    groups: Dict[int, List[int]] = {}
    for index in range(len(events)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def _resources(action: Action) -> Set[str]:
    return set(action.resource_ids or action.target_ids)


def _version_key(value: str) -> Tuple[int, ...]:
    values = [int(part) for part in re.findall(r"\d+", value)]
    return tuple(values or [0])


class ConflictCoordinator:
    def __init__(self, registry: SceneRegistry, tolerance_ms: int = 5000) -> None:
        self.registry = registry
        self.tolerance_ms = int(tolerance_ms)

    def detect(
        self,
        events: Sequence[SemanticEvent],
        decisions: Sequence[DecisionEnvelope],
    ) -> List[ConflictRecord]:
        if len(events) != len(decisions):
            raise ValueError("events and decisions must have the same length")
        conflicts: List[ConflictRecord] = []
        for left_index, right_index in itertools.combinations(range(len(events)), 2):
            left_event = events[left_index]
            right_event = events[right_index]
            if left_event.scene != right_event.scene:
                continue
            if not _time_related(left_event, right_event, self.tolerance_ms):
                continue
            left_decision = decisions[left_index]
            right_decision = decisions[right_index]
            scope_related = scopes_correlated(
                left_event, right_event, self.tolerance_ms
            )
            if (
                scope_related
                and left_decision.policy_version != right_decision.policy_version
            ):
                conflicts.append(
                    ConflictRecord(
                        conflict_id=stable_id(
                            "conflict",
                            left_event.event_id,
                            right_event.event_id,
                            "policy_version_mismatch",
                        ),
                        kind="policy_version_mismatch",
                        scene=left_event.scene,
                        left_index=left_index,
                        right_index=right_index,
                        left_event_id=left_event.event_id,
                        right_event_id=right_event.event_id,
                        left_action_index=-1,
                        right_action_index=-1,
                        shared_resources=sorted(
                            set(left_event.scope.shared_resources)
                            & set(right_event.scope.shared_resources)
                        ),
                        details={
                            "left_version": left_decision.policy_version,
                            "right_version": right_decision.policy_version,
                        },
                    )
                )

            plugin = self.registry.get(left_event.scene)
            for left_action_index, left_action in enumerate(left_decision.actions):
                for right_action_index, right_action in enumerate(right_decision.actions):
                    shared_resources = _resources(left_action) & _resources(right_action)
                    if not shared_resources:
                        continue
                    incompatible, kind = plugin.action_conflict(left_action, right_action)
                    if not incompatible:
                        continue
                    conflicts.append(
                        ConflictRecord(
                            conflict_id=stable_id(
                                "conflict",
                                left_event.event_id,
                                right_event.event_id,
                                left_action_index,
                                right_action_index,
                                kind,
                            ),
                            kind=kind,
                            scene=left_event.scene,
                            left_index=left_index,
                            right_index=right_index,
                            left_event_id=left_event.event_id,
                            right_event_id=right_event.event_id,
                            left_action_index=left_action_index,
                            right_action_index=right_action_index,
                            shared_resources=sorted(shared_resources),
                            details={
                                "scope_correlated": scope_related,
                                "left_action": left_action.action_type,
                                "right_action": right_action.action_type,
                            },
                        )
                    )
        return conflicts

    def coordinate(
        self,
        events: Sequence[SemanticEvent],
        decisions: Sequence[DecisionEnvelope],
        max_rounds: int = 8,
    ) -> CoordinationResult:
        coordinated = list(decisions)
        initial = self.detect(events, coordinated)
        changes: List[Dict[str, Any]] = []
        rounds = 0
        for round_index in range(max_rounds):
            current = self.detect(events, coordinated)
            if not current:
                break
            rounds = round_index + 1
            for conflict in current:
                left_index = conflict.left_index
                right_index = conflict.right_index
                left_decision = coordinated[left_index]
                right_decision = coordinated[right_index]
                if conflict.kind == "policy_version_mismatch":
                    version = max(
                        [left_decision.policy_version, right_decision.policy_version],
                        key=_version_key,
                    )
                    coordinated[left_index] = replace(left_decision, policy_version=version)
                    coordinated[right_index] = replace(right_decision, policy_version=version)
                    reason = "cloud synchronized the policy version"
                else:
                    plugin = self.registry.get(conflict.scene)
                    left_actions = list(left_decision.actions)
                    right_actions = list(right_decision.actions)
                    left_action, right_action, reason = plugin.resolve_action_conflict(
                        left_actions[conflict.left_action_index],
                        right_actions[conflict.right_action_index],
                        events[left_index],
                        events[right_index],
                    )
                    left_actions[conflict.left_action_index] = left_action
                    right_actions[conflict.right_action_index] = right_action
                    left_metadata = dict(left_decision.metadata)
                    right_metadata = dict(right_decision.metadata)
                    left_metadata["globally_coordinated"] = True
                    right_metadata["globally_coordinated"] = True
                    coordinated[left_index] = replace(
                        left_decision,
                        actions=left_actions,
                        reason="global conflict coordination",
                        metadata=left_metadata,
                    )
                    coordinated[right_index] = replace(
                        right_decision,
                        actions=right_actions,
                        reason="global conflict coordination",
                        metadata=right_metadata,
                    )
                changes.append(
                    {
                        "conflict_id": conflict.conflict_id,
                        "kind": conflict.kind,
                        "left_event_id": conflict.left_event_id,
                        "right_event_id": conflict.right_event_id,
                        "reason": reason,
                    }
                )
        residual = self.detect(events, coordinated)
        return CoordinationResult(
            decisions=coordinated,
            initial_conflicts=initial,
            residual_conflicts=residual,
            changes=changes,
            rounds=rounds,
        )
