"""用途：把 ASTGCN 交通事件以紧凑任务特征接入统一云边控制框架。"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from cloud_edge_framework.contracts import (
    Action,
    DecisionEnvelope,
    Evidence,
    EventScope,
    Prediction,
    Risk,
    SemanticEvent,
    Timing,
    Uncertainty,
    build_decision,
)
from cloud_edge_framework.event_envelope import SceneEventEnvelope
from cloud_edge_framework.plugins.base import ScenePlugin
from freeway_traffic_full.edge_llm import TrafficEdgeLLMController
from traffic_system.coordination_policy import (
    ACTION_TO_DECISION,
    DECISION_TO_ACTION,
    CoordinationPolicy,
    verify_policy_package,
)
from traffic_system.global_objective import TrafficGlobalObjective
from traffic_system.road_sets import (
    build_overlap_partial_assessments,
    build_road_set_catalog,
    contexts_for_region,
    fuse_road_set_contexts,
    incident_road_sets,
)
from traffic_system.scene_event import TRAFFIC_DATA_SCHEMA_ID, TRAFFIC_EVENT_TYPE


RISK_PRIORITY = {"low": 0, "medium": 1, "high": 2, "severe": 3}
ACTION_SAFETY_RISK = {
    "traffic_advisory": "low",
    "variable_speed_limit": "medium",
    "ramp_metering": "medium",
    "regional_coordination": "high",
    "reroute": "high",
}
RISK_DEFAULT_SCORE = {"low": 0.2, "medium": 0.5, "high": 0.8, "severe": 1.0}


def _json_size(value: Any) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded_float(
    value: Any,
    default: float = 0.0,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, _safe_float(value, default)))


def _operational_safety_risk(
    payload: Dict[str, Any],
    candidate_actions: Sequence[Action],
) -> Dict[str, Any]:
    """Describe action-consequence risk without reusing traffic congestion severity."""
    action_types = list(dict.fromkeys(action.action_type for action in candidate_actions))
    policy_level = max(
        (ACTION_SAFETY_RISK.get(name, "low") for name in action_types),
        key=RISK_PRIORITY.__getitem__,
        default="low",
    )
    raw = payload.get("operational_safety_risk")
    if raw is not None and not isinstance(raw, dict):
        raise ValueError("traffic operational_safety_risk must be an object")
    if isinstance(raw, dict):
        declared_level = str(raw.get("level", "low"))
        if declared_level not in RISK_PRIORITY:
            raise ValueError("traffic operational safety risk level is invalid")
        level = max(
            (policy_level, declared_level), key=RISK_PRIORITY.__getitem__
        )
        score = _bounded_float(raw.get("score"), RISK_DEFAULT_SCORE[level])
        score = max(score, RISK_DEFAULT_SCORE[level])
        source = "scene_input_with_action_policy_floor"
        declared_source = str(raw.get("source", "scene_input"))
    else:
        declared_level = None
        declared_source = None
        level = policy_level
        score = RISK_DEFAULT_SCORE[level]
        source = "candidate_action_consequence_policy"
    confirmation = "full_cloud" if level in {"high", "severe"} else "local_policy"
    if level == "low":
        confirmation = "local"
    return {
        "level": level,
        "score": round(score, 6),
        "source": source,
        "declared_level": declared_level,
        "declared_source": declared_source,
        "policy_floor_level": policy_level,
        "candidate_action_types": action_types,
        "required_confirmation": confirmation,
    }


def _escalation_expected_gain(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = payload.get("escalation_expected_gain")
    if raw is None:
        return {
            "edge_qwen": 0.0,
            "cloud": 0.0,
            "source": "not_estimated",
        }
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        gain = _bounded_float(raw, 0.0, -1.0, 1.0)
        return {"edge_qwen": gain, "cloud": gain, "source": "scene_input"}
    if not isinstance(raw, dict):
        raise ValueError("traffic escalation_expected_gain must be numeric or an object")
    return {
        "edge_qwen": _bounded_float(raw.get("edge_qwen"), 0.0, -1.0, 1.0),
        "cloud": _bounded_float(raw.get("cloud"), 0.0, -1.0, 1.0),
        "source": str(raw.get("source", "scene_input")),
    }


def _model_uncertainty(
    confidence: float,
    prediction_set: Sequence[str],
    student_confidence_threshold: float,
) -> Dict[str, Any]:
    normalized_set = [str(value) for value in prediction_set]
    ambiguity = 0.0 if len(normalized_set) <= 1 else 1.0 - 1.0 / len(normalized_set)
    score = max(1.0 - confidence, ambiguity)
    return {
        "score": round(_bounded_float(score), 6),
        "perception_confidence": round(_bounded_float(confidence), 6),
        "student_confidence": None,
        "student_confidence_threshold": round(
            _bounded_float(student_confidence_threshold), 6
        ),
        "prediction_set": normalized_set,
        "prediction_set_size": len(normalized_set),
        "student_rule_disagreement": None,
        "requires_review": bool(
            confidence < student_confidence_threshold or len(normalized_set) > 1
        ),
        "source": "perception_calibration",
    }


def _cloud_llm_review_policy(
    payload: Dict[str, Any],
    model_uncertainty: Dict[str, Any],
    expected_gain: Dict[str, Any],
    min_expected_gain: float,
) -> Dict[str, Any]:
    explicit = bool(payload.get("cloud_llm_review_requested", False))
    uncertainty_requires_review = bool(model_uncertainty.get("requires_review", False))
    if "requires_synchronous_review" in model_uncertainty:
        uncertainty_requires_synchronous_review = bool(
            model_uncertainty["requires_synchronous_review"]
        )
    elif "requires_sync_review" in model_uncertainty:
        uncertainty_requires_synchronous_review = bool(
            model_uncertainty["requires_sync_review"]
        )
    else:
        uncertainty_requires_synchronous_review = uncertainty_requires_review
    source = str(expected_gain.get("source", "not_estimated"))
    cloud_gain = _bounded_float(
        expected_gain.get("cloud"), 0.0, -1.0, 1.0
    )
    gain_qualified = (
        source != "not_estimated" and cloud_gain >= min_expected_gain
    )
    eligible = explicit or (
        uncertainty_requires_synchronous_review and gain_qualified
    )
    if explicit:
        reason = "traffic_explicit_cloud_llm_review"
    elif not uncertainty_requires_synchronous_review:
        reason = "traffic_no_model_uncertainty"
    elif source == "not_estimated":
        reason = "traffic_cloud_gain_not_estimated"
    elif not gain_qualified:
        reason = "traffic_cloud_gain_below_threshold"
    else:
        reason = "traffic_uncertainty_with_cloud_gain"
    return {
        "eligible": eligible,
        "reason": reason,
        "explicit_requested": explicit,
        "model_uncertainty_requires_review": uncertainty_requires_review,
        "model_uncertainty_requires_synchronous_review": (
            uncertainty_requires_synchronous_review
        ),
        "expected_gain": round(cloud_gain, 6),
        "expected_gain_source": source,
        "minimum_expected_gain": round(min_expected_gain, 6),
        "legacy_risk_trigger_used": False,
    }


def _evidence_completeness(
    payload: Dict[str, Any],
    evidence: Sequence[Evidence],
    expected_members: Sequence[str],
    aggregation_member: str,
    minimum_level: str,
) -> Dict[str, Any]:
    levels = list(dict.fromkeys(item.level for item in evidence))
    required_available = minimum_level in levels
    members = [str(value) for value in expected_members]
    observed = [aggregation_member] if aggregation_member in members else []
    aggregation_complete = bool(members) and len(observed) == len(members)
    declared = payload.get("evidence_completeness")
    if declared is not None and not isinstance(declared, dict):
        raise ValueError("traffic evidence_completeness must be an object")
    return {
        "available_levels": levels,
        "minimum_required_level": minimum_level,
        "minimum_level_available": required_available,
        "expected_member_count": len(members),
        "observed_member_count": len(observed),
        "aggregation_complete": aggregation_complete,
        "complete": required_available and aggregation_complete,
        "source": "edge_evidence_inventory",
        "declared": dict(declared) if isinstance(declared, dict) else {},
    }


def _traffic_action(action: Dict[str, Any], region_id: str) -> Action:
    action_type = str(action.get("type", "traffic_advisory"))
    raw_targets = list(action.get("target_nodes", [])) + list(action.get("gateway_nodes", []))
    target_ids = ["traffic_node:{}".format(value) for value in raw_targets]
    if not target_ids:
        target_ids = ["traffic_region:{}".format(region_id)]
    resource_ids = list(target_ids)
    if action_type in {"reroute", "regional_coordination"}:
        resource_ids.append("traffic_region:{}".format(region_id))
    ignored = {"type", "target_nodes", "gateway_nodes", "reason"}
    parameters = {key: value for key, value in action.items() if key not in ignored}
    if action_type in {"reroute", "regional_coordination"}:
        parameters.setdefault("requires_cloud_confirmation", True)
    priorities = {
        "traffic_advisory": 30,
        "variable_speed_limit": 60,
        "ramp_metering": 70,
        "regional_coordination": 80,
        "reroute": 90,
    }
    return Action(
        action_type=action_type,
        target_ids=target_ids,
        resource_ids=list(dict.fromkeys(resource_ids)),
        parameters=parameters,
        reason=str(action.get("reason", "traffic control action")),
        priority=priorities.get(action_type, 50),
    )


class TrafficPlugin(ScenePlugin):
    scene = "traffic"
    aliases = ("freeway_traffic_management",)
    event_types = (TRAFFIC_EVENT_TYPE,)
    data_schema_id = TRAFFIC_DATA_SCHEMA_ID

    def __init__(
        self,
        cloud_model_path: Optional[Path] = None,
        edge_student_path: Optional[Path] = None,
        feature_codec_path: Optional[Path] = None,
        topology_path: Optional[Path] = None,
        edge_llm_release_registry_path: Optional[Path] = None,
        edge_llm_runtime_config_path: Optional[Path] = None,
        edge_llm_mode: str = "disabled",
        edge_llm_min_risk_level: str = "medium",
        edge_llm_student_confidence_threshold: float = 0.75,
        edge_llm_deadline_margin_ms: float = 15.0,
        edge_llm_deadline_probe_interval: int = 0,
        edge_llm_runtime_failure_cooldown_seconds: float = 5.0,
        policy_version: str = "traffic-1.9.0",
        edge_llm_min_expected_gain: float = 0.05,
        cloud_llm_min_expected_gain: float = 0.05,
        current_state_edge_student_path: Optional[Path] = None,
        current_state_cloud_model_path: Optional[Path] = None,
        current_state_feature_codec_path: Optional[Path] = None,
        edge_llm_gain_profile_path: Optional[Path] = None,
        current_state_sync_confidence_threshold: float = 0.50,
        global_objective_path: Optional[Path] = None,
        global_optimizer_mode: str = "disabled",
        edge_llm_prompt_prefix: Optional[str] = None,
        coordination_policy_path: Optional[Path] = None,
        coordination_policy_mode: str = "disabled",
        road_set_rule_config_path: Optional[Path] = None,
    ) -> None:
        self.cloud_model_path = Path(cloud_model_path) if cloud_model_path is not None else None
        self.current_state_cloud_model_path = (
            Path(current_state_cloud_model_path)
            if current_state_cloud_model_path is not None
            else None
        )
        self.edge_student_path = (
            Path(edge_student_path) if edge_student_path is not None else None
        )
        self.current_state_edge_student_path = (
            Path(current_state_edge_student_path)
            if current_state_edge_student_path is not None
            else None
        )
        self.feature_codec_path = (
            Path(feature_codec_path) if feature_codec_path is not None else None
        )
        self.current_state_feature_codec_path = (
            Path(current_state_feature_codec_path)
            if current_state_feature_codec_path is not None
            else None
        )
        self.topology_path = Path(topology_path) if topology_path is not None else None
        inferred_road_set_rule_path = (
            self.topology_path.parent / "current_state_perception_v1.json"
            if self.topology_path is not None
            else None
        )
        self.road_set_rule_config_path = (
            Path(road_set_rule_config_path)
            if road_set_rule_config_path is not None
            else inferred_road_set_rule_path
            if inferred_road_set_rule_path is not None
            and inferred_road_set_rule_path.is_file()
            else None
        )
        self._road_set_model_artifact_sha256: Optional[str] = None
        self._road_set_rule_config: Optional[Dict[str, Any]] = None
        self.cloud_llm_min_expected_gain = float(cloud_llm_min_expected_gain)
        if not -1.0 <= self.cloud_llm_min_expected_gain <= 1.0:
            raise ValueError("cloud_llm_min_expected_gain must be in [-1, 1]")
        self.current_state_sync_confidence_threshold = float(
            current_state_sync_confidence_threshold
        )
        if not 0.0 <= self.current_state_sync_confidence_threshold <= 1.0:
            raise ValueError(
                "current_state_sync_confidence_threshold must be in [0, 1]"
            )
        self.policy_version = policy_version
        self.global_objective_path = (
            Path(global_objective_path)
            if global_objective_path is not None
            else None
        )
        self.global_optimizer_mode = str(global_optimizer_mode).strip().lower()
        if self.global_optimizer_mode not in {"disabled", "shadow", "active"}:
            raise ValueError(
                "global_optimizer_mode must be disabled, shadow, or active"
            )
        if self.global_optimizer_mode != "disabled" and self.global_objective_path is None:
            raise ValueError(
                "enabled traffic global optimizer requires global_objective_path"
            )
        self._global_objective: Optional[TrafficGlobalObjective] = None
        self.coordination_policy_path = (
            Path(coordination_policy_path)
            if coordination_policy_path is not None
            else None
        )
        self.coordination_policy_mode = str(
            coordination_policy_mode
        ).strip().lower()
        if self.coordination_policy_mode not in {"disabled", "shadow", "active"}:
            raise ValueError(
                "coordination_policy_mode must be disabled, shadow, or active"
            )
        if (
            self.coordination_policy_mode != "disabled"
            and self.coordination_policy_path is None
        ):
            raise ValueError(
                "enabled coordination policy requires coordination_policy_path"
            )
        self._coordination_policy: Optional[CoordinationPolicy] = None
        self._coordination_policy_package: Optional[Dict[str, Any]] = None
        self._cloud_model: Optional[Dict[str, Any]] = None
        self._current_state_cloud_model: Optional[Dict[str, Any]] = None
        self._edge_student: Optional[Dict[str, Any]] = None
        self._current_state_edge_student: Optional[Dict[str, Any]] = None
        self._feature_codec: Optional[Any] = None
        self._current_state_feature_codec: Optional[Any] = None
        self._cloud_model_sha256: Optional[str] = None
        self._current_state_cloud_model_sha256: Optional[str] = None
        self._topology: Optional[Dict[str, Any]] = None
        self._road_set_catalog: Optional[Dict[str, Any]] = None
        self._payload_schema: Optional[Dict[str, Any]] = None
        self._edge_llm = TrafficEdgeLLMController(
            release_registry_path=edge_llm_release_registry_path,
            runtime_config_path=edge_llm_runtime_config_path,
            mode=edge_llm_mode,
            min_risk_level=edge_llm_min_risk_level,
            student_confidence_threshold=edge_llm_student_confidence_threshold,
            min_expected_gain=edge_llm_min_expected_gain,
            deadline_margin_ms=edge_llm_deadline_margin_ms,
            deadline_probe_interval=edge_llm_deadline_probe_interval,
            runtime_failure_cooldown_seconds=edge_llm_runtime_failure_cooldown_seconds,
            gain_profile_path=edge_llm_gain_profile_path,
            edge_llm_prompt_prefix=edge_llm_prompt_prefix,
        )

    def payload_schema(self) -> Dict[str, Any]:
        if self._payload_schema is None:
            schema_path = (
                Path(__file__).resolve().parents[1]
                / "schemas"
                / "scenes"
                / "traffic_edge_event_v1.schema.json"
            )
            with schema_path.open("r", encoding="utf-8") as file_obj:
                self._payload_schema = json.load(file_obj)
        return dict(self._payload_schema)

    def _load_global_objective(self) -> TrafficGlobalObjective:
        if self.global_objective_path is None:
            raise ValueError("traffic global objective path is not configured")
        if self._global_objective is None:
            self._global_objective = TrafficGlobalObjective.from_path(
                self.global_objective_path
            )
        return self._global_objective

    def _load_coordination_policy(self) -> CoordinationPolicy:
        if self.coordination_policy_path is None:
            raise ValueError("traffic coordination policy path is not configured")
        if self._coordination_policy is None:
            package = verify_policy_package(self.coordination_policy_path)
            if (
                self.coordination_policy_mode == "active"
                and not package["production_activation_allowed"]
            ):
                raise ValueError(
                    "traffic coordination policy did not pass the production gate"
                )
            self._coordination_policy_package = package
            self._coordination_policy = CoordinationPolicy(package["policy"])
        return self._coordination_policy

    def _load_feature_codec(self) -> Any:
        if self.feature_codec_path is None:
            raise ValueError("traffic feature codec path is not configured")
        if self._feature_codec is None:
            from traffic_system.tree_feature_codec import TreeRoutingFeatureCodec

            self._feature_codec = TreeRoutingFeatureCodec(self.feature_codec_path)
        return self._feature_codec

    def _load_current_state_feature_codec(self) -> Any:
        if self.current_state_feature_codec_path is None:
            raise ValueError("traffic current-state feature codec path is not configured")
        if self._current_state_feature_codec is None:
            from traffic_system.tree_feature_codec import TreeRoutingFeatureCodec

            self._current_state_feature_codec = TreeRoutingFeatureCodec(
                self.current_state_feature_codec_path
            )
        return self._current_state_feature_codec

    def _load_edge_student(self) -> Dict[str, Any]:
        if self.edge_student_path is None:
            raise ValueError("traffic edge student path is not configured")
        if self._edge_student is None:
            from traffic_system.edge_student import load_student_model

            self._edge_student = load_student_model(self.edge_student_path)
        return self._edge_student

    def _load_current_state_edge_student(self) -> Dict[str, Any]:
        if self.current_state_edge_student_path is None:
            raise ValueError("traffic current-state edge student path is not configured")
        if self._current_state_edge_student is None:
            from traffic_system.edge_student import load_student_model

            self._current_state_edge_student = load_student_model(
                self.current_state_edge_student_path
            )
        return self._current_state_edge_student

    @staticmethod
    def _uses_current_state_contract(payload: Mapping[str, Any]) -> bool:
        return str(payload.get("perception_mode", "")).lower() == "current_state" or str(
            payload.get("output_type", "")
        ).lower() == "current_state_risk"

    def _feature_codec_for_payload(self, payload: Mapping[str, Any]) -> Any:
        if self._uses_current_state_contract(payload):
            if self.current_state_feature_codec_path is None:
                raise ValueError(
                    "traffic current-state payload requires current_state_feature_codec_path"
                )
            return self._load_current_state_feature_codec()
        return self._load_feature_codec()

    def _cloud_contract_for_payload(
        self,
        payload: Mapping[str, Any],
    ) -> Optional[Tuple[Dict[str, Any], Path, str, str]]:
        from traffic_system.cloud_coordinator import load_cloud_model

        if self._uses_current_state_contract(payload):
            if self.current_state_cloud_model_path is None:
                return None
            if self._current_state_cloud_model is None:
                self._current_state_cloud_model = load_cloud_model(
                    self.current_state_cloud_model_path
                )
            if self._current_state_cloud_model_sha256 is None:
                self._current_state_cloud_model_sha256 = _sha256_file(
                    self.current_state_cloud_model_path
                )
            return (
                self._current_state_cloud_model,
                self.current_state_cloud_model_path,
                self._current_state_cloud_model_sha256,
                "current_state_future_v1",
            )
        if self.cloud_model_path is None:
            return None
        if self._cloud_model is None:
            self._cloud_model = load_cloud_model(self.cloud_model_path)
        if self._cloud_model_sha256 is None:
            self._cloud_model_sha256 = _sha256_file(self.cloud_model_path)
        return (
            self._cloud_model,
            self.cloud_model_path,
            self._cloud_model_sha256,
            "forecast_joint_v1",
        )

    def _edge_student_for_payload(
        self, payload: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], Path, str]:
        if (
            self._uses_current_state_contract(payload)
            and self.current_state_edge_student_path is not None
        ):
            return (
                self._load_current_state_edge_student(),
                self.current_state_edge_student_path,
                "current_state_future_v1",
            )
        if self.edge_student_path is None:
            raise ValueError("traffic edge student path is not configured")
        return self._load_edge_student(), self.edge_student_path, "forecast_joint_v1"

    def _load_topology(self) -> Dict[str, Any]:
        if self.topology_path is None:
            raise ValueError("traffic topology path is not configured")
        if self._topology is None:
            with self.topology_path.open("r", encoding="utf-8") as file_obj:
                topology = json.load(file_obj)
            if not isinstance(topology, dict) or int(topology.get("schema_version", 0)) != 1:
                raise ValueError("invalid traffic region topology")
            neighbors = topology.get("region_neighbors")
            if not isinstance(neighbors, dict):
                raise ValueError("traffic topology region_neighbors must be an object")
            self._topology = topology
        return self._topology

    def _load_road_set_catalog(self) -> Dict[str, Any]:
        if self._road_set_catalog is None:
            self._road_set_catalog = build_road_set_catalog(self._load_topology())
        return self._road_set_catalog

    def _approved_road_set_model_artifact_sha256(self) -> str:
        if self.road_set_rule_config_path is None:
            raise ValueError(
                "canonical road-set observations require a configured rule artifact"
            )
        if self._road_set_model_artifact_sha256 is None:
            if not self.road_set_rule_config_path.is_file():
                raise ValueError("canonical road-set rule artifact is missing")
            self._road_set_model_artifact_sha256 = _sha256_file(
                self.road_set_rule_config_path
            )
        return self._road_set_model_artifact_sha256

    def _load_road_set_rule_config(self) -> Dict[str, Any]:
        self._approved_road_set_model_artifact_sha256()
        if self._road_set_rule_config is None:
            assert self.road_set_rule_config_path is not None
            with self.road_set_rule_config_path.open(
                "r", encoding="utf-8"
            ) as file_obj:
                value = json.load(file_obj)
            if not isinstance(value, dict):
                raise ValueError("canonical road-set rule artifact is invalid")
            self._road_set_rule_config = value
        return dict(self._road_set_rule_config)

    def _attach_boundary_resources(self, action: Action, region_id: str) -> Action:
        """Bind only explicitly road-set-scoped actions.

        Region-level reroute/regional actions are never copied onto every
        incident road set.  Shared actions are produced separately from the
        canonical overlap observation and carry an explicit ``road_set_ids``
        declaration.
        """
        if self.topology_path is None:
            return action
        parameters = dict(action.parameters)
        raw_declared = parameters.get("road_set_ids", [])
        declared = (
            [str(value) for value in raw_declared]
            if isinstance(raw_declared, list)
            else []
        )
        incident = {
            value["road_set_id"]: value
            for value in incident_road_sets(
                self._load_road_set_catalog(), region_id
            )
        }
        invalid = sorted(set(declared) - set(incident))
        if invalid:
            raise ValueError(
                "traffic action declares non-incident road sets: {}".format(
                    ", ".join(invalid)
                )
            )
        road_set_ids = list(dict.fromkeys(value for value in declared if value))
        road_set_resources = [
            incident[road_set_id]["resource_id"] for road_set_id in road_set_ids
        ]
        target_ids = list(action.target_ids)
        resource_ids = list(action.resource_ids)
        target_ids.extend(road_set_resources)
        resource_ids.extend(road_set_resources)
        if road_set_ids:
            parameters["road_set_ids"] = road_set_ids
            if action.action_type in {
                "variable_speed_limit",
                "ramp_metering",
                "regional_coordination",
                "reroute",
            }:
                # The road set is jointly owned by two edge partitions.  A
                # control that changes its shared execution state must never be
                # authorized from one owner's partial view when supplemental
                # evidence is missing, expired, or rejected.
                parameters["requires_cloud_confirmation"] = True
        return replace(
            action,
            target_ids=list(dict.fromkeys(target_ids)),
            resource_ids=list(dict.fromkeys(resource_ids)),
            parameters=parameters,
        )

    def normalize(self, envelope: SceneEventEnvelope) -> SemanticEvent:
        self.validate_envelope(envelope)
        if not isinstance(envelope.data, dict):
            raise ValueError("traffic event data must be an object")
        payload = dict(envelope.data)

        summary = payload.get("region_summary", {})
        if not isinstance(summary, dict):
            raise ValueError("traffic region_summary must be an object")
        risk_level = str(summary.get("region_risk_level", "low"))
        max_node_risk_level = str(
            summary.get("max_node_risk_level", risk_level)
        )
        if risk_level not in RISK_PRIORITY or max_node_risk_level not in RISK_PRIORITY:
            raise ValueError("traffic risk levels must be low, medium, high or severe")
        legacy_risk_level = max(
            (risk_level, max_node_risk_level),
            key=RISK_PRIORITY.__getitem__,
        )
        confidence = _safe_float(summary.get("region_risk_confidence"), 0.5)
        risk_score = _safe_float(summary.get("region_risk_score"), confidence)
        top_nodes = payload.get("top_k_risk_nodes", [])
        if not isinstance(top_nodes, list):
            top_nodes = []
        node_risk_scores = [
            _safe_float(node.get("risk_score"), 0.0)
            for node in top_nodes
            if isinstance(node, dict)
        ]
        legacy_risk_score = max([risk_score] + node_risk_scores)
        calibration = summary.get("region_risk_calibration", {})
        if not isinstance(calibration, dict):
            calibration = {}
        prediction_set = calibration.get("prediction_set", [risk_level])
        if not isinstance(prediction_set, list) or not prediction_set:
            prediction_set = [risk_level]
        occurred_at_ms = envelope.occurred_at_ms
        horizon_ms = int(payload.get("prediction_horizon_minutes", 60)) * 60 * 1000
        region_id = str(payload.get("region_id", "unknown_region"))
        edge_id = envelope.edge_id
        event_id = envelope.event_id
        num_partitions = max(1, int(payload.get("num_partitions", 1)))
        raw_aggregation_members = payload.get("aggregation_expected_members")
        if isinstance(raw_aggregation_members, list) and raw_aggregation_members:
            aggregation_members = [
                str(value) for value in raw_aggregation_members
            ]
        elif num_partitions == 1:
            aggregation_members = [edge_id]
        else:
            aggregation_members = [
                "edge_node_{}".format(index)
                for index in range(num_partitions)
            ]
        aggregation_member = str(
            payload.get("aggregation_member", edge_id)
        )

        from traffic_system.decision_utils import rule_teacher_decision

        reference = rule_teacher_decision(payload, decision_source="traffic_edge_reference_policy")
        actions = [
            self._attach_boundary_resources(_traffic_action(action, region_id), region_id)
            for action in reference.get("actions", [])
        ]
        node_ids = payload.get("managed_node_ids", [])
        raw_overlap_observations = payload.get(
            "road_set_overlap_observations", []
        )
        if not isinstance(raw_overlap_observations, list):
            raise ValueError(
                "traffic road_set_overlap_observations must be a list"
            )
        overlap_observations = [
            dict(value)
            for value in raw_overlap_observations
            if isinstance(value, dict)
        ]
        if overlap_observations:
            expected_dataset = str(payload.get("dataset", ""))
            expected_split = str(payload.get("sample_split", ""))
            expected_sample_id = int(payload.get("sample_id", -1))
            for observation in overlap_observations:
                if (
                    str(observation.get("dataset", "")) != expected_dataset
                    or str(observation.get("split", "")) != expected_split
                    or int(observation.get("sample_id", -2))
                    != expected_sample_id
                    or str(observation.get("window_id", ""))
                    != "{}:{}:{}".format(
                        expected_dataset,
                        expected_split,
                        expected_sample_id,
                    )
                ):
                    raise ValueError(
                        "canonical road-set observation is bound to another event window"
                    )
        road_set_assessments: List[Dict[str, Any]] = []
        road_set_contexts: List[Dict[str, Any]] = []
        road_set_scope_resources: List[str] = []
        road_set_correlation_keys: List[str] = []
        if self.topology_path is not None:
            road_set_catalog = self._load_road_set_catalog()
            incident = incident_road_sets(road_set_catalog, region_id)
            road_set_scope_resources = [
                road_set["resource_id"] for road_set in incident
            ]
            road_set_correlation_keys = [
                "traffic_road_set:{}".format(road_set["road_set_id"])
                for road_set in incident
            ]
            if overlap_observations:
                road_set_assessments = build_overlap_partial_assessments(
                    road_set_catalog,
                    region_id,
                    aggregation_member,
                    overlap_observations,
                    self._approved_road_set_model_artifact_sha256(),
                    self._load_road_set_rule_config(),
                    "edge_normalize_raw",
                )
            road_set_contexts = fuse_road_set_contexts(
                [
                    {
                        "aggregation_member": aggregation_member,
                        "road_set_assessments": road_set_assessments,
                    }
                ]
            )
            # Do not upload a second copy of every partial assessment.  Local
            # contexts carry only routing/audit state; the cloud reconstructs
            # the two-sided context from ``road_set_assessments``.
            road_set_contexts = [
                {
                    key: value
                    for key, value in context.items()
                    if key != "partial_assessments"
                }
                for context in road_set_contexts
            ]
            payload["road_set_assessments"] = road_set_assessments
            payload["road_set_contexts"] = road_set_contexts
        shared_resources = list(
            dict.fromkeys(
                ["traffic_node:{}".format(node) for node in node_ids]
                + road_set_scope_resources
            )
        )
        evidence_summary = {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "confidence": confidence,
            "node_risk_counts": summary.get("node_risk_counts", {}),
            "max_node_risk_level": summary.get("max_node_risk_level", risk_level),
        }
        evidence: List[Evidence] = [
            Evidence(
                evidence_id="{}_summary".format(event_id),
                level="summary",
                modality="traffic_timeseries",
                encoding="json",
                inline=evidence_summary,
                size_bytes=_json_size(evidence_summary),
                content_type="application/json",
            )
        ]
        selected_feature_codec_path = (
            self.current_state_feature_codec_path
            if self._uses_current_state_contract(payload)
            else self.feature_codec_path
        )
        if selected_feature_codec_path is not None:
            from traffic_system.decision_utils import extract_feature_vector

            values, feature_names = extract_feature_vector(payload)
            codec = self._feature_codec_for_payload(payload)
            if list(feature_names) != list(codec.feature_names):
                raise ValueError("traffic feature vector does not match codec schema")
            evidence.append(codec.encode(event_id, values))
        elif isinstance(top_nodes, list) and top_nodes:
            evidence.append(
                Evidence(
                    evidence_id="{}_features".format(event_id),
                    level="feature",
                    modality="traffic_timeseries",
                    encoding="json",
                    inline={"top_k_risk_nodes": top_nodes},
                    size_bytes=len(str(top_nodes).encode("utf-8")),
                    content_type="application/json",
                )
            )
        if overlap_observations:
            overlap_feature_payload = {
                "schema_version": 2,
                "kind": "canonical_road_set_overlap_feature_bundle",
                "observations": [
                    {
                        key: value
                        for key, value in observation.items()
                        if key != "raw_fragment_base64"
                    }
                    for observation in overlap_observations
                ],
            }
            evidence.append(
                Evidence(
                    evidence_id="{}_road_set_overlap_features".format(
                        event_id
                    ),
                    level="feature",
                    modality="traffic_road_set_overlap_features",
                    encoding="json",
                    inline=overlap_feature_payload,
                    size_bytes=_json_size(overlap_feature_payload),
                    content_type="application/json",
                )
            )
            overlap_raw_payload = {
                "schema_version": 2,
                "kind": "canonical_road_set_overlap_raw_bundle",
                "observations": overlap_observations,
            }
            evidence.append(
                Evidence(
                    evidence_id="{}_road_set_overlap_raw".format(event_id),
                    level="raw",
                    modality="traffic_road_set_overlap_raw",
                    encoding="json",
                    inline=overlap_raw_payload,
                    size_bytes=_json_size(overlap_raw_payload),
                    content_type="application/json",
                )
            )
        raw_evidence = payload.get("raw_evidence")
        if isinstance(raw_evidence, dict):
            evidence.append(Evidence.from_dict(raw_evidence))

        cloud_review_requested = bool(payload.get("cloud_review_requested", False))
        evidence_upload_hint = bool(payload.get("upload_required", False))
        upload_level = str(payload.get("upload_level", "")).strip().lower()
        evidence_level_by_upload = {
            "summary": "summary",
            "feature": "feature",
            "regional_context": "feature",
            "sequence": "feature",
            "raw": "raw",
        }
        # A compact semantic summary is the normal data plane. Feature/raw evidence
        # is an explicit escalation, never an implicit consequence of congestion.
        minimum_evidence_level = evidence_level_by_upload.get(upload_level, "summary")

        calibrated_confidence = max(
            0.0,
            min(
                1.0,
                _safe_float(calibration.get("calibrated_confidence"), confidence),
            ),
        )
        regional_state = {
            "level": risk_level,
            "score": round(max(0.0, min(1.0, risk_score)), 6),
            "confidence": round(max(0.0, min(1.0, confidence)), 6),
            "max_node_level": max_node_risk_level,
            "max_node_score": round(max(node_risk_scores, default=0.0), 6),
            "source": "region_summary",
        }
        safety_risk = _operational_safety_risk(payload, actions)
        model_uncertainty = _model_uncertainty(
            calibrated_confidence,
            [str(level) for level in prediction_set],
            self._edge_llm.student_confidence_threshold,
        )
        expected_gain = _escalation_expected_gain(payload)
        cloud_llm_review_policy = _cloud_llm_review_policy(
            payload,
            model_uncertainty,
            expected_gain,
            self.cloud_llm_min_expected_gain,
        )
        evidence_completeness = _evidence_completeness(
            payload,
            evidence,
            aggregation_members,
            aggregation_member,
            minimum_evidence_level,
        )

        return SemanticEvent(
            event_id=event_id,
            scene=self.scene,
            task=str(payload.get("task", "traffic_risk_assessment")),
            edge_id=edge_id,
            occurred_at_ms=occurred_at_ms,
            scope=EventScope(
                entity_id=region_id,
                subsystem="freeway_corridor",
                state_variable="congestion_state",
                region_id=region_id,
                shared_resources=shared_resources,
                correlation_keys=[
                    "traffic_region:{}".format(region_id),
                    "traffic_network:{}".format(payload.get("dataset", "PEMS08")),
                ]
                + road_set_correlation_keys,
                window_start_ms=occurred_at_ms,
                window_end_ms=occurred_at_ms + horizon_ms,
            ),
            prediction=Prediction(
                label=risk_level,
                confidence=max(0.0, min(1.0, confidence)),
                probabilities=dict(summary.get("region_risk_probabilities", {})),
                values={
                    "node_risk_counts": summary.get("node_risk_counts", {}),
                    "mean_node_risk_score": summary.get("mean_node_risk_score", 0.0),
                    "max_node_risk_level": summary.get("max_node_risk_level", risk_level),
                },
            ),
            risk=Risk(
                level=legacy_risk_level,
                score=max(0.0, min(1.0, legacy_risk_score)),
            ),
            uncertainty=Uncertainty(
                confidence=calibrated_confidence,
                calibrated=bool(calibration),
                prediction_set=[str(level) for level in prediction_set],
                method=str(calibration.get("method", "raw_model_confidence")),
            ),
            timing=Timing(
                deadline_ms=_safe_float(payload.get("deadline_ms"), 200.0),
                preprocessing_ms=_safe_float(payload.get("preprocessing_latency_ms"), 0.0),
                edge_inference_ms=_safe_float(payload.get("inference_latency_ms"), 0.0),
            ),
            evidence=evidence,
            candidate_actions=actions,
            model={
                "name": str(payload.get("model", "joint_astgcn")),
                "version": str(
                    payload.get("model_version", payload.get("checkpoint", "unversioned"))
                ),
                "output_type": str(payload.get("output_type", "forecast_and_risk")),
            },
            scene_payload=dict(payload),
            metadata={
                "adapter": "traffic_edge_event_v1",
                "ingress_type": envelope.event_type,
                "ingress_source": envelope.source,
                "ingress_dataschema": envelope.dataschema,
                "reference_edge_decision": str(reference.get("decision", "monitor")),
                "regional_risk_level": risk_level,
                "operational_risk_level": legacy_risk_level,
                "regional_risk_score": risk_score,
                "operational_risk_score": legacy_risk_score,
                "traffic_semantics_version": "2.0",
                "regional_state": regional_state,
                "operational_safety_risk": safety_risk,
                "model_uncertainty": model_uncertainty,
                "escalation_expected_gain": expected_gain,
                "cloud_llm_review_policy": cloud_llm_review_policy,
                "evidence_completeness": evidence_completeness,
                "road_set_contexts": road_set_contexts,
                "road_set_conflict_suspected": False,
                "required_members": [],
                "legacy_risk_semantics": "max(regional_state,max_node_state)",
                "transport_include_scene_payload": False,
                "cloud_review_requested": cloud_review_requested,
                "cloud_review_reason": "traffic_explicit_cloud_review"
                if cloud_review_requested
                else "",
                "evidence_upload_hint": evidence_upload_hint,
                "evidence_upload_level_hint": upload_level,
                "minimum_evidence_level": minimum_evidence_level,
                "aggregation": {
                    "key": "{}:{}:{}".format(
                        payload.get("dataset", "PEMS08"),
                        payload.get("sample_split", "unknown"),
                        payload.get("sample_id", event_id),
                    ),
                    "member": aggregation_member,
                    "expected_members": aggregation_members,
                    "minimum_members": max(
                        1,
                        min(
                            len(aggregation_members),
                            int(payload.get("aggregation_minimum_members", 2)),
                        ),
                    ),
                    "timeout_ms": int(payload.get("aggregation_timeout_ms", 200)),
                },
            },
        )

    def prepare_cloud_event(
        self,
        event: SemanticEvent,
        evidence_level: str,
    ) -> SemanticEvent:
        payload = event.scene_payload
        if not payload:
            return event
        top_nodes = []
        for node in payload.get("top_k_risk_nodes", []):
            if not isinstance(node, dict):
                continue
            compact_node = {
                key: node[key]
                for key in ("node_id", "risk_level", "risk_score")
                if key in node
            }
            top_nodes.append(compact_node)
        top_node_ids = {
            int(node["node_id"])
            for node in top_nodes
            if "node_id" in node and not isinstance(node["node_id"], bool)
        }
        raw_capabilities = payload.get("control_capabilities", {})
        compact_capabilities: Dict[str, Any] = {}
        if isinstance(raw_capabilities, dict):
            for name, fallback_limit in (
                ("variable_speed_limit_nodes", 0),
                ("ramp_meter_nodes", 3),
                ("reroute_gateway_nodes", 5),
            ):
                values = raw_capabilities.get(name, [])
                if not isinstance(values, list):
                    continue
                normalized_values = []
                for value in values:
                    if isinstance(value, bool):
                        continue
                    try:
                        normalized_values.append(int(value))
                    except (TypeError, ValueError):
                        continue
                selected = [
                    value for value in normalized_values if value in top_node_ids
                ]
                if fallback_limit:
                    for value in normalized_values:
                        if value not in selected:
                            selected.append(value)
                        if len(selected) >= fallback_limit:
                            break
                compact_capabilities[name] = selected
        compact_payload = {
            # scene/task/event_id already live in the common SemanticEvent
            # envelope.  Repeating them here made four compact traffic uploads
            # larger than the source float32 window they were meant to replace.
            "edge_id": event.edge_id,
            "region_id": event.scope.region_id,
            "sample_id": payload.get("sample_id"),
            "sample_split": payload.get("sample_split"),
            "partition_id": payload.get("partition_id"),
            "num_partitions": payload.get("num_partitions"),
            "upload_required": bool(payload.get("upload_required", False)),
            "upload_level": payload.get("upload_level"),
            "perception_mode": payload.get("perception_mode", "astgcn"),
            "output_type": payload.get("output_type", event.model.get("output_type")),
            "model_version": payload.get("model_version", event.model.get("version")),
            "prediction_horizon_minutes": payload.get("prediction_horizon_minutes", 60),
            "prediction_steps": payload.get("prediction_steps"),
            "num_managed_nodes": len(payload.get("managed_node_ids", []))
            or payload.get("num_managed_nodes", 1),
            "region_summary": payload.get("region_summary", {}),
            "top_k_risk_nodes": top_nodes,
            "control_capabilities": compact_capabilities,
            "road_set_assessments": payload.get("road_set_assessments", []),
            "road_set_contexts": payload.get("road_set_contexts", []),
        }
        if isinstance(payload.get("current_control_state"), dict):
            compact_payload["current_control_state"] = dict(
                payload["current_control_state"]
            )
        if isinstance(payload.get("neighbor_context"), (dict, list)):
            compact_payload["neighbor_context"] = payload["neighbor_context"]
        # Only cloud-consumed/audited metadata belongs on the data plane.  The
        # original edge event keeps the full diagnostic record locally, while
        # source identity is attached again by EdgeRuntime after this hook.
        cloud_metadata_keys = (
            "trace_id",
            "traffic_semantics_version",
            "regional_state",
            "operational_safety_risk",
            "model_uncertainty",
            "escalation_expected_gain",
            "cloud_llm_review_policy",
            "cloud_review_requested",
            "cloud_review_reason",
            "monitoring_force_cloud_review",
            "monitoring_reasons",
            "edge_runtime_network_available",
            "edge_runtime_network_status",
            "edge_runtime_network_class",
            "edge_runtime_network_rtt_ms",
            "edge_runtime_network_loss_rate",
            "aggregation",
            "road_set_contexts",
            "road_set_conflict_suspected",
            "required_members",
            "required_evidence_level",
        )
        metadata = {
            key: event.metadata[key]
            for key in cloud_metadata_keys
            if key in event.metadata
        }
        metadata.update(
            {
                "transport_include_scene_payload": True,
                "data_plane": "task_feature_evidence",
                "selected_evidence_level": evidence_level,
                "original_scene_payload_bytes": _json_size(payload),
                "cloud_scene_payload_bytes": _json_size(compact_payload),
            }
        )
        return replace(
            event,
            scope=replace(event.scope, shared_resources=[]),
            candidate_actions=[],
            scene_payload=compact_payload,
            metadata=metadata,
        )

    def fuse_cloud_context(
        self,
        events: Sequence[SemanticEvent],
    ) -> Sequence[SemanticEvent]:
        if len(events) < 2:
            return list(events)
        topology = self._load_topology()
        region_neighbors = topology["region_neighbors"]
        fused_road_set_contexts = fuse_road_set_contexts(
            [
                {
                    "region_id": event.scope.region_id,
                    "aggregation_member": event.metadata.get(
                        "aggregation", {}
                    ).get("member", event.edge_id)
                    if isinstance(event.metadata.get("aggregation"), dict)
                    else event.edge_id,
                    "road_set_assessments": self._effective_road_set_assessments(
                        event
                    ),
                }
                for event in events
            ]
        )
        fused_events: List[SemanticEvent] = []
        for event in events:
            allowed_regions = set(region_neighbors.get(event.scope.region_id, []))
            sample_id = event.scene_payload.get("sample_id")
            neighbors = []
            for other in events:
                if other.event_id == event.event_id or other.scope.region_id not in allowed_regions:
                    continue
                other_sample_id = other.scene_payload.get("sample_id")
                if (
                    sample_id is not None
                    and other_sample_id is not None
                    and sample_id != other_sample_id
                ):
                    continue
                if (
                    event.scope.window_end_ms < other.scope.window_start_ms
                    or other.scope.window_end_ms < event.scope.window_start_ms
                ):
                    continue
                other_regional_state = other.metadata.get("regional_state", {})
                other_regional_state = (
                    dict(other_regional_state)
                    if isinstance(other_regional_state, dict)
                    else {}
                )
                neighbors.append(
                    {
                        "event_id": other.event_id,
                        "edge_id": other.edge_id,
                        "region_id": other.scope.region_id,
                        "risk_level": str(
                            other_regional_state.get("level", other.prediction.label)
                        ),
                        "risk_score": _safe_float(
                            other_regional_state.get(
                                "score", other.prediction.confidence
                            )
                        ),
                        "confidence": other.uncertainty.confidence,
                    }
                )
            payload = dict(event.scene_payload)
            if neighbors:
                payload["neighbor_context"] = [
                    {
                        "method": topology.get("method", "road_graph_cut_edges"),
                        "neighbors": neighbors,
                    }
                ]
            event_road_set_contexts = contexts_for_region(
                fused_road_set_contexts, event.scope.region_id
            )
            payload["road_set_contexts"] = event_road_set_contexts
            conflict_contexts = [
                context
                for context in event_road_set_contexts
                if bool(context.get("road_set_conflict_suspected", False))
            ]
            required_members = sorted(
                {
                    str(member)
                    for context in conflict_contexts
                    for member in context.get("required_members", [])
                }
            )
            metadata = dict(event.metadata)
            metadata.update(
                {
                    "topology_fusion": topology.get("method"),
                    "fused_neighbor_count": len(neighbors),
                    "road_set_contexts": event_road_set_contexts,
                    "road_set_conflict_suspected": bool(conflict_contexts),
                    "required_members": required_members,
                    "road_set_conflicts": [
                        {
                            "road_set_id": context.get("road_set_id"),
                            "kind": context.get("conflict_kind"),
                            "reason": context.get("conflict_reason"),
                            "required_members": context.get(
                                "required_members", []
                            ),
                            "required_evidence_level": context.get(
                                "required_evidence_level"
                            ),
                        }
                        for context in conflict_contexts
                    ],
                }
            )
            if conflict_contexts:
                metadata["required_evidence_level"] = "raw"
            else:
                metadata.pop("required_evidence_level", None)
            fused_events.append(replace(event, scene_payload=payload, metadata=metadata))
        return fused_events

    def _effective_road_set_assessments(
        self, event: SemanticEvent
    ) -> List[Dict[str, Any]]:
        """Return the best authenticated road-set view available this pass.

        A summary/feature first pass uses compact declaration references.
        Feature evidence can verify the portable feature calculation but
        cannot independently reconstruct the source-content digest.  Only a
        raw callback rebuilds the complete observation and replaces stale or
        tampered declarations before the cloud reruns the policy.
        """

        raw_default = event.scene_payload.get("road_set_assessments", [])
        default = [
            dict(value)
            for value in raw_default
            if isinstance(value, dict)
        ] if isinstance(raw_default, list) else []
        aggregation = event.metadata.get("aggregation", {})
        member = (
            str(aggregation.get("member", event.edge_id))
            if isinstance(aggregation, dict)
            else event.edge_id
        )
        catalog = self._load_road_set_catalog()
        expected = {
            item["road_set_id"]: item
            for item in incident_road_sets(catalog, event.scope.region_id)
        }
        seen = set()
        for assessment in default:
            road_set_id = str(assessment.get("road_set_id", ""))
            if road_set_id in seen or road_set_id not in expected:
                raise ValueError(
                    "traffic compact road-set assessment coverage is invalid"
                )
            seen.add(road_set_id)
            expected_road_set = expected[road_set_id]
            if (
                assessment.get("assessment_source")
                != "canonical_overlap_observation"
                or str(assessment.get("member_region", ""))
                != event.scope.region_id
                or str(assessment.get("aggregation_member", "")) != member
                or str(assessment.get("resource_id", ""))
                != expected_road_set["resource_id"]
                or sorted(assessment.get("required_regions", []))
                != list(expected_road_set["required_regions"])
            ):
                raise ValueError(
                    "traffic compact road-set assessment owner binding is invalid"
                )
        if seen != set(expected):
            raise ValueError(
                "traffic compact road-set assessment set is incomplete"
            )

        selected_level = str(
            event.metadata.get("selected_evidence_level", "summary")
        ).strip().lower()
        pulled = event.metadata.get("selective_evidence_pull")
        pulled_level = (
            str(pulled.get("requested_level", "")).strip().lower()
            if isinstance(pulled, dict)
            else ""
        )
        raw_materialized = selected_level == "raw" or pulled_level == "raw"
        if not raw_materialized:
            feature_observations = []
            original_by_id = {}
            for assessment in default:
                road_set_id = str(assessment["road_set_id"])
                nested = assessment.get("canonical_feature_observation")
                if not isinstance(nested, dict):
                    raise ValueError(
                        "traffic compact road-set feature declaration is missing"
                    )
                expected_sample_id = int(
                    event.scene_payload.get("sample_id", -1)
                )
                expected_split = str(
                    event.scene_payload.get("sample_split", "")
                )
                if (
                    str(nested.get("dataset", "")) != "PEMS08"
                    or str(nested.get("split", "")) != expected_split
                    or int(nested.get("sample_id", -2))
                    != expected_sample_id
                    or str(nested.get("window_id", ""))
                    != "PEMS08:{}:{}".format(
                        expected_split, expected_sample_id
                    )
                ):
                    raise ValueError(
                        "traffic compact road-set observation window is invalid"
                    )
                feature_observations.append(dict(nested))
                original_by_id[road_set_id] = assessment
            rebuilt = build_overlap_partial_assessments(
                catalog,
                event.scope.region_id,
                member,
                feature_observations,
                self._approved_road_set_model_artifact_sha256(),
                self._load_road_set_rule_config(),
                "cloud_compact_feature_recomputed",
            )
            declaration_fields = (
                "observation_id",
                "window_id",
                "source_dataset_sha256",
                "raw_fragment_sha256",
                "source_content_sha256",
                "feature_content_sha256",
                "preprocess_version",
                "model_id",
                "model_version",
                "model_artifact_sha256",
                "policy_id",
                "policy_version",
                "policy_artifact_sha256",
                "output",
                "output_digest",
                "action",
                "action_digest",
            )
            for assessment in rebuilt:
                original = original_by_id[assessment["road_set_id"]]
                errors = [
                    field
                    for field in declaration_fields
                    if original.get(field) != assessment.get(field)
                ]
                if errors:
                    assessment["compact_self_validation_errors"] = errors
            return rebuilt
        bundles = [
            item
            for item in event.evidence
            if item.level == "raw"
            and item.modality == "traffic_road_set_overlap_raw"
        ]
        if not bundles:
            return default
        if len(bundles) != 1:
            raise ValueError(
                "traffic cloud fusion requires one canonical road-set overlap bundle"
            )
        inline = bundles[0].inline
        if not isinstance(inline, dict):
            raise ValueError("traffic road-set overlap bundle must be inline JSON")
        if (
            int(inline.get("schema_version", 0)) != 2
            or inline.get("kind")
            != "canonical_road_set_overlap_raw_bundle"
        ):
            raise ValueError("traffic road-set overlap bundle contract is invalid")
        raw_observations = inline.get("observations", [])
        if not isinstance(raw_observations, list):
            raise ValueError("traffic road-set overlap observations must be a list")
        expected_sample_id = int(event.scene_payload.get("sample_id", -1))
        expected_split = str(event.scene_payload.get("sample_split", ""))
        for observation in raw_observations:
            if (
                not isinstance(observation, dict)
                or str(observation.get("dataset", "")) != "PEMS08"
                or str(observation.get("split", "")) != expected_split
                or int(observation.get("sample_id", -2))
                != expected_sample_id
                or str(observation.get("window_id", ""))
                != "PEMS08:{}:{}".format(
                    expected_split, expected_sample_id
                )
            ):
                raise ValueError(
                    "pulled road-set raw observation window is invalid"
                )
        return build_overlap_partial_assessments(
            catalog,
            event.scope.region_id,
            member,
            [
                dict(value)
                for value in raw_observations
                if isinstance(value, dict)
            ],
            self._approved_road_set_model_artifact_sha256(),
            self._load_road_set_rule_config(),
            "cloud_pulled_raw",
        )

    def _prepare_cloud_feature_input(
        self,
        event: SemanticEvent,
        cloud_model: Mapping[str, Any],
        cloud_model_sha256: str,
        codec: Any,
    ) -> Dict[str, Any]:
        payload = event.scene_payload
        feature_evidence = [
            item
            for item in event.evidence
            if item.level == "feature" and item.modality == "traffic_task_features"
        ]
        if len(feature_evidence) != 1:
            raise ValueError("cloud traffic inference requires one encoded feature evidence")
        vector = codec.decode(
            feature_evidence[0],
            expected_model_sha256=cloud_model_sha256,
        )
        if list(codec.feature_names) != list(cloud_model.get("feature_names", [])):
            raise ValueError("traffic cloud model and feature codec schemas differ")
        from traffic_system.decision_utils import extract_feature_vector

        fused_values, fused_names = extract_feature_vector(payload)
        if list(fused_names) != list(codec.feature_names):
            raise ValueError("fused traffic context does not match cloud feature schema")
        neighbor_feature_count = 0
        active_neighbor_feature_count = 0
        active_indices = {int(index) for index in codec.active_indices.tolist()}
        for index, name in enumerate(fused_names):
            if not name.startswith("neighbor_"):
                continue
            vector[index] = fused_values[index]
            neighbor_feature_count += 1
            active_neighbor_feature_count += int(index in active_indices)
        return {
            "vector": vector,
            "codec": codec,
            "feature_evidence": feature_evidence[0],
            "neighbor_feature_count": neighbor_feature_count,
            "active_neighbor_feature_count": active_neighbor_feature_count,
        }

    @staticmethod
    def _has_cloud_feature_evidence(event: SemanticEvent) -> bool:
        return sum(
            item.level == "feature"
            and item.modality == "traffic_task_features"
            for item in event.evidence
        ) == 1

    def _cloud_summary_legacy_decision(
        self,
        event: SemanticEvent,
        summary_batch_size: int,
    ) -> Dict[str, Any]:
        """Produce the lightweight cloud baseline from a semantic summary.

        Ordinary online samples intentionally carry no encoded 226-dimensional
        feature vector.  They must still participate in topology fusion and
        conflict coordination instead of entering an impossible model retry
        loop.  The deterministic scene policy consumes only the compact
        semantic payload; feature-bearing events continue to use ExtraTrees.
        """
        from traffic_system.decision_utils import rule_teacher_decision

        legacy = rule_teacher_decision(
            event.scene_payload,
            decision_source="cloud_semantic_summary_coordinator",
        )
        legacy["policy_version"] = self.policy_version
        legacy["framework_metadata"] = {
            "cloud_inference_path": "semantic_summary_policy",
            "cloud_summary_batch_size": int(summary_batch_size),
            "selected_evidence_level": "summary",
            "fused_neighbor_count": event.metadata.get(
                "fused_neighbor_count", 0
            ),
        }
        return legacy

    def _build_cloud_legacy_decision(
        self,
        event: SemanticEvent,
        decision_class: str,
        confidence: float,
        feature_input: Dict[str, Any],
        batch_size: int,
        cloud_model_contract: str,
    ) -> Dict[str, Any]:
        from traffic_system.decision_utils import build_decision_from_student_class

        payload = event.scene_payload
        codec = feature_input["codec"]
        feature_evidence = feature_input["feature_evidence"]
        decision = build_decision_from_student_class(
            payload,
            decision_class,
            confidence,
            decision_source="cloud_extratrees_task_feature_coordinator",
        )
        neighbor_states = []
        for context in payload.get("neighbor_context", []):
            if isinstance(context, dict) and isinstance(context.get("neighbors"), list):
                neighbor_states.extend(
                    item for item in context["neighbors"] if isinstance(item, dict)
                )
        own_confidence = max(0.0, event.uncertainty.confidence)
        own_weight = max(0.05, event.risk.score * own_confidence)
        diversion_limits = []
        for neighbor in neighbor_states:
            neighbor_confidence = _safe_float(neighbor.get("confidence"), 0.0)
            if (
                str(neighbor.get("risk_level")) not in {"high", "severe"}
                or min(own_confidence, neighbor_confidence) < 0.5
            ):
                continue
            neighbor_weight = max(
                0.05,
                _safe_float(neighbor.get("risk_score"), 0.0)
                * neighbor_confidence,
            )
            diversion_limits.append(
                0.5 * own_weight / (own_weight + neighbor_weight)
            )
        pre_coordinated_ratio = None
        if diversion_limits:
            pre_coordinated_ratio = max(0.1, min(0.3, min(diversion_limits)))
            for action in decision.get("actions", []):
                if isinstance(action, dict) and action.get("type") == "reroute":
                    action["diversion_ratio"] = round(pre_coordinated_ratio, 3)
        decision["policy_version"] = self.policy_version
        decision["framework_metadata"] = {
            "feature_codec": codec.metadata["codec"],
            "feature_codec_artifact_id": codec.metadata["artifact_id"],
            "active_feature_count": codec.metadata["active_feature_count"],
            "source_feature_bytes": feature_evidence.codec.get("source_size_bytes"),
            "encoded_feature_bytes": feature_evidence.size_bytes,
            "fused_neighbor_count": event.metadata.get("fused_neighbor_count", 0),
            "neighbor_feature_count": feature_input["neighbor_feature_count"],
            "active_neighbor_feature_count": feature_input[
                "active_neighbor_feature_count"
            ],
            "cloud_inference_batch_size": int(batch_size),
            "cloud_model_contract": cloud_model_contract,
            "topology_precoordinated_diversion_ratio": round(
                pre_coordinated_ratio, 3
            )
            if pre_coordinated_ratio is not None
            else None,
        }
        return decision

    def _cloud_legacy_decisions(
        self,
        events: Sequence[SemanticEvent],
    ) -> List[Dict[str, Any]]:
        normalized = list(events)
        if not normalized:
            return []
        contracts = [
            self._cloud_contract_for_payload(event.scene_payload)
            for event in normalized
        ]
        if any(contract is None for contract in contracts):
            raise ValueError("traffic feature event has no matching cloud model contract")
        assert contracts[0] is not None
        contract_names = {contract[3] for contract in contracts if contract is not None}
        if len(contract_names) != 1:
            raise ValueError("traffic cloud batch mixes incompatible perception contracts")
        cloud_model, cloud_model_path, cloud_model_sha256, cloud_model_contract = (
            contracts[0]
        )
        if not cloud_model_path.is_file():
            raise FileNotFoundError(
                "traffic cloud model not found: {}".format(cloud_model_path)
            )
        codec = self._feature_codec_for_payload(normalized[0].scene_payload)
        feature_inputs = [
            self._prepare_cloud_feature_input(
                event,
                cloud_model,
                cloud_model_sha256,
                codec,
            )
            for event in normalized
        ]
        matrix = np.asarray(
            [feature_input["vector"] for feature_input in feature_inputs],
            dtype=np.float64,
        )
        model = cloud_model["model"]
        probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[0] != len(normalized):
            raise ValueError("traffic cloud model returned an invalid probability matrix")
        class_positions = np.argmax(probabilities, axis=1)
        model_classes = np.asarray(model.classes_)
        decisions = []
        for row_index, (event, feature_input) in enumerate(
            zip(normalized, feature_inputs)
        ):
            class_position = int(class_positions[row_index])
            class_id = int(model_classes[class_position])
            decision_class = str(cloud_model["decision_classes"][class_id])
            decisions.append(
                self._build_cloud_legacy_decision(
                    event,
                    decision_class,
                    float(probabilities[row_index, class_position]),
                    feature_input,
                    len(normalized),
                    cloud_model_contract,
                )
            )
        return decisions

    def _legacy_decision(self, event: SemanticEvent, cloud: bool) -> Dict[str, Any]:
        payload = event.scene_payload
        if not payload:
            return {}
        if cloud and self._cloud_contract_for_payload(payload) is not None:
            return self._cloud_legacy_decisions([event])[0]
        if not cloud and (
            self.edge_student_path is not None
            or self.current_state_edge_student_path is not None
        ):
            from traffic_system.decision_utils import (
                build_decision_from_student_class,
                extract_feature_vector,
            )
            from traffic_system.edge_student import predict_student

            student, selected_student_path, student_contract = (
                self._edge_student_for_payload(payload)
            )
            _, feature_names = extract_feature_vector(payload)
            if list(feature_names) != list(student.get("feature_names", [])):
                raise ValueError("traffic edge student feature schema mismatch")
            decision_class, confidence, probabilities = predict_student(payload, student)
            decision = build_decision_from_student_class(
                payload,
                decision_class,
                confidence,
                decision_source="edge_mlp_distilled_student",
            )
            decision["policy_version"] = self.policy_version
            decision["framework_metadata"] = {
                "edge_student_model": selected_student_path.name,
                "edge_student_type": student.get("model_type", "numpy_mlp"),
                "edge_student_contract": student_contract,
                "edge_student_feature_count": len(feature_names),
                "edge_student_probabilities": {
                    name: round(float(probability), 6)
                    for name, probability in zip(
                        student.get("decision_classes", []), probabilities.tolist()
                    )
                },
            }
            return decision
        from traffic_system.decision_utils import rule_teacher_decision

        source = "traffic_cloud_reference_policy" if cloud else "traffic_edge_reference_policy"
        return rule_teacher_decision(payload, decision_source=source)

    def warmup(self) -> None:
        if self.coordination_policy_mode != "disabled":
            if (
                self.coordination_policy_path is None
                or not self.coordination_policy_path.is_dir()
            ):
                raise FileNotFoundError(
                    "traffic coordination policy not found: {}".format(
                        self.coordination_policy_path
                    )
                )
            self._load_coordination_policy()
        if self.global_optimizer_mode != "disabled":
            if self.global_objective_path is None or not self.global_objective_path.is_file():
                raise FileNotFoundError(
                    "traffic global objective not found: {}".format(
                        self.global_objective_path
                    )
                )
            self._load_global_objective()
        if self.edge_student_path is not None:
            if not self.edge_student_path.is_file():
                raise FileNotFoundError(
                    "traffic edge student not found: {}".format(self.edge_student_path)
                )
            student = self._load_edge_student()
            if list(student.get("feature_names", [])) != list(
                self._load_feature_codec().feature_names
            ):
                raise ValueError("traffic edge student and feature codec schemas differ")
        if self.current_state_edge_student_path is not None:
            if not self.current_state_edge_student_path.is_file():
                raise FileNotFoundError(
                    "traffic current-state edge student not found: {}".format(
                        self.current_state_edge_student_path
                    )
                )
            current_student = self._load_current_state_edge_student()
            if list(current_student.get("feature_names", [])) != list(
                self._load_current_state_feature_codec().feature_names
            ):
                raise ValueError(
                    "traffic current-state student and feature codec schemas differ"
                )
        self._edge_llm.warmup()
        cloud_contracts = []
        if self.cloud_model_path is not None:
            cloud_contracts.append(
                (
                    "forecast_joint_v1",
                    self.cloud_model_path,
                    self._load_feature_codec(),
                    False,
                )
            )
        if self.current_state_cloud_model_path is not None:
            cloud_contracts.append(
                (
                    "current_state_future_v1",
                    self.current_state_cloud_model_path,
                    self._load_current_state_feature_codec(),
                    True,
                )
            )
        if not cloud_contracts:
            return
        if self.topology_path is None:
            raise ValueError("traffic cloud model requires topology_path")
        self._load_topology()
        for contract_name, model_path, codec, current_state in cloud_contracts:
            if not model_path.is_file():
                raise FileNotFoundError(
                    "traffic cloud model not found: {}".format(model_path)
                )
            contract = self._cloud_contract_for_payload(
                {
                    "perception_mode": "current_state" if current_state else "astgcn",
                    "output_type": "current_state_risk"
                    if current_state
                    else "forecast_and_risk",
                }
            )
            if contract is None:
                raise ValueError(
                    "traffic cloud model contract is not configured: {}".format(
                        contract_name
                    )
                )
            model_payload, _, model_sha256, loaded_contract_name = contract
            if loaded_contract_name != contract_name:
                raise ValueError("traffic cloud model contract selection mismatch")
            if codec.metadata.get("source_model_sha256") != model_sha256:
                raise ValueError(
                    "traffic feature codec was exported from another cloud model"
                )
            if list(codec.feature_names) != list(
                model_payload.get("feature_names", [])
            ):
                raise ValueError(
                    "traffic feature codec schema does not match cloud model"
                )

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "cloud_model_configured": self.cloud_model_path is not None,
            "cloud_model_loaded": self._cloud_model is not None,
            "current_state_cloud_model_configured": self.current_state_cloud_model_path
            is not None,
            "current_state_cloud_model_loaded": self._current_state_cloud_model
            is not None,
            "edge_student_configured": self.edge_student_path is not None,
            "edge_student_loaded": self._edge_student is not None,
            "edge_student_type": self._edge_student.get("model_type")
            if self._edge_student is not None
            else None,
            "current_state_edge_student_configured": self.current_state_edge_student_path
            is not None,
            "current_state_edge_student_loaded": self._current_state_edge_student
            is not None,
            "feature_codec_configured": self.feature_codec_path is not None,
            "feature_codec_loaded": self._feature_codec is not None,
            "feature_codec": self._feature_codec.describe()
            if self._feature_codec is not None
            else None,
            "current_state_feature_codec_configured": self.current_state_feature_codec_path
            is not None,
            "current_state_feature_codec_loaded": self._current_state_feature_codec
            is not None,
            "current_state_feature_codec": self._current_state_feature_codec.describe()
            if self._current_state_feature_codec is not None
            else None,
            "topology_configured": self.topology_path is not None,
            "topology_loaded": self._topology is not None,
            "topology_method": self._topology.get("method")
            if self._topology is not None
            else None,
            "policy_version": self.policy_version,
            "cloud_llm_min_expected_gain": self.cloud_llm_min_expected_gain,
            "current_state_sync_confidence_threshold": (
                self.current_state_sync_confidence_threshold
            ),
            "global_optimizer_mode": self.global_optimizer_mode,
            "global_objective_configured": self.global_objective_path is not None,
            "global_objective_loaded": self._global_objective is not None,
            "global_objective_id": self._global_objective.objective_id
            if self._global_objective is not None
            else None,
            "global_objective_sha256": self._global_objective.objective_sha256
            if self._global_objective is not None
            else None,
            "global_objective_definition_sha256": (
                self._global_objective.definition_file_sha256
                if self._global_objective is not None
                else None
            ),
            "coordination_policy_mode": self.coordination_policy_mode,
            "coordination_policy_configured": self.coordination_policy_path
            is not None,
            "coordination_policy_loaded": self._coordination_policy is not None,
            "coordination_policy_version": self._coordination_policy.version
            if self._coordination_policy is not None
            else None,
            "coordination_policy_production_activation_allowed": (
                self._coordination_policy_package[
                    "production_activation_allowed"
                ]
                if self._coordination_policy_package is not None
                else None
            ),
            "edge_llm": self._edge_llm.health(),
        }

    def optimize_global_plan(
        self,
        events: Sequence[SemanticEvent],
        decisions: Sequence[DecisionEnvelope],
    ) -> Tuple[Sequence[DecisionEnvelope], Dict[str, Any]]:
        if self.global_optimizer_mode == "disabled":
            return list(decisions), {}
        objective = self._load_global_objective()
        return objective.optimize(
            events,
            decisions,
            self.action_conflict,
            self.global_optimizer_mode,
        )

    def _attach_local_uncertainty(
        self,
        event: SemanticEvent,
        student_decision: Any,
    ) -> Any:
        """Record deterministic Student/rule signals without selecting a route."""
        from traffic_system.decision_utils import rule_teacher_decision

        rule = rule_teacher_decision(
            event.scene_payload,
            decision_source="traffic_local_safety_policy",
        )
        student_name = str(student_decision.decision)
        rule_name = str(rule["decision"])
        student_confidence = float(student_decision.confidence)
        student_available = bool(
            self.edge_student_path is not None
            or self.current_state_edge_student_path is not None
        )
        uses_current_state_contract = self._uses_current_state_contract(
            event.scene_payload
        )
        disagreement = student_available and student_name != rule_name
        uncertainty = event.metadata.get("model_uncertainty", {})
        uncertainty = dict(uncertainty) if isinstance(uncertainty, dict) else {}
        prediction_set = uncertainty.get(
            "prediction_set", event.uncertainty.prediction_set
        )
        prediction_set = (
            [str(value) for value in prediction_set]
            if isinstance(prediction_set, list)
            else list(event.uncertainty.prediction_set)
        )
        low_student_confidence = bool(
            student_available
            and student_confidence < self._edge_llm.student_confidence_threshold
        )
        uncertainty_update = {
            "score": round(
                max(
                    _bounded_float(uncertainty.get("score"), 0.0),
                    1.0 - student_confidence if student_available else 0.0,
                ),
                6,
            ),
            "student_available": student_available,
            "student_confidence": round(student_confidence, 6)
            if student_available
            else None,
            "student_confidence_threshold": round(
                self._edge_llm.student_confidence_threshold, 6
            ),
            "student_low_confidence": low_student_confidence,
            "prediction_set": prediction_set,
            "prediction_set_size": len(prediction_set),
            "student_rule_disagreement": disagreement,
            "requires_review": bool(
                low_student_confidence
                or len(prediction_set) > 1
                or disagreement
            ),
            "source": "student_rule_signals",
        }
        if uses_current_state_contract:
            synchronous_reasons = []
            if len(prediction_set) > 1:
                synchronous_reasons.append("prediction_set_ambiguous")
            if (
                student_available
                and disagreement
                and student_confidence
                < self.current_state_sync_confidence_threshold
            ):
                synchronous_reasons.append(
                    "student_no_majority_and_rule_disagreement"
                )
            uncertainty_update.update(
                {
                    "requires_synchronous_review": bool(synchronous_reasons),
                    "synchronous_review_reasons": synchronous_reasons,
                    "synchronous_review_confidence_threshold": round(
                        self.current_state_sync_confidence_threshold, 6
                    ),
                    "synchronous_review_resolution": "unresolved"
                    if synchronous_reasons
                    else "not_required",
                }
            )
        uncertainty.update(uncertainty_update)
        metadata = dict(student_decision.metadata)
        metadata.update(
            {
                "traffic_student_candidate_decision": student_name,
                "traffic_student_candidate_confidence": round(
                    student_confidence, 6
                )
                if student_available
                else None,
                "traffic_rule_candidate_decision": rule_name,
                "traffic_student_rule_disagreement": disagreement,
                "model_uncertainty": uncertainty,
                "escalation_expected_gain": dict(
                    event.metadata.get("escalation_expected_gain", {})
                )
                if isinstance(event.metadata.get("escalation_expected_gain"), dict)
                else {},
            }
        )
        return replace(student_decision, metadata=metadata)

    def _resolve_current_state_synchronous_uncertainty(
        self,
        event: SemanticEvent,
        decision: Any,
    ) -> Any:
        """Let an agreeing local Qwen result resolve only Student uncertainty."""
        if not self._uses_current_state_contract(event.scene_payload):
            return decision
        metadata = dict(getattr(decision, "metadata", {}))
        if (
            metadata.get("edge_decision_path") != "edge_qwen"
            or metadata.get("edge_llm_model_disagreement") is not False
            or metadata.get("edge_llm_requires_cloud") is not False
        ):
            return decision
        uncertainty = metadata.get("model_uncertainty", {})
        uncertainty = dict(uncertainty) if isinstance(uncertainty, dict) else {}
        reasons = uncertainty.get("synchronous_review_reasons", [])
        if not isinstance(reasons, list):
            return decision
        reasons = [str(value) for value in reasons]
        student_reason = "student_no_majority_and_rule_disagreement"
        if student_reason not in reasons:
            return decision
        unresolved = [
            reason for reason in reasons if reason != student_reason
        ]
        uncertainty.update(
            {
                "requires_synchronous_review": bool(unresolved),
                "synchronous_review_reasons": unresolved,
                "synchronous_review_resolution": (
                    "edge_qwen_corroborated_student"
                    if not unresolved
                    else "edge_qwen_corroborated_student_with_unresolved_signals"
                ),
            }
        )
        metadata["model_uncertainty"] = uncertainty
        return replace(decision, metadata=metadata)

    def cloud_submission_metadata(
        self,
        event: SemanticEvent,
        local_decision: Any,
    ) -> Dict[str, Any]:
        """Carry Student/rule uncertainty into cloud review policy."""
        event_uncertainty = event.metadata.get("model_uncertainty", {})
        event_uncertainty = (
            dict(event_uncertainty) if isinstance(event_uncertainty, dict) else {}
        )
        local_uncertainty = local_decision.metadata.get("model_uncertainty", {})
        local_uncertainty = (
            dict(local_uncertainty) if isinstance(local_uncertainty, dict) else {}
        )
        uncertainty = {**event_uncertainty, **local_uncertainty}
        local_gain = local_decision.metadata.get("escalation_expected_gain")
        event_gain = event.metadata.get("escalation_expected_gain", {})
        expected_gain = (
            dict(local_gain)
            if isinstance(local_gain, dict)
            else dict(event_gain)
            if isinstance(event_gain, dict)
            else _escalation_expected_gain(event.scene_payload)
        )
        return {
            "model_uncertainty": uncertainty,
            "escalation_expected_gain": expected_gain,
            "cloud_llm_review_policy": _cloud_llm_review_policy(
                event.scene_payload,
                uncertainty,
                expected_gain,
                self.cloud_llm_min_expected_gain,
            ),
        }

    def evidence_advice(
        self,
        event: SemanticEvent,
        local_decision: Any,
        conflict_suspected: bool = False,
    ) -> Dict[str, Any]:
        """Choose traffic evidence independently from the legacy congestion risk."""
        explicit_hint = bool(event.metadata.get("evidence_upload_hint", False))
        explicit_level = str(
            event.metadata.get("evidence_upload_level_hint", "")
        ).strip().lower()
        level_by_hint = {
            "summary": "summary",
            "feature": "feature",
            "regional_context": "feature",
            "sequence": "feature",
            "raw": "raw",
        }
        if explicit_hint or explicit_level:
            return {
                "required_level": level_by_hint.get(explicit_level, "feature"),
                "reason": "traffic_explicit_upload_hint",
            }

        completeness = event.metadata.get("evidence_completeness", {})
        completeness = (
            dict(completeness) if isinstance(completeness, dict) else {}
        )
        declared = completeness.get("declared", {})
        declared = dict(declared) if isinstance(declared, dict) else {}
        declared_level = str(declared.get("required_level", "")).strip().lower()
        if declared_level in level_by_hint:
            return {
                "required_level": level_by_hint[declared_level],
                "reason": "traffic_declared_evidence_requirement",
            }
        if not bool(completeness.get("minimum_level_available", True)):
            return {
                "required_level": "feature",
                "reason": "traffic_required_evidence_missing",
            }

        event_uncertainty = event.metadata.get("model_uncertainty", {})
        event_uncertainty = (
            dict(event_uncertainty) if isinstance(event_uncertainty, dict) else {}
        )
        decision_uncertainty = getattr(local_decision, "metadata", {}).get(
            "model_uncertainty", {}
        )
        decision_uncertainty = (
            dict(decision_uncertainty)
            if isinstance(decision_uncertainty, dict)
            else {}
        )
        uncertainty = {**event_uncertainty, **decision_uncertainty}
        if bool(uncertainty.get("requires_review", False)):
            return {
                "required_level": "feature",
                "reason": "traffic_model_uncertainty",
            }
        if conflict_suspected:
            return {
                "required_level": "feature",
                "reason": "traffic_action_conflict_suspected",
            }
        if bool(event.metadata.get("cloud_review_requested", False)):
            return {
                "required_level": "feature",
                "reason": "traffic_explicit_cloud_review",
            }
        return {
            "required_level": "summary",
            "reason": "traffic_complete_summary_default",
        }

    def _ensure_operational_safety(
        self,
        event: SemanticEvent,
        decision: Any,
    ) -> Any:
        safety_risk = event.metadata.get("operational_safety_risk", {})
        safety_risk = dict(safety_risk) if isinstance(safety_risk, dict) else {}
        event_safety_level = str(safety_risk.get("level", event.risk.level))
        if event_safety_level not in RISK_PRIORITY:
            raise ValueError("traffic operational safety risk level is invalid")
        decision_action_types = list(
            dict.fromkeys(action.action_type for action in decision.actions)
        )
        if decision.decision in ACTION_SAFETY_RISK:
            decision_action_types.append(decision.decision)
        decision_safety_level = max(
            (
                ACTION_SAFETY_RISK.get(action_type, "low")
                for action_type in decision_action_types
            ),
            key=RISK_PRIORITY.__getitem__,
            default="low",
        )
        safety_level = max(
            (event_safety_level, decision_safety_level),
            key=RISK_PRIORITY.__getitem__,
        )
        effective_safety_risk = {
            **safety_risk,
            "level": safety_level,
            "score": max(
                _bounded_float(safety_risk.get("score"), 0.0),
                RISK_DEFAULT_SCORE[safety_level],
            ),
            "decision_action_types": decision_action_types,
            "decision_policy_floor_level": decision_safety_level,
        }
        decision = replace(
            decision,
            metadata={
                **decision.metadata,
                "operational_safety_risk": effective_safety_risk,
            },
        )
        if RISK_PRIORITY[safety_level] < RISK_PRIORITY["high"]:
            return decision
        network_available = bool(
            event.metadata.get("edge_runtime_network_available", True)
        )
        cloud_only_types = {"regional_coordination", "reroute"}
        cloud_only_decision = (
            decision.decision in cloud_only_types
            or any(action.action_type in cloud_only_types for action in decision.actions)
        )
        requires_floor = (
            decision.decision in {"no_action", "abstain", "request_cloud"}
            or not decision.actions
            or (not network_available and cloud_only_decision)
        )
        if not requires_floor:
            return decision

        safe_types = {"traffic_advisory", "variable_speed_limit", "ramp_metering"}
        safe_candidates = [
            action
            for action in event.candidate_actions
            if action.action_type in safe_types
        ]
        advisories = [
            action for action in safe_candidates if action.action_type == "traffic_advisory"
        ]
        controls = [
            action for action in safe_candidates if action.action_type != "traffic_advisory"
        ]
        selected_actions: List[Action] = []
        if advisories:
            selected_actions.append(max(advisories, key=lambda action: action.priority))
        if controls:
            selected_actions.append(max(controls, key=lambda action: action.priority))
        if not selected_actions:
            regional_state = event.metadata.get("regional_state", {})
            regional_state = (
                dict(regional_state) if isinstance(regional_state, dict) else {}
            )
            selected_actions.append(
                Action(
                    action_type="traffic_advisory",
                    target_ids=[event.scope.entity_id],
                    resource_ids=[event.scope.entity_id],
                    parameters={
                        "strategy": "issue_congestion_warning",
                        "warning_level": str(
                            regional_state.get("level", event.prediction.label)
                        ),
                    },
                    reason="Operational risk requires an immediate local warning.",
                    priority=30,
                )
            )

        main_action = selected_actions[-1].action_type
        decision_name = (
            "congestion_warning"
            if main_action == "traffic_advisory"
            else main_action
        )
        safe_decision = build_decision(
            event=event,
            decision=decision_name,
            actions=selected_actions,
            confidence=decision.confidence,
            reason="operational risk requires an immediately executable local action",
            source="traffic_operational_safety_floor",
            policy_version=self.policy_version,
        )
        metadata = dict(decision.metadata)
        previous_source = metadata.get("source")
        metadata.update(
            {
                "source": "traffic_operational_safety_floor",
                "operational_safety_override": True,
                "safety_floor_previous_decision": decision.decision,
                "safety_floor_previous_source": previous_source,
                "safety_floor_network_available": network_available,
                "operational_safety_risk_level": safety_level,
                "operational_safety_risk_source": safety_risk.get(
                    "source", "legacy_event_risk_fallback"
                ),
            }
        )
        return replace(safe_decision, metadata=metadata)

    def _decision_from_legacy(
        self,
        event: SemanticEvent,
        legacy: Dict[str, Any],
        cloud: bool,
    ) -> Any:
        if legacy:
            actions = [
                self._attach_boundary_resources(
                    _traffic_action(action, event.scope.region_id),
                    event.scope.region_id,
                )
                for action in legacy.get("actions", [])
            ]
            source = str(legacy.get("decision_source", "traffic_plugin"))
            decision = build_decision(
                event=event,
                decision=str(legacy.get("decision", "monitor")),
                actions=actions,
                confidence=_safe_float(legacy.get("confidence"), event.prediction.confidence),
                reason=str(legacy.get("reason", "traffic scene decision")),
                source=source,
                policy_version=str(legacy.get("policy_version", self.policy_version)),
            )
            framework_metadata = legacy.get("framework_metadata", {})
            if isinstance(framework_metadata, dict) and framework_metadata:
                metadata = dict(decision.metadata)
                metadata.update(framework_metadata)
                decision = replace(decision, metadata=metadata)
            if not cloud:
                decision = self._attach_local_uncertainty(event, decision)
                local_signal_metadata = {
                    key: value
                    for key, value in decision.metadata.items()
                    if key in {
                        "traffic_student_candidate_decision",
                        "traffic_student_candidate_confidence",
                        "traffic_rule_candidate_decision",
                        "traffic_student_rule_disagreement",
                        "model_uncertainty",
                        "escalation_expected_gain",
                    }
                }
                decision = self._edge_llm.decide(event, decision, self.policy_version)
                if local_signal_metadata:
                    decision = replace(
                        decision,
                        metadata={**decision.metadata, **local_signal_metadata},
                    )
                decision = self._resolve_current_state_synchronous_uncertainty(
                    event,
                    decision,
                )
                decision = self._ensure_operational_safety(event, decision)
            return decision
        source = "traffic_cloud_candidates" if cloud else "traffic_edge_candidates"
        decision = self.decision_from_candidates(
            event, source, event.prediction.confidence
        )
        if cloud:
            return decision
        edge_decision = self._edge_llm.decide(event, decision, self.policy_version)
        return self._ensure_operational_safety(event, edge_decision)

    def _decision(self, event: SemanticEvent, cloud: bool) -> Any:
        return self._decision_from_legacy(
            event,
            self._legacy_decision(event, cloud),
            cloud,
        )

    def _apply_coordination_policy(
        self,
        event: SemanticEvent,
        local_decision: DecisionEnvelope,
    ) -> DecisionEnvelope:
        """Apply the separately versioned coordination layer after local Q8.

        The default is disabled and therefore byte-for-byte preserves the old
        decision object.  Shadow mode records a suggestion only.  Active mode
        is fail-closed at package load time unless the frozen production gate
        explicitly permits activation.
        """

        if self.coordination_policy_mode == "disabled":
            return local_decision
        metadata = dict(local_decision.metadata)
        raw_context = event.scene_payload.get("neighbor_context")
        local_action = DECISION_TO_ACTION.get(local_decision.decision)
        if not isinstance(raw_context, dict) or local_action is None:
            metadata["coordination_policy"] = {
                "mode": self.coordination_policy_mode,
                "applied": False,
                "reason_code": (
                    "NEIGHBOR_CONTEXT_UNAVAILABLE"
                    if not isinstance(raw_context, dict)
                    else "LOCAL_ACTION_OUTSIDE_A_TO_F"
                ),
                "q8_model_input_changed": False,
            }
            return replace(local_decision, metadata=metadata)

        policy = self._load_coordination_policy()
        declared_policy_version = str(
            event.scene_payload.get("coordination_policy_version", "")
        ).strip()
        if declared_policy_version != policy.version:
            metadata["coordination_policy"] = {
                "mode": self.coordination_policy_mode,
                "applied": False,
                "reason_code": "COORDINATION_POLICY_VERSION_MISMATCH",
                "declared_policy_version": declared_policy_version or None,
                "loaded_policy_version": policy.version,
                "q8_model_input_changed": False,
            }
            return replace(local_decision, metadata=metadata)
        recommendation = policy.recommend(local_action, raw_context)
        coordination_metadata = {
            **recommendation.to_dict(),
            "mode": self.coordination_policy_mode,
            "applied": False,
            "q8_model_input_changed": False,
            "q8_model_parameter_changed": False,
            "collaboration_scheduler_changed": False,
        }
        metadata["coordination_policy"] = coordination_metadata
        if (
            self.coordination_policy_mode != "active"
            or not recommendation.changed
        ):
            return replace(local_decision, metadata=metadata)

        from traffic_system.decision_utils import build_decision_from_student_class

        selected_decision = ACTION_TO_DECISION[recommendation.selected_action]
        legacy = build_decision_from_student_class(
            event.scene_payload,
            selected_decision,
            local_decision.confidence,
            decision_source="edge_neighbor_coordination_policy",
        )
        actions = [
            self._attach_boundary_resources(
                _traffic_action(action, event.scope.region_id),
                event.scope.region_id,
            )
            for action in legacy.get("actions", [])
        ]
        adjusted = build_decision(
            event=event,
            decision=selected_decision,
            actions=actions,
            confidence=local_decision.confidence,
            reason=(
                "verified neighbor-aware policy {} changed {} to {} ({})"
            ).format(
                policy.version,
                recommendation.local_action,
                recommendation.selected_action,
                recommendation.reason_code,
            ),
            source="edge_neighbor_coordination_policy",
            policy_version=local_decision.policy_version,
            route=local_decision.route,
            status=local_decision.status,
        )
        coordination_metadata["applied"] = True
        adjusted = replace(
            adjusted,
            metadata={**metadata, "coordination_policy": coordination_metadata},
        )
        # Existing scene safety semantics remain the final action invariant.
        return self._ensure_operational_safety(event, adjusted)

    @staticmethod
    def _action_road_set_ids(action: Action) -> List[str]:
        road_set_ids = []
        raw_parameter_ids = action.parameters.get("road_set_ids", [])
        if isinstance(raw_parameter_ids, list):
            road_set_ids.extend(str(value) for value in raw_parameter_ids)
        for resource_id in action.resource_ids:
            if resource_id.startswith("traffic_road_set:"):
                road_set_ids.append(resource_id.split(":", 1)[1])
        return list(dict.fromkeys(value for value in road_set_ids if value))

    @staticmethod
    def _canonical_road_set_action(
        road_set_id: str, partial: Mapping[str, Any]
    ) -> Optional[Action]:
        raw_action = partial.get("action", {})
        if not isinstance(raw_action, Mapping):
            raise ValueError("canonical road-set action must be an object")
        action_type = str(raw_action.get("action_type", "")).strip()
        if action_type in {"", "no_action"}:
            return None
        raw_parameters = raw_action.get("parameters", {})
        if not isinstance(raw_parameters, Mapping):
            raise ValueError("canonical road-set action parameters must be an object")
        resource_id = "traffic_road_set:{}".format(road_set_id)
        parameters = dict(raw_parameters)
        parameters.update(
            {
                "canonical_road_set_action": True,
                "road_set_ids": [road_set_id],
                "observation_id": partial.get("observation_id"),
                "window_id": partial.get("window_id"),
                "source_dataset_sha256": partial.get(
                    "source_dataset_sha256"
                ),
                "raw_fragment_sha256": partial.get(
                    "raw_fragment_sha256"
                ),
                "source_content_sha256": partial.get(
                    "source_content_sha256"
                ),
                "feature_content_sha256": partial.get(
                    "feature_content_sha256"
                ),
                "preprocess_version": partial.get("preprocess_version"),
                "model_id": partial.get("model_id"),
                "model_version": partial.get("model_version"),
                "model_artifact_sha256": partial.get(
                    "model_artifact_sha256"
                ),
                "policy_id": partial.get("policy_id"),
                "policy_version": partial.get("policy_version"),
                "policy_artifact_sha256": partial.get(
                    "policy_artifact_sha256"
                ),
                "output_digest": partial.get("output_digest"),
                "action_digest": partial.get("action_digest"),
            }
        )
        if action_type in {
            "variable_speed_limit",
            "ramp_metering",
            "regional_coordination",
            "reroute",
        }:
            parameters["requires_cloud_confirmation"] = True
        priority_by_type = {
            "traffic_advisory": 40,
            "variable_speed_limit": 70,
            "ramp_metering": 70,
            "regional_coordination": 80,
            "reroute": 80,
        }
        return Action(
            action_type=action_type,
            target_ids=[resource_id],
            resource_ids=[resource_id],
            parameters=parameters,
            reason=(
                "deterministic policy over the canonical shared overlap "
                "observation"
            ),
            priority=priority_by_type.get(action_type, 50),
        )

    def _attach_road_set_decisions(
        self,
        event: SemanticEvent,
        decision: DecisionEnvelope,
    ) -> DecisionEnvelope:
        """Expose one auditable member decision per shared road-set object."""
        raw_contexts = event.scene_payload.get(
            "road_set_contexts", event.metadata.get("road_set_contexts", [])
        )
        if not isinstance(raw_contexts, list) or not raw_contexts:
            return decision
        aggregation = event.metadata.get("aggregation", {})
        aggregation_member = (
            str(aggregation.get("member", event.edge_id))
            if isinstance(aggregation, dict)
            else event.edge_id
        )
        own_assessments = {
            str(value.get("road_set_id", "")): value
            for value in self._effective_road_set_assessments(event)
            if isinstance(value, dict)
            and str(value.get("member_region", "")) == event.scope.region_id
        }
        actions = list(decision.actions)
        existing_canonical_ids = {
            road_set_id
            for action in actions
            if bool(action.parameters.get("canonical_road_set_action", False))
            for road_set_id in self._action_road_set_ids(action)
        }
        for road_set_id, partial in sorted(own_assessments.items()):
            if not road_set_id or road_set_id in existing_canonical_ids:
                continue
            canonical_action = self._canonical_road_set_action(
                road_set_id, partial
            )
            if canonical_action is not None:
                actions.append(canonical_action)
                existing_canonical_ids.add(road_set_id)
        decision = replace(decision, actions=actions)
        road_set_decisions = []
        for context in raw_contexts:
            if not isinstance(context, dict):
                continue
            road_set_id = str(context.get("road_set_id", ""))
            if not road_set_id:
                continue
            own_partial = own_assessments.get(road_set_id) or next(
                (
                    partial
                    for partial in context.get("partial_assessments", [])
                    if isinstance(partial, dict)
                    and str(partial.get("member_region", ""))
                    == event.scope.region_id
                ),
                {},
            )
            action_indices = [
                index
                for index, action in enumerate(decision.actions)
                if road_set_id in self._action_road_set_ids(action)
            ]
            action_types = [
                decision.actions[index].action_type for index in action_indices
            ]
            if action_indices:
                selected_action = max(
                    (decision.actions[index] for index in action_indices),
                    key=lambda action: action.priority,
                )
                road_set_decision = (
                    "congestion_warning"
                    if selected_action.action_type == "traffic_advisory"
                    else selected_action.action_type
                )
            elif str(
                own_partial.get("action", {}).get("action_type", "")
                if isinstance(own_partial.get("action"), dict)
                else ""
            ) == "no_action":
                road_set_decision = "no_action"
            elif decision.decision in {"no_action", "monitor", "abstain"}:
                road_set_decision = decision.decision
            else:
                # A model may select an interior-node control that does not
                # touch this boundary road set.  Record that fact instead of
                # falsely copying the region decision onto every road set.
                road_set_decision = "no_road_set_action"
            road_set_decisions.append(
                {
                    "schema_version": 2,
                    "road_set_id": road_set_id,
                    "resource_id": "traffic_road_set:{}".format(road_set_id),
                    "member_region": event.scope.region_id,
                    "aggregation_member": own_partial.get(
                        "aggregation_member",
                        aggregation_member,
                    ),
                    "required_regions": list(
                        context.get("required_regions", [])
                    ),
                    "required_members": list(
                        context.get("required_members", [])
                    ),
                    "required_evidence_level": context.get(
                        "required_evidence_level"
                    ),
                    "decision": road_set_decision,
                    "action_indices": action_indices,
                    "action_types": action_types,
                    "confidence": round(
                        _safe_float(
                            own_partial.get("confidence"),
                            decision.confidence,
                        ),
                        6,
                    ),
                    "observation_id": own_partial.get("observation_id"),
                    "window_id": own_partial.get("window_id"),
                    "source_content_sha256": own_partial.get(
                        "source_content_sha256"
                    ),
                    "preprocess_version": own_partial.get(
                        "preprocess_version"
                    ),
                    "model_version": own_partial.get("model_version"),
                    "model_artifact_sha256": own_partial.get(
                        "model_artifact_sha256"
                    ),
                    "policy_version": own_partial.get("policy_version"),
                    "policy_artifact_sha256": own_partial.get(
                        "policy_artifact_sha256"
                    ),
                    "output_digest": own_partial.get("output_digest"),
                    "action_digest": own_partial.get("action_digest"),
                    "raw_fragment_digest_recomputed": bool(
                        own_partial.get(
                            "raw_fragment_digest_recomputed", False
                        )
                    ),
                    "deterministic_policy_recomputed": bool(
                        own_partial.get(
                            "deterministic_policy_recomputed", False
                        )
                    ),
                    "recomputation_source": own_partial.get(
                        "recomputation_source"
                    ),
                    "compact_self_validation_errors": list(
                        own_partial.get(
                            "compact_self_validation_errors", []
                        )
                    ),
                    "road_set_conflict_suspected": bool(
                        context.get("road_set_conflict_suspected", False)
                    ),
                    "conflict_kind": str(context.get("conflict_kind", "")),
                    "conflict_reason": str(
                        context.get("conflict_reason", "")
                    ),
                }
            )
        if not road_set_decisions:
            return decision
        metadata = dict(decision.metadata)
        metadata.update(
            {
                "decision_granularity": "road_set",
                "deployment_partition_region": event.scope.region_id,
                "road_set_decisions": road_set_decisions,
                "road_set_decision_count": len(road_set_decisions),
                "global_action_collapsed": False,
                "road_set_conflict_suspected": any(
                    item["road_set_conflict_suspected"]
                    for item in road_set_decisions
                ),
            }
        )
        return replace(decision, metadata=metadata)

    def edge_decide(self, event: SemanticEvent) -> Any:
        return self._attach_road_set_decisions(
            event,
            self._apply_coordination_policy(
                event,
                self._decision(event, cloud=False),
            ),
        )

    def cloud_decide(self, event: SemanticEvent) -> Any:
        if (
            self.cloud_model_path is None
            and self.current_state_cloud_model_path is None
        ):
            return self._attach_road_set_decisions(
                event, self._decision(event, cloud=True)
            )
        return list(self.cloud_decide_batch([event]))[0]

    def cloud_decide_batch(
        self,
        events: Sequence[SemanticEvent],
    ) -> Sequence[Any]:
        normalized = list(events)
        if not normalized:
            return []
        if (
            self.cloud_model_path is None
            and self.current_state_cloud_model_path is None
        ):
            return [
                self._attach_road_set_decisions(
                    event, self._decision(event, cloud=True)
                )
                for event in normalized
            ]

        decisions: List[Optional[Any]] = [None] * len(normalized)
        model_indices = [
            index
            for index, event in enumerate(normalized)
            if bool(event.scene_payload)
            and self._has_cloud_feature_evidence(event)
            and self._cloud_contract_for_payload(event.scene_payload) is not None
        ]
        if model_indices:
            model_events = [normalized[index] for index in model_indices]
            legacies = self._cloud_legacy_decisions(model_events)
            for index, event, legacy in zip(model_indices, model_events, legacies):
                decisions[index] = self._decision_from_legacy(
                    event,
                    legacy,
                    cloud=True,
                )
        model_index_set = set(model_indices)
        summary_indices = [
            index
            for index, event in enumerate(normalized)
            if bool(event.scene_payload) and index not in model_index_set
        ]
        for index in summary_indices:
            event = normalized[index]
            decisions[index] = self._decision_from_legacy(
                event,
                self._cloud_summary_legacy_decision(
                    event, len(summary_indices)
                ),
                cloud=True,
            )
        for index, event in enumerate(normalized):
            if decisions[index] is None:
                decisions[index] = self._decision_from_legacy(
                    event,
                    {},
                    cloud=True,
                )
        return [
            self._attach_road_set_decisions(event, decision)
            for event, decision in zip(normalized, decisions)
            if decision is not None
        ]

    def action_conflict(self, left: Action, right: Action) -> Tuple[bool, str]:
        shared_road_sets = sorted(
            resource
            for resource in set(left.resource_ids) & set(right.resource_ids)
            if resource.startswith("traffic_road_set:")
        )
        if shared_road_sets:
            left_canonical = bool(
                left.parameters.get("canonical_road_set_action", False)
            )
            right_canonical = bool(
                right.parameters.get("canonical_road_set_action", False)
            )
            if not (left_canonical and right_canonical):
                return True, "overlap_action_digest_mismatch"
            comparisons = (
                (
                    (
                        "window_id",
                        "source_dataset_sha256",
                        "raw_fragment_sha256",
                        "source_content_sha256",
                    ),
                    "overlap_content_digest_mismatch",
                ),
                (
                    ("feature_content_sha256",),
                    "overlap_feature_digest_mismatch",
                ),
                (
                    ("preprocess_version",),
                    "overlap_preprocess_version_mismatch",
                ),
                (
                    (
                        "model_id",
                        "model_version",
                        "model_artifact_sha256",
                    ),
                    "overlap_model_version_mismatch",
                ),
                (
                    (
                        "policy_id",
                        "policy_version",
                        "policy_artifact_sha256",
                    ),
                    "overlap_policy_version_mismatch",
                ),
                (("output_digest",), "overlap_output_digest_mismatch"),
                (("action_digest",), "overlap_action_digest_mismatch"),
            )
            for fields, kind in comparisons:
                if any(
                    left.parameters.get(field)
                    != right.parameters.get(field)
                    for field in fields
                ):
                    return True, kind
            if left.action_type != right.action_type:
                return True, "overlap_action_digest_mismatch"
            left_policy_parameters = {
                key: value
                for key, value in left.parameters.items()
                if key
                not in {
                    "canonical_road_set_action",
                    "road_set_ids",
                    "observation_id",
                    "window_id",
                    "source_dataset_sha256",
                    "raw_fragment_sha256",
                    "source_content_sha256",
                    "feature_content_sha256",
                    "preprocess_version",
                    "model_id",
                    "model_version",
                    "model_artifact_sha256",
                    "policy_id",
                    "policy_version",
                    "policy_artifact_sha256",
                    "output_digest",
                    "action_digest",
                    "requires_cloud_confirmation",
                }
            }
            right_policy_parameters = {
                key: value
                for key, value in right.parameters.items()
                if key
                not in {
                    "canonical_road_set_action",
                    "road_set_ids",
                    "observation_id",
                    "window_id",
                    "source_dataset_sha256",
                    "raw_fragment_sha256",
                    "source_content_sha256",
                    "feature_content_sha256",
                    "preprocess_version",
                    "model_id",
                    "model_version",
                    "model_artifact_sha256",
                    "policy_id",
                    "policy_version",
                    "policy_artifact_sha256",
                    "output_digest",
                    "action_digest",
                    "requires_cloud_confirmation",
                }
            }
            if left_policy_parameters != right_policy_parameters:
                return True, "overlap_action_digest_mismatch"
            return False, ""
        if left.action_type != right.action_type:
            return False, ""
        if left.action_type == "variable_speed_limit":
            delta = abs(
                _safe_float(left.parameters.get("target_speed_mph"))
                - _safe_float(right.parameters.get("target_speed_mph"))
            )
            return (delta > 10.0, "boundary_vsl_discontinuity" if delta > 10.0 else "")
        if left.action_type == "ramp_metering":
            delta = abs(
                _safe_float(left.parameters.get("metering_rate_veh_per_hour"))
                - _safe_float(right.parameters.get("metering_rate_veh_per_hour"))
            )
            return (delta > 180.0, "boundary_ramp_rate_discontinuity" if delta > 180.0 else "")
        if left.action_type == "reroute":
            total = _safe_float(left.parameters.get("diversion_ratio")) + _safe_float(
                right.parameters.get("diversion_ratio")
            )
            return (total > 0.5, "alternate_corridor_overload" if total > 0.5 else "")
        return False, ""

    def resolve_action_conflict(
        self,
        left: Action,
        right: Action,
        left_event: SemanticEvent,
        right_event: SemanticEvent,
    ) -> Tuple[Action, Action, str]:
        if any(
            resource.startswith("traffic_road_set:")
            for resource in set(left.resource_ids) & set(right.resource_ids)
        ):
            # Never average, cap, or otherwise manufacture a shared-boundary
            # control from mismatched canonical inputs/versions.  A successful
            # exact-owner evidence pull reruns the deterministic policy; a
            # failed pull leaves this conflict residual so authorization stays
            # deferred.
            return (
                left,
                right,
                "canonical road-set evidence reconciliation is required",
            )
        if left.action_type == "variable_speed_limit":
            value = round(
                (_safe_float(left.parameters.get("target_speed_mph")) + _safe_float(right.parameters.get("target_speed_mph")))
                / 10.0
            ) * 5
            field_name = "target_speed_mph"
        elif left.action_type == "ramp_metering":
            value = round(
                (_safe_float(left.parameters.get("metering_rate_veh_per_hour")) + _safe_float(right.parameters.get("metering_rate_veh_per_hour")))
                / 120.0
            ) * 60
            field_name = "metering_rate_veh_per_hour"
        elif left.action_type == "reroute":
            total = max(
                0.001,
                _safe_float(left.parameters.get("diversion_ratio"))
                + _safe_float(right.parameters.get("diversion_ratio")),
            )
            left_params = dict(left.parameters)
            right_params = dict(right.parameters)
            left_params["diversion_ratio"] = round(
                _safe_float(left.parameters.get("diversion_ratio")) * 0.5 / total, 3
            )
            right_params["diversion_ratio"] = round(
                _safe_float(right.parameters.get("diversion_ratio")) * 0.5 / total, 3
            )
            reason = "cloud capped the combined diversion ratio at 0.5"
            return replace(left, parameters=left_params, reason=reason), replace(
                right, parameters=right_params, reason=reason
            ), reason
        else:
            return super().resolve_action_conflict(left, right, left_event, right_event)
        left_params = dict(left.parameters)
        right_params = dict(right.parameters)
        left_params[field_name] = int(value)
        right_params[field_name] = int(value)
        reason = "cloud synchronized the boundary traffic parameter"
        return replace(left, parameters=left_params, reason=reason), replace(
            right, parameters=right_params, reason=reason
        ), reason
