"""Traffic-only summary-first evidence escalation.

This module is intentionally an adapter around the frozen traffic plugin and
the framework evidence cache.  It does not replace the perception, risk,
Student/Q4, ExtraTrees, routing, or scheduling policies.  A full normalized
event is retained in the authenticated edge cache, while the first cloud data
plane message always contains summary evidence only.

The cloud may then request feature evidence for an explicitly enumerated
reason.  Raw evidence is not a proactive upload level: it is considered only
after a feature-enriched coordination pass still reports a canonical shared
road-set conflict.
"""

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from cloud_edge_framework.contracts import SemanticEvent
from cloud_edge_framework.evidence import EvidencePlanner
from cloud_edge_framework.selective_evidence_pull import (
    HttpEvidencePullClient,
    PullPlan,
    PullTarget,
    SelectiveEvidencePullPlanner,
    fetch_pull_plan,
    merge_pulled_evidence,
)
from freeway_traffic_full.plugin_impl import TrafficPlugin
from traffic_system.road_sets import (
    build_overlap_partial_assessments,
    incident_road_sets,
)


SUMMARY_FIRST = "SUMMARY_FIRST"
FEATURE_ON_DEMAND = "FEATURE_ON_DEMAND"
RAW_ON_DEMAND = "RAW_ON_DEMAND"
SUMMARY_FIRST_SCHEMA_VERSION = 1
DEFAULT_UNCERTAINTY_CONFIDENCE_THRESHOLD = 0.75

FEATURE_TRIGGER_REASONS = frozenset(
    {
        "high_or_severe_risk",
        "uncertainty_threshold_exceeded",
        "q4_specialist_or_rule_disagreement",
        "cross_region_association_conflict",
        "explicit_cloud_feature_request",
    }
)

# Declaration fields are sufficient for the diagnostic summary pass to compare
# the two owner references and expose the already-computed deterministic road-
# set action.  Portable feature bytes live only in the feature cache response.
_COMPACT_ROAD_SET_FIELDS = (
    "schema_version",
    "road_set_id",
    "resource_id",
    "partial",
    "member_region",
    "aggregation_member",
    "required_regions",
    "required_members",
    "primary_sensor_ownership_changed",
    "subscription_role",
    "assessment_source",
    "risk_level",
    "risk_score",
    "confidence",
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
    "raw_fragment_digest_recomputed",
    "deterministic_policy_recomputed",
    "recomputation_source",
    "road_set_conflict_suspected",
)

_DECLARATION_VALIDATION_FIELDS = (
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


def _aggregation_identity(event: SemanticEvent) -> Tuple[str, str, List[str]]:
    raw = event.metadata.get("aggregation", {})
    if not isinstance(raw, Mapping):
        return "", "", []
    group_key = str(raw.get("key", raw.get("group_key", ""))).strip()
    member = str(raw.get("member", "")).strip()
    expected_raw = raw.get("expected_members", [])
    expected = (
        [str(value).strip() for value in expected_raw if str(value).strip()]
        if isinstance(expected_raw, list)
        else []
    )
    return group_key, member, expected


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bool_field(sources: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    return any(source.get(name) is True for source in sources for name in names)


def feature_pull_reasons(
    event: SemanticEvent,
    local_decision: Any,
    *,
    uncertainty_confidence_threshold: float = DEFAULT_UNCERTAINTY_CONFIDENCE_THRESHOLD,
    cross_region_conflict: bool = False,
    explicit_cloud_request: bool = False,
) -> List[str]:
    """Return the closed set of reasons allowed to request traffic features."""

    reasons: List[str] = []
    regional_state = _mapping(event.metadata.get("regional_state"))
    regional_level = str(
        regional_state.get(
            "level",
            event.metadata.get("regional_risk_level", event.risk.level),
        )
    ).strip().lower()
    if regional_level in {"high", "severe"}:
        reasons.append("high_or_severe_risk")

    event_uncertainty = _mapping(event.metadata.get("model_uncertainty"))
    local_metadata = _mapping(getattr(local_decision, "metadata", {}))
    local_uncertainty = _mapping(local_metadata.get("model_uncertainty"))
    uncertainty_sources = (event_uncertainty, local_uncertainty)
    low_confidence = (
        float(event.uncertainty.confidence) < float(uncertainty_confidence_threshold)
        or len(event.uncertainty.prediction_set) > 1
        or _bool_field(
            uncertainty_sources,
            (
                "student_low_confidence",
                "confidence_threshold_exceeded",
                "uncertainty_threshold_exceeded",
            ),
        )
    )
    if low_confidence:
        reasons.append("uncertainty_threshold_exceeded")

    disagreement_sources = (
        event.metadata,
        local_metadata,
        event_uncertainty,
        local_uncertainty,
    )
    if _bool_field(
        disagreement_sources,
        (
            "traffic_student_rule_disagreement",
            "student_rule_disagreement",
            "edge_llm_model_disagreement",
            "q4_specialist_disagreement",
            "q4_rule_disagreement",
        ),
    ):
        reasons.append("q4_specialist_or_rule_disagreement")

    if cross_region_conflict:
        reasons.append("cross_region_association_conflict")
    cloud_policy = _mapping(event.metadata.get("cloud_llm_review_policy"))
    cloud_requested = bool(
        explicit_cloud_request
        or event.metadata.get("cloud_review_requested") is True
        or event.metadata.get("monitoring_force_cloud_review") is True
        or cloud_policy.get("eligible") is True
    )
    if cloud_requested:
        reasons.append("explicit_cloud_feature_request")
    return list(dict.fromkeys(reasons))


def compact_road_set_declarations(values: Any) -> List[Dict[str, Any]]:
    """Remove portable feature bytes from summary-plane road-set declarations."""

    if not isinstance(values, list):
        return []
    return [
        {key: value[key] for key in _COMPACT_ROAD_SET_FIELDS if key in value}
        for value in values
        if isinstance(value, Mapping)
    ]


@dataclass(frozen=True)
class SummaryFirstDirective:
    state: str
    initial_upload_level: str
    feature_pull_requested: bool
    feature_pull_reasons: List[str]
    raw_pull_requested: bool
    authoritative: bool
    oracle_required_level: str
    oracle_reason: str

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = SUMMARY_FIRST_SCHEMA_VERSION
        return value


@dataclass(frozen=True)
class SummaryFirstCoordination:
    initial_result: Dict[str, Any]
    feature_result: Dict[str, Any]
    final_result: Dict[str, Any]
    effective_events: List[SemanticEvent]
    feature_plan: PullPlan
    raw_plan: PullPlan
    feature_errors: List[str]
    raw_errors: List[str]
    authoritative: bool
    final_state: str


class SummaryFirstTrafficPlugin(TrafficPlugin):
    """Opt-in traffic plugin adapter; the frozen base plugin stays unchanged."""

    def prepare_summary_first_upload(
        self,
        event: SemanticEvent,
        local_decision: Any,
        locator: Mapping[str, Any],
        *,
        conflict_suspected: bool = False,
        explicit_cloud_request: bool = False,
    ) -> Tuple[SemanticEvent, SummaryFirstDirective]:
        """Build one summary-only first message while retaining full cache data."""

        cloud_submission = TrafficPlugin.cloud_submission_metadata(
            self, event, local_decision
        )
        enriched = replace(
            event,
            metadata={**event.metadata, **cloud_submission},
        )
        # The frozen planner remains the oracle used by arm A.  Its result is
        # recorded for audit only and never changes the first upload level.
        oracle_policy = TrafficPlugin.evidence_advice(
            self, enriched, local_decision, conflict_suspected
        )
        oracle_plan = EvidencePlanner().plan(
            enriched,
            conflict_suspected=conflict_suspected,
            scene_policy=oracle_policy,
        )
        eligible_reasons = feature_pull_reasons(
            enriched,
            local_decision,
            cross_region_conflict=conflict_suspected,
            explicit_cloud_request=explicit_cloud_request,
        )
        # Arm B changes transport staging, not the frozen traffic selection
        # oracle.  An oracle-summary member stays summary-only even if a broad
        # diagnostic flag is present; an oracle raw target is downgraded to a
        # feature callback and may reach raw only after cloud feature fusion.
        reasons = (
            eligible_reasons
            if oracle_plan.required_level in {"feature", "raw"}
            else []
        )
        directive = SummaryFirstDirective(
            state=SUMMARY_FIRST,
            initial_upload_level="summary",
            feature_pull_requested=bool(reasons),
            feature_pull_reasons=reasons,
            raw_pull_requested=False,
            authoritative=not bool(reasons),
            oracle_required_level=oracle_plan.required_level,
            oracle_reason=oracle_plan.reason,
        )
        summary_event = replace(
            enriched,
            evidence=[item for item in enriched.evidence if item.level == "summary"],
        )
        cloud = TrafficPlugin.prepare_cloud_event(self, summary_event, "summary")
        payload = dict(cloud.scene_payload)
        payload["road_set_assessments"] = compact_road_set_declarations(
            payload.get("road_set_assessments", [])
        )
        metadata = dict(cloud.metadata)
        metadata.update(
            {
                "selected_evidence_level": "summary",
                "traffic_evidence_upload": directive.to_dict(),
                "traffic_cloud_result_authority": (
                    "authoritative_summary"
                    if directive.authoritative
                    else "non_authoritative_pending_feature"
                ),
                "evidence_pull_locator": dict(locator),
            }
        )
        return replace(cloud, scene_payload=payload, metadata=metadata), directive

    def _effective_road_set_assessments(
        self, event: SemanticEvent
    ) -> List[Dict[str, Any]]:
        """Use declarations on the diagnostic pass and verify pulled features."""

        raw_bundles = [
            item
            for item in event.evidence
            if item.level == "raw"
            and item.modality == "traffic_road_set_overlap_raw"
        ]
        if raw_bundles:
            return TrafficPlugin._effective_road_set_assessments(self, event)

        raw_default = event.scene_payload.get("road_set_assessments", [])
        declarations = [
            dict(value)
            for value in raw_default
            if isinstance(value, Mapping)
        ] if isinstance(raw_default, list) else []
        aggregation = _mapping(event.metadata.get("aggregation"))
        member = str(aggregation.get("member", event.edge_id))
        catalog = self._load_road_set_catalog()
        expected = {
            item["road_set_id"]: item
            for item in incident_road_sets(catalog, event.scope.region_id)
        }
        by_id: Dict[str, Dict[str, Any]] = {}
        for declaration in declarations:
            road_set_id = str(declaration.get("road_set_id", ""))
            expected_road_set = expected.get(road_set_id)
            if road_set_id in by_id or expected_road_set is None:
                raise ValueError("traffic summary road-set coverage is invalid")
            if (
                declaration.get("assessment_source")
                != "canonical_overlap_observation"
                or str(declaration.get("member_region", ""))
                != event.scope.region_id
                or str(declaration.get("aggregation_member", "")) != member
                or str(declaration.get("resource_id", ""))
                != expected_road_set["resource_id"]
                or sorted(declaration.get("required_regions", []))
                != list(expected_road_set["required_regions"])
            ):
                raise ValueError("traffic summary road-set owner binding is invalid")
            by_id[road_set_id] = declaration
        if set(by_id) != set(expected):
            raise ValueError("traffic summary road-set declaration set is incomplete")

        feature_bundles = [
            item
            for item in event.evidence
            if item.level == "feature"
            and item.modality == "traffic_road_set_overlap_features"
        ]
        if not feature_bundles:
            return declarations
        if len(feature_bundles) != 1 or not isinstance(
            feature_bundles[0].inline, Mapping
        ):
            raise ValueError("traffic feature pull requires one overlap bundle")
        inline = dict(feature_bundles[0].inline)
        if (
            int(inline.get("schema_version", 0)) != 2
            or inline.get("kind")
            != "canonical_road_set_overlap_feature_bundle"
            or not isinstance(inline.get("observations"), list)
        ):
            raise ValueError("traffic overlap feature callback contract is invalid")
        expected_sample_id = int(event.scene_payload.get("sample_id", -1))
        expected_split = str(event.scene_payload.get("sample_split", ""))
        observations = []
        for value in inline["observations"]:
            if not isinstance(value, Mapping):
                raise ValueError("traffic overlap feature observation is invalid")
            observation = dict(value)
            if (
                str(observation.get("dataset", "")) != "PEMS08"
                or str(observation.get("split", "")) != expected_split
                or int(observation.get("sample_id", -2)) != expected_sample_id
                or str(observation.get("window_id", ""))
                != "PEMS08:{}:{}".format(expected_split, expected_sample_id)
            ):
                raise ValueError("pulled overlap feature window is invalid")
            observations.append(observation)
        rebuilt = build_overlap_partial_assessments(
            catalog,
            event.scope.region_id,
            member,
            observations,
            self._approved_road_set_model_artifact_sha256(),
            self._load_road_set_rule_config(),
            "cloud_pulled_feature_recomputed",
        )
        for assessment in rebuilt:
            declaration = by_id[assessment["road_set_id"]]
            errors = [
                field
                for field in _DECLARATION_VALIDATION_FIELDS
                if declaration.get(field) != assessment.get(field)
            ]
            if errors:
                assessment["compact_self_validation_errors"] = errors
        return rebuilt


def _empty_pull_plan(reason: str) -> PullPlan:
    return PullPlan(
        triggered=False,
        trigger_reasons=[],
        road_set_ids=[],
        requested_members=[],
        requested_level=None,
        road_set_members={},
        road_set_levels={},
        initial_evidence_sufficient=False,
        targets=[],
        no_pull_reason=reason,
    )


def _unavailable_pull_plan(
    requested_level: str,
    reason: str,
    *,
    trigger_reasons: Sequence[str] = (),
    road_set_ids: Sequence[str] = (),
    requested_members: Sequence[str] = (),
    road_set_members: Optional[Mapping[str, Sequence[str]]] = None,
) -> PullPlan:
    return PullPlan(
        triggered=True,
        trigger_reasons=list(trigger_reasons),
        road_set_ids=list(road_set_ids),
        requested_members=list(requested_members),
        requested_level=requested_level,
        road_set_members={
            key: list(value)
            for key, value in (road_set_members or {}).items()
        },
        road_set_levels={key: requested_level for key in road_set_ids},
        initial_evidence_sufficient=False,
        targets=[],
        no_pull_reason=reason,
    )


class SummaryFirstEvidenceCoordinator:
    """Execute summary -> feature -> optional raw as an auditable state machine."""

    def __init__(
        self,
        cloud_runtime: Any,
        evidence_pull_client: HttpEvidencePullClient,
        *,
        road_set_planner: Optional[SelectiveEvidencePullPlanner] = None,
    ) -> None:
        self.cloud_runtime = cloud_runtime
        self.evidence_pull_client = evidence_pull_client
        self.road_set_planner = road_set_planner or SelectiveEvidencePullPlanner()

    @staticmethod
    def _initial_cross_region_conflicts(
        events: Sequence[SemanticEvent],
        initial_result: Mapping[str, Any],
    ) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
        """Read audited conflict flags without invoking the raw-pull planner."""

        member_by_event_id = {
            event.event_id: _aggregation_identity(event)[1] for event in events
        }
        road_set_members: Dict[str, set] = {}
        raw_decisions = initial_result.get("decisions", [])
        if isinstance(raw_decisions, list):
            for raw_decision in raw_decisions:
                if not isinstance(raw_decision, Mapping):
                    continue
                decision_member = ""
                event_ids = raw_decision.get("event_ids", [])
                if isinstance(event_ids, list):
                    for event_id in event_ids:
                        decision_member = member_by_event_id.get(
                            str(event_id), ""
                        )
                        if decision_member:
                            break
                metadata = _mapping(raw_decision.get("metadata"))
                rows = metadata.get("road_set_decisions", [])
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if (
                        not isinstance(row, Mapping)
                        or row.get("road_set_conflict_suspected") is not True
                    ):
                        continue
                    road_set_id = str(row.get("road_set_id", "")).strip()
                    if not road_set_id:
                        continue
                    state = road_set_members.setdefault(road_set_id, set())
                    raw_members = row.get("required_members", [])
                    if isinstance(raw_members, list):
                        state.update(
                            str(value).strip()
                            for value in raw_members
                            if str(value).strip()
                        )
                    if decision_member:
                        state.add(decision_member)
        normalized = {
            road_set_id: sorted(members)
            for road_set_id, members in sorted(road_set_members.items())
        }
        return (
            sorted({member for members in normalized.values() for member in members}),
            sorted(normalized),
            normalized,
        )

    @staticmethod
    def _feature_plan(
        events: Sequence[SemanticEvent],
        conflict_members: Sequence[str],
        conflict_road_set_ids: Sequence[str],
        conflict_road_set_members: Mapping[str, Sequence[str]],
    ) -> PullPlan:
        requested: Dict[str, SemanticEvent] = {}
        seen_members = set()
        reasons = set()
        group_keys = set()
        for event in events:
            group_key, member, expected = _aggregation_identity(event)
            if not group_key or not member or member not in expected:
                return _unavailable_pull_plan(
                    "feature", "untrusted_aggregation_identity"
                )
            if member in seen_members:
                return _unavailable_pull_plan(
                    "feature", "duplicate_aggregation_member"
                )
            seen_members.add(member)
            group_keys.add(group_key)
            directive = _mapping(event.metadata.get("traffic_evidence_upload"))
            raw_reasons = directive.get("feature_pull_reasons", [])
            event_reasons = (
                [str(value) for value in raw_reasons]
                if isinstance(raw_reasons, list)
                else []
            )
            if any(value not in FEATURE_TRIGGER_REASONS for value in event_reasons):
                return _unavailable_pull_plan(
                    "feature", "invalid_traffic_feature_trigger_reason"
                )
            if directive.get("feature_pull_requested") is True:
                requested[member] = event
                reasons.update(event_reasons)
        if len(group_keys) != 1:
            return _unavailable_pull_plan("feature", "mixed_group_keys")
        event_by_member = {
            _aggregation_identity(event)[1]: event for event in events
        }
        for member in conflict_members:
            event = event_by_member.get(member)
            if event is None:
                return _unavailable_pull_plan(
                    "feature",
                    "conflict_member_not_in_aggregation",
                    trigger_reasons=["cross_region_association_conflict"],
                    road_set_ids=conflict_road_set_ids,
                    requested_members=conflict_members,
                    road_set_members=conflict_road_set_members,
                )
            requested[member] = event
            reasons.add("cross_region_association_conflict")
        if not requested:
            return _empty_pull_plan("no_feature_trigger")
        targets: List[PullTarget] = []
        for member, event in sorted(requested.items()):
            if any(item.level == "feature" for item in event.evidence):
                continue
            locator = event.metadata.get("evidence_pull_locator")
            if not isinstance(locator, Mapping):
                return _unavailable_pull_plan(
                    "feature",
                    "member_locator_unavailable",
                    trigger_reasons=sorted(reasons),
                    road_set_ids=conflict_road_set_ids,
                    requested_members=sorted(requested),
                    road_set_members=conflict_road_set_members,
                )
            group_key, trusted_member, _ = _aggregation_identity(event)
            if (
                str(locator.get("scene", "")) != event.scene
                or str(locator.get("group_key", "")) != group_key
                or str(locator.get("member", "")) != trusted_member
                or str(locator.get("event_id", "")) != event.event_id
                or "feature" not in locator.get("allowed_levels", [])
            ):
                return _unavailable_pull_plan(
                    "feature",
                    "feature_locator_identity_mismatch",
                    trigger_reasons=sorted(reasons),
                    road_set_ids=conflict_road_set_ids,
                    requested_members=sorted(requested),
                    road_set_members=conflict_road_set_members,
                )
            road_set_ids = [
                road_set_id
                for road_set_id, members in conflict_road_set_members.items()
                if member in members
            ]
            targets.append(
                PullTarget(
                    road_set_id=road_set_ids[0] if road_set_ids else "",
                    road_set_ids=sorted(road_set_ids),
                    member=member,
                    event_id=event.event_id,
                    requested_level="feature",
                    locator=dict(locator),
                )
            )
        return PullPlan(
            triggered=True,
            trigger_reasons=sorted(reasons),
            road_set_ids=list(conflict_road_set_ids),
            requested_members=sorted(requested),
            requested_level="feature",
            road_set_members={
                key: list(value)
                for key, value in conflict_road_set_members.items()
            },
            road_set_levels={key: "feature" for key in conflict_road_set_ids},
            initial_evidence_sufficient=not targets,
            targets=targets,
            no_pull_reason=("required_feature_already_present" if not targets else ""),
        )

    @staticmethod
    def _mark_feature_state(
        events: Sequence[SemanticEvent], requested_members: Sequence[str]
    ) -> List[SemanticEvent]:
        requested = set(requested_members)
        result = []
        for event in events:
            member = _aggregation_identity(event)[1]
            if member not in requested:
                result.append(event)
                continue
            metadata = dict(event.metadata)
            directive = _mapping(metadata.get("traffic_evidence_upload"))
            directive.update(
                {
                    "state": FEATURE_ON_DEMAND,
                    "feature_pull_completed": True,
                    "raw_pull_requested": False,
                    "authoritative": True,
                }
            )
            metadata["traffic_evidence_upload"] = directive
            metadata["traffic_cloud_result_authority"] = "authoritative_feature"
            result.append(replace(event, metadata=metadata))
        return result

    @staticmethod
    def _mark_raw_state(
        events: Sequence[SemanticEvent], requested_members: Sequence[str]
    ) -> List[SemanticEvent]:
        requested = set(requested_members)
        result = []
        for event in events:
            member = _aggregation_identity(event)[1]
            if member not in requested:
                result.append(event)
                continue
            metadata = dict(event.metadata)
            directive = _mapping(metadata.get("traffic_evidence_upload"))
            directive.update(
                {
                    "state": RAW_ON_DEMAND,
                    "feature_pull_completed": True,
                    "raw_pull_requested": True,
                    "raw_pull_completed": True,
                    "authoritative": True,
                }
            )
            metadata["traffic_evidence_upload"] = directive
            metadata["traffic_cloud_result_authority"] = "authoritative_raw"
            result.append(replace(event, metadata=metadata))
        return result

    def coordinate(self, events: Sequence[SemanticEvent]) -> SummaryFirstCoordination:
        normalized = list(events)
        if not normalized:
            raise ValueError("summary-first coordination requires events")
        if any(
            {item.level for item in event.evidence} != {"summary"}
            for event in normalized
        ):
            raise ValueError("every first traffic upload must be summary-only")

        initial = dict(self.cloud_runtime.coordinate(normalized))
        (
            conflict_members,
            conflict_road_set_ids,
            conflict_road_set_members,
        ) = self._initial_cross_region_conflicts(normalized, initial)
        feature_plan = self._feature_plan(
            normalized,
            conflict_members,
            conflict_road_set_ids,
            conflict_road_set_members,
        )
        if not feature_plan.triggered:
            authoritative = feature_plan.no_pull_reason == "no_feature_trigger"
            return SummaryFirstCoordination(
                initial_result=initial,
                feature_result=initial,
                final_result=initial,
                effective_events=normalized,
                feature_plan=feature_plan,
                raw_plan=_empty_pull_plan("feature_stage_not_required"),
                feature_errors=[],
                raw_errors=[],
                authoritative=authoritative,
                final_state=SUMMARY_FIRST,
            )

        if not feature_plan.initial_evidence_sufficient and not feature_plan.targets:
            return SummaryFirstCoordination(
                initial_result=initial,
                feature_result=initial,
                final_result=initial,
                effective_events=normalized,
                feature_plan=feature_plan,
                raw_plan=_empty_pull_plan("feature_pull_unavailable"),
                feature_errors=[feature_plan.no_pull_reason or "feature_pull_unavailable"],
                raw_errors=[],
                authoritative=False,
                final_state=SUMMARY_FIRST,
            )

        pulled_feature, feature_errors = fetch_pull_plan(
            self.evidence_pull_client, feature_plan
        )
        if feature_errors or len(pulled_feature) != len(feature_plan.targets):
            return SummaryFirstCoordination(
                initial_result=initial,
                feature_result=initial,
                final_result=initial,
                effective_events=normalized,
                feature_plan=feature_plan,
                raw_plan=_empty_pull_plan("feature_pull_incomplete"),
                feature_errors=feature_errors,
                raw_errors=[],
                authoritative=False,
                final_state=SUMMARY_FIRST,
            )
        feature_events = self._mark_feature_state(
            merge_pulled_evidence(normalized, pulled_feature),
            feature_plan.requested_members,
        )
        feature_result = dict(self.cloud_runtime.coordinate(feature_events))
        raw_plan = self.road_set_planner.plan(feature_events, feature_result)
        if not raw_plan.triggered:
            return SummaryFirstCoordination(
                initial_result=initial,
                feature_result=feature_result,
                final_result=feature_result,
                effective_events=feature_events,
                feature_plan=feature_plan,
                raw_plan=raw_plan,
                feature_errors=[],
                raw_errors=[],
                authoritative=True,
                final_state=FEATURE_ON_DEMAND,
            )
        if not raw_plan.initial_evidence_sufficient and not raw_plan.targets:
            return SummaryFirstCoordination(
                initial_result=initial,
                feature_result=feature_result,
                final_result=feature_result,
                effective_events=feature_events,
                feature_plan=feature_plan,
                raw_plan=raw_plan,
                feature_errors=[],
                raw_errors=[raw_plan.no_pull_reason or "raw_pull_unavailable"],
                authoritative=False,
                final_state=FEATURE_ON_DEMAND,
            )
        if raw_plan.requested_level not in {"raw", "mixed"} or any(
            target.requested_level != "raw" for target in raw_plan.targets
        ):
            return SummaryFirstCoordination(
                initial_result=initial,
                feature_result=feature_result,
                final_result=feature_result,
                effective_events=feature_events,
                feature_plan=feature_plan,
                raw_plan=raw_plan,
                feature_errors=[],
                raw_errors=["road-set conflict did not request raw evidence"],
                authoritative=False,
                final_state=FEATURE_ON_DEMAND,
            )
        pulled_raw, raw_errors = fetch_pull_plan(
            self.evidence_pull_client, raw_plan
        )
        if raw_errors or len(pulled_raw) != len(raw_plan.targets):
            return SummaryFirstCoordination(
                initial_result=initial,
                feature_result=feature_result,
                final_result=feature_result,
                effective_events=feature_events,
                feature_plan=feature_plan,
                raw_plan=raw_plan,
                feature_errors=[],
                raw_errors=raw_errors,
                authoritative=False,
                final_state=FEATURE_ON_DEMAND,
            )
        raw_events = self._mark_raw_state(
            merge_pulled_evidence(feature_events, pulled_raw),
            raw_plan.requested_members,
        )
        final = dict(self.cloud_runtime.coordinate(raw_events))
        residual_plan = self.road_set_planner.plan(raw_events, final)
        residual_conflict = bool(
            residual_plan.triggered
            or int(final.get("initial_conflict_count", 0))
            or int(final.get("residual_conflict_count", 0))
            or final.get("globally_consistent") is not True
        )
        return SummaryFirstCoordination(
            initial_result=initial,
            feature_result=feature_result,
            final_result=final,
            effective_events=raw_events,
            feature_plan=feature_plan,
            raw_plan=raw_plan,
            feature_errors=[],
            raw_errors=(
                ["raw_rerun_did_not_clear_conflict"] if residual_conflict else []
            ),
            authoritative=not residual_conflict,
            final_state=RAW_ON_DEMAND,
        )
