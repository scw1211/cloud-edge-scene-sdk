"""Road-set decision objects derived from the frozen traffic cut-edge topology.

The four METIS partitions remain deployment and sensor-ownership boundaries.
Each topology ``region_pair`` instead becomes one shared road-set decision
object.  Both owners publish a partial assessment under the same stable ID;
the cloud can then audit disagreement without pretending that either owner
observed the other owner's sensors.
"""

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from traffic_system.overlap_observations import (
    recompute_canonical_overlap_observation_from_raw,
    validate_canonical_overlap_observation,
)


RISK_PRIORITY = {"low": 0, "medium": 1, "high": 2, "severe": 3}
ROAD_SET_SCHEMA_VERSION = 1
ROAD_SET_RESOURCE_PREFIX = "traffic_road_set:"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _node_ids(values: Any) -> List[int]:
    if not isinstance(values, (list, tuple, set)):
        return []
    result = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            node_id = int(value)
        except (TypeError, ValueError):
            continue
        if node_id >= 0 and node_id not in result:
            result.append(node_id)
    return sorted(result)


def stable_road_set_id(left_region: str, right_region: str) -> str:
    """Return a direction-independent ID without inventing geographic names."""
    regions = sorted((_text(left_region), _text(right_region)))
    if not regions[0] or not regions[1] or regions[0] == regions[1]:
        raise ValueError("road set requires two distinct non-empty regions")
    return "rs_{}__{}".format(regions[0], regions[1])


def road_set_resource_id(road_set_id: str) -> str:
    road_set_id = _text(road_set_id)
    if not road_set_id:
        raise ValueError("road_set_id must not be empty")
    return ROAD_SET_RESOURCE_PREFIX + road_set_id


def build_road_set_catalog(topology: Mapping[str, Any]) -> Dict[str, Any]:
    """Compile the topology's cut-edge pairs into stable shared road sets."""
    if not isinstance(topology, Mapping):
        raise ValueError("traffic topology must be an object")
    pairs = topology.get("region_pairs", [])
    if not isinstance(pairs, list):
        raise ValueError("traffic topology region_pairs must be a list")

    road_sets: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("traffic topology region pair must be an object")
        declared_left = _text(pair.get("left_region"))
        declared_right = _text(pair.get("right_region"))
        road_set_id = stable_road_set_id(declared_left, declared_right)
        if road_set_id in seen:
            raise ValueError("duplicate traffic road set: {}".format(road_set_id))
        seen.add(road_set_id)

        required_regions = sorted((declared_left, declared_right))
        nodes_by_declared_region = {
            declared_left: _node_ids(pair.get("left_boundary_nodes", [])),
            declared_right: _node_ids(pair.get("right_boundary_nodes", [])),
        }
        member_nodes_by_region = {
            region: nodes_by_declared_region[region]
            for region in required_regions
        }
        if any(not nodes for nodes in member_nodes_by_region.values()):
            raise ValueError(
                "traffic road set {} requires boundary nodes on both sides".format(
                    road_set_id
                )
            )
        member_nodes = sorted(
            {
                node
                for nodes in member_nodes_by_region.values()
                for node in nodes
            }
        )
        road_sets.append(
            {
                "schema_version": ROAD_SET_SCHEMA_VERSION,
                "road_set_id": road_set_id,
                "resource_id": road_set_resource_id(road_set_id),
                "kind": "cut_edge_boundary_road_set",
                "required_regions": required_regions,
                "member_nodes": member_nodes,
                "member_nodes_by_region": member_nodes_by_region,
                "cut_edge_count": max(0, int(pair.get("cut_edge_count", 0))),
                "adjacent_road_set_ids": [],
            }
        )

    for road_set in road_sets:
        own_regions = set(road_set["required_regions"])
        road_set["adjacent_road_set_ids"] = sorted(
            other["road_set_id"]
            for other in road_sets
            if other["road_set_id"] != road_set["road_set_id"]
            and own_regions & set(other["required_regions"])
        )

    by_region: Dict[str, List[str]] = {}
    for road_set in road_sets:
        for region in road_set["required_regions"]:
            by_region.setdefault(region, []).append(road_set["road_set_id"])
    for region in by_region:
        by_region[region].sort()
    return {
        "schema_version": ROAD_SET_SCHEMA_VERSION,
        "source_topology_method": _text(
            topology.get("method", "road_graph_cut_edges")
        ),
        "road_sets": sorted(road_sets, key=lambda value: value["road_set_id"]),
        "road_set_ids_by_region": dict(sorted(by_region.items())),
    }


def incident_road_sets(
    catalog: Mapping[str, Any], region_id: str
) -> List[Dict[str, Any]]:
    region_id = _text(region_id)
    return [
        dict(value)
        for value in catalog.get("road_sets", [])
        if isinstance(value, Mapping)
        and region_id in value.get("required_regions", [])
    ]


def road_sets_for_action(
    catalog: Mapping[str, Any],
    region_id: str,
    target_node_ids: Sequence[int],
    action_type: str,
) -> List[Dict[str, Any]]:
    """Select shared road sets genuinely affected by one owner's action."""
    incident = incident_road_sets(catalog, region_id)
    if _text(action_type) in {"reroute", "regional_coordination"}:
        return incident
    targets = set(_node_ids(target_node_ids))
    if not targets:
        return []
    return [
        road_set
        for road_set in incident
        if targets
        & set(
            road_set.get("member_nodes_by_region", {}).get(region_id, [])
        )
    ]


def _action_view(action: Any, road_set_id: str) -> Dict[str, Any]:
    if isinstance(action, Mapping):
        action_type = _text(action.get("action_type", action.get("type")))
        parameters = action.get("parameters", action)
        resource_ids = action.get("resource_ids", [])
    else:
        action_type = _text(getattr(action, "action_type", ""))
        parameters = getattr(action, "parameters", {})
        resource_ids = getattr(action, "resource_ids", [])
    if not isinstance(parameters, Mapping):
        parameters = {}
    resource_id = road_set_resource_id(road_set_id)
    if resource_id not in list(resource_ids or []):
        return {}
    parameter_names = (
        "target_speed_mph",
        "metering_rate_veh_per_hour",
        "diversion_ratio",
        "duration_seconds",
        "strategy",
    )
    return {
        "action_type": action_type,
        "parameters": {
            name: parameters[name]
            for name in parameter_names
            if name in parameters
        },
    }


def build_partial_assessments(
    catalog: Mapping[str, Any],
    region_id: str,
    aggregation_member: str,
    managed_node_ids: Sequence[int],
    top_k_risk_nodes: Sequence[Mapping[str, Any]],
    region_summary: Mapping[str, Any],
    actions: Sequence[Any] = (),
) -> List[Dict[str, Any]]:
    """Build owner-local partials for every shared road set incident to a region."""
    region_id = _text(region_id)
    aggregation_member = _text(aggregation_member)
    managed = set(_node_ids(managed_node_ids))
    normalized_top_nodes = [
        dict(node)
        for node in top_k_risk_nodes
        if isinstance(node, Mapping)
    ]
    assessments: List[Dict[str, Any]] = []
    for road_set in incident_road_sets(catalog, region_id):
        road_set_id = road_set["road_set_id"]
        owned_nodes = set(
            _node_ids(
                road_set.get("member_nodes_by_region", {}).get(region_id, [])
            )
        )
        if managed:
            owned_nodes &= managed
        relevant_top_nodes = []
        for node in normalized_top_nodes:
            node_ids = _node_ids([node.get("node_id")])
            if node_ids and node_ids[0] in owned_nodes:
                relevant_top_nodes.append(node)
        if relevant_top_nodes:
            representative = max(
                relevant_top_nodes,
                key=lambda node: (
                    RISK_PRIORITY.get(_text(node.get("risk_level")), -1),
                    _safe_float(node.get("risk_score")),
                ),
            )
            risk_level = _text(representative.get("risk_level"))
            risk_score = _safe_float(representative.get("risk_score"))
            confidence = _safe_float(
                representative.get("risk_confidence"),
                _safe_float(region_summary.get("region_risk_confidence"), 0.0),
            )
            assessment_source = "owner_top_k_boundary_nodes"
        else:
            risk_level = _text(region_summary.get("region_risk_level", "low"))
            risk_score = _safe_float(region_summary.get("region_risk_score"))
            confidence = _safe_float(
                region_summary.get("region_risk_confidence")
            )
            assessment_source = "owner_region_summary_proxy"
        if risk_level not in RISK_PRIORITY:
            risk_level = "low"

        action_views = []
        for action in actions:
            view = _action_view(action, road_set_id)
            if view and view not in action_views:
                action_views.append(view)
        observed_top_node_ids = sorted(
            _node_ids([node.get("node_id") for node in relevant_top_nodes])
        )
        assessments.append(
            {
                "schema_version": ROAD_SET_SCHEMA_VERSION,
                "road_set_id": road_set_id,
                "resource_id": road_set["resource_id"],
                "kind": road_set["kind"],
                "partial": True,
                "member_region": region_id,
                "aggregation_member": aggregation_member,
                "required_regions": list(road_set["required_regions"]),
                # This is a local partial, not yet a pull request.  The cloud
                # populates required_members only after comparing both sides.
                "required_members": [],
                "member_nodes": sorted(owned_nodes),
                "all_road_set_nodes": list(road_set["member_nodes"]),
                "observed_top_k_node_ids": observed_top_node_ids,
                "top_k_boundary_coverage": round(
                    len(observed_top_node_ids) / max(1, len(owned_nodes)), 6
                ),
                "risk_level": risk_level,
                "risk_score": round(max(0.0, min(1.0, risk_score)), 6),
                "confidence": round(max(0.0, min(1.0, confidence)), 6),
                "assessment_source": assessment_source,
                "action_views": action_views,
                "adjacent_road_set_ids": list(
                    road_set.get("adjacent_road_set_ids", [])
                ),
                "road_set_conflict_suspected": False,
            }
        )
    return assessments


_OVERLAP_PARTIAL_FIELDS = (
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


def build_overlap_partial_assessments(
    catalog: Mapping[str, Any],
    region_id: str,
    aggregation_member: str,
    overlap_observations: Sequence[Mapping[str, Any]],
    expected_model_artifact_sha256: str = "",
    rule_config: Mapping[str, Any] = None,
    recomputation_source: str = "edge_local_raw",
) -> List[Dict[str, Any]]:
    """Mirror canonical road-set observations into owner-labelled references.

    The owner labels are outside every content/decision digest.  Both members
    of a road set therefore reference the exact same observation bytes while
    their primary sensor partitions remain disjoint.
    """

    region_id = _text(region_id)
    aggregation_member = _text(aggregation_member)
    observations: Dict[str, Dict[str, Any]] = {}
    for value in overlap_observations:
        if not isinstance(value, Mapping):
            raise ValueError("canonical overlap observation must be an object")
        road_set_id = _text(value.get("road_set_id"))
        if not road_set_id or road_set_id in observations:
            raise ValueError(
                "canonical overlap observation IDs must be unique and non-empty"
            )
        observations[road_set_id] = dict(value)
    expected_ids = {
        road_set["road_set_id"]
        for road_set in incident_road_sets(catalog, region_id)
    }
    if set(observations) != expected_ids:
        raise ValueError(
            "canonical overlap observations must exactly cover incident road sets"
        )
    assessments: List[Dict[str, Any]] = []
    for road_set in incident_road_sets(catalog, region_id):
        road_set_id = road_set["road_set_id"]
        observation = observations.get(road_set_id)
        if observation is None:
            raise ValueError(
                "canonical overlap observation is missing for {}".format(
                    road_set_id
                )
            )
        if "raw_fragment_base64" in observation:
            if not isinstance(rule_config, Mapping):
                raise ValueError(
                    "canonical raw overlap observation requires rule config"
                )
            observation = recompute_canonical_overlap_observation_from_raw(
                observation,
                rule_config,
                expected_model_artifact_sha256,
            )
        else:
            observation = validate_canonical_overlap_observation(
                observation,
                require_raw=False,
                expected_model_artifact_sha256=expected_model_artifact_sha256,
            )
        if observation.get("primary_sensor_ownership_changed") is not False:
            raise ValueError(
                "overlap subscription must not change primary sensor ownership"
            )
        if list(observation.get("global_node_ids", [])) != list(
            road_set["member_nodes"]
        ):
            raise ValueError(
                "canonical overlap nodes do not match {}".format(road_set_id)
            )
        required_regions = list(road_set["required_regions"])
        if sorted(observation.get("required_regions", [])) != required_regions:
            raise ValueError(
                "canonical overlap owners do not match {}".format(road_set_id)
            )
        missing = [
            field for field in _OVERLAP_PARTIAL_FIELDS if field not in observation
        ]
        if missing:
            raise ValueError(
                "canonical overlap observation {} is missing {}".format(
                    road_set_id, ", ".join(missing)
                )
            )
        output = observation.get("output", {})
        output = dict(output) if isinstance(output, Mapping) else {}
        action = observation.get("action", {})
        action = dict(action) if isinstance(action, Mapping) else {}
        assessment: Dict[str, Any] = {
            "schema_version": 2,
            "road_set_id": road_set_id,
            "resource_id": road_set["resource_id"],
            "kind": "canonical_shared_overlap_assessment",
            "partial": True,
            "member_region": region_id,
            "aggregation_member": aggregation_member,
            "required_regions": required_regions,
            "required_members": [],
            "member_nodes": list(road_set["member_nodes"]),
            "primary_owned_boundary_nodes": list(
                road_set["member_nodes_by_region"][region_id]
            ),
            "subscription_role": "read_only_boundary_shared_replica",
            "primary_sensor_ownership_changed": False,
            "assessment_source": "canonical_overlap_observation",
            "raw_fragment_digest_recomputed": bool(
                observation.get("raw_fragment_digest_recomputed", False)
            ),
            "deterministic_policy_recomputed": bool(
                observation.get("deterministic_policy_recomputed", False)
                or observation.get("cloud_feature_recomputation_verified", False)
            ),
            "recomputation_source": recomputation_source,
            "canonical_feature_observation": {
                key: value
                for key, value in observation.items()
                if key != "raw_fragment_base64"
            },
            "risk_level": _text(output.get("risk_level", "low")),
            "risk_score": round(_safe_float(output.get("risk_score")), 6),
            "confidence": round(_safe_float(output.get("confidence")), 6),
            "action": action,
            "action_views": [action]
            if _text(action.get("action_type")) not in {"", "no_action"}
            else [],
            "road_set_conflict_suspected": False,
            "adjacent_road_set_ids": list(
                road_set.get("adjacent_road_set_ids", [])
            ),
        }
        assessment.update(
            {field: observation[field] for field in _OVERLAP_PARTIAL_FIELDS}
        )
        assessments.append(assessment)
    return assessments


def _action_conflict(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> Tuple[bool, str, str]:
    action_type = _text(left.get("action_type"))
    if not action_type or action_type != _text(right.get("action_type")):
        return False, "", ""
    left_parameters = left.get("parameters", {})
    right_parameters = right.get("parameters", {})
    if not isinstance(left_parameters, Mapping) or not isinstance(
        right_parameters, Mapping
    ):
        return False, "", ""
    if action_type == "variable_speed_limit":
        delta = abs(
            _safe_float(left_parameters.get("target_speed_mph"))
            - _safe_float(right_parameters.get("target_speed_mph"))
        )
        if delta > 10.0:
            return (
                True,
                "boundary_vsl_discontinuity",
                "same road set has a speed-limit difference above 10 mph",
            )
    elif action_type == "ramp_metering":
        delta = abs(
            _safe_float(left_parameters.get("metering_rate_veh_per_hour"))
            - _safe_float(right_parameters.get("metering_rate_veh_per_hour"))
        )
        if delta > 180.0:
            return (
                True,
                "boundary_ramp_rate_discontinuity",
                "same road set has a ramp-rate difference above 180 veh/h",
            )
    elif action_type == "reroute":
        total = _safe_float(
            left_parameters.get("diversion_ratio")
        ) + _safe_float(right_parameters.get("diversion_ratio"))
        if total > 0.5:
            return (
                True,
                "alternate_corridor_overload",
                "same road set has a combined diversion ratio above 0.5",
            )
    return False, "", ""


def _assessment_conflict(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> Tuple[bool, str, str]:
    if any(
        _text(value.get("assessment_source"))
        != "canonical_overlap_observation"
        for value in (left, right)
    ):
        # Region summaries/top-k nodes describe different primary ownership
        # domains and are never evidence of a shared road-set decision clash.
        return False, "", ""
    if left.get("compact_self_validation_errors") or right.get(
        "compact_self_validation_errors"
    ):
        return (
            True,
            "overlap_compact_declaration_mismatch",
            "a compact road-set declaration disagrees with its verifiable feature policy output",
        )
    comparisons = (
        (
            ("window_id", "source_dataset_sha256", "raw_fragment_sha256", "source_content_sha256"),
            "overlap_content_digest_mismatch",
            "two subscribers do not reference the same canonical overlap bytes",
        ),
        (
            ("feature_content_sha256",),
            "overlap_feature_digest_mismatch",
            "two subscribers do not reference the same canonical overlap features",
        ),
        (
            ("preprocess_version",),
            "overlap_preprocess_version_mismatch",
            "two subscribers use different overlap preprocessing versions",
        ),
        (
            ("model_id", "model_version", "model_artifact_sha256"),
            "overlap_model_version_mismatch",
            "two subscribers use different deterministic road-set models",
        ),
        (
            ("policy_id", "policy_version", "policy_artifact_sha256"),
            "overlap_policy_version_mismatch",
            "two subscribers use different road-set decision policies",
        ),
        (
            ("output_digest",),
            "overlap_output_digest_mismatch",
            "same overlap input/version produced different output digests",
        ),
        (
            ("action_digest",),
            "overlap_action_digest_mismatch",
            "same overlap input/version produced different action digests",
        ),
    )
    for fields, kind, reason in comparisons:
        if any(left.get(field) != right.get(field) for field in fields):
            return True, kind, reason
    if left.get("output") != right.get("output"):
        return (
            True,
            "overlap_output_digest_mismatch",
            "same declared output digest carries different output content",
        )
    if left.get("action") != right.get("action"):
        return (
            True,
            "overlap_action_digest_mismatch",
            "same declared action digest carries different action parameters",
        )
    return False, "", ""


def fuse_road_set_contexts(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Fuse two owner-local partials and emit selective evidence-pull signals.

    ``required_members`` deliberately remains empty for normal or incomplete
    road sets.  A two-member pull request is emitted only for an observed
    disagreement/action conflict, never merely because congestion is severe.
    """
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        default_member = _text(record.get("aggregation_member"))
        trusted_region = _text(record.get("region_id"))
        assessments = record.get("road_set_assessments", [])
        if not isinstance(assessments, list):
            continue
        for raw in assessments:
            if not isinstance(raw, Mapping):
                continue
            assessment = dict(raw)
            road_set_id = _text(assessment.get("road_set_id"))
            region = _text(assessment.get("member_region"))
            member = _text(
                assessment.get("aggregation_member", default_member)
            )
            if trusted_region and region != trusted_region:
                raise ValueError(
                    "road-set assessment member_region is not bound to event scope"
                )
            if default_member and member != default_member:
                raise ValueError(
                    "road-set assessment aggregation_member is not trusted"
                )
            if not road_set_id or not region or not member:
                continue
            assessment["aggregation_member"] = member
            grouped.setdefault(road_set_id, {})[region] = assessment

    contexts: List[Dict[str, Any]] = []
    for road_set_id, by_region in sorted(grouped.items()):
        partials = [by_region[region] for region in sorted(by_region)]
        required_regions = sorted(
            {
                _text(region)
                for partial in partials
                for region in partial.get("required_regions", [])
                if _text(region)
            }
        )
        complete = bool(required_regions) and all(
            region in by_region for region in required_regions
        )
        conflict = False
        conflict_kind = ""
        conflict_reason = ""
        if complete and len(required_regions) == 2:
            left = by_region[required_regions[0]]
            right = by_region[required_regions[1]]
            conflict, conflict_kind, conflict_reason = _assessment_conflict(
                left, right
            )
        required_members = []
        if conflict:
            required_members = [
                _text(by_region[region].get("aggregation_member"))
                for region in required_regions
            ]
        context: Dict[str, Any] = {
            "schema_version": ROAD_SET_SCHEMA_VERSION,
            "road_set_id": road_set_id,
            "resource_id": road_set_resource_id(road_set_id),
            "required_regions": required_regions,
            "required_members": required_members,
            "observed_regions": sorted(by_region),
            "observed_members": sorted(
                {
                    _text(partial.get("aggregation_member"))
                    for partial in partials
                    if _text(partial.get("aggregation_member"))
                }
            ),
            "complete": complete,
            "partial_assessments": partials,
            "road_set_conflict_suspected": conflict,
            "conflict_kind": conflict_kind,
            "conflict_reason": conflict_reason,
        }
        if conflict:
            # Only raw canonical overlap bytes can independently rebuild the
            # source-content digest and deterministic decision after a clash.
            context["required_evidence_level"] = "raw"
        contexts.append(context)
    return contexts


def contexts_for_region(
    contexts: Iterable[Mapping[str, Any]], region_id: str
) -> List[Dict[str, Any]]:
    region_id = _text(region_id)
    return [
        dict(context)
        for context in contexts
        if isinstance(context, Mapping)
        and region_id in context.get("required_regions", [])
    ]
