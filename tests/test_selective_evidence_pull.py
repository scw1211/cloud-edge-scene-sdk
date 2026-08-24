"""Selective edge evidence callback invariants."""

import copy
from dataclasses import replace
import hashlib
import hmac
import json
from types import SimpleNamespace
import time
import unittest

from cloud_edge_framework.cloud_service import CloudApiService
from cloud_edge_framework.contracts import Action, SemanticEvent, build_decision
from cloud_edge_framework.metrics import FrameworkMetrics
from cloud_edge_framework.selective_evidence_pull import (
    BoundedEvidenceCache,
    HttpEvidencePullClient,
    SelectiveEvidencePullPlanner,
)


EXPECTED_MEMBERS = ["region_0", "region_1", "region_2", "region_3"]
GROUP_KEY = "PEMS08:test:7"


def _canonical_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _recompute_bundle_sha256(response):
    value = dict(response)
    value.pop("bundle_sha256", None)
    response["bundle_sha256"] = _canonical_sha256(value)


def _event(
    member,
    road_sets=None,
    include_raw=False,
    event_number=0,
):
    evidence = [
        {
            "evidence_id": "{}_summary".format(member),
            "level": "summary",
            "modality": "traffic_summary",
            "encoding": "json",
            "inline": {"risk": "medium"},
            "content_type": "application/json",
        },
        {
            "evidence_id": "{}_feature".format(member),
            "level": "feature",
            "modality": "traffic_timeseries",
            "encoding": "json",
            "inline": {"vector": [event_number, event_number + 1]},
            "content_type": "application/json",
        },
    ]
    if include_raw:
        evidence.append(
            {
                "evidence_id": "{}_raw".format(member),
                "level": "raw",
                "modality": "traffic_timeseries",
                "encoding": "json",
                "inline": {"window": [[event_number]]},
                "content_type": "application/json",
            }
        )
    return SemanticEvent.from_dict(
        {
            "schema_version": "1.0",
            "event_id": "event_{}_{}".format(member, event_number),
            "scene": "freeway_traffic",
            "task": "traffic_risk_assessment",
            "edge_id": member,
            "occurred_at_ms": 1000,
            "scope": {
                "entity_id": member,
                "subsystem": "freeway_corridor",
                "state_variable": "congestion_state",
                "region_id": member,
                "window_start_ms": 1000,
                "window_end_ms": 2000,
            },
            "prediction": {
                "label": "medium",
                "confidence": 0.8,
                "probabilities": {"medium": 0.8},
            },
            "risk": {"level": "medium", "score": 0.5},
            "uncertainty": {
                "confidence": 0.8,
                "prediction_set": ["medium"],
                "method": "fixture",
            },
            "timing": {"deadline_ms": 200.0},
            "evidence": evidence,
            "candidate_actions": [],
            "scene_payload": {
                "road_set_contexts": list(road_sets or []),
            },
            "metadata": {
                "aggregation": {
                    "key": GROUP_KEY,
                    "member": member,
                    "expected_members": EXPECTED_MEMBERS,
                    "minimum_members": 2,
                    "timeout_ms": 200,
                },
                "road_set_contexts": list(road_sets or []),
            },
        }
    )


def _with_locator(event, cache, retain_levels=("summary",)):
    locator = cache.store(event)
    assert locator is not None
    return replace(
        event,
        evidence=[
            item for item in event.evidence if item.level in set(retain_levels)
        ],
        metadata={**event.metadata, "evidence_pull_locator": locator},
    )


def _decision(event, source="summary"):
    action = Action(
        action_type="variable_speed_limit",
        target_ids=[event.scope.region_id],
        resource_ids=["traffic_road_set:rs_01"],
        parameters={
            "target_speed_mph": 45,
            "requires_cloud_confirmation": True,
        },
    )
    return build_decision(
        event,
        "control",
        [action],
        confidence=0.8,
        reason="fixture",
        source=source,
        policy_version="1",
    ).to_dict()


def _coordination(events, source="summary"):
    return {
        "decisions": [_decision(event, source) for event in events],
        "initial_conflicts": [],
        "residual_conflicts": [],
        "initial_conflict_count": 0,
        "residual_conflict_count": 0,
        "globally_consistent": True,
        "resolution_success_rate": 1.0,
        "event_count": len(events),
    }


class _Runtime:
    def __init__(self):
        self.calls = []

    def coordinate_groups(self, groups):
        materialized = [list(group) for group in groups]
        self.calls.append(materialized)
        source = "pulled" if len(self.calls) > 1 else "summary"
        return [_coordination(events, source) for events in materialized]


def _service(client):
    service = CloudApiService.__new__(CloudApiService)
    service.evidence_pull_client = client
    service.evidence_pull_planner = SelectiveEvidencePullPlanner()
    service.metrics = FrameworkMetrics("cloud-test")
    return service


class BoundedEvidenceCacheTest(unittest.TestCase):
    def test_capability_is_group_bound_and_content_verified(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"x" * 32
        )
        event = _event("region_0")
        locator = cache.store(event)
        self.assertIsNotNone(locator)

        client = HttpEvidencePullClient(
            ["http://edge.test:18101"],
            opener=lambda _url, payload, _timeout, _limit: cache.fetch(payload),
        )
        pulled = client.fetch(locator, "feature")
        self.assertEqual(pulled.member, "region_0")
        self.assertEqual([item.level for item in pulled.evidence], ["feature"])

        tampered = dict(locator)
        tampered["group_key"] = "another-group"
        with self.assertRaises(ValueError):
            client.fetch(tampered, "feature")

    def test_response_hmac_rejects_evidence_tamper_after_bundle_rehash(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"h" * 32
        )
        locator = cache.store(_event("region_0"))
        self.assertIsNotNone(locator)
        self.assertIn("feature", locator["level_content_sha256"])
        self.assertIn("feature", locator["response_auth_tags"])

        def opener(_url, payload, _timeout, _limit):
            response = copy.deepcopy(cache.fetch(payload))
            response["evidence"][0]["inline"]["vector"][0] = 999999
            response["evidence_content_sha256"] = _canonical_sha256(
                response["evidence"]
            )
            # A callback-path attacker can recompute both ordinary digests.
            # It cannot change the HMAC-bound locator commitment.
            _recompute_bundle_sha256(response)
            return response

        client = HttpEvidencePullClient(
            ["http://edge.test:18101"], opener=opener
        )
        with self.assertRaisesRegex(ValueError, "HMAC"):
            client.fetch(locator, "feature")

        tampered_locator = copy.deepcopy(locator)
        tampered_locator["level_content_sha256"]["feature"] = "0" * 64
        with self.assertRaises(ValueError):
            HttpEvidencePullClient(
                ["http://edge.test:18101"],
                opener=lambda _url, payload, _timeout, _limit: cache.fetch(payload),
            ).fetch(tampered_locator, "feature")

    def test_response_hmac_binds_owner_group_level_and_expiry(self):
        signing_key = b"i" * 32
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=signing_key
        )
        locator = cache.store(_event("region_0", include_raw=True))
        self.assertIsNotNone(locator)
        self.assertNotEqual(
            locator["response_auth_tags"]["feature"],
            locator["response_auth_tags"]["raw"],
        )
        feature_auth_material = {
            "schema_version": 1,
            "cache_id": locator["cache_id"],
            "scene": locator["scene"],
            "group_key": locator["group_key"],
            "owner_member": locator["member"],
            "event_id": locator["event_id"],
            "requested_level": "feature",
            "record_content_sha256": locator["content_sha256"],
            "expires_at_ms": locator["expires_at_ms"],
            "evidence_content_sha256": locator["level_content_sha256"][
                "feature"
            ],
        }
        self.assertEqual(
            locator["response_auth_tags"]["feature"],
            hmac.new(
                signing_key,
                _canonical_bytes(feature_auth_material),
                hashlib.sha256,
            ).hexdigest(),
        )

        mutations = {
            "member": "region_1",
            "group_key": "PEMS08:test:other",
            "requested_level": "raw",
            "expires_at_ms": int(locator["expires_at_ms"]) + 1,
        }
        for field, replacement in mutations.items():
            with self.subTest(field=field):
                def opener(_url, payload, _timeout, _limit, *, _field=field):
                    response = copy.deepcopy(cache.fetch(payload))
                    response[_field] = mutations[_field]
                    _recompute_bundle_sha256(response)
                    return response

                client = HttpEvidencePullClient(
                    ["http://edge.test:18101"], opener=opener
                )
                with self.assertRaises(ValueError):
                    client.fetch(locator, "feature")

    def test_ttl_capacity_and_real_raw_unavailability_fail_closed(self):
        now = [int(time.time() * 1000)]
        cache = BoundedEvidenceCache(
            "http://edge.test:18101",
            ttl_seconds=0.01,
            max_entries=1,
            signing_key=b"y" * 32,
            clock_ms=lambda: now[0],
        )
        first = cache.store(_event("region_0", event_number=0))
        second = cache.store(_event("region_1", event_number=1))
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        request = HttpEvidencePullClient._request_payload(first, "feature")
        with self.assertRaises(KeyError):
            cache.fetch(request)

        raw_request = dict(second)
        with self.assertRaises(KeyError):
            HttpEvidencePullClient._request_payload(raw_request, "raw")

        now[0] += 20
        request = HttpEvidencePullClient._request_payload(second, "feature")
        with self.assertRaises(KeyError):
            cache.fetch(request)
        snapshot = cache.snapshot()
        self.assertEqual(snapshot["entry_count"], 0)
        self.assertEqual(snapshot["counters"]["capacity_evictions_total"], 1)
        self.assertEqual(snapshot["counters"]["expired_total"], 1)


class SelectivePullClosedLoopTest(unittest.TestCase):
    def _pair(self, cache, triggered=False, required_level=""):
        context = {
            "road_set_id": "rs_01",
            "required_members": ["region_0", "region_1"] if triggered else [],
            "required_regions": ["region_0", "region_1"] if triggered else [],
            "road_set_conflict_suspected": bool(triggered),
        }
        if required_level:
            context["required_evidence_level"] = required_level
        return [
            _with_locator(_event(member, [context], event_number=index), cache)
            for index, member in enumerate(["region_0", "region_1"])
        ]

    def test_normal_summary_path_makes_zero_fetch_requests(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"n" * 32
        )
        events = self._pair(cache, triggered=False)
        requests = []

        def opener(_url, payload, _timeout, _limit):
            requests.append(payload)
            return cache.fetch(payload)

        runtime = _Runtime()
        results, _ = _service(
            HttpEvidencePullClient(["http://edge.test:18101"], opener=opener)
        )._coordinate_with_selective_evidence_pull(runtime, [events])
        self.assertEqual(requests, [])
        self.assertEqual(len(runtime.calls), 1)
        self.assertFalse(results[0]["evidence_pull"]["triggered"])
        self.assertEqual(results[0]["evidence_pull"]["attempted"], 0)

    def test_level_hint_without_conflict_never_triggers_a_callback(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"h" * 32
        )
        context = {
            "road_set_id": "rs_01",
            "required_members": ["region_0", "region_1"],
            "road_set_conflict_suspected": False,
            "required_evidence_level": "feature",
        }
        events = [
            _with_locator(
                _event(member, [context], event_number=index),
                cache,
            )
            for index, member in enumerate(["region_0", "region_1"])
        ]
        requests = []

        def opener(_url, payload, _timeout, _limit):
            requests.append(payload)
            return cache.fetch(payload)

        runtime = _Runtime()
        results, _ = _service(
            HttpEvidencePullClient(
                ["http://edge.test:18101"], opener=opener
            )
        )._coordinate_with_selective_evidence_pull(runtime, [events])
        diagnostics = results[0]["evidence_pull"]
        self.assertEqual(requests, [])
        self.assertEqual(len(runtime.calls), 1)
        self.assertFalse(diagnostics["triggered"])
        self.assertEqual(diagnostics["attempted"], 0)
        self.assertEqual(
            diagnostics["no_pull_reason"],
            "no_road_set_conflict_trigger",
        )

    def test_trigger_fetches_only_pair_then_reruns(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"p" * 32
        )
        events = self._pair(cache, triggered=True, required_level="feature")
        requested_members = []

        def opener(_url, payload, _timeout, _limit):
            requested_members.append(payload["member"])
            return cache.fetch(payload)

        runtime = _Runtime()
        results, enriched = _service(
            HttpEvidencePullClient(["http://edge.test:18101"], opener=opener)
        )._coordinate_with_selective_evidence_pull(runtime, [events])
        diagnostics = results[0]["evidence_pull"]
        self.assertEqual(sorted(requested_members), ["region_0", "region_1"])
        self.assertEqual(diagnostics["attempted"], 2)
        self.assertEqual(diagnostics["succeeded"], 2)
        self.assertTrue(diagnostics["rerun_applied"])
        self.assertEqual(len(runtime.calls), 2)
        self.assertTrue(
            all(
                any(item.level == "feature" for item in event.evidence)
                for event in enriched[0]
            )
        )

    def test_existing_feature_member_is_not_fetched_again(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"e" * 32
        )
        context = {
            "road_set_id": "rs_01",
            "required_members": ["region_0", "region_1"],
            "road_set_conflict_suspected": True,
            "required_evidence_level": "feature",
        }
        events = [
            _with_locator(
                _event("region_0", [context], event_number=0),
                cache,
                retain_levels=("summary", "feature"),
            ),
            _with_locator(
                _event("region_1", [context], event_number=1), cache
            ),
        ]
        requested_members = []

        def opener(_url, payload, _timeout, _limit):
            requested_members.append(payload["member"])
            return cache.fetch(payload)

        runtime = _Runtime()
        results, _ = _service(
            HttpEvidencePullClient(["http://edge.test:18101"], opener=opener)
        )._coordinate_with_selective_evidence_pull(runtime, [events])
        diagnostics = results[0]["evidence_pull"]
        self.assertEqual(diagnostics["requested_members"], ["region_0", "region_1"])
        self.assertEqual(diagnostics["fetch_target_members"], ["region_1"])
        self.assertEqual(requested_members, ["region_1"])
        self.assertEqual(diagnostics["attempted"], 1)
        self.assertTrue(diagnostics["rerun_applied"])

    def test_both_features_already_present_need_no_fetch_or_rerun(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"a" * 32
        )
        context = {
            "road_set_id": "rs_01",
            "required_members": ["region_0", "region_1"],
            "road_set_conflict_suspected": True,
            "required_evidence_level": "feature",
        }
        events = [
            _with_locator(
                _event(member, [context], event_number=index),
                cache,
                retain_levels=("summary", "feature"),
            )
            for index, member in enumerate(["region_0", "region_1"])
        ]
        requests = []

        def opener(_url, payload, _timeout, _limit):
            requests.append(payload)
            return cache.fetch(payload)

        runtime = _Runtime()
        results, _ = _service(
            HttpEvidencePullClient(["http://edge.test:18101"], opener=opener)
        )._coordinate_with_selective_evidence_pull(runtime, [events])
        diagnostics = results[0]["evidence_pull"]
        self.assertEqual(requests, [])
        self.assertEqual(diagnostics["attempted"], 0)
        self.assertEqual(diagnostics["fetch_target_members"], [])
        self.assertTrue(diagnostics["initial_evidence_sufficient"])
        self.assertFalse(diagnostics["rerun_applied"])
        self.assertIsNone(diagnostics["fallback"])
        self.assertEqual(len(runtime.calls), 1)

    def test_multi_road_set_targets_are_paired_and_member_deduplicated(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"m" * 32
        )
        contexts = {
            "region_0": [
                {
                    "road_set_id": "rs_01",
                    "required_members": ["region_0", "region_1"],
                    "road_set_conflict_suspected": True,
                    "required_evidence_level": "feature",
                }
            ],
            "region_1": [
                {
                    "road_set_id": "rs_01",
                    "required_members": ["region_0", "region_1"],
                    "road_set_conflict_suspected": True,
                    "required_evidence_level": "feature",
                },
                {
                    "road_set_id": "rs_12",
                    "required_members": ["region_1", "region_2"],
                    "road_set_conflict_suspected": True,
                    "required_evidence_level": "feature",
                },
            ],
            "region_2": [
                {
                    "road_set_id": "rs_12",
                    "required_members": ["region_1", "region_2"],
                    "road_set_conflict_suspected": True,
                    "required_evidence_level": "feature",
                }
            ],
        }
        events = [
            _with_locator(_event(member, contexts[member], event_number=index), cache)
            for index, member in enumerate(["region_0", "region_1", "region_2"])
        ]
        plan = SelectiveEvidencePullPlanner().plan(
            events, _coordination(events)
        )
        self.assertEqual(plan.road_set_members["rs_01"], ["region_0", "region_1"])
        self.assertEqual(plan.road_set_members["rs_12"], ["region_1", "region_2"])
        self.assertEqual(
            [target.member for target in plan.targets],
            ["region_0", "region_1", "region_2"],
        )
        self.assertEqual(
            next(target for target in plan.targets if target.member == "region_1").road_set_ids,
            ["rs_01", "rs_12"],
        )

    def test_prefixed_action_conflict_merges_with_canonical_road_set(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"c" * 32
        )
        events = self._pair(cache, triggered=False)
        initial = _coordination(events)
        initial["initial_conflicts"] = [
            {
                "left_event_id": events[0].event_id,
                "right_event_id": events[1].event_id,
                "shared_resources": [
                    "traffic_road_set:rs_01",
                    "traffic_node:99",
                ],
            }
        ]
        plan = SelectiveEvidencePullPlanner().plan(events, initial)
        self.assertEqual(plan.road_set_ids, ["rs_01"])
        self.assertEqual(plan.requested_members, ["region_0", "region_1"])
        self.assertEqual(len(plan.targets), 2)
        self.assertIn("cloud_action_conflict", plan.trigger_reasons)

    def test_canonical_decision_metadata_requests_raw_over_feature(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"w" * 32
        )
        events = [
            _with_locator(
                _event(member, include_raw=True, event_number=index), cache
            )
            for index, member in enumerate(["region_0", "region_1"])
        ]
        initial = _coordination(events)
        for decision, level in zip(
            initial["decisions"], ["feature", "raw"]
        ):
            decision["metadata"]["road_set_decisions"] = [
                {
                    "road_set_id": "rs_01",
                    "required_members": ["region_0", "region_1"],
                    "road_set_conflict_suspected": True,
                    "required_evidence_level": level,
                }
            ]

        plan = SelectiveEvidencePullPlanner().plan(events, initial)

        self.assertTrue(plan.triggered)
        self.assertEqual(plan.road_set_levels, {"rs_01": "raw"})
        self.assertEqual(plan.requested_level, "raw")
        self.assertEqual(
            [target.requested_level for target in plan.targets],
            ["raw", "raw"],
        )

    def test_canonical_decision_metadata_rejects_invalid_level(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"v" * 32
        )
        events = self._pair(cache, triggered=False)
        initial = _coordination(events)
        for decision in initial["decisions"]:
            decision["metadata"]["road_set_decisions"] = [
                {
                    "road_set_id": "rs_01",
                    "required_members": ["region_0", "region_1"],
                    "road_set_conflict_suspected": True,
                    "required_evidence_level": "summary",
                }
            ]

        plan = SelectiveEvidencePullPlanner().plan(events, initial)

        self.assertTrue(plan.triggered)
        self.assertEqual(
            plan.no_pull_reason, "invalid_required_evidence_level"
        )
        self.assertEqual(plan.targets, [])

    def test_one_timeout_keeps_initial_result_and_revokes_global_confirmation(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"t" * 32
        )
        events = self._pair(cache, triggered=True, required_level="feature")

        def opener(_url, payload, _timeout, _limit):
            if payload["member"] == "region_1":
                raise TimeoutError("fixture timeout")
            return cache.fetch(payload)

        runtime = _Runtime()
        results, effective = _service(
            HttpEvidencePullClient(["http://edge.test:18101"], opener=opener)
        )._coordinate_with_selective_evidence_pull(runtime, [events])
        result = results[0]
        diagnostics = result["evidence_pull"]
        self.assertEqual(diagnostics["attempted"], 2)
        self.assertEqual(diagnostics["succeeded"], 1)
        self.assertEqual(diagnostics["failed"], 1)
        self.assertFalse(diagnostics["rerun_applied"])
        self.assertEqual(diagnostics["fallback"], "initial_summary_coordination")
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(effective[0], events)
        self.assertEqual(
            [decision["metadata"]["source"] for decision in result["decisions"]],
            ["summary", "summary"],
        )

        lease = SimpleNamespace(
            group_id="group-1",
            group_key=GROUP_KEY,
            completion_reason="all_expected_members",
            expected_members=["region_0", "region_1"],
            received_members=["region_0", "region_1"],
            missing_members=[],
            result_revision=1,
        )
        marked = CloudApiService._mark_aggregation_finality(result, lease)
        self.assertFalse(marked["global_confirmation"])
        self.assertFalse(marked["globally_consistent"])
        for decision in marked["decisions"]:
            authorization = decision["metadata"]["action_authorization"]
            self.assertFalse(authorization["all_actions_authorized"])
            self.assertEqual(
                authorization["deferred_action_types"],
                ["variable_speed_limit"],
            )

    def test_explicit_raw_without_real_raw_never_starts_a_fetch(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"r" * 32
        )
        events = self._pair(cache, triggered=True, required_level="raw")
        plan = SelectiveEvidencePullPlanner().plan(events, _coordination(events))
        self.assertTrue(plan.triggered)
        self.assertEqual(plan.targets, [])
        self.assertEqual(plan.no_pull_reason, "requested_level_unavailable")


if __name__ == "__main__":
    unittest.main()
