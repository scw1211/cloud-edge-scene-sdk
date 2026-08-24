"""Additional failure-matrix tests for selective traffic evidence pulls."""

import copy
from dataclasses import replace
import hashlib
import json
import time
import unittest

from cloud_edge_framework.selective_evidence_pull import (
    BoundedEvidenceCache,
    HttpEvidencePullClient,
    SelectiveEvidencePullPlanner,
)
from scenes.freeway_traffic.benchmark_real_current_state_e2e import (
    _conflict_metrics,
)
from tests.test_selective_evidence_pull import (
    _Runtime,
    _coordination,
    _event,
    _service,
    _with_locator,
)


def _canonical_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _context(
    road_set_id,
    members,
    *,
    triggered=True,
    required_level="feature",
):
    value = {
        "road_set_id": road_set_id,
        "required_regions": list(members),
        "required_members": list(members) if triggered else [],
        "road_set_conflict_suspected": bool(triggered),
    }
    if required_level:
        value["required_evidence_level"] = required_level
    return value


def _pair(cache, *, required_level="feature"):
    context = _context(
        "rs_01", ["region_0", "region_1"], required_level=required_level
    )
    events = [
        _with_locator(_event(member, [context], event_number=index), cache)
        for index, member in enumerate(["region_0", "region_1"])
    ]
    return [
        replace(
            event,
            evidence=[item for item in event.evidence if item.level == "summary"],
        )
        for event in events
    ]


class SelectivePullFailClosedMatrixTest(unittest.TestCase):
    def assert_initial_core_unchanged(self, initial, result):
        actual = copy.deepcopy(result)
        actual.pop("evidence_pull", None)
        self.assertEqual(_canonical_bytes(initial), _canonical_bytes(actual))

    def _run(self, events, client):
        runtime = _Runtime()
        initial = _coordination(events, "summary")
        results, effective = _service(client)._coordinate_with_selective_evidence_pull(
            runtime, [events]
        )
        return initial, runtime, results[0], effective[0]

    def test_tampered_token_path_does_not_rerun_or_change_initial_core(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"a" * 32
        )
        events = _pair(cache)
        locator = dict(events[0].metadata["evidence_pull_locator"])
        # Identity fields still satisfy the planner; the HMAC-bound content
        # digest is changed so the edge cache must reject the callback.
        locator["content_sha256"] = "0" * 64
        events[0] = replace(
            events[0],
            metadata={**events[0].metadata, "evidence_pull_locator": locator},
        )
        client = HttpEvidencePullClient(
            ["http://edge.test:18101"],
            opener=lambda _url, payload, _timeout, _limit: cache.fetch(payload),
        )

        initial, runtime, result, effective = self._run(events, client)

        diagnostics = result["evidence_pull"]
        self.assertEqual(1, len(runtime.calls))
        self.assertEqual(2, diagnostics["attempted"])
        self.assertEqual(1, diagnostics["succeeded"])
        self.assertEqual(1, diagnostics["failed"])
        self.assertFalse(diagnostics["rerun_applied"])
        self.assertEqual(events, effective)
        self.assert_initial_core_unchanged(initial, result)

    def test_mitm_rehashed_evidence_response_fails_closed_without_rerun(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"m" * 32
        )
        events = _pair(cache)

        def opener(_url, payload, _timeout, _limit):
            response = copy.deepcopy(cache.fetch(payload))
            response["evidence"][0]["inline"]["vector"] = [888, 999]
            response["evidence_content_sha256"] = _canonical_sha256(
                response["evidence"]
            )
            unsigned_bundle = dict(response)
            unsigned_bundle.pop("bundle_sha256", None)
            response["bundle_sha256"] = _canonical_sha256(unsigned_bundle)
            return response

        client = HttpEvidencePullClient(
            ["http://edge.test:18101"], opener=opener
        )
        initial, runtime, result, effective = self._run(events, client)

        diagnostics = result["evidence_pull"]
        self.assertEqual(1, len(runtime.calls))
        self.assertEqual(2, diagnostics["attempted"])
        self.assertEqual(0, diagnostics["succeeded"])
        self.assertEqual(2, diagnostics["failed"])
        self.assertFalse(diagnostics["rerun_applied"])
        self.assertEqual(events, effective)
        self.assert_initial_core_unchanged(initial, result)

    def test_ttl_expiry_does_not_rerun_or_change_initial_core(self):
        now_ms = [int(time.time() * 1000)]
        cache = BoundedEvidenceCache(
            "http://edge.test:18101",
            ttl_seconds=10.0,
            signing_key=b"b" * 32,
            clock_ms=lambda: now_ms[0],
        )
        events = _pair(cache)
        now_ms[0] += 11_000
        client = HttpEvidencePullClient(
            ["http://edge.test:18101"],
            opener=lambda _url, payload, _timeout, _limit: cache.fetch(payload),
        )

        initial, runtime, result, effective = self._run(events, client)

        diagnostics = result["evidence_pull"]
        self.assertEqual(1, len(runtime.calls))
        self.assertEqual(2, diagnostics["attempted"])
        self.assertEqual(0, diagnostics["succeeded"])
        self.assertEqual(2, diagnostics["failed"])
        self.assertFalse(diagnostics["rerun_applied"])
        self.assertEqual(events, effective)
        self.assert_initial_core_unchanged(initial, result)

    def test_required_raw_absence_is_service_level_fail_closed(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"c" * 32
        )
        events = _pair(cache, required_level="raw")
        requests = []

        def opener(_url, payload, _timeout, _limit):
            requests.append(payload)
            return cache.fetch(payload)

        client = HttpEvidencePullClient(
            ["http://edge.test:18101"], opener=opener
        )

        initial, runtime, result, effective = self._run(events, client)

        diagnostics = result["evidence_pull"]
        self.assertEqual([], requests)
        self.assertEqual(1, len(runtime.calls))
        self.assertTrue(diagnostics["triggered"])
        self.assertEqual(0, diagnostics["attempted"])
        self.assertEqual("requested_level_unavailable", diagnostics["no_pull_reason"])
        self.assertFalse(diagnostics["rerun_applied"])
        self.assertEqual(events, effective)
        self.assert_initial_core_unchanged(initial, result)

    def test_missing_member_locator_does_not_fetch_or_rerun(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"f" * 32
        )
        events = _pair(cache)
        metadata = dict(events[1].metadata)
        metadata.pop("evidence_pull_locator")
        events[1] = replace(events[1], metadata=metadata)
        requests = []

        def opener(_url, payload, _timeout, _limit):
            requests.append(payload)
            return cache.fetch(payload)

        client = HttpEvidencePullClient(
            ["http://edge.test:18101"], opener=opener
        )

        initial, runtime, result, effective = self._run(events, client)

        diagnostics = result["evidence_pull"]
        self.assertEqual([], requests)
        self.assertEqual(1, len(runtime.calls))
        self.assertEqual(0, diagnostics["attempted"])
        self.assertEqual("member_locator_unavailable", diagnostics["no_pull_reason"])
        self.assertFalse(diagnostics["rerun_applied"])
        self.assertEqual(events, effective)
        self.assert_initial_core_unchanged(initial, result)

    def test_timeout_with_only_one_success_keeps_initial_core_and_skips_rerun(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"g" * 32
        )
        events = _pair(cache)

        def opener(_url, payload, _timeout, _limit):
            if payload["member"] == "region_1":
                raise TimeoutError("injected member timeout")
            return cache.fetch(payload)

        client = HttpEvidencePullClient(
            ["http://edge.test:18101"], opener=opener
        )

        initial, runtime, result, effective = self._run(events, client)

        diagnostics = result["evidence_pull"]
        self.assertEqual(1, len(runtime.calls))
        self.assertEqual(2, diagnostics["attempted"])
        self.assertEqual(1, diagnostics["succeeded"])
        self.assertEqual(1, diagnostics["failed"])
        self.assertFalse(diagnostics["rerun_applied"])
        self.assertEqual(events, effective)
        self.assert_initial_core_unchanged(initial, result)

    def test_two_road_sets_fetch_three_unique_members_then_rerun_once(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"d" * 32
        )
        first = _context("rs_01", ["region_0", "region_1"])
        second = _context("rs_12", ["region_1", "region_2"])
        contexts = {
            "region_0": [first],
            "region_1": [first, second],
            "region_2": [second],
        }
        events = [
            _with_locator(_event(member, contexts[member], event_number=index), cache)
            for index, member in enumerate(["region_0", "region_1", "region_2"])
        ]
        events = [
            replace(
                event,
                evidence=[
                    item for item in event.evidence if item.level == "summary"
                ],
            )
            for event in events
        ]
        requested = []

        def opener(_url, payload, _timeout, _limit):
            requested.append(payload["member"])
            return cache.fetch(payload)

        runtime = _Runtime()
        results, effective = _service(
            HttpEvidencePullClient(["http://edge.test:18101"], opener=opener)
        )._coordinate_with_selective_evidence_pull(runtime, [events])
        diagnostics = results[0]["evidence_pull"]

        self.assertEqual(2, len(runtime.calls))
        self.assertEqual(
            ["region_0", "region_1", "region_2"], sorted(requested)
        )
        self.assertEqual(3, diagnostics["attempted"])
        self.assertEqual(3, diagnostics["succeeded"])
        self.assertEqual(0, diagnostics["failed"])
        self.assertTrue(diagnostics["rerun_applied"])
        self.assertEqual(
            {
                "rs_01": ["region_0", "region_1"],
                "rs_12": ["region_1", "region_2"],
            },
            diagnostics["road_set_members"],
        )
        enriched_members = [
            event.metadata["selective_evidence_pull"]["member"]
            for event in effective[0]
            if "selective_evidence_pull" in event.metadata
        ]
        self.assertEqual(
            ["region_0", "region_1", "region_2"], enriched_members
        )

    def test_action_conflict_requires_explicit_road_set_resource(self):
        cache = BoundedEvidenceCache(
            "http://edge.test:18101", signing_key=b"e" * 32
        )
        events = [
            _with_locator(_event(member, [], event_number=index), cache)
            for index, member in enumerate(["region_0", "region_1"])
        ]
        events = [
            replace(
                event,
                evidence=[
                    item for item in event.evidence if item.level == "summary"
                ],
            )
            for event in events
        ]
        conflict = {
            "left_event_id": events[0].event_id,
            "right_event_id": events[1].event_id,
            "shared_resources": ["traffic_road_set:rs_01", "traffic_node:5"],
        }
        initial = _coordination(events)
        initial["initial_conflicts"] = [conflict]

        plan = SelectiveEvidencePullPlanner().plan(events, initial)

        self.assertTrue(plan.triggered)
        self.assertEqual(["rs_01"], plan.road_set_ids)
        self.assertEqual(["region_0", "region_1"], plan.requested_members)
        self.assertEqual(2, len(plan.targets))


class ConflictMetricQualificationTest(unittest.TestCase):
    def test_complete_natural_zero_conflicts_marks_resolution_not_evaluated(self):
        metrics = _conflict_metrics(
            initial_conflicts=0,
            residual_conflicts=0,
            coordinated_events=400,
            aggregation_result_count=100,
            expected_aggregation_result_count=100,
            expected_coordinated_event_count=400,
            aggregations_complete=True,
        )

        self.assertTrue(metrics["complete"])
        self.assertEqual(0.0, metrics["conflict_rate"])
        self.assertFalse(metrics["conflict_resolution_evaluated"])
        self.assertIsNone(metrics["conflict_resolution_success_rate"])


if __name__ == "__main__":
    unittest.main()
