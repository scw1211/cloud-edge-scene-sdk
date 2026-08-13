import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import evaluate_competition_targets as evaluator


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return {"path": path.name, "sha256": _sha(path)}


def _write_canonical_json(path, value):
    path.write_text(evaluator._canonical(value) + "\n", encoding="utf-8")
    return {"path": path.name, "sha256": _sha(path)}


def _write_text(path, value):
    path.write_text(value, encoding="utf-8")
    return {"path": path.name, "sha256": _sha(path)}


def _expected(semantics, dataset, models, hardware="test-hardware"):
    return {
        "git_commit": "0123456789abcdef",
        "dataset_id": dataset,
        "model_ids": models,
        "hardware_id": hardware,
        "metric_semantics": semantics,
    }


def _provenance(expected, run_id, *, run_started_at=None):
    value = dict(expected)
    value.update(
        {
            "run_id": run_id,
            "generated_at": "2026-08-08T10:10:00+08:00",
        }
    )
    if run_started_at is not None:
        value["run_started_at"] = run_started_at
    return value


class CompetitionTargetEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self._build_complete_manifest()
        self.manifest_path = self.root / "manifest.json"
        self._save_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def _save_manifest(self):
        self.manifest_path.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load_target_evidence(self, target_name):
        target = self.manifest["targets"][target_name]
        path = self.root / target["evidence"]["path"]
        return path, json.loads(path.read_text(encoding="utf-8"))

    def _store_target_evidence(self, target_name, path, evidence):
        self.manifest["targets"][target_name]["evidence"] = _write_json(
            path, evidence
        )
        self._save_manifest()

    def _evaluate_target(self, target_name):
        report = evaluator.evaluate_manifest(
            self.manifest_path, bootstrap_iterations=300
        )
        return next(
            item for item in report["targets"] if item["target"] == target_name
        )

    def _rewrite_distributed_update(self, evidence):
        target = self.manifest["targets"]["model_update_loop"]
        evidence_path = self.root / target["distributed_evidence"]["path"]
        target["distributed_evidence"] = _write_canonical_json(
            evidence_path, evidence
        )
        completion_path = self.root / target["distributed_completion_marker"]["path"]
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["distributed_evidence_sha256"] = target["distributed_evidence"][
            "sha256"
        ]
        completion["cloud_ack_sha256"] = evaluator._canonical_json_sha256(
            evidence["cloud_ack"]
        )
        target["distributed_completion_marker"] = _write_canonical_json(
            completion_path, completion
        )
        self._save_manifest()

    def _basic_target(self, name, dataset, models, evidence, hardware="test-hardware"):
        semantics = evaluator.EXPECTED_SEMANTICS[name]
        expected = _expected(semantics, dataset, models, hardware=hardware)
        evidence = dict(evidence)
        evidence["provenance"] = _provenance(expected, "run-{}".format(name))
        ref = _write_json(self.root / "{}.json".format(name), evidence)
        return {"evidence": ref, "expected": expected, "criteria": {}}

    def _build_complete_manifest(self):
        targets = {}
        targets["capability_retention"] = self._basic_target(
            "capability_retention",
            "general-v1",
            {"teacher": "teacher-9b", "edge": "edge-0.8b"},
            {
                "teacher_macro_score": 0.80,
                "edge_macro_score": 0.68,
                "category_scores": {
                    "math": {"teacher": 0.80, "edge": 0.68, "sample_count": 1},
                    "code": {"teacher": 0.80, "edge": 0.68, "sample_count": 1},
                    "natural_language_reasoning": {
                        "teacher": 0.80,
                        "edge": 0.68,
                        "sample_count": 1,
                    },
                },
                "category_sample_counts": {
                    "math": 1,
                    "code": 1,
                    "natural_language_reasoning": 1,
                },
                "sample_count": 3,
            },
        )
        targets["ttft_reduction"] = self._basic_target(
            "ttft_reduction",
            "ttft-prompts-v1",
            {"baseline": "teacher-9b", "edge": "edge-0.8b"},
            {
                "baseline_ttft_ms": [400.0, 420.0, 380.0],
                "edge_ttft_ms": [80.0, 85.0, 75.0],
            },
        )
        jetson_hardware = {
            "role": "edge_device",
            "platform": "nvidia_jetson",
            "machine": "aarch64",
            "device_model": "NVIDIA Jetson Orin Nano Developer Kit",
            "hostname": "edge-test",
        }
        targets["single_inference_memory"] = self._basic_target(
            "single_inference_memory",
            "memory-prompts-v1",
            ["edge-0.8b"],
            {
                "peak_memory_mb": 1024.0,
                "memory_semantics": evaluator.EXPECTED_MEMORY_SEMANTICS,
                "measurement_scope": evaluator.EXPECTED_MEMORY_SCOPE,
                "measurement_host": jetson_hardware,
                "measurement_window": {
                    "measured_requests_per_window": 1,
                    "window_count": 3,
                    "sampler_interval_ms": 10.0,
                    "warmup_requests": 2,
                },
                "sample_count": 3,
            },
            hardware=jetson_hardware,
        )
        weak_expected = _expected(
            evaluator.EXPECTED_SEMANTICS["weak_network_retention"],
            "network-matrix-v1",
            {"edge_decision": "edge-0.8b"},
        )
        weak_expected["git_commit"] = "0" * 40
        weak_profiles = {
            "normal": {"delay_ms": 0.0, "jitter_ms": 0.0, "loss_rate": 0.0},
            "mild": {"delay_ms": 40.0, "jitter_ms": 10.0, "loss_rate": 0.01},
            "severe": {"delay_ms": 80.0, "jitter_ms": 30.0, "loss_rate": 0.10},
            "outage": {"delay_ms": 0.0, "jitter_ms": 0.0, "loss_rate": 1.0},
        }
        weak_samples = {
            "normal": [0, 1],
            "mild": [2, 3],
            "severe": [4, 5],
            "outage": [6, 7],
        }
        weak_plan = {
            "schema_version": "1.0",
            "plan_id": "weak-plan-v1",
            "created_at": "2026-08-08T09:00:00+08:00",
            "scene": "freeway_traffic",
            "git_commit": weak_expected["git_commit"],
            "dataset_id": weak_expected["dataset_id"],
            "model_ids": weak_expected["model_ids"],
            "hardware_id": weak_expected["hardware_id"],
            "seed": 20260809,
            "profile_order": ["normal", "mild", "severe", "outage", "recovery"],
            "fault_profiles": weak_profiles,
            "sample_ids_by_profile": weak_samples,
            "business_deadline_ms": 200.0,
            "aggregation_timeout_ms": 150,
            "dispatch_lead_ms": 30.0,
            "split": "test",
            "top_k": 10,
        }
        weak_plan_ref = _write_json(self.root / "weak-plan.json", weak_plan)

        def outbox_snapshot(*, completed=0, active=0):
            return {
                "states": {
                    "pending": active,
                    "inflight": 0,
                    "completed": completed,
                },
                "active": active,
                "reconciliation": {"active": 0},
                "durable_handoff": {
                    "pending": 0,
                    "durable_pending_count": 0,
                },
            }

        def weak_profile(profile_id, successes):
            sample_ids = weak_samples[profile_id]
            events = []
            for sample_id in sample_ids:
                for partition_id in range(4):
                    event_index = len(events)
                    succeeds = event_index < successes
                    outage = profile_id == "outage"
                    response = {
                        "status": "provisional",
                        "route": "local_autonomy" if outage else "cloud_async",
                        "policy_route": (
                            "local_autonomy" if outage else "cloud_async"
                        ),
                        "policy_waits_for_cloud": False,
                        "input_to_response_ms": 80.0 if succeeds else 250.0,
                        "local_actions_authorized": True,
                        "immediate_action_types": ["monitor"],
                        "deferred_action_types": [],
                        "cloud_confirmed": False,
                        "local_autonomy": outage,
                        "summary_delivery_required": True,
                        "summary_persistence_stage": "handoff_durable",
                    }
                    events.append(
                        {
                            "event_id": "{}-{}-{}".format(
                                profile_id, sample_id, partition_id
                            ),
                            "sample_id": sample_id,
                            "profile": profile_id,
                            "success": succeeds,
                            "business_observation": {
                                "response": response,
                                "review": (
                                    {
                                        "state": "queued",
                                        "requested_route": "local_autonomy",
                                    }
                                    if outage
                                    else {}
                                ),
                                "authoritative_observed_ms": None,
                                "deadline_ms": 200.0,
                            },
                        }
                    )
            result = {
                "profile_id": profile_id,
                "fault_parameters": weak_profiles[profile_id],
                "business_attempts": len(events),
                "business_successes": successes,
                "samples": [
                    {
                        "sample_id": sample_id,
                        "business_attempts": 4,
                        "business_successes": sum(
                            row["success"]
                            for row in events
                            if row["sample_id"] == sample_id
                        ),
                    }
                    for sample_id in sample_ids
                ],
                "events": events,
                "outbox_before_profile": [outbox_snapshot()] * 4,
            }
            if not outage:
                result["outbox_after_profile"] = [
                    outbox_snapshot(completed=2)
                ] * 4
                result["authoritative_reviews_after_profile"] = len(events)
            return result

        normal_profile = weak_profile("normal", 8)
        weak_profile_rows = [
            weak_profile("mild", 8),
            weak_profile("severe", 7),
            weak_profile("outage", 7),
        ]
        weak_evidence = {
            "evidence_schema_version": "2.0",
            "provenance": _provenance(
                weak_expected,
                "run-weak_network_retention",
                run_started_at="2026-08-08T10:00:00+08:00",
            ),
            "experiment_plan": {
                "plan_id": weak_plan["plan_id"],
                "path": str(self.root / "weak-plan.json"),
                "sha256": weak_plan_ref["sha256"],
                "created_at": weak_plan["created_at"],
                "loaded_at": "2026-08-08T09:59:00+08:00",
            },
            "formal_eligible": True,
            "configuration": {
                "profile_order": weak_plan["profile_order"],
                "fault_profiles": weak_profiles,
                "sample_ids_by_profile": weak_samples,
                "business_deadline_ms": 200.0,
                "aggregation_timeout_ms": 150,
                "dispatch_lead_ms": 30.0,
                "split": "test",
                "top_k": 10,
                "seed": 20260809,
            },
            "measurement_status": "measured",
            "status": "passed",
            "passed": True,
            "normal_baseline": normal_profile,
            "profiles": weak_profile_rows,
            "aggregate": {
                "business_attempts": 24,
                "business_successes": 22,
                "retention_rate": 22.0 / 24.0,
            },
            "recovery": {
                "passed": True,
                "outbox_before_recovery": [outbox_snapshot(active=2)] * 4,
                "outbox_snapshots": [outbox_snapshot(completed=2)] * 4,
                "review_records": {
                    event["event_id"]: {
                        "state": "completed",
                        "completion_stage": "lightweight_final",
                        "requested_route": "local_autonomy",
                        "final_decision": {
                            "status": "final",
                            "metadata": {
                                "action_authorization": {
                                    "cloud_confirmed": True
                                },
                                "aggregation": {
                                    "group_id": "outage-group-{}".format(
                                        event["sample_id"]
                                    )
                                },
                            },
                        },
                    }
                    for event in weak_profile_rows[-1]["events"]
                },
                "aggregation_records": [
                    {
                        "group_id": "outage-group-{}".format(sample_id),
                        "state": "completed",
                        "completion_reason": "all_expected_members",
                        "evidence_complete": True,
                        "finality": "final",
                        "global_confirmation": True,
                        "expected_members": [
                            "edge_node_{}".format(value) for value in range(4)
                        ],
                        "received_members": [
                            "edge_node_{}".format(value) for value in range(4)
                        ],
                        "missing_members": [],
                    }
                    for sample_id in weak_samples["outage"]
                ],
                "expected_event_ids": [
                    event["event_id"] for event in weak_profile_rows[-1]["events"]
                ],
                "expected_sample_by_event": {
                    event["event_id"]: event["sample_id"]
                    for event in weak_profile_rows[-1]["events"]
                },
                "manual_flush_used": False,
            },
            "attempted_real_event_count": 32,
        }
        targets["weak_network_retention"] = {
            "evidence": _write_json(self.root / "weak-network.json", weak_evidence),
            "experiment_plan": weak_plan_ref,
            "expected": weak_expected,
            "criteria": {},
        }

        latency_expected = _expected(
            evaluator.EXPECTED_SEMANTICS["weighted_business_e2e"],
            "route-test-v1",
            ["edge-0.8b", "cloud-coordinator-v1"],
        )
        routes = {}
        latency_sources = []
        for route_index, route in enumerate(evaluator.REQUIRED_ROUTES):
            run_id = "{}-run".format(route)
            samples = [
                {
                    "group_id": "{}:{}".format(run_id, index),
                    "latency_ms": 90.0 + route_index * 5 + index,
                    "run_id": run_id,
                    "sample_id": index,
                    "partition_id": 0,
                    "policy_route": route,
                    "business_endpoint": (
                        "authoritative_final"
                        if route == "cloud_sync"
                        else "provisional_response_and_durable_summary"
                    ),
                }
                for index in range(5)
            ]
            routes[route] = {
                "attempt_count": len(samples),
                "success_count": len(samples),
                "failure_count": 0,
                "samples": samples,
                "failures": [],
            }
            latency_sources.append(
                {
                    "run_id": run_id,
                    "policy_route": route,
                    "integrity_valid": True,
                    "errors": [],
                    "planned_sample_ids": list(range(5)),
                    "planned_partition_ids": [0],
                }
            )
        latency_evidence = {
            "provenance": _provenance(
                latency_expected,
                "run-latency",
                run_started_at="2026-08-08T10:00:00+08:00",
            ),
            "weight_plan_id": "route-mix-v1",
            "plan_sha256": "c" * 64,
            "integrity_valid": True,
            "integrity_errors": [],
            "errors": [],
            "source_runs": latency_sources,
            "routes": routes,
        }
        weight_plan = {
            "schema_version": 1,
            "weight_plan_id": "route-mix-v1",
            "plan_sha256": "c" * 64,
            "locked_at": "2026-08-08T09:00:00+08:00",
            "routes": {route: 0.25 for route in evaluator.REQUIRED_ROUTES},
        }
        targets["weighted_business_e2e"] = {
            "evidence": _write_json(self.root / "latency.json", latency_evidence),
            "weight_plan": _write_json(self.root / "weights.json", weight_plan),
            "expected": latency_expected,
            "criteria": {},
        }

        objective_expected = _expected(
            evaluator.EXPECTED_SEMANTICS["traffic_global_objective"],
            "objective-test-v1",
            ["cloud-coordinator-v1"],
        )
        definition_ref = _write_json(
            self.root / "objective-definition.json",
            {
                "objective_id": "traffic-joint-utility-v1",
                "expected_members": [
                    "edge_node_0",
                    "edge_node_1",
                    "edge_node_2",
                    "edge_node_3",
                ],
                "terms": ["benefit", "cost"],
            },
        )
        objective_evidence = {
            "provenance": _provenance(
                objective_expected,
                "run-objective",
                run_started_at="2026-08-08T10:00:00+08:00",
            ),
            "objective_id": "traffic-joint-utility-v1",
            "objective_definition_sha256": definition_ref["sha256"],
            "sample_plan": {
                "plan_id": "objective-samples-v1",
                "sha256": "placeholder",
                "locked_at": "2026-08-08T09:00:00+08:00",
                "expected_sample_count": 2,
                "sample_ids": [1, 2],
                "slow_threshold_ms": 200.0,
            },
            "attempts": [
                {
                    "sample_id": 1,
                    "source_experiment_id": "objective-source-run",
                    "outcome": "authoritative_optimized",
                    "aggregation_group_ids": ["g-1"],
                    "global_authoritative_final_ms": 150.0,
                    "slow": False,
                    "errors": [],
                },
                {
                    "sample_id": 2,
                    "source_experiment_id": "objective-source-run",
                    "outcome": "authoritative_optimized",
                    "aggregation_group_ids": ["g-2"],
                    "global_authoritative_final_ms": 250.0,
                    "slow": True,
                    "errors": [],
                },
            ],
            "outcome_counts": {
                "authoritative_optimized": 2,
                "partial": 0,
                "failed": 0,
            },
            "slow_sample_count": 1,
            "records": [
                {
                    "group_id": "g-1",
                    "sample_id": 1,
                    "source_experiment_id": "objective-source-run",
                    "expected_members": [
                        "edge_node_0",
                        "edge_node_1",
                        "edge_node_2",
                        "edge_node_3",
                    ],
                    "observed_members": [
                        "edge_node_0",
                        "edge_node_1",
                        "edge_node_2",
                        "edge_node_3",
                    ],
                    "expected_member_count": 4,
                    "observed_member_count": 4,
                    "objective_definition_sha256": definition_ref["sha256"],
                    "candidate_space_size": 16,
                    "evaluated_candidate_count": 16,
                    "selected_plan_sha256": "a" * 64,
                    "exact_search": True,
                    "candidate_set_complete": True,
                    "constraints_satisfied": True,
                    "mode": "active",
                    "applied": True,
                    "authoritative_input": True,
                    "baseline_utility": 1.0,
                    "selected_utility": 1.2,
                },
                {
                    "group_id": "g-2",
                    "sample_id": 2,
                    "source_experiment_id": "objective-source-run",
                    "expected_members": [
                        "edge_node_0",
                        "edge_node_1",
                        "edge_node_2",
                        "edge_node_3",
                    ],
                    "observed_members": [
                        "edge_node_0",
                        "edge_node_1",
                        "edge_node_2",
                        "edge_node_3",
                    ],
                    "expected_member_count": 4,
                    "observed_member_count": 4,
                    "objective_definition_sha256": definition_ref["sha256"],
                    "candidate_space_size": 16,
                    "evaluated_candidate_count": 16,
                    "selected_plan_sha256": "b" * 64,
                    "exact_search": True,
                    "candidate_set_complete": True,
                    "constraints_satisfied": True,
                    "mode": "active",
                    "applied": True,
                    "authoritative_input": True,
                    "baseline_utility": 2.0,
                    "selected_utility": 2.0,
                },
            ],
        }
        objective_plan = {
            "schema_version": "1.0",
            "plan_id": "objective-samples-v1",
            "locked_at": "2026-08-08T09:00:00+08:00",
            "scene": "freeway_traffic",
            "expected": {
                "git_commit": objective_expected["git_commit"],
                "dataset_id": objective_expected["dataset_id"],
                "model_ids": objective_expected["model_ids"],
                "hardware_id": objective_expected["hardware_id"],
                "objective_id": "traffic-joint-utility-v1",
                "objective_definition_sha256": definition_ref["sha256"],
            },
            "expected_sample_count": 2,
            "sample_ids": [1, 2],
            "slow_threshold_ms": 200.0,
        }
        objective_plan_ref = _write_json(
            self.root / "objective-sample-plan.json", objective_plan
        )
        objective_evidence["sample_plan"]["sha256"] = objective_plan_ref["sha256"]
        targets["traffic_global_objective"] = {
            "evidence": _write_json(self.root / "objective.json", objective_evidence),
            "objective_definition": definition_ref,
            "sample_plan": objective_plan_ref,
            "expected": objective_expected,
            "criteria": {},
        }

        update_expected = _expected(
            evaluator.EXPECTED_SEMANTICS["model_update_loop"],
            "update-smoke-v1",
            {"before": "edge-v1", "applied": "edge-v2"},
        )
        update_expected["llama_server_sha256"] = "f" * 64
        update_provenance = _provenance(update_expected, "run-update")
        stage_provenance = {
            key: update_provenance[key]
            for key in ("git_commit", "run_id", "dataset_id", "hardware_id")
        }
        contract = {
            "scene": "freeway_traffic",
            "base_id": "qwen-base",
            "base_fingerprint": "4" * 64,
            "protocol": "single_token_action/v1",
            "max_input_tokens": 16,
            "max_output_tokens": 1,
            "action_mapping_sha256": "5" * 64,
            "accepted_tokens": ["A", "B"],
        }

        def identity(label, artifact_sha, package_sha, adapter_id, version):
            return {
                "base": {"path": "/assets/base.json", "sha256": "6" * 64, "bytes": 10},
                "package": {
                    "path": "/assets/{}-package".format(label),
                    "sha256": package_sha,
                    "file_count": 4,
                    "bytes": 100,
                },
                "artifact": {
                    "path": "/assets/{}.gguf".format(label),
                    "sha256": artifact_sha,
                    "bytes": 1000,
                },
                "adapter_id": adapter_id,
                "adapter_version": version,
                "scene": "freeway_traffic",
                "base_fingerprint": "4" * 64,
                "gate_results": [{"metric": "accuracy", "passed": True}],
                "decision_contract": dict(contract),
            }

        old_identity = identity("old", "a" * 64, "b" * 64, "traffic-v1", "1.0")
        candidate_identity = identity(
            "candidate", "c" * 64, "d" * 64, "traffic-v2", "2.0"
        )
        history = [
            {
                "sequence": 1,
                "action": "promote",
                "from_release_id": None,
                "to_release_id": "edge-v1",
            },
            {
                "sequence": 2,
                "action": "promote",
                "from_release_id": "edge-v1",
                "to_release_id": "edge-v2",
            },
            {
                "sequence": 3,
                "action": "rollback",
                "from_release_id": "edge-v2",
                "to_release_id": "edge-v1",
                "audit": {
                    "trigger": "candidate_apply_failure",
                    "failed_release_id": "edge-v2",
                    "failed_revision": 2,
                },
            },
        ]
        old_promotion = {
            "status": "promoted",
            "active_release_id": "edge-v1",
            "revision": 1,
            "binding_fingerprint": "1" * 64,
            "artifact_sha256": "a" * 64,
            "package_sha256": "b" * 64,
        }
        candidate_promotion = {
            "status": "promoted",
            "active_release_id": "edge-v2",
            "revision": 2,
            "binding_fingerprint": "2" * 64,
            "artifact_sha256": "c" * 64,
            "package_sha256": "d" * 64,
        }

        def transition(model_id, revision, pid, label, binding):
            selected = old_identity if label == "old" else candidate_identity
            return {
                "status": "active",
                "release_id": model_id,
                "revision": revision,
                "pid": pid,
                "endpoint": "http://127.0.0.1:19290",
                "process_executable": "/runtime/llama-server",
                "artifact_sha256": selected["artifact"]["sha256"],
                "package_sha256": selected["package"]["sha256"],
                "binding_fingerprint": binding,
                "decision_contract": dict(contract),
                "health_verified": True,
                "inference_probe": {
                    "http_status": 200,
                    "prompt_sha256": "9" * 64,
                    "prompt_tokens": 13,
                    "output_tokens": 1,
                    "action_token": "A",
                    "action_mapping_accepted": True,
                },
            }

        transitions = [
            transition("edge-v1", 1, 101, "old", "1" * 64),
            transition("edge-v2", 2, 102, "candidate", "2" * 64),
        ]
        rollback_transition = transition("edge-v1", 3, 103, "old", "1" * 64)
        stage_documents = {
            "package": {
                "schema_version": "edge-llm-update-loop-evidence/v1",
                "stage": "package",
                "status": "completed",
                "execution_mode": "real_llama_server",
                "provenance": stage_provenance,
                "old_release_id": "edge-v1",
                "candidate_release_id": "edge-v2",
                "old": old_identity,
                "candidate": candidate_identity,
            },
            "release": {
                "schema_version": "edge-llm-update-loop-evidence/v1",
                "stage": "release",
                "status": "completed",
                "provenance": stage_provenance,
                "old_release_id": "edge-v1",
                "candidate_release_id": "edge-v2",
                "old_promotion": old_promotion,
                "candidate_promotion": candidate_promotion,
                "release_history": history,
            },
            "apply": {
                "schema_version": "edge-llm-update-loop-evidence/v1",
                "stage": "apply",
                "status": "completed",
                "execution_mode": "real_llama_server",
                "provenance": stage_provenance,
                "old_release_id": "edge-v1",
                "candidate_release_id": "edge-v2",
                "llama_server": {
                    "path": "/runtime/llama-server",
                    "sha256": "f" * 64,
                    "bytes": 100,
                    "version_output": "llama.cpp test",
                },
                "transitions": transitions,
                "candidate_health_and_inference_verified": True,
                "fault_mode": "post_health_inference_gate_failure",
            },
            "rollback": {
                "schema_version": "edge-llm-update-loop-evidence/v1",
                "stage": "rollback",
                "status": "completed",
                "execution_mode": "real_llama_server",
                "provenance": stage_provenance,
                "old_release_id": "edge-v1",
                "candidate_release_id": "edge-v2",
                "fault_mode": "post_health_inference_gate_failure",
                "expected_failure": "controlled failure",
                "rollback_transition": rollback_transition,
                "candidate_process_stopped": True,
                "registry": {
                    "active_release_id": "edge-v1",
                    "revision": 3,
                    "history": history,
                },
                "runtime_config_after_rollback": {"model": "/assets/old.gguf"},
                "supervisor_after_rollback": {
                    "status": "recovered",
                    "process_running": True,
                    "process_healthy": True,
                    "registry_active_release_id": "edge-v1",
                    "applied_release_id": "edge-v1",
                    "registry_revision": 3,
                    "applied_revision": 3,
                    "last_failure": {"rollback": {"status": "rolled_back"}},
                },
                "rollback_verified": True,
                "isolated_runtime_stopped_after_evidence": True,
                "runtime_cleanup": {
                    "tracked_pids": [101, 102, 103],
                    "all_tracked_pids_exited": True,
                    "port_released": True,
                    "passed": True,
                },
            },
        }
        stage_refs = {
            stage: _write_json(self.root / "{}.json".format(stage), document)
            for stage, document in stage_documents.items()
        }
        update_evidence = {
            "schema_version": "edge-llm-update-loop-evidence/v1",
            "provenance": update_provenance,
            "stages": {
                stage: {
                    "status": "completed",
                    "evidence_sha256": stage_refs[stage]["sha256"],
                }
                for stage in stage_refs
            },
            "rollback_verified": True,
            "model_id_before": "edge-v1",
            "model_id_applied": "edge-v2",
            "model_id_after_rollback": "edge-v1",
            "fault_mode": "post_health_inference_gate_failure",
            "execution_mode": "real_llama_server",
        }
        update_ref = _write_json(self.root / "update.json", update_evidence)
        completion = {
            "schema_version": "edge-llm-update-loop-evidence/v1",
            "status": "completed",
            "execution_mode": "real_llama_server",
            "git_commit": update_provenance["git_commit"],
            "run_id": update_provenance["run_id"],
            "main_evidence_sha256": update_ref["sha256"],
            "stage_sha256": {
                stage: ref["sha256"] for stage, ref in stage_refs.items()
            },
            "completed_at_utc": "2026-08-08T10:11:00+08:00",
        }
        completion_ref = _write_json(
            self.root / "FORMAL_EVIDENCE_COMPLETE.json", completion
        )

        def published_identity(identity_value, release_id):
            return {
                "release_id": release_id,
                "adapter_id": identity_value["adapter_id"],
                "adapter_version": identity_value["adapter_version"],
                "scene": identity_value["scene"],
                "base_fingerprint": identity_value["base_fingerprint"],
                "base_sha256": identity_value["base"]["sha256"],
                "package_sha256": identity_value["package"]["sha256"],
                "artifact_sha256": identity_value["artifact"]["sha256"],
                "artifact_bytes": identity_value["artifact"]["bytes"],
                "decision_contract": identity_value["decision_contract"],
                "gate_results": identity_value["gate_results"],
            }

        publication_files = []
        file_specs = (
            ("old", "base_manifest", "old/base/base.json", "6" * 64, 10),
            ("old", "package_file", "old/package/manifest.json", "7" * 64, 100),
            ("old", "gguf", "old/artifact/old.gguf", "a" * 64, 1000),
            ("candidate", "base_manifest", "candidate/base/base.json", "6" * 64, 10),
            (
                "candidate",
                "package_file",
                "candidate/package/manifest.json",
                "8" * 64,
                100,
            ),
            (
                "candidate",
                "gguf",
                "candidate/artifact/candidate.gguf",
                "c" * 64,
                1000,
            ),
        )
        for index, (role, kind, target_path, digest, size) in enumerate(file_specs):
            file_id = "{}-{}-{:04d}".format(role, kind.replace("_", "-"), index)
            publication_files.append(
                {
                    "file_id": file_id,
                    "role": role,
                    "kind": kind,
                    "target_path": target_path,
                    "bytes": size,
                    "sha256": digest,
                    "download_path": "/api/v1/model-updates/run-update/files/{}".format(
                        file_id
                    ),
                }
            )
        publication = {
            "schema_version": "edge-llm-cloud-publication/v1",
            "status": "published",
            "execution_mode": "real_http",
            "run_id": "run-update",
            "publication_nonce": "e" * 64,
            "expected_edge_id": "edge-test",
            "expected_hardware_id": update_expected["hardware_id"],
            "expected_dataset_id": update_expected["dataset_id"],
            "published_at_utc": "2026-08-08T10:09:00+08:00",
            "publisher": {
                "git_commit": update_expected["git_commit"],
                "source_sha256": {
                    "edge_llm_factory/distributed_update_evidence.py": "1" * 64,
                    "scripts/measure_distributed_model_update_loop.py": "2" * 64,
                },
            },
            "release_order": ["edge-v1", "edge-v2", "edge-v1"],
            "releases": {
                "old": published_identity(old_identity, "edge-v1"),
                "candidate": published_identity(candidate_identity, "edge-v2"),
            },
            "files": publication_files,
        }
        publication_sha = evaluator._canonical_json_sha256(publication)
        downloaded_files = [
            {
                "file_id": row["file_id"],
                "target_path": row["target_path"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
            for row in publication_files
        ]
        local_summary = {
            "execution_mode": "real_llama_server",
            "rollback_verified": True,
            "provenance": stage_provenance,
            "model_id_before": "edge-v1",
            "model_id_applied": "edge-v2",
            "model_id_after_rollback": "edge-v1",
            "transition_release_ids": ["edge-v1", "edge-v2", "edge-v1"],
            "transition_revisions": [1, 2, 3],
            "release_actions": ["promote", "promote", "rollback"],
            "candidate_process_stopped": True,
            "runtime_cleanup_passed": True,
            "main_evidence_sha256": update_ref["sha256"],
            "stage_sha256": {
                stage: ref["sha256"] for stage, ref in stage_refs.items()
            },
            "completion_marker_sha256": completion_ref["sha256"],
        }
        receipt_core = {
            "schema_version": "edge-llm-edge-receipt/v1",
            "status": "completed",
            "execution_mode": "real_cloud_edge_http",
            "run_id": "run-update",
            "edge_id": "edge-test",
            "hardware_id": update_expected["hardware_id"],
            "dataset_id": update_expected["dataset_id"],
            "git_commit": update_expected["git_commit"],
            "publication_nonce": publication["publication_nonce"],
            "publication_sha256": publication_sha,
            "downloaded_files": downloaded_files,
            "download_atomic_publish": True,
            "local_execution": local_summary,
            "confirmed_at_utc": "2026-08-08T10:11:01+08:00",
        }
        receipt = {
            **receipt_core,
            "receipt_id": evaluator._canonical_json_sha256(receipt_core),
        }
        receipt_sha = evaluator._canonical_json_sha256(receipt)
        ack = {
            "schema_version": "edge-llm-cloud-ack/v1",
            "status": "accepted",
            "execution_mode": "real_http",
            "run_id": "run-update",
            "edge_id": "edge-test",
            "receipt_id": receipt["receipt_id"],
            "receipt_sha256": receipt_sha,
            "publication_sha256": publication_sha,
            "store_revision": 1,
            "received_at_utc": "2026-08-08T10:11:02+08:00",
        }
        cloud_ack = {
            "url": "http://192.0.2.10:19300/api/v1/model-updates/run-update/receipts",
            "http_status": 200,
            "request_sha256": receipt_sha,
            "ack_sha256": evaluator._canonical_json_sha256(ack),
            "ack": ack,
            "hmac_verified": True,
        }
        distributed = {
            "schema_version": "edge-llm-distributed-update-evidence/v1",
            "status": "completed",
            "execution_mode": "real_cloud_edge_http",
            "provenance": {
                "git_commit": update_expected["git_commit"],
                "run_id": "run-update",
                "edge_id": "edge-test",
                "hardware_id": update_expected["hardware_id"],
                "dataset_id": update_expected["dataset_id"],
                "generated_at": "2026-08-08T10:11:03+08:00",
            },
            "cloud_publication": {
                "manifest_url": "http://192.0.2.10:19300/api/v1/model-updates/run-update/manifest",
                "publication_sha256": publication_sha,
                "hmac_key_id": "1" * 16,
                "manifest_hmac_verified": True,
                "publication": publication,
            },
            "transfer": {
                "protocol": "http",
                "real_http": True,
                "atomic_download_publish": True,
                "cache_reused": False,
                "downloaded_files": downloaded_files,
            },
            "local_execution": local_summary,
            "edge_receipt": receipt,
            "cloud_ack": cloud_ack,
            "cloud_received_edge_confirmation": True,
        }
        distributed_ref = _write_canonical_json(
            self.root / "distributed-update.json", distributed
        )
        distributed_completion = {
            "schema_version": "edge-llm-distributed-update-evidence/v1",
            "status": "completed",
            "execution_mode": "real_cloud_edge_http",
            "run_id": "run-update",
            "distributed_evidence_sha256": distributed_ref["sha256"],
            "cloud_ack_sha256": evaluator._canonical_json_sha256(cloud_ack),
            "completed_at_utc": "2026-08-08T10:11:04+08:00",
        }
        targets["model_update_loop"] = {
            "evidence": update_ref,
            "stage_evidence": stage_refs,
            "completion_marker": completion_ref,
            "distributed_evidence": distributed_ref,
            "distributed_completion_marker": _write_canonical_json(
                self.root / "FORMAL_DISTRIBUTED_EVIDENCE_COMPLETE.json",
                distributed_completion,
            ),
            "expected": update_expected,
            "criteria": {
                "real_cloud_edge_http_required": True,
                "cloud_receipt_ack_required": True,
            },
        }
        return {
            "schema_version": "1.0",
            "evaluation_id": "unit-test",
            "scene": "freeway_traffic",
            "industrial_excluded": True,
            "targets": targets,
        }

    def test_complete_compatible_evidence_passes_all_seven_targets(self):
        report = evaluator.evaluate_manifest(
            self.manifest_path, bootstrap_iterations=300, bootstrap_seed=7
        )
        self.assertEqual(report["gate_status"], "passed")
        self.assertEqual(report["status_counts"]["passed"], 7)
        self.assertTrue(report["industrial_excluded"])
        self.assertTrue(report["policy"]["explicit_evidence_only"])
        latency = next(
            item for item in report["targets"] if item["target"] == "weighted_business_e2e"
        )
        self.assertLess(latency["metrics"]["weighted_mean_ms"], 200.0)
        self.assertLess(latency["metrics"]["bootstrap_95_ci_ms"][1], 200.0)

    def test_model_update_requires_distributed_cloud_edge_evidence(self):
        self.manifest["targets"]["model_update_loop"].pop("distributed_evidence")
        self._save_manifest()
        with self.assertRaisesRegex(evaluator.InvalidEvidence, "distributed_evidence"):
            evaluator.evaluate_manifest(self.manifest_path, bootstrap_iterations=300)

    def test_model_update_rejects_unverified_cloud_ack_even_with_fresh_hashes(self):
        target = self.manifest["targets"]["model_update_loop"]
        path = self.root / target["distributed_evidence"]["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["cloud_ack"]["hmac_verified"] = False
        self._rewrite_distributed_update(evidence)

        result = self._evaluate_target("model_update_loop")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("云端持久接收确认", result["reasons"][0])

    def test_model_update_rejects_loopback_cloud_transport(self):
        target = self.manifest["targets"]["model_update_loop"]
        path = self.root / target["distributed_evidence"]["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["cloud_publication"]["manifest_url"] = (
            "http://127.0.0.1:19300/api/v1/model-updates/run-update/manifest"
        )
        self._rewrite_distributed_update(evidence)

        result = self._evaluate_target("model_update_loop")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("非回环", result["reasons"][0])

    def test_missing_explicit_file_is_not_measured_and_no_old_result_is_discovered(self):
        ref = self.manifest["targets"]["capability_retention"]["evidence"]
        (self.root / ref["path"]).unlink()
        report = evaluator.evaluate_manifest(
            self.manifest_path, bootstrap_iterations=300
        )
        result = report["targets"][0]
        self.assertEqual(result["status"], "not_measured")
        self.assertIn("不存在", result["reasons"][0])

    def test_hash_mismatch_is_invalid_evidence(self):
        ref = self.manifest["targets"]["single_inference_memory"]["evidence"]
        (self.root / ref["path"]).write_text("{}", encoding="utf-8")
        report = evaluator.evaluate_manifest(
            self.manifest_path, bootstrap_iterations=300
        )
        result = next(
            item for item in report["targets"] if item["target"] == "single_inference_memory"
        )
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("SHA-256", result["reasons"][0])

    def test_memory_peak_must_be_strictly_positive(self):
        path, evidence = self._load_target_evidence("single_inference_memory")
        evidence["peak_memory_mb"] = 0.0
        self._store_target_evidence("single_inference_memory", path, evidence)
        result = self._evaluate_target("single_inference_memory")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("大于 0", result["reasons"][0])

    def test_memory_semantics_must_match_exactly(self):
        path, evidence = self._load_target_evidence("single_inference_memory")
        evidence["memory_semantics"] = "model file size"
        self._store_target_evidence("single_inference_memory", path, evidence)
        result = self._evaluate_target("single_inference_memory")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("memory_semantics", result["reasons"][0])

    def test_server_memory_is_rejected_even_when_manifest_matches(self):
        path, evidence = self._load_target_evidence("single_inference_memory")
        server_hardware = {
            "role": "server",
            "platform": "linux_workstation",
            "machine": "x86_64",
            "device_model": "GPU server",
            "hostname": "server-test",
        }
        evidence["provenance"]["hardware_id"] = server_hardware
        evidence["measurement_host"] = server_hardware
        self.manifest["targets"]["single_inference_memory"]["expected"][
            "hardware_id"
        ] = server_hardware
        self._store_target_evidence("single_inference_memory", path, evidence)
        result = self._evaluate_target("single_inference_memory")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("边缘设备", result["reasons"][0])

    def test_memory_requires_one_request_per_measurement_window(self):
        path, evidence = self._load_target_evidence("single_inference_memory")
        evidence["measurement_window"]["measured_requests_per_window"] = 2
        self._store_target_evidence("single_inference_memory", path, evidence)
        result = self._evaluate_target("single_inference_memory")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("一次推理请求", result["reasons"][0])

    def test_capability_requires_all_three_categories(self):
        path, evidence = self._load_target_evidence("capability_retention")
        evidence["category_scores"].pop("natural_language_reasoning")
        evidence["category_sample_counts"].pop("natural_language_reasoning")
        evidence["sample_count"] = 2
        self._store_target_evidence("capability_retention", path, evidence)
        result = self._evaluate_target("capability_retention")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("natural_language_reasoning", result["reasons"][0])

    def test_capability_macro_must_equal_three_category_macro(self):
        path, evidence = self._load_target_evidence("capability_retention")
        evidence["edge_macro_score"] = 0.79
        self._store_target_evidence("capability_retention", path, evidence)
        result = self._evaluate_target("capability_retention")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("宏平均", result["reasons"][0])

    def test_provenance_mismatch_is_invalid_evidence(self):
        target = self.manifest["targets"]["weak_network_retention"]
        path = self.root / target["evidence"]["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["provenance"]["dataset_id"] = "another-dataset"
        target["evidence"] = _write_json(path, evidence)
        self._save_manifest()
        report = evaluator.evaluate_manifest(
            self.manifest_path, bootstrap_iterations=300
        )
        result = next(
            item for item in report["targets"] if item["target"] == "weak_network_retention"
        )
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("dataset_id", result["reasons"][0])

    def test_incomplete_weak_network_run_is_not_measured(self):
        target = self.manifest["targets"]["weak_network_retention"]
        path = self.root / target["evidence"]["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["measurement_status"] = "incomplete"
        evidence["status"] = "incomplete"
        target["evidence"] = _write_json(path, evidence)
        self._save_manifest()
        report = evaluator.evaluate_manifest(
            self.manifest_path, bootstrap_iterations=300
        )
        result = next(
            item for item in report["targets"] if item["target"] == "weak_network_retention"
        )
        self.assertEqual(result["status"], "not_measured")

    def test_weak_network_recovery_failure_cannot_pass_on_retention_alone(self):
        target = self.manifest["targets"]["weak_network_retention"]
        path = self.root / target["evidence"]["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["recovery"]["aggregation_records"][0][
            "global_confirmation"
        ] = False
        # A forged producer summary may not override the raw recovery record.
        evidence["recovery"]["passed"] = True
        evidence["status"] = "passed"
        evidence["passed"] = True
        target["evidence"] = _write_json(path, evidence)
        self._save_manifest()
        report = evaluator.evaluate_manifest(
            self.manifest_path, bootstrap_iterations=300
        )
        result = next(
            item for item in report["targets"] if item["target"] == "weak_network_retention"
        )
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("recovery.passed", result["reasons"][0])

    def test_weak_network_forged_event_success_cannot_override_raw_deadline(self):
        path, evidence = self._load_target_evidence("weak_network_retention")
        event = evidence["profiles"][0]["events"][0]
        event["business_observation"]["response"]["input_to_response_ms"] = 500.0
        # Keep every producer-maintained success/count/pass field forged true.
        event["success"] = True
        self._store_target_evidence("weak_network_retention", path, evidence)
        result = self._evaluate_target("weak_network_retention")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("自报 success", result["reasons"][0])

    def test_weak_network_sample_ids_must_match_preregistered_plan(self):
        path, evidence = self._load_target_evidence("weak_network_retention")
        evidence["profiles"][0]["samples"][0]["sample_id"] = 999
        self._store_target_evidence("weak_network_retention", path, evidence)
        result = self._evaluate_target("weak_network_retention")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("sample_id", result["reasons"][0])

    def test_weak_network_plan_hash_binding_cannot_be_rewritten_after_run(self):
        target = self.manifest["targets"]["weak_network_retention"]
        plan_path = self.root / target["experiment_plan"]["path"]
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["seed"] += 1
        target["experiment_plan"] = _write_json(plan_path, plan)
        self._save_manifest()
        result = self._evaluate_target("weak_network_retention")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("计划", result["reasons"][0])

    def test_weak_network_dirty_development_run_is_not_formal_evidence(self):
        path, evidence = self._load_target_evidence("weak_network_retention")
        evidence["formal_eligible"] = False
        self._store_target_evidence("weak_network_retention", path, evidence)
        result = self._evaluate_target("weak_network_retention")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("脏工作区", result["reasons"][0])

    def test_weight_plan_locked_after_run_is_invalid(self):
        target = self.manifest["targets"]["weighted_business_e2e"]
        path = self.root / target["weight_plan"]["path"]
        plan = json.loads(path.read_text(encoding="utf-8"))
        plan["locked_at"] = "2026-08-08T10:05:00+08:00"
        target["weight_plan"] = _write_json(path, plan)
        self._save_manifest()
        report = evaluator.evaluate_manifest(
            self.manifest_path, bootstrap_iterations=300
        )
        result = next(
            item for item in report["targets"] if item["target"] == "weighted_business_e2e"
        )
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("测量开始前", result["reasons"][0])

    def test_weight_plan_id_must_match_evidence_exactly(self):
        path, evidence = self._load_target_evidence("weighted_business_e2e")
        evidence["weight_plan_id"] = "another-plan"
        self._store_target_evidence("weighted_business_e2e", path, evidence)
        result = self._evaluate_target("weighted_business_e2e")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("weight_plan_id", result["reasons"][0])

    def test_four_route_integrity_flag_must_be_true(self):
        path, evidence = self._load_target_evidence("weighted_business_e2e")
        evidence["integrity_valid"] = False
        self._store_target_evidence("weighted_business_e2e", path, evidence)
        result = self._evaluate_target("weighted_business_e2e")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("integrity_valid", result["reasons"][0])

    def test_four_route_integrity_errors_must_be_empty(self):
        path, evidence = self._load_target_evidence("weighted_business_e2e")
        evidence["integrity_errors"] = ["source_run_mismatch"]
        self._store_target_evidence("weighted_business_e2e", path, evidence)
        result = self._evaluate_target("weighted_business_e2e")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("integrity_errors", result["reasons"][0])

    def test_four_route_producer_errors_must_be_empty(self):
        path, evidence = self._load_target_evidence("weighted_business_e2e")
        evidence["errors"] = ["request_failed"]
        self._store_target_evidence("weighted_business_e2e", path, evidence)
        result = self._evaluate_target("weighted_business_e2e")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("errors", result["reasons"][0])

    def test_four_route_plan_sha_must_match_when_declared_by_plan(self):
        target = self.manifest["targets"]["weighted_business_e2e"]
        path = self.root / target["weight_plan"]["path"]
        plan = json.loads(path.read_text(encoding="utf-8"))
        plan["plan_sha256"] = "d" * 64
        target["weight_plan"] = _write_json(path, plan)
        self._save_manifest()
        result = self._evaluate_target("weighted_business_e2e")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("plan_sha256", result["reasons"][0])

    def test_declared_weight_plan_file_sha_must_match_manifest_ref(self):
        path, evidence = self._load_target_evidence("weighted_business_e2e")
        evidence["weight_plan_sha256"] = "e" * 64
        self._store_target_evidence("weighted_business_e2e", path, evidence)
        result = self._evaluate_target("weighted_business_e2e")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("weight_plan_sha256", result["reasons"][0])

    def test_weight_plan_cannot_hide_a_route_with_zero_weight(self):
        target = self.manifest["targets"]["weighted_business_e2e"]
        path = self.root / target["weight_plan"]["path"]
        plan = json.loads(path.read_text(encoding="utf-8"))
        plan["routes"].update(
            {
                "edge_only": 0.0,
                "local_autonomy": 0.25,
                "cloud_async": 0.25,
                "cloud_sync": 0.50,
            }
        )
        target["weight_plan"] = _write_json(path, plan)
        self._save_manifest()
        report = evaluator.evaluate_manifest(
            self.manifest_path, bootstrap_iterations=300
        )
        result = next(
            item for item in report["targets"] if item["target"] == "weighted_business_e2e"
        )
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("都必须大于", result["reasons"][0])

    def test_weighted_latency_fails_when_mean_and_ci_do_not_meet_target(self):
        target = self.manifest["targets"]["weighted_business_e2e"]
        path = self.root / target["evidence"]["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        for sample in evidence["routes"]["cloud_sync"]["samples"]:
            sample["latency_ms"] = 600.0
        target["evidence"] = _write_json(path, evidence)
        self._save_manifest()
        report = evaluator.evaluate_manifest(
            self.manifest_path, bootstrap_iterations=300, bootstrap_seed=9
        )
        result = next(
            item for item in report["targets"] if item["target"] == "weighted_business_e2e"
        )
        self.assertEqual(result["status"], "failed")
        self.assertGreaterEqual(result["metrics"]["weighted_mean_ms"], 200.0)
        self.assertGreaterEqual(result["metrics"]["bootstrap_95_ci_ms"][1], 200.0)

    def test_weighted_latency_keeps_failed_attempts_in_the_gate(self):
        target = self.manifest["targets"]["weighted_business_e2e"]
        path = self.root / target["evidence"]["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        route = evidence["routes"]["cloud_sync"]
        route["attempt_count"] += 1
        route["failure_count"] += 1
        route["failures"].append(
            {
                "run_id": "cloud_sync-run",
                "sample_id": 5,
                "partition_id": 0,
                "reason": "authoritative_final_missing",
            }
        )
        source = next(
            item
            for item in evidence["source_runs"]
            if item["policy_route"] == "cloud_sync"
        )
        source["planned_sample_ids"].append(5)
        target["evidence"] = _write_json(path, evidence)
        self._save_manifest()
        report = evaluator.evaluate_manifest(
            self.manifest_path, bootstrap_iterations=300, bootstrap_seed=11
        )
        result = next(
            item for item in report["targets"] if item["target"] == "weighted_business_e2e"
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("未到达业务终点", result["reasons"][0])

    def test_global_objective_shadow_record_does_not_pass_active_gate(self):
        target = self.manifest["targets"]["traffic_global_objective"]
        path = self.root / target["evidence"]["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["records"][0]["mode"] = "shadow"
        evidence["records"][0]["applied"] = False
        target["evidence"] = _write_json(path, evidence)
        self._save_manifest()
        report = evaluator.evaluate_manifest(
            self.manifest_path, bootstrap_iterations=300
        )
        result = next(
            item
            for item in report["targets"]
            if item["target"] == "traffic_global_objective"
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("active", " ".join(result["reasons"]))

    def test_global_objective_member_lists_are_required(self):
        path, evidence = self._load_target_evidence("traffic_global_objective")
        evidence["records"][0].pop("observed_members")
        self._store_target_evidence("traffic_global_objective", path, evidence)
        result = self._evaluate_target("traffic_global_objective")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("observed_members", result["reasons"][0])

    def test_global_objective_members_must_be_unique_and_exact(self):
        path, evidence = self._load_target_evidence("traffic_global_objective")
        evidence["records"][0]["observed_members"][-1] = "edge_node_2"
        self._store_target_evidence("traffic_global_objective", path, evidence)
        result = self._evaluate_target("traffic_global_objective")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("重复成员", result["reasons"][0])

    def test_global_objective_expected_and_observed_must_match_exactly(self):
        path, evidence = self._load_target_evidence("traffic_global_objective")
        evidence["records"][0]["observed_members"] = [
            "edge_node_0",
            "edge_node_1",
            "edge_node_2",
        ]
        evidence["records"][0]["observed_member_count"] = 3
        self._store_target_evidence("traffic_global_objective", path, evidence)
        result = self._evaluate_target("traffic_global_objective")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("精确相等", result["reasons"][0])

    def test_global_objective_member_counts_must_match_declarations(self):
        path, evidence = self._load_target_evidence("traffic_global_objective")
        evidence["records"][0]["observed_member_count"] = 3
        self._store_target_evidence("traffic_global_objective", path, evidence)
        result = self._evaluate_target("traffic_global_objective")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("observed_member_count", result["reasons"][0])

    def test_global_objective_member_contract_is_consistent_across_records(self):
        path, evidence = self._load_target_evidence("traffic_global_objective")
        reordered = [
            "edge_node_1",
            "edge_node_0",
            "edge_node_2",
            "edge_node_3",
        ]
        evidence["records"][1]["expected_members"] = list(reordered)
        evidence["records"][1]["observed_members"] = list(reordered)
        self._store_target_evidence("traffic_global_objective", path, evidence)
        result = self._evaluate_target("traffic_global_objective")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("声明不一致", result["reasons"][0])

    def test_global_objective_group_ids_must_be_unique(self):
        path, evidence = self._load_target_evidence("traffic_global_objective")
        evidence["records"][1]["group_id"] = evidence["records"][0]["group_id"]
        self._store_target_evidence("traffic_global_objective", path, evidence)
        result = self._evaluate_target("traffic_global_objective")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("group_id", result["reasons"][0])

    def test_global_objective_missing_planned_sample_is_rejected(self):
        path, evidence = self._load_target_evidence("traffic_global_objective")
        evidence["attempts"].pop()
        self._store_target_evidence("traffic_global_objective", path, evidence)
        result = self._evaluate_target("traffic_global_objective")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("预注册样本总数", result["reasons"][0])

    def test_global_objective_deleted_failed_sample_is_rejected(self):
        path, evidence = self._load_target_evidence("traffic_global_objective")
        evidence["records"].pop()
        evidence["attempts"][1].update(
            {
                "outcome": "failed",
                "aggregation_group_ids": [],
                "global_authoritative_final_ms": None,
                "slow": False,
                "errors": ["timeout"],
            }
        )
        evidence["outcome_counts"] = {
            "authoritative_optimized": 1,
            "partial": 0,
            "failed": 1,
        }
        evidence["slow_sample_count"] = 0
        self._store_target_evidence("traffic_global_objective", path, evidence)
        self.assertEqual(
            self._evaluate_target("traffic_global_objective")["status"], "failed"
        )

        path, evidence = self._load_target_evidence("traffic_global_objective")
        evidence["attempts"].pop()
        evidence["outcome_counts"] = {
            "authoritative_optimized": 1,
            "partial": 0,
            "failed": 0,
        }
        self._store_target_evidence("traffic_global_objective", path, evidence)
        result = self._evaluate_target("traffic_global_objective")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("预注册样本总数", result["reasons"][0])

    def test_global_objective_duplicate_sample_is_rejected(self):
        path, evidence = self._load_target_evidence("traffic_global_objective")
        evidence["attempts"][1]["sample_id"] = 1
        self._store_target_evidence("traffic_global_objective", path, evidence)
        result = self._evaluate_target("traffic_global_objective")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("重复 sample_id", result["reasons"][0])

    def test_global_objective_plan_mismatch_is_rejected(self):
        target = self.manifest["targets"]["traffic_global_objective"]
        plan_path = self.root / target["sample_plan"]["path"]
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["sample_ids"] = [1, 3]
        target["sample_plan"] = _write_json(plan_path, plan)
        self._save_manifest()
        result = self._evaluate_target("traffic_global_objective")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("绑定预注册样本计划", result["reasons"][0])

    def test_markdown_is_chinese_and_keeps_not_measured_visible(self):
        ref = self.manifest["targets"]["ttft_reduction"]["evidence"]
        (self.root / ref["path"]).unlink()
        report = evaluator.evaluate_manifest(
            self.manifest_path, bootstrap_iterations=300
        )
        markdown = evaluator.render_markdown(report)
        self.assertIn("七项统一证据门禁", markdown)
        self.assertIn("未测量", markdown)
        self.assertIn("工业明确排除", markdown)


if __name__ == "__main__":
    unittest.main()
