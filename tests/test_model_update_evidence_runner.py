"""模型更新闭环取证 runner 的逻辑测试。

本文件只使用测试替身验证状态机；它不会、也不能生成正式通过证据。
"""

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
for value in (REPOSITORY_ROOT, TESTS_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from edge_llm_factory.contracts import ManifestError  # noqa: E402
from edge_llm_factory.update_loop_evidence import (  # noqa: E402
    ControlledCandidateGateFailure,
    EvidenceReleaseLlamaServer,
    _preflight,
    _run_real,
    _write_formal_evidence,
    run,
)
from test_edge_llm_release_lifecycle import _make_release  # noqa: E402


class _FakeProcess:
    next_pid = 42000

    def __init__(self) -> None:
        type(self).next_pid += 1
        self.pid = type(self).next_pid
        self.alive = True

    def poll(self):
        return None if self.alive else 0

    def terminate(self) -> None:
        self.alive = False

    def kill(self) -> None:
        self.alive = False

    def wait(self, timeout=None):
        self.alive = False
        return 0


class _FakeEvidenceSupervisor(EvidenceReleaseLlamaServer):
    def _is_process_healthy(self):
        return self.process is not None and self.process.poll() is None

    def _start_record(self, release_id, revision, record):
        self.stop_process()
        self.process = _FakeProcess()
        self.active_release_id = str(release_id)
        self.applied_revision = int(revision)
        self.active_record = dict(record)
        artifact_path = str(Path(record["deployment_artifact"]["path"]).resolve())
        runtime_publication = self._publish_runtime_configuration(
            str(release_id),
            int(revision),
            record,
            Path(artifact_path),
        )
        transition = {
            "status": "active",
            "release_id": str(release_id),
            "revision": int(revision),
            "artifact": artifact_path,
            "endpoint": self.endpoint,
            "pid": self.process.pid,
            "process_executable": str(self.binary),
            "process_command": [str(self.binary), "--model", artifact_path],
            "artifact_sha256": record["deployment_artifact"]["sha256"],
            "health_verified": True,
            "runtime_publication": runtime_publication,
            "inference_probe": {
                "http_status": 200,
                "prompt_tokens": 1,
                "output_tokens": 1,
            },
        }
        self.transitions.append(transition)
        if self.fault_release_id == release_id:
            self.fault_release_id = None
            self.failed_candidate_process = self.process
            raise ControlledCandidateGateFailure(
                "controlled post-health inference gate failure for {}".format(release_id)
            )
        return {
            key: transition[key]
            for key in ("status", "release_id", "revision", "artifact", "endpoint", "pid")
        }


class ModelUpdateEvidenceRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="update-evidence-")
        self.root = Path(self.temporary.name)
        self.old = _make_release(self.root, "1.0.0")
        self.candidate = _make_release(self.root, "2.0.0")
        self.binary = self.root / "llama-server"
        self.binary.write_text("#!/bin/sh\necho 'fake llama version'\n", encoding="utf-8")
        self.binary.chmod(0o700)
        self.output = self.root / "evidence"
        self.config = {
            "repository_root": str(REPOSITORY_ROOT),
            "git_commit": "a" * 40,
            "old_base": str(self.old[0]),
            "old_package": str(self.old[1]),
            "old_artifact": str(self.old[2]),
            "candidate_base": str(self.candidate[0]),
            "candidate_package": str(self.candidate[1]),
            "candidate_artifact": str(self.candidate[2]),
            "old_release_id": "edge-v1",
            "candidate_release_id": "edge-v2",
            "binary": str(self.binary),
            "output_dir": str(self.output),
            "dataset_id": "update-smoke-v1",
            "hardware_id": "test-double",
            "run_id": "unit-test",
            "probe_prompt": "A",
            "fault_mode": "post_health_inference_gate_failure",
            "host": "127.0.0.1",
            "port": 19491,
            "context_tokens": 16,
            "threads": 1,
            "gpu_layers": 0,
            "startup_timeout_seconds": 1.0,
            "min_available_memory_mb": 1.0,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_fake_process_exercises_full_state_machine_but_cannot_write_evidence(self) -> None:
        preflight = _preflight(self.config, check_git=False)
        result = _run_real(
            self.config,
            preflight,
            supervisor_type=_FakeEvidenceSupervisor,
        )

        self.assertTrue(result["rollback_verified"])
        self.assertEqual(
            [row["release_id"] for row in result["transitions"]],
            ["edge-v1", "edge-v2", "edge-v1"],
        )
        self.assertEqual(result["registry"]["active_release_id"], "edge-v1")
        self.assertTrue(result["candidate_process_stopped"])
        self.assertFalse(result["formal_runtime"])
        with self.assertRaisesRegex(ManifestError, "测试替身"):
            _write_formal_evidence(
                self.output,
                result,
                {
                    "git_commit": "a" * 40,
                    "dataset_id": "update-smoke-v1",
                    "model_ids": {"before": "edge-v1", "applied": "edge-v2"},
                    "hardware_id": "test-double",
                    "metric_semantics": "package_release_apply_rollback",
                    "run_id": "unit-test",
                    "generated_at": "2026-08-09T00:00:00Z",
                },
            )
        self.assertFalse((self.output / "model_update_loop.json").exists())

    def test_dry_run_does_not_create_output_or_passed_evidence(self) -> None:
        with mock.patch(
            "edge_llm_factory.update_loop_evidence._git_identity",
            return_value={
                "repository_root": str(REPOSITORY_ROOT),
                "git_commit": "a" * 40,
                "worktree_clean": True,
            },
        ):
            result = run(self.config, dry_run=True)

        self.assertEqual(result["status"], "dry_run_validated_not_executed")
        self.assertFalse(result["formal_evidence_generated"])
        self.assertFalse(self.output.exists())

    def test_preflight_rejects_same_gguf_for_old_and_candidate(self) -> None:
        # Rebuild the candidate package binding against the old artifact would be
        # needed to pass package validation.  Checking after validation is covered
        # by reusing the old package and artifact as one complete input.
        config = dict(self.config)
        config["candidate_package"] = config["old_package"]
        config["candidate_artifact"] = config["old_artifact"]
        with self.assertRaisesRegex(ManifestError, "GGUF SHA-256 必须不同"):
            _preflight(config, check_git=False)


if __name__ == "__main__":
    unittest.main()
