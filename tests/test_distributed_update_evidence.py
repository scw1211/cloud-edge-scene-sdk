"""云端发布、边缘下载和回执闭环的真实 HTTP 逻辑测试。

测试使用小型临时资产和真实 loopback HTTP，不启动 llama-server。正式入口会额外
拒绝 loopback、脏 Git、测试云端和测试执行器，因此这些测试不能生成正式证据。
"""

from pathlib import Path
import hashlib
import hmac
import json
import sys
import tempfile
import threading
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
for value in (REPOSITORY_ROOT, TESTS_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from edge_llm_factory.contracts import ManifestError, read_json_object  # noqa: E402
from edge_llm_factory.distributed_update_evidence import (  # noqa: E402
    CloudUpdateService,
    _build_receipt,
    _canonical_bytes,
    _local_execution_summary,
    _sha256_bytes,
    create_cloud_http_server,
    download_publication,
    post_edge_receipt,
)
from test_edge_llm_release_lifecycle import _make_release  # noqa: E402


class DistributedUpdateEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="distributed-update-")
        self.root = Path(self.temporary.name)
        self.old = _make_release(self.root, "1.0.0")
        self.candidate = _make_release(self.root, "2.0.0")
        self.secret = b"unit-test-shared-secret-32-bytes-minimum"
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
            "run_id": "distributed-unit-test",
            "expected_edge_id": "edge-01",
            "expected_hardware_id": "jetson-test",
            "expected_dataset_id": "dataset-test",
            "storage_dir": str(self.root / "cloud-store"),
        }
        self.service = CloudUpdateService(self.config, self.secret, check_git=False)
        self.server = create_cloud_http_server(self.service, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.cloud_url = "http://127.0.0.1:{}".format(self.server.server_address[1])

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temporary.cleanup()

    def _download(self, name: str = "edge-cache") -> dict:
        return download_publication(
            self.cloud_url,
            self.config["run_id"],
            self.secret,
            self.root / name,
            timeout_seconds=3.0,
        )

    def _local_execution(self) -> dict:
        return {
            "execution_mode": "real_llama_server",
            "rollback_verified": True,
            "provenance": {
                "git_commit": self.config["git_commit"],
                "dataset_id": self.config["expected_dataset_id"],
                "hardware_id": self.config["expected_hardware_id"],
                "run_id": self.config["run_id"],
            },
            "model_id_before": self.config["old_release_id"],
            "model_id_applied": self.config["candidate_release_id"],
            "model_id_after_rollback": self.config["old_release_id"],
            "transition_release_ids": ["edge-v1", "edge-v2", "edge-v1"],
            "transition_revisions": [1, 2, 3],
            "release_actions": ["promote", "promote", "rollback"],
            "candidate_process_stopped": True,
            "runtime_cleanup_passed": True,
            "main_evidence_sha256": "1" * 64,
            "stage_sha256": {
                "package": "2" * 64,
                "release": "3" * 64,
                "apply": "4" * 64,
                "rollback": "5" * 64,
            },
            "completion_marker_sha256": "6" * 64,
        }

    def _receipt(self, transfer: dict) -> dict:
        return _build_receipt(
            transfer["publication"],
            transfer["publication_sha256"],
            self.config["expected_edge_id"],
            self.config["expected_hardware_id"],
            self.config["expected_dataset_id"],
            self.config["git_commit"],
            transfer["downloaded_files"],
            self._local_execution(),
        )

    def test_real_http_download_is_atomic_verified_and_reusable(self) -> None:
        first = self._download()

        self.assertFalse(first["cache_reused"])
        self.assertEqual(len(first["downloaded_files"]), len(self.service.publication["files"]))
        self.assertTrue((Path(first["cache_dir"]) / "DOWNLOAD_COMPLETE.json").is_file())
        first_download_count = self.service.file_download_count
        self.assertGreater(first_download_count, 0)

        second = self._download()
        self.assertTrue(second["cache_reused"])
        self.assertEqual(self.service.file_download_count, first_download_count)
        self.assertEqual(first["publication_sha256"], second["publication_sha256"])
        self.assertEqual(first["downloaded_files"], second["downloaded_files"])

    def test_changed_cloud_source_fails_closed_without_visible_cache(self) -> None:
        file_id = next(
            row["file_id"]
            for row in self.service.publication["files"]
            if row["role"] == "candidate" and row["kind"] == "gguf"
        )
        source = self.service.sources[file_id]
        source.write_bytes(source.read_bytes() + b"changed-after-publication")
        cache = self.root / "corrupt-cache"

        with self.assertRaises((ManifestError, OSError)):
            download_publication(
                self.cloud_url,
                self.config["run_id"],
                self.secret,
                cache,
                timeout_seconds=3.0,
            )

        self.assertTrue(cache.is_dir())
        self.assertEqual(list(cache.iterdir()), [])

    def test_receipt_is_hmac_protected_durable_and_idempotent(self) -> None:
        transfer = self._download()
        receipt = self._receipt(transfer)

        first = post_edge_receipt(
            self.cloud_url,
            receipt,
            self.secret,
            timeout_seconds=3.0,
            attempts=1,
        )
        second = post_edge_receipt(
            self.cloud_url,
            receipt,
            self.secret,
            timeout_seconds=3.0,
            attempts=1,
        )

        self.assertTrue(first["hmac_verified"])
        self.assertEqual(first["ack"], second["ack"])
        store = read_json_object(self.service.receipt_store_path)
        self.assertEqual(store["revision"], 1)
        self.assertEqual(list(store["receipts"]), [self.config["expected_edge_id"]])

        restarted = CloudUpdateService(self.config, self.secret, check_git=False)
        self.assertEqual(restarted.publication_body, self.service.publication_body)
        body, _headers = restarted.accept_receipt(
            _canonical_bytes(receipt),
            hmac.new(
                self.secret, _canonical_bytes(receipt), hashlib.sha256
            ).hexdigest(),
        )
        self.assertEqual(json.loads(body.decode("utf-8")), first["ack"])

    def test_conflicting_receipt_and_bad_hmac_are_rejected(self) -> None:
        receipt = self._receipt(self._download())
        body = _canonical_bytes(receipt)
        with self.assertRaises(PermissionError):
            self.service.accept_receipt(body, "0" * 64)

        post_edge_receipt(
            self.cloud_url,
            receipt,
            self.secret,
            timeout_seconds=3.0,
            attempts=1,
        )
        conflicting = dict(receipt)
        conflicting["confirmed_at_utc"] = "2099-01-01T00:00:00Z"
        core = dict(conflicting)
        core.pop("receipt_id")
        conflicting["receipt_id"] = _sha256_bytes(_canonical_bytes(core))
        with self.assertRaisesRegex(ManifestError, "回执发送失败"):
            post_edge_receipt(
                self.cloud_url,
                conflicting,
                self.secret,
                timeout_seconds=3.0,
                attempts=1,
            )

    def test_test_cloud_and_test_local_evidence_cannot_be_formal(self) -> None:
        transfer = self._download()
        receipt = self._receipt(transfer)
        with self.assertRaisesRegex(ManifestError, "确认内容"):
            post_edge_receipt(
                self.cloud_url,
                receipt,
                self.secret,
                timeout_seconds=3.0,
                attempts=1,
                require_formal_cloud=True,
            )

        with self.assertRaisesRegex(ManifestError, "测试替身"):
            _build_receipt(
                transfer["publication"],
                transfer["publication_sha256"],
                self.config["expected_edge_id"],
                self.config["expected_hardware_id"],
                self.config["expected_dataset_id"],
                self.config["git_commit"],
                transfer["downloaded_files"],
                {"execution_mode": "test_double"},
            )

        main = self.root / "fake-main.json"
        completion = self.root / "fake-completion.json"
        main.write_text(
            json.dumps(
                {
                    "schema_version": "edge-llm-update-evidence/v1",
                    "execution_mode": "test_double",
                    "rollback_verified": True,
                }
            ),
            encoding="utf-8",
        )
        completion.write_text(
            json.dumps({"status": "completed", "execution_mode": "test_double"}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ManifestError, "测试替身"):
            _local_execution_summary(
                {
                    "status": "passed_evidence_generated",
                    "main_evidence": str(main),
                    "completion_marker": str(completion),
                }
            )


if __name__ == "__main__":
    unittest.main()
