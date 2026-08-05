"""Regression tests for the full traffic deployment replay policy."""

from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloud_edge_framework.service_config import load_service_config  # noqa: E402


class FullTrafficDeploymentConfigTests(unittest.TestCase):
    def test_partial_results_are_polled_quickly_without_moving_safety_bounds(self):
        config_path = (
            REPOSITORY_ROOT
            / "scenes"
            / "freeway_traffic"
            / "deployment"
            / "full"
            / "edge_service.json"
        )

        config = load_service_config(
            config_path,
            REPOSITORY_ROOT,
            expected_role="edge",
        )

        self.assertEqual(config.replay.waiting_poll_seconds, 0.025)
        self.assertEqual(config.replay.partial_poll_seconds, 0.05)
        self.assertEqual(config.replay.aggregation_max_wait_seconds, 10.0)
        self.assertEqual(config.replay.reconciliation_poll_seconds, 5.0)
        self.assertEqual(config.replay.reconciliation_max_wait_seconds, 60.0)


if __name__ == "__main__":
    unittest.main()
