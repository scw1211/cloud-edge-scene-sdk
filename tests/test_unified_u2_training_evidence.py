import unittest
from pathlib import Path

from edge_llm_factory.contracts import ManifestError
from edge_llm_factory import train_unified_traffic_general_u2 as u2


class UnifiedU2TrainingEvidenceStubTest(unittest.TestCase):
    def test_training_script_identity_is_stable_and_detects_drift(self) -> None:
        before = u2.training_script_identity()
        report = u2.verify_training_script_stability(before)
        self.assertTrue(report["stable_across_training"])
        self.assertEqual(report["script"], before)
        self.assertEqual(report["script"]["bytes"], Path(u2.__file__).stat().st_size)
        self.assertEqual(report["script"]["sha256"], u2.sha256_file(Path(u2.__file__)))

        changed = {**before, "sha256": "0" * 64}
        with self.assertRaisesRegex(ManifestError, "训练前后发生漂移"):
            u2.verify_training_script_stability(changed)


if __name__ == "__main__":
    unittest.main()
