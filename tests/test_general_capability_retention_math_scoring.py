import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scenes/freeway_traffic/traffic_system/eval_general_capability_retention.py"
)
SPEC = importlib.util.spec_from_file_location("general_capability_retention", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class GeneralCapabilityRetentionMathScoringTests(unittest.TestCase):
    def test_accepts_exact_terminal_final_answer(self):
        result = MODULE.evaluate_math("work\nFINAL: 1,234.5", "1234.5")
        self.assertTrue(result["correct"])
        self.assertEqual(result["prediction"], "1,234.5")

    def test_rejects_expression_after_final_marker(self):
        result = MODULE.evaluate_math("FINAL: $90 - 20 = 70", "90")
        self.assertFalse(result["correct"])
        self.assertIsNone(result["prediction"])

    def test_rejects_output_without_terminal_final_marker(self):
        result = MODULE.evaluate_math("unfinished calculation 12", "12")
        self.assertFalse(result["correct"])
        self.assertIsNone(result["prediction"])

    def test_choice_requires_exact_single_letter(self):
        self.assertTrue(MODULE.evaluate_choice(" c\n", "C")["correct"])
        for output in ("答案是 C", "C because", "A/C", "AB"):
            with self.subTest(output=output):
                result = MODULE.evaluate_choice(output, "C")
                self.assertFalse(result["correct"])
                self.assertIsNone(result["prediction"])


if __name__ == "__main__":
    unittest.main()
