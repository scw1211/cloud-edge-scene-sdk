import unittest

from edge_llm_factory.build_frozen_math_eval import (
    build_math_evaluation,
    exclusion_fingerprints,
    extract_reference,
    prompt_fingerprint,
)
from edge_llm_factory.contracts import ManifestError


class FrozenMathEvaluationTests(unittest.TestCase):
    def _rows(self, count=10):
        return [
            {"question": f"Question {index}?", "answer": f"work\n#### {index:,}"}
            for index in range(count)
        ]

    def test_reference_is_canonical_number(self):
        self.assertEqual(extract_reference("steps\n#### 1,234.5"), "1234.5")

    def test_reference_requires_one_marker(self):
        with self.assertRaises(ManifestError):
            extract_reference("answer 3")
        with self.assertRaises(ManifestError):
            extract_reference("#### 1\n#### 2")

    def test_selection_is_deterministic_and_excludes_prompts(self):
        excluded, _ = exclusion_fingerprints(
            [("old", [{"prompt": "Question 2?"}, {"source_prompt": "Question 3?"}])]
        )
        first, first_report = build_math_evaluation(
            self._rows(), excluded, sample_count=4, seed=7
        )
        second, second_report = build_math_evaluation(
            self._rows(), excluded, sample_count=4, seed=7
        )
        self.assertEqual(first, second)
        self.assertEqual(first_report, second_report)
        fingerprints = {row["prompt_fingerprint"] for row in first}
        self.assertNotIn(prompt_fingerprint("Question 2?"), fingerprints)
        self.assertNotIn(prompt_fingerprint("Question 3?"), fingerprints)
        self.assertEqual(first_report["excluded_prompt_count"], 2)

    def test_user_message_can_supply_exclusion_prompt(self):
        excluded, counts = exclusion_fingerprints(
            [
                (
                    "train",
                    [
                        {
                            "messages": [
                                {"role": "system", "content": "x"},
                                {"role": "user", "content": "Question 4?"},
                                {"role": "assistant", "content": "4"},
                            ]
                        }
                    ],
                )
            ]
        )
        self.assertIn(prompt_fingerprint("Question 4?"), excluded)
        self.assertEqual(counts, {"train": 1})

    def test_fails_when_pool_is_too_small(self):
        with self.assertRaises(ManifestError):
            build_math_evaluation(self._rows(2), set(), sample_count=3, seed=1)


if __name__ == "__main__":
    unittest.main()
