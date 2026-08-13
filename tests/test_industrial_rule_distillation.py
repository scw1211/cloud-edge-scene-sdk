import json
from pathlib import Path
import tempfile
import unittest


from scenes.industrial_anomaly.build_industrial_rule_distillation_dataset import (
    build,
    encode_prompt,
    state_for_score,
)
from scenes.industrial_anomaly.build_industrial_margin_distillation_dataset import (
    encode_margin_prompt,
    margin_position,
    prompt_from_values,
)


ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (
    ROOT
    / "scenes"
    / "industrial_anomaly"
    / "industrial_anomaly"
    / "review_bands.json"
)


class IndustrialRuleDistillationTests(unittest.TestCase):
    def test_prompt_is_decimal16_and_state_contract_is_exact(self):
        prompt = encode_prompt("rgb", 3, 5000, 6000, 7000)
        self.assertEqual(prompt, "1003500060007000")
        self.assertEqual(state_for_score(5999, 6000, 7000), "normal")
        self.assertEqual(state_for_score(6000, 6000, 7000), "review")
        self.assertEqual(state_for_score(6999, 6000, 7000), "review")
        self.assertEqual(state_for_score(7000, 6000, 7000), "anomaly")

    def test_builder_freezes_disjoint_balanced_splits(self):
        with tempfile.TemporaryDirectory(prefix="industrial-kd-") as directory:
            output = Path(directory) / "dataset"
            result = build(THRESHOLDS, output, 20260812)
            self.assertEqual(result["artifacts"]["train"]["rows"], 3840)
            self.assertEqual(result["artifacts"]["validation"]["rows"], 480)
            self.assertEqual(result["artifacts"]["test"]["rows"], 960)
            self.assertEqual(
                result["target_distribution"]["test"],
                {"A": 320, "B": 320, "C": 320},
            )
            prompts = {}
            for split in ("train", "validation", "test"):
                rows = [
                    json.loads(line)
                    for line in (output / (split + ".jsonl"))
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                prompts[split] = {row["messages"][0]["content"] for row in rows}
                self.assertTrue(all(len(value) == 16 and value.isdigit() for value in prompts[split]))
            self.assertFalse(prompts["train"] & prompts["validation"])
            self.assertFalse(prompts["train"] & prompts["test"])
            self.assertFalse(prompts["validation"] & prompts["test"])

    def test_refuses_to_overwrite_dataset(self):
        with tempfile.TemporaryDirectory(prefix="industrial-kd-") as directory:
            output = Path(directory) / "dataset"
            build(THRESHOLDS, output, 20260812)
            with self.assertRaises(FileExistsError):
                build(THRESHOLDS, output, 20260812)

    def test_margin_codec_preserves_review_boundaries(self):
        self.assertEqual(margin_position(0.5, 0.6, 0.7), -1000)
        self.assertEqual(margin_position(0.6, 0.6, 0.7), 0)
        self.assertEqual(margin_position(0.7, 0.6, 0.7), 1000)
        normal = prompt_from_values("rgb", 3, 0.005, 0.006, 0.007)
        review = prompt_from_values("rgb", 3, 0.006, 0.006, 0.007)
        anomaly = prompt_from_values("rgb", 3, 0.007, 0.006, 0.007)
        self.assertEqual(len(normal), 16)
        self.assertEqual(normal[4:8], "3000")
        self.assertEqual(review[4:8], "4000")
        self.assertEqual(anomaly[8:12], "4000")
        self.assertEqual(
            encode_margin_prompt("infrared", 9, 999, 356),
            "2109499939990356",
        )


if __name__ == "__main__":
    unittest.main()
