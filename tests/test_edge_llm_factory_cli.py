import unittest
from unittest.mock import patch

from edge_llm_factory import __main__ as cli


class EdgeLlmFactoryCliTest(unittest.TestCase):
    def test_new_commands_forward_remaining_arguments(self) -> None:
        cases = (
            (
                "build-frozen-math-eval",
                "edge_llm_factory.build_frozen_math_eval.main",
            ),
            (
                "build-frozen-nlr-eval",
                "edge_llm_factory.build_frozen_nlr_eval.main",
            ),
            ("focus-general-kd", "edge_llm_factory.focus_general_kd_dataset.main"),
            (
                "evaluate-general-adapter",
                "edge_llm_factory.evaluate_general_adapter.main",
            ),
            (
                "evaluate-unified-general-teacher",
                "edge_llm_factory.evaluate_unified_general_teacher.main",
            ),
            (
                "gate-unified-general-paired",
                "edge_llm_factory.gate_unified_general_paired.main",
            ),
        )
        for command, target in cases:
            with self.subTest(command=command), patch(target) as command_main:
                result = cli.main([command, "--sentinel", "value"])

                self.assertIsNone(result)
                command_main.assert_called_once_with(["--sentinel", "value"])

    def test_dual_promotion_gate_preserves_failure_exit_code(self) -> None:
        with patch(
            "edge_llm_factory.dual_promotion_gate.main", return_value=2
        ) as command_main:
            result = cli.main(
                ["gate-dual-promotion", "--evidence-json", "evidence.json"]
            )

        self.assertEqual(result, 2)
        command_main.assert_called_once_with(
            ["--evidence-json", "evidence.json"]
        )


if __name__ == "__main__":
    unittest.main()
