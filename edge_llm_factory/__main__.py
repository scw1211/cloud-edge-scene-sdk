"""用途：提供统一命令入口，避免场景团队记忆多个内部脚本路径。"""

import argparse
from typing import Optional


def main(argv: Optional[list] = None) -> Optional[int]:
    parser = argparse.ArgumentParser(prog="python -m edge_llm_factory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "verify-base",
        "export-text-base",
        "verify-text-base",
        "build-calibration",
        "build-imatrix",
        "benchmark-runtime",
        "verify-runtime",
        "probe-runtime",
        "build-general-kd-source",
        "build-general-kd",
        "build-ceval-kd-source",
        "build-frozen-math-eval",
        "build-frozen-nlr-eval",
        "focus-general-kd",
        "train-general-kd",
        "gate-general-kd",
        "evaluate-general-adapter",
        "evaluate-unified-general-teacher",
        "gate-unified-general-paired",
        "gate-dual-promotion",
        "train-sft",
        "evaluate",
        "merge-lora",
        "export-gguf",
        "build-adapter",
        "validate-adapter",
        "release",
        "serve-release",
        "measure-update-loop",
        "measure-distributed-update-loop",
        "run-pipeline",
    ):
        subparsers.add_parser(command, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    if args.command == "verify-base":
        from edge_llm_factory.base_snapshot import main as command_main
    elif args.command in {"export-text-base", "verify-text-base"}:
        from edge_llm_factory.text_base import main as text_base_main

        text_base_command = "export" if args.command == "export-text-base" else "verify"
        command_main = lambda values: text_base_main([text_base_command] + values)
    elif args.command == "build-calibration":
        from edge_llm_factory.general_calibration import main as command_main
    elif args.command == "build-imatrix":
        from edge_llm_factory.build_imatrix import main as command_main
    elif args.command == "benchmark-runtime":
        from edge_llm_factory.benchmark_runtime import main as command_main
    elif args.command in {"verify-runtime", "probe-runtime"}:
        from edge_llm_factory.providers import main as provider_main

        provider_command = "verify" if args.command == "verify-runtime" else "probe"
        command_main = lambda values: provider_main([provider_command] + values)
    elif args.command == "build-general-kd-source":
        from edge_llm_factory.general_kd_source import main as command_main
    elif args.command == "build-general-kd":
        from edge_llm_factory.general_kd_data import main as command_main
    elif args.command == "build-ceval-kd-source":
        from edge_llm_factory.build_ceval_kd_source import main as command_main
    elif args.command == "build-frozen-math-eval":
        from edge_llm_factory.build_frozen_math_eval import main as command_main
    elif args.command == "build-frozen-nlr-eval":
        from edge_llm_factory.build_frozen_nlr_eval import main as command_main
    elif args.command == "focus-general-kd":
        from edge_llm_factory.focus_general_kd_dataset import main as command_main
    elif args.command == "train-general-kd":
        from edge_llm_factory.train_general_kd import main as command_main
    elif args.command == "gate-general-kd":
        from edge_llm_factory.general_kd_gate import main as command_main
    elif args.command == "evaluate-general-adapter":
        from edge_llm_factory.evaluate_general_adapter import main as command_main
    elif args.command == "evaluate-unified-general-teacher":
        from edge_llm_factory.evaluate_unified_general_teacher import main as command_main
    elif args.command == "gate-unified-general-paired":
        from edge_llm_factory.gate_unified_general_paired import main as command_main
    elif args.command == "gate-dual-promotion":
        from edge_llm_factory.dual_promotion_gate import main as command_main
    elif args.command == "train-sft":
        from edge_llm_factory.train_sft import main as command_main
    elif args.command == "evaluate":
        from edge_llm_factory.evaluate_action_tokens import main as command_main
    elif args.command == "merge-lora":
        from edge_llm_factory.merge_lora import main as command_main
    elif args.command == "export-gguf":
        from edge_llm_factory.export_gguf import main as command_main
    elif args.command in {"build-adapter", "validate-adapter"}:
        from edge_llm_factory.adapter_package import main as package_main

        package_command = "build" if args.command == "build-adapter" else "validate"
        command_main = lambda values: package_main([package_command] + values)
    elif args.command == "release":
        from edge_llm_factory.release_store import main as command_main
    elif args.command == "serve-release":
        from edge_llm_factory.serve_release import main as command_main
    elif args.command == "measure-update-loop":
        from edge_llm_factory.update_loop_evidence import main as command_main
    elif args.command == "measure-distributed-update-loop":
        from edge_llm_factory.distributed_update_evidence import main as command_main
    else:
        from edge_llm_factory.pipeline import main as command_main
    result = command_main(remaining)
    return result if isinstance(result, int) else None


if __name__ == "__main__":
    raise SystemExit(main())
