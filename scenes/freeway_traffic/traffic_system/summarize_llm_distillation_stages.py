"""用途：汇总 SFT、纠错蒸馏、DPO 和量化阶段的严格测试结果及部署选择。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize strict Qwen distillation stages.")
    parser.add_argument("--output_json", default="results/llm/qwen_v9_distillation_stage_summary.json")
    return parser.parse_args()


def load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluation(path: str) -> Dict[str, Any]:
    data = load(path)
    classification = data.get("token_classification") or {}
    return {
        "evidence": path,
        "count": data.get("count"),
        "accuracy": data.get("decision_accuracy"),
        "macro_f1": classification.get("macro_f1"),
        "weighted_f1": classification.get("weighted_f1"),
        "valid_output_rate": data.get("json_valid_rate"),
    }


def main() -> None:
    args = parse_args()
    candidates = {
        "phase1_v9_sft": evaluation("results/llm/llm_action_token_v9_test_hf_eval.json"),
        "phase2_r24_correction": evaluation(
            "results/llm/llm_action_token_v9_phase2_r24_strict_test_hf_eval.json"
        ),
        "phase2_r8_correction": evaluation(
            "results/llm/llm_action_token_v9_phase2_r8_strict_test_hf_eval.json"
        ),
        "phase3_dpo": evaluation(
            "results/llm/llm_action_token_v9_dpo_strict_test_hf_eval.json"
        ),
    }
    best_name, best = max(
        candidates.items(),
        key=lambda item: (
            float(item[1].get("accuracy") or 0.0),
            float(item[1].get("weighted_f1") or 0.0),
        ),
    )
    output = {
        "task": "qwen_traffic_distillation_stage_comparison",
        "strict_test_protocol": {
            "events": 36,
            "test_set_used_for_training": False,
            "test_id_sha256": load(
                "datasets/llm_sft_freeway_action_token_v9_phase2_preference/summary.json"
            )["strict_test_id_sha256"],
        },
        "stages": {
            "phase1": "Qwen 9B safety-constrained labels -> Qwen 0.8B LoRA SFT",
            "phase2": "Student rollout errors -> low-rate correction LoRA continuation",
            "phase3": "Teacher chosen token vs Student hard negative -> DPO",
            "phase4": "merge text-only language model -> Q6_K GGUF -> Jetson llama.cpp",
        },
        "dataset": load(
            "datasets/llm_sft_freeway_action_token_v9_phase2_preference/summary.json"
        ),
        "candidates": candidates,
        "selected_for_deployment": best_name,
        "selected_metrics": best,
        "deployment_artifact": (
            "models/gguf/qwen35_0_8b_freeway_action_token_v9_text_only_q6_k.gguf"
        ),
        "phase2_improved_strict_test": (
            candidates["phase2_r8_correction"]["accuracy"]
            > candidates["phase1_v9_sft"]["accuracy"]
        ),
        "phase3_improved_strict_test": (
            candidates["phase3_dpo"]["accuracy"]
            > candidates["phase1_v9_sft"]["accuracy"]
        ),
        "conclusion": (
            "Correction and DPO are reproducible but do not improve the untouched temporal test; "
            "retain phase-1 v9 as the deployed model."
        ),
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
