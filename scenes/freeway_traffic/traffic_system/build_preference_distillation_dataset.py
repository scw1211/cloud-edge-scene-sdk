"""用途：从 Student rollout 构建无测试泄漏的纠错 SFT 与 hard-negative 偏好数据。"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from traffic_system.decision_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict phase-2 and preference datasets.")
    parser.add_argument("--phase1_dir", default="datasets/llm_sft_freeway_action_token_v9")
    parser.add_argument(
        "--train_rollout_json",
        default="results/llm/llm_action_token_v9_train_unique_scores_hf_eval.json",
    )
    parser.add_argument(
        "--development_rollout_json",
        default="results/llm/llm_action_token_v9_val_hf_eval.json",
    )
    parser.add_argument(
        "--cloud_feedback_jsonl",
        default="datasets/cloud_llm_review_feedback.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="datasets/llm_sft_freeway_action_token_v9_phase2_preference",
    )
    parser.add_argument("--correction_repeats", type=int, default=24)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: {}".format(path))
    return value


def unique_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    seen = set()
    for row in rows:
        event_id = str(row.get("event_id"))
        if not event_id or event_id == "None":
            raise ValueError("SFT row is missing event_id.")
        if event_id in seen:
            continue
        seen.add(event_id)
        output.append(row)
    return output


def event_ids(rows: Iterable[Dict[str, Any]]) -> set:
    return {str(row.get("event_id")) for row in rows}


def rollout_examples(path: Path) -> Dict[str, Dict[str, Any]]:
    data = load_json(path)
    examples = data.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Rollout JSON is missing examples: {}".format(path))
    return {str(row.get("event_id")): row for row in examples if isinstance(row, dict)}


def preference_row(source: Dict[str, Any], rollout: Dict[str, Any], split: str) -> Dict[str, Any]:
    chosen = str(source.get("target"))
    rejected = rollout.get("hard_negative") or rollout.get("parsed")
    if not isinstance(rejected, str) or rejected == chosen:
        raise ValueError("Rollout for {} has no valid wrong candidate.".format(source.get("event_id")))
    messages = source.get("messages", [])
    prompt = str(messages[0].get("content", "")) if messages else ""
    return {
        "event_id": source["event_id"],
        "source_split": split,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_source": "qwen3.5:9b_safety_constrained",
        "rejected_source": "student_hard_negative",
        "student_class_probabilities": rollout.get("class_probabilities"),
    }


def hash_ids(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    if args.correction_repeats <= 0:
        raise ValueError("correction_repeats must be positive")
    phase1 = Path(args.phase1_dir)
    output = Path(args.output_dir)
    train = read_jsonl(phase1 / "train.jsonl")
    development = read_jsonl(phase1 / "val.jsonl")
    test = read_jsonl(phase1 / "test.jsonl")
    unique_train = unique_rows(train)
    unique_development = unique_rows(development)
    unique_test = unique_rows(test)

    train_ids = event_ids(unique_train)
    development_ids = event_ids(unique_development)
    test_ids = event_ids(unique_test)
    overlap = {
        "train_development": sorted(train_ids & development_ids),
        "train_test": sorted(train_ids & test_ids),
        "development_test": sorted(development_ids & test_ids),
    }
    if any(overlap.values()):
        raise ValueError("Temporal split overlap detected: {}".format(overlap))

    train_rollouts = rollout_examples(Path(args.train_rollout_json))
    development_rollouts = rollout_examples(Path(args.development_rollout_json))
    missing_train = sorted(train_ids - set(train_rollouts))
    missing_development = sorted(development_ids - set(development_rollouts))
    if missing_train or missing_development:
        raise ValueError(
            "Missing rollout rows: train={}, development={}".format(
                len(missing_train), len(missing_development)
            )
        )

    train_errors = [
        row for row in unique_train if not train_rollouts[row["event_id"]].get("decision_match")
    ]
    development_errors = [
        row
        for row in unique_development
        if not development_rollouts[row["event_id"]].get("decision_match")
    ]
    corrections = []
    for source in train_errors + development_errors:
        split = "train" if source["event_id"] in train_ids else "development"
        rollout = (train_rollouts if split == "train" else development_rollouts)[source["event_id"]]
        corrected = dict(source)
        corrected.update(
            {
                "distillation_stage": "phase2_on_policy_correction",
                "correction_source_split": split,
                "student_wrong_answer": rollout.get("parsed"),
                "teacher_correction": source.get("target"),
            }
        )
        corrections.append(corrected)

    phase2_train = list(train)
    for _ in range(args.correction_repeats):
        phase2_train.extend(dict(row) for row in corrections)

    preferences = []
    for source in unique_train:
        preferences.append(preference_row(source, train_rollouts[source["event_id"]], "train"))
    for source in development_errors:
        rollout = development_rollouts[source["event_id"]]
        if rollout.get("parsed") and rollout.get("parsed") != source.get("target"):
            preferences.append(preference_row(source, rollout, "development"))

    cloud_feedback_path = Path(args.cloud_feedback_jsonl)
    cloud_feedback = read_jsonl(cloud_feedback_path) if cloud_feedback_path.exists() else []
    admitted_feedback = []
    heldout_feedback = []
    for row in cloud_feedback:
        feedback_id = str(row.get("event_id"))
        if feedback_id in train_ids or feedback_id in development_ids:
            admitted_feedback.append(row)
        else:
            heldout_feedback.append(row)
    if admitted_feedback:
        raise ValueError(
            "Cloud feedback conversion is not implemented for compact decisions; refusing silent training use."
        )

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(phase2_train, output / "train.jsonl")
    write_jsonl(development, output / "val.jsonl")
    write_jsonl(preferences, output / "preference_train.jsonl")
    write_jsonl(heldout_feedback, output / "online_feedback_heldout.jsonl")
    write_jsonl(test, output / "strict_test_untouched.jsonl")

    transition_counts = Counter(
        "{}->{}".format(row.get("student_wrong_answer"), row.get("teacher_correction"))
        for row in corrections
    )
    summary = {
        "task": "strict_correction_and_preference_distillation_dataset",
        "phase1_train_rows": len(train),
        "phase1_unique_train_events": len(unique_train),
        "development_events": len(unique_development),
        "strict_test_events": len(unique_test),
        "train_rollout_errors": len(train_errors),
        "development_rollout_errors": len(development_errors),
        "development_split_consumed_for_correction": bool(development_errors),
        "correction_repeats": args.correction_repeats,
        "phase2_train_rows": len(phase2_train),
        "correction_transitions": dict(sorted(transition_counts.items())),
        "preference_pair_count": len(preferences),
        "hard_negative_train_pairs": len(unique_train),
        "online_cloud_feedback_heldout_count": len(heldout_feedback),
        "online_cloud_feedback_used_for_training": False,
        "test_set_used_for_training": False,
        "strict_test_id_sha256": hash_ids(test_ids),
        "split_overlap": overlap,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
