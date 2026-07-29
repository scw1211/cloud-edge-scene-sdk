"""用途：训练使用完整边缘态势的云端专用决策协调分类器。"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from traffic_system.decision_utils import DECISION_CLASSES, labels_from_records, read_jsonl, save_json
from traffic_system.train_edge_student import classification_metrics, split_indices_temporal_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a cloud-side traffic decision coordinator.")
    parser.add_argument("--labels", default="datasets/freeway_teacher_labels_qwen9b_safe_joint_metis4.jsonl")
    parser.add_argument("--model", default="models/cloud_coordinator_extratrees.joblib")
    parser.add_argument("--metrics", default="results/decision/cloud_coordinator_extratrees.json")
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--test_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(Path(args.labels))
    x, y, feature_names, event_ids = labels_from_records(records)
    train_indices, test_indices = split_indices_temporal_group(event_ids, args.test_ratio)
    model = ExtraTreesClassifier(
        n_estimators=args.trees,
        max_features="sqrt",
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(x[train_indices], y[train_indices])

    def evaluate(indices: np.ndarray) -> dict:
        predictions = model.predict(x[indices])
        raw_probabilities = model.predict_proba(x[indices])
        probabilities = np.zeros((len(indices), len(DECISION_CLASSES)), dtype=np.float64)
        for column, class_id in enumerate(model.classes_.tolist()):
            probabilities[:, int(class_id)] = raw_probabilities[:, column]
        return classification_metrics(
            y[indices], predictions, probabilities, [event_ids[int(index)] for index in indices]
        )

    payload = {
        "task": "cloud_global_traffic_decision_coordinator",
        "model_type": "sklearn_extra_trees",
        "trees": args.trees,
        "feature_names": feature_names,
        "decision_classes": DECISION_CLASSES,
        "split_strategy": "strict_temporal_group",
        "num_train": len(train_indices),
        "num_test": len(test_indices),
        "train": evaluate(train_indices),
        "test": evaluate(test_indices),
    }
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": feature_names, "decision_classes": DECISION_CLASSES},
        model_path,
        compress=3,
    )
    save_json(payload, Path(args.metrics))
    print("test accuracy:", payload["test"]["accuracy"])
    print("test weighted F1:", payload["test"]["weighted_f1"])
    print("saved:", args.model)


if __name__ == "__main__":
    main()
