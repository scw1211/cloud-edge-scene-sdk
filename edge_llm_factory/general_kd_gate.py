"""用途：比较通用蒸馏候选与当前基座，阻止能力退化模型进入量化发布。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from edge_llm_factory.contracts import (
    ManifestError,
    read_json_object,
    sha256_file,
    write_json_object,
)


def _summary(result: Mapping[str, Any], label: str) -> Dict[str, Any]:
    models = result.get("models")
    if not isinstance(models, dict) or label not in models:
        raise ManifestError("评测结果缺少模型标签: {}".format(label))
    model = models[label]
    summary = model.get("summary") if isinstance(model, dict) else None
    if not isinstance(summary, dict):
        raise ManifestError("模型 {} 缺少 summary".format(label))
    categories = summary.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ManifestError("模型 {} 缺少分项能力结果".format(label))
    return dict(summary)


def parse_category_improvements(values: Sequence[str]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ManifestError("分项提升门槛必须使用 category=value 格式")
        category, raw_delta = value.split("=", 1)
        category = category.strip()
        if not category or category in parsed:
            raise ManifestError("分项提升门槛类别为空或重复: {}".format(category))
        try:
            delta = float(raw_delta)
        except ValueError as exc:
            raise ManifestError("分项提升门槛不是数字: {}".format(value)) from exc
        if delta < 0:
            raise ManifestError("分项提升门槛不能为负数: {}".format(value))
        parsed[category] = delta
    return parsed


def evaluate_gate(
    result: Mapping[str, Any],
    incumbent_label: str,
    candidate_label: str,
    minimum_macro_delta: float = 0.0,
    max_category_regression: float = 0.0,
    minimum_category_improvements: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    if incumbent_label == candidate_label:
        raise ManifestError("候选模型不能与当前模型相同")
    if max_category_regression < 0:
        raise ManifestError("max_category_regression 不能为负数")
    improvements = dict(minimum_category_improvements or {})
    incumbent = _summary(result, incumbent_label)
    candidate = _summary(result, candidate_label)
    expected_samples = int(result.get("sample_count", 0))
    reasons = []
    for label, summary in ((incumbent_label, incumbent), (candidate_label, candidate)):
        completed = int(summary.get("completed_samples", 0))
        if expected_samples <= 0 or completed != expected_samples:
            reasons.append(
                "{} completed_samples={}，冻结评测要求 {}".format(
                    label, completed, expected_samples
                )
            )

    incumbent_categories = incumbent["categories"]
    candidate_categories = candidate["categories"]
    if set(incumbent_categories) != set(candidate_categories):
        raise ManifestError("候选与当前模型的评测类别不一致")
    unknown = set(improvements) - set(candidate_categories)
    if unknown:
        raise ManifestError("提升门槛包含未知类别: {}".format(sorted(unknown)))

    category_report = {}
    for category in sorted(candidate_categories):
        incumbent_score = float(incumbent_categories[category]["score"])
        candidate_score = float(candidate_categories[category]["score"])
        delta = candidate_score - incumbent_score
        required_delta = float(improvements.get(category, -max_category_regression))
        if required_delta == 0:
            required_delta = 0.0
        passed = delta + 1e-12 >= required_delta
        if not passed:
            reasons.append(
                "{} delta={:.6f}，门槛为 >= {:.6f}".format(
                    category, delta, required_delta
                )
            )
        category_report[category] = {
            "incumbent_score": round(incumbent_score, 6),
            "candidate_score": round(candidate_score, 6),
            "delta": round(delta, 6),
            "required_delta": round(required_delta, 6),
            "passed": passed,
        }

    incumbent_macro = float(incumbent["overall_macro_score"])
    candidate_macro = float(candidate["overall_macro_score"])
    macro_delta = candidate_macro - incumbent_macro
    macro_passed = macro_delta + 1e-12 >= minimum_macro_delta
    if not macro_passed:
        reasons.append(
            "macro delta={:.6f}，门槛为 >= {:.6f}".format(
                macro_delta, minimum_macro_delta
            )
        )
    passed = not reasons
    return {
        "schema_version": "edge-llm-general-kd-gate/v1",
        "task": "general_kd_candidate_promotion_gate",
        "incumbent_label": incumbent_label,
        "candidate_label": candidate_label,
        "sample_count": expected_samples,
        "thresholds": {
            "minimum_macro_delta": minimum_macro_delta,
            "max_category_regression": max_category_regression,
            "minimum_category_improvements": improvements,
        },
        "macro": {
            "incumbent_score": round(incumbent_macro, 6),
            "candidate_score": round(candidate_macro, 6),
            "delta": round(macro_delta, 6),
            "required_delta": round(minimum_macro_delta, 6),
            "passed": macro_passed,
        },
        "categories": category_report,
        "passed": passed,
        "quantization_allowed": passed,
        "reasons": reasons,
    }


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="执行通用蒸馏候选基座发布门禁。")
    parser.add_argument("--evaluation_json", required=True)
    parser.add_argument("--incumbent_label", required=True)
    parser.add_argument("--candidate_label", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--minimum_macro_delta", type=float, default=0.0)
    parser.add_argument("--max_category_regression", type=float, default=0.0)
    parser.add_argument("--minimum_category_improvement", action="append", default=[])
    args = parser.parse_args(argv)

    if args.minimum_macro_delta < 0:
        raise ManifestError("minimum_macro_delta 不能为负数")
    evaluation_path = Path(args.evaluation_json).resolve()
    result = read_json_object(evaluation_path)
    report = evaluate_gate(
        result,
        args.incumbent_label,
        args.candidate_label,
        minimum_macro_delta=args.minimum_macro_delta,
        max_category_regression=args.max_category_regression,
        minimum_category_improvements=parse_category_improvements(
            args.minimum_category_improvement
        ),
    )
    report["evaluation_artifact"] = {
        "path": str(evaluation_path),
        "sha256": sha256_file(evaluation_path),
    }
    output = Path(args.output_json).resolve()
    write_json_object(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
