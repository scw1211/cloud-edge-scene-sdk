"""用途：对比 Teacher 与边缘 Qwen 的数学、代码和中文推理能力保持率。"""

import argparse
import ast
import json
import os
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traffic_system.decision_utils import read_jsonl, save_json  # noqa: E402


DEFAULT_MODELS = [
    "teacher=qwen3.5:9b",
    "base_student=qwen3.5:0.8b",
    "traffic_student=qwen35-freeway-action-general-eval",
]

ALLOWED_IMPORTS = {
    "bisect",
    "collections",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "re",
    "statistics",
    "string",
    "typing",
    "decimal",
    "fractions",
}

DANGEROUS_CALLS = {
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate general-capability retention with Ollama.")
    parser.add_argument("--dataset_jsonl", default="datasets/general_capability_eval/eval.jsonl")
    parser.add_argument("--output_json", default="results/llm/general_capability_retention_v1.json")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--teacher_label", default="teacher")
    parser.add_argument("--num_ctx", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--code_timeout", type=float, default=3.0)
    parser.add_argument("--limit_per_category", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_models(values: Sequence[str]) -> List[Tuple[str, str]]:
    parsed = []
    for value in values or DEFAULT_MODELS:
        if "=" not in value:
            raise ValueError("Model must use label=name format: {}".format(value))
        label, model = value.split("=", 1)
        label = label.strip()
        model = model.strip()
        if not label or not model:
            raise ValueError("Invalid model specification: {}".format(value))
        parsed.append((label, model))
    if len({label for label, _ in parsed}) != len(parsed):
        raise ValueError("Model labels must be unique.")
    return parsed


def select_rows(rows: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return list(rows)
    selected = []
    counts: Dict[str, int] = {}
    for row in rows:
        category = str(row["category"])
        if counts.get(category, 0) >= limit:
            continue
        selected.append(row)
        counts[category] = counts.get(category, 0) + 1
    return selected


def task_prompt(row: Dict[str, Any]) -> Tuple[str, str, int]:
    category = row["category"]
    if category == "math":
        return (
            "You solve grade-school math accurately. Do not provide a long explanation. "
            "End with exactly: FINAL: <number>",
            str(row["prompt"]),
            256,
        )
    if category == "code":
        tests = "\n".join(row.get("test_list", []))
        return (
            "Write a correct Python solution. Output only Python code without Markdown fences or explanation.",
            "{}\nThe function must pass these tests:\n{}".format(row["prompt"], tests),
            384,
        )
    if category == "natural_language_reasoning":
        return (
            "回答中文单项选择题。只输出 A、B、C 或 D，不要解释。",
            str(row["prompt"]),
            4,
        )
    raise ValueError("Unsupported category: {}".format(category))


def ollama_chat(
    host: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    num_ctx: int,
    num_predict: int,
    timeout: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    request = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    wall_ms = round((time.perf_counter() - started) * 1000.0, 4)
    return {
        "text": str(data.get("message", {}).get("content", "")).strip(),
        "wall_time_ms": wall_ms,
        "prompt_tokens": int(data.get("prompt_eval_count", 0)),
        "output_tokens": int(data.get("eval_count", 0)),
        "load_duration_ms": round(data.get("load_duration", 0) / 1_000_000.0, 4),
        "prompt_eval_duration_ms": round(data.get("prompt_eval_duration", 0) / 1_000_000.0, 4),
        "eval_duration_ms": round(data.get("eval_duration", 0) / 1_000_000.0, 4),
    }


def unload_model(host: str, model: str) -> None:
    payload = {"model": model, "keep_alive": 0}
    request = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30):
            pass
    except Exception:  # noqa: BLE001
        return


def normalize_number(value: str) -> Optional[Decimal]:
    clean = value.strip().replace(",", "").replace("$", "")
    clean = clean.rstrip(".")
    try:
        return Decimal(clean)
    except InvalidOperation:
        return None


def evaluate_math(text: str, reference: str) -> Dict[str, Any]:
    final_match = re.search(r"FINAL\s*:\s*([-+$]?[0-9][0-9,]*(?:\.[0-9]+)?)", text, re.I)
    matches = re.findall(r"[-+$]?[0-9][0-9,]*(?:\.[0-9]+)?", text)
    predicted_text = final_match.group(1) if final_match else (matches[-1] if matches else "")
    prediction = normalize_number(predicted_text)
    expected = normalize_number(reference)
    return {
        "prediction": predicted_text or None,
        "reference": reference,
        "correct": prediction is not None and expected is not None and prediction == expected,
    }


def evaluate_choice(text: str, reference: str) -> Dict[str, Any]:
    match = re.search(r"(?<![A-Z])([A-D])(?![A-Z])", text.upper())
    prediction = match.group(1) if match else None
    return {
        "prediction": prediction,
        "reference": reference,
        "correct": prediction == reference,
    }


def extract_code(text: str) -> str:
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    code = fence.group(1).strip() if fence else text.strip()
    lines = code.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"\s*(?:from\s+\w+\s+import|import\s+\w+|def\s+\w+)", line):
            return "\n".join(lines[index:]).strip()
    return code


def validate_code_ast(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return "syntax_error: {}".format(exc)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] not in ALLOWED_IMPORTS:
                    return "blocked_import: {}".format(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if module not in ALLOWED_IMPORTS:
                return "blocked_import: {}".format(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in DANGEROUS_CALLS:
                return "blocked_call: {}".format(node.func.id)
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "blocked_dunder_attribute: {}".format(node.attr)
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            return "blocked_dunder_name: {}".format(node.id)
    return None


def code_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def evaluate_code(text: str, row: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    code = extract_code(text)
    imports = "\n".join(row.get("test_imports", []))
    tests = "\n".join(row.get("test_list", []))
    program = "{}\n{}\n{}\n".format(imports, code, tests)
    validation_error = validate_code_ast(program)
    if validation_error:
        return {
            "prediction": code,
            "reference": "official MBPP tests",
            "correct": False,
            "execution_error": validation_error,
        }
    try:
        with tempfile.TemporaryDirectory(prefix="mbpp_eval_") as directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-"],
                input=program,
                text=True,
                cwd=directory,
                capture_output=True,
                timeout=timeout,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
                preexec_fn=code_limits,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return {
            "prediction": code,
            "reference": "official MBPP tests",
            "correct": False,
            "execution_error": "timeout",
        }
    return {
        "prediction": code,
        "reference": "official MBPP tests",
        "correct": completed.returncode == 0,
        "execution_error": None if completed.returncode == 0 else completed.stderr[-1000:],
    }


def evaluate_response(text: str, row: Dict[str, Any], code_timeout: float) -> Dict[str, Any]:
    category = row["category"]
    if category == "math":
        return evaluate_math(text, str(row["reference_answer"]))
    if category == "code":
        return evaluate_code(text, row, code_timeout)
    return evaluate_choice(text, str(row["reference_answer"]))


def empty_result(args: argparse.Namespace, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "task": "general_capability_retention",
        "dataset_jsonl": args.dataset_jsonl,
        "num_ctx": args.num_ctx,
        "no_thinking": True,
        "sample_count": len(rows),
        "models": {},
        "retention": {},
    }


def load_checkpoint(path: Path, args: argparse.Namespace, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if args.resume and path.exists():
        with path.open("r", encoding="utf-8") as file_obj:
            value = json.load(file_obj)
        if isinstance(value, dict):
            return value
    return empty_result(args, rows)


def model_summary(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    categories = sorted({str(sample["category"]) for sample in samples})
    category_metrics = {}
    for category in categories:
        selected = [sample for sample in samples if sample["category"] == category]
        category_metrics[category] = {
            "total": len(selected),
            "correct": sum(bool(sample["correct"]) for sample in selected),
            "score": round(sum(bool(sample["correct"]) for sample in selected) / len(selected), 6),
            "average_wall_time_ms": round(
                statistics.fmean(sample["wall_time_ms"] for sample in selected), 4
            ),
        }
    return {
        "completed_samples": len(samples),
        "overall_micro_score": round(
            sum(bool(sample["correct"]) for sample in samples) / len(samples), 6
        ),
        "overall_macro_score": round(
            statistics.fmean(metric["score"] for metric in category_metrics.values()), 6
        ),
        "categories": category_metrics,
    }


def update_retention(result: Dict[str, Any], teacher_label: str) -> None:
    models = result.get("models", {})
    teacher = models.get(teacher_label, {}).get("summary")
    if not teacher:
        result["retention"] = {}
        return
    retention = {}
    for label, model_result in models.items():
        if label == teacher_label or "summary" not in model_result:
            continue
        category_retention = {}
        for category, teacher_metric in teacher["categories"].items():
            student_metric = model_result["summary"]["categories"].get(category)
            teacher_score = teacher_metric["score"]
            ratio = None if not student_metric or teacher_score == 0 else student_metric["score"] / teacher_score
            category_retention[category] = {
                "teacher_score": teacher_score,
                "student_score": student_metric["score"] if student_metric else None,
                "retention_ratio": round(ratio, 6) if ratio is not None else None,
                "meets_80_percent": ratio is not None and ratio >= 0.8,
            }
        teacher_macro = teacher["overall_macro_score"]
        student_macro = model_result["summary"]["overall_macro_score"]
        macro_ratio = None if teacher_macro == 0 else student_macro / teacher_macro
        retention[label] = {
            "categories": category_retention,
            "macro_retention_ratio": round(macro_ratio, 6) if macro_ratio is not None else None,
            "all_categories_meet_80_percent": all(
                metric["meets_80_percent"] for metric in category_retention.values()
            ),
        }
    result["retention"] = retention


def main() -> None:
    args = parse_args()
    rows = select_rows(read_jsonl(resolve_path(args.dataset_jsonl)), args.limit_per_category)
    models = parse_models(args.model)
    output_path = resolve_path(args.output_json)
    result = load_checkpoint(output_path, args, rows)

    for label, model in models:
        model_result = result["models"].setdefault(label, {"model": model, "samples": []})
        if model_result.get("model") != model:
            raise ValueError("Checkpoint model mismatch for {}".format(label))
        completed_ids = {sample["sample_id"] for sample in model_result["samples"]}
        try:
            for index, row in enumerate(rows, start=1):
                if row["sample_id"] in completed_ids:
                    continue
                system_prompt, user_prompt, num_predict = task_prompt(row)
                response = ollama_chat(
                    host=args.host,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    num_ctx=args.num_ctx,
                    num_predict=num_predict,
                    timeout=args.timeout,
                )
                evaluation = evaluate_response(response["text"], row, args.code_timeout)
                sample = {
                    "sample_id": row["sample_id"],
                    "benchmark": row["benchmark"],
                    "category": row["category"],
                    "correct": bool(evaluation["correct"]),
                    "prediction": evaluation.get("prediction"),
                    "reference": evaluation.get("reference"),
                    "execution_error": evaluation.get("execution_error"),
                    "raw_output": response.pop("text"),
                    **response,
                }
                model_result["samples"].append(sample)
                model_result["summary"] = model_summary(model_result["samples"])
                update_retention(result, args.teacher_label)
                save_json(result, output_path)
                print(
                    "[{}/{}] {} {} {}".format(
                        index,
                        len(rows),
                        label,
                        row["sample_id"],
                        "PASS" if sample["correct"] else "FAIL",
                    ),
                    flush=True,
                )
        finally:
            unload_model(args.host, model)

    update_retention(result, args.teacher_label)
    save_json(result, output_path)
    printable = {
        label: model_result.get("summary")
        for label, model_result in result["models"].items()
    }
    printable["retention"] = result["retention"]
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
