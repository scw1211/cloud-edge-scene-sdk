"""用途：对比 Teacher 与边缘 Qwen 的数学、代码和中文推理能力保持率。"""

import argparse
import ast
import hashlib
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

NLR_CHOICE_SYSTEM_PROMPT = "回答中文单项选择题。只输出 A、B、C 或 D，不要解释。"
NLR_CHOICE_SCORING = "fullmatch whitespace plus exactly one A/B/C/D token"
ATTESTATION_SCHEMA = "edge-llm-ceval-specialty-evaluation-attestation/v1"
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

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
    parser.add_argument("--evaluation_attestation")
    parser.add_argument("--expected_evaluation_attestation_sha256")
    parser.add_argument("--pre_registration")
    parser.add_argument("--expected_preregistration_sha256")
    parser.add_argument("--dataset_manifest")
    parser.add_argument("--final_candidate_selection")
    parser.add_argument("--expected_final_candidate_selection_sha256")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path, field: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read {} JSON {}: {}".format(field, path, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object: {}".format(field, path))
    return value


def require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError("{} must be a SHA-256 string".format(field))
    result = value.strip().lower().removeprefix("sha256:")
    if SHA256_HEX.fullmatch(result) is None:
        raise ValueError("{} must be a 64-character SHA-256".format(field))
    return result


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


def nlr_choice_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "backend": "Ollama",
        "endpoint": "/api/chat",
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "system_prompt": NLR_CHOICE_SYSTEM_PROMPT,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "num_ctx": int(args.num_ctx),
        "num_predict": 4,
        "timeout_seconds": int(args.timeout),
        "scoring": NLR_CHOICE_SCORING,
    }


def _formal_paths(args: argparse.Namespace) -> Optional[Dict[str, Path]]:
    option_names = (
        "evaluation_attestation",
        "expected_evaluation_attestation_sha256",
        "pre_registration",
        "expected_preregistration_sha256",
        "dataset_manifest",
        "final_candidate_selection",
        "expected_final_candidate_selection_sha256",
    )
    supplied = [getattr(args, name, None) is not None for name in option_names]
    if not any(supplied):
        return None
    if not all(supplied):
        missing = [name for name, present in zip(option_names, supplied) if not present]
        raise ValueError(
            "Formal C-Eval mode requires all attestation options; missing {}".format(
                missing
            )
        )
    return {
        "attestation": resolve_path(args.evaluation_attestation),
        "pre_registration": resolve_path(args.pre_registration),
        "dataset_manifest": resolve_path(args.dataset_manifest),
        "final_candidate_selection": resolve_path(args.final_candidate_selection),
    }


def load_formal_attestation(
    args: argparse.Namespace,
    rows: Sequence[Dict[str, Any]],
    models: Sequence[Tuple[str, str]],
) -> Optional[Dict[str, Any]]:
    paths = _formal_paths(args)
    if paths is None:
        return None
    if args.resume:
        raise ValueError("Formal C-Eval evaluation forbids --resume and result reuse")
    if int(getattr(args, "limit_per_category", 0)) != 0:
        raise ValueError("Formal C-Eval evaluation forbids --limit_per_category")
    if not rows or any(
        row.get("category") != "natural_language_reasoning" for row in rows
    ):
        raise ValueError(
            "Formal C-Eval attestation mode only supports natural_language_reasoning rows"
        )
    for label, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError("Formal {} must be a regular file: {}".format(label, path))

    actual_attestation_sha = sha256_file(paths["attestation"])
    expected_attestation_sha = require_sha256(
        args.expected_evaluation_attestation_sha256,
        "expected_evaluation_attestation_sha256",
    )
    if actual_attestation_sha != expected_attestation_sha:
        raise ValueError("Evaluation attestation does not match its external SHA-256 anchor")
    actual_preregistration_sha = sha256_file(paths["pre_registration"])
    expected_preregistration_sha = require_sha256(
        args.expected_preregistration_sha256,
        "expected_preregistration_sha256",
    )
    if actual_preregistration_sha != expected_preregistration_sha:
        raise ValueError("Pre-registration does not match its external SHA-256 anchor")

    dataset_path = resolve_path(args.dataset_jsonl).resolve()
    actual_dataset_sha = sha256_file(dataset_path)
    actual_manifest_sha = sha256_file(paths["dataset_manifest"])
    actual_final_selection_sha = sha256_file(paths["final_candidate_selection"])
    expected_final_selection_sha = require_sha256(
        args.expected_final_candidate_selection_sha256,
        "expected_final_candidate_selection_sha256",
    )
    if actual_final_selection_sha != expected_final_selection_sha:
        raise ValueError(
            "Final candidate selection does not match its external SHA-256 anchor"
        )
    attestation = read_json_object(paths["attestation"], "evaluation_attestation")
    preregistration = read_json_object(paths["pre_registration"], "pre_registration")
    if attestation.get("schema_version") != ATTESTATION_SCHEMA:
        raise ValueError("Unsupported evaluation attestation schema_version")
    expected_bindings = {
        "pre_registration_sha256": actual_preregistration_sha,
        "dataset_sha256": actual_dataset_sha,
        "dataset_manifest_sha256": actual_manifest_sha,
        "final_candidate_selection_sha256": actual_final_selection_sha,
    }
    for field, expected in expected_bindings.items():
        observed = require_sha256(attestation.get(field), "attestation." + field)
        if observed != expected:
            raise ValueError("Evaluation attestation {} binding mismatch".format(field))

    formal_procedure = preregistration.get("formal_procedure")
    if not isinstance(formal_procedure, dict):
        raise ValueError("Pre-registration lacks formal_procedure")
    if formal_procedure.get("attestation_must_bind_final_selection_sha256") is not True:
        raise ValueError(
            "Pre-registration must require attestation binding to final selection"
        )
    declared_selection = resolve_path(
        str(formal_procedure.get("required_final_selection_record", ""))
    ).resolve()
    if declared_selection != paths["final_candidate_selection"].resolve():
        raise ValueError(
            "Final candidate selection path does not match pre-registration"
        )
    declared_anchor = resolve_path(
        str(
            formal_procedure.get(
                "required_final_selection_external_sha256_anchor", ""
            )
        )
    ).resolve()
    if not declared_anchor.is_file() or declared_anchor.is_symlink():
        raise ValueError(
            "Pre-registered final selection external SHA-256 anchor is missing"
        )
    anchor_tokens = declared_anchor.read_text(encoding="utf-8").split()
    if not anchor_tokens:
        raise ValueError("Final selection external SHA-256 anchor is empty")
    anchored_final_selection_sha = require_sha256(
        anchor_tokens[0], "final_selection_external_sha256_anchor"
    )
    if anchored_final_selection_sha != expected_final_selection_sha:
        raise ValueError(
            "Final selection CLI SHA-256 does not match pre-registered external anchor"
        )

    protocol = nlr_choice_protocol(args)
    if attestation.get("evaluation_protocol") != protocol:
        raise ValueError("Evaluation attestation protocol does not match evaluator runtime")
    if preregistration.get("evaluation_protocol") != protocol:
        raise ValueError("Pre-registration protocol does not match evaluator runtime")

    attested_models = attestation.get("models")
    if not isinstance(attested_models, dict):
        raise ValueError("Evaluation attestation models must be an object")
    actual_models = dict(models)
    if set(attested_models) != set(actual_models):
        raise ValueError("Evaluation attestation model labels do not match --model")
    runtime_digests: Dict[str, str] = {}
    for label, model in models:
        record = attested_models.get(label)
        if not isinstance(record, dict) or record.get("model") != model:
            raise ValueError("Attested model mismatch for {}".format(label))
        manifest = record.get("ollama_manifest")
        if not isinstance(manifest, dict):
            raise ValueError("Attested model {} lacks ollama_manifest".format(label))
        runtime_digests[label] = require_sha256(
            manifest.get("sha256"),
            "attestation.models.{}.ollama_manifest.sha256".format(label),
        )
        declared_runtime_digest = require_sha256(
            record.get("ollama_runtime_digest"),
            "attestation.models.{}.ollama_runtime_digest".format(label),
        )
        if declared_runtime_digest != runtime_digests[label]:
            raise ValueError(
                "Attested Ollama runtime digest must equal manifest SHA for {}".format(
                    label
                )
            )

    return {
        "path": str(paths["attestation"].resolve()),
        "sha256": actual_attestation_sha,
        "pre_registration_path": str(paths["pre_registration"].resolve()),
        "pre_registration_sha256": actual_preregistration_sha,
        "dataset_manifest_path": str(paths["dataset_manifest"].resolve()),
        "dataset_manifest_sha256": actual_manifest_sha,
        "dataset_sha256": actual_dataset_sha,
        "final_candidate_selection_path": str(
            paths["final_candidate_selection"].resolve()
        ),
        "final_candidate_selection_sha256": actual_final_selection_sha,
        "final_candidate_selection_external_anchor_path": str(declared_anchor),
        "final_candidate_selection_external_anchor_sha256": (
            anchored_final_selection_sha
        ),
        "protocol": protocol,
        "runtime_digests": runtime_digests,
    }


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
            NLR_CHOICE_SYSTEM_PROMPT,
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


def load_ollama_model(host: str, model: str, timeout: int = 30) -> None:
    """Load *model* without issuing a scored benchmark prompt.

    Formal evaluation uses this load-only request before consulting ``/api/ps``.
    That makes ``observed_before_sha256`` a genuine pre-inference binding instead
    of a digest observed after the first scored question.
    """

    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "30m",
    }
    request = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def ollama_runtime_digest(host: str, model: str, timeout: int = 30) -> str:
    request = urllib.request.Request(
        host.rstrip("/") + "/api/ps",
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    records = value.get("models") if isinstance(value, dict) else None
    if not isinstance(records, list):
        raise RuntimeError("Ollama /api/ps response lacks models array")
    expected_names = {model}
    if model.endswith(":latest"):
        expected_names.add(model[: -len(":latest")])
    elif ":" not in model.rsplit("/", 1)[-1]:
        expected_names.add(model + ":latest")
    matches = []
    for record in records:
        if not isinstance(record, dict):
            continue
        observed_name = str(record.get("name") or record.get("model") or "")
        if observed_name not in expected_names:
            continue
        matches.append(
            require_sha256(record.get("digest"), "Ollama /api/ps model digest")
        )
    if len(matches) != 1:
        raise RuntimeError(
            "Ollama /api/ps must resolve {} to exactly one loaded model; found {}".format(
                model, len(matches)
            )
        )
    return matches[0]


def verify_ollama_runtime_binding(
    host: str,
    model: str,
    expected_digest: str,
    *,
    timeout: int = 30,
) -> str:
    observed = ollama_runtime_digest(host, model, timeout=timeout)
    if observed != expected_digest:
        raise RuntimeError(
            "Ollama runtime digest mismatch for {}: expected {}, observed {}".format(
                model, expected_digest, observed
            )
        )
    return observed


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
    # Fail closed: the response must end with exactly one numeric answer.
    # Taking the first number from ``FINAL: 90 - 20 = 70`` or the last number
    # from a truncated derivation can incorrectly award a point.
    final_match = re.search(
        r"FINAL\s*:\s*([-+$]?[0-9][0-9,]*(?:\.[0-9]+)?)\s*$",
        text,
        re.I,
    )
    predicted_text = final_match.group(1) if final_match else ""
    prediction = normalize_number(predicted_text)
    expected = normalize_number(reference)
    return {
        "prediction": predicted_text or None,
        "reference": reference,
        "correct": prediction is not None and expected is not None and prediction == expected,
    }


def evaluate_math_legacy(text: str, reference: str) -> Dict[str, Any]:
    """Preserve the historical, permissive scorer used by ordinary runs."""

    final_match = re.search(
        r"FINAL\s*:\s*([-+$]?[0-9][0-9,]*(?:\.[0-9]+)?)", text, re.I
    )
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
    # The prompt contract requires one action token.  Do not award a point by
    # extracting an A-D character from an explanation or a list of options.
    match = re.fullmatch(r"\s*([A-D])\s*", text, re.I)
    prediction = match.group(1) if match else None
    if prediction is not None:
        prediction = prediction.upper()
    return {
        "prediction": prediction,
        "reference": reference,
        "correct": prediction == reference,
    }


def evaluate_choice_legacy(text: str, reference: str) -> Dict[str, Any]:
    """Preserve the historical first-standalone-letter scorer for ordinary runs."""

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


def evaluate_response(
    text: str,
    row: Dict[str, Any],
    code_timeout: float,
    *,
    strict_choice_protocol: bool = False,
) -> Dict[str, Any]:
    category = row["category"]
    if category == "math":
        scorer = evaluate_math if strict_choice_protocol else evaluate_math_legacy
        return scorer(text, str(row["reference_answer"]))
    if category == "code":
        return evaluate_code(text, row, code_timeout)
    scorer = evaluate_choice if strict_choice_protocol else evaluate_choice_legacy
    return scorer(text, str(row["reference_answer"]))


def empty_result(
    args: argparse.Namespace,
    rows: Sequence[Dict[str, Any]],
    formal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = {
        "task": "general_capability_retention",
        "dataset_jsonl": args.dataset_jsonl,
        "num_ctx": args.num_ctx,
        "no_thinking": True,
        "sample_count": len(rows),
        "models": {},
        "retention": {},
    }
    if formal is not None:
        result.update(
            {
                "dataset_jsonl": str(resolve_path(args.dataset_jsonl).resolve()),
                "dataset_sha256": formal["dataset_sha256"],
                "dataset_manifest_sha256": formal["dataset_manifest_sha256"],
                "pre_registration_sha256": formal["pre_registration_sha256"],
                "final_candidate_selection_path": formal[
                    "final_candidate_selection_path"
                ],
                "final_candidate_selection_sha256": formal[
                    "final_candidate_selection_sha256"
                ],
                "evaluation_protocol": dict(formal["protocol"]),
                "evaluation_attestation_sha256": formal["sha256"],
                "evaluator_sha256": sha256_file(Path(__file__).resolve()),
                "formal_execution": {
                    "resume_allowed": False,
                    "preexisting_output_allowed": False,
                    "runtime_model_binding_required": True,
                },
            }
        )
    return result


def load_checkpoint(
    path: Path,
    args: argparse.Namespace,
    rows: Sequence[Dict[str, Any]],
    formal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if args.resume and path.exists():
        with path.open("r", encoding="utf-8") as file_obj:
            value = json.load(file_obj)
        if isinstance(value, dict):
            return value
    return empty_result(args, rows, formal)


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
    formal = load_formal_attestation(args, rows, models)
    if formal is not None and output_path.exists():
        raise ValueError(
            "Formal C-Eval evaluation refuses a pre-existing output file: {}".format(
                output_path
            )
        )
    result = load_checkpoint(output_path, args, rows, formal)

    for label, model in models:
        model_result = result["models"].setdefault(label, {"model": model, "samples": []})
        if model_result.get("model") != model:
            raise ValueError("Checkpoint model mismatch for {}".format(label))
        if formal is not None:
            expected_digest = formal["runtime_digests"][label]
            model_result["runtime_binding"] = {
                "expected_manifest_sha256": expected_digest,
                "observed_before_sha256": None,
                "observed_after_sha256": None,
                "verified": False,
            }
        completed_ids = {sample["sample_id"] for sample in model_result["samples"]}
        try:
            if formal is not None:
                load_ollama_model(
                    args.host,
                    model,
                    timeout=min(args.timeout, 30),
                )
                observed = verify_ollama_runtime_binding(
                    args.host,
                    model,
                    expected_digest,
                    timeout=min(args.timeout, 30),
                )
                model_result["runtime_binding"][
                    "observed_before_sha256"
                ] = observed
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
                evaluation = evaluate_response(
                    response["text"],
                    row,
                    args.code_timeout,
                    strict_choice_protocol=formal is not None,
                )
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
            if formal is not None:
                observed = verify_ollama_runtime_binding(
                    args.host,
                    model,
                    expected_digest,
                    timeout=min(args.timeout, 30),
                )
                model_result["runtime_binding"]["observed_after_sha256"] = observed
                model_result["runtime_binding"]["verified"] = True
                save_json(result, output_path)
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
