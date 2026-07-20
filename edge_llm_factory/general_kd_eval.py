"""用途：安全验收通用 Teacher 的数学、代码和中文选择题输出。"""

import ast
import os
import re
import resource
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Mapping, Optional, Sequence


ALLOWED_IMPORTS = {
    "bisect",
    "collections",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "re",
    "statistics",
    "string",
    "typing",
}

DANGEROUS_CALLS = {
    "__import__",
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
}

TEST_MARKER = "\nThe function must pass these tests:\n"


def assistant_reference(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("通用样本缺少 messages")
    assistant = messages[-1]
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        raise ValueError("通用样本最后一条消息必须是 assistant reference")
    content = str(assistant.get("content", "")).strip()
    if not content:
        raise ValueError("assistant reference 不能为空")
    return content


def user_prompt(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("通用样本缺少 messages")
    users = [message for message in messages if message.get("role") == "user"]
    if len(users) != 1:
        raise ValueError("通用样本必须恰好包含一条 user 消息")
    content = str(users[0].get("content", "")).strip()
    if not content:
        raise ValueError("user prompt 不能为空")
    return content


def normalize_number(value: str) -> Optional[Decimal]:
    clean = value.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return Decimal(clean)
    except InvalidOperation:
        return None


def extract_number(text: str) -> Optional[str]:
    final_match = re.search(
        r"FINAL\s*:\s*([-+$]?[0-9][0-9,]*(?:\.[0-9]+)?)", text, re.IGNORECASE
    )
    matches = re.findall(r"[-+$]?[0-9][0-9,]*(?:\.[0-9]+)?", text)
    if final_match:
        return final_match.group(1)
    return matches[-1] if matches else None


def evaluate_math(text: str, reference: str) -> Dict[str, Any]:
    predicted_text = extract_number(text)
    expected_text = extract_number(reference)
    prediction = normalize_number(predicted_text or "")
    expected = normalize_number(expected_text or "")
    return {
        "prediction": predicted_text,
        "reference": expected_text,
        "correct": prediction is not None and expected is not None and prediction == expected,
    }


def evaluate_choice(text: str, reference: str) -> Dict[str, Any]:
    predicted_match = re.search(r"(?<![A-Z])([A-D])(?![A-Z])", text.upper())
    reference_match = re.search(r"(?<![A-Z])([A-D])(?![A-Z])", reference.upper())
    prediction = predicted_match.group(1) if predicted_match else None
    expected = reference_match.group(1) if reference_match else None
    return {
        "prediction": prediction,
        "reference": expected,
        "correct": prediction is not None and prediction == expected,
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


def _code_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def code_tests(prompt: str) -> Sequence[str]:
    if TEST_MARKER not in prompt:
        return []
    raw = prompt.split(TEST_MARKER, 1)[1]
    return [line.strip() for line in raw.splitlines() if line.strip()]


def evaluate_code(text: str, prompt: str, timeout: float = 3.0) -> Dict[str, Any]:
    code = extract_code(text)
    tests = code_tests(prompt)
    if not tests:
        return {
            "prediction": code,
            "reference": "embedded MBPP tests",
            "correct": False,
            "execution_error": "missing_tests",
        }
    program = "{}\n{}\n".format(code, "\n".join(tests))
    validation_error = validate_code_ast(program)
    if validation_error:
        return {
            "prediction": code,
            "reference": "embedded MBPP tests",
            "correct": False,
            "execution_error": validation_error,
        }
    try:
        with tempfile.TemporaryDirectory(prefix="general_kd_code_") as directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-"],
                input=program,
                text=True,
                cwd=directory,
                capture_output=True,
                timeout=timeout,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
                preexec_fn=_code_limits,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return {
            "prediction": code,
            "reference": "embedded MBPP tests",
            "correct": False,
            "execution_error": "timeout",
        }
    return {
        "prediction": code,
        "reference": "embedded MBPP tests",
        "correct": completed.returncode == 0,
        "execution_error": None if completed.returncode == 0 else completed.stderr[-1000:],
    }


def evaluate_teacher_output(
    text: str, row: Mapping[str, Any], code_timeout: float = 3.0
) -> Dict[str, Any]:
    category = str(row.get("category", ""))
    reference = assistant_reference(row)
    if category == "math":
        return evaluate_math(text, reference)
    if category == "code":
        return evaluate_code(text, user_prompt(row), timeout=code_timeout)
    if category == "natural_language_reasoning":
        return evaluate_choice(text, reference)
    raise ValueError("不支持的通用任务类别: {}".format(category))


def canonical_teacher_target(text: str, evaluation: Mapping[str, Any], category: str) -> str:
    if not evaluation.get("correct"):
        raise ValueError("只能规范化已通过验收的 Teacher 输出")
    if category == "code":
        return extract_code(text)
    if category == "natural_language_reasoning":
        return str(evaluation["prediction"])
    prediction = str(evaluation["prediction"])
    stripped = text.strip()
    if re.search(r"FINAL\s*:", stripped, re.IGNORECASE):
        return stripped
    return "{}\nFINAL: {}".format(stripped, prediction).strip()
