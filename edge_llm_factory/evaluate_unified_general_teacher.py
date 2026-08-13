"""在统一开发集上严格评估固定的 Qwen3.5 9B Ollama Teacher。

本入口只接受统一数据构建器 v1/v2 manifest 锁定的
``general_dev_evaluation.jsonl``。对于 U2/v2，它复用候选评测器对独立晋级集
800 条、数学/中文逻辑各 400 条、训练/选择未使用和所有 test/blind 未加载的
严格校验。它不会读取 GSM8K、LogiQA2.0 或交通场景的任何 test/blind 文件。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
import urllib.error
import urllib.request

from edge_llm_factory.contracts import (
    ManifestError,
    canonical_sha256,
    sha256_file,
    write_json_object,
)
from edge_llm_factory.evaluate_unified_general_adapter import (
    GENERAL_CATEGORIES,
    MAX_NEW_TOKENS,
    _canonical_row_sha256,
    _select_rows,
    evaluate_rows,
    read_evaluation_rows,
    summarize,
    validate_dataset_binding,
)


EVALUATION_SCHEMA_VERSION = "edge-llm-unified-general-teacher-dev-evaluation/v1"
TEACHER_MODEL = "qwen3.5:9b"
MODEL_DIGEST_PATTERN = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$")
JsonTransport = Callable[
    [str, str, Optional[Mapping[str, Any]], float], Mapping[str, Any]
]


def _json_request(
    method: str,
    url: str,
    payload: Optional[Mapping[str, Any]],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    data = None
    headers: Dict[str, str] = {}
    if payload is not None:
        data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(1024).decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Ollama 请求失败: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama 返回的不是有效 JSON") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("Ollama JSON 顶层必须是对象")
    return value


def _normalize_endpoint(value: str) -> str:
    endpoint = value.strip().rstrip("/")
    if not endpoint.startswith(("http://", "https://")):
        raise ManifestError("Ollama endpoint 必须是 http/https URL")
    return endpoint


def _model_digest(value: Any) -> str:
    if not isinstance(value, str):
        raise ManifestError("Ollama 模型 digest 缺失")
    match = MODEL_DIGEST_PATTERN.fullmatch(value.strip().lower())
    if match is None:
        raise ManifestError("Ollama 模型 digest 不是 SHA-256")
    return match.group(1)


def attest_teacher_model(
    endpoint: str,
    timeout_seconds: float,
    transport: JsonTransport = _json_request,
    expected_model_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """锁定实际服务中的 qwen3.5:9b digest、show 元数据和运行时版本。"""
    endpoint = _normalize_endpoint(endpoint)
    tags = transport("GET", endpoint + "/api/tags", None, timeout_seconds)
    models = tags.get("models")
    if not isinstance(models, list):
        raise ManifestError("Ollama /api/tags 缺少 models 数组")
    matches = []
    for item in models:
        if not isinstance(item, Mapping):
            continue
        names = {str(item.get("name", "")), str(item.get("model", ""))}
        if TEACHER_MODEL in names:
            matches.append(item)
    if len(matches) != 1:
        raise ManifestError(
            f"Ollama 必须且只能精确解析一个 {TEACHER_MODEL}，实际为 {len(matches)}"
        )
    digest = _model_digest(matches[0].get("digest"))
    if expected_model_sha256 is not None:
        expected = _model_digest(expected_model_sha256)
        if digest != expected:
            raise ManifestError(
                "Ollama qwen3.5:9b 实际 digest 与预注册 SHA-256 不一致"
            )
    show = transport(
        "POST",
        endpoint + "/api/show",
        {"model": TEACHER_MODEL, "verbose": True},
        timeout_seconds,
    )
    version = transport("GET", endpoint + "/api/version", None, timeout_seconds)
    return {
        "provider": "ollama",
        "endpoint": endpoint,
        "model": TEACHER_MODEL,
        "expected_model_sha256": (
            _model_digest(expected_model_sha256)
            if expected_model_sha256 is not None
            else None
        ),
        "model_sha256": digest,
        "show_response_sha256": canonical_sha256(show),
        "version_response": dict(version),
        "version_response_sha256": canonical_sha256(version),
    }


class OllamaTeacherGenerator:
    def __init__(
        self,
        endpoint: str,
        timeout_seconds: float,
        transport: JsonTransport = _json_request,
        num_ctx: int = 2048,
    ) -> None:
        if timeout_seconds <= 0:
            raise ManifestError("timeout_seconds 必须大于 0")
        if num_ctx <= 0:
            raise ManifestError("num_ctx 必须大于 0")
        self.endpoint = _normalize_endpoint(endpoint)
        self.timeout_seconds = float(timeout_seconds)
        self.transport = transport
        self.num_ctx = int(num_ctx)

    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        max_new_tokens: int,
        row: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        expected = MAX_NEW_TOKENS.get(str(row.get("category")))
        if expected != max_new_tokens:
            raise ManifestError("Teacher 生成长度与类别协议不一致")
        # 不 strip、不重写：与候选评测器使用同一份 system/user 字符串。
        payload = {
            "model": TEACHER_MODEL,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "top_p": 1,
                "seed": 42,
                "num_ctx": self.num_ctx,
                "num_predict": max_new_tokens,
            },
        }
        started = time.perf_counter()
        value = self.transport(
            "POST",
            self.endpoint + "/api/chat",
            payload,
            self.timeout_seconds,
        )
        generation_ms = (time.perf_counter() - started) * 1000.0
        if value.get("model") != TEACHER_MODEL:
            raise RuntimeError(
                f"Ollama /api/chat 返回模型必须精确为 {TEACHER_MODEL}"
            )
        if value.get("done") is not True:
            raise RuntimeError("Ollama /api/chat 未声明 done=true")
        message = value.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise RuntimeError("Ollama /api/chat 缺少 message.content")
        prompt_tokens = value.get("prompt_eval_count")
        output_tokens = value.get("eval_count")
        for name, number in (
            ("prompt_eval_count", prompt_tokens),
            ("eval_count", output_tokens),
        ):
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise RuntimeError(f"Ollama {name} 必须是非负整数")
        return {
            "raw_output": message["content"],
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "generation_ms": generation_ms,
        }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def run_evaluation(
    *,
    dataset: Path,
    dataset_manifest: Path,
    samples_output: Path,
    summary_output: Path,
    endpoint: str,
    expected_model_sha256: str,
    timeout_seconds: float,
    categories: set[str],
    limit_per_category: int,
    num_ctx: int,
    transport: JsonTransport = _json_request,
) -> Dict[str, Any]:
    if samples_output.resolve() == summary_output.resolve():
        raise ManifestError("逐题 JSONL 与 summary JSON 不能使用同一路径")
    for path in (samples_output, summary_output):
        if path.exists():
            raise ManifestError(f"拒绝覆盖已有评测结果: {path}")
    data_identity = validate_dataset_binding(dataset, dataset_manifest)
    rows = _select_rows(
        read_evaluation_rows(dataset), categories, limit_per_category
    )
    identity_before = attest_teacher_model(
        endpoint,
        timeout_seconds,
        expected_model_sha256=expected_model_sha256,
        transport=transport,
    )
    generator = OllamaTeacherGenerator(
        endpoint, timeout_seconds, transport=transport, num_ctx=num_ctx
    )
    samples = evaluate_rows(rows, generator)
    metrics = summarize(samples)
    identity_after = attest_teacher_model(
        endpoint,
        timeout_seconds,
        expected_model_sha256=expected_model_sha256,
        transport=transport,
    )
    stable_fields = (
        "provider",
        "endpoint",
        "model",
        "expected_model_sha256",
        "model_sha256",
        "show_response_sha256",
        "version_response_sha256",
    )
    drifted = [
        field
        for field in stable_fields
        if identity_before.get(field) != identity_after.get(field)
    ]
    if drifted:
        raise ManifestError(
            "Ollama Teacher 在评测期间发生模型或运行时漂移: "
            + ", ".join(drifted)
        )
    _write_jsonl(samples_output, samples)
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "task": "qwen3.5_9b_teacher_math_and_chinese_logic_development_evaluation",
        "development_only": True,
        "formal_blind_test_used": False,
        "dataset": data_identity,
        "selection": {
            "categories": sorted(categories),
            "limit_per_category": limit_per_category,
            "sample_count": len(rows),
            "selected_rows_sha256": canonical_sha256(
                [_canonical_row_sha256(row) for row in rows]
            ),
        },
        "model_identity": {
            **identity_before,
            "stable_across_evaluation": True,
            "attestation_before": identity_before,
            "attestation_after": identity_after,
        },
        "evaluation_protocol": {
            "renderer": "ollama_api_chat",
            "uses_each_row_system_prompt_verbatim": True,
            "uses_each_row_prompt_verbatim": True,
            "max_input_tokens": num_ctx,
            "max_new_tokens": dict(MAX_NEW_TOKENS),
            "decoding": {
                "temperature": 0,
                "top_p": 1,
                "seed": 42,
                "thinking": False,
            },
            "math_scoring": "strict_trailing_FINAL_numeric",
            "logic_scoring": "strict_full_output_A_to_D",
        },
        "metrics": metrics,
        "artifacts": {
            "samples": {
                "path": str(samples_output.resolve()),
                "bytes": samples_output.stat().st_size,
                "sha256": sha256_file(samples_output),
            },
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    }
    write_json_object(summary_output, report)
    return report


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument("--samples_output", required=True)
    parser.add_argument("--summary_output", required=True)
    parser.add_argument("--ollama_endpoint", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--expected_model_sha256",
        required=True,
        help="预注册的 qwen3.5:9b Ollama 模型 digest（64位 SHA-256）",
    )
    parser.add_argument("--timeout_seconds", type=float, default=600.0)
    parser.add_argument("--num_ctx", type=int, default=2048)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--limit_per_category", type=int, default=0)
    args = parser.parse_args(argv)
    categories = {value.strip() for value in args.category if value.strip()}
    if not categories:
        categories = set(GENERAL_CATEGORIES)
    unknown = categories - set(GENERAL_CATEGORIES)
    if unknown:
        raise ManifestError(f"未知类别: {sorted(unknown)}")
    if args.limit_per_category < 0:
        raise ManifestError("limit_per_category 不能为负数")
    report = run_evaluation(
        dataset=Path(args.dataset_jsonl).resolve(),
        dataset_manifest=Path(args.dataset_manifest).resolve(),
        samples_output=Path(args.samples_output).resolve(),
        summary_output=Path(args.summary_output).resolve(),
        endpoint=args.ollama_endpoint,
        expected_model_sha256=args.expected_model_sha256,
        timeout_seconds=args.timeout_seconds,
        categories=categories,
        limit_per_category=args.limit_per_category,
        num_ctx=args.num_ctx,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
