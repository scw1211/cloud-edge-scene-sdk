"""只读汇总冻结 C-Eval 指定专项的 Teacher/候选模型终测结果。

本模块不调用模型，也不修改评测结果、冻结题集或其清单。它重新计算
严格单 token 准确率，校验冻结资产哈希和逐样本配对关系，并把可机器
验证的预注册门禁写入一个新的汇总文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from edge_llm_factory.contracts import ManifestError, read_json_object, sha256_file


SCHEMA_VERSION = "edge-llm-ceval-specialty-summary/v1"
STRICT_CHOICE = re.compile(r"\s*([A-D])\s*", re.IGNORECASE)
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
POLITICS_CIVICS_SUBJECTS = (
    "high_school_politics",
    "mao_zedong_thought",
    "middle_school_politics",
)
POLITICS_CIVICS_SAMPLES_PER_SUBJECT = 20


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestError(
                        f"冻结题集 {path}:{line_number} JSON 无效"
                    ) from exc
                if not isinstance(value, dict):
                    raise ManifestError(
                        f"冻结题集 {path}:{line_number} 必须是 JSON object"
                    )
                rows.append(value)
    except OSError as exc:
        raise ManifestError(f"无法读取冻结题集 {path}: {exc}") from exc
    if not rows:
        raise ManifestError(f"冻结题集为空: {path}")
    return rows


def _require_object(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{field} 必须是 object")
    return dict(value)


def _require_list(value: Any, field: str) -> List[Any]:
    if not isinstance(value, list):
        raise ManifestError(f"{field} 必须是 array")
    return list(value)


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} 必须是非空字符串")
    return value.strip()


def _require_integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError(f"{field} 必须是大于等于 {minimum} 的整数")
    return value


def _require_probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{field} 必须是数字")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ManifestError(f"{field} 必须在 [0, 1] 内")
    return result


def _require_sha256(value: Any, field: str) -> str:
    result = _require_text(value, field).lower()
    if SHA256_HEX.fullmatch(result) is None:
        raise ManifestError(f"{field} 必须是 64 位 SHA-256")
    return result


def _digest(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _subject(row: Mapping[str, Any], location: str) -> str:
    source = _require_object(row.get("source"), f"{location}.source")
    return _require_text(source.get("config"), f"{location}.source.config")


def _resolve_evidence_path(value: Any, evidence_path: Path, field: str) -> Path:
    raw = Path(_require_text(value, field))
    return raw.resolve() if raw.is_absolute() else (evidence_path.parent / raw).resolve()


def _read_sha256_anchor(path: Path, field: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise ManifestError(f"{field} 不是普通文件: {path}")
    try:
        tokens = path.read_text(encoding="utf-8").split()
    except OSError as exc:
        raise ManifestError(f"无法读取 {field}: {path}") from exc
    if not tokens:
        raise ManifestError(f"{field} 为空")
    return _require_sha256(tokens[0], field)


def _load_final_candidate_selection(
    *,
    selection_path: Path,
    expected_selection_sha256: str,
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    procedure = _require_object(
        preregistration.get("formal_procedure"),
        "pre_registration.formal_procedure",
    )
    if procedure.get("attestation_must_bind_final_selection_sha256") is not True:
        raise ManifestError(
            "预注册未要求 attestation 绑定 final candidate selection SHA-256"
        )
    declared_selection = _resolve_evidence_path(
        procedure.get("required_final_selection_record"),
        preregistration_path,
        "pre_registration.formal_procedure.required_final_selection_record",
    )
    actual_selection_path = selection_path.resolve()
    if declared_selection != actual_selection_path:
        raise ManifestError("最终候选选择文件路径与预注册不一致")
    if not actual_selection_path.is_file() or actual_selection_path.is_symlink():
        raise ManifestError(f"最终候选选择文件不是普通文件: {actual_selection_path}")

    expected_sha = _require_sha256(
        expected_selection_sha256,
        "expected_final_candidate_selection_sha256",
    )
    actual_sha = sha256_file(actual_selection_path)
    if actual_sha != expected_sha:
        raise ManifestError("最终候选选择文件 SHA-256 与外部锚点不一致")

    anchor_path = _resolve_evidence_path(
        procedure.get("required_final_selection_external_sha256_anchor"),
        preregistration_path,
        "pre_registration.formal_procedure.required_final_selection_external_sha256_anchor",
    )
    anchored_sha = _read_sha256_anchor(
        anchor_path, "最终候选选择文件外部 SHA-256 锚点"
    )
    if anchored_sha != expected_sha:
        raise ManifestError("最终候选选择文件 CLI SHA-256 与预注册外部锚不一致")
    read_json_object(actual_selection_path)
    return (
        {
            "path": str(actual_selection_path),
            "sha256": actual_sha,
            "expected_sha256_anchor": expected_sha,
            "external_anchor_path": str(anchor_path),
            "external_anchor_sha256": anchored_sha,
        },
        {
            "final_candidate_selection_hash_matches_external_anchor": True,
            "final_candidate_selection_path_matches_pre_registration": True,
            "pre_registered_external_anchor_file_matches_cli_anchor": True,
        },
    )


def _verify_file_record(
    value: Any,
    *,
    evidence_path: Path,
    field: str,
) -> Dict[str, Any]:
    record = _require_object(value, field)
    path = _resolve_evidence_path(record.get("path"), evidence_path, f"{field}.path")
    expected_sha = _require_sha256(record.get("sha256"), f"{field}.sha256")
    if not path.is_file() or path.is_symlink():
        raise ManifestError(f"{field} 不是普通文件: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ManifestError(f"{field} 实际 SHA-256 与取证清单不一致")
    return {
        "path": str(path),
        "sha256": actual_sha,
        "bytes": path.stat().st_size,
    }


def _ollama_layer_digests(value: Mapping[str, Any], field: str) -> Tuple[str, ...]:
    layers = _require_list(value.get("layers"), f"{field}.layers")
    digests: List[str] = []
    for index, raw in enumerate(layers):
        layer = _require_object(raw, f"{field}.layers[{index}]")
        digest = _require_text(
            layer.get("digest"), f"{field}.layers[{index}].digest"
        ).lower().removeprefix("sha256:")
        digests.append(
            _require_sha256(digest, f"{field}.layers[{index}].digest")
        )
    if not digests:
        raise ManifestError(f"{field}.layers 不能为空")
    return tuple(digests)


def _release_record(value: Mapping[str, Any], field: str) -> Dict[str, Any]:
    active_release_id = value.get("active_release_id")
    if active_release_id is not None:
        release_id = _require_text(active_release_id, f"{field}.active_release_id")
        releases = _require_object(value.get("releases"), f"{field}.releases")
        return _require_object(
            releases.get(release_id), f"{field}.releases.{release_id}"
        )
    return dict(value)


def _release_bindings(value: Mapping[str, Any], field: str) -> Dict[str, str]:
    record = _release_record(value, field)
    artifact = _require_object(
        record.get("deployment_artifact"), f"{field}.deployment_artifact"
    )
    return {
        "deployment_sha256": _require_sha256(
            artifact.get("sha256"), f"{field}.deployment_artifact.sha256"
        ),
    }


def _verify_runtime_binding(
    model_result: Mapping[str, Any], expected_manifest_sha256: str, field: str
) -> Dict[str, Any]:
    binding = _require_object(model_result.get("runtime_binding"), field)
    values = {
        name: _require_sha256(binding.get(name), f"{field}.{name}")
        for name in (
            "expected_manifest_sha256",
            "observed_before_sha256",
            "observed_after_sha256",
        )
    }
    if binding.get("verified") is not True:
        raise ManifestError(f"{field}.verified 必须为 true")
    if any(value != expected_manifest_sha256 for value in values.values()):
        raise ManifestError(f"{field} 未证明 Ollama 实际运行模型与取证 manifest 一致")
    return {**values, "verified": True}


def _validate_protocol(value: Any, field: str) -> Dict[str, Any]:
    protocol = _require_object(value, field)
    required = {
        "backend",
        "endpoint",
        "stream",
        "think",
        "keep_alive",
        "system_prompt",
        "temperature",
        "top_p",
        "seed",
        "num_ctx",
        "num_predict",
        "timeout_seconds",
        "scoring",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ManifestError(f"{field} 缺少字段: {missing}")
    if protocol.get("stream") is not False or protocol.get("think") is not False:
        raise ManifestError(f"{field} 必须关闭 stream 与 thinking")
    _require_text(protocol.get("backend"), f"{field}.backend")
    _require_text(protocol.get("endpoint"), f"{field}.endpoint")
    _require_text(protocol.get("keep_alive"), f"{field}.keep_alive")
    _require_text(protocol.get("system_prompt"), f"{field}.system_prompt")
    _require_text(protocol.get("scoring"), f"{field}.scoring")
    _require_probability(protocol.get("temperature"), f"{field}.temperature")
    _require_probability(protocol.get("top_p"), f"{field}.top_p")
    _require_integer(protocol.get("seed"), f"{field}.seed")
    _require_integer(protocol.get("num_ctx"), f"{field}.num_ctx", minimum=1)
    _require_integer(protocol.get("num_predict"), f"{field}.num_predict", minimum=1)
    _require_integer(
        protocol.get("timeout_seconds"), f"{field}.timeout_seconds", minimum=1
    )
    return protocol


def _load_evaluation_attestation(
    *,
    attestation_path: Path,
    expected_attestation_sha256: str,
    actual_preregistration_sha256: str,
    actual_dataset_sha256: str,
    actual_dataset_manifest_sha256: str,
    actual_final_candidate_selection_sha256: str,
    preregistration: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    teacher_label: str,
    candidate_label: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    expected_attestation_sha256 = _require_sha256(
        expected_attestation_sha256,
        "expected_evaluation_attestation_sha256",
    )
    actual_attestation_sha = sha256_file(attestation_path)
    if actual_attestation_sha != expected_attestation_sha256:
        raise ManifestError("评测取证清单 SHA-256 与终测前外部锚点不一致")
    attestation = read_json_object(attestation_path)
    if (
        attestation.get("schema_version")
        != "edge-llm-ceval-specialty-evaluation-attestation/v1"
    ):
        raise ManifestError("评测取证清单 schema_version 不受支持")
    if _require_sha256(
        attestation.get("pre_registration_sha256"),
        "evaluation_attestation.pre_registration_sha256",
    ) != actual_preregistration_sha256:
        raise ManifestError("评测取证清单未绑定本次预注册")
    if _require_sha256(
        attestation.get("dataset_sha256"),
        "evaluation_attestation.dataset_sha256",
    ) != actual_dataset_sha256:
        raise ManifestError("评测取证清单未绑定本次冻结题集")
    if _require_sha256(
        attestation.get("dataset_manifest_sha256"),
        "evaluation_attestation.dataset_manifest_sha256",
    ) != actual_dataset_manifest_sha256:
        raise ManifestError("评测取证清单未绑定本次题集 manifest")
    if _require_sha256(
        attestation.get("final_candidate_selection_sha256"),
        "evaluation_attestation.final_candidate_selection_sha256",
    ) != actual_final_candidate_selection_sha256:
        raise ManifestError("评测取证清单未绑定最终候选选择文件")
    evaluation_bindings = {
        "pre_registration_sha256": actual_preregistration_sha256,
        "dataset_sha256": actual_dataset_sha256,
        "dataset_manifest_sha256": actual_dataset_manifest_sha256,
        "final_candidate_selection_sha256": (
            actual_final_candidate_selection_sha256
        ),
    }
    for name, expected in evaluation_bindings.items():
        if _require_sha256(
            evaluation.get(name), f"evaluation.{name}"
        ) != expected:
            raise ManifestError(f"评测结果 {name} 未绑定本次冻结输入")

    preregistered_protocol = _validate_protocol(
        preregistration.get("evaluation_protocol"),
        "pre_registration.evaluation_protocol",
    )
    attested_protocol = _validate_protocol(
        attestation.get("evaluation_protocol"),
        "evaluation_attestation.evaluation_protocol",
    )
    evaluated_protocol = _validate_protocol(
        evaluation.get("evaluation_protocol"),
        "evaluation.evaluation_protocol",
    )
    if attested_protocol != preregistered_protocol:
        raise ManifestError("评测取证清单的完整协议与预注册不一致")
    if evaluated_protocol != preregistered_protocol:
        raise ManifestError("评测结果记录的完整协议与预注册不一致")
    if evaluation.get("evaluation_attestation_sha256") != actual_attestation_sha:
        raise ManifestError("评测结果未绑定本次评测取证清单")

    evaluator = _verify_file_record(
        attestation.get("evaluator"),
        evidence_path=attestation_path,
        field="evaluation_attestation.evaluator",
    )
    if evaluation.get("evaluator_sha256") != evaluator["sha256"]:
        raise ManifestError("评测结果未绑定取证清单指定的评估器源码")

    evaluation_models = _require_object(evaluation.get("models"), "evaluation.models")
    attested_models = _require_object(
        attestation.get("models"), "evaluation_attestation.models"
    )
    if set(attested_models) != {teacher_label, candidate_label}:
        raise ManifestError("评测取证清单必须且只能包含指定 Teacher 与候选模型")
    teacher_result = _require_object(
        evaluation_models.get(teacher_label), f"evaluation.models.{teacher_label}"
    )
    candidate_result = _require_object(
        evaluation_models.get(candidate_label), f"evaluation.models.{candidate_label}"
    )
    teacher_record = _require_object(
        attested_models.get(teacher_label),
        f"evaluation_attestation.models.{teacher_label}",
    )
    candidate_record = _require_object(
        attested_models.get(candidate_label),
        f"evaluation_attestation.models.{candidate_label}",
    )
    teacher_model = _require_text(
        teacher_result.get("model"), f"evaluation.models.{teacher_label}.model"
    )
    candidate_model = _require_text(
        candidate_result.get("model"), f"evaluation.models.{candidate_label}.model"
    )
    if teacher_model == candidate_model:
        raise ManifestError("候选模型不能与 Teacher 使用同一模型名")
    if teacher_record.get("model") != teacher_model:
        raise ManifestError("Teacher 评测模型名与取证清单不一致")
    if candidate_record.get("model") != candidate_model:
        raise ManifestError("候选评测模型名与取证清单不一致")

    teacher_manifest = _verify_file_record(
        teacher_record.get("ollama_manifest"),
        evidence_path=attestation_path,
        field=f"evaluation_attestation.models.{teacher_label}.ollama_manifest",
    )
    teacher_blob = _verify_file_record(
        teacher_record.get("model_blob"),
        evidence_path=attestation_path,
        field=f"evaluation_attestation.models.{teacher_label}.model_blob",
    )
    candidate_manifest = _verify_file_record(
        candidate_record.get("ollama_manifest"),
        evidence_path=attestation_path,
        field=f"evaluation_attestation.models.{candidate_label}.ollama_manifest",
    )
    candidate_gguf = _verify_file_record(
        candidate_record.get("gguf"),
        evidence_path=attestation_path,
        field=f"evaluation_attestation.models.{candidate_label}.gguf",
    )
    candidate_release = _verify_file_record(
        candidate_record.get("release_manifest"),
        evidence_path=attestation_path,
        field=f"evaluation_attestation.models.{candidate_label}.release_manifest",
    )
    preregistered_teacher = _require_object(
        preregistration.get("teacher"), "pre_registration.teacher"
    )
    if teacher_manifest["sha256"] != _require_sha256(
        preregistered_teacher.get("manifest_sha256"),
        "pre_registration.teacher.manifest_sha256",
    ):
        raise ManifestError("Teacher Ollama manifest SHA-256 与预注册不一致")
    if teacher_blob["sha256"] != _require_sha256(
        preregistered_teacher.get("model_blob_sha256"),
        "pre_registration.teacher.model_blob_sha256",
    ):
        raise ManifestError("Teacher blob SHA-256 与预注册不一致")
    teacher_manifest_json = read_json_object(Path(teacher_manifest["path"]))
    teacher_layer_digests = _ollama_layer_digests(
        teacher_manifest_json, "teacher_ollama_manifest"
    )
    if teacher_blob["sha256"] not in teacher_layer_digests:
        raise ManifestError("Teacher Ollama manifest 未引用所取证的模型 blob")
    candidate_manifest_json = read_json_object(Path(candidate_manifest["path"]))
    candidate_layer_digests = _ollama_layer_digests(
        candidate_manifest_json, "candidate_ollama_manifest"
    )
    if candidate_gguf["sha256"] not in candidate_layer_digests:
        raise ManifestError("候选 Ollama manifest 未引用所取证的 GGUF")
    candidate_release_json = read_json_object(Path(candidate_release["path"]))
    candidate_release_bindings = _release_bindings(
        candidate_release_json, "candidate_release_manifest"
    )
    if candidate_release_bindings["deployment_sha256"] != candidate_gguf["sha256"]:
        raise ManifestError("候选 release manifest 未引用所取证的 GGUF")
    if candidate_gguf["sha256"] == teacher_blob["sha256"]:
        raise ManifestError("候选 GGUF 不能与 Teacher 模型 blob 相同")

    teacher_runtime_digest = _require_sha256(
        teacher_record.get("ollama_runtime_digest"),
        f"evaluation_attestation.models.{teacher_label}.ollama_runtime_digest",
    )
    candidate_runtime_digest = _require_sha256(
        candidate_record.get("ollama_runtime_digest"),
        f"evaluation_attestation.models.{candidate_label}.ollama_runtime_digest",
    )
    if teacher_runtime_digest != teacher_manifest["sha256"]:
        raise ManifestError("Teacher 预期运行时 digest 与 Ollama manifest SHA 不一致")
    if candidate_runtime_digest != candidate_manifest["sha256"]:
        raise ManifestError("候选预期运行时 digest 与 Ollama manifest SHA 不一致")
    teacher_runtime_binding = _verify_runtime_binding(
        teacher_result,
        teacher_runtime_digest,
        f"evaluation.models.{teacher_label}.runtime_binding",
    )
    candidate_runtime_binding = _verify_runtime_binding(
        candidate_result,
        candidate_runtime_digest,
        f"evaluation.models.{candidate_label}.runtime_binding",
    )

    provenance = {
        "path": str(attestation_path),
        "sha256": actual_attestation_sha,
        "expected_sha256_anchor": expected_attestation_sha256,
        "evaluator": evaluator,
        "teacher": {
            "model": teacher_model,
            "ollama_manifest": teacher_manifest,
            "model_blob": teacher_blob,
            "runtime_binding": teacher_runtime_binding,
        },
        "candidate": {
            "model": candidate_model,
            "ollama_manifest": candidate_manifest,
            "gguf": candidate_gguf,
            "release_manifest": candidate_release,
            "runtime_binding": candidate_runtime_binding,
        },
        "evaluation_protocol": preregistered_protocol,
    }
    validation = {
        "attestation_hash_matches_external_anchor": True,
        "attestation_binds_pre_registration": True,
        "attestation_binds_dataset_and_manifest": True,
        "attestation_binds_final_candidate_selection": True,
        "evaluation_binds_final_candidate_selection": True,
        "complete_protocol_matches_pre_registration": True,
        "teacher_model_name_and_assets_bound": True,
        "candidate_model_name_and_assets_bound": True,
        "teacher_candidate_model_names_distinct": True,
        "teacher_candidate_model_assets_distinct": True,
        "teacher_manifest_references_blob": True,
        "candidate_release_manifest_references_gguf": True,
        "candidate_ollama_manifest_references_gguf": True,
        "evaluator_source_hash_bound": True,
        "teacher_runtime_model_bound_to_manifest": True,
        "candidate_runtime_model_bound_to_manifest": True,
    }
    return provenance, validation


def _load_frozen_dataset(
    dataset_path: Path,
    manifest_path: Path,
    preregistration_path: Path,
    expected_preregistration_sha256: str,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Tuple[str, ...],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
]:
    expected_preregistration_sha256 = _require_sha256(
        expected_preregistration_sha256,
        "expected_pre_registration_sha256",
    )
    actual_preregistration_sha = sha256_file(preregistration_path)
    if actual_preregistration_sha != expected_preregistration_sha256:
        raise ManifestError("预注册 SHA-256 与终测前外部锚点不一致")
    manifest = read_json_object(manifest_path)
    preregistration = read_json_object(preregistration_path)
    formal = _require_object(
        preregistration.get("formal_evaluation"),
        "pre_registration.formal_evaluation",
    )
    expected_dataset_sha = _require_text(
        formal.get("sha256"), "pre_registration.formal_evaluation.sha256"
    )
    expected_manifest_sha = _require_text(
        formal.get("manifest_sha256"),
        "pre_registration.formal_evaluation.manifest_sha256",
    )
    actual_dataset_sha = sha256_file(dataset_path)
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_dataset_sha != expected_dataset_sha:
        raise ManifestError("冻结题集 SHA-256 与预注册不一致")
    if actual_manifest_sha != expected_manifest_sha:
        raise ManifestError("冻结题集 manifest SHA-256 与预注册不一致")

    artifact = _require_object(manifest.get("artifact"), "dataset_manifest.artifact")
    artifact_sha = _require_text(
        artifact.get("sha256"), "dataset_manifest.artifact.sha256"
    )
    if artifact_sha != actual_dataset_sha:
        raise ManifestError("冻结题集 SHA-256 与 manifest 不一致")

    expected_count = _require_integer(
        formal.get("sample_count"),
        "pre_registration.formal_evaluation.sample_count",
        minimum=1,
    )
    artifact_rows = _require_integer(
        artifact.get("rows"), "dataset_manifest.artifact.rows", minimum=1
    )
    if artifact_rows != expected_count:
        raise ManifestError("冻结题集行数在 manifest 与预注册之间不一致")

    rows = _read_jsonl(dataset_path)
    if len(rows) != expected_count:
        raise ManifestError(
            f"冻结题集必须包含 {expected_count} 题，实际为 {len(rows)}"
        )
    by_id: Dict[str, Dict[str, Any]] = {}
    subject_counts: Counter[str] = Counter()
    prompt_fingerprints: List[str] = []
    for index, row in enumerate(rows):
        location = f"dataset[{index}]"
        sample_id = _require_text(row.get("sample_id"), f"{location}.sample_id")
        if sample_id in by_id:
            raise ManifestError(f"冻结题集 sample_id 重复: {sample_id}")
        reference = _require_text(
            row.get("reference_answer"), f"{location}.reference_answer"
        ).upper()
        if reference not in {"A", "B", "C", "D"}:
            raise ManifestError(f"{location}.reference_answer 必须是 A-D")
        if row.get("benchmark") != "ceval":
            raise ManifestError(f"{location}.benchmark 必须是 ceval")
        if row.get("category") != "natural_language_reasoning":
            raise ManifestError(
                f"{location}.category 必须是 natural_language_reasoning"
            )
        source = _require_object(row.get("source"), f"{location}.source")
        subject = _require_text(source.get("config"), f"{location}.source.config")
        if source.get("split") != "test":
            raise ManifestError(f"{location}.source.split 必须是 test")
        prompt_fingerprint = _require_text(
            row.get("prompt_fingerprint"), f"{location}.prompt_fingerprint"
        )
        normalized = dict(row)
        normalized["reference_answer"] = reference
        normalized["subject"] = subject
        by_id[sample_id] = normalized
        subject_counts[subject] += 1
        prompt_fingerprints.append(prompt_fingerprint)

    if len(set(prompt_fingerprints)) != len(prompt_fingerprints):
        raise ManifestError("冻结题集 prompt_fingerprint 存在重复")

    expected_subject_count = _require_integer(
        formal.get("subject_count"),
        "pre_registration.formal_evaluation.subject_count",
        minimum=1,
    )
    expected_per_subject = _require_integer(
        formal.get("samples_per_subject"),
        "pre_registration.formal_evaluation.samples_per_subject",
        minimum=1,
    )
    if len(subject_counts) != expected_subject_count:
        raise ManifestError(
            f"冻结题集必须含 {expected_subject_count} 个学科，实际为 {len(subject_counts)}"
        )
    if any(count != expected_per_subject for count in subject_counts.values()):
        raise ManifestError(
            f"冻结题集每科必须为 {expected_per_subject} 题: {dict(subject_counts)}"
        )

    scope = _require_object(preregistration.get("scope"), "pre_registration.scope")
    declared_subjects = tuple(
        sorted(
            _require_text(value, "pre_registration.subjects[]")
            for value in _require_list(
                preregistration.get("subjects"), "pre_registration.subjects"
            )
        )
    )
    actual_subjects = tuple(sorted(subject_counts))
    if len(set(declared_subjects)) != len(declared_subjects):
        raise ManifestError("预注册 subjects 存在重复")
    if declared_subjects != actual_subjects:
        raise ManifestError("冻结题集学科集合与预注册不一致")
    allowed_scope_categories = {
        "natural_language_reasoning.social_science",
        "natural_language_reasoning.politics_civics",
    }
    if scope.get("category") not in allowed_scope_categories:
        raise ManifestError("预注册 scope.category 不是允许的 C-Eval 专项")
    if scope.get("category") == "natural_language_reasoning.politics_civics":
        expected_politics_subjects = tuple(sorted(POLITICS_CIVICS_SUBJECTS))
        if expected_subject_count != len(expected_politics_subjects):
            raise ManifestError("politics_civics 必须精确包含 3 个预注册学科")
        if expected_per_subject != POLITICS_CIVICS_SAMPLES_PER_SUBJECT:
            raise ManifestError("politics_civics 必须每科精确包含 20 道正式题")
        if actual_subjects != expected_politics_subjects:
            raise ManifestError(
                "politics_civics 学科集合必须精确为 high_school_politics、"
                "mao_zedong_thought、middle_school_politics"
            )
        for sample_id, row in by_id.items():
            subject = row["subject"]
            expected_id = re.compile(
                rf"ceval_{re.escape(subject)}_test_[0-9]+"
            )
            if expected_id.fullmatch(sample_id) is None:
                raise ManifestError(
                    f"politics_civics sample_id 与 source.config 不一致: {sample_id}"
                )

    selection = _require_object(
        manifest.get("selection"), "dataset_manifest.selection"
    )
    manifest_subject_counts = _require_object(
        selection.get("selected_subject_counts"),
        "dataset_manifest.selection.selected_subject_counts",
    )
    normalized_manifest_subject_counts: Dict[str, int] = {}
    for subject, count in manifest_subject_counts.items():
        normalized_manifest_subject_counts[str(subject)] = _require_integer(
            count,
            f"dataset_manifest.selection.selected_subject_counts.{subject}",
            minimum=1,
        )
    if normalized_manifest_subject_counts != dict(sorted(subject_counts.items())):
        raise ManifestError("冻结题集逐科学科数与 manifest 不一致")
    if selection.get("selected_rows") != expected_count:
        raise ManifestError("冻结题集 selected_rows 与预注册不一致")
    if selection.get("selected_sample_ids_sha256") != _digest(by_id):
        raise ManifestError("冻结题集 sample_id 摘要与 manifest 不一致")
    if selection.get("selected_prompt_fingerprints_sha256") != _digest(
        prompt_fingerprints
    ):
        raise ManifestError("冻结题集 prompt 摘要与 manifest 不一致")

    provenance = {
        "dataset": {
            "path": str(dataset_path),
            "sha256": actual_dataset_sha,
            "rows": len(rows),
        },
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": actual_manifest_sha,
        },
        "pre_registration": {
            "path": str(preregistration_path),
            "sha256": actual_preregistration_sha,
            "expected_sha256_anchor": expected_preregistration_sha256,
        },
    }
    validation = {
        "dataset_hash_matches_manifest": True,
        "dataset_hash_matches_pre_registration": True,
        "manifest_hash_matches_pre_registration": True,
        "pre_registration_hash_matches_external_anchor": True,
        "sample_count": len(rows),
        "subject_count": len(subject_counts),
        "samples_per_subject": expected_per_subject,
        "sample_ids_unique": True,
        "subject_counts": dict(sorted(subject_counts.items())),
        "sample_id_digest_matches_manifest": True,
        "prompt_fingerprint_digest_matches_manifest": True,
        "prompt_fingerprints_unique": True,
        "all_source_splits_are_test": True,
    }
    return by_id, actual_subjects, preregistration, provenance, validation


def _strict_prediction(raw_output: Any) -> Optional[str]:
    if not isinstance(raw_output, str):
        return None
    match = STRICT_CHOICE.fullmatch(raw_output)
    return match.group(1).upper() if match else None


def _validated_model_samples(
    evaluation: Mapping[str, Any],
    label: str,
    dataset: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    models = _require_object(evaluation.get("models"), "evaluation.models")
    model_result = _require_object(models.get(label), f"evaluation.models.{label}")
    samples = _require_list(
        model_result.get("samples"), f"evaluation.models.{label}.samples"
    )
    by_id: Dict[str, Dict[str, Any]] = {}
    for index, raw_sample in enumerate(samples):
        sample = _require_object(raw_sample, f"evaluation.models.{label}.samples[{index}]")
        sample_id = _require_text(
            sample.get("sample_id"),
            f"evaluation.models.{label}.samples[{index}].sample_id",
        )
        if sample_id in by_id:
            raise ManifestError(f"{label} sample_id 重复: {sample_id}")
        frozen = dataset.get(sample_id)
        if frozen is None:
            raise ManifestError(f"{label} 包含冻结题集外 sample_id: {sample_id}")
        if sample.get("benchmark") != frozen.get("benchmark"):
            raise ManifestError(f"{label}/{sample_id} benchmark 与冻结题集不一致")
        if sample.get("category") != frozen.get("category"):
            raise ManifestError(f"{label}/{sample_id} category 与冻结题集不一致")
        reference = _require_text(
            sample.get("reference"), f"{label}/{sample_id}.reference"
        ).upper()
        if reference != frozen["reference_answer"]:
            raise ManifestError(f"{label}/{sample_id} reference 与冻结题集不一致")
        prediction = _strict_prediction(sample.get("raw_output"))
        recomputed_correct = prediction == reference
        if sample.get("prediction") != prediction:
            raise ManifestError(f"{label}/{sample_id} prediction 与严格重算不一致")
        if sample.get("correct") is not recomputed_correct:
            raise ManifestError(f"{label}/{sample_id} correct 与严格重算不一致")
        execution_error = sample.get("execution_error")
        by_id[sample_id] = {
            "sample_id": sample_id,
            "subject": frozen["subject"],
            "prediction": prediction,
            "reference": reference,
            "strict_output": prediction is not None,
            "correct": recomputed_correct,
            "execution_error": execution_error
            if execution_error not in (None, "")
            else None,
        }
    expected_ids = set(dataset)
    actual_ids = set(by_id)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ManifestError(
            f"{label} 未完成冻结题集全集: missing={missing[:5]}, extra={extra[:5]}"
        )
    return by_id


def _accuracy_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ManifestError("无法汇总空样本")
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    strict = sum(bool(row["strict_output"]) for row in rows)
    errors = sum(row.get("execution_error") is not None for row in rows)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 6),
        "strict_output_count": strict,
        "strict_output_rate": round(strict / total, 6),
        "execution_error_count": errors,
    }


def _model_metrics(
    rows_by_id: Mapping[str, Mapping[str, Any]], subjects: Sequence[str]
) -> Dict[str, Any]:
    rows = [rows_by_id[sample_id] for sample_id in sorted(rows_by_id)]
    by_subject: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject"])].append(row)
    subject_metrics = {
        subject: _accuracy_metrics(by_subject[subject]) for subject in subjects
    }
    return {
        "overall": _accuracy_metrics(rows),
        "subject_macro_accuracy": round(
            statistics.fmean(
                subject_metrics[subject]["accuracy"] for subject in subjects
            ),
            6,
        ),
        "subjects": subject_metrics,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ManifestError("无法对空 bootstrap 分布计算分位数")
    if not 0.0 <= probability <= 1.0:
        raise ManifestError("percentile probability 必须在 [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stratified_paired_retention_bootstrap(
    teacher: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    subjects: Sequence[str],
    *,
    iterations: int = 10000,
    seed: int = 20260813,
) -> Dict[str, Any]:
    if iterations <= 0:
        raise ManifestError("bootstrap iterations 必须大于 0")
    subject_ids: Dict[str, List[str]] = {subject: [] for subject in subjects}
    for sample_id, row in teacher.items():
        subject = str(row["subject"])
        if subject not in subject_ids:
            raise ManifestError(f"Teacher 含未知学科: {subject}")
        subject_ids[subject].append(sample_id)
    for subject in subjects:
        subject_ids[subject].sort()
        if not subject_ids[subject]:
            raise ManifestError(f"学科没有可 bootstrap 样本: {subject}")

    rng = random.Random(seed)
    micro_ratios: List[float] = []
    macro_ratios: List[float] = []
    skipped_micro = 0
    skipped_macro = 0
    for _ in range(iterations):
        teacher_total = 0
        candidate_total = 0
        teacher_subject_accuracy: List[float] = []
        candidate_subject_accuracy: List[float] = []
        for subject in subjects:
            ids = subject_ids[subject]
            selected = [ids[rng.randrange(len(ids))] for _ in range(len(ids))]
            teacher_correct = sum(bool(teacher[sample_id]["correct"]) for sample_id in selected)
            candidate_correct = sum(
                bool(candidate[sample_id]["correct"]) for sample_id in selected
            )
            teacher_total += teacher_correct
            candidate_total += candidate_correct
            teacher_subject_accuracy.append(teacher_correct / len(selected))
            candidate_subject_accuracy.append(candidate_correct / len(selected))
        if teacher_total:
            micro_ratios.append(candidate_total / teacher_total)
        else:
            skipped_micro += 1
        teacher_macro = statistics.fmean(teacher_subject_accuracy)
        candidate_macro = statistics.fmean(candidate_subject_accuracy)
        if teacher_macro:
            macro_ratios.append(candidate_macro / teacher_macro)
        else:
            skipped_macro += 1

    def interval(values: Sequence[float], skipped: int) -> Dict[str, Any]:
        if not values:
            return {
                "lower": None,
                "upper": None,
                "valid_iterations": 0,
                "skipped_zero_teacher": skipped,
            }
        return {
            "lower": round(_percentile(values, 0.025), 6),
            "upper": round(_percentile(values, 0.975), 6),
            "valid_iterations": len(values),
            "skipped_zero_teacher": skipped,
        }

    return {
        "method": "subject_stratified_sample_paired_nonparametric_bootstrap",
        "iterations": iterations,
        "seed": seed,
        "confidence_level": 0.95,
        "strata_sample_counts": {
            subject: len(subject_ids[subject]) for subject in subjects
        },
        "pairing_unit": "sample_id",
        "micro_retention_ratio_95ci": interval(micro_ratios, skipped_micro),
        "subject_macro_retention_ratio_95ci": interval(
            macro_ratios, skipped_macro
        ),
    }


def _pairwise_metrics(
    teacher: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    teacher_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    subjects: Sequence[str],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    counts = Counter()
    for sample_id in sorted(teacher):
        teacher_correct = bool(teacher[sample_id]["correct"])
        candidate_correct = bool(candidate[sample_id]["correct"])
        if teacher_correct and candidate_correct:
            counts["both_correct"] += 1
        elif teacher_correct:
            counts["teacher_correct_candidate_wrong"] += 1
        elif candidate_correct:
            counts["teacher_wrong_candidate_correct"] += 1
        else:
            counts["both_wrong"] += 1

    teacher_accuracy = float(teacher_metrics["overall"]["accuracy"])
    candidate_accuracy = float(candidate_metrics["overall"]["accuracy"])
    teacher_macro = float(teacher_metrics["subject_macro_accuracy"])
    candidate_macro = float(candidate_metrics["subject_macro_accuracy"])
    micro_retention = (
        None if teacher_accuracy == 0 else candidate_accuracy / teacher_accuracy
    )
    macro_retention = None if teacher_macro == 0 else candidate_macro / teacher_macro
    per_subject: Dict[str, Any] = {}
    for subject in subjects:
        teacher_subject = float(
            teacher_metrics["subjects"][subject]["accuracy"]
        )
        candidate_subject = float(
            candidate_metrics["subjects"][subject]["accuracy"]
        )
        per_subject[subject] = {
            "teacher_accuracy": teacher_subject,
            "candidate_accuracy": candidate_subject,
            "candidate_to_teacher_retention_ratio": None
            if teacher_subject == 0
            else round(candidate_subject / teacher_subject, 6),
        }
    bootstrap = stratified_paired_retention_bootstrap(
        teacher,
        candidate,
        subjects,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    return {
        "sample_pairs": len(teacher),
        "both_correct": counts["both_correct"],
        "teacher_correct_candidate_wrong": counts[
            "teacher_correct_candidate_wrong"
        ],
        "teacher_wrong_candidate_correct": counts[
            "teacher_wrong_candidate_correct"
        ],
        "both_wrong": counts["both_wrong"],
        "candidate_to_teacher_micro_retention_ratio": None
        if micro_retention is None
        else round(micro_retention, 6),
        "candidate_to_teacher_subject_macro_retention_ratio": None
        if macro_retention is None
        else round(macro_retention, 6),
        "subjects": per_subject,
        "bootstrap": bootstrap,
    }


def _gate_result(
    preregistration: Mapping[str, Any],
    teacher_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    pairwise: Mapping[str, Any],
) -> Dict[str, Any]:
    gates = _require_object(
        preregistration.get("formal_gates"), "pre_registration.formal_gates"
    )
    checks: List[Dict[str, Any]] = []

    def add(name: str, actual: float, operator: str, expected: float) -> None:
        if operator == ">=":
            passed = actual >= expected
        elif operator == "<=":
            passed = actual <= expected
        elif operator == "==":
            passed = actual == expected
        else:
            raise ManifestError(f"不支持的门禁运算符: {operator}")
        checks.append(
            {
                "name": name,
                "actual": actual,
                "operator": operator,
                "expected": expected,
                "passed": passed,
            }
        )

    teacher_overall = _require_object(teacher_metrics.get("overall"), "teacher.overall")
    candidate_overall = _require_object(
        candidate_metrics.get("overall"), "candidate.overall"
    )
    teacher_accuracy_for_gate = (
        int(teacher_overall["correct"]) / int(teacher_overall["total"])
    )
    candidate_accuracy_for_gate = (
        int(candidate_overall["correct"]) / int(candidate_overall["total"])
    )
    retention_for_gate = (
        None
        if int(teacher_overall["correct"]) == 0
        else candidate_accuracy_for_gate / teacher_accuracy_for_gate
    )
    teacher_minimum = _require_probability(
        gates.get("teacher_minimum_accuracy"),
        "pre_registration.formal_gates.teacher_minimum_accuracy",
    )
    candidate_minimum = _require_probability(
        gates.get("candidate_minimum_accuracy"),
        "pre_registration.formal_gates.candidate_minimum_accuracy",
    )
    retention_minimum = _require_probability(
        gates.get("candidate_to_teacher_retention_minimum"),
        "pre_registration.formal_gates.candidate_to_teacher_retention_minimum",
    )
    teacher_exact = _require_probability(
        gates.get("teacher_exact_output_rate_required"),
        "pre_registration.formal_gates.teacher_exact_output_rate_required",
    )
    candidate_exact = _require_probability(
        gates.get("candidate_exact_output_rate_required"),
        "pre_registration.formal_gates.candidate_exact_output_rate_required",
    )
    completed_required = _require_integer(
        gates.get("completed_samples_required_per_model"),
        "pre_registration.formal_gates.completed_samples_required_per_model",
        minimum=1,
    )
    errors_allowed = _require_integer(
        gates.get("execution_errors_allowed"),
        "pre_registration.formal_gates.execution_errors_allowed",
        minimum=0,
    )
    add("teacher_accuracy", teacher_accuracy_for_gate, ">=", teacher_minimum)
    add(
        "candidate_accuracy",
        candidate_accuracy_for_gate,
        ">=",
        candidate_minimum,
    )
    add(
        "candidate_to_teacher_retention",
        -1.0 if retention_for_gate is None else retention_for_gate,
        ">=",
        retention_minimum,
    )
    add(
        "teacher_strict_output_rate",
        teacher_overall["strict_output_rate"],
        "==",
        teacher_exact,
    )
    add(
        "candidate_strict_output_rate",
        candidate_overall["strict_output_rate"],
        "==",
        candidate_exact,
    )
    add(
        "teacher_completed_samples",
        teacher_overall["total"],
        "==",
        completed_required,
    )
    add(
        "candidate_completed_samples",
        candidate_overall["total"],
        "==",
        completed_required,
    )
    add(
        "teacher_execution_errors",
        teacher_overall["execution_error_count"],
        "<=",
        errors_allowed,
    )
    add(
        "candidate_execution_errors",
        candidate_overall["execution_error_count"],
        "<=",
        errors_allowed,
    )
    measurable_passed = all(check["passed"] for check in checks)
    required_candidate_correct = max(
        math.ceil(candidate_minimum * completed_required),
        math.ceil(
            retention_minimum * int(teacher_overall["correct"])
        ),
    )
    bootstrap_lower = pairwise["bootstrap"][
        "micro_retention_ratio_95ci"
    ]["lower"]
    runtime_after_accuracy = gates.get("runtime_gate_runs_only_after_accuracy_gate_passes")
    if not isinstance(runtime_after_accuracy, bool):
        raise ManifestError(
            "formal_gates.runtime_gate_runs_only_after_accuracy_gate_passes 必须是布尔值"
        )
    return {
        "checks": checks,
        "all_machine_verifiable_gates_passed": measurable_passed,
        "required_candidate_correct_for_observed_teacher": required_candidate_correct,
        "observed_candidate_correct": candidate_overall["correct"],
        "retention_confidence_interpretation": {
            "two_sided_95ci_lower": bootstrap_lower,
            "two_sided_95ci_lower_meets_retention_threshold": bootstrap_lower
            is not None
            and float(bootstrap_lower) >= retention_minimum,
            "note": (
                "预注册正式门禁使用保持率点估计；置信区间下界达标是更强证据，"
                "不替代点估计门禁。"
            ),
        },
        "runtime_gate_allowed": measurable_passed
        if runtime_after_accuracy
        else True,
        "procedural_requirements": {
            "formal_runs_allowed": gates.get("formal_runs_allowed"),
            "result_based_retry_allowed": gates.get("result_based_retry_allowed"),
            "verification_status": "not_verifiable_from_evaluation_result_json",
        },
        "formal_claim_status": "machine_verifiable_gates_passed_procedural_attestation_required"
        if measurable_passed
        else "failed_machine_verifiable_gate",
    }


def summarize(
    *,
    evaluation_path: Path,
    expected_evaluation_sha256: str,
    dataset_path: Path,
    manifest_path: Path,
    preregistration_path: Path,
    expected_preregistration_sha256: str,
    final_candidate_selection_path: Path,
    expected_final_candidate_selection_sha256: str,
    evaluation_attestation_path: Path,
    expected_evaluation_attestation_sha256: str,
    teacher_label: str,
    candidate_label: str,
    bootstrap_iterations: int = 10000,
    bootstrap_seed: int = 20260813,
) -> Dict[str, Any]:
    if teacher_label == candidate_label:
        raise ManifestError("Teacher 与候选模型 label 不能相同")
    dataset, subjects, preregistration, provenance, validation = _load_frozen_dataset(
        dataset_path,
        manifest_path,
        preregistration_path,
        expected_preregistration_sha256,
    )
    final_selection_provenance, final_selection_validation = (
        _load_final_candidate_selection(
            selection_path=final_candidate_selection_path,
            expected_selection_sha256=expected_final_candidate_selection_sha256,
            preregistration_path=preregistration_path,
            preregistration=preregistration,
        )
    )
    provenance["final_candidate_selection"] = final_selection_provenance
    validation.update(final_selection_validation)
    expected_evaluation_sha = _require_sha256(
        expected_evaluation_sha256, "expected_evaluation_sha256"
    )
    actual_evaluation_sha = sha256_file(evaluation_path)
    if actual_evaluation_sha != expected_evaluation_sha:
        raise ManifestError("评测结果 SHA-256 与外部记录不一致")
    evaluation = read_json_object(evaluation_path)
    models = _require_object(evaluation.get("models"), "evaluation.models")
    if set(models) != {teacher_label, candidate_label}:
        raise ManifestError(
            "正式结果必须且只能包含指定 Teacher 与候选模型两个 label"
        )
    if evaluation.get("sample_count") != len(dataset):
        raise ManifestError("evaluation.sample_count 与冻结题集不一致")
    if evaluation.get("task") != "general_capability_retention":
        raise ManifestError("evaluation.task 不是通用能力保持率评测")
    reported_dataset = _require_text(
        evaluation.get("dataset_jsonl"), "evaluation.dataset_jsonl"
    )
    reported_dataset_path = Path(reported_dataset)
    if reported_dataset_path.is_absolute():
        dataset_reference_matches = (
            reported_dataset_path.resolve() == dataset_path.resolve()
        )
    else:
        reported_parts = reported_dataset_path.parts
        actual_parts = dataset_path.resolve().parts
        dataset_reference_matches = (
            bool(reported_parts)
            and len(reported_parts) <= len(actual_parts)
            and actual_parts[-len(reported_parts) :] == reported_parts
        )
    if not dataset_reference_matches:
        raise ManifestError("evaluation.dataset_jsonl 不是本次冻结题集")
    protocol = _validate_protocol(
        preregistration.get("evaluation_protocol"),
        "pre_registration.evaluation_protocol",
    )
    if evaluation.get("num_ctx") != protocol["num_ctx"]:
        raise ManifestError("evaluation.num_ctx 与预注册协议不一致")
    if evaluation.get("no_thinking") is not True:
        raise ManifestError("evaluation 未证明按预注册关闭 thinking")
    reported_final_selection = Path(
        _require_text(
            evaluation.get("final_candidate_selection_path"),
            "evaluation.final_candidate_selection_path",
        )
    )
    if not reported_final_selection.is_absolute():
        reported_final_selection = evaluation_path.parent / reported_final_selection
    if reported_final_selection.resolve() != final_candidate_selection_path.resolve():
        raise ManifestError("evaluation 未引用本次最终候选选择文件")
    validation["evaluation_final_candidate_selection_reference_matches"] = True

    attestation_provenance, attestation_validation = _load_evaluation_attestation(
        attestation_path=evaluation_attestation_path,
        expected_attestation_sha256=expected_evaluation_attestation_sha256,
        actual_preregistration_sha256=provenance["pre_registration"]["sha256"],
        actual_dataset_sha256=provenance["dataset"]["sha256"],
        actual_dataset_manifest_sha256=provenance["dataset_manifest"]["sha256"],
        actual_final_candidate_selection_sha256=final_selection_provenance[
            "sha256"
        ],
        preregistration=preregistration,
        evaluation=evaluation,
        teacher_label=teacher_label,
        candidate_label=candidate_label,
    )
    provenance["evaluation_attestation"] = attestation_provenance
    validation.update(attestation_validation)
    validation["evaluation_dataset_reference_matches"] = True
    teacher = _validated_model_samples(evaluation, teacher_label, dataset)
    candidate = _validated_model_samples(evaluation, candidate_label, dataset)
    if set(teacher) != set(candidate):
        raise ManifestError("Teacher 与候选模型 sample_id 集合不一致")
    validation.update(
        {
            "teacher_completed_all_samples": True,
            "candidate_completed_all_samples": True,
            "teacher_candidate_sample_ids_identical": True,
            "result_contains_exactly_two_declared_models": True,
        }
    )
    provenance["evaluation"] = {
        "path": str(evaluation_path),
        "sha256": actual_evaluation_sha,
        "expected_sha256_anchor": expected_evaluation_sha,
    }
    validation["evaluation_hash_matches_external_record"] = True
    teacher_model = _require_text(
        _require_object(models[teacher_label], f"evaluation.models.{teacher_label}").get(
            "model"
        ),
        f"evaluation.models.{teacher_label}.model",
    )
    candidate_model = _require_text(
        _require_object(
            models[candidate_label], f"evaluation.models.{candidate_label}"
        ).get("model"),
        f"evaluation.models.{candidate_label}.model",
    )
    preregistered_teacher = _require_text(
        _require_object(
            preregistration.get("teacher"), "pre_registration.teacher"
        ).get("model"),
        "pre_registration.teacher.model",
    )
    if teacher_model != preregistered_teacher:
        raise ManifestError("Teacher 模型名与预注册不一致")
    if candidate_model == teacher_model:
        raise ManifestError("候选模型不能与 Teacher 使用同一模型名")
    teacher_metrics = {
        "model": teacher_model,
        **_model_metrics(teacher, subjects),
    }
    candidate_metrics = {
        "model": candidate_model,
        **_model_metrics(candidate, subjects),
    }
    pairwise = _pairwise_metrics(
        teacher,
        candidate,
        teacher_metrics,
        candidate_metrics,
        subjects,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    gate = _gate_result(
        preregistration, teacher_metrics, candidate_metrics, pairwise
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "frozen_ceval_specialty_teacher_candidate_summary",
        "scope_category": _require_object(
            preregistration.get("scope"), "pre_registration.scope"
        )["category"],
        "read_only_inference": True,
        "model_inference_performed": False,
        "teacher_label": teacher_label,
        "candidate_label": candidate_label,
        "provenance": provenance,
        "validation": validation,
        "models": {
            teacher_label: teacher_metrics,
            candidate_label: candidate_metrics,
        },
        "pairwise": pairwise,
        "pre_registered_gate": gate,
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as file_obj:
            json.dump(
                dict(value), file_obj, ensure_ascii=False, indent=2, sort_keys=True
            )
            file_obj.write("\n")
    except FileExistsError as exc:
        raise ManifestError(f"拒绝覆盖已有终测汇总: {path}") from exc


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="只读校验并汇总冻结 C-Eval 社会科学专项终测结果。"
    )
    parser.add_argument("--evaluation_json", required=True)
    parser.add_argument("--expected_evaluation_sha256", required=True)
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument("--pre_registration", required=True)
    parser.add_argument("--expected_preregistration_sha256", required=True)
    parser.add_argument("--final_candidate_selection", required=True)
    parser.add_argument(
        "--expected_final_candidate_selection_sha256", required=True
    )
    parser.add_argument("--evaluation_attestation", required=True)
    parser.add_argument("--expected_evaluation_attestation_sha256", required=True)
    parser.add_argument("--teacher_label", default="teacher")
    parser.add_argument("--candidate_label", required=True)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260813)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args(argv)

    for value, label in (
        (args.evaluation_json, "evaluation_json"),
        (args.dataset_jsonl, "dataset_jsonl"),
        (args.dataset_manifest, "dataset_manifest"),
        (args.pre_registration, "pre_registration"),
        (args.final_candidate_selection, "final_candidate_selection"),
        (args.evaluation_attestation, "evaluation_attestation"),
    ):
        if not Path(value).resolve().is_file():
            raise ManifestError(f"{label} 不存在: {value}")
    if args.bootstrap_iterations != 10000:
        raise ManifestError("正式汇总必须执行预注册的 10000 次 bootstrap")
    result = summarize(
        evaluation_path=Path(args.evaluation_json).resolve(),
        expected_evaluation_sha256=args.expected_evaluation_sha256,
        dataset_path=Path(args.dataset_jsonl).resolve(),
        manifest_path=Path(args.dataset_manifest).resolve(),
        preregistration_path=Path(args.pre_registration).resolve(),
        expected_preregistration_sha256=args.expected_preregistration_sha256,
        final_candidate_selection_path=Path(
            args.final_candidate_selection
        ).resolve(),
        expected_final_candidate_selection_sha256=(
            args.expected_final_candidate_selection_sha256
        ),
        evaluation_attestation_path=Path(args.evaluation_attestation).resolve(),
        expected_evaluation_attestation_sha256=(
            args.expected_evaluation_attestation_sha256
        ),
        teacher_label=args.teacher_label,
        candidate_label=args.candidate_label,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write_json_exclusive(Path(args.output_json).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
