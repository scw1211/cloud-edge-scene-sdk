"""用途：定义边缘大模型基座、场景 LoRA 与单 token 动作协议的强校验规则。"""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
from urllib.parse import urlsplit


BASE_SCHEMA = "edge-llm-base/v1"
ADAPTER_SCHEMA = "edge-llm-adapter/v1"
ACTION_SCHEMA = "edge-llm-action-map/v1"
PACKAGE_SPEC_SCHEMA = "edge-llm-package-spec/v1"
PIPELINE_SCHEMA = "edge-llm-pipeline/v1"

HEX64 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._@-]{1,127}$")
SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
LLM_INPUT_TYPES = {"natural_language", "structured_text", "compact_text_code"}
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "severe": 3}
GATE_OPERATORS = {">=", ">", "<=", "<", "=="}


class ManifestError(ValueError):
    """Raised when a model or adapter contract is incomplete or incompatible."""


def read_json_object(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError("无法读取 JSON {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise ManifestError("JSON 顶层必须是对象: {}".format(path))
    return value


def write_json_object(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _object(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError("{} 必须是对象".format(field))
    return value


def _list(value: Any, field: str, allow_empty: bool = False) -> List[Any]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ManifestError("{} 必须是{}数组".format(field, "非空" if not allow_empty else ""))
    return value


def _text(value: Any, field: str, pattern: re.Pattern = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError("{} 必须是非空字符串".format(field))
    text = value.strip()
    if pattern is not None and not pattern.fullmatch(text):
        raise ManifestError("{} 格式无效: {}".format(field, text))
    return text


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError("{} 必须是布尔值".format(field))
    return value


def _integer(value: Any, field: str, minimum: int = None, maximum: int = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError("{} 必须是整数".format(field))
    if minimum is not None and value < minimum:
        raise ManifestError("{} 不能小于 {}".format(field, minimum))
    if maximum is not None and value > maximum:
        raise ManifestError("{} 不能大于 {}".format(field, maximum))
    return value


def _number(value: Any, field: str, minimum: float = None, maximum: float = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError("{} 必须是数字".format(field))
    number = float(value)
    if not math.isfinite(number):
        raise ManifestError("{} 必须是有限数字".format(field))
    if minimum is not None and number < minimum:
        raise ManifestError("{} 不能小于 {}".format(field, minimum))
    if maximum is not None and number > maximum:
        raise ManifestError("{} 不能大于 {}".format(field, maximum))
    return number


def _hex64(value: Any, field: str) -> str:
    return _text(value, field, HEX64)


def _unique_texts(value: Any, field: str, allow_empty: bool = False) -> List[str]:
    rows = [_text(item, "{}[]".format(field)) for item in _list(value, field, allow_empty)]
    if len(rows) != len(set(rows)):
        raise ManifestError("{} 不能包含重复项".format(field))
    return rows


def safe_relative_path(value: Any, field: str) -> Path:
    text = _text(value, field)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ManifestError("{} 必须是包内相对路径: {}".format(field, text))
    return path


def json_path(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in _text(dotted_path, "metric source path").split("."):
        if not isinstance(current, dict) or part not in current:
            raise ManifestError("评测报告缺少字段路径: {}".format(dotted_path))
        current = current[part]
    return current


def evaluate_gate(actual: float, operator: str, expected: float) -> bool:
    if operator == ">=":
        return actual >= expected
    if operator == ">":
        return actual > expected
    if operator == "<=":
        return actual <= expected
    if operator == "<":
        return actual < expected
    if operator == "==":
        return actual == expected
    raise ManifestError("不支持的门槛运算符: {}".format(operator))


def validate_gates(metrics: Mapping[str, Any], gates: Any) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    names = set()
    for index, row in enumerate(_list(gates, "evaluation.gates")):
        gate = _object(row, "evaluation.gates[{}]".format(index))
        metric = _text(gate.get("metric"), "evaluation.gates[{}].metric".format(index), NAME)
        if metric in names:
            raise ManifestError("同一指标不能配置多个验收门槛: {}".format(metric))
        names.add(metric)
        if metric not in metrics:
            raise ManifestError("验收门槛引用了缺失指标: {}".format(metric))
        operator = _text(gate.get("operator"), "evaluation.gates[{}].operator".format(index))
        if operator not in GATE_OPERATORS:
            raise ManifestError("不支持的门槛运算符: {}".format(operator))
        expected = _number(gate.get("value"), "evaluation.gates[{}].value".format(index))
        actual = _number(metrics[metric], "evaluation.metrics.{}".format(metric))
        results.append(
            {
                "metric": metric,
                "operator": operator,
                "value": expected,
                "actual": actual,
                "passed": evaluate_gate(actual, operator, expected),
            }
        )
    return results


def validate_base_manifest(value: Mapping[str, Any]) -> Dict[str, Any]:
    data = _object(dict(value), "base manifest")
    if data.get("schema_version") != BASE_SCHEMA:
        raise ManifestError("base schema_version 必须是 {}".format(BASE_SCHEMA))
    _text(data.get("base_id"), "base_id", IDENTIFIER)

    source = _object(data.get("source"), "source")
    _text(source.get("model_id"), "source.model_id")
    _text(source.get("revision"), "source.revision", REVISION)
    _hex64(source.get("config_sha256"), "source.config_sha256")
    _hex64(source.get("tokenizer_sha256"), "source.tokenizer_sha256")
    _hex64(source.get("tokenizer_config_sha256"), "source.tokenizer_config_sha256")
    artifacts = _list(source.get("artifacts"), "source.artifacts")
    artifact_paths = set()
    for index, raw in enumerate(artifacts):
        artifact = _object(raw, "source.artifacts[{}]".format(index))
        relative = safe_relative_path(
            artifact.get("path"), "source.artifacts[{}].path".format(index)
        )
        if str(relative) in artifact_paths:
            raise ManifestError("source.artifacts 路径不能重复")
        artifact_paths.add(str(relative))
        _hex64(artifact.get("sha256"), "source.artifacts[{}].sha256".format(index))
        _integer(artifact.get("bytes"), "source.artifacts[{}].bytes".format(index), 1)

    model = _object(data.get("model"), "model")
    _text(model.get("loader"), "model.loader")
    _text(model.get("architecture"), "model.architecture")
    _text(model.get("model_type"), "model.model_type")
    if model.get("modality") != "text_only":
        raise ManifestError("边缘决策基座必须是 text_only")
    _integer(model.get("parameter_count"), "model.parameter_count", 1)
    _integer(model.get("num_hidden_layers"), "model.num_hidden_layers", 1)
    _integer(model.get("hidden_size"), "model.hidden_size", 1)

    policy = _object(data.get("lora_policy"), "lora_policy")
    if policy.get("format") != "peft-safetensors":
        raise ManifestError("lora_policy.format 必须是 peft-safetensors")
    if policy.get("peft_type") != "LORA" or policy.get("task_type") != "CAUSAL_LM":
        raise ManifestError("当前协议只允许 CAUSAL_LM 的 LORA")
    _integer(policy.get("max_rank"), "lora_policy.max_rank", 1, 256)
    _unique_texts(policy.get("allowed_target_modules"), "lora_policy.allowed_target_modules")
    _bool(policy.get("allow_modules_to_save"), "lora_policy.allow_modules_to_save")

    protocol = _object(data.get("decision_protocol"), "decision_protocol")
    if protocol.get("name") != "single_token_action/v1":
        raise ManifestError("decision_protocol.name 必须是 single_token_action/v1")
    _integer(protocol.get("max_input_tokens"), "decision_protocol.max_input_tokens", 1, 4096)
    if _integer(protocol.get("max_output_tokens"), "decision_protocol.max_output_tokens", 1) != 1:
        raise ManifestError("动作协议必须固定输出 1 token")
    if _bool(protocol.get("thinking"), "decision_protocol.thinking"):
        raise ManifestError("边缘动作协议必须关闭 thinking")
    slots = _list(protocol.get("slots"), "decision_protocol.slots")
    slot_names = set()
    token_ids = set()
    for index, raw in enumerate(slots):
        slot = _object(raw, "decision_protocol.slots[{}]".format(index))
        name = _text(slot.get("slot"), "decision_protocol.slots[{}].slot".format(index), NAME)
        token = _text(slot.get("token"), "decision_protocol.slots[{}].token".format(index))
        token_id = _integer(slot.get("token_id"), "decision_protocol.slots[{}].token_id".format(index), 0)
        if len(token) != 1:
            raise ManifestError("动作槽 token 必须是单字符且经 tokenizer 验证为单 token")
        if name in slot_names or token_id in token_ids:
            raise ManifestError("动作槽名称和 token_id 必须唯一")
        slot_names.add(name)
        token_ids.add(token_id)
    reserved = _object(protocol.get("reserved_slots", {}), "decision_protocol.reserved_slots")
    for name, purpose in reserved.items():
        if name not in slot_names:
            raise ManifestError("保留动作槽未在 slots 中定义: {}".format(name))
        if purpose not in {"abstain", "request_cloud"}:
            raise ManifestError("未知保留动作槽用途: {}".format(purpose))

    gates = _object(data.get("deployment_gates"), "deployment_gates")
    _number(gates.get("max_system_ram_mb"), "deployment_gates.max_system_ram_mb", 1)
    _number(gates.get("max_mean_closed_loop_ms"), "deployment_gates.max_mean_closed_loop_ms", 1)
    _number(gates.get("min_valid_output_rate"), "deployment_gates.min_valid_output_rate", 0, 1)
    return data


def base_fingerprint(base: Mapping[str, Any]) -> str:
    data = validate_base_manifest(base)
    identity = {
        "schema_version": data["schema_version"],
        "base_id": data["base_id"],
        "source": data["source"],
        "model": data["model"],
        "lora_policy": data["lora_policy"],
        "decision_protocol": data["decision_protocol"],
    }
    return canonical_sha256(identity)


def base_slots(base: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    data = validate_base_manifest(base)
    return {
        str(row["slot"]): dict(row)
        for row in data["decision_protocol"]["slots"]
    }


def validate_input_contract(
    value: Mapping[str, Any], base: Mapping[str, Any]
) -> Dict[str, Any]:
    data = _object(value, "input_contract")
    allowed_fields = {
        "event_type",
        "data_schema",
        "context_encoder",
        "llm_input_type",
        "max_input_tokens",
        "direct_media_to_llm",
    }
    unknown_fields = sorted(set(data) - allowed_fields)
    if unknown_fields:
        raise ManifestError(
            "input_contract 包含未知字段: {}".format(unknown_fields)
        )
    base_data = validate_base_manifest(base)
    _text(data.get("event_type"), "input_contract.event_type")
    data_schema = _text(data.get("data_schema"), "input_contract.data_schema")
    if not urlsplit(data_schema).scheme:
        raise ManifestError("input_contract.data_schema 必须是绝对 URI")
    _text(data.get("context_encoder"), "input_contract.context_encoder", IDENTIFIER)

    input_type = _text(
        data.get("llm_input_type"), "input_contract.llm_input_type"
    )
    if input_type not in LLM_INPUT_TYPES:
        raise ManifestError(
            "llm_input_type 必须是 natural_language/structured_text/compact_text_code"
        )
    max_tokens = _integer(
        data.get("max_input_tokens"), "input_contract.max_input_tokens", 1
    )
    if max_tokens > int(base_data["decision_protocol"]["max_input_tokens"]):
        raise ManifestError("input_contract 超过基座最大输入 token")
    direct_media = _bool(
        data.get("direct_media_to_llm"), "input_contract.direct_media_to_llm"
    )
    if base_data["model"]["modality"] == "text_only" and direct_media:
        raise ManifestError("纯文本基座禁止图像、点云或音频直接输入 LLM")
    return data


def validate_action_mapping(value: Mapping[str, Any], base: Mapping[str, Any]) -> Dict[str, Any]:
    data = _object(dict(value), "action mapping")
    if data.get("schema_version") != ACTION_SCHEMA:
        raise ManifestError("action mapping schema_version 必须是 {}".format(ACTION_SCHEMA))
    _text(data.get("scene"), "scene", IDENTIFIER)
    if data.get("protocol") != "single_token_action/v1":
        raise ManifestError("action mapping protocol 必须是 single_token_action/v1")
    available = base_slots(base)
    reserved = _object(base["decision_protocol"].get("reserved_slots", {}), "reserved_slots")
    entries = _list(data.get("entries"), "entries")
    used = set()
    for index, raw in enumerate(entries):
        entry = _object(raw, "entries[{}]".format(index))
        slot = _text(entry.get("slot"), "entries[{}].slot".format(index), NAME)
        if slot not in available:
            raise ManifestError("动作映射使用了基座未定义的槽: {}".format(slot))
        if slot in used:
            raise ManifestError("动作槽不能重复映射: {}".format(slot))
        used.add(slot)
        decision = _text(entry.get("decision"), "entries[{}].decision".format(index), IDENTIFIER)
        action_type = entry.get("candidate_action_type")
        if action_type is not None:
            _text(action_type, "entries[{}].candidate_action_type".format(index), IDENTIFIER)
        minimum = _text(entry.get("min_risk_level"), "entries[{}].min_risk_level".format(index))
        maximum = _text(entry.get("max_risk_level"), "entries[{}].max_risk_level".format(index))
        if minimum not in RISK_ORDER or maximum not in RISK_ORDER:
            raise ManifestError("动作映射风险等级必须是 low/medium/high/severe")
        if RISK_ORDER[minimum] > RISK_ORDER[maximum]:
            raise ManifestError("动作映射最小风险等级不能高于最大风险等级")
        _bool(entry.get("requires_cloud"), "entries[{}].requires_cloud".format(index))
        _bool(entry.get("safe_offline"), "entries[{}].safe_offline".format(index))
        if decision == "no_action" and action_type is not None:
            raise ManifestError("no_action 不能绑定执行动作")
        if slot in reserved and decision != reserved[slot]:
            raise ManifestError("保留槽 {} 必须映射为 {}".format(slot, reserved[slot]))
    fallback = _text(data.get("fallback_slot"), "fallback_slot", NAME)
    if fallback not in used:
        raise ManifestError("fallback_slot 必须出现在 entries 中")
    return data


def validate_adapter_manifest(
    value: Mapping[str, Any],
    base: Mapping[str, Any],
    action_mapping: Mapping[str, Any],
) -> Dict[str, Any]:
    data = _object(dict(value), "adapter manifest")
    base_data = validate_base_manifest(base)
    action_data = validate_action_mapping(action_mapping, base_data)
    if data.get("schema_version") != ADAPTER_SCHEMA:
        raise ManifestError("adapter schema_version 必须是 {}".format(ADAPTER_SCHEMA))
    _text(data.get("adapter_id"), "adapter_id", IDENTIFIER)
    scene = _text(data.get("scene"), "scene", IDENTIFIER)
    if scene != action_data["scene"]:
        raise ManifestError("adapter scene 与 action mapping scene 不一致")
    _text(data.get("version"), "version", SEMVER)

    binding = _object(data.get("base"), "base")
    if binding.get("base_id") != base_data["base_id"]:
        raise ManifestError("adapter 绑定了错误的 base_id")
    if binding.get("fingerprint") != base_fingerprint(base_data):
        raise ManifestError("adapter 与基座指纹不兼容")

    artifact = _object(data.get("adapter_artifact"), "adapter_artifact")
    safe_relative_path(artifact.get("path"), "adapter_artifact.path")
    _hex64(artifact.get("sha256"), "adapter_artifact.sha256")
    _integer(artifact.get("bytes"), "adapter_artifact.bytes", 1)
    if artifact.get("format") != "safetensors":
        raise ManifestError("adapter artifact 必须使用 safetensors")

    lora = _object(data.get("lora"), "lora")
    if lora.get("peft_type") != "LORA" or lora.get("task_type") != "CAUSAL_LM":
        raise ManifestError("adapter 只允许 CAUSAL_LM 的 LORA")
    rank = _integer(lora.get("rank"), "lora.rank", 1)
    if rank > int(base_data["lora_policy"]["max_rank"]):
        raise ManifestError("LoRA rank 超过基座允许上限")
    _integer(lora.get("alpha"), "lora.alpha", 1)
    _number(lora.get("dropout"), "lora.dropout", 0, 1)
    targets = set(_unique_texts(lora.get("target_modules"), "lora.target_modules"))
    allowed = set(base_data["lora_policy"]["allowed_target_modules"])
    if not targets.issubset(allowed):
        raise ManifestError("LoRA 包含基座未授权模块: {}".format(sorted(targets - allowed)))
    modules_to_save = _unique_texts(
        lora.get("modules_to_save", []), "lora.modules_to_save", allow_empty=True
    )
    if modules_to_save and not base_data["lora_policy"]["allow_modules_to_save"]:
        raise ManifestError("当前基座禁止 LoRA 携带完整可训练模块")

    mapping_ref = _object(data.get("action_mapping"), "action_mapping")
    safe_relative_path(mapping_ref.get("path"), "action_mapping.path")
    validate_input_contract(data.get("input_contract"), base_data)
    _hex64(mapping_ref.get("sha256"), "action_mapping.sha256")

    training = _object(data.get("training"), "training")
    _text(training.get("teacher_model"), "training.teacher_model")
    methods = _unique_texts(training.get("methods"), "training.methods")
    allowed_methods = {"sft", "on_policy_correction", "dpo", "logit_kd", "prune_distill"}
    if not set(methods).issubset(allowed_methods):
        raise ManifestError("未知蒸馏方法: {}".format(sorted(set(methods) - allowed_methods)))
    _text(training.get("train_dataset_id"), "training.train_dataset_id")
    _text(training.get("test_dataset_id"), "training.test_dataset_id")
    if _bool(training.get("test_set_used_for_training"), "training.test_set_used_for_training"):
        raise ManifestError("验收测试集不得用于训练或纠错蒸馏")

    evaluation = _object(data.get("evaluation"), "evaluation")
    evidence = _object(evaluation.get("evidence"), "evaluation.evidence")
    for name, raw in evidence.items():
        _text(name, "evaluation.evidence name", NAME)
        record = _object(raw, "evaluation.evidence.{}".format(name))
        safe_relative_path(record.get("path"), "evaluation.evidence.{}.path".format(name))
        _hex64(record.get("sha256"), "evaluation.evidence.{}.sha256".format(name))
    metrics = _object(evaluation.get("metrics"), "evaluation.metrics")
    if not metrics:
        raise ManifestError("evaluation.metrics 不能为空")
    for name, metric_value in metrics.items():
        _text(name, "evaluation metric", NAME)
        _number(metric_value, "evaluation.metrics.{}".format(name))
    sources = _object(evaluation.get("metric_sources"), "evaluation.metric_sources")
    if set(sources) != set(metrics):
        raise ManifestError("每个 evaluation metric 必须且只能有一个证据来源")
    for name, raw in sources.items():
        source = _object(raw, "evaluation.metric_sources.{}".format(name))
        if source.get("evidence") not in evidence:
            raise ManifestError("指标 {} 引用了未知证据".format(name))
        _text(source.get("path"), "evaluation.metric_sources.{}.path".format(name))
    calculated = validate_gates(metrics, evaluation.get("gates"))
    if calculated != evaluation.get("gate_results"):
        raise ManifestError("evaluation.gate_results 与指标重新计算结果不一致")

    deployment = _object(data.get("deployment"), "deployment")
    if deployment.get("runtime") not in {"llama.cpp", "ollama", "transformers"}:
        raise ManifestError("不支持的部署运行时")
    if deployment.get("format") not in {"gguf", "safetensors"}:
        raise ManifestError("不支持的部署格式")
    _text(deployment.get("quantization"), "deployment.quantization")
    _hex64(deployment.get("artifact_sha256"), "deployment.artifact_sha256")
    _integer(deployment.get("artifact_bytes"), "deployment.artifact_bytes", 1)
    _integer(deployment.get("max_input_tokens"), "deployment.max_input_tokens", 1)
    if _integer(deployment.get("max_output_tokens"), "deployment.max_output_tokens", 1) != 1:
        raise ManifestError("部署动作模型必须限制为单 token 输出")
    if _bool(deployment.get("thinking"), "deployment.thinking"):
        raise ManifestError("部署动作模型必须关闭 thinking")
    return data


def ensure_required_files(root: Path, relative_paths: Iterable[Path]) -> None:
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise ManifestError("适配器包缺少文件: {}".format(relative))
