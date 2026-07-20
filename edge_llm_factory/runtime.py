"""用途：调用 llama.cpp 单 token 模型，并用场景动作映射做确定性安全解码。"""

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from edge_llm_factory.adapter_package import (
    ACTION_MAP_NAME,
    MANIFEST_NAME,
    validate_adapter_package,
)
from edge_llm_factory.contracts import (
    ManifestError,
    RISK_ORDER,
    base_slots,
    read_json_object,
    validate_action_mapping,
    validate_base_manifest,
)
from edge_llm_factory.providers import (
    GenerationProvider,
    build_provider,
    load_provider,
)


def _risk_level(event: Mapping[str, Any]) -> str:
    risk = event.get("risk")
    if isinstance(risk, dict):
        value = risk.get("level")
    else:
        value = event.get("risk_level")
    level = str(value or "").lower()
    if level not in RISK_ORDER:
        raise ManifestError("事件缺少合法风险等级")
    return level


def _candidate_actions(event: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = event.get("candidate_actions", [])
    if not isinstance(raw, list):
        raise ManifestError("candidate_actions 必须是数组")
    return [dict(action) for action in raw if isinstance(action, dict)]


class ConfiguredActionClient:
    """把任一统一文本 provider 约束为 no-thinking 单 token 动作客户端。"""

    def __init__(self, provider: GenerationProvider) -> None:
        self.provider = provider
        generation = provider.config["generation"]
        if generation["max_output_tokens"] != 1:
            raise ManifestError("动作模型 runtime 必须固定 max_output_tokens=1")
        if generation["thinking"]:
            raise ManifestError("动作模型 runtime 必须关闭 thinking")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ConfiguredActionClient":
        return cls(build_provider(config))

    @classmethod
    def from_path(cls, path: Path) -> "ConfiguredActionClient":
        return cls(load_provider(path))

    def describe(self) -> Dict[str, Any]:
        return self.provider.describe()

    def predict(self, prompt: str, valid_tokens: Mapping[str, str]) -> Dict[str, Any]:
        if not isinstance(valid_tokens, Mapping) or not valid_tokens:
            raise ManifestError("valid_tokens 不能为空")
        token_to_slot = {str(token): str(slot) for slot, token in valid_tokens.items()}
        if len(token_to_slot) != len(valid_tokens):
            raise ManifestError("动作 token 不能重复")
        result = self.provider.generate(prompt)
        output = result.text.strip()
        if output not in token_to_slot:
            raise ManifestError("边缘模型输出不符合单 token 协议: {!r}".format(output))
        normalized = result.to_dict()
        normalized.update({"slot": token_to_slot[output], "token": output})
        normalized.pop("text", None)
        return normalized


class LlamaCppActionClient:
    """Minimal llama.cpp client with a deliberately fixed one-token contract."""

    def __init__(self, endpoint: str, timeout_seconds: float = 5.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def predict(self, prompt: str, valid_tokens: Mapping[str, str]) -> Dict[str, Any]:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be non-empty")
        payload = {
            "prompt": prompt,
            "temperature": 0,
            "top_p": 1,
            "n_predict": 1,
            "stream": False,
            "cache_prompt": False,
        }
        request = urllib.request.Request(
            self.endpoint + "/completion",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        latency_ms = (time.perf_counter() - started) * 1000.0
        output = str(result.get("content", "")).strip()
        token_to_slot = {token: slot for slot, token in valid_tokens.items()}
        if output not in token_to_slot:
            raise ManifestError("边缘模型输出不符合单 token 协议: {!r}".format(output))
        timings = result.get("timings", {})
        return {
            "slot": token_to_slot[output],
            "token": output,
            "latency_ms": round(latency_ms, 4),
            "prompt_tokens": timings.get("prompt_n") if isinstance(timings, dict) else None,
            "predicted_tokens": timings.get("predicted_n") if isinstance(timings, dict) else None,
        }


class ActionDecoder:
    def __init__(self, base: Mapping[str, Any], action_mapping: Mapping[str, Any]) -> None:
        self.base = validate_base_manifest(base)
        self.mapping = validate_action_mapping(action_mapping, self.base)
        self.entries = {str(row["slot"]): dict(row) for row in self.mapping["entries"]}
        self.fallback_slot = str(self.mapping["fallback_slot"])
        self.valid_tokens = {
            slot: str(record["token"])
            for slot, record in base_slots(self.base).items()
            if slot in self.entries
        }

    def _fallback(self, reason: str, original_slot: str) -> Dict[str, Any]:
        fallback = self.entries[self.fallback_slot]
        return {
            "slot": self.fallback_slot,
            "model_slot": original_slot,
            "decision": fallback["decision"],
            "actions": [],
            "requires_cloud": bool(fallback["requires_cloud"]),
            "safety_fallback": True,
            "fallback_reason": reason,
        }

    def decode(
        self,
        slot: str,
        event: Mapping[str, Any],
        network_available: bool,
    ) -> Dict[str, Any]:
        if slot not in self.entries:
            raise ManifestError("动作槽未在当前场景授权: {}".format(slot))
        entry = self.entries[slot]
        risk = _risk_level(event)
        if not (
            RISK_ORDER[entry["min_risk_level"]]
            <= RISK_ORDER[risk]
            <= RISK_ORDER[entry["max_risk_level"]]
        ):
            return self._fallback("risk_level_outside_action_range", slot)
        if entry["requires_cloud"] and not network_available:
            return self._fallback("cloud_required_but_network_unavailable", slot)
        if not entry["safe_offline"] and not network_available:
            return self._fallback("action_not_authorized_offline", slot)

        action_type = entry.get("candidate_action_type")
        selected: List[Dict[str, Any]] = []
        if action_type is not None:
            selected = [
                action
                for action in _candidate_actions(event)
                if action.get("action_type", action.get("type")) == action_type
            ]
            if not selected:
                return self._fallback("authorized_candidate_action_missing", slot)
        return {
            "slot": slot,
            "model_slot": slot,
            "decision": entry["decision"],
            "actions": selected,
            "requires_cloud": bool(entry["requires_cloud"]),
            "safety_fallback": False,
            "fallback_reason": None,
        }


class ValidatedEdgeLLM:
    """A validated adapter package plus model runtime; prompt encoding remains scene-owned."""

    def __init__(
        self,
        base_manifest_path: Path,
        adapter_package: Path,
        endpoint: Optional[str] = None,
        timeout_seconds: float = 5.0,
        runtime_config_path: Optional[Path] = None,
        provider: Optional[GenerationProvider] = None,
    ) -> None:
        self.base_manifest_path = Path(base_manifest_path).resolve()
        self.adapter_package = Path(adapter_package).resolve()
        self.validation = validate_adapter_package(
            self.adapter_package, self.base_manifest_path, require_gates=True
        )
        self.base = read_json_object(self.base_manifest_path)
        self.manifest = read_json_object(self.adapter_package / MANIFEST_NAME)
        self.action_mapping = read_json_object(self.adapter_package / ACTION_MAP_NAME)
        self.decoder = ActionDecoder(self.base, self.action_mapping)
        configured_modes = sum(
            value is not None for value in (endpoint, runtime_config_path, provider)
        )
        if configured_modes != 1:
            raise ValueError(
                "endpoint、runtime_config_path、provider 必须且只能提供一个"
            )
        if provider is not None:
            self.client = ConfiguredActionClient(provider)
        elif runtime_config_path is not None:
            self.client = ConfiguredActionClient.from_path(runtime_config_path)
        else:
            self.client = LlamaCppActionClient(str(endpoint), timeout_seconds)

    def describe(self) -> Dict[str, Any]:
        runtime = self.client.describe() if hasattr(self.client, "describe") else {
            "provider": "legacy_llama_cpp"
        }
        return {
            "adapter_id": self.validation["adapter_id"],
            "scene": self.validation["scene"],
            "version": self.validation["version"],
            "base_fingerprint": self.validation["base_fingerprint"],
            "metrics": dict(self.validation.get("metrics", {})),
            "input_contract": dict(self.manifest.get("input_contract", {})),
            "deployment": dict(self.manifest.get("deployment", {})),
            "runtime": runtime,
        }

    def decide(
        self,
        prompt: str,
        event: Mapping[str, Any],
        network_available: bool,
    ) -> Dict[str, Any]:
        inference = self.client.predict(prompt, self.decoder.valid_tokens)
        decoded = self.decoder.decode(inference["slot"], event, network_available)
        return {"inference": inference, "decision": decoded}
