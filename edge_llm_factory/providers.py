"""用途：用统一配置调用 llama.cpp、Ollama 或 OpenAI-compatible 文本模型服务。"""

import argparse
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit

from edge_llm_factory.contracts import ManifestError, read_json_object


RUNTIME_SCHEMA = "edge-llm-runtime/v1"
PROVIDERS = {"llama_cpp", "ollama", "openai_compatible"}


def _reject_unknown(value: Mapping[str, Any], allowed: set, field: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ManifestError(
            "{} 含未知字段: {}".format(field, ", ".join(unknown))
        )


def _number(
    value: Any,
    field: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
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


def _integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError("{} 必须是整数".format(field))
    if value < minimum or value > maximum:
        raise ManifestError("{} 必须位于 [{}, {}]".format(field, minimum, maximum))
    return value


def _endpoint(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError("runtime endpoint 必须是非空 URL")
    endpoint = value.strip().rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ManifestError("runtime endpoint 只支持 http/https URL")
    if parsed.username or parsed.password:
        raise ManifestError("runtime endpoint 禁止内嵌账号或密码")
    return endpoint


def validate_runtime_config(value: Mapping[str, Any]) -> Dict[str, Any]:
    config = dict(value)
    _reject_unknown(
        config,
        {
            "schema_version",
            "provider",
            "endpoint",
            "model",
            "timeout_seconds",
            "generation",
            "authentication",
        },
        "runtime config",
    )
    if config.get("schema_version") != RUNTIME_SCHEMA:
        raise ManifestError("runtime schema_version 必须是 {}".format(RUNTIME_SCHEMA))
    provider = str(config.get("provider", ""))
    if provider not in PROVIDERS:
        raise ManifestError("runtime provider 必须是 {}".format(sorted(PROVIDERS)))
    endpoint = _endpoint(config.get("endpoint"))
    raw_model = config.get("model", "")
    if not isinstance(raw_model, str):
        raise ManifestError("runtime model 必须是字符串")
    model = raw_model.strip()
    if provider in {"ollama", "openai_compatible"} and not model:
        raise ManifestError("{} runtime 必须声明 model".format(provider))
    timeout = _number(config.get("timeout_seconds", 5.0), "timeout_seconds", 0.01, 600)

    raw_generation = config.get("generation", {})
    if not isinstance(raw_generation, dict):
        raise ManifestError("generation 必须是对象")
    _reject_unknown(
        raw_generation,
        {
            "max_input_tokens",
            "max_output_tokens",
            "temperature",
            "top_p",
            "seed",
            "thinking",
            "keep_alive",
        },
        "generation",
    )
    max_input = _integer(
        raw_generation.get("max_input_tokens", 512),
        "generation.max_input_tokens",
        1,
        262144,
    )
    max_output = _integer(
        raw_generation.get("max_output_tokens", 1),
        "generation.max_output_tokens",
        1,
        4096,
    )
    temperature = _number(
        raw_generation.get("temperature", 0), "generation.temperature", 0, 2
    )
    top_p = _number(raw_generation.get("top_p", 1), "generation.top_p", 0, 1)
    seed = _integer(raw_generation.get("seed", 42), "generation.seed", 0, 2**31 - 1)
    thinking = raw_generation.get("thinking", False)
    if not isinstance(thinking, bool):
        raise ManifestError("generation.thinking 必须是布尔值")
    raw_keep_alive = raw_generation.get("keep_alive", "30m")
    if not isinstance(raw_keep_alive, str):
        raise ManifestError("generation.keep_alive 必须是字符串")
    keep_alive = raw_keep_alive.strip()
    if not keep_alive:
        raise ManifestError("generation.keep_alive 不能为空")

    raw_auth = config.get("authentication", {})
    if not isinstance(raw_auth, dict):
        raise ManifestError("authentication 必须是对象")
    _reject_unknown(raw_auth, {"api_key_env"}, "authentication")
    raw_api_key_env = raw_auth.get("api_key_env", "")
    if not isinstance(raw_api_key_env, str):
        raise ManifestError("authentication.api_key_env 必须是字符串")
    api_key_env = raw_api_key_env.strip()
    if api_key_env and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env) is None:
        raise ManifestError("authentication.api_key_env 格式无效")
    if provider != "openai_compatible" and api_key_env:
        raise ManifestError("只有 openai_compatible provider 支持 api_key_env")

    return {
        "schema_version": RUNTIME_SCHEMA,
        "provider": provider,
        "endpoint": endpoint,
        "model": model,
        "timeout_seconds": timeout,
        "generation": {
            "max_input_tokens": max_input,
            "max_output_tokens": max_output,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "thinking": thinking,
            "keep_alive": keep_alive,
        },
        "authentication": {"api_key_env": api_key_env},
    }


@dataclass(frozen=True)
class GenerationResult:
    provider: str
    model: str
    text: str
    latency_ms: float
    prompt_tokens: Optional[int]
    output_tokens: Optional[int]
    load_duration_ms: Optional[float]
    prompt_duration_ms: Optional[float]
    generation_duration_ms: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GenerationProvider(ABC):
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = validate_runtime_config(config)

    def describe(self) -> Dict[str, Any]:
        generation = dict(self.config["generation"])
        return {
            "schema_version": self.config["schema_version"],
            "provider": self.config["provider"],
            "endpoint": self.config["endpoint"],
            "model": self.config["model"],
            "timeout_seconds": self.config["timeout_seconds"],
            "generation": generation,
            "authenticated": bool(self.config["authentication"]["api_key_env"]),
        }

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> GenerationResult:
        """Generate one normalized text response."""

    def generate_structured(
        self,
        prompt: str,
        schema: Mapping[str, Any],
        system_prompt: str = "",
    ) -> GenerationResult:
        """Generate schema-constrained JSON when the backend supports it."""
        return self.generate(prompt, system_prompt=system_prompt)

    def _post(self, url: str, payload: Mapping[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=float(self.config["timeout_seconds"])
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read(1024).decode("utf-8", errors="replace")
            raise RuntimeError("model runtime HTTP {}: {}".format(exc.code, detail)) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("model runtime request failed: {}".format(exc)) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("model runtime returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise RuntimeError("model runtime response must be a JSON object")
        return result

    @staticmethod
    def _prompt(prompt: str, system_prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a string")
        return prompt.strip()


class LlamaCppProvider(GenerationProvider):
    def generate(self, prompt: str, system_prompt: str = "") -> GenerationResult:
        prompt = self._prompt(prompt, system_prompt)
        rendered = "{}\n\n{}".format(system_prompt.strip(), prompt) if system_prompt.strip() else prompt
        options = self.config["generation"]
        payload = {
            "prompt": rendered,
            "temperature": options["temperature"],
            "top_p": options["top_p"],
            "seed": options["seed"],
            "n_predict": options["max_output_tokens"],
            "stream": False,
            "cache_prompt": False,
        }
        started = time.perf_counter()
        result = self._post(self.config["endpoint"] + "/completion", payload, {})
        latency = (time.perf_counter() - started) * 1000
        timings = result.get("timings", {})
        timings = timings if isinstance(timings, dict) else {}
        return GenerationResult(
            provider="llama_cpp",
            model=self.config["model"] or "llama.cpp",
            text=str(result.get("content", "")).strip(),
            latency_ms=round(latency, 4),
            prompt_tokens=_optional_int(timings.get("prompt_n")),
            output_tokens=_optional_int(timings.get("predicted_n")),
            load_duration_ms=None,
            prompt_duration_ms=_optional_float(timings.get("prompt_ms")),
            generation_duration_ms=_optional_float(timings.get("predicted_ms")),
        )


class OllamaProvider(GenerationProvider):
    def _generate(
        self,
        prompt: str,
        system_prompt: str,
        response_format: Optional[Mapping[str, Any]] = None,
    ) -> GenerationResult:
        prompt = self._prompt(prompt, system_prompt)
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})
        options = self.config["generation"]
        payload = {
            "model": self.config["model"],
            "messages": messages,
            "stream": False,
            "think": options["thinking"],
            "keep_alive": options["keep_alive"],
            "options": {
                "temperature": options["temperature"],
                "top_p": options["top_p"],
                "seed": options["seed"],
                "num_ctx": options["max_input_tokens"],
                "num_predict": options["max_output_tokens"],
            },
        }
        if response_format is not None:
            payload["format"] = dict(response_format)
        started = time.perf_counter()
        result = self._post(self.config["endpoint"] + "/api/chat", payload, {})
        latency = (time.perf_counter() - started) * 1000
        message = result.get("message", {})
        if not isinstance(message, dict):
            raise RuntimeError("Ollama response is missing message")
        return GenerationResult(
            provider="ollama",
            model=self.config["model"],
            text=str(message.get("content", "")).strip(),
            latency_ms=round(latency, 4),
            prompt_tokens=_optional_int(result.get("prompt_eval_count")),
            output_tokens=_optional_int(result.get("eval_count")),
            load_duration_ms=_nanoseconds_ms(result.get("load_duration")),
            prompt_duration_ms=_nanoseconds_ms(result.get("prompt_eval_duration")),
            generation_duration_ms=_nanoseconds_ms(result.get("eval_duration")),
        )


    def generate(self, prompt: str, system_prompt: str = "") -> GenerationResult:
        return self._generate(prompt, system_prompt)

    def generate_structured(
        self,
        prompt: str,
        schema: Mapping[str, Any],
        system_prompt: str = "",
    ) -> GenerationResult:
        return self._generate(prompt, system_prompt, response_format=schema)

class OpenAICompatibleProvider(GenerationProvider):
    def generate(self, prompt: str, system_prompt: str = "") -> GenerationResult:
        prompt = self._prompt(prompt, system_prompt)
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})
        options = self.config["generation"]
        payload = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": options["temperature"],
            "top_p": options["top_p"],
            "seed": options["seed"],
            "max_tokens": options["max_output_tokens"],
            "stream": False,
        }
        headers: Dict[str, str] = {}
        key_name = self.config["authentication"]["api_key_env"]
        if key_name:
            token = os.environ.get(key_name, "").strip()
            if not token:
                raise RuntimeError("runtime API key environment variable is empty: {}".format(key_name))
            headers["Authorization"] = "Bearer " + token
        started = time.perf_counter()
        result = self._post(self.config["endpoint"], payload, headers)
        latency = (time.perf_counter() - started) * 1000
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError("OpenAI-compatible response is missing choices[0]")
        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            raise RuntimeError("OpenAI-compatible response is missing message")
        usage = result.get("usage", {})
        usage = usage if isinstance(usage, dict) else {}
        return GenerationResult(
            provider="openai_compatible",
            model=self.config["model"],
            text=str(message.get("content", "")).strip(),
            latency_ms=round(latency, 4),
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            output_tokens=_optional_int(usage.get("completion_tokens")),
            load_duration_ms=None,
            prompt_duration_ms=None,
            generation_duration_ms=None,
        )


def _optional_int(value: Any) -> Optional[int]:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return round(number, 4) if math.isfinite(number) else None


def _nanoseconds_ms(value: Any) -> Optional[float]:
    number = _optional_float(value)
    return round(number / 1_000_000, 4) if number is not None else None


def build_provider(config: Mapping[str, Any]) -> GenerationProvider:
    validated = validate_runtime_config(config)
    classes = {
        "llama_cpp": LlamaCppProvider,
        "ollama": OllamaProvider,
        "openai_compatible": OpenAICompatibleProvider,
    }
    return classes[validated["provider"]](validated)


def load_provider(path: Path) -> GenerationProvider:
    return build_provider(read_json_object(Path(path).resolve()))


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="校验或探测统一文本模型运行配置。")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--config", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--config", required=True)
    probe.add_argument("--prompt", required=True)
    probe.add_argument("--system_prompt", default="")
    args = parser.parse_args(argv)

    provider = load_provider(Path(args.config))
    if args.command == "verify":
        result = {"status": "valid", "runtime": provider.describe()}
    else:
        result = {
            "status": "ok",
            "runtime": provider.describe(),
            "generation": provider.generate(args.prompt, args.system_prompt).to_dict(),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
