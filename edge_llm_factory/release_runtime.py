"""用途：把 active release、适配器契约和模型 Provider 绑定成可在线使用的 Edge LLM。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from edge_llm_factory.adapter_package import MANIFEST_NAME
from edge_llm_factory.contracts import ManifestError, read_json_object
from edge_llm_factory.providers import validate_runtime_config
from edge_llm_factory.release_store import ReleaseStore
from edge_llm_factory.runtime import ValidatedEdgeLLM


RUNTIME_NAMES = {
    "llama.cpp": "llama_cpp",
    "llama_cpp": "llama_cpp",
    "ollama": "ollama",
    "openai-compatible": "openai_compatible",
    "openai_compatible": "openai_compatible",
}


@dataclass(frozen=True)
class ActiveEdgeLLM:
    release_id: str
    revision: int
    binding_fingerprint: str
    deployment_artifact: Path
    model: ValidatedEdgeLLM

    def describe(self) -> Dict[str, Any]:
        return {
            "release_id": self.release_id,
            "release_revision": self.revision,
            "binding_fingerprint": self.binding_fingerprint,
            "deployment_artifact": str(self.deployment_artifact),
            **self.model.describe(),
        }


def _runtime_matches_release(
    runtime: Dict[str, Any],
    manifest: Dict[str, Any],
    release_id: str,
    artifact: Path,
) -> None:
    deployment = manifest.get("deployment")
    if not isinstance(deployment, dict):
        raise ManifestError("适配器 manifest 缺少 deployment")
    expected_provider = RUNTIME_NAMES.get(str(deployment.get("runtime", "")))
    if expected_provider is None:
        raise ManifestError("适配器声明了未知部署 runtime")
    if runtime["provider"] != expected_provider:
        raise ManifestError(
            "runtime provider 与适配器部署契约不一致: {} != {}".format(
                runtime["provider"], expected_provider
            )
        )
    generation = runtime["generation"]
    for field in ("max_input_tokens", "max_output_tokens", "thinking"):
        if generation[field] != deployment.get(field):
            raise ManifestError("runtime generation.{} 与适配器部署契约不一致".format(field))
    if runtime["provider"] == "llama_cpp":
        model_ref = str(runtime.get("model", "")).strip()
        accepted = {
            release_id,
            artifact.name,
            str(artifact),
            str(artifact.resolve()),
        }
        if model_ref not in accepted:
            raise ManifestError(
                "llama.cpp runtime.model 必须绑定 active release ID 或 GGUF 文件: {}".format(
                    sorted(accepted)
                )
            )


def load_active_edge_llm(
    release_registry_path: Path,
    runtime_config_path: Path,
    expected_scene: Optional[str] = None,
) -> ActiveEdgeLLM:
    store = ReleaseStore(Path(release_registry_path))
    status = store.status(verify_active=True)
    release_id = status.get("active_release_id")
    if not release_id:
        raise ManifestError("release store 没有 active Edge LLM")
    record = status["releases"][release_id]
    base_manifest = Path(record["base_manifest"]["path"])
    adapter_package = Path(record["adapter_package"]["path"])
    artifact = Path(record["deployment_artifact"]["path"])
    manifest = read_json_object(adapter_package / MANIFEST_NAME)
    if expected_scene is not None and manifest.get("scene") != expected_scene:
        raise ManifestError(
            "active adapter scene {!r} 与插件场景 {!r} 不一致".format(
                manifest.get("scene"), expected_scene
            )
        )
    runtime = validate_runtime_config(read_json_object(Path(runtime_config_path)))
    _runtime_matches_release(runtime, manifest, str(release_id), artifact)
    model = ValidatedEdgeLLM(
        base_manifest_path=base_manifest,
        adapter_package=adapter_package,
        runtime_config_path=Path(runtime_config_path),
    )
    return ActiveEdgeLLM(
        release_id=str(release_id),
        revision=int(status["revision"]),
        binding_fingerprint=str(record["binding_fingerprint"]),
        deployment_artifact=artifact.resolve(),
        model=model,
    )
