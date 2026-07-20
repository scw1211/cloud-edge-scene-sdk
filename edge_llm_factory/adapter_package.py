"""用途：把场景 LoRA 构建为可审计发布包，并校验基座、权重、动作和指标证据。"""

import argparse
import json
import os
import shutil
import stat
import struct
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from edge_llm_factory.contracts import (
    ADAPTER_SCHEMA,
    PACKAGE_SPEC_SCHEMA,
    ManifestError,
    base_fingerprint,
    json_path,
    read_json_object,
    safe_relative_path,
    sha256_file,
    validate_action_mapping,
    validate_adapter_manifest,
    validate_base_manifest,
    validate_gates,
    validate_input_contract,
    write_json_object,
)


MANIFEST_NAME = "scene_adapter_manifest.json"
ACTION_MAP_NAME = "action_mapping.json"
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
FORBIDDEN_SUFFIXES = {".bin", ".ckpt", ".pkl", ".pickle", ".pt", ".pth", ".py"}
ALLOWED_ROOT_FILES = {
    MANIFEST_NAME,
    ACTION_MAP_NAME,
    ADAPTER_CONFIG_NAME,
    ADAPTER_WEIGHTS_NAME,
    "README.md",
}


def _resolve(project_root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError("{} 必须是非空路径".format(field))
    path = Path(value)
    resolved = path if path.is_absolute() else project_root / path
    return resolved.resolve()


def _artifact_record(path: Path, relative_path: str) -> Dict[str, Any]:
    if not path.is_file():
        raise ManifestError("缺少产物: {}".format(path))
    return {
        "path": relative_path,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _adapter_lora_record(config: Mapping[str, Any]) -> Dict[str, Any]:
    if config.get("peft_type") != "LORA":
        raise ManifestError("adapter_config.peft_type 必须是 LORA")
    modules_to_save = config.get("modules_to_save")
    if modules_to_save is None:
        modules_to_save = []
    targets = config.get("target_modules")
    if not isinstance(targets, list):
        raise ManifestError("adapter_config.target_modules 必须是数组")
    return {
        "peft_type": config.get("peft_type"),
        "task_type": config.get("task_type"),
        "rank": config.get("r"),
        "alpha": config.get("lora_alpha"),
        "dropout": config.get("lora_dropout"),
        "target_modules": sorted(str(value) for value in targets),
        "modules_to_save": sorted(str(value) for value in modules_to_save),
    }


def inspect_safetensors(path: Path) -> Dict[str, Any]:
    """Read only the safetensors header; pickle or malformed tensors never get loaded."""
    file_size = path.stat().st_size
    if file_size < 10:
        raise ManifestError("safetensors 文件过短: {}".format(path))
    with path.open("rb") as file_obj:
        header_size = struct.unpack("<Q", file_obj.read(8))[0]
        if header_size <= 2 or header_size > min(file_size - 8, 64 * 1024 * 1024):
            raise ManifestError("safetensors header 长度无效: {}".format(path))
        try:
            header = json.loads(file_obj.read(header_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestError("safetensors header 不是合法 JSON: {}".format(path)) from exc
    if not isinstance(header, dict):
        raise ManifestError("safetensors header 必须是对象")
    tensor_count = 0
    payload_size = file_size - 8 - header_size
    for name, raw in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(raw, dict):
            raise ManifestError("safetensors tensor 描述无效: {}".format(name))
        offsets = raw.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] <= offsets[0]
            or offsets[1] > payload_size
        ):
            raise ManifestError("safetensors tensor 偏移无效: {}".format(name))
        if not isinstance(raw.get("shape"), list) or not isinstance(raw.get("dtype"), str):
            raise ManifestError("safetensors tensor shape/dtype 无效: {}".format(name))
        tensor_count += 1
    if tensor_count == 0:
        raise ManifestError("safetensors 不包含任何张量")
    return {
        "header_bytes": header_size,
        "tensor_count": tensor_count,
        "file_bytes": file_size,
    }


def _scan_release_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ManifestError("发布包禁止符号链接: {}".format(path.relative_to(root)))
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ManifestError("发布包禁止可执行或 pickle 权重: {}".format(relative))
        if path.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            raise ManifestError("发布包文件不应带可执行位: {}".format(relative))
        if len(relative.parts) == 1 and path.name not in ALLOWED_ROOT_FILES:
            raise ManifestError("发布包根目录含未知文件: {}".format(relative))
        if len(relative.parts) > 1 and relative.parts[0] != "evidence":
            raise ManifestError("发布包仅允许 evidence/ 子目录: {}".format(relative))
        if relative.parts[0] == "evidence" and path.suffix.lower() != ".json":
            raise ManifestError("证据目录只允许 JSON: {}".format(relative))


def _verify_record(root: Path, record: Mapping[str, Any], field: str) -> Path:
    relative = safe_relative_path(record.get("path"), "{}.path".format(field))
    path = root / relative
    if not path.is_file():
        raise ManifestError("{} 引用文件不存在: {}".format(field, relative))
    if path.stat().st_size != record.get("bytes", path.stat().st_size):
        raise ManifestError("{} 文件大小不匹配".format(field))
    if sha256_file(path) != record.get("sha256"):
        raise ManifestError("{} SHA256 不匹配".format(field))
    return path


def _verify_peft_config(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    base: Mapping[str, Any],
) -> None:
    source_model = str(config.get("base_model_name_or_path", ""))
    if source_model != str(base["source"]["model_id"]):
        raise ManifestError("adapter_config 绑定的模型名与 base manifest 不一致")
    actual = _adapter_lora_record(config)
    expected = dict(manifest["lora"])
    expected["target_modules"] = sorted(expected["target_modules"])
    expected["modules_to_save"] = sorted(expected["modules_to_save"])
    if actual != expected:
        raise ManifestError("adapter_config 与 manifest.lora 不一致")
    if config.get("bias") not in {None, "none"}:
        raise ManifestError("边缘 LoRA 禁止训练 base bias")


def validate_adapter_package(
    package_dir: Path,
    base_manifest_path: Path,
    require_gates: bool = True,
) -> Dict[str, Any]:
    root = package_dir.resolve()
    if not root.is_dir():
        raise ManifestError("适配器包目录不存在: {}".format(root))
    _scan_release_files(root)
    base = validate_base_manifest(read_json_object(base_manifest_path.resolve()))
    manifest = read_json_object(root / MANIFEST_NAME)
    action_path = root / safe_relative_path(
        manifest.get("action_mapping", {}).get("path"), "action_mapping.path"
    )
    action_mapping = read_json_object(action_path)
    validate_action_mapping(action_mapping, base)
    validate_adapter_manifest(manifest, base, action_mapping)

    artifact_path = _verify_record(root, manifest["adapter_artifact"], "adapter_artifact")
    safetensors = inspect_safetensors(artifact_path)
    action_record = manifest["action_mapping"]
    if sha256_file(action_path) != action_record["sha256"]:
        raise ManifestError("action_mapping SHA256 不匹配")

    config_path = root / ADAPTER_CONFIG_NAME
    config = read_json_object(config_path)
    _verify_peft_config(config, manifest, base)

    evidence_documents: Dict[str, Dict[str, Any]] = {}
    for name, record in manifest["evaluation"]["evidence"].items():
        evidence_path = _verify_record(root, record, "evaluation.evidence.{}".format(name))
        evidence_documents[name] = read_json_object(evidence_path)
    for metric, source in manifest["evaluation"]["metric_sources"].items():
        observed = json_path(evidence_documents[source["evidence"]], source["path"])
        expected = manifest["evaluation"]["metrics"][metric]
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise ManifestError("指标证据不是数字: {}".format(metric))
        if abs(float(observed) - float(expected)) > 1e-9:
            raise ManifestError("指标 {} 与证据报告不一致".format(metric))
    gate_results = validate_gates(
        manifest["evaluation"]["metrics"], manifest["evaluation"]["gates"]
    )
    if require_gates and not all(row["passed"] for row in gate_results):
        failed = [row["metric"] for row in gate_results if not row["passed"]]
        raise ManifestError("适配器未通过发布门槛: {}".format(", ".join(failed)))

    return {
        "status": "valid",
        "adapter_id": manifest["adapter_id"],
        "scene": manifest["scene"],
        "version": manifest["version"],
        "base_id": base["base_id"],
        "base_fingerprint": base_fingerprint(base),
        "adapter_sha256": manifest["adapter_artifact"]["sha256"],
        "safetensors": safetensors,
        "metrics": manifest["evaluation"]["metrics"],
        "gate_results": gate_results,
    }


def _validate_spec(value: Mapping[str, Any], base: Mapping[str, Any]) -> Dict[str, Any]:
    spec = dict(value)
    if spec.get("schema_version") != PACKAGE_SPEC_SCHEMA:
        raise ManifestError("package spec schema_version 必须是 {}".format(PACKAGE_SPEC_SCHEMA))
    for field in ("adapter_id", "scene", "version", "adapter_source", "action_mapping"):
        if not isinstance(spec.get(field), str) or not str(spec[field]).strip():
            raise ManifestError("package spec 缺少 {}".format(field))
    training = spec.get("training")
    evaluation = spec.get("evaluation")
    deployment = spec.get("deployment")
    if not isinstance(training, dict) or not isinstance(evaluation, dict) or not isinstance(deployment, dict):
        raise ManifestError("package spec 必须包含 training/evaluation/deployment 对象")
    if not isinstance(evaluation.get("evidence"), dict) or not evaluation["evidence"]:
        raise ManifestError("package spec evaluation.evidence 不能为空")
    if not isinstance(evaluation.get("metric_sources"), dict) or not evaluation["metric_sources"]:
        raise ManifestError("package spec evaluation.metric_sources 不能为空")
    if not isinstance(evaluation.get("gates"), list) or not evaluation["gates"]:
        raise ManifestError("package spec evaluation.gates 不能为空")
    validate_input_contract(spec.get("input_contract"), base)
    return spec


def _copy_evidence(
    project_root: Path,
    target_root: Path,
    evidence_spec: Mapping[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    records: Dict[str, Dict[str, Any]] = {}
    documents: Dict[str, Dict[str, Any]] = {}
    evidence_root = target_root / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    for name, raw_path in sorted(evidence_spec.items()):
        if not isinstance(name, str) or not name.replace("_", "").isalnum():
            raise ManifestError("证据名称只能包含字母、数字和下划线: {}".format(name))
        source = _resolve(project_root, raw_path, "evaluation.evidence.{}".format(name))
        document = read_json_object(source)
        target = evidence_root / "{}.json".format(name)
        shutil.copy2(source, target)
        records[name] = _artifact_record(target, "evidence/{}.json".format(name))
        records[name].pop("bytes")
        documents[name] = document
    return records, documents


def _metric_values(
    documents: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Any],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name, raw in sources.items():
        if not isinstance(raw, dict):
            raise ManifestError("metric source 必须是对象: {}".format(name))
        evidence_name = raw.get("evidence")
        dotted_path = raw.get("path")
        if evidence_name not in documents:
            raise ManifestError("指标 {} 引用了未知证据 {}".format(name, evidence_name))
        observed = json_path(documents[evidence_name], dotted_path)
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise ManifestError("指标 {} 的证据值不是数字".format(name))
        metrics[str(name)] = float(observed)
    return metrics


def _readme(spec: Mapping[str, Any], base: Mapping[str, Any]) -> str:
    return """# {adapter_id}

场景：`{scene}`  
版本：`{version}`  
绑定基座：`{base_id}`

此目录是自动构建的场景 LoRA 发布包。加载前必须执行：

```bash
python -m edge_llm_factory validate-adapter --base /path/to/base_manifest.json --package .
```

包内只包含 PEFT 配置、safetensors LoRA、动作映射和哈希锁定的评测证据；
部署 GGUF 单独分发，并通过 `scene_adapter_manifest.json` 中的 SHA256 校验。
""".format(
        adapter_id=spec["adapter_id"],
        scene=spec["scene"],
        version=spec["version"],
        base_id=base["base_id"],
    )


def build_adapter_package(
    project_root: Path,
    base_manifest_path: Path,
    spec_path: Path,
    output_dir: Path,
    archive_path: Optional[Path] = None,
) -> Dict[str, Any]:
    project = project_root.resolve()
    base = validate_base_manifest(read_json_object(base_manifest_path.resolve()))
    spec = _validate_spec(read_json_object(spec_path.resolve()), base)
    adapter_source = _resolve(project, spec["adapter_source"], "adapter_source")
    action_source = _resolve(project, spec["action_mapping"], "action_mapping")
    deployment_source = _resolve(
        project, spec["deployment"].get("artifact"), "deployment.artifact"
    )
    if not adapter_source.is_dir():
        raise ManifestError("adapter_source 不是目录: {}".format(adapter_source))
    config_source = adapter_source / ADAPTER_CONFIG_NAME
    weights_source = adapter_source / ADAPTER_WEIGHTS_NAME
    config = read_json_object(config_source)
    action_mapping = read_json_object(action_source)
    validate_action_mapping(action_mapping, base)
    if action_mapping.get("scene") != spec["scene"]:
        raise ManifestError("package spec scene 与 action mapping 不一致")
    inspect_safetensors(weights_source)

    output = output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=str(output.parent)))
    try:
        shutil.copy2(config_source, temporary / ADAPTER_CONFIG_NAME)
        shutil.copy2(weights_source, temporary / ADAPTER_WEIGHTS_NAME)
        shutil.copy2(action_source, temporary / ACTION_MAP_NAME)
        evidence_records, evidence_documents = _copy_evidence(
            project, temporary, spec["evaluation"]["evidence"]
        )
        metric_sources = spec["evaluation"]["metric_sources"]
        metrics = _metric_values(evidence_documents, metric_sources)
        gate_results = validate_gates(metrics, spec["evaluation"]["gates"])
        deployment_record = _artifact_record(deployment_source, deployment_source.name)
        manifest = {
            "schema_version": ADAPTER_SCHEMA,
            "adapter_id": spec["adapter_id"],
            "scene": spec["scene"],
            "version": spec["version"],
            "base": {
                "base_id": base["base_id"],
                "fingerprint": base_fingerprint(base),
            },
            "adapter_artifact": {
                **_artifact_record(
                    temporary / ADAPTER_WEIGHTS_NAME, ADAPTER_WEIGHTS_NAME
                ),
                "format": "safetensors",
            },
            "lora": _adapter_lora_record(config),
            "input_contract": dict(spec["input_contract"]),
            "action_mapping": {
                "path": ACTION_MAP_NAME,
                "sha256": sha256_file(temporary / ACTION_MAP_NAME),
            },
            "training": dict(spec["training"]),
            "evaluation": {
                "evidence": evidence_records,
                "metrics": metrics,
                "metric_sources": metric_sources,
                "gates": spec["evaluation"]["gates"],
                "gate_results": gate_results,
            },
            "deployment": {
                "runtime": spec["deployment"].get("runtime"),
                "format": spec["deployment"].get("format"),
                "quantization": spec["deployment"].get("quantization"),
                "artifact_sha256": deployment_record["sha256"],
                "artifact_bytes": deployment_record["bytes"],
                "max_input_tokens": spec["deployment"].get("max_input_tokens"),
                "max_output_tokens": spec["deployment"].get("max_output_tokens"),
                "thinking": spec["deployment"].get("thinking"),
            },
        }
        write_json_object(temporary / MANIFEST_NAME, manifest)
        (temporary / "README.md").write_text(_readme(spec, base), encoding="utf-8")
        validation = validate_adapter_package(temporary, base_manifest_path, require_gates=True)
        if output.exists():
            if not output.is_dir() or not (output / MANIFEST_NAME).is_file():
                raise ManifestError("拒绝覆盖非适配器包目录: {}".format(output))
            shutil.rmtree(output)
        os.replace(str(temporary), str(output))
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)

    archive_record = None
    if archive_path is not None:
        archive = archive_path.resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            archive.unlink()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_obj:
            for path in sorted(item for item in output.rglob("*") if item.is_file()):
                zip_obj.write(path, Path(output.name) / path.relative_to(output))
        archive_record = {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        }
    return {
        "status": "built",
        "package": str(output),
        "validation": validation,
        "archive": archive_record,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建或校验标准场景 LoRA 发布包。")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--base", required=True)
    validate.add_argument("--package", required=True)
    validate.add_argument("--allow_failed_gates", action="store_true")
    build = subparsers.add_parser("build")
    build.add_argument("--project_root", default=".")
    build.add_argument("--base", required=True)
    build.add_argument("--spec", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--archive", default="")
    return parser


def main(argv: Optional[list] = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        result = validate_adapter_package(
            Path(args.package), Path(args.base), require_gates=not args.allow_failed_gates
        )
    else:
        result = build_adapter_package(
            project_root=Path(args.project_root),
            base_manifest_path=Path(args.base),
            spec_path=Path(args.spec),
            output_dir=Path(args.output),
            archive_path=Path(args.archive) if args.archive else None,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
