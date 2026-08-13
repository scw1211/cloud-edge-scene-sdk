"""用途：从独立 SDK 配置、检查并启动真实交通云端或边缘节点。"""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen


SCENE_ROOT = Path(__file__).resolve().parent
SDK_ROOT = SCENE_ROOT.parents[1]
RUNTIME_ROOT = SCENE_ROOT / "runtime"
CATALOG_PATH = SCENE_ROOT / "asset_catalog.json"


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError("{} 必须是 JSON 对象".format(path))
    return value


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _identity(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(8 * 1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _verify_asset(record):
    path = SCENE_ROOT / str(record["file"])
    if not path.is_file():
        raise FileNotFoundError(path)
    size, digest = _identity(path)
    if size != int(record["bytes"]) or digest != str(record["sha256"]):
        raise ValueError("{} 的大小或 SHA-256 不匹配".format(path))
    return {
        "path": str(path.relative_to(SDK_ROOT)),
        "bytes": size,
        "sha256": digest,
    }


def _python_dependencies(role):
    names = {
        "jsonschema": "jsonschema",
        "joblib": "joblib",
        "numpy": "numpy",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
    }
    if role == "edge":
        names["torch"] = "torch"
    versions = {}
    for import_name, distribution_name in names.items():
        __import__(import_name)
        versions[import_name] = metadata.version(distribution_name)
    return versions


def _llama_version(binary):
    completed = subprocess.run(
        [str(binary), "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=15,
    )
    first_line = completed.stdout.strip().splitlines()
    if completed.returncode != 0 or not first_line:
        raise RuntimeError("llama-server --version 执行失败")
    return first_line[0]


def _service_config_template(role, with_cloud_qwen9b, service_config=None):
    if service_config:
        if with_cloud_qwen9b:
            raise ValueError(
                "--service-config 不能与 --with-cloud-qwen9b 同时使用"
            )
        template = Path(str(service_config)).expanduser().resolve()
    else:
        suffix = "_qwen9b" if with_cloud_qwen9b else ""
        template = (
            SCENE_ROOT
            / "deployment"
            / "full"
            / "{}_service{}.json".format(role, suffix)
        )
    if not template.is_file():
        raise FileNotFoundError("服务配置模板不存在: {}".format(template))
    return template


def _generated_service_config(
    role,
    cloud_url,
    with_cloud_qwen9b,
    service_config=None,
):
    template = _service_config_template(
        role,
        with_cloud_qwen9b,
        service_config=service_config,
    )
    config = _read_json(template)
    if config.get("role") != role:
        raise ValueError(
            "服务配置角色不匹配: {} != {}".format(config.get("role"), role)
        )
    if role == "edge":
        if not cloud_url:
            raise ValueError("边缘节点必须提供 --cloud-url")
        cloud = config.get("cloud")
        if not isinstance(cloud, dict):
            raise ValueError("边缘服务配置缺少 cloud 对象")
        cloud["base_url"] = str(cloud_url).rstrip("/")
    output = RUNTIME_ROOT / "generated" / "{}_service.json".format(role)
    _write_json(output, config)
    return output


def _service_readiness_url(config_path):
    config = _read_json(config_path)
    listen = config.get("listen")
    if not isinstance(listen, dict):
        raise ValueError("服务配置缺少 listen 对象")
    port = listen.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("listen.port 必须是 1 到 65535 的整数")
    host = str(listen.get("host", "")).strip().lower()
    if host in {"0.0.0.0", "127.0.0.1", "localhost"}:
        health_host = "127.0.0.1"
    elif host in {"::", "[::]", "::1", "[::1]"}:
        health_host = "[::1]"
    else:
        raise ValueError(
            "listen.host 必须是本机回环或通配地址，不能对远端地址执行就绪探测"
        )
    return "http://{}:{}/ready".format(health_host, port)


def _project_path(raw_path):
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = SDK_ROOT / path
    return path.resolve()


def _service_release_registry(config_path):
    """Resolve the watcher store with the same project-root rules as service."""
    config = _read_json(config_path)
    release_watch = config.get("release_watch")
    if not isinstance(release_watch, dict):
        raise ValueError("边缘服务配置缺少 release_watch")
    raw_registry = release_watch.get("registry")
    if not isinstance(raw_registry, str) or not raw_registry.strip():
        raise ValueError("edge service release_watch.registry 无效")
    return _project_path(raw_registry)


def _verify_runtime_outputs_match_service(config_path, runtime_outputs):
    """Fail closed when the service would read a stale scene runtime file."""
    service = _read_json(config_path)
    plugin_config_raw = service.get("plugin_config")
    if not isinstance(plugin_config_raw, str) or not plugin_config_raw.strip():
        raise ValueError("边缘服务配置缺少 plugin_config")
    plugin_config_path = _project_path(plugin_config_raw)
    plugins_value = _read_json(plugin_config_path).get("plugins")
    if not isinstance(plugins_value, list):
        raise ValueError("场景插件配置缺少 plugins 数组")
    expected_paths = []
    for plugin in plugins_value:
        if not isinstance(plugin, dict) or plugin.get("enabled", True) is not True:
            continue
        options = plugin.get("options", {})
        if not isinstance(options, dict):
            raise ValueError("场景插件 options 必须是对象")
        if str(options.get("edge_llm_mode", "disabled")).strip().lower() == "disabled":
            continue
        runtime_path = options.get("edge_llm_runtime_config_path")
        if not isinstance(runtime_path, str) or not runtime_path.strip():
            raise ValueError("启用 Edge LLM 的插件缺少 runtime 路径")
        expected_paths.append(_project_path(runtime_path))
    if len(expected_paths) != len(set(expected_paths)):
        raise ValueError("启用的多场景 Edge LLM 禁止共用同一 runtime 文件")
    expected = set(expected_paths)
    declared = {Path(item["output"]).resolve() for item in runtime_outputs}
    if declared != expected:
        raise ValueError(
            "runtime output 与启用的场景插件不一致: declared={}, expected={}".format(
                sorted(str(path) for path in declared),
                sorted(str(path) for path in expected),
            )
        )
    return {
        "status": "matched",
        "runtime_outputs": sorted(str(path) for path in declared),
    }


def _child_environment():
    """Build a deterministic child import path without mutating this process."""
    environment = dict(os.environ)
    required = [SDK_ROOT, SCENE_ROOT]
    industrial_scene = SDK_ROOT / "scenes" / "industrial_anomaly"
    if industrial_scene.is_dir():
        required.append(industrial_scene)

    entries = []
    for path in required:
        value = str(path.resolve())
        if value not in entries:
            entries.append(value)
    for raw in environment.get("PYTHONPATH", "").split(os.pathsep):
        raw = raw.strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        value = str(path)
        if value not in entries:
            entries.append(value)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def _positive_int(raw):
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def _verify_edge_downloaded_assets(catalog, edge_release):
    """Verify non-model downloads and bind the model to the active release.

    ``edge_qwen_gguf`` is the legacy, statically catalogued merged model.  A
    newer release may intentionally serve a different immutable GGUF (for
    example a clean base with request-selected LoRAs).  ``ReleaseStore.status``
    has already verified that active deployment artifact and is therefore the
    authoritative model check.  Requiring the superseded catalog file as well
    would turn a retired 630 MB model into an accidental startup dependency.
    """
    active_release_id = edge_release.get("active_release_id")
    releases = edge_release.get("releases")
    if not isinstance(active_release_id, str) or not isinstance(releases, dict):
        raise ValueError("active Edge LLM release metadata is incomplete")
    active = releases.get(active_release_id)
    if not isinstance(active, dict):
        raise ValueError("active Edge LLM release record is missing")
    deployment = active.get("deployment_artifact")
    if not isinstance(deployment, dict):
        raise ValueError("active Edge LLM deployment artifact is missing")

    results = []
    for name, record in catalog["downloaded_assets"].items():
        if name == "edge_qwen_gguf":
            results.append(
                {
                    "catalog_asset": name,
                    "status": "verified_by_active_release",
                    "release_id": active_release_id,
                    "path": deployment["path"],
                    "bytes": deployment["bytes"],
                    "sha256": deployment["sha256"],
                }
            )
            continue
        if record.get("startup_required", True) is False:
            results.append(
                {
                    "catalog_asset": name,
                    "status": "not_required_for_service_startup",
                    "path": str(record["file"]),
                }
            )
            continue
        verified = _verify_asset(record)
        verified["catalog_asset"] = name
        results.append(verified)
    return results


def check_installation(
    role,
    llama_binary=None,
    device="cuda",
    llama_registry=None,
):
    if sys.version_info < (3, 8):
        raise RuntimeError("需要 Python 3.8 或更高版本")
    catalog = _read_json(CATALOG_PATH)
    assets = [
        _verify_asset(record)
        for record in catalog["embedded_assets"].values()
    ]
    result = {
        "status": "ready",
        "role": role,
        "python": sys.version.split()[0],
        "dependencies": _python_dependencies(role),
        "embedded_assets": assets,
    }
    if role == "edge":
        registry = (
            Path(llama_registry).expanduser().resolve()
            if llama_registry
            else RUNTIME_ROOT / "edge_llm_release_store.json"
        )
        if not registry.is_file():
            raise FileNotFoundError(
                "{} 不存在，请先运行 install_full_assets.py --edge".format(
                    registry
                )
            )
        from edge_llm_factory.release_store import ReleaseStore

        edge_release = ReleaseStore(registry).status(
            verify_active=True
        )
        result["edge_release"] = edge_release
        result["downloaded_assets"] = _verify_edge_downloaded_assets(
            catalog, edge_release
        )
        binary = Path(str(llama_binary or "")).expanduser().resolve()
        if not binary.is_file() or not os.access(str(binary), os.X_OK):
            raise FileNotFoundError("llama-server 不存在或不可执行: {}".format(binary))
        result["llama_server"] = {
            "path": str(binary),
            "version": _llama_version(binary),
        }
        import torch

        cuda_available = bool(torch.cuda.is_available())
        result["torch"] = {
            "version": str(torch.__version__),
            "cuda_available": cuda_available,
        }
        if device == "cuda" and not cuda_available:
            raise RuntimeError(
                "当前 Python 环境的 torch 无法使用 CUDA；"
                "请安装与本机 JetPack 对应的 NVIDIA PyTorch"
            )
    return result


def _wait_health(url, timeout_seconds, process):
    deadline = time.monotonic() + timeout_seconds
    last_error = "服务没有响应"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "服务在健康检查前退出，返回码 {}".format(process.returncode)
            )
        try:
            with urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = "{}: {}".format(type(exc).__name__, exc)
        time.sleep(0.1)
    raise TimeoutError("{} 健康检查超时：{}".format(url, last_error))


def _wait_startup_gate(path, timeout_seconds, process):
    """Wait for the exact serve-release process to publish its atomic receipt."""
    receipt_path = Path(path)
    deadline = time.monotonic() + timeout_seconds
    last_error = "启动门禁尚未发布回执"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "serve-release 在启动门禁前退出，返回码 {}".format(
                    process.returncode
                )
            )
        try:
            receipt = _read_json(receipt_path)
            if receipt.get("schema_version") != "edge-llm-startup-gate-receipt/v1":
                raise ValueError("启动门禁回执 schema_version 无效")
            if receipt.get("supervisor_pid") != process.pid:
                raise ValueError("启动门禁回执来自旧 supervisor")
            if receipt.get("status") != "passed":
                raise RuntimeError(
                    "启动门禁未通过: {}".format(receipt.get("status"))
                )
            gate = receipt.get("gate")
            if not isinstance(gate, dict) or gate.get("status") != "passed":
                raise ValueError("启动门禁回执缺少 passed gate")
            return receipt
        except Exception as exc:  # noqa: BLE001
            last_error = "{}: {}".format(type(exc).__name__, exc)
        time.sleep(0.1)
    raise TimeoutError("启动门禁等待超时：{}".format(last_error))


def _stop_processes(processes):
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 8.0
    for process in reversed(processes):
        if process.poll() is not None:
            continue
        remaining = max(0.1, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)


def run_node(args):
    config_path = _generated_service_config(
        args.role,
        args.cloud_url,
        args.with_cloud_qwen9b,
        service_config=getattr(args, "service_config", None),
    )
    service_readiness_url = _service_readiness_url(config_path)
    child_environment = _child_environment()
    service_environment = dict(child_environment)
    service_environment.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    processes = []
    llama_startup_gate = None
    stopping = {"value": False}

    def stop(_signum, _frame):
        stopping["value"] = True
        _stop_processes(processes)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        if args.role == "edge":
            llama_registry = (
                Path(args.llama_registry).expanduser().resolve()
                if getattr(args, "llama_registry", None)
                else RUNTIME_ROOT / "edge_llm_release_store.json"
            )
            service_registry = _service_release_registry(config_path)
            if llama_registry.resolve() != service_registry:
                raise ValueError(
                    "--llama-registry 与 service release_watch.registry 不一致: "
                    "{} != {}".format(llama_registry.resolve(), service_registry)
                )
            check_installation(
                "edge",
                args.llama_binary,
                args.device,
                llama_registry=llama_registry,
            )
            startup_probe_path = None
            if getattr(args, "llama_startup_probes", None):
                startup_probe_path = Path(
                    args.llama_startup_probes
                ).expanduser().resolve()
                from edge_llm_factory.serve_release import load_startup_probe_config

                load_startup_probe_config(startup_probe_path)
            runtime_output_paths = []
            runtime_output_descriptors = []
            if getattr(args, "llama_runtime_output", None):
                from edge_llm_factory.serve_release import (
                    load_runtime_output_descriptor,
                )

                for raw_path in args.llama_runtime_output:
                    descriptor_path = Path(raw_path).expanduser().resolve()
                    descriptor = load_runtime_output_descriptor(descriptor_path)
                    runtime_output_paths.append(descriptor_path)
                    runtime_output_descriptors.append(descriptor)
                _verify_runtime_outputs_match_service(
                    config_path, runtime_output_descriptors
                )
            startup_gate_receipt = (
                RUNTIME_ROOT
                / "generated"
                / "llama_startup_gate_{}.json".format(args.llama_port)
            )
            if startup_gate_receipt.exists():
                startup_gate_receipt.unlink()
            release_command = [
                sys.executable,
                "-m",
                "edge_llm_factory",
                "serve-release",
                "--registry",
                str(llama_registry),
                "--binary",
                str(Path(args.llama_binary).expanduser().resolve()),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.llama_port),
                "--context-tokens",
                str(args.context_tokens),
                "--threads",
                str(args.threads),
                "--batch-size",
                str(getattr(args, "llama_batch_size", 16)),
                "--ubatch-size",
                str(getattr(args, "llama_ubatch_size", 16)),
                "--parallel",
                str(args.parallel),
                "--gpu-layers",
                str(args.gpu_layers if args.device == "cuda" else 0),
                "--startup-gate-receipt",
                str(startup_gate_receipt),
            ]
            if runtime_output_paths:
                for descriptor_path in runtime_output_paths:
                    release_command.extend(
                        ["--runtime-output", str(descriptor_path)]
                    )
            else:
                release_command.extend(
                    [
                        "--runtime-config",
                        str(
                            SCENE_ROOT
                            / "deployment"
                            / "full"
                            / "edge_llm_runtime.json"
                        ),
                    ]
                )
            if getattr(args, "llama_no_mmap", False):
                release_command.append("--no-mmap")
            if startup_probe_path is not None:
                release_command.extend(
                    ["--startup-probe-config", str(startup_probe_path)]
                )
            for adapter in args.llama_lora_adapter:
                release_command.extend(["--lora-adapter", str(Path(adapter).resolve())])
            release_environment = dict(service_environment)
            if getattr(args, "disable_cuda_graphs", False):
                release_environment["GGML_CUDA_DISABLE_GRAPHS"] = "1"
            processes.append(
                subprocess.Popen(
                    release_command,
                    cwd=str(SDK_ROOT),
                    env=release_environment,
                    start_new_session=True,
                )
            )
            llama_startup_gate = _wait_startup_gate(
                startup_gate_receipt,
                args.startup_timeout_seconds,
                processes[-1],
            )
            service_module = "cloud_edge_framework.edge_service"
        else:
            check_installation("cloud")
            service_module = "cloud_edge_framework.cloud_service"

        service_command = [
            sys.executable,
            "-m",
            service_module,
            "--project_root",
            str(SDK_ROOT),
            "--config",
            str(config_path),
        ]
        processes.append(
            subprocess.Popen(
                service_command,
                cwd=str(SDK_ROOT),
                env=service_environment,
                start_new_session=True,
            )
        )
        _wait_health(
            service_readiness_url,
            args.startup_timeout_seconds,
            processes[-1],
        )
        print(
            json.dumps(
                {
                    "status": "running",
                    "role": args.role,
                    "service_config": str(config_path),
                    "pids": [process.pid for process in processes],
                    "llama_startup_gate": llama_startup_gate,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        while not stopping["value"]:
            failed = next(
                (process for process in processes if process.poll() is not None),
                None,
            )
            if failed is not None:
                raise RuntimeError(
                    "节点子进程异常退出，返回码 {}".format(failed.returncode)
                )
            time.sleep(0.5)
    finally:
        _stop_processes(processes)


def main(argv=None):
    parser = argparse.ArgumentParser(description="检查或启动真实交通云边节点")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--role", choices=("edge", "cloud"), required=True)
    check_parser.add_argument("--llama-binary")
    check_parser.add_argument(
        "--llama-registry",
        help="显式 release store；用于隔离候选旁路，默认使用正式 runtime store。",
    )
    check_parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--role", choices=("edge", "cloud"), required=True)
    run_parser.add_argument("--cloud-url")
    run_parser.add_argument("--llama-binary")
    run_parser.add_argument(
        "--llama-registry",
        help="显式传给 serve-release 的隔离 release store 路径。",
    )
    run_parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run_parser.add_argument("--llama-port", type=int, default=18190)
    run_parser.add_argument("--context-tokens", type=int, default=128)
    run_parser.add_argument("--threads", type=int, default=4)
    run_parser.add_argument(
        "--llama-batch-size",
        type=_positive_int,
        default=16,
        help="llama-server 逻辑 batch 上限（默认 16）",
    )
    run_parser.add_argument(
        "--llama-ubatch-size",
        type=_positive_int,
        default=16,
        help="llama-server 物理 micro-batch 上限（默认 16）",
    )
    run_parser.add_argument("--parallel", type=int, default=1)
    run_parser.add_argument("--gpu-layers", type=int, default=99)
    run_parser.add_argument(
        "--llama-no-mmap",
        action="store_true",
        help="通过 serve-release 将 --no-mmap 传给 llama-server。",
    )
    run_parser.add_argument(
        "--disable-cuda-graphs",
        action="store_true",
        help=(
            "仅为 serve-release/llama-server 子进程设置 "
            "GGML_CUDA_DISABLE_GRAPHS=1"
        ),
    )
    service_config_group = run_parser.add_mutually_exclusive_group()
    service_config_group.add_argument(
        "--service-config",
        help=(
            "显式服务配置模板路径；生成运行配置时仍会覆盖 edge cloud.base_url。"
            "未提供时保持 deployment/full 下的默认模板。"
        ),
    )
    service_config_group.add_argument("--with-cloud-qwen9b", action="store_true")
    run_parser.add_argument(
        "--llama-lora-adapter",
        action="append",
        default=[],
        help=(
            "预载 LoRA GGUF，可重复；顺序决定请求级 lora_adapter.id。"
            "未提供时保持现有单 GGUF 生产路径。"
        ),
    )
    run_parser.add_argument(
        "--llama-startup-probes",
        help=(
            "按 release_id 绑定模型/LoRA SHA 的声明式启动 probe JSON。"
            "提供后每个运行时 Adapter 必须通过受约束单 token 预热。"
        ),
    )
    run_parser.add_argument(
        "--llama-runtime-output",
        action="append",
        default=[],
        help=(
            "随 active release 原子生成的场景 runtime output descriptor JSON；"
            "每个场景重复一次。未提供时保持旧单 runtime 路径。"
        ),
    )
    run_parser.add_argument("--startup-timeout-seconds", type=float, default=90.0)
    args = parser.parse_args(argv)

    if args.command == "check":
        print(
            json.dumps(
                check_installation(
                    args.role,
                    args.llama_binary,
                    args.device,
                    llama_registry=args.llama_registry,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.role == "edge" and not args.llama_binary:
        raise SystemExit("边缘节点必须提供 --llama-binary")
    run_node(args)


if __name__ == "__main__":
    main()
