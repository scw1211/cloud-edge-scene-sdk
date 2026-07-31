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


def _generated_service_config(role, cloud_url, with_cloud_qwen9b):
    suffix = "_qwen9b" if with_cloud_qwen9b else ""
    template = (
        SCENE_ROOT
        / "deployment"
        / "full"
        / "{}_service{}.json".format(role, suffix)
    )
    config = _read_json(template)
    if role == "edge":
        if not cloud_url:
            raise ValueError("边缘节点必须提供 --cloud-url")
        config["cloud"]["base_url"] = str(cloud_url).rstrip("/")
    output = RUNTIME_ROOT / "generated" / "{}_service.json".format(role)
    _write_json(output, config)
    return output


def check_installation(role, llama_binary=None, device="cuda"):
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
        for record in catalog["downloaded_assets"].values():
            result.setdefault("downloaded_assets", []).append(
                _verify_asset(record)
            )
        registry = RUNTIME_ROOT / "edge_llm_release_store.json"
        if not registry.is_file():
            raise FileNotFoundError(
                "{} 不存在，请先运行 install_full_assets.py --edge".format(
                    registry
                )
            )
        from edge_llm_factory.release_store import ReleaseStore

        result["edge_release"] = ReleaseStore(registry).status(
            verify_active=True
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
    )
    processes = []
    stopping = {"value": False}

    def stop(_signum, _frame):
        stopping["value"] = True
        _stop_processes(processes)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        if args.role == "edge":
            check_installation("edge", args.llama_binary, args.device)
            release_command = [
                sys.executable,
                "-m",
                "edge_llm_factory",
                "serve-release",
                "--registry",
                str(RUNTIME_ROOT / "edge_llm_release_store.json"),
                "--runtime-config",
                str(
                    SCENE_ROOT
                    / "deployment"
                    / "full"
                    / "edge_llm_runtime.json"
                ),
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
                "--parallel",
                str(args.parallel),
                "--gpu-layers",
                str(args.gpu_layers if args.device == "cuda" else 0),
            ]
            processes.append(
                subprocess.Popen(
                    release_command,
                    cwd=str(SDK_ROOT),
                    start_new_session=True,
                )
            )
            _wait_health(
                "http://127.0.0.1:{}/health".format(args.llama_port),
                args.startup_timeout_seconds,
                processes[-1],
            )
            service_module = "cloud_edge_framework.edge_service"
            service_port = 18101
        else:
            check_installation("cloud")
            service_module = "cloud_edge_framework.cloud_service"
            service_port = 18100

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
                start_new_session=True,
            )
        )
        _wait_health(
            "http://127.0.0.1:{}/ready".format(service_port),
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


def main():
    parser = argparse.ArgumentParser(description="检查或启动真实交通云边节点")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--role", choices=("edge", "cloud"), required=True)
    check_parser.add_argument("--llama-binary")
    check_parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--role", choices=("edge", "cloud"), required=True)
    run_parser.add_argument("--cloud-url")
    run_parser.add_argument("--llama-binary")
    run_parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run_parser.add_argument("--llama-port", type=int, default=18190)
    run_parser.add_argument("--context-tokens", type=int, default=128)
    run_parser.add_argument("--threads", type=int, default=4)
    run_parser.add_argument("--parallel", type=int, default=2)
    run_parser.add_argument("--gpu-layers", type=int, default=99)
    run_parser.add_argument("--startup-timeout-seconds", type=float, default=90.0)
    run_parser.add_argument("--with-cloud-qwen9b", action="store_true")
    args = parser.parse_args()

    if args.command == "check":
        print(
            json.dumps(
                check_installation(
                    args.role,
                    args.llama_binary,
                    args.device,
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
