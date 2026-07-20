"""用途：监督 llama-server，使其始终加载 release store 当前活动的 GGUF。"""

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from edge_llm_factory.contracts import ManifestError, read_json_object
from edge_llm_factory.providers import validate_runtime_config
from edge_llm_factory.release_store import ReleaseStore


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as file_obj:
            json.dump(dict(value), file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


class ActiveReleaseLlamaServer:
    def __init__(
        self,
        registry_path: Path,
        runtime_config_path: Path,
        binary: Path,
        host: str,
        port: int,
        context_tokens: int,
        threads: int,
        gpu_layers: int,
        poll_seconds: float,
        startup_timeout_seconds: float,
    ) -> None:
        self.store = ReleaseStore(registry_path)
        self.runtime_config_path = Path(runtime_config_path).resolve()
        self.binary = Path(binary).resolve()
        if not self.binary.is_file() or not os.access(str(self.binary), os.X_OK):
            raise ManifestError("llama-server 不存在或不可执行: {}".format(self.binary))
        self.host = str(host)
        self.port = int(port)
        self.context_tokens = int(context_tokens)
        self.threads = int(threads)
        self.gpu_layers = int(gpu_layers)
        self.poll_seconds = float(poll_seconds)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        if self.port <= 0 or self.port > 65535:
            raise ValueError("port must be in [1, 65535]")
        if min(self.context_tokens, self.threads) <= 0 or self.gpu_layers < 0:
            raise ValueError("context_tokens/threads must be positive and gpu_layers non-negative")
        if min(self.poll_seconds, self.startup_timeout_seconds) <= 0:
            raise ValueError("poll and startup timeout must be positive")
        self.process: Optional[subprocess.Popen] = None
        self.applied_revision = -1
        self.active_release_id: Optional[str] = None
        self.active_record: Optional[Dict[str, Any]] = None
        self.failed_revision: Optional[int] = None
        self.failed_error: Optional[str] = None
        self.stopping = False

    @property
    def endpoint(self) -> str:
        return "http://{}:{}".format(self.host, self.port)

    def command(self, artifact: Path) -> list:
        return [
            str(self.binary),
            "--model",
            str(artifact),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context_tokens),
            "--threads",
            str(self.threads),
            "--threads-batch",
            str(self.threads),
            "--batch-size",
            "16",
            "--ubatch-size",
            "16",
            "--parallel",
            "1",
            "--gpu-layers",
            str(self.gpu_layers),
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
            "--cache-ram",
            "0",
            "--ctx-checkpoints",
            "0",
            "--no-cache-prompt",
            "--poll",
            "100",
            "--poll-batch",
            "1",
            "--no-webui",
        ]

    def _runtime_config(self, release_id: str, artifact: Path) -> Dict[str, Any]:
        config = validate_runtime_config(read_json_object(self.runtime_config_path))
        if config["provider"] != "llama_cpp":
            raise ManifestError("serve-release 只管理 llama_cpp runtime")
        config["endpoint"] = self.endpoint
        config["model"] = artifact.name
        return config

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        last_error = "server did not answer"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError("llama-server exited with code {}".format(self.process.returncode))
            try:
                with urllib.request.urlopen(self.endpoint + "/health", timeout=0.5) as response:
                    if response.status == 200:
                        return
            except Exception as exc:  # noqa: BLE001
                last_error = "{}: {}".format(type(exc).__name__, exc)
            time.sleep(0.1)
        raise TimeoutError("llama-server startup timed out: {}".format(last_error))

    def stop_process(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)

    def _start_record(
        self,
        release_id: str,
        revision: int,
        record: Mapping[str, Any],
    ) -> Dict[str, Any]:
        artifact = Path(record["deployment_artifact"]["path"])
        self.stop_process()
        self.process = subprocess.Popen(self.command(artifact))
        try:
            self._wait_ready()
            _atomic_write(
                self.runtime_config_path,
                self._runtime_config(release_id, artifact),
            )
        except Exception:
            self.stop_process()
            raise
        self.active_release_id = release_id
        self.applied_revision = revision
        self.active_record = dict(record)
        return {
            "status": "active",
            "release_id": release_id,
            "revision": revision,
            "artifact": str(artifact),
            "endpoint": self.endpoint,
            "pid": self.process.pid,
        }

    def apply_current(self, force: bool = False) -> Dict[str, Any]:
        quick = self.store.status(verify_active=False)
        revision = int(quick["revision"])
        release_id = quick.get("active_release_id")
        if not release_id:
            raise ManifestError("release store 没有 active release")
        if (
            not force
            and revision == self.failed_revision
            and self.process is not None
            and self.process.poll() is None
        ):
            return {
                "status": "unchanged",
                "release_id": self.active_release_id,
                "revision": self.applied_revision,
                "rejected_release_id": release_id,
                "rejected_revision": revision,
                "candidate_error": self.failed_error,
                "pid": self.process.pid,
            }
        if (
            not force
            and revision == self.applied_revision
            and self.process is not None
            and self.process.poll() is None
        ):
            return {
                "status": "unchanged",
                "release_id": release_id,
                "revision": revision,
                "pid": self.process.pid,
            }
        verified = self.store.status(verify_active=True)
        candidate = verified["releases"][release_id]
        previous_id = self.active_release_id
        previous_revision = self.applied_revision
        previous_record = self.active_record
        try:
            result = self._start_record(str(release_id), revision, candidate)
            self.failed_revision = None
            self.failed_error = None
            return result
        except Exception as exc:
            if previous_id is not None and previous_record is not None:
                self._start_record(previous_id, previous_revision, previous_record)
            self.failed_revision = revision
            self.failed_error = "{}: {}".format(type(exc).__name__, exc)
            raise

    def run(self) -> None:
        print(json.dumps(self.apply_current(force=True), ensure_ascii=False))
        while not self.stopping:
            time.sleep(self.poll_seconds)
            if self.stopping:
                break
            try:
                result = self.apply_current()
                if result["status"] != "unchanged":
                    print(json.dumps(result, ensure_ascii=False))
            except Exception as exc:  # noqa: BLE001
                print(
                    json.dumps(
                        {"status": "error", "error": "{}: {}".format(type(exc).__name__, exc)},
                        ensure_ascii=False,
                    )
                )

    def stop(self) -> None:
        self.stopping = True
        self.stop_process()


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Run llama-server for the active Edge LLM release.")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18190)
    parser.add_argument("--context-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args(argv)
    supervisor = ActiveReleaseLlamaServer(
        registry_path=Path(args.registry),
        runtime_config_path=Path(args.runtime_config),
        binary=Path(args.binary),
        host=args.host,
        port=args.port,
        context_tokens=args.context_tokens,
        threads=args.threads,
        gpu_layers=args.gpu_layers,
        poll_seconds=args.poll_seconds,
        startup_timeout_seconds=args.startup_timeout_seconds,
    )
    if args.print_command:
        status = supervisor.store.status(verify_active=True)
        record = status["releases"][status["active_release_id"]]
        print(json.dumps(supervisor.command(Path(record["deployment_artifact"]["path"])), ensure_ascii=False))
        return

    def stop(_signum: int, _frame: Any) -> None:
        supervisor.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        supervisor.run()
    finally:
        supervisor.stop()


if __name__ == "__main__":
    main()
