"""用途：以两个独立进程一键验收框架初版的正常、重复、断网和恢复闭环。"""

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(
    base_url: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    body = None
    method = "GET"
    request_headers = {"Accept": "application/json", **(headers or {})}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
        method = "POST"
    request = Request(
        base_url + path,
        data=body,
        headers=request_headers,
        method=method,
    )
    with urlopen(request, timeout=2.0) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("framework endpoint returned a non-object")
    return value


def _wait_for(description: str, operation, predicate, timeout_seconds: float = 8.0):
    deadline = time.monotonic() + timeout_seconds
    last_error: Optional[Exception] = None
    last_value = None
    while time.monotonic() < deadline:
        try:
            last_value = operation()
            if predicate(last_value):
                return last_value
        except (HTTPError, URLError, OSError, ValueError) as exc:
            last_error = exc
        time.sleep(0.05)
    raise RuntimeError(
        "timed out waiting for {}: value={!r}, error={!r}".format(
            description, last_value, last_error
        )
    )


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _start_service(
    module: str,
    config_path: Path,
    project_root: Path,
    log_path: Path,
):
    log_file = log_path.open("w", encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            module,
            "--config",
            str(config_path),
            "--project_root",
            str(project_root),
        ],
        cwd=str(project_root),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    return process, log_file


def _stop_service(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


def _event(template: Dict[str, Any], event_id: str, edge_id: str, action_value: float):
    value = copy.deepcopy(template)
    value["id"] = event_id
    value["edgeid"] = edge_id
    value["source"] = "urn:edge:{}:fixture".format(edge_id)
    data = value["data"]
    if "entity_id" in data and "action_value" in data:
        data["entity_id"] = event_id + "-entity"
        data["region_id"] = edge_id + "-region"
        data["action_value"] = action_value
    elif "asset_id" in data and "proposed_limit_percent" in data:
        data["asset_id"] = event_id + "-asset"
        data["region_id"] = edge_id + "-region"
        data["shared_resource"] = "acceptance-shared-resource"
        data["proposed_limit_percent"] = int(round(action_value * 100.0))
    else:
        raise ValueError(
            "acceptance event template must expose either "
            "entity_id/action_value or asset_id/proposed_limit_percent"
        )
    return value


def run_acceptance(project_root: Path, output_path: Path) -> Dict[str, Any]:
    project_root = project_root.resolve()
    output_path = output_path.resolve()
    cloud_process = None
    edge_process = None
    log_files = []
    report: Dict[str, Any] = {
        "task": "framework_v0_role_separated_acceptance",
        "passed": False,
        "checks": {},
    }
    with tempfile.TemporaryDirectory(prefix="cloud-edge-v0-") as directory_name:
        directory = Path(directory_name)
        cloud_port = _free_port()
        edge_port = _free_port()
        cloud_url = "http://127.0.0.1:{}".format(cloud_port)
        edge_url = "http://127.0.0.1:{}".format(edge_port)
        plugin_candidates = [
            project_root / "deployment/framework/conformance_plugins.json",
            project_root / "deployment/framework/scene_plugins.json",
        ]
        plugin_config = next(
            (path for path in plugin_candidates if path.is_file()),
            plugin_candidates[-1],
        )
        cloud_config_path = directory / "cloud.json"
        edge_config_path = directory / "edge.json"
        cloud_log = directory / "cloud.log"
        edge_log = directory / "edge.log"
        cloud_config = {
            "schema_version": 1,
            "role": "cloud",
            "plugin_config": str(plugin_config),
            "listen": {"host": "127.0.0.1", "port": cloud_port},
            "storage": {
                "feedback": str(directory / "cloud_feedback.jsonl"),
                "idempotency": str(directory / "cloud_idempotency.sqlite3"),
                "artifacts": str(directory / "cloud_artifacts"),
                "aggregations": str(directory / "cloud_aggregations.sqlite3"),
            },
            "idempotency": {"ttl_seconds": 3600, "max_entries": 1000},
        }
        edge_config = {
            "schema_version": 1,
            "role": "edge",
            "plugin_config": str(plugin_config),
            "listen": {"host": "127.0.0.1", "port": edge_port},
            "storage": {
                "outbox": str(directory / "edge_outbox.sqlite3"),
                "performance_profiles": str(directory / "edge_performance.json"),
                "feedback": str(directory / "edge_feedback.jsonl"),
                "idempotency": str(directory / "edge_idempotency.sqlite3"),
                "reviews": str(directory / "edge_reviews.sqlite3"),
                "monitoring": str(directory / "edge_monitoring.sqlite3"),
            },
            "cloud": {
                "base_url": cloud_url,
                "timeout_seconds": 0.1,
                "max_attempts": 2,
                "retry_backoff_seconds": 0.01,
            },
            "network_probe": {
                "interval_seconds": 0.1,
                "window_size": 5,
                "failure_threshold": 1,
                "cloud_queue_ms": 0.1,
                "cloud_compute_ms": 0.5,
            },
            "replay": {
                "interval_seconds": 0.1,
                "batch_size": 16,
                "lease_seconds": 2.0,
                "max_backoff_seconds": 1.0,
            },
            "idempotency": {"ttl_seconds": 3600, "max_entries": 1000},
        }
        _write_json(cloud_config_path, cloud_config)
        _write_json(edge_config_path, edge_config)
        event_candidates = [
            project_root / "examples/framework_v0/event.json",
            project_root / "scene_plugin_template/sample_event.json",
        ]
        event_path = next(
            (path for path in event_candidates if path.is_file()),
            event_candidates[-1],
        )
        template = json.loads(event_path.read_text(encoding="utf-8"))
        try:
            cloud_process, cloud_handle = _start_service(
                "cloud_edge_framework.cloud_service",
                cloud_config_path,
                project_root,
                cloud_log,
            )
            log_files.append(cloud_handle)
            _wait_for(
                "cloud readiness",
                lambda: _request(cloud_url, "/ready"),
                lambda value: value.get("ready") is True,
            )
            edge_process, edge_handle = _start_service(
                "cloud_edge_framework.edge_service",
                edge_config_path,
                project_root,
                edge_log,
            )
            log_files.append(edge_handle)
            _wait_for(
                "edge cloud probe",
                lambda: _request(edge_url, "/health"),
                lambda value: value.get("cloud_available") is True,
            )

            normal_payload = {
                "event": _event(template, "conformance-normal-001", "edge-1", 0.4)
            }
            first = _request(edge_url, "/api/v1/collaboration/decide", normal_payload)
            duplicate = _request(
                edge_url, "/api/v1/collaboration/decide", normal_payload
            )
            report["checks"]["normal_cloud_sync"] = (
                first["final_decision"]["route"] == "cloud_sync"
            )
            report["checks"]["edge_idempotency"] = (
                not first["idempotency_replay"]
                and duplicate["idempotency_replay"]
                and first["trace_id"] == duplicate["trace_id"]
            )

            internal_payload = {"event": first["event"]}
            idempotency_headers = {"Idempotency-Key": "acceptance-cloud-request"}
            cloud_first = _request(
                cloud_url,
                "/api/v1/collaboration/cloud-decision",
                internal_payload,
                idempotency_headers,
            )
            cloud_duplicate = _request(
                cloud_url,
                "/api/v1/collaboration/cloud-decision",
                internal_payload,
                idempotency_headers,
            )
            report["checks"]["cloud_idempotency"] = (
                not cloud_first["idempotency_replay"]
                and cloud_duplicate["idempotency_replay"]
            )

            try:
                _request(
                    edge_url,
                    "/api/v1/collaboration/cloud-decision",
                    {"event": first["event"]},
                )
                role_status = 200
            except HTTPError as exc:
                role_status = exc.code
            report["checks"]["role_endpoint_isolation"] = role_status == 404

            _stop_service(cloud_process)
            cloud_process = None
            _wait_for(
                "edge outage detection",
                lambda: _request(edge_url, "/health"),
                lambda value: value.get("cloud_available") is False,
            )
            outage_results = []
            for index, action_value in enumerate((0.2, 0.8), start=1):
                outage_results.append(
                    _request(
                        edge_url,
                        "/api/v1/collaboration/decide",
                        {
                            "event": _event(
                                template,
                                "conformance-outage-00{}".format(index),
                                "edge-{}".format(index),
                                action_value,
                            )
                        },
                    )
                )
            outage_health = _request(edge_url, "/health")
            report["checks"]["outage_local_autonomy"] = all(
                item["final_decision"]["route"] == "local_autonomy"
                for item in outage_results
            )
            report["checks"]["outbox_persisted"] = (
                outage_health["outbox"]["active"] == 2
            )

            cloud_process, cloud_handle = _start_service(
                "cloud_edge_framework.cloud_service",
                cloud_config_path,
                project_root,
                cloud_log,
            )
            log_files.append(cloud_handle)
            _wait_for(
                "cloud restart",
                lambda: _request(cloud_url, "/ready"),
                lambda value: value.get("ready") is True,
            )
            recovered = _wait_for(
                "automatic outbox replay",
                lambda: _request(edge_url, "/health"),
                lambda value: (
                    value.get("cloud_available") is True
                    and value["outbox"]["active"] == 0
                ),
            )
            replay_result = recovered["replay"]["last_result"]
            coordination = replay_result.get("coordination") or {}
            report["checks"]["automatic_replay"] = (
                replay_result.get("completed") == 2
                and recovered["outbox"]["states"]["completed"] >= 2
            )
            report["checks"]["conflict_resolution"] = (
                coordination.get("initial_conflict_count", 0) >= 1
                and coordination.get("residual_conflict_count") == 0
            )
            report["edge_health"] = recovered
            report["edge_metrics"] = _request(
                edge_url, "/api/v1/framework/metrics"
            )
            report["cloud_metrics"] = _request(
                cloud_url, "/api/v1/framework/metrics"
            )
            report["passed"] = all(report["checks"].values())
        except Exception as exc:  # noqa: BLE001
            report["error"] = "{}: {}".format(type(exc).__name__, exc)
        finally:
            _stop_service(edge_process)
            _stop_service(cloud_process)
            for file_obj in log_files:
                file_obj.close()
            report["logs"] = {
                "edge": edge_log.read_text(encoding="utf-8", errors="replace")
                if edge_log.exists()
                else "",
                "cloud": cloud_log.read_text(encoding="utf-8", errors="replace")
                if cloud_log.exists()
                else "",
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the role-separated framework acceptance."
    )
    parser.add_argument(
        "--project_root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--output",
        default="results/framework/framework_v0_acceptance.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    report = run_acceptance(root, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
