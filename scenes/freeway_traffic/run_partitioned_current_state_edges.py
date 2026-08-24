#!/usr/bin/env python3
"""在一块机器上以四套独立进程和持久化状态模拟预分配交通边缘节点。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.request import urlopen
import uuid


SCENE_ROOT = Path(__file__).resolve().parent
SDK_ROOT = SCENE_ROOT.parents[1]
DECIDE_PATH = "/api/v1/collaboration/decide"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _percentile(values: Sequence[float], percentile: float) -> float:
    data = sorted(float(value) for value in values)
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * float(percentile) / 100.0
    lower = int(position)
    upper = min(lower + 1, len(data) - 1)
    weight = position - lower
    return data[lower] * (1.0 - weight) + data[upper] * weight


def _summary(values: Iterable[float]) -> Dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        return {"count": 0}
    return {
        "count": len(data),
        "mean": round(statistics.fmean(data), 6),
        "p50": round(_percentile(data, 50), 6),
        "p95": round(_percentile(data, 95), 6),
        "max": round(max(data), 6),
    }


def _business_completion_ms(
    event: Mapping[str, Any],
    authoritative_final_ms: Any,
) -> Any:
    """Return the route-aware business completion point for one event."""

    policy_route = str(event.get("policy_route", "")).strip()
    if policy_route == "cloud_sync":
        return authoritative_final_ms
    if policy_route not in {"edge_only", "local_autonomy", "cloud_async"}:
        raise RuntimeError("unknown policy route: {!r}".format(policy_route))
    if bool(event.get("local_actions_authorized")):
        provisional_ms = event.get("input_to_provisional_ms")
        if provisional_ms is not None:
            return float(provisional_ms)
    return authoritative_final_ms


def _wait_until_ns(target_ns: int) -> None:
    while True:
        remaining = int(target_ns) - time.monotonic_ns()
        if remaining <= 0:
            return
        if remaining > 2_000_000:
            time.sleep((remaining - 1_000_000) / 1_000_000_000.0)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex((host, int(port))) != 0


def _wait_service_ready(
    endpoint: str,
    process: subprocess.Popen,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + float(timeout_seconds)
    last_error = "service did not answer"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                "isolated edge service exited with code {}".format(return_code)
            )
        try:
            with urlopen(endpoint + "/ready", timeout=0.5) as response:
                if int(response.status) == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = "{}: {}".format(type(exc).__name__, exc)
        time.sleep(0.05)
    raise TimeoutError("{} readiness timed out: {}".format(endpoint, last_error))


def _validate_cloud_evidence_pull_allowlist(
    cloud_url: str,
    edge_endpoints: Sequence[str],
    timeout_seconds: float = 2.0,
) -> None:
    """Fail before launch when a running cloud cannot callback every edge.

    The cloud may be an independently managed process, so this launcher never
    rewrites its SSRF allowlist.  A non-default port range requires the operator
    to update/restart the cloud configuration first.
    """

    protocol_url = (
        str(cloud_url).rstrip("/") + "/api/v1/collaboration/schema"
    )
    try:
        with urlopen(protocol_url, timeout=float(timeout_seconds)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "cannot verify cloud selective-evidence allowlist at {}: {}: {}".format(
                protocol_url, type(exc).__name__, exc
            )
        ) from exc
    selective = payload.get("selective_evidence_pull", {})
    if not isinstance(selective, dict) or selective.get("enabled") is not True:
        raise RuntimeError(
            "isolated edges require selective evidence pull enabled on the cloud"
        )
    allowed = selective.get("allowed_edge_base_urls", [])
    if not isinstance(allowed, list):
        raise RuntimeError("cloud selective-evidence allowlist is malformed")
    normalized_allowed = {str(value).rstrip("/") for value in allowed}
    required = {str(value).rstrip("/") for value in edge_endpoints}
    missing = sorted(required - normalized_allowed)
    if missing:
        raise RuntimeError(
            "cloud selective-evidence allowlist is missing {}; update the cloud "
            "evidence_pull.allowed_edge_base_urls and restart it before using "
            "this edge port range".format(", ".join(missing))
        )


def _stop_edge_services(services: Sequence[Mapping[str, Any]]) -> None:
    for service in reversed(list(services)):
        process = service["process"]
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 8.0
    for service in reversed(list(services)):
        process = service["process"]
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


def _launch_isolated_edge_services(
    project_root: Path,
    scene_root: Path,
    experiment_id: str,
    template_path: Path,
    cloud_url: str,
    port_base: int,
    startup_timeout_seconds: float,
) -> List[Dict[str, Any]]:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    if str(template.get("role", "")) != "edge":
        raise ValueError("isolated edge template must have role=edge")
    run_root = (
        scene_root / "runtime" / "partitioned_edge_services" / experiment_id
    ).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    definitions = []
    for partition_id in range(4):
        port = int(port_base) + partition_id
        if not _port_is_free("127.0.0.1", port):
            raise RuntimeError("isolated edge port {} is already in use".format(port))
        state_root = run_root / "edge_node_{}".format(partition_id)
        state_root.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(template)
        config.setdefault("listen", {})
        config["listen"].update({"host": "127.0.0.1", "port": port})
        config.setdefault("cloud", {})["base_url"] = str(cloud_url).rstrip("/")
        evidence_pull = config.get("evidence_pull")
        if isinstance(evidence_pull, dict) and evidence_pull.get("enabled") is True:
            # A capability is bound to the process-local cache.  Reusing the
            # template's port would route edge_node_1..3 to edge_node_0, where
            # the independently generated HMAC token must be rejected.
            evidence_pull["public_base_url"] = (
                "http://127.0.0.1:{}".format(port)
            )
        config["storage"] = {
            "outbox": str(state_root / "outbox.sqlite3"),
            "performance_profiles": str(state_root / "performance.json"),
            "feedback": str(state_root / "feedback.jsonl"),
            "idempotency": str(state_root / "idempotency.sqlite3"),
            "reviews": str(state_root / "reviews.sqlite3"),
            "monitoring": str(state_root / "monitoring.sqlite3"),
        }
        if isinstance(config.get("release_watch"), dict):
            config["release_watch"]["enabled"] = False
        config_path = state_root / "edge_service.json"
        log_path = state_root / "edge_service.log"
        _write_json(config_path, config)
        definitions.append(
            {
                "partition_id": partition_id,
                "port": port,
                "endpoint": "http://127.0.0.1:{}".format(port),
                "config_path": str(config_path),
                "log_path": str(log_path),
                "state_root": str(state_root),
            }
        )

    services = []
    try:
        for definition in definitions:
            log_handle = Path(definition["log_path"]).open("ab")
            service_environment = os.environ.copy()
            inherited_python_path = service_environment.get("PYTHONPATH", "")
            service_environment["PYTHONPATH"] = os.pathsep.join(
                value
                for value in (
                    str(scene_root),
                    str(project_root),
                    inherited_python_path,
                )
                if value
            )
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "cloud_edge_framework.edge_service",
                        "--project_root",
                        str(project_root),
                        "--config",
                        str(definition["config_path"]),
                    ],
                    cwd=str(project_root),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=service_environment,
                )
            finally:
                log_handle.close()
            service = {**definition, "process": process}
            services.append(service)
            _wait_service_ready(
                str(definition["endpoint"]),
                process,
                startup_timeout_seconds,
            )
        return services
    except BaseException:
        _stop_edge_services(services)
        raise


def _edge_worker(
    configuration: Mapping[str, Any],
    command_queue: Any,
    result_queue: Any,
) -> None:
    partition_id = int(configuration["partition_id"])
    client = None
    try:
        for path in (str(SDK_ROOT), str(SCENE_ROOT)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from benchmark_real_current_state_e2e import _PersistentJsonConnection
        from traffic_system.current_state_perception_runtime import (
            PartitionCurrentStateTrafficPerceptionRuntime,
        )
        from traffic_system.scene_event import traffic_event_from_output

        runtime = PartitionCurrentStateTrafficPerceptionRuntime(
            manifest_path=Path(str(configuration["manifest_path"])),
            partition_id=partition_id,
            rule_config_path=Path(str(configuration["rule_config_path"])),
            topology_path=Path(str(configuration["topology_path"])),
            split=str(configuration["split"]),
            top_k=int(configuration["top_k"]),
            verify_sha256=bool(configuration["verify_sha256"]),
        )
        client = _PersistentJsonConnection(
            str(configuration["edge_url"]),
            float(configuration["request_timeout_seconds"]),
        )
        result_queue.put(
            {
                "kind": "ready",
                "partition_id": partition_id,
                "pid": os.getpid(),
                "edge_id": "edge_node_{}".format(partition_id),
                "node_count": len(runtime.managed_node_ids),
                "managed_node_ids": runtime.managed_node_ids,
                "sample_count": runtime.sample_count,
                "load_latency_ms": runtime.load_latency_ms,
                "data_path": str(runtime.data_path),
            }
        )
        while True:
            command = command_queue.get()
            if command.get("kind") == "stop":
                return
            if command.get("kind") != "sample":
                raise ValueError("unknown worker command")
            sample_id = int(command["sample_id"])
            start_at_ns = int(command["start_at_ns"])
            start_at_epoch_ms = float(command["start_at_epoch_ms"])
            _wait_until_ns(start_at_ns)
            started_ns = time.monotonic_ns()
            perception = runtime.infer_sample(sample_id)
            if len(perception.events) != 1:
                raise RuntimeError("partition worker must emit exactly one event")
            native = copy.deepcopy(perception.events[0])
            native["sample_split"] = "{}_{}".format(
                runtime.split, command["experiment_id"]
            )
            native["event_id"] = "{}_{}".format(
                native["event_id"], command["experiment_id"]
            )
            native["aggregation_timeout_ms"] = int(
                command["aggregation_timeout_ms"]
            )
            envelope = traffic_event_from_output(native)
            body = _json_bytes({"event": envelope})
            encode_done_ns = time.monotonic_ns()
            request_started_ns = time.monotonic_ns()
            response, response_bytes, reused, attempts = client.post_json(
                DECIDE_PATH,
                body,
                {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Idempotency-Key": str(envelope["id"]),
                    "X-Trace-Id": "trace_" + str(envelope["id"]),
                    "Prefer": "return=minimal, respond-async",
                },
            )
            finished_ns = time.monotonic_ns()
            response_decision = response.get("final_decision", {})
            decision_status = str(response_decision.get("status", ""))
            response_metadata = response_decision.get("metadata", {})
            response_metadata = (
                dict(response_metadata)
                if isinstance(response_metadata, Mapping)
                else {}
            )
            action_authorization = response_metadata.get(
                "action_authorization", {}
            )
            action_authorization = (
                dict(action_authorization)
                if isinstance(action_authorization, Mapping)
                else {}
            )
            schedule = response.get("schedule", {})
            schedule = dict(schedule) if isinstance(schedule, Mapping) else {}
            data_plane = response.get("data_plane", {})
            data_plane = (
                dict(data_plane) if isinstance(data_plane, Mapping) else {}
            )
            summary_delivery = response.get("summary_delivery", {})
            summary_delivery = (
                dict(summary_delivery)
                if isinstance(summary_delivery, Mapping)
                else {}
            )
            response_review = response.get("review", {})
            response_review = (
                dict(response_review)
                if isinstance(response_review, Mapping)
                else {}
            )
            response_latency_ms = round(
                (finished_ns - start_at_ns) / 1_000_000.0, 6
            )
            result_queue.put(
                {
                    "kind": "sample",
                    "partition_id": partition_id,
                    "pid": os.getpid(),
                    "sample_id": sample_id,
                    "event_id": str(envelope["id"]),
                    "edge_id": str(envelope["edgeid"]),
                    "edge_url": str(configuration["edge_url"]),
                    "managed_node_ids": list(native["managed_node_ids"]),
                    "partition_data_preassigned": bool(
                        native.get("partition_data_preassigned")
                    ),
                    "start_at_ns": start_at_ns,
                    "start_at_epoch_ms": start_at_epoch_ms,
                    "started_ns": started_ns,
                    "finished_ns": finished_ns,
                    "start_offset_ms": round(
                        (started_ns - start_at_ns) / 1_000_000.0, 6
                    ),
                    "perception_ms": perception.perception_ms,
                    "encode_ms": round(
                        (encode_done_ns - started_ns) / 1_000_000.0, 6
                    ),
                    "http_wall_ms": round(
                        (finished_ns - request_started_ns) / 1_000_000.0, 6
                    ),
                    "input_to_response_ms": response_latency_ms,
                    "input_to_provisional_ms": (
                        response_latency_ms
                        if decision_status == "provisional"
                        else None
                    ),
                    "local_actions_authorized": bool(
                        action_authorization.get("all_actions_authorized", False)
                    ),
                    "action_authorization_present": bool(action_authorization),
                    "all_actions_authorized": bool(
                        action_authorization.get("all_actions_authorized", False)
                    ),
                    "immediate_action_types": list(
                        action_authorization.get("immediate_action_types", [])
                    ),
                    "deferred_action_types": list(
                        action_authorization.get("deferred_action_types", [])
                    ),
                    "cloud_confirmed": bool(
                        action_authorization.get("cloud_confirmed", False)
                    ),
                    "local_autonomy": bool(
                        response_metadata.get("local_autonomy", False)
                    ),
                    "policy_route": str(
                        data_plane.get(
                            "scheduler_selected_route",
                            schedule.get("route", ""),
                        )
                    ),
                    "policy_waits_for_cloud": bool(
                        data_plane.get(
                            "scheduler_selected_wait",
                            schedule.get("waits_for_cloud", False),
                        )
                    ),
                    "summary_delivery_required": bool(
                        summary_delivery.get("required", False)
                    ),
                    "summary_persistence_stage": str(
                        summary_delivery.get("persistence_stage", "")
                    ),
                    "review_state": str(response_review.get("state", "")),
                    "review_requested_route": str(
                        response_review.get("requested_route", "")
                    ),
                    "request_bytes": len(body),
                    "response_bytes": int(response_bytes),
                    "connection_reused": bool(reused),
                    "transport_attempts": int(attempts),
                    "route": str(response.get("final_decision", {}).get("route", "")),
                    "status": decision_status,
                }
            )
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(
            {
                "kind": "error",
                "partition_id": partition_id,
                "pid": os.getpid(),
                "error": "{}: {}".format(type(exc).__name__, exc),
            }
        )
    finally:
        if client is not None:
            client.close()


def _collect_messages(
    result_queue: Any,
    kind: str,
    count: int,
    timeout_seconds: float,
) -> List[Dict[str, Any]]:
    deadline = time.monotonic() + float(timeout_seconds)
    values = []
    while len(values) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "timed out waiting for {} {} messages".format(count, kind)
            )
        try:
            value = result_queue.get(timeout=remaining)
        except Empty as exc:
            raise TimeoutError("worker result queue timed out") from exc
        if value.get("kind") == "error":
            raise RuntimeError(
                "edge worker {} failed: {}".format(
                    value.get("partition_id"), value.get("error")
                )
            )
        if value.get("kind") != kind:
            raise RuntimeError("unexpected worker message: {}".format(value))
        values.append(value)
    return values


def _authoritative(review: Mapping[str, Any]) -> bool:
    return bool(
        str(review.get("state", "")) == "completed"
        and str(review.get("completion_stage", ""))
        not in {"partial_final", "local_only_timeout"}
        and isinstance(review.get("final_decision"), dict)
    )


def _four_edge_provisional_barrier_ms(
    events: Sequence[Mapping[str, Any]],
) -> float:
    """Return the last real provisional response, rejecting mixed final rows."""
    if len(events) != 4:
        raise ValueError("four-edge provisional barrier requires exactly four events")
    invalid = [
        "partition {} status={}".format(
            event.get("partition_id"), event.get("status")
        )
        for event in events
        if str(event.get("status", "")) != "provisional"
        or event.get("input_to_provisional_ms") is None
    ]
    if invalid:
        raise RuntimeError(
            "provisional timing cannot include a cloud final response: {}".format(
                ", ".join(invalid)
            )
        )
    return max(float(event["input_to_provisional_ms"]) for event in events)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="四个独立进程读取预切分 METIS 数据并并发提交"
    )
    parser.add_argument("--project-root", default=str(SDK_ROOT))
    parser.add_argument(
        "--manifest",
        default=str(
            SCENE_ROOT / "runtime" / "pems08_metis4_partitions" / "manifest.json"
        ),
    )
    parser.add_argument(
        "--edge-url",
        default="http://127.0.0.1:18101",
        help="共享服务模式使用的边缘接口",
    )
    parser.add_argument(
        "--launch-isolated-edge-services",
        action="store_true",
        help="为四个分区启动独立端口和独立持久化状态的边缘服务",
    )
    parser.add_argument(
        "--edge-service-config-template",
        default=str(SCENE_ROOT / "deployment" / "full" / "edge_service.json"),
    )
    parser.add_argument(
        "--cloud-url",
        help="独立服务模式必填；四个边缘服务共同连接的云端地址",
    )
    parser.add_argument("--edge-port-base", type=int, default=19101)
    parser.add_argument("--edge-service-startup-seconds", type=float, default=30.0)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sample-start", type=int, default=100)
    parser.add_argument("--sample-stop", type=int, default=110)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--request-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--worker-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--final-wait-seconds", type=float, default=15.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.025)
    parser.add_argument("--aggregation-timeout-ms", type=int, default=1000)
    parser.add_argument("--dispatch-lead-ms", type=float, default=30.0)
    parser.add_argument("--skip-sha256", action="store_true")
    parser.add_argument(
        "--output",
        default=str(
            SCENE_ROOT / "evidence" / "partitioned_edges_latest.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_start < 0 or args.sample_stop <= args.sample_start:
        raise ValueError("sample range is invalid")
    if args.dispatch_lead_ms <= 0:
        raise ValueError("dispatch lead must be positive")
    project_root = Path(args.project_root).resolve()
    scene_root = project_root / "scenes" / "freeway_traffic"
    for path in (str(project_root), str(scene_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from benchmark_real_current_state_e2e import _wait_reviews

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("partitions", [])
    partition_ids = sorted(int(record["partition_id"]) for record in records)
    if partition_ids != list(range(4)):
        raise ValueError("formal METIS edge run requires partitions 0,1,2,3")

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    command_queues = [context.Queue() for _ in partition_ids]
    processes = []
    edge_services: List[Dict[str, Any]] = []
    experiment_id = "partitioned-{}-{}".format(
        time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8]
    )
    if args.launch_isolated_edge_services:
        if not args.cloud_url:
            raise ValueError(
                "--cloud-url is required with --launch-isolated-edge-services"
            )
        if args.edge_port_base < 1 or args.edge_port_base + 3 > 65535:
            raise ValueError("edge port range is invalid")
        edge_template_path = Path(args.edge_service_config_template).resolve()
        edge_template = json.loads(
            edge_template_path.read_text(encoding="utf-8")
        )
        evidence_pull = edge_template.get("evidence_pull", {})
        if isinstance(evidence_pull, dict) and evidence_pull.get("enabled") is True:
            _validate_cloud_evidence_pull_allowlist(
                args.cloud_url,
                [
                    "http://127.0.0.1:{}".format(
                        args.edge_port_base + partition_id
                    )
                    for partition_id in partition_ids
                ],
                timeout_seconds=min(
                    2.0, max(0.1, args.edge_service_startup_seconds)
                ),
            )
        edge_services = _launch_isolated_edge_services(
            project_root=project_root,
            scene_root=scene_root,
            experiment_id=experiment_id,
            template_path=edge_template_path,
            cloud_url=args.cloud_url,
            port_base=args.edge_port_base,
            startup_timeout_seconds=args.edge_service_startup_seconds,
        )
        edge_urls = {
            int(service["partition_id"]): str(service["endpoint"])
            for service in edge_services
        }
        service_pids = [int(service["process"].pid) for service in edge_services]
        service_state_roots = [str(service["state_root"]) for service in edge_services]
        if len(set(service_pids)) != 4:
            raise RuntimeError("isolated edge services must have four distinct PIDs")
        if len(set(edge_urls.values())) != 4:
            raise RuntimeError("isolated edge services must have four distinct endpoints")
        if len(set(service_state_roots)) != 4:
            raise RuntimeError("isolated edge services must have four storage roots")
    else:
        edge_urls = {partition_id: str(args.edge_url) for partition_id in partition_ids}
    base_configuration = {
        "manifest_path": str(manifest_path),
        "rule_config_path": str(
            scene_root / "assets" / "models" / "current_state_perception_v1.json"
        ),
        "topology_path": str(
            scene_root / "assets" / "models" / "traffic_region_topology_metis4.json"
        ),
        "split": args.split,
        "top_k": args.top_k,
        "verify_sha256": not args.skip_sha256,
        "request_timeout_seconds": args.request_timeout_seconds,
    }
    try:
        for partition_id, command_queue in zip(partition_ids, command_queues):
            configuration = {
                **base_configuration,
                "partition_id": partition_id,
                "edge_url": edge_urls[partition_id],
            }
            process = context.Process(
                target=_edge_worker,
                args=(configuration, command_queue, result_queue),
                name="traffic-edge-node-{}".format(partition_id),
            )
            process.start()
            processes.append(process)
        ready = _collect_messages(
            result_queue,
            "ready",
            len(partition_ids),
            args.worker_timeout_seconds,
        )
        ready.sort(key=lambda value: int(value["partition_id"]))
        worker_pids = [int(value["pid"]) for value in ready]
        if len(set(worker_pids)) != len(partition_ids):
            raise RuntimeError("each METIS edge must run in a distinct process")
        assigned_nodes = sorted(
            int(node) for value in ready for node in value["managed_node_ids"]
        )
        if assigned_nodes != list(range(170)):
            raise RuntimeError("four worker assignments must cover 170 nodes once")

        samples = []
        for sample_id in range(args.sample_start, args.sample_stop):
            lead_ns = int(args.dispatch_lead_ms * 1_000_000.0)
            start_at_ns = time.monotonic_ns() + lead_ns
            start_at_epoch_ms = time.time() * 1000.0 + args.dispatch_lead_ms
            command = {
                "kind": "sample",
                "sample_id": sample_id,
                "start_at_ns": start_at_ns,
                "start_at_epoch_ms": start_at_epoch_ms,
                "experiment_id": experiment_id,
                "aggregation_timeout_ms": args.aggregation_timeout_ms,
            }
            for command_queue in command_queues:
                command_queue.put(command)
            events = _collect_messages(
                result_queue,
                "sample",
                len(partition_ids),
                args.worker_timeout_seconds,
            )
            events.sort(key=lambda value: int(value["partition_id"]))
            if [int(value["partition_id"]) for value in events] != partition_ids:
                raise RuntimeError("sample returned duplicate or missing partitions")
            event_ids = [str(value["event_id"]) for value in events]
            review_groups: Dict[str, List[str]] = {}
            for event in events:
                review_groups.setdefault(str(event["edge_url"]), []).append(
                    str(event["event_id"])
                )
            reviews: Dict[str, Dict[str, Any]] = {}
            review_observed_perf: Dict[str, float] = {}
            with ThreadPoolExecutor(max_workers=len(review_groups)) as executor:
                futures = [
                    executor.submit(
                        _wait_reviews,
                        endpoint,
                        endpoint_event_ids,
                        args.final_wait_seconds,
                        args.poll_interval_seconds,
                        args.request_timeout_seconds,
                    )
                    for endpoint, endpoint_event_ids in review_groups.items()
                ]
                for future in as_completed(futures):
                    endpoint_reviews, endpoint_observed = future.result()
                    reviews.update(endpoint_reviews)
                    review_observed_perf.update(endpoint_observed)
            sample_reviews = [
                reviews.get(str(event["event_id"]), {}) for event in events
            ]
            authoritative = all(_authoritative(review) for review in sample_reviews)
            observed_at = [
                float(review_observed_perf[str(event["event_id"])])
                for event, review in zip(events, sample_reviews)
                if _authoritative(review)
                and str(event["event_id"]) in review_observed_perf
            ]
            for event, review in zip(events, sample_reviews):
                event_id = str(event["event_id"])
                final_observed_ms = None
                if _authoritative(review) and event_id in review_observed_perf:
                    final_observed_ms = round(
                        (
                            float(review_observed_perf[event_id])
                            - start_at_ns / 1_000_000_000.0
                        )
                        * 1000.0,
                        6,
                    )
                event["input_to_authoritative_final_ms"] = final_observed_ms
                event["input_to_business_completion_ms"] = _business_completion_ms(
                    event,
                    final_observed_ms,
                )
            business_completion_values = [
                float(event["input_to_business_completion_ms"])
                for event in events
                if event.get("input_to_business_completion_ms") is not None
            ]
            samples.append(
                {
                    "sample_id": sample_id,
                    "start_at_epoch_ms": start_at_epoch_ms,
                    "events": events,
                    "dispatch_start_spread_ms": round(
                        (
                            max(int(value["started_ns"]) for value in events)
                            - min(int(value["started_ns"]) for value in events)
                        )
                        / 1_000_000.0,
                        6,
                    ),
                    "four_edge_provisional_ms": round(
                        _four_edge_provisional_barrier_ms(events),
                        6,
                    ),
                    "four_edge_business_completion_ms": (
                        round(max(business_completion_values), 6)
                        if len(business_completion_values) == len(partition_ids)
                        else None
                    ),
                    "authoritative_final": authoritative,
                    "four_edge_global_final_ms": (
                        round(
                            (
                                max(observed_at)
                                - start_at_ns / 1_000_000_000.0
                            )
                            * 1000.0,
                            6,
                        )
                        if len(observed_at) == len(partition_ids)
                        else None
                    ),
                }
            )

        event_rows = [event for sample in samples for event in sample["events"]]
        final_values = [
            float(sample["four_edge_global_final_ms"])
            for sample in samples
            if sample.get("four_edge_global_final_ms") is not None
        ]
        result = {
            "status": "passed"
            if len(final_values) == len(samples)
            else "incomplete",
            "experiment_id": experiment_id,
            "architecture": {
                "physical_host_count": 1,
                "logical_edge_process_count": len(worker_pids),
                "worker_pids": worker_pids,
                "edge_service_mode": (
                    "isolated_per_partition"
                    if args.launch_isolated_edge_services
                    else "shared"
                ),
                "shared_edge_framework_service": (
                    None if args.launch_isolated_edge_services else args.edge_url
                ),
                "edge_service_endpoints": [
                    edge_urls[partition_id] for partition_id in partition_ids
                ],
                "edge_service_pids": [
                    int(service["process"].pid) for service in edge_services
                ],
                "edge_service_state_roots": [
                    str(service["state_root"]) for service in edge_services
                ],
                "independent_edge_storage": bool(
                    args.launch_isolated_edge_services
                ),
                "partitioning": "frozen_metis_preassigned",
                "runtime_repartition": False,
                "each_process_emits_one_event_per_window": True,
            },
            "manifest": str(manifest_path),
            "workers": ready,
            "correctness": {
                "sample_count": len(samples),
                "event_count": len(event_rows),
                "distinct_worker_pid_count": len(set(worker_pids)),
                "distinct_edge_service_pid_count": len(
                    {int(service["process"].pid) for service in edge_services}
                ),
                "distinct_edge_endpoint_count": len(set(edge_urls.values())),
                "preassigned_event_count": sum(
                    bool(event["partition_data_preassigned"]) for event in event_rows
                ),
                "authoritative_sample_count": len(final_values),
                "provisional_event_count": sum(
                    str(event.get("status", "")) == "provisional"
                    for event in event_rows
                ),
                "locally_authorized_event_count": sum(
                    bool(event.get("local_actions_authorized"))
                    for event in event_rows
                ),
                "cloud_confirmed_before_action_event_count": sum(
                    not bool(event.get("local_actions_authorized"))
                    for event in event_rows
                ),
                "transport_retry_count": sum(
                    max(0, int(event["transport_attempts"]) - 1)
                    for event in event_rows
                ),
            },
            "latency_ms": {
                "per_edge_perception": _summary(
                    event["perception_ms"] for event in event_rows
                ),
                "per_edge_http": _summary(event["http_wall_ms"] for event in event_rows),
                "single_edge_input_to_response": _summary(
                    event["input_to_response_ms"] for event in event_rows
                ),
                "single_edge_input_to_provisional": _summary(
                    event["input_to_provisional_ms"] for event in event_rows
                    if event.get("input_to_provisional_ms") is not None
                ),
                "four_edge_dispatch_spread": _summary(
                    sample["dispatch_start_spread_ms"] for sample in samples
                ),
                "four_edge_input_to_provisional": _summary(
                    sample["four_edge_provisional_ms"] for sample in samples
                ),
                "single_edge_input_to_business_completion": _summary(
                    event["input_to_business_completion_ms"]
                    for event in event_rows
                    if event.get("input_to_business_completion_ms") is not None
                ),
                "four_edge_input_to_business_completion": _summary(
                    sample["four_edge_business_completion_ms"]
                    for sample in samples
                    if sample.get("four_edge_business_completion_ms") is not None
                ),
                "four_edge_input_to_global_final": _summary(final_values),
            },
            "timing_contract": {
                "provisional_clock": "worker_host_monotonic",
                "global_final_clock": "orchestrator_observed_monotonic",
                "cross_host_wall_clock_subtraction_used": False,
                "global_final_includes_result_poll_observation_delay": True,
                "edge_response_preference": "Prefer: return=minimal, respond-async",
                "provisional_metric_excludes_direct_final_responses": True,
                "business_completion_rule": (
                    "cloud_sync always finishes at authoritative final; locally "
                    "authorized edge_only/local_autonomy/cloud_async actions finish "
                    "at provisional; other actions finish at authoritative final"
                ),
            },
            "samples": samples,
        }
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({**result, "output": str(output)}, ensure_ascii=False, indent=2))
        if result["status"] != "passed":
            raise RuntimeError("not every four-edge sample reached authoritative final")
    finally:
        for command_queue in command_queues:
            command_queue.put({"kind": "stop"})
        for process in processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        _stop_edge_services(edge_services)


if __name__ == "__main__":
    main()
