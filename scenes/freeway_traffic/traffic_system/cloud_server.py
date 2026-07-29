"""用途：提供接收边缘事件并返回全局交通决策的云端 HTTP 服务。"""

import argparse
import json
import queue
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, urlsplit

from traffic_system.cloud_llm_review import DurableReviewQueue
from traffic_system.cloud_coordinator import cloud_decisions, load_cloud_model
from traffic_system.conflict_coordinator import coordinate_globally
from traffic_system.decision_utils import save_json
from traffic_system.graph_partition import build_undirected_neighbor_map
from traffic_system.infer_joint_risk_astgcn import load_adjacency, load_config
from traffic_system.policy_store import load_json as load_policy_json, verify_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cloud-side HTTP decision service for traffic edge events.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument("--port", type=int, default=18080, help="Bind port.")
    parser.add_argument(
        "--endpoint",
        default="/api/v1/traffic/decision",
        help="HTTP endpoint for edge event decision requests.",
    )
    parser.add_argument("--coordinate_endpoint", default="/api/v1/traffic/coordinate")
    parser.add_argument("--policy_endpoint", default="/api/v1/policy")
    parser.add_argument("--artifact_endpoint", default="/api/v1/artifacts")
    parser.add_argument("--policy_bundle", default="deployment/policy/current_policy.json")
    parser.add_argument("--config", default="configurations/PEMS08_astgcn.conf")
    parser.add_argument(
        "--coordinator_model",
        default="models/cloud_coordinator_future_calibrated.joblib",
    )
    parser.add_argument(
        "--decision_source",
        default="future_calibrated_cloud_coordinator",
        help="decision_source field written into cloud decisions.",
    )
    parser.add_argument(
        "--log_jsonl",
        default="runtime/cloud_http_decision_requests.jsonl",
        help="Append one JSON record per request. Use an empty string to disable.",
    )
    parser.add_argument(
        "--max_body_bytes",
        type=int,
        default=2 * 1024 * 1024,
        help="Maximum request body size.",
    )
    parser.add_argument(
        "--llm_review_queue_dir",
        default="",
        help="Durable asynchronous Qwen review spool. Empty disables review enqueueing.",
    )
    parser.add_argument(
        "--llm_review_decisions",
        default="ramp_metering,regional_coordination,reroute",
        help="Comma-separated fast-decision classes sent to asynchronous Qwen review.",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=8,
        help="Maximum number of concurrent edge requests predicted in one model call.",
    )
    parser.add_argument(
        "--micro_batch_wait_ms",
        type=float,
        default=2.0,
        help="Maximum collection window for a cloud inference micro-batch.",
    )
    parser.add_argument(
        "--access_log",
        action="store_true",
        help="Print one HTTP access line per request. JSONL request logging is configured separately.",
    )
    return parser.parse_args()


def now_ns() -> int:
    return time.time_ns()


def append_jsonl(path: Optional[Path], row: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


class CloudDecisionBatcher:
    def __init__(
        self,
        cloud_model: Dict[str, Any],
        policy_version: str,
        max_batch_size: int,
        wait_ms: float,
    ) -> None:
        if max_batch_size <= 0 or wait_ms < 0:
            raise ValueError("micro-batch size must be positive and wait must be non-negative")
        self.cloud_model = cloud_model
        self.policy_version = policy_version
        self.max_batch_size = max_batch_size
        self.wait_seconds = wait_ms / 1000.0
        self._queue = queue.Queue()
        self._stop = object()
        self._worker = threading.Thread(
            target=self._run,
            name="cloud-decision-micro-batcher",
            daemon=True,
        )
        self._worker.start()

    def decide_many(self, events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items = []
        for event in events:
            item = {
                "event": event,
                "done": threading.Event(),
                "result": None,
                "error": None,
            }
            items.append(item)
            self._queue.put(item)
        for item in items:
            item["done"].wait()
            if item["error"] is not None:
                raise RuntimeError("Cloud micro-batch inference failed") from item["error"]
        return [item["result"] for item in items]

    def decide(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return self.decide_many([event])[0]

    def close(self) -> None:
        if self._worker.is_alive():
            self._queue.put(self._stop)
            self._worker.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is self._stop:
                return
            items = [first]
            stop_after_batch = False
            deadline = time.perf_counter() + self.wait_seconds
            while len(items) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is self._stop:
                    stop_after_batch = True
                    break
                items.append(item)
            try:
                results = cloud_decisions(
                    [item["event"] for item in items],
                    self.cloud_model,
                    self.policy_version,
                )
                for item, result in zip(items, results):
                    item["result"] = result
            except Exception as exc:  # noqa: BLE001
                for item in items:
                    item["error"] = exc
            finally:
                for item in items:
                    item["done"].set()
            if stop_after_batch:
                return


class TrafficThreadingHTTPServer(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True


class CloudDecisionService:
    def __init__(
        self,
        endpoint: str,
        decision_source: str,
        log_jsonl: Optional[Path],
        max_body_bytes: int,
        coordinate_endpoint: str,
        policy_endpoint: str,
        artifact_endpoint: str,
        policy_bundle: Dict[str, Any],
        project_root: Path,
        neighbor_map: Dict[int, list],
        cloud_model: Dict[str, Any],
        micro_batch_size: int,
        micro_batch_wait_ms: float,
        access_log: bool,
        llm_review_queue: Optional[DurableReviewQueue],
        llm_review_decisions: Set[str],
    ) -> None:
        self.endpoint = endpoint
        self.decision_source = decision_source
        self.log_jsonl = log_jsonl
        self.max_body_bytes = max_body_bytes
        self.coordinate_endpoint = coordinate_endpoint
        self.policy_endpoint = policy_endpoint
        self.artifact_endpoint = artifact_endpoint.rstrip("/")
        self.policy_bundle = policy_bundle
        self.project_root = project_root.resolve()
        self.artifact_manifest = {
            str(artifact["path"]): artifact
            for artifact in policy_bundle.get("payload", {}).get("artifacts", [])
        }
        self.neighbor_map = neighbor_map
        self.cloud_model = cloud_model
        self.batch_decider = CloudDecisionBatcher(
            cloud_model,
            str(policy_bundle["policy_version"]),
            micro_batch_size,
            micro_batch_wait_ms,
        )
        self.micro_batch_size = micro_batch_size
        self.micro_batch_wait_ms = micro_batch_wait_ms
        self.access_log = access_log
        self.llm_review_queue = llm_review_queue
        self.llm_review_decisions = llm_review_decisions
        self._log_lock = threading.Lock()

    def extract_event(self, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        request_id = str(payload.get("request_id", ""))
        metadata = {
            "client_sent_at_ns": payload.get("client_sent_at_ns"),
            "edge_perception_latency_ms": payload.get("edge_perception_latency_ms"),
            "edge_student_latency_ms": payload.get("edge_student_latency_ms"),
            "edge_compute_latency_ms": payload.get("edge_compute_latency_ms"),
        }
        event = payload.get("edge_event")
        if isinstance(event, dict):
            return request_id, event, metadata
        if "top_k_risk_nodes" in payload or "region_summary" in payload:
            return request_id, payload, metadata
        raise ValueError("Request JSON must contain an edge_event object.")

    def decide(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return self.batch_decider.decide(event)

    def decide_many(self, events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.batch_decider.decide_many(events)

    def append_log(self, row: Dict[str, Any]) -> None:
        with self._log_lock:
            append_jsonl(self.log_jsonl, row)

    def close(self) -> None:
        self.batch_decider.close()

    def artifact(self, relative_path: str) -> Tuple[Path, Dict[str, Any]]:
        manifest = self.artifact_manifest.get(relative_path)
        if manifest is None:
            raise FileNotFoundError(relative_path)
        path = (self.project_root / relative_path).resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as exc:
            raise FileNotFoundError(relative_path) from exc
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        if path.stat().st_size != int(manifest["size_bytes"]):
            raise ValueError("artifact_size_mismatch")
        return path, manifest

    def enqueue_llm_review(
        self,
        request_id: str,
        event: Dict[str, Any],
        decision: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[str], float]:
        if self.llm_review_queue is None:
            return None, None, 0.0
        if str(decision.get("decision")) not in self.llm_review_decisions:
            return None, None, 0.0
        started = time.perf_counter()
        try:
            job_id = self.llm_review_queue.enqueue(request_id, event, decision)
            return job_id, None, (time.perf_counter() - started) * 1000.0
        except Exception as exc:  # noqa: BLE001
            return None, "{}: {}".format(type(exc).__name__, exc), (
                time.perf_counter() - started
            ) * 1000.0

    def coordinate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        events = payload.get("edge_events")
        decisions = payload.get("edge_decisions")
        if not isinstance(events, list) or len(events) < 2 or not all(isinstance(event, dict) for event in events):
            raise ValueError("edge_events must contain at least two event objects.")
        if decisions is None:
            decisions = self.decide_many(events)
        if not isinstance(decisions, list) or len(decisions) != len(events):
            raise ValueError("edge_decisions must have the same length as edge_events.")
        records = []
        for event, decision in zip(events, decisions):
            if not isinstance(decision, dict):
                raise ValueError("Each edge decision must be an object.")
            candidate = dict(decision)
            candidate.setdefault("policy_version", str(self.policy_bundle["policy_version"]))
            records.append({"event": event, "decision": candidate})
        coordinated = coordinate_globally(records, self.neighbor_map)
        return {
            "policy_version": str(self.policy_bundle["policy_version"]),
            "initial_conflicts": coordinated["initial_conflicts"],
            "residual_conflicts": coordinated["residual_conflicts"],
            "resolution_success_rate": coordinated["resolution_success_rate"],
            "global_rounds": coordinated["rounds"],
            "decisions": [record["decision"] for record in coordinated["records"]],
        }


def build_handler(service: CloudDecisionService):
    class CloudDecisionHandler(BaseHTTPRequestHandler):
        server_version = "TrafficCloudHTTP/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            if not service.access_log:
                return
            print(
                "{} - - [{}] {}".format(
                    self.client_address[0],
                    self.log_date_time_string(),
                    fmt % args,
                )
            )

        def send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_artifact(self, path: Path, manifest: Dict[str, Any]) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("X-Artifact-SHA256", str(manifest["sha256"]))
            self.end_headers()
            with path.open("rb") as file_obj:
                for block in iter(lambda: file_obj.read(1024 * 1024), b""):
                    self.wfile.write(block)

        def do_GET(self) -> None:
            request_path = urlsplit(self.path).path
            if request_path == "/health":
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "service": "traffic_cloud_decision",
                        "endpoint": service.endpoint,
                        "decision_source": service.decision_source,
                        "policy_version": service.policy_bundle["policy_version"],
                        "llm_review_enabled": service.llm_review_queue is not None,
                        "llm_review_pending": (
                            service.llm_review_queue.pending_count()
                            if service.llm_review_queue is not None
                            else 0
                        ),
                        "micro_batch_size": service.micro_batch_size,
                        "micro_batch_wait_ms": service.micro_batch_wait_ms,
                        "artifact_count": len(service.artifact_manifest),
                    },
                )
                return
            if request_path == service.policy_endpoint:
                self.send_json(HTTPStatus.OK, service.policy_bundle)
                return
            artifact_prefix = service.artifact_endpoint + "/"
            if request_path.startswith(artifact_prefix):
                relative_path = unquote(request_path[len(artifact_prefix):])
                try:
                    path, manifest = service.artifact(relative_path)
                    self.send_artifact(path, manifest)
                except FileNotFoundError:
                    self.send_json(
                        HTTPStatus.NOT_FOUND,
                        {"error": "artifact_not_found", "path": relative_path},
                    )
                except ValueError as exc:
                    self.send_json(
                        HTTPStatus.CONFLICT,
                        {"error": "artifact_manifest_mismatch", "detail": str(exc)},
                    )
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found", "path": request_path})

        def do_POST(self) -> None:
            request_received_ns = now_ns()
            if self.path not in {service.endpoint, service.coordinate_endpoint}:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found", "path": self.path})
                return

            content_length = self.headers.get("Content-Length")
            try:
                body_size = int(content_length or "0")
            except ValueError:
                self.send_json(HTTPStatus.LENGTH_REQUIRED, {"error": "invalid_content_length"})
                return
            if body_size <= 0:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "empty_request_body"})
                return
            if body_size > service.max_body_bytes:
                self.send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "body_too_large", "max_body_bytes": service.max_body_bytes},
                )
                return

            try:
                payload = json.loads(self.rfile.read(body_size).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object.")
                if self.path == service.coordinate_endpoint:
                    decision_start = time.perf_counter()
                    coordination = service.coordinate(payload)
                    cloud_decision_ms = (time.perf_counter() - decision_start) * 1000.0
                    response = {
                        "scene": "freeway_traffic_management",
                        "request_id": str(payload.get("request_id", "")),
                        "coordination": coordination,
                        "cloud_metrics": {
                            "request_body_bytes": body_size,
                            "cloud_decision_latency_ms": round(cloud_decision_ms, 6),
                        },
                    }
                    service.append_log(
                        {
                            "request_id": response["request_id"],
                            "request_type": "global_coordination",
                            "edge_count": len(coordination["decisions"]),
                            "initial_conflict_count": len(coordination["initial_conflicts"]),
                            "residual_conflict_count": len(coordination["residual_conflicts"]),
                            "cloud_metrics": response["cloud_metrics"],
                        }
                    )
                    self.send_json(HTTPStatus.OK, response)
                    return
                request_id, event, request_metadata = service.extract_event(payload)
                decision_start = time.perf_counter()
                decision = service.decide(event)
                cloud_decision_ms = (time.perf_counter() - decision_start) * 1000.0
                review_job_id, review_queue_error, review_queue_ms = service.enqueue_llm_review(
                    request_id,
                    event,
                    decision,
                )
                request_finished_ns = now_ns()
                response = {
                    "scene": "freeway_traffic_management",
                    "request_id": request_id,
                    "edge_id": str(event.get("edge_id", "unknown_edge")),
                    "region_id": str(event.get("region_id", "unknown_region")),
                    "decision": decision,
                    "cloud_metrics": {
                        "request_body_bytes": body_size,
                        "cloud_decision_latency_ms": round(cloud_decision_ms, 6),
                        "server_received_at_ns": request_received_ns,
                        "server_finished_at_ns": request_finished_ns,
                        "server_elapsed_wall_ms": round(
                            (request_finished_ns - request_received_ns) / 1_000_000.0,
                            6,
                        ),
                        "llm_review_queue_latency_ms": round(review_queue_ms, 6),
                    },
                    "llm_review": {
                        "mode": "asynchronous",
                        "queued": review_job_id is not None,
                        "job_id": review_job_id,
                        "queue_error": review_queue_error,
                        "affects_current_decision": False,
                    },
                }
                log_row = {
                    "request_id": request_id,
                    "edge_id": response["edge_id"],
                    "region_id": response["region_id"],
                    "client_address": self.client_address[0],
                    "request_metadata": request_metadata,
                    "cloud_metrics": response["cloud_metrics"],
                    "decision": decision.get("decision"),
                    "global_risk_level": decision.get("global_risk_level"),
                    "llm_review": response["llm_review"],
                }
                service.append_log(log_row)
                self.send_json(HTTPStatus.OK, response)
            except json.JSONDecodeError as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json", "detail": str(exc)})
            except Exception as exc:
                self.send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "cloud_decision_failed", "detail": str(exc)},
                )

    return CloudDecisionHandler


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_jsonl) if args.log_jsonl else None
    policy_bundle = load_policy_json(Path(args.policy_bundle))
    valid_policy, policy_reason = verify_bundle(policy_bundle)
    if not valid_policy:
        raise ValueError("Invalid policy bundle: {}".format(policy_reason))
    config = load_config(args.config)
    adjacency, _ = load_adjacency(config)
    neighbor_map = build_undirected_neighbor_map(adjacency, int(config["Data"]["num_of_vertices"]))
    cloud_model = load_cloud_model(Path(args.coordinator_model))
    llm_review_queue = (
        DurableReviewQueue(Path(args.llm_review_queue_dir))
        if args.llm_review_queue_dir
        else None
    )
    llm_review_decisions = {
        part.strip()
        for part in args.llm_review_decisions.split(",")
        if part.strip()
    }
    service = CloudDecisionService(
        endpoint=args.endpoint,
        decision_source=args.decision_source,
        log_jsonl=log_path,
        max_body_bytes=args.max_body_bytes,
        coordinate_endpoint=args.coordinate_endpoint,
        policy_endpoint=args.policy_endpoint,
        artifact_endpoint=args.artifact_endpoint,
        policy_bundle=policy_bundle,
        project_root=Path(__file__).resolve().parents[1],
        neighbor_map=neighbor_map,
        cloud_model=cloud_model,
        micro_batch_size=args.micro_batch_size,
        micro_batch_wait_ms=args.micro_batch_wait_ms,
        access_log=args.access_log,
        llm_review_queue=llm_review_queue,
        llm_review_decisions=llm_review_decisions,
    )
    server = TrafficThreadingHTTPServer((args.host, args.port), build_handler(service))
    print("Cloud HTTP decision service listening on {}:{}".format(args.host, args.port))
    print("Decision endpoint: {}".format(args.endpoint))
    print("Coordinate endpoint: {}".format(args.coordinate_endpoint))
    print("Policy endpoint: {}".format(args.policy_endpoint))
    print("Artifact endpoint: {}/<manifest-path>".format(args.artifact_endpoint.rstrip("/")))
    print("Health endpoint: /health")
    print(
        "Cloud micro-batch: size={}, wait_ms={}".format(
            args.micro_batch_size,
            args.micro_batch_wait_ms,
        )
    )
    if llm_review_queue is not None:
        print("Asynchronous Qwen review queue:", llm_review_queue.root)
        print("Reviewed decision classes:", ", ".join(sorted(llm_review_decisions)))
    if log_path is not None:
        save_json(
            {
                "service": "traffic_cloud_decision",
                "host": args.host,
                "port": args.port,
                "endpoint": args.endpoint,
                "decision_source": args.decision_source,
                "started_at_ns": now_ns(),
            },
            log_path.with_suffix(".meta.json"),
        )
        print("Request log:", log_path)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCloud HTTP decision service stopped.")
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
