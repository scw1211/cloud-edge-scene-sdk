"""用途：接收原始交通窗口，并在云端实际执行 ASTGCN、风险判断和全局协调决策。"""

import argparse
import base64
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from traffic_system.cloud_coordinator import cloud_decision, load_cloud_model
from traffic_system.decision_utils import load_json
from traffic_system.evaluate_future_truth_policy import make_event
from traffic_system.infer_joint_risk_astgcn import (
    build_control_capabilities,
    build_model_from_checkpoint,
    ensure_finite_outputs,
    load_adjacency,
    load_config,
    torch_load_trusted,
)
from traffic_system.policy_store import verify_bundle
from traffic_system.risk_labels import denormalize
from traffic_system.train_joint_risk_astgcn import clip_physical_state, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a measured centralized-cloud traffic baseline.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--endpoint", default="/api/v1/traffic/centralized")
    parser.add_argument("--config", default="configurations/PEMS08_astgcn.conf")
    parser.add_argument("--data_npz", default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz")
    parser.add_argument(
        "--checkpoint",
        default="experiments/PEMS08/joint_risk_astgcn_metis4_flowprio2_frozen/best.pt",
    )
    parser.add_argument(
        "--coordinator_model",
        default="models/cloud_coordinator_future_calibrated.joblib",
    )
    parser.add_argument("--policy_bundle", default="deployment/policy/current_policy.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_body_bytes", type=int, default=256 * 1024)
    return parser.parse_args()


def load_normalization(path: Path) -> Tuple[np.ndarray, np.ndarray, int, int]:
    with np.load(path) as data:
        required = ["mean", "std", "test_x"]
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError("Missing data arrays: {}".format(", ".join(missing)))
        mean = data["mean"]
        std = data["std"]
        in_channels = int(data["test_x"].shape[2])
        output_dim = int(mean.shape[2])
    return mean, std, in_channels, output_dim


def decode_window(payload: Dict[str, Any], expected_shape: Tuple[int, ...]) -> Tuple[np.ndarray, int]:
    if payload.get("encoding") != "base64_float32_le":
        raise ValueError("encoding must be base64_float32_le")
    shape = tuple(int(value) for value in payload.get("input_shape", []))
    if shape != expected_shape:
        raise ValueError("input_shape must be {}, got {}".format(expected_shape, shape))
    encoded = payload.get("input_base64")
    if not isinstance(encoded, str):
        raise ValueError("input_base64 must be a string")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("invalid base64 input") from exc
    expected_bytes = int(np.prod(expected_shape)) * np.dtype("<f4").itemsize
    if len(raw) != expected_bytes:
        raise ValueError("decoded input must contain {} bytes".format(expected_bytes))
    values = np.frombuffer(raw, dtype="<f4").reshape(expected_shape).copy()
    if not np.isfinite(values).all():
        raise ValueError("input contains NaN or Inf")
    return values, len(raw)


class CentralizedTrafficService:
    def __init__(self, args: argparse.Namespace) -> None:
        torch.set_num_threads(args.torch_threads)
        self.device = select_device(args.device)
        self.endpoint = args.endpoint
        self.max_body_bytes = args.max_body_bytes
        self.top_k = args.top_k
        config = load_config(args.config)
        checkpoint = torch_load_trusted(Path(args.checkpoint), self.device)
        adj_mx, _ = load_adjacency(config)
        self.mean, self.std, in_channels, output_dim = load_normalization(Path(args.data_npz))
        self.model = build_model_from_checkpoint(
            config,
            {"in_channels": in_channels, "output_dim": output_dim},
            adj_mx,
            checkpoint,
            self.device,
        )
        self.partitions = [[int(node_id) for node_id in part] for part in checkpoint["partitions"]]
        self.expected_shape = (
            1,
            int(config["Data"]["num_of_vertices"]),
            in_channels,
            int(config["Data"]["len_input"]),
        )
        self.capabilities = [
            build_control_capabilities(self.partitions, adj_mx, partition_id)
            for partition_id in range(len(self.partitions))
        ]
        self.cloud_model = load_cloud_model(Path(args.coordinator_model))
        policy_bundle = load_json(Path(args.policy_bundle))
        valid, reason = verify_bundle(policy_bundle)
        if not valid:
            raise ValueError("Invalid policy bundle: {}".format(reason))
        self.policy_version = str(policy_bundle["policy_version"])
        self._model_lock = threading.Lock()
        self._warmup()

    def _warmup(self) -> None:
        tensor = torch.zeros(self.expected_shape, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        total_started = time.perf_counter()
        decode_started = time.perf_counter()
        values, raw_bytes = decode_window(payload, self.expected_shape)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        partition_id = int(payload.get("partition_id", -1))
        if partition_id < 0 or partition_id >= len(self.partitions):
            raise ValueError("partition_id is outside configured partitions")
        sample_id = int(payload.get("sample_id", -1))
        split = str(payload.get("sample_split", "test"))

        with self._model_lock:
            tensor = torch.from_numpy(values).to(self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            model_started = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(tensor)
            ensure_finite_outputs(outputs)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            model_ms = (time.perf_counter() - model_started) * 1000.0
            forecast_raw = clip_physical_state(
                denormalize(outputs["forecast"].detach().cpu().numpy(), self.mean, self.std)
            )[0]
            node_probs = F.softmax(outputs["node_logits"], dim=-1).detach().cpu().numpy()[0]
            region_probs = F.softmax(outputs["region_logits"], dim=-1).detach().cpu().numpy()[0]
            event_started = time.perf_counter()
            event = make_event(
                split,
                sample_id,
                partition_id,
                self.partitions,
                node_probs,
                region_probs,
                forecast_raw,
                self.capabilities[partition_id],
                self.top_k,
                "centralized_cloud_joint_astgcn",
            )
            event_ms = (time.perf_counter() - event_started) * 1000.0
            coordinator_started = time.perf_counter()
            decision = cloud_decision(event, self.cloud_model, self.policy_version)
            coordinator_ms = (time.perf_counter() - coordinator_started) * 1000.0
        total_ms = (time.perf_counter() - total_started) * 1000.0
        return {
            "scene": "freeway_traffic_management",
            "request_id": str(payload.get("request_id", "")),
            "architecture": "centralized_cloud",
            "edge_id": event["edge_id"],
            "region_id": event["region_id"],
            "decision": decision,
            "cloud_metrics": {
                "raw_window_bytes": raw_bytes,
                "decode_latency_ms": round(decode_ms, 6),
                "astgcn_latency_ms": round(model_ms, 6),
                "event_build_latency_ms": round(event_ms, 6),
                "coordinator_latency_ms": round(coordinator_ms, 6),
                "cloud_total_latency_ms": round(total_ms, 6),
            },
        }


def build_handler(service: CentralizedTrafficService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CentralizedTrafficCloud/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "service": "centralized_traffic_cloud",
                        "endpoint": service.endpoint,
                        "device": str(service.device),
                        "policy_version": service.policy_version,
                        "expected_shape": list(service.expected_shape),
                    },
                )
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path != service.endpoint:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > service.max_body_bytes:
                    raise ValueError("invalid request body size")
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be an object")
                response = service.infer(payload)
                self.send_json(HTTPStatus.OK, response)
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request", "detail": str(exc)})
            except Exception as exc:  # noqa: BLE001
                self.send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "inference_failed", "detail": "{}: {}".format(type(exc).__name__, exc)},
                )

    return Handler


def main() -> None:
    args = parse_args()
    if args.port <= 0 or args.top_k <= 0 or args.max_body_bytes <= 0:
        raise ValueError("port/top_k/max_body_bytes must be positive")
    service = CentralizedTrafficService(args)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(service))
    server.daemon_threads = True
    server.request_queue_size = 128
    print("Centralized traffic cloud listening on http://{}:{}{}".format(args.host, args.port, args.endpoint))
    print("device:", service.device)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
