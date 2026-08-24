#!/usr/bin/env python3
"""Measure paired industrial HTTP transport without changing a running service.

The client runs on the edge host and talks to an isolated cloud-service port.
It records request/response bodies separately from Linux TCP_INFO stream-byte
counters.  Packet/L2 accounting is intentionally left to the independently
filtered pcap so this script never needs packet-capture privileges.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path, PurePosixPath
import socket
import struct
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Tuple
from urllib.parse import urlsplit


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tcp_info(sock: socket.socket) -> Dict[str, int]:
    raw = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 256)

    def u32(offset: int) -> int:
        return int(struct.unpack_from("=I", raw, offset)[0]) if len(raw) >= offset + 4 else 0

    def u64(offset: int) -> int:
        return int(struct.unpack_from("=Q", raw, offset)[0]) if len(raw) >= offset + 8 else 0

    return {
        "struct_bytes": len(raw),
        "total_retrans": u32(100),
        "bytes_acked": u64(120),
        "bytes_received": u64(128),
        "segs_out": u32(136),
        "segs_in": u32(140),
        "data_segs_in": u32(152),
        "data_segs_out": u32(156),
        "bytes_sent": u64(200),
        "bytes_retrans": u64(208),
    }


def _delta(after: Mapping[str, int], before: Mapping[str, int], name: str) -> int:
    return max(0, int(after.get(name, 0)) - int(before.get(name, 0)))


def _http_request(
    host: str,
    port: int,
    method: str,
    path: str,
    body: bytes,
    headers: Mapping[str, str],
    timeout: float,
) -> Dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    connection.connect()
    assert connection.sock is not None
    client_local_ip, client_local_port = connection.sock.getsockname()[:2]
    before = _tcp_info(connection.sock)
    started = time.perf_counter_ns()
    connection.request(method, path, body=body, headers=dict(headers))
    response = connection.getresponse()
    response_body = response.read()
    finished = time.perf_counter_ns()
    assert connection.sock is not None
    after = _tcp_info(connection.sock)
    response_headers = {str(key).lower(): str(value) for key, value in response.getheaders()}
    record = {
        "method": method,
        "path": path,
        "status": int(response.status),
        "client_local_ip": str(client_local_ip),
        "client_local_port": int(client_local_port),
        "request_body_bytes": len(body),
        "request_body_sha256": _sha256_bytes(body),
        "response_body_bytes": len(response_body),
        "response_body_sha256": _sha256_bytes(response_body),
        "content_length_response": int(response_headers.get("content-length", "0")),
        "wall_ms": round((finished - started) / 1_000_000.0, 6),
        "tcp_stream_sent_bytes": _delta(after, before, "bytes_sent"),
        "tcp_stream_received_bytes": _delta(after, before, "bytes_received"),
        "tcp_bytes_acked": _delta(after, before, "bytes_acked"),
        "tcp_bytes_retrans": _delta(after, before, "bytes_retrans"),
        "tcp_total_retrans": _delta(after, before, "total_retrans"),
        "tcp_segments_out": _delta(after, before, "segs_out"),
        "tcp_segments_in": _delta(after, before, "segs_in"),
        "tcp_data_segments_out": _delta(after, before, "data_segs_out"),
        "tcp_data_segments_in": _delta(after, before, "data_segs_in"),
        "tcp_info_struct_bytes": int(after["struct_bytes"]),
    }
    connection.close()
    if record["status"] != 200:
        raise RuntimeError(
            "{} {} returned {}: {}".format(
                method,
                path,
                record["status"],
                response_body[:500].decode("utf-8", errors="replace"),
            )
        )
    return record


def _read_predictions(root: Path, modality: str) -> Iterable[Dict[str, Any]]:
    for product_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        csv_path = product_dir / "predictions.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
            for row_index, row in enumerate(csv.DictReader(file_obj)):
                source = PurePosixPath(str(row["path"]))
                map_path = (product_dir / str(row["map_file"])).resolve()
                yield {
                    "modality": modality,
                    "product": product_dir.name,
                    "row_index": row_index,
                    "category": source.parent.name,
                    "sample_name": source.name,
                    "score": float(row["image_score"]),
                    "map_path": map_path,
                    "predictions_csv": csv_path.resolve(),
                }


def _load_inputs(rgb_root: Path, infrared_root: Path, limit: int) -> List[Dict[str, Any]]:
    rows = list(_read_predictions(rgb_root, "rgb")) + list(
        _read_predictions(infrared_root, "infrared")
    )
    rows.sort(
        key=lambda row: (
            row["product"],
            row["category"],
            row["sample_name"],
            row["modality"],
        )
    )
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError("no prediction rows found")
    for row in rows:
        if row["map_path"].stat().st_size != 160 * 160 * 4:
            raise ValueError("heatmap is not 160x160 float32: {}".format(row["map_path"]))
    return rows


def _build_event(plugin: Any, template: Mapping[str, Any], row: Mapping[str, Any], arm: str, ordinal: int) -> Tuple[Any, str]:
    from cloud_edge_framework.event_envelope import SceneEventEnvelope

    payload = json.loads(json.dumps(template))
    identity = "wire-{:04d}-{}".format(
        ordinal, "arm0" if arm == "full" else "arm1"
    )
    payload["id"] = identity + "-" + str(row["modality"])
    payload["edgeid"] = "industrial-{}-edge".format(row["modality"])
    payload["source"] = "urn:edge:industrial:{}:paired-wire-v1".format(row["modality"])
    payload["subject"] = "{}-{}-{}".format(row["product"], row["category"], row["sample_name"])
    payload["time"] = "2026-08-22T00:00:00Z"
    data = payload["data"]
    data.update(
        {
            "sample_id": payload["subject"],
            "product": row["product"],
            "modality": row["modality"],
            "score": float(row["score"]),
            "raw_uri": "file:///frozen/{}/{}/{}".format(
                row["product"], row["modality"], row["sample_name"]
            ),
            "heatmap_uri": row["map_path"].as_uri(),
            "heatmap_size_bytes": row["map_path"].stat().st_size,
            "heatmap_sha256": _sha256_path(row["map_path"]),
            "inference_ms": 0.0,
            "preprocessing_latency_ms": 0.0,
            "deadline_ms": 200.0,
            "model": {"name": "frozen-industrial10-score", "version": "20260821"},
        }
    )
    event = plugin.normalize_envelope(SceneEventEnvelope.from_dict(payload))
    local = plugin.edge_decide(event)
    event = replace(
        event,
        metadata={
            **event.metadata,
            **plugin.cloud_submission_metadata(event, local),
        },
    )
    return event, str(local.decision)


def _materialized_event(event: Any, include_feature: bool, artifact_sha: str) -> Dict[str, Any]:
    evidence = []
    for item in event.evidence:
        if item.level == "feature" and not include_feature:
            continue
        if item.level == "feature":
            item = replace(item, uri="evidence://" + artifact_sha, sha256=artifact_sha)
        evidence.append(item)
    cloud_event = replace(event, evidence=evidence)
    cloud_event = replace(
        cloud_event,
        scene_payload={},
        metadata={
            **cloud_event.metadata,
            "selected_evidence_level": "feature" if include_feature else "summary",
            "transport_include_scene_payload": False,
        },
    )
    return cloud_event.to_dict(include_scene_payload=False)


def _run_arm(
    host: str,
    port: int,
    timeout: float,
    plugin: Any,
    template: Mapping[str, Any],
    row: Mapping[str, Any],
    arm: str,
    ordinal: int,
) -> Dict[str, Any]:
    event, local_state = _build_event(plugin, template, row, arm, ordinal)
    include_feature = arm == "full" or local_state != "normal"
    artifact_data = row["map_path"].read_bytes()
    artifact_sha = _sha256_bytes(artifact_data)
    transactions = []
    if include_feature:
        transactions.append(
            _http_request(
                host,
                port,
                "PUT",
                "/api/v1/evidence/" + artifact_sha,
                artifact_data,
                {
                    "Accept": "application/json",
                    "Content-Type": "application/octet-stream",
                    "X-Evidence-ID": event.event_id + "_heatmap",
                    "Idempotency-Key": "evidence_" + artifact_sha,
                },
                timeout,
            )
        )
    event_dict = _materialized_event(event, include_feature, artifact_sha)
    decision_body = json.dumps(
        {"event": event_dict}, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    decision_transaction = _http_request(
        host,
        port,
        "POST",
        "/api/v1/collaboration/cloud-decision",
        decision_body,
        {"Accept": "application/json", "Content-Type": "application/json"},
        timeout,
    )
    transactions.append(decision_transaction)
    return {
        "schema_version": "industrial-paired-transport-record/v1",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "ordinal": ordinal,
        "arm": arm,
        "source_identity": {
            "product": row["product"],
            "modality": row["modality"],
            "category": row["category"],
            "sample_name": row["sample_name"],
            "row_index": row["row_index"],
            "score": row["score"],
            "predictions_csv": str(row["predictions_csv"]),
            "heatmap_path": str(row["map_path"]),
            "heatmap_sha256": artifact_sha,
            "heatmap_bytes": len(artifact_data),
        },
        "local_state": local_state,
        "selective_feature_included": include_feature,
        "transaction_count": len(transactions),
        "transactions": transactions,
        "totals": {
            "application_request_body_bytes": sum(item["request_body_bytes"] for item in transactions),
            "http_response_body_bytes": sum(item["response_body_bytes"] for item in transactions),
            "tcp_stream_sent_bytes": sum(item["tcp_stream_sent_bytes"] for item in transactions),
            "tcp_stream_received_bytes": sum(item["tcp_stream_received_bytes"] for item in transactions),
            "tcp_bytes_retrans": sum(item["tcp_bytes_retrans"] for item in transactions),
            "wall_ms": round(sum(item["wall_ms"] for item in transactions), 6),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud-url", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--infrared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output))
    project_root = args.project_root.resolve()
    scene_root = project_root / "scenes/industrial_anomaly"
    for value in (project_root, scene_root):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))
    from industrial_anomaly.plugin import IndustrialAnomalyPlugin

    parsed = urlsplit(args.cloud_url)
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        raise ValueError("cloud-url must be explicit http://host:port")
    if parsed.port == 18100:
        raise ValueError("production port 18100 is forbidden")
    plugin = IndustrialAnomalyPlugin(
        thresholds_path=scene_root / "industrial_anomaly/review_bands.json"
    )
    templates = {
        "rgb": json.loads((scene_root / "samples/rgb_event.json").read_text(encoding="utf-8")),
        "infrared": json.loads((scene_root / "samples/infrared_event.json").read_text(encoding="utf-8")),
    }
    rows = _load_inputs(args.rgb_root.resolve(), args.infrared_root.resolve(), args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file_obj:
        for ordinal, row in enumerate(rows):
            # Balanced AB/BA ordering controls short-term link drift.
            arm_order = ("full", "sele") if ordinal % 2 == 0 else ("sele", "full")
            for arm_code in arm_order:
                arm = "full" if arm_code == "full" else "selective"
                record = _run_arm(
                    parsed.hostname,
                    parsed.port,
                    args.timeout,
                    plugin,
                    templates[row["modality"]],
                    row,
                    arm,
                    ordinal,
                )
                file_obj.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                file_obj.flush()
    print(json.dumps({"output": str(args.output), "source_events": len(rows), "records": len(rows) * 2}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
