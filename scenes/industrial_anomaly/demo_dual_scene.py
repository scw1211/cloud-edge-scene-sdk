#!/usr/bin/env python3
"""Send industrial and traffic examples through one public edge endpoint."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


SCENE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCENE_ROOT.parents[1]


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("sample must contain a JSON object: {}".format(path))
    return value


def _request_json(url: str, payload: Dict[str, Any] = None) -> Dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=10.0) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("{} returned HTTP {}: {}".format(url, exc.code, detail))
    except URLError as exc:
        raise RuntimeError("cannot reach {}: {}".format(url, exc.reason))
    if not isinstance(result, dict):
        raise RuntimeError("{} returned a non-object JSON response".format(url))
    return result


def _with_run_identity(
    source: Dict[str, Any],
    *,
    event_id: str,
    sample_id: str,
) -> Dict[str, Any]:
    event = json.loads(json.dumps(source))
    event["id"] = event_id
    event["subject"] = sample_id
    event["time"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    event["data"]["sample_id"] = sample_id
    return event


def _decision_view(result: Dict[str, Any]) -> Dict[str, Any]:
    decision = result.get("decision") or result.get("local_decision") or {}
    if not isinstance(decision, dict):
        decision = {}
    return {
        "scene": decision.get("scene"),
        "decision": decision.get("decision"),
        "status": decision.get("status"),
        "route": decision.get("route"),
    }


def _wait_for_final(
    edge_url: str,
    event_ids: Iterable[str],
    timeout_seconds: float,
) -> List[Dict[str, Any]]:
    remaining = set(event_ids)
    completed: Dict[str, Dict[str, Any]] = {}
    deadline = time.monotonic() + timeout_seconds
    while remaining and time.monotonic() < deadline:
        for event_id in list(remaining):
            review = _request_json(
                "{}/api/v1/collaboration/reviews/{}".format(
                    edge_url, quote(event_id, safe="")
                )
            )
            final = review.get("final_decision")
            if review.get("state") == "completed" and isinstance(final, dict):
                aggregation = final.get("metadata", {}).get("aggregation", {})
                completed[event_id] = {
                    "event_id": event_id,
                    "scene": final.get("scene"),
                    "decision": final.get("decision"),
                    "status": final.get("status"),
                    "finality": aggregation.get("finality"),
                    "completion_reason": aggregation.get("completion_reason"),
                    "global_confirmation": aggregation.get("global_confirmation"),
                }
                remaining.remove(event_id)
        if remaining:
            time.sleep(0.05)
    if remaining:
        raise TimeoutError(
            "timed out waiting for final decisions: {}".format(sorted(remaining))
        )
    return [completed[event_id] for event_id in event_ids]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demonstrate industrial and traffic routing on one edge API"
    )
    parser.add_argument("--edge-url", default="http://127.0.0.1:18101")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args()
    edge_url = args.edge_url.rstrip("/")
    run_id = str(int(time.time() * 1000))

    definitions = [
        (
            "industrial",
            "industrial-demo-{}".format(run_id),
            [
                SCENE_ROOT / "samples" / "rgb_event.json",
                SCENE_ROOT / "samples" / "infrared_event.json",
            ],
        ),
        (
            "traffic",
            "traffic-demo-{}".format(run_id),
            [
                PROJECT_ROOT / "scenes" / "freeway_traffic" / "samples" / "edge_a_event.json",
                PROJECT_ROOT / "scenes" / "freeway_traffic" / "samples" / "edge_b_event.json",
            ],
        ),
    ]

    report: Dict[str, Any] = {
        "edge_endpoint": edge_url + "/api/v1/collaboration/decide",
        "runs": [],
    }
    for scene_name, sample_id, paths in definitions:
        event_ids: List[str] = []
        provisional: List[Dict[str, Any]] = []
        for index, path in enumerate(paths):
            event_id = "{}-member-{}".format(sample_id, index + 1)
            event = _with_run_identity(
                _read_json(path), event_id=event_id, sample_id=sample_id
            )
            result = _request_json(
                edge_url + "/api/v1/collaboration/decide",
                {
                    "event": event,
                    "response_detail": "compact",
                    "return_provisional_immediately": True,
                },
            )
            event_ids.append(event_id)
            provisional.append({"event_id": event_id, **_decision_view(result)})
        report["runs"].append(
            {
                "input_scene": scene_name,
                "provisional": provisional,
                "final": _wait_for_final(
                    edge_url, event_ids, args.timeout_seconds
                ),
            }
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
