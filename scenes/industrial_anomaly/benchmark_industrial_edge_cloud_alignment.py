#!/usr/bin/env python3
"""Verify the industrial edge-to-cloud path uses the traffic-aligned authority model."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict
from urllib.parse import quote

from benchmark_industrial_closed_loop import (
    _event,
    _request_json,
    _send_group,
    _wait_reviews,
)


LOW_CONFIDENCE_SCORES = {"rgb": 0.00745, "infrared": 0.00802}


def _health(base_url: str) -> Dict[str, Any]:
    value, _, _, _ = _request_json(base_url.rstrip("/") + "/health", timeout=5.0)
    return value


def _review(base_url: str, event_id: str) -> Dict[str, Any]:
    value, _, _, _ = _request_json(
        base_url.rstrip("/")
        + "/api/v1/collaboration/reviews/"
        + quote(event_id, safe=""),
        timeout=5.0,
    )
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-url", default="http://192.168.31.222:18101")
    parser.add_argument("--cloud-url", default="http://127.0.0.1:18100")
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path(__file__).with_name("industrial_anomaly")
        / "review_bands.json",
    )
    parser.add_argument(
        "--rgb-template",
        type=Path,
        default=Path(__file__).with_name("samples") / "rgb_event.json",
    )
    parser.add_argument(
        "--infrared-template",
        type=Path,
        default=Path(__file__).with_name("samples") / "infrared_event.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    edge_health = _health(args.edge_url)
    cloud_health = _health(args.cloud_url)
    bands = json.loads(args.thresholds.read_text(encoding="utf-8"))
    templates = {
        "rgb": json.loads(args.rgb_template.read_text(encoding="utf-8")),
        "infrared": json.loads(
            args.infrared_template.read_text(encoding="utf-8")
        ),
    }
    run_id = "industrial-traffic-aligned-{}".format(time.time_ns())
    events = {}
    for modality in ("rgb", "infrared"):
        event = _event(
            templates[modality],
            bands,
            run_id,
            0,
            modality,
            "review",
            experiment=run_id,
        )
        event["data"]["score"] = LOW_CONFIDENCE_SCORES[modality]
        events[modality] = event

    provisional = _send_group(
        args.edge_url,
        events["rgb"],
        events["infrared"],
        bands,
        async_mode=True,
        force_sync=False,
        timeout=5.0,
    )
    event_ids = [events[modality]["id"] for modality in ("rgb", "infrared")]
    completed, missing = _wait_reviews(args.edge_url, event_ids, 30.0)
    full_reviews = [_review(args.edge_url, event_id) for event_id in event_ids]
    finals = [review["final_decision"] for review in full_reviews]
    metadata = [decision.get("metadata", {}) for decision in finals]
    qwen_reviews = [item.get("cloud_llm_review") for item in metadata]
    aggregations = [item.get("aggregation", {}) for item in metadata]

    gates = {
        "edge_ready_with_two_scenes": edge_health.get("ready") is True
        and edge_health.get("runtime", {}).get("scenes")
        == ["industrial_anomaly", "traffic"],
        "cloud_ready_with_two_scenes": cloud_health.get("ready") is True
        and cloud_health.get("runtime", {}).get("scenes")
        == ["industrial_anomaly", "traffic"],
        "edge_observes_cloud_available": edge_health.get("cloud_available") is True,
        "edge_routes_to_formal_cloud_18100": str(
            edge_health.get("network", {}).get("base_url", "")
        ).endswith(":18100"),
        "edge_provisional_is_review": all(
            item["local_decision"] == "review" for item in provisional["events"]
        ),
        "both_events_reach_authoritative_final": not missing
        and len(completed) == 2
        and all(
            item.get("finality") == "final"
            and item.get("global_confirmation") is True
            for item in aggregations
        ),
        "extratrees_is_online_authority": all(
            decision.get("decision") == "normal"
            and item.get("source") == "industrial_cloud_extratrees_coordinator"
            and item.get("cloud_model") == "industrial_extratrees"
            for decision, item in zip(finals, metadata)
        ),
        "qwen_review_is_group_deduplicated": qwen_reviews[0] is not None
        and qwen_reviews[0] == qwen_reviews[1],
        "qwen_is_advisory_and_non_authoritative": all(
            item.get("cloud_llm_baseline_preserved") is True
            and item.get("cloud_llm_review_role")
            == "advisory_non_authoritative"
            for item in metadata
        ),
        "qwen_challenge_does_not_replace_extratrees": all(
            review is not None
            and review.get("verdict") == "challenge"
            and decision.get("decision") == "normal"
            for review, decision in zip(qwen_reviews, finals)
        ),
    }
    evidence = {
        "schema_version": "industrial-traffic-aligned-edge-cloud-evidence/v1",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "edge_url": args.edge_url,
        "cloud_url": args.cloud_url,
        "edge_observed_cloud_url": edge_health.get("network", {}).get("base_url"),
        "run_id": run_id,
        "flow_contract": {
            "scene_specific": "RGB/infrared perception and industrial action semantics",
            "shared_with_traffic": [
                "semantic event ingress",
                "collaboration scheduler",
                "evidence planning",
                "durable outbox",
                "persistent cloud aggregation",
                "scene ExtraTrees primary decision",
                "cost-gated advisory Qwen review",
                "global conflict coordination",
                "authoritative final backfill",
            ],
        },
        "inputs": {
            "product": "capsule",
            "scores": dict(LOW_CONFIDENCE_SCORES),
            "event_ids": event_ids,
        },
        "edge_provisional": provisional,
        "cloud_final": {
            "decisions": [decision.get("decision") for decision in finals],
            "source": metadata[0].get("source"),
            "cloud_model": metadata[0].get("cloud_model"),
            "cloud_model_confidence": metadata[0].get("cloud_model_confidence"),
            "qwen_review": qwen_reviews[0],
            "qwen_review_role": metadata[0].get("cloud_llm_review_role"),
            "qwen_baseline_preserved": metadata[0].get(
                "cloud_llm_baseline_preserved"
            ),
            "aggregation": aggregations[0],
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if evidence["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
