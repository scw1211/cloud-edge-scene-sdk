"""用途：在边缘节点执行本地调度并通过真实 HTTP 回路请求统一云服务。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from cloud_edge_framework.registry import build_default_registry
from cloud_edge_framework.feedback import DecisionFeedbackStore
from cloud_edge_framework.performance import PerformanceProfileStore
from cloud_edge_framework.review_queue import PendingReviewStore
from cloud_edge_framework.runtime import EdgeRuntime
from cloud_edge_framework.scheduling import NetworkSnapshot
from cloud_edge_framework.transport import HttpCloudClient


NETWORK_PROFILES = {
    "normal": NetworkSnapshot(True, 15.0, 3.0, 0.0, 1.0, 12.0),
    "mild": NetworkSnapshot(True, 45.0, 12.0, 0.01, 3.0, 15.0),
    "severe": NetworkSnapshot(True, 160.0, 40.0, 0.15, 8.0, 20.0),
    "outage": NetworkSnapshot(False, 0.0, 0.0, 1.0, 0.0, 0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one real edge-to-cloud collaboration request.")
    parser.add_argument("--event", default="")
    parser.add_argument("--cloud_base_url", default="http://127.0.0.1:18100")
    parser.add_argument("--network", choices=sorted(NETWORK_PROFILES), default="normal")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--plugin_config", default="deployment/framework/scene_plugins.json")
    parser.add_argument("--review_queue", default="runtime/framework_edge_pending_reviews.jsonl")
    parser.add_argument(
        "--performance_profiles",
        default="runtime/framework_edge_performance_profiles.json",
    )
    parser.add_argument(
        "--feedback",
        default="runtime/framework_edge_decision_feedback.jsonl",
    )
    parser.add_argument("--conflict_suspected", action="store_true")
    parser.add_argument("--model_disagreement", action="store_true")
    parser.add_argument("--flush_pending", action="store_true")
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def load_event(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError("event JSON must contain an object")
    return value


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    plugin_config = Path(args.plugin_config)
    if not plugin_config.is_absolute():
        plugin_config = project_root / plugin_config
    registry = build_default_registry(project_root, plugin_config)
    transport = HttpCloudClient(args.cloud_base_url, args.timeout)
    review_store = PendingReviewStore(Path(args.review_queue) if args.review_queue else None)
    performance_store = PerformanceProfileStore(
        Path(args.performance_profiles) if args.performance_profiles else None,
        synchronous_persistence=False,
    )
    feedback_store = DecisionFeedbackStore(Path(args.feedback) if args.feedback else None)
    runtime = EdgeRuntime(
        registry=registry,
        cloud=transport,
        review_store=review_store,
        performance_store=performance_store,
        feedback_store=feedback_store,
    )
    if args.flush_pending:
        flush_result = runtime.flush_pending()
        print(json.dumps({"flush_pending": flush_result}, ensure_ascii=False, indent=2))
        if not args.event:
            return
    if not args.event:
        raise ValueError("--event is required unless --flush_pending is used")
    result = runtime.process(
        load_event(Path(args.event)),
        NETWORK_PROFILES[args.network],
        conflict_suspected=args.conflict_suspected,
        model_disagreement=args.model_disagreement,
    )
    local_feedback_flush = feedback_store.flush(args.timeout)
    performance_store.flush()
    feedback_sync = transport.flush_feedback(args.timeout)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file_obj:
            json.dump(result, file_obj, ensure_ascii=False, indent=2)
    summary = {
        "scene": result["event"]["scene"],
        "scheduled_route": result["schedule"]["route"],
        "executed_route": result["final_decision"]["route"],
        "decision": result["final_decision"]["decision"],
        "closed_loop_accounting": result["closed_loop_accounting"],
        "transport": result["final_decision"]["metadata"].get("transport"),
        "feedback_sync": feedback_sync,
        "local_feedback_flush": local_feedback_flush,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
