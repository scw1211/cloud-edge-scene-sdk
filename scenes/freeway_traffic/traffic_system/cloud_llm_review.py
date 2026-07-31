"""用途：持久化云端大模型复核任务，并将 Qwen 9B 结果沉淀为纠错和偏好数据。"""

import argparse
import json
import os
import re
import statistics
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Set

from traffic_system.constrain_teacher_labels import constrain_row
from traffic_system.decision_utils import rule_teacher_decision
from traffic_system.generate_teacher_labels import (
    call_ollama_teacher,
    normalize_teacher_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process durable cloud Qwen review jobs.")
    parser.add_argument("--queue_dir", default="runtime/cloud_llm_review")
    parser.add_argument("--feedback_jsonl", default="datasets/cloud_llm_review_feedback.jsonl")
    parser.add_argument("--ollama_url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_predict", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--poll_interval", type=float, default=0.2)
    parser.add_argument("--max_jobs", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--once", action="store_true", help="Exit when the pending queue is empty.")
    parser.add_argument("--summary_json", default="results/decision/cloud_llm_review_summary.json")
    return parser.parse_args()


def action_types(decision: Dict[str, Any]) -> Set[str]:
    return {
        str(action.get("type"))
        for action in decision.get("actions", [])
        if isinstance(action, dict) and action.get("type")
    }


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class DurableReviewQueue:
    """One-file-per-job spool; rename operations provide atomic enqueue and claim."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.pending = root / "pending"
        self.processing = root / "processing"
        self.completed = root / "completed"
        self.failed = root / "failed"
        for directory in (self.pending, self.processing, self.completed, self.failed):
            directory.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        request_id: str,
        event: Dict[str, Any],
        fast_decision: Dict[str, Any],
    ) -> str:
        safe_request = re.sub(r"[^A-Za-z0-9_.-]+", "_", request_id or "request")[:48]
        job_id = "{}-{}-{}".format(time.time_ns(), safe_request, uuid.uuid4().hex[:8])
        payload = {
            "schema_version": 1,
            "job_id": job_id,
            "request_id": request_id,
            "queued_at_ns": time.time_ns(),
            "edge_event": event,
            "fast_decision": fast_decision,
        }
        temporary = self.root / (".{}.tmp".format(job_id))
        destination = self.pending / (job_id + ".json")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(destination)
        return job_id

    def pending_count(self) -> int:
        return sum(1 for _ in self.pending.glob("*.json"))

    def claim_next(self) -> Optional[Path]:
        for source in sorted(self.pending.glob("*.json")):
            destination = self.processing / source.name
            try:
                source.replace(destination)
                return destination
            except FileNotFoundError:
                continue
        return None

    def complete(self, claimed: Path, record: Dict[str, Any]) -> Path:
        destination = self.completed / claimed.name
        temporary = self.root / (".{}.completed.tmp".format(claimed.stem))
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)
        claimed.unlink(missing_ok=True)
        return destination

    def fail(self, claimed: Path, error: str) -> Path:
        payload = json.loads(claimed.read_text(encoding="utf-8"))
        payload["failed_at_ns"] = time.time_ns()
        payload["error"] = error
        destination = self.failed / claimed.name
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        claimed.unlink(missing_ok=True)
        return destination


def build_review_record(
    job: Dict[str, Any],
    teacher_decision: Dict[str, Any],
    teacher_latency_ms: float,
    raw_response: str,
    safety_constraint: Dict[str, Any],
) -> Dict[str, Any]:
    fast_decision = job["fast_decision"]
    event = job.get("edge_event", {})
    event_id = event.get("event_id")
    if not event_id and event.get("sample_id") is not None:
        event_id = "freeway_{}_sample_{:04d}_{}".format(
            str(event.get("sample_split") or "online"),
            int(event["sample_id"]),
            str(event.get("edge_id") or "unknown_edge"),
        )
    class_agreement = fast_decision.get("decision") == teacher_decision.get("decision")
    action_agreement = action_types(fast_decision) == action_types(teacher_decision)
    correction_required = not (class_agreement and action_agreement)
    completed_at_ns = time.time_ns()
    return {
        "schema_version": 1,
        "job_id": job["job_id"],
        "request_id": job.get("request_id", ""),
        "event_id": event_id,
        "queued_at_ns": job["queued_at_ns"],
        "completed_at_ns": completed_at_ns,
        "queue_wait_and_review_ms": round(
            (completed_at_ns - int(job["queued_at_ns"])) / 1_000_000.0,
            3,
        ),
        "teacher_model": "qwen3.5:9b",
        "teacher_latency_ms": round(teacher_latency_ms, 3),
        "edge_event": job["edge_event"],
        "fast_decision": fast_decision,
        "teacher_decision": teacher_decision,
        "decision_class_agreement": class_agreement,
        "action_type_agreement": action_agreement,
        "correction_required": correction_required,
        "safety_constraint": safety_constraint,
        "preference_pair": {
            "chosen": teacher_decision,
            "rejected": fast_decision,
            "usable_for_preference_training": correction_required,
        },
        "raw_teacher_response": raw_response,
    }


def review_one(args: argparse.Namespace, job: Dict[str, Any]) -> Dict[str, Any]:
    event = job.get("edge_event")
    fast_decision = job.get("fast_decision")
    if not isinstance(event, dict) or not isinstance(fast_decision, dict):
        raise ValueError("Review job must contain edge_event and fast_decision objects.")
    teacher_args = SimpleNamespace(
        model=args.model,
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        num_predict=args.num_predict,
        timeout=args.timeout,
    )
    raw, latency_ms, raw_response = call_ollama_teacher(teacher_args, event)
    teacher = normalize_teacher_output(event, raw, source="ollama_teacher")
    constrained = constrain_row(
        {
            "teacher_decision": teacher,
            "rule_decision": rule_teacher_decision(event, decision_source="rule_teacher"),
        }
    )
    return build_review_record(
        job,
        constrained["teacher_decision"],
        latency_ms,
        raw_response,
        constrained["safety_constraint"],
    )


def summarize(records: Iterable[Dict[str, Any]], failures: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    latencies = [float(row["teacher_latency_ms"]) for row in rows]
    return {
        "task": "asynchronous_cloud_qwen9b_review",
        "review_count": len(rows),
        "failure_count": len(failures),
        "success_rate": round(len(rows) / max(1, len(rows) + len(failures)), 6),
        "decision_class_agreement_rate": round(
            sum(bool(row["decision_class_agreement"]) for row in rows) / max(1, len(rows)), 6
        ),
        "action_type_agreement_rate": round(
            sum(bool(row["action_type_agreement"]) for row in rows) / max(1, len(rows)), 6
        ),
        "correction_rate": round(
            sum(bool(row["correction_required"]) for row in rows) / max(1, len(rows)), 6
        ),
        "average_teacher_latency_ms": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        "max_teacher_latency_ms": round(max(latencies), 3) if latencies else 0.0,
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    queue = DurableReviewQueue(Path(args.queue_dir))
    feedback_path = Path(args.feedback_jsonl)
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    while args.max_jobs <= 0 or len(records) + len(failures) < args.max_jobs:
        claimed = queue.claim_next()
        if claimed is None:
            if args.once:
                break
            time.sleep(max(0.02, args.poll_interval))
            continue
        try:
            job = json.loads(claimed.read_text(encoding="utf-8"))
            record = review_one(args, job)
            append_jsonl(feedback_path, record)
            queue.complete(claimed, record)
            records.append(record)
            print(
                "[{}/{}] {} fast={} teacher={} correction={}".format(
                    len(records),
                    args.max_jobs or "unlimited",
                    record["job_id"],
                    record["fast_decision"].get("decision"),
                    record["teacher_decision"].get("decision"),
                    record["correction_required"],
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            error = "{}: {}".format(type(exc).__name__, exc)
            queue.fail(claimed, error)
            failures.append({"job": claimed.name, "error": error})
            print("FAILED {}: {}".format(claimed.name, error), flush=True)

    summary = summarize(records, failures)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
