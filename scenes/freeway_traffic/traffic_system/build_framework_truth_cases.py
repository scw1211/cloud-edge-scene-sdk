"""用途：为正式框架 HTTP 基准生成带未来状态代理真值的交通事件 JSONL。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from traffic_system.decision_utils import rule_teacher_decision, write_jsonl
from traffic_system.evaluate_future_truth_policy import (
    CRITICAL_DECISIONS,
    load_evaluation_arrays,
    make_event,
    one_hot_probabilities,
)
from traffic_system.risk_labels import RISK_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join stored traffic model outputs with future flow/occupancy/speed "
            "proxy references for the deployed framework HTTP benchmark."
        )
    )
    parser.add_argument(
        "--events-dir",
        default="datasets/freeway_events_joint_metis4",
    )
    parser.add_argument(
        "--data-npz",
        default="../data/PEMS08/PEMS08_r1_d0_w0_astcgn_multitask.npz",
    )
    parser.add_argument(
        "--risk-labels",
        default="datasets/risk_labels_pems08_metis4.npz",
    )
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output",
        default="datasets/freeway_events_joint_metis4_truth.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="results/framework/freeway_events_joint_metis4_truth_manifest.json",
    )
    return parser.parse_args()


def _load_native_events(events_dir: Path) -> List[Dict[str, Any]]:
    paths = sorted(path for path in events_dir.glob("*.json") if path.is_file())
    if not paths:
        raise FileNotFoundError("no traffic events found in {}".format(events_dir))
    events = []
    for path in paths:
        with path.open("r", encoding="utf-8") as file_obj:
            event = json.load(file_obj)
        if not isinstance(event, dict):
            raise ValueError("traffic event must be an object: {}".format(path))
        event["_source_path"] = str(path.resolve())
        events.append(event)
    return events


def _partitions(events: List[Dict[str, Any]]) -> List[List[int]]:
    by_id: Dict[int, List[int]] = {}
    for event in events:
        partition_id = int(event["partition_id"])
        nodes = [int(value) for value in event["managed_node_ids"]]
        previous = by_id.setdefault(partition_id, nodes)
        if previous != nodes:
            raise ValueError(
                "managed nodes changed for partition {}".format(partition_id)
            )
    expected = list(range(len(by_id)))
    if sorted(by_id) != expected:
        raise ValueError("partition ids must be contiguous from zero")
    return [by_id[index] for index in expected]


def build_cases(
    events_dir: Path,
    data_npz: Path,
    risk_labels: Path,
    split: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    events = _load_native_events(events_dir)
    partitions = _partitions(events)
    arrays = load_evaluation_arrays(data_npz, risk_labels, split)
    if partitions != arrays["label_partitions"]:
        raise ValueError("stored event partitions differ from frozen risk-label partitions")
    cases: List[Dict[str, Any]] = []
    for event in events:
        sample_id = int(event["sample_id"])
        partition_id = int(event["partition_id"])
        if str(event.get("sample_split", split)) != split:
            raise ValueError(
                "event {} is not in split {}".format(event["event_id"], split)
            )
        truth_event = make_event(
            split,
            sample_id,
            partition_id,
            partitions,
            one_hot_probabilities(
                arrays["node_labels"][sample_id], len(RISK_CLASSES)
            ),
            one_hot_probabilities(
                arrays["region_labels"][sample_id], len(RISK_CLASSES)
            ),
            arrays["split_target"][sample_id],
            event.get("control_capabilities", {}),
            top_k,
            "future_observation_fcm_reference",
        )
        reference = rule_teacher_decision(
            truth_event,
            decision_source="future_truth_policy_reference",
        )
        native = {key: value for key, value in event.items() if key != "_source_path"}
        cases.append(
            {
                "event": native,
                "reference": {
                    "decision": str(reference["decision"]),
                    "critical": str(reference["decision"]) in CRITICAL_DECISIONS,
                    "critical_decisions": sorted(CRITICAL_DECISIONS),
                    "action_types": sorted(
                        {
                            str(action.get("type"))
                            for action in reference.get("actions", [])
                            if isinstance(action, dict) and action.get("type")
                        }
                    ),
                    "source": "future_flow_occupancy_speed_frozen_fcm_policy",
                    "status": (
                        "data-derived proxy reference; not manually annotated "
                        "traffic-control ground truth"
                    ),
                    "sample_id": sample_id,
                    "partition_id": partition_id,
                },
            }
        )
    return cases


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    cases = build_cases(
        Path(args.events_dir),
        Path(args.data_npz),
        Path(args.risk_labels),
        args.split,
        args.top_k,
    )
    output_path = Path(args.output).resolve()
    count = write_jsonl(cases, output_path)
    decisions: Dict[str, int] = {}
    critical = 0
    sample_ids = set()
    regions = set()
    for case in cases:
        reference = case["reference"]
        decision = str(reference["decision"])
        decisions[decision] = decisions.get(decision, 0) + 1
        critical += int(bool(reference["critical"]))
        sample_ids.add(int(reference["sample_id"]))
        regions.add(int(reference["partition_id"]))
    manifest = {
        "schema_version": 1,
        "task": "framework_http_truth_case_generation",
        "output": str(output_path),
        "case_count": count,
        "timestamp_count": len(sample_ids),
        "partition_count": len(regions),
        "reference_decisions": dict(sorted(decisions.items())),
        "critical_count": critical,
        "reference": {
            "source": "future flow/occupancy/speed -> frozen FCM risk -> fixed policy",
            "status": (
                "data-derived proxy reference; not manually annotated "
                "traffic-control ground truth"
            ),
            "split": args.split,
            "test_used_for_fitting": False,
        },
        "inputs": {
            "events_dir": str(Path(args.events_dir).resolve()),
            "data_npz": str(Path(args.data_npz).resolve()),
            "risk_labels": str(Path(args.risk_labels).resolve()),
        },
    }
    manifest_path = Path(args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
