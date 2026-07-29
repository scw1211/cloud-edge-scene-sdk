"""用途：检查两台真实边缘节点的四分区事件是否完成云端汇聚和最终回填。"""

import argparse
import hashlib
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _get(url, timeout_seconds):
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _stable_id(prefix, *parts):
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return "{}_{}".format(prefix, hashlib.sha256(material).hexdigest()[:20])


def main():
    parser = argparse.ArgumentParser(description="核验真实交通双边缘闭环")
    parser.add_argument("--edge-a", required=True)
    parser.add_argument("--edge-b", required=True)
    parser.add_argument("--cloud", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--wait-seconds", type=float, default=20.0)
    args = parser.parse_args()

    sample_split = "{}_{}".format(args.split, args.experiment_id)
    event_ids = {
        0: "freeway_{}_sample_{:04d}_edge_node_0_{}".format(
            args.split, args.sample_id, args.experiment_id
        ),
        1: "freeway_{}_sample_{:04d}_edge_node_1_{}".format(
            args.split, args.sample_id, args.experiment_id
        ),
        2: "freeway_{}_sample_{:04d}_edge_node_2_{}".format(
            args.split, args.sample_id, args.experiment_id
        ),
        3: "freeway_{}_sample_{:04d}_edge_node_3_{}".format(
            args.split, args.sample_id, args.experiment_id
        ),
    }
    group_key = "PEMS08:{}:{}".format(sample_split, args.sample_id)
    group_id = _stable_id(
        "aggregation", "traffic", group_key
    )
    deadline = time.time() + args.wait_seconds
    reviews = {}
    aggregation = None
    while time.time() < deadline:
        reviews = {}
        for partition_id, event_id in event_ids.items():
            edge = args.edge_a if partition_id < 2 else args.edge_b
            reviews[partition_id] = _get(
                edge.rstrip("/")
                + "/api/v1/collaboration/reviews/"
                + event_id,
                3.0,
            )
        aggregation = _get(
            args.cloud.rstrip("/")
            + "/api/v1/collaboration/aggregations/"
            + group_id,
            3.0,
        )
        if (
            aggregation
            and aggregation.get("state") == "completed"
            and all(
                review and review.get("state") == "completed"
                for review in reviews.values()
            )
        ):
            break
        time.sleep(0.2)

    coordination = (
        aggregation.get("result", {}) if isinstance(aggregation, dict) else {}
    )
    passed = bool(
        aggregation
        and aggregation.get("state") == "completed"
        and set(aggregation.get("received_members", []))
        == {"edge_node_0", "edge_node_1", "edge_node_2", "edge_node_3"}
        and all(
            review and review.get("state") == "completed"
            for review in reviews.values()
        )
        and coordination.get("residual_conflict_count") == 0
    )
    result = {
        "status": "passed" if passed else "failed",
        "experiment_id": args.experiment_id,
        "group_id": group_id,
        "group_key": group_key,
        "cloud_aggregation": aggregation,
        "edge_review_states": {
            str(partition_id): (
                review.get("state") if isinstance(review, dict) else "missing"
            )
            for partition_id, review in reviews.items()
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
