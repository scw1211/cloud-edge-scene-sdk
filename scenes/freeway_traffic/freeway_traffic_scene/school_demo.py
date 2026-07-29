"""用途：向学校部署的两台边缘服务发送同一样本并检查最终闭环。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import time
from urllib.request import Request, urlopen


def _request(method, url, payload=None, timeout=3.0):
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_events():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "samples"
    return [
        json.loads((root / "edge_a_event.json").read_text(encoding="utf-8")),
        json.loads((root / "edge_b_event.json").read_text(encoding="utf-8")),
    ]


def main():
    parser = argparse.ArgumentParser(description="运行交通双边缘学校演示")
    parser.add_argument("--edge-a", default="http://127.0.0.1:18101")
    parser.add_argument("--edge-b", default="http://127.0.0.1:18102")
    parser.add_argument("--cloud", default="http://127.0.0.1:18100")
    parser.add_argument("--wait-seconds", type=float, default=8.0)
    args = parser.parse_args()

    suffix = str(int(time.time() * 1000))
    event_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    events = []
    for source in _load_events():
        event = deepcopy(source)
        event["id"] = event["id"] + "_" + suffix
        event["time"] = event_time
        event["data"]["sample_id"] = "school_" + suffix
        events.append(event)

    def submit(index):
        base = args.edge_a if index == 0 else args.edge_b
        return _request(
            "POST",
            base + "/api/v1/collaboration/decide",
            {"event": events[index]},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(submit, [0, 1]))

    deadline = time.time() + args.wait_seconds
    reviews = [None, None]
    while time.time() < deadline:
        for index, base in enumerate((args.edge_a, args.edge_b)):
            reviews[index] = _request(
                "GET",
                base
                + "/api/v1/collaboration/reviews/"
                + events[index]["id"],
            )
        if all(item.get("state") == "completed" for item in reviews):
            break
        time.sleep(0.2)

    aggregation = None
    group_id = (
        reviews[0]
        .get("final_decision", {})
        .get("metadata", {})
        .get("aggregation", {})
        .get("group_id")
    )
    if group_id:
        aggregation = _request(
            "GET",
            args.cloud
            + "/api/v1/collaboration/aggregations/"
            + str(group_id),
        )
    coordination = (
        aggregation.get("result", {})
        if isinstance(aggregation, dict)
        else {}
    )
    result = {
        "status": (
            "passed"
            if (
                all(item.get("state") == "completed" for item in reviews)
                and coordination.get("residual_conflict_count") == 0
            )
            else "incomplete"
        ),
        "edge_initial_results": [
            {
                "event_id": item["event"]["event_id"],
                "route": item["final_decision"]["route"],
                "status": item["final_decision"]["status"],
                "framework_runtime_ms": item["framework_runtime_ms"],
            }
            for item in responses
        ],
        "final_reviews": reviews,
        "cloud_aggregation": {
            "group_id": group_id,
            "state": (
                aggregation.get("state")
                if isinstance(aggregation, dict)
                else None
            ),
            "completion_reason": (
                aggregation.get("completion_reason")
                if isinstance(aggregation, dict)
                else None
            ),
            "initial_conflict_count": coordination.get(
                "initial_conflict_count"
            ),
            "residual_conflict_count": coordination.get(
                "residual_conflict_count"
            ),
            "resolution_success_rate": coordination.get(
                "resolution_success_rate"
            ),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
