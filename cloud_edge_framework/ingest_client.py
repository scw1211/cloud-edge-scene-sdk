"""用途：从文件向独立边缘服务提交一个场景事件信封。"""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit one event to the edge service.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--edge_base_url", default="http://127.0.0.1:18101")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--conflict_suspected", action="store_true")
    parser.add_argument("--model_disagreement", action="store_true")
    parser.add_argument("--idempotency_key", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.event).open("r", encoding="utf-8") as file_obj:
        event = json.load(file_obj)
    if not isinstance(event, dict):
        raise ValueError("event file must contain an object")
    payload = {
        "event": event,
        "conflict_suspected": args.conflict_suspected,
        "model_disagreement": args.model_disagreement,
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if args.idempotency_key:
        headers["Idempotency-Key"] = args.idempotency_key
    request = Request(
        args.edge_base_url.rstrip("/") + "/api/v1/collaboration/decide",
        data=body,
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=args.timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
