"""用途：持续请求边缘 Qwen 动作服务，为并发资源隔离实验生成负载。"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traffic_system.decision_utils import load_json, save_json  # noqa: E402
from traffic_system.build_llm_sft_dataset import build_user_prompt  # noqa: E402
from traffic_system.edge_qwen_action_infer import request_action_token  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sustained asynchronous Qwen traffic load.")
    parser.add_argument("--edge_events", required=True)
    parser.add_argument("--host", default="http://127.0.0.1:18190")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--interval_ms", type=float, default=0.0)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.requests <= 0:
        raise ValueError("requests must be positive")
    input_path = Path(args.edge_events)
    event_paths = (
        sorted(input_path.glob("*.json"))
        if input_path.is_dir()
        else [input_path]
    )
    if not event_paths:
        raise FileNotFoundError("No edge event JSON files found: {}".format(input_path))
    prompts = [
        build_user_prompt(load_json(path), {}, "action_token", max_top_nodes=3)
        for path in event_paths
    ]
    samples: List[Dict[str, Any]] = []
    for index in range(args.requests):
        result = request_action_token(args.host, prompts[index % len(prompts)], args.timeout)
        result["request_id"] = index
        samples.append(result)
        print(
            "[{}/{}] token={} latency={:.3f}ms".format(
                index + 1, args.requests, result["action_token"], result["latency_ms"]
            ),
            flush=True,
        )
        if args.interval_ms > 0:
            time.sleep(args.interval_ms / 1000.0)
    latencies = [sample["latency_ms"] for sample in samples]
    output = {
        "task": "asynchronous_qwen_action_load",
        "host": args.host,
        "requests": args.requests,
        "success_count": len(samples),
        "average_latency_ms": round(statistics.fmean(latencies), 4),
        "max_latency_ms": round(max(latencies), 4),
        "samples": samples,
    }
    save_json(output, Path(args.output_json))
    print(json.dumps({key: output[key] for key in (
        "requests", "success_count", "average_latency_ms", "max_latency_ms"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
