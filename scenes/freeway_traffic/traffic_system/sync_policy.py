"""用途：从云端拉取最新策略包，并在边缘侧完成校验和原子更新。"""

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict

from traffic_system.policy_store import PolicyStore


def fetch_json(url: str, timeout: float) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Cloud policy response must be a JSON object.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronize an edge traffic policy from the cloud.")
    parser.add_argument("--url", default="http://127.0.0.1:18080/api/v1/policy")
    parser.add_argument("--current", default="deployment/policy/current_policy.json")
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        candidate = fetch_json(args.url, args.timeout)
        result = PolicyStore(Path(args.current)).apply(candidate)
    except Exception as exc:  # noqa: BLE001
        current = PolicyStore(Path(args.current)).current()
        result = {
            "applied": False,
            "reason": "network_or_parse_failure",
            "detail": "{}: {}".format(type(exc).__name__, exc),
            "current_version": current.get("policy_version") if current else None,
            "business_continues_with_current_policy": current is not None,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
