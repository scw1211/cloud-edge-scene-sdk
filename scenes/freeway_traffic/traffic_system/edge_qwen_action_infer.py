"""用途：调用边缘 Qwen 服务生成单 token 动作并转换为安全交通决策。"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traffic_system.decision_utils import (  # noqa: E402
    build_decision_from_action_token,
    load_json,
    save_json,
)
from traffic_system.build_llm_sft_dataset import (  # noqa: E402
    ACTION_TOKEN_SYSTEM_PROMPT,
    build_user_prompt,
)
from traffic_system.ultracompact_codec import (  # noqa: E402
    encode_bitpacked_decimal_prompt,
    encode_contextual_decimal_prompt,
    encode_positional_decimal_prompt,
    encode_routing_context_v2_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the distilled Qwen action-token client.")
    parser.add_argument("--edge_event", required=True)
    parser.add_argument("--output_json", default="results/decision/cloud_decision_check.json")
    parser.add_argument("--host", default="http://127.0.0.1:18190")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument(
        "--input_encoding",
        choices=[
            "legacy",
            "positional_decimal",
            "bitpacked_decimal",
            "contextual_decimal",
            "routing_context_v2",
        ],
        default="bitpacked_decimal",
    )
    parser.add_argument("--student_decision", default="congestion_warning")
    parser.add_argument("--rule_decision", default="congestion_warning")
    parser.add_argument("--student_confidence", type=float, default=0.5)
    parser.add_argument("--prediction_set_size", type=int, default=1)
    parser.add_argument(
        "--network_status", choices=("normal", "weak", "offline"), default="normal"
    )
    parser.add_argument(
        "--prompt_format",
        choices=["full_chat", "user_chat", "raw_task"],
        default="raw_task",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def completion_prompt(user_prompt: str, prompt_format: str = "full_chat") -> str:
    if prompt_format == "raw_task":
        return user_prompt
    if prompt_format == "user_chat":
        return "<|im_start|>user\n" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n"
    return (
        "<|im_start|>system\n"
        + ACTION_TOKEN_SYSTEM_PROMPT
        + "<|im_end|>\n<|im_start|>user\n"
        + user_prompt
        + "<|im_end|>\n<|im_start|>assistant\n"
        + "<think>\n\n</think>\n\n"
    )


def request_action_token(
    host: str, prompt: str, timeout: float, prompt_format: str = "full_chat"
) -> Dict[str, Any]:
    payload = {
        "prompt": completion_prompt(prompt, prompt_format),
        "temperature": 0,
        "top_p": 1,
        "n_predict": 1,
        "stream": False,
        "cache_prompt": False,
    }
    request = urllib.request.Request(
        host.rstrip("/") + "/completion",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    latency_ms = round((time.perf_counter() - started) * 1000.0, 4)
    raw_output = str(result.get("content", "")).strip()
    match = re.fullmatch(r"([A-F])", raw_output.upper())
    if not match:
        raise ValueError("Qwen returned an invalid action token: {!r}".format(raw_output))
    return {
        "action_token": match.group(1),
        "latency_ms": latency_ms,
        "raw_output": raw_output,
    }


def build_action_prompt(
    event: Dict[str, Any],
    input_encoding: str = "bitpacked_decimal",
    routing_context: Optional[Mapping[str, Any]] = None,
) -> str:
    compact_prompt = build_user_prompt(
        event=event,
        row={},
        target_schema="action_token",
        max_top_nodes=3,
    )
    if input_encoding == "positional_decimal":
        return encode_positional_decimal_prompt(compact_prompt)
    if input_encoding == "bitpacked_decimal":
        return encode_bitpacked_decimal_prompt(compact_prompt)
    if input_encoding == "contextual_decimal":
        return encode_contextual_decimal_prompt(compact_prompt)
    if input_encoding == "routing_context_v2":
        if routing_context is None:
            raise ValueError("routing_context_v2 requires routing_context")
        return encode_routing_context_v2_prompt(compact_prompt, routing_context)
    if input_encoding != "legacy":
        raise ValueError("Unsupported input encoding: {}".format(input_encoding))
    return compact_prompt


def main() -> None:
    args = parse_args()
    event = load_json(resolve_path(args.edge_event))
    routing_context = None
    if args.input_encoding == "routing_context_v2":
        routing_context = {
            "student_decision": args.student_decision,
            "rule_decision": args.rule_decision,
            "student_confidence": args.student_confidence,
            "prediction_set_size": args.prediction_set_size,
            "network_status": args.network_status,
        }
    compact_prompt = build_action_prompt(
        event, args.input_encoding, routing_context=routing_context
    )
    inference = request_action_token(
        args.host, compact_prompt, args.timeout, prompt_format=args.prompt_format
    )
    decision = build_decision_from_action_token(
        event=event,
        action_token=inference["action_token"],
        confidence=args.confidence,
    )
    output = {
        "model_output": inference,
        "compact_prompt": compact_prompt,
        "input_encoding": args.input_encoding,
        "prompt_format": args.prompt_format,
        "decision": decision,
    }
    save_json(output, resolve_path(args.output_json))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
