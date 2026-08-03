"""用途：将交通态势与路由上下文编码成供边缘 LLM 使用的短任务序列。"""

import re
from typing import Any, Dict, Mapping


FEATURE_KEYS = ["r", "t", "l", "m", "h", "s", "q", "v", "o", "c", "a", "g"]
DECISION_ORDER = (
    "no_action",
    "congestion_warning",
    "variable_speed_limit",
    "ramp_metering",
    "regional_coordination",
    "reroute",
)
NETWORK_ORDER = ("normal", "weak", "offline")


def parse_legacy_action_prompt(prompt: str) -> Dict[str, int]:
    values = {key: int(value) for key, value in re.findall(r"([a-z])(\d+)", prompt)}
    missing = [key for key in FEATURE_KEYS if key not in values]
    if missing:
        raise ValueError("Action prompt missing fields: {}".format(", ".join(missing)))
    return values


def encode_positional_decimal_prompt(prompt: str) -> str:
    values = parse_legacy_action_prompt(prompt)
    digits = [
        min(9, values["r"]),
        min(9, values["t"]),
        min(9, int(round(values["l"] / 5.0))),
        min(9, int(round(values["m"] / 5.0))),
        min(9, int(round(values["h"] / 5.0))),
        min(9, int(round(values["s"] / 5.0))),
    ]
    continuous = "".join(
        "{:02d}".format(min(99, max(0, values[key]))) for key in ("q", "v", "o")
    )
    flags = "".join(str(min(9, values[key])) for key in ("c", "a", "g"))
    return "".join(str(value) for value in digits) + continuous + flags


def encode_bitpacked_decimal_prompt(prompt: str) -> str:
    values = parse_legacy_action_prompt(prompt)
    prefix = encode_positional_decimal_prompt(prompt)[:-3]
    flags = (
        (1 if values["c"] else 0) << 2
        | (1 if values["a"] else 0) << 1
        | (1 if values["g"] else 0)
    )
    return prefix + str(flags)


def _decision_id(value: Any, field_name: str) -> int:
    decision = str(value)
    if decision not in DECISION_ORDER:
        raise ValueError("{} is not a traffic decision: {!r}".format(field_name, value))
    return DECISION_ORDER.index(decision)


def encode_routing_context_v2_prompt(
    prompt: str,
    routing_context: Mapping[str, Any],
) -> str:
    """Encode current traffic plus Student/rule/network context in 16 digits.

    The first 12 digits retain the established traffic-state code.  The final
    four digits jointly encode actuator flags, Student and rule decisions,
    Student confidence, ambiguity, and network state.  The Cartesian product
    has 6912 values, so four decimal digits are sufficient.
    """
    values = parse_legacy_action_prompt(prompt)
    prefix = encode_positional_decimal_prompt(prompt)[:-3]
    flags = (
        (1 if values["c"] else 0) << 2
        | (1 if values["a"] else 0) << 1
        | (1 if values["g"] else 0)
    )
    student_id = _decision_id(
        routing_context.get("student_decision"), "student_decision"
    )
    rule_id = _decision_id(routing_context.get("rule_decision"), "rule_decision")
    confidence = float(routing_context.get("student_confidence", 0.0))
    confidence_bucket = min(3, max(0, int(confidence * 4.0)))
    ambiguous = 1 if int(routing_context.get("prediction_set_size", 1)) > 1 else 0
    network = str(routing_context.get("network_status", "normal"))
    if network not in NETWORK_ORDER:
        raise ValueError("network_status is invalid: {!r}".format(network))
    network_id = NETWORK_ORDER.index(network)

    packed = flags
    for radix, value in (
        (len(DECISION_ORDER), student_id),
        (len(DECISION_ORDER), rule_id),
        (4, confidence_bucket),
        (2, ambiguous),
        (len(NETWORK_ORDER), network_id),
    ):
        packed = packed * radix + value
    if packed > 9999:
        raise AssertionError("routing context no longer fits four decimal digits")
    return prefix + "{:04d}".format(packed)


def decode_routing_context_v2_prompt(code: str) -> Dict[str, Any]:
    if not re.fullmatch(r"\d{16}", str(code)):
        raise ValueError("routing-context-v2 code must contain exactly 16 digits")
    packed = int(str(code)[-4:])
    decoded = []
    for radix in (
        len(NETWORK_ORDER),
        2,
        4,
        len(DECISION_ORDER),
        len(DECISION_ORDER),
    ):
        decoded.append(packed % radix)
        packed //= radix
    network_id, ambiguous, confidence_bucket, rule_id, student_id = decoded
    flags = packed
    if not 0 <= flags <= 7:
        raise ValueError("routing-context-v2 actuator flags are invalid")
    return {
        "traffic_code": str(code)[:12],
        "student_decision": DECISION_ORDER[student_id],
        "rule_decision": DECISION_ORDER[rule_id],
        "student_confidence_bucket": confidence_bucket,
        "prediction_set_ambiguous": bool(ambiguous),
        "prediction_set_size": 2 if ambiguous else 1,
        "network_status": NETWORK_ORDER[network_id],
        "cluster": bool(flags & 4),
        "has_ramp_meter": bool(flags & 2),
        "has_reroute_gateway": bool(flags & 1),
    }


def encode_contextual_decimal_prompt(prompt: str) -> str:
    values = parse_legacy_action_prompt(prompt)
    if "e" not in values or "x" not in values:
        raise ValueError("Contextual action prompt requires edge id 'e' and region score 'x'.")
    edge_id = min(9, max(0, values["e"]))
    region_score = min(99, max(0, values["x"]))
    return "{}{:02d}{}".format(edge_id, region_score, encode_bitpacked_decimal_prompt(prompt))
