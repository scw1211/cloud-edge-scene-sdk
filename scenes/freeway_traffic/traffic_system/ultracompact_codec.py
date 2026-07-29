"""用途：将交通态势的 12 个离散特征编码成供边缘 LLM 使用的短任务序列。"""

import re
from typing import Dict


FEATURE_KEYS = ["r", "t", "l", "m", "h", "s", "q", "v", "o", "c", "a", "g"]
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


def encode_contextual_decimal_prompt(prompt: str) -> str:
    values = parse_legacy_action_prompt(prompt)
    if "e" not in values or "x" not in values:
        raise ValueError("Contextual action prompt requires edge id 'e' and region score 'x'.")
    edge_id = min(9, max(0, values["e"]))
    region_score = min(99, max(0, values["x"]))
    return "{}{:02d}{}".format(edge_id, region_score, encode_bitpacked_decimal_prompt(prompt))
