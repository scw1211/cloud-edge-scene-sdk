"""Stable 16-token input codec for the industrial action adapter."""

from __future__ import annotations


MODALITY_CODE = {"rgb": "0", "infrared": "1"}


def margin_position(score: float, review_low: float, review_high: float) -> int:
    """Return score position in thousandths of the calibrated review band."""
    width = float(review_high) - float(review_low)
    if width <= 0:
        raise ValueError("industrial review band must be positive")
    return int(round((float(score) - float(review_low)) / width * 1000.0))


def _margin_code(value: int) -> int:
    return max(-4000, min(4000, int(value))) + 4000


def encode_margin_prompt(
    modality: str,
    product_index: int,
    position_milli: int,
    width_q6: int,
) -> str:
    """Encode normalized lower/upper threshold distances as decimal16."""
    if modality not in MODALITY_CODE:
        raise ValueError("unsupported industrial modality")
    if not 0 <= product_index <= 99 or not 1 <= width_q6 <= 9999:
        raise ValueError("industrial product/width is outside decimal16 contract")
    lower_margin = _margin_code(position_milli)
    upper_margin = _margin_code(position_milli - 1000)
    prompt = "2{}{:02d}{:04d}{:04d}{:04d}".format(
        MODALITY_CODE[modality],
        product_index,
        lower_margin,
        upper_margin,
        width_q6,
    )
    if len(prompt) != 16 or not prompt.isdigit():
        raise AssertionError("industrial v2 prompt must be decimal16")
    return prompt


def prompt_from_values(
    modality: str,
    product_index: int,
    score: float,
    review_low: float,
    review_high: float,
) -> str:
    """Build the model prompt from the fields already present in SemanticEvent."""
    width_q6 = int(round((float(review_high) - float(review_low)) * 1_000_000.0))
    return encode_margin_prompt(
        modality,
        product_index,
        margin_position(score, review_low, review_high),
        width_q6,
    )
