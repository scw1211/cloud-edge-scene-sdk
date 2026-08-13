"""Deterministic output constraints for single-token action models."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Tuple

from edge_llm_factory.contracts import ManifestError


def normalize_action_tokens(valid_tokens: Any) -> Tuple[str, ...]:
    """Return a stable, duplicate-free tuple of authorized one-character tokens."""

    raw_tokens = valid_tokens.values() if isinstance(valid_tokens, Mapping) else valid_tokens
    if isinstance(raw_tokens, (str, bytes)) or not isinstance(raw_tokens, Sequence):
        # dict_values is iterable but not a Sequence.
        try:
            raw_tokens = tuple(raw_tokens)
        except TypeError as exc:
            raise ManifestError("valid_tokens must be a mapping or token sequence") from exc
    tokens = tuple(str(token) for token in raw_tokens)
    if not tokens:
        raise ManifestError("valid_tokens cannot be empty")
    if any(len(token) != 1 for token in tokens):
        raise ManifestError("action constraint tokens must be single characters")
    if len(set(tokens)) != len(tokens):
        raise ManifestError("action constraint tokens must be unique")
    return tokens


def build_action_token_gbnf(valid_tokens: Any) -> str:
    """Build a llama.cpp GBNF grammar that accepts exactly one authorized token."""

    tokens = normalize_action_tokens(valid_tokens)
    literals = [json.dumps(token, ensure_ascii=False) for token in tokens]
    return "root ::= {}\n".format(" | ".join(literals))


def action_token_constraint_evidence(
    valid_tokens: Any,
    *,
    reserved_tokens: Any = (),
) -> dict:
    """Describe the decoding contract without claiming any post-hoc remapping."""

    tokens = normalize_action_tokens(valid_tokens)
    excluded = normalize_action_tokens(reserved_tokens) if reserved_tokens else ()
    if set(tokens) & set(excluded):
        raise ManifestError("reserved action tokens cannot also be sampled")
    grammar = build_action_token_gbnf(tokens)
    return {
        "enabled": True,
        "backend": "llama.cpp_gbnf",
        "allowed_tokens": list(tokens),
        "reserved_tokens_excluded_from_sampling": list(excluded),
        "grammar": grammar,
        "grammar_sha256": hashlib.sha256(grammar.encode("utf-8")).hexdigest(),
        "applies_before_sampling": True,
        "post_hoc_remapping": False,
    }
