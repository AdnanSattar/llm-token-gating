from __future__ import annotations

from typing import Dict

import tiktoken

from .config import get_settings
from .state import AgentState

_ENCODERS: Dict[str, tiktoken.Encoding] = {}


def _get_encoder(model: str) -> tiktoken.Encoding:
    if model not in _ENCODERS:
        _ENCODERS[model] = tiktoken.encoding_for_model(model)
    return _ENCODERS[model]


def estimate_tokens(text: str, model: str | None = None) -> int:
    """
    Rough token estimate for a piece of text.
    Used for pre-flight budget checks.
    """
    if not text:
        return 0
    settings = get_settings()
    model_name = model or settings.openai_model_name
    encoder = _get_encoder(model_name)
    return len(encoder.encode(text))


def consume_tokens(
    state: AgentState,
    node_name: str,
    estimated_tokens: int,
) -> AgentState:
    """
    Centralized token accounting.

    Mutates and returns the same state dict for convenience.
    """
    remaining = state.get("remaining_tokens", 0)
    remaining -= max(0, estimated_tokens)
    state["remaining_tokens"] = max(remaining, 0)

    tokens_used = state.get("tokens_used") or {}
    prev = tokens_used.get(node_name, 0)
    tokens_used[node_name] = prev + max(0, estimated_tokens)
    state["tokens_used"] = tokens_used

    return state
