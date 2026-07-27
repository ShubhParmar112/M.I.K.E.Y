"""Groq adapter — Tier-1 cloud inference over the OpenAI-compatible API.

Groq serves open models (Llama 3.x etc.) with very fast inference and a free
tier. Privacy-wise it is a cloud provider like Anthropic, not a local runtime.

The protocol lives in `core.models.openai_compat`, which Cerebras and Google's
compatibility endpoint share; only what is specific to Groq stays here.
"""

from __future__ import annotations

from core.models.openai_compat import (  # re-exported: imported by name elsewhere
    MAX_RATE_LIMIT_BACKOFF_S,
    RESAMPLE_FREQUENCY_PENALTY,
    RESAMPLE_PRESENCE_PENALTY,
    RESAMPLE_TEMPERATURE,
    OpenAICompatAdapter,
    _is_daily_limit,
    _parse_inline_tool_calls,
    _rate_limit_reason,
    _retry_after,
)

BASE_URL = "https://api.groq.com/openai/v1"

__all__ = [
    "BASE_URL",
    "MAX_RATE_LIMIT_BACKOFF_S",
    "RESAMPLE_FREQUENCY_PENALTY",
    "RESAMPLE_PRESENCE_PENALTY",
    "RESAMPLE_TEMPERATURE",
    "GroqAdapter",
    "_is_daily_limit",
    "_parse_inline_tool_calls",
    "_rate_limit_reason",
    "_retry_after",
]


class GroqAdapter(OpenAICompatAdapter):
    name = "groq"
    base_url = BASE_URL
    api_key_env = "GROQ_API_KEY"
    local = False  # cloud provider; never eligible to serve Tier-0 data
