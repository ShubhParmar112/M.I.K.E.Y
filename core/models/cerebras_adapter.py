"""Cerebras adapter — a second free cloud tier, so one exhausted quota is not the
end of the good answers for the day.

Why this provider: its free tier allows roughly ten times Groq's daily tokens
(~1M/day vs 100k as of mid-2026), on hardware fast enough that it is a real
alternative rather than a consolation. Its limits bite in a different place —
few requests per minute rather than few tokens per day — which is exactly what
makes it a useful second link: the two tiers fail under different conditions.

Wire format is OpenAI-compatible, so everything but the endpoint is inherited.
"""

from __future__ import annotations

from typing import Any

from core.models.openai_compat import OpenAICompatAdapter

BASE_URL = "https://api.cerebras.ai/v1"

# The free tier allows only a few requests per minute, and one M.I.K.E.Y turn can
# make several model calls. Backing off longer than the default is what keeps a
# multi-call turn on this provider instead of spilling onto the local model
# halfway through it.
RATE_LIMIT_BACKOFF_S = 4.0


class CerebrasAdapter(OpenAICompatAdapter):
    name = "cerebras"
    base_url = BASE_URL
    api_key_env = "CEREBRAS_API_KEY"
    local = False

    def __init__(self, model: str, **kwargs: Any) -> None:
        kwargs.setdefault("rate_limit_backoff_s", RATE_LIMIT_BACKOFF_S)
        super().__init__(model, **kwargs)
