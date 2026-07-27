"""Google Gemini adapter, over Google's OpenAI-compatibility endpoint.

Why this provider: its free tier is metered in *requests* per day (hundreds to a
low thousand for the Flash models) rather than tokens, which is the limit that
suits M.I.K.E.Y worst-case least — a long document in context costs nothing extra
against it. It is also the strongest free model in the chain, so when Groq's
tokens run out the step down is small.

Google publishes an OpenAI-compatible surface, so the wire format is the shared
one; only the endpoint differs.
"""

from __future__ import annotations

from core.models.openai_compat import OpenAICompatAdapter

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


class GeminiAdapter(OpenAICompatAdapter):
    name = "gemini"
    base_url = BASE_URL
    api_key_env = "GEMINI_API_KEY"
    local = False
    # The compatibility layer implements a subset of OpenAI's sampling parameters
    # and rejects the rest outright. The penalties are only used when re-sampling a
    # collapsed reply, so dropping them costs a slightly weaker recovery attempt —
    # a 400 in the middle of one would cost the whole turn.
    supports_penalties = False
