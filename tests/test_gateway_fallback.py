"""Hybrid routing: the gateway serves from the cloud primary and falls back to
a local model only on rate-limit/offline (ModelUnavailable), never on errors a
fallback wouldn't fix (bad key, malformed request)."""

from __future__ import annotations

import httpx
import pytest

from core.models.gateway import (
    ChatMessage,
    ModelGateway,
    ModelResponse,
    ModelUnavailable,
)
from core.models.groq_adapter import GroqAdapter


class _Fake:
    def __init__(self, name: str, response: ModelResponse | None = None,
                 error: Exception | None = None) -> None:
        self.name = name
        self._response = response
        self._error = error
        self.calls = 0

    async def complete(self, system, messages, tools) -> ModelResponse:  # noqa: ANN001
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def _msgs() -> list[ChatMessage]:
    return [ChatMessage(role="user", text="hi")]


async def test_primary_serves_and_fallback_untouched() -> None:
    primary = _Fake("groq", ModelResponse(text="from cloud", tool_calls=[]))
    fallback = _Fake("ollama", ModelResponse(text="from local", tool_calls=[]))
    gw = ModelGateway(primary, fallback=fallback)

    resp = await gw.complete("", _msgs(), [])
    assert resp.text == "from cloud"
    assert gw.last_provider == "groq"
    assert fallback.calls == 0


async def test_falls_back_when_primary_unavailable() -> None:
    primary = _Fake("groq", error=ModelUnavailable("groq", "rate limited (429)"))
    fallback = _Fake("ollama", ModelResponse(text="from local", tool_calls=[]))
    gw = ModelGateway(primary, fallback=fallback)

    resp = await gw.complete("", _msgs(), [])
    assert resp.text == "from local"
    assert gw.last_provider == "ollama"  # trace/health reflect who actually served
    assert primary.calls == 1 and fallback.calls == 1


async def test_no_fallback_gives_actionable_error() -> None:
    primary = _Fake("groq", error=ModelUnavailable("groq", "rate limited (429)", retry_after=8))
    gw = ModelGateway(primary)  # no fallback configured

    with pytest.raises(RuntimeError) as ei:
        await gw.complete("", _msgs(), [])
    msg = str(ei.value)
    assert "groq" in msg and "8s" in msg and "Ollama" in msg


async def test_both_unavailable_reports_both() -> None:
    primary = _Fake("groq", error=ModelUnavailable("groq", "rate limited (429)"))
    fallback = _Fake("ollama", error=ModelUnavailable("ollama", "not running"))
    gw = ModelGateway(primary, fallback=fallback)

    with pytest.raises(RuntimeError) as ei:
        await gw.complete("", _msgs(), [])
    msg = str(ei.value)
    assert "groq" in msg and "ollama" in msg and "not running" in msg


async def test_non_retriable_error_is_not_caught() -> None:
    """A ValueError (or any non-ModelUnavailable) must propagate untouched — the
    fallback path is only for provider-availability failures."""
    boom = ValueError("programming error")
    primary = _Fake("groq", error=boom)
    fallback = _Fake("ollama", ModelResponse(text="local", tool_calls=[]))
    gw = ModelGateway(primary, fallback=fallback)

    with pytest.raises(ValueError):
        await gw.complete("", _msgs(), [])
    assert fallback.calls == 0  # we did NOT mask the bug by falling back


# ---- Groq adapter translates HTTP failures into ModelUnavailable ----


async def test_groq_429_becomes_unavailable_with_retry_after() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "12"},
                              json={"error": {"message": "rate limit"}})

    adapter = GroqAdapter("m", api_key="k", transport=httpx.MockTransport(handler))
    with pytest.raises(ModelUnavailable) as ei:
        await adapter.complete("", _msgs(), [])
    assert ei.value.retry_after == 12.0
    assert "429" in ei.value.reason


async def test_groq_offline_becomes_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("name resolution failed")

    adapter = GroqAdapter("m", api_key="k", transport=httpx.MockTransport(handler))
    with pytest.raises(ModelUnavailable) as ei:
        await adapter.complete("", _msgs(), [])
    assert "unreachable" in ei.value.reason


async def test_groq_auth_error_is_not_a_fallback_trigger() -> None:
    """401 means a bad key — a real config bug. Falling back to local would hide
    it, so it must NOT surface as ModelUnavailable."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    adapter = GroqAdapter("m", api_key="bad", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await adapter.complete("", _msgs(), [])
