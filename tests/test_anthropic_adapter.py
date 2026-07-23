"""The Anthropic adapter must speak the failover protocol: rate-limit / overload
/ offline become ModelUnavailable (so the gateway rolls to the next provider),
while an auth error stays a hard error (a real bug the fallback must not mask)."""

from __future__ import annotations

import anthropic
import httpx
import pytest

from core.models.anthropic_adapter import AnthropicAdapter
from core.models.gateway import ChatMessage, ModelUnavailable

_REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


class _FakeMessages:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def create(self, **kwargs: object) -> object:
        raise self._exc


class _FakeClient:
    def __init__(self, exc: Exception) -> None:
        self.messages = _FakeMessages(exc)


def _adapter(exc: Exception) -> AnthropicAdapter:
    return AnthropicAdapter("claude-x", client=_FakeClient(exc))  # type: ignore[arg-type]


async def _run(adapter: AnthropicAdapter):
    return await adapter.complete("sys", [ChatMessage(role="user", text="hi")], [])


async def test_offline_becomes_unavailable() -> None:
    with pytest.raises(ModelUnavailable) as ei:
        await _run(_adapter(anthropic.APIConnectionError(request=_REQ)))
    assert "unreachable" in ei.value.reason


async def test_rate_limit_becomes_unavailable_with_retry_after() -> None:
    resp = httpx.Response(429, request=_REQ, headers={"retry-after": "7"})
    with pytest.raises(ModelUnavailable) as ei:
        await _run(_adapter(anthropic.RateLimitError("slow down", response=resp, body=None)))
    assert ei.value.retry_after == 7.0


async def test_overloaded_5xx_becomes_unavailable() -> None:
    resp = httpx.Response(529, request=_REQ)
    with pytest.raises(ModelUnavailable):
        await _run(_adapter(anthropic.InternalServerError("overloaded", response=resp, body=None)))


async def test_auth_error_is_not_masked() -> None:
    resp = httpx.Response(401, request=_REQ)
    with pytest.raises(anthropic.AuthenticationError):
        await _run(_adapter(anthropic.AuthenticationError("bad key", response=resp, body=None)))
