"""Model Gateway (review M2): the only door to any LLM.

Modules request a completion; the gateway picks the provider. No other module
may import a provider SDK — that rule is what makes 'hybrid local+cloud'
implementable later (privacy tiers route here at Gen 2+).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatMessage:
    """Provider-neutral message. Adapters translate to vendor wire formats."""

    role: str  # "user" | "assistant" | "tool_result"
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set when role == "tool_result"


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: dict[str, int] = field(default_factory=dict)


class ModelUnavailable(Exception):
    """A provider could not serve the request for a reason another provider might
    survive — rate limits (429), 5xx, or the host being unreachable/offline.

    Adapters raise this (instead of a vendor-specific error) so the gateway can
    fall back without importing any provider's exception types. Errors that a
    fallback would NOT fix — bad API key, malformed request — stay as their own
    exceptions and are not caught here."""

    def __init__(self, provider: str, reason: str, retry_after: float | None = None) -> None:
        super().__init__(f"{provider} unavailable: {reason}")
        self.provider = provider
        self.reason = reason
        self.retry_after = retry_after


class ModelAdapter(Protocol):
    name: str

    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> ModelResponse: ...


class ModelGateway:
    """The one door to any LLM. Given a primary adapter and an optional fallback,
    it serves from the primary and, only when the primary raises ModelUnavailable
    (rate-limited/offline), transparently retries on the fallback — the hybrid
    cloud+local routing the module was always meant to enable."""

    def __init__(self, adapter: ModelAdapter, fallback: ModelAdapter | None = None) -> None:
        self._adapter = adapter
        self._fallback = fallback
        self.total_calls = 0
        # Which adapter actually served the last completion (for traces/health).
        self.last_provider = adapter.name

    @property
    def provider(self) -> str:
        return self._adapter.name

    @property
    def fallback_provider(self) -> str | None:
        return self._fallback.name if self._fallback else None

    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.total_calls += 1
        try:
            resp = await self._adapter.complete(system, messages, tools)
            self.last_provider = self._adapter.name
            return resp
        except ModelUnavailable as primary_err:
            if self._fallback is None:
                # No local safety net: surface a clear, actionable message rather
                # than a raw HTTP error, and keep the retry hint if we have one.
                hint = (
                    f" retry in ~{int(primary_err.retry_after)}s"
                    if primary_err.retry_after
                    else " try again shortly"
                )
                raise RuntimeError(
                    f"{primary_err.provider} is unavailable ({primary_err.reason}) and no "
                    f"local fallback is configured —{hint}, or install Ollama for an offline "
                    "fallback."
                ) from primary_err
            try:
                resp = await self._fallback.complete(system, messages, tools)
                self.last_provider = self._fallback.name
                return resp
            except ModelUnavailable as fb_err:
                raise RuntimeError(
                    f"both providers are unavailable: {primary_err.provider} "
                    f"({primary_err.reason}), and the {fb_err.provider} fallback "
                    f"({fb_err.reason}). If you want offline coverage, make sure Ollama "
                    "is running with a model pulled."
                ) from fb_err
