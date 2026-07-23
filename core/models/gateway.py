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
    """The one door to any LLM. Given a primary adapter and an ordered chain of
    fallbacks, it serves from the primary and, only when a provider raises
    ModelUnavailable (rate-limited/offline), transparently tries the next link —
    e.g. groq → claude → local — so a rate limit or outage never kills a turn."""

    def __init__(
        self,
        adapter: ModelAdapter,
        fallback: ModelAdapter | None = None,
        fallbacks: list[ModelAdapter] | None = None,
    ) -> None:
        self._adapter = adapter
        if fallbacks is not None:
            self._fallbacks = list(fallbacks)
        elif fallback is not None:
            self._fallbacks = [fallback]
        else:
            self._fallbacks = []
        self.total_calls = 0
        # Which adapter actually served the last completion (for traces/health).
        self.last_provider = adapter.name

    @property
    def provider(self) -> str:
        return self._adapter.name

    @property
    def fallback_provider(self) -> str | None:
        return ", ".join(f.name for f in self._fallbacks) or None

    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.total_calls += 1
        errors: list[ModelUnavailable] = []
        for adapter in (self._adapter, *self._fallbacks):
            try:
                resp = await adapter.complete(system, messages, tools)
                self.last_provider = adapter.name
                return resp
            except ModelUnavailable as exc:
                errors.append(exc)  # this provider is down; try the next link

        primary_err = errors[0]
        if len(errors) == 1:  # nothing to fall back to
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
        detail = "; ".join(f"{e.provider} ({e.reason})" for e in errors)
        raise RuntimeError(
            f"all providers are unavailable: {detail}. If you want offline coverage, "
            "make sure Ollama is running with a model pulled."
        ) from errors[-1]
