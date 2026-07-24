"""Model Gateway (review M2): the only door to any LLM.

Modules request a completion; the gateway picks the provider. No other module
may import a provider SDK — that rule is what makes 'hybrid local+cloud'
implementable later (privacy tiers route here at Gen 2+).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from core.events.schema import Tier


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


@dataclass(frozen=True)
class RoutingMeta:
    """Per-request routing hints (architecture 02 §8; ADR-001's
    `complete(request, {tier, capability, budget})`). Carried by the Gateway,
    NOT the adapters — adapters never see it, so adding it breaks nothing.

    Sovereignty S0: only `tier` is enforced today (T0 → local-only, §3 privacy).
    `capability` and `budget` are recorded now and become routing inputs when the
    Router brain and hybrid routing land (sovereignty S1/S3). Absent meta ==
    today's behavior exactly: a T1 request down the existing fallback chain.
    """

    tier: Tier = Tier.T1
    capability: str = "general"  # e.g. "plan" | "code" | "recall" | "chat"
    max_output_tokens: int | None = None  # budget hint; None = adapter default


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
    # Optional convention (read via getattr, default False): True means the
    # adapter runs entirely on-device (no data leaves the machine). The Gateway
    # uses it to enforce the Tier-0 privacy rule. Cloud adapters omit it or set
    # it False; the local (Ollama) and fake adapters set it True.

    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> ModelResponse: ...


def _is_local(adapter: ModelAdapter) -> bool:
    return bool(getattr(adapter, "local", False))


class ModelGateway:
    """The one door to any LLM. Given a primary adapter and an ordered chain of
    fallbacks, it serves from the primary and, only when a provider raises
    ModelUnavailable (rate-limited/offline), transparently tries the next link —
    e.g. groq → claude → local — so a rate limit or outage never kills a turn.

    `routes` (sovereignty S2) pins a capability to a preferred adapter — e.g. the
    critic served by a local model — so brains can be localized one at a time. A
    routed capability still keeps the default chain behind it as fallback, and the
    Tier-0 privacy rule always overrides routing."""

    def __init__(
        self,
        adapter: ModelAdapter,
        fallback: ModelAdapter | None = None,
        fallbacks: list[ModelAdapter] | None = None,
        routes: dict[str, ModelAdapter] | None = None,
    ) -> None:
        self._adapter = adapter
        if fallbacks is not None:
            self._fallbacks = list(fallbacks)
        elif fallback is not None:
            self._fallbacks = [fallback]
        else:
            self._fallbacks = []
        self._routes = dict(routes or {})  # capability -> preferred adapter
        self.total_calls = 0
        # Which adapter actually served the last completion (for traces/health).
        self.last_provider = adapter.name

    @property
    def routed_capabilities(self) -> dict[str, str]:
        """capability -> adapter name, for health/observability."""
        return {cap: a.name for cap, a in self._routes.items()}

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
        meta: RoutingMeta | None = None,
    ) -> ModelResponse:
        self.total_calls += 1
        default_chain: list[ModelAdapter] = [self._adapter, *self._fallbacks]

        # Per-capability routing (S2): a capability pinned to a specific adapter
        # (e.g. a local model for one brain) leads, with the default chain behind it
        # as fallback (deduped by name). No route → today's default chain exactly.
        if meta is not None and meta.capability in self._routes:
            routed = self._routes[meta.capability]
            candidates: list[ModelAdapter] = [routed] + [
                a for a in default_chain if a.name != routed.name
            ]
        else:
            candidates = list(default_chain)

        # Tier-0 privacy is a HARD constraint enforced here, not by convention
        # (architecture 02 §3): private data may only be served by a local model.
        # Refusing is the correct failure — leaking T0 to the cloud is not. This
        # overrides any capability route (a route to a cloud model is dropped for T0).
        if meta is not None and meta.tier is Tier.T0:
            candidates = [a for a in candidates if _is_local(a)]
            if not candidates:
                raise RuntimeError(
                    "refusing to serve Tier-0 (private) data: no local model is "
                    "configured. T0 must never reach a cloud provider — install/start "
                    "Ollama (or set MIKEY_PROVIDER=ollama) so private turns stay on-device."
                )

        errors: list[ModelUnavailable] = []
        for adapter in candidates:
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
