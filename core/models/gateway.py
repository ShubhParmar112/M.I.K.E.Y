"""Model Gateway (review M2): the only door to any LLM.

Modules request a completion; the gateway picks the provider. No other module
may import a provider SDK — that rule is what makes 'hybrid local+cloud'
implementable later (privacy tiers route here at Gen 2+).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from core.events.schema import Tier, now

# How long a provider that reported a DAILY quota is left alone when it didn't say
# when to come back. Long enough to stop paying a failed round-trip per model call
# (a turn makes several), short enough that a quota window rolling over is noticed
# within the hour rather than at midnight.
DEFAULT_DAILY_COOLDOWN = timedelta(hours=1)
MAX_COOLDOWN = timedelta(hours=24)


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
    # Adapter-level notes about how this response was produced — e.g. that a
    # collapsed generation was re-sampled. Recorded in the turn's trace so a
    # degraded reply is visible after the fact instead of silently smoothed over.
    notes: list[str] = field(default_factory=list)


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

    def __init__(
        self,
        provider: str,
        reason: str,
        retry_after: float | None = None,
        daily: bool = False,
    ) -> None:
        super().__init__(f"{provider} unavailable: {reason}")
        self.provider = provider
        self.reason = reason
        self.retry_after = retry_after
        # True when the provider is out for the DAY, not for a moment. The two are
        # operationally opposite: a per-minute spike is worth waiting out, a daily
        # cap means every remaining turn today is served by the fallback and the
        # person should be told plainly rather than left wondering why answers got
        # worse. Retrying a daily cap is pure dead time.
        self.daily = daily


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


class Governor(Protocol):
    """The cost governor, as the gateway needs it (core.cost.governor.CostGovernor).
    Declared structurally so the gateway keeps depending on nothing."""

    def cloud_allowed(self) -> bool: ...

    def record(
        self,
        provider: str,
        model: str,
        usage: dict[str, int],
        capability: str = "general",
        tier: Tier = Tier.T1,
    ) -> float: ...


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
        governor: Governor | None = None,
    ) -> None:
        self._adapter = adapter
        # Cost governor (Gen 3). Optional: without one the gateway behaves exactly
        # as before, which is what tests and one-off tooling want.
        self._governor = governor
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
        # Why the primary was skipped, when it was — so the surface can tell the
        # person "rate-limited for a moment" apart from "out of quota until tomorrow,
        # everything you ask today comes from the weaker model".
        self.last_fallback_reason: str | None = None
        self.last_fallback_daily = False
        # Providers that told us they are out for the day, and when they may be
        # tried again. Without this, every model call of every turn for the rest of
        # the day pays a failed round-trip to a provider we already know has
        # nothing left — several times per turn, since a turn makes several calls.
        self._sidelined: dict[str, datetime] = {}

    @property
    def sidelined(self) -> dict[str, str]:
        """provider -> ISO time it may be tried again, for health/observability."""
        at = now()
        return {n: t.isoformat() for n, t in self._sidelined.items() if t > at}

    def _available(self, candidates: list[ModelAdapter]) -> list[ModelAdapter]:
        """Drop providers known to be out of quota — unless that would leave
        nothing, in which case try them anyway. Our record can be stale (a quota
        window may have rolled over early), and a stale note must never be the
        reason a turn fails outright."""
        at = now()
        live = [a for a in candidates if self._sidelined.get(a.name, at) <= at]
        return live or candidates

    def _sideline(self, adapter: ModelAdapter, exc: ModelUnavailable) -> None:
        """Remember that a provider is finished for now. The provider's own
        retry-after is trusted when it gives one — a daily cap usually comes with
        the time the window rolls over, which is better than any guess."""
        wait = DEFAULT_DAILY_COOLDOWN
        if exc.retry_after and exc.retry_after > 0:
            wait = min(timedelta(seconds=float(exc.retry_after)), MAX_COOLDOWN)
        self._sidelined[adapter.name] = now() + wait

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

        # The monthly budget is enforced HERE, at the same single door as the
        # privacy rule and for the same reason (Gen 3: "spend within budget
        # automatically"). Once the budget is gone the cloud adapters drop out of
        # the chain and the local model carries the load: the meter stops, the
        # assistant does not. Only with no local model at all is this an error.
        if self._governor is not None and not self._governor.cloud_allowed():
            local_only = [a for a in candidates if _is_local(a)]
            if not local_only:
                raise RuntimeError(
                    "this month's model budget is spent and no local model is "
                    "configured to fall back on. Raise MIKEY_MONTHLY_BUDGET_USD, wait "
                    "for the month to roll over, or install/start Ollama so M.I.K.E.Y "
                    "keeps working for free. `mikey spend` shows where it went."
                )
            candidates = local_only

        # A provider that has already told us it is out for the day is skipped
        # rather than asked again. The order of what remains is unchanged.
        candidates = self._available(candidates)

        errors: list[ModelUnavailable] = []
        for adapter in candidates:
            try:
                resp = await adapter.complete(system, messages, tools)
                self.last_provider = adapter.name
                self.last_fallback_reason = errors[0].reason if errors else None
                self.last_fallback_daily = bool(errors and errors[0].daily)
                if adapter.name in self._sidelined:
                    del self._sidelined[adapter.name]  # it's answering again
                if self._governor is not None:
                    # Bill the adapter that actually served, not the one asked —
                    # a fallback to local is genuinely free and must read as free.
                    self._governor.record(
                        adapter.name,
                        str(getattr(adapter, "model", "")),
                        resp.usage,
                        meta.capability if meta is not None else "general",
                        meta.tier if meta is not None else Tier.T1,
                    )
                return resp
            except ModelUnavailable as exc:
                errors.append(exc)  # this provider is down; try the next link
                if exc.daily:
                    self._sideline(adapter, exc)

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
