"""Cost governor (Gen 3): keep monthly spend inside a budget, automatically.

Gen 3's third exit criterion is "monthly spend within budget automatically" —
*automatically* being the whole of it. A dashboard that reports an overspend
after the fact is not a governor; the budget has to be able to say no.

Three properties, in order of importance:

1. **The log is the ledger.** Every model call appends a `model.usage` event with
   its token counts and computed cost, so month-to-date spend is a projection over
   the event log like everything else — it survives a restart, a rebuilt index, and
   a restore-from-backup, and `mikey trace` can explain any line of it.
2. **Enforcement lives at the gateway**, the single door every completion passes
   through — the same place the Tier-0 privacy rule is enforced, for the same
   reason: a rule that each caller must remember to apply is not a rule.
3. **Over budget degrades, it does not break.** A local model costs nothing, so
   the budget stops *cloud* calls and lets Ollama keep serving. M.I.K.E.Y gets
   slower and dumber at the end of an expensive month; it does not stop working,
   and it does not quietly keep spending.

An unpriced model is charged at a deliberately high rate rather than zero: an
unknown model reading as free is exactly how a budget silently fails to bind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from core.events.schema import Event, EventType, Provenance, Tier, now
from core.events.store import EventStore


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""

    input_per_1m: float
    output_per_1m: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_1m + output_tokens * self.output_per_1m
        ) / 1_000_000


# Providers that run on the user's own machine: no marginal cost, ever.
LOCAL_PROVIDERS = frozenset({"ollama", "fake"})

# Published list prices, matched as a substring of the model id. APPROXIMATE and
# dated (early 2026) — vendor pricing drifts, so treat this as a budget guide, not
# an invoice. `mikey spend` labels a month whose total used a fallback price.
PRICES: dict[str, Price] = {
    "llama-3.3-70b": Price(0.59, 0.79),
    "llama-3.1-8b": Price(0.05, 0.08),
    "claude-opus": Price(15.0, 75.0),
    "claude-sonnet": Price(3.0, 15.0),
    "claude-haiku": Price(1.0, 5.0),
}

# What an unrecognized cloud model costs, for budgeting purposes. Set high on
# purpose: over-estimating an unknown model spends the budget early, which is a
# recoverable annoyance; under-estimating it spends the user's money, which is not.
UNKNOWN_CLOUD_PRICE = Price(15.0, 75.0)

# Fraction of the budget at which the user should be told, while there is still
# room to do something about it.
WARN_AT = 0.8


def price_for(provider: str, model: str) -> tuple[Price, bool]:
    """(price, exact). `exact` is False when the model wasn't in the table and the
    conservative fallback was used."""
    if provider in LOCAL_PROVIDERS:
        return Price(0.0, 0.0), True
    key = (model or "").lower()
    for prefix, price in PRICES.items():
        if prefix in key:
            return price, True
    return UNKNOWN_CLOUD_PRICE, False


@dataclass(frozen=True)
class Spend:
    month: str  # "2026-07"
    total_usd: float
    budget_usd: float  # 0.0 = enforcement disabled (tracking continues)
    calls: int
    by_provider: dict[str, float] = field(default_factory=dict)
    estimated: bool = False  # some call was priced with the unknown-model fallback

    @property
    def enforced(self) -> bool:
        return self.budget_usd > 0

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.budget_usd - self.total_usd) if self.enforced else 0.0

    @property
    def fraction(self) -> float:
        return self.total_usd / self.budget_usd if self.enforced else 0.0

    @property
    def over_budget(self) -> bool:
        return self.enforced and self.total_usd >= self.budget_usd

    @property
    def warning(self) -> bool:
        return self.enforced and not self.over_budget and self.fraction >= WARN_AT


def _month_key(at: datetime) -> str:
    return at.strftime("%Y-%m")


class CostGovernor:
    """Tracks model spend in the event log and decides whether cloud calls may
    still be made this month.

    The running total is cached in memory and rebuilt from the log whenever the
    month rolls over or the process restarts — the cache is an optimization, the
    log is the truth."""

    def __init__(
        self,
        events: EventStore,
        budget_usd: float,
        device: str = "dev_desktop_1",
    ) -> None:
        self._events = events
        self._budget = max(0.0, budget_usd)
        self._device = device
        self._month: str | None = None
        self._total = 0.0
        self._calls = 0
        self._by_provider: dict[str, float] = {}
        self._estimated = False

    # ---- ledger ----

    def record(
        self,
        provider: str,
        model: str,
        usage: dict[str, int],
        capability: str = "general",
        tier: Tier = Tier.T1,
    ) -> float:
        """Append one call to the ledger and return what it cost."""
        price, exact = price_for(provider, model)
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cost = price.cost(input_tokens, output_tokens)
        # Refresh the cache from the log BEFORE appending, never after: a reload
        # that included this call would then be added to again, billing it twice.
        at = now()
        if self._month != _month_key(at):
            self._reload(at)
        self._events.append(
            Event(
                type=EventType.MODEL_USAGE.value,
                device=self._device,
                tier=tier,
                provenance=Provenance(source="agent", trusted=True),
                payload={
                    "provider": provider,
                    "model": model,
                    "capability": capability,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost,
                    "priced": "table" if exact else "fallback",
                },
            )
        )
        self._total += cost
        self._calls += 1
        self._by_provider[provider] = self._by_provider.get(provider, 0.0) + cost
        self._estimated = self._estimated or not exact
        return cost

    def _reload(self, at: datetime) -> None:
        start = at.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        self._month = _month_key(at)
        self._total = 0.0
        self._calls = 0
        self._by_provider = {}
        self._estimated = False
        for ev in self._events.since(start.isoformat(), [EventType.MODEL_USAGE.value]):
            cost = float(ev.payload.get("cost_usd", 0.0) or 0.0)
            provider = str(ev.payload.get("provider", "unknown"))
            self._total += cost
            self._calls += 1
            self._by_provider[provider] = self._by_provider.get(provider, 0.0) + cost
            self._estimated = self._estimated or ev.payload.get("priced") == "fallback"

    # ---- the decision ----

    def month_to_date(self, at: datetime | None = None) -> Spend:
        at = at or now()
        if self._month != _month_key(at):
            self._reload(at)
        return Spend(
            month=self._month or _month_key(at),
            total_usd=self._total,
            budget_usd=self._budget,
            calls=self._calls,
            by_provider=dict(self._by_provider),
            estimated=self._estimated,
        )

    def cloud_allowed(self, at: datetime | None = None) -> bool:
        """False once the month's budget is spent. Local inference is unaffected —
        the point is to stop the meter, not to stop the assistant."""
        return not self.month_to_date(at).over_budget
