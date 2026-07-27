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

# Free-tier daily TOKEN allowances, by provider (approximate, early 2026).
#
# On a paid plan the binding constraint is dollars per month, which the budget
# above governs. On a FREE plan it is tokens per DAY — and running out does not
# look like an error: the provider 429s, the gateway falls back, and every
# remaining turn that day is quietly served by a much weaker local model. That is
# how a whole evening's answers got worse with no visible cause. This table
# exists to make the cliff visible while there is still room to do something
# about it.
FREE_TIER_DAILY_TOKENS: dict[str, int] = {"groq": 100_000}

# Fraction of a daily allowance at which to start saying so. Lower than the
# monthly WARN_AT: a day's allowance can go in a single long conversation, so the
# warning has to arrive with real room left.
DAILY_WARN_AT = 0.75


def daily_cap_for(provider: str, override: int = 0) -> int:
    """Tokens this provider allows per day; 0 means "no known cap" (paid plans,
    local models). `override` (MIKEY_DAILY_TOKEN_CAP) replaces the table entry for
    a provider that has one — for when a free tier is upgraded or its limits move."""
    if provider in LOCAL_PROVIDERS:
        return 0
    cap = FREE_TIER_DAILY_TOKENS.get(provider, 0)
    if cap and override > 0:
        return override
    return cap


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


@dataclass(frozen=True)
class ProviderDay:
    """One provider's token consumption so far today, against its daily allowance."""

    provider: str
    calls: int
    input_tokens: int
    output_tokens: int
    cap: int = 0  # 0 = no known daily cap

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def capped(self) -> bool:
        return self.cap > 0

    @property
    def fraction(self) -> float:
        return self.total_tokens / self.cap if self.capped else 0.0

    @property
    def exhausted(self) -> bool:
        # Our count only includes calls M.I.K.E.Y itself recorded, so it is a LOWER
        # bound on what the provider has counted. That asymmetry is why this is safe
        # to report as fact: if our own tally has passed the cap, the provider's has
        # too. The converse does not hold, so being under it proves nothing.
        return self.capped and self.total_tokens >= self.cap

    @property
    def warning(self) -> bool:
        return self.capped and not self.exhausted and self.fraction >= DAILY_WARN_AT

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.cap - self.total_tokens) if self.capped else 0

    @property
    def calls_left(self) -> int | None:
        """Roughly how many more calls fit in what's left, at today's average call
        size. None when there's no cap or nothing to average yet.

        A token count is hard to act on; "about four more exchanges" is not."""
        if not self.capped or self.calls == 0:
            return None
        return int(self.remaining_tokens // max(1, self.total_tokens / self.calls))


@dataclass(frozen=True)
class DayUsage:
    day: str  # "2026-07-27"
    providers: list[ProviderDay] = field(default_factory=list)  # busiest first

    @property
    def total_tokens(self) -> int:
        return sum(p.total_tokens for p in self.providers)

    @property
    def calls(self) -> int:
        return sum(p.calls for p in self.providers)

    @property
    def pressured(self) -> ProviderDay | None:
        """The capped provider closest to running out, once it's worth mentioning."""
        at_risk = [p for p in self.providers if p.exhausted or p.warning]
        return max(at_risk, key=lambda p: p.fraction) if at_risk else None


def _month_key(at: datetime) -> str:
    return at.strftime("%Y-%m")


def _day_key(at: datetime) -> str:
    return at.strftime("%Y-%m-%d")


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
        daily_token_cap: int = 0,
    ) -> None:
        self._events = events
        self._budget = max(0.0, budget_usd)
        self._device = device
        self._daily_cap_override = max(0, daily_token_cap)
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

    def today(self, at: datetime | None = None) -> DayUsage:
        """Today's token consumption per provider, against each daily allowance.

        Read straight from the log every time rather than cached: it is a report
        path called a few times a session, and a stale gauge is worse than none.

        Deliberately NOT wired into `cloud_allowed`. A provider's daily window need
        not line up with the local calendar day, and our tally starts at whatever
        moment usage logging was first deployed — so this number is honest as a
        warning and would be wrong as a gate. The provider's own 429 remains the
        authority on when to stop; this is here so the person sees it coming.
        """
        at = at or now()
        start = at.replace(hour=0, minute=0, second=0, microsecond=0)
        totals: dict[str, list[int]] = {}
        for ev in self._events.since(start.isoformat(), [EventType.MODEL_USAGE.value]):
            provider = str(ev.payload.get("provider", "unknown"))
            row = totals.setdefault(provider, [0, 0, 0])
            row[0] += 1
            row[1] += int(ev.payload.get("input_tokens", 0) or 0)
            row[2] += int(ev.payload.get("output_tokens", 0) or 0)
        providers = [
            ProviderDay(
                provider=provider,
                calls=calls,
                input_tokens=inp,
                output_tokens=out,
                cap=daily_cap_for(provider, self._daily_cap_override),
            )
            for provider, (calls, inp, out) in totals.items()
        ]
        providers.sort(key=lambda p: -p.total_tokens)
        return DayUsage(day=_day_key(at), providers=providers)

    def cloud_allowed(self, at: datetime | None = None) -> bool:
        """False once the month's budget is spent. Local inference is unaffected —
        the point is to stop the meter, not to stop the assistant."""
        return not self.month_to_date(at).over_budget
