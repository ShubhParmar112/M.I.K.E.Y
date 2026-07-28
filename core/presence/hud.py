"""What the HUD says, decided in one pure place.

Everything here is a function of data the gateway already publishes (`/v1/health`,
`/v1/nudges`, `/v1/missions`) plus who actually answered the last turn. No I/O, no
widgets — so the judgements that matter can be tested without a terminal, and a
second surface (a web dashboard, a status line) can reuse them unchanged.

The one judgement worth being careful about is the verdict. A dashboard that is
always amber is wallpaper; a dashboard that says "ok" while a 3B local model is
answering is worse than none. So the precedence is by CONSEQUENCE, not by
severity of the underlying fact: an exhausted allowance with another cloud
provider behind it is a non-event, and the same allowance with nothing behind it
is the single most important thing on the screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.cost.governor import LOCAL_PROVIDERS

OK = "ok"
DEGRADED = "degraded"
BROKEN = "broken"

#: The headline when there is genuinely nothing to say. Anything else, even at
#: verdict OK, is worth a reader's eye — "86% used, ~2 calls left" printed in
#: calm green is how an evening ends on the local model with nobody warned.
NOMINAL = "everything nominal"

# ASCII only, deliberately. A cp1252 console raises mid-render on box-drawing
# characters, and the one thing a status display must never do is crash the
# surface that was about to tell you something was wrong.
BAR_FULL = "#"
BAR_EMPTY = "."


def bar(fraction: float, width: int = 12) -> str:
    """A fixed-width gauge. Clamped at both ends: our token tally is a lower
    bound on what the provider thinks, so it can and does exceed the published
    cap, and a bar longer than its track reads as a rendering bug."""
    f = min(1.0, max(0.0, fraction))
    filled = int(round(f * width))
    if 0.0 < f and filled == 0:
        filled = 1  # something used should never look like nothing used
    return BAR_FULL * filled + BAR_EMPTY * (width - filled)


@dataclass(frozen=True)
class Gauge:
    """One provider's daily allowance."""

    provider: str
    used: int
    cap: int
    unit: str  # "tokens" | "requests"
    fraction: float
    exhausted: bool
    warning: bool
    calls_left: int | None
    covered_by: str | None  # another cloud provider that would take over

    @property
    def line(self) -> str:
        head = f"{self.provider} {bar(self.fraction)} {int(self.fraction * 100)}%"
        tail = f"{self.used:,}/{self.cap:,} {self.unit}"
        if self.exhausted:
            tail += " — spent"
            tail += f", {self.covered_by} covering" if self.covered_by else ", LOCAL ONLY"
        elif self.calls_left:
            tail += f" — ~{self.calls_left} calls left"
        return f"{head}  {tail}"


@dataclass(frozen=True)
class Hud:
    verdict: str = OK
    headline: str = NOMINAL
    answering: str = ""
    answering_is_local: bool = False
    #: True when `answering` is who actually served the last turn. False when no
    #: turn has run yet and this is only who the chain says would take the next
    #: one — a guess, and the panel says so rather than stating it as fact.
    answering_observed: bool = False
    chain: list[str] = field(default_factory=list)
    gauges: list[Gauge] = field(default_factory=list)
    sidelined: list[str] = field(default_factory=list)
    budget: str | None = None
    missions: list[str] = field(default_factory=list)
    nudges: int = 0
    audit_ok: bool = True
    build: str = "?"
    session: str = ""

    @property
    def degraded(self) -> bool:
        return self.verdict != OK

    @property
    def nominal(self) -> bool:
        """Nothing worth looking at. Distinct from `not degraded`: a spent
        allowance that another provider is covering is not a problem, but it is
        still the sentence that explains tonight."""
        return self.verdict == OK and self.headline == NOMINAL


def _cloud(chain: list[str]) -> list[str]:
    return [p for p in chain if p and p not in LOCAL_PROVIDERS]


def chain_of(health: dict[str, Any]) -> list[str]:
    """Every provider that could serve a turn, primary first."""
    out = [str(health.get("provider") or "")]
    out += [f.strip() for f in str(health.get("fallback") or "").split(",") if f.strip()]
    return [p for p in out if p]


def _sidelined_lines(health: dict[str, Any], now: datetime) -> list[str]:
    """Providers currently skipped, and when they are worth trying again."""
    out = []
    for name, until_iso in sorted((health.get("sidelined") or {}).items()):
        try:
            until = datetime.fromisoformat(until_iso)
        except (TypeError, ValueError):
            out.append(f"{name} — standing down")
            continue
        if until.tzinfo and not now.tzinfo:
            out.append(f"{name} — standing down")
            continue
        minutes = max(0, int((until - now).total_seconds() // 60))
        if minutes >= 60:
            out.append(f"{name} — back in {minutes // 60}h{minutes % 60:02d}m")
        elif minutes:
            out.append(f"{name} — back in {minutes}m")
        else:
            out.append(f"{name} — back shortly")  # "back in 0m" reads as a bug
    return out


def _stood_down(health: dict[str, Any]) -> set[str]:
    return set((health.get("sidelined") or {}).keys())


def _gauges(health: dict[str, Any], chain: list[str]) -> list[Gauge]:
    out = []
    for p in (health.get("today") or {}).get("providers") or []:
        name = str(p.get("provider", ""))
        by_calls = p.get("metered_by") == "calls"
        cap = int((p.get("call_cap") if by_calls else p.get("cap")) or 0)
        if cap <= 0:
            continue  # no published free-tier ceiling: a gauge with no track is noise
        # "Covered" means another CLOUD provider is in the chain. Ollama is always
        # last in the chain and is exactly the outcome the gauge is warning about,
        # so it must never count as cover.
        cover = [c for c in _cloud(chain) if c != name]
        out.append(
            Gauge(
                provider=name,
                used=int((p.get("calls") if by_calls else p.get("tokens")) or 0),
                cap=cap,
                unit="requests" if by_calls else "tokens",
                fraction=float(p.get("fraction") or 0.0),
                exhausted=bool(p.get("exhausted")),
                warning=bool(p.get("warning")),
                calls_left=p.get("calls_left"),
                covered_by=cover[0] if cover else None,
            )
        )
    return out


def _budget_line(health: dict[str, Any]) -> str | None:
    spend = health.get("spend") or {}
    budget = spend.get("budget_usd")
    if not budget:
        return None  # 0 == tracking only; a budget of nothing has nothing to report
    return f"${float(spend.get('total_usd') or 0.0):.2f} of ${float(budget):.2f} this month"


def _verdict(
    *,
    audit_ok: bool,
    answering_is_local: bool,
    answering: str,
    gauges: list[Gauge],
    spend: dict[str, Any],
    sidelined: list[str],
) -> tuple[str, str]:
    """The one sentence at the top, and how alarmed to be about it.

    Ordered by what it costs the person sitting there, which is not the same as
    ordering by how broken the machine is.
    """
    if not audit_ok:
        return BROKEN, (
            "the audit chain is broken — what M.I.K.E.Y did can no longer be proven. "
            "Run `mikey doctor` before trusting anything here"
        )
    if answering_is_local:
        # Past tense on purpose, and only when a turn actually came back from the
        # local model. The predicted case ("groq is spent, so the next one will be
        # local") is the stranded-allowance branch below, which says so in the
        # future tense — a dashboard that reports a guess as an observation is one
        # you stop believing the first time it is wrong.
        return DEGRADED, (
            f"answers are coming from {answering}, the local model — it is much weaker "
            "on anything multi-step. `mikey providers` shows how to add another cloud key"
        )
    stranded = [g for g in gauges if g.exhausted and not g.covered_by]
    if stranded:
        g = stranded[0]
        return DEGRADED, (
            f"{g.provider} has spent today's {g.unit} and nothing else is configured — "
            "the next answer comes from the local model"
        )
    if spend.get("over_budget"):
        return DEGRADED, "over the monthly budget — cloud models are switched off until it resets"
    covered = [g for g in gauges if g.exhausted]
    if covered:
        g = covered[0]
        return OK, f"{g.provider} is spent for today; {g.covered_by} is covering"
    if spend.get("warning"):
        return OK, "close to the monthly budget"
    warned = [g for g in gauges if g.warning]
    if warned:
        g = warned[0]
        left = f", ~{g.calls_left} calls left" if g.calls_left else ""
        return OK, f"{g.provider} is {int(g.fraction * 100)}% through today's {g.unit}{left}"
    if sidelined:
        return OK, f"standing down: {sidelined[0]}"
    return OK, NOMINAL


def build_hud(
    health: dict[str, Any],
    *,
    now: datetime,
    session: str = "",
    served_by: str | None = None,
    nudges: int = 0,
    missions: list[dict[str, Any]] | None = None,
) -> Hud:
    """Assemble the dashboard from what the gateway published.

    `served_by` is who answered the LAST turn, which is the fact the banner at the
    top of a chat cannot keep up with — the provider can change mid-conversation
    and that change is the whole reason this display exists.
    """
    chain = chain_of(health)
    gauges = _gauges(health, chain)
    sidelined = _sidelined_lines(health, now)
    if served_by:
        answering = served_by
    else:
        # Before the first turn there is nothing to observe, so name the provider
        # the chain would actually reach: saying "groq" while groq is stood down
        # and cerebras is covering contradicts the headline directly above it.
        down = _stood_down(health) | {g.provider for g in gauges if g.exhausted}
        answering = next((p for p in chain if p not in down), chain[0] if chain else "?")
    audit_ok = bool(health.get("audit_chain_valid"))
    verdict, headline = _verdict(
        audit_ok=audit_ok,
        answering_is_local=bool(served_by) and answering in LOCAL_PROVIDERS,
        answering=answering,
        gauges=gauges,
        spend=health.get("spend") or {},
        sidelined=sidelined,
    )
    return Hud(
        verdict=verdict,
        headline=headline,
        answering=answering,
        answering_is_local=answering in LOCAL_PROVIDERS,
        answering_observed=bool(served_by),
        chain=chain,
        gauges=gauges,
        sidelined=sidelined,
        budget=_budget_line(health),
        missions=[
            f"{m.get('goal', '?')} — step {int(m.get('next_step', 0)) + 1}/"
            f"{int(m.get('steps', 0))} ({m.get('status', '?')})"
            for m in (missions or [])
        ],
        nudges=nudges,
        audit_ok=audit_ok,
        build=str(health.get("build") or "?"),
        session=session,
    )
