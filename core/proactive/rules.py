"""What M.I.K.E.Y notices.

Each rule is a pure function from a snapshot of the world to at most one nudge,
so "would it have said something?" is answerable in a test rather than by waiting
for a Tuesday evening. Gathering the snapshot is `watch.py`'s job.

The bar for adding a rule: it must be something the person would want to be told
*before* they notice it themselves. Everything M.I.K.E.Y already shows on request
— spend, traces, memory — stays on request. What is here is the small set of
things whose first symptom is otherwise "why has it got worse?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from core.proactive.nudge import Nudge, Urgency, new_nudge

# A mission that has not moved in this long has stopped, whatever its status says.
STALLED_AFTER = timedelta(hours=6)
# Enough recent turns on the local model to mean "this is the situation now"
# rather than "one request happened to spill over".
LOCAL_STREAK = 3


@dataclass
class MissionGlance:
    id: str
    goal: str
    last_progress: datetime | None
    status: str


@dataclass
class World:
    """Everything the rules are allowed to look at, gathered once."""

    at: datetime
    # Providers with a key configured, cloud only.
    cloud_providers: list[str] = field(default_factory=list)
    # (provider, fraction_of_daily_allowance_used, calls_left, exhausted)
    provider_pressure: list[tuple[str, float, int | None, bool]] = field(default_factory=list)
    spend_fraction: float = 0.0
    spend_enforced: bool = False
    budget_usd: float = 0.0
    missions: list[MissionGlance] = field(default_factory=list)
    audit_ok: bool = True
    # How many of the last few turns were served by a local model, and how many
    # turns that was out of.
    recent_local_turns: int = 0
    recent_turns: int = 0
    day_key: str = ""


def quota_pressure(world: World) -> Nudge | None:
    """The failure that started all of this: the day's allowance runs out and every
    answer afterwards silently comes from a much weaker model."""
    if not world.provider_pressure:
        return None
    provider, fraction, calls_left, exhausted = max(world.provider_pressure, key=lambda p: p[1])
    if fraction < 0.75:
        return None
    backups = [p for p in world.cloud_providers if p != provider]
    key = f"quota:{world.day_key}:{provider}:{'gone' if exhausted else 'low'}"

    if exhausted:
        if backups:
            return new_nudge(
                "quota",
                f"{provider} is out of allowance for today — {backups[0]} is covering, "
                "so answers should hold up.",
                urgency=Urgency.LOW,
                dedup_key=key,
            )
        return new_nudge(
            "quota",
            f"{provider} has used its allowance for today, so I'm answering from the "
            "local model now — take anything multi-step with a pinch of salt.",
            detail="`mikey providers` shows how to add a second free provider.",
            urgency=Urgency.HIGH,
            dedup_key=key,
        )

    left = f"about {calls_left} more exchanges" if calls_left else "not much left"
    tail = "" if backups else " After that I drop to the local model, which is much weaker."
    return new_nudge(
        "quota",
        f"Heads up: {provider} is at {fraction * 100:.0f}% of today's allowance — {left}.{tail}",
        urgency=Urgency.NORMAL if not backups else Urgency.LOW,
        dedup_key=key,
    )


def serving_locally(world: World) -> Nudge | None:
    """Already degraded, right now — worth saying even before the quota gauge trips,
    because the gauge is a lower bound and this is an observation."""
    if world.recent_turns < LOCAL_STREAK or world.recent_local_turns < LOCAL_STREAK:
        return None
    if not world.cloud_providers:
        return None  # local-only by choice; not news
    return new_nudge(
        "degraded",
        f"The last {world.recent_local_turns} answers came from the local model rather "
        "than the cloud one — that's why they may feel off.",
        detail="Usually a spent daily allowance or a provider outage. `mikey providers`.",
        urgency=Urgency.NORMAL,
        dedup_key=f"degraded:{world.day_key}",
    )


def stalled_mission(world: World) -> Nudge | None:
    """A mission that stopped and never said so. Missions survive reboots, which is
    exactly why one can sit half-finished for a day without anyone noticing."""
    stalled = [
        m
        for m in world.missions
        if m.status in ("pending", "running", "failed")
        and (m.last_progress is None or world.at - m.last_progress > STALLED_AFTER)
    ]
    if not stalled:
        return None
    mission = stalled[0]
    goal = mission.goal if len(mission.goal) <= 60 else mission.goal[:57] + "..."
    extra = f" and {len(stalled) - 1} more" if len(stalled) > 1 else ""
    return new_nudge(
        "mission",
        f"A mission is still unfinished{extra}: “{goal}”. Want me to pick it up?",
        detail=f"`mikey mission-run {mission.id}` resumes it where it stopped.",
        urgency=Urgency.NORMAL,
        dedup_key=f"mission:{mission.id}",
    )


def budget_pressure(world: World) -> Nudge | None:
    if not world.spend_enforced or world.spend_fraction < 0.8:
        return None
    if world.spend_fraction >= 1.0:
        return new_nudge(
            "budget",
            f"This month's ${world.budget_usd:.0f} model budget is spent, so I'm on the "
            "local model until it rolls over.",
            urgency=Urgency.HIGH,
            dedup_key=f"budget:spent:{world.day_key[:7]}",
        )
    return new_nudge(
        "budget",
        f"You're at {world.spend_fraction * 100:.0f}% of this month's model budget.",
        urgency=Urgency.LOW,
        dedup_key=f"budget:warn:{world.day_key[:7]}",
    )


def integrity_broken(world: World) -> Nudge | None:
    """The one thing that is always worth interrupting for: the audit chain is how
    M.I.K.E.Y proves what it did, and a broken one means it can no longer prove it."""
    if world.audit_ok:
        return None
    return new_nudge(
        "integrity",
        "The audit chain no longer verifies — I can't prove what I did or didn't do. "
        "Worth looking at before you rely on anything else I say.",
        detail="`mikey doctor` checks it; a verified backup can be restored with `mikey restore`.",
        urgency=Urgency.HIGH,
        dedup_key="integrity:broken",
    )


RULES = (
    integrity_broken,
    quota_pressure,
    serving_locally,
    stalled_mission,
    budget_pressure,
)


def evaluate(world: World) -> list[Nudge]:
    """Every rule, in order. Raising is the caller's job — and dedup means calling
    this on a timer is safe."""
    found = [rule(world) for rule in RULES]
    return [n for n in found if n is not None]
