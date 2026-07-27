"""Gathering the snapshot the rules judge, and raising what they find.

Everything impure lives here — the stores, the clock, the config — so `rules.py`
stays a set of functions you can reason about. The Sentinel is what a background
tick calls: look, decide, record. It never speaks; delivering is a surface's job,
because only a surface knows whether anyone is there.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from core.config import Config, available_cloud_providers
from core.cost.governor import LOCAL_PROVIDERS, CostGovernor
from core.events.schema import EventType, now
from core.events.store import EventStore
from core.missions.store import MissionStore
from core.proactive.nudge import Nudge, NudgeStore
from core.proactive.rules import MissionGlance, World, evaluate

# How many recent model calls to look at when asking "are we degraded right now?".
# Small enough to reflect the last few minutes rather than the day.
RECENT_CALLS = 6


class Integrity(Protocol):
    """The audit-chain check, as the sentinel needs it (core.policy.PolicyEngine)."""

    def verify_audit_chain(self) -> bool: ...


class Sentinel:
    """Looks at the world on a timer and records what is worth saying."""

    def __init__(
        self,
        config: Config,
        events: EventStore,
        nudges: NudgeStore,
        governor: CostGovernor,
        missions: MissionStore,
        policy: Integrity | None = None,
    ) -> None:
        self._config = config
        self._events = events
        self._nudges = nudges
        self._governor = governor
        self._missions = missions
        self._policy = policy

    def observe(self, at: datetime | None = None) -> World:
        at = at or now()
        day = self._governor.today(at)
        spend = self._governor.month_to_date(at)

        pressure = [
            (p.provider, p.fraction, p.calls_left, p.exhausted)
            for p in day.providers
            if p.capped
        ]

        missions = [
            MissionGlance(
                id=state.id,
                goal=state.goal,
                last_progress=self._last_progress(state.id),
                status=state.status,
            )
            for state in self._missions.active()
        ]

        local, total = self._recent_serving()
        return World(
            at=at,
            cloud_providers=available_cloud_providers(),
            provider_pressure=pressure,
            spend_fraction=spend.fraction,
            spend_enforced=spend.enforced,
            budget_usd=spend.budget_usd,
            missions=missions,
            audit_ok=self._audit_ok(),
            recent_local_turns=local,
            recent_turns=total,
            day_key=day.day,
        )

    def _audit_ok(self) -> bool:
        if self._policy is None:
            return True
        try:
            return self._policy.verify_audit_chain()
        except Exception:  # noqa: BLE001 — a check that itself fails is not a broken chain
            return True

    def _last_progress(self, mission_id: str) -> datetime | None:
        latest: datetime | None = None
        for ev in self._events.recent(
            types=[EventType.MISSION_STEP_RESULT.value], limit=2_000
        ):
            if ev.payload.get("mission_id") == mission_id:
                latest = ev.ts if latest is None else max(latest, ev.ts)
        return latest

    def _recent_serving(self) -> tuple[int, int]:
        """(local calls, total calls) among the last few model calls.

        Returns nothing when brains are deliberately pinned local — being served
        on-device is then the configuration working, not a symptom.
        """
        if self._config.local_brains:
            return 0, 0
        calls = self._events.recent(types=[EventType.MODEL_USAGE.value], limit=RECENT_CALLS)
        if not calls:
            return 0, 0
        local = sum(1 for ev in calls if str(ev.payload.get("provider")) in LOCAL_PROVIDERS)
        return local, len(calls)

    def tick(self, at: datetime | None = None) -> list[Nudge]:
        """One pass: expire what went stale, look, record anything new."""
        at = at or now()
        self._nudges.expire_stale(at)
        raised: list[Nudge] = []
        for nudge in evaluate(self.observe(at)):
            if self._nudges.raise_nudge(nudge) is not None:
                raised.append(nudge)
        return raised
