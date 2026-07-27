"""The brief: what M.I.K.E.Y opens with.

Composed entirely from things already in the log — no model call, so it costs no
quota and cannot hallucinate. That is deliberate: the one moment a summary must
be trustworthy is when nobody asked for it.

It is short on purpose. A brief that lists everything is a report, and a report
volunteered at 8am is not read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from core.events.schema import EventType, now
from core.events.store import EventStore
from core.proactive.nudge import Nudge


@dataclass
class Brief:
    since: datetime
    at: datetime
    conversations: int = 0
    remembered: int = 0
    ingested: int = 0
    actions: int = 0
    missions_open: int = 0
    nudges: list[Nudge] = field(default_factory=list)

    @property
    def quiet(self) -> bool:
        """Nothing happened and nothing is outstanding — worth saying so briefly
        rather than manufacturing three bullet points about it."""
        return not self.nudges and not any(
            (self.conversations, self.remembered, self.ingested, self.actions, self.missions_open)
        )

    def lines(self) -> list[str]:
        """The brief as sentences, most important first."""
        out = [n.text for n in self.nudges]
        activity: list[str] = []
        if self.conversations:
            activity.append(f"{self.conversations} exchange{'s' if self.conversations != 1 else ''}")
        if self.actions:
            activity.append(f"{self.actions} action{'s' if self.actions != 1 else ''}")
        if self.remembered:
            activity.append(f"{self.remembered} thing{'s' if self.remembered != 1 else ''} remembered")
        if self.ingested:
            activity.append(f"{self.ingested} document{'s' if self.ingested != 1 else ''} read")
        if activity:
            out.append("Since we last spoke: " + ", ".join(activity) + ".")
        if self.missions_open:
            out.append(
                f"{self.missions_open} mission{'s are' if self.missions_open != 1 else ' is'} "
                "still open."
            )
        if not out:
            out.append("Nothing outstanding.")
        return out

    def spoken(self) -> str:
        return " ".join(self.lines())


def compose(
    events: EventStore,
    nudges: list[Nudge],
    missions_open: int = 0,
    since: datetime | None = None,
    at: datetime | None = None,
) -> Brief:
    """Build the brief from the log. `since` defaults to the last 24 hours."""
    at = at or now()
    since = since or (at - timedelta(hours=24))
    counted = {
        EventType.USER_MESSAGE.value: 0,
        EventType.MEMORY_NOTE.value: 0,
        EventType.INGEST_DOCUMENT.value: 0,
        EventType.ACTION_EXECUTED.value: 0,
    }
    for ev in events.since(since.isoformat(), list(counted)):
        counted[ev.type] = counted.get(ev.type, 0) + 1

    return Brief(
        since=since,
        at=at,
        conversations=counted[EventType.USER_MESSAGE.value],
        remembered=counted[EventType.MEMORY_NOTE.value],
        ingested=counted[EventType.INGEST_DOCUMENT.value],
        actions=counted[EventType.ACTION_EXECUTED.value],
        missions_open=missions_open,
        nudges=list(nudges),
    )
