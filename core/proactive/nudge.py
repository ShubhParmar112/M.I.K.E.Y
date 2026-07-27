"""A nudge: something M.I.K.E.Y wants to say without being asked.

Durable for the same reason missions are — the thing worth telling you is usually
noticed while you are away, and a note that dies with the process is a note that
never gets delivered. Raised as an event, projected to find what is outstanding,
closed by another event.

Two properties do most of the work:

- **Dedup.** A rule is evaluated on a timer, so "your quota is nearly gone" is
  true on every tick for hours. Without a key that says "this is the same thing I
  already raised", the queue fills with one sentence repeated ninety times, and
  the person stops reading any of it.
- **Expiry.** "You have about four calls left today" is useful for an hour and
  misleading tomorrow. An undelivered nudge that has gone stale is dropped rather
  than shown late, and the drop is recorded — a proactive system that shows you
  yesterday's urgency teaches you to ignore it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from core.events.schema import Event, EventType, Provenance, now, ulid
from core.events.store import EventStore


class Urgency:
    """How much interruption this is worth. Ordered, and compared as such."""

    LOW = "low"  # worth mentioning next time we talk
    NORMAL = "normal"  # worth opening with
    HIGH = "high"  # worth interrupting for, including out of hours

    ORDER = {LOW: 0, NORMAL: 1, HIGH: 2}

    @staticmethod
    def rank(urgency: str) -> int:
        return Urgency.ORDER.get(urgency, 1)


# How long an undelivered nudge stays worth delivering. Urgent things age fastest:
# an emergency that waited a day was not one.
SHELF_LIFE = {
    Urgency.LOW: timedelta(days=3),
    Urgency.NORMAL: timedelta(hours=18),
    Urgency.HIGH: timedelta(hours=6),
}

# How long a dedup key stays spent after it has been raised — whether or not the
# nudge was delivered, dismissed or expired.
#
# Checking only what is still OUTSTANDING is not enough, and the live run showed
# why: dismiss a note and the very next tick raises it again, because nothing
# outstanding matches any more. Dismissing has to mean something. A day is the
# right length because the keys that matter are day-scoped anyway ("today's
# allowance"), so a condition that is still true tomorrow gets to speak up again.
DEDUP_WINDOW = timedelta(hours=24)


@dataclass(frozen=True)
class Nudge:
    """Something to say, and everything needed to decide whether to say it."""

    id: str
    kind: str  # which rule raised it, e.g. "quota" — also what a person mutes
    text: str  # one sentence, as it will be spoken
    detail: str = ""  # optional supporting line, shown but never spoken
    urgency: str = Urgency.NORMAL
    dedup_key: str = ""
    created: datetime = field(default_factory=now)

    def stale_at(self) -> datetime:
        return self.created + SHELF_LIFE.get(self.urgency, SHELF_LIFE[Urgency.NORMAL])

    def is_stale(self, at: datetime | None = None) -> bool:
        return (at or now()) >= self.stale_at()


class NudgeStore:
    """The outstanding-nudge projection, over the append-only log."""

    _TYPES = [EventType.NUDGE_RAISED.value, EventType.NUDGE_DELIVERED.value]
    # Far enough back to cover any shelf life; nudges are low-volume.
    _SCAN = 5_000

    def __init__(self, events: EventStore, device_id: str = "dev_desktop_1") -> None:
        self._events = events
        self._device = device_id
        # Dedup is a check followed by an append, and the two callers of it — the
        # background watcher and a request handler asking for a brief — really do
        # run at the same time. Unguarded, both read an empty queue and both write,
        # which is how the first live run produced the same sentence twice.
        self._lock = threading.Lock()

    def raise_nudge(self, nudge: Nudge) -> Nudge | None:
        """Record something worth saying. Returns None if an equivalent one is
        already outstanding — the same observation twice is not two observations."""
        with self._lock:
            if nudge.dedup_key and nudge.dedup_key in self.spent_keys():
                return None
            self._events.append(
                Event(
                    type=EventType.NUDGE_RAISED.value,
                    device=self._device,
                    provenance=Provenance(source="agent", trusted=True),
                    payload={
                        "nudge_id": nudge.id,
                        "kind": nudge.kind,
                        "text": nudge.text,
                        "detail": nudge.detail,
                        "urgency": nudge.urgency,
                        "dedup_key": nudge.dedup_key,
                    },
                )
            )
        return nudge

    def deliver(self, nudge_id: str, how: str = "chat", outcome: str = "shown") -> None:
        """Close a nudge. `outcome` is "shown" | "dismissed" | "expired" — kept
        because "we said it and they ignored it" and "it went stale unsaid" are
        different failures, and only the log can tell them apart later."""
        self._events.append(
            Event(
                type=EventType.NUDGE_DELIVERED.value,
                device=self._device,
                provenance=Provenance(source="agent", trusted=True),
                payload={"nudge_id": nudge_id, "how": how, "outcome": outcome},
            )
        )

    def pending(self, include_stale: bool = False, at: datetime | None = None) -> list[Nudge]:
        """Outstanding nudges, oldest first. Stale ones are excluded by default."""
        at = at or now()
        raised: dict[str, Nudge] = {}
        closed: set[str] = set()
        for ev in self._events.recent(types=self._TYPES, limit=self._SCAN):
            nudge_id = str(ev.payload.get("nudge_id", ""))
            if not nudge_id:
                continue
            if ev.type == EventType.NUDGE_DELIVERED.value:
                closed.add(nudge_id)
                continue
            raised[nudge_id] = Nudge(
                id=nudge_id,
                kind=str(ev.payload.get("kind", "")),
                text=str(ev.payload.get("text", "")),
                detail=str(ev.payload.get("detail", "")),
                urgency=str(ev.payload.get("urgency", Urgency.NORMAL)),
                dedup_key=str(ev.payload.get("dedup_key", "")),
                created=ev.ts,
            )
        live = [n for n in raised.values() if n.id not in closed]
        if not include_stale:
            live = [n for n in live if not n.is_stale(at)]
        return sorted(live, key=lambda n: n.created)

    def spent_keys(self, at: datetime | None = None) -> set[str]:
        """Dedup keys already used inside the window — including ones whose nudge
        has since been delivered, dismissed or expired. Still-outstanding nudges
        count regardless of age, so nothing can ever be shown twice at once."""
        at = at or now()
        cutoff = at - DEDUP_WINDOW
        spent = {n.dedup_key for n in self.pending(include_stale=True, at=at) if n.dedup_key}
        for ev in self._events.recent(types=[EventType.NUDGE_RAISED.value], limit=self._SCAN):
            key = str(ev.payload.get("dedup_key", ""))
            if key and ev.ts >= cutoff:
                spent.add(key)
        return spent

    def dismissals_by_kind(self) -> dict[str, int]:
        """How often each kind has been actively dismissed.

        Being told twice is a reminder; being told nine times is a reason to stop
        listening to all of it. This is what lets the discipline layer give up on a
        kind of remark rather than wearing out its welcome — the person never has to
        find a setting to turn it off, they just have to keep saying no.
        """
        kinds: dict[str, str] = {}
        counts: dict[str, int] = {}
        for ev in self._events.recent(types=self._TYPES, limit=self._SCAN):
            nudge_id = str(ev.payload.get("nudge_id", ""))
            if ev.type == EventType.NUDGE_RAISED.value:
                kinds[nudge_id] = str(ev.payload.get("kind", ""))
            elif ev.payload.get("outcome") == "dismissed":
                kind = kinds.get(nudge_id, "")
                if kind:
                    counts[kind] = counts.get(kind, 0) + 1
        return counts

    def expire_stale(self, at: datetime | None = None) -> int:
        """Close nudges that waited too long to be worth saying. Returns how many."""
        at = at or now()
        stale = [n for n in self.pending(include_stale=True, at=at) if n.is_stale(at)]
        for nudge in stale:
            self.deliver(nudge.id, how="none", outcome="expired")
        return len(stale)


def new_nudge(
    kind: str,
    text: str,
    detail: str = "",
    urgency: str = Urgency.NORMAL,
    dedup_key: str = "",
) -> Nudge:
    return Nudge(
        id=ulid(),
        kind=kind,
        text=text,
        detail=detail,
        urgency=urgency,
        dedup_key=dedup_key or f"{kind}:{text}",
    )
