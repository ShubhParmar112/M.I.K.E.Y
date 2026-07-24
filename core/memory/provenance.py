"""Calibrated provenance for recalled memories (the "deepen his mind" arc;
architecture review §6.1 — the trust feature of a lifelong system).

A recalled memory is annotated with **where it came from** and **how old it is**,
and a personal fact past a staleness horizon is flagged "possibly outdated". This
lets M.I.K.E.Y say "you told me in March (5 months ago) — possibly out of date —
that X" instead of asserting a months-old fact as current truth.

Reference material (ingested docs, web pages) is NOT flagged stale: a paper's
contents don't drift the way a personal preference or a deadline does.
"""

from __future__ import annotations

from datetime import UTC, datetime

from core.events.schema import now as _now
from core.memory.store import MemoryHit

# A personal fact older than this is flagged "possibly outdated" — a gentle nudge
# to reconfirm, never a hard claim that it's wrong.
STALE_AFTER_DAYS = 120


def _parse(ts_iso: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts_iso)
    except ValueError:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def humanize_age(ts_iso: str, now: datetime) -> str:
    then = _parse(ts_iso)
    if then is None:
        return "unknown age"
    secs = (now - then).total_seconds()
    if secs < 90:
        return "just now"
    mins = secs / 60
    if mins < 60:
        return f"{int(mins)} min ago"
    hours = mins / 60
    if hours < 24:
        return f"{int(hours)}h ago"
    days = int(hours / 24)
    if days < 30:
        return f"{days} day{'s' if days != 1 else ''} ago"
    if days < 365:
        months = round(days / 30)
        return f"{months} month{'s' if months != 1 else ''} ago"
    years = round(days / 365)
    return f"{years} year{'s' if years != 1 else ''} ago"


def is_stale(ts_iso: str, source: str, now: datetime, threshold_days: int = STALE_AFTER_DAYS) -> bool:
    # Reference material doesn't go stale the way personal facts do.
    if source.startswith(("connector:", "web:")):
        return False
    then = _parse(ts_iso)
    if then is None:
        return False
    return (now - then).days >= threshold_days


def source_label(source: str, trusted: bool) -> str:
    if source == "user":
        label = "from you"
    elif source == "agent":
        label = "your earlier note"
    elif source.startswith("connector:file:"):
        label = f"from {source.split(':', 2)[-1]}"
    elif source.startswith("connector:"):
        label = f"from {source.split(':', 1)[1]}"
    elif source.startswith("web:"):
        label = "from the web"
    else:
        label = f"from {source}"
    return label if trusted else f"{label} (unverified)"


def annotate(hit: MemoryHit, now: datetime | None = None) -> str:
    """The provenance header shown before a recalled memory, e.g.
    `01ABC · fact · 5 months ago · from you · possibly outdated`."""
    now = now or _now()
    parts = [hit.event_id]
    if hit.kind in ("fact", "episode", "document"):  # a distinctive tier; "message" is the default
        parts.append(hit.kind)
    parts.append(humanize_age(hit.ts, now))
    # An episode is M.I.K.E.Y's own summary — the "from agent" source is implied.
    if hit.kind != "episode":
        parts.append(source_label(hit.source, hit.trusted))
    # Only personal facts/messages "go stale"; an episode records history, a
    # document's content doesn't drift.
    if hit.kind not in ("episode", "document") and is_stale(hit.ts, hit.source, now):
        parts.append("possibly outdated")
    if not hit.trusted:
        parts.append("UNTRUSTED")
    return " · ".join(parts)
