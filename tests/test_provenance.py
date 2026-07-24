"""Calibrated provenance for recalled memories: humanized age, source labels, and
a staleness flag for personal facts (but not for reference documents).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from core.memory.provenance import annotate, humanize_age, is_stale, source_label
from core.memory.store import MemoryHit

NOW = datetime(2026, 7, 25, 12, 0, 0, tzinfo=UTC)


def _ago(**kw: float) -> str:
    return (NOW - timedelta(**kw)).isoformat()


def test_humanize_age_scales() -> None:
    assert humanize_age(_ago(seconds=5), NOW) == "just now"
    assert humanize_age(_ago(minutes=20), NOW) == "20 min ago"
    assert humanize_age(_ago(hours=5), NOW) == "5h ago"
    assert humanize_age(_ago(days=1), NOW) == "1 day ago"
    assert humanize_age(_ago(days=3), NOW) == "3 days ago"
    assert humanize_age(_ago(days=150), NOW) == "5 months ago"
    assert humanize_age(_ago(days=800), NOW) == "2 years ago"


def test_humanize_age_handles_garbage() -> None:
    assert humanize_age("not-a-date", NOW) == "unknown age"


def test_personal_facts_go_stale_reference_docs_do_not() -> None:
    old = _ago(days=200)
    fresh = _ago(days=3)
    assert is_stale(old, "user", NOW) is True         # a 200-day-old personal fact
    assert is_stale(fresh, "user", NOW) is False      # a recent one
    assert is_stale(old, "agent", NOW) is True        # M.I.K.E.Y's own old note
    # reference material doesn't "go stale" the way personal facts do
    assert is_stale(old, "connector:file:paper.pdf", NOW) is False
    assert is_stale(old, "web:https://x", NOW) is False


def test_source_labels() -> None:
    assert source_label("user", True) == "from you"
    assert source_label("agent", True) == "your earlier note"
    assert source_label("connector:file:mmp.pdf", True) == "from mmp.pdf"
    assert source_label("web:https://x", True) == "from the web"
    assert source_label("connector:file:notes.md", False) == "from notes.md (unverified)"


def _hit(source: str, trusted: bool, ts: str, kind: str = "memory") -> MemoryHit:
    return MemoryHit(event_id="01ABC", source=source, trusted=trusted, ts=ts, text="x",
                     rank=0.0, kind=kind)


def test_annotation_shows_tier_and_episodes_dont_go_stale() -> None:
    fact = annotate(_hit("user", True, _ago(days=200), kind="fact"), NOW)
    assert "01ABC · fact ·" in fact and "possibly outdated" in fact  # old personal fact

    ep = annotate(_hit("agent", True, _ago(days=200), kind="episode"), NOW)
    assert "episode" in ep and "possibly outdated" not in ep  # an episode records history
    assert "your earlier note" not in ep  # source is implied for a summary


def test_annotate_composes_the_header() -> None:
    fresh_user = annotate(_hit("user", True, _ago(days=2)), NOW)
    assert fresh_user == "01ABC · 2 days ago · from you"

    stale_user = annotate(_hit("user", True, _ago(days=200)), NOW)
    assert "possibly outdated" in stale_user and "from you" in stale_user

    untrusted_doc = annotate(_hit("connector:file:seminar.md", False, _ago(days=2)), NOW)
    assert "seminar.md" in untrusted_doc and "UNTRUSTED" in untrusted_doc
    assert "possibly outdated" not in untrusted_doc  # a doc isn't flagged stale
