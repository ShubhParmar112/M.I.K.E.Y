"""Conversation history must not leak between conversations.

From a live session (2026-07-26): the aptitude test ended at 18:46 with a
half-argued 200-metre race problem. At 19:11 a NEW chat asked an unrelated
shopkeeper/TV-set profit question — and M.I.K.E.Y answered with "We also know
that B beats C by 10 meters in a 200-meter race", then concluded it could not
relate the cost price of a television to the distance covered by a runner.

Two defects produced that. `ContextAssembler.assemble` never took a session id at
all, so every turn saw the last 40 messages from every session; and the CLI reused
the id `default` on every run, so there was no boundary to filter on even if it had.
Both are pinned here.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from core.context.assembly import SITTING_GAP, ContextAssembler
from core.events.schema import Event, EventType, now
from core.events.store import EventStore
from core.memory.store import MemoryStore
from core.storage.db import Database

RACE = "In a race of 200 meters, A can beat B by 20 meters, and B can beat C by 10 meters."
TV = "A shopkeeper sold a T.V set for Rs. 17,940 with a discount of 8% and earned a profit of 19.6%."


@pytest.fixture
def store(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    events = EventStore(db)
    memory = MemoryStore(db, events)
    return events, memory, ContextAssembler(events, memory, 10_000)


def _say(events: EventStore, session: str, text: str, ago: timedelta = timedelta()) -> None:
    events.append(
        Event(
            type=EventType.USER_MESSAGE.value,
            ts=now() - ago,
            payload={"text": text, "session_id": session, "turn_id": "t"},
        )
    )


def test_a_new_conversation_does_not_inherit_the_previous_one(store) -> None:
    """The exact live failure: same box, minutes apart, different conversation."""
    events, _memory, assembler = store
    _say(events, "chat-a", RACE, ago=timedelta(minutes=25))

    ctx = assembler.assemble(TV, session_id="chat-b")
    history = " ".join(m.text for m in ctx.messages[:-1])

    assert history == "", "a new session must start with no conversation history"
    assert "200 meters" not in history
    assert ctx.messages[-1].text == TV  # only the question itself


def test_the_same_conversation_still_remembers_itself(store) -> None:
    """The fix must not amputate ordinary follow-ups — "are you sure?" has to keep
    working, which is the whole reason history exists."""
    events, _memory, assembler = store
    _say(events, "chat-a", RACE, ago=timedelta(minutes=2))

    ctx = assembler.assemble("are you sure?", session_id="chat-a")
    history = " ".join(m.text for m in ctx.messages[:-1])

    assert RACE in history
    assert len(ctx.included_events) == 1


def test_history_from_a_previous_sitting_is_dropped(store) -> None:
    """Secondary guard for a session that is resumed much later — you came back the
    next morning, not mid-thought."""
    events, _memory, assembler = store
    _say(events, "chat-a", RACE, ago=SITTING_GAP + timedelta(minutes=5))

    ctx = assembler.assemble(TV, session_id="chat-a")
    assert ctx.included_events == []


def test_the_gap_is_measured_between_messages_not_only_from_now(store) -> None:
    """A run of recent messages preceded by a long silence keeps the recent run and
    drops everything on the far side of the silence."""
    events, _memory, assembler = store
    _say(events, "chat-a", RACE, ago=SITTING_GAP * 2)
    _say(events, "chat-a", "what is 2 + 2?", ago=timedelta(minutes=3))
    _say(events, "chat-a", "and 3 + 3?", ago=timedelta(minutes=1))

    ctx = assembler.assemble("and 4 + 4?", session_id="chat-a")
    history = " ".join(m.text for m in ctx.messages[:-1])

    assert "2 + 2" in history and "3 + 3" in history
    assert "200 meters" not in history


def test_busy_neighbouring_sessions_cannot_starve_this_ones_history(store) -> None:
    """Filtering happens over a wide fetch window, so another session's chatter can
    push this session's own turns out of view."""
    events, _memory, assembler = store
    _say(events, "chat-a", "my own first message", ago=timedelta(minutes=30))
    for i in range(80):
        _say(events, "chat-noisy", f"unrelated chatter {i}", ago=timedelta(minutes=20))

    ctx = assembler.assemble("what did I say first?", session_id="chat-a")
    history = " ".join(m.text for m in ctx.messages[:-1])

    assert "my own first message" in history
    assert "unrelated chatter" not in history


def test_memory_still_crosses_sessions(store) -> None:
    """The split that makes this correct rather than merely quiet: raw history is
    per-conversation, long-term memory is not. A new chat starts clean and can still
    recall what it was told before."""
    _events, memory, assembler = store
    memory.remember("Shubh's dog is named Pixel", source="user", trusted=True)

    ctx = assembler.assemble("what is my dog called?", session_id="a-totally-new-chat")

    assert ctx.included_events == []  # no history
    assert any("Pixel" in h.text for h in ctx.memory_hits)  # but memory works
