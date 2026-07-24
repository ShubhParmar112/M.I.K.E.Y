"""The training exporter (sovereignty S0): event log → per-brain datasets.

Proves the two safety invariants that make exported data trainable: privacy
tiers are respected (Tier-0 excluded by default) and forgetting is respected
(tombstoned events never appear).
"""

from __future__ import annotations

import json
from pathlib import Path

from core.events.schema import Event, EventType, Provenance, Tier, now
from core.events.store import EventStore
from core.storage.db import Database
from training.exporter import TrainingExporter


def _seed(events: EventStore) -> None:
    # Turn A (T1): user → one tool action → assistant reply.
    events.append(Event(type=EventType.USER_MESSAGE.value,
                        payload={"text": "list my files", "session_id": "s1", "turn_id": "A"}))
    events.append(Event(type=EventType.ACTION_EXECUTED.value,
                        provenance=Provenance(source="agent", trusted=True),
                        payload={"tool": "fs_list", "args": {"path": "."}, "ok": True, "turn_id": "A"}))
    events.append(Event(type=EventType.ASSISTANT_MESSAGE.value,
                        provenance=Provenance(source="agent", trusted=True),
                        payload={"text": "here are your files", "turn_id": "A"}))

    # Turn B (T0 private): must be excluded by default.
    events.append(Event(type=EventType.USER_MESSAGE.value, tier=Tier.T0,
                        payload={"text": "my bank pin is secret", "session_id": "s1", "turn_id": "B"}))
    events.append(Event(type=EventType.ASSISTANT_MESSAGE.value, tier=Tier.T0,
                        provenance=Provenance(source="agent", trusted=True),
                        payload={"text": "noted privately", "turn_id": "B"}))

    # Two durable memory notes: one T1, one T0.
    events.append(Event(type=EventType.MEMORY_NOTE.value,
                        payload={"text": "deadline is Nov 15", "supersedes": []}))
    events.append(Event(type=EventType.MEMORY_NOTE.value, tier=Tier.T0,
                        payload={"text": "health record detail", "supersedes": []}))


def test_export_respects_tiers(db: Database, tmp_path: Path) -> None:
    events = EventStore(db)
    _seed(events)
    exp = TrainingExporter(events)

    s = exp.export(tmp_path / "ds", include_t0=False)
    assert s.conversation == 1  # turn A only; T0 turn B excluded
    assert s.tool_use == 1      # turn A had an action
    assert s.memory == 1        # T1 note only
    assert s.skipped_t0_turns == 1 and s.skipped_t0_notes == 1

    # Files are real JSONL and contain what the summary claims.
    convo = [json.loads(x) for x in (tmp_path / "ds" / "conversation.jsonl").read_text().splitlines()]
    assert convo[0]["input"] == "list my files" and convo[0]["tier"] == "T1"
    tools = [json.loads(x) for x in (tmp_path / "ds" / "tool_use.jsonl").read_text().splitlines()]
    assert tools[0]["actions"][0]["tool"] == "fs_list"


def test_include_t0_opt_in(db: Database, tmp_path: Path) -> None:
    events = EventStore(db)
    _seed(events)
    s = TrainingExporter(events).export(tmp_path / "ds", include_t0=True)
    assert s.conversation == 2 and s.memory == 2  # private turn + note now included


def test_forgotten_events_never_exported(db: Database, tmp_path: Path) -> None:
    events = EventStore(db)
    _seed(events)
    # Tombstone the T1 memory note (as MemoryStore.forget does): it must vanish
    # from exports, because EventStore.recent already excludes tombstoned events.
    note = next(e for e in events.recent(types=[EventType.MEMORY_NOTE.value], limit=100)
                if e.tier is Tier.T1)
    with db.conn as conn:
        conn.execute("INSERT INTO tombstones (event_id, ts, reason) VALUES (?, ?, ?)",
                     (note.id, now().isoformat(), "user asked to forget"))

    s = TrainingExporter(events).export(tmp_path / "ds", include_t0=True)
    assert s.memory == 1  # the forgotten T1 note is gone; only the T0 note remains
    mem = [json.loads(x) for x in (tmp_path / "ds" / "memory.jsonl").read_text().splitlines()]
    assert all(m["text"] != "deadline is Nov 15" for m in mem)
