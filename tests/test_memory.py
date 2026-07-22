from __future__ import annotations

from core.events.schema import Event, EventType, Provenance
from core.events.store import EventStore
from core.memory.store import MemoryStore
from core.storage.db import Database


def _memory(db: Database) -> MemoryStore:
    return MemoryStore(db, EventStore(db))


def _doc(text: str, source: str = "connector:file:notes.md", trusted: bool = False) -> Event:
    return Event(
        type=EventType.INGEST_DOCUMENT.value,
        provenance=Provenance(source=source, trusted=trusted),
        payload={"text": text},
    )


def test_record_projects_and_recall_finds(db: Database) -> None:
    memory = _memory(db)
    memory.record(_doc("The MIKEY roadmap has ten generations, ending in bounded autonomy."))
    memory.record(_doc("Groceries: milk, eggs, bread."))
    hits = memory.recall("roadmap generations")
    assert len(hits) >= 1
    assert "ten generations" in hits[0].text
    assert hits[0].source == "connector:file:notes.md"
    assert hits[0].trusted is False


def test_recall_excludes_ids_and_empty_query(db: Database) -> None:
    memory = _memory(db)
    ev = memory.record(_doc("Unique fact about zephyr turbines."))
    assert memory.recall("zephyr turbines", exclude_ids={ev.id}) == []
    assert memory.recall("!!! ...") == []


def test_forget_is_verified_and_total(db: Database) -> None:
    memory = _memory(db)
    ev = memory.record(_doc("Secret: the launch code is in the blue notebook."))
    assert memory.recall("launch code")  # present before
    report = memory.forget(ev.id)
    assert report["verified"] is True
    assert memory.recall("launch code") == []  # gone from retrieval
    # gone from history projection too
    assert ev.id not in [e.id for e in memory.events.recent()]
    # and reindexing does NOT resurrect it
    memory.reindex()
    assert memory.recall("launch code") == []


def test_remember_stores_and_is_recallable(db: Database) -> None:
    memory = _memory(db)
    r = memory.remember("Shubh's dog is named Pixel.")
    assert r.status == "stored"
    hits = memory.recall("dog Pixel")
    assert any(h.event_id == r.event_id for h in hits)


def test_remember_skips_near_duplicate(db: Database) -> None:
    memory = _memory(db)
    first = memory.remember("Shubh's dog is named Pixel.")
    dup = memory.remember("Shubh's dog is named Pixel.")  # same fact again
    assert dup.status == "duplicate"
    assert dup.duplicate_of == first.event_id
    # only one copy actually persisted
    assert len(memory.recent_notes()) == 1


def test_remember_supersedes_tombstones_the_old_fact(db: Database) -> None:
    memory = _memory(db)
    old = memory.remember("Shubh has 2 years of industry experience.")
    new = memory.remember(
        "Shubh has 2.5 years of industry experience.", supersedes=[old.event_id]
    )
    assert new.status == "superseded"
    assert new.superseded == [old.event_id]
    # the stale fact is verifiably gone; the corrected one remains
    ids = [n.id for n in memory.recent_notes()]
    assert old.event_id not in ids and new.event_id in ids
    assert all("2 years" not in h.text for h in memory.recall("industry experience"))


def test_remember_flags_related_but_distinct_fact(db: Database) -> None:
    memory = _memory(db)
    first = memory.remember("Shubh has 2 years of industry experience in data science.")
    # overlapping subject, different content — not a duplicate, worth flagging
    second = memory.remember("Shubh has 2.5 years of industry experience in machine learning.")
    assert second.status == "stored"
    assert first.event_id in second.related


def test_reindex_rebuilds_projection_from_log(db: Database) -> None:
    memory = _memory(db)
    memory.record(_doc("Fact one about quasar alignment."))
    memory.record(_doc("Fact two about quasar drift."))
    with db.conn as conn:  # simulate a lost/corrupted projection
        conn.execute("DELETE FROM memory_fts")
    assert memory.recall("quasar") == []
    reprojected = memory.reindex()
    assert reprojected == 2
    assert len(memory.recall("quasar")) == 2
