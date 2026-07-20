from __future__ import annotations

from core.events.schema import Event, EventType, Provenance, Tier, ulid
from core.events.store import EventStore
from core.storage.db import Database


def test_ulid_is_sortable_and_unique() -> None:
    ids = [ulid() for _ in range(100)]
    assert len(set(ids)) == 100
    assert all(len(i) == 26 for i in ids)


def test_append_and_recent_roundtrip(db: Database) -> None:
    store = EventStore(db)
    for i in range(5):
        store.append(
            Event(
                type=EventType.USER_MESSAGE.value,
                tier=Tier.T0,
                provenance=Provenance(source="user", trusted=True),
                payload={"text": f"msg {i}"},
            )
        )
    got = store.recent(types=[EventType.USER_MESSAGE.value], limit=3)
    assert [e.payload["text"] for e in got] == ["msg 2", "msg 3", "msg 4"]  # oldest-first
    assert got[0].tier == Tier.T0
    assert got[0].provenance.trusted is True


def test_migration_upgrades_older_schema_in_place(tmp_path) -> None:
    """A v1 database (Gen 1 install) must upgrade to v2 on next open."""
    path = tmp_path / "old.db"
    db = Database(path)
    with db.conn as conn:  # rewind to v1
        conn.execute("DROP TABLE memory_fts")
        conn.execute("DROP TABLE tombstones")
        conn.execute("DELETE FROM schema_version WHERE version = 2")
    db.close()
    upgraded = Database(path)
    row = upgraded.conn.execute("SELECT COUNT(*) AS n FROM tombstones").fetchone()
    assert row["n"] == 0  # v2 tables exist again
    versions = [
        r["version"]
        for r in upgraded.conn.execute("SELECT version FROM schema_version ORDER BY version")
    ]
    assert versions == [1, 2]


def test_recent_without_filter_returns_all_types(db: Database) -> None:
    store = EventStore(db)
    store.append(Event(type=EventType.USER_MESSAGE.value, payload={"text": "a"}))
    store.append(Event(type=EventType.ACTION_EXECUTED.value, payload={"tool": "fs_read"}))
    assert len(store.recent()) == 2
