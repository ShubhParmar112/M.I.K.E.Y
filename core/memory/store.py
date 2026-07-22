"""Memory Module — Gen 2 first slice.

The event log remains the source of truth; this module maintains a rebuildable
FTS5 projection over it (`reindex` proves it), retrieval with provenance, and
verified forgetting via tombstones.

Retrieval is keyword (BM25) + recency for now. The vector index slots in here
behind the same `recall()` seam once a local embedding model is available
(ADR-001: embeddings stay local; Groq doesn't serve them).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.events.schema import Event, EventType, now
from core.events.store import EventStore
from core.storage.db import Database

# Event types that become searchable memories.
PROJECTED_TYPES = {
    EventType.USER_MESSAGE.value,
    EventType.ASSISTANT_MESSAGE.value,
    EventType.INGEST_DOCUMENT.value,
    EventType.MEMORY_NOTE.value,
}


@dataclass
class MemoryHit:
    event_id: str
    source: str
    trusted: bool
    ts: str
    text: str
    rank: float


class MemoryStore:
    def __init__(self, db: Database, events: EventStore) -> None:
        self._db = db
        self._events = events

    @property
    def events(self) -> EventStore:
        return self._events

    # ---- write path: log first, projection second ----

    def record(self, event: Event) -> Event:
        self._events.append(event)
        self._project(event)
        return event

    def _project(self, event: Event) -> None:
        text = str(event.payload.get("text", ""))
        if event.type not in PROJECTED_TYPES or not text.strip():
            return
        with self._db.conn as conn:
            conn.execute(
                "INSERT INTO memory_fts (event_id, source, trusted, ts, text) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    event.id,
                    event.provenance.source,
                    int(event.provenance.trusted),
                    event.ts.isoformat(),
                    text,
                ),
            )

    # ---- read path ----

    def recall(
        self, query: str, k: int = 6, exclude_ids: set[str] | None = None
    ) -> list[MemoryHit]:
        terms = re.findall(r"[A-Za-z0-9_]{2,}", query)
        if not terms:
            return []
        match = " OR ".join(terms)
        rows = self._db.conn.execute(
            "SELECT event_id, source, trusted, ts, text, bm25(memory_fts) AS rank "
            "FROM memory_fts WHERE memory_fts MATCH ? "
            "AND event_id NOT IN (SELECT event_id FROM tombstones) "
            "ORDER BY rank LIMIT ?",
            (match, k * 3),
        ).fetchall()
        exclude = exclude_ids or set()
        hits = [
            MemoryHit(
                event_id=r["event_id"],
                source=r["source"],
                trusted=bool(int(r["trusted"])),
                ts=r["ts"],
                text=r["text"],
                rank=float(r["rank"]),
            )
            for r in rows
            if r["event_id"] not in exclude
        ]
        return hits[:k]

    # ---- forgetting: tombstone + purge projections + verify ----

    def forget(self, event_id: str, reason: str = "user request") -> dict[str, Any]:
        with self._db.conn as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tombstones (event_id, ts, reason) VALUES (?, ?, ?)",
                (event_id, now().isoformat(), reason),
            )
            conn.execute("DELETE FROM memory_fts WHERE event_id = ?", (event_id,))
        remaining = self._db.conn.execute(
            "SELECT COUNT(*) AS n FROM memory_fts WHERE event_id = ?", (event_id,)
        ).fetchone()["n"]
        return {
            "event_id": event_id,
            "tombstoned": True,
            "projection_purged": remaining == 0,
            "verified": remaining == 0,
        }

    # ---- the projection is rebuildable, and this proves it ----

    def reindex(self) -> int:
        with self._db.conn as conn:
            conn.execute("DELETE FROM memory_fts")
        count = 0
        for event in self._events.recent(types=list(PROJECTED_TYPES), limit=1_000_000):
            self._project(event)
            count += 1
        return count
