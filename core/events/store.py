"""Append-only event log. Append and query — there is no update, no delete.

(Forgetting arrives in Gen 2 as tombstone events + projection rebuild.)
"""

from __future__ import annotations

import json

from core.events.schema import Event
from core.storage.db import Database


class EventStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def append(self, event: Event) -> Event:
        with self._db.conn as conn:
            conn.execute(
                "INSERT INTO events (id, v, type, ts, device, tier, provenance, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.id,
                    event.v,
                    event.type,
                    event.ts.isoformat(),
                    event.device,
                    event.tier.value,
                    event.provenance.model_dump_json(),
                    json.dumps(event.payload, ensure_ascii=False),
                ),
            )
        return event

    def recent(self, types: list[str] | None = None, limit: int = 50) -> list[Event]:
        """Most recent non-tombstoned events, oldest-first for direct use as history."""
        not_dead = "id NOT IN (SELECT event_id FROM tombstones)"
        if types:
            marks = ",".join("?" * len(types))
            rows = self._db.conn.execute(
                f"SELECT * FROM events WHERE type IN ({marks}) AND {not_dead} "
                "ORDER BY rowid DESC LIMIT ?",
                (*types, limit),
            ).fetchall()
        else:
            rows = self._db.conn.execute(
                f"SELECT * FROM events WHERE {not_dead} ORDER BY rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_event(r) for r in reversed(rows)]

    @staticmethod
    def _row_to_event(row: object) -> Event:
        return Event.model_validate(
            {
                "id": row["id"],  # type: ignore[index]
                "v": row["v"],  # type: ignore[index]
                "type": row["type"],  # type: ignore[index]
                "ts": row["ts"],  # type: ignore[index]
                "device": row["device"],  # type: ignore[index]
                "tier": row["tier"],  # type: ignore[index]
                "provenance": json.loads(row["provenance"]),  # type: ignore[index]
                "payload": json.loads(row["payload"]),  # type: ignore[index]
            }
        )
