"""Run traces (review M7): every turn is a queryable tree of spans.

"Why did you do that?" is answered from this data, never confabulated.
"""

from __future__ import annotations

import json
from typing import Any

from core.events.schema import now, ulid
from core.storage.db import Database


class TraceStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def span(
        self,
        turn_id: str,
        kind: str,
        payload: dict[str, Any],
        parent_id: str | None = None,
    ) -> str:
        span_id = ulid()
        with self._db.conn as conn:
            conn.execute(
                "INSERT INTO traces (turn_id, span_id, parent_id, kind, ts, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (turn_id, span_id, parent_id, kind, now().isoformat(),
                 json.dumps(payload, ensure_ascii=False, default=str)),
            )
        return span_id

    def turn(self, turn_id: str) -> list[dict[str, Any]]:
        rows = self._db.conn.execute(
            "SELECT * FROM traces WHERE turn_id = ? ORDER BY ts, span_id", (turn_id,)
        ).fetchall()
        return [
            {
                "span_id": r["span_id"],
                "parent_id": r["parent_id"],
                "kind": r["kind"],
                "ts": r["ts"],
                "payload": json.loads(r["payload"]),
            }
            for r in rows
        ]

    def recent_turns(self, limit: int = 10) -> list[str]:
        rows = self._db.conn.execute(
            "SELECT turn_id, MAX(ts) AS latest FROM traces GROUP BY turn_id "
            "ORDER BY latest DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [r["turn_id"] for r in rows]
