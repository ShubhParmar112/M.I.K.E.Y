"""The only module that knows SQL exists (ADR-001 Amendment A1.3).

SQLite in WAL mode. Schema is versioned from day one (review M10): migrations
are ordered DDL batches; the current version lives in `schema_version`.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

MIGRATIONS: list[list[str]] = [
    # v1 — Gen 1 baseline
    [
        """CREATE TABLE events (
            id TEXT PRIMARY KEY,
            v INTEGER NOT NULL,
            type TEXT NOT NULL,
            ts TEXT NOT NULL,
            device TEXT NOT NULL,
            tier TEXT NOT NULL,
            provenance TEXT NOT NULL,
            payload TEXT NOT NULL
        )""",
        "CREATE INDEX idx_events_type_ts ON events(type, ts)",
        "CREATE INDEX idx_events_ts ON events(ts)",
        """CREATE TABLE audit (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            decision TEXT NOT NULL,
            reason TEXT NOT NULL,
            payload TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            hash TEXT NOT NULL
        )""",
        """CREATE TABLE traces (
            turn_id TEXT NOT NULL,
            span_id TEXT NOT NULL,
            parent_id TEXT,
            kind TEXT NOT NULL,
            ts TEXT NOT NULL,
            payload TEXT NOT NULL
        )""",
        "CREATE INDEX idx_traces_turn ON traces(turn_id, ts)",
        """CREATE TABLE approvals (
            id TEXT PRIMARY KEY,
            turn_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            request TEXT NOT NULL,
            decision TEXT,
            scope TEXT,
            ts TEXT NOT NULL
        )""",
    ],
    # v2 — Gen 2: memory projection (FTS index over the log) + tombstones.
    # The FTS table is a rebuildable projection; the log stays the truth.
    [
        """CREATE VIRTUAL TABLE memory_fts USING fts5(
            event_id UNINDEXED,
            source UNINDEXED,
            trusted UNINDEXED,
            ts UNINDEXED,
            text
        )""",
        """CREATE TABLE tombstones (
            event_id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            reason TEXT NOT NULL
        )""",
    ],
    # v3 — Gen 2: vector index for semantic retrieval. Another rebuildable
    # projection over the log; embeddings (float32 blobs) stay on-device.
    [
        """CREATE TABLE memory_vectors (
            event_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL
        )""",
    ],
]


class Database:
    """Thread-safe handle to the M.I.K.E.Y SQLite store."""

    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        self._local = threading.local()
        conn = self.conn
        conn.execute("PRAGMA journal_mode=WAL")
        self._migrate(conn)

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        current = row["v"] or 0
        for version, batch in enumerate(MIGRATIONS, start=1):
            if version <= current:
                continue
            with conn:
                for ddl in batch:
                    conn.execute(ddl)
                conn.execute("INSERT INTO schema_version(version) VALUES (?)", (version,))

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
