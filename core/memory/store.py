"""Memory Module — Gen 2 first slice.

The event log remains the source of truth; this module maintains a rebuildable
FTS5 projection over it (`reindex` proves it), retrieval with provenance, and
verified forgetting via tombstones.

Retrieval is keyword (BM25) + recency for now. The vector index slots in here
behind the same `recall()` seam once a local embedding model is available
(ADR-001: embeddings stay local; Groq doesn't serve them).
"""

from __future__ import annotations

import array
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from core.events.schema import Event, EventType, Provenance, Tier, now
from core.events.store import EventStore
from core.models.gateway import ModelUnavailable
from core.storage.db import Database


class Embedder(Protocol):
    name: str

    def embed(self, text: str) -> list[float]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

_WORD_RE = re.compile(r"[a-z0-9]+")

# Dropped from recall queries so retrieval doesn't OR-match on filler like "in"
# or "the" and surface irrelevant chunks (a precision fix the eval harness caught).
STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does down for from had has have
    how i if in into is it its may me might no not of off on or our out over shall should so than
    that the their them then these they this those to up us was we were what when where which who
    whom why will with would you your""".split()
)


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(a: str, b: str) -> float:
    """Cheap, dependency-free text overlap — good enough to catch a fact being
    remembered twice. The vector index will sharpen this later (ADR-001)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

# Event types that become searchable memories.
PROJECTED_TYPES = {
    EventType.USER_MESSAGE.value,
    EventType.ASSISTANT_MESSAGE.value,
    EventType.INGEST_DOCUMENT.value,
    EventType.MEMORY_NOTE.value,
    EventType.MEMORY_EPISODE.value,
}

# Memory tiers (docs/04 §6): the kind of memory a hit is, derived from its event
# type. `fact` = a durable preference/fact (semantic); `episode` = a summary of
# what happened (episodic); `document` = ingested reference; `message` = raw turn.
_TIER_BY_TYPE = {
    EventType.MEMORY_NOTE.value: "fact",
    EventType.MEMORY_EPISODE.value: "episode",
    EventType.INGEST_DOCUMENT.value: "document",
    EventType.USER_MESSAGE.value: "message",
    EventType.ASSISTANT_MESSAGE.value: "message",
}


@dataclass
class MemoryHit:
    event_id: str
    source: str
    trusted: bool
    ts: str
    text: str
    rank: float
    kind: str = "memory"  # tier: fact | episode | document | message (set on recall)


@dataclass
class RememberResult:
    event_id: str
    status: str  # "stored" | "duplicate" | "superseded"
    duplicate_of: str | None = None
    superseded: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    # Pre-existing memory (incl. ingested sources) relevant to this fact, so the
    # caller can verify/cite instead of confabulating or flattering.
    grounding: list["MemoryHit"] = field(default_factory=list)


class MemoryStore:
    def __init__(
        self, db: Database, events: EventStore, embedder: Embedder | None = None
    ) -> None:
        self._db = db
        self._events = events
        self._embedder = embedder  # None → keyword-only retrieval

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

    # ---- read path: keyword (BM25) + semantic (vectors), fused ----

    def recall(
        self, query: str, k: int = 6, exclude_ids: set[str] | None = None
    ) -> list[MemoryHit]:
        """Hybrid retrieval. Keyword catches exact terms; semantic catches
        paraphrase ('who created this' → the authors). Fused with reciprocal-rank
        fusion. Degrades to keyword-only if no embedder is set or it is down, so a
        missing/paused embedding model never breaks recall. Each hit is labeled with
        its memory tier (fact/episode/document/message)."""
        hits = self._recall_hits(query, k, exclude_ids)
        self._label_tiers(hits)
        return hits

    def _recall_hits(
        self, query: str, k: int, exclude_ids: set[str] | None
    ) -> list[MemoryHit]:
        if self._embedder is None:
            return self._keyword_recall(query, k, exclude_ids)
        try:
            semantic = self._semantic_search(query, k * 3, exclude_ids)
        except ModelUnavailable:
            return self._keyword_recall(query, k, exclude_ids)
        keyword = self._keyword_recall(query, k * 3, exclude_ids)
        if not semantic:
            return keyword[:k]
        return self._rrf_merge(keyword, semantic, k)

    def _label_tiers(self, hits: list[MemoryHit]) -> None:
        """Set each hit's tier from its source event type (one query)."""
        ids = [h.event_id for h in hits]
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        rows = self._db.conn.execute(
            f"SELECT id, type FROM events WHERE id IN ({marks})", tuple(ids)
        ).fetchall()
        tier = {r["id"]: _TIER_BY_TYPE.get(r["type"], "memory") for r in rows}
        for h in hits:
            h.kind = tier.get(h.event_id, "memory")

    def _keyword_recall(
        self, query: str, k: int = 6, exclude_ids: set[str] | None = None
    ) -> list[MemoryHit]:
        terms = [
            t for t in re.findall(r"[A-Za-z0-9_]{2,}", query.lower()) if t not in STOPWORDS
        ]
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

    def _semantic_search(
        self, query: str, k: int, exclude_ids: set[str] | None = None
    ) -> list[MemoryHit]:
        assert self._embedder is not None
        qv = self._embedder.embed(query)  # may raise ModelUnavailable
        exclude = exclude_ids or set()
        rows = self._db.conn.execute(
            "SELECT v.event_id, v.vector, m.source, m.trusted, m.ts, m.text "
            "FROM memory_vectors v JOIN memory_fts m ON v.event_id = m.event_id "
            "WHERE v.event_id NOT IN (SELECT event_id FROM tombstones)"
        ).fetchall()
        scored: list[tuple[float, Any]] = []
        for r in rows:
            if r["event_id"] in exclude:
                continue
            vec = array.array("f")
            vec.frombytes(r["vector"])
            scored.append((_cosine(qv, list(vec)), r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            MemoryHit(
                event_id=r["event_id"], source=r["source"],
                trusted=bool(int(r["trusted"])), ts=r["ts"], text=r["text"], rank=-score,
            )
            for score, r in scored[:k]
        ]

    @staticmethod
    def _rrf_merge(
        keyword: list[MemoryHit], semantic: list[MemoryHit], k: int, c: int = 60
    ) -> list[MemoryHit]:
        """Reciprocal-rank fusion: combine two ranked lists without needing their
        scores to be on the same scale."""
        scores: dict[str, float] = {}
        hit_by_id: dict[str, MemoryHit] = {}
        for ranked in (keyword, semantic):
            for rank, h in enumerate(ranked):
                scores[h.event_id] = scores.get(h.event_id, 0.0) + 1.0 / (c + rank + 1)
                hit_by_id.setdefault(h.event_id, h)
        order = sorted(scores, key=lambda e: scores[e], reverse=True)
        return [hit_by_id[e] for e in order[:k]]

    def index_vectors(self) -> int:
        """Embed and store any projected memory chunk that lacks a vector yet
        (incremental — cheap to call repeatedly). No-op without an embedder."""
        if self._embedder is None:
            return 0
        rows = self._db.conn.execute(
            "SELECT event_id, text FROM memory_fts "
            "WHERE event_id NOT IN (SELECT event_id FROM memory_vectors) "
            "AND event_id NOT IN (SELECT event_id FROM tombstones)"
        ).fetchall()
        count = 0
        for r in rows:
            try:
                vec = self._embedder.embed(r["text"])
            except ModelUnavailable:
                break  # embedder down — leave the rest for a later pass
            with self._db.conn as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO memory_vectors (event_id, vector) VALUES (?, ?)",
                    (r["event_id"], array.array("f", vec).tobytes()),
                )
            count += 1
        return count

    # ---- durable notes with hygiene: dedup, supersede, flag conflicts ----

    DUP_THRESHOLD = 0.85  # at/above this overlap, it's the same fact — don't dupe
    RELATED_THRESHOLD = 0.4  # in [RELATED, DUP): possibly conflicting — flag it

    def recent_notes(self, limit: int = 500) -> list[Event]:
        """Durable memory notes only (excludes tombstoned — EventStore.recent does)."""
        return self._events.recent(types=[EventType.MEMORY_NOTE.value], limit=limit)

    def episode_for(self, session_id: str) -> Event | None:
        """The existing episodic summary for a session, if any (for idempotency)."""
        for ev in self._events.recent(types=[EventType.MEMORY_EPISODE.value], limit=100_000):
            if ev.payload.get("session_id") == session_id:
                return ev
        return None

    def record_episode(
        self,
        session_id: str,
        summary: str,
        *,
        tier: Tier = Tier.T1,
        device: str = "dev_desktop_1",
        turn_ids: list[str] | None = None,
    ) -> Event:
        """Record an episodic memory — a summary of what happened in a session.
        Provenance is `agent` (M.I.K.E.Y wrote it); tier mirrors the session's most
        sensitive turn so a private session's summary stays on-device too."""
        return self.record(
            Event(
                type=EventType.MEMORY_EPISODE.value,
                device=device,
                tier=tier,
                provenance=Provenance(source="agent", trusted=True),
                payload={"text": summary, "session_id": session_id, "turns": turn_ids or []},
            )
        )

    def remember(
        self,
        text: str,
        *,
        source: str = "user",
        trusted: bool = True,
        turn_id: str = "",
        device: str = "dev_desktop_1",
        tier: Tier = Tier.T1,
        supersedes: list[str] | None = None,
    ) -> RememberResult:
        """Persist a durable fact while keeping memory clean (Gen 2: contradiction
        flagging + verified forgetting): skip a near-duplicate, tombstone anything
        the caller explicitly replaces, and surface related existing facts so a
        contradiction gets reconciled instead of silently doubling up."""
        text = text.strip()
        scored = sorted(
            ((_jaccard(text, str(n.payload.get("text", ""))), n) for n in self.recent_notes()),
            key=lambda x: x[0],
            reverse=True,
        )
        supersedes = list(supersedes or [])

        # Same fact already on file, and we're not explicitly replacing anything.
        if not supersedes and scored and scored[0][0] >= self.DUP_THRESHOLD:
            return RememberResult(
                event_id=scored[0][1].id, status="duplicate", duplicate_of=scored[0][1].id
            )

        # What do external SOURCES (ingested docs, connectors) say about this?
        # Restricted to citable provenance on purpose: grounding a claim against
        # conversation chatter — including the assistant's own past mistakes — would
        # reinforce errors, not catch them. Captured before recording.
        grounding = [
            h for h in self.recall(text, k=12)
            if h.source.startswith(("connector:", "web:"))
        ][:2]

        known_ids = {n.id for _, n in scored}
        superseded: list[str] = []
        for sid in supersedes:
            if sid in known_ids:  # ignore ids that aren't live notes
                self.forget(sid, reason="superseded by a newer memory")
                superseded.append(sid)

        ev = self.record(
            Event(
                type=EventType.MEMORY_NOTE.value,
                device=device,
                tier=tier,
                provenance=Provenance(source=source, trusted=trusted),
                payload={"text": text, "turn_id": turn_id, "supersedes": superseded},
            )
        )
        related = [
            n.id
            for s, n in scored
            if self.RELATED_THRESHOLD <= s < self.DUP_THRESHOLD and n.id not in superseded
        ][:3]
        return RememberResult(
            event_id=ev.id,
            status="superseded" if superseded else "stored",
            superseded=superseded,
            related=related,
            grounding=grounding,
        )

    # ---- forgetting: tombstone + purge projections + verify ----

    def forget(self, event_id: str, reason: str = "user request") -> dict[str, Any]:
        with self._db.conn as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tombstones (event_id, ts, reason) VALUES (?, ?, ?)",
                (event_id, now().isoformat(), reason),
            )
            conn.execute("DELETE FROM memory_fts WHERE event_id = ?", (event_id,))
            conn.execute("DELETE FROM memory_vectors WHERE event_id = ?", (event_id,))
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
        # Vectors are keyed by stable event id, so they survive the fts rebuild;
        # fill in any that are missing (incremental, no-op without an embedder).
        self.index_vectors()
        return count
