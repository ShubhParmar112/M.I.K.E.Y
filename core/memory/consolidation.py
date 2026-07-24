"""Memory consolidation (the "deepen his mind" arc, slice 2 → 3).

Turns a session's raw turns into ONE **episodic memory** — a short record of what
happened — so M.I.K.E.Y can later recall events ("last time we split the git
history") and not only facts. This is the first consolidation job; Dream Mode
(slice 3) will run it in the background on idle sessions.

Privacy composes for free: a session's episode inherits its most sensitive turn's
tier, so a private session is both summarized on-device (T0 routing) and its
episode is excluded from cloud training.
"""

from __future__ import annotations

from core.events.schema import EventType, Tier
from core.memory.store import MemoryStore
from core.models.gateway import ChatMessage, ModelGateway, RoutingMeta

CONSOLIDATION_PROMPT = """You are M.I.K.E.Y's memory consolidator. Turn the session transcript \
into ONE short episodic memory — a factual record, in the past tense, of what happened. 2-4 \
sentences. Capture what was discussed or done and any decisions or outcomes; skip greetings and \
pleasantries. Do not invent anything that is not in the transcript. Write only the summary."""

_MAX_TURNS = 80
_MAX_CHARS = 6000


class Consolidator:
    def __init__(self, gateway: ModelGateway) -> None:
        self._gateway = gateway

    async def consolidate_session(
        self, memory: MemoryStore, session_id: str, *, min_turns: int = 2, force: bool = False
    ) -> str | None:
        """Summarize a session into an episodic memory. Returns the summary text,
        or None if there was nothing worth consolidating (too short, already done,
        or the model was unavailable)."""
        if not force and memory.episode_for(session_id) is not None:
            return None

        events = [
            e
            for e in memory.events.recent(
                types=[EventType.USER_MESSAGE.value, EventType.ASSISTANT_MESSAGE.value],
                limit=100_000,
            )
            if e.payload.get("session_id") == session_id
        ]
        if len(events) < min_turns:
            return None
        events = events[-_MAX_TURNS:]

        # The episode is as sensitive as the most sensitive turn it summarizes.
        tier = Tier.T0 if any(e.tier is Tier.T0 for e in events) else Tier.T1

        lines = []
        for e in events:
            who = "User" if e.type == EventType.USER_MESSAGE.value else "M.I.K.E.Y"
            lines.append(f"{who}: {e.payload.get('text', '')}")
        transcript = "\n".join(lines)[:_MAX_CHARS]

        try:
            resp = await self._gateway.complete(
                CONSOLIDATION_PROMPT,
                [ChatMessage(role="user", text=transcript)],
                [],
                RoutingMeta(tier=tier, capability="memory"),
            )
        except Exception:
            return None  # a consolidation failure is never fatal
        summary = resp.text.strip()
        if not summary:
            return None

        memory.record_episode(
            session_id, summary, tier=tier, turn_ids=[e.id for e in events]
        )
        return summary
