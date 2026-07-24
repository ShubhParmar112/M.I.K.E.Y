"""Memory consolidation (slice 2): a session becomes a recallable episodic memory,
tagged with the right tier and inheriting the session's privacy.
"""

from __future__ import annotations

from core.events.schema import Event, EventType, Provenance, Tier
from core.events.store import EventStore
from core.memory.consolidation import Consolidator
from core.memory.store import MemoryStore
from core.models.fake_adapter import FakeAdapter
from core.models.gateway import ModelGateway, ModelResponse
from core.storage.db import Database


def _seed_session(memory: MemoryStore, session: str, tier: Tier = Tier.T1) -> None:
    memory.record(Event(
        type=EventType.USER_MESSAGE.value, tier=tier,
        payload={"text": "let's split the git history", "session_id": session, "turn_id": "t1"}))
    memory.record(Event(
        type=EventType.ASSISTANT_MESSAGE.value, tier=tier,
        provenance=Provenance(source="agent", trusted=True),
        payload={"text": "done — split into five commits", "session_id": session, "turn_id": "t1"}))


def _consolidator(summary: str) -> Consolidator:
    return Consolidator(ModelGateway(FakeAdapter([ModelResponse(text=summary, tool_calls=[])])))


async def test_consolidate_creates_a_recallable_episode(db: Database) -> None:
    memory = MemoryStore(db, EventStore(db))
    _seed_session(memory, "s1")

    summary = await _consolidator(
        "We split the git history into five logical commits."
    ).consolidate_session(memory, "s1")
    assert summary == "We split the git history into five logical commits."

    ep = memory.episode_for("s1")
    assert ep is not None and ep.type == EventType.MEMORY_EPISODE.value

    # recallable, and labeled as the episodic tier
    hits = memory.recall("git history commits")
    assert any(h.kind == "episode" and "five logical commits" in h.text for h in hits)


async def test_consolidate_is_idempotent_unless_forced(db: Database) -> None:
    memory = MemoryStore(db, EventStore(db))
    _seed_session(memory, "s1")

    assert await _consolidator("summary one").consolidate_session(memory, "s1") == "summary one"
    # already consolidated → skip
    assert await _consolidator("summary two").consolidate_session(memory, "s1") is None
    # force re-does it
    assert await _consolidator("summary three").consolidate_session(
        memory, "s1", force=True) == "summary three"


async def test_short_session_is_not_consolidated(db: Database) -> None:
    memory = MemoryStore(db, EventStore(db))
    memory.record(Event(type=EventType.USER_MESSAGE.value,
                        payload={"text": "hi", "session_id": "s1"}))
    assert await _consolidator("x").consolidate_session(memory, "s1", min_turns=2) is None


async def test_private_session_episode_inherits_t0(db: Database) -> None:
    memory = MemoryStore(db, EventStore(db))
    _seed_session(memory, "s1", tier=Tier.T0)
    await _consolidator("private summary").consolidate_session(memory, "s1")
    ep = memory.episode_for("s1")
    assert ep is not None and ep.tier is Tier.T0


async def test_consolidate_survives_a_dead_model(db: Database) -> None:
    memory = MemoryStore(db, EventStore(db))
    _seed_session(memory, "s1")

    class _Boom:
        name = "boom"
        local = False

        async def complete(self, system, messages, tools) -> ModelResponse:  # noqa: ANN001
            raise RuntimeError("provider down")

    assert await Consolidator(ModelGateway(_Boom())).consolidate_session(memory, "s1") is None
    assert memory.episode_for("s1") is None  # nothing recorded on failure


def test_remembered_fact_recalls_as_the_semantic_tier(db: Database) -> None:
    memory = MemoryStore(db, EventStore(db))
    memory.remember("my thesis advisor is Dr. Gonsalves")
    hits = memory.recall("thesis advisor")
    assert hits and hits[0].kind == "fact"
