"""Proactivity: what gets noticed, and — mostly — when it is kept quiet.

The rules are the easy half. Most of what is pinned here is restraint: a system
that says true things at wrong moments gets muted, and a muted assistant is worse
than a silent one because you have also stopped trusting it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.config import Config
from core.cost.governor import CostGovernor
from core.events.schema import Event, EventType, now
from core.events.store import EventStore
from core.missions.store import MissionStep, MissionStore
from core.proactive.brief import compose
from core.proactive.discipline import (
    MAX_PER_SESSION,
    MUTE_AFTER_DISMISSALS,
    Attention,
    choose,
    is_quiet_hours,
    may_speak,
)
from core.proactive.nudge import NudgeStore, Urgency, new_nudge
from core.proactive.rules import MissionGlance, World, evaluate
from core.proactive.watch import Sentinel
from core.storage.db import Database


@pytest.fixture
def events(tmp_path) -> EventStore:  # noqa: ANN001
    return EventStore(Database(tmp_path / "t.db"))


def _world(**kwargs) -> World:  # noqa: ANN003
    base = {"at": now(), "day_key": "2026-07-27", "cloud_providers": ["groq"]}
    base.update(kwargs)
    return World(**base)  # type: ignore[arg-type]


# --- what gets noticed --------------------------------------------------------


def test_a_spent_allowance_with_no_backup_is_urgent() -> None:
    """The failure that started all of this: answers silently get worse and nothing
    says why."""
    found = evaluate(_world(provider_pressure=[("groq", 1.0, 0, True)]))
    quota = [n for n in found if n.kind == "quota"]

    assert len(quota) == 1
    assert quota[0].urgency == Urgency.HIGH
    assert "local model" in quota[0].text


def test_the_same_situation_with_a_backup_barely_registers() -> None:
    """Same fact, different consequence — and the urgency has to follow the
    consequence, or the warning is crying wolf."""
    found = evaluate(
        _world(
            cloud_providers=["groq", "cerebras"],
            provider_pressure=[("groq", 1.0, 0, True)],
        )
    )
    quota = [n for n in found if n.kind == "quota"]

    assert quota[0].urgency == Urgency.LOW
    assert "cerebras is covering" in quota[0].text


def test_a_comfortable_day_says_nothing() -> None:
    assert evaluate(_world(provider_pressure=[("groq", 0.3, 40, False)])) == []


def test_being_served_locally_right_now_is_reported() -> None:
    found = evaluate(_world(recent_local_turns=4, recent_turns=6))

    assert any(n.kind == "degraded" for n in found)


def test_a_local_only_setup_is_not_told_it_is_local() -> None:
    """Running on-device by choice is the configuration working, not a symptom."""
    found = evaluate(_world(cloud_providers=[], recent_local_turns=6, recent_turns=6))

    assert not [n for n in found if n.kind == "degraded"]


def test_a_mission_that_stopped_moving_is_surfaced() -> None:
    at = now()
    found = evaluate(
        _world(
            at=at,
            missions=[
                MissionGlance("m1", "tidy the notes folder", at - timedelta(hours=9), "running")
            ],
        )
    )
    mission = [n for n in found if n.kind == "mission"]

    assert mission and "tidy the notes folder" in mission[0].text


def test_a_mission_that_moved_recently_is_left_alone() -> None:
    at = now()
    found = evaluate(
        _world(at=at, missions=[MissionGlance("m1", "goal", at - timedelta(minutes=5), "running")])
    )

    assert not [n for n in found if n.kind == "mission"]


def test_a_broken_audit_chain_is_the_one_thing_worth_interrupting_for() -> None:
    found = evaluate(_world(audit_ok=False))

    assert found[0].kind == "integrity", "it must come first"
    assert found[0].urgency == Urgency.HIGH


# --- restraint ----------------------------------------------------------------


def test_nothing_is_volunteered_mid_turn() -> None:
    """Interrupting the thing you asked for to mention something else is the single
    most irritating behaviour a system can have."""
    nudge = new_nudge("integrity", "urgent", urgency=Urgency.HIGH)
    verdict = may_speak(nudge, Attention(at=_awake(), busy=True))

    assert not verdict.speak
    assert "mid-turn" in verdict.reason


def test_quiet_hours_hold_ordinary_notes_until_morning() -> None:
    at = datetime(2026, 7, 27, 3, 0, tzinfo=UTC)
    assert is_quiet_hours(at)

    ordinary = may_speak(new_nudge("quota", "you're at 80%"), Attention(at=at))
    urgent = may_speak(
        new_nudge("integrity", "chain broken", urgency=Urgency.HIGH), Attention(at=at)
    )

    assert not ordinary.speak
    assert urgent.speak, "genuinely urgent still gets through"


def test_a_session_has_a_ceiling_on_interruptions() -> None:
    attention = Attention(at=_awake(), said_this_session=MAX_PER_SESSION)

    assert not may_speak(new_nudge("quota", "another thing"), attention).speak


def test_urgency_overrides_the_ceiling() -> None:
    attention = Attention(at=_awake(), said_this_session=MAX_PER_SESSION + 5)

    assert may_speak(new_nudge("integrity", "!", urgency=Urgency.HIGH), attention).speak


def test_a_kind_dismissed_often_enough_stops_being_raised() -> None:
    """Being told twice is a reminder; being told nine times is a reason to stop
    listening to any of it. Saying no repeatedly IS the off switch."""
    attention = Attention(
        at=_awake(), dismissals_by_kind={"quota": MUTE_AFTER_DISMISSALS}
    )

    verdict = may_speak(new_nudge("quota", "at 80% again"), attention)

    assert not verdict.speak
    assert "dismissed" in verdict.reason


def test_the_budget_keeps_the_most_pressing_not_the_most_recent() -> None:
    at = _awake()
    old_urgent = new_nudge("integrity", "chain broken", urgency=Urgency.HIGH)
    chatter = [new_nudge("quota", f"note {i}", dedup_key=f"k{i}") for i in range(5)]

    speak, held = choose([*chatter, old_urgent], Attention(at=at))

    assert speak[0].kind == "integrity"
    assert len(speak) == MAX_PER_SESSION
    assert len(held) == 3
    assert all(reason for _n, reason in held), "held items must say why"


def _awake() -> datetime:
    return datetime(2026, 7, 27, 14, 0, tzinfo=UTC)


# --- durability: a nudge survives the process that noticed it -----------------


def test_a_nudge_outlives_the_process_that_raised_it(events: EventStore) -> None:
    """The thing worth telling you is usually noticed while you are away."""
    NudgeStore(events).raise_nudge(new_nudge("quota", "allowance is nearly gone"))

    rebuilt = NudgeStore(events)  # a fresh process, same log

    assert [n.text for n in rebuilt.pending()] == ["allowance is nearly gone"]


def test_the_same_observation_twice_is_not_two_nudges(events: EventStore) -> None:
    """A rule is evaluated on a timer, so it is true on every tick for hours."""
    store = NudgeStore(events)
    first = store.raise_nudge(new_nudge("quota", "gone", dedup_key="quota:today"))
    second = store.raise_nudge(new_nudge("quota", "gone", dedup_key="quota:today"))

    assert first is not None
    assert second is None
    assert len(store.pending()) == 1


def test_a_dismissed_nudge_does_not_come_straight_back(events: EventStore) -> None:
    """Found live: dismissing one and running the watcher again raised it
    immediately, because nothing outstanding matched any more. Dismissing has to
    mean something or the feature is a nag."""
    store = NudgeStore(events)
    nudge = store.raise_nudge(new_nudge("degraded", "on the local model", dedup_key="degraded:day"))
    assert nudge is not None
    store.deliver(nudge.id, outcome="dismissed")

    again = store.raise_nudge(new_nudge("degraded", "on the local model", dedup_key="degraded:day"))

    assert again is None
    assert store.pending() == []


def test_a_condition_still_true_tomorrow_may_speak_again(events: EventStore) -> None:
    """Quiet for a day, not forever: the keys that matter are day-scoped, so a
    problem that outlives the day gets to say so again."""
    store = NudgeStore(events)
    first = store.raise_nudge(new_nudge("quota", "gone", dedup_key="quota:mon"))
    assert first is not None
    store.deliver(first.id, outcome="dismissed")

    tomorrow = now() + timedelta(hours=25)
    spent = store.spent_keys(at=tomorrow)

    assert "quota:mon" not in spent


def test_delivering_closes_it(events: EventStore) -> None:
    store = NudgeStore(events)
    nudge = store.raise_nudge(new_nudge("quota", "gone"))
    assert nudge is not None

    store.deliver(nudge.id, how="chat", outcome="shown")

    assert store.pending() == []


def test_an_undelivered_nudge_goes_stale_rather_than_arriving_late(
    events: EventStore,
) -> None:
    """"About four calls left today" is useful for an hour and misleading tomorrow."""
    store = NudgeStore(events)
    store.raise_nudge(new_nudge("quota", "four left", urgency=Urgency.HIGH))

    later = now() + timedelta(hours=8)

    assert store.pending(at=later) == []
    assert store.expire_stale(at=later) == 1
    assert store.pending(include_stale=True, at=later) == [], "expiry is recorded, not implicit"


def test_two_watchers_racing_still_raise_it_once(events: EventStore) -> None:
    """The background watcher and a request handler asking for a brief genuinely do
    run at the same time — the first live run produced the same sentence twice
    because both read an empty queue before either wrote."""
    import threading

    store = NudgeStore(events)
    barrier = threading.Barrier(4)
    results: list[object] = []
    lock = threading.Lock()

    def racer() -> None:
        barrier.wait()
        outcome = store.raise_nudge(new_nudge("quota", "gone", dedup_key="quota:today"))
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=racer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(1 for r in results if r is not None) == 1
    assert len(store.pending()) == 1


def test_a_nudge_can_be_closed_over_http(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """This endpoint's body model must live at module scope: app.py uses postponed
    annotations, so a model defined inside create_app is invisible when FastAPI
    resolves the handler's types, and the body is silently read off the query string
    instead. The first live POST failed with "field required"."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MIKEY_HOME", str(tmp_path))
    monkeypatch.setenv("MIKEY_VECTORS", "0")
    monkeypatch.setenv("MIKEY_PROACTIVE", "0")  # no background watcher in a test
    config = Config()
    store = NudgeStore(EventStore(Database(config.db_path)), config.device_id)
    raised = store.raise_nudge(new_nudge("quota", "allowance nearly gone"))
    assert raised is not None

    from core.gateway.app import create_app

    with TestClient(create_app(config)) as client:
        listed = client.get("/v1/nudges").json()
        assert [n["text"] for n in listed["nudges"]] == ["allowance nearly gone"]

        closed = client.post(f"/v1/nudges/{raised.id}", json={"outcome": "dismissed"})

        assert closed.status_code == 200, closed.text
        assert client.get("/v1/nudges").json()["nudges"] == []
        assert client.get("/v1/nudges").json()["dismissals"] == {"quota": 1}


def test_reading_the_queue_does_not_consume_it(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """A nudge consumed by a health check is a nudge that was never delivered."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MIKEY_HOME", str(tmp_path))
    monkeypatch.setenv("MIKEY_VECTORS", "0")
    monkeypatch.setenv("MIKEY_PROACTIVE", "0")
    config = Config()
    store = NudgeStore(EventStore(Database(config.db_path)), config.device_id)
    store.raise_nudge(new_nudge("quota", "still here"))

    from core.gateway.app import create_app

    with TestClient(create_app(config)) as client:
        for _ in range(3):
            assert len(client.get("/v1/nudges").json()["nudges"]) == 1


def test_dismissals_are_counted_by_kind(events: EventStore) -> None:
    store = NudgeStore(events)
    for i in range(2):
        nudge = store.raise_nudge(new_nudge("quota", f"n{i}", dedup_key=f"k{i}"))
        assert nudge is not None
        store.deliver(nudge.id, outcome="dismissed")
    shown = store.raise_nudge(new_nudge("mission", "m", dedup_key="m"))
    assert shown is not None
    store.deliver(shown.id, outcome="shown")

    assert store.dismissals_by_kind() == {"quota": 2}


# --- the sentinel, against real stores ----------------------------------------


def test_the_sentinel_notices_and_records_without_speaking(
    events: EventStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MIKEY_LOCAL_BRAINS", raising=False)
    nudges = NudgeStore(events)
    governor = CostGovernor(events, budget_usd=10.0)
    # A day's worth of Groq tokens, all of it.
    governor.record("groq", "llama-3.3-70b-versatile", {"input_tokens": 100_000, "output_tokens": 0})

    sentinel = Sentinel(Config(), events, nudges, governor, MissionStore(events))
    raised = sentinel.tick()

    assert [n.kind for n in raised] == ["quota"]
    assert len(nudges.pending()) == 1
    # Ticking again must not stack the same observation up.
    assert sentinel.tick() == []
    assert len(nudges.pending()) == 1


def test_the_sentinel_survives_a_check_that_itself_fails(events: EventStore) -> None:
    """A watchdog that can be killed by what it is watching is not a watchdog."""

    class _Broken:
        def verify_audit_chain(self) -> bool:
            raise RuntimeError("database is on fire")

    sentinel = Sentinel(
        Config(),
        events,
        NudgeStore(events),
        CostGovernor(events, budget_usd=0.0),
        MissionStore(events),
        _Broken(),
    )

    assert sentinel.tick() == []


# --- the brief ----------------------------------------------------------------


def test_the_brief_counts_what_actually_happened(events: EventStore) -> None:
    for _ in range(3):
        events.append(Event(type=EventType.USER_MESSAGE.value, payload={"text": "hi"}))
    events.append(Event(type=EventType.MEMORY_NOTE.value, payload={"text": "a fact"}))
    events.append(Event(type=EventType.ACTION_EXECUTED.value, payload={"tool": "fs_write"}))

    brief = compose(events, nudges=[], missions_open=1)

    assert brief.conversations == 3
    assert brief.remembered == 1
    assert brief.actions == 1
    assert not brief.quiet
    joined = " ".join(brief.lines())
    assert "3 exchanges" in joined
    assert "1 mission is still open" in joined


def test_a_quiet_day_is_said_briefly_not_padded(events: EventStore) -> None:
    brief = compose(events, nudges=[])

    assert brief.quiet
    assert brief.lines() == ["Nothing outstanding."]


def test_the_brief_leads_with_what_is_wrong(events: EventStore) -> None:
    events.append(Event(type=EventType.USER_MESSAGE.value, payload={"text": "hi"}))
    nudge = new_nudge("integrity", "The audit chain no longer verifies.", urgency=Urgency.HIGH)

    lines = compose(events, nudges=[nudge]).lines()

    assert lines[0] == nudge.text


def test_the_brief_costs_no_model_call(events: EventStore) -> None:
    """It is composed from the log. The one moment a summary must be trustworthy is
    when nobody asked for it — and it must not spend the day's quota to say
    'nothing happened'."""
    before = len(events.recent(types=[EventType.MODEL_USAGE.value], limit=100))

    compose(events, nudges=[], missions_open=0)

    assert len(events.recent(types=[EventType.MODEL_USAGE.value], limit=100)) == before


def test_an_open_mission_shows_up_in_the_brief(events: EventStore) -> None:
    missions = MissionStore(events)
    missions.create("tidy up", [MissionStep("fs_list", {"path": "."})])

    brief = compose(events, nudges=[], missions_open=len(missions.active()))

    assert brief.missions_open == 1
