"""The HUD's judgements, and the one thing its modal must never do.

The pure half (`core.presence.hud`) decides what the dashboard says; the tests
that matter there are about CONSEQUENCE ordering — a spent allowance with another
cloud provider behind it is a non-event, and the same allowance with nothing
behind it is the most important thing on the screen.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from core.presence.hud import BROKEN, DEGRADED, OK, bar, build_hud

NOW = datetime(2026, 7, 28, 12, 0).astimezone()


def health(
    *,
    provider: str = "groq",
    fallback: str | None = "ollama",
    providers: list[dict[str, Any]] | None = None,
    audit: bool = True,
    spend: dict[str, Any] | None = None,
    sidelined: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "fallback": fallback,
        "audit_chain_valid": audit,
        "build": "abc1234",
        "sidelined": sidelined or {},
        "today": {"providers": providers or []},
        "spend": spend or {"total_usd": 0.5, "budget_usd": 10.0},
    }


def gauge(
    provider: str = "groq", *, tokens: int = 0, cap: int = 100_000, **kw: Any
) -> dict[str, Any]:
    base = {
        "provider": provider,
        "tokens": tokens,
        "calls": 0,
        "cap": cap,
        "call_cap": 0,
        "metered_by": "tokens",
        "fraction": tokens / cap if cap else 0.0,
        "calls_left": None,
        "exhausted": False,
        "warning": False,
    }
    base.update(kw)
    return base


# --- the gauge ------------------------------------------------------------


def test_bar_clamps_above_full() -> None:
    """Our tally is a lower bound on the provider's, so it does overshoot the
    published cap — a bar longer than its track reads as a rendering bug."""
    assert bar(1.4, width=10) == "#" * 10
    assert bar(0.0, width=10) == "." * 10


def test_a_little_used_never_looks_like_nothing_used() -> None:
    assert bar(0.01, width=12).startswith("#")


def test_a_provider_with_no_published_cap_gets_no_gauge() -> None:
    """A gauge with no track is noise: it would show a bar that can never move."""
    hud = build_hud(health(providers=[gauge("anthropic", cap=0)]), now=NOW)
    assert hud.gauges == []


# --- the verdict ----------------------------------------------------------


def test_quiet_when_nothing_is_wrong() -> None:
    hud = build_hud(health(providers=[gauge(tokens=1_000)]), now=NOW)
    assert hud.verdict == OK
    assert hud.headline == "everything nominal"


def test_spent_allowance_with_a_cloud_backup_is_a_non_event() -> None:
    hud = build_hud(
        health(
            fallback="cerebras,ollama",
            providers=[gauge(tokens=100_000, exhausted=True, fraction=1.0)],
        ),
        now=NOW,
    )
    assert hud.verdict == OK
    assert "cerebras is covering" in hud.headline


def test_spent_allowance_with_nothing_behind_it_is_the_headline() -> None:
    hud = build_hud(
        health(providers=[gauge(tokens=100_000, exhausted=True, fraction=1.0)]),
        now=NOW,
    )
    assert hud.verdict == DEGRADED
    assert "nothing else is configured" in hud.headline
    assert hud.gauges[0].covered_by is None
    # It knows the next turn will be local — but says so in the future tense,
    # because no turn has actually come back from the local model yet.
    assert hud.answering == "ollama" and not hud.answering_observed
    assert "the next answer comes from the local model" in hud.headline


def test_the_local_model_never_counts_as_cover() -> None:
    """Falling back to the 3B is exactly the outcome the gauge warns about, so it
    cannot also be the reassurance."""
    hud = build_hud(
        health(fallback="ollama", providers=[gauge(tokens=100_000, exhausted=True)]),
        now=NOW,
    )
    assert hud.gauges[0].covered_by is None
    assert hud.verdict == DEGRADED


def test_answering_locally_is_degraded_even_when_all_else_is_healthy() -> None:
    hud = build_hud(health(providers=[gauge(tokens=10)]), now=NOW, served_by="ollama")
    assert hud.verdict == DEGRADED
    assert hud.answering_is_local
    assert "local model" in hud.headline


def test_a_broken_audit_chain_outranks_everything() -> None:
    hud = build_hud(
        health(audit=False, providers=[gauge(tokens=100_000, exhausted=True)]),
        now=NOW,
        served_by="ollama",
    )
    assert hud.verdict == BROKEN
    assert "audit chain" in hud.headline


def test_over_budget_is_degraded() -> None:
    hud = build_hud(
        health(spend={"total_usd": 12.0, "budget_usd": 10.0, "over_budget": True}),
        now=NOW,
    )
    assert hud.verdict == DEGRADED
    assert "budget" in hud.headline


def test_a_warning_gauge_says_how_much_is_left_without_alarming() -> None:
    hud = build_hud(
        health(providers=[gauge(tokens=85_000, fraction=0.85, warning=True, calls_left=4)]),
        now=NOW,
    )
    assert hud.verdict == OK
    assert "85%" in hud.headline and "4 calls left" in hud.headline
    # ...but it is not "nothing to see here": rendered in calm green, this is the
    # warning that gets missed and the evening ends on the local model.
    assert not hud.nominal


def test_the_headline_has_four_weights_not_two() -> None:
    from apps.tui.app import headline_style

    quiet = build_hud(health(providers=[gauge(tokens=10)]), now=NOW)
    warned = build_hud(
        health(providers=[gauge(tokens=85_000, fraction=0.85, warning=True)]), now=NOW
    )
    local = build_hud(health(), now=NOW, served_by="ollama")
    broken = build_hud(health(audit=False), now=NOW)

    assert quiet.nominal and headline_style(quiet) == "green"
    assert headline_style(warned) == "yellow", "a warning must not read as reassurance"
    assert headline_style(local) == "bold black on yellow"
    assert headline_style(broken) == "bold white on red"


# --- the rest of the panel ------------------------------------------------


def test_answering_defaults_to_the_primary_before_any_turn() -> None:
    hud = build_hud(health(), now=NOW)
    assert hud.answering == "groq"
    assert not hud.answering_is_local
    assert not hud.answering_observed, "nothing has been served yet — it is a guess"


def test_before_a_turn_it_names_who_would_actually_answer() -> None:
    """The panel said "answering groq" directly under a headline saying groq was
    spent and cerebras was covering. Both cannot be true."""
    hud = build_hud(
        health(
            fallback="cerebras,ollama",
            sidelined={"groq": (NOW + timedelta(minutes=30)).isoformat()},
            providers=[gauge(tokens=100_000, exhausted=True, fraction=1.0)],
        ),
        now=NOW,
    )
    assert hud.answering == "cerebras"
    assert not hud.answering_observed


def test_what_actually_served_the_turn_always_wins_over_the_guess() -> None:
    hud = build_hud(
        health(fallback="cerebras,ollama", sidelined={"groq": (NOW + timedelta(hours=1)).isoformat()}),
        now=NOW,
        served_by="groq",
    )
    assert hud.answering == "groq" and hud.answering_observed


def test_sidelined_providers_say_when_they_are_back() -> None:
    hud = build_hud(
        health(sidelined={"groq": (NOW + timedelta(minutes=95)).isoformat()}), now=NOW
    )
    assert hud.sidelined == ["groq — back in 1h35m"]


def test_a_sideline_about_to_expire_does_not_read_as_a_bug() -> None:
    hud = build_hud(
        health(sidelined={"groq": (NOW + timedelta(seconds=20)).isoformat()}), now=NOW
    )
    assert hud.sidelined == ["groq — back shortly"]


def test_a_zero_budget_reports_nothing() -> None:
    """0 means tracking-only; a budget of nothing has nothing to say."""
    hud = build_hud(health(spend={"total_usd": 3.0, "budget_usd": 0}), now=NOW)
    assert hud.budget is None


def test_missions_are_shown_with_their_progress() -> None:
    hud = build_hud(
        health(),
        now=NOW,
        missions=[{"goal": "tidy notes", "status": "failed", "steps": 4, "next_step": 2}],
    )
    assert hud.missions == ["tidy notes — step 3/4 (failed)"]


# --- the endpoint the panel reads -----------------------------------------


def test_the_missions_endpoint_reports_failed_ones_too(  # noqa: ANN001
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`MissionStore.active()` excludes failed missions, but a mission that stopped
    on a failed step is exactly the one nobody remembers is still owed — so a HUD
    that only showed `active()` would hide the case it exists to surface."""
    from fastapi.testclient import TestClient

    from core.config import Config
    from core.events.store import EventStore
    from core.missions.store import MissionStep, MissionStore
    from core.storage.db import Database

    monkeypatch.setenv("MIKEY_HOME", str(tmp_path))
    monkeypatch.setenv("MIKEY_VECTORS", "0")
    monkeypatch.setenv("MIKEY_PROACTIVE", "0")
    config = Config()
    missions = MissionStore(EventStore(Database(config.db_path)), config.device_id)

    waiting = missions.create("tidy notes", [MissionStep("fs_write", {"path": "a"})])
    broken = missions.create("push work", [MissionStep("run_command", {"cmd": "git"})])
    missions.record_step_result(broken.id, 0, ok=False, output="no upstream")

    from core.gateway.app import create_app

    with TestClient(create_app(config)) as client:
        body = client.get("/v1/missions").json()

    by_id = {m["id"]: m for m in body["missions"]}
    assert by_id[waiting.id]["status"] == "pending"
    assert by_id[broken.id]["status"] == "failed"
    assert by_id[broken.id]["steps"] == 1


# --- the modal ------------------------------------------------------------


@pytest.mark.asyncio
async def test_enter_does_not_approve_an_action() -> None:
    """The single most important behaviour on this surface.

    An action that reached the approval card is one the policy engine decided a
    person must decide on. Enter is the key someone leans on to dismiss things, so
    it must not be an approval — only an explicit y/s is, and escape denies.
    """
    from apps.tui.app import ApprovalScreen
    from textual.app import App, ComposeResult
    from textual.widgets import Label

    decided: list[tuple[bool, str]] = []

    class Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield Label("harness")

        def on_mount(self) -> None:
            self.push_screen(
                ApprovalScreen(
                    {
                        "tool": "fs_write",
                        "args": {"path": "notes.md"},
                        "reason": "policy: ASK",
                        "approval_id": "ap_1",
                        "preview": {"destructive": True, "reversible": False, "simulated": True,
                                    "summary": "overwrites notes.md"},
                    }
                ),
                callback=lambda r: decided.append(r),
            )

    async with Harness().run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        assert decided == [], "Enter must never approve"
        await pilot.press("escape")
        await pilot.pause()
        assert decided == [(False, "once")]


@pytest.mark.asyncio
async def test_an_approval_pauses_the_turn_until_it_is_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HUD must not be able to run ahead of a decision.

    The failure this pins is a surface that renders the card, keeps consuming the
    stream, and posts the answer afterwards — by which point the gateway has been
    left waiting on an approval nobody is blocking on.
    """
    import apps.tui.app as tui

    order: list[str] = []

    def fake_stream(client: Any, session: str, text: str) -> Any:
        yield {"kind": "status", "turn_id": "t1", "brain": "operator"}
        yield {
            "kind": "approval_request",
            "tool": "fs_write",
            "args": {"path": "a.txt"},
            "approval_id": "ap_1",
            "preview": {"destructive": False, "simulated": True, "summary": "creates a.txt"},
        }
        order.append("stream resumed")
        yield {"kind": "final", "text": "done", "served_by": "ollama"}

    sent: list[tuple[str, bool, str]] = []

    def fake_send(client: Any, approval_id: str, approved: bool, scope: str) -> None:
        order.append("answered")
        sent.append((approval_id, approved, scope))

    monkeypatch.setattr(tui, "stream_turn", fake_stream)
    monkeypatch.setattr(tui, "send_approval", fake_send)
    monkeypatch.setattr(tui, "get", lambda path, **kw: {})

    app = tui.MikeyHud(session="s1", primary="groq", refresh_s=3600)
    async with app.run_test() as pilot:
        await pilot.press(*"hi")
        await pilot.press("enter")
        await pilot.pause()
        for _ in range(40):  # wait for the modal the worker pushed
            if isinstance(app.screen, tui.ApprovalScreen):
                break
            await pilot.pause(0.05)
        assert isinstance(app.screen, tui.ApprovalScreen), "the card never came up"
        assert order == [], "the turn ran on past an unanswered approval"
        await pilot.press("y")
        for _ in range(40):
            if app._served_by:
                break
            await pilot.pause(0.05)

    assert order == ["answered", "stream resumed"]
    assert sent == [("ap_1", True, "once")]
    # And who actually answered is now HUD state, not just a subtitle that scrolls away.
    assert app._served_by == "ollama"


@pytest.mark.asyncio
async def test_y_approves_once_and_s_approves_for_the_session() -> None:
    from apps.tui.app import ApprovalScreen
    from textual.app import App, ComposeResult
    from textual.widgets import Label

    for key, expected in (("y", (True, "once")), ("s", (True, "session")), ("n", (False, "once"))):
        decided: list[tuple[bool, str]] = []

        class Harness(App[None]):
            def compose(self) -> ComposeResult:
                yield Label("harness")

            def on_mount(self) -> None:
                self.push_screen(
                    ApprovalScreen({"tool": "run_command", "args": {}, "approval_id": "a"}),
                    callback=lambda r: decided.append(r),
                )

        async with Harness().run_test() as pilot:
            await pilot.press(key)
            await pilot.pause()
        assert decided == [expected], key
