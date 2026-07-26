"""Independent verification of reasoning answers.

From a live session (2026-07-27, served by the local 3B) on "A beats B by 20 m,
B beats C by 10 m — by how much does A beat C?":

    ...200 - 191.8 = 8.2 meters
    However, I made another mistake earlier.
    Let me re-evaluate again...
     Ah-ha! I found it!
    The correct answer is indeed: 29

29 is the right answer. It was recalled, not derived — which is why the reply could
not survive "how did you get 29?". Neither existing guard fires on it, and the
reasoning brain's own self-check is what produced "Ah-ha! I found it!" in the first
place. So: detect the shape, then hand it to a *different* brain to check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import Config
from core.events.store import EventStore
from core.memory.store import MemoryStore
from core.models.degeneration import collapse_score, trim_restart, unsupported_answer
from core.models.fake_adapter import FakeAdapter
from core.models.gateway import ModelGateway, ModelResponse
from core.orchestrator.critic import Critic
from core.orchestrator.loop import ApprovalRegistry, Orchestrator
from core.policy.engine import PolicyEngine
from core.storage.db import Database
from core.trace.store import TraceStore

THE_LIVE_REPLY = """I made a mistake in my previous calculation. Let's go back to the beginning:

We know that A beats B by 20 meters and B beats C by 10 meters.

When A finishes the race (200 meters), B has covered 180 meters.

Since B covers 190 meters when C finishes, we can set up a proportion:

(190 / x) = (180 / 200)

x = 191.8

200 - 191.8 = 8.2 meters

However, I made another mistake earlier.

Let me re-evaluate again...

 Ah-ha! I found it!

The correct answer is indeed: 29"""

A_REAL_DERIVATION = """When A runs 200 m, B runs 180 m, so B/A = 180/200 = 0.9.
When B runs 200 m, C runs 190 m, so C/B = 190/200 = 0.95.
When A finishes 200 m, C has run 200 * 0.9 * 0.95 = 171 m.
So A beats C by 200 - 171 = 29 metres."""


# ---- the detector ----------------------------------------------------------


def test_the_live_reply_slips_past_both_existing_guards() -> None:
    """Pinned so the gap this closes stays visible: it is not a repetition collapse,
    and trimming it at the restart would have kept the WRONG answer (8.2)."""
    assert collapse_score(THE_LIVE_REPLY) == 0.0
    assert trim_restart(THE_LIVE_REPLY)[1] is False


def test_an_asserted_answer_is_flagged() -> None:
    assert unsupported_answer(THE_LIVE_REPLY) is True


def test_a_real_derivation_is_not_flagged() -> None:
    assert unsupported_answer(A_REAL_DERIVATION) is False


def test_a_correction_that_shows_its_working_is_not_flagged() -> None:
    """Retracting is fine — retracting and then asserting is not. This is the
    false positive that would make the check useless if it fired."""
    text = (
        "I made a mistake above: I used 190/200 where it should have been 180/200.\n"
        "Redoing it: 200 * 0.9 * 0.95 = 171, so the answer is 200 - 171 = 29 metres."
    )
    assert unsupported_answer(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "The answer is 29 metres.",  # no retraction at all
        "Let me re-evaluate. Actually 3 * 4 = 12, so the final answer is 12.",
        "```\nlet me try again\nthe answer is 5\n```",  # fenced code is exempt
    ],
)
def test_quiet_on_healthy_replies(text: str) -> None:
    assert unsupported_answer(text) is False


# ---- the verifier in the turn loop -----------------------------------------


QUESTION = "In a 200 m race A beats B by 20 m, and B beats C by 10 m. By how much does A beat C?"


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MIKEY_WORKSPACE", raising=False)
    monkeypatch.setenv("MIKEY_HOME", str(tmp_path))
    config = Config(home=tmp_path)
    config.ensure_dirs()
    return config, Database(config.db_path)


def _orchestrator(config: Config, db: Database, script: list[ModelResponse], verifier_says):
    """Turn loop with a stubbed verifier, so the test pins OUR logic rather than a
    model's judgement."""
    memory = MemoryStore(db, EventStore(db))
    traces = TraceStore(db)
    gateway = ModelGateway(FakeAdapter(script))

    class _Critic(Critic):
        def __init__(self) -> None:
            super().__init__(gateway)
            self.seen: list[str] = []

        async def verify_answer(self, *, question, answer, tier=None):  # type: ignore[override]
            self.seen.append(answer)
            return verifier_says(answer)

    critic = _Critic()
    orch = Orchestrator(
        config, memory, traces, PolicyEngine(db), gateway, None, ApprovalRegistry(), critic=critic
    )
    return orch, critic, traces


async def _final(orch: Orchestrator) -> dict:
    out: dict = {}
    async for ev in orch.run_turn("s1", QUESTION):
        if ev.kind == "final":
            out = ev.data
    return out


async def test_an_asserted_answer_is_re_derived_before_the_person_sees_it(env) -> None:
    from core.orchestrator.critic import Verdict

    config, db = env
    script = [
        ModelResponse(text=THE_LIVE_REPLY, tool_calls=[]),
        ModelResponse(text=A_REAL_DERIVATION, tool_calls=[]),  # the re-derivation
    ]
    orch, critic, traces = _orchestrator(
        config, db, script,
        lambda answer: Verdict(sound="171" in answer, note="29 is asserted, never computed."),
    )
    final = await _final(orch)

    assert final["verified"] is True and final["re_derived"] is True
    assert "171" in final["text"], "the person should get a derivation, not an assertion"
    assert "Ah-ha" not in final["text"]
    assert len(critic.seen) == 2  # checked, re-derived, checked again
    kinds = [s["kind"] for s in traces.turn(final["turn_id"])]
    assert kinds.count("verify_answer") == 2  # the whole exchange is in the trace


async def test_an_answer_that_still_cannot_be_verified_says_so(env) -> None:
    """The honest outcome. A number nobody could check, presented as certain, is the
    failure being fixed — so when the check keeps failing, that has to reach the user."""
    from core.orchestrator.critic import Verdict

    config, db = env
    script = [
        ModelResponse(text=THE_LIVE_REPLY, tool_calls=[]),
        ModelResponse(text="Still not sure. The correct answer is indeed: 29", tool_calls=[]),
    ]
    orch, _critic, _traces = _orchestrator(
        config, db, script, lambda answer: Verdict(sound=False, note="No arithmetic produces 29.")
    )
    final = await _final(orch)

    assert final["verified"] is False
    assert "could not confirm" in final["text"]
    assert "No arithmetic produces 29." in final["text"]


async def test_a_healthy_derivation_costs_no_extra_call(env) -> None:
    from core.orchestrator.critic import Verdict

    config, db = env
    orch, critic, traces = _orchestrator(
        config, db, [ModelResponse(text=A_REAL_DERIVATION, tool_calls=[])],
        lambda answer: Verdict(sound=True, note="fine"),
    )
    final = await _final(orch)

    assert critic.seen == [], "an unflagged answer must not spend a verification call"
    assert final["text"] == A_REAL_DERIVATION
    assert "verify_answer" not in [s["kind"] for s in traces.turn(final["turn_id"])]


async def test_a_verifier_that_gives_no_verdict_is_not_treated_as_confirmation(env) -> None:
    """Found by running the real 3B: asked to check a wildly wrong answer it never
    emitted `OK:`/`CONCERN:` at all, and the default-to-sound parse reported that as
    `verified=True`. A verifier too weak to be understood rubber-stamps everything,
    which is worse than no verifier — so "no usable verdict" must read as unchecked.
    """
    from core.orchestrator.critic import _parse

    rambling = _parse("Let me work through this. x = 192.5 metres, I think that's it.")
    assert rambling.parsed is False, "an answer with no verdict line is not a verdict"

    config, db = env
    orch, _critic, _traces = _orchestrator(
        config, db, [ModelResponse(text=THE_LIVE_REPLY, tool_calls=[])],
        lambda answer: rambling,
    )
    final = await _final(orch)

    assert final["verified"] is None, "unchecked must never be reported as verified"
    assert "could not get an independent check" in final["text"]


def test_an_explicit_ok_still_confirms() -> None:
    from core.orchestrator.critic import _parse

    verdict = _parse("OK: 200 * 0.9 * 0.95 = 171, so 29 is right and the working shows it.")
    assert verdict.parsed is True and verdict.sound is True


def test_the_action_critic_stays_fail_open() -> None:
    """The action critic is advisory and must never turn a garbled reply into a false
    block — only the answer-verification path changes behaviour on an unparsed verdict."""
    from core.orchestrator.critic import _parse

    assert _parse("uh, sure, whatever you think").sound is True


FALSE_VERIFICATION = """Let M be the Marked Price.
M = 34,240
Verify: (0.92 × 34,240) = Rs. 17,940 (True)
The profit with no discount would be 20%."""


async def test_provably_wrong_arithmetic_re_derives_without_asking_a_model(env) -> None:
    """Arithmetic that does not hold is proof, not opinion. Asking the model that
    just wrote `0.92 x 34,240 = 17,940` whether it is right is the one judgement we
    already know we cannot trust — so the verifier call is skipped entirely."""
    from core.orchestrator.critic import Verdict

    config, db = env
    script = [
        ModelResponse(text=FALSE_VERIFICATION, tool_calls=[]),
        ModelResponse(text="Cost price = 17940 / 1.196 = 15000, marked = 17940 / 0.92 = 19500, "
                           "so profit = (4500 / 15000) * 100 = 30%", tool_calls=[]),
    ]
    orch, critic, traces = _orchestrator(
        config, db, script, lambda answer: Verdict(sound=True, note="ok")
    )
    final = await _final(orch)

    assert "30%" in final["text"] and final["re_derived"] is True
    assert len(critic.seen) == 1, "only the RETRY is worth a verifier call, not the proven-bad one"
    kinds = [s["kind"] for s in traces.turn(final["turn_id"])]
    assert "arithmetic_audit" in kinds


async def test_a_retry_that_is_still_wrong_ships_with_the_false_step_named(env) -> None:
    from core.orchestrator.critic import Verdict

    config, db = env
    orch, critic, _traces = _orchestrator(
        config, db,
        [ModelResponse(text=FALSE_VERIFICATION, tool_calls=[]),
         ModelResponse(text="Sorry — (0.92 × 34,240) = Rs. 17,940 still. It is 20%.",
                       tool_calls=[])],
        lambda answer: Verdict(sound=True, note="ok"),
    )
    final = await _final(orch)

    assert final["verified"] is False
    assert "arithmetic that does not hold" in final["text"]
    assert "31,500.80" in final["text"]  # the person is told what it actually equals
    assert critic.seen == [], "a second provably-wrong answer needs no second opinion either"


async def test_verification_can_be_turned_off(env, monkeypatch) -> None:
    from core.orchestrator.critic import Verdict

    config, db = env
    monkeypatch.setenv("MIKEY_VERIFY_REASONING", "off")
    config = Config(home=config.home)
    orch, critic, _traces = _orchestrator(
        config, db, [ModelResponse(text=THE_LIVE_REPLY, tool_calls=[])],
        lambda answer: Verdict(sound=False, note="nope"),
    )
    final = await _final(orch)

    assert critic.seen == []
    assert final["text"] == THE_LIVE_REPLY  # exactly the old behaviour
