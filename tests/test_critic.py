"""The Critic / Verifier (sovereignty S1): an independent review of a proposed
action, parsed into a sound/concern verdict, and never allowed to break the turn.
"""

from __future__ import annotations

from core.models.fake_adapter import FakeAdapter
from core.models.gateway import ChatMessage, ModelGateway, ModelResponse
from core.orchestrator.critic import Critic, _parse


def _critic(text: str) -> Critic:
    return Critic(ModelGateway(FakeAdapter([ModelResponse(text=text, tool_calls=[])])))


async def test_ok_verdict_is_sound() -> None:
    c = _critic("OK: writes exactly the file the user asked for")
    v = await c.review(
        user_request="write hello.txt", tool="fs_write",
        args={"path": "hello.txt", "content": "hi"}, tainted=False,
    )
    assert v.sound is True
    assert "asked for" in v.note


async def test_concern_verdict_is_flagged() -> None:
    c = _critic("CONCERN: deletes a memory the user never mentioned")
    v = await c.review(
        user_request="that was it, bye", tool="memory_forget",
        args={"event_id": "01ABC"}, tainted=False,
    )
    assert v.sound is False
    assert "memory" in v.note


async def test_critic_survives_a_dead_model() -> None:
    """A verifier that is down must degrade to 'no second opinion', never crash."""

    class _Boom:
        name = "boom"
        local = False

        async def complete(self, system, messages, tools) -> ModelResponse:  # noqa: ANN001
            raise RuntimeError("provider exploded")

    v = await Critic(ModelGateway(_Boom())).review(
        user_request="x", tool="fs_write", args={}, tainted=False,
    )
    assert v.sound is True
    assert "unavailable" in v.note


def test_parse_defaults_to_no_concern_on_garbage() -> None:
    assert _parse("").sound is True          # empty → advisory pass
    assert _parse("blah blah").sound is True  # unparseable → no false block
    assert _parse("CONCERN: x").sound is False
    assert _parse("ok: fine").sound is True


async def test_tainted_turn_is_flagged_to_the_critic() -> None:
    """When the turn is tainted, the critic's prompt gets the injection warning."""
    fake = FakeAdapter([ModelResponse(text="OK: fine", tool_calls=[])])
    c = Critic(ModelGateway(fake))
    await c.review(user_request="fetch it", tool="web_fetch",
                   args={"url": "http://x"}, tainted=True)
    # the user-message the critic saw carried the untrusted-content warning
    sent: list[ChatMessage] = fake.calls[0]
    assert "UNTRUSTED" in sent[-1].text
