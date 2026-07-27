"""Several free tiers behind one gateway.

The failure being fixed is specific and was lived through: one cloud key, its
daily allowance spent by evening, and every answer after that served by a 3B
local model — with nothing to show for it but worse answers. These tests pin the
three things that make that not happen again: other providers exist, a provider
that says it is finished for the day is not asked again, and the surface can tell
"another cloud model took over" apart from "we are on the weak local one".
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from core.config import Config
from core.cost.governor import ProviderDay, daily_call_cap_for, daily_cap_for
from core.models.cerebras_adapter import CerebrasAdapter
from core.models.gateway import (
    ChatMessage,
    ModelGateway,
    ModelResponse,
    ModelUnavailable,
)
from core.models.gemini_adapter import GeminiAdapter
from core.models.openai_compat import _is_daily_limit

CLOUD_KEYS = ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY", "GEMINI_API_KEY")


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine with no provider keys, so a test says what it configures."""
    for env in (*CLOUD_KEYS, "MIKEY_PROVIDER"):
        monkeypatch.delenv(env, raising=False)


def _capture(reply: dict[str, Any], captured: dict[str, Any]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=reply)

    return httpx.MockTransport(handler)


TEXT_REPLY = {"choices": [{"message": {"content": "hello"}}], "usage": {}}


# --- the new providers speak the shared wire format ---------------------------


async def test_cerebras_posts_to_its_own_endpoint() -> None:
    captured: dict[str, Any] = {}
    adapter = CerebrasAdapter("gpt-oss-120b", api_key="k", transport=_capture(TEXT_REPLY, captured))

    resp = await adapter.complete("sys", [ChatMessage(role="user", text="hi")], [])

    assert captured["url"] == "https://api.cerebras.ai/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer k"
    assert captured["body"]["model"] == "gpt-oss-120b"
    assert resp.text == "hello"


async def test_gemini_posts_to_the_openai_compatibility_endpoint() -> None:
    captured: dict[str, Any] = {}
    adapter = GeminiAdapter("gemini-2.5-flash", api_key="k", transport=_capture(TEXT_REPLY, captured))

    await adapter.complete("sys", [ChatMessage(role="user", text="hi")], [])

    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert captured["headers"]["authorization"] == "Bearer k"


async def test_gemini_resamples_without_penalties() -> None:
    """Google's compatibility layer implements a subset of OpenAI's parameters and
    rejects the rest. A 400 in the middle of recovering a collapsed reply would cost
    the whole turn — so the recovery goes ahead without the penalties."""
    bodies: list[dict[str, Any]] = []
    collapsed = "no it is not 7 and 14. " * 12

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": collapsed}}], "usage": {}}
        )

    adapter = GeminiAdapter("gemini-2.5-flash", api_key="k", transport=httpx.MockTransport(handler))
    await adapter.complete("sys", [ChatMessage(role="user", text="solve it")], [])

    assert len(bodies) == 2, "the collapsed reply should still be re-sampled"
    assert bodies[1]["temperature"] > bodies[0]["temperature"]
    assert "frequency_penalty" not in bodies[1]
    assert "presence_penalty" not in bodies[1]


@pytest.mark.parametrize(
    "message",
    [
        "Rate limit reached ... on tokens per day (TPD): Limit 100000, Used 98372",
        "You exceeded your current quota: GenerateRequestsPerDayPerProjectPerModel",
        "Request limit exceeded: 1M tokens per day",
    ],
)
def test_each_provider_wording_for_a_daily_cap_is_recognised(message: str) -> None:
    """The answer to a daily cap (stop asking today) is the opposite of the answer
    to a per-minute one (wait a moment), so the wording has to be read right for
    every provider in the chain — they all phrase it differently."""
    assert _is_daily_limit(message)


def test_a_per_minute_limit_is_not_mistaken_for_a_daily_one() -> None:
    assert not _is_daily_limit("Rate limit reached ... on tokens per minute (TPM): Limit 12000")


async def test_a_request_too_long_for_one_provider_rolls_to_the_next() -> None:
    """Free tiers cap context well below the model's real window, so a long
    document in memory can be refused by the narrowest link in the chain. That must
    be a reason to try the next provider, not to end the turn."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            400,
            json={"error": {"message": "Please reduce the length of the messages: "
                                       "context_length_exceeded (8192)"}},
        )
    )
    adapter = CerebrasAdapter("gpt-oss-120b", api_key="k", transport=transport)

    with pytest.raises(ModelUnavailable) as exc:
        await adapter.complete("sys", [ChatMessage(role="user", text="a long one")], [])
    assert "context too small" in exc.value.reason
    assert exc.value.daily is False  # it may well answer the very next, shorter turn


async def test_a_genuine_bad_request_is_still_a_hard_error() -> None:
    """A malformed request is a bug in M.I.K.E.Y. Falling back on it would hide the
    bug behind a slightly worse answer from the next provider, every single turn."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(400, json={"error": {"message": "unknown model 'nope'"}})
    )
    adapter = CerebrasAdapter("nope", api_key="k", transport=transport)

    with pytest.raises(RuntimeError, match="unknown model"):
        await adapter.complete("sys", [ChatMessage(role="user", text="hi")], [])


# --- the gateway stops asking a provider that is out for the day --------------


class _Fake:
    def __init__(
        self, name: str, response: ModelResponse | None = None, error: Exception | None = None
    ) -> None:
        self.name = name
        self._response = response
        self._error = error
        self.calls = 0

    async def complete(self, system, messages, tools) -> ModelResponse:  # noqa: ANN001
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def _msgs() -> list[ChatMessage]:
    return [ChatMessage(role="user", text="hi")]


async def test_a_daily_exhausted_provider_is_not_asked_again() -> None:
    """A turn makes several model calls. Without this, every one of them pays a
    failed round-trip to a provider that already said it has nothing left today."""
    spent = _Fake("groq", error=ModelUnavailable("groq", "tokens per day (TPD)", daily=True))
    second = _Fake("cerebras", ModelResponse(text="still good", tool_calls=[]))
    gw = ModelGateway(spent, fallbacks=[second])

    for _ in range(3):
        resp = await gw.complete("", _msgs(), [])
        assert resp.text == "still good"

    assert spent.calls == 1, "asked once, then left alone"
    assert second.calls == 3
    assert "groq" in gw.sidelined


async def test_a_per_minute_limit_does_not_sideline_a_provider() -> None:
    """A spike clears in seconds; sidelining for it would hand the rest of the
    conversation to a weaker model for no reason."""
    flaky = _Fake("groq", error=ModelUnavailable("groq", "rate limited (429)"))
    second = _Fake("cerebras", ModelResponse(text="ok", tool_calls=[]))
    gw = ModelGateway(flaky, fallbacks=[second])

    await gw.complete("", _msgs(), [])
    await gw.complete("", _msgs(), [])

    assert flaky.calls == 2
    assert gw.sidelined == {}


async def test_sidelining_never_leaves_the_gateway_with_nothing() -> None:
    """Our note about a provider can be stale — a quota window may have rolled over
    early. A stale note must never be the reason a turn fails outright."""
    spent = _Fake("groq", error=ModelUnavailable("groq", "tokens per day", daily=True))
    gw = ModelGateway(spent)

    with pytest.raises(RuntimeError):
        await gw.complete("", _msgs(), [])
    assert "groq" in gw.sidelined

    spent._error = None
    spent._response = ModelResponse(text="quota reset", tool_calls=[])
    resp = await gw.complete("", _msgs(), [])

    assert resp.text == "quota reset"
    assert gw.sidelined == {}, "a provider that answers is no longer sidelined"


async def test_the_local_model_is_only_reached_when_every_cloud_is_out() -> None:
    a = _Fake("groq", error=ModelUnavailable("groq", "tokens per day", daily=True))
    b = _Fake("cerebras", error=ModelUnavailable("cerebras", "rate limited (429)"))
    local = _Fake("ollama", ModelResponse(text="from local", tool_calls=[]))
    gw = ModelGateway(a, fallbacks=[b, local])

    resp = await gw.complete("", _msgs(), [])

    assert resp.text == "from local"
    assert gw.last_provider == "ollama"
    assert gw.last_fallback_daily is True  # the surface can say "out of quota", truthfully


# --- the chain is built from whatever keys are present ------------------------


def test_a_single_key_gives_a_cloud_primary_and_a_local_net(
    no_keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.gateway.app import _make_fallbacks

    monkeypatch.setenv("GROQ_API_KEY", "x")
    config = Config()

    assert config.provider == "groq"
    assert [a.name for a in _make_fallbacks(config)] == ["ollama"]


def test_every_extra_key_becomes_a_link_before_the_local_model(
    no_keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shape that removes the cliff: two more cloud providers between the
    primary running out and the 3B model."""
    from core.gateway.app import _make_fallbacks

    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("CEREBRAS_API_KEY", "y")
    monkeypatch.setenv("GEMINI_API_KEY", "z")
    config = Config()

    assert config.provider == "groq"
    assert [a.name for a in _make_fallbacks(config)] == ["cerebras", "gemini", "ollama"]


def test_a_named_provider_wins_and_is_not_duplicated_in_the_chain(
    no_keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.gateway.app import _make_fallbacks

    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "z")
    monkeypatch.setenv("MIKEY_PROVIDER", "gemini")
    config = Config()

    assert config.provider == "gemini"
    assert [a.name for a in _make_fallbacks(config)] == ["groq", "ollama"]


# --- the gauge knows each free tier is metered differently --------------------


def test_cerebras_has_its_own_daily_token_allowance() -> None:
    assert daily_cap_for("cerebras") == 1_000_000
    assert daily_cap_for("groq") == 100_000


def test_a_request_metered_provider_is_gauged_by_requests() -> None:
    """Google counts requests per day, not tokens. Gauged on tokens it would read
    0% right up to the moment it stopped answering."""
    assert daily_call_cap_for("gemini") > 0

    day = ProviderDay(
        provider="gemini",
        calls=daily_call_cap_for("gemini"),
        input_tokens=5_000,
        output_tokens=500,
        cap=0,
        call_cap=daily_call_cap_for("gemini"),
    )

    assert day.capped
    assert day.metered_by == "calls"
    assert day.exhausted
    assert day.calls_left == 0
