"""Groq adapter wire-format test via httpx.MockTransport — verifies request
translation (system, tool schemas, tool results) and response parsing
(text + tool_calls) without touching the network."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from core.models.gateway import ChatMessage, ModelUnavailable, ToolCall
from core.models.groq_adapter import GroqAdapter

TOOLS = [
    {
        "name": "fs_read",
        "description": "read a file",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
]


def _mock(response_body: dict[str, Any], captured: dict[str, Any]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=response_body)

    return httpx.MockTransport(handler)


TOOL_USE_FAILED = {
    "error": {
        "code": "tool_use_failed",
        "message": "Failed to call a function.",
        "failed_generation": '<function=fs_read[]{"path": "C:\\\\hosts"}</function>',
    }
}


def _flaky(fail_times: int, requests: list[dict[str, Any]]) -> httpx.MockTransport:
    """400/tool_use_failed for the first `fail_times` calls, then 200."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) <= fail_times:
            return httpx.Response(400, json=TOOL_USE_FAILED)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "recovered"}}], "usage": {}}
        )

    return httpx.MockTransport(handler)


async def test_request_translation_and_tool_call_parsing() -> None:
    captured: dict[str, Any] = {}
    adapter = GroqAdapter(
        model="llama-3.3-70b-versatile",
        api_key="test-key",
        transport=_mock(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "fs_read",
                                        "arguments": '{"path": "a.txt"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            captured,
        ),
    )

    resp = await adapter.complete("be helpful", [ChatMessage(role="user", text="read a.txt")], TOOLS)

    body = captured["body"]
    assert body["messages"][0] == {"role": "system", "content": "be helpful"}
    assert body["messages"][1] == {"role": "user", "content": "read a.txt"}
    assert body["tools"][0]["function"]["name"] == "fs_read"
    assert body["tools"][0]["function"]["parameters"]["required"] == ["path"]
    assert captured["headers"]["authorization"] == "Bearer test-key"

    assert resp.tool_calls == [ToolCall(id="call_1", name="fs_read", arguments={"path": "a.txt"})]
    assert resp.usage == {"input_tokens": 10, "output_tokens": 5}


async def test_tool_result_roundtrip_and_text_response() -> None:
    captured: dict[str, Any] = {}
    adapter = GroqAdapter(
        model="m",
        api_key="k",
        transport=_mock(
            {"choices": [{"message": {"content": "The file says hi."}}], "usage": {}},
            captured,
        ),
    )
    messages = [
        ChatMessage(role="user", text="read a.txt"),
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="call_1", name="fs_read", arguments={"path": "a.txt"})],
        ),
        ChatMessage(role="tool_result", text="hi", tool_call_id="call_1"),
    ]
    resp = await adapter.complete("sys", messages, TOOLS)

    wire = captured["body"]["messages"]
    assert wire[2]["tool_calls"][0]["function"]["arguments"] == '{"path": "a.txt"}'
    assert wire[3] == {"role": "tool", "tool_call_id": "call_1", "content": "hi"}
    assert resp.text == "The file says hi."
    assert resp.tool_calls == []


async def test_inline_function_tag_is_recovered_as_tool_call() -> None:
    """Llama sometimes returns a tool call as literal text in `content` instead of
    a structured tool_call. The adapter must recover it (so it executes) and strip
    it from the visible text (so the user never sees the raw tag)."""
    captured: dict[str, Any] = {}
    content = (
        'Sure, let me look that up. '
        '<function=memory_recall>{"query": "dog name", "k": "6"}</function>'
    )
    adapter = GroqAdapter(
        model="m", api_key="k",
        transport=_mock({"choices": [{"message": {"content": content}}], "usage": {}}, captured),
    )
    resp = await adapter.complete("sys", [ChatMessage(role="user", text="hi")], TOOLS)

    assert resp.text == "Sure, let me look that up."  # tag stripped from the reply
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "memory_recall"
    assert resp.tool_calls[0].arguments == {"query": "dog name", "k": "6"}


async def test_structured_tool_calls_win_over_inline_text() -> None:
    """If Groq returns a real structured tool_call, that is authoritative even if
    the content also happens to contain a function-like tag."""
    captured: dict[str, Any] = {}
    adapter = GroqAdapter(
        model="m", api_key="k",
        transport=_mock(
            {
                "choices": [{"message": {
                    "content": "<function=ignored>{}</function>",
                    "tool_calls": [{"id": "c1", "type": "function",
                                    "function": {"name": "fs_read",
                                                 "arguments": '{"path": "a.txt"}'}}],
                }}],
                "usage": {},
            },
            captured,
        ),
    )
    resp = await adapter.complete("sys", [ChatMessage(role="user", text="hi")], TOOLS)
    assert [tc.name for tc in resp.tool_calls] == ["fs_read"]


async def test_tool_use_failed_is_retried() -> None:
    requests: list[dict[str, Any]] = []
    adapter = GroqAdapter(model="m", api_key="k", transport=_flaky(1, requests))
    resp = await adapter.complete("sys", [ChatMessage(role="user", text="hi")], TOOLS)
    assert resp.text == "recovered"
    assert len(requests) == 2
    assert requests[1]["tool_choice"] == "auto"  # plain retry first


async def test_final_retry_degrades_to_text_only() -> None:
    requests: list[dict[str, Any]] = []
    adapter = GroqAdapter(model="m", api_key="k", transport=_flaky(2, requests))
    resp = await adapter.complete("sys", [ChatMessage(role="user", text="hi")], TOOLS)
    assert resp.text == "recovered"
    assert len(requests) == 3
    assert requests[2]["tool_choice"] == "none"  # graceful degradation to text


async def test_persistent_tool_use_failure_raises_with_detail() -> None:
    requests: list[dict[str, Any]] = []
    adapter = GroqAdapter(model="m", api_key="k", transport=_flaky(99, requests))
    try:
        await adapter.complete("sys", [ChatMessage(role="user", text="hi")], TOOLS)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "tool_use_failed" in str(exc)
        assert "fs_read" in str(exc)  # failed_generation surfaced for debugging
    assert len(requests) == 3


# --- repetition-collapse re-sample -------------------------------------------


def _sequence(bodies: list[dict[str, Any]], contents: list[str]) -> httpx.MockTransport:
    """Return `contents[i]` for the i-th request, recording each request body."""

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        text = contents[min(len(bodies) - 1, len(contents) - 1)]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": text}}], "usage": {}}
        )

    return httpx.MockTransport(handler)


def _reply(bodies: list[dict[str, Any]], content: str) -> httpx.MockTransport:
    return _sequence(bodies, [content])


COLLAPSED = (
    "After reevaluating the problem, I think the correct solution is indeed 7 and 14 is not "
    "correct, but 12 and 24 is also not correct, the correct answer is actually 7 is not correct "
    "but 12 is not correct, the correct answer is 7 is not the answer but one of the answer is 12 "
    "is not correct but 7 is not the answer the correct answer is actually 12 is not correct but "
    "one of the answer is 7 is not correct but the correct answer is actually 35 and 71 and 7 and "
    "14 is not the answer, 12 and 24 is not the answer but 7 and 14 is not correct."
)
CLEAN = "Both conditions check out: 36:72 = 1:2 and 30:66 = 5:11, so the numbers are 35 and 71."


async def test_request_is_capped_and_uses_configured_temperature() -> None:
    """An unbounded reply is what let the live repetition loop run for a full
    paragraph; every request now carries a ceiling."""
    bodies: list[dict[str, Any]] = []
    adapter = GroqAdapter(
        "llama-3.3-70b-versatile",
        api_key="k",
        transport=_reply(bodies, "hi"),
        temperature=0.35,
        max_output_tokens=900,
    )
    await adapter.complete("sys", [ChatMessage(role="user", text="hey")], [])

    assert bodies[0]["max_tokens"] == 900
    assert bodies[0]["temperature"] == 0.35
    # Penalties must NOT ride on a normal request: they also penalize the repeated
    # punctuation of well-formed JSON tool arguments.
    assert "frequency_penalty" not in bodies[0]


async def test_a_collapsed_reply_is_resampled_and_replaced() -> None:
    bodies: list[dict[str, Any]] = []
    adapter = GroqAdapter(
        "llama-3.3-70b-versatile",
        api_key="k",
        transport=_sequence(bodies, [COLLAPSED, CLEAN]),
    )
    resp = await adapter.complete("sys", [ChatMessage(role="user", text="solve it")], TOOLS)

    assert resp.text == CLEAN
    assert len(bodies) == 2, "the collapsed generation should have been re-sampled once"
    # The re-sample leans the other way on purpose: low temperature is what sharpens
    # a repetition loop in the first place.
    assert bodies[1]["temperature"] > bodies[0]["temperature"]
    assert bodies[1]["frequency_penalty"] > 0
    assert bodies[1]["tool_choice"] == "none"  # we already know this turn ends in prose
    assert any("re-sampled" in n for n in resp.notes), "the recovery must be traceable"


async def test_a_healthy_reply_is_never_resampled() -> None:
    """One model call for a good answer — the guard must not double every turn's cost."""
    bodies: list[dict[str, Any]] = []
    adapter = GroqAdapter("m", api_key="k", transport=_reply(bodies, CLEAN))
    resp = await adapter.complete("sys", [ChatMessage(role="user", text="solve it")], [])

    assert len(bodies) == 1
    assert resp.text == CLEAN
    assert resp.notes == []


async def test_a_collapsed_resample_keeps_whichever_is_better() -> None:
    """A re-sample can collapse too; the adapter keeps the less repetitive of the two
    rather than trusting the second call blindly."""
    bodies: list[dict[str, Any]] = []
    worse = "no it is not 7 and 14. " * 12
    adapter = GroqAdapter("m", api_key="k", transport=_sequence(bodies, [COLLAPSED, worse]))
    resp = await adapter.complete("sys", [ChatMessage(role="user", text="solve it")], [])

    assert len(bodies) == 2
    assert resp.text == COLLAPSED, "kept the original: the re-sample looped harder"


async def test_a_tool_call_is_never_resampled() -> None:
    """Tool calls are structured output judged by the API, not prose to score."""
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": COLLAPSED,  # rambling text alongside the call
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "fs_read",
                                        "arguments": '{"path": "notes.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {},
            },
        )

    adapter = GroqAdapter("m", api_key="k", transport=httpx.MockTransport(handler))
    resp = await adapter.complete("sys", [ChatMessage(role="user", text="read it")], TOOLS)

    assert len(bodies) == 1
    assert [tc.name for tc in resp.tool_calls] == ["fs_read"]


async def test_an_unavailable_resample_degrades_to_the_original() -> None:
    """A repetitive answer still beats failing the turn outright."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": COLLAPSED}}], "usage": {}}
            )
        return httpx.Response(429, headers={"retry-after": "30"})

    adapter = GroqAdapter(
        "m", api_key="k", transport=httpx.MockTransport(handler), rate_limit_retries=0
    )
    resp = await adapter.complete("sys", [ChatMessage(role="user", text="solve it")], [])

    assert resp.text == COLLAPSED
    assert any("re-sample unavailable" in n for n in resp.notes)


ANSWER_THEN_RESTART = (
    "Let the two numbers be x and y. From (x + 1) : (y + 1) = 1 : 2 we get y = 2x + 1, and from "
    "(x - 5) : (y - 5) = 5 : 11 we get 11x - 5y = 30. Substituting gives x = 35 and y = 71.\n"
    "VERIFICATION: 36 : 72 = 1 : 2 and 30 : 66 = 5 : 11. Both conditions hold.\n"
    "The final answer is: the two numbers are 35 and 71.\n"
    "However we should also check another pair of numbers, let's try to solve the system again.\n"
    "Let's try x = 12 and y = 24: (12 + 1) / (24 + 1) = 13 / 25, which is not 1 / 2.\n"
    "But what about x = 6 and y = 12: (6 + 1) / (12 + 1) = 7 / 13, which is not 1 / 2."
)


async def test_a_restart_after_the_answer_is_trimmed_without_a_second_call() -> None:
    """The reply already contains a correct, verified answer — it just didn't stop.
    Cutting the tail is the fix; re-sampling it is not (live, that produced a second
    runaway), so this must cost only the one call."""
    bodies: list[dict[str, Any]] = []
    adapter = GroqAdapter("m", api_key="k", transport=_reply(bodies, ANSWER_THEN_RESTART))
    resp = await adapter.complete("sys", [ChatMessage(role="user", text="solve it")], [])

    assert len(bodies) == 1, "trimming must not trigger a re-sample"
    assert resp.text.endswith("the two numbers are 35 and 71.")
    assert "12 and 24" not in resp.text
    assert any("trimmed a restart" in n for n in resp.notes)


async def test_a_healthy_reply_is_not_trimmed() -> None:
    bodies: list[dict[str, Any]] = []
    adapter = GroqAdapter("m", api_key="k", transport=_reply(bodies, CLEAN))
    resp = await adapter.complete("sys", [ChatMessage(role="user", text="solve it")], [])

    assert resp.text == CLEAN
    assert resp.notes == []


TPD_EXHAUSTED = {
    "error": {
        "message": (
            "Rate limit reached for model `llama-3.3-70b-versatile` in organization "
            "`org_abc` service tier `on_demand` on tokens per day (TPD): Limit 100000, "
            "Used 96999, Requested 4294. Please try again in 18m37.152s. Need more "
            "tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing"
        ),
        "type": "tokens",
        "code": "rate_limit_exceeded",
    }
}


async def test_a_429_carries_which_limit_was_hit() -> None:
    """A per-minute spike clears in seconds; a per-day cap means the key is done until
    the window rolls over. Collapsing both to "rate limited (429)" hides the one fact
    that decides what to do next — it sent this session chasing a pacing fix for a
    daily-quota exhaustion."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(429, json=TPD_EXHAUSTED, headers={"retry-after": "1118"})
    )
    adapter = GroqAdapter("m", api_key="k", transport=transport, rate_limit_retries=0)

    with pytest.raises(ModelUnavailable) as exc:
        await adapter.complete("sys", [ChatMessage(role="user", text="hi")], [])

    assert "tokens per day (TPD)" in exc.value.reason
    assert "Limit 100000, Used 96999" in exc.value.reason
    assert exc.value.retry_after == 1118.0
    assert "Upgrade to Dev Tier" not in exc.value.reason, "the upsell is not diagnosis"


async def test_a_429_without_a_body_still_reports_cleanly() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(429, text="nope"))
    adapter = GroqAdapter("m", api_key="k", transport=transport, rate_limit_retries=0)

    with pytest.raises(ModelUnavailable) as exc:
        await adapter.complete("sys", [ChatMessage(role="user", text="hi")], [])
    assert exc.value.reason == "rate limited (429)"
