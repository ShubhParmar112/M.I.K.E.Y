"""Groq adapter wire-format test via httpx.MockTransport — verifies request
translation (system, tool schemas, tool results) and response parsing
(text + tool_calls) without touching the network."""

from __future__ import annotations

import json
from typing import Any

import httpx

from core.models.gateway import ChatMessage, ToolCall
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
