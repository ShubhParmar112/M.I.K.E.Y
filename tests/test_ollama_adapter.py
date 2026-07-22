"""Ollama adapter wire-format test via httpx.MockTransport — request translation,
response parsing, the warm-keep/output-cap options, and availability signalling
(offline / model-not-pulled -> ModelUnavailable so the gateway can react)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from core.models.gateway import ChatMessage, ModelUnavailable
from core.models.ollama_adapter import OllamaAdapter

TOOLS = [
    {
        "name": "fs_read",
        "description": "read a file",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
    }
]


def _mock(body: dict[str, Any], captured: dict[str, Any]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


async def test_request_body_and_response_parsing() -> None:
    captured: dict[str, Any] = {}
    adapter = OllamaAdapter(
        "http://localhost:11434", "llama3.2",
        transport=_mock({"message": {"content": "hello there"}}, captured),
    )
    resp = await adapter.complete("be nice", [ChatMessage(role="user", text="hi")], TOOLS)

    body = captured["body"]
    assert body["messages"][0] == {"role": "system", "content": "be nice"}
    assert body["keep_alive"] == "30m"  # stays resident: no cold reloads between turns
    assert body["options"]["num_predict"] == 512  # reply length capped for latency
    assert body["tools"][0]["function"]["name"] == "fs_read"
    assert resp.text == "hello there"
    assert resp.tool_calls == []


async def test_tool_call_parsing() -> None:
    captured: dict[str, Any] = {}
    adapter = OllamaAdapter(
        "http://localhost:11434", "llama3.2",
        transport=_mock(
            {"message": {"content": "",
                         "tool_calls": [{"function": {"name": "fs_read",
                                                      "arguments": {"path": "a.txt"}}}]}},
            captured,
        ),
    )
    resp = await adapter.complete("sys", [ChatMessage(role="user", text="read a.txt")], TOOLS)
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "fs_read"
    assert resp.tool_calls[0].arguments == {"path": "a.txt"}


async def test_offline_becomes_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    adapter = OllamaAdapter("http://localhost:11434", "llama3.2",
                            transport=httpx.MockTransport(handler))
    with pytest.raises(ModelUnavailable) as ei:
        await adapter.complete("sys", [ChatMessage(role="user", text="hi")], [])
    assert "not running" in ei.value.reason


async def test_model_not_pulled_becomes_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    adapter = OllamaAdapter("http://localhost:11434", "llama3.2",
                            transport=httpx.MockTransport(handler))
    with pytest.raises(ModelUnavailable) as ei:
        await adapter.complete("sys", [ChatMessage(role="user", text="hi")], [])
    assert "not pulled" in ei.value.reason
    assert "ollama pull llama3.2" in ei.value.reason
