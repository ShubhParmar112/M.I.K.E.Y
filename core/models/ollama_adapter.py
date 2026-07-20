"""Ollama adapter — Tier-0 local inference (no data leaves the machine)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from core.models.gateway import ChatMessage, ModelResponse, ToolCall
from core.events.schema import ulid


class OllamaAdapter:
    name = "ollama"

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "user":
                wire.append({"role": "user", "content": m.text})
            elif m.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.text}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {"function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in m.tool_calls
                    ]
                wire.append(msg)
            elif m.role == "tool_result":
                wire.append({"role": "tool", "content": m.text})

        body: dict[str, Any] = {"model": self._model, "messages": wire, "stream": False}
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=body)
            resp.raise_for_status()
            data = resp.json()

        message = data.get("message", {})
        tool_calls = [
            ToolCall(
                id=ulid(),
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"]
                if isinstance(tc["function"]["arguments"], dict)
                else json.loads(tc["function"]["arguments"]),
            )
            for tc in message.get("tool_calls", [])
        ]
        return ModelResponse(text=message.get("content", ""), tool_calls=tool_calls)
