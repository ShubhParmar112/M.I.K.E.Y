"""Groq adapter — Tier-1 cloud inference over the OpenAI-compatible API.

Groq serves open models (Llama 3.x etc.) with very fast inference and a free
tier. Privacy-wise it is a cloud provider like Anthropic, not a local runtime.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from core.models.gateway import ChatMessage, ModelResponse, ToolCall

BASE_URL = "https://api.groq.com/openai/v1"


class GroqAdapter:
    name = "groq"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._transport = transport  # injectable for tests

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
                msg: dict[str, Any] = {"role": "assistant", "content": m.text or None}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                wire.append(msg)
            elif m.role == "tool_result":
                wire.append(
                    {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.text}
                )

        body: dict[str, Any] = {"model": self._model, "messages": wire}
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
            body["tool_choice"] = "auto"

        async with httpx.AsyncClient(
            timeout=120.0, transport=self._transport
        ) as client:
            resp = await client.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        message = data["choices"][0]["message"]
        tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"]["arguments"] or "{}"),
            )
            for tc in message.get("tool_calls") or []
        ]
        usage = data.get("usage") or {}
        return ModelResponse(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        )
