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

        body: dict[str, Any] = {"model": self._model, "messages": wire, "temperature": 0.2}
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

        # Llama occasionally emits malformed tool-call syntax, which Groq rejects
        # as 400/tool_use_failed. That's a transient generation failure, not a
        # request error: retry, and on the last attempt force a text-only answer
        # so the turn degrades gracefully instead of dying (failure taxonomy M12).
        async with httpx.AsyncClient(
            timeout=120.0, transport=self._transport
        ) as client:
            last_failed_generation = ""
            for attempt in range(3):
                attempt_body = dict(body)
                if attempt == 2 and "tools" in body:
                    attempt_body["tool_choice"] = "none"
                resp = await client.post(
                    f"{BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=attempt_body,
                )
                if resp.status_code == 400:
                    try:
                        err = resp.json().get("error", {})
                    except ValueError:
                        err = {}
                    if err.get("code") == "tool_use_failed":
                        last_failed_generation = str(err.get("failed_generation", ""))[:300]
                        continue
                    raise RuntimeError(f"groq 400: {err.get('message', resp.text[:300])}")
                resp.raise_for_status()
                data = resp.json()
                break
            else:
                raise RuntimeError(
                    "groq: model repeatedly produced malformed tool calls "
                    f"(tool_use_failed); last generation: {last_failed_generation!r}"
                )

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
