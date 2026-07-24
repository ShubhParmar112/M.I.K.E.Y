"""Anthropic adapter — Tier-1 cloud inference."""

from __future__ import annotations

from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from core.models.gateway import ChatMessage, ModelResponse, ModelUnavailable, ToolCall


def _retry_after(exc: anthropic.APIStatusError) -> float | None:
    try:
        raw = exc.response.headers.get("retry-after")
        return float(raw) if raw else None
    except (AttributeError, TypeError, ValueError):
        return None


class AnthropicAdapter:
    name = "anthropic"
    local = False  # cloud provider; never eligible to serve Tier-0 data

    def __init__(self, model: str, client: AsyncAnthropic | None = None) -> None:
        self._client = client or AsyncAnthropic()  # reads ANTHROPIC_API_KEY
        self._model = model

    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        wire: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "user":
                wire.append({"role": "user", "content": m.text})
            elif m.role == "assistant":
                content: list[dict[str, Any]] = []
                if m.text:
                    content.append({"type": "text", "text": m.text})
                for tc in m.tool_calls:
                    content.append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    )
                wire.append({"role": "assistant", "content": content})
            elif m.role == "tool_result":
                wire.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.text,
                            }
                        ],
                    }
                )

        anthropic_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in tools
        ]

        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system,
                messages=wire,
                tools=anthropic_tools or None,
            )
        except anthropic.APIConnectionError as exc:
            # offline / unreachable — a fallback provider can still answer
            raise ModelUnavailable("anthropic", f"unreachable ({type(exc).__name__})") from exc
        except anthropic.RateLimitError as exc:
            raise ModelUnavailable("anthropic", "rate limited (429)", _retry_after(exc)) from exc
        except anthropic.APIStatusError as exc:
            # 5xx / 529-overloaded are what the fallback exists for; auth/4xx are
            # real bugs and must not be masked by silently switching providers.
            if exc.status_code >= 500:
                raise ModelUnavailable("anthropic", f"server error ({exc.status_code})") from exc
            raise

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        )
