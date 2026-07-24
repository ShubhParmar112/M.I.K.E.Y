"""Groq adapter — Tier-1 cloud inference over the OpenAI-compatible API.

Groq serves open models (Llama 3.x etc.) with very fast inference and a free
tier. Privacy-wise it is a cloud provider like Anthropic, not a local runtime.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import httpx

from core.events.schema import ulid
from core.models.gateway import ChatMessage, ModelResponse, ModelUnavailable, ToolCall

BASE_URL = "https://api.groq.com/openai/v1"
# The local fallback is a weak 3B model that hallucinates, so it's worth waiting
# out a per-minute rate spike on the strong cloud model rather than conceding fast.
MAX_RATE_LIMIT_BACKOFF_S = 15.0

# Llama on Groq sometimes emits a tool call as literal text inside the message
# content — `<function=name>{json args}</function>` — instead of a structured
# tool_call. Left alone it shows up as garbage in the reply AND never executes.
_INLINE_CALL_RE = re.compile(
    r"<function=([A-Za-z0-9_]+)\s*>\s*(\{.*?\})\s*(?:</function>)?", re.DOTALL
)


def _parse_inline_tool_calls(content: str) -> tuple[str, list[ToolCall]]:
    """Recover inline `<function=...>` calls from content and strip them from the
    visible text. Malformed calls are dropped (not fired) but still stripped."""
    calls: list[ToolCall] = []

    def _take(m: re.Match[str]) -> str:
        try:
            args = json.loads(m.group(2))
        except ValueError:
            return ""
        if isinstance(args, dict):
            calls.append(ToolCall(id=ulid(), name=m.group(1), arguments=args))
        return ""

    return _INLINE_CALL_RE.sub(_take, content).strip(), calls


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class GroqAdapter:
    name = "groq"
    local = False  # cloud provider; never eligible to serve Tier-0 data

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        rate_limit_retries: int = 4,
        rate_limit_backoff_s: float = 2.0,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._transport = transport  # injectable for tests
        self._rl_retries = rate_limit_retries
        self._rl_backoff = rate_limit_backoff_s

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
                # Post, with a short backoff on 429 before conceding to the
                # fallback: a per-minute spike usually clears in a second or two,
                # which beats dropping onto a much slower local model.
                for rl in range(self._rl_retries + 1):
                    try:
                        resp = await client.post(
                            f"{BASE_URL}/chat/completions",
                            headers={"Authorization": f"Bearer {self._api_key}"},
                            json=attempt_body,
                        )
                    except httpx.TransportError as exc:
                        # DNS/connect/timeout ~ offline or Groq unreachable. A local
                        # fallback can still answer, so signal it rather than dying.
                        raise ModelUnavailable(
                            "groq", f"unreachable ({type(exc).__name__})"
                        ) from exc
                    if resp.status_code == 429 and rl < self._rl_retries:
                        delay = min(
                            _retry_after(resp) or self._rl_backoff * (rl + 1),
                            MAX_RATE_LIMIT_BACKOFF_S,
                        )
                        await asyncio.sleep(delay)
                        continue
                    break
                # Rate limit (after backoff) and server errors are what the local
                # fallback exists for; auth/other 4xx are not (that hides a real bug).
                if resp.status_code == 429:
                    raise ModelUnavailable(
                        "groq", "rate limited (429)", retry_after=_retry_after(resp)
                    )
                if resp.status_code >= 500:
                    raise ModelUnavailable("groq", f"server error ({resp.status_code})")
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
        content = message.get("content") or ""
        tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"]["arguments"] or "{}"),
            )
            for tc in message.get("tool_calls") or []
        ]
        # Only reach for the text form when Groq gave us no structured calls —
        # the structured field is authoritative when present.
        if not tool_calls and "<function=" in content:
            content, tool_calls = _parse_inline_tool_calls(content)
        usage = data.get("usage") or {}
        return ModelResponse(
            text=content,
            tool_calls=tool_calls,
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        )
