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
from core.models.degeneration import collapse_score, is_degenerate, trim_restart
from core.models.gateway import ChatMessage, ModelResponse, ModelUnavailable, ToolCall

BASE_URL = "https://api.groq.com/openai/v1"
# The local fallback is a weak 3B model that hallucinates, so it's worth waiting
# out a per-minute rate spike on the strong cloud model rather than conceding fast.
MAX_RATE_LIMIT_BACKOFF_S = 15.0

# Re-sample settings for a reply that collapsed into a repetition loop. Warmer than
# the default on purpose — a *low* temperature is what sharpens the loop in the first
# place — plus frequency/presence penalties that make repeating a phrase expensive.
# Penalties are confined to this path: applied to a tool-bearing request they also
# penalize the repeated punctuation of well-formed JSON arguments.
RESAMPLE_TEMPERATURE = 0.7
RESAMPLE_FREQUENCY_PENALTY = 0.7
RESAMPLE_PRESENCE_PENALTY = 0.3

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


def _rate_limit_reason(resp: httpx.Response) -> str:
    """Groq's own words for WHY it refused, not just "429".

    This distinction is operationally the whole story and collapsing it loses real
    information: a per-minute (TPM) spike clears in seconds and pacing fixes it, while
    a per-day (TPD) cap means the key is finished until the window rolls over and no
    amount of pacing or retrying will help. The message names which limit was hit and
    the numbers behind it — carry it through.
    """
    try:
        message = str(resp.json().get("error", {}).get("message", "")).strip()
    except ValueError:
        message = ""
    if not message:
        return "rate limited (429)"
    # Drop the upsell tail; keep the diagnosis.
    message = message.split("Need more tokens?")[0].strip()
    return f"rate limited (429) — {message}"


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
        temperature: float = 0.3,
        max_output_tokens: int = 1536,
        resample_on_collapse: bool = True,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._transport = transport  # injectable for tests
        self._rl_retries = rate_limit_retries
        self._rl_backoff = rate_limit_backoff_s
        self._temperature = temperature
        # An unbounded reply is what let a repetition loop run for a full paragraph
        # of self-negating text. Cap it: a collapsed generation gets cut off, and a
        # legitimate derivation fits comfortably.
        self._max_output_tokens = max_output_tokens
        self._resample = resample_on_collapse

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

        body: dict[str, Any] = {
            "model": self._model,
            "messages": wire,
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
        }
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
            answer = self._clean(self._parse(await self._post(client, body)))

            # A reply that collapsed into a repetition loop is a failed generation,
            # not an answer — re-sample it once with anti-repetition settings rather
            # than handing the user a paragraph of self-negating text. Only for final
            # text answers: a tool call is structured output, judged by the API.
            if self._resample and not answer.tool_calls and is_degenerate(answer.text):
                retry_body = dict(body)
                retry_body["temperature"] = RESAMPLE_TEMPERATURE
                retry_body["frequency_penalty"] = RESAMPLE_FREQUENCY_PENALTY
                retry_body["presence_penalty"] = RESAMPLE_PRESENCE_PENALTY
                if "tools" in retry_body:
                    retry_body["tool_choice"] = "none"  # we know this turn ends in prose
                try:
                    second = self._clean(self._parse(await self._post(client, retry_body)))
                except (ModelUnavailable, RuntimeError):
                    # Nothing better on offer; a repetitive answer still beats
                    # failing the turn outright.
                    answer.notes.append("collapsed reply; re-sample unavailable")
                    return answer
                # Keep whichever is less repetitive — a re-sample can collapse too.
                better = min((second, answer), key=lambda r: collapse_score(r.text))
                better.notes.append(
                    "re-sampled after repetition collapse "
                    f"({collapse_score(answer.text):.2f} -> {collapse_score(second.text):.2f})"
                )
                return better
            return answer

    def _clean(self, resp: ModelResponse) -> ModelResponse:
        """Drop a trailing restart — the reply finished an answer and then began
        hunting for another one. Applied before the collapse check, since cutting the
        runaway tail often makes the surviving answer perfectly good and saves a
        needless re-sample."""
        if resp.tool_calls:
            return resp  # structured output; not prose to trim
        trimmed, cut = trim_restart(resp.text)
        if cut:
            resp.text = trimmed
            resp.notes.append("trimmed a restart after the answer was already given")
        return resp

    async def _post(self, client: httpx.AsyncClient, body: dict[str, Any]) -> dict[str, Any]:
        """One completion, with the two retry policies this API needs.

        Llama occasionally emits malformed tool-call syntax, which Groq rejects as
        400/tool_use_failed. That's a transient generation failure, not a request
        error: retry, and on the last attempt force a text-only answer so the turn
        degrades gracefully instead of dying (failure taxonomy M12).
        """
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
                    "groq", _rate_limit_reason(resp), retry_after=_retry_after(resp)
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
            data: dict[str, Any] = resp.json()
            return data
        raise RuntimeError(
            "groq: model repeatedly produced malformed tool calls "
            f"(tool_use_failed); last generation: {last_failed_generation!r}"
        )

    def _parse(self, data: dict[str, Any]) -> ModelResponse:
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
