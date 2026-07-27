"""The OpenAI-compatible chat-completions adapter — one implementation, several
providers.

Groq, Cerebras and Google (via its OpenAI-compatibility endpoint) all speak the
same wire format, and the hard-won behaviour around it is provider-independent:
recovering a tool call the model emitted as literal text, retrying a malformed
tool call, re-sampling a reply that collapsed into a repetition loop, telling a
per-minute rate spike apart from a daily quota that is gone until tomorrow.

That last distinction is why this module exists at all. A single free tier is a
single point of failure: when the day's tokens run out, every remaining answer
comes from a 3B local model and quality falls off a cliff with no error to show
for it. Several independent free tiers behind one gateway means the cliff is a
step down to another cloud model instead — the local model goes back to being
what it was meant to be, the offline safety net.

Subclasses supply the endpoint and a name. They should not need to touch the
protocol.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, ClassVar

import httpx

from core.events.schema import ulid
from core.models.degeneration import collapse_score, is_degenerate, trim_restart
from core.models.gateway import ChatMessage, ModelResponse, ModelUnavailable, ToolCall

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
    """The provider's own words for WHY it refused, not just "429".

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


# Each provider words a daily cap differently: Groq says "tokens per day (TPD)",
# Cerebras "tokens per day", Google "GenerateRequestsPerDayPerProjectPerModel" or
# "quota exceeded ... per day". Matching all three matters because the answer to a
# daily cap (stop asking this provider today) is the opposite of the answer to a
# per-minute one (wait a few seconds).
_DAILY_LIMIT_RE = re.compile(r"per\s*day|\bTPD\b|\bRPD\b|\bperday\b|\bdaily\b", re.IGNORECASE)


def _is_daily_limit(reason: str) -> bool:
    """Whether a 429 means "finished for today" rather than "slow down"."""
    return bool(_DAILY_LIMIT_RE.search(reason))


# "This request is longer than the model can take." Providers disagree on the
# wording and on the status code, but they agree on the shape of the problem.
_TOO_LONG_RE = re.compile(
    r"context[ _-]?(length|window)|too many tokens|maximum context|"
    r"reduce the length|input is too long|exceeds? the .{0,20}(context|token) limit",
    re.IGNORECASE,
)


def _is_too_long(message: str) -> bool:
    """Whether a 400 means the request didn't fit — which another provider with a
    bigger window may well accept.

    Free tiers cap context far below the model's real window (Cerebras' is a
    fraction of Groq's), so this is not exotic: one long document in memory and the
    smaller provider refuses a request the next one in the chain would answer
    fine. Treating it as a hard error would end the turn on the narrowest link.
    """
    return bool(_TOO_LONG_RE.search(message))


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class OpenAICompatAdapter:
    """Chat completions over the OpenAI wire format. Subclass and set the class
    attributes; the protocol handling below is shared."""

    name: ClassVar[str] = "openai-compatible"
    base_url: ClassVar[str] = ""
    api_key_env: ClassVar[str] = ""
    local: ClassVar[bool] = False  # cloud provider; never eligible to serve Tier-0
    # Some compatibility endpoints reject sampling parameters they don't implement.
    # Only the re-sample path uses them, so a provider that refuses them simply
    # re-samples at a higher temperature without the penalties.
    supports_penalties: ClassVar[bool] = True

    @property
    def model(self) -> str:
        """The model id actually being billed — read by the cost governor."""
        return self._model

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
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get(self.api_key_env, "")
        self._base_url = (base_url or self.base_url).rstrip("/")
        self._transport = transport  # injectable for tests
        self._rl_retries = rate_limit_retries
        self._rl_backoff = rate_limit_backoff_s
        self._temperature = temperature
        # An unbounded reply is what let a repetition loop run for a full paragraph
        # of self-negating text. Cap it: a collapsed generation gets cut off, and a
        # legitimate derivation fits comfortably.
        self._max_output_tokens = max_output_tokens
        self._resample = resample_on_collapse

    # ---- wire format -------------------------------------------------------

    def _wire_messages(self, system: str, messages: list[ChatMessage]) -> list[dict[str, Any]]:
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
        return wire

    def _body(
        self, system: str, messages: list[ChatMessage], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": self._wire_messages(system, messages),
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
        return body

    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        body = self._body(system, messages, tools)

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
                if self.supports_penalties:
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

        Open-weight models occasionally emit malformed tool-call syntax, which the
        provider rejects as 400/tool_use_failed. That's a transient generation
        failure, not a request error: retry, and on the last attempt force a
        text-only answer so the turn degrades gracefully instead of dying (failure
        taxonomy M12).
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
                        f"{self._base_url}/chat/completions",
                        headers=self._headers(),
                        json=attempt_body,
                    )
                except httpx.TransportError as exc:
                    # DNS/connect/timeout ~ offline or the provider unreachable. A
                    # local fallback can still answer, so signal it rather than dying.
                    raise ModelUnavailable(
                        self.name, f"unreachable ({type(exc).__name__})"
                    ) from exc
                if (
                    resp.status_code == 429
                    and rl < self._rl_retries
                    # Waiting out a per-minute spike is worth ~15s for the better
                    # model. Waiting out a DAILY cap is 15s of dead time on every
                    # remaining turn of the day, and it still ends in the fallback.
                    and not _is_daily_limit(_rate_limit_reason(resp))
                ):
                    delay = min(
                        _retry_after(resp) or self._rl_backoff * (rl + 1),
                        MAX_RATE_LIMIT_BACKOFF_S,
                    )
                    await asyncio.sleep(delay)
                    continue
                break
            # Rate limit (after backoff) and server errors are what the fallback
            # chain exists for; auth/other 4xx are not (that hides a real bug).
            if resp.status_code == 429:
                reason = _rate_limit_reason(resp)
                raise ModelUnavailable(
                    self.name, reason, retry_after=_retry_after(resp),
                    daily=_is_daily_limit(reason),
                )
            if resp.status_code >= 500:
                raise ModelUnavailable(self.name, f"server error ({resp.status_code})")
            if resp.status_code == 400:
                try:
                    err = resp.json().get("error", {})
                except ValueError:
                    err = {}
                if err.get("code") == "tool_use_failed":
                    last_failed_generation = str(err.get("failed_generation", ""))[:300]
                    continue
                message = str(err.get("message", resp.text[:300]))
                if _is_too_long(message):
                    raise ModelUnavailable(self.name, f"context too small ({message[:120]})")
                raise RuntimeError(f"{self.name} 400: {message}")
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            return data
        raise RuntimeError(
            f"{self.name}: model repeatedly produced malformed tool calls "
            f"(tool_use_failed); last generation: {last_failed_generation!r}"
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

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
        # Only reach for the text form when the provider gave us no structured
        # calls — the structured field is authoritative when present.
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
