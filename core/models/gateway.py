"""Model Gateway (review M2): the only door to any LLM.

Modules request a completion; the gateway picks the provider. No other module
may import a provider SDK — that rule is what makes 'hybrid local+cloud'
implementable later (privacy tiers route here at Gen 2+).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatMessage:
    """Provider-neutral message. Adapters translate to vendor wire formats."""

    role: str  # "user" | "assistant" | "tool_result"
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set when role == "tool_result"


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: dict[str, int] = field(default_factory=dict)


class ModelAdapter(Protocol):
    name: str

    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> ModelResponse: ...


class ModelGateway:
    def __init__(self, adapter: ModelAdapter) -> None:
        self._adapter = adapter
        self.total_calls = 0

    @property
    def provider(self) -> str:
        return self._adapter.name

    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.total_calls += 1
        return await self._adapter.complete(system, messages, tools)
