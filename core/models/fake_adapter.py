"""Scripted adapter for tests and offline demo — deterministic by design."""

from __future__ import annotations

from typing import Any

from core.models.gateway import ChatMessage, ModelResponse


class FakeAdapter:
    name = "fake"
    local = True  # offline/deterministic; treated as on-device for Tier-0 routing

    def __init__(self, script: list[ModelResponse] | None = None) -> None:
        self._script = list(script or [])
        self.calls: list[list[ChatMessage]] = []
        self.systems: list[str] = []
        # What each call was actually offered. A toolless brain's guarantee is that
        # the model never *sees* the tool — asserting on this checks the offer, not
        # just the absence of an action the model may simply have declined to take.
        self.tools: list[list[dict[str, Any]]] = []

    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.calls.append(list(messages))
        self.systems.append(system)
        self.tools.append(list(tools))
        if self._script:
            return self._script.pop(0)
        return ModelResponse(
            text="(fake model) No provider configured. Set ANTHROPIC_API_KEY or run Ollama.",
            tool_calls=[],
        )
