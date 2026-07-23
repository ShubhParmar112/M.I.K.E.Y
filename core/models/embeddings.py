"""Local text embeddings via Ollama (ADR-001: embeddings stay on-device — Groq
does not serve them). Used to add semantic retrieval alongside keyword search.

Synchronous on purpose: a single short embedding is fast, and it keeps the
memory write/recall paths (which are sync) simple. Failures surface as
ModelUnavailable so callers can degrade to keyword-only rather than break.
"""

from __future__ import annotations

import httpx

from core.models.gateway import ModelUnavailable


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str = "nomic-embed-text",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._transport = transport  # injectable for tests

    @property
    def name(self) -> str:
        return self._model

    def embed(self, text: str) -> list[float]:
        try:
            with httpx.Client(timeout=60.0, transport=self._transport) as client:
                resp = client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
        except httpx.TransportError as exc:
            raise ModelUnavailable("ollama-embed", "not running") from exc
        if resp.status_code == 404:
            raise ModelUnavailable(
                "ollama-embed", f"model '{self._model}' not pulled (ollama pull {self._model})"
            )
        resp.raise_for_status()
        vector = resp.json().get("embedding")
        if not vector:
            raise ModelUnavailable("ollama-embed", "empty embedding response")
        return [float(x) for x in vector]
