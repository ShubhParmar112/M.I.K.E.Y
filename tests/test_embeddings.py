"""OllamaEmbedder: returns a vector, and signals ModelUnavailable when the server
is down or the model isn't pulled (so recall can degrade to keyword-only)."""

from __future__ import annotations

import httpx
import pytest

from core.models.embeddings import OllamaEmbedder
from core.models.gateway import ModelUnavailable


def test_returns_vector() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})

    e = OllamaEmbedder("http://localhost:11434", transport=httpx.MockTransport(handler))
    assert e.embed("hello") == [0.1, 0.2, 0.3]


def test_offline_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    e = OllamaEmbedder("http://localhost:11434", transport=httpx.MockTransport(handler))
    with pytest.raises(ModelUnavailable):
        e.embed("hello")


def test_model_not_pulled_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    e = OllamaEmbedder("http://localhost:11434", transport=httpx.MockTransport(handler))
    with pytest.raises(ModelUnavailable) as ei:
        e.embed("hello")
    assert "not pulled" in ei.value.reason
