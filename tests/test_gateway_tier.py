"""Tier-0 privacy enforcement at the Gateway (sovereignty S0).

Private (T0) data may ONLY be served by a local model — enforced here, not by
convention (architecture 02 §3). A T1 request is unaffected: absent/normal meta
behaves exactly as before.
"""

from __future__ import annotations

import pytest

from core.events.schema import Tier
from core.models.gateway import ChatMessage, ModelGateway, ModelResponse, RoutingMeta


class _Adapter:
    def __init__(self, name: str, local: bool, text: str) -> None:
        self.name = name
        self.local = local
        self._text = text
        self.calls = 0

    async def complete(self, system, messages, tools) -> ModelResponse:  # noqa: ANN001
        self.calls += 1
        return ModelResponse(text=self._text, tool_calls=[])


def _msgs() -> list[ChatMessage]:
    return [ChatMessage(role="user", text="something private")]


async def test_t0_routes_to_local_and_skips_cloud() -> None:
    cloud = _Adapter("groq", local=False, text="from cloud")
    local = _Adapter("ollama", local=True, text="from local")
    gw = ModelGateway(cloud, fallbacks=[local])

    resp = await gw.complete("", _msgs(), [], RoutingMeta(tier=Tier.T0))
    assert resp.text == "from local"
    assert gw.last_provider == "ollama"
    assert cloud.calls == 0  # T0 never touched the cloud provider


async def test_t0_with_no_local_refuses_rather_than_leaks() -> None:
    cloud = _Adapter("groq", local=False, text="from cloud")
    gw = ModelGateway(cloud)  # no local adapter anywhere

    with pytest.raises(RuntimeError) as ei:
        await gw.complete("", _msgs(), [], RoutingMeta(tier=Tier.T0))
    assert "Tier-0" in str(ei.value)
    assert cloud.calls == 0  # refused; nothing leaked


async def test_t1_is_unaffected_by_meta() -> None:
    cloud = _Adapter("groq", local=False, text="from cloud")
    local = _Adapter("ollama", local=True, text="from local")
    gw = ModelGateway(cloud, fallbacks=[local])

    # Explicit T1 and no-meta both serve from the primary, as before.
    r1 = await gw.complete("", _msgs(), [], RoutingMeta(tier=Tier.T1))
    r2 = await gw.complete("", _msgs(), [])
    assert r1.text == "from cloud" and r2.text == "from cloud"
    assert local.calls == 0


def test_routing_meta_defaults_to_t1() -> None:
    assert RoutingMeta().tier is Tier.T1
