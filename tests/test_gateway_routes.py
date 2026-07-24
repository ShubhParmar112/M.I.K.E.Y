"""Per-capability model routing (sovereignty S2): pin a brain to a local model,
one at a time, with cloud fallback preserved and the Tier-0 rule still overriding.
"""

from __future__ import annotations

from core.events.schema import Tier
from core.models.gateway import (
    ChatMessage,
    ModelGateway,
    ModelResponse,
    ModelUnavailable,
    RoutingMeta,
)


class _Adapter:
    def __init__(self, name: str, text: str, local: bool = False,
                 error: Exception | None = None) -> None:
        self.name = name
        self._text = text
        self.local = local
        self._error = error
        self.calls = 0

    async def complete(self, system, messages, tools) -> ModelResponse:  # noqa: ANN001
        self.calls += 1
        if self._error is not None:
            raise self._error
        return ModelResponse(text=self._text, tool_calls=[])


def _msgs() -> list[ChatMessage]:
    return [ChatMessage(role="user", text="hi")]


async def test_routed_capability_is_served_locally_others_stay_cloud() -> None:
    cloud = _Adapter("groq", "cloud")
    local = _Adapter("ollama", "local", local=True)
    gw = ModelGateway(cloud, fallbacks=[local], routes={"verify": local})

    # the routed capability (critic) is served by the local model
    r = await gw.complete("", _msgs(), [], RoutingMeta(capability="verify"))
    assert r.text == "local" and gw.last_provider == "ollama"
    assert cloud.calls == 0

    # a different capability still uses the cloud primary
    r2 = await gw.complete("", _msgs(), [], RoutingMeta(capability="general"))
    assert r2.text == "cloud"


async def test_no_meta_is_unchanged_default_chain() -> None:
    cloud = _Adapter("groq", "cloud")
    local = _Adapter("ollama", "local", local=True)
    gw = ModelGateway(cloud, fallbacks=[local], routes={"verify": local})
    assert (await gw.complete("", _msgs(), [])).text == "cloud"


async def test_routed_local_falls_back_to_cloud_on_failure() -> None:
    cloud = _Adapter("groq", "cloud")
    local = _Adapter("ollama", "x", local=True, error=ModelUnavailable("ollama", "down"))
    gw = ModelGateway(cloud, fallbacks=[local], routes={"verify": local})

    r = await gw.complete("", _msgs(), [], RoutingMeta(capability="verify"))
    assert r.text == "cloud" and gw.last_provider == "groq"  # local led, failed, cloud caught it
    assert local.calls == 1 and cloud.calls == 1


async def test_tier0_overrides_a_cloud_route() -> None:
    """Routing must never defeat the privacy rule: a T0 request stays local even if
    its capability is routed to the cloud."""
    cloud = _Adapter("groq", "cloud")
    local = _Adapter("ollama", "local", local=True)
    gw = ModelGateway(cloud, fallbacks=[local], routes={"general": cloud})

    r = await gw.complete("", _msgs(), [], RoutingMeta(tier=Tier.T0, capability="general"))
    assert r.text == "local" and cloud.calls == 0


def test_routed_capabilities_are_visible() -> None:
    cloud = _Adapter("groq", "c")
    local = _Adapter("ollama", "l", local=True)
    gw = ModelGateway(cloud, fallbacks=[local], routes={"verify": local, "chat": local})
    assert gw.routed_capabilities == {"verify": "ollama", "chat": "ollama"}
