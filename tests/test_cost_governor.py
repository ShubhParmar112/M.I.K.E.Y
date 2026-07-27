"""Cost governor (Gen 3 exit criterion: monthly spend within budget automatically).

The criterion is about *automatically*, so the tests that matter are the ones
where the budget actually refuses something: an exhausted budget must push work
onto the free local model rather than keep spending, and must survive a restart —
the ledger is the event log, not a counter in memory.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from core.cost.governor import CostGovernor, daily_cap_for, price_for
from core.events.schema import EventType, Tier
from core.events.store import EventStore
from core.models.gateway import (
    ChatMessage,
    ModelGateway,
    ModelResponse,
    ModelUnavailable,
    RoutingMeta,
)
from core.storage.db import Database


@pytest.fixture
def events(tmp_path: Path) -> EventStore:
    return EventStore(Database(tmp_path / "t.db"))


class _Adapter:
    """A model that reports a fixed token usage per call."""

    def __init__(self, name: str, model: str, local: bool, tokens: int = 1_000_000) -> None:
        self.name = name
        self.model = model
        self.local = local
        self.calls = 0
        self._tokens = tokens

    async def complete(
        self, system: str, messages: list[ChatMessage], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            text=f"from {self.name}",
            tool_calls=[],
            usage={"input_tokens": self._tokens, "output_tokens": 0},
        )


async def _ask(gateway: ModelGateway) -> str:
    resp = await gateway.complete("sys", [ChatMessage(role="user", text="hi")], [])
    return resp.text


# ---- pricing ---------------------------------------------------------------


def test_local_inference_is_free_and_cloud_is_not() -> None:
    local, exact = price_for("ollama", "llama3.2")
    assert local.cost(1_000_000, 1_000_000) == 0.0 and exact

    cloud, exact = price_for("groq", "llama-3.3-70b-versatile")
    assert cloud.cost(1_000_000, 0) > 0 and exact


def test_an_unpriced_cloud_model_is_charged_high_not_free() -> None:
    """A model missing from the table must never read as free — that is exactly
    how a budget silently stops binding."""
    price, exact = price_for("anthropic", "some-unreleased-model-2027")
    assert exact is False
    assert price.cost(1_000_000, 0) > 0


# ---- the ledger ------------------------------------------------------------


def test_spend_is_a_projection_over_the_log_not_in_memory_state(events: EventStore) -> None:
    gov = CostGovernor(events, budget_usd=10.0)
    gov.record("groq", "llama-3.3-70b-versatile", {"input_tokens": 1_000_000, "output_tokens": 0})
    first = gov.month_to_date().total_usd
    assert first == pytest.approx(0.59)

    # a fresh governor — as after a restart — reads the same total back from the log
    assert CostGovernor(events, budget_usd=10.0).month_to_date().total_usd == pytest.approx(first)
    assert [e.type for e in events.recent()] == [EventType.MODEL_USAGE.value]


def test_the_budget_resets_when_the_month_rolls_over(events: EventStore) -> None:
    gov = CostGovernor(events, budget_usd=0.10)
    gov.record("groq", "llama-3.3-70b-versatile", {"input_tokens": 1_000_000, "output_tokens": 0})
    assert gov.month_to_date().over_budget is True

    # 45 days on is always a later month; this month's spend must not follow it there
    later = datetime.now(UTC) + timedelta(days=45)
    assert gov.month_to_date(later).total_usd == 0.0
    assert gov.cloud_allowed(later) is True


# ---- enforcement at the gateway --------------------------------------------


async def test_exhausted_budget_falls_back_to_local_instead_of_spending(
    events: EventStore,
) -> None:
    cloud = _Adapter("groq", "llama-3.3-70b-versatile", local=False)
    local = _Adapter("ollama", "llama3.2", local=True)
    # 1M input tokens ≈ $0.59, so a $1 budget survives one call and not two.
    gov = CostGovernor(events, budget_usd=1.0)
    gateway = ModelGateway(cloud, fallbacks=[local], governor=gov)

    assert await _ask(gateway) == "from groq"
    assert gov.month_to_date().over_budget is False

    # second call takes it over; the third must not reach the cloud at all
    assert await _ask(gateway) == "from groq"
    assert gov.month_to_date().over_budget is True

    assert await _ask(gateway) == "from ollama"
    assert cloud.calls == 2, "no cloud call may be made once the budget is spent"
    # and the local turns cost nothing, so the total stops moving
    assert gov.month_to_date().total_usd == pytest.approx(1.18)


async def test_over_budget_with_no_local_model_is_a_clear_error_not_a_silent_charge(
    events: EventStore,
) -> None:
    cloud = _Adapter("groq", "llama-3.3-70b-versatile", local=False)
    gov = CostGovernor(events, budget_usd=0.10)
    gateway = ModelGateway(cloud, governor=gov)

    await _ask(gateway)  # spends past the budget
    with pytest.raises(RuntimeError, match="budget is spent"):
        await _ask(gateway)
    assert cloud.calls == 1


async def test_a_zero_budget_disables_enforcement_but_keeps_tracking(
    events: EventStore,
) -> None:
    cloud = _Adapter("groq", "llama-3.3-70b-versatile", local=False)
    gov = CostGovernor(events, budget_usd=0.0)
    gateway = ModelGateway(cloud, governor=gov)

    for _ in range(3):
        await _ask(gateway)
    spend = gov.month_to_date()
    assert cloud.calls == 3 and spend.enforced is False
    assert spend.total_usd == pytest.approx(1.77)  # still measured


async def test_a_fallback_to_local_is_billed_to_the_model_that_served(
    events: EventStore,
) -> None:
    class _RateLimited:
        name, local, model = "groq", False, "llama-3.3-70b-versatile"

        async def complete(self, system, messages, tools):  # type: ignore[no-untyped-def]
            raise ModelUnavailable("groq", "429")

    local = _Adapter("ollama", "llama3.2", local=True)
    gov = CostGovernor(events, budget_usd=10.0)
    gateway = ModelGateway(_RateLimited(), fallbacks=[local], governor=gov)

    assert await _ask(gateway) == "from ollama"
    spend = gov.month_to_date()
    assert spend.by_provider == {"ollama": 0.0}  # the rate-limited call is not charged
    assert spend.total_usd == 0.0


async def test_usage_events_carry_the_turns_privacy_tier(events: EventStore) -> None:
    local = _Adapter("ollama", "llama3.2", local=True)
    gateway = ModelGateway(local, governor=CostGovernor(events, budget_usd=1.0))
    await gateway.complete(
        "sys", [ChatMessage(role="user", text="private")], [], RoutingMeta(tier=Tier.T0)
    )
    usage = events.recent(types=[EventType.MODEL_USAGE.value])
    assert len(usage) == 1 and usage[0].tier is Tier.T0


def test_warning_fires_before_the_budget_is_gone(events: EventStore) -> None:
    gov = CostGovernor(events, budget_usd=1.0)
    gov.record("groq", "llama-3.3-70b-versatile", {"input_tokens": 1_500_000, "output_tokens": 0})
    spend = gov.month_to_date()  # ~$0.885 of $1.00
    assert spend.warning is True and spend.over_budget is False


# ---- the other budget: tokens per day --------------------------------------
#
# The failure these cover is not overspending, it is the opposite: a FREE plan
# whose daily token allowance runs out mid-evening, after which every answer
# comes from a much weaker local model with nothing on screen to say so.


def test_a_free_tier_provider_has_a_daily_token_cap_and_a_local_one_never_does() -> None:
    assert daily_cap_for("groq") == 100_000
    assert daily_cap_for("ollama") == 0, "local inference has no allowance to run out of"
    assert daily_cap_for("anthropic") == 0, "no known free daily cap — don't invent one"
    # moving off the free tier is a config change, not a code change
    assert daily_cap_for("groq", override=500_000) == 500_000
    assert daily_cap_for("ollama", override=500_000) == 0


def test_today_counts_tokens_per_provider_against_the_daily_allowance(
    events: EventStore,
) -> None:
    gov = CostGovernor(events, budget_usd=10.0)
    for _ in range(4):
        gov.record("groq", "llama-3.3-70b-versatile", {"input_tokens": 5_000, "output_tokens": 500})
    gov.record("ollama", "llama3.2", {"input_tokens": 9_000, "output_tokens": 9_000})

    day = gov.today()
    assert day.calls == 5
    groq, ollama = day.providers[0], day.providers[1]
    assert groq.provider == "groq"  # busiest first
    assert (groq.calls, groq.total_tokens, groq.cap) == (4, 22_000, 100_000)
    assert groq.warning is False and groq.exhausted is False
    assert ollama.capped is False and ollama.total_tokens == 18_000
    # a rough "how much conversation is left", which a raw token count is not
    assert groq.calls_left == 14  # 78,000 left at 5,500 a call


def test_the_daily_gauge_warns_with_room_left_then_reports_exhaustion(
    events: EventStore,
) -> None:
    gov = CostGovernor(events, budget_usd=10.0)
    gov.record("groq", "llama-3.3-70b-versatile", {"input_tokens": 80_000, "output_tokens": 0})
    hot = gov.today().pressured
    assert hot is not None and hot.warning is True and hot.exhausted is False
    assert hot.remaining_tokens == 20_000

    gov.record("groq", "llama-3.3-70b-versatile", {"input_tokens": 25_000, "output_tokens": 0})
    hot = gov.today().pressured
    assert hot is not None and hot.exhausted is True and hot.remaining_tokens == 0


def test_yesterdays_tokens_do_not_count_against_today(events: EventStore) -> None:
    """The allowance resets; a gauge that never forgets would cry wolf every day."""
    gov = CostGovernor(events, budget_usd=10.0)
    gov.record("groq", "llama-3.3-70b-versatile", {"input_tokens": 99_000, "output_tokens": 0})
    assert gov.today().pressured is not None

    tomorrow = datetime.now(UTC) + timedelta(days=1)
    day = gov.today(tomorrow)
    assert day.providers == [] and day.pressured is None


def test_the_daily_gauge_never_gates_a_call(events: EventStore) -> None:
    """Deliberate: our tally starts whenever logging was deployed and a provider's
    daily window need not match the local calendar day, so this number is honest as
    a warning and would be wrong as a gate. The provider's 429 stays the authority."""
    gov = CostGovernor(events, budget_usd=10.0)
    gov.record("groq", "llama-3.3-70b-versatile", {"input_tokens": 200_000, "output_tokens": 0})
    assert gov.today().pressured is not None  # visibly out of daily tokens
    assert gov.cloud_allowed() is True  # and still allowed to try
