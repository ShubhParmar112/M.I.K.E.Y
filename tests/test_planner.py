"""The Planner (sovereignty S1): a goal becomes a validated, durable mission plan.

Driven by scripted FakeAdapters so decomposition, validation, and the fallbacks
are tested deterministically and offline.
"""

from __future__ import annotations

from typing import Any

from core.events.store import EventStore
from core.missions.store import MissionStore
from core.models.fake_adapter import FakeAdapter
from core.models.gateway import ModelGateway, ModelResponse, ToolCall
from core.orchestrator.planner import PLANNABLE_TOOLS, Planner
from core.storage.db import Database


def _planner_proposing(steps: list[dict[str, Any]]) -> Planner:
    call = ToolCall(id="p", name="propose_plan", arguments={"steps": steps})
    return Planner(ModelGateway(FakeAdapter([ModelResponse(text="", tool_calls=[call])])))


async def test_plan_builds_ordered_valid_steps() -> None:
    p = _planner_proposing([
        {"tool": "fs_list", "args": {"path": "."}, "rationale": "see what exists"},
        {"tool": "fs_write", "args": {"path": "README.md", "content": "# hi"}, "rationale": "scaffold"},
    ])
    result = await p.plan("scaffold a readme")
    assert result.ok
    assert [s.tool for s in result.steps] == ["fs_list", "fs_write"]
    assert result.steps[1].args["path"] == "README.md"
    assert result.rejected == []


async def test_plan_drops_un_runnable_tools() -> None:
    """A mission can only run executor tools; memory ops / invented tools are dropped."""
    p = _planner_proposing([
        {"tool": "memory_forget", "args": {"event_id": "x"}},   # not runnable in a mission
        {"tool": "teleport", "args": {}},                       # hallucinated
        {"tool": "fs_write", "args": {"path": "a.txt", "content": "y"}},
    ])
    result = await p.plan("do things")
    assert [s.tool for s in result.steps] == ["fs_write"]
    assert set(result.rejected) == {"memory_forget", "teleport"}


async def test_plan_from_json_text_fallback() -> None:
    """If the model answers with a JSON body instead of the tool call, still parse it."""
    body = '{"steps": [{"tool": "web_fetch", "args": {"url": "http://x"}}]}'
    p = Planner(ModelGateway(FakeAdapter([ModelResponse(text=body, tool_calls=[])])))
    result = await p.plan("grab a page")
    assert result.ok and result.steps[0].tool == "web_fetch"


async def test_plan_without_a_plan_is_not_ok() -> None:
    p = Planner(ModelGateway(FakeAdapter([ModelResponse(text="I can't do that", tool_calls=[])])))
    result = await p.plan("???")
    assert result.ok is False
    assert result.steps == []


async def test_plan_survives_a_dead_model() -> None:
    class _Boom:
        name = "boom"
        local = False

        async def complete(self, system, messages, tools) -> ModelResponse:  # noqa: ANN001
            raise RuntimeError("provider down")

    result = await Planner(ModelGateway(_Boom())).plan("x")
    assert result.ok is False
    assert "unavailable" in result.notes


async def test_plan_becomes_a_durable_mission(db: Database) -> None:
    """The plan integrates with the real mission store: a created mission carries
    the planned steps and starts pending — ready for the (already-tested) runner."""
    p = _planner_proposing([
        {"tool": "fs_write", "args": {"path": "out/a.txt", "content": "x"}, "rationale": "r"},
    ])
    result = await p.plan("write a file")
    missions = MissionStore(EventStore(db))
    mission = missions.create("write a file", result.steps)

    reloaded = missions.state(mission.id)
    assert reloaded is not None
    assert reloaded.steps[0].tool == "fs_write"
    assert reloaded.steps[0].args["path"] == "out/a.txt"
    assert reloaded.status == "pending"


def test_plannable_tools_are_exactly_the_executor_tools() -> None:
    assert PLANNABLE_TOOLS == {"fs_read", "fs_write", "fs_list", "run_command", "web_fetch"}
