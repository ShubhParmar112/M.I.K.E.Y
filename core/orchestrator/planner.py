"""The Planner (sovereignty S1: decompose the monolith).

Turns a natural-language goal into an ordered list of concrete tool steps — a
durable **mission** the existing MissionRunner executes and can resume after a
reboot (core/missions). This is the "hardest reasoning" brain (docs/04 §5); this
first slice produces a linear plan (a DAG is a later refinement).

Two properties make the output safe to hand to the executor:

- **Structured, not free text.** The planner must call `propose_plan`, so steps
  arrive as typed {tool, args} rather than prose we'd have to scrape.
- **Validated against what a mission can actually run.** Missions execute only the
  sandbox executor tools; any step naming another tool (memory ops, ingest, or a
  hallucinated name) is dropped and reported, never handed to the runner.

Every step still passes the policy engine and its approval card at run time — the
planner proposes; it does not gain any authority to act.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from core.missions.store import MissionStep
from core.models.gateway import ChatMessage, ModelGateway, ModelResponse, RoutingMeta
from core.orchestrator.brains import PLANNER

# The only tools a mission can execute (MissionRunner calls the executor directly;
# in-process tools like memory_* and ingest are not runnable there).
PLANNABLE_TOOLS = frozenset({"fs_read", "fs_write", "fs_list", "run_command", "web_fetch"})

PROPOSE_PLAN_TOOL: dict[str, Any] = {
    "name": "propose_plan",
    "description": "Emit the ordered plan as a list of concrete executor-tool steps.",
    "input_schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "description": "one of: fs_read, fs_write, fs_list, "
                            "run_command, web_fetch",
                        },
                        "args": {"type": "object", "description": "arguments for the tool"},
                        "rationale": {"type": "string", "description": "why this step (one line)"},
                    },
                    "required": ["tool", "args"],
                },
            }
        },
        "required": ["steps"],
    },
}


@dataclass
class PlanResult:
    ok: bool
    steps: list[MissionStep]
    notes: str = ""
    rejected: list[str] = field(default_factory=list)  # step tools dropped as un-runnable


class Planner:
    def __init__(self, gateway: ModelGateway) -> None:
        self._gateway = gateway

    async def plan(self, goal: str) -> PlanResult:
        try:
            resp = await self._gateway.complete(
                PLANNER.system_prompt,
                [ChatMessage(role="user", text=f"Goal: {goal}")],
                [PROPOSE_PLAN_TOOL],
                RoutingMeta(tier=PLANNER.tier, capability=PLANNER.capability),
            )
        except Exception as exc:
            return PlanResult(ok=False, steps=[], notes=f"planner unavailable: {exc}")

        raw = _extract_steps(resp)
        if raw is None:
            return PlanResult(
                ok=False, steps=[], notes=resp.text.strip()[:300] or "no plan was produced"
            )

        steps: list[MissionStep] = []
        rejected: list[str] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool", "")).strip()
            args = item.get("args", {})
            if tool not in PLANNABLE_TOOLS or not isinstance(args, dict):
                rejected.append(tool or "(missing tool)")
                continue
            steps.append(MissionStep(tool=tool, args=args))

        return PlanResult(ok=bool(steps), steps=steps, rejected=rejected)


def _extract_steps(resp: ModelResponse) -> list[Any] | None:
    """Prefer the structured propose_plan call; fall back to a JSON body if the
    model answered in text. None means no plan could be recovered."""
    for tc in resp.tool_calls:
        if tc.name == "propose_plan" and isinstance(tc.arguments.get("steps"), list):
            return list(tc.arguments["steps"])
    text = resp.text.strip()
    if text:
        try:
            data = json.loads(text)
        except ValueError:
            return None
        if isinstance(data, dict) and isinstance(data.get("steps"), list):
            return list(data["steps"])
        if isinstance(data, list):
            return data
    return None
