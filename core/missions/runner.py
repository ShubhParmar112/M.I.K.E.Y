"""Mission runner: executes a mission's remaining steps, one at a time, through
the SAME policy engine, approval flow, and sandboxed executor as a normal turn —
recording each outcome as a durable event. Because progress lives in the log, a
runner started fresh after a reboot resumes exactly where the last one stopped.

Gen 3 first slice: linear steps over the executor tools (fs_*, run_command,
web_fetch). Model-planned DAGs and in-process tools come in later slices.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from core.config import Config
from core.executor_client import ExecResult, ExecutorClient
from core.missions.store import MissionStore
from core.orchestrator.loop import ApprovalRegistry, StreamEvent
from core.policy.engine import ActionRequest, Decision, PolicyEngine
from core.events.schema import ulid


class MissionRunner:
    def __init__(
        self,
        config: Config,
        missions: MissionStore,
        policy: PolicyEngine,
        executor: ExecutorClient,
        approvals: ApprovalRegistry,
    ) -> None:
        self._config = config
        self._missions = missions
        self._policy = policy
        self._executor = executor
        self._approvals = approvals

    async def run(self, mission_id: str) -> AsyncIterator[StreamEvent]:
        state = self._missions.state(mission_id)
        if state is None:
            yield StreamEvent("error", {"message": f"no such mission: {mission_id}"})
            return
        if state.status == "completed":
            yield StreamEvent("final", {"mission_id": mission_id, "status": "completed"})
            return

        session_id = f"mission:{mission_id}"
        start = state.next_step
        yield StreamEvent(
            "status",
            {"mission_id": mission_id, "goal": state.goal,
             "resuming_at": start, "total": len(state.steps)},
        )

        for i in range(start, len(state.steps)):
            step = state.steps[i]
            yield StreamEvent(
                "action", {"mission_id": mission_id, "step": i, "tool": step.tool, "args": step.args}
            )

            req = ActionRequest(
                tool=step.tool, args=step.args, turn_id=mission_id, session_id=session_id
            )
            decision = self._policy.evaluate(req).decision
            if decision is Decision.ASK:
                approval_id = ulid()
                fut = self._approvals.create(approval_id)
                yield StreamEvent(
                    "approval_request",
                    {"approval_id": approval_id, "mission_id": mission_id, "step": i,
                     "tool": step.tool, "args": step.args},
                )
                try:
                    approved, scope = await asyncio.wait_for(fut, timeout=600.0)
                except TimeoutError:
                    approved, scope = False, "once"
                self._policy.record_user_decision(req, approved)
                if approved and scope == "session":
                    self._policy.grant_session(req)
                decision = Decision.ALLOW if approved else Decision.DENY

            if decision is Decision.DENY:
                self._missions.record_step_result(mission_id, i, ok=False, output="denied")
                yield StreamEvent("action_result", {"step": i, "ok": False, "output": "denied"})
                yield StreamEvent(
                    "error", {"message": f"mission paused: step {i} ({step.tool}) was denied"}
                )
                return

            try:
                result = await self._executor.call(step.tool, step.args)
            except Exception as exc:
                result = ExecResult(False, f"executor failure: {exc}", False)

            self._missions.record_step_result(mission_id, i, ok=result.ok, output=result.output)
            yield StreamEvent(
                "action_result", {"step": i, "ok": result.ok, "output": result.output[:500]}
            )
            if not result.ok:
                yield StreamEvent(
                    "error",
                    {"message": f"mission failed at step {i} ({step.tool}); "
                     f"fix and resume. detail: {result.output[:200]}"},
                )
                return

        yield StreamEvent("final", {"mission_id": mission_id, "status": "completed"})
