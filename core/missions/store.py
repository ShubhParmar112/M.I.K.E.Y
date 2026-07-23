"""Durable missions (Gen 3): a multi-step plan whose plan AND per-step progress
are events in the log, so its state is a projection — always reconstructable, and
therefore able to survive a reboot and resume exactly where it stopped.

Status is derived, never stored: leading successful steps determine the next step;
a failed step marks the mission failed (and resuming re-runs from there).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.events.schema import Event, EventType, Provenance, ulid
from core.events.store import EventStore


@dataclass
class MissionStep:
    tool: str
    args: dict[str, Any]


@dataclass
class MissionState:
    id: str
    goal: str
    steps: list[MissionStep]
    step_results: dict[int, dict[str, Any]] = field(default_factory=dict)  # index -> {ok, output}

    @property
    def next_step(self) -> int:
        """Index of the first step not yet completed successfully (== len(steps)
        when the mission is done)."""
        for i in range(len(self.steps)):
            r = self.step_results.get(i)
            if r is None or not r["ok"]:
                return i
        return len(self.steps)

    @property
    def status(self) -> str:
        n = self.next_step
        if n >= len(self.steps):
            return "completed"
        r = self.step_results.get(n)
        if r is not None and not r["ok"]:
            return "failed"  # tried and failed; resume re-runs this step
        return "running" if self.step_results else "pending"


class MissionStore:
    _TYPES = [EventType.MISSION_CREATED.value, EventType.MISSION_STEP_RESULT.value]

    def __init__(self, events: EventStore, device_id: str = "dev_desktop_1") -> None:
        self._events = events
        self._device = device_id

    def create(self, goal: str, steps: list[MissionStep]) -> MissionState:
        mission_id = ulid()
        self._events.append(
            Event(
                type=EventType.MISSION_CREATED.value,
                device=self._device,
                provenance=Provenance(source="user", trusted=True),
                payload={
                    "mission_id": mission_id,
                    "goal": goal,
                    "steps": [{"tool": s.tool, "args": s.args} for s in steps],
                },
            )
        )
        state = self.state(mission_id)
        assert state is not None
        return state

    def record_step_result(self, mission_id: str, step: int, ok: bool, output: str) -> None:
        self._events.append(
            Event(
                type=EventType.MISSION_STEP_RESULT.value,
                device=self._device,
                provenance=Provenance(source="agent", trusted=True),
                payload={"mission_id": mission_id, "step": step, "ok": ok, "output": output[:4000]},
            )
        )

    def state(self, mission_id: str) -> MissionState | None:
        created: Event | None = None
        results: dict[int, dict[str, Any]] = {}
        for ev in self._events.recent(types=self._TYPES, limit=1_000_000):  # oldest-first
            if ev.payload.get("mission_id") != mission_id:
                continue
            if ev.type == EventType.MISSION_CREATED.value:
                created = ev
            else:  # latest result per step wins (supports retry-on-resume)
                results[int(ev.payload["step"])] = {
                    "ok": bool(ev.payload["ok"]),
                    "output": ev.payload.get("output", ""),
                }
        if created is None:
            return None
        steps = [MissionStep(s["tool"], dict(s.get("args", {}))) for s in created.payload["steps"]]
        return MissionState(mission_id, created.payload["goal"], steps, results)

    def active(self) -> list[MissionState]:
        """Missions that are unfinished (resumable) — pending or running."""
        out: list[MissionState] = []
        for ev in self._events.recent(types=[EventType.MISSION_CREATED.value], limit=1_000_000):
            state = self.state(ev.payload["mission_id"])
            if state is not None and state.status in ("pending", "running"):
                out.append(state)
        return out
