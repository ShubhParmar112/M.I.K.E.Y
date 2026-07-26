"""Durable missions (Gen 3). The centerpiece is the reboot drill: run a mission
partway, throw away the runner (as a crash/reboot would), rebuild it from the log
alone, resume — and prove every step ran exactly once and the mission completes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import Config
from core.events.store import EventStore
from core.executor_client import ExecutorClient
from core.missions.runner import MissionRunner
from core.missions.store import MissionState, MissionStep, MissionStore
from core.orchestrator.loop import ApprovalRegistry
from core.policy.engine import PolicyEngine
from core.storage.db import Database


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MIKEY_WORKSPACE", raising=False)
    monkeypatch.setenv("MIKEY_HOME", str(tmp_path))
    config = Config(home=tmp_path)
    config.ensure_dirs()
    return config, Database(config.db_path)


def _write_steps(n: int) -> list[MissionStep]:
    return [MissionStep("fs_write", {"path": f"out/f{i}.txt", "content": f"step{i}"}) for i in range(n)]


def test_mission_state_projects_progress_and_status(env) -> None:
    _config, db = env
    missions = MissionStore(EventStore(db))
    m = missions.create("do three things", _write_steps(3))
    assert m.status == "pending" and m.next_step == 0

    missions.record_step_result(m.id, 0, ok=True, output="ok")
    missions.record_step_result(m.id, 1, ok=True, output="ok")
    assert missions.state(m.id).next_step == 2
    assert missions.state(m.id).status == "running"

    missions.record_step_result(m.id, 2, ok=True, output="ok")
    assert missions.state(m.id).status == "completed"


def test_failed_step_marks_failed_and_active_list(env) -> None:
    _config, db = env
    missions = MissionStore(EventStore(db))
    m = missions.create("will fail", _write_steps(3))
    missions.record_step_result(m.id, 0, ok=True, output="ok")
    missions.record_step_result(m.id, 1, ok=False, output="boom")
    state = missions.state(m.id)
    assert state.status == "failed" and state.next_step == 1  # resume re-runs step 1
    assert m.id not in [s.id for s in missions.active()]  # failed is not "resumable" until retried


async def _drive(runner: MissionRunner, mission_id: str, stop_after: int | None = None,
                 approvals: ApprovalRegistry | None = None) -> int:
    """Consume a run; auto-approve steps; optionally 'crash' after N results."""
    done = 0
    async for ev in runner.run(mission_id):
        if ev.kind == "approval_request" and approvals is not None:
            approvals.resolve(ev.data["approval_id"], approved=True, scope="once")
        if ev.kind == "action_result":
            done += 1
            if stop_after is not None and done >= stop_after:
                return done
    return done


async def test_mission_survives_reboot_and_completes(env) -> None:
    """The Gen 3 exit criterion, exactly as written: a 10+ step mission survives a
    mid-mission reboot and completes — every step run once, nothing repeated."""
    config, db = env
    missions = MissionStore(EventStore(db))
    policy = PolicyEngine(db)
    approvals = ApprovalRegistry()
    executor = ExecutorClient(config.workspace)
    steps = 10
    mission = missions.create("write ten files", _write_steps(steps))

    try:
        runner = MissionRunner(config, missions, policy, executor, approvals)
        await _drive(runner, mission.id, stop_after=6, approvals=approvals)  # crash after 6
        # exactly six side effects so far
        assert (config.workspace / "out" / "f5.txt").exists()
        assert not (config.workspace / "out" / "f6.txt").exists()

        # --- reboot: brand-new store + runner, state comes only from the log ---
        missions2 = MissionStore(EventStore(db))
        resumed: MissionState = missions2.state(mission.id)
        assert resumed.next_step == 6 and resumed.status == "running"

        runner2 = MissionRunner(config, missions2, policy, executor, approvals)
        await _drive(runner2, mission.id, approvals=approvals)  # resume to the end
    finally:
        await executor.close()

    # every step ran exactly once, mission complete
    for i in range(steps):
        assert (config.workspace / "out" / f"f{i}.txt").read_text() == f"step{i}"
    assert missions2.state(mission.id).status == "completed"
    assert policy.verify_audit_chain() is True


async def test_a_mission_step_that_clobbers_a_file_is_previewed_before_approval(env) -> None:
    """Unattended multi-step work is where an unpreviewed overwrite would slip
    through unnoticed, so mission steps get the same simulate-first treatment."""
    config, db = env
    (config.workspace / "out").mkdir(parents=True, exist_ok=True)
    (config.workspace / "out" / "f0.txt").write_text("PRECIOUS", encoding="utf-8")

    missions = MissionStore(EventStore(db))
    approvals = ApprovalRegistry()
    executor = ExecutorClient(config.workspace)
    mission = missions.create("overwrite a file", _write_steps(1))
    runner = MissionRunner(config, missions, PolicyEngine(db), executor, approvals)

    previews: list[dict] = []
    try:
        async for ev in runner.run(mission.id):
            if ev.kind == "approval_request":
                previews.append(ev.data["preview"])
                approvals.resolve(ev.data["approval_id"], approved=False, scope="once")
    finally:
        await executor.close()

    assert len(previews) == 1
    assert previews[0]["destructive"] is True
    assert "-PRECIOUS" in previews[0]["detail"]  # the user sees what would be lost
    # denied, so the file is untouched
    assert (config.workspace / "out" / "f0.txt").read_text(encoding="utf-8") == "PRECIOUS"
