"""Simulate-first previews (Gen 3 exit criterion: no destructive action without
a preview). Two things are under test: that the preview tells the truth about
what an action would do, and that the system cannot be talked out of showing it —
in particular that a standing session grant does not quietly cover an overwrite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.config import Config
from core.events.store import EventStore
from core.executor_client import ExecResult, ExecutorClient
from core.memory.store import MemoryStore
from core.models.fake_adapter import FakeAdapter
from core.models.gateway import ModelGateway, ModelResponse, ToolCall
from core.orchestrator.loop import ApprovalRegistry, Orchestrator
from core.policy.engine import PolicyEngine
from core.policy.preview import Previewer, classify
from core.storage.db import Database
from core.trace.store import TraceStore


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MIKEY_WORKSPACE", raising=False)
    monkeypatch.setenv("MIKEY_HOME", str(tmp_path))
    config = Config(home=tmp_path)
    config.ensure_dirs()
    return config, Database(config.db_path)


# ---- static classification -------------------------------------------------


@pytest.mark.parametrize(
    "tool,args,destructive",
    [
        ("fs_write", {"path": "a.txt", "content": "x"}, True),  # conservative: may clobber
        ("memory_forget", {"event_id": "01H"}, True),
        ("fs_read", {"path": "a.txt"}, False),
        ("memory_recall", {"query": "dog"}, False),
        ("run_command", {"command": ["git", "clean", "-fd"]}, True),
        ("run_command", {"command": ["git", "reset", "--hard"]}, True),
        ("run_command", {"command": ["git", "checkout", "--", "."]}, True),
        ("run_command", {"command": ["git", "push", "--force"]}, True),
        ("run_command", {"command": ["pip", "uninstall", "requests"]}, True),
        # already a simulation — nothing is deleted by a dry run
        ("run_command", {"command": ["git", "clean", "-fd", "--dry-run"]}, False),
        # routine work must stay frictionless
        ("run_command", {"command": ["git", "status"]}, False),
        ("run_command", {"command": ["python", "-m", "pytest"]}, False),
    ],
)
def test_classify_flags_destruction_and_spares_routine_work(
    tool: str, args: dict[str, Any], destructive: bool
) -> None:
    assert classify(tool, args).destructive is destructive


def test_uninstall_is_destructive_but_recoverable() -> None:
    impact = classify("run_command", {"command": ["pip", "uninstall", "requests"]})
    assert impact.destructive and impact.reversible  # you can reinstall a package


# ---- simulation ------------------------------------------------------------


async def test_write_preview_separates_a_create_from_a_clobber(env) -> None:
    config, _db = env
    executor = ExecutorClient(config.workspace)
    previewer = Previewer(executor)
    try:
        fresh = await previewer.preview(
            "fs_write", {"path": "notes.md", "content": "line one\nline two\n"}
        )
        assert fresh is not None
        assert fresh.destructive is False and fresh.simulated is True
        assert "creates notes.md" in fresh.summary

        (config.workspace / "notes.md").write_text("line one\nold line\n", encoding="utf-8")
        clobber = await previewer.preview(
            "fs_write", {"path": "notes.md", "content": "line one\nline two\n"}
        )
        assert clobber is not None
        assert clobber.destructive is True and clobber.reversible is False
        # the user is shown exactly what disappears, not just that "a write happens"
        assert "-old line" in clobber.detail and "+line two" in clobber.detail
    finally:
        await executor.close()


async def test_rewriting_identical_content_is_not_destructive(env) -> None:
    config, _db = env
    executor = ExecutorClient(config.workspace)
    (config.workspace / "same.txt").write_text("unchanged", encoding="utf-8")
    try:
        preview = await Previewer(executor).preview(
            "fs_write", {"path": "same.txt", "content": "unchanged"}
        )
        assert preview is not None and preview.destructive is False
        assert "changes nothing" in preview.summary
    finally:
        await executor.close()


class _RecordingExecutor:
    """Stands in for the sandbox so the dry-run form can be asserted without
    needing a real git repository on the test machine."""

    def __init__(self, output: str = "") -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._output = output

    async def call(self, name: str, arguments: dict[str, Any]) -> ExecResult:
        self.calls.append((name, arguments))
        return ExecResult(True, self._output, False)


async def test_destructive_command_is_simulated_with_its_dry_run_form() -> None:
    executor = _RecordingExecutor(output="Would remove build/\nWould remove tmp.log")
    preview = await Previewer(executor).preview(
        "run_command", {"command": ["git", "clean", "-fd"]}
    )
    assert preview is not None
    assert preview.destructive is True and preview.simulated is True
    # the command that ACTUALLY ran was the no-op form, never the real one
    assert executor.calls == [("run_command", {"command": ["git", "clean", "-fd", "--dry-run"]})]
    assert "Would remove build/" in preview.detail


async def test_unsimulatable_destructive_command_says_so_rather_than_pretending() -> None:
    executor = _RecordingExecutor()
    preview = await Previewer(executor).preview(
        "run_command", {"command": ["pip", "uninstall", "-y", "requests"]}
    )
    assert preview is not None
    assert preview.destructive is True
    assert preview.simulated is False  # honest: no dry run exists for this
    assert executor.calls == []  # and nothing was run to find out


async def test_forget_preview_shows_the_memory_it_would_delete(env) -> None:
    _config, db = env
    memory = MemoryStore(db, EventStore(db))
    stored = memory.remember("Pixel is the dog's name", source="user", trusted=True)

    previewer = Previewer(_RecordingExecutor(), memory)
    preview = await previewer.preview("memory_forget", {"event_id": stored.event_id})
    assert preview is not None
    assert preview.destructive is True and preview.reversible is False
    assert "Pixel is the dog's name" in preview.detail

    memory.forget(stored.event_id)
    gone = await previewer.preview("memory_forget", {"event_id": stored.event_id})
    assert gone is not None and gone.destructive is False  # nothing left to lose


async def test_a_failing_preview_still_produces_a_card_marked_unsimulated() -> None:
    class _Broken:
        async def call(self, name: str, arguments: dict[str, Any]) -> ExecResult:
            raise RuntimeError("sandbox is down")

    preview = await Previewer(_Broken()).preview(
        "fs_write", {"path": "x.txt", "content": "y"}
    )
    # The one outcome the exit criterion forbids is silence: a broken preview must
    # still reach the user, still flagged destructive.
    assert preview is not None
    assert preview.destructive is True and preview.simulated is False
    assert "sandbox is down" in preview.summary


# ---- the invariant: a session grant cannot smuggle a clobber past the user ----


def _write_call(path: str, content: str) -> ModelResponse:
    return ModelResponse(
        text="",
        tool_calls=[ToolCall(id="t1", name="fs_write",
                             arguments={"path": path, "content": content})],
    )


async def test_session_grant_covers_new_files_but_never_an_overwrite(env) -> None:
    config, db = env
    script = [
        _write_call("notes.md", "first version"),
        ModelResponse(text="wrote it", tool_calls=[]),
        _write_call("other.md", "a different, new file"),
        ModelResponse(text="wrote that too", tool_calls=[]),
        _write_call("notes.md", "SECOND version — clobbers the first"),
        ModelResponse(text="overwrote it", tool_calls=[]),
    ]
    memory = MemoryStore(db, EventStore(db))
    policy = PolicyEngine(db)
    approvals = ApprovalRegistry()
    executor = ExecutorClient(config.workspace)
    orch = Orchestrator(
        config, memory, TraceStore(db), policy,
        ModelGateway(FakeAdapter(script)), executor, approvals,
    )

    async def turn(text: str, scope: str) -> int:
        """Run one turn, approving anything asked. Returns how many cards appeared."""
        cards = 0
        async for ev in orch.run_turn("s1", text):
            if ev.kind == "approval_request":
                cards += 1
                assert ev.data["preview"]["simulated"] is True
                approvals.resolve(ev.data["approval_id"], approved=True, scope=scope)
        return cards

    try:
        # 1. first write: asked, and granted for the whole session
        assert await turn("write notes", scope="session") == 1
        # 2. a NEW file rides the grant — the grant still means something
        assert await turn("write another file", scope="once") == 0
        # 3. overwriting the first file comes back to the user despite the grant
        cards = await turn("rewrite notes", scope="once")
    finally:
        await executor.close()

    assert cards == 1, "a destructive overwrite must re-ask even under a session grant"
    assert (config.workspace / "notes.md").read_text(encoding="utf-8") == (
        "SECOND version — clobbers the first"
    )
    # the escalation is in the audit chain, not just in the UI
    rows = db.conn.execute(
        "SELECT reason FROM audit WHERE decision = 'ask' AND reason LIKE '%session grant does not%'"
    ).fetchall()
    assert len(rows) == 1
    assert policy.verify_audit_chain() is True
