"""End-to-end turn-loop test: scripted model → policy ask → approval → real
sandboxed executor subprocess → final answer, with events, traces, audit all
written. This is the Gen 1 spine under test."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import Config
from core.events.schema import Event, EventType, Provenance, ulid
from core.events.store import EventStore
from core.executor_client import ExecutorClient
from core.memory.store import MemoryStore
from core.models.fake_adapter import FakeAdapter
from core.models.gateway import ModelGateway, ModelResponse, ToolCall
from core.orchestrator.loop import ApprovalRegistry, Orchestrator
from core.policy.engine import PolicyEngine
from core.storage.db import Database
from core.trace.store import TraceStore


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MIKEY_WORKSPACE", raising=False)
    monkeypatch.setenv("MIKEY_HOME", str(tmp_path))
    config = Config(home=tmp_path)
    config.ensure_dirs()
    db = Database(config.db_path)
    return config, db


def _orchestrator(config: Config, db: Database, script: list[ModelResponse]):
    memory = MemoryStore(db, EventStore(db))
    traces = TraceStore(db)
    policy = PolicyEngine(db)
    approvals = ApprovalRegistry()
    executor = ExecutorClient(config.workspace)
    orch = Orchestrator(
        config, memory, traces, policy, ModelGateway(FakeAdapter(script)), executor, approvals
    )
    return orch, memory.events, traces, policy, approvals, executor


async def test_full_turn_with_approval(env) -> None:
    config, db = env
    script = [
        ModelResponse(
            text="",
            tool_calls=[
                ToolCall(id=ulid(), name="fs_write",
                         arguments={"path": "hello.txt", "content": "hi from mikey"})
            ],
        ),
        ModelResponse(text="Done — wrote hello.txt.", tool_calls=[]),
    ]
    orch, events, traces, policy, approvals, executor = _orchestrator(config, db, script)

    seen: list[str] = []
    final_text = ""
    turn_id = ""
    try:
        gen = orch.run_turn("s1", "write a greeting file")
        async for ev in gen:
            seen.append(ev.kind)
            if ev.kind == "status":
                turn_id = ev.data["turn_id"]
            if ev.kind == "approval_request":
                assert ev.data["tool"] == "fs_write"
                approvals.resolve(ev.data["approval_id"], approved=True, scope="once")
            if ev.kind == "final":
                final_text = ev.data["text"]
    finally:
        await executor.close()

    assert seen == ["status", "action", "approval_request", "action_result", "final"]
    assert final_text == "Done — wrote hello.txt."
    # side effect really happened, inside the workspace
    assert (config.workspace / "hello.txt").read_text() == "hi from mikey"
    # event log captured the turn
    types = [e.type for e in events.recent()]
    assert EventType.USER_MESSAGE.value in types
    assert EventType.ACTION_EXECUTED.value in types
    assert EventType.ASSISTANT_MESSAGE.value in types
    # trace answers "why": context → model_call → policy → approval → tool_call
    kinds = [s["kind"] for s in traces.turn(turn_id)]
    for expected in ("context", "model_call", "policy_decision", "approval", "tool_call"):
        assert expected in kinds
    # audit chain intact
    assert policy.verify_audit_chain() is True


async def test_denied_action_is_not_executed(env) -> None:
    config, db = env
    script = [
        ModelResponse(
            text="",
            tool_calls=[
                ToolCall(id=ulid(), name="fs_write",
                         arguments={"path": "evil.txt", "content": "nope"})
            ],
        ),
        ModelResponse(text="Understood, I won't write the file.", tool_calls=[]),
    ]
    orch, events, traces, policy, approvals, executor = _orchestrator(config, db, script)

    try:
        async for ev in orch.run_turn("s1", "write it"):
            if ev.kind == "approval_request":
                approvals.resolve(ev.data["approval_id"], approved=False, scope="once")
    finally:
        await executor.close()

    assert not (config.workspace / "evil.txt").exists()
    # denial recorded, no action.executed event
    assert EventType.ACTION_EXECUTED.value not in [e.type for e in events.recent()]


async def test_retry_of_denied_action_is_auto_denied_without_reasking(env) -> None:
    """Reproduces the live Gen 1 incident: model retries the exact action the
    user just denied. The system must auto-deny without showing a second
    approval card, and stop the turn if the model keeps insisting."""
    config, db = env
    write = ToolCall(id=ulid(), name="fs_write",
                     arguments={"path": "hello.txt", "content": "Hello, World!"})
    script = [
        ModelResponse(text="", tool_calls=[write]),                      # user denies
        ModelResponse(text="", tool_calls=[
            ToolCall(id=ulid(), name="fs_write", arguments=dict(write.arguments))
        ]),                                                              # retry -> auto-deny
        ModelResponse(text="Understood — I won't write the file.", tool_calls=[]),
    ]
    orch, events, _traces, policy, approvals, executor = _orchestrator(config, db, script)

    approval_requests = 0
    auto_denied = 0
    finals = []
    try:
        async for ev in orch.run_turn("s1", "write a greeting file"):
            if ev.kind == "approval_request":
                approval_requests += 1
                approvals.resolve(ev.data["approval_id"], approved=False, scope="once")
            if ev.kind == "action_result" and "DENIED (auto)" in ev.data["output"]:
                auto_denied += 1
            if ev.kind == "final":
                finals.append(ev.data["text"])
    finally:
        await executor.close()

    assert approval_requests == 1  # the user is never asked twice for the same denial
    assert auto_denied == 1
    assert finals == ["Understood — I won't write the file."]
    assert not (config.workspace / "hello.txt").exists()
    assert policy.verify_audit_chain() is True


async def test_persistent_retry_stops_the_turn(env) -> None:
    config, db = env
    def write() -> ToolCall:
        return ToolCall(id=ulid(), name="fs_write",
                        arguments={"path": "hello.txt", "content": "Hello, World!"})
    script = [
        ModelResponse(text="", tool_calls=[write()]),
        ModelResponse(text="", tool_calls=[write()]),
        ModelResponse(text="", tool_calls=[write()]),
        ModelResponse(text="should never be reached", tool_calls=[]),
    ]
    orch, _events, _traces, _policy, approvals, executor = _orchestrator(config, db, script)

    kinds = []
    try:
        async for ev in orch.run_turn("s1", "write it"):
            kinds.append(ev.kind)
            if ev.kind == "approval_request":
                approvals.resolve(ev.data["approval_id"], approved=False, scope="once")
    finally:
        await executor.close()

    assert kinds[-1] == "error"  # turn stopped, model never got its "final" say
    assert "final" not in kinds
    assert not (config.workspace / "hello.txt").exists()


async def test_untrusted_memory_is_injected_and_taints_the_turn(env) -> None:
    """Gen 2 spine: an ingested (untrusted) memory must (a) appear in the
    system prompt with its source, and (b) taint the turn so even auto-allowed
    reads escalate to an approval card."""
    config, db = env
    memory = MemoryStore(db, EventStore(db))
    doc = memory.record(
        Event(
            type=EventType.INGEST_DOCUMENT.value,
            provenance=Provenance(source="connector:file:seminar.md", trusted=False),
            payload={"text": "The seminar on quantum error correction is on Friday."},
        )
    )
    adapter = FakeAdapter(
        [
            ModelResponse(
                text="",
                tool_calls=[ToolCall(id=ulid(), name="fs_read", arguments={"path": "notes.txt"})],
            ),
            ModelResponse(text="Checked.", tool_calls=[]),
        ]
    )
    traces = TraceStore(db)
    policy = PolicyEngine(db)
    approvals = ApprovalRegistry()
    executor = ExecutorClient(config.workspace)
    orch = Orchestrator(
        config, memory, traces, policy, ModelGateway(adapter), executor, approvals
    )

    approval_for_read = False
    try:
        async for ev in orch.run_turn("s1", "when is the quantum seminar?"):
            if ev.kind == "approval_request" and ev.data["tool"] == "fs_read":
                approval_for_read = True
                approvals.resolve(ev.data["approval_id"], approved=True, scope="once")
    finally:
        await executor.close()

    # (a) memory injected with provenance annotation
    assert "quantum error correction" in adapter.systems[0]
    assert "connector:file:seminar.md" in adapter.systems[0]
    assert "UNTRUSTED" in adapter.systems[0]
    assert doc.id in adapter.systems[0]
    # (b) normally auto-allowed fs_read required approval because turn was tainted
    assert approval_for_read is True


async def test_plain_answer_no_tools(env) -> None:
    config, db = env
    script = [ModelResponse(text="2 + 2 = 4", tool_calls=[])]
    orch, _events, _traces, _policy, _approvals, executor = _orchestrator(config, db, script)
    finals = []
    try:
        async for ev in orch.run_turn("s1", "what is 2+2?"):
            if ev.kind == "final":
                finals.append(ev.data["text"])
    finally:
        await executor.close()
    assert finals == ["2 + 2 = 4"]
