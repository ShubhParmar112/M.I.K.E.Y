"""End-to-end turn-loop test: scripted model → policy ask → approval → real
sandboxed executor subprocess → final answer, with events, traces, audit all
written. This is the Gen 1 spine under test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.config import Config
from core.events.schema import Event, EventType, Provenance, Tier, ulid
from core.events.store import EventStore
from core.executor_client import ExecutorClient
from core.memory.store import MemoryStore
from core.models.fake_adapter import FakeAdapter
from core.models.gateway import ModelGateway, ModelResponse, ToolCall
from core.orchestrator.critic import Critic
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


async def test_remember_then_recall_across_turns(env) -> None:
    """The Jarvis loop: the user tells M.I.K.E.Y a fact, it persists it with
    memory_remember (no approval — it only touches its own state), and a later
    turn retrieves it with memory_recall instead of shelling out to the CLI."""
    config, db = env
    script = [
        # turn 1: remember a durable fact
        ModelResponse(text="", tool_calls=[
            ToolCall(id=ulid(), name="memory_remember",
                     arguments={"text": "The user's dog is named Pixel."})
        ]),
        ModelResponse(text="Got it — I'll remember that.", tool_calls=[]),
        # turn 2: recall it
        ModelResponse(text="", tool_calls=[
            ToolCall(id=ulid(), name="memory_recall", arguments={"query": "dog name"})
        ]),
        ModelResponse(text="Your dog is named Pixel.", tool_calls=[]),
    ]
    orch, events, _traces, policy, approvals, executor = _orchestrator(config, db, script)

    try:
        approvals_asked = 0
        async for ev in orch.run_turn("s1", "remember my dog is named Pixel"):
            if ev.kind == "approval_request":
                approvals_asked += 1
        # remembering a fact on a clean turn is auto-allowed
        assert approvals_asked == 0
        assert EventType.MEMORY_NOTE.value in [e.type for e in events.recent()]

        recall_output = ""
        async for ev in orch.run_turn("s1", "what is my dog's name?"):
            if ev.kind == "action_result" and ev.data["tool"] == "memory_recall":
                recall_output = ev.data["output"]
    finally:
        await executor.close()

    # the persisted fact came back through the recall tool, carrying provenance
    assert "Pixel" in recall_output
    assert "trusted" in recall_output
    assert policy.verify_audit_chain() is True


async def test_memory_forget_requires_approval(env) -> None:
    """Forgetting is destructive, so the model's memory_forget must surface an
    approval card; approving it verifiably removes the memory."""
    config, db = env
    memory = MemoryStore(db, EventStore(db))
    note = memory.remember("The wifi password is hunter2.")
    script = [
        ModelResponse(text="", tool_calls=[
            ToolCall(id=ulid(), name="memory_forget", arguments={"event_id": note.event_id})
        ]),
        ModelResponse(text="Done — I've forgotten it.", tool_calls=[]),
    ]
    traces = TraceStore(db)
    policy = PolicyEngine(db)
    approvals = ApprovalRegistry()
    executor = ExecutorClient(config.workspace)
    orch = Orchestrator(
        config, memory, traces, policy, ModelGateway(FakeAdapter(script)), executor, approvals
    )

    asked = False
    status_brain = ""
    try:
        async for ev in orch.run_turn("s1", "forget the wifi password"):
            if ev.kind == "status":
                status_brain = ev.data["brain"]
            if ev.kind == "approval_request" and ev.data["tool"] == "memory_forget":
                asked = True
                approvals.resolve(ev.data["approval_id"], approved=True, scope="once")
    finally:
        await executor.close()

    assert status_brain == "memory"  # forgetting is routed to the dedicated memory brain
    assert asked is True
    # the note is verifiably gone (a term unique to it no longer retrieves it)
    assert memory.recall("hunter2") == []
    assert note.event_id not in [h.event_id for h in memory.recall("wifi password")]
    assert policy.verify_audit_chain() is True


async def test_operator_is_refused_if_it_tries_to_forget(env) -> None:
    """Authority is enforced, not merely un-offered (S1 memory brain): if the
    operator model emits memory_forget anyway, the loop refuses it — no approval
    card, no deletion — because only the memory brain holds that tool."""
    config, db = env
    memory = MemoryStore(db, EventStore(db))
    note = memory.remember("The wifi password is hunter2.")
    # A question routes to the operator; the (mis)behaving model tries to forget.
    script = [
        ModelResponse(text="", tool_calls=[
            ToolCall(id=ulid(), name="memory_forget", arguments={"event_id": note.event_id})
        ]),
        ModelResponse(text="I can't delete memories in this mode.", tool_calls=[]),
    ]
    traces = TraceStore(db)
    policy = PolicyEngine(db)
    approvals = ApprovalRegistry()
    executor = ExecutorClient(config.workspace)
    orch = Orchestrator(
        config, memory, traces, policy, ModelGateway(FakeAdapter(script)), executor, approvals
    )

    turn_id = ""
    status_brain = ""
    approvals_asked = 0
    refused = False
    try:
        async for ev in orch.run_turn("s1", "what's the wifi password?"):
            if ev.kind == "status":
                turn_id = ev.data["turn_id"]
                status_brain = ev.data["brain"]
            if ev.kind == "approval_request":
                approvals_asked += 1
            if ev.kind == "action_result" and "not available to the operator" in ev.data["output"]:
                refused = True
    finally:
        await executor.close()

    assert status_brain == "operator"     # the turn was handled by the operator
    assert refused is True                # its forget attempt was refused by authority
    assert approvals_asked == 0           # the user was never even asked
    assert memory.recall("hunter2")       # the memory is intact
    assert "authority_denied" in [s["kind"] for s in traces.turn(turn_id)]


async def test_ingest_tool_reads_a_file_and_requires_approval(env, tmp_path: Path) -> None:
    """M.I.K.E.Y can ingest a document from a path (even outside the workspace)
    via the ingest tool — approval-gated — and its contents become recallable."""
    config, db = env
    doc = tmp_path / "paper.md"
    doc.write_text("The Zorblax theorem proves quibits stabilize at dawn.", encoding="utf-8")
    memory = MemoryStore(db, EventStore(db))
    script = [
        ModelResponse(text="", tool_calls=[
            ToolCall(id=ulid(), name="ingest", arguments={"path": str(doc)})
        ]),
        ModelResponse(text="Ingested — ask me anything about it.", tool_calls=[]),
    ]
    traces = TraceStore(db)
    policy = PolicyEngine(db)
    approvals = ApprovalRegistry()
    executor = ExecutorClient(config.workspace)
    orch = Orchestrator(
        config, memory, traces, policy, ModelGateway(FakeAdapter(script)), executor, approvals
    )

    asked = False
    try:
        async for ev in orch.run_turn("s1", "ingest my paper"):
            if ev.kind == "approval_request" and ev.data["tool"] == "ingest":
                asked = True
                approvals.resolve(ev.data["approval_id"], approved=True, scope="once")
    finally:
        await executor.close()

    assert asked is True  # reading a file off disk is approval-gated
    zorblax = [h for h in memory.recall("Zorblax theorem quibits") if "Zorblax" in h.text]
    assert zorblax  # the document's contents are now in memory
    assert all(not h.trusted for h in zorblax)  # and correctly marked untrusted


async def test_recall_of_untrusted_memory_taints_the_turn(env) -> None:
    """memory_recall that surfaces an untrusted memory must taint the turn, so a
    normally auto-allowed fs_read later in the same turn escalates to approval."""
    config, db = env
    memory = MemoryStore(db, EventStore(db))
    memory.record(
        Event(
            type=EventType.INGEST_DOCUMENT.value,
            provenance=Provenance(source="connector:file:notes.md", trusted=False),
            payload={"text": "The launch codeword is Bluebird."},
        )
    )
    adapter = FakeAdapter([
        ModelResponse(text="", tool_calls=[
            ToolCall(id=ulid(), name="memory_recall", arguments={"query": "launch codeword"})
        ]),
        ModelResponse(text="", tool_calls=[
            ToolCall(id=ulid(), name="fs_read", arguments={"path": "x.txt"})
        ]),
        ModelResponse(text="Done.", tool_calls=[]),
    ])
    traces = TraceStore(db)
    policy = PolicyEngine(db)
    approvals = ApprovalRegistry()
    executor = ExecutorClient(config.workspace)
    orch = Orchestrator(
        config, memory, traces, policy, ModelGateway(adapter), executor, approvals
    )

    read_needed_approval = False
    try:
        async for ev in orch.run_turn("s1", "what's the codeword?"):
            if ev.kind == "approval_request" and ev.data["tool"] == "fs_read":
                read_needed_approval = True
                approvals.resolve(ev.data["approval_id"], approved=True, scope="once")
    finally:
        await executor.close()

    assert read_needed_approval is True


async def test_related_memory_hint_steers_away_from_forget(env) -> None:
    """Regression for the live memory_forget cascade: when a new fact is similar
    to an existing one, the tool result must point the model at `supersedes` — never
    dangle the related ids as things to delete — and must forbid unsolicited forgets."""
    config, db = env
    memory = MemoryStore(db, EventStore(db))
    seed = memory.remember("Pixel is the user's pet dog.")
    traces = TraceStore(db)
    policy = PolicyEngine(db)
    approvals = ApprovalRegistry()
    executor = ExecutorClient(config.workspace)
    orch = Orchestrator(
        config, memory, traces, policy, ModelGateway(FakeAdapter([])), executor, approvals
    )
    try:
        # Moderately similar (Jaccard ~0.42) → "related", not a duplicate.
        result = orch._call_inprocess_tool(
            "memory_remember",
            {"text": "The user's dog Pixel likes to run in the park."},
            False,
            "t1",
        )
    finally:
        await executor.close()

    out = result.output
    assert seed.event_id in out               # the related memory is surfaced...
    assert "supersedes" in out                # ...as something to supersede, not delete
    assert "do NOT use memory_forget" in out
    assert "unless the user explicitly asks" in out

    # Defence in depth: the same standing rule lives in the system prompt.
    from core.context.assembly import SYSTEM_PROMPT

    assert "memory_forget" in SYSTEM_PROMPT and "explicitly asks" in SYSTEM_PROMPT


async def test_social_turn_uses_toolless_conversation_brain(env) -> None:
    """S1 routing: a pure sign-off is handled by the conversation brain — its own
    prompt is used, the routing is traced, the event is tagged, and it gets a
    single-shot reply with NO tools or actions (the memory_forget incident class)."""
    config, db = env
    fake = FakeAdapter([ModelResponse(text="Anytime — catch you later, Shubh.", tool_calls=[])])
    memory = MemoryStore(db, EventStore(db))
    traces = TraceStore(db)
    policy = PolicyEngine(db)
    approvals = ApprovalRegistry()
    executor = ExecutorClient(config.workspace)
    orch = Orchestrator(config, memory, traces, policy, ModelGateway(fake), executor, approvals)

    turn_id = ""
    status_brain = ""
    finals: list[str] = []
    try:
        async for ev in orch.run_turn("s1", "that was it for today, ttyl mikey"):
            if ev.kind == "status":
                turn_id = ev.data["turn_id"]
                status_brain = ev.data["brain"]
            if ev.kind == "final":
                finals.append(ev.data["text"])
    finally:
        await executor.close()

    assert status_brain == "conversation"
    assert finals == ["Anytime — catch you later, Shubh."]
    # the conversation brain's own (toolless) prompt was used, not the operator's
    assert "casual conversation" in fake.systems[0]
    # routing is recorded in the trace ("why did it go there?")
    route_spans = [s for s in traces.turn(turn_id) if s["kind"] == "route"]
    assert route_spans and route_spans[0]["payload"]["brain"] == "conversation"
    # nothing was executed, and the assistant event is tagged with its brain
    assert EventType.ACTION_EXECUTED.value not in [e.type for e in memory.events.recent()]
    asst = [e for e in memory.events.recent() if e.type == EventType.ASSISTANT_MESSAGE.value]
    assert asst and asst[0].payload.get("brain") == "conversation"


async def test_critic_reviews_a_risky_action_before_approval(env) -> None:
    """S1 critic: a proposed ASK action is reviewed by an independent brain, and its
    verdict rides on the approval card — here a CONCERN is surfaced before the user
    decides. The critic runs on its own gateway, so it doesn't touch the operator's
    script."""
    config, db = env
    script = [
        ModelResponse(text="", tool_calls=[
            ToolCall(id=ulid(), name="fs_write", arguments={"path": "x.txt", "content": "hi"})
        ]),
        ModelResponse(text="ok, leaving it.", tool_calls=[]),
    ]
    memory = MemoryStore(db, EventStore(db))
    traces = TraceStore(db)
    policy = PolicyEngine(db)
    approvals = ApprovalRegistry()
    executor = ExecutorClient(config.workspace)
    critic = Critic(ModelGateway(FakeAdapter(
        [ModelResponse(text="CONCERN: writes a file the user never asked to create", tool_calls=[])]
    )))
    orch = Orchestrator(
        config, memory, traces, policy, ModelGateway(FakeAdapter(script)),
        executor, approvals, critic=critic,
    )

    turn_id = ""
    card: dict[str, Any] = {}
    try:
        async for ev in orch.run_turn("s1", "write a file"):
            if ev.kind == "status":
                turn_id = ev.data["turn_id"]
            if ev.kind == "approval_request":
                card = ev.data
                approvals.resolve(ev.data["approval_id"], approved=False, scope="once")
    finally:
        await executor.close()

    assert card.get("critic_sound") is False
    assert "never asked" in card.get("critic_note", "")
    assert "critic" in [s["kind"] for s in traces.turn(turn_id)]  # traced too
    assert not (config.workspace / "x.txt").exists()  # user denied → nothing written


async def test_private_turn_is_classified_t0_and_kept_on_device(env) -> None:
    """S3: a turn with private data is Tier-0 — served by the local model (never the
    cloud) and tagged T0 on the log (so the exporter excludes it from cloud training)."""
    config, db = env

    class _Cloud:  # a cloud adapter that must never be reached on a T0 turn
        name = "cloud"
        local = False

        async def complete(self, system, messages, tools) -> ModelResponse:  # noqa: ANN001
            raise AssertionError("Tier-0 data must not reach the cloud")

    local = FakeAdapter([ModelResponse(text="Noted privately.", tool_calls=[])])  # local=True
    gateway = ModelGateway(_Cloud(), fallbacks=[local])
    memory = MemoryStore(db, EventStore(db))
    traces = TraceStore(db)
    policy = PolicyEngine(db)
    approvals = ApprovalRegistry()
    executor = ExecutorClient(config.workspace)
    orch = Orchestrator(config, memory, traces, policy, gateway, executor, approvals)

    finals: list[str] = []
    tier_in_status = ""
    try:
        async for ev in orch.run_turn("s1", "my banking password is hunter2, keep it safe"):
            if ev.kind == "status":
                tier_in_status = ev.data["tier"]
            if ev.kind == "final":
                finals.append(ev.data["text"])
    finally:
        await executor.close()

    assert tier_in_status == "T0"
    assert finals == ["Noted privately."]
    assert gateway.last_provider == "fake"  # served locally; the cloud was never called
    user_ev = next(e for e in memory.events.recent() if e.type == EventType.USER_MESSAGE.value)
    assert user_ev.tier is Tier.T0


async def test_benign_turn_stays_t1_and_uses_cloud(env) -> None:
    config, db = env

    class _Cloud:
        name = "cloud"
        local = False

        async def complete(self, system, messages, tools) -> ModelResponse:  # noqa: ANN001
            return ModelResponse(text="here you go", tool_calls=[])

    local = FakeAdapter([ModelResponse(text="local", tool_calls=[])])
    gateway = ModelGateway(_Cloud(), fallbacks=[local])
    memory = MemoryStore(db, EventStore(db))
    traces = TraceStore(db)
    policy = PolicyEngine(db)
    approvals = ApprovalRegistry()
    executor = ExecutorClient(config.workspace)
    orch = Orchestrator(config, memory, traces, policy, gateway, executor, approvals)

    finals: list[str] = []
    try:
        async for ev in orch.run_turn("s1", "what is the capital of France?"):
            if ev.kind == "final":
                finals.append(ev.data["text"])
    finally:
        await executor.close()

    assert finals == ["here you go"]  # the cloud primary served an ordinary turn
    assert gateway.last_provider == "cloud"
    user_ev = next(e for e in memory.events.recent() if e.type == EventType.USER_MESSAGE.value)
    assert user_ev.tier is Tier.T1


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
