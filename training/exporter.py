"""The data flywheel (sovereignty S0): event log → per-brain training datasets.

M.I.K.E.Y is already generating its own training corpus every time it is used —
every turn, tool call, and remembered fact is an immutable event. This exporter
turns that log into JSONL datasets for the future local brain fleet
(docs/04-intelligence-sovereignty.md §8). It is read-only over the log and
touches no network: cloud-primary operation *is* the data-collection phase.

Two invariants make the exported data safe to train on:

- **Forgetting is respected for free.** We read through `EventStore.recent`,
  which already excludes tombstoned events — a memory the user asked to forget
  never enters a dataset.
- **Privacy tiers are respected.** Tier-0 (private) turns are excluded unless a
  caller explicitly opts in (`include_t0=True`), which is only ever done for
  strictly on-device training. A T0 turn is defined as any turn whose events
  carry tier T0.

The datasets map to the structural brains:
- `conversation.jsonl`  → Conversation brain   (input → reply)
- `tool_use.jsonl`      → Planner/Execution     (input → tool trajectory → reply)
- `memory.jsonl`        → Memory brain          (what was remembered, and how)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.events.schema import EventType, Tier
from core.events.store import EventStore

# Everything non-tombstoned; the log is small at personal scale.
_ALL_LIMIT = 1_000_000_000


@dataclass
class ExportSummary:
    out_dir: str
    conversation: int
    tool_use: int
    memory: int
    turns_seen: int
    skipped_t0_turns: int
    skipped_t0_notes: int
    files: list[str] = field(default_factory=list)


@dataclass
class _Turn:
    """Accumulated events for one turn_id, in arrival order."""

    turn_id: str
    session_id: str = ""
    user_text: str = ""
    user_source: str = "user"
    assistant_text: str = ""
    has_assistant: bool = False
    actions: list[dict[str, Any]] = field(default_factory=list)
    private: bool = False  # any constituent event was Tier-0


class TrainingExporter:
    def __init__(self, events: EventStore) -> None:
        self._events = events

    # ---- grouping ----

    def _turns(self) -> list[_Turn]:
        """Group conversation + action events by turn_id, preserving order."""
        turns: dict[str, _Turn] = {}
        order: list[str] = []
        for ev in self._events.recent(
            types=[
                EventType.USER_MESSAGE.value,
                EventType.ASSISTANT_MESSAGE.value,
                EventType.ACTION_EXECUTED.value,
            ],
            limit=_ALL_LIMIT,
        ):
            turn_id = str(ev.payload.get("turn_id", ""))
            if not turn_id:
                continue
            if turn_id not in turns:
                turns[turn_id] = _Turn(turn_id=turn_id)
                order.append(turn_id)
            t = turns[turn_id]
            if ev.tier is Tier.T0:
                t.private = True
            if ev.type == EventType.USER_MESSAGE.value:
                t.user_text = str(ev.payload.get("text", ""))
                t.session_id = str(ev.payload.get("session_id", ""))
                t.user_source = ev.provenance.source
            elif ev.type == EventType.ASSISTANT_MESSAGE.value:
                t.assistant_text = str(ev.payload.get("text", ""))
                t.has_assistant = True
            elif ev.type == EventType.ACTION_EXECUTED.value:
                t.actions.append(
                    {
                        "tool": ev.payload.get("tool"),
                        "args": ev.payload.get("args", {}),
                        "ok": bool(ev.payload.get("ok", False)),
                    }
                )
        return [turns[tid] for tid in order]

    # ---- datasets ----

    def conversation_pairs(self, include_t0: bool = False) -> tuple[list[dict[str, Any]], int]:
        rows: list[dict[str, Any]] = []
        skipped = 0
        for t in self._turns():
            if not (t.user_text and t.has_assistant and t.assistant_text):
                continue
            if t.private and not include_t0:
                skipped += 1
                continue
            rows.append(
                {
                    "turn_id": t.turn_id,
                    "session_id": t.session_id,
                    "input": t.user_text,
                    "output": t.assistant_text,
                    "source": t.user_source,
                    "tier": Tier.T0.value if t.private else Tier.T1.value,
                }
            )
        return rows, skipped

    def tool_use_trajectories(self, include_t0: bool = False) -> tuple[list[dict[str, Any]], int]:
        rows: list[dict[str, Any]] = []
        skipped = 0
        for t in self._turns():
            if not t.actions:
                continue
            if t.private and not include_t0:
                skipped += 1
                continue
            rows.append(
                {
                    "turn_id": t.turn_id,
                    "input": t.user_text,
                    "actions": t.actions,
                    "output": t.assistant_text,
                    "tier": Tier.T0.value if t.private else Tier.T1.value,
                }
            )
        return rows, skipped

    def memory_decisions(self, include_t0: bool = False) -> tuple[list[dict[str, Any]], int]:
        rows: list[dict[str, Any]] = []
        skipped = 0
        notes = self._events.recent(types=[EventType.MEMORY_NOTE.value], limit=_ALL_LIMIT)
        for ev in notes:
            if ev.tier is Tier.T0 and not include_t0:
                skipped += 1
                continue
            rows.append(
                {
                    "event_id": ev.id,
                    "text": str(ev.payload.get("text", "")),
                    "source": ev.provenance.source,
                    "trusted": ev.provenance.trusted,
                    "supersedes": ev.payload.get("supersedes", []),
                    "tier": ev.tier.value,
                }
            )
        return rows, skipped

    # ---- write ----

    def export(self, out_dir: Path | str, include_t0: bool = False) -> ExportSummary:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        turns = self._turns()
        convo, skip_convo = self.conversation_pairs(include_t0)
        tools, skip_tools = self.tool_use_trajectories(include_t0)
        mem, skip_mem = self.memory_decisions(include_t0)

        files: list[str] = []
        for name, rows in (
            ("conversation", convo),
            ("tool_use", tools),
            ("memory", mem),
        ):
            path = out / f"{name}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            files.append(str(path))

        # A T0 turn is skipped identically by conversation and tool-use passes;
        # report it once as a turn count rather than double-counting.
        skipped_t0_turns = max(skip_convo, skip_tools)
        return ExportSummary(
            out_dir=str(out),
            conversation=len(convo),
            tool_use=len(tools),
            memory=len(mem),
            turns_seen=len(turns),
            skipped_t0_turns=skipped_t0_turns,
            skipped_t0_notes=skip_mem,
            files=files,
        )
