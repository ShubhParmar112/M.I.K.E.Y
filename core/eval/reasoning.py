"""Reasoning / tool-use eval + shadow-compare harness (sovereignty S0).

This is the instrument that makes brain replacement *safe*: you cannot swap a
cloud brain for a local one you cannot measure (docs/04-intelligence-sovereignty
§8.2). It scores a single model call against the real system prompt + TOOLS —
"did the model reach for the right tool, or correctly answer with none?" — and it
can run two adapters side by side (shadow mode) to compare a candidate against the
incumbent WITHOUT promoting anything. Promotion stays a human decision until the
numbers earn it.

Deterministic and offline when driven by scripted `FakeAdapter`s (that is how the
test suite exercises it); it hits the network only when a caller passes a real
cloud/local adapter, e.g. from `mikey reasoning-eval`.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.context.assembly import SYSTEM_PROMPT
from core.models.gateway import ChatMessage, ModelAdapter, ModelResponse
from core.orchestrator.tools import TOOLS

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "evals" / "reasoning_golden.json"

# Below this fraction of cases answered, a run is reported as inconclusive rather than
# as a score. Two thirds is a judgement call, not a derived constant — it is simply
# high enough that a mostly-rate-limited run can never masquerade as a measurement.
MIN_COVERAGE = 2 / 3


@dataclass
class ReasoningCase:
    id: str
    input: str
    category: str = "general"
    expect_tool: str | None = None
    expect_no_tool: bool = False
    expect_contains: list[str] = field(default_factory=list)


@dataclass
class CaseScore:
    id: str
    category: str
    passed: bool
    tools_called: list[str]
    detail: str
    error: str | None = None


@dataclass
class ReasoningReport:
    adapter: str
    results: list[CaseScore]
    pass_rate: float
    by_category: dict[str, float]

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def errored(self) -> list[CaseScore]:
        """Cases the provider never answered — rate limits, outages, timeouts.

        Kept separate from the pass rate on purpose. A run that scored "reason 0%"
        because every call came back 429 says nothing about reasoning, and reading it
        as a quality signal is how you'd "fix" a problem that was never there.
        """
        return [r for r in self.results if r.error]

    @property
    def scored(self) -> list[CaseScore]:
        return [r for r in self.results if not r.error]

    @property
    def coverage(self) -> float:
        """Fraction of cases the provider actually answered.

        The pass rate is meaningless without this. Excluding unanswered cases from the
        rate stops a rate limit from *understating* quality, but it lets it overstate
        just as easily: a run where 14 of 15 cases 429'd reports "pass 100%" off a
        single case. Callers must not present a rate whose coverage is this thin.
        """
        return len(self.scored) / self.n if self.n else 0.0

    @property
    def conclusive(self) -> bool:
        """Whether enough cases were answered for the rate to mean anything."""
        return self.coverage >= MIN_COVERAGE


@dataclass
class ShadowReport:
    """A candidate measured against the incumbent. Promotes nothing."""

    incumbent: ReasoningReport
    candidate: ReasoningReport
    agreement: float  # fraction of cases where both made the same tool decision
    regressions: list[str]  # incumbent passed, candidate failed
    improvements: list[str]  # candidate passed, incumbent failed


def load_reasoning_golden(path: Path = GOLDEN_PATH) -> list[ReasoningCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        ReasoningCase(
            id=c["id"],
            input=c["input"],
            category=c.get("category", "general"),
            expect_tool=c.get("expect_tool"),
            expect_no_tool=bool(c.get("expect_no_tool", False)),
            expect_contains=list(c.get("expect_contains", [])),
        )
        for c in raw["cases"]
    ]


def score_response(case: ReasoningCase, response: ModelResponse) -> CaseScore:
    tools_called = [tc.name for tc in response.tool_calls]
    called = set(tools_called)
    checks: list[bool] = []
    notes: list[str] = []

    if case.expect_tool is not None:
        ok = case.expect_tool in called
        checks.append(ok)
        notes.append(
            f"wanted {case.expect_tool}, got {sorted(called) or 'no tool'}"
        )
    if case.expect_no_tool:
        ok = len(called) == 0
        checks.append(ok)
        notes.append("wanted no tool" + (f", got {sorted(called)}" if called else ", got none"))
    if case.expect_contains:
        text = response.text.lower()
        missing = [s for s in case.expect_contains if s.lower() not in text]
        checks.append(not missing)
        if missing:
            notes.append(f"missing text {missing}")

    passed = all(checks) if checks else True
    return CaseScore(
        id=case.id,
        category=case.category,
        passed=passed,
        tools_called=tools_called,
        detail="; ".join(notes),
    )


async def run_reasoning_eval(
    adapter: ModelAdapter,
    cases: list[ReasoningCase],
    system: str = SYSTEM_PROMPT,
    tools: list[dict[str, Any]] = TOOLS,
    pace_s: float = 0.0,
) -> ReasoningReport:
    """Score `cases` against one adapter.

    `pace_s` waits between cases. A free-tier cloud key has a per-minute token budget
    that a back-to-back run exhausts, after which every remaining case returns 429 —
    a self-inflicted outage that reads exactly like a quality collapse in the table.
    Defaults to 0 so scripted/offline runs stay instant.
    """
    results: list[CaseScore] = []
    for i, case in enumerate(cases):
        if pace_s and i:
            await asyncio.sleep(pace_s)
        messages = [ChatMessage(role="user", text=case.input)]
        try:
            resp = await adapter.complete(system, messages, tools)
        except Exception as exc:  # ModelUnavailable or any provider fault → a failed case, never a crash
            results.append(
                CaseScore(case.id, case.category, False, [], "adapter error", error=str(exc))
            )
            continue
        results.append(score_response(case, resp))

    # Rate over the cases the provider actually answered. An unanswered case is an
    # infrastructure fact, not a wrong answer; averaging it in understates quality and
    # invites "fixing" a problem the run never measured.
    scored = [r for r in results if not r.error]
    pass_rate = sum(r.passed for r in scored) / len(scored) if scored else 0.0
    return ReasoningReport(
        adapter=getattr(adapter, "name", "?"),
        results=results,
        pass_rate=pass_rate,
        by_category=_by_category(scored),
    )


async def shadow_compare(
    incumbent: ModelAdapter,
    candidate: ModelAdapter,
    cases: list[ReasoningCase],
    system: str = SYSTEM_PROMPT,
    tools: list[dict[str, Any]] = TOOLS,
    pace_s: float = 0.0,
) -> ShadowReport:
    """Run BOTH adapters over the same cases and compare. Never promotes."""
    inc = await run_reasoning_eval(incumbent, cases, system, tools, pace_s)
    cand = await run_reasoning_eval(candidate, cases, system, tools, pace_s)

    inc_by_id = {r.id: r for r in inc.results}
    cand_by_id = {r.id: r for r in cand.results}
    shared = [cid for cid in inc_by_id if cid in cand_by_id]

    agree = 0
    regressions: list[str] = []
    improvements: list[str] = []
    for cid in shared:
        a, b = inc_by_id[cid], cand_by_id[cid]
        if set(a.tools_called) == set(b.tools_called):
            agree += 1
        if a.passed and not b.passed:
            regressions.append(cid)
        if b.passed and not a.passed:
            improvements.append(cid)

    agreement = agree / len(shared) if shared else 0.0
    return ShadowReport(
        incumbent=inc,
        candidate=cand,
        agreement=agreement,
        regressions=regressions,
        improvements=improvements,
    )


def _by_category(results: list[CaseScore]) -> dict[str, float]:
    buckets: dict[str, list[bool]] = {}
    for r in results:
        buckets.setdefault(r.category, []).append(r.passed)
    return {cat: sum(vs) / len(vs) for cat, vs in buckets.items()}
