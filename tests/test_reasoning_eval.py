"""Reasoning/tool-use eval + shadow harness (sovereignty S0).

Driven by scripted FakeAdapters so the scoring and shadow-comparison logic is
tested deterministically and offline — exactly how a local candidate will be
measured against a cloud incumbent later, minus the network.
"""

from __future__ import annotations

from core.eval.reasoning import (
    ReasoningCase,
    load_reasoning_golden,
    run_reasoning_eval,
    score_response,
    shadow_compare,
)
from core.models.fake_adapter import FakeAdapter
from core.models.gateway import ModelResponse, ModelUnavailable, ToolCall


def _tool(name: str) -> ModelResponse:
    return ModelResponse(text="", tool_calls=[ToolCall(id="t", name=name, arguments={})])


def _text(t: str) -> ModelResponse:
    return ModelResponse(text=t, tool_calls=[])


_CASES = [
    ReasoningCase(id="greet", input="hi", expect_no_tool=True, category="chat"),
    ReasoningCase(id="recall", input="what did I say?", expect_tool="memory_recall", category="memory"),
]


def test_scoring_tool_and_no_tool() -> None:
    assert score_response(_CASES[0], _text("hello there")).passed
    assert not score_response(_CASES[0], _tool("fs_read")).passed  # tool when none wanted
    assert score_response(_CASES[1], _tool("memory_recall")).passed
    assert not score_response(_CASES[1], _tool("fs_read")).passed  # wrong tool


def test_scoring_expect_contains() -> None:
    case = ReasoningCase(id="calc", input="17*3", expect_no_tool=True, expect_contains=["51"])
    assert score_response(case, _text("that's 51")).passed
    assert not score_response(case, _text("that's fifty-one")).passed


async def test_run_eval_pass_rate_and_categories() -> None:
    good = FakeAdapter(script=[_text("hey!"), _tool("memory_recall")])
    report = await run_reasoning_eval(good, _CASES)
    assert report.pass_rate == 1.0
    assert report.by_category == {"chat": 1.0, "memory": 1.0}


async def test_adapter_error_is_a_failed_case_not_a_crash() -> None:
    class _Boom:
        name = "boom"

        async def complete(self, system, messages, tools) -> ModelResponse:  # noqa: ANN001
            raise RuntimeError("provider down")

    report = await run_reasoning_eval(_Boom(), _CASES)
    assert report.pass_rate == 0.0
    assert all(r.error for r in report.results)


async def test_shadow_compare_flags_regressions_without_promoting() -> None:
    good = FakeAdapter(script=[_text("hey!"), _tool("memory_recall")])
    bad = FakeAdapter(script=[_tool("fs_read"), _text("no idea")])

    rep = await shadow_compare(incumbent=good, candidate=bad, cases=_CASES)
    assert rep.incumbent.pass_rate == 1.0
    assert rep.candidate.pass_rate == 0.0
    assert rep.agreement == 0.0
    assert set(rep.regressions) == {"greet", "recall"}
    assert rep.improvements == []


async def test_shadow_compare_flags_improvements() -> None:
    bad = FakeAdapter(script=[_tool("fs_read"), _text("no idea")])
    good = FakeAdapter(script=[_text("hey!"), _tool("memory_recall")])

    rep = await shadow_compare(incumbent=bad, candidate=good, cases=_CASES)
    assert set(rep.improvements) == {"greet", "recall"}
    assert rep.regressions == []


def test_committed_golden_set_loads_and_is_wellformed() -> None:
    cases = load_reasoning_golden()
    assert len(cases) >= 8
    for c in cases:
        assert c.input.strip()
        # each case must assert *something* to be a real test
        assert c.expect_tool or c.expect_no_tool or c.expect_contains


async def test_unanswered_cases_are_excluded_from_the_pass_rate() -> None:
    """A provider that runs out of per-minute tokens must not look like a model that
    reasons badly. This bit live: a run where 11 of 15 cases came back 429 reported
    "reason 0%", which is a statement about the rate limit and nothing else."""

    class _Flaky:
        name = "flaky"

        def __init__(self) -> None:
            self.n = 0

        async def complete(self, system, messages, tools) -> ModelResponse:  # noqa: ANN001
            self.n += 1
            if self.n == 1:
                return ModelResponse(text="hey!", tool_calls=[])
            raise ModelUnavailable("flaky", "rate limited (429)")

    report = await run_reasoning_eval(_Flaky(), _CASES)

    assert report.n == 2
    assert len(report.scored) == 1 and len(report.errored) == 1
    assert report.pass_rate == 1.0, "the one answered case passed; the 429 is not a failure"
    assert "memory" not in report.by_category, "an unanswered category is absent, not 0%"


async def test_all_unanswered_is_zero_not_a_crash() -> None:
    class _Down:
        name = "down"

        async def complete(self, system, messages, tools) -> ModelResponse:  # noqa: ANN001
            raise ModelUnavailable("down", "server error (503)")

    report = await run_reasoning_eval(_Down(), _CASES)
    assert report.pass_rate == 0.0 and report.scored == []
    assert len(report.errored) == 2


async def test_thin_coverage_is_inconclusive_not_a_score() -> None:
    """The other half of the honesty fix. Excluding unanswered cases stops a rate
    limit from understating quality — but it lets it overstate just as easily: a run
    where only one case was answered reported "pass 100%". Coverage is what makes the
    rate readable, so the report carries it."""

    class _OneThenDown:
        name = "onethendown"

        def __init__(self) -> None:
            self.n = 0

        async def complete(self, system, messages, tools) -> ModelResponse:  # noqa: ANN001
            self.n += 1
            if self.n == 1:
                return ModelResponse(text="hey!", tool_calls=[])
            raise ModelUnavailable("onethendown", "rate limited (429)")

    cases = _CASES * 5  # 10 cases, only the first is answered
    report = await run_reasoning_eval(_OneThenDown(), cases)

    assert report.pass_rate == 1.0          # every answered case passed...
    assert report.coverage == 0.1           # ...but that is one case in ten
    assert not report.conclusive, "a 10%-coverage run must not read as a score"


async def test_a_full_run_is_conclusive() -> None:
    good = FakeAdapter(script=[_text("hey!"), _tool("memory_recall")])
    report = await run_reasoning_eval(good, _CASES)
    assert report.coverage == 1.0 and report.conclusive
