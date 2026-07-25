"""The Router (sovereignty S1): which brain handles a turn.

Safety-biased: anything that might need a tool, memory op, action, or a factual
answer goes to the full operator; memory *curation* (forgetting/removing) goes to
the narrow memory brain; only clearly social turns go to the toolless conversation
brain. Two non-negotiables: a goodbye must never reach a brain that can touch
memory, and only the memory brain may forget.
"""

from __future__ import annotations

import pytest

from core.orchestrator.brains import CONVERSATION, MEMORY, OPERATOR, REASONING, Router
from core.orchestrator.tools import TOOLS


@pytest.fixture
def router() -> Router:
    return Router()


@pytest.mark.parametrize(
    "text",
    [
        "hey mikey, morning",
        "thanks, appreciate it",
        "ok cool",
        "goodnight mikey",
        "yeah so that was it for this conversation ig, will ttyl mikey",  # the live incident
    ],
)
def test_social_turns_route_to_conversation(router: Router, text: str) -> None:
    assert router.route(text).brain is CONVERSATION


@pytest.mark.parametrize(
    "text",
    [
        "remember that my deadline is Nov 15",  # appending a fact stays on operator
        "read notes.md and summarise it",
        "run the tests",
        "what's my dog's name?",             # question → may need memory
        "how many files are in the repo?",   # question + actiony
        "fetch https://example.com",
        "delete the temp file in the workspace",  # file delete, NOT memory forget
    ],
)
def test_actionable_turns_route_to_operator(router: Router, text: str) -> None:
    assert router.route(text).brain is OPERATOR


@pytest.mark.parametrize(
    "text",
    [
        "forget what I told you about the wifi",
        "forget my old address",
        "delete that note about the meeting",
        "erase what you remember about my password",
        "unremember the deadline",
    ],
)
def test_memory_curation_routes_to_memory_brain(router: Router, text: str) -> None:
    assert router.route(text).brain is MEMORY


def test_goodbye_cannot_reach_a_memory_capable_brain(router: Router) -> None:
    """The exact failure mode from the live session: a wind-down sign-off must land
    on a brain that holds no tools, so it *cannot* fire memory_forget."""
    brain = router.route("that was it for today, ttyl mikey").brain
    assert brain is CONVERSATION
    assert brain.tools == []


def test_only_the_memory_brain_may_forget() -> None:
    """memory_forget authority is exclusive to the memory brain."""
    op_tools = {t["name"] for t in OPERATOR.tools}
    assert "memory_forget" not in op_tools                    # operator can never forget
    assert op_tools == {t["name"] for t in TOOLS} - {"memory_forget"}

    mem_tools = {t["name"] for t in MEMORY.tools}
    assert mem_tools == {"memory_recall", "memory_remember", "memory_forget"}

    assert CONVERSATION.tools == []


# --- the reasoning brain (closed problems) ----------------------------------
#
# The live failure: a ratio word problem reached the full operator (because
# _ACTIONY matches "find"), which answered it by firing memory_recall and then
# guessing "12 and 24" — the right answer is 35 and 71. A closed problem needs
# derivation, not a lookup, so it goes to a brain with no tools at all.

LIVE_PROBLEM = (
    "if 1 is added to each of two certain numbers, their ratio is 1:2; and if 5 is "
    "subtracted from each of the two numbers their ratio becomes 5:11. Find the numbers"
)


@pytest.mark.parametrize(
    "text",
    [
        LIVE_PROBLEM,
        "solve for x: 3(x - 4) = 2x + 7",
        "A father is three times as old as his son. In 12 years he will be twice as old "
        "as his son. How old are they now?",
        "what's the probability of drawing two aces from a standard deck?",
        "simplify (x^2 - 9) / (x + 3)",
        "if a train covers 240 km in 3 hours, what is the average speed?",
    ],
)
def test_closed_problems_route_to_reasoning(router: Router, text: str) -> None:
    assert router.route(text).brain is REASONING


def test_the_reasoning_brain_holds_no_tools() -> None:
    """Toolless for the same reason conversation is: the failure being fixed IS a
    tool call. A brain with no tools cannot substitute a lookup for thinking."""
    assert REASONING.tools == []


@pytest.mark.parametrize(
    "text",
    [
        "solve the equations in problem.txt",          # the problem is in a file
        "read notes.md and work out the average",      # ditto
        "run the script and tell me the ratio it prints",
        "what did I tell you the deadline ratio was?",  # memory, not derivation
        "fetch https://example.com/puzzle and solve it",
    ],
)
def test_a_problem_needing_the_world_stays_on_operator(router: Router, text: str) -> None:
    """The reasoning brain is toolless, so a turn that needs a file, a command, the
    network or memory must never be stranded there — however mathematical it sounds."""
    assert router.route(text).brain is OPERATOR


@pytest.mark.parametrize(
    "text",
    [
        "are you sure about that, and what about 35 and 71?",  # the live follow-up
        "so what's the correct answer?",
        "show me the working",
        "why?",
        "that's wrong",
        "double-check it please",
    ],
)
def test_follow_ups_stay_with_the_brain_that_has_the_derivation(
    router: Router, text: str
) -> None:
    """A problem and the challenge that follows it are one piece of work. Without the
    hint these carry none of the problem's vocabulary and switch brains mid-derivation
    — which is how the live session ended up thrashing."""
    assert router.route(text, last_brain="reasoning").brain is REASONING


@pytest.mark.parametrize(
    "text",
    [
        "are you sure about that?",
        "so what's the correct answer?",
    ],
)
def test_the_same_follow_ups_do_not_pull_other_turns_into_reasoning(
    router: Router, text: str
) -> None:
    """Stickiness applies only after a reasoning turn; with no such history these are
    ordinary questions for the capable brain."""
    assert router.route(text).brain is OPERATOR
    assert router.route(text, last_brain="operator").brain is OPERATOR


def test_a_real_request_escapes_a_sticky_reasoning_session(router: Router) -> None:
    """Stickiness must not trap a session: anything needing a tool leaves at once."""
    assert router.route("ok now read notes.md", last_brain="reasoning").brain is OPERATOR
    assert (
        router.route("forget what I told you about the wifi", last_brain="reasoning").brain
        is MEMORY
    )


def test_routing_without_the_hint_is_unchanged(router: Router) -> None:
    """`last_brain` is optional, so callers that don't track it behave exactly as
    before — the hint may only add stickiness, never change a first-turn decision."""
    for text in (LIVE_PROBLEM, "hey mikey", "read notes.md", "forget my old address"):
        assert router.route(text).brain is router.route(text, last_brain=None).brain


@pytest.mark.parametrize(
    "text",
    [
        "hey mikey",
        "thanks!",
        "ok cool",
        "night mikey, ttyl",
    ],
)
def test_a_social_turn_never_sticks_to_reasoning(router: Router, text: str) -> None:
    """Signing off after a solved problem is a goodbye, not a follow-up. Without this
    the short-turn rule would hand "thanks, night" to the reasoning brain."""
    assert router.route(text, last_brain="reasoning").brain is CONVERSATION


def test_a_social_opener_that_is_also_a_follow_up_still_sticks(router: Router) -> None:
    """"ok" makes this look social, but it is plainly still about the problem."""
    assert router.route("ok so what's the answer then?", last_brain="reasoning").brain is REASONING
