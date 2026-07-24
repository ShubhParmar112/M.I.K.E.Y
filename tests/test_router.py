"""The Router (sovereignty S1): which brain handles a turn.

Safety-biased: anything that might need a tool, memory op, action, or a factual
answer goes to the full operator; only clearly social turns go to the toolless
conversation brain. The one non-negotiable is the incident class — a goodbye must
never reach a brain that can touch memory.
"""

from __future__ import annotations

import pytest

from core.orchestrator.brains import CONVERSATION, OPERATOR, Router
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
        "remember that my deadline is Nov 15",
        "forget what I told you about the wifi",
        "read notes.md and summarise it",
        "run the tests",
        "what's my dog's name?",          # question → may need memory
        "how many files are in the repo?",  # question + actiony
        "fetch https://example.com",
    ],
)
def test_actionable_turns_route_to_operator(router: Router, text: str) -> None:
    assert router.route(text).brain is OPERATOR


def test_goodbye_cannot_reach_a_memory_capable_brain(router: Router) -> None:
    """The exact failure mode from the live session: a wind-down sign-off must land
    on a brain that holds no tools, so it *cannot* fire memory_forget."""
    brain = router.route("that was it for today, ttyl mikey").brain
    assert brain is CONVERSATION
    assert brain.tools == []


def test_brain_tool_scoping() -> None:
    assert CONVERSATION.tools == []
    assert len(OPERATOR.tools) == len(TOOLS)  # operator keeps the full suite
    assert {t["name"] for t in OPERATOR.tools} == {t["name"] for t in TOOLS}
