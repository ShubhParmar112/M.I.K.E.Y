"""Repetition-collapse detection: the guard on the failure mode where an
open-weight model stops answering and starts looping.

The positive case is the verbatim reply from the live session that motivated this.
The negatives matter just as much, and one of them is the whole reason the detector
is locality-aware rather than a flat n-gram count: a templated memory dump repeats
its template as heavily as a loop repeats itself, and must not be re-sampled.
"""

from __future__ import annotations

import pytest

from core.models.degeneration import (
    DEFAULT_THRESHOLD,
    collapse_score,
    duplicate_sentence_ratio,
    intra_sentence_repetition,
    is_degenerate,
    trim_restart,
)

# Verbatim from the live transcript (2026-07-26): one run-on sentence eating itself.
COLLAPSED = (
    "After reevaluating the problem, I think the correct solution is indeed 7 and 14 is not "
    "correct, but 12 and 24 is also not correct, the correct answer is actually 7 is not correct "
    "but 12 is not correct, the correct answer is 7 is not the answer but one of the answer is 12 "
    "is not correct but 7 is not the answer the correct answer is actually 12 is not correct but "
    "one of the answer is 7 is not correct but the correct answer is actually 35 and 71 and 7 and "
    "14 is not the answer, 12 and 24 is not the answer but 7 and 14 is not correct but the correct "
    "answer is 35 and 71 and 7 is not correct but 12 is not correct but the correct answer is "
    "actually 35 and 71."
)

# The other shape of loop: no single sentence is long, but they repeat wholesale.
SHORT_SENTENCE_LOOP = (
    "It is not 7 and 14. It is not 12 and 24. It is not 7 and 14. It is not 12 and 24. "
    "The answer is not 7 and 14. It is not 12 and 24. It is not 7 and 14."
)

# What that turn should have looked like — reuses phrasing, still healthy.
GOOD_DERIVATION = (
    "Let the two numbers be x and y. The first condition says (x + 1) : (y + 1) = 1 : 2, so "
    "2(x + 1) = y + 1, which gives y = 2x + 1. The second condition says (x - 5) : (y - 5) = "
    "5 : 11, so 11(x - 5) = 5(y - 5). Substituting y = 2x + 1 into that: 11x - 55 = 5(2x + 1) - 25 "
    "= 10x - 20, so x = 35, and therefore y = 2(35) + 1 = 71. Checking both conditions against "
    "the original wording: adding 1 to each gives 36 and 72, and 36 : 72 = 1 : 2 as required; "
    "subtracting 5 from each gives 30 and 66, and 30 : 66 = 5 : 11 as required. Both conditions "
    "hold, so the numbers are 35 and 71."
)

BULLETED_LISTING = """Here are the files in your workspace:
- notes.md — 4 KB, last modified yesterday
- thesis.md — 120 KB, last modified last Tuesday
- budget.csv — 2 KB, last modified last month
- draft.txt — 8 KB, last modified today
- refs.bib — 15 KB, last modified last week
- outline.md — 3 KB, last modified two days ago
- README.md — 9 KB, last modified today
Let me know which one you want me to open."""

# The adversarial negative: M.I.K.E.Y's own provenance-annotated memory recall. Every
# entry repeats the same template, so a flat duplicate-n-gram count over the whole
# reply scores this ~0.38 — identical to COLLAPSED. It is a perfectly good answer.
TEMPLATED_MEMORY_DUMP = "\n".join(
    f"You told me that {fact} (from you, {i % 9 + 1} months ago)."
    for i, fact in enumerate(
        [
            "the deadline is November 15th",
            "your advisor is Dr Rao",
            "you prefer Groq for speed",
            "the project is called MIKEY",
            "you work best at night",
            "your lab is on the third floor",
            "you use Windows 11",
            "your thesis is on machine learning",
            "you dislike long meetings",
            "your commute takes 40 minutes",
            "you drink your coffee black",
            "you play badminton on Sundays",
            "you read before bed",
            "your favourite colour is green",
            "you have a cat named Pixel",
            "your rent is due on the 5th",
            "you would rather not use PowerPoint",
            "your gym closes at 10pm",
        ]
    )
)

HEALTHY = {
    "good derivation": GOOD_DERIVATION,
    "bulleted listing": BULLETED_LISTING,
    "templated memory dump": TEMPLATED_MEMORY_DUMP,
}


@pytest.mark.parametrize("text", [COLLAPSED, SHORT_SENTENCE_LOOP], ids=["run-on", "short-sentence"])
def test_detects_both_shapes_of_collapse(text: str) -> None:
    assert is_degenerate(text)
    assert collapse_score(text) >= DEFAULT_THRESHOLD


@pytest.mark.parametrize("name", sorted(HEALTHY))
def test_healthy_replies_are_not_flagged(name: str) -> None:
    """The fix must not reject the answers it exists to produce."""
    text = HEALTHY[name]
    assert not is_degenerate(text), f"{name} was wrongly flagged"
    assert collapse_score(text) < DEFAULT_THRESHOLD


def test_templated_output_has_real_headroom() -> None:
    """The nearest legitimate miss stays well clear of the threshold — this is the
    case that a flat whole-reply n-gram count could not distinguish from a loop."""
    assert collapse_score(TEMPLATED_MEMORY_DUMP) < DEFAULT_THRESHOLD / 2


def test_the_two_signals_catch_different_shapes() -> None:
    """Each signal covers a loop the other misses, so both are load-bearing."""
    assert intra_sentence_repetition(COLLAPSED) >= DEFAULT_THRESHOLD
    assert duplicate_sentence_ratio(COLLAPSED) < DEFAULT_THRESHOLD  # one long sentence

    assert duplicate_sentence_ratio(SHORT_SENTENCE_LOOP) >= DEFAULT_THRESHOLD
    assert intra_sentence_repetition(SHORT_SENTENCE_LOOP) == 0.0  # no long sentence


@pytest.mark.parametrize(
    "text",
    [
        "",
        "The numbers are 35 and 71.",
        "Yes — that checks out. 36:72 = 1:2 and 30:66 = 5:11, so 35 and 71 is right.",
    ],
)
def test_short_replies_are_never_flagged(text: str) -> None:
    """Too little text to tell a loop from ordinary English; guessing here would
    reject perfectly good short answers."""
    assert not is_degenerate(text)
    assert collapse_score(text) == 0.0


def test_pure_loop_scores_near_one() -> None:
    assert collapse_score("the answer is not correct " * 30) > 0.9


def test_fenced_code_is_exempt() -> None:
    """Repeated lines are normal in code, and re-sampling code on a prose heuristic
    would corrupt more than it fixes."""
    code = "Here is the test:\n```python\n" + "assert compute(x) == expected_value\n" * 12 + "```\n"
    assert collapse_score(code) >= DEFAULT_THRESHOLD  # the score does fire...
    assert not is_degenerate(code)                    # ...but the guard declines


# --- trimming a restart after the answer -------------------------------------
#
# The other runaway shape, also observed live on the same problem: a correct,
# verified derivation, then "let's try another approach" and several hundred tokens
# of guessed pairs until the token cap. Everything after the restart is not an answer.

ANSWER_THEN_RESTART = (
    "Let the two numbers be x and y. From (x + 1) : (y + 1) = 1 : 2 we get y = 2x + 1, and from "
    "(x - 5) : (y - 5) = 5 : 11 we get 11x - 5y = 30. Substituting gives x = 35 and y = 71.\n"
    "VERIFICATION: 36 : 72 = 1 : 2 and 30 : 66 = 5 : 11. Both conditions hold.\n"
    "The final answer is: the two numbers are 35 and 71.\n"
    "However we should also check another pair of numbers, let's try to solve the system again.\n"
    "Let's try x = 12 and y = 24: (12 + 1) / (24 + 1) = 13 / 25, which is not 1 / 2.\n"
    "But what about x = 6 and y = 12: (6 + 1) / (12 + 1) = 7 / 13, which is not 1 / 2.\n"
    "However, let's try another pair, x = 4 and y = 8, and see if the ratios are correct."
)


def test_a_restart_after_the_answer_is_cut() -> None:
    trimmed, cut = trim_restart(ANSWER_THEN_RESTART)
    assert cut
    assert trimmed.endswith("the two numbers are 35 and 71.")
    assert "12 and 24" not in trimmed
    assert "another pair" not in trimmed
    # what survives is the complete, verified derivation
    assert "x = 35 and y = 71" in trimmed and "Both conditions hold" in trimmed


def test_trimming_catches_what_the_collapse_score_misses() -> None:
    """Why both guards exist. A restart isn't repetitive — each guessed pair is new
    text — so it slips past the collapse score entirely while still being useless
    output. Trimming is the guard for it; re-sampling is not (and, live, re-sampling
    this shape just produced a second runaway)."""
    assert not is_degenerate(ANSWER_THEN_RESTART)
    trimmed, cut = trim_restart(ANSWER_THEN_RESTART)
    assert cut and not is_degenerate(trimmed)
    assert len(trimmed) < len(ANSWER_THEN_RESTART) / 2


def test_a_mid_derivation_next_step_is_not_a_restart() -> None:
    """"let's check the other condition" is ordinary progress through a derivation, not
    a reply starting over — trimming it would truncate a correct answer."""
    text = (
        "Let the two numbers be x and y. The first condition gives y = 2x + 1. "
        "Now let's check the other condition: 11(x - 5) = 5(y - 5), so 11x - 5y = 30. "
        "Substituting, x = 35 and y = 71. Verification: 36 : 72 = 1 : 2, and 30 : 66 = 5 : 11. "
        "The final answer is 35 and 71."
    )
    trimmed, cut = trim_restart(text)
    assert not cut and trimmed == text


def test_a_restart_before_any_answer_is_left_alone() -> None:
    """With nothing established yet, there is no good prefix to keep — leave the reply
    whole and let the collapse guard deal with it."""
    text = "Let me try another approach to this problem, since the first one did not work out."
    trimmed, cut = trim_restart(text)
    assert not cut and trimmed == text


@pytest.mark.parametrize("name", sorted(HEALTHY))
def test_healthy_replies_are_never_trimmed(name: str) -> None:
    trimmed, cut = trim_restart(HEALTHY[name])
    assert not cut and trimmed == HEALTHY[name]


def test_fenced_code_is_never_trimmed() -> None:
    text = (
        "The final answer is 35 and 71.\n```python\n# let's try another value to double-check\n"
        "assert solve() == (35, 71)\n```"
    )
    trimmed, cut = trim_restart(text)
    assert not cut and trimmed == text
