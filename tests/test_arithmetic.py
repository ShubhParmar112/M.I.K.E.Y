"""Deterministic arithmetic auditing.

From a live turn (2026-07-27, local 3B) on "sold a TV for Rs 17,940 at 8% discount
with 19.6% profit — profit if no discount?" (the answer is 30%): the model asserted
a cost price of 16,300, a marked price of 34,240, answered 20%, and then wrote a
"verification" whose own equations were false — including `(0.92 x 34,240) = Rs.
17,940 (True)`, which is 31,500.80.

No prompt and no self-check catches that, because the model doing the checking is
the one that got it wrong. A computer can just do the sum.
"""

from __future__ import annotations

import pytest

from core.models.arithmetic import arithmetic_errors

THE_LIVE_REPLY = """Let M be the Marked Price.
M - (16,300) = 17,940
M = 34,240

(B) Verify: Substitute our result back into the ORIGINAL wording of each condition.

1. Selling price is 92% of marked price:
(0.92 × 34,240) = Rs. 17,940 (True)

2. Profit percentage is 19.6%:
(16,300 + 3,176) = Rs. 17,940

The percentage of profit earned if no discount was offered is 20%."""

CORRECT_WORKING = """Cost price = 17940 / 1.196 = 15000
Marked price = 17940 / 0.92 = 19500
Profit with no discount = 19500 - 15000 = 4500
Profit percentage = (4500 / 15000) * 100 = 30%"""


def test_the_live_false_verification_is_caught() -> None:
    errors = arithmetic_errors(THE_LIVE_REPLY)
    claims = " | ".join(e.claim for e in errors)

    assert len(errors) == 2
    assert "0.92 * 34,240" in claims and "16,300 + 3,176" in claims
    assert any(abs(e.actual - 31_500.80) < 0.01 for e in errors)


def test_correct_working_raises_nothing() -> None:
    assert arithmetic_errors(CORRECT_WORKING) == []


def test_rounding_is_not_an_error() -> None:
    """A derivation that rounds as it goes is normal and must not be accused."""
    assert arithmetic_errors("= (3200 / 16300) * 100 = 19.63%") == []
    assert arithmetic_errors("200 * 0.9 * 0.95 = 171, so 200 - 171 = 29 metres.") == []


def test_claims_about_symbols_are_left_alone() -> None:
    """`M` is the model's to define — only pure arithmetic is checkable, and
    skipping is always the safe direction."""
    assert arithmetic_errors("M - (16,300) = 17,940") == []
    assert arithmetic_errors("Cost Price + (19.6/100) * Cost Price = 17940") == []


@pytest.mark.parametrize(
    "text,caught",
    [
        ("2 + 2 = 5", True),
        ("2 + 2 = 4", False),
        ("(100 - 8) / 2 = 46", False),
        ("(100 - 8) / 2 = 50", True),
        ("17940 / 0.92 = 19500", False),
        ("17940 / 0.92 = 21000", True),
        ("Rs. 1,000 + Rs. 2,000 = Rs. 3,000", False),
        ("Rs. 1,000 + Rs. 2,000 = Rs. 4,000", True),
        ("10 / 0 = 0", False),  # undefined, not "wrong" — never accuse on a div by zero
    ],
)
def test_verdicts(text: str, caught: bool) -> None:
    assert bool(arithmetic_errors(text)) is caught


def test_code_blocks_are_exempt() -> None:
    """Fenced code is not the model's own arithmetic — `x = 2 + 2` there is an
    assignment, and flagging it would be nonsense."""
    assert arithmetic_errors("```python\nassert 2 + 2 == 5\ntotal = 1 + 1 = 3\n```") == []


def test_nothing_in_a_reply_can_execute() -> None:
    """The evaluator walks an AST allowlist — no names, no calls, no attributes — so
    a reply cannot reach anything by writing it as 'arithmetic'."""
    hostile = "__import__('os').system('echo pwned') = 1\nopen('x','w') = 2\n(2).__class__ = 3"
    assert arithmetic_errors(hostile) == []


def test_only_the_first_few_errors_are_reported() -> None:
    """One wrong step usually causes the rest; naming three is actionable, naming
    thirty is noise."""
    spam = "\n".join(f"{i} + {i} = {i * 3 + 7}" for i in range(1, 20))
    assert len(arithmetic_errors(spam)) == 3
