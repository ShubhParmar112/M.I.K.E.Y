"""Deterministic arithmetic auditing of a model's own working.

A language model writing "(0.92 × 34,240) = Rs. 17,940 (True)" is not doing
arithmetic, it is producing text shaped like arithmetic. Live, on the local 3B:

    M - (16,300) = 17,940  →  M = 34,240
    Verify: (0.92 × 34,240) = Rs. 17,940 (True)      ← actually 31,500.80
            (16,300 + 3,176) = Rs. 17,940            ← actually 19,476

Every step is asserted and the "verification" confirms itself. No prompt fixes
this and no amount of self-checking catches it, because the model doing the check
is the model that got it wrong.

But a computer can just *do the sum*. This module extracts the equalities a reply
claims and evaluates them, so a false arithmetic claim is caught mechanically —
free, instant, and identically reliable whether the answer came from a 70B or a 3B.
It is the same instinct as the rest of the system: never rely on the model's
obedience for something the orchestrator can enforce.

Deliberately narrow. Only claims of the form `<pure arithmetic> = <number>` are
checked; anything containing a symbol (`M - 16300 = 17940`) is skipped, because
the value of `M` is the model's to define. Skipping is always safe — the cost of
a miss is the status quo, while a false accusation would teach the person to
ignore the warnings.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

# Rounding shown in a derivation is normal ("= 19.63%" for 19.6319…), so a claim is
# only wrong when it is wrong by more than plausible rounding. 1% relative, with an
# absolute floor so tiny numbers don't trip on their own rounding.
REL_TOLERANCE = 0.01
ABS_TOLERANCE = 0.01
# Guards against pathological input: an expression this long is not a derivation
# step, and evaluating it is not worth the risk.
MAX_EXPR_CHARS = 120

_CLEANUP = {
    "×": "*", "·": "*", "✕": "*", "х": "*",
    "÷": "/", "−": "-", "–": "-", "—": "-",
    "≈": "=", "≃": "=", "＝": "=",
}
# Currency and unit noise that sits inside an otherwise arithmetic claim.
_NOISE_RE = re.compile(r"(?:rs\.?|inr|usd|₹|\$|£|€)\s*", re.IGNORECASE)

# An expression: starts with a digit, contains at least one operator, and holds
# nothing but digits, separators, operators and parentheses — never a letter, so a
# claim about an undefined symbol is left alone.
_EXPR = r"\(?\s*\d[\d,. ]*(?:[-+*/]\s*\(?\s*[\d,. ]*\d[\d,. ]*\)?\s*)+\)?"
_VALUE = r"\d[\d,]*(?:\.\d+)?"
_CLAIM_RE = re.compile(rf"({_EXPR})=\s*({_VALUE})")


@dataclass(frozen=True)
class BadClaim:
    claim: str  # as written, e.g. "(0.92 * 34240) = 17940"
    actual: float  # what it really evaluates to

    def __str__(self) -> str:
        return f"{self.claim} — that actually evaluates to {self.actual:,.2f}"


def _normalise(text: str) -> str:
    for bad, good in _CLEANUP.items():
        text = text.replace(bad, good)
    return _NOISE_RE.sub("", text)


def _to_number(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").strip())
    except ValueError:
        return None


def _safe_eval(expr: str) -> float | None:
    """Evaluate a pure arithmetic expression, or None if it isn't one.

    Parsed to an AST and walked with an explicit node allowlist — no name lookup,
    no calls, no attribute access — so nothing in a model's reply can execute here.
    """
    expr = expr.replace(",", "").strip()
    if not expr or len(expr) > MAX_EXPR_CHARS:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, MemoryError):
        return None

    def walk(node: ast.AST) -> float | None:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value) if isinstance(node.value, (int, float)) else None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand)
            return None if value is None else (value if isinstance(node.op, ast.UAdd) else -value)
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return None if right == 0 else left / right
            return None
        return None

    try:
        return walk(tree)
    except (OverflowError, ValueError, ZeroDivisionError):
        return None


def _matches(actual: float, claimed: float) -> bool:
    return abs(actual - claimed) <= max(ABS_TOLERANCE, abs(actual) * REL_TOLERANCE)


def arithmetic_errors(text: str, limit: int = 3) -> list[BadClaim]:
    """Arithmetic the reply claims that does not hold.

    Returns at most `limit` — the first wrong step is usually the cause and the
    rest are its consequences, so naming three is enough to be actionable without
    burying the person in a list.
    """
    if "```" in text:  # code blocks are not the model's own arithmetic
        return []
    found: list[BadClaim] = []
    for line in _normalise(text).splitlines():
        for expr, value in _CLAIM_RE.findall(line):
            claimed = _to_number(value)
            actual = _safe_eval(expr)
            if claimed is None or actual is None or _matches(actual, claimed):
                continue
            written = f"{expr.strip()} = {value.strip()}"
            found.append(BadClaim(claim=written, actual=actual))
            if len(found) >= limit:
                return found
    return found
