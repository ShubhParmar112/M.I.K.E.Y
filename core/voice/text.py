"""Turning a written reply into something worth hearing.

A reply that reads well on screen is often terrible aloud: markdown asterisks
become "star", a URL becomes ninety seconds of letters, a code block becomes
noise, and "A = P(1 + r/n)^(nt)" becomes gibberish. The screen keeps the full
text — this module decides what the voice actually says.

Two rules shape everything here:

1. **Never invent, never omit silently.** Where something is dropped, the speech
   says so ("there's a code block on screen"), so the person always knows the
   spoken version is a summary of what they can see rather than the whole of it.
2. **Short.** Being read a nine-paragraph answer is worse than being read the
   first paragraph and told where the rest is.
"""

from __future__ import annotations

import re

# About how much speech is comfortable before it should stop and defer to the
# screen — roughly 45 seconds at a normal speaking rate.
DEFAULT_MAX_CHARS = 700

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]+)`")
_URL = re.compile(r"https?://\S+|www\.\S+")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s{0,4}[-*+]\s+", re.MULTILINE)
_NUMBERED = re.compile(r"^\s{0,4}(\d+)[.)]\s+", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_RULE = re.compile(r"^\s*([-*_])\1{2,}\s*$", re.MULTILINE)
_SPACES = re.compile(r"[ \t]{2,}")
_BLANK_LINES = re.compile(r"\n{3,}")

# Arithmetic read as symbols is unintelligible, and this assistant does a lot of
# arithmetic. Each substitution is deliberately narrow: only where a symbol sits
# between things that make it unambiguously an operator, so "and/or" survives and
# "35/71" becomes "35 over 71".
_MATH: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<=[\w\)])\s*\^\s*(?=[\w\(])"), " to the power of "),
    (re.compile(r"(?<=\d)\s*/\s*(?=\d)"), " over "),
    # Only with spaces on both sides and digits either end: "200 - 171" is a
    # subtraction, "well-known" and "2026-07-27" are not.
    (re.compile(r"(?<=\d) - (?=\d)"), " minus "),
    (re.compile(r"(?<=\d) \+ (?=\d)"), " plus "),
    (re.compile(r"(?<=[\w\)])\s*[×*]\s*(?=[\w\(])"), " times "),
    (re.compile(r"(?<=[\w\)])\s*÷\s*(?=[\w\(])"), " divided by "),
    (re.compile(r"\s*≈\s*"), " is about "),
    (re.compile(r"\s*≥\s*"), " is at least "),
    (re.compile(r"\s*≤\s*"), " is at most "),
    (re.compile(r"\s*!=\s*|\s*≠\s*"), " is not "),
    (re.compile(r"(?<=[\w\)\]])\s*=\s*(?=[\w\-\(])"), " equals "),
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _strip_markup(text: str) -> tuple[str, list[str]]:
    """Remove what cannot be spoken, collecting a note for each kind removed."""
    notes: list[str] = []

    if _CODE_FENCE.search(text):
        notes.append("there's a code block on screen")
        text = _CODE_FENCE.sub(" ", text)
    if _TABLE_ROW.search(text):
        notes.append("there's a table on screen")
        text = _TABLE_ROW.sub(" ", text)
    if _URL.search(text):
        text = _URL.sub("a link", text)

    text = _RULE.sub(" ", text)
    text = _HEADING.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _EMPHASIS.sub(r"\2", text)
    # A bullet is a sentence boundary when spoken; without the full stop the voice
    # runs every item together into one breathless clause.
    text = _BULLET.sub("", text)
    text = _NUMBERED.sub("", text)
    return text, notes


def _verbalize_math(text: str) -> str:
    for pattern, replacement in _MATH:
        text = pattern.sub(replacement, text)
    return text


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    """Cut at a sentence boundary at or before `max_chars`. Returns (text, cut)."""
    if len(text) <= max_chars:
        return text, False
    kept: list[str] = []
    total = 0
    for sentence in _SENTENCE_END.split(text):
        if total + len(sentence) > max_chars and kept:
            break
        kept.append(sentence)
        total += len(sentence) + 1
    if not kept:  # one enormous sentence — cut it on a word instead
        return text[:max_chars].rsplit(" ", 1)[0], True
    return " ".join(kept), True


def speakable(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """The spoken form of a written reply. Empty in, empty out."""
    if not text or not text.strip():
        return ""

    body, notes = _strip_markup(text)
    body = _verbalize_math(body)
    body = _SPACES.sub(" ", body)
    body = _BLANK_LINES.sub("\n\n", body).strip()

    body, cut = _truncate(body, max_chars)
    if cut:
        notes.append("the rest is on screen")

    if not body.strip():
        # Everything spoken-unfriendly: say what it was rather than nothing at all,
        # which would read as M.I.K.E.Y having ignored the question.
        return notes[0].capitalize() + "." if notes else ""
    if notes:
        tail = "; ".join(notes)
        separator = " " if body.endswith((".", "!", "?")) else ". "
        return f"{body}{separator}({tail}.)"
    return body
