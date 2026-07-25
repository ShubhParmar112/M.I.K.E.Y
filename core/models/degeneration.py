"""Repetition-collapse detection (failure taxonomy M12: a bad generation is an
*action* failure, not a request error).

Open-weight models — Llama on Groq especially, and the local 3B fallback more so —
sometimes fall into a repetition attractor and emit a self-negating loop instead of
an answer. From a live session:

    "the correct answer is indeed 7 and 14 is not correct, but 12 and 24 is also
     not correct, the correct answer is actually 7 is not correct but 12 is not
     correct, the correct answer is 7 is not the answer but one of the answer..."

Sampling params alone don't prevent it (low temperature makes it *more* likely, by
sharpening the loop), so the adapters detect it after the fact and re-sample once.
The detector is a pure function over text — cheap, and testable offline.

Counting duplicate n-grams over the whole reply does NOT work, and the reason is
worth keeping: a legitimately templated answer scores just as high as a collapse.
A 20-entry memory dump ("You told me that X (from you, N months ago)." twenty times
over) hits the same 0.38 as the loop above, because the template really does repeat.
What separates them is *where* the repetition sits:

- a collapse repeats inside one run-on sentence, which is never how prose works;
- a template repeats across sentences, each of which is internally unique.

So the score is the worse of two locality-aware signals: repetition *within* a
single sentence, and outright duplicate sentences (which catches the other shape of
loop — "It is not 7. It is not 12. It is not 7." — where no single sentence is long).
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[\w']+")
_SENTENCE_SPLIT = re.compile(r"[.!?;\n]+")

NGRAM = 4
# A sentence shorter than this can repeat a 4-gram innocently ("as far as I can
# tell, as far as I know"); only a long run-on is diagnostic on its own.
MIN_SENTENCE_WORDS = 30
# Fragments below this are ignored when looking for duplicate sentences, so a reply
# with several "Yes." / "Correct." beats isn't read as a loop.
MIN_CLAUSE_WORDS = 4
# Duplicate-sentence detection needs enough sentences for the fraction to mean
# something; two identical lines in a three-line reply is not evidence.
MIN_SENTENCES = 6
# Healthy replies measure ~0.00–0.05 on this score (verified in tests against
# derivations, bulleted listings and templated memory dumps); the live collapse
# measures ~0.38. 0.22 sits with roughly 2x headroom on both sides.
DEFAULT_THRESHOLD = 0.22


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _duplicate_ngram_ratio(words: list[str], n: int) -> float:
    """Fraction of n-gram positions that repeat an earlier n-gram. 0.0 when the
    window is too short for the fraction to be meaningful."""
    if len(words) < n * 4:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def intra_sentence_repetition(text: str, n: int = NGRAM) -> float:
    """Worst n-gram repetition found *inside* any one long sentence.

    This is the signal that fires on a loop and stays quiet on a template, because
    a template's repetition lives across sentence boundaries, not within one.
    """
    worst = 0.0
    for sentence in _SENTENCE_SPLIT.split(text):
        words = _words(sentence)
        if len(words) < MIN_SENTENCE_WORDS:
            continue
        worst = max(worst, _duplicate_ngram_ratio(words, n))
    return worst


def duplicate_sentence_ratio(text: str) -> float:
    """Fraction of sentences that repeat an earlier sentence verbatim.

    Catches the loop shape that hides from `intra_sentence_repetition` by using
    short sentences: "It is not 7. It is not 12. It is not 7. It is not 12."
    """
    sentences = [
        " ".join(w)
        for w in (_words(s) for s in _SENTENCE_SPLIT.split(text))
        if len(w) >= MIN_CLAUSE_WORDS
    ]
    if len(sentences) < MIN_SENTENCES:
        return 0.0
    return 1.0 - len(set(sentences)) / len(sentences)


def collapse_score(text: str) -> float:
    """How much this reply looks like a repetition collapse: 0.0 = not at all.

    Also used to compare two samples, so it must be a plain magnitude rather than
    a verdict — the caller keeps whichever generation scores lower.
    """
    return max(intra_sentence_repetition(text), duplicate_sentence_ratio(text))


# The other way a reply runs away: it finishes a correct derivation, then announces
# a fresh attempt and keeps going until the token cap. Observed live on the same
# problem — a verified "the numbers are 35 and 71", followed by "let's try another
# approach" and several hundred tokens of guessed pairs. Everything after the restart
# is not an answer, so it is cut.
_RESTART_RE = re.compile(
    r"^\s*(?:but |however,? |now,? )?(?:also,? )?"
    r"(?:let'?s|let me|we (?:should|can|could|will)|i'?ll|i will)\s+"
    r"(?:also\s+)?(?:try|check|solve|re-?solve|re-?evaluate|re-?examine|look for|find|"
    r"attempt|test|consider|verify)\b"
    r".*?\b(?:another|again|other|different|yet another|one more|next)\b",
    re.IGNORECASE | re.MULTILINE,
)
# A restart only counts once the reply has actually delivered something. Without
# this, "let's check another condition" mid-derivation would truncate a good answer.
_ANSWERED_RE = re.compile(
    r"\b(?:final answer|the answer is|the numbers are|so the (?:answer|numbers|value)|"
    r"therefore|verification|both conditions hold|this condition holds|checks? out)\b",
    re.IGNORECASE,
)
# Keep nothing shorter than this — if the "answer" before the restart is a stub, the
# reply is broken in a way trimming cannot fix, so leave it whole for the caller.
MIN_KEPT_CHARS = 200


def trim_restart(text: str) -> tuple[str, bool]:
    """Cut a reply at the point it abandons a finished answer and starts over.

    Returns `(text, trimmed)`. Only fires when the part being kept already contains a
    stated answer or a verification, so a mid-derivation "let's check the other
    condition" is left alone.
    """
    if "```" in text:
        return text, False
    for match in _RESTART_RE.finditer(text):
        head = text[: match.start()].rstrip()
        if len(head) >= MIN_KEPT_CHARS and _ANSWERED_RE.search(head):
            return head, True
    return text, False


def is_degenerate(text: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """True when `text` looks like a repetition collapse rather than an answer.

    Deliberately conservative: a false positive costs one extra model call (the
    caller re-samples and keeps whichever reply scores lower), while a false
    negative ships word salad to the user. Fenced code is exempt — repeated lines
    are normal there, and re-sampling code on a prose heuristic would do more harm
    than good.
    """
    if "```" in text:
        return False
    return collapse_score(text) >= threshold
