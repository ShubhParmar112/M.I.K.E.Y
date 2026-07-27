"""When *not* to speak.

This is the load-bearing half of proactivity. Noticing things is easy; the reason
most proactive software ends up muted is that it says true things at wrong
moments, and being muted is worse than being silent — a muted assistant is one
you have also stopped trusting.

Four rules, in the order they bite:

1. **Quiet hours.** Outside waking hours only something genuinely urgent gets
   through. Everything else waits for morning, which costs nothing: it was going
   to be read in the morning either way.
2. **A budget of interruptions.** However much is happening, there is a ceiling
   per session and per hour. Past it, the rest waits. A queue of nine things is
   not nine times as useful as the most important one — it is noise.
3. **Never mid-thought.** Nothing is volunteered while a turn is in flight or
   while the person is mid-sentence. Interrupting the thing you asked for to tell
   you something else is the single most irritating behaviour a system can have.
4. **Muted kinds stay muted.** If a kind of nudge has been dismissed repeatedly,
   stop raising it. Being told twice is a reminder; being told nine times is a
   reason to stop listening entirely.

All of it is pure: given the clock and a little state, decide. That makes the
judgement testable without waiting until 3am to find out what happens at 3am.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from core.proactive.nudge import Nudge, Urgency

# Waking hours, local. Deliberately conservative at both ends: the cost of holding
# something until 08:00 is one morning; the cost of a 06:30 interruption is that
# the feature gets turned off.
WAKING_START_HOUR = 8
WAKING_END_HOUR = 22

# At most this many volunteered remarks in one sitting, and this many per hour.
MAX_PER_SESSION = 3
MAX_PER_HOUR = 4

# After this many dismissals of the same kind, stop raising it at all.
MUTE_AFTER_DISMISSALS = 3


@dataclass
class Attention:
    """What is known about the person's availability right now.

    Defaults describe the ordinary case — awake, not mid-turn, nothing said yet —
    so a caller only sets what it actually knows.
    """

    at: datetime
    busy: bool = False  # a turn is running, or they are mid-sentence
    said_this_session: int = 0
    recent_deliveries: list[datetime] = field(default_factory=list)
    dismissals_by_kind: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    speak: bool
    reason: str


def is_quiet_hours(at: datetime) -> bool:
    return not (WAKING_START_HOUR <= at.hour < WAKING_END_HOUR)


def may_speak(nudge: Nudge, attention: Attention) -> Verdict:
    """Whether this nudge may be volunteered right now, and why not if not.

    The reason is returned rather than logged privately because "why didn't you
    tell me?" deserves the same quality of answer as "why did you do that?".
    """
    urgent = Urgency.rank(nudge.urgency) >= Urgency.rank(Urgency.HIGH)

    if attention.dismissals_by_kind.get(nudge.kind, 0) >= MUTE_AFTER_DISMISSALS:
        return Verdict(False, f"you've dismissed '{nudge.kind}' notes {MUTE_AFTER_DISMISSALS}+ times")

    if attention.busy:
        # Even urgent things wait for a gap. Nothing here is worth talking over
        # the answer the person actually asked for.
        return Verdict(False, "mid-turn — will mention it at the next gap")

    if is_quiet_hours(attention.at) and not urgent:
        return Verdict(False, "outside waking hours — holding it until morning")

    if attention.said_this_session >= MAX_PER_SESSION and not urgent:
        return Verdict(False, f"already volunteered {MAX_PER_SESSION} things this session")

    hour_ago = attention.at - timedelta(hours=1)
    recent = [t for t in attention.recent_deliveries if t >= hour_ago]
    if len(recent) >= MAX_PER_HOUR and not urgent:
        return Verdict(False, f"{MAX_PER_HOUR} interruptions in the last hour already")

    return Verdict(True, "")


def choose(nudges: list[Nudge], attention: Attention) -> tuple[list[Nudge], list[tuple[Nudge, str]]]:
    """Split what is outstanding into what to say now and what to hold, applying
    the budget as it goes.

    Most urgent first, then oldest — so a ceiling of three drops the least
    pressing three, not the three that happened to be raised last.
    """
    ordered = sorted(nudges, key=lambda n: (-Urgency.rank(n.urgency), n.created))
    speak: list[Nudge] = []
    hold: list[tuple[Nudge, str]] = []
    budget = Attention(
        at=attention.at,
        busy=attention.busy,
        said_this_session=attention.said_this_session,
        recent_deliveries=list(attention.recent_deliveries),
        dismissals_by_kind=dict(attention.dismissals_by_kind),
    )
    for nudge in ordered:
        verdict = may_speak(nudge, budget)
        if verdict.speak:
            speak.append(nudge)
            budget.said_this_session += 1
            budget.recent_deliveries.append(budget.at)
        else:
            hold.append((nudge, verdict.reason))
    return speak, hold
