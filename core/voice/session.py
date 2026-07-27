"""The voice loop's decisions, separated from its plumbing.

Everything here is about *what should happen* when M.I.K.E.Y is being spoken to —
which is testable with a fake microphone and a fake mouth, and is where the rules
that matter live. The CLI supplies the real audio.

Two rules are worth stating plainly, because they are safety properties rather
than preferences:

1. **A spoken word never approves an action.** A voice in the room is not proof of
   who is in the room: a television, a housemate, or a video call can all say
   "yes". So when an action needs approval, M.I.K.E.Y says out loud what it wants
   to do and then waits for the keyboard. Voice widens what you can ask for; it
   does not widen what can be authorised.
2. **Silence is not a question.** A transcriber handed a cough or a closing door
   returns confident, plausible text ("Thank you." is Whisper's favourite). Sending
   that as a turn spends quota and puts words in the person's mouth. Anything not
   clearly speech is dropped, and said to be dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# Whisper's stock hallucinations on near-silence. Matched whole, case- and
# punctuation-insensitively: a real turn is never exactly one of these, and a
# person who genuinely says only "thank you" gets a reply from the next thing
# they say rather than a wrong answer to a door closing.
_PHANTOMS = frozenset(
    {
        "thank you", "thanks", "you", "bye", "okay", "ok", "yeah", "uh", "um",
        "thank you for watching", "thanks for watching", "please subscribe",
        "subscribe", "the end", "music", "applause", "silence", "beep",
    }
)
_STRIP = re.compile(r"[\s.,!?\-–—\"'()\[\]]+")


class Decision(Enum):
    """What to do with something that was heard."""

    SPEAK_TO_MIKEY = "turn"  # send it as a turn
    IGNORE = "ignore"  # not speech, or not meant for us
    STOP_TALKING = "barge_in"  # cut the current reply off
    END_SESSION = "quit"


@dataclass(frozen=True)
class Utterance:
    text: str
    decision: Decision

    @property
    def actionable(self) -> bool:
        return self.decision is Decision.SPEAK_TO_MIKEY


# Matched against the punctuation-stripped form, so "that's" arrives as "that s".
_QUIT = re.compile(
    r"^(good ?bye|bye|good ?night|stop listening|exit|quit|"
    r"that ?'?s all( for now)?)\b",
    re.IGNORECASE,
)
_HUSH = re.compile(r"^(stop|shush|quiet|be quiet|shut up|enough|never ?mind)\b", re.IGNORECASE)


def _bare(text: str) -> str:
    return _STRIP.sub(" ", text).strip().lower()


def interpret(heard: str, *, speaking: bool = False) -> Utterance:
    """Decide what a transcription means.

    `speaking` says whether M.I.K.E.Y is talking right now — which changes the
    reading of a short interjection entirely. "Stop" said into silence is a strange
    request; "stop" said over a reply is someone asking it to shut up, and it has
    to work immediately or the voice is unbearable.
    """
    text = heard.strip()
    bare = _bare(text)
    if not bare:
        return Utterance(text, Decision.IGNORE)
    if bare in _PHANTOMS and len(bare.split()) <= 3:
        return Utterance(text, Decision.IGNORE)
    if _QUIT.match(bare):
        return Utterance(text, Decision.END_SESSION)
    if speaking and _HUSH.match(bare):
        return Utterance(text, Decision.STOP_TALKING)
    return Utterance(text, Decision.SPEAK_TO_MIKEY)


def approval_announcement(tool: str, reason: str = "", destructive: bool = False) -> str:
    """What M.I.K.E.Y says aloud when an action needs approving.

    It reads the request out and then points at the keyboard. Saying "shall I?"
    would invite a spoken yes, which is exactly the thing that must not authorise
    it — see rule 1 above.
    """
    what = tool.replace("_", " ")
    lead = "This one would change or delete something" if destructive else "I need your approval"
    tail = f" — {reason.rstrip('.')}." if reason else "."
    return f"{lead}: {what}{tail} It's on screen; approve it there."
