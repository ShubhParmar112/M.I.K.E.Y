"""Voice: the decisions, not the audio.

None of this needs a microphone. What it pins is the part that decides whether
being spoken to is usable — what gets said aloud, what a stray noise is allowed to
do, and the rule that a spoken word never approves an action.
"""

from __future__ import annotations

import pytest

from core.events.schema import Tier
from core.voice.mouth import Mouth
from core.voice.session import Decision, approval_announcement, interpret
from core.voice.synth import LocalSynth, SynthUnavailable, TieredSynth, make_synth
from core.voice.text import speakable

# --- what actually gets said --------------------------------------------------


def test_a_plain_answer_is_spoken_as_written() -> None:
    assert speakable("The answer is 29 metres.") == "The answer is 29 metres."


def test_a_code_block_is_described_not_read_out() -> None:
    spoken = speakable("Here's the fix:\n\n```python\nfor i in range(10):\n    print(i)\n```\n")

    assert "range" not in spoken
    assert "code block on screen" in spoken


def test_a_url_becomes_a_link() -> None:
    spoken = speakable("I found it at https://example.com/a/very/long/path?q=1 — have a look.")

    assert "https" not in spoken
    assert "a link" in spoken


def test_markdown_is_flattened_rather_than_pronounced() -> None:
    spoken = speakable("## Findings\n\n- **first** point\n- second `value` here\n")

    assert "#" not in spoken
    assert "*" not in spoken
    assert "`" not in spoken
    assert "first point" in spoken


def test_arithmetic_is_verbalised() -> None:
    """This assistant does a lot of arithmetic, and symbols read aloud are noise."""
    spoken = speakable("So c = 180 × 190 / 200 = 171, and 200 - 171 = 29.")

    assert "equals" in spoken
    assert "times" in spoken
    assert "over" in spoken


def test_subtraction_is_verbalised_but_hyphens_are_left_alone() -> None:
    spoken = speakable("On 2026-07-27 the well-known result was 200 - 171 = 29.")

    assert "200 minus 171" in spoken
    assert "2026-07-27" in spoken, "a date is not a subtraction"
    assert "well-known" in spoken


def test_prose_punctuation_is_not_mistaken_for_arithmetic() -> None:
    """A narrow rule beats a clever one: 'and/or' must not become 'and over or'."""
    spoken = speakable("Bring a pen and/or a pencil — it's 5 minutes away.")

    assert "and/or" in spoken
    assert "over" not in spoken


def test_a_long_answer_stops_and_says_where_the_rest_is() -> None:
    long_answer = " ".join(f"This is sentence number {i}." for i in range(80))
    spoken = speakable(long_answer)

    assert len(spoken) < len(long_answer) / 2
    assert "the rest is on screen" in spoken
    assert spoken.count("sentence number") >= 1  # it still says something


def test_a_truncated_answer_is_cut_at_a_sentence() -> None:
    spoken = speakable(" ".join(f"Sentence {i} here." for i in range(200)), max_chars=100)
    body = spoken.split("(the rest")[0].strip()

    assert body.endswith(".")


def test_an_answer_that_is_only_code_still_says_something() -> None:
    """Silence would read as M.I.K.E.Y having ignored the question."""
    spoken = speakable("```\nnothing but code\n```")

    assert spoken.strip()
    assert "code block" in spoken.lower()


def test_empty_in_empty_out() -> None:
    assert speakable("") == ""
    assert speakable("   \n  ") == ""


# --- what a noise in the room is allowed to do --------------------------------


@pytest.mark.parametrize("phantom", ["Thank you.", "you", "Thanks for watching!", " Bye "])
def test_a_transcribed_room_noise_is_not_sent_as_a_turn(phantom: str) -> None:
    """A transcriber handed a cough or a closing door returns confident, plausible
    text. Sending it spends quota and puts words in the person's mouth."""
    said = interpret(phantom)

    assert not said.actionable


def test_a_real_question_is_sent() -> None:
    said = interpret("What's the compound interest on twelve thousand rupees?")

    assert said.decision is Decision.SPEAK_TO_MIKEY
    assert said.actionable


def test_stop_means_stop_only_while_it_is_talking() -> None:
    """Said over a reply it's "be quiet"; said into silence it's a strange request
    that should reach the assistant like anything else."""
    assert interpret("stop", speaking=True).decision is Decision.STOP_TALKING
    assert interpret("stop", speaking=False).decision is Decision.SPEAK_TO_MIKEY


def test_goodbye_ends_the_session() -> None:
    assert interpret("goodbye").decision is Decision.END_SESSION
    assert interpret("that's all for now").decision is Decision.END_SESSION


def test_a_long_sentence_starting_with_stop_is_still_a_request() -> None:
    said = interpret("stop the mission and tell me what it did", speaking=True)

    assert said.decision is Decision.STOP_TALKING, (
        "a leading 'stop' during speech is an interruption; the rest is said again"
    )


# --- the rule that a spoken word never approves anything ----------------------


def test_the_approval_announcement_points_at_the_keyboard() -> None:
    """It must not ask a yes/no question out loud: a television can say yes."""
    spoken = approval_announcement("fs_write", "writes notes.md", destructive=False)

    assert "approve it there" in spoken
    assert "?" not in spoken


def test_a_destructive_action_is_announced_as_destructive() -> None:
    spoken = approval_announcement("fs_write", "overwrites notes.md", destructive=True)

    assert "delete" in spoken or "change" in spoken


# --- privacy: a cloud voice never speaks a private turn -----------------------


class _FakeSynth:
    def __init__(self, name: str, local: bool, audio: bytes = b"audio") -> None:
        self.name = name
        self.local = local
        self._audio = audio
        self.spoke: list[str] = []

    def render(self, text: str) -> bytes:
        self.spoke.append(text)
        return self._audio


def test_a_private_turn_is_never_sent_to_a_cloud_voice() -> None:
    """The gateway refuses to send T0 text to a cloud model. Speaking it through a
    cloud voice would leak it out the far end of the same turn."""
    cloud = _FakeSynth("edge", local=False)
    local = _FakeSynth("local", local=True)
    voice = TieredSynth(cloud, local)

    assert voice.render("my bank balance is 12", Tier.T0) == b"audio"
    assert cloud.spoke == [], "the cloud voice must never see private text"
    assert local.spoke == ["my bank balance is 12"]
    assert voice.last_used == "local"


def test_a_normal_turn_uses_the_preferred_voice() -> None:
    cloud = _FakeSynth("edge", local=False)
    local = _FakeSynth("local", local=True)
    voice = TieredSynth(cloud, local)

    voice.render("the answer is 29", Tier.T1)

    assert cloud.spoke == ["the answer is 29"]
    assert local.spoke == []


def test_a_failing_voice_falls_through_to_the_other_one() -> None:
    class _Broken(_FakeSynth):
        def render(self, text: str) -> bytes:
            raise SynthUnavailable("no network")

    voice = TieredSynth(_Broken("edge", local=False), _FakeSynth("local", local=True))

    assert voice.render("hello") == b"audio"
    assert voice.last_used == "local"


def test_no_voice_at_all_is_silence_not_an_error() -> None:
    class _Broken(_FakeSynth):
        def render(self, text: str) -> bytes:
            raise SynthUnavailable("nothing works")

    voice = TieredSynth(_Broken("local", local=True))

    assert voice.render("hello") is None
    assert voice.last_used is None


def test_voice_off_builds_nothing() -> None:
    assert make_synth("off") is None
    assert isinstance(make_synth("local"), TieredSynth)


# --- the mouth never costs a turn ---------------------------------------------


class _SilentPlayer:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.stopped = False

    def play(self, audio: bytes, blocking: bool = True) -> None:
        self.played.append(audio)

    def stop(self) -> None:
        self.stopped = True


def test_the_mouth_speaks_the_shaped_text_not_the_raw_reply() -> None:
    synth = _FakeSynth("local", local=True)
    player = _SilentPlayer()
    mouth = Mouth(TieredSynth(synth), player)  # type: ignore[arg-type]

    assert mouth.say("Done — see ```code``` for details.")
    assert "```" not in synth.spoke[0]
    assert player.played == [b"audio"]


def test_a_voice_failure_is_reported_but_never_raised() -> None:
    class _Exploding(_FakeSynth):
        def render(self, text: str) -> bytes:
            raise OSError("the sound card is on fire")

    mouth = Mouth(TieredSynth(_Exploding("local", local=True)), _SilentPlayer())  # type: ignore[arg-type]

    assert mouth.say("hello") is False
    assert mouth.last_error


def test_an_empty_reply_is_not_spoken() -> None:
    player = _SilentPlayer()
    mouth = Mouth(TieredSynth(_FakeSynth("local", local=True)), player)  # type: ignore[arg-type]

    assert mouth.say("   ") is False
    assert player.played == []


def test_a_quote_in_a_configured_value_cannot_end_the_powershell_literal() -> None:
    """The local voice is driven through PowerShell. What M.I.K.E.Y *says* never
    touches that command line at all — it is written to a file and read back,
    because model output can be steered by an ingested document. Configured values
    like a voice name do go inline, so they are escaped."""
    from core.voice.synth import _ps_quote

    assert _ps_quote("David's") == "David''s"
    assert _ps_quote("'; Remove-Item C:\\ -Recurse; '") == (
        "''; Remove-Item C:\\ -Recurse; ''"
    )
    assert isinstance(LocalSynth("David"), LocalSynth)
