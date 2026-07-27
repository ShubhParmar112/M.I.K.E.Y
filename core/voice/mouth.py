"""Saying something: shape it, synthesize it, play it — and never fail a turn."""

from __future__ import annotations

import threading

from core.events.schema import Tier
from core.voice.play import AudioUnavailable, Player
from core.voice.synth import TieredSynth
from core.voice.text import speakable


class Mouth:
    """The voice, as the CLI uses it.

    Speaking is always best-effort. A missing audio device, a voice that won't
    load, a network hiccup on the neural voice — none of them are reasons to lose
    the answer, which is on screen regardless. Failures are recorded in
    `last_error` for a surface that wants to mention it once.
    """

    def __init__(self, synth: TieredSynth, player: Player | None = None) -> None:
        self._synth = synth
        self._player = player or Player()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    @property
    def speaking(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def say(self, text: str, tier: Tier = Tier.T1, blocking: bool = True) -> bool:
        """Speak a written reply. Returns whether anything was actually said."""
        spoken = speakable(text)
        if not spoken:
            return False
        try:
            audio = self._synth.render(spoken, tier)
        except Exception as exc:  # noqa: BLE001 — a voice failure must never end a turn
            self.last_error = str(exc)
            return False
        if audio is None:
            self.last_error = (
                "no voice available for this turn"
                if tier is not Tier.T0
                else "private turn — the neural voice is not allowed to speak it"
            )
            return False

        def _play() -> None:
            try:
                self._player.play(audio)
            except AudioUnavailable as exc:
                self.last_error = str(exc)

        if blocking:
            _play()
            return True
        self._thread = threading.Thread(target=_play, daemon=True)
        self._thread.start()
        return True

    def hush(self) -> None:
        """Stop mid-sentence."""
        self._player.stop()
