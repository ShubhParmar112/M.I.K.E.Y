"""Playing audio, interruptibly.

Interruptible is the whole reason this is not one line. An assistant that cannot
be stopped mid-sentence is exhausting to use: the one thing you want when it
starts reading you the wrong answer is to cut it off. So playback runs on a
stream we own and `stop()` ends it immediately.
"""

from __future__ import annotations

import io
import threading
from typing import Any


class AudioUnavailable(RuntimeError):
    """No usable audio output — no device, no PortAudio, an undecodable format.
    Always survivable: the reply is on screen either way."""


class Player:
    """Plays decoded audio through the default output device."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def play(self, audio: bytes, blocking: bool = True) -> None:
        """Play encoded audio (WAV, MP3 — whatever libsndfile decodes)."""
        try:
            import numpy as np
            import sounddevice as sd
            import soundfile as sf
        except (ImportError, OSError) as exc:
            # OSError: sounddevice raises it when PortAudio itself is missing.
            raise AudioUnavailable(f"audio output unavailable: {exc}") from exc

        try:
            data, samplerate = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        except Exception as exc:  # noqa: BLE001 — any decode failure is the same to us
            raise AudioUnavailable(f"could not decode audio: {exc}") from exc
        if not len(data):
            return

        with self._lock:
            self._stop.clear()
            try:
                self._play_array(sd, np, data, samplerate, blocking)
            except Exception as exc:  # noqa: BLE001 — device errors vary by backend
                raise AudioUnavailable(f"playback failed: {exc}") from exc

    def _play_array(
        self, sd: Any, np: Any, data: Any, samplerate: int, blocking: bool
    ) -> None:
        if not blocking:
            sd.play(data, samplerate)
            return
        # Played in blocks rather than one call so `stop()` takes effect within a
        # few hundredths of a second instead of at the end of the sentence.
        block = max(1024, samplerate // 20)
        with sd.OutputStream(
            samplerate=samplerate, channels=data.shape[1], dtype="float32"
        ) as stream:
            for start in range(0, len(data), block):
                if self._stop.is_set():
                    break
                stream.write(np.ascontiguousarray(data[start : start + block]))

    def stop(self) -> None:
        """Cut playback off now."""
        self._stop.set()
        try:
            import sounddevice as sd

            sd.stop()
        except (ImportError, OSError):
            pass
