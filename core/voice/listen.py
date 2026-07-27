"""Hearing: microphone in, text out — entirely on this machine.

Speech recognition runs locally (faster-whisper on the CPU) rather than through a
cloud speech API, and not only for privacy: a microphone is the one input that
picks things up you did not mean to send. Whatever is said in this room stays in
it unless the turn itself goes to a cloud model, which the existing tier rules
already govern.

Recording stops on silence rather than on a keypress. Having to press a key to
stop talking is the difference between a conversation and a walkie-talkie.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Protocol

SAMPLE_RATE = 16_000  # what Whisper expects; resampling later would only lose fidelity
BLOCK_MS = 30

# Speech is roughly two orders of magnitude louder than room tone. This threshold
# is on RMS amplitude of float32 samples and is deliberately low: the cost of
# picking up a quiet sentence is nothing, the cost of missing one is repeating
# yourself.
SILENCE_RMS = 0.012
# How long a pause has to last before it counts as "finished talking". Shorter and
# it cuts you off mid-thought; longer and every exchange has a dead second in it.
SILENCE_HANG_S = 1.2
# Give up waiting for speech that never comes.
MAX_WAIT_S = 20.0
MAX_UTTERANCE_S = 60.0
# Anything shorter than this is a cough, a chair, or a door.
MIN_SPEECH_S = 0.35


class MicrophoneUnavailable(RuntimeError):
    """No usable input device. The keyboard still works — never fatal."""


class Transcriber(Protocol):
    def transcribe(self, audio: Any) -> str: ...


@dataclass
class Heard:
    text: str
    seconds: float

    @property
    def empty(self) -> bool:
        return not self.text.strip()


class WhisperTranscriber:
    """faster-whisper, loaded once and kept warm.

    `tiny.en` is the default because this machine has four CPU cores and no usable
    GPU: a bigger model turns a two-second sentence into an eight-second wait, and
    the wait is what makes voice control feel broken. Set MIKEY_STT_MODEL to
    `base.en` or `small.en` if accuracy matters more than latency.
    """

    def __init__(self, model: str = "tiny.en", compute_type: str = "int8") -> None:
        self._model_name = model
        self._compute_type = compute_type
        self._model: Any = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the model now — the first load downloads it, which is not something
        to discover in the middle of the first sentence someone speaks."""
        with self._lock:
            if self._model is not None:
                return
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise MicrophoneUnavailable(
                    "faster-whisper is not installed (uv sync --extra voice)"
                ) from exc
            self._model = WhisperModel(
                self._model_name, device="cpu", compute_type=self._compute_type
            )

    def transcribe(self, audio: Any) -> str:
        self.load()
        assert self._model is not None
        options: dict[str, Any] = {
            "language": "en",
            "beam_size": 1,  # greedy: on a CPU this is most of the latency
            "condition_on_previous_text": False,  # each utterance stands alone
            "vad_filter": True,
        }
        try:
            segments, _info = self._model.transcribe(audio.flatten(), **options)
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except RuntimeError:
            # The VAD filter needs onnxruntime, which is optional and does not load
            # on every machine. It only trims silence Whisper would otherwise
            # hallucinate over — and the microphone above already cut this clip on
            # silence — so losing it degrades quality slightly rather than at all.
            options["vad_filter"] = False
            segments, _info = self._model.transcribe(audio.flatten(), **options)
            text = " ".join(segment.text.strip() for segment in segments).strip()
        return text


class Microphone:
    """Records one utterance: waits for speech, then stops when it ends."""

    def __init__(
        self,
        samplerate: int = SAMPLE_RATE,
        silence_rms: float = SILENCE_RMS,
        hang_s: float = SILENCE_HANG_S,
    ) -> None:
        self._samplerate = samplerate
        self._silence = silence_rms
        self._hang = hang_s

    def record(self, max_wait_s: float = MAX_WAIT_S) -> Any:
        """Capture one utterance as a float32 array. Empty array if nothing was said."""
        try:
            import numpy as np
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise MicrophoneUnavailable(f"no microphone support: {exc}") from exc

        block = int(self._samplerate * BLOCK_MS / 1000)
        collected: list[Any] = []
        speaking = False
        silent_for = 0.0
        waited = 0.0
        step = BLOCK_MS / 1000

        try:
            with sd.InputStream(
                samplerate=self._samplerate, channels=1, dtype="float32", blocksize=block
            ) as stream:
                while True:
                    chunk, _overflowed = stream.read(block)
                    level = float(np.sqrt(np.mean(np.square(chunk))))
                    if level >= self._silence:
                        speaking = True
                        silent_for = 0.0
                        collected.append(chunk.copy())
                    elif speaking:
                        silent_for += step
                        collected.append(chunk.copy())  # keep the tail of the word
                        if silent_for >= self._hang:
                            break
                    else:
                        waited += step
                        if waited >= max_wait_s:
                            break
                    if len(collected) * step >= MAX_UTTERANCE_S:
                        break
        except Exception as exc:  # noqa: BLE001 — backend errors vary
            raise MicrophoneUnavailable(f"could not record: {exc}") from exc

        if not collected:
            return np.zeros((0, 1), dtype="float32")
        return np.concatenate(collected)


class Ears:
    """Microphone plus transcriber: `listen()` gives you what was said."""

    def __init__(self, mic: Microphone, transcriber: Transcriber) -> None:
        self._mic = mic
        self._transcriber = transcriber

    def listen(self, max_wait_s: float = MAX_WAIT_S) -> Heard:
        audio = self._mic.record(max_wait_s)
        seconds = len(audio) / SAMPLE_RATE
        if seconds < MIN_SPEECH_S:
            # Too short to be a sentence. Transcribing it produces confident
            # nonsense ("Thank you.", "you") that would then be sent as a turn.
            return Heard(text="", seconds=seconds)
        return Heard(text=self._transcriber.transcribe(audio), seconds=seconds)
