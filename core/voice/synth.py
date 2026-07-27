"""Speech synthesis: text in, audio bytes out.

Two synthesizers, and the choice between them is a privacy decision rather than a
taste one:

- **LocalSynth** uses the speech voices Windows already has. Nothing leaves the
  machine, it works offline, and it sounds like a 2010 satnav.
- **EdgeSynth** uses Microsoft's neural voices over the network. It sounds like a
  person — and it means the text of every spoken reply is sent to Microsoft.

That second point is why this module knows about privacy tiers at all. A Tier-0
turn is one the gateway refuses to send to any cloud model; speaking its answer
through a cloud voice would leak out the far end of the same turn. So a cloud
synth REFUSES T0 text and the local voice takes it — degraded, not disclosed.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from core.events.schema import Tier

# A dry, measured English voice reads better for an assistant than the bright
# customer-service default. Override with MIKEY_VOICE_NAME.
DEFAULT_EDGE_VOICE = "en-GB-RyanNeural"
DEFAULT_LOCAL_VOICE = ""  # empty = whatever Windows considers its default


def _ps_quote(value: str) -> str:
    """Escape a value for a PowerShell single-quoted literal."""
    return value.replace("'", "''")


class SynthUnavailable(RuntimeError):
    """This synthesizer cannot speak right now — offline, missing voice, no
    speech support on the platform. The caller falls back or stays silent; it is
    never a reason to fail the turn."""


class Synth(Protocol):
    name: str
    local: bool

    def render(self, text: str) -> bytes:
        """Audio for `text`, in a container `soundfile` can decode."""
        ...


class LocalSynth:
    """Windows' built-in speech, rendered to a WAV.

    Driven through PowerShell's System.Speech rather than a Python TTS binding on
    purpose: no extra dependency, no COM lifetime quirks, and rendering to a file
    (instead of speaking directly) means playback is ours to interrupt — which is
    what makes talking over M.I.K.E.Y possible.
    """

    name = "local"
    local = True

    def __init__(self, voice: str = DEFAULT_LOCAL_VOICE, rate: int = 1) -> None:
        self._voice = voice
        # System.Speech rate is -10..10; slightly above neutral reads as attentive
        # rather than sleepy, and shortens every reply a little.
        self._rate = max(-10, min(10, rate))

    def render(self, text: str) -> bytes:
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell is None:
            raise SynthUnavailable("no PowerShell available to reach Windows speech")

        work = Path(tempfile.mkdtemp(prefix="mikey-voice-"))
        out = work / "say.wav"
        said = work / "say.txt"
        # The text is written to a file and read back by the script rather than
        # embedded in it. What M.I.K.E.Y says is model output, and model output can
        # be steered by an ingested document — any quoting scheme that puts it
        # inside a command line is one crafted sentence away from being a command.
        said.write_text(text, encoding="utf-8")
        select_voice = f"$s.SelectVoice('{_ps_quote(self._voice)}');" if self._voice else ""
        script = (
            "Add-Type -AssemblyName System.Speech;"
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            f"{select_voice}"
            f"$s.Rate = {self._rate};"
            f"$s.SetOutputToWaveFile('{_ps_quote(str(out))}');"
            f"$t = [IO.File]::ReadAllText('{_ps_quote(str(said))}', [Text.Encoding]::UTF8);"
            "$s.Speak($t);"
            "$s.Dispose();"
        )
        try:
            done = subprocess.run(
                [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SynthUnavailable(f"windows speech failed: {exc}") from exc
        try:
            if done.returncode != 0 or not out.exists():
                detail = done.stderr.decode("utf-8", "replace").strip()[:200]
                raise SynthUnavailable(f"windows speech failed: {detail or 'no audio produced'}")
            return out.read_bytes()
        finally:
            shutil.rmtree(work, ignore_errors=True)


class EdgeSynth:
    """Microsoft's neural voices. Sounds real; needs the network; sends the text.

    Refuses Tier-0 text by construction — see the module docstring.
    """

    name = "edge"
    local = False

    def __init__(self, voice: str = DEFAULT_EDGE_VOICE) -> None:
        self._voice = voice

    def render(self, text: str) -> bytes:
        try:
            import edge_tts
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise SynthUnavailable("edge-tts is not installed (uv sync --extra voice)") from exc

        async def _render() -> bytes:
            audio = bytearray()
            communicate = edge_tts.Communicate(text, self._voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio.extend(chunk["data"])
            return bytes(audio)

        try:
            audio = asyncio.run(_render())
        except Exception as exc:  # noqa: BLE001 — any network/protocol failure is the same to us
            raise SynthUnavailable(f"neural voice unreachable: {exc}") from exc
        if not audio:
            raise SynthUnavailable("neural voice returned no audio")
        return audio


class TieredSynth:
    """The voice as the rest of the system sees it: `say(text, tier)`.

    Holds a preferred synth and the local one, and enforces the two rules that
    must not depend on remembering them at the call site — private text never
    reaches a cloud voice, and a voice that fails is silent rather than fatal.
    """

    def __init__(self, preferred: Synth, local: Synth | None = None) -> None:
        self._preferred = preferred
        self._local = local if local is not None else (preferred if preferred.local else LocalSynth())
        self.last_used: str | None = None

    def render(self, text: str, tier: Tier = Tier.T1) -> bytes | None:
        """Audio for `text`, or None if nothing could speak it."""
        if not text.strip():
            return None
        chain: list[Synth] = [self._preferred]
        if self._preferred is not self._local:
            chain.append(self._local)
        for synth in chain:
            if tier is Tier.T0 and not synth.local:
                # The same rule the model gateway enforces, at the other end of the
                # turn: private stays on the device, in every direction.
                continue
            try:
                audio = synth.render(text)
            except SynthUnavailable:
                continue
            self.last_used = synth.name
            return audio
        self.last_used = None
        return None


def make_synth(kind: str, voice: str = "") -> TieredSynth | None:
    """Build the configured voice. `kind` is "local" | "edge" | "off"."""
    if kind == "off":
        return None
    local = LocalSynth(voice if kind == "local" else DEFAULT_LOCAL_VOICE)
    if kind == "edge":
        return TieredSynth(EdgeSynth(voice or DEFAULT_EDGE_VOICE), local)
    return TieredSynth(local, local)
