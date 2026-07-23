"""Central configuration. Everything overridable via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_home() -> Path:
    return Path(os.environ.get("MIKEY_HOME", str(Path.home() / ".mikey")))


def _detect_provider() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    return "ollama"


@dataclass(frozen=True)
class Config:
    home: Path = field(default_factory=_default_home)
    port: int = field(default_factory=lambda: int(os.environ.get("MIKEY_PORT", "43110")))
    provider: str = field(default_factory=lambda: os.environ.get("MIKEY_PROVIDER", _detect_provider()))
    anthropic_model: str = field(
        default_factory=lambda: os.environ.get("MIKEY_MODEL", "claude-sonnet-5")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get("MIKEY_OLLAMA_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.environ.get("MIKEY_OLLAMA_MODEL", "llama3.2")
    )
    groq_model: str = field(
        default_factory=lambda: os.environ.get("MIKEY_GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    # Hybrid routing: when a cloud provider is primary, fall back to a local
    # Ollama model on rate-limit/offline. Set MIKEY_LOCAL_FALLBACK=0 to disable.
    local_fallback: bool = field(
        default_factory=lambda: os.environ.get("MIKEY_LOCAL_FALLBACK", "1") != "0"
    )
    fallback_ollama_model: str = field(
        default_factory=lambda: os.environ.get(
            "MIKEY_FALLBACK_MODEL", os.environ.get("MIKEY_OLLAMA_MODEL", "llama3.2")
        )
    )
    # Semantic retrieval via a local embedding model (degrades to keyword-only if
    # the model/Ollama is unavailable). Set MIKEY_VECTORS=0 to disable entirely.
    local_vectors: bool = field(
        default_factory=lambda: os.environ.get("MIKEY_VECTORS", "1") != "0"
    )
    embed_model: str = field(
        default_factory=lambda: os.environ.get("MIKEY_EMBED_MODEL", "nomic-embed-text")
    )
    # Approximate context budget for conversation history, in characters
    # (~4 chars/token). Kept lean so a turn's several model calls stay under the
    # provider's per-minute token limit and don't get bounced to the local model.
    context_budget_chars: int = field(
        default_factory=lambda: int(os.environ.get("MIKEY_CONTEXT_CHARS", "10000"))
    )
    device_id: str = field(default_factory=lambda: os.environ.get("MIKEY_DEVICE", "dev_desktop_1"))

    @property
    def db_path(self) -> Path:
        return self.home / "mikey.db"

    @property
    def workspace(self) -> Path:
        return Path(os.environ.get("MIKEY_WORKSPACE", str(self.home / "workspace")))

    def ensure_dirs(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
