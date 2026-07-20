"""Central configuration. Everything overridable via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_home() -> Path:
    return Path(os.environ.get("MIKEY_HOME", str(Path.home() / ".mikey")))


@dataclass(frozen=True)
class Config:
    home: Path = field(default_factory=_default_home)
    port: int = field(default_factory=lambda: int(os.environ.get("MIKEY_PORT", "43110")))
    provider: str = field(
        default_factory=lambda: os.environ.get(
            "MIKEY_PROVIDER", "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "ollama"
        )
    )
    anthropic_model: str = field(
        default_factory=lambda: os.environ.get("MIKEY_MODEL", "claude-sonnet-5")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get("MIKEY_OLLAMA_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.environ.get("MIKEY_OLLAMA_MODEL", "llama3.2")
    )
    # Approximate context budget for assembly, in characters (~4 chars/token).
    context_budget_chars: int = 24_000
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
