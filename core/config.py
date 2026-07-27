"""Central configuration. Everything overridable via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_home() -> Path:
    return Path(os.environ.get("MIKEY_HOME", str(Path.home() / ".mikey")))


# Cloud providers M.I.K.E.Y can speak to, in the order they are preferred when no
# provider is named — and, past the first, the order of the failover chain. The
# key of each is what decides whether it is available at all.
#
# The ordering is "best answer first, then whatever is still free": Anthropic when
# it is paid for, then Groq (fastest, but the smallest free daily allowance by an
# order of magnitude), then the roomier free tiers. Running out of one is then a
# step sideways to another cloud model rather than a fall onto the 3B local one.
CLOUD_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("groq", "GROQ_API_KEY"),
    ("cerebras", "CEREBRAS_API_KEY"),
    ("gemini", "GEMINI_API_KEY"),
)


def available_cloud_providers() -> list[str]:
    """Cloud providers with a key present, in preference order."""
    return [name for name, env in CLOUD_PROVIDERS if os.environ.get(env)]


def _detect_provider() -> str:
    providers = available_cloud_providers()
    return providers[0] if providers else "ollama"


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
    # The other two free tiers. Both are far roomier than Groq's 100k tokens/day —
    # which is the whole reason they are here: one free tier is a single point of
    # failure, and its failure mode is every answer for the rest of the day coming
    # from a 3B local model.
    cerebras_model: str = field(
        default_factory=lambda: os.environ.get("MIKEY_CEREBRAS_MODEL", "gpt-oss-120b")
    )
    gemini_model: str = field(
        default_factory=lambda: os.environ.get("MIKEY_GEMINI_MODEL", "gemini-2.5-flash")
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
    # Per-brain localization (sovereignty S2): brain names served by the local
    # model instead of the cloud primary, e.g. MIKEY_LOCAL_BRAINS=conversation,critic.
    # Cloud fallback is preserved for each. Empty by default — opt in per brain as
    # its local quality passes the shadow/eval gate (`mikey reasoning-eval`).
    local_brains: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            b.strip() for b in os.environ.get("MIKEY_LOCAL_BRAINS", "").split(",") if b.strip()
        )
    )
    # Privacy-tier classification (sovereignty S3): mark plainly-private turns Tier-0
    # so the gateway keeps them on-device and the exporter excludes them from cloud
    # training. On by default; set MIKEY_TIER_CLASSIFY=0 to treat every turn as T1.
    tier_classify: bool = field(
        default_factory=lambda: os.environ.get("MIKEY_TIER_CLASSIFY", "1") != "0"
    )
    # Sampling. Temperature stays low for a factual assistant, but NOT at 0.2 —
    # open-weight models fall into repetition loops most readily at very low
    # temperature, which is how a live turn produced a paragraph of self-negating
    # text. The cap bounds any loop that still starts (and a real derivation fits).
    temperature: float = field(
        default_factory=lambda: float(os.environ.get("MIKEY_TEMPERATURE", "0.3"))
    )
    max_output_tokens: int = field(
        default_factory=lambda: int(os.environ.get("MIKEY_MAX_OUTPUT_TOKENS", "1536"))
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
    # Independent verification of reasoning answers. "flagged" (default) spends a
    # second model call only when the reply looks like an asserted-not-derived answer;
    # "always" checks every reasoning answer (slower, costlier, catches more); "off"
    # restores self-verification only. MIKEY_VERIFY_REASONING.
    verify_reasoning: str = field(
        default_factory=lambda: os.environ.get("MIKEY_VERIFY_REASONING", "flagged").lower()
    )
    # Cost governor (Gen 3): USD of cloud inference allowed per calendar month.
    # Once it's spent, the gateway serves from the local model instead of the cloud
    # — spend stops, M.I.K.E.Y keeps working. Set MIKEY_MONTHLY_BUDGET_USD=0 to
    # disable enforcement entirely (usage is still tracked, so `mikey spend` works).
    monthly_budget_usd: float = field(
        default_factory=lambda: float(os.environ.get("MIKEY_MONTHLY_BUDGET_USD", "10"))
    )
    # The other budget, and on a free plan the one that actually bites: tokens per
    # DAY. 0 uses the built-in free-tier table (core.cost.governor); set
    # MIKEY_DAILY_TOKEN_CAP to your real allowance if you've moved off the free
    # tier, so the gauge tells the truth instead of crying wolf.
    daily_token_cap: int = field(
        default_factory=lambda: int(os.environ.get("MIKEY_DAILY_TOKEN_CAP", "0"))
    )
    # --- voice (optional: `uv sync --extra voice`) ---
    # Which voice speaks. "local" = Windows' own speech: offline, private, robotic.
    # "edge" = Microsoft's neural voices: sounds human, but sends the text of every
    # spoken reply over the network — so it is never used for a Tier-0 turn, which
    # falls back to the local voice instead. "off" = text only.
    voice_synth: str = field(
        default_factory=lambda: os.environ.get("MIKEY_VOICE", "local").lower()
    )
    voice_name: str = field(default_factory=lambda: os.environ.get("MIKEY_VOICE_NAME", ""))
    # Speech recognition always runs on-device. tiny.en answers in well under a
    # second on this CPU; base.en is more accurate and roughly twice the wait.
    stt_model: str = field(default_factory=lambda: os.environ.get("MIKEY_STT_MODEL", "tiny.en"))
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
