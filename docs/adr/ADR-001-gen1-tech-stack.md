# ADR-001: Generation 1–3 Technology Stack

**Status:** Accepted · **Date:** 2026-07-21 · **Scope:** Gen 1–3 (desktop core). Revisit at Gen 4 (sync/mobile) and Gen 6 (speech).

## Decision drivers

1. One developer; iteration speed is survival.
2. ML/agent ecosystem gravity (Python is where every model, SDK, and paper lands first).
3. Primary dev/runtime platform is Windows 11; must not preclude Linux/macOS.
4. The architecture (02) demands: strict module boundaries, two process boundaries, local-first storage, provider-agnostic models.
5. Prefer boring, embedded, zero-ops technology. Every server you don't run is a server that can't be down.

## Decisions

### Core language & runtime
- **Python 3.12+**, `uv` for dependency/project management, **`mypy --strict`** + **ruff** enforced in CI.
- **import-linter** contracts enforce the modular-monolith boundaries (core modules may not import each other except through declared interfaces). *This is the mechanism that keeps the monolith modular — without it the architecture erodes in months.*
- **Rejected:** Rust core (velocity cost too high solo; ML ecosystem friction), Node core (weaker ML story), Java/Kotlin (no gravity here).

### Agent loop, models, tools
- **Model Gateway** built over provider SDKs with a thin internal interface: `complete(request, {tier, capability, budget})`.
  - Cloud tier-1: **Anthropic Claude** (Sonnet-class default; strongest tool-use/agentic behavior) via the **Claude Agent SDK** for the agentic loop rather than hand-rolling plan-act-verify.
  - Local tier-0: **Ollama** (llama.cpp under the hood) hosting a small instruct model + embedding model. Simplest consumer-hardware path; the inference host is already a separate process, satisfying the crash boundary for free.
- **MCP (Model Context Protocol) as the universal tool/connector protocol** — including *internally*: the Executor Sandbox exposes its tools as an MCP server. One protocol from Gen 1 through the Gen 8 public plugin SDK; the vision's "plugin manager" is MCP client + registry + policy wrapper.
- **Embeddings:** local by default (privacy + cost): `nomic-embed-text` or BGE via Ollama.
- **Rejected:** LangChain/CrewAI/AutoGen as foundations (abstraction churn, debugging opacity; they optimize demo speed, we optimize trace clarity). Raw provider APIs only — acceptable fallback if the Agent SDK constrains us.

### Storage (all embedded, all local)
- **SQLite (WAL mode)** — event log, task/mission state, policy rules, audit chain, graph tables, config. Accessed through a single storage module; SQLAlchemy Core (not ORM) for typed queries and future Postgres escape hatch.
- **sqlite-vec** for vector search + **FTS5** for keyword — hybrid retrieval in one file, one backup unit.
- **Rejected:** Postgres (ops burden on a laptop, no multi-user yet), Neo4j (graph is a projection in SQLite until proven otherwise — review §1.2), Chroma/Qdrant/LanceDB servers (another process to babysit; sqlite-vec suffices at personal scale), Redis/Celery (SQLite-backed durable queue tables instead).

### Executor Sandbox (separate process)
- **Go**, single static binary, tools exposed over **MCP (stdio)**. Small trusted computing base, memory-safe, trivial cross-compile, easy Windows job-object / restricted-token integration for real OS-level constraint.
- Capability tokens (scoped filesystem roots, egress allowlist, per-tool grants, TTL) validated *in the executor*, not just in core — the sandbox does not trust the core's politeness.
- **Concession:** if Go slows Gen 1 unacceptably, a Python executor process is permitted *only* behind the same MCP contract and process boundary, with the Go rewrite scheduled for Gen 3. The boundary and protocol are the architecture; the language is an implementation detail.

### Session Gateway & API
- **FastAPI** + **Pydantic v2** (Pydantic models double as the event-schema source of truth, versioned per 02 §4), SSE for token/action streaming. Runs on localhost only in Gen 1; the same API later serves web dashboard and mobile.

### Desktop app
- **Tauri 2 + React + TypeScript + Vite + Tailwind**, talking only to the Gateway API. Chat, approval cards, mission board, memory browser, trace viewer.
- **Rejected:** Electron (RAM/size; Tauri's Rust shell is also a nice place for future OS integrations), pure terminal UI (approval cards and trace trees need real UI), PyQt (web stack reuses directly for the Gen 4 dashboard and Gen 8 extension).
- **CLI:** Python + Typer, same Gateway API — ships before the GUI does.

### Security primitives
- Secrets: **OS keychain** (Windows Credential Manager / DPAPI via `keyring`) — never in SQLite, never in model context; injected into tool calls at execution time.
- Audit chain: hash-chained rows (SHA-256) written only by the policy engine.
- Sync crypto (Gen 4): **libsodium** (PyNaCl) — sealed boxes per device key; relay = any dumb blob store (self-hosted mini-service or S3-compatible), decided in ADR at Gen 4.

### Perception (Gen 6 pre-commitments, revisit then)
- ASR: **faster-whisper** local · TTS: **Piper/Kokoro** local · Wake word: **openWakeWord** · OCR/PDF: **PyMuPDF** + vision-capable model via Gateway for layout understanding.

### Dev infrastructure
- **GitHub + Actions**: lint, type-check, tests, import-linter contracts, eval smoke suite on every PR. **pytest** (+ trace-replay fixtures). Conventional commits; ADR per contested decision.

## Consequences

- Two languages in anger (Python, TypeScript) + Go confined to one small binary — acceptable solo surface.
- Everything embedded → whole system backs up as files; restore drill (Gen 1 exit criterion) is `copy` + verify.
- MCP everywhere means Gen 8's "ecosystem" is an unlock, not a rewrite.
- The Postgres/service escape hatches exist at every seam but are paid for only when forced.

## Amendment A1 — 2026-07-21 (accepted)

1. **TypeScript deferred out of Gen 1–2.** The Gen 1–2 user surface is **CLI (Typer) + Textual TUI** — pure Python. Tauri/React enters at Gen 3+ only when approval cards and the trace viewer measurably outgrow the terminal. This is deferral, not deletion: browser surfaces (Gen 4 dashboard, Gen 8 extension) and the mobile app remain web-stack, and all UI still speaks only the Gateway API, so the swap costs nothing architecturally.
2. **Executor ships Python-first** under the concession already recorded above: same separate process, same MCP-style stdio contract, capability enforcement inside the executor. Go rewrite scheduled for Gen 3 when the tool surface grows.
3. **stdlib `sqlite3` behind a single storage module** instead of SQLAlchemy Core for Gen 1. The event log is append-only with a handful of queries; SQLAlchemy Core is adopted when schema complexity warrants it. The storage module is the only file that knows SQL exists.
