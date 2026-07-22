# M.I.K.E.Y — Multimodal Intelligent Knowledge & Execution Engine for You

A personal AI cognitive operating system: perceives, remembers, reasons, executes, and improves over a multi-year horizon, across devices, under strict user control.

This repository currently contains the **engineering foundation** — the design documents that everything else will be built on.

## Documentation index

| Doc | Purpose |
|---|---|
| [docs/00-vision.md](docs/00-vision.md) | The original vision document (unedited input) |
| [docs/01-architecture-review.md](docs/01-architecture-review.md) | Critical review of the vision: missing subsystems, weaknesses, security threat model, feasibility flags |
| [docs/02-system-architecture.md](docs/02-system-architecture.md) | Proposed production architecture: trust boundaries, data model, context pipeline, execution safety, sync design |
| [docs/03-roadmap.md](docs/03-roadmap.md) | Generation 1 → 10 development roadmap with exit criteria |

## Reading order

1. Review (01) — understand what the vision gets wrong and why.
2. Architecture (02) — the corrected design.
3. Roadmap (03) — the order of construction. **Generation 1 is deliberately small.**

## Gen 1 — quick start

```powershell
uv sync                      # install
uv run pytest                # verify (16 tests)

# pick a model provider (any of the three):
$env:ANTHROPIC_API_KEY = "sk-ant-..."       # cloud (Claude), or:
$env:GROQ_API_KEY = "gsk_..."               # cloud (Groq, free tier, Llama 3.3), or:
# install Ollama + `ollama pull llama3.2`   # local / private

# hybrid (recommended): a cloud primary with a local fallback. If Ollama is
# installed, M.I.K.E.Y auto-routes to it when the cloud model is rate-limited
# (429) or unreachable (offline) — so a Groq free-tier limit is no longer fatal.
#   MIKEY_LOCAL_FALLBACK=0        disable the fallback
#   MIKEY_FALLBACK_MODEL=qwen2.5:3b   choose the local fallback model

uv run mikey chat            # interactive chat with approval cards
uv run mikey trace           # "why did you do that?" — trace tree of the last turn
uv run mikey events          # inspect the event log

# Gen 2 — memory
uv run mikey ingest <path>   # ingest text files/folders into memory (marked untrusted)
uv run mikey recall "query"  # search memory — results carry source, date, trust
uv run mikey forget <id>     # tombstone a memory; verified gone from all projections
uv run mikey reindex         # rebuild the memory index from the event log
```

Data lives in `~/.mikey/` (event log, audit chain, traces); the agent's sandbox is `~/.mikey/workspace/`. Reads are auto-allowed; writes and commands require approval (`y` once / `s` for the session); unknown tools are denied. Web content is taint-marked and can never authorize actions.

M.I.K.E.Y also reaches its memory *during* a conversation: `memory_recall` searches long-term memory on demand (not just the handful pre-loaded each turn) and `memory_remember` persists a durable fact when you ask it to — so "remember my dog is named Pixel" now sticks, and "what's my dog's name?" later retrieves it with provenance. Recalling an untrusted memory taints the turn just like ingested files do.

Remembered facts are kept clean: a near-duplicate is skipped, a correction can `supersede` (tombstone) the stale fact it replaces, and a related-but-conflicting memory is flagged so it gets reconciled rather than silently doubling up. `memory_forget` lets M.I.K.E.Y drop a specific memory on request — a verified deletion from every projection, gated behind an approval card because it's destructive.

## Gen 1 status

| Component | State |
|---|---|
| Event log (append-only, versioned schema, SQLite WAL) | ✅ `core/events/` |
| Context assembly (recent-history v0, traced) | ✅ `core/context/` |
| Model gateway (Anthropic / Ollama / fake) | ✅ `core/models/` |
| Policy engine + hash-chained audit + taint rule | ✅ `core/policy/` |
| Executor sandbox (separate process, path confinement, command allowlist) | ✅ `executor/` |
| Turn loop (plan → policy → act → trace) | ✅ `core/orchestrator/` |
| Session gateway API (SSE streaming, approvals, traces) | ✅ `core/gateway/` |
| CLI with approval cards + trace viewer | ✅ `apps/cli/` |
| CI (ruff + pytest on every push) | ✅ `.github/workflows/` |
| Memory: FTS retrieval, ingestion, verified forgetting, taint | ✅ `core/memory/`, `core/ingest/` |
| Textual TUI, vector retrieval, memory tiers, contradiction flags | ⏳ next |

## Non-negotiable principles

- **Local-first, user-owned data.** The user's memory store lives on their devices; cloud is an optional accelerator, never the source of truth.
- **Every action is mediated by a policy engine.** No agent touches the OS, network, or user data without a permission decision that is logged.
- **The event log is the source of truth.** Memory, knowledge graph, and sync state are all derived, rebuildable indexes.
- **Measured self-improvement only.** No "reflection" without an evaluation harness; a system that cannot measure itself cannot improve itself.
- **Modular monolith first, services later.** Process boundaries where safety demands them (execution sandbox), not where fashion suggests them.
