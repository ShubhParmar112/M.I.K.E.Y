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

uv run mikey chat            # interactive chat with approval cards
uv run mikey trace           # "why did you do that?" — trace tree of the last turn
uv run mikey events          # inspect the event log
```

Data lives in `~/.mikey/` (event log, audit chain, traces); the agent's sandbox is `~/.mikey/workspace/`. Reads are auto-allowed; writes and commands require approval (`y` once / `s` for the session); unknown tools are denied. Web content is taint-marked and can never authorize actions.

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
| Textual TUI, retrieval, memory tiers | ⏳ next |

## Non-negotiable principles

- **Local-first, user-owned data.** The user's memory store lives on their devices; cloud is an optional accelerator, never the source of truth.
- **Every action is mediated by a policy engine.** No agent touches the OS, network, or user data without a permission decision that is logged.
- **The event log is the source of truth.** Memory, knowledge graph, and sync state are all derived, rebuildable indexes.
- **Measured self-improvement only.** No "reflection" without an evaluation harness; a system that cannot measure itself cannot improve itself.
- **Modular monolith first, services later.** Process boundaries where safety demands them (execution sandbox), not where fashion suggests them.
