# M.I.K.E.Y — Multimodal Intelligent Knowledge & Execution Engine for You

A personal AI cognitive operating system: perceives, remembers, reasons, executes, and improves over a multi-year horizon, across devices, under strict user control.

This repository contains a working **Gen 1–3 core** — append-only event log, policy-gated executor sandbox, long-term memory, and durable missions — plus a decomposed multi-brain orchestration layer and the design corpus that guides it.

## Documentation index

| Doc | Purpose |
|---|---|
| [docs/00-vision.md](docs/00-vision.md) | The original vision document (unedited input) |
| [docs/01-architecture-review.md](docs/01-architecture-review.md) | Critical review of the vision: missing subsystems, weaknesses, security threat model, feasibility flags |
| [docs/02-system-architecture.md](docs/02-system-architecture.md) | Proposed production architecture: trust boundaries, data model, context pipeline, execution safety, sync design |
| [docs/03-roadmap.md](docs/03-roadmap.md) | Generation 1 → 10 development roadmap with exit criteria |
| [docs/04-intelligence-sovereignty.md](docs/04-intelligence-sovereignty.md) | Migrating from API-driven cognition to a self-hosted brain fleet: bottlenecks, model strategy, training pipeline, S0–S6 plan |

## Reading order

1. Review (01) — understand what the vision gets wrong and why.
2. Architecture (02) — the corrected design.
3. Roadmap (03) — the order of construction. **Generation 1 is deliberately small.**

## Quick start

```powershell
uv sync                      # install
uv run pytest                # verify (136 passing)

# pick a model provider (any of the three):
$env:GROQ_API_KEY = "gsk_..."               # cloud (Groq, free tier, Llama 3.3), or:
$env:ANTHROPIC_API_KEY = "sk-ant-..."       # cloud (Claude), or:
# install Ollama + `ollama pull llama3.2`   # local / private

uv run mikey doctor          # check providers, local models, brain routing, integrity
uv run mikey chat            # interactive chat with approval cards + trace
```

After `uv sync`, run any command as `uv run mikey <command>` (or `mikey <command>` with the venv active). Run `mikey doctor` anytime to see your effective setup.

Data lives in `~/.mikey/` (event log, audit chain, traces); the agent's sandbox is `~/.mikey/workspace/`. Reads are auto-allowed; writes and commands require approval (`y` once / `s` for the session); unknown tools are denied. Web content is taint-marked and can never authorize actions.

M.I.K.E.Y also reaches its memory *during* a conversation: `memory_recall` searches long-term memory on demand (not just the handful pre-loaded each turn) and `memory_remember` persists a durable fact when you ask it to — so "remember my dog is named Pixel" now sticks, and "what's my dog's name?" later retrieves it with provenance. Recalling an untrusted memory taints the turn just like ingested files do.

Remembered facts are kept clean: a near-duplicate is skipped, a correction can `supersede` (tombstone) the stale fact it replaces, and a related-but-conflicting memory is flagged so it gets reconciled rather than silently doubling up. `memory_forget` lets M.I.K.E.Y drop a specific memory on request — a verified deletion from every projection, gated behind an approval card because it's destructive.

## Commands

**Chat & reasoning**

| Command | What it does |
|---|---|
| `mikey chat` | Interactive session — routes each turn through the brain fleet, streams actions + approval cards (with the critic's note), header shows which brains run locally. `/trace` and `/quit` work inside. |
| `mikey serve` | Run the gateway in the foreground (for a separate terminal). |
| `mikey trace [turn_id]` | "Why did you do that?" — the full reasoning tree for a turn (route → model call → policy → tool). Defaults to the last turn. |
| `mikey events [--limit N]` | Recent raw events from the append-only log. |

**Memory**

| Command | What it does |
|---|---|
| `mikey ingest <path>` | Read a file/folder (text or PDF, anywhere on disk) into long-term memory (marked untrusted). |
| `mikey recall "<query>" [--k N]` | Search memory; results carry source, date, and trust level. |
| `mikey forget <event_id>` | Tombstone a memory; verified gone from every projection. |
| `mikey reindex` | Rebuild the memory index from the event log (projections are disposable). |

**Missions & planning**

| Command | What it does |
|---|---|
| `mikey plan "<goal>" [--run]` | Decompose a goal into a validated, durable mission; `--run` executes it immediately. |
| `mikey missions` | List unfinished (resumable) missions. |
| `mikey mission-run <id>` | Run or resume a mission, approving steps as they come (survives reboot). |

**Setup, sovereignty & quality**

| Command | What it does |
|---|---|
| `mikey doctor` | Setup check: cloud providers, local model host, which brain runs where (+ localization advice), audit-chain integrity. |
| `mikey reasoning-eval [--against <provider>]` | Score tool-use on the golden set; `--against ollama` shadow-compares cloud vs local (the gate before localizing a reasoning brain). |
| `mikey export [--out DIR] [--include-t0]` | Export the event log → per-brain training datasets (respects tombstones + privacy tiers). |
| `mikey eval [--update-baseline]` | Retrieval-quality eval against the golden set. |

**Ops & safety**

| Command | What it does |
|---|---|
| `mikey backup` | Verified snapshot of the whole store (log + audit chain). |
| `mikey restore <path> [--yes]` | Restore from a backup (verifies it, snapshots current state first). |

## Brains & local-first routing

Every turn is routed to one of a small fleet of **brains** — each a capability profile (prompt + tool allowlist), not a separate model:

| Brain | Role | Tools |
|---|---|---|
| `conversation` | greetings, sign-offs, small talk | none |
| `operator` | actionable turns, questions, recall/remember | all except `memory_forget` |
| `memory` | forgetting / correcting stored memory | recall, remember, forget |
| `critic` | reviews a risky action before you approve it | none (judges) |
| `planner` | turns a goal into a durable mission | none (proposes) |
| `router` | picks the brain per turn | heuristic (always local) |

Brains are served by a **cloud primary with a local (Ollama) fallback** — auto-routing to Ollama on rate-limit/offline. Any brain can also be pinned **on-device**, one at a time, so its calls never leave the machine:

```powershell
$env:MIKEY_LOCAL_BRAINS = "conversation"   # serve chit-chat locally; keep reasoning on cloud
$env:MIKEY_OLLAMA_MODEL = "llama3.2"       # the local model used for pinned brains
```

Useful env knobs: `MIKEY_LOCAL_BRAINS` (brains to run locally), `MIKEY_OLLAMA_MODEL` (local model), `MIKEY_LOCAL_FALLBACK=0` (disable the fallback), `MIKEY_PROVIDER` / `MIKEY_MODEL` (cloud choice), `MIKEY_HOME`, `MIKEY_WORKSPACE`. `mikey doctor` prints the effective result. See [docs/04-intelligence-sovereignty.md](docs/04-intelligence-sovereignty.md) for the full local-migration plan.

## Status

| Component | State |
|---|---|
| Event log (append-only, versioned schema, SQLite WAL) | ✅ `core/events/` |
| Context assembly (history + provenance-annotated memory, traced) | ✅ `core/context/` |
| Model gateway (Groq / Anthropic / Ollama / fake · tier + capability routing) | ✅ `core/models/` |
| Policy engine + hash-chained audit + taint rule | ✅ `core/policy/` |
| Executor sandbox (separate process, path confinement, command allowlist) | ✅ `executor/` |
| Turn loop + brain fleet (router · conversation · operator · memory · critic · planner) | ✅ `core/orchestrator/` |
| Durable missions (multi-step, policy-gated, resume after reboot) | ✅ `core/missions/` |
| Memory: hybrid FTS + vector retrieval, ingestion, verified forgetting, taint | ✅ `core/memory/`, `core/ingest/` |
| Session gateway API (SSE streaming, approvals, traces) | ✅ `core/gateway/` |
| CLI with approval cards, trace viewer, doctor, planner | ✅ `apps/cli/` |
| Sovereignty S0–S2: T0 enforcement · data exporter · reasoning eval · per-brain local routing | ✅ `training/`, `core/eval/`, `core/models/` |
| CI (ruff + pytest on every push) | ✅ `.github/workflows/` |
| Local reasoning brains, training/distillation pipeline, knowledge graph | ⏳ next |

## Non-negotiable principles

- **Local-first, user-owned data.** The user's memory store lives on their devices; cloud is an optional accelerator, never the source of truth.
- **Every action is mediated by a policy engine.** No agent touches the OS, network, or user data without a permission decision that is logged.
- **The event log is the source of truth.** Memory, knowledge graph, and sync state are all derived, rebuildable indexes.
- **Measured self-improvement only.** No "reflection" without an evaluation harness; a system that cannot measure itself cannot improve itself.
- **Modular monolith first, services later.** Process boundaries where safety demands them (execution sandbox), not where fashion suggests them.
