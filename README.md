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
uv run pytest                # verify (262 passing)

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
| `mikey reasoning-eval [--against <provider>] [--pace N]` | Score tool-use + reasoning on the golden set; `--against ollama` shadow-compares cloud vs local (the gate before localizing a reasoning brain). Cases the provider never answered (429 / outage) are excluded from the pass rate and reported separately, and a run with thin coverage is labelled **inconclusive** rather than scored — a rate limit must not be readable as a quality number in either direction. `--pace` (default 20s) keeps a free-tier key under its **per-minute** token budget (Groq free tier: 12k TPM, ~2.4k input per case, so ~3 cases/min — the full set takes ~5 min). A **per-day** cap (100k TPD) is a different thing entirely and pacing cannot help it; the 429's reason names which limit was hit. |
| `mikey export [--out DIR] [--include-t0]` | Export the event log → per-brain training datasets (respects tombstones + privacy tiers). |
| `mikey eval [--update-baseline]` | Retrieval-quality eval against the golden set. |

**Ops & safety**

| Command | What it does |
|---|---|
| `mikey backup` | Verified snapshot of the whole store (log + audit chain). |
| `mikey restore <path> [--yes]` | Restore from a backup (verifies it, snapshots current state first). |
| `mikey spend` | This month's model spend per provider against the budget, and what happens when it runs out. |

## Simulate first, then ask

Nothing that can destroy data is approved from its arguments alone. Before a
destructive action reaches an approval card, M.I.K.E.Y works out what it would
actually do and shows you that:

| Action | What the card shows |
|---|---|
| `fs_write` over an existing file | the unified diff — the lines that disappear, not just "a write happens" |
| `git clean`, `git rm` | the real `--dry-run`, so you see the file list before anything is deleted |
| `git reset --hard`, `git checkout --` | the uncommitted work that would be discarded |
| `memory_forget` | the text of the memory being deleted, with its source and date |

A write that turns out to *create* a file is marked safe and stays frictionless.
The invariant is enforced, not merely offered: a standing "approve for this
session" grant covers routine writes but is **withdrawn** the moment a preview
shows data would be lost — and that withdrawal is written to the audit chain, so
it is as reconstructable as the grant it overrode. A preview that fails still
produces a card, flagged unsimulated; silence is the one outcome that is not
allowed. Mission steps get the same treatment — unattended work is exactly where
an unpreviewed overwrite would go unnoticed.

## The cost governor

Model spend is a projection over the event log: every call appends a
`model.usage` event with its tokens and cost, so the month-to-date total survives
a restart, a rebuilt index, and a restore-from-backup. The budget is enforced at
the model gateway — the same single door as the Tier-0 privacy rule — and when
it runs out, **cloud adapters drop out of the chain and the local model keeps
serving**: the meter stops, the assistant doesn't. Only with no local model at
all does it become an error.

```powershell
$env:MIKEY_MONTHLY_BUDGET_USD = "10"   # default; "0" disables enforcement (tracking continues)
```

Prices are a dated, approximate table (`core/cost/governor.py`) — a model that
isn't in it is charged at a deliberately high fallback rate rather than zero,
because an unpriced model reading as free is how a budget silently stops binding.
`mikey spend` says when a month's total was estimated that way.

## Brains & local-first routing

Every turn is routed to one of a small fleet of **brains** — each a capability profile (prompt + tool allowlist), not a separate model:

| Brain | Role | Tools |
|---|---|---|
| `conversation` | greetings, sign-offs, small talk | none |
| `reasoning` | closed problems — maths, logic, puzzles: derive, then verify | none |
| `operator` | actionable turns, questions, recall/remember | all except `memory_forget` |
| `memory` | forgetting / correcting stored memory | recall, remember, forget |
| `critic` | reviews a risky action before you approve it | none (judges) |
| `planner` | turns a goal into a durable mission | none (proposes) |
| `router` | picks the brain per turn | heuristic (always local) |

A self-contained problem goes to `reasoning`, which holds **no tools** — so a lookup can't stand in for the algebra — and is prompted to derive, substitute its answer back into the original conditions, and stop once the check passes. Routing is sticky across a problem: the "are you sure?" that follows stays on the same brain instead of switching mid-derivation.

Model replies are also screened for the two ways an open-weight model runs away instead of answering: a **repetition collapse** (re-sampled once with anti-repetition settings) and a **restart after the answer** — "let's try another approach" followed by guessed values — which is trimmed. Both are recorded in the turn's trace, so a recovered reply is visible rather than silently smoothed over.

Brains are served by a **cloud primary with a local (Ollama) fallback** — auto-routing to Ollama on rate-limit/offline. Any brain can also be pinned **on-device**, one at a time, so its calls never leave the machine:

```powershell
$env:MIKEY_LOCAL_BRAINS = "conversation"   # serve chit-chat locally; keep reasoning on cloud
$env:MIKEY_OLLAMA_MODEL = "llama3.2"       # the local model used for pinned brains
```

Useful env knobs: `MIKEY_LOCAL_BRAINS` (brains to run locally), `MIKEY_OLLAMA_MODEL` (local model), `MIKEY_LOCAL_FALLBACK=0` (disable the fallback), `MIKEY_PROVIDER` / `MIKEY_MODEL` (cloud choice), `MIKEY_TEMPERATURE` (default `0.3` — very low values make open models loop), `MIKEY_MAX_OUTPUT_TOKENS` (default `1536`), `MIKEY_HOME`, `MIKEY_WORKSPACE`. `mikey doctor` prints the effective result. See [docs/04-intelligence-sovereignty.md](docs/04-intelligence-sovereignty.md) for the full local-migration plan.

## Status

| Component | State |
|---|---|
| Event log (append-only, versioned schema, SQLite WAL) | ✅ `core/events/` |
| Context assembly (history + provenance-annotated memory, traced) | ✅ `core/context/` |
| Model gateway (Groq / Anthropic / Ollama / fake · tier + capability routing) | ✅ `core/models/` |
| Policy engine + hash-chained audit + taint rule | ✅ `core/policy/` |
| Executor sandbox (separate process, path confinement, command allowlist) | ✅ `executor/` |
| Turn loop + brain fleet (router · conversation · reasoning · operator · memory · critic · planner) | ✅ `core/orchestrator/` |
| Durable missions (multi-step, policy-gated, resume after reboot) | ✅ `core/missions/` |
| Simulate-first previews for destructive actions (diff / dry-run / memory text) | ✅ `core/policy/preview.py` |
| Cost governor (spend ledger in the log, monthly budget enforced at the gateway) | ✅ `core/cost/` |
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
