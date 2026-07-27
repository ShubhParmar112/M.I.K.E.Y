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
uv run pytest                # verify (328 passing)

# pick a model provider — and ideally more than one (see below):
$env:GROQ_API_KEY = "gsk_..."               # cloud, free: fastest, ~100k tokens/day
$env:CEREBRAS_API_KEY = "csk-..."           # cloud, free: ~1M tokens/day
$env:GEMINI_API_KEY = "AIza..."             # cloud, free: metered in requests/day
$env:ANTHROPIC_API_KEY = "sk-ant-..."       # cloud, paid: the strongest of them
# install Ollama + `ollama pull llama3.2`   # local / private / offline

uv run mikey providers       # who can answer, in what order, and what's missing
uv run mikey doctor          # check providers, local models, brain routing, integrity
uv run mikey chat            # interactive chat with approval cards + trace
```

**Configure more than one.** A single free tier is a single point of failure with
a silent failure mode: its daily allowance runs out mid-evening, every remaining
answer is served by the 3B local model, and nothing announces it except the
answers getting worse. Every key you add is an independent daily allowance —
M.I.K.E.Y walks the chain (`groq → cerebras → gemini → ollama`), and a provider
that reports it is out for the day is set aside rather than asked again on every
call. The local model goes back to being what it should be: the offline net.

After `uv sync`, run any command as `uv run mikey <command>` (or `mikey <command>` with the venv active). Run `mikey doctor` anytime to see your effective setup.

Data lives in `~/.mikey/` (event log, audit chain, traces); the agent's sandbox is `~/.mikey/workspace/`. Reads are auto-allowed; writes and commands require approval (`y` once / `s` for the session); unknown tools are denied. Web content is taint-marked and can never authorize actions.

M.I.K.E.Y also reaches its memory *during* a conversation: `memory_recall` searches long-term memory on demand (not just the handful pre-loaded each turn) and `memory_remember` persists a durable fact when you ask it to — so "remember my dog is named Pixel" now sticks, and "what's my dog's name?" later retrieves it with provenance. Recalling an untrusted memory taints the turn just like ingested files do.

Remembered facts are kept clean: a near-duplicate is skipped, a correction can `supersede` (tombstone) the stale fact it replaces, and a related-but-conflicting memory is flagged so it gets reconciled rather than silently doubling up. `memory_forget` lets M.I.K.E.Y drop a specific memory on request — a verified deletion from every projection, gated behind an approval card because it's destructive.

## Commands

**Chat & reasoning**

| Command | What it does |
|---|---|
| `mikey chat` | Interactive session — routes each turn through the brain fleet, streams actions + approval cards (with the critic's note), header shows which brains run locally. **Each run is a new conversation**; `--continue` resumes the last one, `--session <id>` names one. `/new`, `/trace` and `/quit` work inside. |
| `mikey voice` | Talk to it and hear it back. Hearing and (by default) speaking run **on this machine**; same brains, same approval cards, same traces as `mikey chat`. `--synth edge` for a neural voice, `--mute` to listen but reply in text. Needs `uv sync --extra voice`. |
| `mikey serve` | Run the gateway in the foreground (for a separate terminal). |
| `mikey trace [turn_id]` | "Why did you do that?" — the full reasoning tree for a turn (route → model call → policy → tool). Defaults to the last turn. |
| `mikey events [--limit N]` | Recent raw events from the append-only log. |
| `mikey brief [--hours N] [--speak]` | What happened and anything worth knowing — composed from the log, so it costs no quota and can't invent anything. |
| `mikey nudges [--dismiss <id\|kind>]` | What M.I.K.E.Y is waiting to tell you, and what it has stopped mentioning. Dismissing a kind enough times mutes it for good. |

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
| `mikey spend` | This month's model spend per provider against the budget, and today's usage against each free-tier **daily** allowance — the limit that actually runs out and silently drops answers onto the weak local model. Providers are gauged by whichever allowance binds first (Groq counts tokens/day, Gemini counts requests/day). |
| `mikey providers` | The whole answer chain: who is primary, who backs it up, which keys are missing, and what each free tier is worth. Tells you if you are one exhausted quota away from the 3B model. |

## Speaking first

The gateway is the only thing always running, so it is the only thing that can
notice something while nobody is watching. Every few minutes it looks at the log
— no model call, so this costs nothing and cannot hallucinate — and records a
**nudge** when it finds one of a small set of things whose first symptom is
otherwise "why has it got worse?": a daily allowance about to run out, answers
already coming from the local model, a mission that stopped moving, a spent
budget, a broken audit chain.

A nudge is an event, so what is outstanding is a projection like everything else:
it survives a restart, and "why did you tell me that?" has an answer in the log.

The interesting part is the restraint (`core/proactive/discipline.py`), because a
system that says true things at wrong moments gets muted — and a muted assistant
is worse than a silent one, since you have also stopped trusting it:

- **Never mid-turn.** Nothing is volunteered while you're waiting on an answer.
- **Quiet hours** (22:00–08:00) hold everything except the genuinely urgent.
- **A ceiling** of three volunteered remarks per session and four per hour; past
  it the rest waits, most pressing first rather than most recent.
- **Dismissing means something.** A dismissed nudge stays gone for a day rather
  than returning on the next tick, and a kind you wave away three times stops
  being raised at all — no setting to find, you just keep saying no.
- **Stale notes are dropped, not shown late.** "About four calls left today" is
  useful for an hour and misleading tomorrow.

Set `MIKEY_PROACTIVE=0` to keep it strictly reactive, or
`MIKEY_PROACTIVE_INTERVAL` to change how often it looks (default 300s).

## Voice

```powershell
uv sync --extra voice          # speech model + audio bindings (optional)
uv run mikey voice             # speak; it speaks back
```

Speech recognition runs locally (faster-whisper on the CPU): **what you say never
leaves the machine.** The default voice is Windows' own — offline, private, and
frankly robotic. `--synth edge` swaps in Microsoft's neural voices, which sound
human but send the text of each spoken reply over the network; a **Tier-0 (private)
turn is never given to a cloud voice** and falls back to the local one, the same
rule the model gateway enforces at the other end of the turn.

Three behaviours are worth knowing, because they're deliberate:

- **A spoken word never approves an action.** When something needs approval,
  M.I.K.E.Y reads the request aloud and then waits for the keyboard. A television,
  a housemate, or a video call can all say "yes"; none of them are you.
- **Silence isn't a question.** Handed a cough or a door, a transcriber returns
  confident text ("Thank you." is the classic). Anything that isn't clearly speech
  is dropped and shown as dropped, rather than spent as a turn.
- **You can talk over it.** Say "stop" while it's speaking and it stops mid-word;
  "goodbye" ends the session.

What it says is not what it wrote: code blocks, tables and URLs are described
rather than read out, arithmetic is verbalised ("180 × 190 / 200" → "180 times 190
over 200"), and a long answer stops early and says the rest is on screen.

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

### The other budget: tokens per day

On a paid plan the binding limit is dollars per month. On a **free** plan it is
**tokens per day**, and running out doesn't look like an error — the provider
429s, the gateway falls back, and every remaining turn that day is served by a
much weaker local model. That is what an evening of unexplainably bad answers
actually was. So the same ledger is also read per-day, per-provider, against the
free-tier allowance (Groq: 100k tokens/day):

```powershell
$env:MIKEY_DAILY_TOKEN_CAP = "0"   # 0 = use the built-in free-tier table; set your real allowance if you've upgraded
```

`mikey spend` shows today's consumption and roughly how many calls are left at
today's average call size, and `mikey chat` says so in the banner once the day's
allowance is running out — before the answers get worse, not after. It is a
gauge, deliberately **not** a gate: the count only includes calls M.I.K.E.Y made
(a lower bound on the provider's own tally) and a provider's daily window need
not match the local calendar day, so the 429 remains the authority on when to
stop.

The gauge tells you the cliff is coming. What stops you falling off it is having
somewhere else to go: **configure a second free provider** (`mikey providers`).
When one reports a daily cap, M.I.K.E.Y sets it aside — for the time the provider
itself named, or an hour if it named none — and the next cloud model in the chain
answers instead, at comparable quality. Only when every cloud provider is out or
offline does the local 3B model take a turn, and the chat banner says which of
those two things happened, because they mean very different things about how much
to trust the answer.

## Brains & local-first routing

Every turn is routed to one of a small fleet of **brains** — each a capability profile (prompt + tool allowlist), not a separate model:

| Brain | Role | Tools |
|---|---|---|
| `conversation` | greetings, sign-offs, small talk | none |
| `reasoning` | closed problems — maths, logic, puzzles: derive, then verify | none |
| `operator` | actionable turns, questions, recall/remember | all except `memory_forget` |
| `memory` | forgetting / correcting stored memory | recall, remember, forget |
| `critic` | reviews a risky action before you approve it | none (judges) |
| `verifier` | independently re-checks a reasoning answer before you see it | none (re-derives) |
| `planner` | turns a goal into a durable mission | none (proposes) |
| `router` | picks the brain per turn | heuristic (always local) |

A self-contained problem goes to `reasoning`, which holds **no tools** — so a lookup can't stand in for the algebra — and is prompted to derive, substitute its answer back into the original conditions, and stop once the check passes. Routing is sticky across a problem: the "are you sure?" that follows stays on the same brain instead of switching mid-derivation.

Model replies are also screened for the three ways an open-weight model runs away instead of answering: a **repetition collapse** (re-sampled once with anti-repetition settings), a **restart after the answer** — "let's try another approach" followed by guessed values — which is trimmed, and an **asserted answer**: the reply retracts its own working and then simply announces a result it never computed ("Let me re-evaluate again… Ah-ha! I found it! The answer is indeed 29"). All three are recorded in the turn's trace, so a recovered reply is visible rather than silently smoothed over.

An asserted answer is the dangerous one, because it can be *right* — recalled from training data rather than derived — and it only falls apart when you ask "how did you get that?". So a flagged answer is handed to the **verifier**, a separate brain that gets the original problem and none of the first brain's reasoning, and is asked whether the working actually establishes the answer. If it doesn't, M.I.K.E.Y re-derives once with the concern fed back; if the check still fails, the answer ships with the concern attached rather than as a confident number nobody can check. `MIKEY_VERIFY_REASONING` = `flagged` (default — a second call only when the reply looks asserted), `always`, or `off`.

Before any of that costs a model call, the reply's own arithmetic is **audited
deterministically** (`core/models/arithmetic.py`): every claim of the form
`<arithmetic> = <number>` is extracted and evaluated, so `(0.92 × 34,240) = Rs.
17,940 (True)` is caught as the false step it is — instantly, free, and just as
reliably on a 3B as on a 70B. Arithmetic that doesn't hold is proof rather than
opinion, so it goes straight to a re-derivation naming the bad steps instead of
asking a model to grade the sum it just got wrong. Claims involving a symbol
(`M − 16,300 = 17,940`) are skipped: a false accusation would teach you to ignore
the warnings, so the check only fires on what it can prove.

Verification is deliberately three-valued — **confirmed**, **checked and found wanting**, and **could not be checked** — because a verifier that is down, or too weak to return a verdict at all, must never be mistaken for a clean bill of health. Only an explicit verdict counts as confirmation; anything else reaches you as "treat this as unverified". A weak local model verifying itself is the case this guards against, and it is a real one.

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
| Model gateway (Groq / Cerebras / Gemini / Anthropic / Ollama / fake · failover chain · tier + capability routing) | ✅ `core/models/` |
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
