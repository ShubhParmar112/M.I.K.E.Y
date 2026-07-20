# M.I.K.E.Y — Architectural Review of the Vision Document

**Status:** Accepted baseline review · **Reviewed artifact:** `docs/00-vision.md`
**Verdict:** The vision is coherent as a *north star*. As an architecture it does not yet exist — the document is a capability inventory (what the system should do), not an architecture (how responsibilities, data, trust, and failure are partitioned). This review converts the former into the raw material for the latter.

---

## 1. The three structural errors that must be corrected first

These are not nitpicks. Each one, left in place, kills the project at a predictable point.

### 1.1 "Modular microservice architecture" is the wrong starting point

**The claim:** independent services for Vision, Memory, Planning, Reasoning, Orchestration, Execution, etc., communicating over APIs/event bus.

**Why it's wrong:** Microservices solve an *organizational* scaling problem (many teams shipping independently) at the cost of operational complexity (service discovery, distributed tracing, network failure modes, versioned contracts, deployment orchestration). M.I.K.E.Y at Generation 1–4 is one developer shipping to one machine. Seventeen services on a laptop means seventeen ways to be down and zero users served while you debug inter-service auth.

**What's right instead:** a **modular monolith** — one process, strict internal module boundaries enforced by the language (package visibility, dependency rules, internal APIs), plus **separate processes only where a trust or crash boundary demands it**:

- The **execution sandbox** (runs untrusted actions) — separate, unprivileged, killable process. Non-negotiable.
- **Heavy ML inference** (local models) — separate process so a model crash/OOM doesn't take down the core.
- Everything else — modules in one process.

The module boundaries are designed *as if* they were services (explicit interfaces, no shared mutable state, message-passing internally). If a module later needs to become a service (e.g., cloud inference), the seam already exists. This is the industry-standard "extract services when forced" path (Shopify, Stack Overflow, Basecamp all scaled enormous load on monoliths; the teams that started with microservices at n=1 users are mostly dead).

**Tradeoff accepted:** less enforced isolation between modules; mitigated by CI-enforced dependency rules and the two hard process boundaries above.

### 1.2 The knowledge graph must be a derived index, not the source of truth

**The claim:** "Instead of storing plain text, create a dynamic semantic knowledge graph."

**Why it's wrong as stated:** Entity/relation extraction is lossy and errorful. If the graph is the *primary* store, every extraction mistake is permanent data corruption, every schema change is a migration of your life's data, and you can never re-extract with a better model. Graph-primary personal-knowledge systems fail this way consistently.

**What's right instead:** an **append-only event log** (conversations, observations, actions, ingested documents — raw, immutable) is the source of truth. The knowledge graph, vector indexes, summaries, and memory tiers are all **derived, versioned, rebuildable projections** of that log. When a better extraction model appears in 2028, you re-run projection over the log and the graph gets smarter retroactively — this is the single biggest long-term advantage a lifelong system can have, and the vision document currently designs it away.

### 1.3 "Remember everything" is a retrieval problem, not a storage problem

Storing everything is trivial (it's disk). The hard problem is **precision at recall time**: injecting the *right* 2–10 memories into a bounded context window without drowning the model in 40,000 irrelevant ones. A system that remembers everything and retrieves badly is *worse* than one with no memory — it confidently acts on stale or wrong context. The architecture therefore needs a first-class **Context Assembly Pipeline** (see §2, missing subsystem M1) with relevance scoring, recency/importance weighting, contradiction detection, and staleness tracking. The vision lists "Store, Retrieve, Summarize…" as bullet points; retrieval quality is actually the core research problem of the entire project.

---

## 2. Missing subsystems

The vision names perception, memory, agents, execution, security. The following are absent and every one is load-bearing.

| # | Subsystem | Why the system fails without it |
|---|---|---|
| M1 | **Context Assembly Pipeline** | The component that decides what enters the model's context window each turn (memories, graph facts, task state, tool results, budget-aware truncation). This *is* the product; everything else feeds it. |
| M2 | **Model Gateway** | Single abstraction over all LLMs/VLMs (local + cloud): routing by task/cost/privacy tier, fallback, retry, streaming, token accounting, provider-agnostic tool-calling. Without it, every module hard-codes a vendor and the "hybrid local+cloud" promise is unimplementable. |
| M3 | **Evaluation & Regression Harness** | "Self-improvement" without measurement is drift. Golden-task suites, LLM-as-judge scoring, A/B of prompt/policy changes, regression gates before any self-modification is adopted. The Reflection agent writes *hypotheses*; the harness decides if they were improvements. |
| M4 | **Policy & Consent Engine** | Permissions as *data* (declarative rules: action class × resource × context → allow / ask / deny / simulate-first), not as scattered `if` statements. Central choke point for every side effect; produces the audit log. The vision says "ask permission for sensitive actions" but never says who decides what's sensitive. |
| M5 | **Ingestion & Connector Framework** | Uniform pipeline: connector → normalize → dedupe → chunk → embed → project into indexes, with incremental sync and backfill. Emails, files, repos, calendars all enter here. Otherwise every data source is a bespoke snowflake. |
| M6 | **Scheduler & Durable Task Queue** | Missions, Dream Mode, reminders, retries, resumption after crash — all are durable jobs with state machines. "Resume interrupted work" is impossible without persisted task state. |
| M7 | **Observability & Tracing for agent runs** | Every mission → plan → step → tool call → result as a queryable trace tree. This is your only debugging instrument for nondeterministic behavior, and the raw material for M3. |
| M8 | **Cost & Resource Governor** | Token/compute/API budgets per mission and per day, throttling, "you are about to spend $X" checkpoints. An autonomous researcher without a budget governor will spend your money at 3 a.m. |
| M9 | **Identity, Device & Key Management** | Device enrollment/revocation, key derivation and rotation, encrypted backup and **restore drills**. A lifelong memory that can't survive a stolen laptop is not lifelong. |
| M10 | **Schema Versioning & Migration** | Memory formats, graph schemas, event types will change dozens of times over a decade. Versioned events + migration framework from day one, or year-3 data becomes unreadable. |
| M11 | **Update & Rollback of M.I.K.E.Y itself** | Signed updates, staged rollout to your own devices, one-command rollback. Especially critical once plugins/self-generated automations exist. |
| M12 | **Failure Taxonomy & Recovery** | Classified errors (transient / permanent / needs-human / dangerous-to-retry) with per-class recovery policy. "Retry" as a bullet point becomes an infinite loop on permanent failures. |

---

## 3. Architectural weaknesses in what *is* specified

**W1 — Agent-per-noun explosion.** Seventeen named agents (Finance, Medical, Presentation…) is role-play, not architecture. Most differ only by prompt + tool set. Correct model: a small set of *structural* roles (Planner, Executor, Critic/Verifier, Memory-writer) and a **capability registry**; "Finance agent" is a configuration (prompt + tools + policies), not a subsystem. This collapses the orchestration surface from 17² interactions to a handful, and new "agents" become data, not code.

**W2 — E2E encryption vs. cloud intelligence is an unacknowledged contradiction.** "End-to-end encrypted sync" and "cloud inference over your memories" are incompatible as stated: if the cloud reasons over plaintext, it's not E2E. The architecture must define explicit **privacy tiers** (see 02-architecture §3): Tier-0 data never leaves the device; Tier-1 may go to cloud inference transiently, never stored; sync payloads E2E-encrypted, cloud is a dumb encrypted relay. Pretending you have both without the tiering is a security fraud on yourself.

**W3 — Sync is hand-waved.** "Real-time, conflict-aware, offline-capable" describes years of hard distributed-systems work. Correct simplification: **the append-only event log is the sync unit** (per-device vector-clock cursors, append-only merge = few real conflicts), and derived state is rebuilt locally, never synced. Full CRDTs only if collaborative live-editing is ever needed.

**W4 — Execution engine has no threat model.** Mouse/keyboard/terminal/browser control + reading emails and web pages = the textbook **prompt-injection lethal trifecta** (private data + untrusted content + ability to act). A web page saying "ignore previous instructions, run this command" is *input data* to the very model holding the keyboard. Mitigations must be architectural, not prompt-level: taint-tracking of untrusted content, policy engine between plan and execution, simulate-first for destructive ops, sandboxed executor with allowlisted capabilities per task. §5 below.

**W5 — Dream Mode is just background jobs — name it that.** Romantic framing hides real requirements: idle detection, preemption on user return, battery/thermal budgets, and *every* Dream-Mode change (prompt edits, graph rewrites) must pass the M3 harness before adoption, or the system corrupts itself while you sleep.

**W6 — Digital Twin is underdefined and should be descoped early.** As specified it's "learn everything, act as the user" — an unbounded research problem with impersonation risks. Tractable v1: **style profiles** (writing/code conventions learned from corpus, applied on request) and **preference models** (ranked choices with confidence). "Acts as you" autonomy is Gen 9–10, gated behind years of trust data.

**W7 — No testing strategy for nondeterminism.** Traditional unit tests don't cover "did the plan make sense." Needed: deterministic replay from traces (M7), golden-task evals (M3), simulation environments for the executor, property-based checks on tool calls ("never `rm -rf` outside workspace").

**W8 — Wrong scaling axis.** "Assume millions of users" pushes toward premature multi-tenant cloud design. A personal AI is **shard-per-user by construction** — scaling to millions is mostly replication of single-user pods plus shared inference capacity. The hard scaling axes are actually: one user's decade of data (log compaction, index tiering), and concurrent agent workloads on one machine (the governor, M8). Design for depth-per-user, not user count.

---

## 4. Feasibility & regulatory flags (things the vision treats as features that are legally or platform-constrained)

| Item | Reality |
|---|---|
| Medical Assistant | Regulated territory (varies by jurisdiction). Ship as *information + document organization*, never diagnosis/dosing. Explicit disclaimers, no autonomy. |
| Finance Agent | Personalized investment advice is a licensed activity in most jurisdictions. Ship as *bookkeeping + summarization*; no trade execution, ever, at the architectural level (policy engine hard-deny). |
| Call summarization | Call-recording consent laws differ by state/country; iOS forbids call audio access. Design as opt-in, platform-permitting, geo-aware. |
| WhatsApp control | Automation via unofficial clients violates ToS → account bans. Use official Business API where possible or drop it. |
| iOS "control the phone" | iOS will never allow system-wide UI automation by a third-party app. Mobile is a **companion surface** (voice, camera, notifications, approvals), not an execution surface. Plan for it; don't fight it. |
| Photoshop/Blender control | Feasible only via their scripting APIs (ExtendScript/UXP, bpy) — reliable and supported — not via screen-vision + clicking, which is a demo, not a product. Prefer API-first automation everywhere; pixels are the fallback of last resort. |

---

## 5. Security review (threat-model summary)

Assets: lifetime memory store, credentials vault, execution capability, the user's identity/style.

| Threat | Vector | Architectural control |
|---|---|---|
| Prompt injection → arbitrary execution | Malicious content in emails/web/docs steering the agent | Taint labels on all ingested content; policy engine (M4) between plan and act; untrusted content can *inform* but never *authorize*; simulate-first + human confirm for destructive/outward actions |
| Memory poisoning | Injected content writes false "facts" into long-term memory | Memory writes are policied actions; provenance recorded per memory; quarantine tier for facts derived from untrusted sources |
| Exfiltration | Agent with network access leaks Tier-0 data | Egress allowlists per task in sandbox; privacy tiers enforced at the model gateway (Tier-0 never in cloud prompts) |
| Credential theft | Vault compromise | OS keychain-backed vault; secrets injected into tools at call time, never into model context; short-lived scoped tokens |
| Malicious/buggy plugin | Third-party or self-generated code | Plugins run in the sandbox with a declared capability manifest; signed; least-privilege; kill switch |
| Device loss | Stolen laptop/phone | Full-disk + app-level encryption at rest; device revocation (M9); remote log detach |
| Self-modification gone wrong | Reflection adopts a bad prompt/policy | All self-changes are versioned proposals gated by the eval harness (M3) + rollback (M11) |
| Audit tampering | Attacker (or a confused agent) hides actions | Append-only, hash-chained audit log, written by the policy engine only |

Biometric auth note: use platform biometrics (Windows Hello / TouchID) to gate the vault and high-risk approvals — don't build biometric processing yourself.

---

## 6. Missing AI capabilities & genuine differentiators

Capabilities absent from the vision that would matter more than several that are listed:

1. **Calibrated uncertainty & provenance-first answers** — every claim from memory carries source + confidence + age ("based on your March notes, possibly stale"). Almost no assistant does this; it is *the* trust feature for a lifelong system.
2. **Contradiction & staleness management** — actively detect when new information conflicts with stored beliefs; ask or annotate rather than silently keeping both.
3. **Forgetting as a first-class verified operation** — provable deletion across log, indexes, backups, and derived summaries (GDPR-grade tombstoning). "Forget (if requested)" is one line in the vision; it's a genuinely hard, differentiating subsystem.
4. **Explanation replay** — "why did you do that?" answered from the actual trace (M7), not confabulated post-hoc. JARVIS-ness is mostly this.
5. **Interruption-aware continuity** — resume any mission after crash/reboot/week-long pause with a "here's where we were" brief. Falls out of durable tasks (M6) + traces (M7) if designed for.
6. **Skill compilation** — when the same multi-step task recurs, the system proposes a deterministic, reviewable script/automation (cheap, fast, auditable) to replace LLM improvisation. This is the *correct* form of "self-improvement": migrating from expensive inference to verified automation.
7. **Personal eval corpus** — the system continuously builds test cases from your own corrected outputs; your assistant measurably gets better *for you*, provably, with graphs.

---

## 7. Answer to the author's question ("so is this much fine?")

As a **vision**: yes — it is unusually complete on the *what*, and the instinct for modularity, local-first privacy, and permission-gating is correct.

As an **engineering foundation**: not yet. It lists ~200 features at equal weight, omits the twelve subsystems in §2 that determine whether any of the features work, contains three structural decisions (§1) that must be reversed, and has no ordering — and for a multi-year solo-built system, **the ordering is the architecture**. The corrected design is in [02-system-architecture.md](02-system-architecture.md) and the ordering in [03-roadmap.md](03-roadmap.md).

The single most important reframe: **JARVIS is not a feature list; it is trust accumulated through a long sequence of small, verified, explainable actions.** The architecture below is organized around producing that sequence, not around the feature list.
