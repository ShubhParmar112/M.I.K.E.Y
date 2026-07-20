# M.I.K.E.Y — Development Roadmap: Generation 1 → 10

**Rule of the roadmap:** each generation ships something you *use daily* and has measurable exit criteria. A generation is not "done" when its features exist; it is done when its exit criteria hold for 30 consecutive days of real personal use. Trust is the product; the roadmap is a trust-accumulation schedule.

Later generations assume the model landscape keeps improving — the event-log architecture (02 §4) is what lets M.I.K.E.Y absorb better models retroactively.

---

## Gen 1 — The Trustworthy Core (foundation)
The spine, end to end, tiny surface: desktop app + CLI, chat with one strong cloud model + one local model behind the **Model Gateway**; **event log**; basic **Context Assembly** (recent turns + simple retrieval); **Policy Engine** with approval cards; **Executor Sandbox** with three tools only — filesystem (scoped), terminal (allowlisted), web fetch; **trace viewer** ("why did you do that" works from day one).
*Deliberately absent:* voice, vision, graph, missions, mobile, plugins.
**Exit criteria:** 30 days daily use; zero unauthorized side effects; every action reconstructable from the audit trail; restore-from-backup drill passes.

## Gen 2 — Memory That Deserves the Name
**Ingestion framework** + first connectors (files, git, calendar, one mail account); memory tiers with promotion, provenance, confidence, staleness; **verified forgetting**; contradiction flagging; personal-corpus beginnings of the **eval harness**.
**Exit criteria:** "what do you know about X and where from?" answers with sources; a deleted memory is provably gone from every projection; retrieval precision measured on a 50-case personal golden set.

## Gen 3 — Real Work (execution depth)
Full tool suite on desktop via **API-first automation** (git/GitHub, VS Code/LSP, office docs, browser via DevTools protocol); **durable missions** (multi-step DAGs, resume after reboot); failure taxonomy + recovery policies; **cost governor**; simulate-first for destructive ops.
**Exit criteria:** a 10+ step mission (e.g., "set up repo, scaffold app, write tests, open PR") survives a mid-mission reboot and completes; zero destructive actions without preview; monthly spend within budget automatically.

## Gen 4 — Everywhere (sync + mobile companion)
E2E-encrypted **event-log sync** through a dumb relay; device enrollment/revocation; mobile companion app: voice + camera capture → ingestion, notifications, memory/mission views, and **remote approval cards** for desktop actions.
**Exit criteria:** start a task on desktop, approve its risky step from the phone, review the result on the web dashboard — with the relay server fully untrusted (verified: server sees only ciphertext).

## Gen 5 — Connected Knowledge (graph + research)
**Knowledge graph as derived projection** with per-edge provenance; graph-aware context assembly ("who/what is related"); **research profile**: literature search, paper ingestion, citation graph, comparison tables, report + deck generation. First **capability-profile registry** (research, coding as profiles, not code).
**Exit criteria:** graph fully rebuilds from the log with a newer extractor and measurably improves on the golden set; one real research survey produced end-to-end with verifiable citations.

## Gen 6 — Ears and Eyes (perception)
Local wake word + streaming ASR + TTS (T0 audio path stays on-device); screen understanding on explicit request; OCR/PDF/handwriting through the ingestion pipeline; camera-based capture on mobile matured.
**Exit criteria:** hands-free daily driving of Gen 1–5 features; perception errors degrade gracefully (system states uncertainty instead of acting on misreads).

## Gen 7 — The Improvement Engine (measured self-improvement)
Eval harness matured into a gate: **reflection proposals** (prompt/profile changes, **skill compilation** of recurring tasks into deterministic scripts) adopted only on eval pass, with rollback; Dream-Mode maintenance jobs (compaction, re-embedding, re-extraction) under resource budgets.
**Exit criteria:** ≥3 self-proposed changes adopted through the gate with measured improvement; ≥1 recurring workflow compiled to a reviewed automation that runs without LLM calls; zero regressions shipped by Dream Mode.

## Gen 8 — The Ecosystem (plugins + SDK)
Public module contract (MCP-based): manifest-declared capabilities, sandboxed execution, signing, kill switch; browser extension; REST API/SDK stabilized (versioned, documented); community-installable connectors.
**Exit criteria:** a third party (or you, cleanly) builds a working plugin without touching core; a malicious test plugin is contained by the sandbox in a red-team exercise.

## Gen 9 — The Companion (proactivity + style)
**Style profiles** (writing/code/communication) applied on request and measurably matching your corpus; calibrated proactivity: suggestions with confidence, learned from accept/reject history; predictive assistance (deadline risk, meeting prep) with a strict interruption budget; smartwatch/IoT surfaces as notification+approval endpoints.
**Exit criteria:** proactive suggestion acceptance rate tracked and >50%; style-profile drafts accepted with minor edits most of the time; interruption budget never exceeded.

## Gen 10 — Bounded Autonomy (the JARVIS threshold)
Standing delegations ("handle my routine PRs", "triage my inbox") executed autonomously **within policy envelopes earned from years of audit history**; digital-twin assistance for drafting-as-you with mandatory review on anything outward-facing; graceful degradation of autonomy whenever confidence drops.
**Exit criteria:** weeks-long standing missions run with zero policy violations; every autonomous action remains explainable from traces; the user reports — this is the real metric — that they *trust it*.

---

## Sequencing rationale

- **Safety before capability** (Gen 1 before 3): an executor without the policy spine is a liability compounding daily.
- **Memory before graph** (Gen 2 before 5): the graph is a projection; it needs a log worth projecting.
- **Sync before perception** (Gen 4 before 6): multi-device is architectural; voice is a feature on top.
- **Evals before autonomy** (Gen 7 before 10): autonomy is granted by measurement, not by optimism.
- Regulatory-hazard profiles (medical, finance) remain information-only at every generation; the policy engine hard-denies execution classes in those domains permanently (review §4).
