"""Session Gateway — the only API surface. Every client (CLI today; TUI, web,
mobile later) speaks these endpoints and nothing else (architecture 02 §12).

Binds to localhost only in Gen 1.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.config import CONFIG, Config, available_cloud_providers
from core.cost.governor import CostGovernor
from core.events.schema import EventType
from core.events.store import EventStore
from core.executor_client import ExecutorClient
from core.ingest.files import FileIngestor
from core.memory.store import MemoryStore
from core.models.fake_adapter import FakeAdapter
from core.models.gateway import ModelAdapter, ModelGateway
from core.missions.store import MissionStore
from core.orchestrator.critic import Critic
from core.orchestrator.loop import ApprovalRegistry, Orchestrator, stream_event_json
from core.policy.engine import PolicyEngine
from core.proactive.nudge import NudgeStore
from core.proactive.watch import Sentinel
from core.reach.projects import ProjectRegistry
from core.storage.db import Database
from core.trace.store import TraceStore


def _make_adapter(config: Config, provider: str | None = None) -> ModelAdapter:
    """Build the adapter for `provider` (default: the configured one). The
    override lets tooling — e.g. `mikey reasoning-eval --against ollama` — build a
    second provider's adapter to shadow-compare, without touching config."""
    provider = provider or config.provider
    if provider == "anthropic":
        from core.models.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(config.anthropic_model)
    if provider == "groq":
        from core.models.groq_adapter import GroqAdapter

        return GroqAdapter(
            config.groq_model,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
        )
    if provider == "cerebras":
        from core.models.cerebras_adapter import CerebrasAdapter

        return CerebrasAdapter(
            config.cerebras_model,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
        )
    if provider == "gemini":
        from core.models.gemini_adapter import GeminiAdapter

        return GeminiAdapter(
            config.gemini_model,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
        )
    if provider == "ollama":
        from core.models.ollama_adapter import OllamaAdapter

        return OllamaAdapter(
            config.ollama_base_url,
            config.ollama_model,
            temperature=config.temperature,
            num_predict=config.max_output_tokens,
        )
    return FakeAdapter()


def _make_fallbacks(config: Config) -> list[ModelAdapter]:
    """An ordered failover chain: every OTHER cloud provider whose key is present,
    in preference order, then the local model last for offline coverage — e.g.
    groq → cerebras → gemini → ollama.

    The shape matters more than the order. With one cloud key, running out of its
    daily allowance means every answer for the rest of the day comes from a 3B
    local model, which is the difference between an assistant and a toy; that is
    what actually happened here on a Monday evening. Each additional free tier is
    an independent daily allowance, and they run out under different conditions
    (Groq counts tokens per day, Gemini counts requests, Cerebras limits requests
    per minute), so the chain degrades in steps rather than off a cliff.
    """
    chain: list[ModelAdapter] = []
    for provider in available_cloud_providers():
        if provider != config.provider:
            chain.append(_make_adapter(config, provider))
    # The local model is the last link, and only behind a cloud primary: it costs
    # nothing and works offline, which is exactly what a safety net should be — and
    # exactly what it should not be asked to do while any cloud model can answer.
    if config.local_fallback and config.provider != "ollama":
        from core.models.ollama_adapter import OllamaAdapter

        chain.append(
            OllamaAdapter(
                config.ollama_base_url,
                config.fallback_ollama_model,
                temperature=config.temperature,
                num_predict=config.max_output_tokens,
            )
        )
    return chain


def _make_routes(config: Config) -> dict[str, ModelAdapter]:
    """Pin the configured brains to the local model (sovereignty S2). Config lists
    brain names (conversation, memory, critic, ...); we resolve each to its
    capability — the key RoutingMeta carries — so the gateway serves it locally,
    with cloud fallback preserved. Empty unless MIKEY_LOCAL_BRAINS is set."""
    if not config.local_brains:
        return {}
    from core.models.ollama_adapter import OllamaAdapter
    from core.orchestrator.brains import BRAINS

    local = OllamaAdapter(
        config.ollama_base_url,
        config.ollama_model,
        temperature=config.temperature,
        num_predict=config.max_output_tokens,
    )
    routes: dict[str, ModelAdapter] = {}
    for name in config.local_brains:
        brain = BRAINS.get(name)
        if brain is not None:
            routes[brain.capability] = local
    return routes


def _make_governor(config: Config, events: EventStore) -> CostGovernor:
    """The monthly budget, backed by the event log as its ledger (Gen 3)."""
    return CostGovernor(
        events, config.monthly_budget_usd, config.device_id, config.daily_token_cap
    )


def _make_gateway(
    config: Config,
    adapter: ModelAdapter | None = None,
    events: EventStore | None = None,
) -> ModelGateway:
    """The one place a real gateway is assembled — used by the server and the CLI
    planner alike, so both honor the same primary/fallback chain, S2 routes, and
    monthly budget. Pass `events` so the calls are billed to the shared ledger;
    without it the gateway runs ungoverned (tests, one-off tooling)."""
    governor = _make_governor(config, events) if events is not None else None
    if adapter is not None:
        return ModelGateway(adapter, governor=governor)
    return ModelGateway(
        _make_adapter(config),
        fallbacks=_make_fallbacks(config),
        routes=_make_routes(config),
        governor=governor,
    )


def build_id() -> str:
    """Short git hash of the running code, so a stale gateway is identifiable."""
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class TurnRequest(BaseModel):
    session_id: str = "default"
    input: str


class ApprovalDecision(BaseModel):
    approved: bool
    scope: str = "once"  # "once" | "session"


class NudgeDecision(BaseModel):
    # Module level, like the model above and unlike the request bodies declared
    # inside create_app: this file uses `from __future__ import annotations`, so a
    # handler's annotations are strings resolved against module globals. A model
    # defined in a local scope is invisible there, and FastAPI silently falls back
    # to reading the parameter off the query string — which is how the first live
    # POST to this endpoint failed with "field required".
    outcome: str = "shown"  # "shown" | "dismissed"
    how: str = "chat"


def create_app(config: Config = CONFIG, adapter: ModelAdapter | None = None) -> FastAPI:
    config.ensure_dirs()
    db = Database(config.db_path)
    events = EventStore(db)
    embedder = None
    if config.local_vectors:
        from core.models.embeddings import OllamaEmbedder

        embedder = OllamaEmbedder(config.ollama_base_url, config.embed_model)
    memory = MemoryStore(db, events, embedder)
    ingestor = FileIngestor(memory, config.device_id)
    traces = TraceStore(db)
    policy = PolicyEngine(db)
    approvals = ApprovalRegistry()
    executor = ExecutorClient(config.workspace)
    # A caller-supplied adapter (tests) runs solo; the real server gets the
    # cloud→local hybrid + any per-brain S2 routes so a rate limit or dropped
    # connection isn't fatal and localized brains are served locally.
    gateway = _make_gateway(config, adapter, events)
    projects = ProjectRegistry(events, config.device_id)
    orch = Orchestrator(
        config, memory, traces, policy, gateway, executor, approvals,
        critic=Critic(gateway), projects=projects,
    )

    # Proactivity (Gen 4): the gateway is the only thing that is always running, so
    # it is the only thing that can notice something while nobody is watching.
    nudges = NudgeStore(events, config.device_id)
    missions = MissionStore(events, config.device_id)
    sentinel = Sentinel(
        config, events, nudges, _make_governor(config, events), missions, policy
    )

    app = FastAPI(title="M.I.K.E.Y Gateway", version="0.1.0")
    app.state.policy = policy
    build = build_id()

    @app.post("/v1/turns")
    async def run_turn(req: TurnRequest) -> StreamingResponse:
        async def sse() -> Any:
            async for ev in orch.run_turn(req.session_id, req.input):
                yield f"data: {stream_event_json(ev)}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.post("/v1/approvals/{approval_id}")
    async def decide(approval_id: str, decision: ApprovalDecision) -> dict[str, bool]:
        ok = approvals.resolve(approval_id, decision.approved, decision.scope)
        if not ok:
            raise HTTPException(404, "no such pending approval")
        return {"ok": True}

    @app.get("/v1/traces")
    async def list_turns(limit: int = 10) -> dict[str, list[str]]:
        return {"turns": traces.recent_turns(limit)}

    @app.get("/v1/traces/{turn_id}")
    async def get_trace(turn_id: str) -> dict[str, Any]:
        spans = traces.turn(turn_id)
        if not spans:
            raise HTTPException(404, "unknown turn")
        return {"turn_id": turn_id, "spans": spans}

    class IngestRequest(BaseModel):
        path: str

    class RecallRequest(BaseModel):
        q: str
        k: int = 6

    class ForgetRequest(BaseModel):
        event_id: str
        reason: str = "user request"

    @app.post("/v1/ingest")
    async def ingest(req: IngestRequest) -> dict[str, Any]:
        return ingestor.ingest_path(req.path)

    @app.post("/v1/memory/query")
    async def memory_query(req: RecallRequest) -> dict[str, Any]:
        hits = memory.recall(req.q, k=req.k)
        return {
            "hits": [
                {"event_id": h.event_id, "source": h.source, "trusted": h.trusted,
                 "ts": h.ts, "text": h.text, "rank": h.rank, "kind": h.kind}
                for h in hits
            ]
        }

    @app.post("/v1/memory/forget")
    async def memory_forget(req: ForgetRequest) -> dict[str, Any]:
        return memory.forget(req.event_id, req.reason)

    @app.post("/v1/memory/reindex")
    async def memory_reindex() -> dict[str, int]:
        return {"reprojected": memory.reindex()}

    @app.get("/v1/events")
    async def get_events(limit: int = 20) -> dict[str, Any]:
        return {"events": [e.model_dump(mode="json") for e in events.recent(limit=limit)]}

    @app.get("/v1/missions")
    async def list_missions() -> dict[str, Any]:
        """Unfinished missions, for a surface that wants to show what is still owed.

        `active()` deliberately excludes failed ones, but a mission that stopped on
        a failed step is precisely the one nobody remembers is waiting — so this
        reports both, with the status attached.
        """
        open_ = list(missions.active())
        seen = {m.id for m in open_}
        for ev in events.recent(types=[EventType.MISSION_CREATED.value], limit=1000):
            mid = ev.payload["mission_id"]
            if mid in seen:
                continue
            state = missions.state(mid)
            if state is not None and state.status == "failed":
                open_.append(state)
        return {
            "missions": [
                {
                    "id": m.id,
                    "goal": m.goal,
                    "status": m.status,
                    "steps": len(m.steps),
                    "next_step": m.next_step,
                }
                for m in open_
            ]
        }

    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        governor = _make_governor(config, events)
        spend = governor.month_to_date()
        day = governor.today()
        return {
            # Today's token consumption per provider, so a surface can warn before
            # the free-tier daily allowance runs out and answers silently get worse.
            "today": {
                "day": day.day,
                "tokens": day.total_tokens,
                "calls": day.calls,
                "providers": [
                    {
                        "provider": p.provider,
                        "calls": p.calls,
                        "tokens": p.total_tokens,
                        "cap": p.cap,
                        "call_cap": p.call_cap,
                        "metered_by": p.metered_by,
                        "fraction": round(p.fraction, 3),
                        "calls_left": p.calls_left,
                        "exhausted": p.exhausted,
                        "warning": p.warning,
                    }
                    for p in day.providers
                ],
            },
            "ok": True,
            "provider": gateway.provider,
            "fallback": gateway.fallback_provider,
            # Providers currently skipped because they said they were out of quota,
            # and when they may be tried again.
            "sidelined": gateway.sidelined,
            "local_brains": list(config.local_brains),  # brains served on-device (S2)
            "routed_capabilities": gateway.routed_capabilities,
            "build": build,
            "audit_chain_valid": policy.verify_audit_chain(),
            "spend": {
                "month": spend.month,
                "total_usd": round(spend.total_usd, 4),
                "budget_usd": spend.budget_usd,
                "over_budget": spend.over_budget,
                "warning": spend.warning,
            },
        }

    @app.get("/v1/nudges")
    async def list_nudges() -> dict[str, Any]:
        """What M.I.K.E.Y would say unprompted, and why it hasn't yet.

        Reading this does NOT deliver them: a surface decides whether anyone is
        actually there, and confirms with the POST below. Otherwise a health check
        would silently consume the thing you were meant to be told."""
        pending = nudges.pending()
        return {
            # So the surface can stop raising a kind the person keeps waving away.
            "dismissals": nudges.dismissals_by_kind(),
            "nudges": [
                {
                    "id": n.id,
                    "kind": n.kind,
                    "text": n.text,
                    "detail": n.detail,
                    "urgency": n.urgency,
                    "created": n.created.isoformat(),
                }
                for n in pending
            ]
        }

    @app.post("/v1/nudges/{nudge_id}")
    async def close_nudge(nudge_id: str, decision: NudgeDecision) -> dict[str, bool]:
        nudges.deliver(nudge_id, how=decision.how, outcome=decision.outcome)
        return {"ok": True}

    @app.get("/v1/brief")
    async def get_brief(hours: int = 24) -> dict[str, Any]:
        from datetime import timedelta

        from core.events.schema import now as _now
        from core.proactive.brief import compose

        sentinel.tick()  # look before briefing, so it reflects this moment
        at = _now()
        brief = compose(
            events,
            nudges.pending(),
            missions_open=len(missions.active()),
            since=at - timedelta(hours=max(1, hours)),
            at=at,
        )
        return {
            "since": brief.since.isoformat(),
            "quiet": brief.quiet,
            "lines": brief.lines(),
            "spoken": brief.spoken(),
            "nudges": [{"id": n.id, "kind": n.kind, "urgency": n.urgency} for n in brief.nudges],
        }

    @app.on_event("startup")
    async def start_sentinel() -> None:
        import asyncio

        async def watch_loop() -> None:
            # Looking is cheap (a few indexed reads) and costs no model call, so the
            # interval is set by how stale a warning may be, not by expense.
            while True:
                try:
                    await asyncio.to_thread(sentinel.tick)
                except Exception:  # noqa: BLE001 — a watchdog must not take the gateway down
                    pass
                await asyncio.sleep(config.proactive_interval_s)

        if config.proactive:
            app.state.sentinel_task = asyncio.create_task(watch_loop())

    @app.on_event("shutdown")
    async def shutdown() -> None:
        task = getattr(app.state, "sentinel_task", None)
        if task is not None:
            task.cancel()
        await executor.close()

    return app
