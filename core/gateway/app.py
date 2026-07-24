"""Session Gateway — the only API surface. Every client (CLI today; TUI, web,
mobile later) speaks these endpoints and nothing else (architecture 02 §12).

Binds to localhost only in Gen 1.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.config import CONFIG, Config
from core.events.store import EventStore
from core.executor_client import ExecutorClient
from core.ingest.files import FileIngestor
from core.memory.store import MemoryStore
from core.models.fake_adapter import FakeAdapter
from core.models.gateway import ModelAdapter, ModelGateway
from core.orchestrator.critic import Critic
from core.orchestrator.loop import ApprovalRegistry, Orchestrator, stream_event_json
from core.policy.engine import PolicyEngine
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

        return GroqAdapter(config.groq_model)
    if provider == "ollama":
        from core.models.ollama_adapter import OllamaAdapter

        return OllamaAdapter(config.ollama_base_url, config.ollama_model)
    return FakeAdapter()


def _make_fallbacks(config: Config) -> list[ModelAdapter]:
    """An ordered failover chain: a second cloud model (if its key is present and
    it isn't already primary), then the local model last for offline coverage —
    e.g. groq → claude → ollama. So a rate limit rolls to the next link, not to a
    hard error and not straight to the weak local model."""
    chain: list[ModelAdapter] = []
    if config.provider != "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        from core.models.anthropic_adapter import AnthropicAdapter

        chain.append(AnthropicAdapter(config.anthropic_model))
    if config.provider != "groq" and os.environ.get("GROQ_API_KEY"):
        from core.models.groq_adapter import GroqAdapter

        chain.append(GroqAdapter(config.groq_model))
    if config.local_fallback and config.provider in ("groq", "anthropic"):
        from core.models.ollama_adapter import OllamaAdapter

        chain.append(OllamaAdapter(config.ollama_base_url, config.fallback_ollama_model))
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

    local = OllamaAdapter(config.ollama_base_url, config.ollama_model)
    routes: dict[str, ModelAdapter] = {}
    for name in config.local_brains:
        brain = BRAINS.get(name)
        if brain is not None:
            routes[brain.capability] = local
    return routes


def _make_gateway(config: Config, adapter: ModelAdapter | None = None) -> ModelGateway:
    """The one place a real gateway is assembled — used by the server and the CLI
    planner alike, so both honor the same primary/fallback chain and S2 routes."""
    if adapter is not None:
        return ModelGateway(adapter)
    return ModelGateway(
        _make_adapter(config),
        fallbacks=_make_fallbacks(config),
        routes=_make_routes(config),
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
    gateway = _make_gateway(config, adapter)
    orch = Orchestrator(
        config, memory, traces, policy, gateway, executor, approvals, critic=Critic(gateway)
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
                 "ts": h.ts, "text": h.text, "rank": h.rank}
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

    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "provider": gateway.provider,
            "fallback": gateway.fallback_provider,
            "local_brains": list(config.local_brains),  # brains served on-device (S2)
            "routed_capabilities": gateway.routed_capabilities,
            "build": build,
            "audit_chain_valid": policy.verify_audit_chain(),
        }

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await executor.close()

    return app
