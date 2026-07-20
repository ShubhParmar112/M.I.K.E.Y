"""Session Gateway — the only API surface. Every client (CLI today; TUI, web,
mobile later) speaks these endpoints and nothing else (architecture 02 §12).

Binds to localhost only in Gen 1.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.config import CONFIG, Config
from core.events.store import EventStore
from core.executor_client import ExecutorClient
from core.models.fake_adapter import FakeAdapter
from core.models.gateway import ModelAdapter, ModelGateway
from core.orchestrator.loop import ApprovalRegistry, Orchestrator, stream_event_json
from core.policy.engine import PolicyEngine
from core.storage.db import Database
from core.trace.store import TraceStore


def _make_adapter(config: Config) -> ModelAdapter:
    if config.provider == "anthropic":
        from core.models.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(config.anthropic_model)
    if config.provider == "groq":
        from core.models.groq_adapter import GroqAdapter

        return GroqAdapter(config.groq_model)
    if config.provider == "ollama":
        from core.models.ollama_adapter import OllamaAdapter

        return OllamaAdapter(config.ollama_base_url, config.ollama_model)
    return FakeAdapter()


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
    traces = TraceStore(db)
    policy = PolicyEngine(db)
    approvals = ApprovalRegistry()
    executor = ExecutorClient(config.workspace)
    gateway = ModelGateway(adapter or _make_adapter(config))
    orch = Orchestrator(config, events, traces, policy, gateway, executor, approvals)

    app = FastAPI(title="M.I.K.E.Y Gateway", version="0.1.0")
    app.state.policy = policy

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

    @app.get("/v1/events")
    async def get_events(limit: int = 20) -> dict[str, Any]:
        return {"events": [e.model_dump(mode="json") for e in events.recent(limit=limit)]}

    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "provider": gateway.provider,
            "audit_chain_valid": policy.verify_audit_chain(),
        }

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await executor.close()

    return app
