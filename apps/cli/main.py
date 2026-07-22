"""M.I.K.E.Y CLI — the Gen 1 user surface.

`mikey chat` starts (or reuses) the local gateway and opens an interactive
session with live approval cards. `mikey trace` answers "why did you do that?".
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any

import httpx
import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from core.config import CONFIG

app = typer.Typer(help="M.I.K.E.Y — personal AI cognitive operating system (Gen 1)")
console = Console()

BASE = f"http://127.0.0.1:{CONFIG.port}"


def _server_running() -> bool:
    try:
        return httpx.get(f"{BASE}/v1/health", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


def _warn_if_stale() -> None:
    """A reused gateway may be running older code than this CLI — say so loudly."""
    from core.gateway.app import build_id

    try:
        running = httpx.get(f"{BASE}/v1/health", timeout=2.0).json().get("build", "?")
    except httpx.HTTPError:
        return
    local = build_id()
    if running != local:
        console.print(
            Panel(
                f"gateway is running build [bold]{running}[/bold] but your code is "
                f"[bold]{local}[/bold].\nQuit any open 'mikey chat' windows and rerun "
                "so the gateway restarts on current code.",
                title="STALE GATEWAY",
                border_style="red",
            )
        )


def _start_server_in_thread() -> None:
    from core.gateway.app import create_app

    server = uvicorn.Server(
        uvicorn.Config(create_app(), host="127.0.0.1", port=CONFIG.port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        if _server_running():
            return
        time.sleep(0.2)
    raise RuntimeError("gateway failed to start")


def _ensure_server() -> None:
    if not _server_running():
        console.print("[dim]starting local gateway…[/dim]")
        _start_server_in_thread()
    else:
        _warn_if_stale()


@app.command()
def serve() -> None:
    """Run the gateway in the foreground (for a separate terminal)."""
    from core.gateway.app import create_app

    uvicorn.run(create_app(), host="127.0.0.1", port=CONFIG.port)


def _served_tag(ev: dict[str, Any], primary: str) -> str:
    """Mark an event that a non-primary (local fallback) model produced."""
    served = ev.get("served_by")
    return f" [yellow](via {served})[/yellow]" if served and served != primary else ""


def _fallback_subtitle(ev: dict[str, Any], primary: str) -> str | None:
    served = ev.get("served_by")
    if served and served != primary:
        return f"[yellow]on local model ({served}) — {primary} was rate-limited/offline[/yellow]"
    return None


def _handle_approval(client: httpx.Client, ev: dict[str, Any]) -> None:
    args = json.dumps(ev.get("args", {}), ensure_ascii=False)
    console.print(
        Panel(
            f"[bold]{ev['tool']}[/bold]\n{args}\n[dim]{ev.get('reason', '')}[/dim]",
            title="approval required",
            border_style="yellow",
        )
    )
    answer = console.input("[yellow]approve? \\[y]es / \\[n]o / \\[s]ession: [/yellow]").strip().lower()
    approved = answer in ("y", "yes", "s", "session")
    scope = "session" if answer in ("s", "session") else "once"
    client.post(
        f"{BASE}/v1/approvals/{ev['approval_id']}",
        json={"approved": approved, "scope": scope},
    )


@app.command()
def chat(session: str = typer.Option("default", help="session id")) -> None:
    """Interactive chat with approval cards."""
    _ensure_server()
    health = httpx.get(f"{BASE}/v1/health", timeout=5.0).json()
    primary = health["provider"]
    fallback = health.get("fallback")
    provider_line = f"provider: [bold]{health['provider']}[/bold]"
    if fallback:
        provider_line += f" [dim](+{fallback} fallback)[/dim]"
    console.print(
        Panel(
            f"{provider_line} · "
            f"build: {health.get('build', '?')} · "
            f"audit chain: {'[green]valid[/green]' if health['audit_chain_valid'] else '[red]BROKEN[/red]'} · "
            f"workspace: {CONFIG.workspace}",
            title="M.I.K.E.Y",
        )
    )
    last_turn: str | None = None
    with httpx.Client(timeout=None) as client:
        while True:
            try:
                user_input = console.input("[bold cyan]you>[/bold cyan] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]bye[/dim]")
                return
            if not user_input:
                continue
            if user_input in ("/quit", "/exit"):
                return
            if user_input == "/trace":
                if last_turn:
                    _print_trace(last_turn)
                else:
                    console.print("[dim]no turn yet[/dim]")
                continue

            # A spinner so a slow turn (e.g. a cold local-model fallback) reads as
            # "working", not "frozen". Ctrl+C here cancels the turn — closing the
            # stream disconnects the client, which cancels the turn server-side —
            # and drops back to the prompt instead of killing the whole session.
            status = console.status("[dim]thinking…[/dim]", spinner="dots")
            try:
                status.start()
                with client.stream(
                    "POST", f"{BASE}/v1/turns", json={"session_id": session, "input": user_input}
                ) as resp:
                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        ev = json.loads(line[6:])
                        kind = ev["kind"]
                        status.stop()
                        if kind == "status":
                            last_turn = ev["turn_id"]
                        elif kind == "action":
                            args = json.dumps(ev["args"], ensure_ascii=False)[:120]
                            console.print(f"[dim]→ {ev['tool']} {args}[/dim]{_served_tag(ev, primary)}")
                        elif kind == "approval_request":
                            _handle_approval(client, ev)
                        elif kind == "action_result":
                            mark = "[green]ok[/green]" if ev["ok"] else "[red]failed[/red]"
                            console.print(f"[dim]← {ev['tool']} {mark}[/dim]")
                        elif kind == "final":
                            console.print(Panel(
                                ev["text"], border_style="cyan", title="mikey",
                                subtitle=_fallback_subtitle(ev, primary),
                            ))
                        elif kind == "error":
                            console.print(f"[red]error:[/red] {ev['message']}")
                        if kind not in ("final", "error"):
                            status.start()  # resume the spinner while the turn continues
            except KeyboardInterrupt:
                console.print(
                    "\n[dim](turn canceled — any in-flight action may still finish)[/dim]"
                )
            except httpx.HTTPError as exc:
                console.print(
                    f"[red]turn aborted:[/red] {type(exc).__name__}: {exc} — "
                    "the gateway may have restarted; try again"
                )
            finally:
                status.stop()


def _print_trace(turn_id: str) -> None:
    data = httpx.get(f"{BASE}/v1/traces/{turn_id}", timeout=5.0).json()
    tree = Tree(f"[bold]turn {turn_id}[/bold]")
    nodes: dict[str, Tree] = {}
    for s in data["spans"]:
        label = f"[bold]{s['kind']}[/bold] [dim]{s['ts']}[/dim]\n{json.dumps(s['payload'], ensure_ascii=False)[:300]}"
        parent = nodes.get(s["parent_id"] or "", tree)
        nodes[s["span_id"]] = parent.add(label)
    console.print(tree)


@app.command()
def trace(turn_id: str = typer.Argument(None)) -> None:
    """Show the trace tree for a turn (defaults to the most recent)."""
    _ensure_server()
    if turn_id is None:
        turns = httpx.get(f"{BASE}/v1/traces?limit=1", timeout=5.0).json()["turns"]
        if not turns:
            console.print("[dim]no turns recorded yet[/dim]")
            return
        turn_id = turns[0]
    _print_trace(turn_id)


@app.command()
def events(limit: int = 20) -> None:
    """Show recent events from the event log."""
    _ensure_server()
    data = httpx.get(f"{BASE}/v1/events?limit={limit}", timeout=5.0).json()
    for ev in data["events"]:
        console.print(
            f"[dim]{ev['ts']}[/dim] [bold]{ev['type']}[/bold] "
            f"{json.dumps(ev['payload'], ensure_ascii=False)[:120]}"
        )


@app.command()
def ingest(path: str) -> None:
    """Ingest a text file or directory into memory."""
    _ensure_server()
    report = httpx.post(f"{BASE}/v1/ingest", json={"path": path}, timeout=300.0).json()
    if not report.get("ok"):
        console.print(f"[red]{report.get('error', 'ingest failed')}[/red]")
        return
    console.print(
        f"[green]ingested[/green] {report['files_ingested']} files, "
        f"{report['chunks']} chunks"
        + (f" · skipped: {', '.join(report['skipped'])}" if report["skipped"] else "")
    )


@app.command()
def recall(query: str, k: int = 6) -> None:
    """Search memory; results carry source, date, and trust level."""
    _ensure_server()
    data = httpx.post(f"{BASE}/v1/memory/query", json={"q": query, "k": k}, timeout=30.0).json()
    if "hits" not in data:
        console.print(f"[red]server error:[/red] {data} — is the gateway on an old build?")
        return
    hits = data["hits"]
    if not hits:
        console.print("[dim]no memories matched[/dim]")
        return
    for h in hits:
        trust = "[green]trusted[/green]" if h["trusted"] else "[yellow]untrusted[/yellow]"
        console.print(
            Panel(
                h["text"][:500],
                title=f"{h['event_id']} · {h['ts'][:10]} · {h['source']} · {trust}",
                border_style="magenta",
            )
        )


@app.command()
def forget(event_id: str) -> None:
    """Tombstone a memory and verify it is gone from every projection."""
    _ensure_server()
    report = httpx.post(
        f"{BASE}/v1/memory/forget", json={"event_id": event_id}, timeout=30.0
    ).json()
    mark = "[green]verified forgotten[/green]" if report["verified"] else "[red]NOT VERIFIED[/red]"
    console.print(f"{report['event_id']}: {mark}")


@app.command()
def reindex() -> None:
    """Rebuild the memory index from the event log (projections are disposable)."""
    _ensure_server()
    report = httpx.post(f"{BASE}/v1/memory/reindex", timeout=300.0).json()
    console.print(f"[green]reprojected[/green] {report['reprojected']} events")


def main() -> None:
    if os.environ.get("MIKEY_SANDBOXED") == "1":
        # Running inside M.I.K.E.Y's own executor sandbox: refuse recursion.
        print(
            "mikey cannot run inside mikey's sandbox. "
            "Run this command in your own terminal (the PS> prompt, not you>)."
        )
        sys.exit(1)
    app()


if __name__ == "__main__":
    main()
