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
from pathlib import Path
from typing import Any

import httpx
import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
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
                            # Surface when a non-default brain handled the turn, so
                            # the routing (S1) is visible — e.g. a sign-off going to
                            # the toolless conversation brain.
                            if ev.get("brain") and ev["brain"] != "operator":
                                console.print(f"[dim]· {ev['brain']} brain[/dim]")
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


@app.command()
def backup() -> None:
    """Create a verified backup snapshot of the whole store (log + audit chain)."""
    from core.backup.store import create_backup
    from core.gateway.app import build_id
    from core.storage.db import Database

    path, m = create_backup(Database(CONFIG.db_path), CONFIG.home / "backups", build_id())
    console.print(
        Panel(
            f"[green]backup created[/green]\n{path}\n"
            f"events: [bold]{m.event_count}[/bold] · audit entries: [bold]{m.audit_count}[/bold] · "
            f"chain: {'[green]valid[/green]' if m.audit_valid else '[red]BROKEN[/red]'}\n"
            f"sha256: [dim]{m.sha256[:16]}…[/dim]",
            title="M.I.K.E.Y backup",
        )
    )


@app.command()
def restore(
    backup_path: str = typer.Argument(..., help="path to a mikey-*.db backup file"),
    yes: bool = typer.Option(False, "--yes", help="skip the confirmation prompt"),
) -> None:
    """Restore the store from a backup: verifies it, snapshots current state, then
    replaces the DB and rebuilds projections from the log."""
    from core.backup.store import create_backup, restore_backup, verify_backup
    from core.gateway.app import build_id
    from core.storage.db import Database

    if _server_running():
        console.print("[red]Stop the running gateway (close any 'mikey chat') before restoring.[/red]")
        raise typer.Exit(1)

    ok, issues = verify_backup(Path(backup_path))
    if not ok:
        console.print(f"[red]backup failed verification:[/red] {'; '.join(issues)}")
        raise typer.Exit(1)

    if not yes:
        ans = console.input(
            f"[yellow]This overwrites {CONFIG.db_path}. Proceed? \\[y/N]: [/yellow]"
        ).strip().lower()
        if ans not in ("y", "yes"):
            console.print("[dim]aborted[/dim]")
            return

    if CONFIG.db_path.exists():  # safety net: snapshot current state before overwriting
        pre, _ = create_backup(Database(CONFIG.db_path), CONFIG.home / "backups", build_id())
        console.print(f"[dim]current state saved to {pre} first[/dim]")

    report = restore_backup(Path(backup_path), CONFIG.db_path)
    if report.ok:
        console.print(
            Panel(
                f"[green]restored[/green] · events: [bold]{report.event_count}[/bold] · "
                f"reprojected: [bold]{report.reprojected}[/bold] · "
                f"chain: {'[green]valid[/green]' if report.audit_valid else '[red]BROKEN[/red]'}",
                title="M.I.K.E.Y restore",
            )
        )
    else:
        console.print(f"[red]restore failed:[/red] {'; '.join(report.issues)}")
        raise typer.Exit(1)


@app.command("eval")
def run_eval_cmd(
    update_baseline: bool = typer.Option(
        False, "--update-baseline", help="save current results as the regression baseline"
    ),
) -> None:
    """Measure retrieval quality against the golden set (Gen 2 exit criterion)."""
    from core.eval.retrieval import load_golden, run_eval, save_baseline

    report = run_eval(load_golden())
    console.print(
        Panel(
            f"hit@1 [bold]{report.hit_at[1]:.0%}[/bold] · "
            f"hit@3 [bold]{report.hit_at[3]:.0%}[/bold] · "
            f"hit@6 [bold]{report.hit_at[6]:.0%}[/bold] · "
            f"MRR [bold]{report.mrr:.2f}[/bold] · "
            f"false-positive [bold]{report.false_positive_rate:.0%}[/bold]\n"
            f"{report.n_positive} positive + {report.n_negative} negative cases",
            title=f"retrieval eval — {'[green]PASS[/green]' if report.passed else '[red]FAIL[/red]'}",
            border_style="green" if report.passed else "red",
        )
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("case")
    table.add_column("result")
    table.add_column("top hit")
    for c in report.results:
        result = "[dim]neg[/dim]" if c.negative else (
            f"rank {c.first_relevant_rank}" if c.first_relevant_rank else "[red]miss[/red]"
        )
        mark = "[green]ok[/green]" if c.passed else "[red]XX[/red]"
        table.add_row(f"{mark} {c.id}", result, (c.top_source or "-").replace("connector:file:", ""))
    console.print(table)
    if report.regressions:
        console.print("[red]regressions vs baseline:[/red]")
        for r in report.regressions:
            console.print(f"  {r}")
    if update_baseline:
        save_baseline(report)
        console.print("[dim]baseline updated[/dim]")


@app.command("export")
def export_cmd(
    out: str = typer.Option(None, help="output dir (default: <MIKEY_HOME>/datasets)"),
    include_t0: bool = typer.Option(
        False, "--include-t0", help="include Tier-0 (private) turns — for on-device training only"
    ),
) -> None:
    """Export the event log into per-brain training datasets (sovereignty S0).

    Read-only over the log; respects tombstones (forgotten memories never appear)
    and privacy tiers (Tier-0 excluded unless --include-t0)."""
    from core.events.store import EventStore
    from core.storage.db import Database
    from training.exporter import TrainingExporter

    CONFIG.ensure_dirs()  # so the store opens whether or not MIKEY_HOME exists yet
    out_dir = Path(out) if out else CONFIG.home / "datasets"
    db = Database(CONFIG.db_path)
    try:
        s = TrainingExporter(EventStore(db)).export(out_dir, include_t0=include_t0)
    finally:
        db.close()
    console.print(
        Panel(
            f"[green]exported[/green] to {s.out_dir}\n"
            f"conversation: [bold]{s.conversation}[/bold] · "
            f"tool-use: [bold]{s.tool_use}[/bold] · "
            f"memory: [bold]{s.memory}[/bold]  (from {s.turns_seen} turns)\n"
            f"[dim]skipped Tier-0: {s.skipped_t0_turns} turns, {s.skipped_t0_notes} notes[/dim]",
            title="M.I.K.E.Y training export",
        )
    )


@app.command("reasoning-eval")
def reasoning_eval_cmd(
    against: str = typer.Option(
        None, help="provider to shadow-compare against, e.g. ollama (cloud vs local)"
    ),
) -> None:
    """Score tool-use reasoning on the golden set; optionally shadow-compare a
    second provider (sovereignty S0). Promotes nothing — it only measures."""
    import asyncio

    from core.eval.reasoning import load_reasoning_golden, run_reasoning_eval, shadow_compare
    from core.gateway.app import _make_adapter

    cases = load_reasoning_golden()
    primary = _make_adapter(CONFIG)

    def _cats(report: Any) -> str:
        return " · ".join(f"{c} {r:.0%}" for c, r in sorted(report.by_category.items()))

    if against:
        rep = asyncio.run(shadow_compare(primary, _make_adapter(CONFIG, against), cases))
        inc, cand = rep.incumbent, rep.candidate
        console.print(
            Panel(
                f"incumbent [bold]{inc.adapter}[/bold] {inc.pass_rate:.0%} "
                f"vs candidate [bold]{cand.adapter}[/bold] {cand.pass_rate:.0%}\n"
                f"agreement [bold]{rep.agreement:.0%}[/bold] · "
                f"regressions {rep.regressions or 'none'} · "
                f"improvements {rep.improvements or 'none'}\n"
                f"[dim]shadow only — nothing promoted[/dim]",
                title="reasoning shadow compare",
                border_style="cyan",
            )
        )
        def _mark(r: Any) -> str:
            return "[green]ok[/green]" if r and r.passed else "[red]XX[/red]"

        table = Table(show_header=True, header_style="bold")
        table.add_column("case")
        table.add_column(inc.adapter)
        table.add_column(cand.adapter)
        cand_by_id = {r.id: r for r in cand.results}
        for a in inc.results:
            b = cand_by_id.get(a.id)
            table.add_row(
                a.id,
                f"{_mark(a)} {','.join(a.tools_called) or '-'}",
                f"{_mark(b)} {','.join(b.tools_called) if b else '-'}",
            )
        console.print(table)
        return

    rep_one = asyncio.run(run_reasoning_eval(primary, cases))
    console.print(
        Panel(
            f"pass [bold]{rep_one.pass_rate:.0%}[/bold] "
            f"({sum(r.passed for r in rep_one.results)}/{rep_one.n}) · "
            f"adapter [bold]{rep_one.adapter}[/bold]\n{_cats(rep_one)}",
            title="reasoning eval",
            border_style="green" if rep_one.pass_rate >= 0.8 else "yellow",
        )
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("case")
    table.add_column("result")
    table.add_column("tools called")
    table.add_column("detail")
    for r in rep_one.results:
        mark = "[green]ok[/green]" if r.passed else "[red]XX[/red]"
        detail = r.error or r.detail
        table.add_row(f"{mark} {r.id}", r.category, ",".join(r.tools_called) or "-", detail[:60])
    console.print(table)


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
