"""The gateway client every surface shares.

There are three ways to talk to M.I.K.E.Y now — text chat, voice, and the HUD —
and the failure mode to design against is a surface that quietly grows its own
turn loop and, six months later, is missing the approval card or the "this came
from the local model" warning that the other two have. So the wire protocol lives
here exactly once, and a surface's only job is to render what it yields.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from core.config import CONFIG

BASE = f"http://127.0.0.1:{CONFIG.port}"


def stream_turn(
    client: httpx.Client, session: str, user_input: str
) -> Iterator[dict[str, Any]]:
    """Yield each event of one turn as the gateway emits it.

    Closing this generator early (a surface cancelling a turn) disconnects the
    stream, which cancels the turn server-side. That is the intended way to stop
    a turn — there is no separate cancel endpoint to forget to call.
    """
    with client.stream(
        "POST", f"{BASE}/v1/turns", json={"session_id": session, "input": user_input}
    ) as resp:
        for line in resp.iter_lines():
            if line.startswith("data: "):
                yield json.loads(line[6:])


def send_approval(
    client: httpx.Client, approval_id: str, approved: bool, scope: str = "once"
) -> None:
    client.post(
        f"{BASE}/v1/approvals/{approval_id}",
        json={"approved": approved, "scope": scope},
    )


def get(path: str, *, timeout: float = 5.0, **params: Any) -> dict[str, Any]:
    """A read that never raises: a dashboard panel that cannot reach the gateway
    should go blank, not take the surface down with it."""
    try:
        return dict(httpx.get(f"{BASE}{path}", params=params, timeout=timeout).json())
    except (httpx.HTTPError, ValueError):
        return {}


# --- shared rendering -------------------------------------------------------
# These produce rich markup, which both the console and the TUI render. They live
# beside the wire protocol for the same reason it does: the approval card and the
# "this did not come from the usual model" warning are safety furniture, and a
# surface that reimplements them is a surface that will one day render a weaker
# version of them.


def served_tag(ev: dict[str, Any], primary: str) -> str:
    """Mark an event that a non-primary (local fallback) model produced."""
    served = ev.get("served_by")
    return f" [yellow](via {served})[/yellow]" if served and served != primary else ""


def preview_block(preview: dict[str, Any] | None) -> tuple[str, str]:
    """Render an action's simulated effect for an approval card, plus the border
    color the card should use. Gen 3's rule is that nothing destructive is approved
    from its arguments alone — so when a preview says data will be lost, the card
    says so loudly and shows the diff/dry-run underneath."""
    from rich.markup import escape

    if not preview:
        return "", "yellow"
    detail = str(preview.get("detail", "")).rstrip()
    # ASCII only: a cp1252 console cannot encode U+26A0 and would raise mid-render,
    # and an approval card is the last place that may fail to print.
    if preview.get("destructive"):
        head = "[bold red]DESTRUCTIVE[/bold red]"
        if not preview.get("reversible"):
            head += " [red](cannot be undone)[/red]"
        color = "red"
    else:
        head = "[green]safe — nothing is overwritten[/green]"
        color = "yellow"
    if not preview.get("simulated"):
        head += " [yellow](not simulated — effect unknown)[/yellow]"
    block = f"\n{head}\n[bold]{preview.get('summary', '')}[/bold]"
    if detail:
        # escape() so a diff line starting with "[" is not eaten as rich markup
        block += f"\n[dim]{escape(detail)}[/dim]"
    return block, color


def approval_body(ev: dict[str, Any]) -> tuple[str, str]:
    """The full text of an approval card, and its border colour."""
    args = json.dumps(ev.get("args", {}), ensure_ascii=False)
    body = f"[bold]{ev['tool']}[/bold]\n{args}\n[dim]{ev.get('reason', '')}[/dim]"
    # A second brain's read of the action, when present (S1 critic).
    note = ev.get("critic_note")
    if note:
        color = "green" if ev.get("critic_sound") else "red"
        label = "critic" if ev.get("critic_sound") else "CRITIC!"  # ASCII: see preview_block
        body += f"\n[{color}]{label}: {note}[/{color}]"
    block, border = preview_block(ev.get("preview"))
    return body + block, border
