"""The HUD: a full-screen surface that stays up between questions.

What it adds over `mikey chat` is not prettiness — it is the right-hand panel.
Every "M.I.K.E.Y got worse tonight" in this project turned out to be a fact that
WAS knowable (the daily allowance was spent, a 3B local model had taken over, the
audit chain was broken) and was printed once in a banner that had long since
scrolled away. The HUD keeps that on screen, refreshed while you work.

The conversation half deliberately runs the SAME turn driver as the CLI
(`apps.surface.stream_turn`) and the same approval card text, so this surface
cannot quietly end up with a weaker version of the safety furniture.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx
from rich.panel import Panel
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Label, RichLog, Static

from apps.surface import BASE, approval_body, get, send_approval, served_tag, stream_turn
from core.presence.hud import BROKEN, DEGRADED, Hud, build_hud

if TYPE_CHECKING:
    from core.proactive.nudge import Nudge

URGENCY_STYLE = {"high": "red", "normal": "yellow", "low": "dim"}


def headline_style(hud: Hud) -> str:
    """Four weights, not two. A banner reversed out in amber every evening stops
    being read; a calm green "86% used, ~2 calls left" is not read either. So:
    reversed for the states that change what you should do, plain amber text for
    the ones worth noticing, green only when there is truly nothing to say."""
    if hud.verdict == BROKEN:
        return "bold white on red"
    if hud.verdict == DEGRADED:
        return "bold black on yellow"
    return "green" if hud.nominal else "yellow"


class ApprovalScreen(ModalScreen[tuple[bool, str]]):
    """The approval card, as a modal that blocks the turn until answered.

    Bindings only — there is no default button and Enter does nothing. An action
    that reached this screen is one the policy engine decided a person must decide
    on, so it must cost a deliberate keystroke; escape denies, because the safe
    reading of "walked away" is no.
    """

    BINDINGS = [
        Binding("y", "decide('once')", "approve once"),
        Binding("s", "decide('session')", "approve for session"),
        Binding("n", "deny", "deny"),
        Binding("escape", "deny", "deny"),
    ]

    def __init__(self, event: dict[str, Any]) -> None:
        super().__init__()
        self._event = event

    def compose(self) -> ComposeResult:
        body, border = approval_body(self._event)
        with Vertical(id="approval-box"):
            yield Static(
                Panel(body, title="approval required", border_style=border, expand=True),
                id="approval-card",
            )
            yield Label("[y] approve once   [s] approve for this session   [n]/esc deny")

    def action_decide(self, scope: str) -> None:
        self.dismiss((True, scope))

    def action_deny(self) -> None:
        self.dismiss((False, "once"))


class HudPanel(Static):
    """The right-hand column. Rendered from the pure `Hud` — this only draws."""

    #: what was last drawn, kept so a test (or a person debugging) can read the
    #: panel back without going through the terminal.
    rendered: Text | None = None
    last: Hud | None = None

    def show(self, hud: Hud) -> None:
        t = Text()
        t.append(f" {hud.headline} \n", style=headline_style(hud))
        t.append("\nanswering  ", style="bold")
        t.append(
            hud.answering + (" (local)" if hud.answering_is_local else ""),
            style="yellow" if hud.answering_is_local else "cyan",
        )
        if not hud.answering_observed:
            # Say it is a guess. Nothing has been served yet, and a dashboard that
            # states a prediction as an observation is how you stop trusting it.
            t.append(" (expected)", style="dim")
        if hud.chain:
            t.append(f"\nchain      {' → '.join(hud.chain)}", style="dim")

        if hud.gauges:
            t.append("\n\ntoday\n", style="bold")
            for g in hud.gauges:
                t.append(
                    g.line + "\n",
                    style="red" if g.exhausted and not g.covered_by
                    else "yellow" if g.exhausted or g.warning else "dim",
                )
        if hud.sidelined:
            t.append("\nstood down\n", style="bold")
            for line in hud.sidelined:
                t.append(line + "\n", style="dim")
        if hud.budget:
            t.append(f"\nbudget     {hud.budget}", style="dim")

        t.append("\n\naudit      ", style="bold")
        t.append("valid" if hud.audit_ok else "BROKEN", style="green" if hud.audit_ok else "red")

        if hud.missions:
            t.append("\n\nmissions\n", style="bold")
            for m in hud.missions:
                t.append(m + "\n", style="dim")
        if hud.nudges:
            t.append(f"\n\n{hud.nudges} thing(s) waiting to be mentioned", style="dim")

        t.append(f"\n\nsession    {hud.session}\nbuild      {hud.build}", style="dim")
        self.rendered, self.last = t, hud
        self.update(t)


class MikeyHud(App[None]):
    """`mikey hud`."""

    CSS = """
    Screen { layers: base modal; }
    #body { height: 1fr; }
    #left { width: 3fr; }
    #right { width: 44; border-left: solid $panel-lighten-2; padding: 0 1; }
    #log { height: 1fr; }
    #hud { height: auto; }
    Input { dock: bottom; }
    #approval-box {
        layer: modal; width: 84; max-width: 90%; height: auto;
        margin: 2 4; padding: 1 2; background: $surface; border: thick $warning;
    }
    #approval-card { height: auto; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "quit"),
        Binding("ctrl+n", "new_session", "new conversation"),
        Binding("f5", "refresh_hud", "refresh"),
    ]

    def __init__(self, session: str, primary: str, refresh_s: float = 10.0) -> None:
        super().__init__()
        self._session = session
        self._primary = primary
        self._refresh_s = refresh_s
        self._served_by: str | None = None
        self._busy = False
        # How much this sitting has already been interrupted — the same restraint
        # the chat CLI applies, for the same reason: only the surface knows whether
        # anyone is actually here.
        from apps.cli.main import _Sitting

        self._sitting = _Sitting()

    # --- layout ---

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield RichLog(id="log", markup=True, wrap=True, highlight=False)
            with VerticalScroll(id="right"):
                yield HudPanel(id="hud")
        yield Input(placeholder="ask M.I.K.E.Y…  (ctrl+n new conversation, ctrl+c quit)")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "M.I.K.E.Y"
        self.sub_title = self._session
        self.query_one(Input).focus()
        self.refresh_hud()
        self.set_interval(self._refresh_s, self.refresh_hud)

    # --- the dashboard ---

    def action_refresh_hud(self) -> None:
        self.refresh_hud()

    @work(thread=True, group="hud", exclusive=True)
    def refresh_hud(self) -> None:
        """Poll the gateway off the UI thread; a slow health call must never make
        typing stutter."""
        health = get("/v1/health")
        nudges = get("/v1/nudges").get("nudges") or []
        missions = get("/v1/missions").get("missions") or []
        hud = build_hud(
            health,
            now=datetime.now().astimezone(),
            session=self._session,
            served_by=self._served_by,
            nudges=len(nudges),
            missions=missions,
        )
        self.call_from_thread(self.query_one(HudPanel).show, hud)

    # --- conversation ---

    def action_new_session(self) -> None:
        from apps.cli.main import _new_session_id

        self._session = _new_session_id()
        self.sub_title = self._session
        self._write(
            f"[green]new conversation[/green] {self._session} "
            "[dim]— previous turns are no longer in context (memory is)[/dim]"
        )
        self.refresh_hud()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self._busy:
            return
        event.input.value = ""
        if text in ("/quit", "/exit"):
            self.exit()
            return
        if text == "/new":
            self.action_new_session()
            return
        self._write(f"[bold cyan]you>[/bold cyan] {text}")
        self._busy = True
        self.run_turn(text)

    @work(thread=True, group="turn", exclusive=True)
    def run_turn(self, user_input: str) -> None:
        """One turn, on the shared driver, in a worker thread.

        Blocking httpx is used on purpose: it is the same code path the CLI and
        voice surfaces exercise, so there is one turn protocol in this codebase
        rather than one per surface.
        """
        try:
            with httpx.Client(timeout=None) as client:
                for ev in stream_turn(client, self._session, user_input):
                    self._render(client, ev)
        except httpx.HTTPError as exc:
            self._write(
                f"[red]turn aborted:[/red] {type(exc).__name__}: {exc} — "
                "the gateway may have restarted; try again"
            )
        finally:
            self._busy = False
            self.refresh_hud()
            self.call_from_thread(self._offer_nudges)

    def _render(self, client: httpx.Client, ev: dict[str, Any]) -> None:
        from apps.cli.main import _fallback_subtitle

        kind = ev["kind"]
        if kind == "status":
            if ev.get("brain") and ev["brain"] != "operator":
                self._write(f"[dim]· {ev['brain']} brain[/dim]")
            if ev.get("tier") == "T0":
                self._write("[green]· private — kept on-device[/green]")
        elif kind == "action":
            import json

            args = json.dumps(ev["args"], ensure_ascii=False)[:120]
            self._write(f"[dim]→ {ev['tool']} {args}[/dim]{served_tag(ev, self._primary)}")
        elif kind == "approval_request":
            approved, scope = self.call_from_thread(self._ask_approval, ev)
            send_approval(client, ev["approval_id"], approved, scope)
            self._write(
                f"[green]approved ({scope})[/green]" if approved else "[red]denied[/red]"
            )
        elif kind == "action_result":
            mark = "[green]ok[/green]" if ev["ok"] else "[red]failed[/red]"
            self._write(f"[dim]← {ev['tool']} {mark}[/dim]")
        elif kind == "final":
            # Who answered is HUD state, not just a subtitle: it stays on the panel
            # after this reply scrolls away.
            self._served_by = ev.get("served_by") or self._served_by
            self._write(
                Panel(
                    ev["text"],
                    border_style="cyan",
                    title="mikey",
                    subtitle=_fallback_subtitle(ev, self._primary),
                )
            )
        elif kind == "error":
            self._write(f"[red]error:[/red] {ev['message']}")

    async def _ask_approval(self, ev: dict[str, Any]) -> tuple[bool, str]:
        return await self.push_screen_wait(ApprovalScreen(ev))

    # --- proactive ---

    def _offer_nudges(self) -> None:
        """Anything noticed while that turn ran — now, between turns, never during.

        The same `choose()` the chat CLI uses, so the restraint (quiet hours, per
        session and per hour caps, muting a kind that keeps being waved away) is
        one implementation rather than two.
        """
        from core.proactive.discipline import Attention, choose
        from core.proactive.nudge import Nudge

        body = get("/v1/nudges")
        pending = body.get("nudges") or []
        if not pending:
            return
        now = datetime.now().astimezone()
        candidates: list[Nudge] = [
            Nudge(
                id=n["id"],
                kind=n.get("kind", ""),
                text=n.get("text", ""),
                detail=n.get("detail", ""),
                urgency=n.get("urgency", "normal"),
                created=datetime.fromisoformat(n["created"]),
            )
            for n in pending
        ]
        speak, _held = choose(
            candidates,
            Attention(
                at=now,
                said_this_session=self._sitting.said,
                recent_deliveries=list(self._sitting.delivered_at),
                dismissals_by_kind=body.get("dismissals", {}),
            ),
        )
        by_id = {n["id"]: n for n in pending}
        for nudge in speak:
            style = URGENCY_STYLE.get(nudge.urgency, "yellow")
            self._write(f"[{style}]· {nudge.text}[/{style}]")
            detail = by_id[nudge.id].get("detail")
            if detail:
                self._write(f"  [dim]{detail}[/dim]")
            self._sitting.said += 1
            self._sitting.delivered_at.append(now)
            self._deliver(nudge.id)

    @work(thread=True, group="nudge")
    def _deliver(self, nudge_id: str) -> None:
        """Close a nudge only after it has actually been put in front of someone."""
        try:
            httpx.post(
                f"{BASE}/v1/nudges/{nudge_id}",
                json={"outcome": "shown", "how": "hud"},
                timeout=5.0,
            )
        except httpx.HTTPError:
            pass

    # --- plumbing ---

    def _write(self, renderable: Any) -> None:
        """Write to the transcript from any thread."""
        log = self.query_one(RichLog)
        if self._in_ui_thread():
            log.write(renderable)
        else:
            self.call_from_thread(log.write, renderable)

    def _in_ui_thread(self) -> bool:
        import threading

        return threading.current_thread() is threading.main_thread()
