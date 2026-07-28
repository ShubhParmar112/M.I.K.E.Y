"""Presence (the JARVIS fourth): a surface that is *there* between questions.

The chat CLI tells you the state of the system once, in a banner, and then that
banner scrolls away. Every "M.I.K.E.Y got stupid tonight" in this project's
history was really "the free daily allowance ran out four turns ago and nothing
said so". The HUD exists to keep that answer on screen.

`hud.py` is pure — it decides what the dashboard says. `apps/tui` only draws it.
"""

from core.presence.hud import Gauge, Hud, bar, build_hud

__all__ = ["Gauge", "Hud", "bar", "build_hud"]
