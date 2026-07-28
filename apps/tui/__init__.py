"""The HUD — M.I.K.E.Y's persistent surface. `mikey hud`.

Import is deliberately lazy everywhere it is used: textual is an optional extra
(`uv sync --extra tui`), and the rest of the CLI must keep working on a machine
that never installs it.
"""
