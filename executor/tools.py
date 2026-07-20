"""Executor tools with capability enforcement.

This module does not trust its caller (architecture 02 §7): every path is
confined to the workspace root, every command checked against the allowlist —
here, inside the sandbox process, regardless of what the core asked for.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

COMMAND_ALLOWLIST = {"git", "python", "py", "uv", "pip", "where", "whoami"}
MAX_FETCH_BYTES = 100_000
MAX_FILE_BYTES = 1_000_000
COMMAND_TIMEOUT_S = 60


@dataclass
class ToolResult:
    ok: bool
    output: str
    tainted: bool = False  # content originates from an untrusted source


class CapabilityError(Exception):
    pass


class Tools:
    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()
        self._workspace.mkdir(parents=True, exist_ok=True)

    # ---- confinement ----

    def _confine(self, raw: str) -> Path:
        candidate = (self._workspace / raw).resolve() if not Path(raw).is_absolute() else Path(
            raw
        ).resolve()
        if candidate != self._workspace and self._workspace not in candidate.parents:
            raise CapabilityError(f"path escapes workspace: {raw}")
        return candidate

    # ---- tools ----

    def fs_read(self, path: str) -> ToolResult:
        p = self._confine(path)
        if not p.is_file():
            return ToolResult(False, f"not a file: {path}")
        if p.stat().st_size > MAX_FILE_BYTES:
            return ToolResult(False, f"file exceeds {MAX_FILE_BYTES} bytes: {path}")
        return ToolResult(True, p.read_text(encoding="utf-8", errors="replace"))

    def fs_write(self, path: str, content: str) -> ToolResult:
        p = self._confine(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return ToolResult(True, f"wrote {len(content)} chars to {p.relative_to(self._workspace)}")

    def fs_list(self, path: str = ".") -> ToolResult:
        p = self._confine(path)
        if not p.is_dir():
            return ToolResult(False, f"not a directory: {path}")
        entries = sorted(
            f"{'d' if e.is_dir() else 'f'} {e.name}" for e in p.iterdir()
        )
        return ToolResult(True, "\n".join(entries) or "(empty)")

    def run_command(self, command: list[str]) -> ToolResult:
        if not command:
            raise CapabilityError("empty command")
        binary = Path(command[0]).name.lower().removesuffix(".exe")
        if binary not in COMMAND_ALLOWLIST:
            raise CapabilityError(f"binary '{binary}' not in allowlist {sorted(COMMAND_ALLOWLIST)}")
        try:
            proc = subprocess.run(
                command,
                cwd=self._workspace,
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT_S,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"command timed out after {COMMAND_TIMEOUT_S}s")
        except FileNotFoundError:
            return ToolResult(False, f"binary not found: {command[0]}")
        out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        return ToolResult(proc.returncode == 0, out.strip() or f"(exit {proc.returncode})")

    def web_fetch(self, url: str) -> ToolResult:
        if not url.lower().startswith(("http://", "https://")):
            raise CapabilityError("only http(s) URLs allowed")
        try:
            resp = httpx.get(url, follow_redirects=True, timeout=30.0)
        except httpx.HTTPError as exc:
            return ToolResult(False, f"fetch failed: {exc}", tainted=True)
        text = resp.text[:MAX_FETCH_BYTES]
        return ToolResult(True, text, tainted=True)

    def call(self, name: str, arguments: dict[str, object]) -> ToolResult:
        try:
            match name:
                case "fs_read":
                    return self.fs_read(str(arguments["path"]))
                case "fs_write":
                    return self.fs_write(str(arguments["path"]), str(arguments["content"]))
                case "fs_list":
                    return self.fs_list(str(arguments.get("path", ".")))
                case "run_command":
                    cmd = arguments.get("command")
                    if not isinstance(cmd, list):
                        raise CapabilityError("command must be an argv array")
                    return self.run_command([str(c) for c in cmd])
                case "web_fetch":
                    return self.web_fetch(str(arguments["url"]))
                case _:
                    return ToolResult(False, f"unknown tool: {name}")
        except CapabilityError as exc:
            return ToolResult(False, f"capability violation: {exc}")
        except KeyError as exc:
            return ToolResult(False, f"missing argument: {exc}")
