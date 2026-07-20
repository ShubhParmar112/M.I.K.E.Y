"""Core-side client for the executor sandbox process."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    ok: bool
    output: str
    tainted: bool


class ExecutorClient:
    """Spawns and talks to the sandbox over stdio JSON lines."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._next_id = 0

    async def _ensure_started(self) -> asyncio.subprocess.Process:
        if self._proc is None or self._proc.returncode is not None:
            self._proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "executor.main",
                "--workspace",
                str(self._workspace),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
        return self._proc

    async def call(self, name: str, arguments: dict[str, object]) -> ExecResult:
        async with self._lock:
            proc = await self._ensure_started()
            assert proc.stdin is not None and proc.stdout is not None
            self._next_id += 1
            req = {
                "id": self._next_id,
                "method": "call",
                "params": {"name": name, "arguments": arguments},
            }
            proc.stdin.write((json.dumps(req, ensure_ascii=False) + "\n").encode())
            await proc.stdin.drain()
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=120.0)
            if not line:
                self._proc = None
                return ExecResult(False, "executor process died", False)
            resp = json.loads(line)
            if "error" in resp:
                return ExecResult(False, f"executor error: {resp['error']}", False)
            r = resp["result"]
            return ExecResult(bool(r["ok"]), str(r["output"]), bool(r.get("tainted", False)))

    async def close(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except TimeoutError:
                self._proc.kill()
        self._proc = None
