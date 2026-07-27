"""Core-side client for the executor sandbox process."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path


# One response is one JSON line, and asyncio's default StreamReader buffer is
# 64 KiB — so every reply bigger than that failed with "Separator is not found,
# and chunk exceed the limit" and, worse, left the rest of the line in the pipe
# for the NEXT call to read as its own response. The executor deliberately allows
# a 1 MB file read and a 100 KB fetch, so most of that range was unreachable:
# reading a large file or fetching a real web page returned a cryptic executor
# failure. The buffer must comfortably exceed the executor's own limits, with
# room for JSON escaping expanding the payload.
EXECUTOR_STREAM_LIMIT = 8 * 1024 * 1024


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
                limit=EXECUTOR_STREAM_LIMIT,
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
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=120.0)
            except TimeoutError:
                # The sandbox is wedged (e.g. a child holding pipes). A stuck
                # executor can never be reused — responses would desync. Kill it;
                # the next call spawns a fresh one.
                await self._kill()
                return ExecResult(
                    False, "executor timed out and was restarted; the action did not complete",
                    False,
                )
            except ValueError:
                # A response longer than the buffer above. readline() gives up with
                # the remainder of the line still in the pipe, which the next call
                # would read as ITS response — so this connection is unusable and
                # the executor is replaced rather than reused. Same rule as the
                # timeout: never continue on a stream that may be out of step.
                await self._kill()
                return ExecResult(
                    False,
                    "the result was too large to return from the sandbox; ask for a "
                    "smaller slice (a specific file, or a filtered command)",
                    False,
                )
            if not line:
                self._proc = None
                return ExecResult(False, "executor process died", False)
            resp = json.loads(line)
            if "error" in resp:
                return ExecResult(False, f"executor error: {resp['error']}", False)
            r = resp["result"]
            return ExecResult(bool(r["ok"]), str(r["output"]), bool(r.get("tainted", False)))

    async def _kill(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.kill()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except TimeoutError:
                pass
        self._proc = None

    async def close(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except TimeoutError:
                self._proc.kill()
        self._proc = None
