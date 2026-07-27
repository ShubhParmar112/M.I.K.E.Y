"""Executor sandbox entry point — a separate OS process, JSON-RPC over stdio
(MCP-style; formal MCP server arrives with the Gen 3 rewrite, same seam).

Run: python -m executor.main --workspace <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from executor.tools import Tools


def serve(workspace: Path) -> None:
    tools = Tools(workspace)
    # The protocol is UTF-8 in both directions, and must be said out loud: the core
    # writes UTF-8 bytes down the pipe, but a child process on Windows decodes stdio
    # in the system codepage (cp1252) unless told otherwise. Without this, every
    # file M.I.K.E.Y wrote containing an em-dash, a curly quote, an accent or any
    # non-Latin script was silently corrupted on its way INTO the sandbox, and
    # reading such a file back returned mojibake to the model. `replace` rather than
    # strict, because malformed input must not kill the sandbox loop.
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            result = tools.call(req["params"]["name"], req["params"].get("arguments", {}))
            resp = {
                "id": req.get("id"),
                "result": {"ok": result.ok, "output": result.output, "tainted": result.tainted},
            }
        except Exception as exc:  # never crash the sandbox loop on a bad request
            resp = {"id": None, "error": str(exc)}
        try:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except OSError:
            return  # parent is gone; exit quietly instead of a zombie traceback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    serve(Path(args.workspace))


if __name__ == "__main__":
    main()
