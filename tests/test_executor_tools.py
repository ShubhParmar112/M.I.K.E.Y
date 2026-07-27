from __future__ import annotations

from pathlib import Path

from core.executor_client import ExecutorClient
from executor.tools import Tools


def test_fs_write_read_list_roundtrip(tmp_path: Path) -> None:
    tools = Tools(tmp_path)
    assert tools.call("fs_write", {"path": "notes/a.txt", "content": "hello"}).ok
    r = tools.call("fs_read", {"path": "notes/a.txt"})
    assert r.ok and r.output == "hello"
    listing = tools.call("fs_list", {"path": "notes"})
    assert listing.ok and "f a.txt" in listing.output


def test_path_escape_is_blocked(tmp_path: Path) -> None:
    tools = Tools(tmp_path / "ws")
    for evil in ("..\\outside.txt", "../outside.txt", "C:\\Windows\\evil.txt"):
        r = tools.call("fs_write", {"path": evil, "content": "x"})
        assert not r.ok and "capability violation" in r.output
    r = tools.call("fs_read", {"path": "..\\..\\secrets.txt"})
    assert not r.ok and "capability violation" in r.output


def test_command_allowlist_enforced(tmp_path: Path) -> None:
    tools = Tools(tmp_path)
    r = tools.call("run_command", {"command": ["powershell", "-c", "whoami"]})
    assert not r.ok and "not in allowlist" in r.output
    r = tools.call("run_command", {"command": ["cmd.exe", "/c", "del"]})
    assert not r.ok and "not in allowlist" in r.output
    r = tools.call("run_command", {"command": ["python", "-c", "print('ok')"]})
    assert r.ok and "ok" in r.output


def test_drive_letter_paths_denied_everywhere(tmp_path: Path) -> None:
    """C:\\-style paths must be escape attempts on every OS — on Linux they are
    otherwise legal filenames, which is how this suite broke CI on ubuntu."""
    tools = Tools(tmp_path / "ws")
    r = tools.call("fs_read", {"path": "D:\\data\\secrets.txt"})
    assert not r.ok and "capability violation" in r.output


def test_command_timeout_kills_process_tree(tmp_path: Path, monkeypatch) -> None:
    import executor.tools as et

    monkeypatch.setattr(et, "COMMAND_TIMEOUT_S", 1)
    tools = Tools(tmp_path)
    r = tools.call("run_command", {"command": ["python", "-c", "import time; time.sleep(60)"]})
    assert not r.ok and "timed out" in r.output and "killed" in r.output


def test_run_command_children_are_marked_sandboxed(tmp_path: Path) -> None:
    tools = Tools(tmp_path)
    r = tools.call(
        "run_command",
        {"command": ["python", "-c", "import os; print(os.environ.get('MIKEY_SANDBOXED'))"]},
    )
    assert r.ok and r.output.strip() == "1"


def test_web_fetch_rejects_non_http(tmp_path: Path) -> None:
    tools = Tools(tmp_path)
    r = tools.call("web_fetch", {"url": "file:///C:/Windows/system.ini"})
    assert not r.ok and "capability violation" in r.output


def test_unknown_tool_rejected(tmp_path: Path) -> None:
    tools = Tools(tmp_path)
    r = tools.call("rm_rf", {})
    assert not r.ok and "unknown tool" in r.output


async def test_a_large_result_crosses_the_sandbox_boundary_intact(tmp_path: Path) -> None:
    """asyncio's default stream buffer is 64 KiB, so every reply above that used to
    fail with "Separator is not found" — and leave the rest of the line in the pipe
    for the next call to read as its own response. The executor deliberately allows
    a 1 MB read and a 100 KB fetch, so most of that range was unreachable.

    The second call is the real assertion: it proves the stream is still in step.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    big = "START\n" + ("payload line\n" * 30_000) + "END"
    assert len(big) > 64 * 1024
    (ws / "big.txt").write_text(big, encoding="utf-8")
    (ws / "small.txt").write_text("small", encoding="utf-8")

    client = ExecutorClient(ws)
    try:
        first = await client.call("fs_read", {"path": "big.txt"})
        second = await client.call("fs_read", {"path": "small.txt"})
    finally:
        await client.close()

    assert first.ok and first.output == big
    assert second.ok and second.output == "small", "the next response must not be a leftover"


async def test_text_survives_the_sandbox_boundary_in_both_directions(tmp_path: Path) -> None:
    """The stdio protocol is UTF-8; a child process on Windows assumes the system
    codepage unless told. Until it was told, every file M.I.K.E.Y wrote containing an
    em-dash, a curly quote or any non-Latin script was corrupted on the way in — and
    reading one back handed the model mojibake to reason about.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    text = "an em-dash — a curly ‘quote’ · देवनागरी · 日本語 · émoji 🚀"

    client = ExecutorClient(ws)
    try:
        wrote = await client.call("fs_write", {"path": "unicode.txt", "content": text})
        read_back = await client.call("fs_read", {"path": "unicode.txt"})
    finally:
        await client.close()

    assert wrote.ok
    assert (ws / "unicode.txt").read_text(encoding="utf-8") == text  # in
    assert read_back.ok and read_back.output == text  # and back out
