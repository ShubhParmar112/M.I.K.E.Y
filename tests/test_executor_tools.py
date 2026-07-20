from __future__ import annotations

from pathlib import Path

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


def test_web_fetch_rejects_non_http(tmp_path: Path) -> None:
    tools = Tools(tmp_path)
    r = tools.call("web_fetch", {"url": "file:///C:/Windows/system.ini"})
    assert not r.ok and "capability violation" in r.output


def test_unknown_tool_rejected(tmp_path: Path) -> None:
    tools = Tools(tmp_path)
    r = tools.call("rm_rf", {})
    assert not r.ok and "unknown tool" in r.output
