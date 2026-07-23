from __future__ import annotations

from core.policy.engine import ActionRequest, Decision, PolicyEngine
from core.storage.db import Database


def _req(tool: str, tainted: bool = False, args: dict | None = None) -> ActionRequest:
    return ActionRequest(
        tool=tool, args=args or {}, turn_id="t1", session_id="s1", tainted=tainted
    )


def test_reads_allowed_writes_ask_unknown_denied(db: Database) -> None:
    policy = PolicyEngine(db)
    assert policy.evaluate(_req("fs_read")).decision is Decision.ALLOW
    assert policy.evaluate(_req("fs_write")).decision is Decision.ASK
    assert policy.evaluate(_req("run_command")).decision is Decision.ASK
    assert policy.evaluate(_req("format_disk")).decision is Decision.DENY


def test_taint_escalates_allow_to_ask(db: Database) -> None:
    policy = PolicyEngine(db)
    assert policy.evaluate(_req("fs_read", tainted=True)).decision is Decision.ASK
    # web_fetch included: a tainted turn fetching a crafted URL is the classic
    # exfiltration channel and must go through the user
    assert policy.evaluate(_req("web_fetch", tainted=True)).decision is Decision.ASK
    assert policy.evaluate(_req("web_fetch", tainted=False)).decision is Decision.ALLOW


def test_memory_recall_is_taint_safe(db: Database) -> None:
    """A pure internal memory read can't exfiltrate, so it stays auto-allowed even
    on a tainted turn — no approval fatigue once a document is in play."""
    policy = PolicyEngine(db)
    assert policy.evaluate(_req("memory_recall", tainted=True)).decision is Decision.ALLOW
    # but the exfil channels are still gated on a tainted turn
    assert policy.evaluate(_req("web_fetch", tainted=True)).decision is Decision.ASK


def test_session_grant_converts_ask_to_allow(db: Database) -> None:
    policy = PolicyEngine(db)
    req = _req("fs_write")
    assert policy.evaluate(req).decision is Decision.ASK
    policy.grant_session(req)
    assert policy.evaluate(req).decision is Decision.ALLOW
    # grants are per-session
    other = ActionRequest(tool="fs_write", args={}, turn_id="t2", session_id="s2")
    assert policy.evaluate(other).decision is Decision.ASK


def test_run_command_grant_is_per_binary(db: Database) -> None:
    policy = PolicyEngine(db)
    git = _req("run_command", args={"command": ["git", "status"]})
    policy.grant_session(git)
    assert policy.evaluate(git).decision is Decision.ALLOW
    python = _req("run_command", args={"command": ["python", "-c", "1"]})
    assert policy.evaluate(python).decision is Decision.ASK


def test_audit_chain_verifies_and_detects_tampering(db: Database) -> None:
    policy = PolicyEngine(db)
    for _ in range(3):
        policy.evaluate(_req("fs_read"))
    assert policy.verify_audit_chain() is True
    with db.conn as conn:
        conn.execute("UPDATE audit SET decision = 'deny' WHERE seq = 2")
    assert policy.verify_audit_chain() is False
