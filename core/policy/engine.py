"""Policy & Consent Engine (review M4).

Every side effect passes through `evaluate()`. Decisions are data (rules),
every evaluation is written to the hash-chained audit log, and untrusted
(tainted) input can never escalate an action to auto-allow.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from core.events.schema import now
from core.storage.db import Database


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class ActionRequest:
    tool: str
    args: dict[str, Any]
    turn_id: str
    session_id: str
    tainted: bool = False  # derived from untrusted content (web, ingested docs)


@dataclass(frozen=True)
class PolicyResult:
    decision: Decision
    reason: str


# Gen 1 rule table: action class -> decision. Reads observe, writes ask.
# Unknown tools are denied — the allowlist is the boundary, not the model's judgment.
RULES: dict[str, Decision] = {
    "fs_read": Decision.ALLOW,
    "fs_list": Decision.ALLOW,
    "fs_write": Decision.ASK,
    "run_command": Decision.ASK,
    "web_fetch": Decision.ALLOW,
    "memory_recall": Decision.ALLOW,
    # Remembering writes only to M.I.K.E.Y's own state (no outside effect), so it
    # auto-allows on a clean turn — but a tainted turn escalates it to ASK below,
    # closing the memory-poisoning channel where untrusted content plants a "fact".
    "memory_remember": Decision.ALLOW,
    # Forgetting is destructive and irreversible: always confirm with the user.
    "memory_forget": Decision.ASK,
    # Ingest reads a user-named file from anywhere on disk (outside the sandbox)
    # into memory — confirm the path with the user first.
    "ingest": Decision.ASK,
}

# Auto-allowed tools that stay allowed even on a tainted turn: they only READ
# M.I.K.E.Y's own memory and cannot exfiltrate or cause an external side effect,
# so escalating them to an approval card is pure friction (the exfil channels —
# web_fetch, run_command, fs_write — remain gated regardless).
TAINT_SAFE_TOOLS = {"memory_recall"}


class PolicyEngine:
    def __init__(self, db: Database) -> None:
        self._db = db
        # session_id -> set of action signatures granted for the session
        self._session_grants: dict[str, set[str]] = {}

    def evaluate(self, req: ActionRequest) -> PolicyResult:
        base = RULES.get(req.tool)
        if base is None:
            result = PolicyResult(Decision.DENY, f"tool '{req.tool}' is not in the policy table")
        elif req.tainted and base is Decision.ALLOW and req.tool not in TAINT_SAFE_TOOLS:
            # Untrusted content may inform but never authorize (review W4). This
            # includes web_fetch: a tainted turn fetching a crafted URL is the
            # classic exfiltration channel, so it must go through the user.
            result = PolicyResult(Decision.ASK, "input derived from untrusted content")
        elif base is Decision.ASK and self._signature(req) in self._session_grants.get(
            req.session_id, set()
        ):
            result = PolicyResult(Decision.ALLOW, "standing session grant")
        else:
            result = PolicyResult(base, f"rule for '{req.tool}'")
        self._audit("policy", req, result.decision.value, result.reason)
        return result

    def grant_session(self, req: ActionRequest) -> None:
        self._session_grants.setdefault(req.session_id, set()).add(self._signature(req))
        self._audit("user", req, "session_grant", "user granted for session")

    def record_auto_denial(self, req: ActionRequest) -> None:
        self._audit("policy", req, "deny", "auto-denied: repeat of user-denied action")

    def record_user_decision(self, req: ActionRequest, approved: bool) -> None:
        self._audit("user", req, "approved" if approved else "denied", "explicit user decision")

    @staticmethod
    def _signature(req: ActionRequest) -> str:
        """Grant key: tool name only for run_command would be too broad — include
        the command binary; for fs_write, the workspace-relative directory."""
        if req.tool == "run_command":
            argv = req.args.get("command") or []
            return f"run_command:{argv[0] if argv else '?'}"
        return req.tool

    # ---- hash-chained audit (review §5) ----

    def _audit(self, actor: str, req: ActionRequest, decision: str, reason: str) -> None:
        payload = json.dumps(
            {"tool": req.tool, "args": req.args, "turn_id": req.turn_id, "tainted": req.tainted},
            ensure_ascii=False,
            sort_keys=True,
        )
        with self._db.conn as conn:
            row = conn.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            prev_hash = row["hash"] if row else "GENESIS"
            ts = now().isoformat()
            digest = hashlib.sha256(
                "|".join([prev_hash, ts, actor, req.tool, decision, reason, payload]).encode()
            ).hexdigest()
            conn.execute(
                "INSERT INTO audit (ts, actor, action, decision, reason, payload, prev_hash, hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, actor, req.tool, decision, reason, payload, prev_hash, digest),
            )

    def verify_audit_chain(self) -> bool:
        return audit_chain_valid(self._db.conn)


def audit_chain_valid(conn: Any) -> bool:
    """Recompute the hash chain over the audit table (any sqlite connection with a
    Row factory). Shared by the policy engine and the backup verifier so the
    integrity check has exactly one implementation."""
    prev = "GENESIS"
    for row in conn.execute("SELECT * FROM audit ORDER BY seq"):
        expected = hashlib.sha256(
            "|".join(
                [
                    prev,
                    row["ts"],
                    row["actor"],
                    row["action"],
                    row["decision"],
                    row["reason"],
                    row["payload"],
                ]
            ).encode()
        ).hexdigest()
        if row["hash"] != expected or row["prev_hash"] != prev:
            return False
        prev = row["hash"]
    return True
