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
    # True when the ALLOW came from a standing session grant rather than the rule
    # table. The caller re-checks these against the action's simulated preview: a
    # grant for "write files this session" must not silently cover an overwrite
    # that destroys existing content (Gen 3: no destructive action without preview).
    via_session_grant: bool = False


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
    # Reach (Gen 3 automation). `git` splits by action below — reading a repository
    # is nothing like pushing one. GitHub is read-only and returns other people's
    # text, so it allows but taints. Opening a file or a URL launches a program and
    # is visible on screen, so it asks.
    "git": Decision.ASK,  # the floor; ACTION_RULES relaxes the read actions
    "github": Decision.ALLOW,
    "open": Decision.ASK,
}

# Tools whose verdict depends on WHICH operation is being asked for. Without this,
# a single decision would have to cover both `git status` and `git push` — and
# whichever way that went would be wrong: asking for every status read trains
# people to approve without reading, and allowing pushes does not bear thinking
# about. An action not named here is DENIED, so the allowlist stays the boundary.
ACTION_RULES: dict[str, dict[str, Decision]] = {
    "git": {
        "status": Decision.ALLOW,
        "log": Decision.ALLOW,
        "diff": Decision.ALLOW,
        "branches": Decision.ALLOW,
        "unpushed": Decision.ALLOW,
        "commit": Decision.ASK,
        "push": Decision.ASK,
    },
}

# Auto-allowed tools that stay allowed even on a tainted turn: they only READ
# M.I.K.E.Y's own memory and cannot exfiltrate or cause an external side effect,
# so escalating them to an approval card is pure friction (the exfil channels —
# web_fetch, run_command, fs_write — remain gated regardless).
TAINT_SAFE_TOOLS = {"memory_recall"}


def base_decision(tool: str, args: dict[str, Any]) -> Decision | None:
    """The rule-table verdict for an action, before taint and grants are applied.
    None means "no rule" — which the caller turns into a denial."""
    per_action = ACTION_RULES.get(tool)
    if per_action is not None:
        return per_action.get(str(args.get("action", "")).strip().lower())
    return RULES.get(tool)


class PolicyEngine:
    def __init__(self, db: Database) -> None:
        self._db = db
        # session_id -> set of action signatures granted for the session
        self._session_grants: dict[str, set[str]] = {}

    def evaluate(self, req: ActionRequest) -> PolicyResult:
        base = base_decision(req.tool, req.args)
        if base is None:
            what = (
                f"'{req.tool}' has no rule for action '{req.args.get('action')}'"
                if req.tool in ACTION_RULES
                else f"tool '{req.tool}' is not in the policy table"
            )
            result = PolicyResult(Decision.DENY, what)
        elif req.tainted and base is Decision.ALLOW and req.tool not in TAINT_SAFE_TOOLS:
            # Untrusted content may inform but never authorize (review W4). This
            # includes web_fetch: a tainted turn fetching a crafted URL is the
            # classic exfiltration channel, so it must go through the user.
            result = PolicyResult(Decision.ASK, "input derived from untrusted content")
        elif base is Decision.ASK and self._signature(req) in self._session_grants.get(
            req.session_id, set()
        ):
            result = PolicyResult(Decision.ALLOW, "standing session grant", via_session_grant=True)
        else:
            result = PolicyResult(base, f"rule for '{req.tool}'")
        self._audit("policy", req, result.decision.value, result.reason)
        return result

    def grant_session(self, req: ActionRequest) -> None:
        self._session_grants.setdefault(req.session_id, set()).add(self._signature(req))
        self._audit("user", req, "session_grant", "user granted for session")

    def record_preview_escalation(self, req: ActionRequest, why: str) -> None:
        """A session grant was withdrawn because the action's preview showed it
        destroys data. Audited, so the escalation is as reconstructable as the
        grant that it overrode."""
        self._audit("policy", req, "ask", f"session grant does not cover a destructive action: {why}")

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
        if req.tool in ACTION_RULES:
            # A grant has to name the action too. "Approve git for this session"
            # after reading a status must not quietly cover a push.
            return f"{req.tool}:{str(req.args.get('action', '')).strip().lower()}"
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
