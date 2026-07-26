"""Simulate-first previews for destructive actions (Gen 3).

Gen 3's exit criterion is *zero destructive actions without preview*: before a
change that cannot be undone reaches the approval card, the user sees what it
will actually do — the diff a write would apply, the files a `git clean` would
delete, the text of the memory a forget would drop. An approval card that says
only `fs_write {"path": "notes.md", ...}` is a consent ritual, not consent.

Two pieces, deliberately separate:

* `classify()` is pure and synchronous. It answers "could this destroy data?"
  **conservatively** — `fs_write` counts, because a write to an existing path is
  an overwrite and nothing static can tell which it is. Its job is to force a
  preview, not to be the final word.
* `Previewer` runs the simulation, which needs I/O, and returns the *resolved*
  truth: a write that turns out to create a new file is not destructive.

Nothing here mutates anything. Every file read goes through the executor's own
path confinement, and every command run is a no-op form (`--dry-run`,
`git status`) of the command being previewed — never the command itself.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Protocol

MAX_DIFF_LINES = 80
MAX_DETAIL_CHARS = 4000

# Tools whose effect is worth showing before it happens. Everything else on the
# tool surface is a read (fs_read, fs_list, web_fetch, memory_recall) or an
# append to M.I.K.E.Y's own state (memory_remember, ingest) — nothing to lose.
PREVIEWABLE = frozenset({"fs_write", "run_command", "memory_forget"})


@dataclass(frozen=True)
class Impact:
    """Static, conservative read of an action. `destructive` here means "treat as
    destructive until a preview proves otherwise", which is why `fs_write` is
    always True: only reading the target says whether it is a create or a clobber."""

    destructive: bool
    reversible: bool
    why: str


@dataclass(frozen=True)
class Preview:
    """The resolved simulation, attached to the approval card and the trace."""

    tool: str
    summary: str  # one line: what changes
    detail: str  # the diff / dry-run output / the memory text
    destructive: bool
    reversible: bool
    simulated: bool  # True = detail came from actually simulating, not guessing

    def as_card(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "detail": self.detail,
            "destructive": self.destructive,
            "reversible": self.reversible,
            "simulated": self.simulated,
        }


# ---- static classification -------------------------------------------------


def _binary(argv: list[str]) -> str:
    return PurePath(argv[0]).name.lower().removesuffix(".exe")


def _subcommand(argv: list[str]) -> str:
    return next((a for a in argv[1:] if not a.startswith("-")), "")


def _git_impact(argv: list[str]) -> Impact | None:
    sub = _subcommand(argv)
    flags = {a for a in argv if a.startswith("-")}
    dry_already = bool(flags & {"-n", "--dry-run"})

    if sub == "clean":
        if dry_already:
            return None  # already a simulation
        return Impact(True, False, "deletes untracked files from the working tree")
    if sub == "rm":
        if dry_already:
            return None
        return Impact(True, False, "removes files from the working tree and the index")
    if sub == "reset" and "--hard" in flags:
        return Impact(True, False, "discards every uncommitted change in the working tree")
    if sub == "checkout" and ("--" in argv or bool(flags & {"-f", "--force"})):
        return Impact(True, False, "overwrites local changes in the named paths")
    if sub == "restore":
        return Impact(True, False, "overwrites local changes in the named paths")
    if sub == "branch" and "-D" in flags:
        return Impact(True, False, "force-deletes a branch — unmerged commits become unreachable")
    if sub == "push" and bool(flags & {"-f", "--force", "--force-with-lease"}):
        return Impact(True, False, "force-pushes: rewrites history on the remote")
    if sub == "stash" and ("drop" in argv or "clear" in argv):
        return Impact(True, False, "discards stashed changes")
    if sub in ("filter-branch", "gc", "prune"):
        return Impact(True, False, f"`git {sub}` rewrites or prunes repository history")
    return None


def _package_impact(binary: str, argv: list[str]) -> Impact | None:
    words = [a for a in argv[1:] if not a.startswith("-")]
    if "uninstall" in words or (binary == "uv" and "remove" in words):
        # Recoverable — a package can be reinstalled — but still a removal the
        # user should see coming, because it can break the environment now.
        return Impact(True, True, "removes installed packages")
    return None


# Binaries that execute arbitrary code. Not classified destructive (that would
# put every `python -m pytest` behind a diff nobody can produce), but the preview
# says plainly that their effect cannot be simulated.
OPAQUE_BINARIES = frozenset({"python", "py", "uv", "pip"})


def classify(tool: str, args: dict[str, Any]) -> Impact:
    if tool == "fs_write":
        return Impact(True, False, "overwrites the file if it already exists")
    if tool == "memory_forget":
        return Impact(True, False, "tombstones a memory — gone from every projection")
    if tool == "run_command":
        raw = args.get("command") or []
        argv = [str(c) for c in raw] if isinstance(raw, list) else []
        if not argv:
            return Impact(False, True, "empty command")
        binary = _binary(argv)
        impact = _git_impact(argv) if binary == "git" else _package_impact(binary, argv)
        if impact is not None:
            return impact
        if binary in OPAQUE_BINARIES:
            return Impact(False, True, "runs code — its effect cannot be simulated in advance")
        return Impact(False, True, "no known destructive effect")
    return Impact(False, True, "read-only, or writes only to M.I.K.E.Y's own state")


# ---- simulation ------------------------------------------------------------


def _dry_run_form(argv: list[str]) -> tuple[str, list[str]] | None:
    """A genuinely side-effect-free command that reveals what `argv` would do.

    Either the command's own `--dry-run`, or a read-only command that shows what
    stands to be lost. Returns None when no honest simulation exists — in which
    case the card says so rather than implying one was run."""
    if _binary(argv) != "git":
        return None
    git, sub = argv[0], _subcommand(argv)
    if sub in ("clean", "rm"):
        return "would remove:", [*argv, "--dry-run"]
    if sub == "reset":
        return "uncommitted work that would be discarded:", [git, "status", "--porcelain"]
    if sub in ("checkout", "restore"):
        return "local changes that would be overwritten:", [git, "diff", "--stat"]
    if sub == "stash":
        return "stashes that would be dropped:", [git, "stash", "list"]
    return None


def _clip(text: str, limit: int = MAX_DETAIL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… ({len(text) - limit} more characters)"


def _clip_lines(lines: list[str], limit: int = MAX_DIFF_LINES) -> str:
    if len(lines) <= limit:
        return "\n".join(lines)
    return "\n".join(lines[:limit]) + f"\n… ({len(lines) - limit} more lines)"


class _Result(Protocol):
    ok: bool
    output: str


class _Executor(Protocol):
    async def call(self, name: str, arguments: dict[str, Any]) -> _Result: ...


class _NoteSource(Protocol):
    def note(self, event_id: str) -> Any: ...


class Previewer:
    """Builds the preview for an action, without performing it.

    `executor` is the same sandboxed client the real action would use, so a
    preview cannot read outside the workspace either; `memory` supplies the text
    behind a `memory_forget` id."""

    def __init__(self, executor: _Executor, memory: _NoteSource | None = None) -> None:
        self._executor = executor
        self._memory = memory

    async def preview(self, tool: str, args: dict[str, Any]) -> Preview | None:
        """None means "nothing to preview" (a read). Never raises: a preview that
        fails must still produce a card saying so — silently skipping it is the
        one outcome the exit criterion forbids."""
        if tool not in PREVIEWABLE:
            return None
        try:
            if tool == "fs_write":
                return await self._preview_write(args)
            if tool == "run_command":
                return await self._preview_command(args)
            return self._preview_forget(args)
        except Exception as exc:  # noqa: BLE001 — a broken preview must not eat the turn
            impact = classify(tool, args)
            return Preview(
                tool=tool,
                summary=f"could not simulate this action: {type(exc).__name__}: {exc}",
                detail="Approve only if you are certain what it will do.",
                destructive=impact.destructive,
                reversible=impact.reversible,
                simulated=False,
            )

    async def _preview_write(self, args: dict[str, Any]) -> Preview:
        path = str(args.get("path", ""))
        new = str(args.get("content", ""))
        current = await self._executor.call("fs_read", {"path": path})
        if not current.ok:
            # Nothing there to lose. (If the read failed for another reason —
            # confinement, size cap — the write hits the same wall, so still
            # nothing is destroyed.)
            body = _clip_lines(new.splitlines())
            return Preview(
                tool="fs_write",
                summary=f"creates {path} — {len(new)} chars, {len(new.splitlines())} lines",
                detail=body,
                destructive=False,
                reversible=True,
                simulated=True,
            )
        old = current.output
        if old == new:
            return Preview(
                tool="fs_write",
                summary=f"{path} already contains exactly this — the write changes nothing",
                detail="",
                destructive=False,
                reversible=True,
                simulated=True,
            )
        diff = list(
            difflib.unified_diff(
                old.splitlines(),
                new.splitlines(),
                fromfile=f"{path} (now)",
                tofile=f"{path} (after)",
                lineterm="",
            )
        )
        added = sum(1 for ln in diff if ln.startswith("+") and not ln.startswith("+++"))
        removed = sum(1 for ln in diff if ln.startswith("-") and not ln.startswith("---"))
        return Preview(
            tool="fs_write",
            summary=f"overwrites {path}: +{added} / -{removed} lines "
            f"({len(old)} existing chars replaced)",
            detail=_clip_lines(diff),
            destructive=True,
            reversible=False,
            simulated=True,
        )

    async def _preview_command(self, args: dict[str, Any]) -> Preview:
        raw = args.get("command") or []
        argv = [str(c) for c in raw] if isinstance(raw, list) else []
        impact = classify("run_command", {"command": argv})
        shown = " ".join(argv)
        if not impact.destructive:
            return Preview(
                tool="run_command",
                summary=f"runs `{shown}` in the workspace",
                detail=impact.why,
                destructive=False,
                reversible=True,
                simulated=False,
            )
        dry = _dry_run_form(argv)
        if dry is None:
            return Preview(
                tool="run_command",
                summary=f"`{shown}` — {impact.why}",
                detail="This command has no no-op form, so its effect cannot be simulated. "
                "Read it carefully before approving.",
                destructive=True,
                reversible=impact.reversible,
                simulated=False,
            )
        label, dry_argv = dry
        result = await self._executor.call("run_command", {"command": dry_argv})
        body = result.output.strip() or "(nothing)"
        return Preview(
            tool="run_command",
            summary=f"`{shown}` — {impact.why}",
            detail=f"{label}\n$ {' '.join(dry_argv)}\n{_clip(body)}",
            destructive=True,
            reversible=impact.reversible,
            simulated=True,
        )

    def _preview_forget(self, args: dict[str, Any]) -> Preview:
        event_id = str(args.get("event_id", "")).strip()
        note = self._memory.note(event_id) if self._memory is not None else None
        if note is None:
            return Preview(
                tool="memory_forget",
                summary=f"no live memory has id {event_id} — nothing would be deleted",
                detail="",
                destructive=False,
                reversible=True,
                simulated=True,
            )
        trust = "trusted" if note.trusted else "untrusted"
        return Preview(
            tool="memory_forget",
            summary=f"permanently drops memory {event_id} from every projection",
            detail=f"[{note.source} · {trust} · {note.ts[:10]}]\n{_clip(note.text)}",
            destructive=True,
            reversible=False,
            simulated=True,
        )
