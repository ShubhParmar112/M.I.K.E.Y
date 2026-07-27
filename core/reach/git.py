"""Git, on a real repository rather than the sandbox.

Every command is an argv list run with `cwd` set to a resolved project directory
— never a shell string, so nothing here can be turned into a command by a
carefully worded commit message or filename.

The write surface is deliberately narrow and deliberately boring:

* **Commit stages nothing by surprise.** It commits what is already staged, or
  exactly the paths it is given. `git commit -a` is not available: a model that
  can sweep every modified file into a commit will eventually sweep in the file
  you were halfway through.
* **Push never forces.** `--force` and `--force-with-lease` are not reachable
  through this tool at all. Rewriting a remote's history is not something to get
  wrong on someone's behalf, and the escape hatch — a person typing it
  themselves — is right there.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

# A read of a large repository can be slow; a hung one must not hold a turn open.
TIMEOUT_S = 60
# Enough of a log or diff to be useful, bounded so one call cannot fill the
# model's context (a tool result is re-sent on every remaining step of a turn).
MAX_OUTPUT_CHARS = 6_000


class GitError(RuntimeError):
    """git itself refused. The message is git's own — it is nearly always the
    most useful thing anyone could say about the failure."""


@dataclass(frozen=True)
class GitResult:
    ok: bool
    output: str


def _clip(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n… ({len(text) - MAX_OUTPUT_CHARS} more characters)"


class Repo:
    """One repository, at a path the registry has already approved."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def is_repo(self) -> bool:
        return (self.path / ".git").exists()

    def run(self, argv: list[str]) -> GitResult:
        if not self.is_repo:
            return GitResult(False, f"{self.path} is not a git repository")
        try:
            done = subprocess.run(
                ["git", *argv],
                cwd=self.path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TIMEOUT_S,
            )
        except FileNotFoundError:
            return GitResult(False, "git is not installed or not on PATH")
        except subprocess.TimeoutExpired:
            return GitResult(False, f"git {argv[0]} timed out after {TIMEOUT_S}s")
        output = _clip((done.stdout or "") + ("\n" + done.stderr if done.stderr.strip() else ""))
        return GitResult(done.returncode == 0, output or "(no output)")

    # ---- reads ----

    def status(self) -> GitResult:
        branch = self.run(["rev-parse", "--abbrev-ref", "HEAD"])
        state = self.run(["status", "--short", "--branch"])
        if not state.ok:
            return state
        head = branch.output if branch.ok else "?"
        return GitResult(True, f"on branch {head}\n{state.output}")

    def log(self, limit: int = 10, path: str = "") -> GitResult:
        argv = ["log", f"-{max(1, min(limit, 100))}", "--pretty=format:%h %ad %an — %s",
                "--date=short"]
        if path:
            argv += ["--", path]
        return self.run(argv)

    def diff(self, path: str = "", staged: bool = False) -> GitResult:
        """A named path gives the full diff for it; no path gives the summary —
        a whole-repo diff is usually thousands of lines nobody asked for."""
        argv = ["diff", *(["--cached"] if staged else [])]
        argv += ["--", path] if path else ["--stat"]
        return self.run(argv)

    def branches(self) -> GitResult:
        return self.run(["branch", "--all", "-vv"])

    def unpushed(self) -> GitResult:
        """Commits that a push would send. Also the push preview."""
        return self.run(["log", "@{u}..HEAD", "--pretty=format:%h %s"])

    # ---- writes ----

    def commit(self, message: str, paths: list[str] | None = None) -> GitResult:
        if not message.strip():
            return GitResult(False, "a commit needs a message")
        if paths:
            staged = self.run(["add", "--", *paths])
            if not staged.ok:
                return staged
        pending = self.run(["diff", "--cached", "--name-only"])
        if pending.ok and not pending.output.strip().replace("(no output)", ""):
            return GitResult(
                False,
                "nothing is staged, so there is nothing to commit. Stage the files "
                "first, or name the paths to include.",
            )
        return self.run(["commit", "-m", message])

    def push(self, remote: str = "", branch: str = "") -> GitResult:
        argv = ["push"]
        if remote:
            argv.append(remote)
            if branch:
                argv.append(branch)
        return self.run(argv)
