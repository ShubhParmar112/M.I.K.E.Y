"""Reach: acting outside the sandbox, and the boundary that makes it safe.

The sandbox was the whole safety story until now, and this package deliberately
opens doors in it. So most of what is pinned here is the lock: what gets refused,
that the model cannot widen its own reach, and that the two actions which leave a
mark — commit and push — cannot happen without a simulation the user has seen.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import httpx
import pytest

from core.events.store import EventStore
from core.orchestrator.tools import TOOLS
from core.policy.engine import ActionRequest, Decision, PolicyEngine
from core.policy.preview import PREVIEWABLE, classify
from core.reach.desktop import OpenRefused, open_url
from core.reach.git import Repo
from core.reach.github import GitHub, GitHubError
from core.reach.projects import OutsideReach, ProjectRegistry
from core.storage.db import Database


@pytest.fixture
def registry(tmp_path: Path) -> ProjectRegistry:
    return ProjectRegistry(EventStore(Database(tmp_path / "t.db")))


def _git(path: Path, *argv: str) -> subprocess.CompletedProcess[bytes]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "GIT_AUTHOR_NAME": "T",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "T",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    return subprocess.run(["git", *argv], cwd=path, capture_output=True, env=env, check=True)


@pytest.fixture
def repo_dir(tmp_path: Path) -> Path:
    """A real git repository with one commit."""
    path = tmp_path / "proj"
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "T")
    (path / "README.md").write_text("hello\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "first")
    return path


# --- the boundary -------------------------------------------------------------


def test_nothing_is_reachable_until_something_is_registered(registry: ProjectRegistry) -> None:
    """The default is the old one: the sandbox and nothing else."""
    with pytest.raises(OutsideReach) as exc:
        registry.resolve("C:/Windows/System32")

    assert "no projects are registered" in str(exc.value)
    assert "mikey project add" in str(exc.value), "a refusal should say what would fix it"


def test_a_registered_project_and_its_contents_resolve(
    registry: ProjectRegistry, tmp_path: Path
) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "a.py").write_text("x = 1", encoding="utf-8")
    registry.register(work)

    assert registry.resolve(str(work)) == work.resolve()
    assert registry.resolve(str(work / "src" / "a.py")) == (work / "src" / "a.py").resolve()
    assert registry.resolve("src/a.py") == (work / "src" / "a.py").resolve()


def test_a_sibling_directory_is_still_refused(
    registry: ProjectRegistry, tmp_path: Path
) -> None:
    """Registering one directory opens that directory, not its neighbourhood."""
    (tmp_path / "work").mkdir()
    (tmp_path / "secrets").mkdir()
    registry.register(tmp_path / "work")

    with pytest.raises(OutsideReach):
        registry.resolve(str(tmp_path / "secrets"))


def test_dot_dot_cannot_climb_out(registry: ProjectRegistry, tmp_path: Path) -> None:
    """The oldest escape there is. Paths are compared after resolution, never as
    strings, so `work/../secrets` is `secrets` before anything looks at it."""
    (tmp_path / "work").mkdir()
    (tmp_path / "secrets").mkdir()
    registry.register(tmp_path / "work")

    with pytest.raises(OutsideReach):
        registry.resolve(str(tmp_path / "work" / ".." / "secrets"))


def test_a_symlink_out_of_a_project_is_refused(
    registry: ProjectRegistry, tmp_path: Path
) -> None:
    """Resolution follows links first — otherwise a link inside a project is a
    door to anywhere on the disk."""
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("shh", encoding="utf-8")
    try:
        (work / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need elevation on this machine")
    registry.register(work)

    with pytest.raises(OutsideReach):
        registry.resolve(str(work / "link" / "secret.txt"))


def test_closing_a_project_takes_effect_immediately(
    registry: ProjectRegistry, tmp_path: Path
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    registry.register(work)
    assert registry.resolve(str(work)) == work.resolve()

    assert registry.forget(work) is True

    with pytest.raises(OutsideReach):
        registry.resolve(str(work))


def test_the_registry_survives_the_process(registry: ProjectRegistry, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    registry.register(work)

    rebuilt = ProjectRegistry(registry._events)  # a fresh process, same log

    assert [p.path for p in rebuilt.projects()] == [work.resolve()]


def test_no_tool_can_register_a_project() -> None:
    """The load-bearing rule of the whole package: M.I.K.E.Y can use the reach it
    was given and cannot give itself more. A model that can widen its own boundary
    has no boundary — every document it has ever read becomes a way to ask."""
    names = {str(t["name"]) for t in TOOLS}
    actions: set[str] = set()
    for tool in TOOLS:
        enum = tool["input_schema"].get("properties", {}).get("action", {}).get("enum", [])
        actions.update(str(a) for a in enum)

    forbidden = {"register", "project_add", "add_project", "grant", "allow", "trust"}
    assert not (names & forbidden)
    assert not (actions & forbidden)
    assert not any("project" in name for name in names)


# --- policy: reading a repo is nothing like pushing one -----------------------


def _req(tool: str, args: dict[str, object], session: str = "s1") -> ActionRequest:
    return ActionRequest(tool=tool, args=args, turn_id="t1", session_id=session)


def test_reads_are_allowed_and_writes_ask(tmp_path: Path) -> None:
    policy = PolicyEngine(Database(tmp_path / "p.db"))

    assert policy.evaluate(_req("git", {"action": "status"})).decision is Decision.ALLOW
    assert policy.evaluate(_req("git", {"action": "log"})).decision is Decision.ALLOW
    assert policy.evaluate(_req("git", {"action": "commit"})).decision is Decision.ASK
    assert policy.evaluate(_req("git", {"action": "push"})).decision is Decision.ASK
    assert policy.evaluate(_req("open", {"target": "a.py"})).decision is Decision.ASK


def test_an_unknown_git_action_is_denied(tmp_path: Path) -> None:
    """The allowlist stays the boundary: anything not named is refused, so a new
    git subcommand can never arrive pre-approved."""
    policy = PolicyEngine(Database(tmp_path / "p.db"))

    result = policy.evaluate(_req("git", {"action": "reset", "hard": True}))

    assert result.decision is Decision.DENY
    assert "no rule for action" in result.reason


def test_approving_git_for_the_session_does_not_cover_a_push(tmp_path: Path) -> None:
    """"Approve for the session" on a commit must not silently become permission to
    publish. The grant names the action, not just the tool."""
    policy = PolicyEngine(Database(tmp_path / "p.db"))
    commit = _req("git", {"action": "commit", "message": "wip"})
    policy.grant_session(commit)

    assert policy.evaluate(commit).decision is Decision.ALLOW
    assert policy.evaluate(_req("git", {"action": "push"})).decision is Decision.ASK


def test_github_reads_without_asking_but_never_writes() -> None:
    """It is read-only, so it allows — and what it returns is tainted, which is what
    escalates anything the turn does afterwards."""
    names = {t["name"] for t in TOOLS}
    github = next(t for t in TOOLS if t["name"] == "github")
    actions = github["input_schema"]["properties"]["action"]["enum"]

    assert "github" in names
    assert set(actions) == {"repo", "pulls", "issues"}, "no write actions on this surface"


# --- simulate first ------------------------------------------------------------


def test_a_push_is_previewed_as_irreversible() -> None:
    """The only action on the whole surface that leaves the machine. Once it is on
    someone else's server it is in everyone else's next fetch."""
    assert "git" in PREVIEWABLE

    impact = classify("git", {"action": "push"})

    assert impact.destructive
    assert not impact.reversible
    assert "cannot be unsent" in impact.why


def test_a_commit_is_previewed_but_read_as_recoverable() -> None:
    impact = classify("git", {"action": "commit", "message": "x"})

    assert impact.destructive, "it writes history — the user should see what goes in"
    assert impact.reversible, "but it is local, and git reset exists"


def test_reading_a_repository_is_not_destructive() -> None:
    for action in ("status", "log", "diff", "branches", "unpushed"):
        assert not classify("git", {"action": action}).destructive


# --- git, on a real repository -------------------------------------------------


def test_status_and_log_read_a_real_repository(repo_dir: Path) -> None:
    repo = Repo(repo_dir)

    assert repo.is_repo
    assert "main" in repo.status().output
    assert "first" in repo.log().output


def test_a_commit_with_nothing_staged_refuses_rather_than_inventing_one(
    repo_dir: Path,
) -> None:
    """Sweeping every modified file into a commit is exactly the behaviour that
    catches the file you were halfway through."""
    (repo_dir / "unstaged.txt").write_text("not ready", encoding="utf-8")
    repo = Repo(repo_dir)

    result = repo.commit("tidy up")

    assert not result.ok
    assert "nothing is staged" in result.output
    assert "unstaged.txt" not in Repo(repo_dir).log().output


def test_a_commit_records_exactly_the_named_paths(repo_dir: Path) -> None:
    (repo_dir / "wanted.txt").write_text("yes", encoding="utf-8")
    (repo_dir / "untouched.txt").write_text("no", encoding="utf-8")
    repo = Repo(repo_dir)

    assert repo.commit("add wanted", ["wanted.txt"]).ok

    listed = repo.run(["show", "--name-only", "--pretty=format:", "HEAD"]).output
    assert "wanted.txt" in listed
    assert "untouched.txt" not in listed


def test_push_never_forces(repo_dir: Path) -> None:
    """Force-pushing rewrites a remote's history. It is not reachable through this
    tool at all — the escape hatch is a person typing it themselves."""
    calls: list[list[str]] = []

    class _Recorded(Repo):
        def run(self, argv: list[str]) -> object:  # type: ignore[override]
            calls.append(argv)
            return super().run(argv)

    _Recorded(repo_dir).push("origin", "main")

    assert calls, "push should have run something"
    flat = " ".join(" ".join(c) for c in calls)
    assert "--force" not in flat
    assert "-f " not in flat + " "


def test_a_missing_repository_fails_cleanly(tmp_path: Path) -> None:
    result = Repo(tmp_path).status()

    assert not result.ok
    assert "not a git repository" in result.output


# --- github --------------------------------------------------------------------


def _gh(payload: object, status: int = 200) -> GitHub:
    return GitHub(transport=httpx.MockTransport(lambda r: httpx.Response(status, json=payload)))


def test_pull_requests_are_listed_and_marked_untrusted() -> None:
    """A PR title is written by someone else. One of them will eventually contain a
    sentence addressed to an assistant rather than a person."""
    client = _gh([
        {"number": 7, "title": "Fix the parser", "user": {"login": "someone"},
         "head": {"ref": "fix-parser"}},
    ])

    result = client.pulls("owner/repo")

    assert "#7 Fix the parser" in result.output
    assert "someone" in result.output
    assert result.tainted is True


def test_pull_requests_are_not_reported_as_issues() -> None:
    """GitHub returns PRs from the issues endpoint; nobody means that by "issues"."""
    client = _gh([
        {"number": 1, "title": "a real issue", "user": {"login": "a"}},
        {"number": 2, "title": "a pull request", "user": {"login": "b"}, "pull_request": {}},
    ])

    result = client.issues("owner/repo")

    assert "a real issue" in result.output
    assert "a pull request" not in result.output


def test_a_missing_repo_says_so_usefully() -> None:
    client = _gh({"message": "Not Found"}, status=404)

    with pytest.raises(GitHubError, match="no such repository"):
        client.repo("owner/nope")


def test_a_rate_limited_anonymous_read_names_the_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    client = _gh({"message": "rate limited"}, status=403)

    with pytest.raises(GitHubError, match="GITHUB_TOKEN"):
        client.repo("owner/repo")


# --- opening things ------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///C:/Windows/System32/drivers/etc/hosts",
        "javascript:alert(1)",
        "vbscript:msgbox",
        "data:text/html;base64,PHNjcmlwdD4=",
    ],
)
def test_only_web_addresses_are_opened(url: str) -> None:
    """A "link" that runs local code is the oldest trick there is, and a model that
    has just read a web page is exactly who would be talked into opening one."""
    with pytest.raises(OpenRefused):
        open_url(url)


def test_a_url_without_a_host_is_refused() -> None:
    with pytest.raises(OpenRefused):
        open_url("https://")
