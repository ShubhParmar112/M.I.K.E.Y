"""GitHub, read-only, over the REST API.

No `gh` CLI dependency: it isn't installed here, and shelling out to a tool that
holds its own credentials is a worse boundary than an HTTP call with a token this
process was given deliberately. `GITHUB_TOKEN` is used when present; without one
public repositories still work, at a much lower rate limit.

**Everything this returns is untrusted.** A pull request title, an issue body, a
review comment — these are written by other people, and one of them may
eventually contain a sentence addressed to an AI assistant rather than to a
human. So results are tainted the same way fetched web pages are, which is what
makes the policy engine escalate any action taken afterwards in that turn.

Read-only on purpose. Opening an issue or a pull request publishes something
under the user's name, and that deserves its own preview showing the exact text —
a separate piece of work rather than a flag on this one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

API = "https://api.github.com"
TIMEOUT_S = 20
MAX_ITEMS = 20
BODY_CHARS = 500


class GitHubError(RuntimeError):
    """The API refused, or could not be reached."""


@dataclass(frozen=True)
class GitHubResult:
    ok: bool
    output: str
    tainted: bool = True  # always: this is other people's text


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _clip(text: str, limit: int = BODY_CHARS) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


class GitHub:
    """A thin read client. `transport` is injectable so tests never touch the network."""

    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self._transport = transport

    @property
    def authenticated(self) -> bool:
        return bool(os.environ.get("GITHUB_TOKEN"))

    def _get(self, path: str, params: dict[str, str] | None = None) -> object:
        try:
            with httpx.Client(timeout=TIMEOUT_S, transport=self._transport) as client:
                resp = client.get(f"{API}{path}", headers=_headers(), params=params or {})
        except httpx.HTTPError as exc:
            raise GitHubError(f"could not reach GitHub ({type(exc).__name__})") from exc
        if resp.status_code == 404:
            raise GitHubError("no such repository, or it is private and the token can't see it")
        if resp.status_code in (401, 403):
            hint = (
                "the rate limit for unauthenticated requests is low — set GITHUB_TOKEN"
                if not self.authenticated
                else "the token was refused or lacks the scope for this"
            )
            raise GitHubError(f"GitHub refused the request ({resp.status_code}) — {hint}")
        if resp.status_code >= 400:
            raise GitHubError(f"GitHub returned {resp.status_code}")
        return resp.json()

    def repo(self, full_name: str) -> GitHubResult:
        data = self._get(f"/repos/{full_name}")
        if not isinstance(data, dict):
            raise GitHubError("unexpected response shape")
        lines = [
            f"{data.get('full_name')} — {_clip(str(data.get('description') or ''), 200)}",
            f"default branch: {data.get('default_branch')} · "
            f"{data.get('stargazers_count', 0)} stars · "
            f"{data.get('open_issues_count', 0)} open issues/PRs",
            f"pushed: {str(data.get('pushed_at') or '')[:10]}",
        ]
        return GitHubResult(True, "\n".join(lines))

    def pulls(self, full_name: str, state: str = "open") -> GitHubResult:
        data = self._get(f"/repos/{full_name}/pulls", {"state": state, "per_page": str(MAX_ITEMS)})
        if not isinstance(data, list):
            raise GitHubError("unexpected response shape")
        if not data:
            return GitHubResult(True, f"no {state} pull requests")
        lines = [
            f"#{p.get('number')} {_clip(str(p.get('title')), 120)} "
            f"[{(p.get('user') or {}).get('login', '?')} · "
            f"{(p.get('head') or {}).get('ref', '?')}]"
            for p in data
        ]
        return GitHubResult(True, "\n".join(lines))

    def issues(self, full_name: str, state: str = "open") -> GitHubResult:
        data = self._get(f"/repos/{full_name}/issues", {"state": state, "per_page": str(MAX_ITEMS)})
        if not isinstance(data, list):
            raise GitHubError("unexpected response shape")
        # GitHub returns pull requests from the issues endpoint too; they are not
        # what anyone means by "issues".
        real = [i for i in data if "pull_request" not in i]
        if not real:
            return GitHubResult(True, f"no {state} issues")
        lines = [
            f"#{i.get('number')} {_clip(str(i.get('title')), 120)} "
            f"[{(i.get('user') or {}).get('login', '?')}]"
            for i in real
        ]
        return GitHubResult(True, "\n".join(lines))
