"""Opening things: a file in the editor, a page in the browser.

Small, and the smallness is the point — this launches a program with an argument,
which is exactly the shape of a problem if the argument came from somewhere
untrusted. So:

* A **path** must already have passed the project registry; this module only ever
  receives resolved ones.
* A **URL** must be http or https. `file:`, `javascript:`, `vbscript:` and the
  rest are refused outright — a "link" that runs local code is the oldest trick
  there is, and a model that has just read a web page is precisely who would be
  talked into opening one.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

SAFE_SCHEMES = frozenset({"http", "https"})
LAUNCH_TIMEOUT_S = 15


class OpenRefused(ValueError):
    """The target is not something worth opening on someone's behalf."""


@dataclass(frozen=True)
class OpenResult:
    ok: bool
    output: str


def editor_command() -> list[str] | None:
    """The editor to open files in, if one is on PATH."""
    for candidate in ("code", "code-insiders", "subl", "notepad++"):
        found = shutil.which(candidate)
        if found:
            return [found]
    return None


def open_in_editor(path: Path, line: int = 0) -> OpenResult:
    """Open a resolved path in the editor, at a line if one is given."""
    if not path.exists():
        return OpenResult(False, f"{path} does not exist")
    editor = editor_command()
    if editor is None:
        # Falling back to the OS association is better than refusing: the person
        # asked to see the thing, and Windows knows what opens a .pdf.
        try:
            os.startfile(str(path))  # noqa: S606 — a user-registered path, by request
        except (OSError, AttributeError) as exc:
            return OpenResult(False, f"no editor found and the OS could not open it: {exc}")
        return OpenResult(True, f"opened {path.name} with the default application")

    target = f"{path}:{line}" if line and path.is_file() else str(path)
    argv = [*editor, *(["-g", target] if line and path.is_file() else [target])]
    try:
        subprocess.run(argv, timeout=LAUNCH_TIMEOUT_S, capture_output=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return OpenResult(False, f"could not launch the editor: {exc}")
    where = f" at line {line}" if line and path.is_file() else ""
    return OpenResult(True, f"opened {path.name}{where} in {Path(editor[0]).stem}")


def open_url(url: str) -> OpenResult:
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in SAFE_SCHEMES:
        raise OpenRefused(
            f"'{parsed.scheme or url[:20]}' is not a web address. I only open http and https — "
            "other schemes can run programs rather than show a page."
        )
    if not parsed.netloc:
        raise OpenRefused(f"'{url}' has no host — that isn't a page I can open")
    opened = webbrowser.open(url)
    if not opened:
        return OpenResult(False, "no browser is available to open it")
    return OpenResult(True, f"opened {parsed.netloc}{parsed.path or ''} in your browser")
