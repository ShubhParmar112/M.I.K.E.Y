"""The reach boundary: directories the user has explicitly opened up.

This is the whole safety story of the package, so it is worth being exact about
what it does and does not promise.

**It promises:** a path handed to any reach tool is refused unless it resolves
inside a directory the user registered from the CLI. Resolution happens after
following symlinks and after normalising `..`, so neither is a way out. The
registry lives in the event log, so what was opened, when, and by whom is
reconstructable and appears in the audit trail like everything else.

**It does not promise** anything about what happens inside a registered project.
Registering a directory means "you may run git here"; it is a grant of authority,
and the way to keep that grant small is to register the repository you are
working on rather than your home directory. `mikey project add` says so.

There is no tool for registering a project. That is the point: the model can use
the reach it has been given and cannot give itself more, so a document it reads
cannot talk it into widening the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.events.schema import Event, EventType, Provenance
from core.events.store import EventStore


class OutsideReach(PermissionError):
    """A path outside every registered project. Refusing is the correct outcome —
    the message names what would make it work, because a person hitting this is
    usually one `mikey project add` away from what they wanted."""


@dataclass(frozen=True)
class Project:
    name: str  # the directory's own name, used as a short handle
    path: Path
    registered: datetime

    @property
    def is_git_repo(self) -> bool:
        return (self.path / ".git").exists()


def _normalise(raw: str | Path) -> Path:
    """Absolute, symlink-resolved, `..` collapsed. Everything compares in this form
    — a boundary that compares strings is not a boundary."""
    return Path(raw).expanduser().resolve()


class ProjectRegistry:
    """Which directories are reachable, as a projection over the log."""

    _TYPES = [EventType.PROJECT_REGISTERED.value, EventType.PROJECT_FORGOTTEN.value]
    _SCAN = 10_000

    def __init__(self, events: EventStore, device_id: str = "dev_desktop_1") -> None:
        self._events = events
        self._device = device_id

    # ---- the user's side (CLI only) ----

    def register(self, raw: str | Path) -> Project:
        path = _normalise(raw)
        if not path.is_dir():
            raise NotADirectoryError(f"{path} is not a directory")
        self._events.append(
            Event(
                type=EventType.PROJECT_REGISTERED.value,
                device=self._device,
                # "user", always: this event is only ever appended from the CLI, and
                # the provenance is what an audit reads to see that.
                provenance=Provenance(source="user", trusted=True),
                payload={"path": str(path), "name": path.name},
            )
        )
        found = self.get(path.name)
        assert found is not None
        return found

    def forget(self, raw: str | Path) -> bool:
        path = _normalise(raw)
        if not any(p.path == path for p in self.projects()):
            return False
        self._events.append(
            Event(
                type=EventType.PROJECT_FORGOTTEN.value,
                device=self._device,
                provenance=Provenance(source="user", trusted=True),
                payload={"path": str(path), "name": path.name},
            )
        )
        return True

    # ---- the boundary ----

    def projects(self) -> list[Project]:
        """Currently registered projects, in registration order."""
        live: dict[str, Project] = {}
        for ev in self._events.recent(types=self._TYPES, limit=self._SCAN):
            path = str(ev.payload.get("path", ""))
            if not path:
                continue
            if ev.type == EventType.PROJECT_FORGOTTEN.value:
                live.pop(path, None)
            else:
                live[path] = Project(
                    name=str(ev.payload.get("name") or Path(path).name),
                    path=Path(path),
                    registered=ev.ts,
                )
        return list(live.values())

    def get(self, name_or_path: str) -> Project | None:
        """A project by its short name or by any path inside it."""
        projects = self.projects()
        for project in projects:
            if project.name == name_or_path:
                return project
        try:
            wanted = _normalise(name_or_path)
        except OSError:
            return None
        for project in projects:
            if wanted == project.path or project.path in wanted.parents:
                return project
        return None

    def resolve(self, raw: str) -> Path:
        """The absolute path for `raw`, or refuse.

        Accepts a project name, a project-relative path (`M.I.K.E.Y/core`), or an
        absolute path — but every form has to land inside a registered project.
        """
        registered = self.projects()
        if not registered:
            raise OutsideReach(
                "no projects are registered, so I can't reach anything outside the "
                "workspace. `mikey project add <path>` opens one up."
            )

        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            # A relative path is read against each project in turn, so "core/x.py"
            # works without the person having to type the whole thing.
            head = candidate.parts[0] if candidate.parts else ""
            for project in registered:
                base = project.path if project.name != head else project.path.parent
                resolved = _normalise(base / candidate)
                if self._inside(resolved, registered):
                    return resolved
        else:
            resolved = _normalise(candidate)
            if self._inside(resolved, registered):
                return resolved

        names = ", ".join(p.name for p in registered)
        raise OutsideReach(
            f"'{raw}' is outside every project I can reach (I have: {names}). "
            "`mikey project add <path>` opens another one up."
        )

    @staticmethod
    def _inside(path: Path, projects: list[Project]) -> bool:
        return any(path == p.path or p.path in path.parents for p in projects)
