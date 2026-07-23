"""Backup & restore (Gen 1 exit criterion: "restore-from-backup drill passes").

A backup is a consistent online snapshot of the whole SQLite store. Restore is
deliberately more than a file copy: it rebuilds the memory projection from the
log (reindex) and re-verifies the audit chain — exercising, not just trusting,
M.I.K.E.Y's core claim that the event log is the truth and everything else is a
derivable projection.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.events.schema import now
from core.events.store import EventStore
from core.memory.store import MemoryStore
from core.policy.engine import audit_chain_valid
from core.storage.db import Database


@dataclass
class BackupManifest:
    created_at: str
    schema_version: int
    event_count: int
    audit_count: int
    audit_head: str
    audit_valid: bool
    sha256: str
    build: str


@dataclass
class RestoreReport:
    ok: bool
    event_count: int
    audit_valid: bool
    integrity_ok: bool
    reprojected: int
    issues: list[str] = field(default_factory=list)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _manifest_path(backup_path: Path) -> Path:
    return backup_path.with_name(backup_path.name + ".manifest.json")


def _ro_conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def create_backup(db: Database, backups_dir: Path, build: str = "unknown") -> tuple[Path, BackupManifest]:
    """Consistent online snapshot (safe even with a live gateway / WAL) plus a
    manifest of what it should contain, so a restore can be trusted."""
    backups_dir.mkdir(parents=True, exist_ok=True)
    dest = backups_dir / f"mikey-{now().strftime('%Y%m%dT%H%M%SZ')}.db"
    target = sqlite3.connect(str(dest))
    try:
        db.conn.backup(target)  # SQLite online backup API — atomic, WAL-safe
    finally:
        target.close()

    con = _ro_conn(dest)
    try:
        schema_version = con.execute("SELECT MAX(version) v FROM schema_version").fetchone()["v"] or 0
        event_count = con.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
        audit_count = con.execute("SELECT COUNT(*) c FROM audit").fetchone()["c"]
        head = con.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
        manifest = BackupManifest(
            created_at=now().isoformat(),
            schema_version=schema_version,
            event_count=event_count,
            audit_count=audit_count,
            audit_head=head["hash"] if head else "GENESIS",
            audit_valid=audit_chain_valid(con),
            sha256=_sha256_file(dest),
            build=build,
        )
    finally:
        con.close()

    _manifest_path(dest).write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return dest, manifest


def read_manifest(backup_path: Path) -> BackupManifest | None:
    mp = _manifest_path(Path(backup_path))
    if not mp.exists():
        return None
    return BackupManifest(**json.loads(mp.read_text(encoding="utf-8")))


def verify_backup(backup_path: Path) -> tuple[bool, list[str]]:
    """Check a backup without restoring it: file checksum (at-rest corruption),
    SQLite integrity, audit-chain validity, and count vs manifest. Read-only."""
    backup_path = Path(backup_path)
    if not backup_path.exists():
        return False, [f"backup file not found: {backup_path}"]

    issues: list[str] = []
    manifest = read_manifest(backup_path)
    if manifest is None:
        issues.append("manifest missing — cannot check integrity claims")
    elif _sha256_file(backup_path) != manifest.sha256:
        issues.append("checksum mismatch — backup file is corrupted or altered")

    try:
        con = _ro_conn(backup_path)
    except sqlite3.DatabaseError as exc:
        return False, issues + [f"cannot open backup as a database: {exc}"]
    try:
        integ = con.execute("PRAGMA integrity_check").fetchone()[0]
        if integ != "ok":
            issues.append(f"integrity_check failed: {integ}")
        if not audit_chain_valid(con):
            issues.append("audit chain does not verify")
        count = con.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
        if manifest and count != manifest.event_count:
            issues.append(f"event count {count} != manifest {manifest.event_count}")
    except sqlite3.DatabaseError as exc:
        issues.append(f"backup unreadable: {exc}")
    finally:
        con.close()
    return (not issues), issues


def restore_backup(backup_path: Path, target_db_path: Path) -> RestoreReport:
    """Refuse to restore an invalid backup; otherwise replace the store, rebuild
    the projection from the log, and verify the result."""
    backup_path, target_db_path = Path(backup_path), Path(target_db_path)
    ok, issues = verify_backup(backup_path)
    if not ok:
        return RestoreReport(False, 0, False, False, 0, ["refusing to restore an invalid backup:"] + issues)

    target_db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(backup_path, target_db_path)
    for suffix in ("-wal", "-shm"):  # drop stale sidecars from any prior db
        sidecar = Path(str(target_db_path) + suffix)
        if sidecar.exists():
            sidecar.unlink()

    db = Database(target_db_path)
    try:
        reprojected = MemoryStore(db, EventStore(db)).reindex()  # rebuild projection from truth
        audit_valid = audit_chain_valid(db.conn)
        integrity_ok = db.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        event_count = db.conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
    finally:
        db.close()
    return RestoreReport(
        ok=audit_valid and integrity_ok,
        event_count=event_count,
        audit_valid=audit_valid,
        integrity_ok=integrity_ok,
        reprojected=reprojected,
    )
