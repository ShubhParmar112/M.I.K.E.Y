"""The restore-from-backup drill (Gen 1 exit criterion). Build a real store,
back it up, destroy the database completely, restore, and prove nothing was lost:
event count matches, the audit chain still verifies, and a canary memory is
recallable again — i.e. the projection was rebuilt from the log."""

from __future__ import annotations

from pathlib import Path

from core.backup.store import create_backup, restore_backup, verify_backup
from core.events.schema import Event, EventType, Provenance
from core.events.store import EventStore
from core.memory.store import MemoryStore
from core.policy.engine import ActionRequest, PolicyEngine
from core.storage.db import Database


def _seed(db: Database) -> int:
    memory = MemoryStore(db, EventStore(db))
    policy = PolicyEngine(db)
    policy.evaluate(ActionRequest("fs_read", {}, "t1", "s1"))  # writes an audit entry
    policy.evaluate(ActionRequest("fs_write", {}, "t1", "s1"))
    memory.remember("Canary: the zephyr turbines hum at dawn.")
    memory.record(Event(
        type=EventType.INGEST_DOCUMENT.value,
        provenance=Provenance(source="connector:file:notes.md", trusted=False),
        payload={"text": "An ingested source about quibit stabilization."},
    ))
    return db.conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]


def test_backup_restore_drill(tmp_path: Path) -> None:
    db_path = tmp_path / "mikey.db"
    db = Database(db_path)
    events_before = _seed(db)
    assert MemoryStore(db, EventStore(db)).recall("zephyr turbines")  # present before

    backup_path, manifest = create_backup(db, tmp_path / "backups", build="testbuild")
    assert manifest.audit_valid and manifest.event_count == events_before
    ok, issues = verify_backup(backup_path)
    assert ok, issues
    db.close()

    # --- catastrophe: the database is gone ---
    db_path.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(db_path) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    assert not db_path.exists()

    # --- restore ---
    report = restore_backup(backup_path, db_path)
    assert report.ok, report.issues
    assert report.audit_valid and report.integrity_ok
    assert report.event_count == events_before
    assert report.reprojected > 0  # projection was rebuilt from the log

    # --- prove the restored store actually works ---
    db2 = Database(db_path)
    assert MemoryStore(db2, EventStore(db2)).recall("zephyr turbines")  # canary survived
    assert PolicyEngine(db2).verify_audit_chain() is True
    db2.close()


def test_restore_refuses_corrupted_backup(tmp_path: Path) -> None:
    db_path = tmp_path / "mikey.db"
    db = Database(db_path)
    _seed(db)
    backup_path, _ = create_backup(db, tmp_path / "backups", build="testbuild")
    db.close()

    # corrupt the backup bytes (leaves the manifest's checksum no longer matching)
    data = bytearray(backup_path.read_bytes())
    for i in range(200, 400):
        data[i] ^= 0xFF
    backup_path.write_bytes(bytes(data))

    ok, issues = verify_backup(backup_path)
    assert not ok and issues  # corruption detected

    target = tmp_path / "restored.db"
    report = restore_backup(backup_path, target)
    assert report.ok is False  # refused, target not left in a half-restored state
    assert not target.exists()
