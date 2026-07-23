from __future__ import annotations

from pathlib import Path

from core.events.store import EventStore
from core.ingest.files import FileIngestor, chunk_text
from core.memory.store import MemoryStore
from core.storage.db import Database


def test_chunk_text_packs_paragraphs_and_splits_oversized() -> None:
    assert chunk_text("one\n\ntwo") == ["one\n\ntwo"]
    chunks = chunk_text("a" * 3200, chunk_chars=1500)
    assert len(chunks) == 3
    assert all(len(c) <= 1500 for c in chunks)


def test_ingest_directory_marks_untrusted_and_is_recallable(
    db: Database, tmp_path: Path
) -> None:
    (tmp_path / "notes.md").write_text(
        "MIKEY design notes.\n\nThe policy engine mediates every side effect.",
        encoding="utf-8",
    )
    (tmp_path / "binary.exe").write_bytes(b"\x00\x01")
    memory = MemoryStore(db, EventStore(db))
    report = FileIngestor(memory, "dev_test").ingest_path(tmp_path)
    assert report["ok"] and report["files_ingested"] == 1
    assert "binary.exe" in report["skipped"]

    hits = memory.recall("policy engine side effect")
    assert hits and hits[0].trusted is False
    assert hits[0].source == "connector:file:notes.md"


def test_ingest_pdf_extracts_text_and_is_recallable(db: Database, tmp_path: Path) -> None:
    import pymupdf

    pdf_path = tmp_path / "paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72), "Abstract: We study retrieval augmented memory for personal agents."
    )
    doc.save(pdf_path)
    doc.close()

    memory = MemoryStore(db, EventStore(db))
    report = FileIngestor(memory, "dev_test").ingest_path(pdf_path)
    assert report["ok"] and report["files_ingested"] == 1

    hits = memory.recall("retrieval augmented memory")
    assert hits and hits[0].source == "connector:file:paper.pdf"
    assert hits[0].trusted is False


def test_ingest_missing_path_reports_error(db: Database, tmp_path: Path) -> None:
    memory = MemoryStore(db, EventStore(db))
    report = FileIngestor(memory, "dev_test").ingest_path(tmp_path / "nope")
    assert report["ok"] is False


def test_reingesting_same_file_is_skipped_not_duplicated(db: Database, tmp_path: Path) -> None:
    """A file already in memory must not be ingested again (no bloat, no re-work),
    unless force=True — this is what stopped the 4x-ingest / approval spam."""
    f = tmp_path / "paper.md"
    f.write_text("The theorem proves quibits stabilize.\n\nA second paragraph here.",
                 encoding="utf-8")
    ingestor = FileIngestor(MemoryStore(db, EventStore(db)), "dev_test")

    first = ingestor.ingest_path(f)
    assert first["files_ingested"] == 1 and first["chunks"] >= 1

    second = ingestor.ingest_path(f)  # same file again
    assert second["files_ingested"] == 0
    assert "paper.md" in second["already_ingested"]

    forced = ingestor.ingest_path(f, force=True)  # explicit re-ingest of a changed file
    assert forced["files_ingested"] == 1
