"""File ingestion — the first connector (review M5).

Text files become chunked `ingest.document` events in the log, projected into
the memory index. Ingested content is marked untrusted: it may *inform* the
model but can never *authorize* actions (taint rule, review W4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.events.schema import Event, EventType, Provenance, Tier
from core.memory.store import MemoryStore

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".py", ".json", ".toml", ".yaml", ".yml", ".csv"}
PDF_EXTENSION = ".pdf"
MAX_FILE_BYTES = 50_000_000
CHUNK_CHARS = 1_500


def extract_pdf_text(path: Path) -> str:
    """Plain-text extraction, page by page. Scanned PDFs (no text layer) come
    back empty — OCR is a later Gen 2 slice."""
    import pymupdf

    with pymupdf.open(path) as doc:
        return "\n\n".join(page.get_text() for page in doc)


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS) -> list[str]:
    """Split on paragraph boundaries, packing paragraphs up to the chunk size."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        if current and len(current) + len(p) + 2 > chunk_chars:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
        while len(current) > chunk_chars:  # single oversized paragraph
            chunks.append(current[:chunk_chars])
            current = current[chunk_chars:]
    if current:
        chunks.append(current)
    return chunks


class FileIngestor:
    def __init__(self, memory: MemoryStore, device_id: str) -> None:
        self._memory = memory
        self._device_id = device_id

    def ingest_path(self, path: str | Path) -> dict[str, Any]:
        root = Path(path).expanduser().resolve()
        if not root.exists():
            return {"ok": False, "error": f"path does not exist: {root}"}
        files = [root] if root.is_file() else sorted(
            p for p in root.rglob("*") if p.is_file()
        )
        ingested: list[str] = []
        skipped: list[str] = []
        chunks_total = 0
        for f in files:
            suffix = f.suffix.lower()
            if suffix not in TEXT_EXTENSIONS and suffix != PDF_EXTENSION:
                skipped.append(f.name)
                continue
            if f.stat().st_size > MAX_FILE_BYTES:
                skipped.append(f"{f.name} (too large)")
                continue
            if suffix == PDF_EXTENSION:
                try:
                    text = extract_pdf_text(f)
                except Exception as exc:
                    skipped.append(f"{f.name} (pdf error: {exc})")
                    continue
                if not text.strip():
                    skipped.append(f"{f.name} (no text layer — scanned PDF? OCR comes later)")
                    continue
            else:
                text = f.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                self._memory.record(
                    Event(
                        type=EventType.INGEST_DOCUMENT.value,
                        device=self._device_id,
                        tier=Tier.T1,
                        provenance=Provenance(source=f"connector:file:{f.name}", trusted=False),
                        payload={
                            "text": chunk,
                            "file": str(f),
                            "chunk": i,
                            "total_chunks": len(chunks),
                        },
                    )
                )
            ingested.append(f.name)
            chunks_total += len(chunks)
        return {
            "ok": True,
            "files_ingested": len(ingested),
            "chunks": chunks_total,
            "skipped": skipped[:20],
        }
