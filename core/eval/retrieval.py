"""Retrieval eval harness (Gen 2 exit criterion: "retrieval precision measured
on a personal golden set").

Builds a throwaway index from a fixed corpus, runs each golden query through the
real `MemoryStore.recall` path, and scores it — so "is memory trustworthy?" is a
number (hit@k, MRR, false-positive rate) with a pass gate and regression diff,
not a vibe. Deterministic and offline: no network, no touching the live DB.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.events.store import EventStore
from core.ingest.files import FileIngestor
from core.memory.store import MemoryStore
from core.storage.db import Database

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "evals" / "golden_set.json"
BASELINE_PATH = REPO_ROOT / "evals" / "baseline.json"

RECALL_K = 6
HIT_AT = (1, 3, 6)
HIT3_PASS = 0.80  # ≥80% of positive cases must surface a relevant hit in the top 3
FP_PASS = 0.10  # ≤10% of negative cases may return a spurious hit


@dataclass
class CaseResult:
    id: str
    query: str
    negative: bool
    first_relevant_rank: int | None  # 1-based; None if no relevant hit retrieved
    top_source: str | None
    passed: bool


@dataclass
class EvalReport:
    results: list[CaseResult]
    hit_at: dict[int, float]
    mrr: float
    false_positive_rate: float
    n_positive: int
    n_negative: int
    passed: bool
    regressions: list[str] = field(default_factory=list)


def load_golden(path: Path = GOLDEN_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_of(doc_id: str) -> str:
    return f"connector:file:{doc_id}.txt"


def run_eval(golden: dict[str, Any]) -> EvalReport:
    # mkdtemp + manual rmtree: on Windows the open SQLite file can't be deleted
    # by TemporaryDirectory's auto-cleanup, so close the db first and tolerate a
    # lingering WAL file.
    tmp = tempfile.mkdtemp(prefix="mikey-eval-")
    root = Path(tmp)
    db: Database | None = None
    try:
        corpus_dir = root / "corpus"
        corpus_dir.mkdir()
        for doc in golden["corpus"]:
            (corpus_dir / f"{doc['id']}.txt").write_text(doc["text"], encoding="utf-8")

        db = Database(root / "eval.db")
        memory = MemoryStore(db, EventStore(db))
        FileIngestor(memory, "eval").ingest_path(corpus_dir)

        results: list[CaseResult] = []
        hits = {k: 0 for k in HIT_AT}
        rr_sum = 0.0
        n_pos = n_neg = fp = 0

        for case in golden["cases"]:
            relevant = {_source_of(d) for d in case["relevant"]}
            hitlist = memory.recall(case["query"], k=RECALL_K)
            top_source = hitlist[0].source if hitlist else None

            if not relevant:  # negative case: nothing should be surfaced
                n_neg += 1
                is_fp = len(hitlist) > 0
                fp += int(is_fp)
                results.append(CaseResult(case["id"], case["query"], True, None, top_source, not is_fp))
                continue

            n_pos += 1
            ranks = [i + 1 for i, h in enumerate(hitlist) if h.source in relevant]
            first = min(ranks) if ranks else None
            for k in HIT_AT:
                hits[k] += int(first is not None and first <= k)
            rr_sum += (1.0 / first) if first else 0.0
            results.append(
                CaseResult(case["id"], case["query"], False, first, top_source,
                           first is not None and first <= 3)
            )

        hit_at = {k: (hits[k] / n_pos if n_pos else 0.0) for k in HIT_AT}
        mrr = rr_sum / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        passed = hit_at[3] >= HIT3_PASS and fpr <= FP_PASS
        report = EvalReport(results, hit_at, mrr, fpr, n_pos, n_neg, passed)
        report.regressions = _diff_baseline(report)
        return report
    finally:
        if db is not None:
            db.close()
        shutil.rmtree(root, ignore_errors=True)


# ---- regression baseline: per-case first-relevant rank, so a drop is visible ----


def _rank_map(report: EvalReport) -> dict[str, Any]:
    return {r.id: r.first_relevant_rank for r in report.results if not r.negative}


def _diff_baseline(report: EvalReport, path: Path = BASELINE_PATH) -> list[str]:
    if not path.exists():
        return []
    baseline = json.loads(path.read_text(encoding="utf-8"))
    now = _rank_map(report)
    worse: list[str] = []
    for cid, base_rank in baseline.items():
        cur = now.get(cid)
        # a case regresses if it used to find a relevant hit and now finds it later (or not at all)
        if base_rank is not None and (cur is None or cur > base_rank):
            worse.append(f"{cid}: rank {base_rank} -> {cur}")
    return worse


def save_baseline(report: EvalReport, path: Path = BASELINE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_rank_map(report), indent=2), encoding="utf-8")
