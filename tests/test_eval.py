"""The retrieval eval harness is itself a regression gate: it must run on the
committed golden set and clear the quality bar. As harder (paraphrase) cases are
added and metrics dip, this test is where a retrieval regression shows up."""

from __future__ import annotations

from core.eval.retrieval import FP_PASS, HIT3_PASS, load_golden, run_eval


def test_golden_set_is_wellformed() -> None:
    golden = load_golden()
    assert golden["corpus"] and golden["cases"]
    doc_ids = {d["id"] for d in golden["corpus"]}
    for case in golden["cases"]:
        assert case["query"].strip()
        # every referenced relevant doc must exist in the corpus
        assert all(d in doc_ids for d in case["relevant"])


def test_retrieval_eval_meets_the_gate() -> None:
    report = run_eval(load_golden())
    assert report.n_positive >= 8  # a meaningful seed, not one case
    assert report.hit_at[3] >= HIT3_PASS
    assert report.false_positive_rate <= FP_PASS
    assert report.passed is True
