from __future__ import annotations

from email_thread_rag.rag.fusion import weighted_rrf


def test_exact_rrf_formula():
    fused = weighted_rrf(["a", "b"], ["b", "a"], k=60, lexical_weight=1.0, dense_weight=1.0)
    by_id = {chunk_id: score for chunk_id, score, _, _ in fused}
    # a: lexical_rank=1, dense_rank=2 -> 1/(60+1) + 1/(60+2)
    assert by_id["a"] == 1.0 / 61 + 1.0 / 62
    # b: lexical_rank=2, dense_rank=1 -> 1/(60+2) + 1/(60+1)
    assert by_id["b"] == 1.0 / 62 + 1.0 / 61


def test_weights_scale_each_branch_independently():
    fused = weighted_rrf(["a"], ["a"], k=60, lexical_weight=2.0, dense_weight=0.5)
    by_id = {chunk_id: score for chunk_id, score, _, _ in fused}
    assert by_id["a"] == 2.0 / 61 + 0.5 / 61


def test_chunk_in_both_branches_gets_both_contributions():
    fused = weighted_rrf(["shared"], ["shared"], k=60)
    chunk_id, score, lexical_rank, dense_rank = fused[0]
    assert chunk_id == "shared"
    assert lexical_rank == 1 and dense_rank == 1
    assert score == 1.0 / 61 + 1.0 / 61


def test_chunk_in_only_one_branch_remains_eligible():
    fused = weighted_rrf(["lexical-only"], [], k=60)
    assert len(fused) == 1
    chunk_id, score, lexical_rank, dense_rank = fused[0]
    assert chunk_id == "lexical-only"
    assert lexical_rank == 1
    assert dense_rank is None
    assert score == 1.0 / 61


def test_duplicate_ids_fused_once():
    # Same chunk_id appearing at multiple ranks within a single branch's list
    # collapses to its first (best) rank via the rank_by_id dict construction.
    fused = weighted_rrf(["x", "x", "y"], [], k=60)
    ids = [chunk_id for chunk_id, *_ in fused]
    assert ids.count("x") == 1
    assert sorted(ids) == ["x", "y"]


def test_ranks_start_at_one():
    fused = weighted_rrf(["a", "b", "c"], [], k=60)
    ranks = {chunk_id: lexical_rank for chunk_id, _, lexical_rank, _ in fused}
    assert ranks["a"] == 1
    assert ranks["b"] == 2
    assert ranks["c"] == 3


def test_deterministic_tie_break_by_chunk_id():
    # Both "a" and "b" tie in score (only lexical branch, adjacent ranks would
    # differ, so force an exact tie via two disjoint single-branch hits with
    # matching rank position across separate calls is not comparable; instead
    # verify equal-score chunks sort by chunk_id ascending).
    fused = weighted_rrf(["z"], ["a"], k=60)  # both rank 1 in their own branch -> equal score
    ids = [chunk_id for chunk_id, *_ in fused]
    assert ids == sorted(ids)
    assert ids == ["a", "z"]
