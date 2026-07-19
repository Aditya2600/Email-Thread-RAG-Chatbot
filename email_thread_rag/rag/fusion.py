from __future__ import annotations

from collections import defaultdict

from email_thread_rag.app.schemas import RetrievalHit


def reciprocal_rank_fusion(
    bm25_hits: list[RetrievalHit],
    dense_hits: list[RetrievalHit],
    *,
    k: int = 60,
    top_k: int = 10,
) -> list[RetrievalHit]:
    merged: dict[str, RetrievalHit] = {}
    scores = defaultdict(float)

    for label, hits in (("bm25", bm25_hits), ("dense", dense_hits)):
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            if chunk_id not in merged:
                merged[chunk_id] = hit.model_copy(deep=True)
            else:
                existing = merged[chunk_id]
                existing.metrics.bm25_score_raw = max(existing.metrics.bm25_score_raw, hit.metrics.bm25_score_raw)
                existing.metrics.bm25_score_norm = max(existing.metrics.bm25_score_norm, hit.metrics.bm25_score_norm)
                existing.metrics.dense_score_raw = max(existing.metrics.dense_score_raw, hit.metrics.dense_score_raw)
                existing.metrics.dense_score_norm = max(existing.metrics.dense_score_norm, hit.metrics.dense_score_norm)
                existing.source_lists = sorted(set(existing.source_lists + hit.source_lists))
            scores[chunk_id] += 1.0 / (k + rank)
            if label not in merged[chunk_id].source_lists:
                merged[chunk_id].source_lists.append(label)

    fused = sorted(merged.values(), key=lambda item: scores[item.chunk.chunk_id], reverse=True)[:top_k]
    for rank, hit in enumerate(fused, start=1):
        hit.metrics.rrf_score = float(scores[hit.chunk.chunk_id])
        hit.retrieval_rank = rank
        hit.source_lists = sorted(set(hit.source_lists))
    return fused


def weighted_rrf(
    lexical_ranked_ids: list[str],
    dense_ranked_ids: list[str],
    *,
    k: int = 60,
    lexical_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> list[tuple[str, float, int | None, int | None]]:
    """Stage-2 ParadeDB fusion: weighted RRF over two ranked chunk_id lists.

    fused_score = lexical_weight / (k + lexical_rank) + dense_weight / (k + dense_rank)

    Ranks start at 1. A chunk_id present in both lists gets both terms; one
    present in only one list is still eligible with just that term. Returned
    as (chunk_id, fused_score, lexical_rank, dense_rank), sorted by fused_score
    descending with a deterministic chunk_id-ascending tie-break. Pure and
    DB-agnostic so the exact math is unit-testable without Postgres.
    """
    lexical_rank_by_id = {chunk_id: rank for rank, chunk_id in enumerate(lexical_ranked_ids, start=1)}
    dense_rank_by_id = {chunk_id: rank for rank, chunk_id in enumerate(dense_ranked_ids, start=1)}
    all_ids = set(lexical_rank_by_id) | set(dense_rank_by_id)

    fused: list[tuple[str, float, int | None, int | None]] = []
    for chunk_id in all_ids:
        lexical_rank = lexical_rank_by_id.get(chunk_id)
        dense_rank = dense_rank_by_id.get(chunk_id)
        score = 0.0
        if lexical_rank is not None:
            score += lexical_weight / (k + lexical_rank)
        if dense_rank is not None:
            score += dense_weight / (k + dense_rank)
        fused.append((chunk_id, score, lexical_rank, dense_rank))

    fused.sort(key=lambda item: (-item[1], item[0]))
    return fused


def weighted_rrf_multi(
    branches: dict[str, list[str]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float, dict[str, int | None]]]:
    """Weighted RRF over N named ranked branches -- the same math as
    ``weighted_rrf`` generalized past two lists so Stage-6 can fuse
    bm25 + dense + graph without a second fusion system.

    fused_score = sum over branches of weight[branch] / (k + rank_in_branch)

    Ranks start at 1. A chunk present in several branches gets one term per
    branch (this is how the same chunk found by two branches is deduplicated
    into a single, higher-scored entry). Returns
    ``(chunk_id, fused_score, {branch: rank_or_None})`` -- the per-branch ranks
    carry the provenance of which branch found the chunk -- sorted by
    fused_score descending with a deterministic chunk_id-ascending tie-break.
    """
    weights = weights or {}
    rank_by_branch = {
        name: {chunk_id: rank for rank, chunk_id in enumerate(ids, start=1)}
        for name, ids in branches.items()
    }
    all_ids: set[str] = set()
    for ids in branches.values():
        all_ids.update(ids)

    fused: list[tuple[str, float, dict[str, int | None]]] = []
    for chunk_id in all_ids:
        score = 0.0
        present: dict[str, int | None] = {}
        for name, ranks in rank_by_branch.items():
            rank = ranks.get(chunk_id)
            present[name] = rank
            if rank is not None:
                score += weights.get(name, 1.0) / (k + rank)
        fused.append((chunk_id, score, present))

    fused.sort(key=lambda item: (-item[1], item[0]))
    return fused

