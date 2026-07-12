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

