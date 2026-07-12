from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from email_thread_rag.app.schemas import ChunkRecord, RetrievalHit
from email_thread_rag.rag.utils import tokenize


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    minimum = min(values)
    if maximum == minimum:
        return [1.0 if maximum > 0 else 0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


class BM25Index:
    def __init__(self, chunks: list[ChunkRecord]):
        self.chunks = chunks
        self.corpus_tokens = [tokenize(chunk.text.lower()) for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"chunks": [chunk.model_dump(mode="json") for chunk in self.chunks]}, handle)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        chunks = [ChunkRecord.model_validate(chunk) for chunk in payload["chunks"]]
        return cls(chunks)

    def search(self, query: str, *, top_k: int, thread_id: str | None = None) -> list[RetrievalHit]:
        if self.bm25 is None:
            return []
        query_tokens = tokenize(query.lower())
        scores = list(self.bm25.get_scores(query_tokens))
        indexed = list(enumerate(scores))
        if thread_id is not None:
            indexed = [(idx, score) for idx, score in indexed if self.chunks[idx].thread_id == thread_id]
        indexed.sort(key=lambda item: item[1], reverse=True)
        indexed = indexed[:top_k]
        normalized = _normalize_scores([score for _, score in indexed])
        hits: list[RetrievalHit] = []
        for rank, ((index, raw_score), norm_score) in enumerate(zip(indexed, normalized), start=1):
            hit = RetrievalHit(chunk=self.chunks[index], retrieval_rank=rank, source_lists=["bm25"])
            hit.metrics.bm25_score_raw = float(raw_score)
            hit.metrics.bm25_score_norm = float(norm_score)
            hits.append(hit)
        return hits
