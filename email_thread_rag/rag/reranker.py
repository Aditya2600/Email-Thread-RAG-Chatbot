from __future__ import annotations

import math
from typing import Protocol

from email_thread_rag.app.schemas import RetrievalHit
from email_thread_rag.config import Settings
from email_thread_rag.rag.utils import tokenize


class RerankScorer(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]:
        ...


class OverlapRerankScorer:
    def score(self, query: str, passages: list[str]) -> list[float]:
        query_tokens = set(tokenize(query.lower()))
        scores: list[float] = []
        for passage in passages:
            passage_tokens = set(tokenize(passage.lower()))
            if not query_tokens or not passage_tokens:
                scores.append(0.0)
                continue
            overlap = len(query_tokens & passage_tokens)
            scores.append(overlap / max(len(query_tokens), 1))
        return scores


class CrossEncoderScorer:
    def __init__(self, settings: Settings, model_name: str | None = None):
        self.settings = settings
        self.model_name = model_name or settings.reranker_model_name
        self._model = None
        self.fallback = OverlapRerankScorer()

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        try:
            model = self._load_model()
            pairs = [(query, passage) for passage in passages]
            return [float(value) for value in model.predict(pairs, show_progress_bar=False)]
        except Exception:
            return self.fallback.score(query, passages)


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    minimum = min(values)
    if math.isclose(maximum, minimum):
        return [1.0 if maximum > 0 else 0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


class CrossEncoderReranker:
    def __init__(self, settings: Settings, scorer: RerankScorer | None = None):
        self.settings = settings
        self.scorer = scorer or CrossEncoderScorer(settings)

    def rerank(self, query: str, hits: list[RetrievalHit], *, top_k: int) -> list[RetrievalHit]:
        if not hits:
            return []
        scores = self.scorer.score(query, [hit.chunk.text for hit in hits])
        normalized = _normalize_scores(scores)
        reranked = [hit.model_copy(deep=True) for hit in hits]
        for hit, raw_score, norm_score in zip(reranked, scores, normalized):
            hit.metrics.rerank_score_raw = float(raw_score)
            hit.metrics.rerank_score_norm = float(norm_score)
            hit.metrics.chunk_support_score = float(
                0.5 * hit.metrics.rerank_score_norm
                + 0.3 * hit.metrics.dense_score_norm
                + 0.2 * hit.metrics.bm25_score_norm
            )
        reranked.sort(key=lambda item: item.metrics.rerank_score_norm, reverse=True)
        reranked = reranked[:top_k]
        for rank, hit in enumerate(reranked, start=1):
            hit.rerank_rank = rank
        return reranked

