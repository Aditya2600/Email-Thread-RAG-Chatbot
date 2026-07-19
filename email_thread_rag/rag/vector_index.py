from __future__ import annotations

import math
import pickle
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np

from email_thread_rag.app.schemas import ChunkRecord, RetrievalHit
from email_thread_rag.config import Settings
from email_thread_rag.rag.utils import tokenize

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    faiss = None


class TextEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        ...


class HashingEncoder:
    def __init__(self, dim: int = 768):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in tokenize(text.lower()):
                # Builtin hash() is PYTHONHASHSEED-randomized per process, which
                # made this fallback embedding space (and any index persisted
                # across restarts) non-reproducible. crc32 is stable everywhere.
                bucket = zlib.crc32(token.encode("utf-8")) % self.dim
                vectors[row, bucket] += 1.0
        return _l2_normalize(vectors)


# BGE models are asymmetric: the query side wants this instruction prefix, the
# passage side must go in bare. Keyed off the model name rather than a config
# knob so a symmetric encoder (the current gte-modernbert-base, MiniLM, the
# hashing fallback) cannot silently inherit a prefix meant for another model --
# a wrong prefix is not an error anywhere, it just quietly degrades recall.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEncoder:
    def __init__(self, settings: Settings, model_name: str | None = None):
        self.settings = settings
        self.model_name = model_name or settings.embedding_model_name
        self._model = None
        # Fallback must emit the same width as the real model, or every write
        # into the vector column fails once the model is unavailable.
        self.fallback = HashingEncoder(dim=settings.embedding_dim)

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        try:
            model = self._load_model()
            embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return np.asarray(embeddings, dtype="float32")
        except Exception:
            return self.fallback.encode(texts)

    def encode_query(self, texts: list[str]) -> np.ndarray:
        # GTE (and every symmetric model) embeds queries as raw text, same path
        # as passages. Only BGE takes the prefix.
        if "bge" not in self.model_name.lower():
            return self.encode(texts)
        return self.encode([_BGE_QUERY_PREFIX + text for text in texts])


def encode_query(encoder: TextEncoder, query: str) -> np.ndarray:
    """Embed a query, using the encoder's query-side path when it has one.

    Encoders without encode_query (the hashing fallback, test doubles) are
    symmetric and just get encode().
    """
    encode = getattr(encoder, "encode_query", encoder.encode)
    return encode([query])[0]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    minimum = min(values)
    if math.isclose(maximum, minimum):
        return [1.0 if maximum > 0 else 0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


class VectorIndex:
    def __init__(self, chunks: list[ChunkRecord], embeddings: np.ndarray, encoder: TextEncoder):
        self.chunks = chunks
        self.embeddings = _l2_normalize(embeddings.astype("float32"))
        self.encoder = encoder
        self.index = None
        if faiss is not None and len(chunks) > 0:
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings)

    @classmethod
    def build(cls, chunks: list[ChunkRecord], settings: Settings, encoder: TextEncoder | None = None) -> "VectorIndex":
        effective_encoder = encoder or SentenceTransformerEncoder(settings)
        embeddings = effective_encoder.encode([(chunk.embed_text or chunk.text) for chunk in chunks]) if chunks else np.zeros((0, settings.embedding_dim), dtype="float32")
        return cls(chunks, embeddings, effective_encoder)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "chunks": [chunk.model_dump(mode="json") for chunk in self.chunks],
                    "embeddings": self.embeddings,
                },
                handle,
            )

    @classmethod
    def load(cls, path: Path, settings: Settings, encoder: TextEncoder | None = None) -> "VectorIndex":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        chunks = [ChunkRecord.model_validate(chunk) for chunk in payload["chunks"]]
        return cls(chunks, np.asarray(payload["embeddings"], dtype="float32"), encoder or SentenceTransformerEncoder(settings))

    def _score_subset(self, query_embedding: np.ndarray, indexes: list[int], top_k: int) -> list[tuple[int, float]]:
        if not indexes:
            return []
        subset = self.embeddings[indexes]
        scores = subset @ query_embedding.reshape(-1, 1)
        flattened = scores.reshape(-1)
        pairs = [(indexes[idx], float(score)) for idx, score in enumerate(flattened)]
        pairs.sort(key=lambda item: item[1], reverse=True)
        return pairs[:top_k]

    def search(self, query: str, *, top_k: int, thread_id: str | None = None) -> list[RetrievalHit]:
        if not self.chunks:
            return []
        query_embedding = encode_query(self.encoder, query)
        query_embedding = _l2_normalize(query_embedding.reshape(1, -1))[0]
        if thread_id is not None:
            candidate_indexes = [idx for idx, chunk in enumerate(self.chunks) if chunk.thread_id == thread_id]
            ranked = self._score_subset(query_embedding, candidate_indexes, top_k)
        elif self.index is not None:
            scores, indexes = self.index.search(query_embedding.reshape(1, -1), min(top_k, len(self.chunks)))
            ranked = [(int(index), float(score)) for index, score in zip(indexes[0], scores[0]) if index >= 0]
        else:
            ranked = self._score_subset(query_embedding, list(range(len(self.chunks))), top_k)

        normalized = _normalize_scores([score for _, score in ranked])
        hits: list[RetrievalHit] = []
        for rank, ((index, raw_score), norm_score) in enumerate(zip(ranked, normalized), start=1):
            hit = RetrievalHit(chunk=self.chunks[index], retrieval_rank=rank, source_lists=["dense"])
            hit.metrics.dense_score_raw = float(raw_score)
            hit.metrics.dense_score_norm = float(norm_score)
            hits.append(hit)
        return hits

