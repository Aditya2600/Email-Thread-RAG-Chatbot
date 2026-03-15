from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from email_thread_rag.app.schemas import ChunkRecord, RetrievalHit
from email_thread_rag.config import Settings
from email_thread_rag.rag.bm25_index import BM25Index
from email_thread_rag.rag.fusion import reciprocal_rank_fusion
from email_thread_rag.rag.reranker import CrossEncoderReranker
from email_thread_rag.rag.vector_index import SentenceTransformerEncoder, VectorIndex


@dataclass
class RetrievalResult:
    query: str
    bm25_hits: list[RetrievalHit]
    dense_hits: list[RetrievalHit]
    fused_hits: list[RetrievalHit]
    reranked_hits: list[RetrievalHit]


def load_chunks(path: Path) -> list[ChunkRecord]:
    if not path.exists():
        return []
    chunks: list[ChunkRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunks.append(ChunkRecord.model_validate_json(line))
    return chunks


class HybridRetriever:
    def __init__(
        self,
        chunks: list[ChunkRecord],
        settings: Settings,
        *,
        bm25_index: BM25Index | None = None,
        vector_index: VectorIndex | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.settings = settings
        self.chunks = chunks
        self.bm25_index = bm25_index or BM25Index(chunks)
        self.vector_index = vector_index or VectorIndex.build(chunks, settings, encoder=SentenceTransformerEncoder(settings))
        self.reranker = reranker or CrossEncoderReranker(settings)

    @classmethod
    def from_chunk_store(cls, settings: Settings) -> "HybridRetriever":
        chunks = load_chunks(settings.chunk_store_path)
        bm25_path = settings.index_dir / "bm25.pkl"
        vector_path = settings.index_dir / "vector.pkl"
        bm25_index = BM25Index.load(bm25_path) if bm25_path.exists() else None
        vector_index = (
            VectorIndex.load(vector_path, settings, encoder=SentenceTransformerEncoder(settings))
            if vector_path.exists()
            else None
        )
        return cls(chunks, settings, bm25_index=bm25_index, vector_index=vector_index)

    def available_threads(self) -> list[str]:
        return sorted({chunk.thread_id for chunk in self.chunks})

    def search(self, query: str, *, thread_id: str | None = None) -> RetrievalResult:
        bm25_hits = self.bm25_index.search(query, top_k=self.settings.bm25_top_k, thread_id=thread_id)
        dense_hits = self.vector_index.search(query, top_k=self.settings.dense_top_k, thread_id=thread_id)
        fused_hits = reciprocal_rank_fusion(
            bm25_hits,
            dense_hits,
            k=60,
            top_k=self.settings.fused_top_k,
        )
        reranked_hits = self.reranker.rerank(query, fused_hits, top_k=self.settings.evidence_top_k)
        return RetrievalResult(
            query=query,
            bm25_hits=bm25_hits,
            dense_hits=dense_hits,
            fused_hits=fused_hits,
            reranked_hits=reranked_hits,
        )
