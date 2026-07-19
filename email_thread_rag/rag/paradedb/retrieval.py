"""Narrow lexical/dense/hybrid retrieval interfaces over the ParadeDB schema.

``RetrievedChunk`` is the narrow, psycopg-independent record ``LexicalRetriever``
/``DenseRetriever``/``HybridRetriever`` hand back. ``ParadeDBEngineRetriever``
(bottom of file) is the Stage-2.5 adapter: it wraps those three around the
*same* ``RetrievalResult``/``RetrievalHit`` shape ``rag.engine.RAGEngine``
already speaks, and reuses the existing ``CrossEncoderReranker`` unchanged --
so the engine's answer/citation pipeline (coupled to the in-memory
``ChunkRecord``/``RetrievalHit`` pydantic models) doesn't need to change at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Sequence

import psycopg

from email_thread_rag.app.schemas import ChunkRecord, RetrievalHit
from email_thread_rag.config import Settings
from email_thread_rag.rag.fusion import weighted_rrf, weighted_rrf_multi
from email_thread_rag.rag.paradedb.repository import vector_literal
from email_thread_rag.rag.planner import plan_query
from email_thread_rag.rag.vector_index import encode_query


@dataclass
class RetrievalFilters:
    """Mandatory tenant/mailbox scope; optional narrowing filters."""

    tenant_id: str
    mailbox_id: str
    thread_id: str | None = None


@dataclass
class RetrievedChunk:
    chunk_id: str
    message_id: str
    thread_id: str | None
    text: str
    source_start: int | None
    source_end: int | None
    metadata: dict[str, Any] = field(default_factory=dict)
    # First-class columns needed to rebuild a canonical ChunkRecord (kind,
    # sender, subject, date) without a second round-trip to the DB.
    kind: str = "email"
    sender: str | None = None
    subject: str | None = None
    sent_at: datetime | None = None
    embed_text: str | None = None
    lexical_rank: int | None = None
    dense_rank: int | None = None
    lexical_score: float | None = None
    dense_score: float | None = None
    fused_score: float = 0.0


_ROW_COLUMNS = (
    "chunk_id, message_id, thread_id, text, embed_text, source_start, source_end, "
    "metadata, chunk_kind, sender, subject, sent_at"
)


def _row_to_chunk(row: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row["chunk_id"],
        message_id=row["message_id"],
        thread_id=row["thread_id"],
        text=row["text"],
        embed_text=row["embed_text"],
        source_start=row["source_start"],
        source_end=row["source_end"],
        metadata=row["metadata"] or {},
        kind=row["chunk_kind"],
        sender=row["sender"],
        subject=row["subject"],
        sent_at=row["sent_at"],
    )


class LexicalRetriever:
    """BM25 (pg_search) retrieval over ``embed_text``. Never returns embed_text as evidence."""

    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def search(self, query: str, filters: RetrievalFilters, limit: int) -> list[RetrievedChunk]:
        rows = self.conn.execute(
            f"""
            SELECT {_ROW_COLUMNS}, pdb.score(id) AS score
            FROM email_chunks
            WHERE embed_text ||| %(query)s
              AND tenant_id = %(tenant_id)s
              AND mailbox_id = %(mailbox_id)s
              AND (%(thread_id)s::text IS NULL OR thread_id = %(thread_id)s)
            ORDER BY score DESC, id ASC
            LIMIT %(limit)s
            """,
            {
                "query": query,
                "tenant_id": filters.tenant_id,
                "mailbox_id": filters.mailbox_id,
                "thread_id": filters.thread_id,
                "limit": limit,
            },
        ).fetchall()
        hits = []
        for rank, row in enumerate(rows, start=1):
            chunk = _row_to_chunk(row)
            chunk.lexical_rank = rank
            chunk.lexical_score = float(row["score"])
            hits.append(chunk)
        return hits


class DenseRetriever:
    """pgvector cosine retrieval. Ignores rows with a NULL embedding."""

    def __init__(self, conn: psycopg.Connection, *, embedding_dim: int = 768, use_iterative_scan: bool = True):
        self.conn = conn
        self.embedding_dim = embedding_dim
        self.use_iterative_scan = use_iterative_scan

    def search(
        self, query_embedding: Sequence[float], filters: RetrievalFilters, limit: int
    ) -> list[RetrievedChunk]:
        embedding_literal = vector_literal(query_embedding, expected_dim=self.embedding_dim)
        with self.conn.transaction():
            if self.use_iterative_scan:
                # Tenant/mailbox filters narrow an HNSW scan a lot for a small
                # tenant on a shared index; iterative scan keeps recall honest
                # instead of returning fewer than `limit` filtered candidates.
                self.conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
            rows = self.conn.execute(
                f"""
                SELECT {_ROW_COLUMNS}, embedding <=> %(embedding)s::vector AS distance
                FROM email_chunks
                WHERE embedding IS NOT NULL
                  AND tenant_id = %(tenant_id)s
                  AND mailbox_id = %(mailbox_id)s
                  AND (%(thread_id)s::text IS NULL OR thread_id = %(thread_id)s)
                ORDER BY distance ASC, id ASC
                LIMIT %(limit)s
                """,
                {
                    "embedding": embedding_literal,
                    "tenant_id": filters.tenant_id,
                    "mailbox_id": filters.mailbox_id,
                    "thread_id": filters.thread_id,
                    "limit": limit,
                },
            ).fetchall()
        hits = []
        for rank, row in enumerate(rows, start=1):
            chunk = _row_to_chunk(row)
            chunk.dense_rank = rank
            chunk.dense_score = float(row["distance"])
            hits.append(chunk)
        return hits


class HybridRetriever:
    """Weighted-RRF fusion of lexical + dense candidates. Search uses embed_text
    (and its embedding); returned evidence is always the exact ``text``."""

    def __init__(
        self,
        conn: psycopg.Connection,
        settings: Settings,
        *,
        encoder,
        lexical: LexicalRetriever | None = None,
        dense: DenseRetriever | None = None,
    ):
        self.settings = settings
        self.encoder = encoder
        self.lexical = lexical or LexicalRetriever(conn)
        self.dense = dense or DenseRetriever(conn, embedding_dim=settings.embedding_dim)

    def search(self, query: str, filters: RetrievalFilters, top_k: int) -> list[RetrievedChunk]:
        candidate_limit = max(self.settings.hybrid_candidate_limit, top_k * 4)
        lexical_hits = self.lexical.search(query, filters, candidate_limit)
        # Query embedding computed once and reused for the dense branch only.
        query_embedding = encode_query(self.encoder, query)
        dense_hits = self.dense.search(query_embedding, filters, candidate_limit)

        lexical_by_id = {hit.chunk_id: hit for hit in lexical_hits}
        dense_by_id = {hit.chunk_id: hit for hit in dense_hits}
        fused = weighted_rrf(
            [hit.chunk_id for hit in lexical_hits],
            [hit.chunk_id for hit in dense_hits],
            k=self.settings.hybrid_rrf_k,
            lexical_weight=self.settings.hybrid_lexical_weight,
            dense_weight=self.settings.hybrid_dense_weight,
        )

        results: list[RetrievedChunk] = []
        for chunk_id, fused_score, lexical_rank, dense_rank in fused[:top_k]:
            base = lexical_by_id.get(chunk_id) or dense_by_id.get(chunk_id)
            lexical_hit = lexical_by_id.get(chunk_id)
            dense_hit = dense_by_id.get(chunk_id)
            results.append(
                replace(
                    base,
                    lexical_rank=lexical_rank,
                    dense_rank=dense_rank,
                    lexical_score=lexical_hit.lexical_score if lexical_hit else None,
                    dense_score=dense_hit.dense_score if dense_hit else None,
                    fused_score=fused_score,
                )
            )
        return results


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    maximum, minimum = max(values), min(values)
    if maximum == minimum:
        return [1.0 if maximum else 0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def _to_retrieval_hit(chunk: RetrievedChunk, *, source_lists: list[str], retrieval_rank: int | None) -> RetrievalHit:
    """Rebuild a canonical ``RetrievalHit``/``ChunkRecord`` from a Postgres row.

    ``doc_id``/``source_path``/``source_type``/``token_count``/``ocr_used``/
    ``attachment_name``/``page_no`` aren't first-class ``email_chunks``
    columns (Stage 2's schema doesn't need them); ``ParadeDBRepository``
    round-trips them through the existing ``metadata`` jsonb column instead
    of a schema change. ``text`` is always the exact authored evidence;
    ``embed_text``'s headers never reach this record's citation-facing field.
    """
    metadata = dict(chunk.metadata or {})
    doc_id = metadata.pop("_doc_id", None) or chunk.message_id.strip("<>")
    source_path = metadata.pop("_source_path", None) or f"paradedb://{chunk.chunk_id}"
    source_type = metadata.pop("_source_type", None) or "manual"
    token_count = metadata.pop("_token_count", None) or len(chunk.text.split())
    ocr_used = bool(metadata.pop("_ocr_used", False))
    attachment_name = metadata.pop("_attachment_name", None)
    page_no = metadata.pop("_page_no", None)

    record = ChunkRecord(
        chunk_id=chunk.chunk_id,
        doc_id=doc_id,
        thread_id=chunk.thread_id or chunk.message_id,
        message_id=chunk.message_id,
        kind=chunk.kind,
        attachment_name=attachment_name,
        page_no=page_no,
        sender=chunk.sender,
        date=chunk.sent_at or datetime.now(timezone.utc),
        subject=chunk.subject,
        text=chunk.text,
        embed_text=chunk.embed_text or chunk.text,
        source_start=chunk.source_start,
        source_end=chunk.source_end,
        ocr_used=ocr_used,
        token_count=int(token_count),
        source_path=source_path,
        source_type=source_type,
        metadata=metadata,
    )
    return RetrievalHit(chunk=record, retrieval_rank=retrieval_rank, source_lists=list(source_lists))


class ParadeDBEngineRetriever:
    """Adapts Lexical/Dense/weighted-RRF onto the exact ``RetrievalResult`` /
    ``RetrievalHit`` shape ``rag.engine.RAGEngine`` already consumes, and
    reuses the existing ``CrossEncoderReranker`` unchanged -- so the engine's
    answer-builder/citation-validator pipeline needs no changes to run
    against ParadeDB. Scoped to one tenant/mailbox for its whole lifetime;
    there is no unscoped search method.
    """

    def __init__(self, conn: psycopg.Connection, settings: Settings, *, encoder, reranker=None):
        self.conn = conn
        self.settings = settings
        self.encoder = encoder
        self.lexical = LexicalRetriever(conn)
        self.dense = DenseRetriever(conn, embedding_dim=settings.embedding_dim)
        self._graph_store_cache = None
        if reranker is None:
            from email_thread_rag.rag.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(settings)
        self.reranker = reranker

    def available_threads(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT thread_id FROM email_chunks "
            "WHERE tenant_id = %s AND mailbox_id = %s AND thread_id IS NOT NULL",
            (self.settings.tenant_id, self.settings.mailbox_id),
        ).fetchall()
        return sorted(row["thread_id"] for row in rows)

    def _graph_store(self):
        # Lazy: memory deployments never build this, and importing here keeps
        # the graph package off the module-level import graph.
        if self._graph_store_cache is None:
            from email_thread_rag.graph.repository import PostgresGraphStore

            self._graph_store_cache = PostgresGraphStore(self.conn)
        return self._graph_store_cache

    def _load_graph_chunks(self, filters: RetrievalFilters, chunk_ids: list[str]) -> list[RetrievedChunk]:
        """Load canonical chunk rows for graph-sourced ids, re-applying the
        tenant/mailbox/thread scope (defense in depth) and preserving the graph
        branch's deterministic order. Returns real chunks, never fact strings."""
        if not chunk_ids:
            return []
        rows = self.conn.execute(
            f"""
            SELECT {_ROW_COLUMNS} FROM email_chunks
            WHERE tenant_id = %(tenant_id)s AND mailbox_id = %(mailbox_id)s
              AND chunk_id = ANY(%(ids)s)
              AND (%(thread_id)s::text IS NULL OR thread_id = %(thread_id)s)
            """,
            {
                "tenant_id": filters.tenant_id,
                "mailbox_id": filters.mailbox_id,
                "ids": list(chunk_ids),
                "thread_id": filters.thread_id,
            },
        ).fetchall()
        by_id = {row["chunk_id"]: _row_to_chunk(row) for row in rows}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def search(self, query: str, *, thread_id: str | None = None, evidence_top_k: int | None = None):
        from email_thread_rag.rag.retrieval import RetrievalResult

        filters = RetrievalFilters(
            tenant_id=self.settings.tenant_id, mailbox_id=self.settings.mailbox_id, thread_id=thread_id
        )
        candidate_limit = max(self.settings.hybrid_candidate_limit, self.settings.fused_top_k * 4)

        lexical_hits = self.lexical.search(query, filters, candidate_limit)
        query_embedding = encode_query(self.encoder, query)
        dense_hits = self.dense.search(query_embedding, filters, candidate_limit)

        lexical_by_id = {hit.chunk_id: hit for hit in lexical_hits}
        dense_by_id = {hit.chunk_id: hit for hit in dense_hits}
        lexical_norm = _normalize([hit.lexical_score for hit in lexical_hits])
        dense_norm = _normalize([hit.dense_score for hit in dense_hits])

        bm25_retrieval_hits = []
        for rank, (hit, norm) in enumerate(
            zip(lexical_hits[: self.settings.bm25_top_k], lexical_norm), start=1
        ):
            rh = _to_retrieval_hit(hit, source_lists=["bm25"], retrieval_rank=rank)
            rh.metrics.bm25_score_raw = hit.lexical_score or 0.0
            rh.metrics.bm25_score_norm = norm
            bm25_retrieval_hits.append(rh)

        dense_retrieval_hits = []
        for rank, (hit, norm) in enumerate(zip(dense_hits[: self.settings.dense_top_k], dense_norm), start=1):
            rh = _to_retrieval_hit(hit, source_lists=["dense"], retrieval_rank=rank)
            rh.metrics.dense_score_raw = hit.dense_score or 0.0
            rh.metrics.dense_score_norm = norm
            dense_retrieval_hits.append(rh)

        # Stage-6 deterministic planner + evidence-backed graph branch.
        plan = plan_query(
            query,
            tenant_id=self.settings.tenant_id,
            mailbox_id=self.settings.mailbox_id,
            thread_id=thread_id,
            settings=self.settings,
        )
        graph_chunks: list[RetrievedChunk] = []
        fallback_reason: str | None = None
        if plan.uses_graph:
            from email_thread_rag.graph.retrieval import collect_graph_chunk_ids

            graph_chunk_ids = collect_graph_chunk_ids(self._graph_store(), plan)
            graph_chunks = self._load_graph_chunks(filters, graph_chunk_ids)
            if not graph_chunks:
                fallback_reason = "no_graph_evidence"  # falls back to pure hybrid
        graph_by_id = {chunk.chunk_id: chunk for chunk in graph_chunks}
        graph_retrieval_hits = [
            _to_retrieval_hit(chunk, source_lists=["graph"], retrieval_rank=rank)
            for rank, chunk in enumerate(graph_chunks, start=1)
        ]

        branches: dict[str, list[str]] = {
            "bm25": [hit.chunk_id for hit in lexical_hits],
            "dense": [hit.chunk_id for hit in dense_hits],
        }
        if graph_chunks:
            branches["graph"] = [chunk.chunk_id for chunk in graph_chunks]
        fused = weighted_rrf_multi(
            branches,
            k=self.settings.hybrid_rrf_k,
            weights={
                "bm25": self.settings.hybrid_lexical_weight,
                "dense": self.settings.hybrid_dense_weight,
                "graph": self.settings.graph_branch_weight,
            },
        )[: self.settings.fused_top_k]

        fused_retrieval_hits = []
        for rank, (chunk_id, fused_score, present) in enumerate(fused, start=1):
            lexical_hit = lexical_by_id.get(chunk_id)
            dense_hit = dense_by_id.get(chunk_id)
            graph_chunk = graph_by_id.get(chunk_id)
            base = lexical_hit or dense_hit or graph_chunk
            source_lists = [name for name in ("bm25", "dense", "graph") if present.get(name) is not None]
            rh = _to_retrieval_hit(base, source_lists=source_lists, retrieval_rank=rank)
            rh.metrics.bm25_score_raw = lexical_hit.lexical_score if lexical_hit else 0.0
            rh.metrics.dense_score_raw = dense_hit.dense_score if dense_hit else 0.0
            rh.metrics.rrf_score = fused_score
            fused_retrieval_hits.append(rh)

        # Stage-7 grounded answering widens this on its one bounded retry; the
        # default (None) preserves existing behavior exactly.
        reranked_hits = self.reranker.rerank(
            query, fused_retrieval_hits, top_k=evidence_top_k or self.settings.evidence_top_k
        )

        return RetrievalResult(
            query=query,
            bm25_hits=bm25_retrieval_hits,
            dense_hits=dense_retrieval_hits,
            fused_hits=fused_retrieval_hits,
            reranked_hits=reranked_hits,
            graph_hits=graph_retrieval_hits,
            plan=plan,
            fallback_reason=fallback_reason,
        )
