from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from email_thread_rag.app.schemas import ChunkRecord, EmailRecord
from email_thread_rag.config import Settings
from email_thread_rag.rag.paradedb.repository import EmbeddedChunk, ParadeDBRepository
from email_thread_rag.rag.paradedb.retrieval import (
    DenseRetriever,
    HybridRetriever,
    LexicalRetriever,
    RetrievalFilters,
)
from email_thread_rag.rag.vector_index import HashingEncoder

pytestmark = pytest.mark.integration

ENCODER = HashingEncoder(dim=768)


def _email(message_id: str, thread_id: str, authored_text: str) -> EmailRecord:
    return EmailRecord(
        doc_id=message_id.strip("<>"),
        message_id=message_id,
        thread_id=thread_id,
        date=datetime(2026, 5, 3, tzinfo=timezone.utc),
        sender="bob@acme.com",
        to=["sarah@acme.com"],
        subject="Q3 Budget",
        body_text=authored_text,
        authored_text=authored_text,
        source_path="/tmp/x.eml",
        source_type="fixture",
    )


def _chunk(chunk_id: str, message_id: str, thread_id: str, text: str, embed_text: str | None = None) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=message_id.strip("<>"),
        thread_id=thread_id,
        message_id=message_id,
        kind="email",
        date=datetime(2026, 5, 3, tzinfo=timezone.utc),
        text=text,
        embed_text=embed_text or text,
        token_count=len(text.split()),
        source_path="/tmp/x.eml",
        source_type="fixture",
    )


def _repo(conn) -> ParadeDBRepository:
    return ParadeDBRepository(conn, embedding_dim=768)


@pytest.fixture
def smoke_corpus(db_conn):
    """Chunk A (lexical), B (dense-close, different wording), C (other tenant,
    must never leak), D (null embedding, exact BM25 match) -- per the Stage-2
    offline smoke corpus spec.
    """
    repo = _repo(db_conn)

    query = "budget authorization"
    query_embedding = ENCODER.encode([query])[0]
    # Chunk B: worded completely differently from the query, but its stored
    # vector is deliberately crafted close to the query embedding -- this is
    # what lets dense retrieval surface it despite zero lexical overlap.
    close_vector = 0.98 * query_embedding + 0.02 * ENCODER.encode(["unrelated filler noise token"])[0]
    close_vector = close_vector / np.linalg.norm(close_vector)

    email_a = _email("<msg-a@acme.com>", "thread-1", "Project Atlas rollout is approved for Q3.")
    chunk_a = _chunk("chunk-a", "<msg-a@acme.com>", "thread-1", "Project Atlas rollout is approved for Q3.")

    email_b = _email("<msg-b@acme.com>", "thread-1", "Fiscal spending clearance for the department is finalized.")
    chunk_b = _chunk(
        "chunk-b", "<msg-b@acme.com>", "thread-1", "Fiscal spending clearance for the department is finalized."
    )

    email_c = _email("<msg-c@other.com>", "thread-2", "budget authorization approved for $500,000, Project Atlas final sign-off")
    chunk_c = _chunk(
        "chunk-c",
        "<msg-c@other.com>",
        "thread-2",
        "budget authorization approved for $500,000, Project Atlas final sign-off",
    )

    email_d = _email("<msg-d@acme.com>", "thread-3", "Q3 budget authorization number BUD-2026-0099 approved.")
    chunk_d = _chunk("chunk-d", "<msg-d@acme.com>", "thread-3", "Q3 budget authorization number BUD-2026-0099 approved.")

    msg_a_id = repo.upsert_message(email_a, tenant_id="acme", mailbox_id="inbox")
    repo.upsert_chunks(
        msg_a_id,
        [EmbeddedChunk(chunk_a, embedding=ENCODER.encode([chunk_a.embed_text])[0].tolist())],
        tenant_id="acme",
        mailbox_id="inbox",
    )

    msg_b_id = repo.upsert_message(email_b, tenant_id="acme", mailbox_id="inbox")
    repo.upsert_chunks(
        msg_b_id,
        [EmbeddedChunk(chunk_b, embedding=close_vector.tolist())],
        tenant_id="acme",
        mailbox_id="inbox",
    )

    msg_c_id = repo.upsert_message(email_c, tenant_id="other-tenant", mailbox_id="inbox")
    repo.upsert_chunks(
        msg_c_id,
        [EmbeddedChunk(chunk_c, embedding=ENCODER.encode([chunk_c.embed_text])[0].tolist())],
        tenant_id="other-tenant",
        mailbox_id="inbox",
    )

    msg_d_id = repo.upsert_message(email_d, tenant_id="acme", mailbox_id="inbox")
    repo.upsert_chunks(msg_d_id, [EmbeddedChunk(chunk_d, embedding=None)], tenant_id="acme", mailbox_id="inbox")
    db_conn.commit()

    return {
        "query": query,
        "query_embedding": query_embedding,
        "acme": RetrievalFilters(tenant_id="acme", mailbox_id="inbox"),
        "other_tenant": RetrievalFilters(tenant_id="other-tenant", mailbox_id="inbox"),
    }


# 1. pg_search / vector extensions exist.
def test_extensions_exist(db_conn):
    rows = db_conn.execute(
        "SELECT name, installed_version FROM pg_available_extensions WHERE name IN ('pg_search','vector')"
    ).fetchall()
    versions = {row["name"]: row["installed_version"] for row in rows}
    assert versions.get("pg_search")
    assert versions.get("vector")


# 2. Migrations can be applied to a clean database (the migrated_database_url
# fixture itself is a freshly CREATE DATABASE'd instance migrated from empty).
def test_migrations_applied_to_clean_database(db_conn):
    row = db_conn.execute("SELECT to_regclass('email_chunks') AS reg").fetchone()
    assert row["reg"] == "email_chunks"


# 3. Re-running migrations is safe.
def test_migrations_are_idempotent(migrated_database_url):
    import psycopg
    from psycopg.rows import dict_row

    from email_thread_rag.rag.paradedb.repository import apply_migrations

    conn = psycopg.connect(migrated_database_url, row_factory=dict_row)
    try:
        applied = apply_migrations(conn)
        assert applied == []
    finally:
        conn.close()


# 14. BM25 and HNSW indexes exist.
def test_bm25_and_hnsw_indexes_exist(db_conn):
    rows = db_conn.execute(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'email_chunks'"
    ).fetchall()
    names = {row["indexname"] for row in rows}
    assert "email_chunks_bm25_idx" in names
    assert "email_chunks_embedding_hnsw_idx" in names


# 4 & 5. Idempotent re-ingestion + transactional stale-chunk cleanup.
def test_reingest_is_idempotent_and_reprocessing_deletes_stale_chunks(db_conn):
    repo = _repo(db_conn)
    email = _email("<msg-r@acme.com>", "thread-r", "First authored version.")
    chunk_1 = _chunk("chunk-r-0", "<msg-r@acme.com>", "thread-r", "First authored version.")
    chunk_2 = _chunk("chunk-r-1", "<msg-r@acme.com>", "thread-r", "Second authored paragraph.")

    msg_id = repo.reprocess_message(
        email, [EmbeddedChunk(chunk_1), EmbeddedChunk(chunk_2)], tenant_id="acme", mailbox_id="inbox"
    )
    count = db_conn.execute(
        "SELECT count(*) AS n FROM email_chunks WHERE message_db_id = %s", (msg_id,)
    ).fetchone()["n"]
    assert count == 2

    # Re-ingest identical chunks: no duplicates.
    repo.reprocess_message(
        email, [EmbeddedChunk(chunk_1), EmbeddedChunk(chunk_2)], tenant_id="acme", mailbox_id="inbox"
    )
    count = db_conn.execute(
        "SELECT count(*) AS n FROM email_chunks WHERE message_db_id = %s", (msg_id,)
    ).fetchone()["n"]
    assert count == 2

    # Reprocess with only chunk_1 (chunker produced fewer chunks this time):
    # chunk_2 must be deleted.
    repo.reprocess_message(email, [EmbeddedChunk(chunk_1)], tenant_id="acme", mailbox_id="inbox")
    remaining = db_conn.execute(
        "SELECT chunk_id FROM email_chunks WHERE message_db_id = %s", (msg_id,)
    ).fetchall()
    assert [row["chunk_id"] for row in remaining] == ["chunk-r-0"]
    db_conn.rollback()


# 6. BM25 retrieves an exact sender/subject/number match.
def test_bm25_exact_number_match(db_conn, smoke_corpus):
    lexical = LexicalRetriever(db_conn)
    hits = lexical.search("BUD-2026-0099", smoke_corpus["acme"], limit=5)
    assert any(hit.chunk_id == "chunk-d" for hit in hits)


# 7. Vector retrieval returns the nearest deterministic fixture vector.
def test_dense_retrieval_returns_nearest_fixture_vector(db_conn, smoke_corpus):
    dense = DenseRetriever(db_conn, embedding_dim=768)
    hits = dense.search(smoke_corpus["query_embedding"], smoke_corpus["acme"], limit=5)
    assert hits, "expected at least one embedded row"
    assert hits[0].chunk_id == "chunk-b"


# 8. Hybrid retrieval includes both lexical and semantic candidates.
def test_hybrid_includes_both_lexical_and_dense_candidates(db_conn, smoke_corpus):
    hybrid = HybridRetriever(db_conn, Settings(rag_backend="paradedb"), encoder=ENCODER)
    results = hybrid.search("Project Atlas budget authorization", smoke_corpus["acme"], top_k=5)
    chunk_ids = {hit.chunk_id for hit in results}
    assert "chunk-a" in chunk_ids  # lexical: "Project Atlas"
    assert "chunk-b" in chunk_ids  # dense: crafted close vector
    lexical_ranked = [hit for hit in results if hit.lexical_rank is not None]
    dense_ranked = [hit for hit in results if hit.dense_rank is not None]
    assert lexical_ranked and dense_ranked


# 9. Metadata/date/thread filters work.
def test_thread_filter_narrows_results(db_conn, smoke_corpus):
    lexical = LexicalRetriever(db_conn)
    scoped = RetrievalFilters(tenant_id="acme", mailbox_id="inbox", thread_id="thread-3")
    hits = lexical.search("budget authorization", scoped, limit=10)
    assert all(hit.thread_id == "thread-3" for hit in hits)
    assert any(hit.chunk_id == "chunk-d" for hit in hits)
    assert not any(hit.chunk_id == "chunk-a" for hit in hits)


# 10. Cross-tenant and cross-mailbox chunks never leak.
def test_cross_tenant_isolation(db_conn, smoke_corpus):
    lexical = LexicalRetriever(db_conn)
    hits = lexical.search("Project Atlas final sign-off", smoke_corpus["acme"], limit=10)
    assert not any(hit.chunk_id == "chunk-c" for hit in hits), "other tenant's chunk leaked into acme results"

    dense = DenseRetriever(db_conn, embedding_dim=768)
    dense_hits = dense.search(smoke_corpus["query_embedding"], smoke_corpus["acme"], limit=10)
    assert not any(hit.chunk_id == "chunk-c" for hit in dense_hits)


# 11. Null-embedding chunks remain BM25-searchable; dense excludes them.
def test_null_embedding_chunk_lexical_searchable_dense_excluded(db_conn, smoke_corpus):
    lexical = LexicalRetriever(db_conn)
    lexical_hits = lexical.search("BUD-2026-0099", smoke_corpus["acme"], limit=5)
    assert any(hit.chunk_id == "chunk-d" for hit in lexical_hits)

    dense = DenseRetriever(db_conn, embedding_dim=768)
    dense_hits = dense.search(smoke_corpus["query_embedding"], smoke_corpus["acme"], limit=10)
    assert not any(hit.chunk_id == "chunk-d" for hit in dense_hits)


# 12. Returned text and source offsets remain citation-safe.
def test_citation_safety_embed_text_searched_text_returned(db_conn):
    repo = _repo(db_conn)
    authored = "The approved amount is $120,000."
    email = _email("<msg-cite@acme.com>", "thread-cite", authored)
    chunk = _chunk(
        "chunk-cite",
        "<msg-cite@acme.com>",
        "thread-cite",
        text=authored,
        embed_text=f"Subject: Q3 Budget\n\n{authored}",
    )
    chunk.source_start = 0
    chunk.source_end = len(authored)
    repo.reprocess_message(email, [EmbeddedChunk(chunk)], tenant_id="acme", mailbox_id="inbox")
    db_conn.commit()

    lexical = LexicalRetriever(db_conn)
    filters = RetrievalFilters(tenant_id="acme", mailbox_id="inbox")
    hits = lexical.search("Q3 Budget", filters, limit=5)
    assert any(hit.chunk_id == "chunk-cite" for hit in hits)
    hit = next(hit for hit in hits if hit.chunk_id == "chunk-cite")
    assert "Subject:" not in hit.text
    assert hit.text == authored
    assert authored[chunk.source_start : chunk.source_end] == hit.text


# 13. Deterministic query ordering is preserved.
def test_deterministic_query_ordering(db_conn, smoke_corpus):
    lexical = LexicalRetriever(db_conn)
    first = [hit.chunk_id for hit in lexical.search("budget authorization", smoke_corpus["acme"], limit=10)]
    second = [hit.chunk_id for hit in lexical.search("budget authorization", smoke_corpus["acme"], limit=10)]
    assert first == second


# Offline smoke corpus narrative: query -> BM25 ranks -> dense ranks -> RRF -> final.
def test_offline_smoke_corpus_end_to_end(db_conn, smoke_corpus, capsys):
    filters = smoke_corpus["acme"]
    query = "Project Atlas budget authorization"
    lexical = LexicalRetriever(db_conn)
    dense = DenseRetriever(db_conn, embedding_dim=768)

    lexical_hits = lexical.search(query, filters, limit=10)
    dense_hits = dense.search(smoke_corpus["query_embedding"], filters, limit=10)

    from email_thread_rag.rag.fusion import weighted_rrf

    fused = weighted_rrf(
        [hit.chunk_id for hit in lexical_hits],
        [hit.chunk_id for hit in dense_hits],
        k=60,
    )

    print("BM25 candidates:", [(hit.chunk_id, hit.lexical_rank) for hit in lexical_hits])
    print("Dense candidates:", [(hit.chunk_id, hit.dense_rank) for hit in dense_hits])
    print("Fused ranking:", fused)

    fused_ids = [chunk_id for chunk_id, *_ in fused]
    assert "chunk-c" not in fused_ids  # other tenant's highly relevant chunk excluded
    assert "chunk-a" in fused_ids or "chunk-d" in fused_ids  # lexical candidates present

    hybrid = HybridRetriever(db_conn, Settings(rag_backend="paradedb"), encoder=ENCODER)
    final = hybrid.search(query, filters, top_k=3)
    assert all("Subject:" not in hit.text for hit in final)
    assert "chunk-c" not in {hit.chunk_id for hit in final}
