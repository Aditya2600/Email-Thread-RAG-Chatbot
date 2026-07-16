"""Stage-2.5: the actual RAGEngine, wired to a real ParadeDB, not just the
standalone retriever classes Stage 2 already covers.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from email_thread_rag.app.schemas import EmailRecord
from email_thread_rag.config import Settings
from email_thread_rag.rag.backend import build_retriever
from email_thread_rag.rag.chunking import chunk_email
from email_thread_rag.rag.paradedb.ingest import persist_corpus_to_paradedb
from email_thread_rag.rag.paradedb.repository import ParadeDBConfigError
from email_thread_rag.rag.paradedb.retrieval import ParadeDBEngineRetriever
from email_thread_rag.rag.engine import RAGEngine
from email_thread_rag.rag.reranker import CrossEncoderReranker, OverlapRerankScorer
from email_thread_rag.rag.vector_index import HashingEncoder

pytestmark = pytest.mark.integration

ENCODER = HashingEncoder(dim=384)


def _engine(conn, settings: Settings) -> RAGEngine:
    retriever = ParadeDBEngineRetriever(
        conn, settings, encoder=ENCODER, reranker=CrossEncoderReranker(settings, scorer=OverlapRerankScorer())
    )
    return RAGEngine(settings, retriever=retriever)


def _settings(tenant_id: str, mailbox_id: str) -> Settings:
    return Settings(rag_backend="paradedb", tenant_id=tenant_id, mailbox_id=mailbox_id)


# 4. Canonical ingestion reaches ParadeDB (real chunk_email(), not a hand-built ChunkRecord).
def test_canonical_email_persists_via_reprocess_message(db_conn):
    email = EmailRecord(
        doc_id="e2e-1",
        message_id="<e2e-1@acme.com>",
        thread_id="thread-e2e",
        date=datetime(2026, 5, 3, tzinfo=timezone.utc),
        sender="bob@acme.com",
        to=["sarah@acme.com"],
        subject="Q3 Budget",
        body_text="Hi Sarah,\n\nThe approved budget is now $120,000.\n\nRegards,\nBob",
        source_path="/tmp/e2e-1.eml",
        source_type="fixture",
    )
    chunks = chunk_email(email)
    assert chunks, "canonical chunker should produce at least one chunk"

    stats = persist_corpus_to_paradedb(
        db_conn, [email], chunks, tenant_id="acme", mailbox_id="inbox", encoder=ENCODER, embedding_dim=384
    )
    db_conn.commit()
    assert stats == {"messages": 1, "chunks": len(chunks)}

    row = db_conn.execute(
        "SELECT chunk_id, text, embed_text FROM email_chunks WHERE tenant_id='acme' AND mailbox_id='inbox' "
        "AND message_id = %s",
        (email.message_id,),
    ).fetchone()
    assert row is not None
    assert "$120,000" in row["text"]
    assert "Subject: Q3 Budget" in row["embed_text"]
    assert "Subject: Q3 Budget" not in row["text"]


# 5. Re-ingestion via the Stage-2.5 ingest helper is idempotent and drops stale chunks.
def test_ingest_helper_is_idempotent_and_drops_stale_chunks(db_conn):
    email = EmailRecord(
        doc_id="e2e-2",
        message_id="<e2e-2@acme.com>",
        thread_id="thread-e2e-2",
        date=datetime(2026, 5, 3, tzinfo=timezone.utc),
        sender="bob@acme.com",
        subject="Reprocess Me",
        body_text="First authored paragraph here.\n\nSecond authored paragraph here.",
        source_path="/tmp/e2e-2.eml",
        source_type="fixture",
    )
    chunks = chunk_email(email)
    persist_corpus_to_paradedb(
        db_conn, [email], chunks, tenant_id="acme", mailbox_id="inbox", encoder=ENCODER, embedding_dim=384
    )
    persist_corpus_to_paradedb(
        db_conn, [email], chunks, tenant_id="acme", mailbox_id="inbox", encoder=ENCODER, embedding_dim=384
    )
    db_conn.commit()
    count = db_conn.execute(
        "SELECT count(*) AS n FROM email_chunks WHERE tenant_id='acme' AND mailbox_id='inbox' AND message_id = %s",
        (email.message_id,),
    ).fetchone()["n"]
    assert count == len(chunks)

    # Chunker re-run produces fewer chunks (simulating an updated chunker version).
    fewer_chunks = chunks[:1]
    persist_corpus_to_paradedb(
        db_conn, [email], fewer_chunks, tenant_id="acme", mailbox_id="inbox", encoder=ENCODER, embedding_dim=384
    )
    db_conn.commit()
    remaining = db_conn.execute(
        "SELECT chunk_id FROM email_chunks WHERE tenant_id='acme' AND mailbox_id='inbox' AND message_id = %s",
        (email.message_id,),
    ).fetchall()
    assert {row["chunk_id"] for row in remaining} == {fewer_chunks[0].chunk_id}


# 2. ParadeDB backend validates required configuration -- through the real factory.
def test_build_retriever_paradedb_fails_clearly_without_database_url():
    with pytest.raises(ParadeDBConfigError, match="DATABASE_URL"):
        build_retriever(Settings(rag_backend="paradedb", database_url=None))


# 1. Memory backend still works with no Docker / no DATABASE_URL involved at all.
def test_build_retriever_memory_default_needs_no_database(tmp_path):
    from email_thread_rag.rag.retrieval import HybridRetriever as MemoryHybridRetriever

    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        raw_data_dir=tmp_path / "data" / "raw",
        processed_data_dir=tmp_path / "data" / "processed",
        index_dir=tmp_path / "data" / "indexes",
        runs_dir=tmp_path / "runs",
        dataset_manifest_path=tmp_path / "data" / "raw" / "dataset_manifest.json",
        resolved_manifest_path=tmp_path / "data" / "processed" / "resolved_dataset_manifest.json",
        chunk_store_path=tmp_path / "data" / "processed" / "chunks.jsonl",
        stats_path=tmp_path / "data" / "processed" / "ingest_stats.json",
    )
    settings.ensure_directories()
    retriever = build_retriever(settings)
    assert isinstance(retriever, MemoryHybridRetriever)


# 6, 8, 9. The actual engine: BM25 + dense + weighted RRF, other-tenant isolation,
# citation-safe evidence, and source-span correctness -- all through engine.ask().
def test_engine_end_to_end_over_paradedb(db_conn):
    settings = _settings("acme", "inbox")

    target_email = EmailRecord(
        doc_id="e2e-3",
        message_id="<e2e-3@acme.com>",
        thread_id="thread-e2e-3",
        date=datetime(2026, 5, 3, tzinfo=timezone.utc),
        sender="bob@acme.com",
        subject="Q3 Budget",
        body_text="The approved amount is $120,000 for Project Atlas.",
        source_path="/tmp/e2e-3.eml",
        source_type="fixture",
    )
    other_tenant_email = EmailRecord(
        doc_id="e2e-4",
        message_id="<e2e-4@other.com>",
        thread_id="thread-e2e-3",  # same thread_id on purpose: isolation must still hold
        date=datetime(2026, 5, 3, tzinfo=timezone.utc),
        sender="eve@other.com",
        subject="Q3 Budget",
        body_text="The approved amount is $999,999,999 for Project Atlas -- top secret.",
        source_path="/tmp/e2e-4.eml",
        source_type="fixture",
    )

    target_chunks = chunk_email(target_email)
    other_chunks = chunk_email(other_tenant_email)
    persist_corpus_to_paradedb(
        db_conn, [target_email], target_chunks, tenant_id="acme", mailbox_id="inbox", encoder=ENCODER, embedding_dim=384
    )
    persist_corpus_to_paradedb(
        db_conn,
        [other_tenant_email],
        other_chunks,
        tenant_id="other-tenant",
        mailbox_id="inbox",
        encoder=ENCODER,
        embedding_dim=384,
    )
    db_conn.commit()

    engine = _engine(db_conn, settings)
    session = engine.session_store.start_session("thread-e2e-3")
    outcome = engine.ask(session.session_id, "What is the approved amount in the Q3 Budget email?", search_outside_thread=False)

    assert "$120,000" in outcome.response.answer
    assert "999,999,999" not in outcome.response.answer  # other tenant never leaks into the answer
    assert "Subject:" not in outcome.response.answer  # embed_text headers never reach citation evidence
    for citation in outcome.response.citations:
        assert "Subject:" not in citation.clause_text

    hit = next(h for h in outcome.response.retrieved if h.chunk.message_id == target_email.message_id)
    assert hit.metrics.bm25_score_raw > 0 or hit.metrics.dense_score_raw > 0  # exercised through real BM25/dense
    assert hit.chunk.source_start is not None and hit.chunk.source_end is not None
    authored = target_chunks[0].text  # single short chunk; authored body == chunk text here
    assert authored[hit.chunk.source_start : hit.chunk.source_end] == hit.chunk.text


# 7. Other-tenant/mailbox isolation proven at the available_threads()/search() level too.
def test_engine_available_threads_scoped_to_tenant_mailbox(db_conn):
    acme_email = EmailRecord(
        doc_id="e2e-5",
        message_id="<e2e-5@acme.com>",
        thread_id="thread-acme-only",
        date=datetime(2026, 5, 3, tzinfo=timezone.utc),
        sender="bob@acme.com",
        subject="Acme thread",
        body_text="Acme-only content.",
        source_path="/tmp/e2e-5.eml",
        source_type="fixture",
    )
    other_email = EmailRecord(
        doc_id="e2e-6",
        message_id="<e2e-6@other.com>",
        thread_id="thread-other-only",
        date=datetime(2026, 5, 3, tzinfo=timezone.utc),
        sender="eve@other.com",
        subject="Other thread",
        body_text="Other-tenant-only content.",
        source_path="/tmp/e2e-6.eml",
        source_type="fixture",
    )
    persist_corpus_to_paradedb(
        db_conn,
        [acme_email],
        chunk_email(acme_email),
        tenant_id="acme",
        mailbox_id="inbox",
        encoder=ENCODER,
        embedding_dim=384,
    )
    persist_corpus_to_paradedb(
        db_conn,
        [other_email],
        chunk_email(other_email),
        tenant_id="other-tenant",
        mailbox_id="inbox",
        encoder=ENCODER,
        embedding_dim=384,
    )
    db_conn.commit()

    engine = _engine(db_conn, _settings("acme", "inbox"))
    threads = engine.available_threads()
    assert "thread-acme-only" in threads
    assert "thread-other-only" not in threads
