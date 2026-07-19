"""Stage-6 against the real ParadeDB container: the deterministic planner and
evidence-backed graph branch, fused into the canonical ParadeDB retrieval path
and reached through the engine's retriever entry point.

Covers: graph-entity retrieval returns clean authored chunk text with exact
offsets (never synthetic fact strings), the current/latest temporal route,
hybrid fallback when the graph has no citable evidence, tenant isolation across
every branch, and that a metadata-only relation cannot become an authored-text
citation. The extraction provider is always a fake -- no model, no network.
"""

from __future__ import annotations

from datetime import datetime, timezone

import psycopg
import pytest
from psycopg.rows import dict_row

from email_thread_rag.app.schemas import EmailRecord
from email_thread_rag.config import Settings
from email_thread_rag.graph.fakes import FakeGraphProvider, graph_json
from email_thread_rag.graph.fingerprint import PROMPT_VERSION, SCHEMA_VERSION
from email_thread_rag.graph.repository import PostgresGraphStore
from email_thread_rag.graph.worker import GraphWorker
from email_thread_rag.rag.chunking import chunk_email
from email_thread_rag.rag.paradedb.ingest import persist_corpus_to_paradedb
from email_thread_rag.rag.paradedb.retrieval import ParadeDBEngineRetriever
from email_thread_rag.rag.planner import RetrievalRoute
from email_thread_rag.rag.reranker import CrossEncoderReranker, OverlapRerankScorer
from email_thread_rag.rag.vector_index import HashingEncoder

pytestmark = pytest.mark.integration

ENCODER = HashingEncoder(dim=768)
TENANT = "acme"
MAILBOX = "inbox"
MODEL = "fake-graph-model"
BODY = "Final budget attached. The approved amount is $1200 for Acme Supplies."


@pytest.fixture
def autocommit_conn(migrated_database_url):
    conn = psycopg.connect(migrated_database_url, row_factory=dict_row, autocommit=True)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def clean_tables(autocommit_conn):
    autocommit_conn.execute(
        "TRUNCATE graph_extraction_jobs, chunk_entity_mentions, relation_observations, "
        "fact_evidence, facts, graph_entities, email_chunks, email_messages RESTART IDENTITY CASCADE"
    )
    yield


def graph_settings(*, tenant_id=TENANT, **overrides) -> Settings:
    kwargs = dict(
        rag_backend="paradedb", tenant_id=tenant_id, mailbox_id=MAILBOX,
        graph_extraction_enabled=True, graph_base_url="http://fake.invalid/v1", graph_model=MODEL,
        graph_schema_version=SCHEMA_VERSION, graph_prompt_version=PROMPT_VERSION, embedding_dim=768,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


def make_email(*, message_id="<msg-2@example.com>", body=BODY, thread_id="thread-alpha") -> EmailRecord:
    return EmailRecord(
        doc_id="msg-2", message_id=message_id, thread_id=thread_id,
        date=datetime(2024, 1, 7, tzinfo=timezone.utc), sender="bob@corp.com",
        to=["alice@corp.com"], subject="Re: Budget Review", body_text=body,
        source_path="/tmp/msg-2.json", source_type="fixture",
    )


def persist(conn, email, *, settings, tenant_id=TENANT) -> dict:
    return persist_corpus_to_paradedb(
        conn, [email], chunk_email(email), tenant_id=tenant_id, mailbox_id=MAILBOX,
        encoder=ENCODER, embedding_dim=768, settings=settings,
    )


def drain(conn, responder):
    GraphWorker(
        PostgresGraphStore(conn), FakeGraphProvider(responder=responder, model_id=MODEL),
        schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION,
    ).drain()


def _retriever(conn, settings) -> ParadeDBEngineRetriever:
    return ParadeDBEngineRetriever(
        conn, settings, encoder=ENCODER,
        reranker=CrossEncoderReranker(settings, scorer=OverlapRerankScorer()),
    )


def _ingest_and_extract(conn, *, tenant_id=TENANT, body=BODY, responder=None):
    settings = graph_settings(tenant_id=tenant_id)
    persist(conn, make_email(body=body), settings=settings, tenant_id=tenant_id)
    # Only the target tenant's jobs are extracted in these tests.
    conn.execute("DELETE FROM graph_extraction_jobs WHERE tenant_id <> %s", (tenant_id,))
    drain(conn, responder or (lambda ei: graph_json(
        entities=[{"name": "Acme Supplies", "type": "ORG", "evidence": "Acme Supplies"}],
        facts=[{"subject": "approved amount", "predicate": "is", "object": "$1200",
                "evidence": "approved amount is $1200"}],
    )))
    return settings


def _chunk_id(conn, tenant_id=TENANT) -> str:
    return conn.execute(
        "SELECT chunk_id FROM email_chunks WHERE tenant_id = %s ORDER BY id LIMIT 1", (tenant_id,)
    ).fetchone()["chunk_id"]


# 2/3/4. Entity query fuses graph evidence; the hit is the clean authored chunk
# with exact source offsets -- never a synthesized fact string.
def test_entity_query_returns_clean_authored_chunk_with_exact_offsets(autocommit_conn):
    settings = _ingest_and_extract(autocommit_conn)
    chunk_id = _chunk_id(autocommit_conn)

    result = _retriever(autocommit_conn, settings).search("who is Acme Supplies?", thread_id=None)

    assert RetrievalRoute.GRAPH_ENTITY in result.plan.routes
    graph_ids = {hit.chunk.chunk_id for hit in result.graph_hits}
    assert chunk_id in graph_ids
    assert result.fallback_reason is None

    hit = next(h for h in result.graph_hits if h.chunk.chunk_id == chunk_id)
    assert hit.chunk.text == BODY  # authored body, not a graph row / fact string
    assert "Subject:" not in hit.chunk.text
    assert hit.chunk.source_start is not None
    assert hit.chunk.text[0:] == BODY
    assert BODY[hit.chunk.source_start:hit.chunk.source_end] == hit.chunk.text

    # The graph-sourced chunk survives fusion and carries branch provenance.
    fused = next(h for h in result.fused_hits if h.chunk.chunk_id == chunk_id)
    assert "graph" in fused.source_lists


# 5. Current/latest route surfaces active-fact evidence.
def test_current_temporal_route_surfaces_active_fact_evidence(autocommit_conn):
    settings = _ingest_and_extract(autocommit_conn)
    chunk_id = _chunk_id(autocommit_conn)

    result = _retriever(autocommit_conn, settings).search(
        "what is the current approved amount?", thread_id=None
    )
    assert RetrievalRoute.GRAPH_CURRENT in result.plan.routes
    assert chunk_id in {hit.chunk.chunk_id for hit in result.graph_hits}


# 7. No graph match -> explicit fallback to hybrid retrieval.
def test_no_graph_match_falls_back_to_hybrid(autocommit_conn):
    settings = _ingest_and_extract(autocommit_conn)

    result = _retriever(autocommit_conn, settings).search("status of Zephyr Holdings?", thread_id=None)
    assert result.plan.uses_graph is True  # "Zephyr Holdings" is an entity term
    assert result.graph_hits == []
    assert result.fallback_reason == "no_graph_evidence"


# 8. Tenant isolation across the graph branch: another tenant's evidence never
# surfaces, even for the same entity name.
def test_graph_branch_is_tenant_isolated(autocommit_conn):
    # Acme has the extracted entity; Globex ingests the same words but is engine-scoped out.
    _ingest_and_extract(autocommit_conn, tenant_id="acme")
    persist(autocommit_conn, make_email(body="Globex talks about Acme Supplies too."),
            settings=graph_settings(tenant_id="globex"), tenant_id="globex")

    globex_settings = graph_settings(tenant_id="globex")
    result = _retriever(autocommit_conn, globex_settings).search("who is Acme Supplies?", thread_id=None)
    # Globex has no extracted graph rows, so the graph branch is empty for it.
    assert result.graph_hits == []
    assert result.fallback_reason == "no_graph_evidence"
    # And Acme's authored body never appears in Globex's results (chunk_ids can
    # collide across tenants by message_id -- isolation is by tenant_id, so we
    # assert on the authored text, which differs).
    assert all(hit.chunk.text != BODY for hit in result.fused_hits)


# 9. A metadata-only relation (SENT/CC/REPLY_TO) helps retrieve the email but the
# citation is always the chunk's own clean authored text -- never a synthetic
# "A SENT B" string or a fabricated span.
def test_metadata_relation_yields_only_clean_chunk_text(autocommit_conn):
    # Extraction with NO text entities/facts: only the deterministic SENT edge lands.
    settings = _ingest_and_extract(autocommit_conn, responder=lambda ei: graph_json())
    # The sender is an entity term; its metadata edges point at the chunk.
    result = _retriever(autocommit_conn, settings).search('email from "bob@corp.com"', thread_id=None)
    for hit in result.graph_hits:
        assert hit.chunk.text == BODY  # authored text only
        assert "SENT" not in hit.chunk.text and "bob@corp.com SENT" not in hit.chunk.text
