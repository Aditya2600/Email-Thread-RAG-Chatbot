"""Stage-7 grounded answering against the real ParadeDB container, with a fake
provider (no model, no network).

Covers: a graph/current-routed answer cites the real underlying email chunk with
exact clean-text quotes (never a synthetic fact row), tenant isolation through
the whole answer path, safe abstention when the mailbox has no evidence, and the
engine's /ask entry point routed through the grounded flow.
"""

from __future__ import annotations

import json
import re
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
from email_thread_rag.rag.engine import RAGEngine
from email_thread_rag.rag.grounded_answer import GroundedAnswerer
from email_thread_rag.rag.paradedb.ingest import persist_corpus_to_paradedb
from email_thread_rag.rag.paradedb.retrieval import ParadeDBEngineRetriever
from email_thread_rag.rag.reranker import CrossEncoderReranker, OverlapRerankScorer
from email_thread_rag.rag.vector_index import HashingEncoder

pytestmark = pytest.mark.integration

ENCODER = HashingEncoder(dim=768)
TENANT = "acme"
MAILBOX = "inbox"
MODEL = "fake-graph-model"
BODY = "Final budget attached. The approved amount is $1200 for Acme Supplies."


class EchoAnswerProvider:
    """A faithful fake: cites the first evidence chunk with a verbatim quote it
    copies out of that chunk's own text. Proves the pipeline end-to-end without a
    real model. It can only ever cite what retrieval actually placed in the pack."""

    model_id = "fake-answer-model"

    def generate(self, messages):
        user = messages[-1]["content"]
        match = re.search(r'<evidence id="([^"]+)">\n(.*?)\n</evidence>', user, re.DOTALL)
        if not match:
            return json.dumps(
                {"answer": "no evidence", "claims": [], "is_relevant": False,
                 "is_supported": False, "is_useful": False, "needs_more_evidence": True}
            )
        chunk_id, text = match.group(1), match.group(2)
        quote = " ".join(text.split()[:4])  # verbatim leading span of the clean text
        return json.dumps(
            {
                "answer": f"According to the email: {quote}",
                "claims": [{"text": quote, "citations": [{"chunk_id": chunk_id, "quote": quote}]}],
                "is_relevant": True, "is_supported": True, "is_useful": True, "needs_more_evidence": False,
            }
        )


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


def _settings(*, tenant_id=TENANT, **overrides) -> Settings:
    kwargs = dict(
        rag_backend="paradedb", tenant_id=tenant_id, mailbox_id=MAILBOX,
        graph_extraction_enabled=True, graph_base_url="http://fake.invalid/v1", graph_model=MODEL,
        graph_schema_version=SCHEMA_VERSION, graph_prompt_version=PROMPT_VERSION, embedding_dim=768,
        answer_generation_enabled=True, answer_evidence_budget=4,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


def _make_email(*, body=BODY, thread_id="thread-alpha", message_id="<msg-2@example.com>") -> EmailRecord:
    return EmailRecord(
        doc_id="msg-2", message_id=message_id, thread_id=thread_id,
        date=datetime(2024, 1, 7, tzinfo=timezone.utc), sender="bob@corp.com",
        to=["alice@corp.com"], subject="Re: Budget Review", body_text=body,
        source_path="/tmp/msg-2.json", source_type="fixture",
    )


def _persist(conn, email, *, settings, tenant_id=TENANT):
    return persist_corpus_to_paradedb(
        conn, [email], chunk_email(email), tenant_id=tenant_id, mailbox_id=MAILBOX,
        encoder=ENCODER, embedding_dim=768, settings=settings,
    )


def _retriever(conn, settings) -> ParadeDBEngineRetriever:
    return ParadeDBEngineRetriever(
        conn, settings, encoder=ENCODER,
        reranker=CrossEncoderReranker(settings, scorer=OverlapRerankScorer()),
    )


def _ingest_and_extract(conn, *, tenant_id=TENANT, body=BODY):
    settings = _settings(tenant_id=tenant_id)
    _persist(conn, _make_email(body=body), settings=settings, tenant_id=tenant_id)
    conn.execute("DELETE FROM graph_extraction_jobs WHERE tenant_id <> %s", (tenant_id,))
    GraphWorker(
        PostgresGraphStore(conn),
        FakeGraphProvider(responder=lambda ei: graph_json(
            entities=[{"name": "Acme Supplies", "type": "ORG", "evidence": "Acme Supplies"}],
            facts=[{"subject": "approved amount", "predicate": "is", "object": "$1200",
                    "evidence": "approved amount is $1200"}],
        ), model_id=MODEL),
        schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION,
    ).drain()
    return settings


def _chunk_id(conn, tenant_id=TENANT) -> str:
    return conn.execute(
        "SELECT chunk_id FROM email_chunks WHERE tenant_id = %s ORDER BY id LIMIT 1", (tenant_id,)
    ).fetchone()["chunk_id"]


# A current/latest-routed question answers by citing the real email chunk, with
# exact clean-text offsets -- never a synthesized fact string.
def test_grounded_answer_cites_real_source_chunk(autocommit_conn):
    settings = _ingest_and_extract(autocommit_conn)
    chunk_id = _chunk_id(autocommit_conn)
    answerer = GroundedAnswerer(_retriever(autocommit_conn, settings), EchoAnswerProvider(), settings)

    result = answerer.answer("what is the current approved amount?", thread_id=None)

    assert result.status == "answered"
    assert result.citations
    for citation in result.citations:
        assert citation.chunk_id == chunk_id
        assert citation.quote in BODY
        assert BODY[citation.quote_start : citation.quote_end] == citation.quote
        assert "Subject:" not in citation.quote


# Another tenant's evidence never surfaces through the answer path, even for the
# same question -- the provider can only cite what this tenant's retrieval packed.
def test_answer_path_is_tenant_isolated(autocommit_conn):
    _ingest_and_extract(autocommit_conn, tenant_id="acme")
    globex_settings = _settings(tenant_id="globex")
    _persist(
        autocommit_conn,
        _make_email(body="Globex says the approved amount is $55 for Globex Inc."),
        settings=globex_settings, tenant_id="globex",
    )

    answerer = GroundedAnswerer(_retriever(autocommit_conn, globex_settings), EchoAnswerProvider(), globex_settings)
    result = answerer.answer("what is the current approved amount?", thread_id=None)

    assert "$1200" not in result.answer  # acme's amount never leaks
    for citation in result.citations:
        assert "$1200" not in citation.quote
        assert "Acme Supplies" not in citation.quote


def test_answer_abstains_when_mailbox_has_no_evidence(autocommit_conn):
    empty_settings = _settings(tenant_id="empty-tenant")
    answerer = GroundedAnswerer(_retriever(autocommit_conn, empty_settings), EchoAnswerProvider(), empty_settings)

    result = answerer.answer("what is the current approved amount?", thread_id=None)
    assert result.status == "abstained"
    assert result.abstain_reason == "no_evidence"
    assert result.citations == []


def test_engine_ask_routes_through_the_grounded_answer_path(autocommit_conn):
    settings = _ingest_and_extract(autocommit_conn)
    retriever = _retriever(autocommit_conn, settings)
    engine = RAGEngine(
        settings, retriever=retriever,
        grounded_answerer=GroundedAnswerer(retriever, EchoAnswerProvider(), settings),
    )
    session = engine.session_store.start_session("thread-alpha")

    outcome = engine.ask(session.session_id, "what is the approved amount?", search_outside_thread=False)

    assert outcome.response.answer_status == "answered"
    assert outcome.response.citations
    for citation in outcome.response.citations:
        assert "Subject:" not in citation.clause_text  # embed_text headers never reach a citation
