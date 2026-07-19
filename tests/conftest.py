from __future__ import annotations

import socket
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from email_thread_rag.app.schemas import AttachmentPage, AttachmentRecord, EmailRecord
from email_thread_rag.config import Settings
from email_thread_rag.rag.chunking import chunk_corpus
from email_thread_rag.rag.engine import RAGEngine
from email_thread_rag.rag.reranker import CrossEncoderReranker, OverlapRerankScorer
from email_thread_rag.rag.retrieval import HybridRetriever
from email_thread_rag.rag.rewrite import RewriteResult
from email_thread_rag.rag.vector_index import HashingEncoder, VectorIndex


# Env vars a developer's local .env sets that would otherwise silently switch a
# unit test onto the real Gmail/Postgres integrations. Ordinary unit tests must
# declare the configuration they need explicitly (via Settings(...)), not inherit
# it from the ambient .env; integration tests are exempt and set DATABASE_URL
# themselves. See config.py's module-level load_dotenv: these land in os.environ
# and Settings' default_factory reads them lazily at construction, so clearing
# them here (before any Settings() is built) is what restores isolation.
_AMBIENT_INTEGRATION_ENV = (
    "DATABASE_URL",
    "RAG_BACKEND",
    "GMAIL_CLIENT_ID",
    "GMAIL_CLIENT_SECRET",
    "GMAIL_REDIRECT_URI",
    "GMAIL_PUBSUB_TOPIC",
    "GMAIL_PUBSUB_SUBSCRIPTION",
    "GMAIL_PUBSUB_AUDIENCE",
    "GMAIL_PUBSUB_SERVICE_ACCOUNT",
    "GMAIL_TOKEN_ENCRYPTION_KEY",
    "GMAIL_TOKEN_KEY_ID",
)


@pytest.fixture(autouse=True)
def isolate_ambient_integration_env(request, monkeypatch):
    """Keep a developer's .env from silently activating Gmail/DB in unit tests."""
    if request.node.get_closest_marker("integration"):
        return
    for name in _AMBIENT_INTEGRATION_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_network_in_gmail_unit_tests(request, monkeypatch):
    """Fail any Gmail or context unit test that opens a real socket.

    The Stage-3/Stage-4 tests must run against fakes only. This exists because
    it is easy to *think* a fake is injected while a default argument still
    points at the real HTTP client -- that mistake passes silently otherwise,
    and quietly calls Google (Stage 3) or a live LLM endpoint (Stage 4).
    FastAPI's TestClient uses an in-process ASGI transport and opens no socket,
    so it is unaffected. Integration tests (which need Postgres) are exempt.
    """
    if not request.module.__name__.startswith(("test_gmail", "test_context", "test_graph")):
        return

    def blocked(self, *args, **kwargs):
        raise AssertionError(
            "this test opened a network connection; Gmail unit tests must use fakes only"
        )

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


class RuleOnlyRewriter:
    def rewrite(self, user_text, session):
        target = session.memory_slots.correction_override or session.memory_slots.last_attachment or session.memory_slots.last_subject
        rewritten = user_text
        if session.memory_slots.correction_override:
            rewritten = f"approved amount in {target}"
        elif target:
            rewritten = rewritten.replace("it", target)
        return RewriteResult(query=rewritten, mode="rules", token_counts={"rewrite_prompt_tokens": 0, "rewrite_output_tokens": 0})


def make_settings(tmp_path: Path) -> Settings:
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
    return settings


@pytest.fixture
def sample_records(tmp_path: Path):
    settings = make_settings(tmp_path)
    emails = [
        EmailRecord(
            doc_id="msg-1",
            message_id="<msg-1@example.com>",
            thread_id="thread-alpha",
            date=datetime(2024, 1, 5, tzinfo=timezone.utc),
            sender="alice@corp.com",
            to=["bob@corp.com"],
            subject="Budget Review",
            body_text="Please review the draft budget. The summary email says the draft budget is $1000 for Acme Supplies.",
            attachment_ids=["msg-1-att-1"],
            source_path=str(tmp_path / "msg-1.json"),
            source_type="fixture",
        ),
        EmailRecord(
            doc_id="msg-2",
            message_id="<msg-2@example.com>",
            thread_id="thread-alpha",
            date=datetime(2024, 1, 7, tzinfo=timezone.utc),
            sender="bob@corp.com",
            to=["alice@corp.com"],
            subject="Re: Budget Review",
            body_text="Final budget attached. The summary email says the approved amount is $1200.",
            attachment_ids=["msg-2-att-1"],
            in_reply_to="<msg-1@example.com>",
            references=["<msg-1@example.com>"],
            source_path=str(tmp_path / "msg-2.json"),
            source_type="fixture",
        ),
        EmailRecord(
            doc_id="msg-3",
            message_id="<msg-3@example.com>",
            thread_id="thread-beta",
            date=datetime(2024, 2, 1, tzinfo=timezone.utc),
            sender="carol@corp.com",
            to=["ops@corp.com"],
            subject="Phoenix Invoice",
            body_text="Phoenix invoice approved by Carol Finance for $900.",
            attachment_ids=["msg-3-att-1"],
            source_path=str(tmp_path / "msg-3.json"),
            source_type="fixture",
        ),
    ]
    attachments = [
        AttachmentRecord(
            attachment_id="msg-1-att-1",
            message_id="<msg-1@example.com>",
            thread_id="thread-alpha",
            filename="budget_draft.pdf",
            media_type="application/pdf",
            source_path=str(tmp_path / "budget_draft.pdf"),
            pages=[
                AttachmentPage(
                    page_no=1,
                    text="Draft budget for Acme Supplies amount $1000 approved by Alice Manager.",
                    ocr_used=False,
                    text_density=1.0,
                    alnum_count=50,
                )
            ],
        ),
        AttachmentRecord(
            attachment_id="msg-2-att-1",
            message_id="<msg-2@example.com>",
            thread_id="thread-alpha",
            filename="budget_final.pdf",
            media_type="application/pdf",
            source_path=str(tmp_path / "budget_final.pdf"),
            pages=[
                AttachmentPage(
                    page_no=1,
                    text="Final budget for Acme Supplies amount $1500 approved by Bob Director.",
                    ocr_used=False,
                    text_density=1.0,
                    alnum_count=50,
                )
            ],
        ),
        AttachmentRecord(
            attachment_id="msg-3-att-1",
            message_id="<msg-3@example.com>",
            thread_id="thread-beta",
            filename="phoenix_invoice.pdf",
            media_type="application/pdf",
            source_path=str(tmp_path / "phoenix_invoice.pdf"),
            pages=[
                AttachmentPage(
                    page_no=1,
                    text="Phoenix invoice amount $900 approved by Carol Finance.",
                    ocr_used=False,
                    text_density=1.0,
                    alnum_count=40,
                )
            ],
        ),
    ]
    chunks = chunk_corpus(emails, attachments)
    return settings, emails, attachments, chunks


@pytest.fixture
def test_engine(sample_records):
    settings, _, _, chunks = sample_records
    vector_index = VectorIndex.build(chunks, settings, encoder=HashingEncoder())
    retriever = HybridRetriever(
        chunks,
        settings,
        vector_index=vector_index,
        reranker=CrossEncoderReranker(settings, scorer=OverlapRerankScorer()),
    )
    return RAGEngine(settings, retriever=retriever, rewriter=RuleOnlyRewriter())


@pytest.fixture
def session_id(test_engine):
    session = test_engine.session_store.start_session("thread-alpha")
    return session.session_id


@pytest.fixture
def api_client(test_engine):
    from email_thread_rag.app import main as main_module

    main_module.engine = test_engine
    main_module.settings = test_engine.settings
    client = TestClient(main_module.app)
    return client
