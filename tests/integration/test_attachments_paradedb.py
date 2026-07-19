"""Stage-8 PDF attachment extraction against the real ParadeDB container, with a
fake Gmail client (scripted bytes) and a fake OCR backend -- no model, no network.

Covers: the worker persists page chunks into email_chunks; retrieval finds them
through embed_text but returns the clean page text; a Stage-7 grounded answer
cites the attachment page with an exact quote; tenant isolation; and the Postgres
queue's idempotency + stale replacement.
"""

from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone

import fitz
import psycopg
import pytest
from psycopg.rows import dict_row

from email_thread_rag.app.schemas import EmailRecord
from email_thread_rag.config import Settings
from email_thread_rag.rag.attachments.models import AttachmentMeta
from email_thread_rag.rag.attachments.repository import PostgresAttachmentJobStore
from email_thread_rag.rag.attachments.worker import AttachmentExtractionWorker
from email_thread_rag.rag.chunking import chunk_email
from email_thread_rag.rag.grounded_answer import GroundedAnswerer
from email_thread_rag.rag.paradedb.ingest import persist_corpus_to_paradedb
from email_thread_rag.rag.paradedb.repository import ParadeDBRepository
from email_thread_rag.rag.paradedb.retrieval import ParadeDBEngineRetriever
from email_thread_rag.rag.reranker import CrossEncoderReranker, OverlapRerankScorer
from email_thread_rag.rag.vector_index import HashingEncoder

pytestmark = pytest.mark.integration

ENCODER = HashingEncoder(dim=384)
TENANT = "acme"
MAILBOX = "inbox"
MESSAGE_ID = "<msg-att@example.com>"
GMAIL_ATT_ID = "att-pdf-1"
PAGE_TEXT = "The approved amount is $1200 for Acme Supplies."


class EchoAnswerProvider:
    model_id = "fake-answer-model"

    def generate(self, messages):
        user = messages[-1]["content"]
        match = re.search(r'<evidence id="([^"]+)">\n(.*?)\n</evidence>', user, re.DOTALL)
        if not match:
            return json.dumps({"answer": "no evidence", "claims": [], "is_relevant": False,
                               "is_supported": False, "is_useful": False, "needs_more_evidence": True})
        chunk_id, text = match.group(1), match.group(2)
        quote = " ".join(text.split()[:5])
        return json.dumps({
            "answer": f"According to the attachment: {quote}",
            "claims": [{"text": quote, "citations": [{"chunk_id": chunk_id, "quote": quote}]}],
            "is_relevant": True, "is_supported": True, "is_useful": True, "needs_more_evidence": False,
        })


@pytest.fixture
def autocommit_conn(migrated_database_url):
    conn = psycopg.connect(migrated_database_url, row_factory=dict_row, autocommit=True)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def clean_tables(autocommit_conn):
    autocommit_conn.execute(
        "TRUNCATE attachment_extraction_jobs, email_attachments, email_chunks, email_messages "
        "RESTART IDENTITY CASCADE"
    )
    yield


def _settings(*, tenant_id=TENANT, **overrides) -> Settings:
    kwargs = dict(
        rag_backend="paradedb", tenant_id=tenant_id, mailbox_id=MAILBOX, embedding_dim=384,
        attachment_extraction_enabled=True, answer_generation_enabled=True, answer_evidence_budget=4,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


def _text_pdf(body=PAGE_TEXT) -> bytes:
    document = fitz.open()
    document.new_page(width=420, height=420).insert_text((40, 60), body, fontsize=13)
    return document.tobytes()


def _image_only_pdf(text) -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (520, 200), color="white")
    ImageDraw.Draw(image).text((20, 80), text, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    document = fitz.open()
    page = document.new_page(width=520, height=200)
    page.insert_image(page.rect, stream=buffer.getvalue())
    return document.tobytes()


class FakeGmailAttachments:
    def __init__(self, blobs):
        self.blobs = blobs

    def get_attachment(self, *, message_id, attachment_id):
        return self.blobs.get((message_id, attachment_id))


def _persist_parent(conn, settings, *, tenant_id=TENANT, message_id=MESSAGE_ID, thread_id="thread-att"):
    email = EmailRecord(
        doc_id="msg-att", message_id=message_id, thread_id=thread_id,
        date=datetime(2024, 1, 7, tzinfo=timezone.utc), sender="bob@corp.com", to=["alice@corp.com"],
        subject="Budget with attachment", body_text="See attached budget.",
        source_path="/tmp/msg-att.json", source_type="gmail",
    )
    persist_corpus_to_paradedb(
        conn, [email], chunk_email(email), tenant_id=tenant_id, mailbox_id=MAILBOX,
        encoder=ENCODER, embedding_dim=384, settings=settings,
    )
    return email


def _extract(conn, settings, pdf_bytes, *, tenant_id=TENANT, message_id=MESSAGE_ID, thread_id="thread-att",
             ocr_backend=None, byte_size=None):
    store = PostgresAttachmentJobStore(conn)
    store.enqueue(
        AttachmentMeta(gmail_attachment_id=GMAIL_ATT_ID, filename="budget.pdf",
                       media_type="application/pdf", byte_size=byte_size or len(pdf_bytes)),
        message_id=message_id, thread_id=thread_id, tenant_id=tenant_id, mailbox_id=MAILBOX,
    )
    worker = AttachmentExtractionWorker(
        store,
        FakeGmailAttachments({(message_id, GMAIL_ATT_ID): pdf_bytes}),
        ParadeDBRepository(conn, embedding_dim=384),
        encoder=ENCODER, settings=settings, ocr_backend=ocr_backend,
    )
    processed = worker.drain()
    return store, processed


def _retriever(conn, settings) -> ParadeDBEngineRetriever:
    return ParadeDBEngineRetriever(
        conn, settings, encoder=ENCODER,
        reranker=CrossEncoderReranker(settings, scorer=OverlapRerankScorer()),
    )


def test_worker_persists_native_pdf_page_chunks(autocommit_conn):
    settings = _settings()
    _persist_parent(autocommit_conn, settings)
    store, processed = _extract(autocommit_conn, settings, _text_pdf())
    assert processed == 1

    rows = autocommit_conn.execute(
        "SELECT chunk_id, text, embed_text, metadata FROM email_chunks "
        "WHERE tenant_id = %s AND chunk_kind = 'attachment' ORDER BY chunk_id", (TENANT,)
    ).fetchall()
    assert rows, "attachment page chunk was not persisted"
    row = rows[0]
    assert "$1200" in row["text"]              # clean page text
    assert "Subject:" not in row["text"]        # no headers in citable text
    assert "Page: 1" in row["embed_text"]       # page marker only in embed_text
    assert row["metadata"]["extraction_method"] == "native_pdf"

    att = autocommit_conn.execute(
        "SELECT extraction_status, extraction_method, content_hash FROM email_attachments "
        "WHERE tenant_id = %s", (TENANT,)
    ).fetchone()
    assert att["extraction_status"] == "done"
    assert att["extraction_method"] == "native_pdf"
    assert att["content_hash"]


def test_retrieval_returns_clean_page_text_but_matches_on_embed_text(autocommit_conn):
    settings = _settings()
    _persist_parent(autocommit_conn, settings)
    _extract(autocommit_conn, settings, _text_pdf())

    result = _retriever(autocommit_conn, settings).search("approved amount Acme", thread_id=None)
    attachment_hits = [h for h in result.reranked_hits if h.chunk.kind == "attachment"]
    assert attachment_hits, "attachment chunk was not retrievable"
    hit = attachment_hits[0]
    assert hit.chunk.text == PAGE_TEXT           # clean page text returned
    assert hit.chunk.page_no == 1
    assert "Page: 1" in (hit.chunk.embed_text or "")


def test_grounded_answer_cites_attachment_page_and_exact_quote(autocommit_conn):
    settings = _settings()
    _persist_parent(autocommit_conn, settings)
    _extract(autocommit_conn, settings, _text_pdf())
    retriever = _retriever(autocommit_conn, settings)
    answerer = GroundedAnswerer(retriever, EchoAnswerProvider(), settings)

    result = answerer.answer("what is the approved amount in the attachment?", thread_id=None)
    assert result.status == "answered"
    attachment_citations = [c for c in result.citations if c.page_no is not None]
    assert attachment_citations, "answer did not cite an attachment page"
    citation = attachment_citations[0]
    assert citation.page_no == 1
    assert citation.attachment_name == "budget.pdf"
    assert citation.quote in PAGE_TEXT
    assert citation.extraction_method == "native_pdf"


def test_scanned_page_uses_fake_ocr_and_is_labeled(autocommit_conn):
    settings = _settings()
    _persist_parent(autocommit_conn, settings)

    class FakeOCR:
        def image_to_text(self, png_bytes):
            return "Scanned approved amount $2222"

    _extract(autocommit_conn, settings, _image_only_pdf("Scanned approved amount $2222"), ocr_backend=FakeOCR())
    row = autocommit_conn.execute(
        "SELECT text, metadata FROM email_chunks WHERE tenant_id = %s AND chunk_kind = 'attachment'", (TENANT,)
    ).fetchone()
    assert "$2222" in row["text"]
    assert row["metadata"]["extraction_method"] == "ocr"
    att = autocommit_conn.execute(
        "SELECT extraction_method FROM email_attachments WHERE tenant_id = %s", (TENANT,)
    ).fetchone()
    assert att["extraction_method"] == "ocr"


def test_attachment_path_is_tenant_isolated(autocommit_conn):
    acme = _settings(tenant_id="acme")
    _persist_parent(autocommit_conn, acme, tenant_id="acme")
    _extract(autocommit_conn, acme, _text_pdf(), tenant_id="acme")

    globex = _settings(tenant_id="globex")
    _persist_parent(autocommit_conn, globex, tenant_id="globex")
    _extract(autocommit_conn, globex, _text_pdf("Globex amount is $55 only."), tenant_id="globex")

    result = _retriever(autocommit_conn, globex).search("approved amount", thread_id=None)
    for hit in result.reranked_hits:
        assert "$1200" not in hit.chunk.text  # acme's attachment never leaks to globex


def test_idempotent_reprocess_and_stale_replacement(autocommit_conn):
    settings = _settings()
    _persist_parent(autocommit_conn, settings)
    store, _ = _extract(autocommit_conn, settings, _text_pdf(), byte_size=1000)

    def attachment_chunk_texts():
        return [
            r["text"]
            for r in autocommit_conn.execute(
                "SELECT text FROM email_chunks WHERE tenant_id = %s AND chunk_kind = 'attachment'", (TENANT,)
            ).fetchall()
        ]

    first = attachment_chunk_texts()
    assert any("$1200" in t for t in first)

    # Same attachment (same size) -> idempotent, no new job.
    dup = store.enqueue(
        AttachmentMeta(gmail_attachment_id=GMAIL_ATT_ID, filename="budget.pdf",
                       media_type="application/pdf", byte_size=1000),
        message_id=MESSAGE_ID, thread_id="thread-att", tenant_id=TENANT, mailbox_id=MAILBOX,
    )
    assert dup is None

    # Changed attachment (new bytes + size) -> fresh job; stale page chunks replaced.
    _extract(autocommit_conn, settings, _text_pdf("Revised approved amount is $9999 total."), byte_size=2000)
    after = attachment_chunk_texts()
    assert any("$9999" in t for t in after)
    assert not any("$1200" in t for t in after)  # stale chunk gone
