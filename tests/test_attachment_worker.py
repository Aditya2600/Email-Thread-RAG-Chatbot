"""Stage-8 extraction worker, exercised with a fake Gmail client, a fake OCR
backend, an in-memory queue, and a fake repository -- no Postgres, no network,
no real OCR engine.
"""

from __future__ import annotations

from datetime import datetime, timezone

import fitz
import pytest
from PIL import Image, ImageDraw

from email_thread_rag.config import Settings
from email_thread_rag.gmail.fakes import FakeGmailClient
from email_thread_rag.rag.attachments.models import AttachmentMeta
from email_thread_rag.rag.attachments.store import InMemoryAttachmentJobStore
from email_thread_rag.rag.attachments.worker import AttachmentExtractionWorker
from email_thread_rag.rag.vector_index import HashingEncoder

MESSAGE_ID = "<m1@x>"
GMAIL_ATT_ID = "att-abc"
PARENT = {"sender": "bob@corp.com", "subject": "Budget", "sent_at": datetime(2024, 1, 7, tzinfo=timezone.utc)}


class FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class FakeConn:
    def __init__(self, parent_row):
        self.parent_row = parent_row

    def execute(self, sql, params=None):
        return FakeCursor(self.parent_row)


class FakeRepository:
    def __init__(self, parent_row=PARENT):
        self.conn = FakeConn(parent_row)
        self.embedding_dim = 384
        self.replaced: list = []

    def replace_attachment_chunks(self, message_id, attachment_id, embedded, *, tenant_id, mailbox_id):
        self.replaced.append((message_id, attachment_id, embedded))
        return len(embedded)


class FakeOCRBackend:
    def __init__(self, text):
        self.text = text

    def image_to_text(self, png_bytes):
        return self.text


def _text_pdf(body="Approved amount is $1200 for Acme."):
    document = fitz.open()
    page = document.new_page(width=400, height=400)
    page.insert_text((40, 60), body, fontsize=14)
    return document.tobytes()


def _image_only_pdf(text="Scanned amount $2222"):
    import io

    image = Image.new("RGB", (500, 200), color="white")
    ImageDraw.Draw(image).text((20, 80), text, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    document = fitz.open()
    page = document.new_page(width=500, height=200)
    page.insert_image(page.rect, stream=buffer.getvalue())
    return document.tobytes()


def _setup(pdf_bytes, *, ocr_backend=None, settings=None, meta_size=None, parent=PARENT):
    settings = settings or Settings()
    store = InMemoryAttachmentJobStore()
    client = FakeGmailClient()
    if pdf_bytes is not None:
        client.attachments[(MESSAGE_ID, GMAIL_ATT_ID)] = pdf_bytes
    repo = FakeRepository(parent)
    store.enqueue(
        AttachmentMeta(gmail_attachment_id=GMAIL_ATT_ID, filename="budget.pdf",
                       media_type="application/pdf", byte_size=meta_size or (len(pdf_bytes) if pdf_bytes else 0)),
        message_db_id=1, message_id=MESSAGE_ID, thread_id="thread-1",
        tenant_id="acme", mailbox_id="inbox",
    )
    worker = AttachmentExtractionWorker(
        store, client, repo, encoder=HashingEncoder(dim=384), settings=settings, ocr_backend=ocr_backend,
    )
    return worker, store, client, repo


def _only_attachment(store):
    return next(iter(store.attachments.values()))


def test_native_pdf_produces_page_chunks_and_marks_done():
    worker, store, client, repo = _setup(_text_pdf())
    assert worker.run_once() is True
    att = _only_attachment(store)
    assert att.extraction_status == "done"
    assert att.extraction_method == "native_pdf"
    assert att.content_hash  # sha256 of the bytes recorded
    # One replace call with real page chunks citing the attachment/page.
    message_id, attachment_id, embedded = repo.replaced[0]
    assert attachment_id == GMAIL_ATT_ID
    chunk = embedded[0].chunk
    assert chunk.kind == "attachment"
    assert chunk.page_no == 1
    assert "1200" in chunk.text
    assert chunk.metadata["extraction_method"] == "native_pdf"
    assert ("get_attachment", (MESSAGE_ID, GMAIL_ATT_ID)) in client.calls


def test_scanned_ocr_labeled_and_chunks_flagged():
    worker, store, client, repo = _setup(_image_only_pdf(), ocr_backend=FakeOCRBackend("Scanned amount $2222"))
    worker.run_once()
    chunk = repo.replaced[0][2][0].chunk
    assert chunk.ocr_used is True
    assert chunk.metadata["extraction_method"] == "ocr"
    assert "$2222" in chunk.text


def test_ocr_disabled_scanned_page_yields_no_chunk_and_no_invented_text():
    worker, store, client, repo = _setup(_image_only_pdf(), ocr_backend=None)
    worker.run_once()
    att = _only_attachment(store)
    assert att.extraction_status == "done"
    # No usable native text and no OCR -> the page produced no chunk at all.
    assert repo.replaced[0][2] == []


def test_encrypted_pdf_is_terminal_without_retry(tmp_path):
    document = fitz.open()
    document.new_page().insert_text((40, 60), "secret", fontsize=14)
    path = tmp_path / "enc.pdf"
    document.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    worker, store, client, repo = _setup(path.read_bytes())
    worker.run_once()
    att = _only_attachment(store)
    assert att.extraction_status == "failed"
    assert att.extraction_error == "encrypted"
    job = store.get_job(1)
    assert job.status == "failed"  # deterministic: no retry


def test_malformed_pdf_is_terminal():
    worker, store, client, repo = _setup(b"%PDF-1.4 garbage", meta_size=16)
    worker.run_once()
    att = _only_attachment(store)
    assert att.extraction_status == "failed"
    assert att.extraction_error == "malformed"


def test_fetch_failure_is_transient_and_retries():
    worker, store, client, repo = _setup(_text_pdf())
    client.fail_get_attachment_ids.add(GMAIL_ATT_ID)
    worker.run_once()
    job = store.get_job(1)
    assert job.status == "pending"  # transient: back to the queue
    att = _only_attachment(store)
    assert att.extraction_status == "pending"  # not marked terminal


def test_missing_attachment_bytes_are_terminal_not_found():
    worker, store, client, repo = _setup(None)  # no scripted bytes -> get_attachment returns None
    worker.run_once()
    att = _only_attachment(store)
    assert att.extraction_status == "failed"
    assert att.extraction_error == "not_found"


def test_reprocess_is_idempotent_for_unchanged_attachment():
    worker, store, client, repo = _setup(_text_pdf())
    assert worker.run_once() is True
    # Re-enqueue the exact same attachment: same input hash, no new job.
    dup = store.enqueue(
        AttachmentMeta(gmail_attachment_id=GMAIL_ATT_ID, filename="budget.pdf",
                       media_type="application/pdf", byte_size=_only_attachment(store).byte_size),
        message_db_id=1, message_id=MESSAGE_ID, thread_id="thread-1",
        tenant_id="acme", mailbox_id="inbox",
    )
    assert dup is None
    assert worker.run_once() is False  # nothing new to do
