"""Stage-8 Gmail sync side: PDF attachment metadata is persisted/enqueued during
sync, non-PDF parts are ignored, and attachment *bytes* are never fetched on the
sync path (that is the extraction worker's job). Sockets are guarded by conftest.
"""

from __future__ import annotations

from email_thread_rag.gmail.fakes import FakeGmailClient
from email_thread_rag.gmail.message import gmail_pdf_attachments
from email_thread_rag.gmail.sink import InMemoryChunkSink
from email_thread_rag.gmail.sync import run_full_sync


def _message_with_parts():
    return {
        "id": "g1",
        "threadId": "t1",
        "internalDate": "1704614400000",
        "payload": {
            "headers": [
                {"name": "From", "value": "bob@corp.com"},
                {"name": "Subject", "value": "Budget"},
                {"name": "Date", "value": "Sun, 07 Jan 2024 00:00:00 +0000"},
            ],
            "parts": [
                {"mimeType": "text/plain", "body": {"data": "aGVsbG8"}},  # body, not attachment
                {"filename": "budget.pdf", "mimeType": "application/pdf",
                 "body": {"attachmentId": "att-pdf", "size": 2048}},
                {"filename": "logo.png", "mimeType": "image/png",
                 "body": {"attachmentId": "att-png", "size": 64}},
                {"filename": "notes.docx",
                 "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                 "body": {"attachmentId": "att-doc", "size": 128}},
            ],
        },
    }


def test_only_pdf_parts_are_surfaced():
    metas = gmail_pdf_attachments(_message_with_parts())
    assert [m.gmail_attachment_id for m in metas] == ["att-pdf"]
    assert metas[0].filename == "budget.pdf"
    assert metas[0].media_type == "application/pdf"
    assert metas[0].byte_size == 2048


def test_sync_persists_attachment_metadata_and_never_fetches_bytes():
    client = FakeGmailClient(messages={"g1": _message_with_parts()}, profile_history_id=10)
    sink = InMemoryChunkSink()

    run_full_sync(client, sink, email_address="user@example.com")

    # PDF metadata recorded during sync...
    metas = sink.attachments_by_message["g1"]
    assert [m.gmail_attachment_id for m in metas] == ["att-pdf"]
    # ...but the bytes are never fetched here -- that is the worker's job.
    assert not any(name == "get_attachment" for name, _ in client.calls)
