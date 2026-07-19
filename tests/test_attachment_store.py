"""Stage-8 attachment queue semantics against the in-memory store.

Idempotency by input hash, stale reprocess when the attachment changes, and the
claim/commit/fail lifecycle -- the same behaviour the Postgres store must match
(exercised for real in tests/integration/test_attachments_paradedb.py).
"""

from __future__ import annotations

from email_thread_rag.rag.attachments.models import AttachmentMeta
from email_thread_rag.rag.attachments.store import InMemoryAttachmentJobStore


def _meta(size=1000, filename="budget.pdf"):
    return AttachmentMeta(gmail_attachment_id="att-abc", filename=filename, media_type="application/pdf", byte_size=size)


def _enqueue(store, meta):
    return store.enqueue(
        meta, message_db_id=1, message_id="<m1@x>", thread_id="thread-1",
        tenant_id="acme", mailbox_id="inbox",
    )


def test_enqueue_is_idempotent_by_input_hash():
    store = InMemoryAttachmentJobStore()
    assert _enqueue(store, _meta()) is not None
    assert _enqueue(store, _meta()) is None  # same attachment, no duplicate job
    assert len(store.attachments) == 1


def test_changed_attachment_reprocesses():
    store = InMemoryAttachmentJobStore()
    assert _enqueue(store, _meta(size=1000)) is not None
    # A changed attachment (new byte size) hashes differently -> a fresh job.
    job2 = _enqueue(store, _meta(size=2048))
    assert job2 is not None
    assert len(store.attachments) == 1  # same attachment row, updated


def test_claim_commit_lifecycle():
    store = InMemoryAttachmentJobStore()
    job = _enqueue(store, _meta())
    claimed = store.claim_job(owner="w1")
    assert claimed.id == job.id and claimed.status == "running"
    assert store.claim_job(owner="w1") is None  # nothing else claimable

    assert store.commit_extraction(claimed, content_hash="deadbeef", method="native_pdf", status="done") is True
    att = store.load_attachment(claimed.attachment_db_id)
    assert att.extraction_status == "done"
    assert att.extraction_method == "native_pdf"
    assert att.content_hash == "deadbeef"
    assert store.get_job(job.id).status == "done"


def test_fail_job_retries_then_gives_up():
    store = InMemoryAttachmentJobStore()
    job = _enqueue(store, _meta())
    store.claim_job(owner="w1")
    store.fail_job(job.id, "boom", error_rule="fetch_error", max_attempts=2)
    assert store.get_job(job.id).status == "pending"  # one attempt, retry

    store.claim_job(owner="w1")  # attempts -> 2
    store.fail_job(job.id, "boom again", error_rule="fetch_error", max_attempts=2)
    assert store.get_job(job.id).status == "failed"  # spent


def test_mark_attachment_terminal():
    store = InMemoryAttachmentJobStore()
    job = _enqueue(store, _meta())
    store.mark_attachment(job.attachment_db_id, status="unsupported", error="encrypted")
    att = store.load_attachment(job.attachment_db_id)
    assert att.extraction_status == "unsupported"
    assert att.extraction_error == "encrypted"
