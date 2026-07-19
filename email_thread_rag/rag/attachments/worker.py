"""The PDF attachment extraction worker: claim, fetch, extract, chunk, commit.

Ordering mirrors the Stage 4/5 workers:

    claim (txn) -> [no txn held] fetch bytes + parse/OCR -> replace page chunks
    (txn) -> commit (txn)

No DB transaction is open across the Gmail attachment fetch or the parse/OCR, so
a slow document can never pin a Postgres connection.

Failure handling:
  * Gmail fetch raised (network/5xx) -> transient. fail_job, retry.
  * unsupported / oversized / encrypted / malformed -> deterministic. Mark the
    attachment terminal ('failed'/'unsupported'), retire the job with no retry,
    and never let the document enter retrieval.
  * OCR disabled/unavailable on a scanned page -> the page yields no text, so it
    simply produces no chunk. Never invented.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time

from email_thread_rag.rag.attachments.pdf import extract_pdf
from email_thread_rag.rag.chunking import chunk_attachment

logger = logging.getLogger(__name__)


class AttachmentExtractionWorker:
    def __init__(
        self,
        store,
        gmail_client,
        repository,
        *,
        encoder,
        settings,
        ocr_backend=None,
        owner: str = "attachment-worker",
        max_attempts: int = 3,
    ):
        self.store = store
        self.gmail_client = gmail_client
        self.repository = repository
        self.encoder = encoder
        self.settings = settings
        self.ocr_backend = ocr_backend
        self.owner = owner
        self.max_attempts = max_attempts

    def run_once(self) -> bool:
        """Process at most one job. Returns False when the queue is empty."""
        job = self.store.claim_job(owner=self.owner)
        if job is None:
            return False

        att = self.store.load_attachment(job.attachment_db_id)
        if att is None:
            self.store.fail_job(job.id, "attachment row gone", error_rule="missing", max_attempts=0)
            return True

        try:
            data = self.gmail_client.get_attachment(
                message_id=att.message_id, attachment_id=att.gmail_attachment_id
            )
        except Exception as exc:  # noqa: BLE001 - transient Gmail/network failure
            self.store.fail_job(job.id, str(exc), error_rule="fetch_error", max_attempts=self.max_attempts)
            return True

        if data is None:
            # 404: the attachment no longer exists. Deterministic; do not retry.
            self.store.mark_attachment(att.id, status="failed", error="not_found")
            self.store.fail_job(job.id, "attachment not found", error_rule="not_found", max_attempts=0)
            return True

        result = extract_pdf(
            data,
            attachment_id=att.attachment_id,
            filename=att.filename,
            message_id=att.message_id,
            thread_id=att.thread_id,
            media_type=att.media_type,
            settings=self.settings,
            ocr_backend=self.ocr_backend,
        )
        content_hash = hashlib.sha256(data).hexdigest()

        if result.status != "done" or result.record is None:
            # unsupported / oversized / encrypted / malformed: terminal, no retry.
            # Also drop any stale page chunks from a prior good version.
            self._replace_chunks(att, [])
            self.store.mark_attachment(att.id, status=result.status, error=result.error)
            self.store.fail_job(job.id, result.error or "extraction_failed",
                                error_rule=result.error, max_attempts=0)
            return True

        parent = self._parent_message(att)
        if parent is None:
            # Parent email removed mid-flight; nothing to attach to. Retry later.
            self.store.fail_job(job.id, "parent message missing", error_rule="missing_parent",
                                max_attempts=self.max_attempts)
            return True

        chunks = chunk_attachment(
            result.record,
            message_date=parent["sent_at"],
            sender=parent["sender"],
            subject=parent["subject"],
            source_type="gmail",
        )
        self._replace_chunks(att, chunks)
        self.store.commit_extraction(job, content_hash=content_hash, method=result.method, status="done")
        # Attachment page chunks are new chunks in this message: let the existing
        # context/graph queues pick them up, exactly as email-body ingestion does.
        self._enqueue_downstream(att)
        return True

    def _parent_message(self, att):
        return self.repository.conn.execute(
            "SELECT sender, subject, sent_at FROM email_messages "
            "WHERE tenant_id = %s AND mailbox_id = %s AND message_id = %s",
            (att.tenant_id, att.mailbox_id, att.message_id),
        ).fetchone()

    def _replace_chunks(self, att, chunks) -> None:
        embedded = []
        if chunks:
            encoder_name = getattr(self.encoder, "model_name", self.encoder.__class__.__name__)
            from email_thread_rag.rag.paradedb.repository import EmbeddedChunk

            embeddings = self.encoder.encode([c.embed_text or c.text for c in chunks])
            embedded = [
                EmbeddedChunk(chunk=c, embedding=list(embeddings[i]), embedding_model=encoder_name)
                for i, c in enumerate(chunks)
            ]
        self.repository.replace_attachment_chunks(
            att.message_id, att.attachment_id, embedded,
            tenant_id=att.tenant_id, mailbox_id=att.mailbox_id,
        )

    def _enqueue_downstream(self, att) -> None:
        from email_thread_rag.context.enqueue import enqueue_message_context
        from email_thread_rag.graph.enqueue import enqueue_message_graph

        enqueue_message_context(
            self.repository.conn, att.message_id,
            tenant_id=att.tenant_id, mailbox_id=att.mailbox_id,
            settings=self.settings, embedding_dim=self.repository.embedding_dim,
        )
        enqueue_message_graph(
            self.repository.conn, att.message_id,
            tenant_id=att.tenant_id, mailbox_id=att.mailbox_id, settings=self.settings,
        )

    def drain(self, *, max_jobs: int = 1000) -> int:
        processed = 0
        while processed < max_jobs and self.run_once():
            processed += 1
        return processed


def build_production_worker(settings, *, owner: str = "attachment-worker"):
    """Wire a worker from configuration. Imports stay local so this module is
    importable without psycopg / a Gmail client installed."""
    from email_thread_rag.gmail.cipher import build_token_cipher
    from email_thread_rag.gmail.client import HttpxGmailClient
    from email_thread_rag.gmail.oauth import refresh_access_token
    from email_thread_rag.gmail.repository import PostgresSyncStore
    from email_thread_rag.rag.attachments.ocr import build_ocr_backend
    from email_thread_rag.rag.attachments.repository import PostgresAttachmentJobStore
    from email_thread_rag.rag.paradedb.repository import ParadeDBRepository, connect
    from email_thread_rag.rag.vector_index import SentenceTransformerEncoder

    conn = connect(settings.database_url, autocommit=True)
    sync_store = PostgresSyncStore(conn)
    mailbox = sync_store.get_mailbox(tenant_id=settings.tenant_id, mailbox_id=settings.mailbox_id)
    if mailbox is None or not mailbox.refresh_token_ciphertext:
        raise RuntimeError("No connected mailbox with a refresh token for attachment extraction.")
    cipher = build_token_cipher(settings)
    refresh_token = cipher.decrypt(mailbox.refresh_token_ciphertext, key_id=mailbox.token_key_id)
    client = HttpxGmailClient(
        refresh_access_token(
            refresh_token=refresh_token,
            client_id=settings.gmail_client_id,
            client_secret=settings.gmail_client_secret,
        )
    )
    return AttachmentExtractionWorker(
        PostgresAttachmentJobStore(conn),
        client,
        ParadeDBRepository(conn, embedding_dim=settings.embedding_dim),
        encoder=SentenceTransformerEncoder(settings),
        settings=settings,
        ocr_backend=build_ocr_backend(settings),
        owner=owner,
    )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - operational entrypoint
    parser = argparse.ArgumentParser(description="Process the PDF attachment extraction queue.")
    parser.add_argument("--once", action="store_true", help="Drain the queue and exit.")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--max-jobs", type=int, default=1000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from email_thread_rag.config import get_settings

    worker = build_production_worker(get_settings())
    if args.once:
        count = worker.drain(max_jobs=args.max_jobs)
        logger.info("processed %d attachment job(s)", count)
        return 0

    logger.info("attachment worker polling every %.1fs", args.poll_interval)
    while True:
        if worker.drain(max_jobs=args.max_jobs) == 0:
            time.sleep(args.poll_interval)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
