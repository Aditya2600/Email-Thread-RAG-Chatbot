"""Where synced mail lands: the Stage-1 chunker + Stage-2.5 persistence path.

``ChunkSink`` is deliberately tiny (persist one email, delete one message) so
the sync state machine can be unit-tested against ``InMemoryChunkSink`` with no
Postgres, while production uses ``ParadeDBChunkSink`` -- which does no chunking
of its own: it calls the same ``chunk_email`` and
``ParadeDBRepository.reprocess_message`` that Stage 2.5 already uses.
"""

from __future__ import annotations

from typing import Protocol

from email_thread_rag.app.schemas import ChunkRecord, EmailRecord
from email_thread_rag.rag.chunking import chunk_email


class ChunkSink(Protocol):
    def persist(self, email: EmailRecord) -> int:
        """Chunk and store one email. Idempotent: re-persisting replaces its chunks."""

    def delete_message(self, message_id: str) -> int:
        """Remove a message's indexed chunks so retrieval can never return them."""

    def persist_attachments(self, message: dict, *, email: EmailRecord) -> int:
        """Persist PDF attachment metadata and enqueue extraction. Never blocks on
        parsing/OCR -- it only records metadata and queues work. Returns the
        number of extraction jobs enqueued (0 when disabled or no PDF parts)."""


class InMemoryChunkSink:
    """Test/demo sink. Same idempotency and deletion semantics as ParadeDB."""

    def __init__(self):
        self.chunks_by_message: dict[str, list[ChunkRecord]] = {}
        self.emails: dict[str, EmailRecord] = {}
        self.deleted: list[str] = []
        self.attachments_by_message: dict[str, list] = {}

    def persist(self, email: EmailRecord) -> int:
        chunks = chunk_email(email)
        self.chunks_by_message[email.message_id] = chunks
        self.emails[email.message_id] = email
        return len(chunks)

    def persist_attachments(self, message: dict, *, email: EmailRecord) -> int:
        from email_thread_rag.gmail.message import gmail_pdf_attachments

        metas = gmail_pdf_attachments(message)
        self.attachments_by_message[email.message_id] = metas
        return len(metas)

    def delete_message(self, message_id: str) -> int:
        removed = len(self.chunks_by_message.pop(message_id, []))
        self.emails.pop(message_id, None)
        self.deleted.append(message_id)
        return removed

    def all_chunks(self) -> list[ChunkRecord]:
        return [chunk for chunks in self.chunks_by_message.values() for chunk in chunks]


class ParadeDBChunkSink:
    """Canonical Stage-1 chunking + Stage-2.5 ParadeDB persistence, scoped to
    one tenant/mailbox. No Gmail concepts reach this layer."""

    def __init__(
        self,
        conn,
        *,
        tenant_id: str,
        mailbox_id: str,
        encoder,
        embedding_dim: int = 384,
        settings=None,
    ):
        self.conn = conn
        self.tenant_id = tenant_id
        self.mailbox_id = mailbox_id
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        # None => Stage-4 contextualization off, which is the default.
        self.settings = settings

    def persist(self, email: EmailRecord) -> int:
        from email_thread_rag.context.enqueue import enqueue_message_context
        from email_thread_rag.graph.enqueue import enqueue_message_graph
        from email_thread_rag.rag.paradedb.repository import EmbeddedChunk, ParadeDBRepository

        chunks = chunk_email(email)
        if not chunks:
            return 0
        repo = ParadeDBRepository(self.conn, embedding_dim=self.embedding_dim)
        encoder_name = getattr(self.encoder, "model_name", self.encoder.__class__.__name__)
        # Embeddings come from embed_text (headers + authored text), never from
        # `text` -- the citation-facing evidence stays untouched. Same rule as
        # Stage 2.5's persist_corpus_to_paradedb.
        embeddings = self.encoder.encode([chunk.embed_text or chunk.text for chunk in chunks])
        embedded = [
            EmbeddedChunk(chunk=chunk, embedding=list(embeddings[index]), embedding_model=encoder_name)
            for index, chunk in enumerate(chunks)
        ]
        repo.reprocess_message(email, embedded, tenant_id=self.tenant_id, mailbox_id=self.mailbox_id)
        # Queue context work; never call the provider here. Sync must not block
        # on an LLM, and a stalled model must not stall Gmail ingestion.
        enqueue_message_context(
            self.conn,
            email.message_id,
            tenant_id=self.tenant_id,
            mailbox_id=self.mailbox_id,
            settings=self.settings,
            embedding_dim=self.embedding_dim,
        )
        enqueue_message_graph(
            self.conn,
            email.message_id,
            tenant_id=self.tenant_id,
            mailbox_id=self.mailbox_id,
            settings=self.settings,
        )
        return len(chunks)

    def delete_message(self, message_id: str) -> int:
        from email_thread_rag.rag.paradedb.repository import ParadeDBRepository

        repo = ParadeDBRepository(self.conn, embedding_dim=self.embedding_dim)
        return repo.delete_message(message_id, tenant_id=self.tenant_id, mailbox_id=self.mailbox_id)

    def persist_attachments(self, message: dict, *, email: EmailRecord) -> int:
        # Enqueuing only: the extraction worker fetches bytes and parses/OCRs
        # later, off the sync path. Disabled -> nothing persisted, nothing queued.
        if self.settings is None or not getattr(self.settings, "attachment_extraction_enabled", False):
            return 0

        from email_thread_rag.gmail.message import gmail_pdf_attachments
        from email_thread_rag.rag.attachments.repository import PostgresAttachmentJobStore

        metas = gmail_pdf_attachments(message)
        if not metas:
            return 0
        store = PostgresAttachmentJobStore(self.conn)
        enqueued = 0
        for meta in metas:
            if store.enqueue(
                meta,
                message_id=email.message_id,
                thread_id=email.thread_id,
                tenant_id=self.tenant_id,
                mailbox_id=self.mailbox_id,
            ):
                enqueued += 1
        return enqueued
