"""Persist canonically-chunked emails to ParadeDB (Stage 2.5 ingestion wiring).

Groups ``ChunkRecord``s by ``message_id`` (an email's own chunks and its
attachment chunks share the parent email's ``message_id``) and reprocesses
each message through the existing idempotent
``ParadeDBRepository.reprocess_message`` -- re-running ingestion is safe and
removes chunks the chunker no longer produces.
"""

from __future__ import annotations

from email_thread_rag.app.schemas import ChunkRecord, EmailRecord
from email_thread_rag.rag.paradedb.repository import EmbeddedChunk, ParadeDBRepository


def persist_corpus_to_paradedb(
    conn,
    emails: list[EmailRecord],
    chunks: list[ChunkRecord],
    *,
    tenant_id: str,
    mailbox_id: str,
    encoder,
    embedding_dim: int,
) -> dict[str, int]:
    repo = ParadeDBRepository(conn, embedding_dim=embedding_dim)
    chunks_by_message: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        chunks_by_message.setdefault(chunk.message_id, []).append(chunk)

    encoder_name = getattr(encoder, "model_name", encoder.__class__.__name__)
    persisted_messages = 0
    persisted_chunks = 0
    for email in emails:
        message_chunks = chunks_by_message.get(email.message_id, [])
        if not message_chunks:
            continue
        # Embeddings come from embed_text (header + authored text), never
        # from injected citation output -- text stays untouched by the model.
        texts = [chunk.embed_text or chunk.text for chunk in message_chunks]
        embeddings = encoder.encode(texts)
        embedded_chunks = [
            EmbeddedChunk(chunk=chunk, embedding=list(embeddings[index]), embedding_model=encoder_name)
            for index, chunk in enumerate(message_chunks)
        ]
        repo.reprocess_message(email, embedded_chunks, tenant_id=tenant_id, mailbox_id=mailbox_id)
        persisted_messages += 1
        persisted_chunks += len(message_chunks)

    return {"messages": persisted_messages, "chunks": persisted_chunks}
