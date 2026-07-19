"""Backfill: queue contextualization for chunks that predate Stage 4.

Resumable and idempotent, by construction rather than by bookkeeping:

  * resumable -- it pages by ascending chunk id and keeps a cursor, so an
    interrupted run restarts from the last page rather than the beginning;
  * idempotent -- enqueuing collides on the job's unique fingerprint, and
    already-contextualized chunks are excluded by the scan itself. Running it
    twice queues nothing the second time.

Enqueue only. The worker does the LLM calls, so a backfill of a large mailbox
cannot be throttled by, or take down, the model endpoint.
"""

from __future__ import annotations

import argparse
import logging

from email_thread_rag.context.enqueue import context_identity

logger = logging.getLogger(__name__)


def backfill_context_jobs(
    conn,
    *,
    tenant_id: str,
    mailbox_id: str,
    settings,
    batch_size: int = 100,
    max_chunks: int | None = None,
    embedding_dim: int = 768,
) -> int:
    """Queue every not-yet-contextualized chunk. Returns jobs created."""
    from email_thread_rag.context.repository import PostgresContextJobStore

    prompt_version, model_id = context_identity(settings)
    store = PostgresContextJobStore(conn, embedding_dim=embedding_dim)

    queued = 0
    scanned = 0
    after_id = 0
    while True:
        states = store.chunks_needing_context(
            tenant_id=tenant_id, mailbox_id=mailbox_id, limit=batch_size, after_id=after_id
        )
        if not states:
            break
        for state in states:
            if store.enqueue(state, prompt_version=prompt_version, model_id=model_id):
                queued += 1
            scanned += 1
            # The cursor is what makes this resumable: it only ever moves
            # forward, so a crash costs at most the current page.
            after_id = state.chunk_db_id
            if max_chunks is not None and scanned >= max_chunks:
                return queued
    return queued


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Queue LLM contextualization for existing chunks.")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--mailbox-id", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    from email_thread_rag.config import get_settings
    from email_thread_rag.rag.paradedb.repository import connect

    settings = get_settings()
    if not settings.context_enabled:
        # Refuse rather than queue work nothing will consume.
        parser.error("CONTEXT_ENABLED is false; enable contextualization before backfilling.")

    conn = connect(settings.database_url, autocommit=True)
    queued = backfill_context_jobs(
        conn,
        tenant_id=args.tenant_id,
        mailbox_id=args.mailbox_id,
        settings=settings,
        batch_size=args.batch_size,
        max_chunks=args.max_chunks,
        embedding_dim=settings.embedding_dim,
    )
    logger.info("queued %d context job(s); run the context worker to process them", queued)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
