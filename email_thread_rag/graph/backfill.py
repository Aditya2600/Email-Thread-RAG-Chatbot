"""Backfill: queue graph extraction for chunks that predate Stage 5.

Resumable and idempotent by construction, not bookkeeping:
  * resumable -- pages by ascending chunk id and keeps a cursor, so an
    interrupted run restarts from the last page;
  * idempotent -- enqueuing collides on the job's unique fingerprint, and
    already-extracted chunks are excluded by the scan (graph_input_hash IS NULL).

Enqueue only. The worker does the LLM calls, so backfilling a large mailbox
cannot throttle or take down the model endpoint.
"""

from __future__ import annotations

import argparse
import logging

from email_thread_rag.graph.enqueue import graph_identity

logger = logging.getLogger(__name__)


def backfill_graph_jobs(
    conn, *, tenant_id, mailbox_id, settings, batch_size=100, max_chunks=None
) -> int:
    """Queue every not-yet-extracted chunk. Returns jobs created."""
    from email_thread_rag.graph.repository import PostgresGraphStore

    schema_version, prompt_version, model_id = graph_identity(settings)
    store = PostgresGraphStore(conn)

    queued = 0
    scanned = 0
    after_id = 0
    while True:
        states = store.chunks_needing_graph(
            tenant_id=tenant_id, mailbox_id=mailbox_id, limit=batch_size, after_id=after_id
        )
        if not states:
            break
        for state in states:
            if store.enqueue(
                state, schema_version=schema_version, prompt_version=prompt_version, model_id=model_id
            ):
                queued += 1
            scanned += 1
            after_id = state.chunk_db_id  # forward-only cursor => resumable
            if max_chunks is not None and scanned >= max_chunks:
                return queued
    return queued


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Queue graph extraction for existing chunks.")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--mailbox-id", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from email_thread_rag.config import get_settings
    from email_thread_rag.rag.paradedb.repository import connect

    settings = get_settings()
    if not settings.graph_extraction_enabled:
        parser.error("GRAPH_EXTRACTION_ENABLED is false; enable graph extraction before backfilling.")

    conn = connect(settings.database_url, autocommit=True)
    queued = backfill_graph_jobs(conn, tenant_id=args.tenant_id, mailbox_id=args.mailbox_id, settings=settings,
                                 batch_size=args.batch_size, max_chunks=args.max_chunks)
    logger.info("queued %d graph job(s); run the graph worker to process them", queued)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
