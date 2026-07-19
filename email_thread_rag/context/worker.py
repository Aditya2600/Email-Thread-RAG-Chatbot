"""The contextualization worker: claim, call, validate, commit.

The ordering is the whole design:

    claim (txn) -> [no txn held] LLM call -> validate -> commit (txn)

No DB transaction is open while the provider is called, so a slow or hung model
cannot pin a Postgres connection. The chunk may change during that window, which
is exactly what ``commit_context``'s fingerprint recheck is for.

Two failure modes, deliberately handled differently:

  * provider raised (network, HTTP 5xx, timeout) -> transient. fail_job, retry.
  * output failed validation -> deterministic. temperature=0 means a retry
    produces the identical bad output, so retrying is a guaranteed-useless loop.
    Fall back to the Stage-1 deterministic embed_text and retire the job.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

from email_thread_rag.context.fingerprint import PROMPT_VERSION
from email_thread_rag.context.models import ChunkContextState
from email_thread_rag.context.prompt import ContextValidationError, validate_output
from email_thread_rag.context.provider import ContextProvider, ContextProviderError
from email_thread_rag.context.store import ContextJobStore
from email_thread_rag.rag.email_segmentation import build_embed_text

logger = logging.getLogger(__name__)


def rebuild_embed_text(state: ChunkContextState, prefix: Optional[str]) -> str:
    """The canonical assembly, reused. Never re-implement it here."""
    return build_embed_text(
        state.text,
        sender=state.sender,
        to=state.to,
        cc=state.cc,
        date=state.date,
        subject=state.subject,
        thread_id=state.thread_id,
        in_reply_to=state.in_reply_to,
        context_prefix=prefix,
    )


class ContextWorker:
    def __init__(
        self,
        store: ContextJobStore,
        provider: ContextProvider,
        *,
        encoder,
        owner: str = "context-worker",
        prompt_version: str = PROMPT_VERSION,
        max_attempts: int = 3,
    ):
        self.store = store
        self.provider = provider
        self.encoder = encoder
        self.owner = owner
        self.prompt_version = prompt_version
        self.max_attempts = max_attempts

    def run_once(self) -> bool:
        """Process at most one job. Returns False when the queue is empty."""
        job = self.store.claim_job(owner=self.owner)
        if job is None:
            return False

        state = self.store.load_chunk_state(job.chunk_db_id)
        if state is None:
            # Chunk deleted (message removed) while the job waited. Nothing to
            # contextualize; the ON DELETE CASCADE usually beats us here.
            self.store.fail_job(job.id, "chunk no longer exists", max_attempts=0)
            return True

        try:
            raw = self.provider.generate(state.as_context_input())
        except ContextProviderError as exc:
            # Transient: keep the job, let it retry.
            self.store.fail_job(job.id, str(exc), max_attempts=self.max_attempts)
            return True

        try:
            prefix = validate_output(raw)
            method = "llm"
        except ContextValidationError as exc:
            # Deterministic fallback. The chunk stays perfectly retrievable via
            # its Stage-1 embed_text; it simply gains no context prefix.
            logger.warning("context validation failed for %s: %s", state.chunk_id, exc)
            prefix, method = None, "deterministic"

        embed_text = rebuild_embed_text(state, prefix)
        embedding = None
        if self.encoder is not None:
            embedding = list(self.encoder.encode([embed_text])[0])

        self.store.commit_context(
            job,
            prefix=prefix,
            method=method,
            context_version=self.prompt_version,
            model_id=self.provider.model_id,
            prompt_version=self.prompt_version,
            embed_text=embed_text,
            embedding=embedding,
            **self._embedding_model_kwarg(),
        )
        return True

    def _embedding_model_kwarg(self) -> dict:
        # The in-memory store has no embedding_model column; only pass it where
        # it means something.
        if self.encoder is None or not hasattr(self.store, "embedding_dim"):
            return {}
        return {"embedding_model": getattr(self.encoder, "model_name", self.encoder.__class__.__name__)}

    def drain(self, *, max_jobs: int = 1000) -> int:
        processed = 0
        while processed < max_jobs and self.run_once():
            processed += 1
        return processed


def build_production_worker(settings, *, owner: str = "context-worker"):
    """Wire a worker from configuration. Imports stay local so this module is
    importable without psycopg or a provider installed."""
    from email_thread_rag.context.provider import build_provider
    from email_thread_rag.context.repository import PostgresContextJobStore
    from email_thread_rag.rag.paradedb.repository import connect
    from email_thread_rag.rag.vector_index import SentenceTransformerEncoder

    provider = build_provider(settings)
    if provider is None:
        raise ContextProviderError(
            "Contextualization is disabled. Set CONTEXT_ENABLED=true to run the worker."
        )
    # autocommit + explicit transaction blocks: this is what keeps a DB
    # transaction from sitting open across the LLM call.
    conn = connect(settings.database_url, autocommit=True)
    store = PostgresContextJobStore(conn, embedding_dim=settings.embedding_dim)
    # The same encoder ingestion used, so re-embedding stays in one vector space.
    return ContextWorker(store, provider, encoder=SentenceTransformerEncoder(settings), owner=owner)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process the chunk contextualization queue.")
    parser.add_argument("--once", action="store_true", help="Drain the queue and exit.")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--max-jobs", type=int, default=1000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from email_thread_rag.config import get_settings

    worker = build_production_worker(get_settings())
    if args.once:
        count = worker.drain(max_jobs=args.max_jobs)
        logger.info("processed %d context job(s)", count)
        return 0

    logger.info("context worker polling every %.1fs", args.poll_interval)
    while True:
        if worker.drain(max_jobs=args.max_jobs) == 0:
            time.sleep(args.poll_interval)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
