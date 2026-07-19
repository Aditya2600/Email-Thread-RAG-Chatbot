"""The extraction worker: claim, call, validate, ground-in-evidence, commit.

The ordering is the whole design, identical to Stage 4's context worker:

    claim (txn) -> [no txn held] LLM call -> validate -> resolve -> commit (txn)

No DB transaction is open while the provider is called. The chunk may change
during that window, which is exactly what ``commit_graph``'s hash recheck is for.

Two failure modes, handled differently:
  * provider raised (network/5xx/timeout) -> transient. fail_job, retry.
  * output failed to parse -> deterministic. temperature=0 means a retry yields
    the identical bad output, so we do not loop: we commit the metadata-only
    graph (which needs no LLM) and mark the chunk extracted.
"""

from __future__ import annotations

import argparse
import logging
import time

from email_thread_rag.graph.extract import resolve_extraction
from email_thread_rag.graph.fingerprint import PROMPT_VERSION, SCHEMA_VERSION
from email_thread_rag.graph.prompt import GraphValidationError, LLMExtraction, validate_extraction
from email_thread_rag.graph.provider import GraphProvider, GraphProviderError
from email_thread_rag.graph.store import GraphStore

logger = logging.getLogger(__name__)


class GraphWorker:
    def __init__(
        self,
        store: GraphStore,
        provider: GraphProvider,
        *,
        owner: str = "graph-worker",
        schema_version: str = SCHEMA_VERSION,
        prompt_version: str = PROMPT_VERSION,
        max_attempts: int = 3,
    ):
        self.store = store
        self.provider = provider
        self.owner = owner
        self.schema_version = schema_version
        self.prompt_version = prompt_version
        self.max_attempts = max_attempts

    def run_once(self) -> bool:
        """Process at most one job. Returns False when the queue is empty."""
        job = self.store.claim_job(owner=self.owner)
        if job is None:
            return False

        state = self.store.load_chunk_state(job.chunk_db_id)
        if state is None:
            self.store.fail_job(job.id, "chunk no longer exists", error_rule="missing_chunk", max_attempts=0)
            return True

        try:
            raw = self.provider.generate(state.as_extraction_input())
        except GraphProviderError as exc:
            self.store.fail_job(job.id, str(exc), error_rule="provider", max_attempts=self.max_attempts)
            return True

        try:
            extraction = validate_extraction(raw)
            method = "llm"
        except GraphValidationError as exc:
            # Deterministic: no usable LLM graph, but metadata relations still
            # apply. Commit those and mark the chunk extracted rather than loop.
            logger.warning("graph validation failed for %s: %s", state.chunk_id, exc)
            extraction, method = LLMExtraction(), "deterministic"

        resolved = resolve_extraction(extraction, state)
        self.store.commit_graph(
            job,
            resolved=resolved,
            method=method,
            extraction_version=self.schema_version,
            schema_version=self.schema_version,
            prompt_version=self.prompt_version,
            model_id=self.provider.model_id,
        )
        return True

    def drain(self, *, max_jobs: int = 1000) -> int:
        processed = 0
        while processed < max_jobs and self.run_once():
            processed += 1
        return processed


def build_production_worker(settings, *, owner: str = "graph-worker"):
    """Wire a worker from configuration. Imports stay local so this module is
    importable without psycopg or a provider installed."""
    from email_thread_rag.graph.provider import build_provider
    from email_thread_rag.graph.repository import PostgresGraphStore
    from email_thread_rag.rag.paradedb.repository import connect

    provider = build_provider(settings)
    if provider is None:
        raise GraphProviderError(
            "Graph extraction is disabled. Set GRAPH_EXTRACTION_ENABLED=true to run the worker."
        )
    conn = connect(settings.database_url, autocommit=True)
    return GraphWorker(PostgresGraphStore(conn), provider, owner=owner)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process the graph-extraction queue.")
    parser.add_argument("--once", action="store_true", help="Drain the queue and exit.")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--max-jobs", type=int, default=1000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from email_thread_rag.config import get_settings

    worker = build_production_worker(get_settings())
    if args.once:
        logger.info("processed %d graph job(s)", worker.drain(max_jobs=args.max_jobs))
        return 0

    logger.info("graph worker polling every %.1fs", args.poll_interval)
    while True:
        if worker.drain(max_jobs=args.max_jobs) == 0:
            time.sleep(args.poll_interval)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
