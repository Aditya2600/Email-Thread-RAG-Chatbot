"""DB-backed sync worker + its local entry point.

The ``gmail_sync_jobs`` table is the queue. No Celery, no Redis, no broker:
``claim_job`` leases with ``FOR UPDATE SKIP LOCKED``, so running two of these
processes is already safe, and a worker that dies mid-job has its lease expire
and the job reclaimed.

Run it locally:

    python -m email_thread_rag.gmail.worker --once
    python -m email_thread_rag.gmail.worker --poll-interval 15
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import time
from typing import Callable, Optional

from email_thread_rag.gmail.client import GmailClient
from email_thread_rag.gmail.models import Mailbox
from email_thread_rag.gmail.sink import ChunkSink
from email_thread_rag.gmail.store import SyncStore
from email_thread_rag.gmail.sync import SyncOutcome, run_sync

logger = logging.getLogger(__name__)


class SyncWorker:
    """Claims one job at a time and runs it. Client/sink construction is
    injected so tests drive fakes and production builds real ones."""

    def __init__(
        self,
        store: SyncStore,
        *,
        client_factory: Callable[[Mailbox], GmailClient],
        sink_factory: Callable[[Mailbox], ChunkSink],
        owner: str | None = None,
        lease_seconds: int = 300,
        max_attempts: int = 5,
    ):
        self.store = store
        self.client_factory = client_factory
        self.sink_factory = sink_factory
        self.owner = owner or f"{socket.gethostname()}:{os.getpid()}"
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts

    def run_once(self) -> Optional[SyncOutcome]:
        """Claim and run one job. Returns None when the queue is empty."""
        job = self.store.claim_job(owner=self.owner, lease_seconds=self.lease_seconds)
        if job is None:
            return None

        mailbox = self.store.get_mailbox_by_db_id(job.mailbox_db_id)
        if mailbox is None or mailbox.status == "disconnected":
            logger.info("job %s targets a missing/disconnected mailbox; dropping", job.id)
            self.store.complete_job(job.id)
            return None

        try:
            # Both factories and every Gmail call below run with no DB
            # transaction open: the store commits each statement on its own.
            outcome = run_sync(
                self.store,
                self.client_factory(mailbox),
                self.sink_factory(mailbox),
                mailbox=mailbox,
                job=job,
            )
        except Exception as exc:  # noqa: BLE001 - a failed job must not kill the worker
            # The cursor was never advanced: run_sync commits it only after
            # everything is persisted. The retry re-reads the same window.
            reason = f"{type(exc).__name__}: {exc}"
            logger.exception("gmail sync job %s failed", job.id)
            self.store.fail_job(job.id, reason, max_attempts=self.max_attempts)
            self.store.set_mailbox_status(mailbox.id, "error", error=reason)
            return None

        self.store.complete_job(job.id)
        logger.info(
            "job %s synced mailbox %s: +%s messages, -%s messages, %s chunks, cursor=%s%s",
            job.id,
            mailbox.mailbox_id,
            outcome.added,
            outcome.deleted,
            outcome.chunks_written,
            outcome.committed_history_id,
            " (full sync)" if outcome.full_sync else "",
        )
        return outcome

    def drain(self, *, limit: int = 100) -> list[SyncOutcome]:
        """Run queued jobs until empty. Handy for tests and manual catch-up."""
        outcomes: list[SyncOutcome] = []
        for _ in range(limit):
            outcome = self.run_once()
            if outcome is None:
                break
            outcomes.append(outcome)
        return outcomes


def build_production_worker(settings) -> SyncWorker:
    """Wire the worker to Postgres + real Gmail. Never imported by tests."""
    from email_thread_rag.gmail.cipher import build_token_cipher
    from email_thread_rag.gmail.client import HttpxGmailClient
    from email_thread_rag.gmail.oauth import refresh_access_token
    from email_thread_rag.gmail.repository import PostgresSyncStore
    from email_thread_rag.gmail.sink import ParadeDBChunkSink
    from email_thread_rag.rag.paradedb.repository import connect
    from email_thread_rag.rag.vector_index import SentenceTransformerEncoder

    # autocommit + explicit transaction blocks: this is what keeps a DB
    # transaction from sitting open across a Gmail HTTP call.
    conn = connect(settings.database_url, autocommit=True)
    store = PostgresSyncStore(conn)
    cipher = build_token_cipher(settings)
    encoder = SentenceTransformerEncoder(settings)

    def client_factory(mailbox: Mailbox) -> GmailClient:
        if not mailbox.refresh_token_ciphertext:
            raise RuntimeError(f"mailbox {mailbox.mailbox_id} has no stored credential; reconnect it")
        access_token = refresh_access_token(
            refresh_token=cipher.decrypt(mailbox.refresh_token_ciphertext),
            client_id=settings.gmail_client_id,
            client_secret=settings.gmail_client_secret,
        )
        return HttpxGmailClient(access_token)

    def sink_factory(mailbox: Mailbox) -> ChunkSink:
        return ParadeDBChunkSink(
            conn,
            tenant_id=mailbox.tenant_id,
            mailbox_id=mailbox.mailbox_id,
            encoder=encoder,
            embedding_dim=settings.embedding_dim,
            # Lets the sink queue Stage-4 context work. Inert unless enabled;
            # the sync worker never calls the LLM itself.
            settings=settings,
        )

    return SyncWorker(store, client_factory=client_factory, sink_factory=sink_factory)


def renew_watches(settings) -> int:
    """Renew every watch nearing expiry. Run from cron/systemd at least daily:
    a Gmail watch dies after 7 days and the mailbox goes silent with no error."""
    from email_thread_rag.gmail.cipher import build_token_cipher
    from email_thread_rag.gmail.client import HttpxGmailClient
    from email_thread_rag.gmail.oauth import refresh_access_token
    from email_thread_rag.gmail.repository import PostgresSyncStore
    from email_thread_rag.gmail.service import renew_expiring_watches
    from email_thread_rag.rag.paradedb.repository import connect

    store = PostgresSyncStore(connect(settings.database_url, autocommit=True))
    cipher = build_token_cipher(settings)

    def client_factory(mailbox: Mailbox) -> GmailClient:
        return HttpxGmailClient(
            refresh_access_token(
                refresh_token=cipher.decrypt(mailbox.refresh_token_ciphertext),
                client_id=settings.gmail_client_id,
                client_secret=settings.gmail_client_secret,
            )
        )

    renewed = renew_expiring_watches(store, client_factory, topic_name=settings.gmail_pubsub_topic)
    return len(renewed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gmail sync worker (the DB table is the queue).")
    parser.add_argument("--once", action="store_true", help="Drain the queue and exit.")
    parser.add_argument("--poll-interval", type=float, default=10.0, help="Seconds between empty polls.")
    parser.add_argument(
        "--renew-watches", action="store_true", help="Renew watches nearing expiry and exit."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    from email_thread_rag.config import get_settings

    settings = get_settings()

    if args.renew_watches:
        print(f"Renewed {renew_watches(settings)} watch(es).")
        return

    worker = build_production_worker(settings)

    if args.once:
        outcomes = worker.drain()
        print(f"Processed {len(outcomes)} job(s).")
        return

    # Plain poll loop, no scheduler: `--once` under cron/systemd covers
    # scheduling when that's wanted.
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False
        logger.info("shutting down after the current job")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while running:
        if worker.run_once() is None:
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
