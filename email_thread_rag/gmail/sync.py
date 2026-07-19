"""Gmail delta/full sync: the cursor state machine.

Cursor rule, enforced in exactly one place (``run_sync``): the mailbox's
``last_committed_history_id`` advances only after every change in the run has
been persisted. Any failure -- Gmail error, sink error, worker crash -- leaves
the old cursor intact, so the retry re-reads the same history window. Gmail
history is replayed against an idempotent sink (upsert by message ID), so
re-reading a window is safe.

No DB transaction is held across a Gmail call: each sink write commits on its
own, and the cursor commit is a separate statement at the end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

from email_thread_rag.gmail.client import GmailClient, GmailHistoryExpired
from email_thread_rag.gmail.message import gmail_message_to_email_record
from email_thread_rag.gmail.models import Mailbox, SyncJob
from email_thread_rag.gmail.sink import ChunkSink
from email_thread_rag.gmail.store import SyncStore

logger = logging.getLogger(__name__)

HISTORY_TYPES = ("messageAdded", "messageDeleted")


@dataclass
class SyncOutcome:
    added: int = 0
    deleted: int = 0
    chunks_written: int = 0
    committed_history_id: int | None = None
    full_sync: bool = False


@dataclass
class HistoryDiff:
    """Net effect of a history window: message IDs to add, message IDs to drop."""

    added_ids: list[str]
    deleted_ids: list[str]
    max_history_id: int


def collect_history(client: GmailClient, *, start_history_id: int) -> HistoryDiff:
    """Fold every history page into one net diff.

    Gmail returns history records in chronological order and the same message
    can appear on several pages, so this dedups across pages and lets the last
    event for a message win: added-then-deleted is a delete, deleted-then-added
    (a restore) is an add.
    """
    # dict-as-ordered-set: dedup while keeping Gmail's chronological order.
    added: dict[str, None] = {}
    deleted: dict[str, None] = {}
    max_history_id = start_history_id

    for page in client.list_history(start_history_id=start_history_id, history_types=HISTORY_TYPES):
        max_history_id = max(max_history_id, _as_history_id(page.get("historyId")))
        for record in page.get("history", []):
            max_history_id = max(max_history_id, _as_history_id(record.get("id")))
            for item in record.get("messagesAdded", []):
                message_id = item.get("message", {}).get("id")
                if message_id:
                    deleted.pop(message_id, None)
                    added[message_id] = None
            for item in record.get("messagesDeleted", []):
                message_id = item.get("message", {}).get("id")
                if message_id:
                    added.pop(message_id, None)
                    deleted[message_id] = None

    return HistoryDiff(added_ids=list(added), deleted_ids=list(deleted), max_history_id=max_history_id)


def _as_history_id(value: Any) -> int:
    # Gmail sends history IDs as JSON strings; int() keeps every comparison
    # numeric. '10' < '9' as text, which would silently rewind a cursor.
    return int(value) if value not in (None, "") else 0


def _ingest_messages(
    client: GmailClient, sink: ChunkSink, message_ids: Iterable[str], *, email_address: str
) -> tuple[int, int]:
    added = 0
    chunks_written = 0
    for message_id in message_ids:
        message = client.get_message(message_id)
        if message is None:
            # Deleted between the history page and this fetch. Not an error:
            # a later messageDeleted record (or the next run) covers it.
            logger.info("gmail message %s vanished before fetch; skipping", message_id)
            continue
        email = gmail_message_to_email_record(message, email_address=email_address)
        chunks_written += sink.persist(email)
        # Persist PDF attachment metadata + enqueue extraction. Never blocks the
        # sync: the worker fetches bytes and parses/OCRs off this path.
        sink.persist_attachments(message, email=email)
        added += 1
    return added, chunks_written


def run_delta_sync(
    client: GmailClient, sink: ChunkSink, *, start_history_id: int, email_address: str
) -> SyncOutcome:
    """Apply one history window. Raises GmailHistoryExpired if the cursor is too old."""
    diff = collect_history(client, start_history_id=start_history_id)
    added, chunks_written = _ingest_messages(client, sink, diff.added_ids, email_address=email_address)
    for message_id in diff.deleted_ids:
        sink.delete_message(message_id)
    return SyncOutcome(
        added=added,
        deleted=len(diff.deleted_ids),
        chunks_written=chunks_written,
        committed_history_id=diff.max_history_id,
    )


def run_full_sync(client: GmailClient, sink: ChunkSink, *, email_address: str) -> SyncOutcome:
    """Rebuild the mailbox, then close the gap the rebuild itself opened.

    Order matters: the checkpoint is taken *before* the scan, so mail that
    arrives or is deleted while messages.list pages through is picked up by the
    replay afterwards. Taking it after the scan would silently lose those
    changes. Every step is idempotent (upsert by message ID), so a crash
    anywhere just means the whole full sync runs again.
    """
    checkpoint = int(client.get_profile()["historyId"])

    scanned_ids: list[str] = []
    for page in client.list_messages():
        scanned_ids.extend(item["id"] for item in page.get("messages", []))

    added, chunks_written = _ingest_messages(client, sink, scanned_ids, email_address=email_address)
    outcome = SyncOutcome(
        added=added, chunks_written=chunks_written, committed_history_id=checkpoint, full_sync=True
    )

    try:
        replay = run_delta_sync(client, sink, start_history_id=checkpoint, email_address=email_address)
    except GmailHistoryExpired:
        # The checkpoint aged out during a very long scan. The scan itself is
        # still valid up to `checkpoint`; commit that and let the next run
        # decide whether another full sync is needed.
        logger.warning("gmail history expired while replaying full-sync checkpoint %s", checkpoint)
        return outcome

    outcome.added += replay.added
    outcome.deleted += replay.deleted
    outcome.chunks_written += replay.chunks_written
    outcome.committed_history_id = max(checkpoint, replay.committed_history_id or checkpoint)
    return outcome


def run_sync(
    store: SyncStore, client: GmailClient, sink: ChunkSink, *, mailbox: Mailbox, job: SyncJob
) -> SyncOutcome:
    """Run one job to completion and commit the cursor exactly once, at the end."""
    needs_full = (
        job.needs_full_sync
        or mailbox.status == "needs_full_sync"
        or mailbox.last_committed_history_id is None
    )

    if needs_full:
        outcome = run_full_sync(client, sink, email_address=mailbox.email_address)
    else:
        try:
            outcome = run_delta_sync(
                client,
                sink,
                start_history_id=mailbox.last_committed_history_id,
                email_address=mailbox.email_address,
            )
        except GmailHistoryExpired:
            # Mark first, then rebuild: if this process dies right here, the
            # persisted marks make the retry a full sync, and the old cursor is
            # still untouched.
            store.mark_job_needs_full_sync(job.id)
            store.set_mailbox_status(
                mailbox.id, "needs_full_sync", error="Gmail history expired; full sync required"
            )
            outcome = run_full_sync(client, sink, email_address=mailbox.email_address)

    # The only cursor advance in the codebase, and the last thing this run does.
    if outcome.committed_history_id is not None:
        store.commit_history_cursor(mailbox.id, outcome.committed_history_id)
    store.set_mailbox_status(mailbox.id, "active")
    return outcome
