"""``SyncStore`` over Postgres (migration 0002_gmail.sql).

Held to the same contract as ``InMemorySyncStore`` -- see
``tests/gmail_store_contract.py``, which both implementations run.

psycopg is imported lazily by ``build_store``; importing this module without
the postgres extra is fine as long as you never construct the store.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from email_thread_rag.gmail.models import GmailNotification, Mailbox, OAuthState, SyncJob

# Gmail history counters are unsigned 64-bit. Postgres numeric(20,0) round-trips
# them through psycopg as Decimal; int() is exact for integral Decimals.
_MAILBOX_COLUMNS = (
    "id, tenant_id, mailbox_id, email_address, status, last_committed_history_id, "
    "watch_topic, watch_expiration, last_error, refresh_token_ciphertext, token_key_id"
)
_JOB_COLUMNS = (
    "id, mailbox_db_id, tenant_id, mailbox_id, requested_history_id, status, attempts, "
    "needs_full_sync, leased_until, lease_owner, last_error, completed_at"
)


def _history_id(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _row_to_mailbox(row: dict[str, Any] | None) -> Optional[Mailbox]:
    if row is None:
        return None
    return Mailbox(
        id=row["id"],
        tenant_id=row["tenant_id"],
        mailbox_id=row["mailbox_id"],
        email_address=row["email_address"],
        status=row["status"],
        last_committed_history_id=_history_id(row["last_committed_history_id"]),
        watch_topic=row["watch_topic"],
        watch_expiration=row["watch_expiration"],
        last_error=row["last_error"],
        refresh_token_ciphertext=bytes(row["refresh_token_ciphertext"])
        if row["refresh_token_ciphertext"] is not None
        else None,
        token_key_id=row["token_key_id"],
    )


def _row_to_job(row: dict[str, Any] | None) -> Optional[SyncJob]:
    if row is None:
        return None
    return SyncJob(
        id=row["id"],
        mailbox_db_id=row["mailbox_db_id"],
        tenant_id=row["tenant_id"],
        mailbox_id=row["mailbox_id"],
        requested_history_id=int(row["requested_history_id"]),
        status=row["status"],
        attempts=row["attempts"],
        needs_full_sync=row["needs_full_sync"],
        leased_until=row["leased_until"],
        lease_owner=row["lease_owner"],
        last_error=row["last_error"],
        completed_at=row["completed_at"],
    )


class PostgresSyncStore:
    def __init__(self, conn):
        self.conn = conn

    # --- OAuth state -----------------------------------------------------
    def create_oauth_state(self, state: OAuthState) -> None:
        with self.conn.transaction():
            self.conn.execute(
                "INSERT INTO gmail_oauth_states "
                "(state, tenant_id, mailbox_id, code_verifier, redirect_uri, expires_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    state.state,
                    state.tenant_id,
                    state.mailbox_id,
                    state.code_verifier,
                    state.redirect_uri,
                    state.expires_at,
                ),
            )

    def consume_oauth_state(self, state: str) -> Optional[OAuthState]:
        """Atomic single-use consume: the conditional UPDATE is the check.

        A read-then-write would let two concurrent callbacks both pass the
        "unconsumed?" test; here the second one updates zero rows.
        """
        with self.conn.transaction():
            row = self.conn.execute(
                "UPDATE gmail_oauth_states SET consumed_at = now() "
                "WHERE state = %s AND consumed_at IS NULL AND expires_at > now() "
                "RETURNING state, tenant_id, mailbox_id, code_verifier, redirect_uri, expires_at",
                (state,),
            ).fetchone()
        if row is None:
            return None
        return OAuthState(
            state=row["state"],
            tenant_id=row["tenant_id"],
            mailbox_id=row["mailbox_id"],
            code_verifier=row["code_verifier"],
            redirect_uri=row["redirect_uri"],
            expires_at=row["expires_at"],
        )

    # --- Mailboxes -------------------------------------------------------
    def upsert_mailbox(
        self,
        *,
        tenant_id: str,
        mailbox_id: str,
        email_address: str,
        refresh_token_ciphertext: bytes,
        token_key_id: str,
    ) -> Mailbox:
        with self.conn.transaction():
            row = self.conn.execute(
                f"""
                INSERT INTO gmail_mailboxes (
                    tenant_id, mailbox_id, email_address, refresh_token_ciphertext,
                    token_key_id, status, updated_at
                ) VALUES (%s, %s, %s, %s, %s, 'pending', now())
                ON CONFLICT (tenant_id, mailbox_id) DO UPDATE SET
                    email_address = EXCLUDED.email_address,
                    refresh_token_ciphertext = EXCLUDED.refresh_token_ciphertext,
                    token_key_id = EXCLUDED.token_key_id,
                    status = 'pending',
                    last_error = NULL,
                    updated_at = now()
                RETURNING {_MAILBOX_COLUMNS}
                """,
                (tenant_id, mailbox_id, email_address, refresh_token_ciphertext, token_key_id),
            ).fetchone()
        return _row_to_mailbox(row)

    def get_mailbox(self, tenant_id: str, mailbox_id: str) -> Optional[Mailbox]:
        return _row_to_mailbox(
            self.conn.execute(
                f"SELECT {_MAILBOX_COLUMNS} FROM gmail_mailboxes "
                "WHERE tenant_id = %s AND mailbox_id = %s",
                (tenant_id, mailbox_id),
            ).fetchone()
        )

    def get_mailbox_by_db_id(self, mailbox_db_id: int) -> Optional[Mailbox]:
        return _row_to_mailbox(
            self.conn.execute(
                f"SELECT {_MAILBOX_COLUMNS} FROM gmail_mailboxes WHERE id = %s", (mailbox_db_id,)
            ).fetchone()
        )

    def find_live_mailbox_by_address(self, email_address: str) -> Optional[Mailbox]:
        return _row_to_mailbox(
            self.conn.execute(
                f"SELECT {_MAILBOX_COLUMNS} FROM gmail_mailboxes "
                "WHERE email_address = %s AND status <> 'disconnected'",
                (email_address,),
            ).fetchone()
        )

    def activate_watch(
        self, mailbox_db_id: int, *, history_id: int, topic: str, expiration: datetime
    ) -> Mailbox:
        """Persist watch state and mark the mailbox active, in one transaction.

        COALESCE seeds the cursor from the watch's historyId only on first
        connect: a renewal must never rewind or skip past a cursor the worker
        has already committed.
        """
        with self.conn.transaction():
            row = self.conn.execute(
                f"""
                UPDATE gmail_mailboxes SET
                    watch_topic = %s,
                    watch_expiration = %s,
                    last_committed_history_id = COALESCE(last_committed_history_id, %s),
                    status = 'active',
                    last_error = NULL,
                    updated_at = now()
                WHERE id = %s
                RETURNING {_MAILBOX_COLUMNS}
                """,
                (topic, expiration, history_id, mailbox_db_id),
            ).fetchone()
        return _row_to_mailbox(row)

    def set_mailbox_status(self, mailbox_db_id: int, status: str, *, error: str | None = None) -> None:
        with self.conn.transaction():
            self.conn.execute(
                "UPDATE gmail_mailboxes SET status = %s, last_error = %s, updated_at = now() WHERE id = %s",
                (status, error, mailbox_db_id),
            )

    def disconnect_mailbox(self, mailbox_db_id: int) -> None:
        with self.conn.transaction():
            self.conn.execute(
                "UPDATE gmail_mailboxes SET status = 'disconnected', refresh_token_ciphertext = NULL, "
                "token_key_id = NULL, watch_topic = NULL, watch_expiration = NULL, updated_at = now() "
                "WHERE id = %s",
                (mailbox_db_id,),
            )

    def commit_history_cursor(self, mailbox_db_id: int, history_id: int) -> None:
        with self.conn.transaction():
            # GREATEST on numeric: an out-of-order run can only move the cursor
            # forward, never rewind it. This is why history IDs are not text.
            self.conn.execute(
                "UPDATE gmail_mailboxes SET "
                "last_committed_history_id = GREATEST(COALESCE(last_committed_history_id, 0), %s), "
                "updated_at = now() WHERE id = %s",
                (history_id, mailbox_db_id),
            )

    def mailboxes_due_for_watch_renewal(self, *, before: datetime) -> list[Mailbox]:
        rows = self.conn.execute(
            f"SELECT {_MAILBOX_COLUMNS} FROM gmail_mailboxes "
            "WHERE status IN ('active', 'needs_full_sync') AND watch_expiration IS NOT NULL "
            "AND watch_expiration <= %s ORDER BY watch_expiration ASC",
            (before,),
        ).fetchall()
        return [_row_to_mailbox(row) for row in rows]

    # --- Jobs ------------------------------------------------------------
    def record_notification(self, notification: GmailNotification) -> Optional[SyncJob]:
        """Dedup + create/coalesce in ONE transaction, so the webhook can only
        return 200 after the job is durable. Returns None for a redelivered
        Pub/Sub message or an address with no live mailbox (both: ack, no work).
        """
        with self.conn.transaction():
            inserted = self.conn.execute(
                "INSERT INTO gmail_pubsub_messages (pubsub_message_id) VALUES (%s) "
                "ON CONFLICT (pubsub_message_id) DO NOTHING RETURNING pubsub_message_id",
                (notification.pubsub_message_id,),
            ).fetchone()
            if inserted is None:
                return None
            mailbox = self.find_live_mailbox_by_address(notification.email_address)
            if mailbox is None:
                return None
            row = self.conn.execute(
                f"""
                INSERT INTO gmail_sync_jobs (
                    mailbox_db_id, tenant_id, mailbox_id, requested_history_id, status, needs_full_sync
                ) VALUES (%s, %s, %s, %s, 'pending', %s)
                ON CONFLICT (mailbox_db_id) WHERE status = 'pending' DO UPDATE SET
                    requested_history_id = GREATEST(
                        gmail_sync_jobs.requested_history_id, EXCLUDED.requested_history_id
                    ),
                    updated_at = now()
                RETURNING {_JOB_COLUMNS}
                """,
                (
                    mailbox.id,
                    mailbox.tenant_id,
                    mailbox.mailbox_id,
                    notification.history_id,
                    mailbox.status == "needs_full_sync",
                ),
            ).fetchone()
        return _row_to_job(row)

    def claim_job(self, *, owner: str, lease_seconds: int = 300) -> Optional[SyncJob]:
        """Lease one job. SKIP LOCKED lets N workers claim disjoint jobs; the
        expired-lease clause reclaims work from a worker that died mid-run."""
        with self.conn.transaction():
            candidate = self.conn.execute(
                "SELECT id FROM gmail_sync_jobs "
                "WHERE status = 'pending' OR (status = 'running' AND leased_until <= now()) "
                "ORDER BY id ASC FOR UPDATE SKIP LOCKED LIMIT 1"
            ).fetchone()
            if candidate is None:
                return None
            row = self.conn.execute(
                f"""
                UPDATE gmail_sync_jobs SET
                    status = 'running',
                    attempts = attempts + 1,
                    lease_owner = %s,
                    leased_until = now() + make_interval(secs => %s),
                    updated_at = now()
                WHERE id = %s
                RETURNING {_JOB_COLUMNS}
                """,
                (owner, lease_seconds, candidate["id"]),
            ).fetchone()
        return _row_to_job(row)

    def get_job(self, job_id: int) -> Optional[SyncJob]:
        return _row_to_job(
            self.conn.execute(
                f"SELECT {_JOB_COLUMNS} FROM gmail_sync_jobs WHERE id = %s", (job_id,)
            ).fetchone()
        )

    def complete_job(self, job_id: int) -> None:
        with self.conn.transaction():
            self.conn.execute(
                "UPDATE gmail_sync_jobs SET status = 'done', completed_at = now(), leased_until = NULL, "
                "lease_owner = NULL, last_error = NULL, updated_at = now() WHERE id = %s",
                (job_id,),
            )

    def fail_job(self, job_id: int, error: str, *, max_attempts: int = 5) -> None:
        """Release for retry without touching the mailbox cursor.

        If a newer notification queued a pending job for the same mailbox while
        this one ran, fold this job's requested history ID into it instead of
        returning to pending -- the partial unique index allows only one pending
        job per mailbox, and the successor already covers the work.
        """
        with self.conn.transaction():
            job = self.conn.execute(
                f"SELECT {_JOB_COLUMNS} FROM gmail_sync_jobs WHERE id = %s FOR UPDATE", (job_id,)
            ).fetchone()
            if job is None:
                return
            successor = self.conn.execute(
                "SELECT id FROM gmail_sync_jobs WHERE mailbox_db_id = %s AND status = 'pending' AND id <> %s",
                (job["mailbox_db_id"], job_id),
            ).fetchone()
            if job["attempts"] >= max_attempts or successor is not None:
                if successor is not None:
                    self.conn.execute(
                        "UPDATE gmail_sync_jobs SET requested_history_id = GREATEST("
                        "requested_history_id, %s), needs_full_sync = needs_full_sync OR %s, "
                        "updated_at = now() WHERE id = %s",
                        (job["requested_history_id"], job["needs_full_sync"], successor["id"]),
                    )
                self.conn.execute(
                    "UPDATE gmail_sync_jobs SET status = 'failed', last_error = %s, leased_until = NULL, "
                    "lease_owner = NULL, completed_at = now(), updated_at = now() WHERE id = %s",
                    (error, job_id),
                )
                return
            self.conn.execute(
                "UPDATE gmail_sync_jobs SET status = 'pending', last_error = %s, leased_until = NULL, "
                "lease_owner = NULL, updated_at = now() WHERE id = %s",
                (error, job_id),
            )

    def mark_job_needs_full_sync(self, job_id: int) -> None:
        with self.conn.transaction():
            self.conn.execute(
                "UPDATE gmail_sync_jobs SET needs_full_sync = true, updated_at = now() WHERE id = %s",
                (job_id,),
            )


def build_store(database_url: str | None):
    from email_thread_rag.rag.paradedb.repository import connect

    return PostgresSyncStore(connect(database_url))
