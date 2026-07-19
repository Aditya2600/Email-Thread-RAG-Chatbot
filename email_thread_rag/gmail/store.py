"""The sync state machine's storage contract, plus an in-memory implementation.

``SyncStore`` is the only thing the webhook and worker know about. Two
implementations exist and are held to the same contract tests
(``tests/gmail_store_contract.py``):

* ``InMemorySyncStore`` -- unit tests, no Postgres, no configuration.
* ``PostgresSyncStore`` (``gmail/repository.py``) -- production; the same
  operations as SQL, with the durability the in-memory one only simulates.

Every method that takes or returns a history ID uses ``int``. Ordering is
numeric at both layers; a history ID is never compared as text.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

from email_thread_rag.gmail.models import GmailNotification, Mailbox, OAuthState, SyncJob


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SyncStore(Protocol):
    def create_oauth_state(self, state: OAuthState) -> None: ...
    def consume_oauth_state(self, state: str) -> Optional[OAuthState]: ...

    def upsert_mailbox(
        self,
        *,
        tenant_id: str,
        mailbox_id: str,
        email_address: str,
        refresh_token_ciphertext: bytes,
        token_key_id: str,
    ) -> Mailbox: ...
    def get_mailbox(self, tenant_id: str, mailbox_id: str) -> Optional[Mailbox]: ...
    def get_mailbox_by_db_id(self, mailbox_db_id: int) -> Optional[Mailbox]: ...
    def find_live_mailbox_by_address(self, email_address: str) -> Optional[Mailbox]: ...
    def activate_watch(
        self, mailbox_db_id: int, *, history_id: int, topic: str, expiration: datetime
    ) -> Mailbox: ...
    def set_mailbox_status(self, mailbox_db_id: int, status: str, *, error: str | None = None) -> None: ...
    def disconnect_mailbox(self, mailbox_db_id: int) -> None: ...
    def commit_history_cursor(self, mailbox_db_id: int, history_id: int) -> None: ...
    def mailboxes_due_for_watch_renewal(self, *, before: datetime) -> list[Mailbox]: ...

    def record_notification(self, notification: GmailNotification) -> Optional[SyncJob]: ...
    def claim_job(self, *, owner: str, lease_seconds: int = 300) -> Optional[SyncJob]: ...
    def get_job(self, job_id: int) -> Optional[SyncJob]: ...
    def complete_job(self, job_id: int) -> None: ...
    def fail_job(self, job_id: int, error: str, *, max_attempts: int = 5) -> None: ...
    def mark_job_needs_full_sync(self, job_id: int) -> None: ...


class InMemorySyncStore:
    """Dict-backed ``SyncStore``. Same semantics as the Postgres one, minus
    durability and cross-process leases."""

    def __init__(self):
        self._lock = threading.RLock()
        self._mailboxes: dict[int, Mailbox] = {}
        self._jobs: dict[int, SyncJob] = {}
        self._oauth_states: dict[str, tuple[OAuthState, Optional[datetime]]] = {}
        self._seen_pubsub_ids: set[str] = set()
        self._next_mailbox_id = 1
        self._next_job_id = 1

    # --- OAuth state -----------------------------------------------------
    def create_oauth_state(self, state: OAuthState) -> None:
        with self._lock:
            if state.state in self._oauth_states:
                raise ValueError("duplicate oauth state")
            self._oauth_states[state.state] = (state, None)

    def consume_oauth_state(self, state: str) -> Optional[OAuthState]:
        """Return the state exactly once, and only before it expires."""
        with self._lock:
            entry = self._oauth_states.get(state)
            if entry is None:
                return None
            record, consumed_at = entry
            if consumed_at is not None or record.expires_at <= utcnow():
                return None
            self._oauth_states[state] = (record, utcnow())
            return record

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
        with self._lock:
            existing = self.get_mailbox(tenant_id, mailbox_id)
            if existing is not None:
                existing.email_address = email_address
                existing.refresh_token_ciphertext = refresh_token_ciphertext
                existing.token_key_id = token_key_id
                existing.status = "pending"
                return existing
            live = self.find_live_mailbox_by_address(email_address)
            if live is not None:
                raise ValueError(f"{email_address} is already connected to another mailbox")
            mailbox = Mailbox(
                id=self._next_mailbox_id,
                tenant_id=tenant_id,
                mailbox_id=mailbox_id,
                email_address=email_address,
                status="pending",
                refresh_token_ciphertext=refresh_token_ciphertext,
                token_key_id=token_key_id,
            )
            self._mailboxes[mailbox.id] = mailbox
            self._next_mailbox_id += 1
            return mailbox

    def get_mailbox(self, tenant_id: str, mailbox_id: str) -> Optional[Mailbox]:
        with self._lock:
            for mailbox in self._mailboxes.values():
                if mailbox.tenant_id == tenant_id and mailbox.mailbox_id == mailbox_id:
                    return mailbox
            return None

    def get_mailbox_by_db_id(self, mailbox_db_id: int) -> Optional[Mailbox]:
        with self._lock:
            return self._mailboxes.get(mailbox_db_id)

    def find_live_mailbox_by_address(self, email_address: str) -> Optional[Mailbox]:
        with self._lock:
            for mailbox in self._mailboxes.values():
                if mailbox.email_address == email_address and mailbox.status != "disconnected":
                    return mailbox
            return None

    def activate_watch(
        self, mailbox_db_id: int, *, history_id: int, topic: str, expiration: datetime
    ) -> Mailbox:
        with self._lock:
            mailbox = self._mailboxes[mailbox_db_id]
            mailbox.watch_topic = topic
            mailbox.watch_expiration = expiration
            # The watch's historyId seeds the cursor only on first connect. A
            # renewal must not rewind (or fast-forward past) a cursor the
            # worker has already committed.
            if mailbox.last_committed_history_id is None:
                mailbox.last_committed_history_id = history_id
            mailbox.status = "active"
            mailbox.last_error = None
            return mailbox

    def set_mailbox_status(self, mailbox_db_id: int, status: str, *, error: str | None = None) -> None:
        with self._lock:
            mailbox = self._mailboxes[mailbox_db_id]
            mailbox.status = status
            mailbox.last_error = error

    def disconnect_mailbox(self, mailbox_db_id: int) -> None:
        with self._lock:
            mailbox = self._mailboxes[mailbox_db_id]
            mailbox.status = "disconnected"
            mailbox.refresh_token_ciphertext = None
            mailbox.token_key_id = None
            mailbox.watch_expiration = None
            mailbox.watch_topic = None

    def commit_history_cursor(self, mailbox_db_id: int, history_id: int) -> None:
        with self._lock:
            mailbox = self._mailboxes[mailbox_db_id]
            current = mailbox.last_committed_history_id
            # Numeric max: an out-of-order or replayed job never rewinds a
            # cursor that has already moved further forward.
            if current is None or history_id > current:
                mailbox.last_committed_history_id = history_id

    def mailboxes_due_for_watch_renewal(self, *, before: datetime) -> list[Mailbox]:
        with self._lock:
            return [
                mailbox
                for mailbox in self._mailboxes.values()
                if mailbox.status in ("active", "needs_full_sync")
                and mailbox.watch_expiration is not None
                and mailbox.watch_expiration <= before
            ]

    # --- Jobs ------------------------------------------------------------
    def record_notification(self, notification: GmailNotification) -> Optional[SyncJob]:
        """Dedup the Pub/Sub message, then create or coalesce one pending job.

        Returns None when the notification is a redelivery or the address has
        no live mailbox -- both are "ack, no work", never an error.
        """
        with self._lock:
            if notification.pubsub_message_id in self._seen_pubsub_ids:
                return None
            self._seen_pubsub_ids.add(notification.pubsub_message_id)
            mailbox = self.find_live_mailbox_by_address(notification.email_address)
            if mailbox is None:
                return None
            for job in self._jobs.values():
                if job.mailbox_db_id == mailbox.id and job.status == "pending":
                    job.requested_history_id = max(job.requested_history_id, notification.history_id)
                    return job
            job = SyncJob(
                id=self._next_job_id,
                mailbox_db_id=mailbox.id,
                tenant_id=mailbox.tenant_id,
                mailbox_id=mailbox.mailbox_id,
                requested_history_id=notification.history_id,
                status="pending",
                needs_full_sync=mailbox.status == "needs_full_sync",
            )
            self._jobs[job.id] = job
            self._next_job_id += 1
            return job

    def claim_job(self, *, owner: str, lease_seconds: int = 300) -> Optional[SyncJob]:
        with self._lock:
            now = utcnow()
            for job in sorted(self._jobs.values(), key=lambda j: j.id):
                claimable = job.status == "pending" or (
                    job.status == "running" and job.leased_until is not None and job.leased_until <= now
                )
                if not claimable:
                    continue
                job.status = "running"
                job.attempts += 1
                job.lease_owner = owner
                job.leased_until = now + timedelta(seconds=lease_seconds)
                return job
            return None

    def get_job(self, job_id: int) -> Optional[SyncJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def complete_job(self, job_id: int) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "done"
            job.completed_at = utcnow()
            job.leased_until = None
            job.last_error = None

    def fail_job(self, job_id: int, error: str, *, max_attempts: int = 5) -> None:
        """Release the job for retry; the mailbox cursor is never touched here.

        If a newer notification already queued a pending job for this mailbox
        while this one was running, fold this job's work into that one rather
        than returning to pending -- at most one pending job per mailbox is the
        invariant the Postgres partial unique index enforces.
        """
        with self._lock:
            job = self._jobs[job_id]
            job.last_error = error
            job.leased_until = None
            job.lease_owner = None
            if job.attempts >= max_attempts:
                job.status = "failed"
                job.completed_at = utcnow()
                return
            successor = next(
                (
                    other
                    for other in self._jobs.values()
                    if other.mailbox_db_id == job.mailbox_db_id and other.status == "pending"
                ),
                None,
            )
            if successor is not None:
                successor.requested_history_id = max(
                    successor.requested_history_id, job.requested_history_id
                )
                successor.needs_full_sync = successor.needs_full_sync or job.needs_full_sync
                job.status = "failed"
                job.completed_at = utcnow()
                return
            job.status = "pending"

    def mark_job_needs_full_sync(self, job_id: int) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.needs_full_sync = True
