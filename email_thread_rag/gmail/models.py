"""Plain records shared by the store, webhook, and worker.

History IDs are ``int`` everywhere in Python and ``numeric(20,0)`` in
Postgres. They are never compared as strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


MailboxStatus = str  # pending | active | needs_full_sync | disconnected | error
JobStatus = str  # pending | running | done | failed


@dataclass
class Mailbox:
    id: int
    tenant_id: str
    mailbox_id: str
    email_address: str
    status: MailboxStatus = "pending"
    last_committed_history_id: Optional[int] = None
    watch_topic: Optional[str] = None
    watch_expiration: Optional[datetime] = None
    last_error: Optional[str] = None
    # Ciphertext only. The plaintext refresh token exists solely inside
    # TokenCipher.decrypt()'s caller; it is never a field on this record, so it
    # cannot reach a repr, a log line, or a response body by accident.
    refresh_token_ciphertext: Optional[bytes] = None
    token_key_id: Optional[str] = None

    def __repr__(self) -> str:  # pragma: no cover - trivial
        # Explicit: the default dataclass repr would print the ciphertext into
        # any log line or pytest failure that formats a Mailbox.
        return (
            f"Mailbox(id={self.id!r}, tenant_id={self.tenant_id!r}, "
            f"mailbox_id={self.mailbox_id!r}, email_address={self.email_address!r}, "
            f"status={self.status!r}, last_committed_history_id={self.last_committed_history_id!r})"
        )


@dataclass
class SyncJob:
    id: int
    mailbox_db_id: int
    tenant_id: str
    mailbox_id: str
    requested_history_id: int
    status: JobStatus = "pending"
    attempts: int = 0
    needs_full_sync: bool = False
    leased_until: Optional[datetime] = None
    lease_owner: Optional[str] = None
    last_error: Optional[str] = None
    completed_at: Optional[datetime] = None


@dataclass
class OAuthState:
    state: str
    tenant_id: str
    mailbox_id: str
    code_verifier: str
    redirect_uri: str
    expires_at: datetime


@dataclass
class GmailNotification:
    """Decoded Pub/Sub push payload."""

    email_address: str
    history_id: int
    pubsub_message_id: str
