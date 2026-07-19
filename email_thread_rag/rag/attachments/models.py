"""Plain records shared by the attachment store, sink, and worker."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional


@dataclass
class AttachmentMeta:
    """PDF attachment metadata lifted from a Gmail message part (no bytes yet)."""

    gmail_attachment_id: str
    filename: str
    media_type: str
    byte_size: Optional[int] = None

    @property
    def attachment_id(self) -> str:
        """Stable canonical id that page chunk ids are namespaced under."""
        return self.gmail_attachment_id


@dataclass
class StoredAttachment:
    id: int
    tenant_id: str
    mailbox_id: str
    message_id: str
    thread_id: Optional[str]
    gmail_attachment_id: str
    attachment_id: str
    filename: str
    media_type: str
    byte_size: Optional[int] = None
    content_hash: Optional[str] = None
    extraction_status: str = "pending"
    extraction_method: Optional[str] = None
    extraction_error: Optional[str] = None


@dataclass
class AttachmentJob:
    id: int
    attachment_db_id: int
    tenant_id: str
    mailbox_id: str
    attachment_id: str
    extraction_input_hash: str
    status: str = "pending"
    attempts: int = 0
    leased_until: object = None
    lease_owner: Optional[str] = None
    last_error: Optional[str] = None
    error_rule: Optional[str] = None
    completed_at: object = None


def extraction_input_hash(meta: AttachmentMeta) -> str:
    """The queue's idempotency key. Folds the Gmail attachmentId and byte size:
    a re-synced unchanged attachment hashes identically (no duplicate work); a
    changed one (new size) hashes differently and re-extracts, replacing stale
    page chunks."""
    basis = f"{meta.gmail_attachment_id}|{meta.byte_size if meta.byte_size is not None else ''}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()
