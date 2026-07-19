"""Plain records for the contextualization queue.

``ContextInput`` is the *whole* of what the prefix is derived from: whatever is
in here is what the fingerprint covers, and therefore what a change to
invalidates a running job. Adding a field here without adding it to
``fingerprint_of`` would silently weaken the stale-job guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

JobStatus = str  # pending | running | done | failed
ContextMethod = str  # none | deterministic | llm


@dataclass(frozen=True)
class ContextInput:
    """Everything the prefix is allowed to be derived from, and nothing else.

    ``parent_message_id``/``parent_subject`` are populated only when the parent
    is locally available -- Stage 4 does not fetch a parent it does not already
    have, and a missing parent is a normal case, not an error.
    """

    chunk_id: str
    text: str
    subject: Optional[str] = None
    sender: Optional[str] = None
    thread_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    parent_subject: Optional[str] = None


@dataclass
class ContextJob:
    id: int
    chunk_db_id: int
    tenant_id: str
    mailbox_id: str
    chunk_id: str
    context_input_hash: str
    status: JobStatus = "pending"
    attempts: int = 0
    leased_until: Optional[datetime] = None
    lease_owner: Optional[str] = None
    last_error: Optional[str] = None
    completed_at: Optional[datetime] = None


@dataclass
class ChunkContextState:
    """The chunk fields Stage 4 reads and writes. Deliberately narrow: `text`
    and the source offsets are inputs here, never outputs."""

    chunk_db_id: int
    chunk_id: str
    tenant_id: str
    mailbox_id: str
    text: str
    subject: Optional[str] = None
    sender: Optional[str] = None
    thread_id: Optional[str] = None
    date: Optional[datetime] = None
    to: list[str] = field(default_factory=list)
    cc: list[str] = field(default_factory=list)
    in_reply_to: Optional[str] = None
    parent_subject: Optional[str] = None
    context_prefix: Optional[str] = None
    context_method: Optional[str] = None
    context_version: Optional[str] = None
    context_input_hash: Optional[str] = None

    def as_context_input(self) -> ContextInput:
        return ContextInput(
            chunk_id=self.chunk_id,
            text=self.text,
            subject=self.subject,
            sender=self.sender,
            thread_id=self.thread_id,
            parent_message_id=self.in_reply_to,
            parent_subject=self.parent_subject,
        )
