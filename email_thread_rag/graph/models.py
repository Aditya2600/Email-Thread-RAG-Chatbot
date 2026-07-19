"""Plain records for the graph-extraction queue and the resolved graph rows.

``ExtractionInput`` is the *whole* of what an extraction is derived from: what is
in here is what the fingerprint covers, and therefore what a change to
invalidates a running job. Adding a field here without adding it to
``extraction_hash_of`` would silently weaken the stale-job guard.

The ``Resolved*`` records are what the worker hands the store *after* it has
located every LLM evidence string in the chunk's own ``text`` and derived offsets
itself. By the time a record is a ``Resolved*`` its offsets are proven to index
real chunk text; the raw LLM output never reaches the store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

JobStatus = str  # pending | running | done | failed


@dataclass(frozen=True)
class ExtractionInput:
    """Everything an extraction is allowed to be derived from, and nothing else.

    Only clean chunk ``text``, safe email metadata, and deterministic thread
    links. Never quoted history, signature/disclaimer, ``embed_text``, or the
    Stage-4 context prefix.
    """

    chunk_id: str
    text: str
    subject: Optional[str] = None
    sender: Optional[str] = None
    thread_id: Optional[str] = None
    recipients: tuple[str, ...] = ()
    cc: tuple[str, ...] = ()
    in_reply_to: Optional[str] = None
    parent_sender: Optional[str] = None


@dataclass
class GraphJob:
    id: int
    chunk_db_id: int
    tenant_id: str
    mailbox_id: str
    chunk_id: str
    extraction_input_hash: str
    status: JobStatus = "pending"
    attempts: int = 0
    leased_until: Optional[datetime] = None
    lease_owner: Optional[str] = None
    last_error: Optional[str] = None
    error_rule: Optional[str] = None
    completed_at: Optional[datetime] = None


@dataclass
class ChunkGraphState:
    """The chunk fields Stage 5 reads. Deliberately read-only w.r.t. evidence:
    ``text`` and the source offsets are inputs here, never outputs."""

    chunk_db_id: int
    chunk_id: str
    tenant_id: str
    mailbox_id: str
    text: str
    subject: Optional[str] = None
    sender: Optional[str] = None
    thread_id: Optional[str] = None
    date: Optional[datetime] = None
    source_start: Optional[int] = None
    recipients: list[str] = field(default_factory=list)
    cc: list[str] = field(default_factory=list)
    in_reply_to: Optional[str] = None
    parent_sender: Optional[str] = None
    graph_input_hash: Optional[str] = None

    def as_extraction_input(self) -> ExtractionInput:
        return ExtractionInput(
            chunk_id=self.chunk_id,
            text=self.text,
            subject=self.subject,
            sender=self.sender,
            thread_id=self.thread_id,
            recipients=tuple(self.recipients),
            cc=tuple(self.cc),
            in_reply_to=self.in_reply_to,
            parent_sender=self.parent_sender,
        )


# --- resolved rows: offsets already proven against chunk.text ----------------


@dataclass(frozen=True)
class ResolvedMention:
    entity_type: str
    canonical_name: str
    normalized_name: str
    mention_text: str
    chunk_start: int
    chunk_end: int
    source_start: Optional[int]
    source_end: Optional[int]

    @property
    def entity_key(self) -> tuple[str, str]:
        return (self.entity_type, self.normalized_name)


@dataclass(frozen=True)
class ResolvedRelation:
    subject_key: tuple[str, str]
    predicate: str
    object_key: tuple[str, str]
    evidence_kind: str  # 'text' | 'metadata'
    mention_text: Optional[str] = None
    chunk_start: Optional[int] = None
    chunk_end: Optional[int] = None


@dataclass(frozen=True)
class ResolvedFact:
    subject: str
    predicate: str
    object_value: str
    normalized_subject: str
    normalized_predicate: str
    evidence_text: str
    chunk_start: int
    chunk_end: int
    source_start: Optional[int]
    source_end: Optional[int]
    evidence_hash: str
    has_update_cue: bool


@dataclass(frozen=True)
class ResolvedGraph:
    """The complete, evidence-checked result for one chunk. Empty is valid: a
    chunk with no locatable entities produces an empty graph, not an error."""

    mentions: tuple[ResolvedMention, ...] = ()
    relations: tuple[ResolvedRelation, ...] = ()
    facts: tuple[ResolvedFact, ...] = ()

    def is_empty(self) -> bool:
        return not (self.mentions or self.relations or self.facts)
