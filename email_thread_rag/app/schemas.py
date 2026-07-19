from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


ChunkKind = Literal["email", "attachment"]
SourceType = Literal["eml", "enron_archive", "fixture", "gmail", "manual"]


class AttachmentPage(BaseModel):
    page_no: int
    text: str
    ocr_used: bool = False
    text_density: float = 0.0
    alnum_count: int = 0


class AttachmentRecord(BaseModel):
    attachment_id: str
    message_id: str
    thread_id: str
    filename: str
    media_type: str
    source_path: str
    pages: List[AttachmentPage] = Field(default_factory=list)


class EmailRecord(BaseModel):
    doc_id: str
    message_id: str
    thread_id: str
    date: datetime
    sender: str
    to: List[str] = Field(default_factory=list)
    cc: List[str] = Field(default_factory=list)
    subject: str
    body_text: str
    attachment_ids: List[str] = Field(default_factory=list)
    in_reply_to: Optional[str] = None
    references: List[str] = Field(default_factory=list)
    source_path: str
    source_type: SourceType = "manual"
    # Stage-1 email-native segmentation (additive; None until segmented). Only
    # ``authored_text`` feeds the normal retrieval path; the rest is for audit.
    authored_text: Optional[str] = None
    quoted_text: Optional[str] = None
    signature_text: Optional[str] = None
    disclaimer_text: Optional[str] = None


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    thread_id: str
    message_id: str
    kind: ChunkKind
    attachment_name: Optional[str] = None
    page_no: Optional[int] = None
    sender: Optional[str] = None
    date: datetime
    subject: Optional[str] = None
    text: str
    # Compact retrieval text (headers + exact authored text) used for BM25/vector
    # indexing. Defaults to ``text`` for old fixtures/records lacking it, so the
    # public contract stays backward compatible.
    embed_text: Optional[str] = None
    # Optional citation provenance: offsets into the normalized authored body.
    source_start: Optional[int] = None
    source_end: Optional[int] = None
    ocr_used: bool = False
    token_count: int
    source_path: str
    source_type: SourceType = "manual"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _default_embed_text(self) -> "ChunkRecord":
        if self.embed_text is None:
            self.embed_text = self.text
        return self


class RetrievalMetrics(BaseModel):
    bm25_score_raw: float = 0.0
    bm25_score_norm: float = 0.0
    dense_score_raw: float = 0.0
    dense_score_norm: float = 0.0
    rrf_score: float = 0.0
    rerank_score_raw: float = 0.0
    rerank_score_norm: float = 0.0
    chunk_support_score: float = 0.0


class RetrievalHit(BaseModel):
    chunk: ChunkRecord
    metrics: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    retrieval_rank: Optional[int] = None
    rerank_rank: Optional[int] = None
    source_lists: List[str] = Field(default_factory=list)


class Citation(BaseModel):
    message_id: str
    page_no: Optional[int] = None
    chunk_id: str
    clause_text: str
    clause_support_score: float
    formatted: str


AnswerStatus = Literal["answered", "abstained"]


class AnswerCitation(BaseModel):
    """A Stage-7 grounded citation. ``quote`` is copied verbatim from the cited
    chunk's clean authored ``ChunkRecord.text`` (never embed_text, headers,
    metadata, graph prose, or quoted history), and ``quote_start``/``quote_end``
    are its exact offsets into that clean text."""

    chunk_id: str
    message_id: str
    quote: str
    quote_start: int
    quote_end: int
    # Stage-8 attachment citation identity (None for email-body chunks). An
    # OCR-derived quote (``ocr_used=True``) is labeled as such and must never be
    # presented as byte-perfect original document text.
    page_no: Optional[int] = None
    attachment_name: Optional[str] = None
    ocr_used: bool = False
    extraction_method: Optional[str] = None


class AnswerClaim(BaseModel):
    """One factual claim in a grounded answer, with the citations that validated
    it. Every claim that survives validation has >=1 valid citation."""

    text: str
    citations: List[AnswerCitation] = Field(default_factory=list)


class AnswerResult(BaseModel):
    """Stage-7 grounded answering outcome. Either an ``answered`` result whose
    every claim is citation-backed by clean source text, or an explicit
    ``abstained`` result -- never an unsupported answer. ``trace`` is body-free
    (routes, candidate counts, validation rule, attempt count); it never carries
    email text."""

    status: AnswerStatus
    answer: str
    claims: List[AnswerClaim] = Field(default_factory=list)
    citations: List[AnswerCitation] = Field(default_factory=list)
    attempts: int = 0
    abstain_reason: Optional[str] = None
    trace: Dict[str, Any] = Field(default_factory=dict)


class ClauseValidation(BaseModel):
    clause_text: str
    kept: bool
    support_score: float = 0.0
    citations: List[Citation] = Field(default_factory=list)
    token_overlap_f1: float = 0.0
    entity_value_match: float = 0.0


class MemorySlots(BaseModel):
    people: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    amounts: List[str] = Field(default_factory=list)
    filenames: List[str] = Field(default_factory=list)
    message_ids: List[str] = Field(default_factory=list)
    current_focus: Optional[str] = None
    last_user_intent: Optional[str] = None
    last_answer_focus: Optional[str] = None
    last_attachment: Optional[str] = None
    last_subject: Optional[str] = None
    last_decision: Optional[str] = None
    comparison_target: Optional[str] = None
    correction_override: Optional[str] = None


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    timestamp: datetime


class SessionState(BaseModel):
    session_id: str
    thread_id: str
    created_at: datetime
    updated_at: datetime
    recent_turns: List[Turn] = Field(default_factory=list)
    memory_slots: MemorySlots = Field(default_factory=MemorySlots)


class AskRequest(BaseModel):
    session_id: str
    text: str
    search_outside_thread: bool = False


class StartSessionRequest(BaseModel):
    thread_id: str


class SwitchThreadRequest(BaseModel):
    session_id: str
    thread_id: str


class ResetSessionRequest(BaseModel):
    session_id: str


class MetricsResponse(BaseModel):
    answer_support_score: float = 0.0
    citation_coverage: float = 0.0
    evidence_count: int = 0
    top_thread_support_score: float = 0.0


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    rewrite: str
    rewrite_mode: str
    retrieved: List[RetrievalHit]
    trace_id: str
    outside_thread_used: bool = False
    metrics: MetricsResponse = Field(default_factory=MetricsResponse)
    # Stage-7: "answered" or "abstained" when grounded answering is enabled;
    # None on the default deterministic answer path.
    answer_status: Optional[AnswerStatus] = None


class TraceRecord(BaseModel):
    trace_id: str
    session_id: str
    thread_id: str
    user_text: str
    rewrite: str
    rewrite_mode: str
    retrieved_items: List[Dict[str, Any]]
    fused_ranking: List[Dict[str, Any]]
    reranked_items: List[Dict[str, Any]]
    used_chunks: List[str]
    final_answer: str
    citations: List[Dict[str, Any]]
    latency_ms: float
    token_counts: Dict[str, int] = Field(default_factory=dict)
    flags: Dict[str, Any] = Field(default_factory=dict)
    clause_validations: List[ClauseValidation] = Field(default_factory=list)
    metrics: MetricsResponse = Field(default_factory=MetricsResponse)
    fallback_trigger_reason: Optional[str] = None
