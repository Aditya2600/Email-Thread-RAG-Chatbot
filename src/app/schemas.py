from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


SourceType = Literal["eml", "enron_archive", "fixture", "manual"]
ChunkSourceType = Literal["email", "attachment"]
QueryIntent = Literal["thread_qa", "comparison", "timeline", "metadata_lookup", "analytics", "abstain"]


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
    pages: list[AttachmentPage] = Field(default_factory=list)


class EmailRecord(BaseModel):
    doc_id: str
    message_id: str
    thread_id: str
    date: datetime
    sender: str
    to: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    subject: str
    body_text: str
    attachment_ids: list[str] = Field(default_factory=list)
    in_reply_to: str | None = None
    references: list[str] = Field(default_factory=list)
    source_path: str
    source_type: SourceType = "manual"


class EmailChunk(BaseModel):
    id: str
    message_id: str
    thread_id: str
    chunk_index: int
    page_number: int | None = None
    source_type: ChunkSourceType
    filename: str | None = None
    text: str
    sender: str
    sent_at: str
    subject: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(EmailChunk):
    bm25_score: float | None = None
    vector_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    final_score: float | None = None


class CitationRef(BaseModel):
    message_id: str
    page_number: int | None = None

    def __str__(self) -> str:
        if self.page_number is not None:
            return f"[msg:{self.message_id}, page:{self.page_number}]"
        return f"[msg:{self.message_id}]"


class GroundingResult(BaseModel):
    clause: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    best_chunk_id: str | None = None


class EntityMemory(BaseModel):
    people: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    last_cited_message_ids: list[str] = Field(default_factory=list)


class Turn(BaseModel):
    role: Literal["user", "assistant", "system_summary"]
    text: str
    timestamp: datetime


class SessionState(BaseModel):
    session_id: str
    thread_id: str
    created_at: datetime
    updated_at: datetime
    recent_turns: list[Turn] = Field(default_factory=list)
    entity_memory: EntityMemory = Field(default_factory=EntityMemory)


class ChatSessionState(BaseModel):
    session_id: str
    thread_id: str
    context_memory: EntityMemory = Field(default_factory=EntityMemory)
    token_count: int = 0
    created_at: datetime
    updated_at: datetime


class AskRequest(BaseModel):
    session_id: str
    text: str
    search_outside_thread: bool = False
    stream: bool = False


class StartSessionRequest(BaseModel):
    thread_id: str


class StartSessionResponse(BaseModel):
    session_id: str
    thread_id: str
    message: str = "Session started"


class SwitchThreadRequest(BaseModel):
    session_id: str
    thread_id: str


class ResetSessionRequest(BaseModel):
    session_id: str


class QueryPlan(BaseModel):
    intent: QueryIntent
    rewritten_query: str
    requires_retrieval: bool = True
    search_outside_thread: bool = False
    filters: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class MetricsResponse(BaseModel):
    answer_support_score: float = 0.0
    citation_coverage: float = 0.0
    evidence_count: int = 0
    top_thread_support_score: float = 0.0


class AnswerResponse(BaseModel):
    answer: str
    citations: list[CitationRef] = Field(default_factory=list)
    grounding_results: list[GroundingResult] = Field(default_factory=list)
    abstained: bool = False
    rewritten_query: str
    trace_id: str
    latency_ms: int


class AskResponse(AnswerResponse):
    query_plan: QueryPlan | None = None
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    outside_thread_used: bool = False
    metrics: MetricsResponse = Field(default_factory=MetricsResponse)


class TraceLog(BaseModel):
    trace_id: str
    timestamp: str
    user_query: str
    rewritten_query: str
    intent: str
    retrieved_chunk_ids: list[str]
    rrf_scores: dict[str, float]
    reranked_chunk_ids: list[str]
    answer_draft: str
    clauses_removed: list[str]
    final_answer: str
    citations: list[CitationRef]
    abstained: bool
    prompt_hash: str
    latency_ms: int


class ThreadSummary(BaseModel):
    thread_id: str
    subject: str | None = None
    message_count: int = 0
    attachment_count: int = 0
    participants: list[str] = Field(default_factory=list)
    first_message_at: datetime | None = None
    last_message_at: datetime | None = None


class InboxStats(BaseModel):
    total_threads: int
    total_messages: int
    total_attachments: int
    last_sync_at: datetime | None = None


class StreamTokenEvent(BaseModel):
    event: Literal["token"] = "token"
    token: str


class StreamDoneEvent(BaseModel):
    event: Literal["done"] = "done"
    response: AskResponse


class StreamErrorEvent(BaseModel):
    event: Literal["error"] = "error"
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    trace_id: str | None = None
