from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterable
from uuid import uuid4

from email_thread_rag.app.schemas import AskResponse, ClauseValidation, MetricsResponse, RetrievalHit, TraceRecord
from email_thread_rag.app.sessions import SessionStore
from email_thread_rag.config import Settings, get_settings
from email_thread_rag.rag.answer import AnswerBuilder, DraftAnswer
from email_thread_rag.rag.citation_validator import CitationValidator, ValidationResult
from email_thread_rag.rag.memory import MemoryManager
from email_thread_rag.rag.retrieval import HybridRetriever, RetrievalResult
from email_thread_rag.rag.rewrite import QueryRewriter
from email_thread_rag.rag.utils import append_jsonl


@dataclass
class AskOutcome:
    response: AskResponse
    trace: TraceRecord


class RAGEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        session_store: SessionStore | None = None,
        retriever: HybridRetriever | None = None,
        memory_manager: MemoryManager | None = None,
        rewriter: QueryRewriter | None = None,
        answer_builder: AnswerBuilder | None = None,
        citation_validator: CitationValidator | None = None,
    ):
        self.settings = settings or get_settings()
        self.session_store = session_store or SessionStore(self.settings)
        self.retriever = retriever or HybridRetriever.from_chunk_store(self.settings)
        self.memory_manager = memory_manager or MemoryManager()
        self.rewriter = rewriter or QueryRewriter(self.settings)
        self.answer_builder = answer_builder or AnswerBuilder()
        self.citation_validator = citation_validator or CitationValidator()
        self.run_dir = self.settings.runs_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir / "trace.jsonl"

    def available_threads(self) -> list[str]:
        return self.retriever.available_threads()

    def ask(self, session_id: str, user_text: str, *, search_outside_thread: bool) -> AskOutcome:
        started = perf_counter()
        trace_id = str(uuid4())
        session = self.session_store.get(session_id)
        session = self.memory_manager.update_from_user_text(session, user_text)

        rewrite_result = self.rewriter.rewrite(user_text, session)
        comparison = self.answer_builder.is_comparison_query(user_text)

        in_thread_payload = self._retrieve_and_answer(
            rewritten_query=rewrite_result.query,
            session=session,
            comparison=comparison,
            thread_id=session.thread_id,
        )
        top_thread_support_score = (
            max((hit.metrics.chunk_support_score for hit in in_thread_payload["hits"]), default=0.0)
            if in_thread_payload["hits"]
            else 0.0
        )
        outside_thread_used = False
        fallback_trigger_reason = None
        final_payload = in_thread_payload

        if search_outside_thread:
            fallback_trigger_reason = self._outside_thread_reason(
                in_thread_payload["hits"],
                in_thread_payload["validation"],
            )
            if fallback_trigger_reason:
                final_payload = self._retrieve_and_answer(
                    rewritten_query=rewrite_result.query,
                    session=session,
                    comparison=comparison,
                    thread_id=None,
                )
                outside_thread_used = True
                second_reason = self._outside_thread_reason(final_payload["hits"], final_payload["validation"])
                if second_reason:
                    final_payload["validation"].answer = "I could not confirm that even after searching outside the active thread."

        validation = final_payload["validation"]
        validation.metrics.top_thread_support_score = top_thread_support_score
        session = self.memory_manager.update_from_hits(session, final_payload["hits"])
        session = self.memory_manager.update_from_answer(session, validation.answer)
        self.session_store.save(session)
        self.session_store.append_turn(session_id, "user", user_text)
        self.session_store.append_turn(session_id, "assistant", validation.answer)

        response = AskResponse(
            answer=validation.answer,
            citations=validation.citations,
            rewrite=rewrite_result.query,
            rewrite_mode=rewrite_result.mode,
            retrieved=final_payload["hits"],
            trace_id=trace_id,
            outside_thread_used=outside_thread_used,
            metrics=validation.metrics,
        )
        trace = self._build_trace(
            trace_id=trace_id,
            session_id=session_id,
            thread_id=session.thread_id,
            user_text=user_text,
            rewrite_query=rewrite_result.query,
            rewrite_mode=rewrite_result.mode,
            retrieval_payload=final_payload,
            validation=validation,
            latency_ms=(perf_counter() - started) * 1000.0,
            token_counts=rewrite_result.token_counts,
            search_outside_thread=search_outside_thread,
            outside_thread_used=outside_thread_used,
            fallback_trigger_reason=fallback_trigger_reason,
        )
        append_jsonl(self.trace_path, trace.model_dump(mode="json"))
        return AskOutcome(response=response, trace=trace)

    def _outside_thread_reason(self, hits: list[RetrievalHit], validation: ValidationResult) -> str | None:
        thresholds = self.settings.retrieval_thresholds
        supported_hits = [hit for hit in hits if hit.metrics.chunk_support_score >= thresholds.min_chunk_support]
        top_rerank = max((hit.metrics.rerank_score_norm for hit in hits), default=0.0)
        if len(supported_hits) < thresholds.min_supported_chunks:
            return "insufficient_supported_chunks"
        if top_rerank < thresholds.min_top_rerank:
            return "top_rerank_below_threshold"
        if validation.metrics.citation_coverage < thresholds.min_citation_coverage:
            return "citation_coverage_below_threshold"
        return None

    def _retrieve_and_answer(
        self,
        *,
        rewritten_query: str,
        session,
        comparison: bool,
        thread_id: str | None,
    ) -> dict[str, object]:
        if comparison:
            earlier_query, final_query = self.answer_builder.build_comparison_queries(rewritten_query, session)
            earlier_result = self.retriever.search(earlier_query, thread_id=thread_id)
            final_result = self.retriever.search(final_query, thread_id=thread_id)
            draft = self.answer_builder.build_comparison(earlier_result.reranked_hits, final_result.reranked_hits)
            hits = self._unique_hits(earlier_result.reranked_hits + final_result.reranked_hits)
            validation = self.citation_validator.validate(draft, hits)
            return {
                "draft": draft,
                "validation": validation,
                "hits": hits,
                "results": [
                    {"role": "earlier", "result": earlier_result},
                    {"role": "final", "result": final_result},
                ],
            }

        result = self.retriever.search(rewritten_query, thread_id=thread_id)
        draft = (
            self.answer_builder.build_timeline(result.reranked_hits)
            if self.answer_builder.is_timeline_query(rewritten_query)
            else self.answer_builder.build_direct(rewritten_query, result.reranked_hits)
        )
        validation = self.citation_validator.validate(draft, result.reranked_hits)
        return {
            "draft": draft,
            "validation": validation,
            "hits": result.reranked_hits,
            "results": [{"role": "main", "result": result}],
        }

    def _unique_hits(self, hits: Iterable[RetrievalHit]) -> list[RetrievalHit]:
        seen: set[str] = set()
        unique: list[RetrievalHit] = []
        for hit in hits:
            if hit.chunk.chunk_id in seen:
                continue
            seen.add(hit.chunk.chunk_id)
            unique.append(hit)
        return unique

    def _flatten_result(self, role: str, result: RetrievalResult, attr: str) -> list[dict]:
        hits = getattr(result, attr)
        flattened = []
        for hit in hits:
            flattened.append(
                {
                    "role": role,
                    "chunk_id": hit.chunk.chunk_id,
                    "thread_id": hit.chunk.thread_id,
                    "message_id": hit.chunk.message_id,
                    "page_no": hit.chunk.page_no,
                    "kind": hit.chunk.kind,
                    "scores": hit.metrics.model_dump(),
                }
            )
        return flattened

    def _build_trace(
        self,
        *,
        trace_id: str,
        session_id: str,
        thread_id: str,
        user_text: str,
        rewrite_query: str,
        rewrite_mode: str,
        retrieval_payload: dict[str, object],
        validation: ValidationResult,
        latency_ms: float,
        token_counts: dict[str, int],
        search_outside_thread: bool,
        outside_thread_used: bool,
        fallback_trigger_reason: str | None,
    ) -> TraceRecord:
        result_entries = retrieval_payload["results"]
        retrieved_items: list[dict] = []
        fused_ranking: list[dict] = []
        reranked_items: list[dict] = []
        for entry in result_entries:
            role = entry["role"]
            result = entry["result"]
            retrieved_items.extend(self._flatten_result(role, result, "bm25_hits"))
            retrieved_items.extend(self._flatten_result(role, result, "dense_hits"))
            fused_ranking.extend(self._flatten_result(role, result, "fused_hits"))
            reranked_items.extend(self._flatten_result(role, result, "reranked_hits"))

        return TraceRecord(
            trace_id=trace_id,
            session_id=session_id,
            thread_id=thread_id,
            user_text=user_text,
            rewrite=rewrite_query,
            rewrite_mode=rewrite_mode,
            retrieved_items=retrieved_items,
            fused_ranking=fused_ranking,
            reranked_items=reranked_items,
            used_chunks=[hit.chunk.chunk_id for hit in retrieval_payload["hits"]],
            final_answer=validation.answer,
            citations=[citation.model_dump() for citation in validation.citations],
            latency_ms=latency_ms,
            token_counts=token_counts,
            flags={
                "search_outside_thread": search_outside_thread,
                "use_cloud_rewrite": self.settings.enable_cloud_rewrite,
                "ocr_used_anywhere": any(hit.chunk.ocr_used for hit in retrieval_payload["hits"]),
                "outside_thread_used": outside_thread_used,
            },
            clause_validations=validation.clause_validations,
            metrics=validation.metrics,
            fallback_trigger_reason=fallback_trigger_reason,
        )
