from __future__ import annotations

import re
from dataclasses import dataclass

from email_thread_rag.app.schemas import Citation, ClauseValidation, MetricsResponse, RetrievalHit
from email_thread_rag.rag.answer import DraftAnswer, DraftClause
from email_thread_rag.rag.utils import tokenize


AMOUNT_RE = re.compile(r"\$?\d[\d,]*(?:\.\d{2})?")
DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s+\d{4})?)\b",
    re.IGNORECASE,
)
FILENAME_RE = re.compile(r"\b[\w.\- ]+\.(?:pdf|docx|txt|html|htm)\b", re.IGNORECASE)
PERSON_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")


@dataclass
class ValidationResult:
    answer: str
    citations: list[Citation]
    clause_validations: list[ClauseValidation]
    metrics: MetricsResponse


def _token_overlap_f1(clause_text: str, evidence_text: str) -> float:
    clause_tokens = tokenize(clause_text.lower())
    if not clause_tokens:
        return 0.0
    clause_set = set(clause_tokens)
    fragments = re.split(r"(?<=[.!?])\s+|\n+", evidence_text)
    best = 0.0
    for fragment in fragments:
        evidence_tokens = tokenize(fragment.lower())
        if not evidence_tokens:
            continue
        evidence_set = set(evidence_tokens)
        overlap = len(clause_set & evidence_set)
        if overlap == 0:
            continue
        precision = overlap / len(clause_set)
        recall = overlap / len(evidence_set)
        if precision + recall == 0:
            continue
        score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def _extract_entities(text: str) -> set[str]:
    entities: set[str] = set()
    for amount in AMOUNT_RE.findall(text):
        normalized = amount.replace("$", "").replace(",", "").strip().lower()
        if normalized:
            entities.add(normalized)
    for date in DATE_RE.findall(text):
        normalized = date.strip().lower()
        if normalized:
            entities.add(normalized)
    for filename in FILENAME_RE.findall(text):
        normalized = filename.strip().lower()
        if normalized:
            entities.add(normalized)
    for person in PERSON_RE.findall(text):
        normalized = person.strip().lower()
        if normalized:
            entities.add(normalized)
    return entities


def _entity_value_match(clause_text: str, evidence_text: str) -> float:
    clause_entities = _extract_entities(clause_text)
    if not clause_entities:
        return 1.0
    evidence_entities = _extract_entities(evidence_text)
    matches = len(clause_entities & evidence_entities)
    return matches / len(clause_entities)


def _format_citation(hit: RetrievalHit, clause_text: str, support_score: float) -> Citation:
    page_no = hit.chunk.page_no if hit.chunk.kind == "attachment" else None
    if page_no is not None:
        formatted = f"[msg: {hit.chunk.message_id}, page: {page_no}]"
    else:
        formatted = f"[msg: {hit.chunk.message_id}]"
    return Citation(
        message_id=hit.chunk.message_id,
        page_no=page_no,
        chunk_id=hit.chunk.chunk_id,
        clause_text=clause_text,
        clause_support_score=support_score,
        formatted=formatted,
    )


class CitationValidator:
    support_threshold = 0.65

    def _support_text(self, hit: RetrievalHit) -> str:
        metadata_lines: list[str] = []
        for key, value in (hit.chunk.metadata or {}).items():
            if value in (None, "", [], {}):
                continue
            if isinstance(value, list):
                rendered = ", ".join(str(item) for item in value if str(item).strip())
            else:
                rendered = str(value)
            if rendered:
                metadata_lines.append(f"{key}: {rendered}")
        metadata_bits = [
            hit.chunk.text,
            hit.chunk.attachment_name or "",
            hit.chunk.subject or "",
            f"from: {hit.chunk.sender}" if hit.chunk.sender else "",
            hit.chunk.sender or "",
            hit.chunk.date.date().isoformat(),
            *metadata_lines,
        ]
        return "\n".join(bit for bit in metadata_bits if bit)

    def validate(self, draft: DraftAnswer, default_hits: list[RetrievalHit]) -> ValidationResult:
        rendered_lines: list[str] = []
        all_citations: list[Citation] = []
        clause_validations: list[ClauseValidation] = []
        factual_count = 0
        kept_factual = 0
        support_scores: list[float] = []

        for clause in draft.clauses:
            if not clause.factual:
                rendered_lines.append(clause.text)
                clause_validations.append(ClauseValidation(clause_text=clause.text, kept=True))
                continue

            factual_count += 1
            content_text = clause.text.split(":", 1)[-1].strip()
            candidates = clause.supporting_hits or default_hits
            scored_hits: list[tuple[float, float, float, RetrievalHit]] = []
            for hit in candidates:
                support_text = self._support_text(hit)
                overlap_f1 = _token_overlap_f1(content_text, support_text)
                entity_match = _entity_value_match(content_text, support_text)
                support_score = 0.6 * overlap_f1 + 0.4 * entity_match
                scored_hits.append((support_score, overlap_f1, entity_match, hit))
            scored_hits.sort(key=lambda item: item[0], reverse=True)

            needed = 2 if clause.require_dual_citation else 1
            if clause.require_dual_citation:
                kept_hits = scored_hits[:needed]
                if len(kept_hits) < needed:
                    clause_validations.append(
                        ClauseValidation(
                            clause_text=clause.text,
                            kept=False,
                            support_score=scored_hits[0][0] if scored_hits else 0.0,
                            token_overlap_f1=scored_hits[0][1] if scored_hits else 0.0,
                            entity_value_match=scored_hits[0][2] if scored_hits else 0.0,
                        )
                    )
                    continue
                combined_text = " ".join(self._support_text(hit) for _, _, _, hit in kept_hits)
                combined_overlap = _token_overlap_f1(content_text, combined_text)
                combined_entity = _entity_value_match(content_text, combined_text)
                combined_support = 0.6 * combined_overlap + 0.4 * combined_entity
                if combined_support < self.support_threshold:
                    clause_validations.append(
                        ClauseValidation(
                            clause_text=clause.text,
                            kept=False,
                            support_score=combined_support,
                            token_overlap_f1=combined_overlap,
                            entity_value_match=combined_entity,
                        )
                    )
                    continue
            else:
                kept_hits = [item for item in scored_hits if item[0] >= self.support_threshold][:needed]
            if len(kept_hits) < needed:
                clause_validations.append(
                    ClauseValidation(
                        clause_text=clause.text,
                        kept=False,
                        support_score=scored_hits[0][0] if scored_hits else 0.0,
                        token_overlap_f1=scored_hits[0][1] if scored_hits else 0.0,
                        entity_value_match=scored_hits[0][2] if scored_hits else 0.0,
                    )
                )
                continue

            kept_factual += 1
            if clause.require_dual_citation:
                support_score = combined_support
                avg_overlap = combined_overlap
                avg_entity_match = combined_entity
            else:
                support_score = sum(item[0] for item in kept_hits) / len(kept_hits)
                avg_overlap = sum(item[1] for item in kept_hits) / len(kept_hits)
                avg_entity_match = sum(item[2] for item in kept_hits) / len(kept_hits)
            support_scores.append(support_score)
            citations = [_format_citation(hit, clause.text, score) for score, _, _, hit in kept_hits]
            clause_validations.append(
                ClauseValidation(
                    clause_text=clause.text,
                    kept=True,
                    support_score=support_score,
                    citations=citations,
                    token_overlap_f1=avg_overlap,
                    entity_value_match=avg_entity_match,
                )
            )
            all_citations.extend(citations)
            rendered_lines.append(f"{clause.text} {' '.join(citation.formatted for citation in citations)}")

        citation_coverage = kept_factual / factual_count if factual_count else 0.0
        if factual_count and citation_coverage < 0.70:
            answer = "I could not support enough of that answer from the selected evidence."
        else:
            separator = "\n" if draft.kind == "comparison" else " "
            answer = separator.join(rendered_lines).strip()
        metrics = MetricsResponse(
            answer_support_score=(sum(support_scores) / len(support_scores)) if support_scores else 0.0,
            citation_coverage=citation_coverage,
            evidence_count=len({citation.chunk_id for citation in all_citations}),
        )
        return ValidationResult(
            answer=answer,
            citations=all_citations,
            clause_validations=clause_validations,
            metrics=metrics,
        )
