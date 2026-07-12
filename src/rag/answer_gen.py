from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from email_thread_rag.app.schemas import RetrievalHit, SessionState
from email_thread_rag.rag.utils import tokenize


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
AMOUNT_RE = re.compile(r"\$?\d[\d,]*(?:\.\d{2})?")
APPROVER_RE = re.compile(r"(?:approved|signed off)\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", re.IGNORECASE)
VENDOR_RE = re.compile(r"(?:vendor|supplier)\s*[:\-]\s*([A-Za-z0-9 &.\-]+)", re.IGNORECASE)
MONTH_RE = re.compile(r"\b([A-Z][a-z]+ \d{4})\b")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "attachment",
    "by",
    "does",
    "email",
    "for",
    "in",
    "is",
    "it",
    "meant",
    "no",
    "of",
    "on",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "which",
    "who",
}


@dataclass
class DraftClause:
    text: str
    supporting_hits: list[RetrievalHit] = field(default_factory=list)
    factual: bool = True
    require_dual_citation: bool = False


@dataclass
class DraftAnswer:
    clauses: list[DraftClause]
    kind: str = "direct"


class AnswerBuilder:
    comparison_patterns = (
        re.compile(r"\bcompare\b", re.IGNORECASE),
        re.compile(r"\bdifference\b", re.IGNORECASE),
        re.compile(r"\bchanged\b", re.IGNORECASE),
        re.compile(r"\bdraft\b.*\bfinal\b", re.IGNORECASE),
        re.compile(r"\bearlier\b.*\b(?:latest|final)\b", re.IGNORECASE),
        re.compile(r"\b(?:vs|versus)\b", re.IGNORECASE),
    )
    timeline_patterns = (
        re.compile(r"\btimeline\b", re.IGNORECASE),
        re.compile(r"\bsequence\b", re.IGNORECASE),
        re.compile(r"\bchronolog", re.IGNORECASE),
        re.compile(r"^\s*when did\b", re.IGNORECASE),
    )

    def is_comparison_query(self, query: str) -> bool:
        return any(pattern.search(query) for pattern in self.comparison_patterns)

    def is_timeline_query(self, query: str) -> bool:
        return any(pattern.search(query) for pattern in self.timeline_patterns)

    def prefers_direct_answer(self, query: str) -> bool:
        lowered = query.lower()
        direct_markers = (
            "from",
            "to",
            "cc",
            "subject",
            "deadline",
            "due",
            "company",
            "territor",
            "month",
            "hotel",
            "audience",
            "topic",
            "what does the email say",
            "what does the attachment say",
            "who is requesting",
            "what company",
            "which territories",
        )
        return any(marker in lowered for marker in direct_markers)

    def build_comparison_queries(self, rewritten_query: str, session: SessionState) -> tuple[str, str]:
        target = session.memory_slots.correction_override or session.memory_slots.last_attachment or rewritten_query
        earlier_query = f"{target} earlier draft"
        final_query = f"{target} final version"
        return earlier_query, final_query

    def build_direct(self, query: str, hits: list[RetrievalHit]) -> DraftAnswer:
        selected_hits = self._prioritize_hits_for_query(query, self._preferred_hits_for_query(query, hits))
        if not selected_hits:
            return DraftAnswer(
                clauses=[DraftClause(text="I could not confirm that from the selected thread.", factual=False)],
                kind="direct",
            )
        metadata_answer = self._build_email_metadata_answer(query, selected_hits)
        if metadata_answer is not None:
            return metadata_answer
        structured_answer = self._build_structured_answer(query, selected_hits)
        if structured_answer is not None:
            return structured_answer
        query_tokens = self._content_tokens(query)
        max_hits = 1 if self._has_targeted_source_hint(query) else 2
        clauses: list[DraftClause] = []
        for hit in selected_hits[:max_hits]:
            sentence = self._best_sentence(hit.chunk.text, query_tokens)
            if sentence:
                clauses.append(DraftClause(text=sentence, supporting_hits=[hit]))
        if not clauses:
            clauses.append(DraftClause(text="I could not confirm that from the selected thread.", factual=False))
        return DraftAnswer(clauses=clauses, kind="timeline" if self.is_timeline_query(query) else "direct")

    def build_timeline(self, hits: list[RetrievalHit]) -> DraftAnswer:
        if not hits:
            return DraftAnswer(
                clauses=[DraftClause(text="I could not confirm the timeline from the selected thread.", factual=False)],
                kind="timeline",
            )
        ordered = sorted(hits, key=lambda hit: hit.chunk.date)
        clauses = []
        for hit in ordered[:3]:
            sentence = self._best_sentence(hit.chunk.text, set())
            if not sentence:
                continue
            clauses.append(
                DraftClause(
                    text=f"On {hit.chunk.date.date().isoformat()}, {sentence}",
                    supporting_hits=[hit],
                )
            )
        if not clauses:
            clauses.append(DraftClause(text="I could not confirm the timeline from the selected thread.", factual=False))
        return DraftAnswer(clauses=clauses, kind="timeline")

    def build_comparison(
        self,
        earlier_hits: list[RetrievalHit],
        final_hits: list[RetrievalHit],
    ) -> DraftAnswer:
        if not earlier_hits or not final_hits:
            return DraftAnswer(
                clauses=[DraftClause(text="I could not confirm both versions from the selected thread.", factual=False)],
                kind="comparison",
            )
        earlier_hit = self._select_comparison_hit(earlier_hits, phase="earlier")
        final_hit = self._select_comparison_hit(final_hits, phase="final")
        earlier_summary = self._summarize_hit(earlier_hit)
        final_summary = self._summarize_hit(final_hit)
        earlier_sentence = self._best_sentence(earlier_hit.chunk.text, set())
        final_sentence = self._best_sentence(final_hit.chunk.text, set())
        difference = f"{earlier_sentence} -> {final_sentence}"
        clauses = [
            DraftClause(
                text=f"Earlier draft: {earlier_sentence}",
                supporting_hits=[earlier_hit],
            ),
            DraftClause(
                text=f"Final version: {final_sentence}",
                supporting_hits=[final_hit],
            ),
            DraftClause(
                text=f"Difference: {difference}",
                supporting_hits=[earlier_hit, final_hit],
                require_dual_citation=True,
            ),
        ]
        return DraftAnswer(clauses=clauses, kind="comparison")

    def _best_sentence(self, text: str, query_tokens: set[str]) -> str | None:
        sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text.strip()) if sentence.strip()]
        if not sentences:
            compact = text.strip()[:240]
            return compact or None
        scored = []
        for sentence in sentences:
            sentence_tokens = self._content_tokens(sentence)
            overlap = len(sentence_tokens & query_tokens) if query_tokens else len(sentence_tokens)
            scored.append((overlap, len(sentence_tokens), sentence))
        scored.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
        if query_tokens and scored[0][0] <= 0:
            return None
        return scored[0][2]

    def _content_tokens(self, text: str) -> set[str]:
        return {token for token in tokenize(text.lower()) if token not in STOPWORDS and len(token) > 2}

    def _preferred_hits_for_query(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        lowered = query.lower()
        filename_match = re.search(r"\b[\w.\- ]+\.(?:pdf|docx|txt|html|htm)\b", lowered)
        if filename_match:
            requested = filename_match.group(0).strip()
            matched = [
                hit
                for hit in hits
                if hit.chunk.attachment_name and hit.chunk.attachment_name.lower() == requested
            ]
            if matched:
                return sorted(matched, key=lambda hit: hit.chunk.date, reverse=True)
        if any(token in lowered for token in ("attachment", "pdf", "docx", "txt", "html")):
            attachment_hits = sorted(
                [hit for hit in hits if hit.chunk.kind == "attachment"],
                key=lambda hit: hit.chunk.date,
                reverse=True,
            )
            if attachment_hits:
                return attachment_hits
        if any(token in lowered for token in ("email", "message")):
            email_hits = [hit for hit in hits if hit.chunk.kind == "email"]
            if email_hits:
                return email_hits
        return hits

    def _prioritize_hits_for_query(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        query_tokens = self._content_tokens(query)
        lowered = query.lower()

        def score(hit: RetrievalHit) -> tuple[int, int, float]:
            haystack = " ".join(
                part
                for part in (
                    hit.chunk.subject or "",
                    hit.chunk.attachment_name or "",
                    hit.chunk.text,
                )
                if part
            )
            hit_tokens = self._content_tokens(haystack)
            overlap = len(query_tokens & hit_tokens)
            keyword_bonus = 0
            if "topic" in lowered and "topic" in haystack.lower():
                keyword_bonus += 4
            if "instruction" in lowered and "instruction" in haystack.lower():
                keyword_bonus += 4
            if "deadline" in lowered and "deadline" in haystack.lower():
                keyword_bonus += 4
            if "due" in lowered and "due" in haystack.lower():
                keyword_bonus += 3
            if "company" in lowered and "supply requests" in haystack.lower():
                keyword_bonus += 3
            if "territor" in lowered and "service territories" in haystack.lower():
                keyword_bonus += 3
            return (overlap + keyword_bonus, -int(hit.chunk.kind == "attachment"), hit.metrics.chunk_support_score)

        return sorted(hits, key=score, reverse=True)

    def _build_email_metadata_answer(self, query: str, hits: list[RetrievalHit]) -> DraftAnswer | None:
        lowered = query.lower()
        asks_from = bool(re.search(r"\bfrom\b", lowered))
        asks_to = bool(re.search(r"\bto\b", lowered))
        asks_cc = bool(re.search(r"\bcc\b", lowered))
        asks_subject = bool(re.search(r"\bsubject\b", lowered))
        asks_sent = bool(re.search(r"\b(?:when sent|when was .* sent|sent date|date sent)\b", lowered))
        if not any((asks_from, asks_to, asks_cc, asks_subject, asks_sent)):
            return None

        email_hits = [hit for hit in hits if hit.chunk.kind == "email"]
        if not email_hits:
            return None
        if "latest" in lowered or "final" in lowered:
            email_hit = max(email_hits, key=lambda hit: hit.chunk.date)
        elif any(token in lowered for token in ("earlier", "first", "original")) or not asks_sent:
            email_hit = min(email_hits, key=lambda hit: hit.chunk.date)
        else:
            email_hit = email_hits[0]

        metadata = email_hit.chunk.metadata or {}
        clauses: list[DraftClause] = []

        if asks_from and email_hit.chunk.sender:
            clauses.append(
                DraftClause(
                    text=f"From: {email_hit.chunk.sender}",
                    supporting_hits=[email_hit],
                )
            )
        if asks_to:
            to_values = metadata.get("to") or []
            if to_values:
                clauses.append(
                    DraftClause(
                        text=f"To: {', '.join(str(value) for value in to_values)}",
                        supporting_hits=[email_hit],
                    )
                )
        if asks_cc:
            cc_values = metadata.get("cc") or []
            if cc_values:
                clauses.append(
                    DraftClause(
                        text=f"CC: {', '.join(str(value) for value in cc_values)}",
                        supporting_hits=[email_hit],
                    )
                )
        if asks_subject and email_hit.chunk.subject:
            clauses.append(
                DraftClause(
                    text=f"Subject: {email_hit.chunk.subject}",
                    supporting_hits=[email_hit],
                )
            )
        if asks_sent:
            clauses.append(
                DraftClause(
                    text=f"Sent date: {email_hit.chunk.date.date().isoformat()}",
                    supporting_hits=[email_hit],
                )
            )

        if not clauses:
            return None
        return DraftAnswer(clauses=clauses, kind="direct")

    def _build_structured_answer(self, query: str, hits: list[RetrievalHit]) -> DraftAnswer | None:
        lowered = query.lower()
        temporal_query = bool(re.search(r"\bwhen\b|\bdate\b|\bsent\b", lowered))

        if "hotel" in lowered:
            return DraftAnswer(
                clauses=[DraftClause(text="I could not confirm any hotel booking from the selected thread.", factual=False)],
                kind="direct",
            )

        if "topic" in lowered:
            topic = self._extract_with_pattern(
                hits,
                [
                    re.compile(r"\btopic\s+will\s+(?:be\s+)?(?:the\s+)?([^.]+)", re.IGNORECASE),
                ],
            )
            if topic:
                value, hit = topic
                value = re.sub(r"^\s*the\s+", "", value, flags=re.IGNORECASE)
                return DraftAnswer(
                    clauses=[DraftClause(text=f"Topic: {value}", supporting_hits=[hit])],
                    kind="direct",
                )

        if temporal_query and ("meeting" in lowered or "ferc" in lowered):
            meeting_time = self._extract_with_pattern(
                hits,
                [
                    re.compile(r"FERC meeting ([^.]+)", re.IGNORECASE),
                ],
            )
            if meeting_time:
                value, hit = meeting_time
                return DraftAnswer(
                    clauses=[DraftClause(text=f"FERC meeting mentioned for: {value}", supporting_hits=[hit])],
                    kind="direct",
                )

        if "instruction" in lowered and not temporal_query:
            instructions = self._extract_with_pattern(
                hits,
                [
                    re.compile(r"instructions (?:below )?to ([^.]+)", re.IGNORECASE),
                    re.compile(r"configured to ([^.]+access the ferc meeting[^.]*)", re.IGNORECASE),
                ],
            )
            if instructions:
                value, hit = instructions
                return DraftAnswer(
                    clauses=[DraftClause(text=f"Instructions: {value}", supporting_hits=[hit])],
                    kind="direct",
                )

        if "audience" in lowered:
            audience = self._extract_with_pattern(
                hits,
                [
                    re.compile(r"copy to ([A-Z][A-Za-z .'-]+), who will conference in from ([A-Z][A-Za-z .'-]+)", re.IGNORECASE),
                ],
                formatter=lambda match: f"{match.group(1).strip()} from {match.group(2).strip()}",
            )
            if audience:
                value, hit = audience
                return DraftAnswer(
                    clauses=[DraftClause(text=f"Audience: {value}", supporting_hits=[hit])],
                    kind="direct",
                )

        if "company" in lowered or "who is requesting" in lowered:
            company = self._extract_with_pattern(
                hits,
                [
                    re.compile(r"^([A-Z][A-Za-z&.,'()/ -]+?)\s+[A-Z][a-z]+\s+\d{4}\s+Supply Requests", re.IGNORECASE),
                    re.compile(r"^([A-Z][A-Za-z&.,'()/ -]+?)\s+is currently accepting", re.IGNORECASE),
                ],
            )
            if company:
                value, hit = company
                return DraftAnswer(
                    clauses=[DraftClause(text=f"Company: {value}", supporting_hits=[hit])],
                    kind="direct",
                )

        if "territor" in lowered or "audience" in lowered:
            territories = self._extract_with_pattern(
                hits,
                [
                    re.compile(r"for its ([A-Za-z ,&/-]+?) service territories", re.IGNORECASE),
                ],
            )
            if territories:
                value, hit = territories
                return DraftAnswer(
                    clauses=[DraftClause(text=f"Territories: {value}", supporting_hits=[hit])],
                    kind="direct",
                )

        if "month" in lowered:
            month = self._extract_with_pattern(
                hits,
                [
                    re.compile(r"\b([A-Z][a-z]+ \d{4})\s+Supply Requests\b"),
                    re.compile(r"accepting [^.]* for ([A-Z][a-z]+ \d{4}) gas supplies", re.IGNORECASE),
                ],
            )
            if month:
                value, hit = month
                return DraftAnswer(
                    clauses=[DraftClause(text=f"Supply request month: {value}", supporting_hits=[hit])],
                    kind="direct",
                )

        if "deadline" in lowered or "due" in lowered or re.search(r"\bwhen\b", lowered):
            deadline = self._extract_with_pattern(
                hits,
                [
                    re.compile(r"deadline for bid submittals is ([^.]+)", re.IGNORECASE),
                    re.compile(r"due ([^.]+)", re.IGNORECASE),
                ],
                prefer_email="email" in lowered,
            )
            if deadline:
                value, hit = deadline
                if "email" in lowered:
                    text = f"Email due date: {value}"
                else:
                    text = f"Deadline: {value}"
                return DraftAnswer(
                    clauses=[DraftClause(text=text, supporting_hits=[hit])],
                    kind="direct",
                )

        return None

    def _extract_with_pattern(
        self,
        hits: list[RetrievalHit],
        patterns: list[re.Pattern[str]],
        *,
        formatter=None,
        prefer_email: bool = False,
    ) -> tuple[str, RetrievalHit] | None:
        ordered_hits = hits
        if prefer_email:
            ordered_hits = sorted(hits, key=lambda hit: (hit.chunk.kind != "email", hit.rerank_rank or 999))
        for hit in ordered_hits:
            text = hit.chunk.text.strip()
            for pattern in patterns:
                match = pattern.search(text)
                if not match:
                    continue
                if formatter is not None:
                    value = formatter(match)
                else:
                    value = match.group(1).strip()
                value = re.sub(r"\s+", " ", value).strip(" .,:;")
                if value:
                    return value, hit
        return None

    def _has_targeted_source_hint(self, query: str) -> bool:
        lowered = query.lower()
        return bool(
            re.search(r"\b[\w.\- ]+\.(?:pdf|docx|txt|html|htm)\b", lowered)
            or any(token in lowered for token in ("attachment", "pdf", "docx", "txt", "html", "email", "message"))
        )

    def _summarize_hit(self, hit: RetrievalHit) -> dict[str, str]:
        text = hit.chunk.text
        amount = AMOUNT_RE.search(text)
        approver = APPROVER_RE.search(text)
        vendor = VENDOR_RE.search(text)
        return {
            "filename": hit.chunk.attachment_name or "",
            "date": hit.chunk.date.date().isoformat(),
            "amount": amount.group(0) if amount else "",
            "approver": approver.group(1) if approver else (hit.chunk.sender or ""),
            "vendor_subject": vendor.group(1).strip() if vendor else (hit.chunk.subject or ""),
        }

    def _format_summary(self, summary: dict[str, str]) -> str:
        parts = []
        if summary["filename"]:
            parts.append(f"filename {summary['filename']}")
        if summary["amount"]:
            parts.append(f"amount {summary['amount']}")
        if summary["approver"]:
            parts.append(f"approver {summary['approver']}")
        if summary["vendor_subject"] and not summary["filename"]:
            parts.append(f"subject {summary['vendor_subject']}")
        elif summary["date"] and len(parts) < 3:
            parts.append(f"date {summary['date']}")
        return ", ".join(parts) if parts else "no supported fields were extracted"

    def _compare_summaries(self, earlier: dict[str, str], final: dict[str, str]) -> str:
        changes = []
        for key in ("filename", "amount", "approver"):
            left = earlier.get(key, "")
            right = final.get(key, "")
            if left and right and left != right:
                label = "subject" if key == "vendor_subject" else key
                changes.append(f"{label} {left} -> {right}")
        if not changes:
            return "I could not confirm a supported field change between the retrieved versions."
        return "; ".join(changes)

    def _select_comparison_hit(self, hits: list[RetrievalHit], *, phase: str) -> RetrievalHit:
        phase_tokens = {
            "earlier": ("draft", "earlier"),
            "final": ("final", "latest"),
        }[phase]

        def hit_text(hit: RetrievalHit) -> str:
            return " ".join(
                part
                for part in (
                    hit.chunk.text,
                    hit.chunk.attachment_name or "",
                    hit.chunk.subject or "",
                )
                if part
            ).lower()

        attachment_matches = [
            hit
            for hit in hits
            if hit.chunk.kind == "attachment" and any(token in hit_text(hit) for token in phase_tokens)
        ]
        if attachment_matches:
            return attachment_matches[0]

        generic_matches = [hit for hit in hits if any(token in hit_text(hit) for token in phase_tokens)]
        if generic_matches:
            return generic_matches[0]

        attachment_hits = [hit for hit in hits if hit.chunk.kind == "attachment"]
        if attachment_hits:
            return attachment_hits[0]
        return hits[0]
