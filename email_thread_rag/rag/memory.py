from __future__ import annotations

import re

from email_thread_rag.app.schemas import RetrievalHit, SessionState


AMOUNT_RE = re.compile(r"\$?\d[\d,]*(?:\.\d{2})?")
DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s+\d{4})?)\b",
    re.IGNORECASE,
)
FILENAME_RE = re.compile(r"\b[\w.\- ]+\.(?:pdf|docx|txt|html|htm)\b", re.IGNORECASE)
MESSAGE_ID_RE = re.compile(r"<[^>]+>")
PEOPLE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


class MemoryManager:
    def update_from_user_text(self, session: SessionState, text: str) -> SessionState:
        slots = session.memory_slots
        follow_up = self._is_follow_up(text)
        slots.amounts = _dedupe_keep_order(slots.amounts + AMOUNT_RE.findall(text))
        slots.dates = _dedupe_keep_order(slots.dates + DATE_RE.findall(text))
        slots.filenames = _dedupe_keep_order(slots.filenames + FILENAME_RE.findall(text))
        slots.message_ids = _dedupe_keep_order(slots.message_ids + MESSAGE_ID_RE.findall(text))
        slots.people = _dedupe_keep_order(slots.people + PEOPLE_RE.findall(text))
        detected_intent = self.detect_intent(text)
        if not (follow_up and detected_intent in {"time", "general"} and slots.last_user_intent):
            slots.last_user_intent = detected_intent

        lowered = text.lower()
        if any(word in lowered for word in ("compare", "difference", "changed", "draft", "final")):
            slots.comparison_target = text.strip()

        correction = self.detect_correction(text, session)
        if correction:
            slots.correction_override = correction
            slots.current_focus = correction
        elif follow_up:
            slots.current_focus = (
                slots.correction_override
                or slots.last_answer_focus
                or slots.last_attachment
                or slots.last_subject
                or slots.current_focus
            )
        else:
            explicit_focus = self.extract_focus(text)
            if explicit_focus:
                slots.current_focus = explicit_focus
        return session

    def update_from_hits(self, session: SessionState, hits: list[RetrievalHit]) -> SessionState:
        if not hits:
            return session
        slots = session.memory_slots
        top = hits[0].chunk
        attachment_hits = [hit for hit in hits if hit.chunk.attachment_name]
        attachment_names = [hit.chunk.attachment_name for hit in attachment_hits if hit.chunk.attachment_name]
        if attachment_names:
            newest_attachment = max(attachment_hits, key=lambda hit: hit.chunk.date)
            slots.last_attachment = newest_attachment.chunk.attachment_name
            slots.filenames = _dedupe_keep_order(slots.filenames + attachment_names)
            slots.current_focus = newest_attachment.chunk.attachment_name
        if top.subject:
            slots.last_subject = top.subject
            if not slots.current_focus:
                slots.current_focus = top.subject
        slots.message_ids = _dedupe_keep_order(slots.message_ids + [hit.chunk.message_id for hit in hits])
        slots.people = _dedupe_keep_order(slots.people + [hit.chunk.sender for hit in hits if hit.chunk.sender])
        return session

    def update_from_answer(self, session: SessionState, answer: str) -> SessionState:
        slots = session.memory_slots
        decision_match = re.search(r"\b(approved|rejected|finalized|signed off|selected)\b", answer, re.IGNORECASE)
        if decision_match:
            slots.last_decision = decision_match.group(1).lower()
        answer_focus = self.extract_answer_focus(answer)
        if answer_focus:
            slots.last_answer_focus = answer_focus
            slots.current_focus = answer_focus
        return session

    def detect_intent(self, text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in ("compare", "difference", "changed", "draft", "final")):
            return "comparison"
        if "instruction" in lowered:
            return "instructions"
        if "topic" in lowered:
            return "topic"
        if "audience" in lowered:
            return "audience"
        if "deadline" in lowered or "due" in lowered:
            return "deadline"
        if "company" in lowered or "who is requesting" in lowered:
            return "company"
        if "territor" in lowered:
            return "territories"
        if "month" in lowered:
            return "month"
        if "from" in lowered or "to" in lowered or "cc" in lowered or "subject" in lowered:
            return "metadata"
        if "when" in lowered or "date" in lowered or "sent" in lowered:
            return "time"
        return "general"

    def extract_focus(self, text: str) -> str | None:
        cleaned = text.strip().rstrip("?.!")
        lowered = cleaned.lower()

        patterns = [
            r"what were the (?P<focus>.+?) about$",
            r"what was the (?P<focus>.+?) about$",
            r"what does the (?P<focus>attachment|email|pdf|docx|document) say$",
            r"what company is requesting bids(?: in the attachment)?$",
            r"which (?P<focus>territories) are mentioned(?: in the attachment)?$",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered, re.IGNORECASE)
            if match:
                focus = match.groupdict().get("focus")
                if focus:
                    return focus.strip()

        if lowered.startswith(("and ", "when ", "who ", "what about")):
            return None

        noun_match = re.search(r"\b(?:about|regarding|for)\s+(.+)$", cleaned, re.IGNORECASE)
        if noun_match:
            return noun_match.group(1).strip()
        return None

    def extract_answer_focus(self, answer: str) -> str | None:
        cleaned = re.sub(r"\s*\[msg:[^\]]+\]", "", answer).strip()
        if not cleaned or cleaned.lower().startswith("i could not confirm"):
            return None
        prefix_match = re.match(
            r"^(Instructions|Topic|Audience|Company|Territories|Supply request month|Deadline|Email due date|From|To|CC|Subject|Sent date|FERC meeting mentioned for):\s*(.+)$",
            cleaned,
            re.IGNORECASE,
        )
        if prefix_match:
            return prefix_match.group(2).strip()
        first_clause = cleaned.split(".")[0].strip()
        return first_clause or None

    def _is_follow_up(self, text: str) -> bool:
        lowered = text.lower().strip()
        follow_up_patterns = (
            r"^and\s+\w+",
            r"^(what|when|who)\s+about\b",
            r"^(and when|and who|and where)\??$",
            r"^(no,\s*)?i meant\b",
        )
        return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in follow_up_patterns)

    def detect_correction(self, text: str, session: SessionState) -> str | None:
        lowered = text.lower().strip()
        patterns = [
            r"^(?:no,\s*)?i meant (?P<target>.+)$",
            r"^not that (?P<target>.+)$",
            r"^the (?P<target>earlier attachment|earlier email|latest attachment|final version)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered, re.IGNORECASE)
            if match:
                return match.group("target").strip()
        if "pdf" in lowered:
            filenames = [name for name in session.memory_slots.filenames if name.lower().endswith(".pdf")]
            if filenames:
                return filenames[-1]
            return "pdf"
        if "docx" in lowered:
            filenames = [name for name in session.memory_slots.filenames if name.lower().endswith(".docx")]
            if filenames:
                return filenames[-1]
        return None
