from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from email_thread_rag.app.schemas import SessionState
from email_thread_rag.config import Settings


@dataclass
class RewriteResult:
    query: str
    mode: str
    token_counts: dict[str, int]


class QueryRewriter:
    INTENT_ANCHORS: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"\binstructions?\b", re.IGNORECASE), "instructions"),
        (re.compile(r"\btopic\b", re.IGNORECASE), "topic"),
        (re.compile(r"\baudience\b", re.IGNORECASE), "audience"),
        (re.compile(r"\bdeadline\b", re.IGNORECASE), "deadline"),
        (re.compile(r"\bdue\b", re.IGNORECASE), "due"),
        (re.compile(r"\bcompany\b", re.IGNORECASE), "company"),
        (re.compile(r"\bterritor(?:y|ies)\b", re.IGNORECASE), "territories"),
        (re.compile(r"\bmonth\b", re.IGNORECASE), "month"),
        (re.compile(r"\bhotel\b", re.IGNORECASE), "hotel"),
        (re.compile(r"\bfrom\b", re.IGNORECASE), "from"),
        (re.compile(r"\bto\b", re.IGNORECASE), "to"),
        (re.compile(r"\bcc\b", re.IGNORECASE), "cc"),
        (re.compile(r"\bsubject\b", re.IGNORECASE), "subject"),
        (re.compile(r"\battachment\b", re.IGNORECASE), "attachment"),
        (re.compile(r"\bemail\b", re.IGNORECASE), "email"),
        (re.compile(r"\bpdf\b", re.IGNORECASE), "pdf"),
    )

    def __init__(self, settings: Settings):
        self.settings = settings
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.settings.rewrite_model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.settings.rewrite_model_name)
        return self._tokenizer, self._model

    def rewrite(self, user_text: str, session: SessionState) -> RewriteResult:
        fallback_query = self._rule_based_rewrite(user_text, session)
        local_result: RewriteResult
        try:
            tokenizer, model = self._load_model()
            prompt = self._build_prompt(user_text, session)
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
            output = model.generate(**encoded, max_new_tokens=64)
            rewritten = tokenizer.decode(output[0], skip_special_tokens=True).strip()
            if not self._is_usable_rewrite(rewritten, user_text):
                raise ValueError("rewrite was empty or unchanged")
            rewritten = self._preserve_intent_anchors(user_text, rewritten)
            if self._should_prefer_follow_up_fallback(user_text, rewritten, fallback_query):
                raise ValueError("rewrite dropped follow-up intent")
            local_result = RewriteResult(
                query=rewritten,
                mode="t5",
                token_counts={
                    "rewrite_prompt_tokens": int(encoded["input_ids"].shape[-1]),
                    "rewrite_output_tokens": int(output.shape[-1]),
                },
            )
        except Exception:
            local_result = RewriteResult(
                query=fallback_query,
                mode="rules",
                token_counts={"rewrite_prompt_tokens": 0, "rewrite_output_tokens": 0},
            )
        cloud_result = self._maybe_enhance_with_cloud(user_text, session, local_result)
        return cloud_result or local_result

    def _maybe_enhance_with_cloud(
        self,
        user_text: str,
        session: SessionState,
        local_result: RewriteResult,
    ) -> RewriteResult | None:
        if not self.settings.enable_cloud_rewrite:
            return None
        if (self.settings.cloud_rewrite_provider or "").lower() != "gemini":
            return None
        if not self.settings.gemini_api_key:
            return None
        try:
            rewritten = self._rewrite_with_gemini(user_text, session, local_result.query)
            if not self._is_usable_rewrite(rewritten, user_text):
                return None
            rewritten = self._preserve_intent_anchors(user_text, rewritten)
            token_counts = dict(local_result.token_counts)
            token_counts["cloud_rewrite_input_chars"] = len(self._build_cloud_prompt(user_text, session, local_result.query))
            token_counts["cloud_rewrite_output_chars"] = len(rewritten)
            return RewriteResult(
                query=rewritten,
                mode=f"{local_result.mode}+gemini",
                token_counts=token_counts,
            )
        except Exception:
            return None

    def _is_usable_rewrite(self, rewritten: str, user_text: str) -> bool:
        if not rewritten:
            return False
        normalized = rewritten.strip()
        if not normalized or normalized.lower() == user_text.strip().lower():
            return False
        lowered = normalized.lower()
        leaked_markers = (
            "active thread:",
            "recent turns:",
            "current focus:",
            "memory:",
            "question:",
            "rewrite:",
            "user:",
            "assistant:",
            "people=",
            "dates=",
            "amounts=",
            "filenames=",
            "last_attachment=",
            "last_user_intent=",
            "last_answer_focus=",
            "current_focus=",
            "comparison_target=",
            "correction_override=",
        )
        if any(marker in lowered for marker in leaked_markers):
            return False
        if len(normalized) > 240 or len(normalized.split()) > 40:
            return False
        if normalized.count("\n") > 1:
            return False
        if not re.search(r"[a-z0-9]", lowered):
            return False
        return True

    def _build_cloud_prompt(self, user_text: str, session: SessionState, local_query: str) -> str:
        memory = session.memory_slots
        recent_turns = "\n".join(
            f"{turn.role}: {turn.text}"
            for turn in session.recent_turns[-self.settings.rewrite_turn_window :]
        )
        return (
            "Rewrite the request into one concise self-contained retrieval query.\n"
            "Return only the rewritten query, with no explanation.\n"
            f"Active thread: {session.thread_id}\n"
            f"Recent turns:\n{recent_turns}\n"
            f"Current focus: {memory.current_focus}\n"
            "Memory:\n"
            f"people={memory.people}\n"
            f"dates={memory.dates}\n"
            f"amounts={memory.amounts}\n"
            f"filenames={memory.filenames}\n"
            f"last_attachment={memory.last_attachment}\n"
            f"last_subject={memory.last_subject}\n"
            f"last_decision={memory.last_decision}\n"
            f"last_user_intent={memory.last_user_intent}\n"
            f"last_answer_focus={memory.last_answer_focus}\n"
            f"current_focus={memory.current_focus}\n"
            f"comparison_target={memory.comparison_target}\n"
            f"correction_override={memory.correction_override}\n"
            f"Local draft rewrite: {local_query}\n"
            f"User question: {user_text}\n"
            "Rewritten query:"
        )

    def _rewrite_with_gemini(self, user_text: str, session: SessionState, local_query: str) -> str:
        model_name = self.settings.cloud_rewrite_model or "gemini-2.5-flash"
        prompt = self._build_cloud_prompt(user_text, session, local_query)
        response = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
            headers={"x-goog-api-key": self.settings.gemini_api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 64,
                },
            },
            timeout=self.settings.rewrite_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        rewritten = " ".join(part.get("text", "").strip() for part in parts if part.get("text")).strip()
        if not rewritten:
            raise ValueError("Gemini returned empty text")
        return rewritten

    def _build_prompt(self, user_text: str, session: SessionState) -> str:
        memory = session.memory_slots
        recent_turns = "\n".join(
            f"{turn.role}: {turn.text}"
            for turn in session.recent_turns[-self.settings.rewrite_turn_window :]
        )
        return (
            "Rewrite the user request into a self-contained retrieval query.\n"
            f"Active thread: {session.thread_id}\n"
            f"Recent turns:\n{recent_turns}\n"
            f"Current focus: {memory.current_focus}\n"
            "Memory:\n"
            f"people={memory.people}\n"
            f"dates={memory.dates}\n"
            f"amounts={memory.amounts}\n"
            f"filenames={memory.filenames}\n"
            f"last_attachment={memory.last_attachment}\n"
            f"last_subject={memory.last_subject}\n"
            f"last_decision={memory.last_decision}\n"
            f"last_user_intent={memory.last_user_intent}\n"
            f"last_answer_focus={memory.last_answer_focus}\n"
            f"current_focus={memory.current_focus}\n"
            f"comparison_target={memory.comparison_target}\n"
            f"correction_override={memory.correction_override}\n"
            f"Question: {user_text}\n"
            "Rewrite:"
        )

    def _rule_based_rewrite(self, user_text: str, session: SessionState) -> str:
        text = user_text.strip()
        memory = session.memory_slots
        target = (
            memory.correction_override
            or memory.current_focus
            or memory.last_answer_focus
            or memory.last_attachment
            or memory.last_subject
            or memory.last_decision
            or memory.comparison_target
            or session.thread_id
        )

        if re.match(r"^and when\??$", text, re.IGNORECASE):
            focus_text = " ".join(filter(None, [memory.current_focus, memory.last_answer_focus, memory.last_subject]))
            if "instruction" in (memory.last_subject or "").lower() and "ferc" in focus_text.lower():
                return f"When was the FERC meeting mentioned in {memory.last_subject}?"
            if memory.last_user_intent == "deadline" and target:
                return f"When is the deadline for {target}?"
            if target:
                return f"When was {target} mentioned in {memory.last_subject or session.thread_id}?"
            return f"When did {session.thread_id} happen?"
        if re.match(r"^compare it\??$", text, re.IGNORECASE):
            return f"Compare {target} earlier draft versus final version"
        if re.match(r"^and who\??$", text, re.IGNORECASE):
            if target:
                return f"Who is associated with {target} in {memory.last_subject or session.thread_id}?"
            return f"Who is associated with {session.thread_id}?"

        rewritten = text
        if target:
            rewritten = re.sub(r"\b(it|that|this|those)\b", str(target), rewritten, flags=re.IGNORECASE)

        temporal_replacements = {
            r"\bearlier\b": "earlier version",
            r"\bprevious\b": "previous message",
            r"\blatest\b": "latest version",
            r"\bfinal\b": "final version",
        }
        for pattern, replacement in temporal_replacements.items():
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)

        if memory.correction_override:
            rewritten = f"{rewritten} focus on {memory.correction_override}"

        return rewritten

    def _should_prefer_follow_up_fallback(self, user_text: str, rewritten: str, fallback_query: str) -> bool:
        lowered_user = user_text.strip().lower()
        lowered_rewrite = rewritten.lower()
        lowered_fallback = fallback_query.lower()

        explicit_follow_ups = (
            r"^and when\??$",
            r"^and who\??$",
            r"^compare it\??$",
            r"^and where\??$",
        )
        if any(re.match(pattern, lowered_user, re.IGNORECASE) for pattern in explicit_follow_ups):
            if "when" in lowered_user and "when" not in lowered_rewrite and "when" in lowered_fallback:
                return True
            if "who" in lowered_user and "who" not in lowered_rewrite and "who" in lowered_fallback:
                return True
            if "compare" in lowered_user and "compare" not in lowered_rewrite and "compare" in lowered_fallback:
                return True
            if "where" in lowered_user and "where" not in lowered_rewrite and "where" in lowered_fallback:
                return True
        return False

    def _preserve_intent_anchors(self, user_text: str, rewritten: str) -> str:
        anchors = [label for pattern, label in self.INTENT_ANCHORS if pattern.search(user_text)]
        if not anchors:
            return rewritten
        lowered = rewritten.lower()
        missing = [anchor for anchor in anchors if anchor not in lowered]
        if not missing:
            return rewritten
        return " ".join([*missing, rewritten]).strip()
