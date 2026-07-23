"""LLM (medha) query rewrite: the injected OpenAI-compatible provider resolves a
follow-up, and every failure mode falls back to the deterministic rule-based
draft. All with a fake provider -- no model, no network."""

from __future__ import annotations

from datetime import datetime, timezone

from email_thread_rag.app.schemas import MemorySlots, SessionState, Turn
from email_thread_rag.config import Settings
from email_thread_rag.rag.rewrite import QueryRewriter


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _session() -> SessionState:
    return SessionState(
        session_id="s1",
        thread_id="thread-alpha",
        created_at=_now(),
        updated_at=_now(),
        recent_turns=[
            Turn(role="user", text="What is the Project Phoenix budget?", timestamp=_now()),
            Turn(role="assistant", text="The Project Phoenix budget is 475000.", timestamp=_now()),
        ],
        memory_slots=MemorySlots(current_focus="Project Phoenix budget"),
    )


class FakeProvider:
    def __init__(self, output):
        self.output = output
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return self.output


def test_medha_rewrite_resolves_followup():
    provider = FakeProvider("Who approved the Project Phoenix budget?")
    rewriter = QueryRewriter(Settings(), provider)
    result = rewriter.rewrite("Who approved it?", _session())
    assert result.query == "Who approved the Project Phoenix budget?"
    assert result.mode.endswith("+medha")
    assert provider.calls == 1


def test_medha_provider_error_falls_back_to_rules():
    class Boom:
        def generate(self, messages):
            raise RuntimeError("medha down")

    rewriter = QueryRewriter(Settings(), Boom())
    result = rewriter.rewrite("Who approved it?", _session())
    assert result.mode == "rules"
    # rule-based pronoun swap still substitutes the focus for "it"
    assert "Project Phoenix budget" in result.query


def test_medha_unusable_output_falls_back_to_rules():
    # An output that just echoes the question is rejected by _is_usable_rewrite.
    provider = FakeProvider("Who approved it?")
    rewriter = QueryRewriter(Settings(), provider)
    result = rewriter.rewrite("Who approved it?", _session())
    assert result.mode == "rules"


def test_no_provider_is_rules_only():
    rewriter = QueryRewriter(Settings(), None)
    result = rewriter.rewrite("Who approved it?", _session())
    assert result.mode == "rules"
