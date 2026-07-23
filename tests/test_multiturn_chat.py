"""Multi-turn chat at the engine level: fresh retrieval per turn, follow-up
resolution using the prior topic, preserved history, and the additive
original/resolved query fields -- all with fakes, no model or network.

The grounding boundary is proven structurally: the answerer only ever receives
a query string and a thread scope, never a previous answer, so a prior answer
can clarify the query (via the rewriter/memory) but can never enter the
evidence pack.
"""

from __future__ import annotations

from email_thread_rag.app.schemas import AnswerCitation, AnswerClaim, AnswerResult
from email_thread_rag.config import Settings
from email_thread_rag.rag.engine import RAGEngine


class FakeRetriever:
    def available_threads(self):
        return ["thread-alpha"]

    def search(self, *args, **kwargs):  # unused; grounded answerer is faked
        raise AssertionError("retriever.search should not be called in this test")


class FakeGroundedAnswerer:
    """Records every (query, thread_id) it is asked -- one call == one fresh
    retrieval. It never sees a prior answer, so answers can't become evidence."""

    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []

    def answer(self, query, *, thread_id=None):
        self.calls.append((query, thread_id))
        return AnswerResult(
            status="answered",
            answer="The Customer Support Automation budget approved is $840,000.",
            claims=[
                AnswerClaim(
                    text="Approved budget is $840,000.",
                    citations=[
                        AnswerCitation(
                            chunk_id=f"c{len(self.calls)}",
                            message_id="<m1@x>",
                            quote="approved is $840,000",
                            quote_start=0,
                            quote_end=20,
                        )
                    ],
                )
            ],
            citations=[
                AnswerCitation(
                    chunk_id=f"c{len(self.calls)}",
                    message_id="<m1@x>",
                    quote="approved is $840,000",
                    quote_start=0,
                    quote_end=20,
                )
            ],
            attempts=1,
        )


def _engine():
    answerer = FakeGroundedAnswerer()
    engine = RAGEngine(
        Settings(),
        retriever=FakeRetriever(),
        grounded_answerer=answerer,
    )
    return engine, answerer


def test_followup_resolves_history_preserved_and_evidence_is_fresh():
    engine, answerer = _engine()
    session = engine.session_store.start_session("thread-alpha")
    sid = session.session_id

    q1 = "What is the approved budget for Customer Support Automation?"
    q2 = "When does it go live?"

    r1 = engine.ask(sid, q1, search_outside_thread=False)
    r2 = engine.ask(sid, q2, search_outside_thread=False)

    # Fresh retrieval on every turn: one answerer call per question.
    assert len(answerer.calls) == 2
    q1_query, q1_scope = answerer.calls[0]
    q2_query, q2_scope = answerer.calls[1]

    # Follow-up resolved to a standalone query carrying the Q1 topic; "it" is gone.
    assert "Customer Support Automation" in q2_query
    assert q2_query != q2
    assert " it " not in f" {q2_query.lower()} "

    # Authorization applied every turn: retrieval stayed scoped to the mailbox thread.
    assert q1_scope == "thread-alpha" and q2_scope == "thread-alpha"

    # Additive fields: original verbatim, resolved is what actually retrieved.
    assert r1.response.original_query == q1
    assert r2.response.original_query == q2
    assert r2.response.resolved_query == q2_query

    # History preserved: both exchanges still present, oldest first.
    turns = engine.session_store.get(sid).recent_turns
    assert [t.role for t in turns] == ["user", "assistant", "user", "assistant"]
    assert turns[0].text == q1

    # Each turn owns its own citations (distinct freshly-retrieved chunks).
    assert r1.response.citations[0].chunk_id != r2.response.citations[0].chunk_id


def test_single_turn_ask_still_works_with_defaults():
    """Existing single-turn callers keep working; the new fields default safely."""
    engine, answerer = _engine()
    sid = engine.session_store.start_session("thread-alpha").session_id
    r = engine.ask(sid, "What was approved?", search_outside_thread=False)
    assert r.response.answer
    assert r.response.original_query == "What was approved?"
    assert r.response.resolved_query  # non-empty
    assert len(answerer.calls) == 1
