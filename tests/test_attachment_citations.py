from __future__ import annotations


def test_attachment_page_citations_are_preserved(test_engine, session_id):
    outcome = test_engine.ask(session_id, "What amount is in the budget_final.pdf attachment?", search_outside_thread=False)
    assert "[msg: <msg-2@example.com>, page: 1]" in outcome.response.answer
    assert any(citation.page_no == 1 for citation in outcome.response.citations)

