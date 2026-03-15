from __future__ import annotations


def test_thread_scope_stays_inside_active_thread_by_default(test_engine, session_id):
    outcome = test_engine.ask(session_id, "Who approved the Phoenix invoice?", search_outside_thread=False)
    assert outcome.response.outside_thread_used is False
    assert "Carol Finance" not in outcome.response.answer

