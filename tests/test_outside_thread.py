from __future__ import annotations


def test_search_outside_thread_opt_in_behavior(test_engine, session_id):
    in_thread = test_engine.ask(session_id, "Who approved the Phoenix invoice?", search_outside_thread=False)
    assert in_thread.response.outside_thread_used is False
    assert "Carol Finance" not in in_thread.response.answer

    global_fallback = test_engine.ask(session_id, "Who approved the Phoenix invoice?", search_outside_thread=True)
    assert global_fallback.response.outside_thread_used is True
    assert "Carol Finance" in global_fallback.response.answer

