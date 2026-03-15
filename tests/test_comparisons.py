from __future__ import annotations


def test_comparison_answers_cite_both_sides(test_engine, session_id):
    outcome = test_engine.ask(session_id, "Compare the earlier draft and final version of the budget.", search_outside_thread=False)
    lines = outcome.response.answer.splitlines()
    assert lines[0].startswith("Earlier draft:")
    assert lines[1].startswith("Final version:")
    assert lines[2].startswith("Difference:")
    assert "[msg: <msg-1@example.com>, page: 1]" in lines[0]
    assert "[msg: <msg-2@example.com>, page: 1]" in lines[1]
    assert lines[2].count("[msg:") >= 2

