from __future__ import annotations


def test_correction_override_forces_new_retrieval(test_engine, session_id):
    first = test_engine.ask(session_id, "What does the summary email say the approved amount is?", search_outside_thread=False)
    assert "$1200" in first.response.answer

    corrected = test_engine.ask(session_id, "no, I meant the PDF", search_outside_thread=False)
    assert "$1500" in corrected.response.answer
    assert "$1200" not in corrected.response.answer

