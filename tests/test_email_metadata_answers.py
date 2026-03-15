from __future__ import annotations


def test_email_header_answer_uses_message_metadata(test_engine, session_id):
    outcome = test_engine.ask(
        session_id,
        "What are the From and To fields in the email message?",
        search_outside_thread=False,
    )
    assert "From: alice@corp.com" in outcome.response.answer
    assert "To: bob@corp.com" in outcome.response.answer
    assert "[msg: <msg-1@example.com>]" in outcome.response.answer
