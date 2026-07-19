from __future__ import annotations

from email_thread_rag.app.sessions import SessionStore
from email_thread_rag.rag.memory import MemoryManager
from email_thread_rag.rag.rewrite import QueryRewriter


def test_session_store_keeps_last_eight_turns(sample_records):
    settings, _, _, _ = sample_records
    store = SessionStore(settings)
    session = store.start_session("thread-alpha")

    for index in range(10):
        role = "user" if index % 2 == 0 else "assistant"
        store.append_turn(session.session_id, role, f"turn {index}")

    saved = store.get(session.session_id)
    assert len(saved.recent_turns) == 8
    assert saved.recent_turns[0].text == "turn 2"
    assert saved.recent_turns[-1].text == "turn 9"


def test_follow_up_rewrite_uses_preserved_focus_for_and_when(sample_records):
    settings, _, _, _ = sample_records
    store = SessionStore(settings)
    memory = MemoryManager()
    rewriter = QueryRewriter(settings)

    session = store.start_session("allen-p:instructions for ferc meetings")
    session = memory.update_from_user_text(session, "What were the instructions about?")
    session.memory_slots.last_subject = "Instructions for FERC Meetings"
    session = memory.update_from_answer(
        session,
        "Instructions: get access to view FERC meetings [msg: 1470ef1f41430243b4f19782582e071a]",
    )
    session = memory.update_from_user_text(session, "And when?")

    assert session.memory_slots.last_user_intent == "instructions"
    assert session.memory_slots.current_focus == "get access to view FERC meetings"

    result = rewriter.rewrite("And when?", session)

    assert result.mode == "rules"
    assert result.query == "When was the FERC meeting mentioned in Instructions for FERC Meetings?"


def test_generic_follow_up_rewrite_falls_back_to_context_aware_rule(sample_records):
    settings, _, _, _ = sample_records
    store = SessionStore(settings)
    memory = MemoryManager()
    rewriter = QueryRewriter(settings)

    session = store.start_session("allen-p:instructions for ferc meetings")
    session = memory.update_from_user_text(session, "What were the instructions about?")
    session.memory_slots.last_subject = "Instructions for FERC Meetings"
    session = memory.update_from_answer(
        session,
        "Instructions: get access to view FERC meetings [msg: 1470ef1f41430243b4f19782582e071a]",
    )
    session = memory.update_from_user_text(session, "And when?")

    result = rewriter.rewrite("And when?", session)

    assert result.mode == "rules"
    assert result.query == "When was the FERC meeting mentioned in Instructions for FERC Meetings?"
