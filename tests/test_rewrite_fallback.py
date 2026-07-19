from __future__ import annotations

from email_thread_rag.rag.rewrite import QueryRewriter


def test_rule_based_rewrite_resolves_pronoun(sample_records):
    settings, _, _, _ = sample_records
    rewriter = QueryRewriter(settings)
    from email_thread_rag.app.sessions import SessionStore

    session = SessionStore(settings).start_session("thread-alpha")
    session.memory_slots.last_attachment = "budget_final.pdf"
    result = rewriter.rewrite("compare it", session)
    assert result.mode == "rules"
    assert "budget_final.pdf" in result.query


def test_gemini_cloud_rewrite_enhances_local_rewrite(sample_records, monkeypatch):
    settings, _, _, _ = sample_records
    settings.enable_cloud_rewrite = True
    settings.cloud_rewrite_provider = "gemini"
    settings.cloud_rewrite_model = "gemini-2.5-flash"
    settings.gemini_api_key = "test-key"
    rewriter = QueryRewriter(settings)
    monkeypatch.setattr(
        rewriter,
        "_rewrite_with_gemini",
        lambda user_text, session, local_query: "approved amount in budget_final.pdf",
    )

    from email_thread_rag.app.sessions import SessionStore

    session = SessionStore(settings).start_session("thread-alpha")
    result = rewriter.rewrite("What is it?", session)
    assert result.mode == "rules+gemini"
    assert result.query == "approved amount in budget_final.pdf"
