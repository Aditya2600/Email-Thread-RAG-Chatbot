from __future__ import annotations

from email_thread_rag.rag.rewrite import QueryRewriter


def test_rule_based_rewrite_fallback_activates_on_model_failure(sample_records, monkeypatch):
    settings, _, _, _ = sample_records
    rewriter = QueryRewriter(settings)
    monkeypatch.setattr(rewriter, "_load_model", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    from email_thread_rag.app.sessions import SessionStore

    session = SessionStore(settings).start_session("thread-alpha")
    session.memory_slots.last_attachment = "budget_final.pdf"
    result = rewriter.rewrite("compare it", session)
    assert result.mode == "rules"
    assert "budget_final.pdf" in result.query


def test_rule_based_rewrite_fallback_activates_on_prompt_leak(sample_records, monkeypatch):
    settings, _, _, _ = sample_records
    rewriter = QueryRewriter(settings)

    class FakeIds(list):
        @property
        def shape(self):
            return (1, len(self[0]))

    class FakeTokenizer:
        def __call__(self, prompt, return_tensors="pt", truncation=True):
            return {"input_ids": FakeIds([[1, 2, 3]])}

        def decode(self, *_args, **_kwargs):
            return "user: compare it assistant: earlier answer"

    class FakeModel:
        def generate(self, **_kwargs):
            return [[1, 2, 3]]

    monkeypatch.setattr(rewriter, "_load_model", lambda: (FakeTokenizer(), FakeModel()))

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

    class FakeIds(list):
        @property
        def shape(self):
            return (1, len(self[0]))

    class FakeTokenizer:
        def __call__(self, prompt, return_tensors="pt", truncation=True):
            return {"input_ids": FakeIds([[1, 2, 3]])}

        def decode(self, *_args, **_kwargs):
            return "amount in budget_final.pdf"

    class FakeModel:
        def generate(self, **_kwargs):
            return FakeIds([[1, 2, 3]])

    monkeypatch.setattr(rewriter, "_load_model", lambda: (FakeTokenizer(), FakeModel()))
    monkeypatch.setattr(
        rewriter,
        "_rewrite_with_gemini",
        lambda user_text, session, local_query: "approved amount in budget_final.pdf",
    )

    from email_thread_rag.app.sessions import SessionStore

    session = SessionStore(settings).start_session("thread-alpha")
    result = rewriter.rewrite("What is it?", session)
    assert result.mode == "t5+gemini"
    assert result.query == "approved amount in budget_final.pdf"


def test_local_rewrite_preserves_instruction_anchor(sample_records, monkeypatch):
    settings, _, _, _ = sample_records
    rewriter = QueryRewriter(settings)

    class FakeIds(list):
        @property
        def shape(self):
            return (1, len(self[0]))

    class FakeTokenizer:
        def __call__(self, prompt, return_tensors="pt", truncation=True):
            return {"input_ids": FakeIds([[1, 2, 3]])}

        def decode(self, *_args, **_kwargs):
            return "ferc meetings"

    class FakeModel:
        def generate(self, **_kwargs):
            return FakeIds([[1, 2, 3]])

    monkeypatch.setattr(rewriter, "_load_model", lambda: (FakeTokenizer(), FakeModel()))

    from email_thread_rag.app.sessions import SessionStore

    session = SessionStore(settings).start_session("allen-p:instructions for ferc meetings")
    result = rewriter.rewrite("What were the instructions about?", session)
    assert result.mode == "t5"
    assert result.query == "instructions ferc meetings"
