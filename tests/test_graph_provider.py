"""The graph provider seam. All fakes; no model download, no real endpoint call."""

from __future__ import annotations

import pytest

from email_thread_rag.config import Settings
from email_thread_rag.graph.fakes import (
    ExplodingGraphProvider,
    FakeGraphProvider,
    UnavailableGraphProvider,
    graph_json,
)
from email_thread_rag.graph.models import ExtractionInput
from email_thread_rag.graph.provider import (
    GraphProviderError,
    MedhaGraphExtractor,
    build_provider,
)


def _input():
    return ExtractionInput(chunk_id="c-1", text="Alice approved the budget.", subject="Budget")


def test_build_provider_returns_none_when_disabled(tmp_path):
    settings = Settings(project_root=tmp_path, graph_extraction_enabled=False)
    assert build_provider(settings) is None


def test_build_provider_requires_url_and_model_when_enabled(tmp_path):
    settings = Settings(project_root=tmp_path, graph_extraction_enabled=True,
                        graph_base_url=None, graph_model=None)
    with pytest.raises(GraphProviderError):
        build_provider(settings)


def test_build_provider_constructs_openai_compatible_client(tmp_path):
    settings = Settings(project_root=tmp_path, graph_extraction_enabled=True,
                        graph_base_url="http://fake.invalid/v1", graph_model="fake-model")
    provider = build_provider(settings)
    assert isinstance(provider, MedhaGraphExtractor)
    assert provider.model_id == "fake-model"


def test_api_key_never_appears_in_repr():
    provider = MedhaGraphExtractor(base_url="http://x/v1", model="m", api_key="super-secret-key")
    assert "super-secret-key" not in repr(provider)


def test_fake_provider_records_calls_and_returns_canned_output():
    provider = FakeGraphProvider(responder=lambda ei: graph_json(
        entities=[{"name": "Alice", "type": "PERSON", "evidence": "Alice"}]))
    out = provider.generate(_input())
    assert "Alice" in out and len(provider.calls) == 1


def test_unavailable_provider_raises_provider_error():
    with pytest.raises(GraphProviderError):
        UnavailableGraphProvider().generate(_input())


def test_exploding_provider_asserts_it_should_never_be_called():
    with pytest.raises(AssertionError):
        ExplodingGraphProvider().generate(_input())
