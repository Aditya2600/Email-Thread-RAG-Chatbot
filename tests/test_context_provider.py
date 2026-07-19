"""The provider seam. Fakes and a stubbed transport only -- no model, no network.

The autouse socket guard in conftest.py fails this module if anything here
opens a real connection, so a "fake" that accidentally reaches the live Medha
endpoint cannot pass silently.
"""

from __future__ import annotations

import json

import pytest

from email_thread_rag.config import Settings
from email_thread_rag.context.fakes import (
    ExplodingContextProvider,
    FakeContextProvider,
    UnavailableContextProvider,
    context_json,
)
from email_thread_rag.context.models import ContextInput
from email_thread_rag.context.provider import (
    DEFAULT_MAX_TOKENS,
    ContextProviderError,
    MedhaContextualizer,
    build_provider,
)

API_KEY = "sk-medha-super-secret-key"
CONTEXT_INPUT = ContextInput(
    chunk_id="msg-2-email-0",
    text="The approved amount is $1200.",
    subject="Re: Budget Review",
    sender="bob@corp.com",
)


class StubResponse:
    def __init__(self, *, status_code=200, payload=None, text_body=None):
        self.status_code = status_code
        self._payload = payload
        self._text_body = text_body

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def captured_post(monkeypatch):
    """Replace httpx.post and record what the provider would have sent."""
    import httpx

    calls: list[dict] = []

    def fake_post(url, *, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return StubResponse(
            payload={"choices": [{"message": {"content": context_json("This chunk concerns the budget.")}}]}
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


def medha(**overrides) -> MedhaContextualizer:
    kwargs = dict(base_url="http://164.52.192.196:8002/v1", model="Medha", api_key=API_KEY)
    kwargs.update(overrides)
    return MedhaContextualizer(**kwargs)


# --- request shape -------------------------------------------------------
def test_the_provider_posts_to_chat_completions(captured_post):
    medha().generate(CONTEXT_INPUT)
    assert captured_post[0]["url"] == "http://164.52.192.196:8002/v1/chat/completions"


def test_a_trailing_slash_on_the_base_url_does_not_double_up(captured_post):
    medha(base_url="http://164.52.192.196:8002/v1/").generate(CONTEXT_INPUT)
    assert captured_post[0]["url"] == "http://164.52.192.196:8002/v1/chat/completions"


def test_the_request_pins_the_deterministic_settings(captured_post):
    medha().generate(CONTEXT_INPUT)
    body = captured_post[0]["json"]
    assert body["model"] == "Medha"
    assert body["temperature"] == 0
    assert body["max_tokens"] == DEFAULT_MAX_TOKENS == 96
    assert body["stream"] is False


def test_the_api_key_is_sent_as_a_bearer_token(captured_post):
    medha().generate(CONTEXT_INPUT)
    assert captured_post[0]["headers"]["Authorization"] == f"Bearer {API_KEY}"


def test_no_authorization_header_when_no_key_is_configured(captured_post):
    medha(api_key=None).generate(CONTEXT_INPUT)
    assert "Authorization" not in captured_post[0]["headers"]


def test_max_tokens_and_timeout_are_configuration(captured_post):
    medha(max_tokens=48, timeout=7.5).generate(CONTEXT_INPUT)
    assert captured_post[0]["json"]["max_tokens"] == 48
    assert captured_post[0]["timeout"] == 7.5


def test_the_provider_returns_raw_output_for_the_caller_to_validate(captured_post):
    raw = medha().generate(CONTEXT_INPUT)
    # Raw, unvalidated: validation is the worker's job so every provider is held
    # to the same contract.
    assert json.loads(raw)["context"] == "This chunk concerns the budget."


# --- secret handling -----------------------------------------------------
def test_the_api_key_is_never_in_the_provider_repr():
    assert API_KEY not in repr(medha())


def test_the_api_key_is_not_a_public_attribute():
    provider = medha()
    assert API_KEY not in json.dumps({k: str(v) for k, v in vars(provider).items() if not k.startswith("_")})


def test_provider_errors_never_carry_the_key_or_the_body(monkeypatch):
    import httpx

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: StubResponse(status_code=401, payload={"error": API_KEY})
    )
    with pytest.raises(ContextProviderError) as excinfo:
        medha().generate(CONTEXT_INPUT)
    assert API_KEY not in str(excinfo.value)
    assert "HTTP 401" in str(excinfo.value)


# --- failure modes -------------------------------------------------------
def test_a_non_200_becomes_a_provider_error(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "post", lambda *a, **k: StubResponse(status_code=503))
    with pytest.raises(ContextProviderError, match="HTTP 503"):
        medha().generate(CONTEXT_INPUT)


def test_a_transport_failure_becomes_a_provider_error(monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(ContextProviderError, match="request failed"):
        medha().generate(CONTEXT_INPUT)


def test_an_unexpected_response_shape_becomes_a_provider_error(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "post", lambda *a, **k: StubResponse(payload={"unexpected": True}))
    with pytest.raises(ContextProviderError, match="unexpected response shape"):
        medha().generate(CONTEXT_INPUT)


# --- build_provider: disabled by default ---------------------------------
def test_contextualization_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CONTEXT_ENABLED", raising=False)
    monkeypatch.delenv("MEDHA_BASE_URL", raising=False)
    monkeypatch.delenv("CONTEXT_BASE_URL", raising=False)
    settings = Settings()
    assert settings.context_enabled is False
    # Disabled means no provider at all: nothing to call, nothing to import.
    assert build_provider(settings) is None


def test_enabled_without_an_endpoint_fails_loudly(monkeypatch):
    monkeypatch.setenv("CONTEXT_ENABLED", "true")
    monkeypatch.delenv("CONTEXT_BASE_URL", raising=False)
    monkeypatch.delenv("MEDHA_BASE_URL", raising=False)
    monkeypatch.delenv("CONTEXT_MODEL", raising=False)
    monkeypatch.delenv("MEDHA_MODEL", raising=False)
    with pytest.raises(ContextProviderError, match="requires CONTEXT_BASE_URL"):
        build_provider(Settings())


def test_the_medha_env_vars_configure_the_provider(monkeypatch):
    monkeypatch.setenv("CONTEXT_ENABLED", "true")
    monkeypatch.setenv("MEDHA_BASE_URL", "http://164.52.192.196:8002/v1")
    monkeypatch.setenv("MEDHA_MODEL", "Medha")
    monkeypatch.setenv("MEDHA_API_KEY", API_KEY)

    provider = build_provider(Settings())

    assert isinstance(provider, MedhaContextualizer)
    assert provider.base_url == "http://164.52.192.196:8002/v1"
    assert provider.model_id == "Medha"
    assert API_KEY not in repr(provider)


def test_nothing_is_hard_coded_to_medha(monkeypatch):
    # Any OpenAI-compatible endpoint works purely by configuration.
    monkeypatch.setenv("CONTEXT_ENABLED", "true")
    monkeypatch.setenv("CONTEXT_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("CONTEXT_MODEL", "some-other-model")
    provider = build_provider(Settings())
    assert provider.base_url == "http://localhost:11434/v1"
    assert provider.model_id == "some-other-model"


# --- the fakes themselves ------------------------------------------------
def test_the_fake_provider_records_calls_and_returns_valid_json():
    provider = FakeContextProvider()
    raw = provider.generate(CONTEXT_INPUT)
    assert provider.calls == [CONTEXT_INPUT]
    assert json.loads(raw)["context"] == "This chunk concerns Re: Budget Review."


def test_the_exploding_provider_fails_any_synchronous_call():
    with pytest.raises(AssertionError, match="must not be called"):
        ExplodingContextProvider().generate(CONTEXT_INPUT)


def test_the_unavailable_provider_raises_a_provider_error():
    with pytest.raises(ContextProviderError):
        UnavailableContextProvider().generate(CONTEXT_INPUT)
