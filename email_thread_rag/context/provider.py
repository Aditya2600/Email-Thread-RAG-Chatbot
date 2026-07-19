"""The provider seam: one narrow interface, one OpenAI-compatible client.

``ContextProvider`` has a single method so tests can substitute a fake without
any HTTP machinery. ``MedhaContextualizer`` speaks the OpenAI Chat Completions
API, which means the same class works against any OpenAI-compatible endpoint
(vLLM, llama.cpp, Ollama's compat layer) purely by configuration -- no model is
downloaded, no SDK is required, and nothing is called unless explicitly enabled.

httpx is imported inside the client, not at module import: importing this module
must stay free for the disabled/memory path.
"""

from __future__ import annotations

from typing import Optional, Protocol

from email_thread_rag.context.models import ContextInput

DEFAULT_MAX_TOKENS = 96  # 80-token prefix budget + room for the JSON wrapper
DEFAULT_TIMEOUT_SECONDS = 30.0


class ContextProviderError(RuntimeError):
    """Provider call failed. Carries a status/reason, never the API key, the
    prompt, or the response body -- this message is logged and stored on the
    job row."""


class ContextProvider(Protocol):
    def generate(self, context_input: ContextInput) -> Optional[str]:
        """Return the model's raw output for validation, or None if unavailable.

        Implementations must not validate: the caller does that deterministically
        so every provider is held to the identical contract.
        """

    @property
    def model_id(self) -> str:
        """Identifies the model in the fingerprint. Changing it re-contextualizes."""


class MedhaContextualizer:
    """OpenAI-compatible Chat Completions client (POST {base_url}/chat/completions).

    temperature=0 and stream=False are not configurable: the fingerprint assumes
    a given input maps to a stable output, and a sampled or streamed response
    breaks the "reprocessing an unchanged chunk does no new work" property.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Held for the Authorization header only. Never rendered into an error,
        # a log line, or this object's repr.
        self._api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"MedhaContextualizer(base_url={self.base_url!r}, model={self.model!r})"

    @property
    def model_id(self) -> str:
        return self.model

    def generate(self, context_input: ContextInput) -> Optional[str]:
        import httpx

        from email_thread_rag.context.prompt import build_messages

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self.model,
            "messages": build_messages(context_input),
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            # str(exc) on an httpx error is a message/URL, never our headers.
            raise ContextProviderError(f"context provider request failed: {exc.__class__.__name__}") from None

        if response.status_code != 200:
            # Status only: a 4xx body from an unknown endpoint is not something
            # we want to copy into logs or the job row.
            raise ContextProviderError(f"context provider returned HTTP {response.status_code}")

        try:
            body = response.json()
            return body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise ContextProviderError("context provider returned an unexpected response shape") from None


def build_provider(settings) -> Optional[ContextProvider]:
    """Construct the configured provider, or None when contextualization is off.

    Disabled is the default and returns None, which is what keeps ingestion from
    enqueuing work nothing will ever consume.
    """
    if not settings.context_enabled:
        return None
    if not settings.context_base_url or not settings.context_model:
        raise ContextProviderError(
            "CONTEXT_ENABLED=true requires CONTEXT_BASE_URL and CONTEXT_MODEL to be set."
        )
    return MedhaContextualizer(
        base_url=settings.context_base_url,
        model=settings.context_model,
        api_key=settings.context_api_key,
        timeout=settings.context_timeout_seconds,
        max_tokens=settings.context_max_tokens,
    )
