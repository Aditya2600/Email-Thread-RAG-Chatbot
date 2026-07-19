"""The provider seam: one narrow interface, one OpenAI-compatible client.

Mirror of ``context.provider``. httpx is imported inside the client, never at
module import, so this module stays importable on the disabled/memory path. The
same class works against any OpenAI-compatible endpoint by configuration; no
model is downloaded and nothing is called unless graph extraction is enabled.
"""

from __future__ import annotations

from typing import Optional, Protocol

from email_thread_rag.graph.models import ExtractionInput

DEFAULT_MAX_TOKENS = 800  # room for entities + relations + facts JSON
DEFAULT_TIMEOUT_SECONDS = 60.0


class GraphProviderError(RuntimeError):
    """Provider call failed. Carries a status/reason, never the API key, the
    prompt, or the response body."""


class GraphProvider(Protocol):
    def generate(self, extraction_input: ExtractionInput) -> Optional[str]:
        """Return the model's raw output for parsing, or None if unavailable.

        Implementations must not parse or locate evidence: the caller does that
        deterministically, so every provider is held to the identical contract.
        """

    @property
    def model_id(self) -> str:
        """Identifies the model in the fingerprint. Changing it re-extracts."""


class MedhaGraphExtractor:
    """OpenAI-compatible Chat Completions client (POST {base_url}/chat/completions).

    temperature=0 and stream=False are fixed: the fingerprint assumes a given
    input maps to a stable output.
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
        return f"MedhaGraphExtractor(base_url={self.base_url!r}, model={self.model!r})"

    @property
    def model_id(self) -> str:
        return self.model

    def generate(self, extraction_input: ExtractionInput) -> Optional[str]:
        import httpx

        from email_thread_rag.graph.prompt import build_messages

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self.model,
            "messages": build_messages(extraction_input),
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            raise GraphProviderError(f"graph provider request failed: {exc.__class__.__name__}") from None

        if response.status_code != 200:
            raise GraphProviderError(f"graph provider returned HTTP {response.status_code}")

        try:
            body = response.json()
            return body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise GraphProviderError("graph provider returned an unexpected response shape") from None


def build_provider(settings) -> Optional[GraphProvider]:
    """Construct the configured provider, or None when graph extraction is off.

    Disabled is the default and returns None, which is what keeps ingestion from
    enqueuing work nothing will ever consume.
    """
    if not settings.graph_extraction_enabled:
        return None
    if not settings.graph_base_url or not settings.graph_model:
        raise GraphProviderError(
            "GRAPH_EXTRACTION_ENABLED=true requires GRAPH_BASE_URL and GRAPH_MODEL to be set."
        )
    return MedhaGraphExtractor(
        base_url=settings.graph_base_url,
        model=settings.graph_model,
        api_key=settings.graph_api_key,
        timeout=settings.graph_timeout_seconds,
        max_tokens=settings.graph_max_tokens,
    )
