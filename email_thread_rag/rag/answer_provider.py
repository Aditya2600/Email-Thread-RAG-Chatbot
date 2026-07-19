"""The Stage-7 answer provider seam: one narrow interface, one OpenAI-compatible
client. Mirror of ``graph.provider`` / ``context.provider``.

httpx is imported inside ``generate``, never at module import, so this module is
safe to import on the disabled/memory path. The same class works against any
OpenAI-compatible endpoint by configuration; no model is downloaded and nothing
is called unless answer generation is enabled.
"""

from __future__ import annotations

from typing import Optional, Protocol

DEFAULT_MAX_TOKENS = 800
DEFAULT_TIMEOUT_SECONDS = 60.0


class AnswerProviderError(RuntimeError):
    """Provider call failed. Carries a status/reason, never the API key, the
    prompt, or the response body."""


class AnswerProvider(Protocol):
    def generate(self, messages: list[dict]) -> Optional[str]:
        """Return the model's raw text output for parsing, or None if empty.

        The provider never parses or validates: the caller does that
        deterministically and authoritatively, so every provider (real or fake)
        is held to the identical contract.
        """

    @property
    def model_id(self) -> str:
        ...


class OpenAICompatibleAnswerProvider:
    """OpenAI-compatible Chat Completions client (POST {base_url}/chat/completions).

    temperature=0 and stream=False are fixed so the grounded flow is as
    reproducible as the model allows.
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
        return f"OpenAICompatibleAnswerProvider(base_url={self.base_url!r}, model={self.model!r})"

    @property
    def model_id(self) -> str:
        return self.model

    def generate(self, messages: list[dict]) -> Optional[str]:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            raise AnswerProviderError(f"answer provider request failed: {exc.__class__.__name__}") from None

        if response.status_code != 200:
            raise AnswerProviderError(f"answer provider returned HTTP {response.status_code}")

        try:
            body = response.json()
            return body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise AnswerProviderError("answer provider returned an unexpected response shape") from None


def build_answer_provider(settings) -> Optional[AnswerProvider]:
    """Construct the configured provider, or None when answer generation is off.

    Disabled is the default and returns None -- which is what makes the grounded
    answerer abstain safely with no provider and no network.
    """
    if not settings.answer_generation_enabled:
        return None
    if not settings.answer_base_url or not settings.answer_model:
        raise AnswerProviderError(
            "ANSWER_GENERATION_ENABLED=true requires ANSWER_BASE_URL and ANSWER_MODEL to be set."
        )
    return OpenAICompatibleAnswerProvider(
        base_url=settings.answer_base_url,
        model=settings.answer_model,
        api_key=settings.answer_api_key,
        timeout=settings.answer_timeout_seconds,
        max_tokens=settings.answer_max_tokens,
    )
