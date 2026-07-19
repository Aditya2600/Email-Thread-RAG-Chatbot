"""Fakes for the provider seam. No HTTP, no model, no network.

Importable from tests and from a local demo; every Stage-4 test uses these.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

from email_thread_rag.context.models import ContextInput
from email_thread_rag.context.provider import ContextProviderError


def context_json(text: str) -> str:
    """A well-formed provider response body, the way a compliant model replies."""
    return json.dumps({"context": text})


class FakeContextProvider:
    """Records calls; returns a canned or computed response.

    ``responder`` receives the ContextInput and returns the raw model output, so
    a test can simulate malformed JSON, an oversized prefix, or an injection
    attempt just as easily as a good reply.
    """

    def __init__(
        self,
        *,
        responder: Callable[[ContextInput], Optional[str]] | None = None,
        model_id: str = "fake-context-model",
        fail_with: Exception | None = None,
    ):
        self.calls: list[ContextInput] = []
        self._responder = responder or (lambda ci: context_json(f"This chunk concerns {ci.subject or 'an email'}."))
        self._model_id = model_id
        self._fail_with = fail_with

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, context_input: ContextInput) -> Optional[str]:
        self.calls.append(context_input)
        if self._fail_with is not None:
            raise self._fail_with
        return self._responder(context_input)


class ExplodingContextProvider:
    """Fails every call. For asserting that ingestion/webhook paths never call
    the provider synchronously -- if they do, the test dies loudly."""

    model_id = "exploding-context-model"

    def generate(self, context_input: ContextInput) -> Optional[str]:
        raise AssertionError("the provider must not be called on this path")


class UnavailableContextProvider:
    """Simulates a provider that is configured but down."""

    model_id = "unavailable-context-model"

    def generate(self, context_input: ContextInput) -> Optional[str]:
        raise ContextProviderError("context provider returned HTTP 503")
