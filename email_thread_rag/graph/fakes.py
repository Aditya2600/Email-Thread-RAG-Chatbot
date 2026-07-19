"""Fakes for the graph provider seam. No HTTP, no model, no network.

Importable from tests and a local demo; every Stage-5 provider test uses these.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

from email_thread_rag.graph.models import ExtractionInput
from email_thread_rag.graph.provider import GraphProviderError


def graph_json(*, entities=None, relations=None, facts=None) -> str:
    """A well-formed extraction response body, the way a compliant model replies."""
    return json.dumps(
        {"entities": entities or [], "relations": relations or [], "facts": facts or []}
    )


class FakeGraphProvider:
    """Records calls; returns a canned or computed response.

    ``responder`` receives the ExtractionInput and returns raw model output, so a
    test can simulate malformed JSON, hallucinated entities, or evidence strings
    that do not appear in the text just as easily as a good reply.
    """

    def __init__(
        self,
        *,
        responder: Callable[[ExtractionInput], Optional[str]] | None = None,
        model_id: str = "fake-graph-model",
        fail_with: Exception | None = None,
    ):
        self.calls: list[ExtractionInput] = []
        self._responder = responder or (lambda ei: graph_json())
        self._model_id = model_id
        self._fail_with = fail_with

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, extraction_input: ExtractionInput) -> Optional[str]:
        self.calls.append(extraction_input)
        if self._fail_with is not None:
            raise self._fail_with
        return self._responder(extraction_input)


class ExplodingGraphProvider:
    """Fails every call. For asserting that ingestion/webhook paths never call
    the provider synchronously -- if they do, the test dies loudly."""

    model_id = "exploding-graph-model"

    def generate(self, extraction_input: ExtractionInput) -> Optional[str]:
        raise AssertionError("the provider must not be called on this path")


class UnavailableGraphProvider:
    """Simulates a provider that is configured but down."""

    model_id = "unavailable-graph-model"

    def generate(self, extraction_input: ExtractionInput) -> Optional[str]:
        raise GraphProviderError("graph provider returned HTTP 503")
