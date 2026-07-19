"""The query-side prefix is model-specific and fails silently when wrong.

A prefix applied to a model that does not want one (GTE, MiniLM) raises no
error -- it just embeds a different sentence than the user asked about and
quietly loses recall. Same for a missing prefix on BGE. Nothing else in the
stack notices, so it is checked here.
"""

from __future__ import annotations

import numpy as np

from email_thread_rag.config import Settings
from email_thread_rag.rag.vector_index import SentenceTransformerEncoder, encode_query


class _RecordingEncoder(SentenceTransformerEncoder):
    """Records what reached the model instead of loading one."""

    def __init__(self, model_name: str):
        super().__init__(Settings(), model_name=model_name)
        self.seen: list[str] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.seen.extend(texts)
        return np.zeros((len(texts), self.settings.embedding_dim), dtype="float32")


def test_gte_queries_are_embedded_as_raw_text():
    encoder = _RecordingEncoder("Alibaba-NLP/gte-modernbert-base")
    encoder.encode_query(["who approved the budget?"])
    assert encoder.seen == ["who approved the budget?"]


def test_bge_queries_keep_their_retrieval_prefix():
    encoder = _RecordingEncoder("BAAI/bge-base-en-v1.5")
    encoder.encode_query(["who approved the budget?"])
    assert encoder.seen[0].endswith("who approved the budget?")
    assert encoder.seen[0] != "who approved the budget?"


def test_passages_never_get_a_prefix():
    for model_name in ("Alibaba-NLP/gte-modernbert-base", "BAAI/bge-base-en-v1.5"):
        encoder = _RecordingEncoder(model_name)
        encoder.encode(["budget approved by finance"])
        assert encoder.seen == ["budget approved by finance"]


def test_encode_query_falls_back_to_encode_on_symmetric_encoders():
    """Test doubles and the hashing fallback have no encode_query at all."""

    class _Symmetric:
        def __init__(self):
            self.seen: list[str] = []

        def encode(self, texts: list[str]) -> np.ndarray:
            self.seen.extend(texts)
            return np.zeros((len(texts), 768), dtype="float32")

    encoder = _Symmetric()
    vector = encode_query(encoder, "who approved the budget?")
    assert encoder.seen == ["who approved the budget?"]
    assert vector.shape == (768,)


def test_fallback_encoder_matches_the_configured_dimension():
    settings = Settings()
    encoder = SentenceTransformerEncoder(settings)
    assert encoder.fallback.dim == settings.embedding_dim
    assert encoder.fallback.encode(["hi"]).shape == (1, settings.embedding_dim)
