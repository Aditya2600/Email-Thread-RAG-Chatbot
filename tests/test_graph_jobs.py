"""The in-memory GraphStore against the shared contract. No DB, no network."""

from __future__ import annotations

import pytest

from email_thread_rag.graph.models import ChunkGraphState
from email_thread_rag.graph.store import InMemoryGraphStore

from tests.graph_store_contract import GraphStoreContract


class TestInMemoryGraphStore(GraphStoreContract):
    @pytest.fixture
    def store(self):
        return InMemoryGraphStore()

    @pytest.fixture
    def make_chunk(self, store):
        counter = {"next_id": 1}
        by_key: dict[tuple[str, str], ChunkGraphState] = {}

        def _make(*, chunk_id, text, tenant_id="acme", mailbox_id="inbox", replace=False):
            key = (tenant_id, chunk_id)
            if replace and key in by_key:
                by_key[key].text = text
                return by_key[key]
            state = ChunkGraphState(
                chunk_db_id=counter["next_id"], chunk_id=chunk_id, tenant_id=tenant_id,
                mailbox_id=mailbox_id, text=text, subject="Budget Review",
                sender="alice@corp.com", thread_id="thread-alpha", source_start=0,
                recipients=["bob@corp.com"], cc=["carol@corp.com"],
            )
            counter["next_id"] += 1
            store.add_chunk(state)
            by_key[key] = state
            return state

        return _make

    @pytest.fixture
    def read_state(self, store):
        return lambda state: store.load_chunk_state(state.chunk_db_id)

    @pytest.fixture
    def mutate_chunk(self, store):
        def _mutate(state, *, text):
            store.load_chunk_state(state.chunk_db_id).text = text

        return _mutate
