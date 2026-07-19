"""The in-memory ContextJobStore against the shared contract. No DB, no network."""

from __future__ import annotations

import pytest

from email_thread_rag.context.models import ChunkContextState
from email_thread_rag.context.store import InMemoryContextJobStore

from tests.context_store_contract import ContextStoreContract


class TestInMemoryContextJobStore(ContextStoreContract):
    @pytest.fixture
    def store(self):
        return InMemoryContextJobStore()

    @pytest.fixture
    def make_chunk(self, store):
        counter = {"next_id": 1}
        by_key: dict[tuple[str, str], ChunkContextState] = {}

        def _make(*, chunk_id, text, tenant_id="acme", mailbox_id="inbox", replace=False):
            key = (tenant_id, chunk_id)
            if replace and key in by_key:
                existing = by_key[key]
                existing.text = text
                return existing
            state = ChunkContextState(
                chunk_db_id=counter["next_id"],
                chunk_id=chunk_id,
                tenant_id=tenant_id,
                mailbox_id=mailbox_id,
                text=text,
                subject="Budget Review",
                sender="alice@corp.com",
                thread_id="thread-alpha",
            )
            counter["next_id"] += 1
            store.add_chunk(state)
            by_key[key] = state
            return state

        return _make

    @pytest.fixture
    def read_back(self, store):
        # The in-memory store hands out the live object; re-reading is the
        # same lookup Postgres would do.
        return lambda state: store.load_chunk_state(state.chunk_db_id)

    @pytest.fixture
    def mutate_chunk(self, store):
        def _mutate(state, *, text):
            live = store.load_chunk_state(state.chunk_db_id)
            live.text = text

        return _mutate
