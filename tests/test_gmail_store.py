from __future__ import annotations

import pytest

from email_thread_rag.gmail.store import InMemorySyncStore
from gmail_store_contract import SyncStoreContract


class TestInMemorySyncStore(SyncStoreContract):
    """The full store contract, with no Postgres and no configuration.

    tests/integration/test_gmail_paradedb.py runs the same contract against
    PostgresSyncStore.
    """

    @pytest.fixture
    def store(self):
        return InMemorySyncStore()
