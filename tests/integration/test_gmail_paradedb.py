"""Stage-3 against the real ParadeDB container.

Two jobs here:
1. Run the *same* SyncStore contract the in-memory store passes, against
   PostgresSyncStore -- so the fast suite's store cannot drift from the real one.
2. Smoke-test the whole path: fake Pub/Sub push -> durable job -> fake Gmail
   history/message -> canonical chunks in ParadeDB -> hybrid retrieval returns
   clean citation text.

Gmail itself is still faked. No credentials, no network.
"""

from __future__ import annotations

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from email_thread_rag.config import Settings
from email_thread_rag.gmail.fakes import (
    FakeGmailClient,
    FakePubSubVerifier,
    build_gmail_message,
    pubsub_push_body,
)
from email_thread_rag.gmail.repository import PostgresSyncStore
from email_thread_rag.gmail.sink import ParadeDBChunkSink
from email_thread_rag.gmail.store import utcnow
from email_thread_rag.gmail.webhook import build_router
from email_thread_rag.gmail.worker import SyncWorker
from email_thread_rag.rag.paradedb.retrieval import ParadeDBEngineRetriever, RetrievalFilters
from email_thread_rag.rag.paradedb.retrieval import HybridRetriever as ParadeDBHybridRetriever
from email_thread_rag.rag.vector_index import HashingEncoder
from gmail_store_contract import SyncStoreContract, connect_mailbox, notify

pytestmark = pytest.mark.integration

ENCODER = HashingEncoder(dim=768)
SUBSCRIPTION = "projects/demo/subscriptions/gmail-push"
AUTH = {"Authorization": "Bearer fake-oidc-token"}
ADDRESS = "user@example.com"
GMAIL_TABLES = ("gmail_sync_jobs", "gmail_pubsub_messages", "gmail_oauth_states", "gmail_mailboxes")


@pytest.fixture
def gmail_conn(migrated_database_url):
    """autocommit + explicit transaction blocks: the same connection mode the
    worker uses, which is what keeps a transaction from spanning a Gmail call."""
    conn = psycopg.connect(migrated_database_url, autocommit=True, row_factory=dict_row)
    # The store commits for real, so clean up rather than relying on rollback.
    conn.execute(f"TRUNCATE {', '.join(GMAIL_TABLES)} RESTART IDENTITY CASCADE")
    conn.execute("TRUNCATE email_chunks, email_messages RESTART IDENTITY CASCADE")
    yield conn
    conn.close()


@pytest.fixture
def store(gmail_conn):
    return PostgresSyncStore(gmail_conn)


class TestPostgresSyncStore(SyncStoreContract):
    """The full store contract, against Postgres. Same assertions the
    in-memory store passes in tests/test_gmail_store.py."""


# --- Schema ---------------------------------------------------------------
def test_gmail_migration_created_every_table(gmail_conn):
    rows = gmail_conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' "
        "AND table_name LIKE 'gmail%'"
    ).fetchall()
    assert set(GMAIL_TABLES) <= {row["table_name"] for row in rows}


def test_history_ids_are_stored_numerically_not_as_text(gmail_conn):
    rows = gmail_conn.execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'public' AND column_name IN "
        "('last_committed_history_id', 'requested_history_id')"
    ).fetchall()
    assert rows
    for row in rows:
        assert row["data_type"] == "numeric", f"{row['table_name']}.{row['column_name']} is not numeric"


def test_only_one_pending_job_per_mailbox_is_possible(gmail_conn, store):
    mailbox = connect_mailbox(store, address=ADDRESS)
    notify(store, ADDRESS, 100, "pubsub-1")
    # The partial unique index is the invariant the coalescing relies on;
    # bypass the store to prove the database itself enforces it.
    with pytest.raises(psycopg.errors.UniqueViolation):
        gmail_conn.execute(
            "INSERT INTO gmail_sync_jobs (mailbox_db_id, tenant_id, mailbox_id, requested_history_id, status) "
            "VALUES (%s, 'acme', 'inbox', 200, 'pending')",
            (mailbox.id,),
        )


def test_two_live_mailboxes_cannot_share_an_address(gmail_conn, store):
    connect_mailbox(store, tenant="acme", mailbox="inbox", address=ADDRESS)
    with pytest.raises(psycopg.errors.UniqueViolation):
        gmail_conn.execute(
            "INSERT INTO gmail_mailboxes (tenant_id, mailbox_id, email_address, status) "
            "VALUES ('globex', 'inbox', %s, 'active')",
            (ADDRESS,),
        )


def test_large_history_ids_round_trip_exactly(store):
    """Gmail history IDs are unsigned 64-bit; numeric(20,0) must not round them."""
    mailbox = connect_mailbox(store, address=ADDRESS, history_id=100)
    huge = 18_446_744_073_709_551_615
    store.commit_history_cursor(mailbox.id, huge)
    assert store.get_mailbox_by_db_id(mailbox.id).last_committed_history_id == huge


# --- End-to-end smoke ------------------------------------------------------
def settings_for(tenant="acme", mailbox_id="inbox") -> Settings:
    return Settings(rag_backend="paradedb", tenant_id=tenant, mailbox_id=mailbox_id, embedding_dim=768)


def gmail_message(gmail_id, *, body, thread="t-1", subject="Budget Review"):
    return build_gmail_message(
        gmail_id=gmail_id,
        thread_id=thread,
        history_id=200,
        sender="alice@corp.com",
        to="bob@corp.com",
        subject=subject,
        body=body,
    )


def history_page(*, history_id, added=(), deleted=()):
    records = [{"id": str(rid), "messagesAdded": [{"message": {"id": mid}}]} for rid, mid in added]
    records += [{"id": str(rid), "messagesDeleted": [{"message": {"id": mid}}]} for rid, mid in deleted]
    return {"historyId": str(history_id), "history": records}


def build_worker(store, conn, client, *, tenant="acme"):
    return SyncWorker(
        store,
        client_factory=lambda _m: client,
        sink_factory=lambda m: ParadeDBChunkSink(
            conn, tenant_id=m.tenant_id, mailbox_id=m.mailbox_id, encoder=ENCODER
        ),
    )


def push_client(store):
    app = FastAPI()
    app.include_router(
        build_router(
            store_factory=lambda: store, verifier=FakePubSubVerifier(expected_subscription=SUBSCRIPTION)
        )
    )
    return TestClient(app)


def test_smoke_push_to_job_to_paradedb_to_hybrid_retrieval(gmail_conn, store):
    """Fake Pub/Sub notification -> durable job -> fake Gmail history/message ->
    parsed chunks in ParadeDB -> hybrid retrieval returns clean citation text."""
    mailbox = connect_mailbox(store, address=ADDRESS, history_id=100)

    # 1. Pub/Sub push -> durable job (no Gmail call in the request path).
    response = push_client(store).post(
        "/gmail/pubsub/push",
        json=pubsub_push_body(ADDRESS, 200, message_id="pubsub-1", subscription=SUBSCRIPTION),
        headers=AUTH,
    )
    assert response.status_code == 200
    job = gmail_conn.execute("SELECT * FROM gmail_sync_jobs WHERE status = 'pending'").fetchone()
    assert job is not None and int(job["requested_history_id"]) == 200

    # 2. Worker drains it against fake Gmail.
    client = FakeGmailClient(
        email_address=ADDRESS,
        messages={
            "m-1": gmail_message(
                "m-1",
                body=(
                    "Confirming the approved amount is $1200 for Acme Supplies.\n\n"
                    "On Mon, Jan 5, 2026 at 9:00 AM Bob <bob@corp.com> wrote:\n"
                    "> The draft budget was $1000.\n"
                ),
            )
        },
        history_pages=[history_page(history_id=200, added=[(150, "m-1")])],
    )
    outcome = build_worker(store, gmail_conn, client).run_once()
    assert outcome.added == 1 and outcome.chunks_written >= 1

    # 3. Chunks landed in ParadeDB, scoped to the tenant/mailbox.
    rows = gmail_conn.execute(
        "SELECT * FROM email_chunks WHERE tenant_id = 'acme' AND mailbox_id = 'inbox'"
    ).fetchall()
    assert rows
    assert rows[0]["embedding"] is not None
    assert rows[0]["thread_id"] == "t-1"

    # 4. Cursor advanced only after all of that persisted.
    assert store.get_mailbox_by_db_id(mailbox.id).last_committed_history_id == 200

    # 5. Hybrid retrieval returns the exact authored text -- no embed_text
    #    headers, no quoted reply -- so a citation quotes what the sender wrote.
    hybrid = ParadeDBHybridRetriever(gmail_conn, settings_for(), encoder=ENCODER)
    hits = hybrid.search(
        "approved amount", RetrievalFilters(tenant_id="acme", mailbox_id="inbox"), top_k=5
    )
    assert hits
    text = hits[0].text
    assert "approved amount is $1200" in text
    assert "Subject:" not in text and "From:" not in text
    assert "$1000" not in text

    # 6. The engine-facing retriever answers from the same rows.
    engine_retriever = ParadeDBEngineRetriever(gmail_conn, settings_for(), encoder=ENCODER)
    assert engine_retriever.available_threads() == ["t-1"]
    result = engine_retriever.search("approved amount", thread_id="t-1")
    assert result.reranked_hits
    assert "$1200" in result.reranked_hits[0].chunk.text


def test_deleted_gmail_message_disappears_from_hybrid_retrieval(gmail_conn, store):
    connect_mailbox(store, address=ADDRESS, history_id=100)
    client = FakeGmailClient(
        email_address=ADDRESS,
        messages={"m-1": gmail_message("m-1", body="Phoenix invoice approved for $900.")},
        history_pages=[history_page(history_id=200, added=[(150, "m-1")])],
    )
    worker = build_worker(store, gmail_conn, client)
    notify(store, ADDRESS, 200, "pubsub-1")
    worker.run_once()

    filters = RetrievalFilters(tenant_id="acme", mailbox_id="inbox")
    hybrid = ParadeDBHybridRetriever(gmail_conn, settings_for(), encoder=ENCODER)
    assert hybrid.search("Phoenix invoice", filters, top_k=5)

    # Gmail says it's gone.
    client.history_pages = [history_page(history_id=300, deleted=[(250, "m-1")])]
    notify(store, ADDRESS, 300, "pubsub-2")
    worker.run_once()

    assert hybrid.search("Phoenix invoice", filters, top_k=5) == []
    assert gmail_conn.execute("SELECT count(*) AS n FROM email_chunks").fetchone()["n"] == 0
    # The parent message row goes too, not just its chunks.
    assert gmail_conn.execute("SELECT count(*) AS n FROM email_messages").fetchone()["n"] == 0


def test_reingesting_the_same_message_is_idempotent(gmail_conn, store):
    connect_mailbox(store, address=ADDRESS, history_id=100)
    client = FakeGmailClient(
        email_address=ADDRESS,
        messages={"m-1": gmail_message("m-1", body="Approved amount is $1200.")},
        history_pages=[history_page(history_id=200, added=[(150, "m-1")])],
    )
    worker = build_worker(store, gmail_conn, client)

    notify(store, ADDRESS, 200, "pubsub-1")
    worker.run_once()
    first = gmail_conn.execute("SELECT count(*) AS n FROM email_chunks").fetchone()["n"]

    # A replayed history window (retry, redelivery, overlapping page) must not
    # duplicate chunks.
    notify(store, ADDRESS, 200, "pubsub-2")
    worker.run_once()

    assert gmail_conn.execute("SELECT count(*) AS n FROM email_chunks").fetchone()["n"] == first
    assert gmail_conn.execute("SELECT count(*) AS n FROM email_messages").fetchone()["n"] == 1


def test_tenant_isolation_holds_through_retrieval(gmail_conn, store):
    acme = connect_mailbox(store, tenant="acme", mailbox="inbox", address="a@example.com", history_id=100)
    globex = connect_mailbox(store, tenant="globex", mailbox="inbox", address="b@example.com", history_id=100)

    clients = {
        acme.id: FakeGmailClient(
            email_address="a@example.com",
            messages={"m-acme": gmail_message("m-acme", body="Acme secret budget is $1200.", thread="t-acme")},
            history_pages=[history_page(history_id=200, added=[(150, "m-acme")])],
        ),
        globex.id: FakeGmailClient(
            email_address="b@example.com",
            messages={
                "m-globex": gmail_message("m-globex", body="Globex secret budget is $9900.", thread="t-globex")
            },
            history_pages=[history_page(history_id=300, added=[(250, "m-globex")])],
        ),
    }
    worker = SyncWorker(
        store,
        client_factory=lambda m: clients[m.id],
        sink_factory=lambda m: ParadeDBChunkSink(
            gmail_conn, tenant_id=m.tenant_id, mailbox_id=m.mailbox_id, encoder=ENCODER
        ),
    )
    notify(store, "a@example.com", 200, "pubsub-acme")
    notify(store, "b@example.com", 300, "pubsub-globex")
    worker.drain()

    hybrid = ParadeDBHybridRetriever(gmail_conn, settings_for(), encoder=ENCODER)
    acme_hits = hybrid.search("secret budget", RetrievalFilters(tenant_id="acme", mailbox_id="inbox"), top_k=10)
    globex_hits = hybrid.search(
        "secret budget", RetrievalFilters(tenant_id="globex", mailbox_id="inbox"), top_k=10
    )

    assert {hit.message_id for hit in acme_hits} == {"m-acme"}
    assert {hit.message_id for hit in globex_hits} == {"m-globex"}
    assert all("9900" not in hit.text for hit in acme_hits)
    assert all("1200" not in hit.text for hit in globex_hits)


def test_worker_failure_leaves_cursor_and_retries_against_postgres(gmail_conn, store):
    mailbox = connect_mailbox(store, address=ADDRESS, history_id=100)
    store.commit_history_cursor(mailbox.id, 250)
    job = notify(store, ADDRESS, 400, "pubsub-1")

    client = FakeGmailClient(
        email_address=ADDRESS,
        messages={"m-1": gmail_message("m-1", body="Approved amount is $1200.")},
        history_pages=[history_page(history_id=400, added=[(300, "m-1")])],
    )
    client.fail_get_message_ids = {"m-1"}
    worker = build_worker(store, gmail_conn, client)
    worker.run_once()

    assert store.get_mailbox_by_db_id(mailbox.id).last_committed_history_id == 250
    assert store.get_job(job.id).status == "pending"
    assert gmail_conn.execute("SELECT count(*) AS n FROM email_chunks").fetchone()["n"] == 0

    client.fail_get_message_ids = set()
    worker.run_once()

    assert client.history_calls() == [250, 250]
    assert store.get_mailbox_by_db_id(mailbox.id).last_committed_history_id == 400
    assert store.get_job(job.id).status == "done"
    assert gmail_conn.execute("SELECT count(*) AS n FROM email_chunks").fetchone()["n"] >= 1


def test_history_404_full_sync_against_postgres(gmail_conn, store):
    mailbox = connect_mailbox(store, address=ADDRESS, history_id=100)
    store.commit_history_cursor(mailbox.id, 250)
    notify(store, ADDRESS, 900, "pubsub-1")

    client = FakeGmailClient(
        email_address=ADDRESS,
        profile_history_id=800,
        messages={
            "m-1": gmail_message("m-1", body="Approved amount is $1200."),
            "m-2": gmail_message("m-2", body="Phoenix invoice approved for $900.", thread="t-2"),
        },
    )
    client.history_expired = True
    outcome = build_worker(store, gmail_conn, client).run_once()

    assert outcome.full_sync is True
    assert store.get_mailbox_by_db_id(mailbox.id).last_committed_history_id == 800
    assert store.get_mailbox_by_db_id(mailbox.id).status == "active"
    assert gmail_conn.execute("SELECT count(*) AS n FROM email_messages").fetchone()["n"] == 2
