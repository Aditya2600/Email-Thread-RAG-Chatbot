"""End-to-end proof that a Pub/Sub push results in stored emails and non-empty
threads -- the exact path that was silently broken because nothing ran the
worker.

Pub/Sub push (webhook router) -> gmail_sync_jobs -> worker claim/process ->
FakeGmailClient.messages.get -> ChunkSink persistence -> thread availability.

No network, no Postgres: InMemorySyncStore + InMemoryChunkSink + FakeGmailClient
stand in for the durable pair (tests/integration/test_gmail_paradedb.py runs the
real one). The point here is the wiring, not the storage engine.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from email_thread_rag.gmail.fakes import (
    AllowAllPubSubVerifier,
    FakeGmailClient,
    build_gmail_message,
    pubsub_push_body,
)
from email_thread_rag.gmail.sink import InMemoryChunkSink
from email_thread_rag.gmail.store import InMemorySyncStore
from email_thread_rag.gmail.webhook import build_router
from email_thread_rag.gmail.worker import InlineWorker, SyncWorker
from gmail_store_contract import connect_mailbox

ADDRESS = "user@example.com"
SUBSCRIPTION = "projects/demo/subscriptions/gmail-push"


def message(gmail_id: str, *, thread: str) -> dict:
    return build_gmail_message(
        gmail_id=gmail_id,
        thread_id=thread,
        history_id=200,
        sender="alice@corp.com",
        to="bob@corp.com",
        subject="Budget Review",
        body="The approved amount is $1200.",
    )


def history_page(history_id: int, added: list[tuple[int, str]]) -> dict:
    return {
        "historyId": str(history_id),
        "history": [
            {"id": str(rid), "messagesAdded": [{"message": {"id": mid}}]} for rid, mid in added
        ],
    }


@pytest.fixture
def store():
    store = InMemorySyncStore()
    connect_mailbox(store, address=ADDRESS, history_id=100)
    return store


@pytest.fixture
def webhook_client(store):
    app = FastAPI()
    app.include_router(build_router(store_factory=lambda: store, verifier=AllowAllPubSubVerifier()))
    return TestClient(app)


def push(webhook_client, *, history_id: int, message_id: str):
    return webhook_client.post(
        "/gmail/pubsub/push",
        json=pubsub_push_body(ADDRESS, history_id, message_id=message_id, subscription=SUBSCRIPTION),
    )


def test_push_queues_a_job_then_worker_stores_mail_and_threads_are_non_empty(store, webhook_client):
    # 1. Pub/Sub push -> webhook returns 200 only after the job is durable.
    assert push(webhook_client, history_id=200, message_id="pubsub-1").status_code == 200

    # 2. The job actually landed in the queue (the link that was missing when
    #    no worker ran was consumption, not production).
    job = store.claim_job(owner="assert")
    assert job is not None and job.status == "running"
    store.fail_job(job.id, "released for the worker", max_attempts=5)  # hand it back

    # 3. Worker claims + processes it against a scripted Gmail.
    sink = InMemoryChunkSink()
    client = FakeGmailClient(
        email_address=ADDRESS,
        messages={"m-1": message("m-1", thread="t-1")},
        history_pages=[history_page(200, added=[(150, "m-1")])],
    )
    worker = SyncWorker(store, client_factory=lambda _m: client, sink_factory=lambda _m: sink)
    outcomes = worker.drain()

    # 4. Stored emails.
    assert [o.added for o in outcomes] == [1]
    assert "m-1" in sink.emails
    assert sink.emails["m-1"].thread_id == "t-1"

    # 5. Non-empty threads -- this is what /threads reads (DISTINCT thread_id over
    #    the persisted chunks in the ParadeDB path).
    threads = sorted({chunk.thread_id for chunk in sink.all_chunks()})
    assert threads == ["t-1"]


def test_duplicate_push_does_not_double_store(store, webhook_client):
    assert push(webhook_client, history_id=200, message_id="pubsub-dup").status_code == 200
    # Redelivery of the same Pub/Sub message: acked, no second job.
    assert push(webhook_client, history_id=200, message_id="pubsub-dup").status_code == 200

    sink = InMemoryChunkSink()
    client = FakeGmailClient(
        email_address=ADDRESS,
        messages={"m-1": message("m-1", thread="t-1")},
        history_pages=[history_page(200, added=[(150, "m-1")])],
    )
    SyncWorker(store, client_factory=lambda _m: client, sink_factory=lambda _m: sink).drain()

    # One job, one message fetched.
    assert [c for c in client.calls if c[0] == "get_message"] == [("get_message", "m-1")]


def test_inline_worker_thread_drains_a_pushed_job(store, webhook_client):
    """The production wiring: a background thread drains the queue with no manual
    drain() call -- the fix for jobs piling up unclaimed."""
    assert push(webhook_client, history_id=200, message_id="pubsub-1").status_code == 200

    sink = InMemoryChunkSink()
    client = FakeGmailClient(
        email_address=ADDRESS,
        messages={"m-1": message("m-1", thread="t-1")},
        history_pages=[history_page(200, added=[(150, "m-1")])],
    )
    worker = SyncWorker(store, client_factory=lambda _m: client, sink_factory=lambda _m: sink)
    inline = InlineWorker(worker, poll_interval=0.01).start()
    try:
        _wait_until(lambda: "m-1" in sink.emails)
    finally:
        inline.stop(timeout=2.0)

    assert sink.emails["m-1"].thread_id == "t-1"


def _wait_until(predicate, *, timeout: float = 3.0, interval: float = 0.01) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition not met before timeout")
