"""Pub/Sub push endpoint tests: authentication, decoding, durability, dedup.

Everything runs against fakes through FastAPI's TestClient: no Google, no
network, no credentials.
"""

from __future__ import annotations

import base64
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from email_thread_rag.gmail.fakes import FakePubSubVerifier, pubsub_push_body
from email_thread_rag.gmail.store import InMemorySyncStore
from email_thread_rag.gmail.webhook import InvalidPushPayload, build_router, decode_notification
from gmail_store_contract import connect_mailbox

SUBSCRIPTION = "projects/demo/subscriptions/gmail-push"
AUTH = {"Authorization": "Bearer fake-oidc-token"}


@pytest.fixture
def store():
    return InMemorySyncStore()


@pytest.fixture
def client(store):
    app = FastAPI()
    app.include_router(
        build_router(
            store_factory=lambda: store,
            verifier=FakePubSubVerifier(expected_subscription=SUBSCRIPTION),
        )
    )
    return TestClient(app)


def push(client, *, address="user@example.com", history_id=150, message_id="pubsub-1", headers=AUTH):
    return client.post(
        "/gmail/pubsub/push",
        json=pubsub_push_body(address, history_id, message_id=message_id, subscription=SUBSCRIPTION),
        headers=headers,
    )


# --- Decoding -------------------------------------------------------------
def test_decode_base64url_notification():
    body = pubsub_push_body("user@example.com", 4242, message_id="m-1", subscription=SUBSCRIPTION)
    notification = decode_notification(body)

    assert notification.email_address == "user@example.com"
    assert notification.history_id == 4242
    assert notification.pubsub_message_id == "m-1"


def test_decode_accepts_a_string_history_id():
    # Gmail sends historyId as a JSON number or string depending on the client.
    data = base64.urlsafe_b64encode(
        json.dumps({"emailAddress": "user@example.com", "historyId": "99"}).encode()
    ).decode()
    notification = decode_notification({"message": {"data": data, "messageId": "m-1"}})
    assert notification.history_id == 99 and isinstance(notification.history_id, int)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"message": {}},
        {"message": {"messageId": "m-1"}},
        {"message": {"data": "!!!not-base64!!!", "messageId": "m-1"}},
        {"message": {"data": base64.urlsafe_b64encode(b"{}").decode(), "messageId": "m-1"}},
    ],
)
def test_malformed_payloads_are_rejected(body):
    with pytest.raises(InvalidPushPayload):
        decode_notification(body)


# --- Authentication -------------------------------------------------------
def test_unauthenticated_push_is_rejected_and_creates_no_job(client, store):
    connect_mailbox(store)
    response = push(client, headers={})

    assert response.status_code == 403
    assert store.claim_job(owner="w") is None


def test_push_from_an_unexpected_subscription_is_rejected(client, store):
    connect_mailbox(store)
    response = client.post(
        "/gmail/pubsub/push",
        json=pubsub_push_body("user@example.com", 150, message_id="m-1", subscription="projects/x/subscriptions/evil"),
        headers=AUTH,
    )

    assert response.status_code == 403
    assert store.claim_job(owner="w") is None


def test_malformed_body_is_dropped_not_retried_forever(client, store):
    connect_mailbox(store)
    response = client.post(
        "/gmail/pubsub/push",
        json={"message": {"messageId": "m-1"}, "subscription": SUBSCRIPTION},
        headers=AUTH,
    )
    assert response.status_code == 400


# --- Durability + dedup ---------------------------------------------------
def test_valid_push_creates_a_durable_job_before_acking(client, store):
    mailbox = connect_mailbox(store)
    response = push(client, history_id=150)

    assert response.status_code == 200
    # The 200 is only meaningful if the job is already readable from the store.
    job = store.claim_job(owner="w")
    assert job is not None
    assert job.mailbox_db_id == mailbox.id
    assert job.requested_history_id == 150


def test_duplicate_delivery_does_not_create_duplicate_work(client, store):
    connect_mailbox(store)
    assert push(client, message_id="pubsub-same").status_code == 200
    assert push(client, message_id="pubsub-same").status_code == 200

    assert store.claim_job(owner="w") is not None
    assert store.claim_job(owner="w") is None


def test_repeated_notifications_coalesce_to_the_highest_history_id(client, store):
    connect_mailbox(store)
    push(client, history_id=9, message_id="pubsub-a")
    push(client, history_id=10, message_id="pubsub-b")

    job = store.claim_job(owner="w")
    # Numeric, not lexicographic: '10' would sort below '9' as text.
    assert job.requested_history_id == 10
    assert store.claim_job(owner="w") is None


def test_push_for_an_unknown_address_is_acked_without_work(client, store):
    connect_mailbox(store)
    assert push(client, address="stranger@example.com").status_code == 200
    assert store.claim_job(owner="w") is None


def test_webhook_never_calls_gmail_or_indexes(client, store, monkeypatch):
    """The push path must not do Gmail I/O -- that is what the ack deadline is for."""
    import email_thread_rag.gmail.sync as sync_module

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the webhook must not sync inline")

    monkeypatch.setattr(sync_module, "run_sync", explode)
    monkeypatch.setattr(sync_module, "run_delta_sync", explode)

    connect_mailbox(store)
    assert push(client).status_code == 200
