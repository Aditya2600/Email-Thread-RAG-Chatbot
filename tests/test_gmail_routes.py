"""OAuth routes + lazy route mounting + watch lifecycle.

No real OAuth flow: the token endpoint and Gmail are fakes.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from email_thread_rag.config import Settings
from email_thread_rag.gmail.cipher import AesGcmTokenCipher
from email_thread_rag.gmail.fakes import FakeGmailClient
from email_thread_rag.gmail.routes import build_oauth_router
from email_thread_rag.gmail.service import (
    connect_mailbox,
    disconnect_mailbox,
    renew_expiring_watches,
    renew_watch,
)
from email_thread_rag.gmail.store import InMemorySyncStore, utcnow

REFRESH_TOKEN = "1//super-secret-refresh-token"
TOPIC = "projects/demo/topics/gmail-sync"


@pytest.fixture
def store():
    return InMemorySyncStore()


@pytest.fixture
def cipher():
    return AesGcmTokenCipher(os.urandom(32), key_id="test-key")


@pytest.fixture
def gmail_client():
    return FakeGmailClient(email_address="user@example.com", watch_history_id=500)


def configured_settings(**overrides) -> Settings:
    values = dict(
        gmail_client_id="client-id",
        gmail_client_secret="client-secret",
        gmail_redirect_uri="https://app.example.com/gmail/oauth/callback",
        gmail_pubsub_topic=TOPIC,
        gmail_pubsub_subscription="projects/demo/subscriptions/gmail-push",
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def client(store, cipher, gmail_client):
    from email_thread_rag.gmail.client import GMAIL_READONLY_SCOPE

    app = FastAPI()
    app.include_router(
        build_oauth_router(
            settings=configured_settings(),
            store_factory=lambda: store,
            client_factory=lambda _refresh_token: gmail_client,
            cipher=cipher,
            # Injected, not monkeypatched: the callback must never reach Google.
            token_exchanger=lambda form: {"refresh_token": REFRESH_TOKEN, "scope": GMAIL_READONLY_SCOPE},
        )
    )
    return TestClient(app)


# --- OAuth routes ---------------------------------------------------------
def test_start_returns_a_consent_url(client):
    response = client.get("/gmail/oauth/start", params={"tenant_id": "acme", "mailbox_id": "inbox"})
    assert response.status_code == 200

    url = response.json()["authorization_url"]
    assert "gmail.readonly" in url
    assert "code_challenge_method=S256" in url


def test_callback_connects_the_mailbox_without_exposing_a_token(client, store, cipher):
    start = client.get("/gmail/oauth/start", params={"tenant_id": "acme", "mailbox_id": "inbox"})
    state = start.json()["authorization_url"].split("state=")[1].split("&")[0]

    response = client.get("/gmail/oauth/callback", params={"code": "4/auth-code", "state": state})
    assert response.status_code == 200

    body = response.json()
    assert body["email_address"] == "user@example.com"
    assert body["status"] == "active"
    # The response body must never carry the credential.
    assert REFRESH_TOKEN not in response.text
    assert "refresh_token" not in body

    stored = store.get_mailbox("acme", "inbox")
    assert cipher.decrypt(stored.refresh_token_ciphertext) == REFRESH_TOKEN
    assert stored.last_committed_history_id == 500


def test_callback_rejects_a_replayed_state(client):
    start = client.get("/gmail/oauth/start", params={"tenant_id": "acme", "mailbox_id": "inbox"})
    state = start.json()["authorization_url"].split("state=")[1].split("&")[0]

    assert client.get("/gmail/oauth/callback", params={"code": "4/c", "state": state}).status_code == 200
    replay = client.get("/gmail/oauth/callback", params={"code": "4/c", "state": state})
    assert replay.status_code == 400
    assert "4/c" not in replay.text  # the code never echoes back


def test_callback_rejects_an_unknown_state(client):
    response = client.get("/gmail/oauth/callback", params={"code": "4/c", "state": "never-issued"})
    assert response.status_code == 400


def test_routes_report_missing_configuration(store, cipher, gmail_client):
    app = FastAPI()
    app.include_router(
        build_oauth_router(
            settings=configured_settings(gmail_client_id=None),
            store_factory=lambda: store,
            client_factory=lambda _t: gmail_client,
            cipher=cipher,
            token_exchanger=lambda form: {},
        )
    )
    response = TestClient(app).get(
        "/gmail/oauth/start", params={"tenant_id": "acme", "mailbox_id": "inbox"}
    )
    assert response.status_code == 503
    assert "GMAIL_CLIENT_ID" in response.json()["detail"]


# --- Lazy mounting --------------------------------------------------------
def test_gmail_routes_are_not_mounted_without_configuration():
    from email_thread_rag.app.main import mount_gmail_routes

    app = FastAPI()
    # Memory backend, no Gmail config: no Gmail routes, no DB connection attempt.
    assert mount_gmail_routes(app, Settings(rag_backend="memory", database_url=None)) is False
    assert [route.path for route in app.routes if "gmail" in route.path] == []


# --- Watch lifecycle ------------------------------------------------------
def test_connect_persists_watch_state_before_marking_active(store, cipher, gmail_client):
    mailbox = connect_mailbox(
        store,
        gmail_client,
        cipher,
        tenant_id="acme",
        mailbox_id="inbox",
        refresh_token=REFRESH_TOKEN,
        topic_name=TOPIC,
    )
    assert mailbox.status == "active"
    assert mailbox.last_committed_history_id == 500
    assert mailbox.watch_expiration is not None
    assert ("watch", TOPIC) in gmail_client.calls


def test_renewal_does_not_move_a_cursor_the_worker_already_advanced(store, cipher, gmail_client):
    mailbox = connect_mailbox(
        store,
        gmail_client,
        cipher,
        tenant_id="acme",
        mailbox_id="inbox",
        refresh_token=REFRESH_TOKEN,
        topic_name=TOPIC,
    )
    store.commit_history_cursor(mailbox.id, 900)

    gmail_client.watch_history_id = 1000
    renewed = renew_watch(store, gmail_client, mailbox, topic_name=TOPIC)

    # Gmail's current historyId must not skip the unsynced window 900..1000.
    assert renewed.last_committed_history_id == 900


def test_renew_expiring_watches_only_touches_the_ones_near_expiry(store, cipher, gmail_client):
    connect_mailbox(
        store,
        gmail_client,
        cipher,
        tenant_id="acme",
        mailbox_id="inbox",
        refresh_token=REFRESH_TOKEN,
        topic_name=TOPIC,
    )
    mailbox = store.get_mailbox("acme", "inbox")

    # Fresh watch: nothing due.
    assert renew_expiring_watches(store, lambda _m: gmail_client, topic_name=TOPIC) == []

    mailbox.watch_expiration = utcnow() + timedelta(hours=2)
    renewed = renew_expiring_watches(store, lambda _m: gmail_client, topic_name=TOPIC)
    assert [m.id for m in renewed] == [mailbox.id]


def test_a_failing_renewal_does_not_stop_the_others(store, cipher):
    for name, address in (("a", "a@example.com"), ("b", "b@example.com")):
        client = FakeGmailClient(email_address=address)
        connect_mailbox(
            store,
            client,
            cipher,
            tenant_id="acme",
            mailbox_id=name,
            refresh_token=REFRESH_TOKEN,
            topic_name=TOPIC,
        )
        store.get_mailbox("acme", name).watch_expiration = utcnow() + timedelta(hours=1)

    def client_factory(mailbox):
        if mailbox.mailbox_id == "a":
            raise RuntimeError("gmail is down for this mailbox")
        return FakeGmailClient(email_address=mailbox.email_address)

    renewed = renew_expiring_watches(store, client_factory, topic_name=TOPIC)

    assert [m.mailbox_id for m in renewed] == ["b"]
    assert store.get_mailbox("acme", "a").status == "error"
    assert store.get_mailbox("acme", "b").status == "active"


def test_disconnect_stops_the_watch_and_drops_the_credential(store, cipher, gmail_client):
    mailbox = connect_mailbox(
        store,
        gmail_client,
        cipher,
        tenant_id="acme",
        mailbox_id="inbox",
        refresh_token=REFRESH_TOKEN,
        topic_name=TOPIC,
    )
    disconnect_mailbox(store, gmail_client, mailbox)

    assert gmail_client.stop_watch_calls == 1
    stored = store.get_mailbox("acme", "inbox")
    assert stored.status == "disconnected"
    assert stored.refresh_token_ciphertext is None


def test_disconnect_drops_the_credential_even_if_stop_fails(store, cipher):
    class StopFailsClient(FakeGmailClient):
        def stop_watch(self):
            raise RuntimeError("gmail unreachable")

    client = StopFailsClient(email_address="user@example.com")
    mailbox = connect_mailbox(
        store,
        client,
        cipher,
        tenant_id="acme",
        mailbox_id="inbox",
        refresh_token=REFRESH_TOKEN,
        topic_name=TOPIC,
    )
    disconnect_mailbox(store, client, mailbox)

    # Discarding the local credential is what actually revokes our access.
    assert store.get_mailbox("acme", "inbox").refresh_token_ciphertext is None
