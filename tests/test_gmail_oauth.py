"""OAuth + refresh-token-at-rest tests. No real OAuth flow, no network.

The token endpoint is a fake callable; Google is never contacted.
"""

from __future__ import annotations

import base64
import hashlib
import os
from datetime import timedelta
from urllib.parse import parse_qs, urlparse

import pytest

from email_thread_rag.gmail.cipher import AesGcmTokenCipher, TokenCipherError
from email_thread_rag.gmail.client import GMAIL_READONLY_SCOPE
from email_thread_rag.gmail.fakes import FakeGmailClient
from email_thread_rag.gmail.models import OAuthState
from email_thread_rag.gmail.oauth import (
    OAuthError,
    exchange_code,
    generate_pkce_pair,
    refresh_access_token,
    start_authorization,
)
from email_thread_rag.gmail.service import connect_mailbox
from email_thread_rag.gmail.store import InMemorySyncStore, utcnow

REFRESH_TOKEN = "1//super-secret-refresh-token"
AUTH_CODE = "4/secret-authorization-code"


@pytest.fixture
def store():
    return InMemorySyncStore()


@pytest.fixture
def cipher():
    return AesGcmTokenCipher(os.urandom(32), key_id="test-key")


def fake_exchanger(response: dict):
    """A stand-in token endpoint that records the form it was posted."""
    calls: list[dict] = []

    def exchanger(form: dict) -> dict:
        calls.append(form)
        return response

    exchanger.calls = calls
    return exchanger


def start(store, **overrides):
    kwargs = dict(
        client_id="client-id.apps.googleusercontent.com",
        redirect_uri="https://app.example.com/oauth/callback",
        tenant_id="acme",
        mailbox_id="inbox",
    )
    kwargs.update(overrides)
    return start_authorization(store, **kwargs)


# --- Authorization request ------------------------------------------------
def test_authorization_url_requests_only_gmail_readonly_with_pkce(store):
    request = start(store)
    query = parse_qs(urlparse(request.url).query)

    assert query["scope"] == [GMAIL_READONLY_SCOPE]
    assert query["code_challenge_method"] == ["S256"]
    assert query["access_type"] == ["offline"]
    assert query["state"] == [request.state]
    assert "code_challenge" in query


def test_pkce_challenge_is_the_s256_of_the_verifier():
    verifier, challenge = generate_pkce_pair()
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge == expected


def test_states_are_random_and_unguessable(store):
    states = {start(store).state for _ in range(20)}
    assert len(states) == 20
    assert all(len(state) >= 32 for state in states)


def test_verifier_never_appears_in_the_authorization_url(store):
    request = start(store)
    verifier = store.consume_oauth_state(request.state).code_verifier
    # PKCE's whole point: the verifier stays server-side until the exchange.
    assert verifier not in request.url


# --- State lifecycle ------------------------------------------------------
def test_state_is_single_use_across_the_exchange(store):
    request = start(store)
    exchanger = fake_exchanger({"refresh_token": REFRESH_TOKEN, "scope": GMAIL_READONLY_SCOPE})

    record, refresh_token = exchange_code(
        store,
        state=request.state,
        code=AUTH_CODE,
        client_id="cid",
        client_secret="csecret",
        exchanger=exchanger,
    )
    assert refresh_token == REFRESH_TOKEN
    assert record.tenant_id == "acme"

    # Replaying the same callback must fail, and must not hit the token endpoint.
    with pytest.raises(OAuthError, match="invalid, expired, or already used"):
        exchange_code(
            store,
            state=request.state,
            code=AUTH_CODE,
            client_id="cid",
            client_secret="csecret",
            exchanger=exchanger,
        )
    assert len(exchanger.calls) == 1


def test_expired_state_is_rejected(store):
    request = start(store)
    store._oauth_states[request.state] = (
        OAuthState(
            state=request.state,
            tenant_id="acme",
            mailbox_id="inbox",
            code_verifier="v",
            redirect_uri="https://app.example.com/oauth/callback",
            expires_at=utcnow() - timedelta(seconds=1),
        ),
        None,
    )
    exchanger = fake_exchanger({"refresh_token": REFRESH_TOKEN})
    with pytest.raises(OAuthError):
        exchange_code(
            store, state=request.state, code=AUTH_CODE, client_id="c", client_secret="s", exchanger=exchanger
        )
    assert exchanger.calls == []


def test_exchange_sends_the_pkce_verifier(store):
    request = start(store)
    exchanger = fake_exchanger({"refresh_token": REFRESH_TOKEN})
    exchange_code(
        store, state=request.state, code=AUTH_CODE, client_id="c", client_secret="s", exchanger=exchanger
    )
    form = exchanger.calls[0]
    assert form["code_verifier"]
    assert form["grant_type"] == "authorization_code"


def test_exchange_rejects_a_response_without_the_readonly_scope(store):
    request = start(store)
    exchanger = fake_exchanger({"refresh_token": REFRESH_TOKEN, "scope": "https://mail.google.com/"})
    with pytest.raises(OAuthError, match="gmail.readonly"):
        exchange_code(
            store, state=request.state, code=AUTH_CODE, client_id="c", client_secret="s", exchanger=exchanger
        )


def test_exchange_rejects_a_response_without_a_refresh_token(store):
    request = start(store)
    exchanger = fake_exchanger({"access_token": "ya29.something"})
    with pytest.raises(OAuthError, match="no refresh token"):
        exchange_code(
            store, state=request.state, code=AUTH_CODE, client_id="c", client_secret="s", exchanger=exchanger
        )


def test_refresh_access_token_uses_the_refresh_grant():
    exchanger = fake_exchanger({"access_token": "ya29.access"})
    token = refresh_access_token(
        refresh_token=REFRESH_TOKEN, client_id="c", client_secret="s", exchanger=exchanger
    )
    assert token == "ya29.access"
    assert exchanger.calls[0]["grant_type"] == "refresh_token"


# --- Encryption at rest ---------------------------------------------------
def test_refresh_token_round_trips_through_the_cipher(cipher):
    ciphertext = cipher.encrypt(REFRESH_TOKEN)
    assert cipher.decrypt(ciphertext) == REFRESH_TOKEN


def test_ciphertext_does_not_contain_the_plaintext(cipher):
    ciphertext = cipher.encrypt(REFRESH_TOKEN)
    assert REFRESH_TOKEN.encode() not in ciphertext
    assert REFRESH_TOKEN not in base64.b64encode(ciphertext).decode()


def test_same_token_encrypts_differently_each_time(cipher):
    # Random nonce per encrypt: equal ciphertexts must not reveal equal tokens.
    assert cipher.encrypt(REFRESH_TOKEN) != cipher.encrypt(REFRESH_TOKEN)


def test_wrong_key_cannot_decrypt(cipher):
    ciphertext = cipher.encrypt(REFRESH_TOKEN)
    other = AesGcmTokenCipher(os.urandom(32))
    with pytest.raises(TokenCipherError):
        other.decrypt(ciphertext)


def test_tampered_ciphertext_is_rejected(cipher):
    ciphertext = bytearray(cipher.encrypt(REFRESH_TOKEN))
    ciphertext[-1] ^= 0xFF
    with pytest.raises(TokenCipherError):
        cipher.decrypt(bytes(ciphertext))


def test_cipher_errors_never_echo_key_or_token(cipher):
    with pytest.raises(TokenCipherError) as excinfo:
        AesGcmTokenCipher(b"too-short")
    assert "too-short" not in str(excinfo.value)

    with pytest.raises(TokenCipherError) as excinfo:
        AesGcmTokenCipher.from_base64_key("not base64!!")
    assert "not base64!!" not in str(excinfo.value)


def test_connected_mailbox_stores_only_ciphertext(store, cipher):
    client = FakeGmailClient(email_address="user@example.com", watch_history_id=777)
    mailbox = connect_mailbox(
        store,
        client,
        cipher,
        tenant_id="acme",
        mailbox_id="inbox",
        refresh_token=REFRESH_TOKEN,
        topic_name="projects/demo/topics/gmail-sync",
    )

    stored = store.get_mailbox("acme", "inbox")
    assert stored.refresh_token_ciphertext is not None
    assert REFRESH_TOKEN.encode() not in stored.refresh_token_ciphertext
    assert cipher.decrypt(stored.refresh_token_ciphertext) == REFRESH_TOKEN
    # Neither the record's repr nor the mailbox object leaks the token.
    assert REFRESH_TOKEN not in repr(stored)
    assert REFRESH_TOKEN not in repr(mailbox)
    assert mailbox.status == "active"
    assert mailbox.last_committed_history_id == 777


def test_oauth_state_repr_hides_the_pkce_verifier(store):
    request = start(store)
    record = store.consume_oauth_state(request.state)
    assert record.code_verifier not in repr(record)


def test_oauth_errors_never_contain_the_authorization_code(store):
    request = start(store)
    exchanger = fake_exchanger({})
    with pytest.raises(OAuthError) as excinfo:
        exchange_code(
            store, state=request.state, code=AUTH_CODE, client_id="c", client_secret="s", exchanger=exchanger
        )
    assert AUTH_CODE not in str(excinfo.value)
