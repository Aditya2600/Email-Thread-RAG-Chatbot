"""Gmail OAuth: authorization URL, PKCE, single-use state, token exchange.

Scope is exactly ``gmail.readonly`` -- this stage reads mail and never sends,
modifies, or deletes it, so nothing broader is requested.

Nothing here logs or returns an authorization code, access token, or refresh
token. The refresh token leaves this module only as ciphertext.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable
from urllib.parse import urlencode

from email_thread_rag.gmail.client import GMAIL_READONLY_SCOPE, GOOGLE_TOKEN_URL
from email_thread_rag.gmail.models import OAuthState
from email_thread_rag.gmail.store import SyncStore, utcnow

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
STATE_TTL = timedelta(minutes=10)


class OAuthError(RuntimeError):
    """OAuth failure. Message is safe to log: never contains code or tokens."""


def _urlsafe_token(n_bytes: int = 32) -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(n_bytes)).rstrip(b"=").decode("ascii")


def generate_pkce_pair() -> tuple[str, str]:
    """(code_verifier, code_challenge) for PKCE S256."""
    verifier = _urlsafe_token(32)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


@dataclass
class AuthorizationRequest:
    url: str
    state: str


def start_authorization(
    store: SyncStore,
    *,
    client_id: str,
    redirect_uri: str,
    tenant_id: str,
    mailbox_id: str,
    ttl: timedelta = STATE_TTL,
) -> AuthorizationRequest:
    """Persist an expiring, single-use state + PKCE verifier, return the consent URL."""
    state = _urlsafe_token(32)  # CSPRNG; the state is the CSRF defence
    verifier, challenge = generate_pkce_pair()
    store.create_oauth_state(
        OAuthState(
            state=state,
            tenant_id=tenant_id,
            mailbox_id=mailbox_id,
            code_verifier=verifier,
            redirect_uri=redirect_uri,
            expires_at=utcnow() + ttl,
        )
    )
    query = urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": GMAIL_READONLY_SCOPE,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            # Required to receive a refresh token at all; consent forces a new
            # one even if the user has approved this client before.
            "access_type": "offline",
            "prompt": "consent",
            "include_granted_scopes": "false",
        }
    )
    return AuthorizationRequest(url=f"{GOOGLE_AUTH_URL}?{query}", state=state)


TokenExchanger = Callable[[dict[str, str]], dict[str, Any]]


def http_token_exchanger(form: dict[str, str]) -> dict[str, Any]:
    """Real token endpoint call. Tests never reach this; they inject a fake."""
    import httpx

    response = httpx.post(GOOGLE_TOKEN_URL, data=form, timeout=30.0)
    if response.status_code >= 400:
        # Google's error body names the failure mode (invalid_grant, ...) and
        # does not echo the code back; the request form (which holds the code
        # and client secret) is never included.
        raise OAuthError(f"Token exchange failed with HTTP {response.status_code}.")
    return response.json()


def exchange_code(
    store: SyncStore,
    *,
    state: str,
    code: str,
    client_id: str,
    client_secret: str,
    exchanger: TokenExchanger | None = None,
) -> tuple[OAuthState, str]:
    """Consume the state (once) and swap the code for a refresh token.

    Returns (state_record, refresh_token). The caller encrypts the refresh
    token immediately; it is never persisted or logged in plaintext.

    ``exchanger`` defaults to the real token endpoint, but is resolved at call
    time rather than bound as a default argument -- a default would capture the
    real HTTP function at import and let a test that thought it had substituted
    a fake quietly call Google for real.
    """
    exchanger = exchanger or http_token_exchanger
    record = store.consume_oauth_state(state)
    if record is None:
        # One message for unknown/expired/already-used: an attacker probing
        # states learns nothing from the difference.
        raise OAuthError("OAuth state is invalid, expired, or already used.")

    payload = exchanger(
        {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": record.redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": record.code_verifier,
        }
    )
    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        raise OAuthError(
            "Token response contained no refresh token (the account may already be "
            "authorized; re-consent with prompt=consent)."
        )
    granted = payload.get("scope", GMAIL_READONLY_SCOPE).split()
    if GMAIL_READONLY_SCOPE not in granted:
        raise OAuthError("Consent did not grant the gmail.readonly scope.")
    return record, refresh_token


def refresh_access_token(
    *,
    refresh_token: str,
    client_id: str,
    client_secret: str,
    exchanger: TokenExchanger | None = None,
) -> str:
    exchanger = exchanger or http_token_exchanger
    payload = exchanger(
        {
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
        }
    )
    access_token = payload.get("access_token")
    if not access_token:
        raise OAuthError("Token refresh response contained no access token.")
    return access_token
