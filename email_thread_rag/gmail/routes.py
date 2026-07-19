"""OAuth connect/callback routes.

API only -- no UI is part of this stage. Responses never contain a token, an
authorization code, or a raw credential: the callback returns the connected
address and status, nothing more.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from email_thread_rag.gmail.oauth import OAuthError, exchange_code, start_authorization

logger = logging.getLogger(__name__)


def build_oauth_router(
    *, settings, store_factory, client_factory, cipher, token_exchanger=None, prefix: str = "/gmail/oauth"
):
    """``client_factory(refresh_token)`` builds a GmailClient for the freshly
    connected account; injected so tests never construct a real one.

    ``token_exchanger`` defaults to the real Google token endpoint; tests pass a
    fake so the callback never leaves the process.
    """
    router = APIRouter(prefix=prefix)

    def _require_config() -> None:
        missing = [
            name
            for name, value in (
                ("GMAIL_CLIENT_ID", settings.gmail_client_id),
                ("GMAIL_CLIENT_SECRET", settings.gmail_client_secret),
                ("GMAIL_REDIRECT_URI", settings.gmail_redirect_uri),
                ("GMAIL_PUBSUB_TOPIC", settings.gmail_pubsub_topic),
            )
            if not value
        ]
        if missing:
            raise HTTPException(status_code=503, detail=f"Gmail is not configured: missing {', '.join(missing)}")

    @router.get("/start")
    def start(tenant_id: str = Query(...), mailbox_id: str = Query(...)) -> dict:
        _require_config()
        request = start_authorization(
            store_factory(),
            client_id=settings.gmail_client_id,
            redirect_uri=settings.gmail_redirect_uri,
            tenant_id=tenant_id,
            mailbox_id=mailbox_id,
        )
        # The state is in the URL the user is about to follow; returning it is
        # not a leak. The PKCE verifier stays server-side.
        return {"authorization_url": request.url}

    @router.get("/callback")
    def callback(code: str = Query(...), state: str = Query(...)) -> dict:
        _require_config()
        from email_thread_rag.gmail.service import connect_mailbox

        store = store_factory()
        try:
            record, refresh_token = exchange_code(
                store,
                state=state,
                code=code,
                client_id=settings.gmail_client_id,
                client_secret=settings.gmail_client_secret,
                exchanger=token_exchanger,
            )
        except OAuthError as exc:
            # exc is written to never contain the code or a token.
            logger.warning("gmail oauth callback rejected: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from None

        mailbox = connect_mailbox(
            store,
            client_factory(refresh_token),
            cipher,
            tenant_id=record.tenant_id,
            mailbox_id=record.mailbox_id,
            refresh_token=refresh_token,
            topic_name=settings.gmail_pubsub_topic,
        )
        # Deliberately narrow: address + status + watch expiry, never a token.
        return {
            "tenant_id": mailbox.tenant_id,
            "mailbox_id": mailbox.mailbox_id,
            "email_address": mailbox.email_address,
            "status": mailbox.status,
            "watch_expiration": mailbox.watch_expiration.isoformat() if mailbox.watch_expiration else None,
        }

    return router
