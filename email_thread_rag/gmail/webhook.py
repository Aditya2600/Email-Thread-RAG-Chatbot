"""Pub/Sub push endpoint.

The webhook does exactly one thing: verify the push, then durably record a
sync job. It never calls Gmail and never parses or indexes mail -- that is the
worker's job. A push handler that did network I/O would hold Pub/Sub's ack
deadline open and get the notification redelivered under load.

HTTP 200 is returned only after the job transaction has committed, so an ack
always means the work is durable.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from typing import Any, Callable

from fastapi import APIRouter, Header, Request, Response

from email_thread_rag.gmail.client import PubSubPushVerifier, PubSubVerificationError
from email_thread_rag.gmail.models import GmailNotification
from email_thread_rag.gmail.store import SyncStore

logger = logging.getLogger(__name__)


class InvalidPushPayload(ValueError):
    """Body is not a Gmail notification we can act on."""


def decode_notification(body: dict[str, Any]) -> GmailNotification:
    """Decode Base64URL ``message.data`` into emailAddress + historyId."""
    message = body.get("message")
    if not isinstance(message, dict):
        raise InvalidPushPayload("push body has no message object")
    pubsub_message_id = message.get("messageId") or message.get("message_id")
    if not pubsub_message_id:
        raise InvalidPushPayload("push message has no messageId")
    data = message.get("data")
    if not data:
        raise InvalidPushPayload("push message has no data")
    try:
        padding = "=" * (-len(data) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(data + padding))
    except (binascii.Error, ValueError) as exc:
        raise InvalidPushPayload("push message data is not base64url-encoded JSON") from exc

    email_address = decoded.get("emailAddress")
    history_id = decoded.get("historyId")
    if not email_address or history_id is None:
        raise InvalidPushPayload("notification is missing emailAddress or historyId")
    return GmailNotification(
        email_address=email_address,
        # Gmail sends historyId as a number or a string depending on the
        # client; int() keeps every later comparison numeric.
        history_id=int(history_id),
        pubsub_message_id=str(pubsub_message_id),
    )


def build_router(
    *, store_factory: Callable[[], SyncStore], verifier: PubSubPushVerifier, path: str = "/gmail/pubsub/push"
) -> APIRouter:
    router = APIRouter()

    @router.post(path)
    async def gmail_push(
        request: Request, authorization: str | None = Header(default=None)
    ) -> Response:
        body = await request.json()
        try:
            verifier.verify(
                authorization_header=authorization, subscription=body.get("subscription")
            )
        except PubSubVerificationError as exc:
            logger.warning("rejected gmail push: %s", exc)
            # 403: Pub/Sub will retry, but an unauthenticated caller should
            # never be told whether the address exists.
            return Response(status_code=403)

        try:
            notification = decode_notification(body)
        except InvalidPushPayload as exc:
            # 400 and ack: retrying a malformed body forever helps nobody.
            logger.warning("dropping malformed gmail push: %s", exc)
            return Response(status_code=400)

        job = store_factory().record_notification(notification)
        if job is None:
            # Redelivered message, or an address with no live mailbox. Both are
            # "nothing to do" -- ack so Pub/Sub stops retrying.
            return Response(status_code=200)

        # The job transaction has committed by now; acking is safe.
        logger.info(
            "queued gmail sync job %s for mailbox %s at historyId %s",
            job.id,
            job.mailbox_db_id,
            job.requested_history_id,
        )
        return Response(status_code=200)

    return router
