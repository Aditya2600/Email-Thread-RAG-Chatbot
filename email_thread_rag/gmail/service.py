"""Mailbox lifecycle: connect (watch), renew before expiry, disconnect (stop).

Watch state is persisted *before* the mailbox is treated as active, so a
notification can never arrive for a mailbox whose historyId we failed to store.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from email_thread_rag.gmail.cipher import TokenCipher
from email_thread_rag.gmail.client import GmailClient
from email_thread_rag.gmail.models import Mailbox
from email_thread_rag.gmail.store import SyncStore, utcnow

logger = logging.getLogger(__name__)

# Gmail expires a watch after 7 days and documents re-calling watch at least
# every 7 days; renewing a day early leaves room for a failed run to retry
# before the mailbox goes silent.
WATCH_RENEWAL_MARGIN = timedelta(days=1)


def _expiration_to_datetime(expiration_ms: int) -> datetime:
    return datetime.fromtimestamp(int(expiration_ms) / 1000, tz=timezone.utc)


def connect_mailbox(
    store: SyncStore,
    client: GmailClient,
    cipher: TokenCipher,
    *,
    tenant_id: str,
    mailbox_id: str,
    refresh_token: str,
    topic_name: str,
) -> Mailbox:
    """Store the encrypted token, start the Gmail watch, persist its state.

    Order is deliberate: the mailbox row (status 'pending') exists before
    users.watch, and only becomes 'active' after the returned historyId and
    expiration are committed. A crash between the two leaves a pending mailbox
    that a retry re-watches -- never an active mailbox with no cursor.
    """
    profile = client.get_profile()
    email_address = profile["emailAddress"]

    mailbox = store.upsert_mailbox(
        tenant_id=tenant_id,
        mailbox_id=mailbox_id,
        email_address=email_address,
        # Plaintext ends here: only ciphertext is passed on, and the local
        # `refresh_token` name is never logged or returned.
        refresh_token_ciphertext=cipher.encrypt(refresh_token),
        token_key_id=cipher.key_id,
    )

    watch = client.watch(topic_name=topic_name)
    return store.activate_watch(
        mailbox.id,
        history_id=int(watch["historyId"]),
        topic=topic_name,
        expiration=_expiration_to_datetime(watch["expiration"]),
    )


def renew_watch(store: SyncStore, client: GmailClient, mailbox: Mailbox, *, topic_name: str) -> Mailbox:
    """Re-call users.watch and persist the new expiration.

    The cursor is untouched: ``activate_watch`` only seeds it when it is NULL,
    so a renewal cannot skip past history the worker has not synced yet.
    """
    watch = client.watch(topic_name=topic_name)
    return store.activate_watch(
        mailbox.id,
        history_id=int(watch["historyId"]),
        topic=topic_name,
        expiration=_expiration_to_datetime(watch["expiration"]),
    )


def renew_expiring_watches(
    store: SyncStore, client_factory, *, topic_name: str, margin: timedelta = WATCH_RENEWAL_MARGIN
) -> list[Mailbox]:
    """Renew every watch expiring within ``margin``. One bad mailbox does not
    stop the others."""
    due = store.mailboxes_due_for_watch_renewal(before=utcnow() + margin)
    renewed: list[Mailbox] = []
    for mailbox in due:
        try:
            renewed.append(renew_watch(store, client_factory(mailbox), mailbox, topic_name=topic_name))
        except Exception as exc:  # noqa: BLE001 - keep renewing the rest
            logger.warning("watch renewal failed for mailbox %s: %s", mailbox.id, type(exc).__name__)
            store.set_mailbox_status(mailbox.id, "error", error=f"watch renewal failed: {type(exc).__name__}")
    return renewed


def disconnect_mailbox(store: SyncStore, client: GmailClient, mailbox: Mailbox) -> None:
    """Stop the Gmail watch and drop the stored refresh token.

    users.stop is best-effort: if it fails, the local credential is still
    discarded, which is what actually revokes our access to the mailbox.
    """
    try:
        client.stop_watch()
    except Exception as exc:  # noqa: BLE001
        logger.warning("users.stop failed for mailbox %s: %s", mailbox.id, type(exc).__name__)
    store.disconnect_mailbox(mailbox.id)
