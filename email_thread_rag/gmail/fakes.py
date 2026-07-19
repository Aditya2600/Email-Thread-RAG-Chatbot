"""In-process fakes for GmailClient / PubSubPushVerifier / TokenCipher.

Every test in the suite runs against these: no credentials, no network, no
Google packages. They ship in the package (not tests/) because the local demo
worker can also run against them without a connected mailbox.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Iterator, Optional, Sequence

from email_thread_rag.gmail.client import GmailHistoryExpired


def encode_pubsub_data(email_address: str, history_id: int) -> str:
    """Base64URL-encode a Gmail notification body exactly as Pub/Sub delivers it."""
    payload = json.dumps({"emailAddress": email_address, "historyId": history_id}).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii")


def pubsub_push_body(email_address: str, history_id: int, *, message_id: str, subscription: str) -> dict:
    return {
        "message": {
            "data": encode_pubsub_data(email_address, history_id),
            "messageId": message_id,
            "publishTime": "2026-01-01T00:00:00.000Z",
        },
        "subscription": subscription,
    }


def build_gmail_message(
    *,
    gmail_id: str,
    thread_id: str,
    history_id: int,
    sender: str,
    to: str,
    subject: str,
    body: str,
    date: str = "Mon, 5 Jan 2026 09:00:00 +0000",
    rfc_message_id: str | None = None,
    in_reply_to: str | None = None,
) -> dict[str, Any]:
    """A minimal but real-shaped ``messages.get(format=full)`` resource."""
    headers = [
        {"name": "From", "value": sender},
        {"name": "To", "value": to},
        {"name": "Subject", "value": subject},
        {"name": "Date", "value": date},
    ]
    if rfc_message_id:
        headers.append({"name": "Message-ID", "value": rfc_message_id})
    if in_reply_to:
        headers.append({"name": "In-Reply-To", "value": in_reply_to})
    return {
        "id": gmail_id,
        "threadId": thread_id,
        "historyId": str(history_id),
        "labelIds": ["INBOX"],
        "payload": {
            "mimeType": "text/plain",
            "headers": headers,
            "body": {"data": base64.urlsafe_b64encode(body.encode("utf-8")).decode("ascii")},
        },
    }


class FakeGmailClient:
    """Scriptable Gmail. Records every call so tests can assert on arguments
    (notably: which startHistoryId the worker actually asked for)."""

    def __init__(
        self,
        *,
        email_address: str = "user@example.com",
        profile_history_id: int = 1000,
        messages: dict[str, dict[str, Any]] | None = None,
        history_pages: list[dict[str, Any]] | None = None,
        watch_history_id: int = 1000,
        watch_expiration_ms: int = 1_800_000_000_000,
    ):
        self.email_address = email_address
        self.profile_history_id = profile_history_id
        self.messages: dict[str, dict[str, Any]] = messages or {}
        self.history_pages = history_pages or []
        self.watch_history_id = watch_history_id
        self.watch_expiration_ms = watch_expiration_ms

        self.history_expired = False
        self.fail_get_message_ids: set[str] = set()
        # Scripted attachment bytes, keyed by (message_id, attachment_id).
        self.attachments: dict[tuple[str, str], bytes] = {}
        self.fail_get_attachment_ids: set[str] = set()
        self.calls: list[tuple[str, Any]] = []
        self.stop_watch_calls = 0

    def get_profile(self) -> dict[str, Any]:
        self.calls.append(("get_profile", None))
        return {"emailAddress": self.email_address, "historyId": self.profile_history_id}

    def watch(self, *, topic_name: str) -> dict[str, Any]:
        self.calls.append(("watch", topic_name))
        return {"historyId": self.watch_history_id, "expiration": self.watch_expiration_ms}

    def stop_watch(self) -> None:
        self.calls.append(("stop_watch", None))
        self.stop_watch_calls += 1

    def list_history(
        self, *, start_history_id: int, history_types: Sequence[str] | None = None
    ) -> Iterator[dict[str, Any]]:
        self.calls.append(("list_history", start_history_id))
        if self.history_expired:
            raise GmailHistoryExpired(f"history gone from {start_history_id}")
        for page in self.history_pages:
            yield page

    def list_messages(self, *, query: str | None = None) -> Iterator[dict[str, Any]]:
        self.calls.append(("list_messages", query))
        yield {"messages": [{"id": mid, "threadId": m.get("threadId")} for mid, m in self.messages.items()]}

    def get_message(self, message_id: str) -> Optional[dict[str, Any]]:
        self.calls.append(("get_message", message_id))
        if message_id in self.fail_get_message_ids:
            raise RuntimeError(f"simulated Gmail failure for {message_id}")
        return self.messages.get(message_id)

    def get_attachment(self, *, message_id: str, attachment_id: str) -> Optional[bytes]:
        self.calls.append(("get_attachment", (message_id, attachment_id)))
        if attachment_id in self.fail_get_attachment_ids:
            raise RuntimeError(f"simulated Gmail attachment failure for {attachment_id}")
        return self.attachments.get((message_id, attachment_id))

    def history_calls(self) -> list[int]:
        return [arg for name, arg in self.calls if name == "list_history"]


class AllowAllPubSubVerifier:
    """Accepts any push. For tests that are not about push authentication."""

    def verify(self, *, authorization_header: str | None, subscription: str | None) -> None:
        return None


class FakePubSubVerifier:
    """Checks a fixed token + subscription, mirroring what the real verifier
    enforces (identity and expected subscription) without any network."""

    def __init__(self, *, expected_token: str = "fake-oidc-token", expected_subscription: str = "sub-1"):
        self.expected_token = expected_token
        self.expected_subscription = expected_subscription

    def verify(self, *, authorization_header: str | None, subscription: str | None) -> None:
        from email_thread_rag.gmail.client import PubSubVerificationError

        if subscription != self.expected_subscription:
            raise PubSubVerificationError("Push is not from the expected subscription.")
        if authorization_header != f"Bearer {self.expected_token}":
            raise PubSubVerificationError("Push identity token failed verification.")


class InMemoryTokenCipher:
    """Reversible, obviously-not-secret cipher for tests that only need the
    encrypt/decrypt contract. Tests that assert real encryption use
    AesGcmTokenCipher directly."""

    key_id = "in-memory-test"

    def encrypt(self, plaintext: str) -> bytes:
        return base64.b64encode(plaintext.encode("utf-8"))

    def decrypt(self, ciphertext: bytes) -> str:
        return base64.b64decode(ciphertext).decode("utf-8")
