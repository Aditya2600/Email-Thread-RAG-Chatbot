"""Narrow Gmail + Pub/Sub interfaces, plus httpx-backed production clients.

The interfaces are what the worker/webhook depend on. ``fakes.py`` implements
them for tests; nothing in the test suite touches the classes below.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional, Protocol, Sequence

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_API_ROOT = "https://gmail.googleapis.com/gmail/v1"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


class GmailApiError(RuntimeError):
    """Any non-success Gmail response. Never carries the access token."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"Gmail API error {status_code}: {message}")


class GmailHistoryExpired(Exception):
    """history.list returned 404: startHistoryId is older than Gmail's window.

    Signals "full sync required". The old cursor is left untouched.
    """


class GmailClient(Protocol):
    def get_profile(self) -> dict[str, Any]:
        """-> {'emailAddress': str, 'historyId': int}"""

    def watch(self, *, topic_name: str) -> dict[str, Any]:
        """-> {'historyId': int, 'expiration': int (epoch ms)}"""

    def stop_watch(self) -> None: ...

    def list_history(
        self, *, start_history_id: int, history_types: Sequence[str] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield raw history.list response pages. Raises GmailHistoryExpired on 404."""

    def list_messages(self, *, query: str | None = None) -> Iterator[dict[str, Any]]:
        """Yield raw messages.list response pages."""

    def get_message(self, message_id: str) -> Optional[dict[str, Any]]:
        """format=full message resource, or None if it no longer exists (404)."""


class PubSubPushVerifier(Protocol):
    def verify(self, *, authorization_header: str | None, subscription: str | None) -> None:
        """Raise PubSubVerificationError unless this is a genuine push from the
        expected subscription, signed for the expected audience."""


class PubSubVerificationError(Exception):
    """Push request is not an authenticated delivery from the expected subscription."""


class HttpxGmailClient:
    """Gmail REST over httpx (already a project dependency; no Google SDK).

    Holds a short-lived access token minted from the mailbox's refresh token.
    Never logs or stringifies that token.
    """

    def __init__(self, access_token: str, *, user_id: str = "me", timeout: float = 30.0):
        self._access_token = access_token
        self.user_id = user_id
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> Any:
        import httpx

        url = f"{GMAIL_API_ROOT}/users/{self.user_id}{path}"
        response = httpx.request(
            method,
            url,
            headers={"Authorization": f"Bearer {self._access_token}"},
            timeout=self.timeout,
            **kwargs,
        )
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            # response.text is Google's error JSON; the request (and its bearer
            # header) is never included.
            raise GmailApiError(response.status_code, response.text[:500])
        return response.json()

    def get_profile(self) -> dict[str, Any]:
        payload = self._request("GET", "/profile")
        return {"emailAddress": payload["emailAddress"], "historyId": int(payload["historyId"])}

    def watch(self, *, topic_name: str) -> dict[str, Any]:
        payload = self._request(
            "POST", "/watch", json={"topicName": topic_name, "labelFilterBehavior": "INCLUDE"}
        )
        return {"historyId": int(payload["historyId"]), "expiration": int(payload["expiration"])}

    def stop_watch(self) -> None:
        self._request("POST", "/stop")

    def list_history(
        self, *, start_history_id: int, history_types: Sequence[str] | None = None
    ) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] = {"startHistoryId": str(start_history_id)}
        if history_types:
            params["historyTypes"] = list(history_types)
        page_token = None
        while True:
            if page_token:
                params["pageToken"] = page_token
            page = self._request("GET", "/history", params=params)
            if page is None:
                # 404 here means only one thing: startHistoryId predates Gmail's
                # history window. Full sync, cursor untouched.
                raise GmailHistoryExpired(
                    f"Gmail history is no longer available from historyId {start_history_id}"
                )
            yield page
            page_token = page.get("nextPageToken")
            if not page_token:
                return

    def list_messages(self, *, query: str | None = None) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] = {"maxResults": 500}
        if query:
            params["q"] = query
        page_token = None
        while True:
            if page_token:
                params["pageToken"] = page_token
            page = self._request("GET", "/messages", params=params) or {}
            yield page
            page_token = page.get("nextPageToken")
            if not page_token:
                return

    def get_message(self, message_id: str) -> Optional[dict[str, Any]]:
        return self._request("GET", f"/messages/{message_id}", params={"format": "full"})


class GooglePubSubPushVerifier:
    """Verifies the OIDC token Pub/Sub attaches to an authenticated push.

    Checks the signature/audience via google-auth and pins the expected
    subscription name, so an attacker cannot drive syncs by POSTing a
    hand-written body at the webhook.
    """

    def __init__(self, *, audience: str, expected_subscription: str, expected_service_account: str | None = None):
        self.audience = audience
        self.expected_subscription = expected_subscription
        self.expected_service_account = expected_service_account

    def verify(self, *, authorization_header: str | None, subscription: str | None) -> None:
        if subscription != self.expected_subscription:
            raise PubSubVerificationError("Push is not from the expected subscription.")
        if not authorization_header or not authorization_header.lower().startswith("bearer "):
            raise PubSubVerificationError("Push is missing its bearer identity token.")
        token = authorization_header.split(" ", 1)[1]
        try:
            from google.auth.transport import requests as google_requests
            from google.oauth2 import id_token
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise PubSubVerificationError(
                "Pub/Sub push verification requires the 'gmail' extra: pip install -e '.[gmail]'"
            ) from exc
        try:
            claims = id_token.verify_oauth2_token(token, google_requests.Request(), self.audience)
        except Exception as exc:  # noqa: BLE001 - never echo the token
            raise PubSubVerificationError("Push identity token failed verification.") from exc
        if self.expected_service_account and claims.get("email") != self.expected_service_account:
            raise PubSubVerificationError("Push identity is not the expected service account.")
        if self.expected_service_account and not claims.get("email_verified"):
            raise PubSubVerificationError("Push identity email is not verified.")
