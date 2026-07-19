"""Gmail message resource -> canonical EmailRecord."""

from __future__ import annotations

import base64
from datetime import timezone

from email_thread_rag.gmail.fakes import build_gmail_message
from email_thread_rag.gmail.message import gmail_message_to_email_record
from email_thread_rag.rag.chunking import chunk_email


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def test_plain_message_maps_to_canonical_fields():
    email = gmail_message_to_email_record(
        build_gmail_message(
            gmail_id="m-1",
            thread_id="t-1",
            history_id=200,
            sender="Alice <alice@corp.com>",
            to="bob@corp.com",
            subject="Budget Review",
            body="Approved amount is $1200.",
        ),
        email_address="user@example.com",
    )

    # message_id is Gmail's ID, not the RFC header: messageDeleted history
    # records name only this, so it has to be the indexed key.
    assert email.message_id == "m-1"
    assert email.doc_id == "m-1"
    assert email.thread_id == "t-1"
    assert email.to == ["bob@corp.com"]
    assert email.subject == "Budget Review"
    assert email.body_text == "Approved amount is $1200."
    assert email.source_type == "gmail"
    assert email.source_path == "gmail://user@example.com/m-1"


def test_headers_are_matched_case_insensitively():
    email = gmail_message_to_email_record(
        {
            "id": "m-1",
            "threadId": "t-1",
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "FROM", "value": "alice@corp.com"},
                    {"name": "subject", "value": "Lowercase header"},
                    {"name": "To", "value": "bob@corp.com, carol@corp.com"},
                ],
                "body": {"data": b64("Body.")},
            },
        },
        email_address="user@example.com",
    )
    assert email.sender == "alice@corp.com"
    assert email.subject == "Lowercase header"
    assert email.to == ["bob@corp.com", "carol@corp.com"]


def test_multipart_alternative_prefers_plain_text_and_skips_attachments():
    email = gmail_message_to_email_record(
        {
            "id": "m-1",
            "threadId": "t-1",
            "payload": {
                "mimeType": "multipart/mixed",
                "headers": [{"name": "Subject", "value": "Mixed"}],
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": b64("The plain body.")}},
                    {"mimeType": "text/html", "body": {"data": b64("<p>The HTML body.</p>")}},
                    {
                        "mimeType": "application/pdf",
                        "filename": "invoice.pdf",
                        "body": {"attachmentId": "att-1"},
                    },
                ],
            },
        },
        email_address="user@example.com",
    )
    assert email.body_text == "The plain body."


def test_html_only_message_falls_back_to_stripped_text():
    email = gmail_message_to_email_record(
        {
            "id": "m-1",
            "threadId": "t-1",
            "payload": {
                "mimeType": "text/html",
                "headers": [{"name": "Subject", "value": "HTML only"}],
                "body": {"data": b64("<html><body><p>Approved: $900.</p></body></html>")},
            },
        },
        email_address="user@example.com",
    )
    assert "Approved: $900." in email.body_text
    assert "<p>" not in email.body_text


def test_base64url_body_without_padding_decodes():
    # Gmail strips '=' padding; the decoder has to put it back.
    payload = {"mimeType": "text/plain", "headers": [], "body": {"data": b64("abcde")}}
    email = gmail_message_to_email_record(
        {"id": "m-1", "threadId": "t-1", "payload": payload}, email_address="user@example.com"
    )
    assert email.body_text == "abcde"


def test_date_header_is_parsed_as_utc():
    email = gmail_message_to_email_record(
        build_gmail_message(
            gmail_id="m-1",
            thread_id="t-1",
            history_id=1,
            sender="a@corp.com",
            to="b@corp.com",
            subject="S",
            body="B",
            date="Mon, 5 Jan 2026 09:00:00 +0200",
        ),
        email_address="user@example.com",
    )
    assert email.date.tzinfo is not None
    assert email.date.astimezone(timezone.utc).hour == 7


def test_missing_date_header_falls_back_to_internal_date():
    email = gmail_message_to_email_record(
        {
            "id": "m-1",
            "threadId": "t-1",
            "internalDate": "1767610800000",
            "payload": {"mimeType": "text/plain", "headers": [], "body": {"data": b64("B")}},
        },
        email_address="user@example.com",
    )
    assert email.date.year == 2026


def test_malformed_date_header_falls_back_to_internal_date():
    email = gmail_message_to_email_record(
        {
            "id": "m-1",
            "threadId": "t-1",
            "internalDate": "1767610800000",
            "payload": {
                "mimeType": "text/plain",
                "headers": [{"name": "Date", "value": "not a date"}],
                "body": {"data": b64("B")},
            },
        },
        email_address="user@example.com",
    )
    assert email.date.year == 2026


def test_empty_body_produces_no_chunks():
    email = gmail_message_to_email_record(
        {"id": "m-1", "threadId": "t-1", "payload": {"mimeType": "text/plain", "headers": [], "body": {}}},
        email_address="user@example.com",
    )
    assert email.body_text == ""
    assert chunk_email(email) == []


def test_converted_message_chunks_through_the_stage_1_segmenter():
    """Quoted reply text must not become the authored evidence -- the same rule
    the .eml path already enforces, with no Gmail-specific handling."""
    email = gmail_message_to_email_record(
        build_gmail_message(
            gmail_id="m-2",
            thread_id="t-1",
            history_id=300,
            sender="bob@corp.com",
            to="alice@corp.com",
            subject="Re: Budget Review",
            body=(
                "Confirming the final number is $1200.\n\n"
                "On Mon, Jan 5, 2026 at 9:00 AM Alice <alice@corp.com> wrote:\n"
                "> The draft budget is $1000.\n"
            ),
        ),
        email_address="user@example.com",
    )
    chunks = chunk_email(email)

    assert len(chunks) == 1
    assert "Confirming the final number is $1200." in chunks[0].text
    assert "$1000" not in chunks[0].text
    assert email.quoted_text and "$1000" in email.quoted_text
