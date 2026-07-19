"""Gmail ``messages.get(format=full)`` -> canonical ``EmailRecord``.

This is the only Gmail-shaped code in the ingestion path. Once a message is an
``EmailRecord``, Stage 1's segmenter/chunker and Stage 2.5's ParadeDB
persistence handle it exactly as they handle a parsed .eml -- no Gmail-specific
branches downstream.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from email.utils import getaddresses, parsedate_to_datetime
from typing import Any

from email_thread_rag.app.schemas import EmailRecord
from email_thread_rag.rag.attachments.models import AttachmentMeta

PDF_MEDIA_TYPE = "application/pdf"


def gmail_pdf_attachments(message: dict[str, Any]) -> list[AttachmentMeta]:
    """PDF attachment parts of a Gmail message, as metadata only (no bytes).

    Stage 8 is PDF-only: parts that are not application/pdf (and don't end in
    .pdf) are ignored here and never enter the attachment pipeline. The bytes
    are fetched later, in the extraction worker, via messages.attachments.get.
    """
    metas: list[AttachmentMeta] = []
    stack = [message.get("payload", {})]
    while stack:
        part = stack.pop(0)
        stack.extend(part.get("parts", []))
        filename = part.get("filename") or ""
        body = part.get("body", {})
        attachment_id = body.get("attachmentId")
        if not attachment_id:
            continue
        mime = part.get("mimeType") or ""
        if mime != PDF_MEDIA_TYPE and not filename.lower().endswith(".pdf"):
            continue
        metas.append(
            AttachmentMeta(
                gmail_attachment_id=attachment_id,
                filename=filename or f"{attachment_id}.pdf",
                media_type=PDF_MEDIA_TYPE,
                byte_size=body.get("size"),
            )
        )
    return metas


def _decode_body_data(data: str | None) -> str:
    if not data:
        return ""
    # Gmail base64url-encodes body data and strips padding.
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding).decode("utf-8", errors="ignore")


def _headers(payload: dict[str, Any]) -> dict[str, str]:
    """Header names are case-insensitive; Gmail's casing is not guaranteed."""
    return {header.get("name", "").lower(): header.get("value", "") for header in payload.get("headers", [])}


def _collect_parts(payload: dict[str, Any], mime_type: str) -> list[str]:
    """Depth-first collect of decoded bodies matching ``mime_type``, skipping attachments."""
    found: list[str] = []
    stack = [payload]
    while stack:
        part = stack.pop(0)
        if part.get("filename"):  # an attachment, not inline body text
            continue
        if part.get("mimeType") == mime_type:
            text = _decode_body_data(part.get("body", {}).get("data"))
            if text.strip():
                found.append(text)
        stack.extend(part.get("parts", []))
    return found


def _html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "lxml").get_text("\n").strip()


def extract_body_text(payload: dict[str, Any]) -> str:
    plain = _collect_parts(payload, "text/plain")
    if plain:
        return "\n".join(plain).strip()
    html_parts = _collect_parts(payload, "text/html")
    if html_parts:
        return _html_to_text("\n".join(html_parts))
    return ""


def _parse_date(headers: dict[str, str], internal_date_ms: str | None) -> datetime:
    raw = headers.get("date")
    if raw:
        try:
            parsed = parsedate_to_datetime(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except (TypeError, ValueError):
            pass  # malformed Date header: fall through to Gmail's internalDate
    if internal_date_ms:
        return datetime.fromtimestamp(int(internal_date_ms) / 1000, tz=timezone.utc)
    return datetime.now(timezone.utc)


def _address_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [address for _name, address in getaddresses([value]) if address]


def gmail_message_to_email_record(message: dict[str, Any], *, email_address: str) -> EmailRecord:
    """Convert one Gmail message resource into the canonical record.

    ``message_id`` is Gmail's own message ID, not the RFC 5322 Message-ID
    header: a ``messageDeleted`` history record names only the Gmail ID, so it
    has to be the key the indexed chunks are addressable by.

    ``thread_id`` is Gmail's ``threadId`` -- authoritative, so no
    subject-normalization threading heuristic runs on synced mail.
    """
    payload = message.get("payload", {})
    headers = _headers(payload)
    gmail_id = message["id"]

    references = [ref for ref in (headers.get("references") or "").split() if ref]
    return EmailRecord(
        doc_id=gmail_id,
        message_id=gmail_id,
        thread_id=message.get("threadId") or gmail_id,
        date=_parse_date(headers, message.get("internalDate")),
        sender=headers.get("from", ""),
        to=_address_list(headers.get("to")),
        cc=_address_list(headers.get("cc")),
        subject=headers.get("subject", ""),
        body_text=extract_body_text(payload),
        attachment_ids=[meta.attachment_id for meta in gmail_pdf_attachments(message)],
        in_reply_to=headers.get("in-reply-to"),
        references=references,
        source_path=f"gmail://{email_address}/{gmail_id}",
        source_type="gmail",
    )
