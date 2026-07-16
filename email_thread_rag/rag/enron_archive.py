from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dateutil import parser as date_parser

from email_thread_rag.app.schemas import AttachmentRecord, EmailRecord
from email_thread_rag.config import Settings
from email_thread_rag.rag.parse_attachments import parse_attachment
from email_thread_rag.rag.utils import coerce_list, normalize_subject


def _coerce_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = date_parser.parse(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_enron_message(
    record: dict[str, Any],
    *,
    message_path: Path,
    attachment_root: Path,
    settings: Settings,
) -> tuple[EmailRecord, list[AttachmentRecord]]:
    message_id = record.get("message_id") or record.get("Message-ID") or record.get("id")
    if not message_id:
        raise ValueError(f"Missing message_id in {message_path}")

    thread_id = record.get("thread_id") or normalize_subject(record.get("subject", "")) or message_id
    email = EmailRecord(
        doc_id=str(record.get("doc_id") or str(message_id).strip("<>")),
        message_id=str(message_id),
        thread_id=str(thread_id),
        date=_coerce_datetime(record.get("date")),
        sender=str(record.get("from") or record.get("sender") or ""),
        to=coerce_list(record.get("to")),
        cc=coerce_list(record.get("cc")),
        subject=str(record.get("subject") or ""),
        body_text=str(record.get("body") or record.get("plain_text_body") or record.get("text") or ""),
        attachment_ids=[],
        in_reply_to=record.get("in_reply_to") or record.get("In-Reply-To"),
        references=coerce_list(record.get("references") or record.get("References")),
        source_path=str(message_path),
        source_type="enron_archive",
    )

    attachments: list[AttachmentRecord] = []
    for index, attachment_meta in enumerate(record.get("attachments", []), start=1):
        filename = attachment_meta.get("filename") or attachment_meta.get("name") or f"attachment-{index}"
        relative_path = attachment_meta.get("relative_path") or attachment_meta.get("path") or filename
        attachment_path = attachment_root / relative_path
        attachment_id = attachment_meta.get("attachment_id") or f"{email.doc_id}-att-{index}"
        attachment = parse_attachment(
            attachment_path,
            attachment_id=str(attachment_id),
            message_id=email.message_id,
            thread_id=email.thread_id,
            settings=settings,
            media_type=attachment_meta.get("media_type"),
        )
        attachments.append(attachment)
        email.attachment_ids.append(attachment.attachment_id)

    return email, attachments

