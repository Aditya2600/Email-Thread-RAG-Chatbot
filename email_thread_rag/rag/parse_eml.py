from __future__ import annotations

from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Optional
from uuid import uuid4

from email.utils import getaddresses, parsedate_to_datetime

from email_thread_rag.app.schemas import AttachmentRecord, EmailRecord
from email_thread_rag.config import Settings
from email_thread_rag.rag.parse_attachments import parse_attachment
from email_thread_rag.rag.utils import coerce_list, normalize_subject


def _parse_datetime(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    dt = parsedate_to_datetime(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_body(message) -> str:
    if message.is_multipart():
        parts: list[str] = []
        for part in message.walk():
            if part.get_content_disposition() == "attachment":
                continue
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                parts.append(payload.decode(charset, errors="ignore"))
        if parts:
            return "\n".join(parts).strip()
    payload = message.get_payload(decode=True)
    if payload is None:
        if isinstance(message.get_payload(), str):
            return message.get_payload().strip()
        return ""
    charset = message.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="ignore").strip()


def _participant_list(message, header: str) -> list[str]:
    values = message.get_all(header, [])
    return [addr for _, addr in getaddresses(values) if addr]


def parse_eml(
    path: Path,
    *,
    settings: Settings,
    attachment_output_dir: Optional[Path] = None,
    default_thread_id: Optional[str] = None,
) -> tuple[EmailRecord, list[AttachmentRecord]]:
    raw_message = path.read_bytes()
    message = BytesParser(policy=policy.default).parsebytes(raw_message)

    message_id = message.get("Message-ID") or f"<generated-{uuid4()}>"
    references = coerce_list(message.get_all("References", []))
    flattened_references: list[str] = []
    for item in references:
        flattened_references.extend(part for part in item.split() if part)
    subject = message.get("Subject", "").strip()
    thread_id = default_thread_id or normalize_subject(subject) or message_id.strip("<>")
    body_text = _extract_body(message)

    email_record = EmailRecord(
        doc_id=message_id.strip("<>"),
        message_id=message_id,
        thread_id=thread_id,
        date=_parse_datetime(message.get("Date")),
        sender=(message.get("From") or "").strip(),
        to=_participant_list(message, "To"),
        cc=_participant_list(message, "Cc"),
        subject=subject,
        body_text=body_text,
        attachment_ids=[],
        in_reply_to=message.get("In-Reply-To"),
        references=flattened_references,
        source_path=str(path),
        source_type="eml",
    )

    attachments: list[AttachmentRecord] = []
    for index, part in enumerate(message.walk(), start=1):
        if part.get_content_disposition() != "attachment":
            continue
        filename = part.get_filename() or f"attachment-{index}"
        payload = part.get_payload(decode=True) or b""
        if attachment_output_dir is None:
            attachment_output_dir = path.parent / f"{path.stem}_attachments"
        attachment_output_dir.mkdir(parents=True, exist_ok=True)
        attachment_path = attachment_output_dir / filename
        attachment_path.write_bytes(payload)
        attachment_id = f"{email_record.doc_id}-att-{index}"
        attachment_record = parse_attachment(
            attachment_path,
            attachment_id=attachment_id,
            message_id=message_id,
            thread_id=thread_id,
            settings=settings,
            media_type=part.get_content_type(),
        )
        attachments.append(attachment_record)
        email_record.attachment_ids.append(attachment_id)

    return email_record, attachments

