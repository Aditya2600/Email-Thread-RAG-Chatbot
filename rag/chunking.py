from __future__ import annotations

from uuid import uuid4

from datetime import datetime

from email_thread_rag.app.schemas import AttachmentRecord, ChunkRecord, EmailRecord
from email_thread_rag.rag.utils import count_tokens, sliding_text_chunks


def chunk_email(email: EmailRecord) -> list[ChunkRecord]:
    metadata = {
        "to": email.to,
        "cc": email.cc,
        "attachment_ids": email.attachment_ids,
        "in_reply_to": email.in_reply_to,
        "references": email.references,
    }
    token_count = count_tokens(email.body_text)
    if token_count <= 900:
        return [
            ChunkRecord(
                chunk_id=f"{email.doc_id}-email-0",
                doc_id=email.doc_id,
                thread_id=email.thread_id,
                message_id=email.message_id,
                kind="email",
                sender=email.sender,
                date=email.date,
                subject=email.subject,
                text=email.body_text,
                ocr_used=False,
                token_count=token_count,
                source_path=email.source_path,
                source_type=email.source_type,
                metadata=metadata,
            )
        ]

    paragraphs = [paragraph.strip() for paragraph in email.body_text.split("\n\n") if paragraph.strip()]
    chunks: list[ChunkRecord] = []
    current_parts: list[str] = []
    current_tokens = 0
    chunk_index = 0
    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph)
        if current_parts and current_tokens + paragraph_tokens > 900:
            text = "\n\n".join(current_parts)
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{email.doc_id}-email-{chunk_index}",
                    doc_id=email.doc_id,
                    thread_id=email.thread_id,
                    message_id=email.message_id,
                    kind="email",
                    sender=email.sender,
                    date=email.date,
                    subject=email.subject,
                    text=text,
                    ocr_used=False,
                    token_count=count_tokens(text),
                    source_path=email.source_path,
                    source_type=email.source_type,
                    metadata=metadata,
                )
            )
            chunk_index += 1
            current_parts = [paragraph]
            current_tokens = paragraph_tokens
        else:
            current_parts.append(paragraph)
            current_tokens += paragraph_tokens
    if current_parts:
        text = "\n\n".join(current_parts)
        chunks.append(
            ChunkRecord(
                chunk_id=f"{email.doc_id}-email-{chunk_index}",
                doc_id=email.doc_id,
                thread_id=email.thread_id,
                message_id=email.message_id,
                kind="email",
                sender=email.sender,
                date=email.date,
                subject=email.subject,
                text=text,
                ocr_used=False,
                    token_count=count_tokens(text),
                    source_path=email.source_path,
                    source_type=email.source_type,
                    metadata=metadata,
                )
            )
    return chunks


def chunk_attachment(
    attachment: AttachmentRecord,
    *,
    message_date: datetime,
    sender: str | None,
    subject: str | None,
    source_type: str,
    chunk_size: int = 250,
    overlap: int = 50,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for page in attachment.pages:
        page_chunks = list(sliding_text_chunks(page.text, chunk_size=chunk_size, overlap=overlap)) or [page.text]
        for index, text in enumerate(page_chunks):
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{attachment.attachment_id}-page-{page.page_no}-chunk-{index}",
                    doc_id=attachment.attachment_id,
                    thread_id=attachment.thread_id,
                    message_id=attachment.message_id,
                    kind="attachment",
                    attachment_name=attachment.filename,
                    page_no=page.page_no,
                    date=message_date,
                    sender=sender,
                    subject=subject,
                    text=text.strip(),
                    ocr_used=page.ocr_used,
                    token_count=count_tokens(text),
                    source_path=attachment.source_path,
                    source_type=source_type,
                    metadata={"media_type": attachment.media_type},
                )
            )
    return chunks


def chunk_corpus(emails: list[EmailRecord], attachments: list[AttachmentRecord]) -> list[ChunkRecord]:
    attachment_by_message = {}
    for attachment in attachments:
        attachment_by_message.setdefault(attachment.message_id, []).append(attachment)

    chunks: list[ChunkRecord] = []
    for email in emails:
        chunks.extend(chunk_email(email))
        for attachment in attachment_by_message.get(email.message_id, []):
            chunks.extend(
                chunk_attachment(
                    attachment,
                    message_date=email.date,
                    sender=email.sender,
                    subject=email.subject,
                    source_type=email.source_type,
                )
            )
    return [
        chunk.model_copy(update={"chunk_id": chunk.chunk_id or f"chunk-{uuid4()}"})
        for chunk in chunks
        if chunk.text.strip()
    ]
