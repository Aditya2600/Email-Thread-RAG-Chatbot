from __future__ import annotations

from uuid import uuid4

from datetime import datetime

from email_thread_rag.app.schemas import AttachmentRecord, ChunkRecord, EmailRecord
from email_thread_rag.rag.email_segmentation import (
    build_embed_text,
    segment_email_body,
)
from email_thread_rag.rag.utils import count_tokens, sliding_text_chunks


# Email-aware chunk sizing. Emails are short and semantically dense, so we target
# a small window and split on paragraph boundaries rather than a fixed token grid.
MAX_CHUNK_TOKENS = 450
OVERLAP_TOKENS = 50


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of blank-line-separated paragraphs within ``text``."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for block in text.split("\n\n"):
        start = text.find(block, pos)
        if start < 0:  # defensive; shouldn't happen for a plain split
            start = pos
        end = start + len(block)
        pos = end
        if block.strip():
            spans.append((start, end))
    return spans


class EmailAwareChunker:
    """Split authored email content into citation-safe chunks.

    Short authored emails stay a single chunk. Longer ones split on paragraph
    boundaries with a small token overlap. Each chunk keeps exact source offsets
    into the authored body so citations map back to real text.
    """

    def __init__(self, *, max_tokens: int = MAX_CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS):
        self.max_tokens = max_tokens
        self.overlap = overlap

    def split(self, authored_text: str) -> list[tuple[str, int, int]]:
        """Return list of (chunk_text, source_start, source_end)."""
        authored_text = authored_text or ""
        if not authored_text.strip():
            return []
        if count_tokens(authored_text) <= self.max_tokens:
            return [(authored_text, 0, len(authored_text))]

        spans = _paragraph_spans(authored_text)
        chunks: list[tuple[str, int, int]] = []
        current: list[tuple[int, int]] = []
        current_tokens = 0

        def flush() -> None:
            if not current:
                return
            start = current[0][0]
            end = current[-1][1]
            chunks.append((authored_text[start:end], start, end))

        for span in spans:
            para = authored_text[span[0] : span[1]]
            para_tokens = count_tokens(para)
            if para_tokens > self.max_tokens:
                # Single oversized paragraph: fall back to sliding token windows.
                # ponytail: offsets pinned to the paragraph span (word-window
                # offsets aren't worth reconstructing); citations still land in
                # the right paragraph. Upgrade to exact spans if a validator needs it.
                flush()
                current, current_tokens = [], 0
                for window in sliding_text_chunks(para, chunk_size=self.max_tokens, overlap=self.overlap):
                    chunks.append((window, span[0], span[1]))
                continue
            if current and current_tokens + para_tokens > self.max_tokens:
                flush()
                # Seed the next chunk with trailing paragraphs for ~overlap tokens.
                overlap_parts: list[tuple[int, int]] = []
                overlap_tokens = 0
                for prev in reversed(current):
                    t = count_tokens(authored_text[prev[0] : prev[1]])
                    if overlap_tokens + t > self.overlap:
                        break
                    overlap_parts.insert(0, prev)
                    overlap_tokens += t
                current = overlap_parts + [span]
                current_tokens = overlap_tokens + para_tokens
            else:
                current.append(span)
                current_tokens += para_tokens
        flush()
        return chunks


def chunk_email(email: EmailRecord, chunker: EmailAwareChunker | None = None) -> list[ChunkRecord]:
    """Segment the email, then chunk only the sender's authored content.

    ``text`` holds exact authored evidence (no injected headers); ``embed_text``
    carries a compact header block plus that same text for indexing.
    """
    chunker = chunker or EmailAwareChunker()
    segments = segment_email_body(email.body_text)

    # Backfill audit segments on the record (additive; safe if already set).
    email.authored_text = segments.authored_text
    email.quoted_text = segments.quoted_text or None
    email.signature_text = segments.signature_text or None
    email.disclaimer_text = segments.disclaimer_text or None

    metadata = {
        "to": email.to,
        "cc": email.cc,
        "attachment_ids": email.attachment_ids,
        "in_reply_to": email.in_reply_to,
        "references": email.references,
    }

    chunks: list[ChunkRecord] = []
    for index, (text, start, end) in enumerate(chunker.split(segments.authored_text)):
        embed_text = build_embed_text(
            text,
            sender=email.sender,
            to=email.to,
            cc=email.cc,
            date=email.date,
            subject=email.subject,
            thread_id=email.thread_id,
            in_reply_to=email.in_reply_to,
        )
        chunks.append(
            ChunkRecord(
                chunk_id=f"{email.doc_id}-email-{index}",
                doc_id=email.doc_id,
                thread_id=email.thread_id,
                message_id=email.message_id,
                kind="email",
                sender=email.sender,
                date=email.date,
                subject=email.subject,
                text=text,
                embed_text=embed_text,
                source_start=start,
                source_end=end,
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
            clean = text.strip()
            # embed_text carries parent-email provenance + filename/page so the
            # attachment chunk is findable by email context; text stays pure.
            embed_text = build_embed_text(
                clean,
                sender=sender,
                date=message_date,
                subject=subject,
            )
            header_extra = f"Attachment: {attachment.filename} (page {page.page_no})"
            embed_text = f"{header_extra}\n{embed_text}" if embed_text != clean else f"{header_extra}\n\n{clean}"
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
                    text=clean,
                    embed_text=embed_text,
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
