from __future__ import annotations

from pathlib import Path

from email_thread_rag.app.schemas import AttachmentRecord, ChunkRecord, EmailRecord
from email_thread_rag.config import Settings
from email_thread_rag.rag.chunking import chunk_corpus
from email_thread_rag.rag.enron_archive import parse_enron_message
from email_thread_rag.rag.parse_eml import parse_eml
from email_thread_rag.rag.threading import reconstruct_threads
from email_thread_rag.rag.utils import read_json, write_json, write_jsonl


def ingest_corpus(settings: Settings) -> tuple[list[EmailRecord], list[AttachmentRecord], list[ChunkRecord], dict]:
    manifest = read_json(settings.resolved_manifest_path if settings.resolved_manifest_path.exists() else settings.dataset_manifest_path)
    emails: list[EmailRecord] = []
    attachments: list[AttachmentRecord] = []

    for thread in manifest.get("threads", []):
        for message_meta in thread.get("messages", []):
            relative_path = message_meta.get("local_path") or message_meta.get("relative_path")
            if not relative_path:
                continue
            message_path = settings.raw_data_dir / relative_path
            if not message_path.exists():
                raise FileNotFoundError(f"Message file not found: {message_path}")

            if message_path.suffix.lower() == ".eml":
                email_record, attachment_records = parse_eml(
                    message_path,
                    settings=settings,
                    default_thread_id=thread.get("thread_key") or thread.get("thread_id"),
                )
            else:
                payload = read_json(message_path)
                email_record, attachment_records = parse_enron_message(
                    payload,
                    message_path=message_path,
                    attachment_root=settings.raw_data_dir,
                    settings=settings,
                )
            emails.append(email_record)
            attachments.extend(attachment_records)

    threaded_emails = reconstruct_threads(emails)
    thread_lookup = {email.message_id: email.thread_id for email in threaded_emails}
    normalized_attachments = [
        attachment.model_copy(update={"thread_id": thread_lookup.get(attachment.message_id, attachment.thread_id)})
        for attachment in attachments
    ]
    chunks = chunk_corpus(threaded_emails, normalized_attachments)

    stats = {
        "thread_count": len({email.thread_id for email in threaded_emails}),
        "message_count": len(threaded_emails),
        "attachment_count": len(normalized_attachments),
        "chunk_count": len(chunks),
        "indexed_text_size": sum(len(chunk.text) for chunk in chunks),
        "ocr_triggered_pages": sum(
            1
            for attachment in normalized_attachments
            for page in attachment.pages
            if page.ocr_used
        ),
    }

    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(settings.chunk_store_path, [chunk.model_dump(mode="json") for chunk in chunks])
    write_json(settings.stats_path, stats)
    write_json(
        settings.processed_data_dir / "emails.json",
        [email.model_dump(mode="json") for email in threaded_emails],
    )
    write_json(
        settings.processed_data_dir / "attachments.json",
        [attachment.model_dump(mode="json") for attachment in normalized_attachments],
    )
    return threaded_emails, normalized_attachments, chunks, stats

