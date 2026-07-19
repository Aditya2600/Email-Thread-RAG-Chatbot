"""Postgres-backed attachment metadata + extraction queue.

Mirrors graph/repository.py: no ORM, plain parameterized statements, and the
claim path uses ``FOR UPDATE SKIP LOCKED`` so concurrent workers never grab the
same job. No transaction is held across the Gmail attachment fetch -- claim
commits, the worker fetches/extracts with no txn open, then commit writes.
"""

from __future__ import annotations

from typing import Optional

from email_thread_rag.rag.attachments.models import (
    AttachmentJob,
    AttachmentMeta,
    StoredAttachment,
    extraction_input_hash,
)

_JOB_COLUMNS = (
    "id, attachment_db_id, tenant_id, mailbox_id, attachment_id, extraction_input_hash, "
    "status, attempts, leased_until, lease_owner, last_error, error_rule, completed_at"
)
_ATT_COLUMNS = (
    "id, tenant_id, mailbox_id, message_id, thread_id, gmail_attachment_id, attachment_id, "
    "filename, media_type, byte_size, content_hash, extraction_status, extraction_method, "
    "extraction_error"
)


def _row_to_job(row) -> Optional[AttachmentJob]:
    return AttachmentJob(**row) if row is not None else None


def _row_to_attachment(row) -> Optional[StoredAttachment]:
    return StoredAttachment(**row) if row is not None else None


class PostgresAttachmentJobStore:
    def __init__(self, conn):
        self.conn = conn

    def enqueue(
        self, meta: AttachmentMeta, *, message_db_id=None, message_id, thread_id, tenant_id, mailbox_id
    ) -> Optional[AttachmentJob]:
        if message_db_id is None:
            found = self.conn.execute(
                "SELECT id FROM email_messages WHERE tenant_id = %s AND mailbox_id = %s AND message_id = %s",
                (tenant_id, mailbox_id, message_id),
            ).fetchone()
            message_db_id = found["id"] if found else None

        att = self.conn.execute(
            """
            INSERT INTO email_attachments (
                tenant_id, mailbox_id, message_db_id, message_id, thread_id,
                gmail_attachment_id, attachment_id, filename, media_type, byte_size
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (tenant_id, mailbox_id, message_id, gmail_attachment_id) DO UPDATE SET
                filename = EXCLUDED.filename,
                media_type = EXCLUDED.media_type,
                byte_size = EXCLUDED.byte_size,
                message_db_id = EXCLUDED.message_db_id,
                updated_at = now()
            RETURNING id
            """,
            (
                tenant_id, mailbox_id, message_db_id, message_id, thread_id,
                meta.gmail_attachment_id, meta.attachment_id, meta.filename,
                meta.media_type, meta.byte_size,
            ),
        ).fetchone()
        attachment_db_id = att["id"]

        digest = extraction_input_hash(meta)
        row = self.conn.execute(
            f"""
            INSERT INTO attachment_extraction_jobs (
                attachment_db_id, tenant_id, mailbox_id, attachment_id, extraction_input_hash
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (tenant_id, mailbox_id, attachment_id, extraction_input_hash) DO NOTHING
            RETURNING {_JOB_COLUMNS}
            """,
            (attachment_db_id, tenant_id, mailbox_id, meta.attachment_id, digest),
        ).fetchone()
        if row is not None:
            # A genuinely new job (changed or first-seen attachment): move the
            # attachment back to pending so a prior 'done' doesn't hide new work.
            self.conn.execute(
                "UPDATE email_attachments SET extraction_status = 'pending', extraction_error = NULL, "
                "updated_at = now() WHERE id = %s",
                (attachment_db_id,),
            )
        return _row_to_job(row)

    def claim_job(self, *, owner, lease_seconds=600) -> Optional[AttachmentJob]:
        with self.conn.transaction():
            candidate = self.conn.execute(
                "SELECT id FROM attachment_extraction_jobs "
                "WHERE status = 'pending' OR (status = 'running' AND leased_until <= now()) "
                "ORDER BY id ASC FOR UPDATE SKIP LOCKED LIMIT 1"
            ).fetchone()
            if candidate is None:
                return None
            row = self.conn.execute(
                f"""
                UPDATE attachment_extraction_jobs SET
                    status = 'running', attempts = attempts + 1, lease_owner = %s,
                    leased_until = now() + make_interval(secs => %s), updated_at = now()
                WHERE id = %s RETURNING {_JOB_COLUMNS}
                """,
                (owner, lease_seconds, candidate["id"]),
            ).fetchone()
        return _row_to_job(row)

    def get_job(self, job_id) -> Optional[AttachmentJob]:
        return _row_to_job(
            self.conn.execute(
                f"SELECT {_JOB_COLUMNS} FROM attachment_extraction_jobs WHERE id = %s", (job_id,)
            ).fetchone()
        )

    def load_attachment(self, attachment_db_id) -> Optional[StoredAttachment]:
        return _row_to_attachment(
            self.conn.execute(
                f"SELECT {_ATT_COLUMNS} FROM email_attachments WHERE id = %s", (attachment_db_id,)
            ).fetchone()
        )

    def commit_extraction(self, job, *, content_hash, method, status) -> bool:
        with self.conn.transaction():
            current = self.conn.execute(
                "SELECT status FROM attachment_extraction_jobs WHERE id = %s FOR UPDATE", (job.id,)
            ).fetchone()
            if current is None or current["status"] != "running":
                return False  # superseded / already retired
            self.conn.execute(
                "UPDATE email_attachments SET content_hash = %s, extraction_method = %s, "
                "extraction_status = %s, extraction_error = NULL, updated_at = now() WHERE id = %s",
                (content_hash, method, status, job.attachment_db_id),
            )
            self.conn.execute(
                "UPDATE attachment_extraction_jobs SET status = 'done', completed_at = now(), "
                "leased_until = NULL, lease_owner = NULL, last_error = NULL, updated_at = now() "
                "WHERE id = %s",
                (job.id,),
            )
        return True

    def fail_job(self, job_id, error, *, error_rule=None, max_attempts=3) -> None:
        with self.conn.transaction():
            self.conn.execute(
                """
                UPDATE attachment_extraction_jobs SET
                    status = CASE WHEN attempts >= %s THEN 'failed' ELSE 'pending' END,
                    last_error = %s, error_rule = %s, leased_until = NULL, lease_owner = NULL,
                    updated_at = now()
                WHERE id = %s
                """,
                (max_attempts, error, error_rule, job_id),
            )

    def mark_attachment(self, attachment_db_id, *, status, error=None) -> None:
        self.conn.execute(
            "UPDATE email_attachments SET extraction_status = %s, extraction_error = %s, "
            "updated_at = now() WHERE id = %s",
            (status, error, attachment_db_id),
        )
