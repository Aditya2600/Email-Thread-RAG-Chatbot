"""The attachment extraction queue: a Protocol plus an in-memory implementation.

Same shape and semantics as the Stage 4/5 job stores: one Protocol satisfied by
both a dict-backed fake (fast suite) and a Postgres store (see repository.py),
exercised by a shared contract test so the fake cannot drift from the real one.

The database table is the queue -- no broker.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

from email_thread_rag.rag.attachments.models import (
    AttachmentJob,
    AttachmentMeta,
    StoredAttachment,
    extraction_input_hash,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AttachmentJobStore(Protocol):
    def enqueue(
        self,
        meta: AttachmentMeta,
        *,
        message_db_id: Optional[int],
        message_id: str,
        thread_id: Optional[str],
        tenant_id: str,
        mailbox_id: str,
    ) -> Optional[AttachmentJob]:
        """Persist attachment metadata and queue extraction. Idempotent by input
        hash: re-syncing an unchanged attachment creates no duplicate job."""

    def claim_job(self, *, owner: str, lease_seconds: int = 600) -> Optional[AttachmentJob]: ...

    def get_job(self, job_id: int) -> Optional[AttachmentJob]: ...

    def load_attachment(self, attachment_db_id: int) -> Optional[StoredAttachment]: ...

    def commit_extraction(
        self, job: AttachmentJob, *, content_hash: str, method: Optional[str], status: str
    ) -> bool:
        """Record the extraction outcome on the attachment and retire the job.

        Returns False if the job is stale (the attachment's queued input hash
        changed under it), in which case nothing is written."""

    def fail_job(
        self, job_id: int, error: str, *, error_rule: Optional[str] = None, max_attempts: int = 3
    ) -> None: ...

    def mark_attachment(
        self, attachment_db_id: int, *, status: str, error: Optional[str] = None
    ) -> None:
        """Set a terminal status on an attachment without a job commit (used for
        'unsupported' / hard 'failed' before any bytes are fetched)."""


class InMemoryAttachmentJobStore:
    """Dict-backed store with Postgres-equivalent semantics. Test/demo only."""

    def __init__(self):
        self.attachments: dict[int, StoredAttachment] = {}
        self._jobs: dict[int, AttachmentJob] = {}
        self._next_attachment_id = 1
        self._next_job_id = 1

    def _find_attachment(self, tenant_id, mailbox_id, message_id, gmail_attachment_id):
        for att in self.attachments.values():
            if (
                att.tenant_id == tenant_id
                and att.mailbox_id == mailbox_id
                and att.message_id == message_id
                and att.gmail_attachment_id == gmail_attachment_id
            ):
                return att
        return None

    def enqueue(
        self, meta, *, message_db_id, message_id, thread_id, tenant_id, mailbox_id
    ) -> Optional[AttachmentJob]:
        att = self._find_attachment(tenant_id, mailbox_id, message_id, meta.gmail_attachment_id)
        if att is None:
            att = StoredAttachment(
                id=self._next_attachment_id,
                tenant_id=tenant_id,
                mailbox_id=mailbox_id,
                message_id=message_id,
                thread_id=thread_id,
                gmail_attachment_id=meta.gmail_attachment_id,
                attachment_id=meta.attachment_id,
                filename=meta.filename,
                media_type=meta.media_type,
                byte_size=meta.byte_size,
            )
            self.attachments[att.id] = att
            self._next_attachment_id += 1
        else:
            att.filename = meta.filename
            att.media_type = meta.media_type
            att.byte_size = meta.byte_size

        digest = extraction_input_hash(meta)
        for job in self._jobs.values():
            if (
                job.tenant_id == tenant_id
                and job.mailbox_id == mailbox_id
                and job.attachment_id == meta.attachment_id
                and job.extraction_input_hash == digest
            ):
                return None  # mirrors ON CONFLICT DO NOTHING
        job = AttachmentJob(
            id=self._next_job_id,
            attachment_db_id=att.id,
            tenant_id=tenant_id,
            mailbox_id=mailbox_id,
            attachment_id=meta.attachment_id,
            extraction_input_hash=digest,
        )
        self._jobs[job.id] = job
        self._next_job_id += 1
        return job

    def claim_job(self, *, owner, lease_seconds=600) -> Optional[AttachmentJob]:
        now = utcnow()
        for job in sorted(self._jobs.values(), key=lambda j: j.id):
            expired = job.status == "running" and job.leased_until is not None and job.leased_until <= now
            if job.status == "pending" or expired:
                job.status = "running"
                job.attempts += 1
                job.lease_owner = owner
                job.leased_until = now + timedelta(seconds=lease_seconds)
                return job
        return None

    def get_job(self, job_id) -> Optional[AttachmentJob]:
        return self._jobs.get(job_id)

    def load_attachment(self, attachment_db_id) -> Optional[StoredAttachment]:
        return self.attachments.get(attachment_db_id)

    def commit_extraction(self, job, *, content_hash, method, status) -> bool:
        current = self._jobs.get(job.id)
        att = self.attachments.get(job.attachment_db_id)
        if current is None or att is None or current.extraction_input_hash != job.extraction_input_hash:
            self._retire(job.id)
            return False
        att.content_hash = content_hash
        att.extraction_method = method
        att.extraction_status = status
        att.extraction_error = None
        self._retire(job.id)
        return True

    def _retire(self, job_id) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = "done"
        job.completed_at = utcnow()
        job.leased_until = None
        job.lease_owner = None
        job.last_error = None

    def fail_job(self, job_id, error, *, error_rule=None, max_attempts=3) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.last_error = error
        job.error_rule = error_rule
        job.leased_until = None
        job.lease_owner = None
        job.status = "failed" if job.attempts >= max_attempts else "pending"

    def mark_attachment(self, attachment_db_id, *, status, error=None) -> None:
        att = self.attachments.get(attachment_db_id)
        if att is not None:
            att.extraction_status = status
            att.extraction_error = error
