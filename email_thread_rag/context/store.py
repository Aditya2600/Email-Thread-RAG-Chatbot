"""The context job queue, as an interface plus an in-memory implementation.

Same shape as Stage 3's SyncStore: a Protocol both a dict-backed fake and a
Postgres-backed store satisfy, exercised by one shared contract test so the
fake used by the fast suite cannot drift from the real one.

The database table is the queue. No Redis, no Celery, no broker.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol, Sequence

from email_thread_rag.context.fingerprint import fingerprint_of
from email_thread_rag.context.models import ChunkContextState, ContextJob


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ContextJobStore(Protocol):
    def enqueue(
        self, state: ChunkContextState, *, prompt_version: str, model_id: str
    ) -> Optional[ContextJob]:
        """Queue contextualization for a chunk. Returns None if already queued.

        Idempotent by fingerprint: re-persisting an unchanged chunk collides on
        (tenant, mailbox, chunk_id, hash) and creates no duplicate work.
        """

    def claim_job(self, *, owner: str, lease_seconds: int = 300) -> Optional[ContextJob]:
        """Lease one claimable job, or None."""

    def get_job(self, job_id: int) -> Optional[ContextJob]: ...

    def load_chunk_state(self, chunk_db_id: int) -> Optional[ChunkContextState]: ...

    def commit_context(
        self,
        job: ContextJob,
        *,
        prefix: Optional[str],
        method: str,
        context_version: str,
        model_id: str,
        prompt_version: str,
        embed_text: str,
        embedding: Optional[Sequence[float]],
    ) -> bool:
        """Atomically write prefix + embed_text + embedding, or refuse if stale.

        Returns False when the chunk's inputs changed since the job was
        enqueued: the job is retired without writing, because its prefix
        describes text that no longer exists.
        """

    def fail_job(self, job_id: int, error: str, *, max_attempts: int = 3) -> None:
        """Return the job to pending for retry, or mark it failed once spent."""

    def chunks_needing_context(
        self, *, tenant_id: str, mailbox_id: str, limit: int = 100, after_id: int = 0
    ) -> list[ChunkContextState]: ...


class InMemoryContextJobStore:
    """Dict-backed store with the same semantics as Postgres. Test/demo only."""

    def __init__(self):
        self.chunks: dict[int, ChunkContextState] = {}
        self.embed_texts: dict[int, str] = {}
        self.embeddings: dict[int, list[float]] = {}
        self._jobs: dict[int, ContextJob] = {}
        self._next_job_id = 1

    # --- test/demo seam --------------------------------------------------
    def add_chunk(self, state: ChunkContextState, *, embed_text: str = "") -> ChunkContextState:
        self.chunks[state.chunk_db_id] = state
        self.embed_texts[state.chunk_db_id] = embed_text
        return state

    # --- ContextJobStore -------------------------------------------------
    def enqueue(
        self, state: ChunkContextState, *, prompt_version: str, model_id: str
    ) -> Optional[ContextJob]:
        digest = fingerprint_of(state.as_context_input(), prompt_version=prompt_version, model_id=model_id)
        # Already applied to the chunk: nothing to do, and no row to create.
        if state.context_input_hash == digest:
            return None
        for job in self._jobs.values():
            if (
                job.tenant_id == state.tenant_id
                and job.mailbox_id == state.mailbox_id
                and job.chunk_id == state.chunk_id
                and job.context_input_hash == digest
            ):
                return None  # mirrors the UNIQUE index / ON CONFLICT DO NOTHING
        job = ContextJob(
            id=self._next_job_id,
            chunk_db_id=state.chunk_db_id,
            tenant_id=state.tenant_id,
            mailbox_id=state.mailbox_id,
            chunk_id=state.chunk_id,
            context_input_hash=digest,
        )
        self._jobs[job.id] = job
        self._next_job_id += 1
        return job

    def claim_job(self, *, owner: str, lease_seconds: int = 300) -> Optional[ContextJob]:
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

    def get_job(self, job_id: int) -> Optional[ContextJob]:
        return self._jobs.get(job_id)

    def load_chunk_state(self, chunk_db_id: int) -> Optional[ChunkContextState]:
        return self.chunks.get(chunk_db_id)

    def commit_context(
        self,
        job: ContextJob,
        *,
        prefix: Optional[str],
        method: str,
        context_version: str,
        model_id: str,
        prompt_version: str,
        embed_text: str,
        embedding: Optional[Sequence[float]],
    ) -> bool:
        state = self.chunks.get(job.chunk_db_id)
        if state is None:
            self._retire(job)
            return False
        # Recompute from the chunk as it is NOW. A slow LLM call may have been
        # overtaken by a re-ingest; this is where that is caught.
        current = fingerprint_of(
            state.as_context_input(), prompt_version=prompt_version, model_id=model_id
        )
        if current != job.context_input_hash:
            self._retire(job)
            return False

        state.context_prefix = prefix
        state.context_method = method
        state.context_version = context_version
        state.context_input_hash = job.context_input_hash
        self.embed_texts[job.chunk_db_id] = embed_text
        if embedding is not None:
            self.embeddings[job.chunk_db_id] = list(embedding)
        self._retire(job)
        return True

    def _retire(self, job: ContextJob) -> None:
        job.status = "done"
        job.completed_at = utcnow()
        job.leased_until = None
        job.lease_owner = None
        job.last_error = None

    def fail_job(self, job_id: int, error: str, *, max_attempts: int = 3) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.last_error = error
        job.leased_until = None
        job.lease_owner = None
        job.status = "failed" if job.attempts >= max_attempts else "pending"

    def chunks_needing_context(
        self, *, tenant_id: str, mailbox_id: str, limit: int = 100, after_id: int = 0
    ) -> list[ChunkContextState]:
        matches = [
            state
            for state in self.chunks.values()
            if state.tenant_id == tenant_id
            and state.mailbox_id == mailbox_id
            and state.context_input_hash is None
            and state.chunk_db_id > after_id
        ]
        return sorted(matches, key=lambda s: s.chunk_db_id)[:limit]
