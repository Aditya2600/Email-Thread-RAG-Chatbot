"""Postgres-backed context job queue. Satisfies ContextJobStore.

psycopg is imported at module level here (this module IS the Postgres one and is
never imported by the memory path), but every caller reaches it lazily.

Connections are expected in autocommit mode with explicit ``transaction()``
blocks -- the same discipline Stage 3 uses -- so no transaction is ever held
open across an LLM call.
"""

from __future__ import annotations

from typing import Optional, Sequence

from psycopg.types.json import Jsonb  # noqa: F401  (kept for symmetry with repo conventions)

from email_thread_rag.context.fingerprint import fingerprint_of
from email_thread_rag.context.models import ChunkContextState, ContextJob
from email_thread_rag.rag.paradedb.repository import content_hash, vector_literal

_JOB_COLUMNS = (
    "id, chunk_db_id, tenant_id, mailbox_id, chunk_id, context_input_hash, status, "
    "attempts, leased_until, lease_owner, last_error, completed_at"
)

_CHUNK_STATE_SELECT = """
    SELECT c.id AS chunk_db_id, c.chunk_id, c.tenant_id, c.mailbox_id, c.text,
           c.sender, c.subject, c.thread_id, c.sent_at, c.metadata,
           c.context_prefix, c.context_method, c.context_version, c.context_input_hash,
           parent.subject AS parent_subject
    FROM email_chunks c
    LEFT JOIN email_messages parent
      ON parent.tenant_id = c.tenant_id
     AND parent.mailbox_id = c.mailbox_id
     AND parent.message_id = c.metadata->>'in_reply_to'
"""


def _row_to_job(row) -> Optional[ContextJob]:
    if row is None:
        return None
    return ContextJob(**row)


def _row_to_state(row) -> Optional[ChunkContextState]:
    if row is None:
        return None
    metadata = row.get("metadata") or {}
    return ChunkContextState(
        chunk_db_id=row["chunk_db_id"],
        chunk_id=row["chunk_id"],
        tenant_id=row["tenant_id"],
        mailbox_id=row["mailbox_id"],
        text=row["text"],
        subject=row["subject"],
        sender=row["sender"],
        thread_id=row["thread_id"],
        date=row["sent_at"],
        to=list(metadata.get("to") or []),
        cc=list(metadata.get("cc") or []),
        in_reply_to=metadata.get("in_reply_to"),
        # NULL unless the parent message is already persisted in this mailbox:
        # Stage 4 never fetches a parent it does not have.
        parent_subject=row.get("parent_subject"),
        context_prefix=row["context_prefix"],
        context_method=row["context_method"],
        context_version=row["context_version"],
        context_input_hash=row["context_input_hash"],
    )


class PostgresContextJobStore:
    def __init__(self, conn, *, embedding_dim: int = 768):
        self.conn = conn
        self.embedding_dim = embedding_dim

    def enqueue(
        self, state: ChunkContextState, *, prompt_version: str, model_id: str
    ) -> Optional[ContextJob]:
        digest = fingerprint_of(state.as_context_input(), prompt_version=prompt_version, model_id=model_id)
        if state.context_input_hash == digest:
            return None  # already contextualized for exactly these inputs
        row = self.conn.execute(
            f"""
            INSERT INTO chunk_context_jobs (
                chunk_db_id, tenant_id, mailbox_id, chunk_id, context_input_hash
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (tenant_id, mailbox_id, chunk_id, context_input_hash) DO NOTHING
            RETURNING {_JOB_COLUMNS}
            """,
            (state.chunk_db_id, state.tenant_id, state.mailbox_id, state.chunk_id, digest),
        ).fetchone()
        # DO NOTHING returns no row: the job already exists, which is the point.
        return _row_to_job(row)

    def enqueue_message(self, message_id: str, *, tenant_id: str, mailbox_id: str, prompt_version: str, model_id: str) -> int:
        """Enqueue every chunk of one message. Used right after persistence."""
        rows = self.conn.execute(
            _CHUNK_STATE_SELECT + " WHERE c.message_id = %s AND c.tenant_id = %s AND c.mailbox_id = %s",
            (message_id, tenant_id, mailbox_id),
        ).fetchall()
        queued = 0
        for row in rows:
            if self.enqueue(_row_to_state(row), prompt_version=prompt_version, model_id=model_id):
                queued += 1
        return queued

    def claim_job(self, *, owner: str, lease_seconds: int = 300) -> Optional[ContextJob]:
        """Lease one job. SKIP LOCKED lets N workers take disjoint jobs; the
        expired-lease clause reclaims work from a worker that died mid-call."""
        with self.conn.transaction():
            candidate = self.conn.execute(
                "SELECT id FROM chunk_context_jobs "
                "WHERE status = 'pending' OR (status = 'running' AND leased_until <= now()) "
                "ORDER BY id ASC FOR UPDATE SKIP LOCKED LIMIT 1"
            ).fetchone()
            if candidate is None:
                return None
            row = self.conn.execute(
                f"""
                UPDATE chunk_context_jobs SET
                    status = 'running',
                    attempts = attempts + 1,
                    lease_owner = %s,
                    leased_until = now() + make_interval(secs => %s),
                    updated_at = now()
                WHERE id = %s
                RETURNING {_JOB_COLUMNS}
                """,
                (owner, lease_seconds, candidate["id"]),
            ).fetchone()
        return _row_to_job(row)

    def get_job(self, job_id: int) -> Optional[ContextJob]:
        return _row_to_job(
            self.conn.execute(
                f"SELECT {_JOB_COLUMNS} FROM chunk_context_jobs WHERE id = %s", (job_id,)
            ).fetchone()
        )

    def load_chunk_state(self, chunk_db_id: int) -> Optional[ChunkContextState]:
        return _row_to_state(
            self.conn.execute(_CHUNK_STATE_SELECT + " WHERE c.id = %s", (chunk_db_id,)).fetchone()
        )

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
        embedding_model: Optional[str] = None,
    ) -> bool:
        with self.conn.transaction():
            # FOR UPDATE: serialize against a concurrent re-ingest of this chunk
            # so the fingerprint we check is the one we write against.
            row = self.conn.execute(
                _CHUNK_STATE_SELECT + " WHERE c.id = %s FOR UPDATE OF c", (job.chunk_db_id,)
            ).fetchone()
            state = _row_to_state(row)
            if state is None:
                self._retire(job.id)
                return False

            current = fingerprint_of(
                state.as_context_input(), prompt_version=prompt_version, model_id=model_id
            )
            if current != job.context_input_hash:
                # The chunk changed under a slow LLM call. This prefix describes
                # text that no longer exists; a newer job already covers the new
                # text, so drop this result rather than overwrite.
                self._retire(job.id)
                return False

            self.conn.execute(
                """
                UPDATE email_chunks SET
                    context_prefix = %s,
                    context_method = %s,
                    context_version = %s,
                    context_input_hash = %s,
                    context_model = %s,
                    context_updated_at = now(),
                    embed_text = %s,
                    embedding = COALESCE(%s::vector, embedding),
                    embedding_model = COALESCE(%s, embedding_model),
                    content_hash = %s,
                    updated_at = now()
                WHERE id = %s
                """,
                (
                    prefix,
                    method,
                    context_version,
                    job.context_input_hash,
                    model_id,
                    embed_text,
                    vector_literal(embedding, expected_dim=self.embedding_dim) if embedding else None,
                    embedding_model,
                    content_hash(embed_text),
                    job.chunk_db_id,
                ),
            )
            self._retire(job.id)
        return True

    def _retire(self, job_id: int) -> None:
        self.conn.execute(
            "UPDATE chunk_context_jobs SET status = 'done', completed_at = now(), "
            "leased_until = NULL, lease_owner = NULL, last_error = NULL, updated_at = now() "
            "WHERE id = %s",
            (job_id,),
        )

    def fail_job(self, job_id: int, error: str, *, max_attempts: int = 3) -> None:
        with self.conn.transaction():
            self.conn.execute(
                """
                UPDATE chunk_context_jobs SET
                    status = CASE WHEN attempts >= %s THEN 'failed' ELSE 'pending' END,
                    last_error = %s,
                    leased_until = NULL,
                    lease_owner = NULL,
                    updated_at = now()
                WHERE id = %s
                """,
                (max_attempts, error, job_id),
            )

    def chunks_needing_context(
        self, *, tenant_id: str, mailbox_id: str, limit: int = 100, after_id: int = 0
    ) -> list[ChunkContextState]:
        rows = self.conn.execute(
            _CHUNK_STATE_SELECT
            + """
            WHERE c.tenant_id = %s AND c.mailbox_id = %s
              AND c.context_input_hash IS NULL
              AND c.id > %s
            ORDER BY c.id ASC
            LIMIT %s
            """,
            (tenant_id, mailbox_id, after_id, limit),
        ).fetchall()
        return [_row_to_state(row) for row in rows]


def build_store(conn, *, embedding_dim: int = 768) -> PostgresContextJobStore:
    return PostgresContextJobStore(conn, embedding_dim=embedding_dim)
