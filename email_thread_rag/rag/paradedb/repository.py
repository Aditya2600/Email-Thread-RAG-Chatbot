"""Thin, transactional psycopg3 repository for the ParadeDB backend.

No ORM: the schema is two tables (see migrations/0001_init.sql) and every
query here is a plain parameterized statement. Vectors are passed as
pgvector text-literals (``'[0.1,0.2,...]'::vector``) so we don't need the
extra ``pgvector`` Python package on top of ``psycopg``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from email_thread_rag.app.schemas import ChunkRecord, EmailRecord

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class ParadeDBConfigError(RuntimeError):
    """Raised when RAG_BACKEND=paradedb is selected but misconfigured.

    An explicit ParadeDB selection must fail loudly, never fall back to the
    memory backend silently.
    """


def _require_database_url(database_url: str | None) -> str:
    if not database_url:
        raise ParadeDBConfigError(
            "RAG_BACKEND=paradedb requires DATABASE_URL to be set (e.g. "
            "postgresql://user:pass@localhost:5433/email_rag)."
        )
    return database_url


def connect(database_url: str | None) -> psycopg.Connection:
    """Open a connection, failing clearly if config or connectivity is bad."""
    url = _require_database_url(database_url)
    try:
        return psycopg.connect(url, row_factory=dict_row)
    except psycopg.OperationalError as exc:
        raise ParadeDBConfigError(f"Could not connect to ParadeDB at DATABASE_URL: {exc}") from exc


def verify_extensions(conn: psycopg.Connection) -> dict[str, str | None]:
    """Return installed versions for pg_search/vector; raise if either is missing."""
    rows = conn.execute(
        "SELECT name, installed_version FROM pg_available_extensions "
        "WHERE name IN ('pg_search', 'vector')"
    ).fetchall()
    found = {row["name"]: row["installed_version"] for row in rows}
    for required in ("pg_search", "vector"):
        if not found.get(required):
            raise ParadeDBConfigError(
                f"Required extension '{required}' is not available on this Postgres instance."
            )
    return found


def apply_migrations(conn: psycopg.Connection, *, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply each .sql file in order, tracked in schema_migrations. Idempotent.

    No migration framework: filenames sort lexicographically (0001_, 0002_, ...)
    and each file's own DDL is written with IF NOT EXISTS, so re-running this
    against an already-migrated database is a safe no-op.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "filename text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    applied_now: list[str] = []
    for path in sorted(migrations_dir.glob("*.sql")):
        already = conn.execute(
            "SELECT 1 FROM schema_migrations WHERE filename = %s", (path.name,)
        ).fetchone()
        if already:
            continue
        conn.execute(path.read_text(encoding="utf-8"))
        conn.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,))
        applied_now.append(path.name)
    conn.commit()
    return applied_now


def vector_literal(embedding: Sequence[float] | None, *, expected_dim: int) -> str | None:
    if embedding is None:
        return None
    if len(embedding) != expected_dim:
        raise ValueError(f"embedding has dimension {len(embedding)}, expected {expected_dim}")
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class EmbeddedChunk:
    """A ``ChunkRecord`` plus the embedding to persist alongside it."""

    chunk: ChunkRecord
    embedding: Sequence[float] | None = None
    embedding_model: str | None = None
    embedding_version: str | None = None


class ParadeDBRepository:
    """Tenant/mailbox-scoped, transactional persistence for emails + chunks."""

    def __init__(self, conn: psycopg.Connection, *, embedding_dim: int = 384):
        self.conn = conn
        self.embedding_dim = embedding_dim

    def upsert_message(self, email: EmailRecord, *, tenant_id: str, mailbox_id: str) -> int:
        row = self.conn.execute(
            """
            INSERT INTO email_messages (
                tenant_id, mailbox_id, message_id, thread_id, sender, recipients, cc,
                subject, sent_at, authored_text, quoted_text, signature_text,
                disclaimer_text, metadata, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (tenant_id, mailbox_id, message_id) DO UPDATE SET
                thread_id = EXCLUDED.thread_id,
                sender = EXCLUDED.sender,
                recipients = EXCLUDED.recipients,
                cc = EXCLUDED.cc,
                subject = EXCLUDED.subject,
                sent_at = EXCLUDED.sent_at,
                authored_text = EXCLUDED.authored_text,
                quoted_text = EXCLUDED.quoted_text,
                signature_text = EXCLUDED.signature_text,
                disclaimer_text = EXCLUDED.disclaimer_text,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            RETURNING id
            """,
            (
                tenant_id,
                mailbox_id,
                email.message_id,
                email.thread_id,
                email.sender,
                email.to,
                email.cc,
                email.subject,
                email.date,
                email.authored_text or "",
                email.quoted_text,
                email.signature_text,
                email.disclaimer_text,
                Jsonb({}),
            ),
        ).fetchone()
        return row["id"]

    def upsert_chunks(
        self,
        message_db_id: int,
        embedded_chunks: Iterable[EmbeddedChunk],
        *,
        tenant_id: str,
        mailbox_id: str,
    ) -> list[str]:
        current_ids: list[str] = []
        for index, item in enumerate(embedded_chunks):
            chunk = item.chunk
            current_ids.append(chunk.chunk_id)
            embedding_literal = vector_literal(item.embedding, expected_dim=self.embedding_dim)
            # doc_id/source_path/source_type/token_count/ocr_used/attachment_name/
            # page_no aren't first-class email_chunks columns (Stage 2's schema
            # doesn't need them for BM25/vector search); round-trip them through
            # metadata instead of a schema change, so the Stage-2.5 engine
            # adapter can rebuild a full canonical ChunkRecord on the way out.
            round_trip_metadata = {
                **chunk.metadata,
                "_doc_id": chunk.doc_id,
                "_source_path": chunk.source_path,
                "_source_type": chunk.source_type,
                "_token_count": chunk.token_count,
                "_ocr_used": chunk.ocr_used,
                "_attachment_name": chunk.attachment_name,
                "_page_no": chunk.page_no,
            }
            self.conn.execute(
                """
                INSERT INTO email_chunks (
                    chunk_id, message_db_id, tenant_id, mailbox_id, message_id, thread_id,
                    chunk_index, chunk_kind, sender, subject, sent_at, text, embed_text,
                    source_start, source_end, embedding, embedding_model, embedding_version,
                    content_hash, metadata, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s::vector, %s, %s, %s, %s, now()
                )
                ON CONFLICT (tenant_id, mailbox_id, chunk_id) DO UPDATE SET
                    message_db_id = EXCLUDED.message_db_id,
                    thread_id = EXCLUDED.thread_id,
                    chunk_index = EXCLUDED.chunk_index,
                    chunk_kind = EXCLUDED.chunk_kind,
                    sender = EXCLUDED.sender,
                    subject = EXCLUDED.subject,
                    sent_at = EXCLUDED.sent_at,
                    text = EXCLUDED.text,
                    embed_text = EXCLUDED.embed_text,
                    source_start = EXCLUDED.source_start,
                    source_end = EXCLUDED.source_end,
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_version = EXCLUDED.embedding_version,
                    content_hash = EXCLUDED.content_hash,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                (
                    chunk.chunk_id,
                    message_db_id,
                    tenant_id,
                    mailbox_id,
                    chunk.message_id,
                    chunk.thread_id,
                    index,
                    chunk.kind,
                    chunk.sender,
                    chunk.subject,
                    chunk.date,
                    chunk.text,
                    chunk.embed_text or chunk.text,
                    chunk.source_start,
                    chunk.source_end,
                    embedding_literal,
                    item.embedding_model,
                    item.embedding_version,
                    content_hash(chunk.embed_text or chunk.text),
                    Jsonb(round_trip_metadata),
                ),
            )
        return current_ids

    def delete_stale_chunks(
        self,
        message_db_id: int,
        current_chunk_ids: list[str],
        *,
        tenant_id: str,
        mailbox_id: str,
    ) -> int:
        result = self.conn.execute(
            """
            DELETE FROM email_chunks
            WHERE message_db_id = %s AND tenant_id = %s AND mailbox_id = %s
              AND NOT (chunk_id = ANY(%s))
            """,
            (message_db_id, tenant_id, mailbox_id, current_chunk_ids),
        )
        return result.rowcount

    def reprocess_message(
        self,
        email: EmailRecord,
        embedded_chunks: list[EmbeddedChunk],
        *,
        tenant_id: str,
        mailbox_id: str,
    ) -> int:
        """Upsert message + current chunks, delete stale ones, atomically."""
        with self.conn.transaction():
            message_db_id = self.upsert_message(email, tenant_id=tenant_id, mailbox_id=mailbox_id)
            current_ids = self.upsert_chunks(
                message_db_id, embedded_chunks, tenant_id=tenant_id, mailbox_id=mailbox_id
            )
            self.delete_stale_chunks(message_db_id, current_ids, tenant_id=tenant_id, mailbox_id=mailbox_id)
        return message_db_id

    def load_chunk(self, chunk_id: str, *, tenant_id: str, mailbox_id: str) -> dict[str, Any] | None:
        return self.conn.execute(
            "SELECT * FROM email_chunks WHERE chunk_id = %s AND tenant_id = %s AND mailbox_id = %s",
            (chunk_id, tenant_id, mailbox_id),
        ).fetchone()
