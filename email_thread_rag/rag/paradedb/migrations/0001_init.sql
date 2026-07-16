-- Stage 2: email_messages + email_chunks on ParadeDB (pg_search + pgvector).
-- Idempotent by construction (IF NOT EXISTS everywhere) so re-running this
-- file against an already-migrated database is a safe no-op.
--
-- Embedding dimension is pinned at 384 (sentence-transformers/all-MiniLM-L6-v2
-- and the deterministic HashingEncoder fallback both emit 384-dim vectors).
-- Changing this requires a new migration + backfill; do not point a different
-- dimension encoder at this column without one.

CREATE EXTENSION IF NOT EXISTS pg_search;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS email_messages (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id       text NOT NULL,
    mailbox_id      text NOT NULL,
    message_id      text NOT NULL,
    thread_id       text,
    sender          text,
    recipients      text[] NOT NULL DEFAULT '{}',
    cc              text[] NOT NULL DEFAULT '{}',
    subject         text,
    sent_at         timestamptz,
    authored_text   text NOT NULL,
    quoted_text     text,
    signature_text  text,
    disclaimer_text text,
    metadata        jsonb NOT NULL DEFAULT '{}',
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, mailbox_id, message_id)
);

CREATE TABLE IF NOT EXISTS email_chunks (
    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chunk_id          text NOT NULL,
    message_db_id     bigint REFERENCES email_messages (id) ON DELETE CASCADE,
    tenant_id         text NOT NULL,
    mailbox_id        text NOT NULL,
    message_id        text NOT NULL,
    thread_id         text,
    chunk_index       integer NOT NULL,
    chunk_kind        text NOT NULL DEFAULT 'email_body',
    sender            text,
    subject           text,
    sent_at           timestamptz,
    text              text NOT NULL,
    embed_text        text NOT NULL,
    source_start      integer,
    source_end        integer,
    embedding         vector(384),
    embedding_model   text,
    embedding_version text,
    content_hash      text NOT NULL,
    -- Nullable extension points for future contextual retrieval (Stage 3+).
    -- No LLM contextual prefix is generated in Stage 2; these stay unused.
    context_prefix    text,
    context_method    text,
    context_version   text,
    metadata          jsonb NOT NULL DEFAULT '{}',
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, mailbox_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS email_chunks_tenant_mailbox_idx ON email_chunks (tenant_id, mailbox_id);
CREATE INDEX IF NOT EXISTS email_chunks_message_id_idx ON email_chunks (message_id);
CREATE INDEX IF NOT EXISTS email_chunks_thread_id_idx ON email_chunks (thread_id);
CREATE INDEX IF NOT EXISTS email_chunks_sent_at_idx ON email_chunks (sent_at);

-- BM25 lexical index (pg_search, current post-0.20 syntax: no
-- paradedb.create_bm25 helper). key_field must be the numeric primary key.
CREATE INDEX IF NOT EXISTS email_chunks_bm25_idx ON email_chunks
USING bm25 (
    id,
    embed_text,
    tenant_id,
    mailbox_id,
    message_id,
    thread_id,
    sender,
    subject,
    sent_at,
    chunk_kind
)
WITH (key_field = 'id');

-- Dense cosine index. Rows with a NULL embedding (lexical-only, e.g. no
-- embedding model available) are simply excluded by dense queries, never by
-- this index definition.
CREATE INDEX IF NOT EXISTS email_chunks_embedding_hnsw_idx ON email_chunks
USING hnsw (embedding vector_cosine_ops);
