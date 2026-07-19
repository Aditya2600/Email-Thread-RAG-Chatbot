-- Stage 4: asynchronous LLM contextualization of persisted chunks.
--
-- email_chunks already carries context_prefix/context_method/context_version
-- (reserved as extension points by 0001_init.sql), so this migration adds only
-- the genuinely missing pieces: the input fingerprint, the model identity, an
-- audit timestamp, and the job queue itself.
--
-- Idempotent by construction (IF NOT EXISTS everywhere), same as 0001/0002.

-- The fingerprint of the inputs the prefix was derived from (clean text +
-- selected metadata + parent id + prompt version + model id). This is the
-- stale-job guard: a worker may only write a prefix if the fingerprint it
-- computed at claim time still matches the chunk's inputs at commit time.
ALTER TABLE email_chunks ADD COLUMN IF NOT EXISTS context_input_hash text;
-- Which model produced the prefix. Auditable without storing the prompt or the
-- raw response, neither of which is ever persisted.
ALTER TABLE email_chunks ADD COLUMN IF NOT EXISTS context_model text;
ALTER TABLE email_chunks ADD COLUMN IF NOT EXISTS context_updated_at timestamptz;

-- 'none' = never contextualized; 'deterministic' = Stage-1 headers only (the
-- fallback when a model is unavailable or its output fails validation);
-- 'llm' = a validated model prefix is present.
ALTER TABLE email_chunks DROP CONSTRAINT IF EXISTS email_chunks_context_method_check;
ALTER TABLE email_chunks ADD CONSTRAINT email_chunks_context_method_check
    CHECK (context_method IS NULL OR context_method IN ('none', 'deterministic', 'llm'));

-- Backfill scans claim work by "has this chunk been contextualized yet?", which
-- is a partial-index question, not a full-table one.
CREATE INDEX IF NOT EXISTS email_chunks_needs_context_idx
    ON email_chunks (tenant_id, mailbox_id, id)
    WHERE context_input_hash IS NULL;

CREATE TABLE IF NOT EXISTS chunk_context_jobs (
    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chunk_db_id       bigint NOT NULL REFERENCES email_chunks (id) ON DELETE CASCADE,
    tenant_id         text NOT NULL,
    mailbox_id        text NOT NULL,
    chunk_id          text NOT NULL,
    -- Job identity. The hash already folds in prompt version + model id, so
    -- this one column is what makes a job unique per chunk/version/input.
    context_input_hash text NOT NULL,
    status            text NOT NULL DEFAULT 'pending'
                      CHECK (status IN ('pending', 'running', 'done', 'failed')),
    attempts          integer NOT NULL DEFAULT 0,
    leased_until      timestamptz,
    lease_owner       text,
    -- Validation/provider failure reason only. Never a prompt, never a raw
    -- model response: this column is read by humans and by tests.
    last_error        text,
    completed_at      timestamptz,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    -- Re-persisting an unchanged chunk produces the same fingerprint, so the
    -- enqueue collides here and does no duplicate work.
    UNIQUE (tenant_id, mailbox_id, chunk_id, context_input_hash)
);

CREATE INDEX IF NOT EXISTS chunk_context_jobs_claimable_idx
    ON chunk_context_jobs (id)
    WHERE status IN ('pending', 'running');
