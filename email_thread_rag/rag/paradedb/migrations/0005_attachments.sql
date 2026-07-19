-- Stage 8: PDF attachment metadata + extraction queue.
--
-- Two tables, mirroring the Stage 4/5 job pattern (context/graph):
--   * email_attachments  -- one row per Gmail attachment we know about, carrying
--     its metadata plus the extraction status/method/error and a content hash.
--   * attachment_extraction_jobs -- the queue (pending -> running -> done|failed),
--     leased/claimed, keyed by an input hash so re-syncing an unchanged
--     attachment collides and does no duplicate work.
--
-- Idempotent by construction (IF NOT EXISTS everywhere), same as 0001-0004.
-- Scope: PDF only. Non-PDF attachments are never inserted here; nothing else in
-- the schema needs to know attachments exist -- extracted page text lands in the
-- existing email_chunks table as chunk_kind='attachment'.

CREATE TABLE IF NOT EXISTS email_attachments (
    id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id           text NOT NULL,
    mailbox_id          text NOT NULL,
    message_db_id       bigint REFERENCES email_messages (id) ON DELETE CASCADE,
    message_id          text NOT NULL,
    thread_id           text,
    -- Gmail's own attachmentId (stable per message part). The canonical
    -- attachment_id is what chunk ids are namespaced under.
    gmail_attachment_id text NOT NULL,
    attachment_id       text NOT NULL,
    filename            text NOT NULL,
    media_type          text NOT NULL,
    byte_size           bigint,
    -- sha256 of the decoded bytes, filled in by the worker after fetch. NULL
    -- until a first successful (or attempted) extraction.
    content_hash        text,
    extraction_status   text NOT NULL DEFAULT 'pending'
                        CHECK (extraction_status IN (
                            'pending', 'running', 'done', 'failed', 'unsupported')),
    -- native_pdf | ocr | mixed, or NULL when not yet / never extracted.
    extraction_method   text,
    -- Safe-to-log reason string (a rule name / status), never document bytes.
    extraction_error    text,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, mailbox_id, message_id, gmail_attachment_id)
);

CREATE INDEX IF NOT EXISTS email_attachments_tenant_mailbox_idx
    ON email_attachments (tenant_id, mailbox_id);
CREATE INDEX IF NOT EXISTS email_attachments_message_idx
    ON email_attachments (message_id);

CREATE TABLE IF NOT EXISTS attachment_extraction_jobs (
    id                    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    attachment_db_id      bigint NOT NULL REFERENCES email_attachments (id) ON DELETE CASCADE,
    tenant_id             text NOT NULL,
    mailbox_id            text NOT NULL,
    attachment_id         text NOT NULL,
    -- Folds the Gmail attachmentId + byte size: a re-synced, unchanged
    -- attachment collides here (ON CONFLICT DO NOTHING); a changed one (new
    -- size) makes a fresh job, which replaces the stale page chunks.
    extraction_input_hash text NOT NULL,
    status                text NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('pending', 'running', 'done', 'failed')),
    attempts              integer NOT NULL DEFAULT 0,
    leased_until          timestamptz,
    lease_owner           text,
    last_error            text,
    error_rule            text,
    completed_at          timestamptz,
    created_at            timestamptz NOT NULL DEFAULT now(),
    updated_at            timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, mailbox_id, attachment_id, extraction_input_hash)
);

CREATE INDEX IF NOT EXISTS attachment_extraction_jobs_claimable_idx
    ON attachment_extraction_jobs (id)
    WHERE status IN ('pending', 'running');
