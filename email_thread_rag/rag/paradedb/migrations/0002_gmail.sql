-- Stage 3: Gmail OAuth + Pub/Sub push + durable delta sync.
-- Idempotent by construction (IF NOT EXISTS everywhere), same as 0001.
--
-- Gmail history IDs are unsigned 64-bit counters. They are stored as
-- numeric(20,0) and compared with numeric operators (>, GREATEST) only --
-- never as text, where '10' would sort before '9'.

CREATE TABLE IF NOT EXISTS gmail_mailboxes (
    id                       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id                text NOT NULL,
    mailbox_id               text NOT NULL,
    email_address            text NOT NULL,
    -- AES-GCM ciphertext produced by gmail.cipher.TokenCipher. The plaintext
    -- refresh token is never stored, logged, or returned by any endpoint.
    refresh_token_ciphertext bytea,
    token_key_id             text,
    status                   text NOT NULL DEFAULT 'pending',
    last_committed_history_id numeric(20, 0),
    watch_topic              text,
    watch_expiration         timestamptz,
    last_error               text,
    created_at               timestamptz NOT NULL DEFAULT now(),
    updated_at               timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, mailbox_id),
    CONSTRAINT gmail_mailboxes_status_check CHECK (
        status IN ('pending', 'active', 'needs_full_sync', 'disconnected', 'error')
    )
);

-- A Pub/Sub push carries only emailAddress, so an address must resolve to at
-- most one live mailbox. Partial index so a disconnected mailbox does not block
-- reconnecting the same address later.
CREATE UNIQUE INDEX IF NOT EXISTS gmail_mailboxes_live_address_idx
    ON gmail_mailboxes (email_address) WHERE status <> 'disconnected';

CREATE TABLE IF NOT EXISTS gmail_sync_jobs (
    id                   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    mailbox_db_id        bigint NOT NULL REFERENCES gmail_mailboxes (id) ON DELETE CASCADE,
    tenant_id            text NOT NULL,
    mailbox_id           text NOT NULL,
    requested_history_id numeric(20, 0) NOT NULL,
    status               text NOT NULL DEFAULT 'pending',
    attempts             integer NOT NULL DEFAULT 0,
    needs_full_sync      boolean NOT NULL DEFAULT false,
    leased_until         timestamptz,
    lease_owner          text,
    last_error           text,
    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now(),
    completed_at         timestamptz,
    CONSTRAINT gmail_sync_jobs_status_check CHECK (
        status IN ('pending', 'running', 'done', 'failed')
    )
);

-- Coalescing key: at most one pending job per mailbox. A second notification
-- for a mailbox that already has pending work raises requested_history_id to
-- the numeric max instead of queueing duplicate work.
CREATE UNIQUE INDEX IF NOT EXISTS gmail_sync_jobs_pending_mailbox_idx
    ON gmail_sync_jobs (mailbox_db_id) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS gmail_sync_jobs_claim_idx
    ON gmail_sync_jobs (status, leased_until);

-- Pub/Sub redelivers on any missed ack. Its message_id is the dedup key: the
-- webhook inserts here in the same transaction that creates/coalesces the job,
-- so a redelivered notification is a no-op rather than duplicate work.
CREATE TABLE IF NOT EXISTS gmail_pubsub_messages (
    pubsub_message_id text PRIMARY KEY,
    received_at       timestamptz NOT NULL DEFAULT now()
);

-- OAuth state: single-use and expiring. Consumption is an atomic conditional
-- UPDATE (see PostgresSyncStore.consume_oauth_state), never read-then-write.
-- code_verifier is a short-lived PKCE secret bound to one state row; it is
-- deleted with the row and never leaves the server.
CREATE TABLE IF NOT EXISTS gmail_oauth_states (
    state         text PRIMARY KEY,
    tenant_id     text NOT NULL,
    mailbox_id    text NOT NULL,
    code_verifier text NOT NULL,
    redirect_uri  text NOT NULL,
    expires_at    timestamptz NOT NULL,
    consumed_at   timestamptz,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS gmail_oauth_states_expires_idx ON gmail_oauth_states (expires_at);
