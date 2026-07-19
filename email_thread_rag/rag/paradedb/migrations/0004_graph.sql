-- Stage 5: evidence-backed entity/relation/temporal-fact extraction.
--
-- Every row in these tables is either (a) traceable to an exact span of a
-- chunk's immutable `text` (evidence_kind='text', mention/fact offsets), or
-- (b) a deterministic metadata observation that must never be presented as
-- authored-text proof (evidence_kind='metadata', offsets NULL, enforced by a
-- CHECK). The LLM never writes offsets: it supplies evidence *strings*, code
-- locates each string in `text` itself, and anything it cannot locate is
-- dropped before it ever reaches this schema.
--
-- Idempotent by construction (IF NOT EXISTS everywhere), same as 0001-0003.

-- The stale-job guard, mirrored from Stage 4's context_input_hash: the hash of
-- everything the extraction was derived from (clean text + selected metadata +
-- schema version + prompt version + model id). A worker may only write graph
-- rows if the hash it computed at claim time still matches the chunk at commit.
ALTER TABLE email_chunks ADD COLUMN IF NOT EXISTS graph_input_hash text;
ALTER TABLE email_chunks ADD COLUMN IF NOT EXISTS graph_extracted_at timestamptz;

-- Backfill asks "has this chunk been graph-extracted yet?" -- a partial-index
-- question, not a full-table scan.
CREATE INDEX IF NOT EXISTS email_chunks_needs_graph_idx
    ON email_chunks (tenant_id, mailbox_id, id)
    WHERE graph_input_hash IS NULL;

-- The canonical entity per (tenant, mailbox). normalized_name is the casefolded,
-- Unicode-normalized, whitespace-collapsed uniqueness key; canonical_name keeps
-- the first-seen display form. The UNIQUE key is per-mailbox on purpose: people
-- and organizations are never fuzzy-merged across mailboxes.
CREATE TABLE IF NOT EXISTS graph_entities (
    entity_id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id       text NOT NULL,
    mailbox_id      text NOT NULL,
    entity_type     text NOT NULL CHECK (entity_type IN (
                        'PERSON', 'ORG', 'PROJECT', 'TOPIC', 'DOCUMENT',
                        'MEETING', 'COMMITMENT', 'DATE', 'MONEY')),
    canonical_name  text NOT NULL,
    normalized_name text NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, mailbox_id, entity_type, normalized_name)
);

-- One entity mention grounded in one chunk. chunk_start/chunk_end index into the
-- immutable chunk `text`; source_start/source_end (optional) index into the
-- authored email body when the chunk carries a body offset. The UNIQUE key makes
-- re-extraction of the same span a no-op.
CREATE TABLE IF NOT EXISTS chunk_entity_mentions (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id          text NOT NULL,
    mailbox_id         text NOT NULL,
    chunk_db_id        bigint NOT NULL REFERENCES email_chunks (id) ON DELETE CASCADE,
    chunk_id           text NOT NULL,
    entity_id          bigint NOT NULL REFERENCES graph_entities (entity_id) ON DELETE CASCADE,
    mention_text       text NOT NULL,
    chunk_start        integer NOT NULL,
    chunk_end          integer NOT NULL,
    source_start       integer,
    source_end         integer,
    extraction_method  text NOT NULL,
    extraction_version text NOT NULL,
    extraction_model   text NOT NULL,
    created_at         timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, mailbox_id, chunk_db_id, entity_id, chunk_start, chunk_end)
);

-- A subject-predicate-object observation. evidence_kind='text' carries exact
-- offsets into the chunk; evidence_kind='metadata' carries none. The CHECK is
-- the schema-level guarantee that a metadata observation (SENT/CC/REPLY_TO) can
-- never masquerade as an authored-text span.
CREATE TABLE IF NOT EXISTS relation_observations (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id          text NOT NULL,
    mailbox_id         text NOT NULL,
    subject_entity_id  bigint NOT NULL REFERENCES graph_entities (entity_id) ON DELETE CASCADE,
    predicate          text NOT NULL CHECK (predicate IN (
                          'MENTIONS', 'WORKS_ON', 'ASSIGNED_TO', 'APPROVED',
                          'REJECTED', 'REFERS_TO',            -- semantic (text)
                          'SENT', 'CC', 'REPLY_TO')),         -- metadata only
    object_entity_id   bigint NOT NULL REFERENCES graph_entities (entity_id) ON DELETE CASCADE,
    chunk_db_id        bigint NOT NULL REFERENCES email_chunks (id) ON DELETE CASCADE,
    chunk_id           text NOT NULL,
    chunk_start        integer,
    chunk_end          integer,
    mention_text       text,
    evidence_kind      text NOT NULL CHECK (evidence_kind IN ('text', 'metadata')),
    extraction_method  text NOT NULL,
    extraction_version text NOT NULL,
    extraction_model   text NOT NULL,
    created_at         timestamptz NOT NULL DEFAULT now(),
    CHECK (
        (evidence_kind = 'text'     AND chunk_start IS NOT NULL AND chunk_end IS NOT NULL)
     OR (evidence_kind = 'metadata' AND chunk_start IS NULL     AND chunk_end IS NULL)
    )
);

-- A fact is an observation, not automatic truth. New facts land 'active'; a fact
-- is only ever moved to 'superseded' when a later fact's *evidence text* carries
-- an explicit update cue (see extract.UPDATE_CUE) -- never merely because a newer
-- email exists. supersedes_fact_id points at the fact this one replaced.
CREATE TABLE IF NOT EXISTS facts (
    fact_id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id           text NOT NULL,
    mailbox_id          text NOT NULL,
    subject             text NOT NULL,
    predicate           text NOT NULL,
    object_value        text NOT NULL,
    normalized_subject  text NOT NULL,
    normalized_predicate text NOT NULL,
    status              text NOT NULL DEFAULT 'active'
                        CHECK (status IN ('candidate', 'active', 'superseded')),
    effective_date      timestamptz,
    supersedes_fact_id  bigint REFERENCES facts (fact_id) ON DELETE SET NULL,
    extraction_method   text NOT NULL,
    extraction_version  text NOT NULL,
    extraction_model    text NOT NULL,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now()
);

-- Exact evidence for a fact: which chunk, which span, and a hash of the evidence
-- text so tampering with the source is detectable.
CREATE TABLE IF NOT EXISTS fact_evidence (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    fact_id       bigint NOT NULL REFERENCES facts (fact_id) ON DELETE CASCADE,
    tenant_id     text NOT NULL,
    mailbox_id    text NOT NULL,
    chunk_db_id   bigint NOT NULL REFERENCES email_chunks (id) ON DELETE CASCADE,
    chunk_id      text NOT NULL,
    chunk_start   integer NOT NULL,
    chunk_end     integer NOT NULL,
    source_start  integer,
    source_end    integer,
    evidence_text text NOT NULL,
    evidence_hash text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);

-- The extraction queue. Identical mechanics to Stage 4's chunk_context_jobs:
-- the input hash folds in schema version + prompt version + model id, so this
-- one column is what makes a job unique per chunk/version/input, and re-ingesting
-- an unchanged chunk collides here and does no duplicate work.
CREATE TABLE IF NOT EXISTS graph_extraction_jobs (
    id                    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chunk_db_id           bigint NOT NULL REFERENCES email_chunks (id) ON DELETE CASCADE,
    tenant_id             text NOT NULL,
    mailbox_id            text NOT NULL,
    chunk_id              text NOT NULL,
    extraction_input_hash text NOT NULL,
    status                text NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('pending', 'running', 'done', 'failed')),
    attempts              integer NOT NULL DEFAULT 0,
    leased_until          timestamptz,
    lease_owner           text,
    -- Failure reason, safe to log: a rule name / status code, never a prompt or
    -- a raw model response.
    last_error            text,
    error_rule            text,
    completed_at          timestamptz,
    created_at            timestamptz NOT NULL DEFAULT now(),
    updated_at            timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, mailbox_id, chunk_id, extraction_input_hash)
);

CREATE INDEX IF NOT EXISTS graph_extraction_jobs_claimable_idx
    ON graph_extraction_jobs (id)
    WHERE status IN ('pending', 'running');

CREATE INDEX IF NOT EXISTS graph_entities_lookup_idx
    ON graph_entities (tenant_id, mailbox_id, entity_type, normalized_name);
CREATE INDEX IF NOT EXISTS chunk_entity_mentions_entity_idx
    ON chunk_entity_mentions (tenant_id, mailbox_id, entity_id);
CREATE INDEX IF NOT EXISTS chunk_entity_mentions_chunk_idx
    ON chunk_entity_mentions (chunk_db_id);
CREATE INDEX IF NOT EXISTS relation_observations_subject_idx
    ON relation_observations (tenant_id, mailbox_id, subject_entity_id);
CREATE INDEX IF NOT EXISTS relation_observations_chunk_idx
    ON relation_observations (chunk_db_id);
CREATE INDEX IF NOT EXISTS facts_lookup_idx
    ON facts (tenant_id, mailbox_id, normalized_subject, normalized_predicate, status);
CREATE INDEX IF NOT EXISTS fact_evidence_fact_idx
    ON fact_evidence (fact_id);
CREATE INDEX IF NOT EXISTS fact_evidence_chunk_idx
    ON fact_evidence (chunk_db_id);
