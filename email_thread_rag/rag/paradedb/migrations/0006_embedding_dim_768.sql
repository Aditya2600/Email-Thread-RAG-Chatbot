-- Stage 2 follow-up: widen the embedding column 384 -> 768 for
-- BAAI/bge-base-en-v1.5 (0001 pinned 384 for the bge-small/MiniLM era).
--
-- DESTRUCTIVE: every existing embedding is discarded. A 384-dim vector has no
-- meaningful projection into a 768-dim model's space, so these cannot be
-- converted -- only recomputed. Dense retrieval returns nothing for a chunk
-- until it is re-embedded, because the HNSW index simply excludes NULL rows
-- (lexical/BM25 retrieval is unaffected and keeps working throughout).
--
-- After applying, re-embed:
--   python -m email_thread_rag.scripts.ingest_corpus
-- and, if contextual retrieval is enabled, re-run the context worker.

-- The HNSW index is bound to the column's dimension and blocks the ALTER.
DROP INDEX IF EXISTS email_chunks_embedding_hnsw_idx;

-- USING NULL rather than a cast: pgvector cannot reinterpret 384-dim data as
-- 768-dim, and a silent partial conversion would be worse than an empty column.
ALTER TABLE email_chunks
    ALTER COLUMN embedding TYPE vector(768) USING NULL;

-- embedding_model/_version described vectors that no longer exist. Clearing
-- them keeps the columns honest about what is actually stored (NULL = never
-- embedded) so a partially re-embedded table is still readable.
UPDATE email_chunks
SET embedding_model = NULL, embedding_version = NULL
WHERE embedding_model IS NOT NULL OR embedding_version IS NOT NULL;

CREATE INDEX IF NOT EXISTS email_chunks_embedding_hnsw_idx ON email_chunks
USING hnsw (embedding vector_cosine_ops);
