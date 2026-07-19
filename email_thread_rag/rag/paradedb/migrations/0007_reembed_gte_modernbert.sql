-- Embedder swap: BAAI/bge-base-en-v1.5 -> Alibaba-NLP/gte-modernbert-base.
--
-- The dimension does NOT change: both models emit 768, so the vector(768)
-- column from 0006 is left exactly as it is and there is no ALTER, no index
-- drop, and no new dimension migration. Same width, however, is not
-- compatibility -- the two models have unrelated vector spaces, so a BGE
-- vector scored against a GTE query embedding produces a plausible-looking
-- cosine number that means nothing. Silently wrong retrieval is worse than no
-- retrieval, so every stale vector is cleared here.
--
-- DESTRUCTIVE: every existing embedding is discarded. Dense retrieval returns
-- nothing for a chunk until it is re-embedded (the HNSW index simply excludes
-- NULL rows); lexical/BM25 retrieval is unaffected and keeps working throughout.
--
-- This is a migration rather than a re-ingest because re-ingest does not cover
-- every row: ingest_corpus only re-embeds messages present in the current
-- corpus, so Gmail-synced rows and chunks no longer in the corpus would keep
-- their BGE vectors. Clearing at the table level is the only way to guarantee
-- the two models' vectors are never mixed.
--
-- After applying, re-embed:
--   python -m email_thread_rag.scripts.ingest_corpus
-- and, if contextual retrieval is enabled, re-run the context worker.

-- embedding_model/_version described vectors that no longer exist. Clearing
-- them with the vector keeps the columns honest about what is actually stored
-- (NULL = never embedded) so a partially re-embedded table is still readable.
-- Note context/repository.py writes these with COALESCE(%s, embedding_model),
-- which never clears -- so this statement is the only thing that resets them.
UPDATE email_chunks
SET embedding = NULL, embedding_model = NULL, embedding_version = NULL
WHERE embedding IS NOT NULL
   OR embedding_model IS NOT NULL
   OR embedding_version IS NOT NULL;
