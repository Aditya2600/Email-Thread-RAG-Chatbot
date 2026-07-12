Plan: Upgrade Inbox-Copilot to a Level-3 Email-Thread RAG
Context
inbox-copilot is an email-thread RAG chatbot (Enron slice + attachments). Today it is a Level-2 system: hybrid BM25 + dense retrieval, RRF fusion, cross-encoder reranking, and a deterministic citation validator — but answers are template/rule-based (no LLM), vectors live in FAISS pickle files, there is no HyDE and no self-correction.

The goal is Level 3: persistent pgvector storage, HyDE, header-injected chunks for sender/date-aware search, a Self-RAG–inspired verification layer, and real LLM generation via the remote Medha vLLM service (Google Gemma 26B-A4B AWQ-4bit, OpenAI-compatible, on an A100). Endpoint verified working: http://164.52.192.196:8001/v1 (model Medha, 14336 ctx, streaming).

Current repo is mid-refactor and not runnable
The working tree has two parallel trees. src/ is a half-done restructure that broke the package: the email_thread_rag import root has no packaging (empty pyproject.toml), src/ingest/chunker.py is truncated, src/retrieval/hyde.py + src/rag/grounding.py + src/rag/citations.py + src/db/* are empty stubs, and src/app/schemas.py was rewritten into names that diverge from what the engine and tests import.

Key recovery insight: HEAD (13b1d95) still contains a complete, coherent, test-backed generation at the repo root — app/, rag/, scripts/, ui/, config.py — whose module names (rag.chunking, rag.answer, rag.vector_index, rag.bm25_index, rag.fusion, rag.reranker, rag.retrieval, rag.rewrite, rag.engine) and schema names (ChunkRecord, RetrievalHit, RetrievalMetrics, TraceRecord, ClauseValidation, AskResponse) exactly match the imports in tests/ and rag/engine.py. The 19 tests in tests/ are the executable spec.

So we rebuild the baseline from HEAD's coherent modules (not from the broken src/ stubs), package it properly, get tests green, then layer the five features on top.

Decisions (confirmed with user)
Scope: full repair to runnable + all features.
Vector + BM25 store: single ParadeDB Postgres container (paradedb/paradedb) — bundles pgvector (dense) and pg_search (real BM25). One store, fused with RRF.
Postgres: runs as a docker-compose service.
Answers: Medha grounded generation with enforced inline citations; deterministic builder kept as abstain fallback.
Phase 0 — Consolidate to one installable package (make it run)
Package layout. Create email_thread_rag/ as the single canonical package and add a real pyproject.toml (setuptools; deps from requirements.txt + sqlalchemy[asyncio], asyncpg, pgvector, httpx, sentence-transformers, spacy). Install editable: pip install -e ..
Recover coherent modules from HEAD, not the broken src/ stubs: app/ (schemas.py, main.py, sessions.py, streaming.py), rag/ (chunking.py, answer.py, vector_index.py, bm25_index.py, fusion.py, reranker.py, retrieval.py, rewrite.py, engine.py, citation_validator.py, corpus.py, memory.py, parse_*.py, utils.py, enron_archive.py, threading.py), config.py, scripts/ (build_dataset_slice.py, ingest_corpus.py, run_demo.py), ui/app.py. Source via git show HEAD:<path>.
Salvage selectively from src/ only where clearly better and working (e.g. UI polish in src/ui/gradio_app.py); otherwise prefer HEAD. Delete the broken duplicate src/ and the stale root dupes once recovered, ending the two-tree confusion.
Green the baseline: pytest passes against the recovered Level-2 contract before adding features. The tests/conftest.py fixtures (memory backend: HashingEncoder + in-memory HybridRetriever) must keep working without a live DB — this constrains every later change.
Gate: do not start Phase 1+ until pip install -e . && pytest is green.

Phase 1 — pgvector + pg_search store (ParadeDB)
docker-compose: add a db service paradedb/paradedb:latest with POSTGRES_*, a named volume, and a healthcheck; wire DATABASE_URL into api/ui services and into Settings. Update infra/docker-compose.prod.yml similarly. (Also fix the existing compose, which still references email_thread_rag.scripts... — now correct.)
email_thread_rag/db/ (new): engine.py (async SQLAlchemy engine / asyncpg pool from DATABASE_URL), schema.sql + a migration that runs CREATE EXTENSION IF NOT EXISTS vector; and CREATE EXTENSION IF NOT EXISTS pg_search;, creates the chunks table, an HNSW index (vector_cosine_ops) on embedding, and a bm25 index over embed_text (+ sender, subject) via pg_search with key_field='chunk_id'.
chunks table: chunk_id PK, doc_id, thread_id, message_id, kind, sender, recipients, date timestamptz, subject, attachment_name, page_no, text, embed_text, embedding vector(768), token_count, ocr_used, source_path, source_type, metadata jsonb.
Embeddings: computed locally with sentence-transformers BAAI/bge-base-en-v1.5 (768-dim, matches config.embed_model). Medha is chat-only — no embeddings endpoint.
Pluggable retrieval backend. Extend rag/retrieval.py:HybridRetriever with backend="memory"|"postgres":
memory (unchanged): bm25_index (rank-bm25) + VectorIndex/HashingEncoder — keeps conftest.py and unit tests DB-free.
postgres (new, production): dense via pgvector (embedding <=> :qvec cosine, thread-filtered in SQL) + lexical via pg_search (embed_text @@@ :q, paradedb.score(chunk_id), thread-filtered) → RRF-fuse in app (reuse rag/fusion.py) → cross-encoder rerank (reuse rag/reranker.py).
Ingest: scripts/ingest_corpus.py / rag/corpus.py:ingest_corpus upsert rows (with embedding + embed_text) into Postgres by chunk_id, in addition to the existing chunks.jsonl (kept for the memory/test path).
Phase 2 — Header-injected chunks (sender/date-aware)
In rag/chunking.py: build embed_text = header + "\n\n" + body and store both text (raw body, for display/citation) and embed_text (embedded and BM25-indexed). Header = From / To / Cc / Date / Subject / Thread-ID. Fix the cc=to bug seen in the src draft (src/ingest/chunker.py:207 copies email.to into Cc).
Attachments: header = parent email's From/Date/Subject + Attachment: <filename> (page N) then page text. Token-window chunking with overlap (reuse rag/utils.sliding_text_chunks, count_tokens).
Add embed_text field to ChunkRecord (additive, defaults to text so existing tests/fixtures stay valid).
Optional precision path: the query planner extracts structured filters (sender, date range) → SQL WHERE predicates; header injection covers the soft/semantic path.
Phase 3 — Medha LLM client + grounded answers
email_thread_rag/llm/medha.py (new): async OpenAI-compatible client (httpx) for /v1/chat/completions with complete() (non-stream) and stream() (SSE). Config: medha_base_url (http://164.52.192.196:8001/v1), medha_api_key (env MEDHA_API_KEY, default to the provided token), medha_model="Medha", max_tokens, temperature (0.2), timeout, ctx cap 14336. Centralized retry/timeout/error handling.
rag/rewrite.py: route query rewrite through Medha (drop T5/Gemini); keep RewriteResult shape and the rule-based fallback (tests use RuleOnlyRewriter).
rag/answer.py: add a grounded LLM path — prompt Medha with the packed, citation-tagged context to produce an answer with inline [msg:<id>] / [msg:<id>, page:N] citations; stream tokens through existing app/streaming.py. Keep the deterministic AnswerBuilder as fallback/abstain. Respect token_budget against the 14336 ctx (header injection inflates context).
Config: replace ollama_* knobs with medha_*; update .env.example and compose env.
Phase 4 — Self-RAG–inspired verification layer
Implement email_thread_rag/rag/grounding.py (currently empty) as reflection on top of the existing CitationValidator. Engine flow becomes: plan → retrieve → ISREL filter → generate → ISSUP/validate → ISUSE → (bounded) correct → finalize.

ISREL (relevance): after retrieval, a cheap batched Medha call judges each top chunk's relevance; drop irrelevant chunks before generation and feed the signal into the existing outside-thread fallback decision.
ISSUP (support): keep the deterministic CitationValidator (token-F1 + entity match, 0.70 coverage gate) as the fast gate, and add a Medha verifier labeling each answer sentence Supported/Partial/Unsupported against its cited chunks; strip or flag unsupported sentences.
ISUSE (usefulness): Medha rates 1–5 whether the answer addresses the question.
Correction loop (max 1–2 iters): if coverage/support below threshold → (a) trigger the existing outside-thread re-retrieval (engine._outside_thread_reason thresholds in config.RetrievalThresholds), or (b) regenerate with stricter "state only what citations support" instruction, else abstain with the existing message.
Surface reflection scores in RetrievalMetrics/TraceRecord and the Gradio debug panel.
Phase 5 — Eval harness (substantiates "Level 3")
evals/ is empty. Add a small labeled question set over the Enron slice + scripts/run_eval.py reporting: retrieval hit@k, citation coverage, support/faithfulness (Medha-as-judge), and abstain correctness. Lets us show before/after numbers for the upgrade.
Critical files
Area	Files
Packaging	pyproject.toml (new), recover email_thread_rag/** from HEAD, delete broken src/
pgvector/pg_search	email_thread_rag/db/{engine,schema.sql,migrations} (new), docker-compose.yml, infra/docker-compose.prod.yml
Retrieval	rag/retrieval.py (pluggable backend), reuse rag/{fusion,reranker,bm25_index,vector_index}.py
HyDE	rag/hyde.py (new), wired into rag/retrieval.py
Chunking	rag/chunking.py, app/schemas.py (ChunkRecord.embed_text)
Medha LLM	email_thread_rag/llm/medha.py (new), rag/rewrite.py, rag/answer.py, config.py, .env.example
Self-RAG	rag/grounding.py (new), rag/citation_validator.py, rag/engine.py, ui/app.py
Evals	evals/*, scripts/run_eval.py (new)
Reused (do not rewrite)
RRF: rag/fusion.py · Cross-encoder rerank: rag/reranker.py · BM25 (memory path): rag/bm25_index.py · Deterministic grounding: rag/citation_validator.py · Chunk tokenization: rag/utils.py · Memory: rag/memory.py · Streaming SSE: app/streaming.py · Outside-thread thresholds: config.RetrievalThresholds + engine._outside_thread_reason.
HyDE detail
rag/hyde.py:generate_hypothetical(query, *, llm, thread_context) -> str: Medha drafts a short hypothetical email/answer passage → embed with bge → use as (or blend 0.5/0.5 with) the raw query embedding for the dense branch. Config enable_hyde, hyde_max_tokens; skip for metadata_lookup intent; cache per query.

Verification
Baseline gate: pip install -e . && pytest green on the recovered Level-2 contract (memory backend, no DB).
DB up: docker compose up -d db; confirm vector + pg_search extensions and chunks indexes created.
Ingest: run scripts/ingest_corpus.py → rows present in Postgres with non-null embedding and header-injected embed_text.
Retrieval (postgres backend): a sender/date query (e.g. "what did Bob send in January about the budget") returns thread-correct chunks; verify both BM25 (paradedb.score) and pgvector contribute via the trace's fused_ranking.
HyDE on/off: toggle enable_hyde; confirm dense hits and final answer change and HyDE text appears in the trace.
Medha generation: /ask returns an LLM answer with inline [msg:...] citations; streaming works through app/streaming.py; curl health on /v1/models as a smoke check.
Self-RAG: craft an unsupported-claim query → verify ISSUP strips/flags it, the correction loop fires (re-retrieve or stricter regen), and ISREL/ISSUP/ISUSE scores show in the Gradio debug panel and TraceRecord.
Evals: scripts/run_eval.py prints hit@k, citation coverage, faithfulness, abstain-correctness over the labeled set.
Full pytest green again after features (memory path unaffected; add DB-backed tests guarded by DATABASE_URL).