# Email Thread RAG Chatbot

This project builds a local-first chatbot for a single selected email thread and its attachments. It uses mandatory hybrid retrieval with BM25, dense vectors, RRF, a reranker, deterministic evidence-bound answering, clause-level citation validation, short session memory, OCR fallback for low-text PDFs, and streaming responses in both the API and UI.

## Stage 0 — Recovered baseline (in-memory)

Stage 0 is the consolidated, test-backed Level-2 baseline. It runs entirely on an
**in-memory backend** (in-memory BM25 + `VectorIndex` with a `HashingEncoder`, RRF
fusion, overlap reranker fallback, deterministic answering, and rule-based query
rewrite). It needs **no Postgres, Docker, Gmail account, LLM, or network access**.

Canonical import root: `email_thread_rag`.

```text
email_thread_rag/
├── __init__.py
├── config.py
├── app/       # FastAPI app + schemas
├── rag/       # chunking, indexes, fusion, reranker, retrieval, rewrite, engine
├── scripts/   # dataset slice, ingest, demo
└── ui/        # Gradio UI
```

Install and test (from the repo root):

```bash
python -m pip install -e .          # core, in-memory baseline
python -m pip install -e '.[dev]'   # + pytest
pytest -q                           # 24 tests, no external services
```

Smoke checks:

```bash
python -c "import email_thread_rag"
python -c "from email_thread_rag.rag.engine import RAGEngine"
```

Run the API over the already-ingested local corpus (no rebuild, no network):

```bash
python -m pip install -e '.[serve]'                       # adds uvicorn + gradio
python -m email_thread_rag.scripts.run_demo --skip-build  # serves data/processed
```

Optional extras:
- `.[models]` — `torch`, `transformers`, `sentence-transformers`, `faiss-cpu`,
  `huggingface-hub`. Only needed for the model-backed encoder/reranker/rewriter;
  the baseline lazily falls back to the in-memory path without them.
- `.[serve]` — `uvicorn`, `gradio` for the HTTP API and UI.

**Deferred to Stage 1+ (not implemented):** ParadeDB/Postgres, `pgvector`/`pg_search`,
Gmail OAuth + `users.watch` sync, Medha / LLM generation, HyDE, Self-RAG, GraphRAG /
Apache AGE, entity extraction, RAPTOR summaries, agentic routing. The partial `src/`
tree is a noncanonical Stage-1 scaffold; it is excluded from packaging and imported by
nothing in the working baseline.

## Setup
All commands below assume you are running from the parent folder that contains the `email_thread_rag/` directory. If you are already inside `email_thread_rag/`, run `cd ..` first.

1. Create a Python environment.
2. Install dependencies (editable package; see the Stage 0 section above for extras):

```bash
pip install -e .
```

3. Install Tesseract locally if you are not using Docker.
   On macOS:

```bash
brew install tesseract
```

4. Install `antiword` locally if you want native `.doc` extraction outside Docker.

```bash
brew install antiword
```

5. Copy the environment template:

```bash
cp email_thread_rag/.env.example email_thread_rag/.env
```

6. The checked-in manifest already points at the Enron Archive mailbox dataset and auto-fetches a laptop-sized slice on first ingest.
   `.eml` support remains available for tests and secondary/manual ingestion only.

## Run commands
Build the dataset slice and indexes:

```bash
python -m email_thread_rag.scripts.ingest_corpus
```

Run the API:

```bash
uvicorn email_thread_rag.app.main:app --reload
```

Run the UI:

```bash
python -m email_thread_rag.ui.app
```

Run everything with Docker:

```bash
docker compose up
```

## Environment variables
- `EMAIL_RAG_DATASET_MANIFEST_PATH`: raw pinned manifest path.
- `EMAIL_RAG_RESOLVED_MANIFEST_PATH`: resolved copied/downloaded manifest path.
- `EMAIL_RAG_CHUNK_STORE_PATH`: chunk output file path.
- `EMAIL_RAG_STATS_PATH`: ingest statistics output file path.
- `EMAIL_RAG_API_BASE_URL`: UI target for API calls.
- `EMAIL_RAG_EMBEDDING_MODEL_NAME`: embedding model override.
- `EMAIL_RAG_RERANKER_MODEL_NAME`: reranker override.
- `EMAIL_RAG_REWRITE_MODEL_NAME`: rewrite model override.
- `EMAIL_RAG_ENABLE_CLOUD_REWRITE`: optional cloud rewrite flag.
- `EMAIL_RAG_CLOUD_REWRITE_PROVIDER`: set to `gemini` for Gemini rewrite enhancement.
- `EMAIL_RAG_CLOUD_REWRITE_MODEL`: Gemini model name, such as `gemini-2.5-flash`.
- `EMAIL_RAG_API_PORT`: API port.
- `EMAIL_RAG_UI_PORT`: UI port.
- `HF_TOKEN`: optional Hugging Face token for higher download limits.
- `GEMINI_API_KEY`: optional Gemini API key used only when cloud rewrite is enabled.

Example `.env`:

```dotenv
EMAIL_RAG_DATASET_MANIFEST_PATH=/absolute/path/to/email_thread_rag/data/raw/dataset_manifest.json
EMAIL_RAG_EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMAIL_RAG_RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
EMAIL_RAG_REWRITE_MODEL_NAME=t5-small
EMAIL_RAG_ENABLE_CLOUD_REWRITE=false
EMAIL_RAG_CLOUD_REWRITE_PROVIDER=gemini
EMAIL_RAG_CLOUD_REWRITE_MODEL=gemini-2.5-flash
EMAIL_RAG_API_BASE_URL=http://localhost:8000
HF_TOKEN=
GEMINI_API_KEY=
```

## Which dataset to use
- Use `enronarchive/enron-mail` as the main public dataset.
- Do not use the old Enron WebMail login site.
- The default checked-in manifest fetches mailbox JSON from the Enron Archive GitHub repo and selects a deterministic laptop-sized slice on first ingest.
- The default slice is currently configured for:
  - mailbox: `allen-p`
  - date window: `2000-12-01` to `2001-05-31`
  - target size: `20` threads, about `100+` messages, and about `20-50` attachments
- The main corpus path is mailbox JSON plus selected attachments. `.eml` is only a secondary/manual ingestion path.
- To rebuild the slice explicitly, run:

```bash
python -m email_thread_rag.scripts.build_dataset_slice --force
python -m email_thread_rag.scripts.ingest_corpus --build-slice
```

The ingest script auto-builds the slice if no resolved manifest exists yet.

## How ingestion works
- `build_dataset_slice.py` reads `data/raw/dataset_manifest.json`, downloads the selected mailbox JSON files from `enronarchive/enron-mail`, deterministically selects a small thread slice, downloads only the needed attachments, and writes `data/processed/resolved_dataset_manifest.json`.
- `ingest_corpus.py` auto-builds that slice on first run, then normalizes it into email records, attachment records, and chunk records.
- Enron mailbox JSON is the primary corpus path.
- `.eml` parsing is supported through `rag/parse_eml.py` for tests and secondary/manual ingestion only.
- Attachments are parsed page-aware. PDFs stay per page, and low-text pages trigger OCR.
- Legacy `.doc`, `.xls`, and `.rtf` attachments in the Enron slice are also extracted in the default path.

## Stage 1 — Email-aware parsing & chunking
- **Why fixed-window chunking fails for email.** A generic sliding window mixes the sender's new words with quoted history, signatures, and legal boilerplate, then splits mid-sentence on a token grid. Retrieval then matches on text the sender never wrote, and citations point at the wrong message. `rag/email_segmentation.py` first splits each body into `authored_text` / `quoted_text` / `signature_text` / `disclaimer_text`, and `EmailAwareChunker` (`rag/chunking.py`) chunks only the authored text on paragraph boundaries (~450 token target, ~50 token overlap).
- **Why quoted reply history is excluded.** Copied-in previous emails belong to their own message and are already indexed there. Indexing them again in the reply inflates the corpus with duplicates and lets a query "hit" a message that only quoted the relevant text. Quoted history is retained on `EmailRecord.quoted_text` for audit but never chunked.
- **`text` vs `embed_text`.** `ChunkRecord.text` is the exact authored evidence used for display and citation validation — no injected headers. `ChunkRecord.embed_text` is what BM25 and the vector index consume: a compact `From/To/Cc/Date/Subject/Thread-ID` header block plus that same authored text. `Cc` is built only from real Cc data. `embed_text` defaults to `text` for legacy records, so old fixtures/chunk stores keep working.
- **How source spans enable citations.** Each chunk carries `source_start`/`source_end`, offsets into the normalized authored body, so a citation maps back to the precise span of real text.
- **Deferred.** Gmail sync, LLM/HyDE/Self-RAG, attachment OCR service — remain later stages. Attachment chunks get parent-email provenance in `embed_text` but OCR extraction is unchanged. (ParadeDB/pgvector persistence is now Stage 2, below.)
- **Known stripping limitations.** Bottom-posted quotes without `>` or a recognized marker, signatures without a `-- ` delimiter or known sign-off, and disclaimers outside the marker set are not stripped. When both a sign-off and a `-- ` delimiter exist, the cut is at `-- `, so the sign-off line may survive (conservative "keep authored content" bias). A body reduced to nothing by over-eager signature/disclaimer stripping falls back to the full normalized body — but only when no quote block was detected; a quote-only email (pure forward, no new words) legitimately produces zero chunks rather than resurrecting the quoted history.

## Stage 2 — ParadeDB persistence + hybrid retrieval
- **Why BM25 and dense retrieval solve different problems.** BM25 (`pg_search`) rewards exact terms — sender names, dollar amounts, filenames, reference numbers — that a bag-of-embeddings model tends to blur. Dense retrieval (`pgvector`) rewards paraphrase and topical closeness that lexical matching misses entirely. Running both and fusing beats either alone for the mix of "find this exact number" and "find the email about this" queries email search needs.
- **Why RRF combines ranks, not raw scores.** A BM25 score and a cosine distance live on unrelated scales, so adding them directly is meaningless. Reciprocal Rank Fusion converts each branch to a rank, then combines `weight / (k + rank)` per branch — comparable regardless of how either branch's raw scores are distributed. See `weighted_rrf()` in `rag/fusion.py`, unit-tested against the literal formula in `tests/test_rrf_fusion.py`.
- **`embed_text` searched, `text` cited.** Both the BM25 index and the embedding are built from `embed_text` (headers + authored body), so a query like "Q3 Budget" can match on injected metadata. The row's `text` column — pure authored evidence — is always what's returned as citation evidence; the header block never reaches the reader.
- **Versions.** ParadeDB `paradedb/paradedb:0.24.1` (pinned, not `latest`), bundling `pg_search 0.24.1` and `pgvector 0.8.2` on Postgres 18.4. Bring up: `docker compose up -d db`, then `python -m email_thread_rag.scripts.apply_paradedb_migrations`.
- **Schema.** `email_messages` (one row per email, `UNIQUE (tenant_id, mailbox_id, message_id)`) and `email_chunks` (one row per chunk, `UNIQUE (tenant_id, mailbox_id, chunk_id)`, FK to `email_messages`, `vector(384)` embedding column, BM25 index over `embed_text` + metadata columns, HNSW cosine index over `embedding`). See `rag/paradedb/migrations/0001_init.sql`. Re-ingesting a message upserts it, upserts its current chunks, and deletes chunks the chunker no longer produces — all in one transaction (`ParadeDBRepository.reprocess_message`).
- **memory vs paradedb.** `RAG_BACKEND=memory` (default) needs no database and is what the whole test suite runs against by default. `RAG_BACKEND=paradedb` requires `DATABASE_URL`; an explicit paradedb selection with a missing/unreachable database or missing extensions raises `ParadeDBConfigError` immediately rather than silently continuing on the memory backend.
- **Embedding dimension policy.** Pinned at 384 (`EMBEDDING_DIM`), matching both `sentence-transformers/all-MiniLM-L6-v2` and the deterministic `HashingEncoder` fallback. Changing it requires a new migration plus a full re-embedding backfill — never point a different-dimension encoder at existing rows.
- **Tenant/mailbox filtering.** Every repository write and every `LexicalRetriever`/`DenseRetriever`/`HybridRetriever` query requires an explicit `tenant_id` + `mailbox_id` (`RetrievalFilters`); there is no unscoped search method. Enforced in every SQL `WHERE` clause rather than Postgres RLS in this stage — see "deferred" below.
- **Running the tests.** `python -m pytest -q -m "not integration"` runs the full suite with no database. `docker compose up -d db && python -m email_thread_rag.scripts.apply_paradedb_migrations && python -m pytest -m integration -q` runs the real ParadeDB integration suite (`tests/integration/`) against the pinned container; it creates and drops its own throwaway database per session so it never touches your working data.
- **Deferred.** Postgres row-level security (isolation is enforced in repository/query code instead). LLM contextual prefixes (`context_prefix`/`context_method`/`context_version` columns exist as extension points, unused). Gmail sync, GraphRAG, OCR, HyDE, Self-RAG, reranking models, agentic query planning. (`RAGEngine` wiring to ParadeDB is now Stage 2.5, below.)

## Stage 2.5 — Wiring ParadeDB hybrid retrieval into the engine
- **What changed.** `RAGEngine` no longer hardcodes the in-memory retriever. `rag/backend.py`'s `build_retriever(settings)` picks `memory` (default, unchanged) or `paradedb` based on `RAG_BACKEND`, and `rag/paradedb/retrieval.py`'s `ParadeDBEngineRetriever` adapts `LexicalRetriever`/`DenseRetriever`/`weighted_rrf` onto the exact `RetrievalResult`/`RetrievalHit` shape the engine already speaks — so the existing reranker, answer-builder, and citation-validator run completely unchanged against Postgres-sourced hits.
- **Start ParadeDB:** `docker compose up -d db`. **Run migrations:** `python -m email_thread_rag.scripts.apply_paradedb_migrations`. **Ingest into it:** set `RAG_BACKEND=paradedb` and run `python -m email_thread_rag.scripts.ingest_corpus`, which persists every parsed message/chunk via the same idempotent `reprocess_message` Stage 2 validated, using the same encoder the local vector index already built (no second model load).
- **`RAG_BACKEND=memory` vs `RAG_BACKEND=paradedb`.** Both produce a drop-in retriever for `RAGEngine(settings)` — nothing else about `/ask` changes. `memory` needs no database. `paradedb` requires `DATABASE_URL`, `TENANT_ID`, `MAILBOX_ID`; an explicit `paradedb` selection that's misconfigured raises `ParadeDBConfigError` immediately (`rag/backend.py` never silently falls back to memory).
- **`embed_text` indexed, `text` cited — still true end-to-end.** `ParadeDBEngineRetriever` builds every `ChunkRecord` it returns with `text` as the exact authored evidence and `embed_text` as the searched-but-never-cited field; `doc_id`/`source_path`/`source_type`/`token_count`/`ocr_used`/`attachment_name`/`page_no` (not first-class `email_chunks` columns) round-trip through the existing `metadata` jsonb column rather than a schema change.
- **Deferred.** Gmail sync, GraphRAG, RAPTOR, OCR, LLM generation, reranking models, Postgres RLS — unchanged from Stage 2's list.

## How retrieval works
- Retrieval always starts inside the active thread.
- BM25 returns top 15 lexical hits.
- Dense retrieval returns top 15 vector hits.
- RRF fuses the two lists.
- A cross-encoder reranks the fused top 10.
- The final evidence set is the top 5 reranked hits.
- If `search_outside_thread=true`, the engine retries globally only when the in-thread result fails the explicit support thresholds.

## How rewrite works
- Every `/ask` call rewrites the query before retrieval.
- Primary path: local `t5-small`.
- Fallback path: deterministic rule-based rewrite for pronouns, ellipsis, temporal references, and corrections.
- Optional cloud enhancement path: Gemini, behind `EMAIL_RAG_ENABLE_CLOUD_REWRITE=true`.
- When Gemini is enabled, the app still produces a local rewrite first, then optionally refines that query with Gemini using only the session context and local draft rewrite.
- `rewrite_mode` is returned and logged as `t5`, `rules`, `t5+gemini`, or `rules+gemini`.

## How correction handling works
- Corrections such as `no, I meant the PDF` update `correction_override`.
- The previous target interpretation is invalidated.
- The query is rewritten against the corrected target.
- Retrieval reruns from scratch.
- The final answer uses only corrected evidence.

## How citation validation works
- Answer generation is deterministic and evidence-bound.
- Every factual clause is validated against retrieved evidence.
- Each clause gets a `clause_support_score = 0.6 * token_overlap_f1 + 0.4 * entity_value_match`.
- Unsupported clauses are dropped.
- If fewer than 70% of factual clauses survive validation, the bot abstains or asks one short follow-up.

## API
- `POST /start_session` with `{"thread_id": "..."}`.
- `POST /switch_thread` with `{"session_id": "...", "thread_id": "..."}`.
- `POST /reset_session` with `{"session_id": "..."}`.
- `POST /ask` with `{"session_id": "...", "text": "...", "search_outside_thread": false}`.

`/ask` content negotiation:
- `application/json`: normal structured response.
- `text/event-stream`: SSE with `delta` events followed by one `final` event.

Final `/ask` payload fields:
- `answer`
- `citations`
- `rewrite`
- `rewrite_mode`
- `retrieved`
- `trace_id`
- `outside_thread_used`
- `metrics`

## UI
- Gradio provides:
  - thread selector
  - streaming chat area
  - outside-thread toggle
  - debug panel for rewrite, hits, scores, reranked results, citations, trace ID, and fallback usage

## Tests
Run:

```bash
pytest email_thread_rag/tests
```

Included coverage:
- thread scoping
- attachment page citations
- correction override behavior
- comparison path
- SSE response shape
- citation validator filtering
- OCR fallback
- outside-thread fallback
- rewrite fallback
- end-to-end manifest build + ingest path

## Demo

https://github.com/Aditya2600/Email-Thread-RAG-Chatbot-Nexux-Ocean/raw/main/Screen_Recording/Screen%20Recording%202026-03-15%20at%2010.34.27%E2%80%AFPM.mov

> The recording demonstrates: thread selection, streaming chat, pronoun/ellipsis follow-up handling, attachment citations, correction override, and graceful abstention on out-of-scope questions.

## Sample questions
For one clean attachment-bearing thread in the default Allen slice, use:
- Chosen thread: `allen-p:bid solicitation`
- Email message IDs in the selected slice: `036c4137cd99dc2c513e96916a32f624` and `c90ecfb887b787e72b8ce2131cab5fab`
- Attachment filename: `mobidltr.501.doc`

Sample questions with expected citation shapes:
- `What does the email say is due and when?`
  Expected citation: `[msg: 036c4137cd99dc2c513e96916a32f624]` or `[msg: c90ecfb887b787e72b8ce2131cab5fab]`
- `What company is requesting bids in the attachment?`
  Expected citation: `[msg: 036c4137cd99dc2c513e96916a32f624, page: 1]` or `[msg: c90ecfb887b787e72b8ce2131cab5fab, page: 1]`
- `Which territories are mentioned in the attachment?`
  Expected citation: `[msg: 036c4137cd99dc2c513e96916a32f624, page: 1]` or `[msg: c90ecfb887b787e72b8ce2131cab5fab, page: 1]`
- `And when is the deadline?`
  Expected behavior: follow-up rewrite resolves the prior attachment/email context and returns a cited date
- `No, I meant the attachment. What month are the supply requests for?`
  Expected behavior: correction override forces new retrieval and returns an attachment-grounded answer with `[msg: ..., page: 1]`
- `What hotel was booked for the meeting?`
  Expected behavior: graceful abstention or a concise missing-evidence answer because that thread does not discuss hotel bookings

## Known limitations
- The checked-in manifest points at the real Enron Archive GitHub repo, but this sandbox cannot verify a live network fetch during implementation.
- If you want a stricter pin than `revision=main`, set a specific dataset revision in `data/raw/dataset_manifest.json`.
- If local model downloads are unavailable, the project falls back to deterministic rewrite and lightweight test encoders/rerankers; the production path still targets the configured open-source models.
- The real Enron slice contains many legacy Office attachments; native `.doc` extraction outside Docker depends on `antiword`.
