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
python -m pip install -e '.[serve]'                       # adds uvicorn
python -m email_thread_rag.scripts.run_demo --skip-build  # serves data/processed
```

Optional extras:
- `.[models]` — `torch`, `sentence-transformers`, `faiss-cpu`,
  `huggingface-hub`. Only needed for the model-backed encoder/reranker;
  the baseline lazily falls back to the in-memory path without them.
- `.[serve]` — `uvicorn` for the HTTP API.

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
cd frontend && npm install && npm run dev
```

Run everything with Docker:

```bash
docker compose -f infra/docker-compose.yml up
```

## Environment variables
- `EMAIL_RAG_DATASET_MANIFEST_PATH`: raw pinned manifest path.
- `EMAIL_RAG_RESOLVED_MANIFEST_PATH`: resolved copied/downloaded manifest path.
- `EMAIL_RAG_CHUNK_STORE_PATH`: chunk output file path.
- `EMAIL_RAG_STATS_PATH`: ingest statistics output file path.
- `EMAIL_RAG_API_BASE_URL`: UI target for API calls.
- `EMAIL_RAG_EMBEDDING_MODEL_NAME`: embedding model override.
- `EMAIL_RAG_RERANKER_MODEL_NAME`: reranker override.
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
EMAIL_RAG_EMBEDDING_MODEL_NAME=Alibaba-NLP/gte-modernbert-base
EMAIL_RAG_RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
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
- **Versions.** ParadeDB `paradedb/paradedb:0.24.1` (pinned, not `latest`), bundling `pg_search 0.24.1` and `pgvector 0.8.2` on Postgres 18.4. Bring up: `docker compose -f infra/docker-compose.yml up -d db`, then `python -m email_thread_rag.scripts.apply_paradedb_migrations`.
- **Schema.** `email_messages` (one row per email, `UNIQUE (tenant_id, mailbox_id, message_id)`) and `email_chunks` (one row per chunk, `UNIQUE (tenant_id, mailbox_id, chunk_id)`, FK to `email_messages`, `vector(768)` embedding column, BM25 index over `embed_text` + metadata columns, HNSW cosine index over `embedding`). See `rag/paradedb/migrations/0001_init.sql` (widened to 768 by `0006_embedding_dim_768.sql`; embeddings cleared for the GTE swap by `0007_reembed_gte_modernbert.sql`). Re-ingesting a message upserts it, upserts its current chunks, and deletes chunks the chunker no longer produces — all in one transaction (`ParadeDBRepository.reprocess_message`).
- **memory vs paradedb.** `RAG_BACKEND=memory` (default) needs no database and is what the whole test suite runs against by default. `RAG_BACKEND=paradedb` requires `DATABASE_URL`; an explicit paradedb selection with a missing/unreachable database or missing extensions raises `ParadeDBConfigError` immediately rather than silently continuing on the memory backend.
- **Embedding dimension policy.** Pinned at 768 (`EMBEDDING_DIM`), matching both `Alibaba-NLP/gte-modernbert-base` and the deterministic `HashingEncoder` fallback. Changing it requires a new migration plus a full re-embedding backfill — never point a different-dimension encoder at existing rows. Changing the *model* at a fixed dimension needs the same full re-embed: equal width is not a compatible vector space, and mixing two models' embeddings in one column yields cosine scores that look valid and mean nothing. See `0007_reembed_gte_modernbert.sql`.
- **Tenant/mailbox filtering.** Every repository write and every `LexicalRetriever`/`DenseRetriever`/`HybridRetriever` query requires an explicit `tenant_id` + `mailbox_id` (`RetrievalFilters`); there is no unscoped search method. Enforced in every SQL `WHERE` clause rather than Postgres RLS in this stage — see "deferred" below.
- **Running the tests.** `python -m pytest -q -m "not integration"` runs the full suite with no database. `docker compose -f infra/docker-compose.yml up -d db && python -m email_thread_rag.scripts.apply_paradedb_migrations && python -m pytest -m integration -q` runs the real ParadeDB integration suite (`tests/integration/`) against the pinned container; it creates and drops its own throwaway database per session so it never touches your working data.
- **Deferred.** Postgres row-level security (isolation is enforced in repository/query code instead). LLM contextual prefixes (`context_prefix`/`context_method`/`context_version` columns exist as extension points, unused). Gmail sync, GraphRAG, OCR, HyDE, Self-RAG, reranking models, agentic query planning. (`RAGEngine` wiring to ParadeDB is now Stage 2.5, below.)

## Stage 2.5 — Wiring ParadeDB hybrid retrieval into the engine
- **What changed.** `RAGEngine` no longer hardcodes the in-memory retriever. `rag/backend.py`'s `build_retriever(settings)` picks `memory` (default, unchanged) or `paradedb` based on `RAG_BACKEND`, and `rag/paradedb/retrieval.py`'s `ParadeDBEngineRetriever` adapts `LexicalRetriever`/`DenseRetriever`/`weighted_rrf` onto the exact `RetrievalResult`/`RetrievalHit` shape the engine already speaks — so the existing reranker, answer-builder, and citation-validator run completely unchanged against Postgres-sourced hits.
- **Start ParadeDB:** `docker compose -f infra/docker-compose.yml up -d db`. **Run migrations:** `python -m email_thread_rag.scripts.apply_paradedb_migrations`. **Ingest into it:** set `RAG_BACKEND=paradedb` and run `python -m email_thread_rag.scripts.ingest_corpus`, which persists every parsed message/chunk via the same idempotent `reprocess_message` Stage 2 validated, using the same encoder the local vector index already built (no second model load).
- **`RAG_BACKEND=memory` vs `RAG_BACKEND=paradedb`.** Both produce a drop-in retriever for `RAGEngine(settings)` — nothing else about `/ask` changes. `memory` needs no database. `paradedb` requires `DATABASE_URL`, `TENANT_ID`, `MAILBOX_ID`; an explicit `paradedb` selection that's misconfigured raises `ParadeDBConfigError` immediately (`rag/backend.py` never silently falls back to memory).
- **`embed_text` indexed, `text` cited — still true end-to-end.** `ParadeDBEngineRetriever` builds every `ChunkRecord` it returns with `text` as the exact authored evidence and `embed_text` as the searched-but-never-cited field; `doc_id`/`source_path`/`source_type`/`token_count`/`ocr_used`/`attachment_name`/`page_no` (not first-class `email_chunks` columns) round-trip through the existing `metadata` jsonb column rather than a schema change.
- **Deferred.** Gmail sync, GraphRAG, RAPTOR, OCR, LLM generation, reranking models, Postgres RLS — unchanged from Stage 2's list.

## Stage 3 — Gmail OAuth, Pub/Sub push, and durable delta sync

Real mail flows in along this path, and each arrow is a separate, restartable step:

`Gmail OAuth` → `users.watch` → `Pub/Sub push webhook` → `durable sync job (Postgres)` → `background worker` → `history.list` → `messages.get` → **Stage-1 parsing/chunking** → **Stage-2.5 ParadeDB persistence**

Nothing about Stage 1 or 2.5 changed to make this work: `gmail/message.py` converts a Gmail message into the same canonical `EmailRecord` the `.eml` parser produces, and from there the existing segmenter, chunker, and `reprocess_message` handle it with no Gmail-specific branches.

### Setup expectations

Stage 3 is entirely optional. `RAG_BACKEND=memory` still runs with no Gmail packages, no credentials, no Docker, and no network — that boundary is enforced by tests (`tests/test_gmail_independence.py`), not just convention.

1. `pip install -e ".[gmail]"` (adds `cryptography` for token encryption and `google-auth` for verifying Pub/Sub pushes; Gmail REST itself goes over `httpx`, already a core dependency).
2. In Google Cloud: create an OAuth client (web application) and a Pub/Sub topic + **push** subscription pointing at `https://<your-host>/gmail/pubsub/push`. Grant `gmail-api-push@system.gserviceaccount.com` the Publisher role on the topic, and enable an OIDC service account on the push subscription — the webhook rejects any push it cannot verify.
3. Generate a token encryption key: `python -c "import base64,os;print(base64.b64encode(os.urandom(32)).decode())"`.
4. `docker compose -f infra/docker-compose.yml up -d db && python -m email_thread_rag.scripts.apply_paradedb_migrations` (applies `0002_gmail.sql`).
5. Connect a mailbox: `GET /gmail/oauth/start?tenant_id=...&mailbox_id=...` returns a consent URL; Google redirects back to `/gmail/oauth/callback`, which stores the encrypted refresh token, calls `users.watch`, and persists the watch state. The Gmail routes only mount when `GMAIL_PUBSUB_SUBSCRIPTION` and `DATABASE_URL` are set.
6. Run the worker: `python -m email_thread_rag.gmail.worker` (or `--once` under cron).

Scope requested is exactly `gmail.readonly` — this stage reads mail and never sends or modifies it.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `GMAIL_CLIENT_ID` / `GMAIL_CLIENT_SECRET` | OAuth client credentials. |
| `GMAIL_REDIRECT_URI` | Must match the OAuth client exactly, e.g. `https://app.example.com/gmail/oauth/callback`. |
| `GMAIL_PUBSUB_TOPIC` | Topic passed to `users.watch`, e.g. `projects/demo/topics/gmail-sync`. |
| `GMAIL_PUBSUB_SUBSCRIPTION` | Expected push subscription; a push naming anything else is rejected. |
| `GMAIL_PUBSUB_AUDIENCE` | Expected OIDC audience (defaults to `EMAIL_RAG_API_BASE_URL`). |
| `GMAIL_PUBSUB_SERVICE_ACCOUNT` | Optional: pin the pushing service account's email. |
| `GMAIL_TOKEN_ENCRYPTION_KEY` | Base64 32-byte AES-256 key for refresh tokens at rest. |
| `GMAIL_TOKEN_KEY_ID` | Label recorded next to each ciphertext, so a future key rotation can tell which key encrypted what. Defaults to `local`. |

### Cursor semantics — the rule that keeps sync correct

`gmail_mailboxes.last_committed_history_id` is the only cursor, and it advances in exactly one place (`sync.run_sync`), as the **last** step of a run, after every message in that window is persisted.

- **A failure never advances it.** A Gmail error, a ParadeDB error, or a worker crash all leave the old cursor intact; the retry re-reads the same window. That is safe because replaying a window is idempotent — every message upserts by ID, so re-ingesting produces the same rows rather than duplicates.
- **It never rewinds.** Commits use a numeric `GREATEST`, so an out-of-order or replayed job can only move it forward.
- **History IDs are numeric, never text.** They live in `numeric(20,0)` columns and are compared with numeric operators. Compared as strings, `'10'` sorts below `'9'` — which would silently rewind a cursor and skip mail.
- **A watch renewal does not touch it.** `activate_watch` seeds the cursor only when it is NULL, so re-watching cannot skip past a window the worker has not synced.

### Watch renewal

A Gmail watch expires after 7 days, and an expired watch fails silently — mail simply stops arriving. Run `python -m email_thread_rag.gmail.worker --renew-watches` at least daily (cron/systemd); it re-calls `users.watch` for anything expiring within 24 hours. One mailbox failing to renew does not stop the others. Disconnecting calls `users.stop` and discards the stored refresh token — and discards it even if `users.stop` fails, since dropping the credential is what actually ends our access.

### History expiry (404) → full sync

Gmail only keeps history for a limited window. If `history.list` returns 404, the cursor is too old:

1. Mark the job and mailbox `needs_full_sync` **first**, so a crash mid-rebuild still retries as a full sync — and never advance the old cursor.
2. Take a fresh history checkpoint from `users.getProfile` **before** scanning.
3. Paginate `messages.list` + `messages.get` through the canonical ingestion path.
4. Replay history since that checkpoint, then commit the new cursor.

Step 2's ordering is the point: taking the checkpoint *after* the scan would lose every change that happened while the scan was running.

### Job queue and delivery semantics

The `gmail_sync_jobs` table **is** the queue — no Celery, Redis, or Kafka in this stage. Workers claim with `FOR UPDATE SKIP LOCKED` plus a lease, so several can run at once and a dead worker's job is reclaimed when its lease expires.

- The webhook does no Gmail I/O and no indexing. It verifies the push, then commits a job — and returns 200 only after that commit, so an ack always means the work is durable.
- Pub/Sub redelivers; `gmail_pubsub_messages` dedups by message ID in the same transaction, so a redelivery is a no-op.
- At most one pending job per mailbox (a partial unique index). A second notification raises the pending job's `requested_history_id` to the numeric max instead of queueing duplicate work.

### Deleted mail

A `messageDeleted` history record removes the message and its chunks outright (not a tombstone flag). Tombstoned rows would stay in the BM25 and HNSW indexes and every query would have to remember to exclude them; deleting is what actually makes them unretrievable. `tests/integration/test_gmail_paradedb.py` asserts a deleted message disappears from hybrid retrieval.

### Local test workflow (fakes only)

Every Stage-3 test runs against `FakeGmailClient` / `FakePubSubVerifier` (`gmail/fakes.py`). No credentials, no network, no real OAuth flow:

```bash
# Fast suite: no database, no Gmail config, no network.
python -m pytest -q -m "not integration"

# Against the real ParadeDB container (Gmail still faked).
docker compose -f infra/docker-compose.yml up -d db
export DATABASE_URL=postgresql://email_rag:email_rag_local_dev@localhost:5433/email_rag
python -m email_thread_rag.scripts.apply_paradedb_migrations
python -m pytest -q -m integration
```

`InMemorySyncStore` and `PostgresSyncStore` are held to the same contract (`tests/gmail_store_contract.py`, run against both), so the store the fast tests use cannot drift from the real one. A conftest guard fails any Gmail unit test that opens a socket.

- **Deferred (not in this stage).** GraphRAG/RAPTOR, OCR of Gmail attachments (synced messages ingest body text only), a Gmail UI, query routing, Postgres RLS, and any `src/` cleanup. LLM contextualization arrived in Stage 4, below.

## Stage 4 — Asynchronous LLM contextualization

Chunks are persisted first and contextualized later. A background worker asks a small LLM what a chunk *concerns*, puts that sentence in front of the chunk's retrieval text, and re-embeds it — so a chunk becomes findable by concepts its author never spelled out.

**The invariant, first:** `text`, `source_start`, and `source_end` never change. The model's words only ever enter `embed_text`, which is what the indexes search. What gets displayed and cited is still the exact authored evidence. A prefix cannot become a citation.

```
embed_text = compact headers + optional context_prefix + exact text
```

That assembly lives in exactly one function — `build_embed_text` in `rag/email_segmentation.py`, the same one Stage 1 already used. With no prefix it returns byte-identical Stage-1 output, which is what lets a chunk be contextualized without its deterministic form drifting.

### Setup

Off by default. Nothing below is read, and no LLM client is imported, unless you enable it. Requires `RAG_BACKEND=paradedb`, since contextualization rewrites persisted rows.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONTEXT_ENABLED` | `false` | Master switch. Off = ingestion queues nothing. |
| `CONTEXT_BASE_URL` | — | Any OpenAI-compatible endpoint. `MEDHA_BASE_URL` is an alias. |
| `CONTEXT_MODEL` | — | Model name sent in the request. `MEDHA_MODEL` is an alias. |
| `CONTEXT_API_KEY` | — | Sent as `Authorization: Bearer`. Environment only. `MEDHA_API_KEY` is an alias. |
| `CONTEXT_TIMEOUT_SECONDS` | `30` | Per-request timeout. |
| `CONTEXT_MAX_TOKENS` | `96` | 80-token prefix budget + JSON wrapper. |
| `CONTEXT_PROMPT_VERSION` | built-in | Part of the fingerprint; bumping re-contextualizes everything. |

```bash
export RAG_BACKEND=paradedb CONTEXT_ENABLED=true
export CONTEXT_BASE_URL=http://164.52.192.196:8002/v1 CONTEXT_MODEL=Medha CONTEXT_API_KEY=...
python -m email_thread_rag.scripts.apply_paradedb_migrations   # adds 0003_context.sql

python -m email_thread_rag.context.worker --once               # drain the queue
python -m email_thread_rag.context.backfill --tenant-id acme --mailbox-id inbox
```

No model is downloaded and Ollama is not a dependency: any OpenAI-compatible `/chat/completions` endpoint works by configuration alone.

### The prompt contract

The model is asked for one or two factual sentences, ≤80 tokens, saying what the chunk concerns — using only the supplied subject/thread/parent/text. It must not answer questions, invent facts, add citations, or echo instructions.

Email content is **untrusted data**. It arrives inside explicit `<email_chunk>` delimiters and the system prompt says so, so a body reading *"ignore previous instructions"* is data, not a directive. But a prompt is a request, not a guarantee — so nothing returned is trusted either. `validate_output` independently re-checks every constraint (strict JSON, token budget, sentence count, no citation/link markers). Anything that fails becomes a **deterministic fallback**: the chunk keeps its exact Stage-1 `embed_text`, stays fully retrievable, and simply gains no prefix.

Prompts and raw model responses are never stored. Only the validated prefix, the model id, and a rule-name error are persisted.

### Cursor and job semantics

The database table is the queue — no Redis, Celery, or Kafka. Jobs follow Stage 3's lease model: `pending → running → done | failed`, claimed with `FOR UPDATE SKIP LOCKED`.

A job's identity is a **fingerprint** over the chunk's clean text, its selected metadata, the locally-available parent id/subject, the prompt version, and the model id:

- **No duplicate work.** Re-persisting an unchanged message produces the same fingerprint, collides on a unique index, and queues nothing.
- **Stale jobs cannot overwrite newer state.** The worker recomputes the fingerprint from the chunk's *current* row inside the commit transaction. If the chunk was re-ingested while the LLM was thinking, the result is discarded — the prefix describes text that no longer exists, and a newer job already covers the new text.
- **Change anything covered and it re-contextualizes.** Editing the body, or bumping `CONTEXT_PROMPT_VERSION`, mints a new fingerprint and new work.

No DB transaction is held open across an LLM call: the worker uses autocommit plus explicit transaction blocks, and claims, calls, and commits in three separate steps.

**Retry vs. fallback** is a deliberate split. A provider outage (timeout, HTTP 5xx) is transient, so the job returns to `pending` and retries, writing nothing — an outage must not degrade a chunk. Invalid model output is deterministic: at `temperature=0` a retry produces the identical bad output, so retrying is a guaranteed-useless loop. Those fall back instead.

### Backfill

`python -m email_thread_rag.context.backfill --tenant-id X --mailbox-id Y` queues pre-existing chunks. Idempotent and resumable by construction, not by bookkeeping: it pages forward by chunk id, and already-contextualized chunks are excluded by the scan itself. Run it twice and the second run queues nothing. It only enqueues — the worker does the calling, so backfilling a large mailbox cannot take down the model endpoint.

### Local test workflow (fakes only)

Every Stage-4 test uses `FakeContextProvider` (`context/fakes.py`). No model download, no remote call; a conftest guard fails any context unit test that opens a socket.

```bash
python -m pytest -q -m "not integration"    # contextualization disabled, no DB/model config
python -m pytest -q -m integration          # ParadeDB container, fake contextualizer
```

`InMemoryContextJobStore` and `PostgresContextJobStore` are held to one shared contract (`tests/context_store_contract.py`, run against both), so the store the fast tests use cannot drift from the real one.

- **Deferred (not in this stage).** Answer-generation LLMs, HyDE, Self-RAG, GraphRAG/RAPTOR, thread summaries, OCR, a Gmail UI, query routing, Postgres RLS, and any `src/` cleanup.

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
- Primary path: deterministic rule-based rewrite for pronouns, ellipsis, temporal references, and corrections.
- Optional cloud enhancement path: Gemini, behind `EMAIL_RAG_ENABLE_CLOUD_REWRITE=true`.
- When Gemini is enabled, the app still produces a local rule-based rewrite first, then optionally refines that query with Gemini using only the session context and local draft rewrite.
- `rewrite_mode` is returned and logged as `rules` or `rules+gemini`.

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

## Stage 5 — Evidence-backed graph extraction

Stage 5 builds a tenant- and mailbox-isolated evidence graph from clean chunk
`text`: entities (`PERSON/ORG/PROJECT/…`), relation observations, and temporal
facts. It runs as an optional async job queue (`GRAPH_EXTRACTION_ENABLED=true`),
inert when off — no LLM package is imported and nothing is enqueued.

- **Graph facts are evidence-backed retrieval data, not independently verified
  truth.** The LLM supplies only evidence *strings*; code locates each string in
  the chunk's own immutable `text` and derives the offsets itself, dropping
  anything it cannot locate verbatim. Every entity/relation/fact traces to an
  exact span of authored text.
- **`active` is the latest retained assertion under an explicit supersession
  rule.** A fact is moved to `superseded` only when a later fact's evidence text
  carries an explicit update cue (`replaces`, `updated from`, `now`, `instead
  of`) — never merely because a newer email exists. A date alone never
  supersedes.
- Metadata edges (`SENT/CC/REPLY_TO`) are stored as `evidence_kind=metadata`
  with no offsets, and a DB `CHECK` forbids them ever carrying an authored-text
  span.
- **Run it:** `python -m email_thread_rag.graph.worker` (drains the queue) and
  `python -m email_thread_rag.graph.backfill` (idempotent, resumable). Both
  refuse to run unless graph extraction is enabled.
- **Deferred until Stage 6:** graph reads were not wired into retrieval.

## Stage 6 — Deterministic query planning + evidence-backed graph retrieval

Stage 6 adds a deterministic query planner that fuses the existing hybrid
retriever with graph-evidence retrieval. It decides *how* to retrieve; it does
not generate answers.

- **Deterministic routing, no model.** `rag/planner.py` classifies a query with
  regex + literal token rules only — never an LLM, embeddings, spaCy, or a
  network call — and returns a typed `RetrievalPlan` (routes, tenant/mailbox
  scope, parsed `as_of`, bounded limits, and rule/fallback labels for tracing).
- **Route types:**
  - *Generic* → existing BM25 + dense hybrid.
  - *Entity/relationship* → matching graph entities, relations, and facts, whose
    evidence chunks are fused with the hybrid candidates.
  - *Current/latest* (`current`, `latest`, `now`, `updated`, `replaced`) → only
    `active` facts for the query's subject scope.
  - *`as of <date>`* → facts with a real `effective_date` at or before an
    explicit, unambiguous date; undated facts are never treated as historically
    valid.
- **Graph retrieval always returns source email evidence.** Every graph branch
  resolves through mentions / relation evidence / `fact_evidence` to real email
  chunks; the returned hits are canonical `RetrievalHit`/`ChunkRecord`s with
  exact source spans — never synthetic fact strings, headers, or prose. A
  metadata edge can help *retrieve* a related email but the citation is always
  the chunk's own clean authored text.
- **Reused fusion, bounded graph weight.** bm25 + dense + graph are fused by the
  same weighted-RRF implementation (`weighted_rrf_multi`), deduplicated by chunk
  identity with per-branch provenance kept in `source_lists`. Weight and
  candidate limits are config (`GRAPH_BRANCH_WEIGHT`, `GRAPH_CANDIDATE_LIMIT`,
  `GRAPH_TEMPORAL_CANDIDATE_LIMIT`).
- **Hybrid fallback.** If a graph route yields no citable chunks, retrieval falls
  back to the hybrid retriever and records the reason in the trace.
- **Safe by default.** `GRAPH_PLANNER_ENABLED=true` but inert without graph data
  (every route falls back to hybrid, so existing deployments retrieve
  identically). The memory backend is untouched and imports no ParadeDB/graph
  code; only the ParadeDB retrieval path consults the planner.
- **No answer generation and no Self-RAG in this stage** — nor HyDE, RAPTOR, OCR,
  UI, or RLS.

## Stage 7 — Grounded answering + bounded Self-RAG
Disabled by default (`ANSWER_GENERATION_ENABLED=false`); when off, the
deterministic answer path is used and no provider or HTTP client is imported.
When enabled, an LLM drafts an answer *on top of* Stage-6 retrieval, and local
validation — not the model — decides whether it ships.

- **Answer flow.** `query → Stage-6 retrieval → clean evidence pack → structured
  LLM draft → local validation → accept | one retry | abstain`. The evidence pack
  is built solely from clean `ChunkRecord.text`, deduplicated and bounded
  (`ANSWER_EVIDENCE_BUDGET`). Email bodies are wrapped as untrusted, delimited
  data; sender/date/subject are display metadata, never factual proof.
- **Citation contract.** Every factual claim carries ≥1 citation, and each
  citation resolves to a chunk in the *current* retrieval result and quotes that
  chunk's clean authored text verbatim (with exact offsets). Citations never
  resolve to `embed_text`, headers, graph prose, quoted history, signatures, or
  metadata. Graph facts remain retrieval cues only: answers cite the underlying
  email chunks, never synthetic fact rows.
- **Self-RAG is advisory.** The provider returns `is_relevant` / `is_supported` /
  `is_useful` / `needs_more_evidence`, but local validation is authoritative.
  Malformed JSON, invented ids/quotes, wrong offsets, uncited claims, or
  metadata-only "evidence" reject the draft — as does any content injected inside
  an email body trying to change the contract.
- **Bounded retry.** On rejection (or a model that flags its own draft
  unsupported/not useful), retrieval is re-run once with a wider, still-bounded
  candidate budget and the model drafts again. The ceiling is fixed at two
  attempts in code, not configurable.
- **Abstention.** If the second attempt also fails, there is no evidence, or the
  provider is disabled/fails, the result is an explicit `abstained` — never an
  unsupported answer. Traces stay body-free: route, candidate counts, the
  validation rule, and the attempt count only.

## Stage 8 — PDF attachment extraction, local OCR, and page-level citations

Adds end-to-end attachment support, deliberately narrow: **PDF only**. Non-PDF
attachments (DOCX/XLSX/images/…) are never entered into the pipeline.

- **Flow.** Gmail attachment metadata → extraction job → fetch/decode bytes →
  native PDF text per page → OCR fallback only for pages with no usable native
  text → page chunks → the existing embedding/retrieval/graph/context pipelines.
- **PDF-only boundary.** Only `application/pdf` is handled. Unsupported,
  oversized, encrypted/password-protected, or malformed files fail safely
  (`extraction_status` = `failed`/`unsupported`, a safe reason string) and never
  enter retrieval.
- **Sync never blocks.** Gmail sync persists attachment metadata and enqueues
  idempotent extraction work (`email_attachments` + `attachment_extraction_jobs`,
  the same `pending → running → done|failed`, leased-claim, input-hash,
  tenant/mailbox-scoped pattern as Stages 4/5). The attachment **bytes** are
  fetched (`messages.attachments.get`) only in the extraction worker, off the
  sync path — so a slow parse/OCR never stalls sync.
- **Native extraction vs OCR fallback.** Each page is extracted natively first.
  A page with no usable native text (image-only/scanned) is OCR'd **only if** a
  local OCR backend is available. OCR is optional (install the `ocr` extra and
  set `ATTACHMENT_OCR_ENABLED=true`); it uses a local Tesseract binary — no cloud
  OCR, no API key, no model download. When OCR is disabled or unavailable, the
  page is recorded as **unavailable** and produces no chunk — text is never
  invented.
- **Page chunks & the citation format.** One page becomes one or more canonical
  `chunk_kind='attachment'` chunks. `text` is the exact extracted page text;
  `embed_text` carries compact parent-email metadata, the filename, and
  `Page: N`. Each chunk keeps a page-local span and the attachment citation
  identity: attachment id, filename, 1-indexed page number, chunk/page offsets,
  and extraction method (`native_pdf` or `ocr`). An OCR-derived citation is
  labeled as such (`[budget.pdf, page: 1 (OCR)]`) and is never presented as
  byte-perfect original text.
- **Untrusted content.** Attachment page text is untrusted input and flows
  through the same Stage-7 prompt-injection boundary as email bodies; the parent
  email body and the attachment page text stay separate evidence sources.
- **Reuse, not reinvention.** Attachment chunks ride the existing chunk/embed
  helpers, the one hybrid/graph retriever, and the Stage-7 answer path — no second
  vector store, citation system, or answer path. A Stage-7 answer cites the
  attachment page/chunk with an exact quote, never a synthetic summary.
- **Idempotency & stale replacement.** The queue is idempotent by attachment
  content/input hash: re-syncing an unchanged attachment does no duplicate work;
  a changed one re-extracts and replaces only that attachment's stale page chunks
  (the parent email's body chunks are untouched), then re-triggers the existing
  context/graph enqueues.
- **Unsupported attachment behavior.** Anything that is not a usable PDF is
  recorded terminally and excluded from retrieval; deterministic failures
  (encrypted/malformed/oversized/unsupported) are not retried, transient Gmail
  fetch errors are.

Run the worker with `python -m email_thread_rag.rag.attachments.worker --once`.

## Known limitations
- The checked-in manifest points at the real Enron Archive GitHub repo, but this sandbox cannot verify a live network fetch during implementation.
- If you want a stricter pin than `revision=main`, set a specific dataset revision in `data/raw/dataset_manifest.json`.
- If local model downloads are unavailable, the project falls back to deterministic rewrite and lightweight test encoders/rerankers; the production path still targets the configured open-source models.
- The real Enron slice contains many legacy Office attachments; native `.doc` extraction outside Docker depends on `antiword`.
